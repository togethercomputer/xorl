import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from xorl.server.launcher import load_server_arguments, parse_server_overrides, validate_server_overrides
from xorl.server.removed_config import reject_removed_configuration_fields


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _assert_yaml_rejects_removed_fields_before_filtering(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    cases = [
        (
            "flat adapter ownership",
            {
                "model_path": "Qwen/Qwen3-8B",
                "adapter_gradient_ownership_mode": "observe",
                "shared_config_key_not_owned_by_server": True,
            },
            "adapter_gradient_ownership_mode.*authoritative-only",
        ),
        (
            "nested ZORL field",
            {"model": {"model_path": "Qwen/Qwen3-8B"}, "train": {"zorl_b_sigma": 0.01}},
            "train.zorl_b_sigma.*ZORL was removed",
        ),
        (
            "nested ZORL section",
            {"model": {"model_path": "Qwen/Qwen3-8B"}, "zorl": {"enabled": True, "sigma": 0.01}},
            "zorl.*ZORL was removed",
        ),
        (
            "nested NeMo FP8 alias",
            {"model": {"model_path": "Qwen/Qwen3-8B"}, "train": {"fp8_cfg": {"enabled": True}}},
            "train.fp8_cfg.*native fp8_training",
        ),
    ]

    for label, payload, error_pattern in cases:
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=error_pattern):
            load_server_arguments(str(config_path))


def _assert_load_server_arguments_rejects_removed_cli_override(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(yaml.safe_dump({"model_path": "Qwen/Qwen3-8B"}), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_gradient_ownership_shadow_canary.*authoritative-only"):
        load_server_arguments(
            str(config_path),
            overrides={"adapter_gradient_ownership_shadow_canary": True},
        )


def _assert_removed_field_inventory_allows_unrelated_unknown_fields():
    payload = {"shared_config_key_not_owned_by_server": {"future_nested_key": True}}
    assert reject_removed_configuration_fields(payload, context="test config") is payload


_SHIPPED_MOE_LORA_CONFIGS = (
    "examples/server/configs/lora/qwen3_5_35b_a3b_lora.yaml",
    "examples/server/configs/lora/qwen3_coder_30b_a3b_lora.yaml",
)
_SHIPPED_QWEN_MOE_QLORA_CONFIGS = (
    "examples/server/configs/qlora/qwen3_235b_a22b_qlora_nf4.yaml",
    "examples/server/configs/qlora/qwen3_235b_a22b_qlora_nvfp4.yaml",
    "examples/server/configs/qlora/qwen3_30b_a3b_qlora_nf4.yaml",
    "examples/server/configs/qlora/qwen3_30b_a3b_qlora_nvfp4.yaml",
    "examples/server/configs/qlora/qwen3_coder_30b_a3b_qlora.yaml",
)


@pytest.fixture(scope="module")
def clean_shipped_adapter_arguments():
    """Parse shipped configs in a clean process, outside this module's launcher stubs."""

    root = Path(__file__).resolve().parents[2]
    relative_paths = _SHIPPED_MOE_LORA_CONFIGS + _SHIPPED_QWEN_MOE_QLORA_CONFIGS
    script = """
import json
import sys
from xorl.server.launcher import load_server_arguments

result = {}
for path in sys.argv[1:]:
    args = load_server_arguments(path)
    result[path] = {
        "moe_implementation": args.moe_implementation,
        "moe_hybrid_shared_lora": args.moe_hybrid_shared_lora,
        "lora_target_modules": args.lora_target_modules,
    }
print("XORL_SHIPPED_CONFIG_ARGS=" + json.dumps(result, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-c", script, *(str(root / path) for path in relative_paths)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    marker = "XORL_SHIPPED_CONFIG_ARGS="
    payload = next(line.removeprefix(marker) for line in completed.stdout.splitlines() if line.startswith(marker))
    parsed = json.loads(payload)
    return {path: parsed[str(root / path)] for path in relative_paths}


def _assert_shipped_moe_lora_examples_restore_certified_quack(clean_shipped_adapter_arguments):
    root = Path(__file__).resolve().parents[2]
    for relative_path in _SHIPPED_MOE_LORA_CONFIGS:
        config = yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        parsed = clean_shipped_adapter_arguments[relative_path]
        assert config["moe_implementation"] == parsed["moe_implementation"] == "quack", relative_path
        assert config.get("moe_hybrid_shared_lora", False) is parsed["moe_hybrid_shared_lora"] is False


def _assert_shipped_qwen_quantized_moe_examples_restore_quack_expert_targets(clean_shipped_adapter_arguments):
    expected_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    root = Path(__file__).resolve().parents[2]
    for relative_path in _SHIPPED_QWEN_MOE_QLORA_CONFIGS:
        config = yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        parsed = clean_shipped_adapter_arguments[relative_path]
        assert config["moe_implementation"] == parsed["moe_implementation"] == "quack", relative_path
        assert config["lora_target_modules"] == parsed["lora_target_modules"] == expected_targets
        assert config["moe_hybrid_shared_lora"] is parsed["moe_hybrid_shared_lora"] is True


def _assert_canonical_moe_and_rope_auto_defaults_serialize(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_path": "synthetic"},
                "train": {"output_dir": str(tmp_path / "outputs")},
            }
        ),
        encoding="utf-8",
    )
    args = load_server_arguments(str(config_path))
    model_config = args.to_config_dict()["model"]
    assert model_config["attn_implementation"] is None
    assert model_config["router_fp32"] is None
    assert model_config["lm_head_fp32"] is None
    assert model_config["rmsnorm_mode"] is None
    assert model_config["rope_native"] is None
    assert model_config["rope_class_b"] is None
    assert model_config["sparse_mla_enabled"] is None
    assert model_config["train_router"] is False
    assert args.to_config_dict()["train"]["ce_mode"] is None


def _assert_load_server_arguments_preserves_nested_runtime_controls(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    for optimizer in ("signsgd", "distsignsgd"):
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_path": "Qwen/Qwen3-8B"},
                    "train": {
                        "optimizer": optimizer,
                        "output_dir": str(tmp_path / "outputs"),
                        "load_checkpoint_path": "/tmp/initial-dcp",
                        "load_optimizer": False,
                        "enable_forward_prefetch": False,
                        "enable_backward_prefetch": True,
                        "data_parallel_replicate_size": 2,
                        "data_parallel_shard_size": 4,
                        "defer_grad_sync_in_accumulation": True,
                        "enable_activation_offload": True,
                        "activation_gpu_limit": 0.25,
                    },
                    "data": {"sample_packing_sequence_len": 16384, "pad_to_multiple_of": 4096},
                    "lora": {"enable_lora": True, "adapter_state_load_mode": "rank0_broadcast"},
                },
            ),
            encoding="utf-8",
        )

        args = load_server_arguments(str(config_path))
        config = args.to_config_dict()
        train_config = config["train"]

        assert args.optimizer == train_config["optimizer"] == optimizer
        assert args.load_weights_mode == train_config["load_weights_mode"] == "grouped"
        assert args.load_optimizer is train_config["load_optimizer"] is False
        assert train_config["load_checkpoint_path"] == "/tmp/initial-dcp"
        assert args.enable_forward_prefetch is train_config["enable_forward_prefetch"] is False
        assert args.enable_backward_prefetch is train_config["enable_backward_prefetch"] is True
        assert args.defer_grad_sync_in_accumulation is train_config["defer_grad_sync_in_accumulation"] is True
        assert args.pad_to_multiple_of == train_config["pad_to_multiple_of"] == 4096
        assert args.activation_gpu_limit == train_config["activation_gpu_limit"] == pytest.approx(0.25)
        assert args.adapter_state_load_mode == config["lora"]["adapter_state_load_mode"] == "rank0_broadcast"


def _assert_server_arguments_r3_payload_transport_admission(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    payload_dir = tmp_path / "r3-payloads"
    cases = [
        (
            {"r3_payload_transport": "mooncake", "r3_payload_keep": True, "r3_payload_namespace_prefix": "tests/r3"},
            {"r3_payload_transport": "mooncake", "r3_payload_keep": True, "r3_payload_namespace_prefix": "tests/r3"},
        ),
        (
            {"r3_payload_transport": "filesystem", "r3_payload_dir": str(payload_dir), "r3_payload_keep": True},
            {"r3_payload_transport": "filesystem", "r3_payload_dir": str(payload_dir), "r3_payload_keep": True},
        ),
    ]

    for train_payload, expected in cases:
        config_path.write_text(
            yaml.safe_dump({"model": {"model_path": "Qwen/Qwen3-8B"}, "train": train_payload}),
            encoding="utf-8",
        )
        args = load_server_arguments(str(config_path))
        train_config = args.to_config_dict()["train"]
        for field, value in expected.items():
            assert getattr(args, field) == value
            assert train_config[field] == value

    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "r3_payload_dir": str(tmp_path / "r3-payloads"),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="r3_payload_dir requires r3_payload_transport='filesystem'"):
        load_server_arguments(str(config_path))


def _assert_load_server_arguments_threads_fp8_training_into_train_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_fp8_training": True,
                    "fp8_training_block_size": 128,
                    "fp8_training_backward": "fp8",
                    "fp8_training_smoothquant_alpha": 0.5,
                    "fp8_training_lm_head_smoothquant_alpha": 0.4,
                    "fp8_training_activation_amax_scale": 0.875,
                    "fp8_training_weight_amax_scale": 1.125,
                    "fp8_training_correction_mode": "activation2",
                    "fp8_training_module_overrides": {
                        "model.layers.3[4-5].mlp.down_proj": {"block_size": 32, "correction_mode": "first_order"},
                        "lm_head": {"smoothquant_alpha": 0.5, "correction_mode": "full"},
                    },
                    "fp8_training_moe_grouped_backend": "triton_grouped",
                    "fp8_training_target_modules": ["q_proj", "k_proj"],
                    "fp8_training_exclude_modules": ["lm_head"],
                    "fp8_training_allow_bf16_fallback": False,
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_block_size"] == 128
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_smoothquant_alpha"] == 0.5
    assert train_config["fp8_training_lm_head_smoothquant_alpha"] == 0.4
    assert train_config["fp8_training_activation_amax_scale"] == 0.875
    assert train_config["fp8_training_weight_amax_scale"] == 1.125
    assert train_config["fp8_training_correction_mode"] == "activation2"
    assert train_config["fp8_training_module_overrides"] == {
        "model.layers.3[4-5].mlp.down_proj": {"block_size": 32, "correction_mode": "first_order"},
        "lm_head": {"smoothquant_alpha": 0.5, "correction_mode": "full"},
    }
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_target_modules"] == ["q_proj", "k_proj"]
    assert train_config["fp8_training_exclude_modules"] == ["lm_head"]
    assert train_config["fp8_training_allow_bf16_fallback"] is False


def _assert_load_server_arguments_threads_glm52_block_fp8_qlora_mode(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "zai-org/GLM-5.2-FP8",
                    "moe_implementation": "triton",
                    "ep_dispatch": "deepep",
                    "merge_qkv": True,
                },
                "train": {
                    "freeze_router": True,
                    "output_dir": str(tmp_path / "outputs"),
                },
                "lora": {
                    "enable_lora": True,
                    "enable_qlora": True,
                    "block_fp8_qlora_training": True,
                    "quant_format": "block_fp8",
                    "quant_group_size": 128,
                    "moe_hybrid_shared_lora": True,
                    "lora_export_format": "sglang_shared_outer",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()

    assert args.block_fp8_qlora_training is True
    assert config["lora"]["enable_lora"] is True
    assert config["lora"]["enable_qlora"] is True
    assert config["lora"]["block_fp8_qlora_training"] is True
    assert config["lora"]["quant_format"] == "block_fp8"
    assert config["lora"]["quant_group_size"] == 128
    assert config["lora"]["moe_hybrid_shared_lora"] is True


def _exact_glm52_rank1_server_config(tmp_path):
    return {
        "model": {
            "model_path": "zai-org/GLM-5.2-FP8",
            "moe_implementation": "triton",
            "ep_dispatch": "alltoall",
            "merge_qkv": True,
        },
        "train": {
            "freeze_router": True,
            "output_dir": str(tmp_path / "outputs"),
            "expert_parallel_size": 16,
            "ulysses_parallel_size": 16,
            "lm_head_tensor_parallel_size": 16,
            "fsdp_sharded_lm_head_loss": True,
        },
        "lora": {
            "enable_lora": True,
            "enable_qlora": True,
            "block_fp8_qlora_training": True,
            "lora_rank": 1,
            "max_lora_rank": 1,
            "lora_alpha": 1,
            "quant_format": "block_fp8",
            "quant_group_size": 128,
            "moe_hybrid_shared_lora": True,
            "lora_export_format": "sglang_shared_outer",
        },
    }


def _assert_load_server_arguments_exact_glm52_rank1_topology_admission(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(yaml.safe_dump(_exact_glm52_rank1_server_config(tmp_path)), encoding="utf-8")

    args = load_server_arguments(str(config_path))

    assert (args.lora_rank, args.max_lora_rank, args.lora_alpha) == (1, 1, 1)
    assert args.ep_dispatch == "alltoall"
    assert (args.expert_parallel_size, args.ulysses_parallel_size) == (16, 16)
    assert args.lm_head_tensor_parallel_size == 16
    assert args.fsdp_sharded_lm_head_loss is True
    assert args.get_total_gpus() == 16

    payload = _exact_glm52_rank1_server_config(tmp_path)
    payload["train"]["lm_head_tensor_parallel_size"] = 1
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lm_head_tensor_parallel_size=1"):
        load_server_arguments(str(config_path))


def _assert_load_server_arguments_threads_qarl_into_train_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_qarl": True,
                    "qarl_quant_cfg": "FP8_DEFAULT_CFG",
                    "qarl_sync_format": "fp8",
                    "qarl_calib_data": str(tmp_path / "calib.json"),
                    "qarl_calib_size": 4,
                    "qarl_quant_sequence_length": 16,
                    "qarl_target_modules": ["q_proj", "k_proj"],
                    "qarl_exclude_modules": ["lm_head"],
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.enable_qarl is True
    assert train_config["enable_qarl"] is True
    assert train_config["qarl_quant_cfg"] == {
        "format": "fp8_e4m3",
        "weight": True,
        "activation": True,
        "dynamic": True,
        "weight_block_size": [128, 128],
    }
    assert train_config["qarl_sync_format"] == "fp8"
    assert train_config["qarl_calib_data"] == str(tmp_path / "calib.json")
    assert train_config["qarl_calib_size"] == 4
    assert train_config["qarl_quant_sequence_length"] == 16
    assert train_config["qarl_target_modules"] == ["q_proj", "k_proj"]
    assert train_config["qarl_exclude_modules"] == ["lm_head"]


def _assert_load_server_arguments_rejects_quantized_training_incompatible_scopes_and_sources(tmp_path):
    qarl_lora_path = tmp_path / "qarl_lora.yaml"
    qarl_lora_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_qarl": True,
                },
                "lora": {
                    "enable_lora": True,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="enable_qarl is a full-weight mode"):
        load_server_arguments(str(qarl_lora_path))

    fp8_lora_path = tmp_path / "fp8_lora.yaml"
    fp8_lora_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_fp8_training": True,
                },
                "lora": {
                    "enable_lora": True,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="enable_fp8_training is a full-weight mode"):
        load_server_arguments(str(fp8_lora_path))

    qarl_fp8_path = tmp_path / "qarl_fp8.yaml"
    qarl_fp8_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_qarl": True,
                    "enable_fp8_training": True,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="enable_qarl cannot be combined with enable_fp8_training"):
        load_server_arguments(str(qarl_fp8_path))

    qarl_bad_calib_path = tmp_path / "qarl_bad_calib.yaml"
    qarl_bad_calib_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_qarl": True,
                    "qarl_quant_sequence_length": 16,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="require qarl_calib_data"):
        load_server_arguments(str(qarl_bad_calib_path))

    qarl_mtp_path = tmp_path / "qarl_mtp.yaml"
    qarl_mtp_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "foundation": {"text_config": {"num_nextn_predict_layers": 1}},
                },
                "train": {"enable_qarl": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MTP/speculative and Mamba"):
        load_server_arguments(str(qarl_mtp_path))

    model_dir = tmp_path / "mamba-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mamba", "architectures": ["MambaForCausalLM"]}),
        encoding="utf-8",
    )
    qarl_mamba_path = tmp_path / "qarl_mamba.yaml"
    qarl_mamba_path.write_text(
        yaml.safe_dump({"model": {"model_path": str(model_dir)}, "train": {"enable_qarl": True}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config_json.model_type=mamba"):
        load_server_arguments(str(qarl_mamba_path))

    nemo_modelopt_path = tmp_path / "nemo_modelopt_qarl.yaml"
    nemo_modelopt_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "policy": {"generation": {"quant_cfg": "FP8_DEFAULT_CFG"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ModelOpt QARL"):
        load_server_arguments(str(nemo_modelopt_path))


def _assert_load_server_arguments_rejects_vllm_fp8_runtime_knobs(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "generation": {"vllm_cfg": {"precision": "fp8"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vLLM FP8 receiver"):
        load_server_arguments(str(config_path))

    quantization_path = tmp_path / "server_config_quantization.yaml"
    quantization_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "generation": {"vllm_cfg": {"quantization": "fp8"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vLLM FP8 receiver"):
        load_server_arguments(str(quantization_path))

    ignored_layers_path = tmp_path / "server_config_ignored_layers.yaml"
    ignored_layers_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "generation": {"vllm_cfg": {"quantization_ignored_layer_kws": ["a_proj"]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quantization_ignored_layer_kws"):
        load_server_arguments(str(ignored_layers_path))

    kv_cache_path = tmp_path / "server_config_kv.yaml"
    kv_cache_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "generation": {"vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receiver_kv_cache_dtype"):
        load_server_arguments(str(kv_cache_path))

    pow2_path = tmp_path / "server_config_pow2.yaml"
    pow2_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "generation": {"vllm_cfg": {"pow2_activation_scaling_factors": True}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pow2_activation_scaling_factors"):
        load_server_arguments(str(pow2_path))


def _assert_load_server_arguments_threads_receiver_kv_cache_dtype(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "receiver_kv_cache_dtype": "FP8_E4M3",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.receiver_kv_cache_dtype == "fp8_e4m3"
    assert args.to_config_dict()["train"]["receiver_kv_cache_dtype"] == "fp8_e4m3"


def _assert_adapter_gradient_transport_bucket_configuration(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_path": "synthetic"},
                "train": {"adapter_gradient_ownership_bucket_bytes": 1024},
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    assert args.adapter_gradient_ownership_bucket_bytes == 1024
    assert args.to_config_dict()["train"]["adapter_gradient_ownership_bucket_bytes"] == 1024

    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_path": "synthetic"},
                "train": {"adapter_gradient_ownership_bucket_bytes": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bucket_bytes"):
        load_server_arguments(str(config_path))


def _assert_load_server_arguments_threads_lm_head_tp_loss_fields(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "ulysses_parallel_size": 8,
                    "lm_head_tensor_parallel_size": 8,
                    "fsdp_sharded_lm_head_loss": True,
                    "fsdp_sharded_lm_head_loss_num_chunks": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.lm_head_tensor_parallel_size == 8
    assert args.fsdp_sharded_lm_head_loss is True
    assert args.fsdp_sharded_lm_head_loss_num_chunks == 4
    assert train_config["lm_head_tensor_parallel_size"] == 8
    assert train_config["fsdp_sharded_lm_head_loss"] is True
    assert train_config["fsdp_sharded_lm_head_loss_num_chunks"] == 4


def _assert_model_tensor_parallel_lora_is_rejected_but_output_head_tp_remains_distinct(tmp_path):
    rejected_path = tmp_path / "rejected.yaml"
    rejected_path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_path": "synthetic"},
                "train": {"tensor_parallel_size": 2},
                "lora": {"enable_lora": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
        load_server_arguments(str(rejected_path))

    admitted_path = tmp_path / "output_head.yaml"
    admitted_path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_path": "synthetic"},
                "train": {"lm_head_tensor_parallel_size": 2},
                "lora": {"enable_lora": True},
            }
        ),
        encoding="utf-8",
    )
    args = load_server_arguments(str(admitted_path))
    assert args.tensor_parallel_size == 1
    assert args.lm_head_tensor_parallel_size == 2


def _assert_server_fp8_training_defaults_to_fail_fast_fallback(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_fp8_training": True,
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.enable_fp8_training is True
    assert train_config["fp8_training_allow_bf16_fallback"] is False


def _assert_load_server_arguments_rejects_broadcast_load_weights_mode(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "load_weights_mode": "broadcast",
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported load_weights_mode"):
        load_server_arguments(str(config_path))


def _assert_load_server_arguments_rejects_unsupported_multi_adapter_modes(tmp_path):
    merge_interval_path = tmp_path / "merge_interval.yaml"
    merge_interval_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "lora": {
                    "enable_lora": True,
                    "merge_lora_interval": 16,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="merge_lora_interval is not supported"):
        load_server_arguments(str(merge_interval_path))

    pipeline_parallel_path = tmp_path / "pipeline_parallel.yaml"
    pipeline_parallel_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "pipeline_parallel_size": 2,
                },
                "lora": {
                    "enable_lora": True,
                    "adapter_state_load_mode": "rank0_broadcast",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pipeline_parallel_size > 1 is not supported"):
        load_server_arguments(str(pipeline_parallel_path))


def _assert_load_server_arguments_threads_muon_gram_newton_schulz_through_nested_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "optimizer": "muon",
                    "optimizer_dtype": "bf16",
                    "muon_lr": 0.03,
                    "muon_ns_algorithm": "gram_newton_schulz",
                    "muon_ns_use_quack_kernels": False,
                    "muon_gram_ns_num_restarts": 2,
                    "muon_gram_ns_restart_iterations": [2],
                    "muon_grouped_gram_ns_fp32_byte_limit": 23,
                    "muon_fallback_optimizer": "sgd",
                    "muon_grad_dtype": "fp32",
                    "muon_update_dtype": "bf16",
                    "muon_force_momentum_path": True,
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    optimizer_kwargs = args.to_config_dict()["train"]["optimizer_kwargs"]
    assert optimizer_kwargs["muon_lr"] == pytest.approx(0.03)
    assert optimizer_kwargs["muon_ns_algorithm"] == "gram_newton_schulz"
    assert optimizer_kwargs["muon_ns_use_quack_kernels"] is False
    assert optimizer_kwargs["muon_gram_ns_num_restarts"] == 2
    assert optimizer_kwargs["muon_gram_ns_restart_iterations"] == [2]
    assert optimizer_kwargs["muon_momentum_dtype"] == torch.bfloat16
    assert optimizer_kwargs["muon_grad_dtype"] == torch.float32
    assert optimizer_kwargs["muon_update_dtype"] == torch.bfloat16
    assert optimizer_kwargs["muon_force_momentum_path"] is True

    train_config = args.to_config_dict()["train"]
    assert args.optimizer == "muon"
    assert args.muon_ns_algorithm == "gram_newton_schulz"
    assert train_config["muon_ns_algorithm"] == "gram_newton_schulz"
    assert train_config["muon_ns_use_quack_kernels"] is False
    assert train_config["muon_gram_ns_num_restarts"] == 2
    assert train_config["muon_gram_ns_restart_iterations"] == [2]
    assert train_config["muon_grouped_gram_ns_fp32_byte_limit"] == 23
    assert train_config["muon_fallback_optimizer"] == "sgd"
    assert args.muon_fallback_optimizer == "sgd"
    assert args.muon_grad_dtype == "fp32"
    assert args.muon_update_dtype == "bf16"
    assert args.muon_force_momentum_path is True
    assert train_config["muon_grad_dtype"] == "fp32"
    assert train_config["muon_update_dtype"] == "bf16"
    assert train_config["muon_force_momentum_path"] is True


def _assert_load_server_arguments_preserves_runner_compatibility_fields(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "record_routing_weights": False,
                    "rmsnorm_mode": "sglang",
                },
                "train": {
                    "enable_full_determinism": True,
                    "optimizer": "muon",
                    "cautious_weight_decay": True,
                    "muon_distributed_mode": "full_gradient",
                    "moe_grad_reduce_mode": "bf16_a2a_fp32_sum",
                    "output_dir": str(tmp_path / "outputs"),
                },
                "lora": {
                    "lora_export_format": "sglang_shared_outer",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()

    assert config["model"]["record_routing_weights"] is False
    assert config["model"]["rmsnorm_mode"] == "sglang"
    assert config["train"]["enable_full_determinism"] is True
    assert config["train"]["cautious_weight_decay"] is True
    assert config["train"]["muon_distributed_mode"] == "full_gradient"
    assert config["train"]["moe_grad_reduce_mode"] == "bf16_a2a_fp32_sum"
    assert config["train"]["optimizer_kwargs"]["muon_distributed_mode"] == "full_gradient"
    assert config["lora"]["lora_export_format"] == "sglang_shared_outer"


def _assert_load_server_arguments_threads_sparse_mla_into_model_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "zai-org/GLM-5",
                "sparse_mla_enabled": True,
                "sparse_mla_backend": "flashmla",
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    model_config = args.to_config_dict()["model"]

    assert args.sparse_mla_enabled is True
    assert args.sparse_mla_backend == "flashmla"
    assert model_config["sparse_mla_enabled"] is True
    assert model_config["sparse_mla_backend"] == "flashmla"


def _assert_server_arguments_moe_routing_weights_before_down_explicit_and_auto(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    for configured, expected in ((True, True), (None, "auto")):
        model = {"model_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct"}
        if configured is not None:
            model["moe_routing_weights_before_down"] = configured
        config_path.write_text(
            yaml.safe_dump({"model": model, "train": {"output_dir": str(tmp_path / "outputs")}}),
            encoding="utf-8",
        )

        args = load_server_arguments(str(config_path))

        assert args.moe_routing_weights_before_down == expected
        assert args.to_config_dict()["model"]["moe_routing_weights_before_down"] == expected


def test_removed_configuration_boundary(tmp_path):
    _assert_yaml_rejects_removed_fields_before_filtering(tmp_path)
    _assert_load_server_arguments_rejects_removed_cli_override(tmp_path)
    _assert_removed_field_inventory_allows_unrelated_unknown_fields()
    _assert_server_override_parse_and_validation_policy()
    unsupported_root = tmp_path / "unsupported"
    unsupported_root.mkdir()
    _assert_server_rejects_unsupported_training_configurations(unsupported_root)


def _assert_server_override_parse_and_validation_policy():
    _, overrides = parse_server_overrides(["--server.bogus_key=42"])
    assert overrides == {"bogus_key": 42}

    with pytest.raises(ValueError, match="enable_zorl.*ZORL was removed"):
        validate_server_overrides({"enable_zorl": True})


def test_server_runtime_configuration_round_trip(tmp_path):
    _assert_canonical_moe_and_rope_auto_defaults_serialize(tmp_path)
    _assert_load_server_arguments_preserves_nested_runtime_controls(tmp_path)
    _assert_load_server_arguments_threads_receiver_kv_cache_dtype(tmp_path)
    _assert_server_arguments_r3_payload_transport_admission(tmp_path)
    _assert_adapter_gradient_transport_bucket_configuration(tmp_path)
    _assert_server_optimizer_and_runner_compatibility_contract(tmp_path)
    _assert_server_model_specific_configuration_contract(tmp_path)


def test_server_quantized_training_configuration_contract(tmp_path, clean_shipped_adapter_arguments):
    _assert_load_server_arguments_threads_fp8_training_into_train_config(tmp_path)
    _assert_load_server_arguments_threads_glm52_block_fp8_qlora_mode(tmp_path)
    _assert_load_server_arguments_threads_qarl_into_train_config(tmp_path)
    _assert_server_fp8_training_defaults_to_fail_fast_fallback(tmp_path)
    _assert_shipped_moe_lora_examples_restore_certified_quack(clean_shipped_adapter_arguments)
    _assert_shipped_qwen_quantized_moe_examples_restore_quack_expert_targets(clean_shipped_adapter_arguments)


def test_server_parallel_topology_configuration_contract(tmp_path):
    _assert_load_server_arguments_exact_glm52_rank1_topology_admission(tmp_path)
    _assert_load_server_arguments_threads_lm_head_tp_loss_fields(tmp_path)
    _assert_model_tensor_parallel_lora_is_rejected_but_output_head_tp_remains_distinct(tmp_path)


def _assert_server_rejects_unsupported_training_configurations(tmp_path):
    _assert_load_server_arguments_rejects_quantized_training_incompatible_scopes_and_sources(tmp_path)
    _assert_load_server_arguments_rejects_vllm_fp8_runtime_knobs(tmp_path)
    _assert_load_server_arguments_rejects_broadcast_load_weights_mode(tmp_path)
    _assert_load_server_arguments_rejects_unsupported_multi_adapter_modes(tmp_path)


def _assert_server_optimizer_and_runner_compatibility_contract(tmp_path):
    _assert_load_server_arguments_threads_muon_gram_newton_schulz_through_nested_config(tmp_path)
    _assert_load_server_arguments_preserves_runner_compatibility_fields(tmp_path)


def _assert_server_model_specific_configuration_contract(tmp_path):
    _assert_load_server_arguments_threads_sparse_mla_into_model_config(tmp_path)
    _assert_server_arguments_moe_routing_weights_before_down_explicit_and_auto(tmp_path)
