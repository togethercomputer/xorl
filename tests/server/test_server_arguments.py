import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _load_server_arguments_fn():
    module_path = Path(__file__).resolve().parents[2] / "src" / "xorl" / "server" / "launcher.py"
    spec = importlib.util.spec_from_file_location("xorl_test_launcher", module_path)
    assert spec is not None and spec.loader is not None

    fake_api_server_pkg = types.ModuleType("xorl.server.api_server")
    fake_api_server_pkg.__path__ = []
    fake_api_server_mod = types.ModuleType("xorl.server.api_server.server")
    fake_api_server_mod.APIServer = object
    fake_api_server_pkg.server = fake_api_server_mod

    fake_orchestrator_pkg = types.ModuleType("xorl.server.orchestrator")
    fake_orchestrator_pkg.__path__ = []
    fake_orchestrator_mod = types.ModuleType("xorl.server.orchestrator.orchestrator")
    fake_orchestrator_mod.Orchestrator = object
    fake_orchestrator_pkg.orchestrator = fake_orchestrator_mod

    fake_utils_pkg = types.ModuleType("xorl.server.utils")
    fake_utils_pkg.__path__ = []
    fake_network_mod = types.ModuleType("xorl.server.utils.network")
    fake_network_mod.read_address_file = lambda *args, **kwargs: None
    fake_utils_pkg.network = fake_network_mod
    fake_session_spec_mod = types.ModuleType("xorl.server.session_spec")
    fake_session_spec_mod.build_default_session_spec = lambda *args, **kwargs: None

    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "xorl.server.api_server": fake_api_server_pkg,
            "xorl.server.api_server.server": fake_api_server_mod,
            "xorl.server.orchestrator": fake_orchestrator_pkg,
            "xorl.server.orchestrator.orchestrator": fake_orchestrator_mod,
            "xorl.server.session_spec": fake_session_spec_mod,
            "xorl.server.utils": fake_utils_pkg,
            "xorl.server.utils.network": fake_network_mod,
        },
    ):
        spec.loader.exec_module(module)

    return module.load_server_arguments


_load_server_arguments_impl = _load_server_arguments_fn()


def load_server_arguments(config_path, *args, **kwargs):
    """Skip (rather than error) when a referenced config file is absent.

    The K3 / FP8 config-contract tests load real experiment configs from
    experiments/k3_tests/configs/, which live outside the src+tests PR scope and
    are not present in this tree. tmp_path-authored and examples/ configs exist
    and proceed normally.
    """
    if not Path(str(config_path)).exists():
        pytest.skip(f"config '{config_path}' not present (experiments/ configs are outside the src+tests merge scope)")
    return _load_server_arguments_impl(config_path, *args, **kwargs)


def test_load_server_arguments_threads_signsgd_through_nested_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "optimizer": "signsgd",
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.optimizer == "signsgd"
    assert args.load_weights_mode == "grouped"
    assert args.to_config_dict()["train"]["optimizer"] == "signsgd"
    assert args.to_config_dict()["train"]["load_weights_mode"] == "grouped"


def test_load_server_arguments_threads_distsignsgd_through_nested_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "optimizer": "distsignsgd",
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.optimizer == "distsignsgd"
    assert args.to_config_dict()["train"]["optimizer"] == "distsignsgd"


def test_load_server_arguments_threads_adapter_state_load_mode_into_lora_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "lora": {
                    "enable_lora": True,
                    "adapter_state_load_mode": "rank0_broadcast",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.adapter_state_load_mode == "rank0_broadcast"
    assert args.to_config_dict()["lora"]["adapter_state_load_mode"] == "rank0_broadcast"


def test_load_server_arguments_threads_activation_gpu_limit_into_train_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_activation_offload": True,
                    "activation_gpu_limit": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.activation_gpu_limit == pytest.approx(0.25)
    assert args.to_config_dict()["train"]["activation_gpu_limit"] == pytest.approx(0.25)


def test_load_server_arguments_threads_fp8_training_into_train_config(tmp_path):
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


def test_load_server_arguments_threads_qarl_into_train_config(tmp_path):
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


def test_load_server_arguments_rejects_qarl_with_adapters_or_fp8(tmp_path):
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


def test_load_server_arguments_rejects_qarl_with_mtp_model_metadata(tmp_path):
    config_path = tmp_path / "qarl_mtp.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "foundation": {"text_config": {"num_nextn_predict_layers": 1}},
                },
                "train": {
                    "enable_qarl": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MTP/speculative and Mamba"):
        load_server_arguments(str(config_path))


def test_load_server_arguments_rejects_qarl_with_mamba_config_json(tmp_path):
    model_dir = tmp_path / "mamba-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mamba", "architectures": ["MambaForCausalLM"]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "qarl_mamba.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": str(model_dir),
                },
                "train": {
                    "enable_qarl": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_json.model_type=mamba"):
        load_server_arguments(str(config_path))


def test_load_server_arguments_accepts_fp8_cfg_alias_and_layer_islands(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "fp8_cfg": {
                        "enabled": True,
                        "fp8": "e4m3",
                        "fp8_recipe": "blockwise",
                        "fp8_param": False,
                    },
                    "fp8_training_num_first_layers_bf16": 1,
                    "fp8_training_num_last_layers_bf16": 2,
                    "fp8_training_allow_blackwell": True,
                    "fp8_training_blackwell_validation_artifact": "artifact.json",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_num_first_layers_bf16"] == 1
    assert train_config["fp8_training_num_last_layers_bf16"] == 2
    assert train_config["fp8_training_allow_blackwell"] is True
    assert train_config["fp8_training_blackwell_validation_artifact"] == "artifact.json"


def test_load_server_arguments_accepts_nemo_policy_fp8_cfg_alias(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "policy": {
                    "megatron_cfg": {
                        "fp8_cfg": {
                            "enabled": True,
                            "fp8": "e4m3",
                            "fp8_recipe": "blockwise",
                            "fp8_param": False,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.fp8_cfg == {"enabled": True, "fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": False}
    assert args.enable_fp8_training is True


def test_load_server_arguments_rejects_vllm_fp8_runtime_knobs(tmp_path):
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


def test_load_server_arguments_rejects_nemo_modelopt_qarl_configs(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "policy": {
                    "generation": {
                        "quant_cfg": "FP8_DEFAULT_CFG",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ModelOpt QARL"):
        load_server_arguments(str(config_path))


def test_load_server_arguments_threads_receiver_kv_cache_dtype(tmp_path):
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


def test_qwen3_8b_fp8_bf16_islands_example_config_loads():
    config_path = Path(__file__).resolve().parents[2] / "examples/server/configs/full/qwen3_8b_fp8_bf16_islands.yaml"

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_num_first_layers_bf16"] == 1
    assert train_config["fp8_training_num_last_layers_bf16"] == 1
    assert train_config["fp8_training_allow_blackwell"] is False


def test_qwen3_8b_fp8_sglang_kv_cache_example_config_loads():
    config_path = Path(__file__).resolve().parents[2] / "examples/server/configs/full/qwen3_8b_fp8_sglang_kv_cache.yaml"

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_num_first_layers_bf16"] == 1
    assert train_config["fp8_training_num_last_layers_bf16"] == 1
    assert train_config["receiver_kv_cache_dtype"] == "fp8_e4m3"
    assert train_config["fp8_training_allow_blackwell"] is False


def test_qwen3_8b_qarl_fp8_fake_quant_example_config_loads():
    config_path = Path(__file__).resolve().parents[2] / "examples/server/configs/full/qwen3_8b_qarl_fp8_fake_quant.yaml"

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert train_config["enable_qarl"] is True
    assert train_config["qarl_quant_cfg"] == {
        "format": "fp8_e4m3",
        "weight": True,
        "activation": True,
        "dynamic": True,
        "weight_block_size": [128, 128],
    }
    assert train_config["qarl_sync_format"] == "fp8"
    assert train_config["qarl_exclude_modules"] == ["lm_head"]


def test_k3_fp8_training_smoke_config_enables_full_model_fp8():
    config_path = Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3-8b_fp8-training-smoke.yaml"

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["data_parallel_shard_size"] == 1


def test_k3_fp8_training_block64_smoke_config_enables_full_model_fp8_with_smaller_blocks():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3-8b_fp8-training-block64-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_block_size"] == 64
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["output_dir"] == "outputs/k3-test-fp8-training-block64-smoke"


def test_k3_qlora_nf4_fp8_sync_smoke_config_enables_adapter_manager_and_qlora():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3-0.6b_qlora-nf4-fp8-sync-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    train_config = config["train"]
    lora_config = config["lora"]

    assert args.model_path == "Qwen/Qwen3-0.6B"
    assert train_config["enable_fp8_training"] is False
    assert lora_config["enable_lora"] is True
    assert lora_config["enable_qlora"] is True
    assert lora_config["quant_format"] == "nf4"
    assert lora_config["quant_group_size"] == 64
    assert lora_config["lora_rank"] == 8
    assert lora_config["max_lora_rank"] == 8
    assert lora_config["lora_target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


@pytest.mark.parametrize(
    ("filename", "model_path", "quant_format", "quant_group_size"),
    [
        (
            "qwen3-8b_qlora-nvfp4-fp8-sync-smoke.yaml",
            "nvidia/Qwen3-8B-NVFP4",
            "nvfp4",
            16,
        ),
        (
            "qwen3-8b_qlora-block-fp8-sync-smoke.yaml",
            "Qwen/Qwen3-8B-FP8",
            "block_fp8",
            128,
        ),
    ],
)
def test_k3_prequantized_qlora_fp8_sync_configs_pin_matching_format(
    filename,
    model_path,
    quant_format,
    quant_group_size,
):
    config_path = Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs" / filename

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    train_config = config["train"]
    lora_config = config["lora"]

    assert args.model_path == model_path
    assert train_config["enable_fp8_training"] is False
    assert lora_config["enable_lora"] is True
    assert lora_config["enable_qlora"] is True
    assert lora_config["quant_format"] == quant_format
    assert lora_config["quant_group_size"] == quant_group_size
    assert lora_config["lora_rank"] == 8
    assert lora_config["max_lora_rank"] == 8
    assert lora_config["lora_target_modules"] == [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]


def test_qwen3_6_ep8_deepep_fp8_training_config_matches_promotion_gate_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3_6_35b_ep8_deepep_fp8_training.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "deepep"
    assert model_config["train_router"] is False
    assert model_config["deepep_buffer_size_gb"] == pytest.approx(2.0)
    assert model_config["deepep_num_sms"] == 72
    assert model_config["deepep_async_combine"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 8
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"
    assert train_config["sync_inference_method"] == "p2p"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep8_deepep_bf16_training_config_matches_control_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3_6_35b_ep8_deepep_bf16_training.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert args.enable_fp8_training is False
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "deepep"
    assert model_config["train_router"] is False
    assert model_config["deepep_buffer_size_gb"] == pytest.approx(2.0)
    assert model_config["deepep_num_sms"] == 72
    assert model_config["deepep_async_combine"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 8
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is True
    assert train_config["enable_fp8_training"] is False
    assert train_config["load_weights_mode"] == "grouped"
    assert train_config["sync_inference_method"] == "p2p"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep8_deepep_bf16_native_swiglu_config_matches_control_topology():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3_6_35b_ep8_deepep_bf16_training_native_swiglu.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert args.enable_fp8_training is False
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "deepep"
    assert model_config["train_router"] is False
    assert model_config["deepep_buffer_size_gb"] == pytest.approx(2.0)
    assert model_config["deepep_num_sms"] == 72
    assert model_config["deepep_async_combine"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False
    assert model_config["activation_native"] is True

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 8
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is True
    assert train_config["enable_fp8_training"] is False
    assert train_config["load_weights_mode"] == "grouped"
    assert train_config["sync_inference_method"] == "p2p"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep8_quack_alltoall_config_matches_deepep_control_discriminator():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3_6_35b_ep8_quack_alltoall.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert args.enable_fp8_training is False
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 8
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep1_fsdp8_bf16_reference_config_matches_component_reference_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3_6_35b_ep1_fsdp8_bf16_reference.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert args.enable_fp8_training is False
    assert model_config["moe_implementation"] == "eager"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 1
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 8
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.get_total_gpus() == 8
    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep8_deepep_fp8_training_moe_experts_bf16_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep8_deepep_fp8_training_moe_experts_bf16.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "deepep"
    assert model_config["train_router"] is False
    assert model_config["deepep_buffer_size_gb"] == pytest.approx(2.0)
    assert model_config["deepep_num_sms"] == 72
    assert model_config["deepep_async_combine"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 8
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.experts"]
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"
    assert train_config["sync_inference_method"] == "p2p"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_quack_alltoall_bf16_config_matches_reference_probe_contract():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3_6_35b_ep4_quack_alltoall_bf16_training.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert args.enable_fp8_training is False
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_quack_alltoall_bf16_native_swiglu_config_matches_reference_probe_topology():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3_6_35b_ep4_quack_alltoall_bf16_training_native_swiglu.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert args.enable_fp8_training is False
    assert model_config["moe_implementation"] == "quack"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False
    assert model_config["activation_native"] is True

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall_fp8_training.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_block64_smoothquant_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_triton_alltoall_fp8_training_block64_smoothquant_a04_actcorr.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_block_size"] == 64
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_smoothquant_alpha"] == pytest.approx(0.4)
    assert train_config["fp8_training_lm_head_smoothquant_alpha"] == pytest.approx(0.45)
    assert train_config["fp8_training_correction_mode"] == "activation"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_late_attn_head_b32_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_triton_alltoall_fp8_training_late_attn_head_b32.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_module_overrides"] == {
        "model.layers.3[2-9].linear_attn.[qkv]_proj": {"block_size": 32},
        "lm_head": {"block_size": 32},
    }
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_late25_attn_head_b32_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_triton_alltoall_fp8_training_late25_attn_head_b32.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_module_overrides"] == {
        "model.layers.2[5-9].linear_attn.[qkv]_proj": {"block_size": 32},
        "model.layers.3[0-9].linear_attn.[qkv]_proj": {"block_size": 32},
        "lm_head": {"block_size": 32},
    }
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_lm_head_bf16_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3_6_35b_ep4_triton_alltoall_fp8_training_lm_head_bf16.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_exclude_modules"] == ["lm_head"]
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_linear_attn_qkv_bf16_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_triton_alltoall_fp8_training_linear_attn_qkv_bf16.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_exclude_modules"] == [
        "model.layers.*.linear_attn.q_proj",
        "model.layers.*.linear_attn.k_proj",
        "model.layers.*.linear_attn.v_proj",
    ]
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_router_gate_bf16_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_triton_alltoall_fp8_training_router_gate_bf16.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.gate"]
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_triton_fp8_training_moe_experts_bf16_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_triton_alltoall_fp8_training_moe_experts_bf16.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "triton"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.experts"]
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_qwen3_6_ep4_native_fp8_training_moe_experts_bf16_config_matches_diagnostic_contract():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/"
        "qwen3_6_35b_ep4_native_alltoall_fp8_training_moe_experts_bf16.yaml"
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()
    model_config = config["model"]
    train_config = config["train"]

    assert args.model_path == "Qwen/Qwen3.6-35B-A3B"
    assert model_config["moe_implementation"] == "native"
    assert model_config["ep_dispatch"] == "alltoall"
    assert model_config["train_router"] is False
    assert model_config["router_fp32"] is False
    assert model_config["lm_head_fp32"] is False

    assert train_config["data_parallel_mode"] == "fsdp2"
    assert train_config["expert_parallel_size"] == 4
    assert train_config["data_parallel_replicate_size"] == 1
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["tensor_parallel_size"] == 1
    assert train_config["ulysses_parallel_size"] == 1
    assert train_config["ringattn_parallel_size"] == 1
    assert train_config["enable_compile"] is False
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["fp8_training_moe_grouped_backend"] == "triton_grouped"
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.experts"]
    assert train_config["fp8_training_allow_bf16_fallback"] is False
    assert train_config["load_weights_mode"] == "grouped"

    assert args.sample_packing_sequence_len == 8193
    assert args.enable_packing is True


def test_k3_bf16_training_smoke_config_matches_fp8_control_topology():
    config_path = Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3-8b_bf16-training-smoke.yaml"

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is False
    assert train_config["enable_fp8_training"] is False
    assert train_config["data_parallel_shard_size"] == 1
    assert train_config["output_dir"] == "outputs/k3-test-bf16-training-smoke"


def test_server_fp8_training_defaults_to_fail_fast_fallback(tmp_path):
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


def test_k3_fp8_training_lm_head_bf16_config_excludes_only_lm_head():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3-8b_fp8-training-lm-head-bf16-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_exclude_modules"] == ["lm_head"]


def test_k3_fp8_training_attention_bf16_config_excludes_attention_linears():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3-8b_fp8-training-attn-bf16-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.self_attn.*"]
    assert train_config["output_dir"] == "outputs/k3-test-fp8-training-attn-bf16-smoke"


def test_k3_fp8_training_mlp_bf16_config_excludes_mlp_linears():
    config_path = (
        Path(__file__).resolve().parents[2] / "experiments/k3_tests/configs/qwen3-8b_fp8-training-mlp-bf16-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.*"]
    assert train_config["output_dir"] == "outputs/k3-test-fp8-training-mlp-bf16-smoke"


def test_k3_fp8_training_mlp_gate_up_bf16_config_excludes_gate_up_only():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3-8b_fp8-training-mlp-gate-up-bf16-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.gate_up_proj"]
    assert train_config["output_dir"] == "outputs/k3-test-fp8-training-mlp-gate-up-bf16-smoke"


def test_k3_fp8_training_mlp_down_bf16_config_excludes_down_only():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/k3_tests/configs/qwen3-8b_fp8-training-mlp-down-bf16-smoke.yaml"
    )

    args = load_server_arguments(str(config_path))
    train_config = args.to_config_dict()["train"]

    assert args.model_path == "Qwen/Qwen3-8B"
    assert args.enable_fp8_training is True
    assert train_config["enable_fp8_training"] is True
    assert train_config["fp8_training_exclude_modules"] == ["model.layers.*.mlp.down_proj"]
    assert train_config["output_dir"] == "outputs/k3-test-fp8-training-mlp-down-bf16-smoke"
    assert train_config["fp8_training_backward"] == "fp8"
    assert train_config["data_parallel_shard_size"] == 1


def test_load_server_arguments_rejects_broadcast_load_weights_mode(tmp_path):
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


def test_load_server_arguments_rejects_merge_lora_interval_for_server_multi_adapter(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
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
        load_server_arguments(str(config_path))


def test_load_server_arguments_rejects_pipeline_parallel_multi_adapter_lora(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
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
        load_server_arguments(str(config_path))


def test_load_server_arguments_threads_muon_gram_newton_schulz_through_nested_config(tmp_path):
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


def test_load_server_arguments_preserves_runner_compatibility_fields(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "record_routing_weights": False,
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
    assert config["train"]["enable_full_determinism"] is True
    assert config["train"]["cautious_weight_decay"] is True
    assert config["train"]["muon_distributed_mode"] == "full_gradient"
    assert config["train"]["moe_grad_reduce_mode"] == "bf16_a2a_fp32_sum"
    assert config["train"]["optimizer_kwargs"]["muon_distributed_mode"] == "full_gradient"
    assert config["lora"]["lora_export_format"] == "sglang_shared_outer"
