import json
import sys

import pytest
import torch
import yaml

import xorl.arguments as arguments_module
from xorl.arguments import Arguments, parse_args


pytestmark = [pytest.mark.cpu]


def test_parse_args_optimizer_packing_and_numeric_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    for optimizer in ("signsgd", "distsignsgd"):
        config_path = tmp_path / f"{optimizer}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "model_path": "Qwen/Qwen3-8B",
                        "router_fp32": False,
                        "lm_head_fp32": False,
                        "rmsnorm_mode": "sglang",
                        "activation_native": True,
                        "rope_native": True,
                        "attention_cast_bf16": True,
                    },
                    "data": {
                        "datasets": [{"path": "dummy", "type": "tokenized"}],
                        "sample_packing_method": "multipack",
                        "sample_packing_sequence_len": 4096,
                        "sample_packing_group_size": 64,
                        "sample_packing_bin_size": 16,
                    },
                    "train": {
                        "init_device": "meta",
                        "output_dir": str(tmp_path / "outputs"),
                        "optimizer": optimizer,
                        "fsdp_reduce_dtype": "bf16",
                        "skip_param_upcast": True,
                        "use_wandb": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

        args = parse_args(Arguments)

        assert args.train.optimizer == optimizer
        assert args.train.optimizer_kwargs == {}
        assert args.train.load_weights_mode == "grouped"
        assert args.train.fsdp_reduce_dtype == "bf16"
        assert args.train.skip_param_upcast is True
        assert args.data.sample_packing_method == "multipack"
        assert args.data.sample_packing_sequence_len == 4096
        assert args.data.sample_packing_group_size == 64
        assert args.data.sample_packing_bin_size == 16
        assert args.model.router_fp32 is False
        assert args.model.lm_head_fp32 is False
        assert args.model.rmsnorm_mode == "sglang"
        assert args.model.activation_native is True
        assert args.model.rope_native is True
        assert args.model.attention_cast_bf16 is True
        assert args.model.deepep_native_exact is False
        assert args.lora.lora_serving_mode is None

    muon_root = tmp_path / "muon"
    muon_root.mkdir()
    _assert_parse_args_wires_muon_kwargs(muon_root, monkeypatch)
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    with monkeypatch.context() as checkpoint_patch:
        _assert_parse_args_checkpoint_policy(checkpoint_root, checkpoint_patch)
    low_precision_root = tmp_path / "low-precision"
    low_precision_root.mkdir()
    with monkeypatch.context() as low_precision_patch:
        _assert_parse_args_low_precision_configuration_policy(low_precision_root, low_precision_patch)


def _assert_parse_args_wires_muon_kwargs(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "optimizer": "muon",
                    "optimizer_dtype": "bf16",
                    "muon_ns_algorithm": "gram_newton_schulz",
                    "muon_ns_use_quack_kernels": False,
                    "muon_gram_ns_num_restarts": 2,
                    "muon_gram_ns_restart_iterations": [2],
                    "muon_grouped_gram_ns_fp32_byte_limit": 23,
                    "muon_fallback_optimizer": "sgd",
                    "muon_grad_dtype": "fp32",
                    "muon_update_dtype": "fp32",
                    "muon_force_momentum_path": True,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.optimizer == "muon"
    assert args.train.optimizer_kwargs["muon_ns_algorithm"] == "gram_newton_schulz"
    assert args.train.optimizer_kwargs["muon_ns_use_quack_kernels"] is False
    assert args.train.optimizer_kwargs["muon_gram_ns_num_restarts"] == 2
    assert args.train.optimizer_kwargs["muon_gram_ns_restart_iterations"] == [2]
    assert args.train.optimizer_kwargs["muon_grouped_gram_ns_fp32_byte_limit"] == 23
    assert args.train.optimizer_kwargs["muon_fallback_optimizer"] == "sgd"
    assert args.train.optimizer_kwargs["muon_momentum_dtype"] is torch.bfloat16
    assert args.train.optimizer_kwargs["muon_grad_dtype"] is torch.float32
    assert args.train.optimizer_kwargs["muon_update_dtype"] is torch.float32
    assert args.train.optimizer_kwargs["muon_force_momentum_path"] is True


def _assert_parse_args_checkpoint_policy(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "gradient_checkpointing_method": "recompute_before_dispatch",
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.gradient_checkpointing_method == "recompute_before_dispatch"
    assert args.train.moe_recomputed is False

    native_config_path = tmp_path / "native-exact-default.yaml"
    native_config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "deepep_native_exact": True,
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "native-outputs"),
                    "expert_parallel_size": 2,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train.py", str(native_config_path)])
    native_args = parse_args(Arguments)

    assert native_args.train.gradient_checkpointing_method == "recompute_before_dispatch"
    assert native_args.train.moe_recomputed is False

    auto_root = tmp_path / "auto-checkpoint"
    auto_root.mkdir()
    _assert_parse_args_resolves_auto_checkpoint_before_validation(auto_root, monkeypatch)
    optimizer_root = tmp_path / "load-optimizer"
    optimizer_root.mkdir()
    _assert_parse_args_load_optimizer_flag(optimizer_root, monkeypatch)


def test_native_exact_training_rejects_ep1(tmp_path, monkeypatch):
    config_path = tmp_path / "native-exact-ep1.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "deepep_native_exact": True,
                },
                "data": {"datasets": [{"path": "dummy", "type": "tokenized"}]},
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "expert_parallel_size": 1,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="expert_parallel_size > 1"):
        parse_args(Arguments)


def _assert_parse_args_low_precision_configuration_policy(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_fp8_training": True,
                    "fp8_training_backward": "fp8",
                    "fp8_training_allow_bf16_fallback": False,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.enable_fp8_training is True
    assert args.train.fp8_training_module_overrides is None
    assert args.train.fp8_training_allow_bf16_fallback is False

    vllm_root = tmp_path / "vllm-rejections"
    vllm_root.mkdir()
    _assert_parse_args_rejects_vllm_fp8_runtime_knobs(vllm_root, monkeypatch)
    default_root = tmp_path / "fallback-default"
    default_root.mkdir()
    _assert_parse_args_fp8_defaults_to_fail_fast_fallback(default_root, monkeypatch)
    low_precision_root = tmp_path / "low-precision-modes"
    low_precision_root.mkdir()
    _assert_parse_args_low_precision_mode_policy(low_precision_root, monkeypatch)


def _assert_parse_args_rejects_vllm_fp8_runtime_knobs(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "generation": {
                    "vllm_cfg": {
                        "precision": "fp8",
                    }
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="vLLM FP8 receiver"):
        parse_args(Arguments)

    quantization_path = tmp_path / "config_quantization.yaml"
    quantization_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "generation": {
                    "vllm_cfg": {
                        "quantization": "fp8",
                    }
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train.py", str(quantization_path)])

    with pytest.raises(ValueError, match="vLLM FP8 receiver"):
        parse_args(Arguments)

    vllm_island_path = tmp_path / "config_vllm_island.yaml"
    vllm_island_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "generation": {
                    "vllm_cfg": {
                        "num_first_layers_in_bf16": 1,
                    }
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train.py", str(vllm_island_path)])

    with pytest.raises(ValueError, match="num_first_layers_in_bf16"):
        parse_args(Arguments)

    kv_cache_path = tmp_path / "config_kv.yaml"
    kv_cache_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "generation": {
                    "vllm_cfg": {
                        "kv_cache_dtype": "fp8_e4m3",
                    }
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train.py", str(kv_cache_path)])

    with pytest.raises(ValueError, match="receiver_kv_cache_dtype"):
        parse_args(Arguments)

    pow2_path = tmp_path / "config_pow2.yaml"
    pow2_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "generation": {
                    "vllm_cfg": {
                        "pow2_activation_scaling_factors": True,
                    }
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train.py", str(pow2_path)])

    with pytest.raises(ValueError, match="pow2_activation_scaling_factors"):
        parse_args(Arguments)


def _assert_parse_args_low_precision_mode_policy(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "policy": {
                    "quant_cfg": "FP8_DEFAULT_CFG",
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="ModelOpt QARL"):
        parse_args(Arguments)

    qlora_root = tmp_path / "block-fp8-qlora"
    qlora_root.mkdir()
    _assert_parse_args_accepts_glm52_block_fp8_qlora_mode(qlora_root, monkeypatch)
    qarl_root = tmp_path / "qarl"
    qarl_root.mkdir()
    _assert_parse_args_accepts_qarl_quant_cfg(qarl_root, monkeypatch)
    rejection_root = tmp_path / "rejections"
    rejection_root.mkdir()
    _assert_parse_args_rejects_incompatible_low_precision_configs(rejection_root, monkeypatch)


def _assert_parse_args_fp8_defaults_to_fail_fast_fallback(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_fp8_training": True,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.enable_fp8_training is True
    assert args.train.fp8_training_allow_bf16_fallback is False


def _assert_parse_args_accepts_glm52_block_fp8_qlora_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "zai-org/GLM-5.2-FP8",
                    "moe_implementation": "triton",
                    "ep_dispatch": "deepep",
                    "freeze_router": True,
                    "merge_qkv": True,
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "use_wandb": False,
                },
                "lora": {
                    "enable_lora": True,
                    "enable_qlora": True,
                    "block_fp8_qlora_training": True,
                    "quant_format": "block_fp8",
                    "quant_group_size": 128,
                    "moe_hybrid_shared_lora": True,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.lora.block_fp8_qlora_training is True
    assert args.lora.enable_lora is True
    assert args.lora.enable_qlora is True
    assert args.lora.quant_format == "block_fp8"
    assert args.lora.quant_group_size == 128
    assert args.lora.moe_hybrid_shared_lora is True


def _assert_parse_args_accepts_qarl_quant_cfg(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_qarl": True,
                    "qarl_quant_cfg": {"format": "fp8_e4m3", "activation": False},
                    "qarl_sync_format": "fp8",
                    "qarl_calib_data": str(tmp_path / "calib.json"),
                    "qarl_calib_size": 8,
                    "qarl_quant_sequence_length": 32,
                    "qarl_target_modules": ["q_proj", "k_proj"],
                    "qarl_exclude_modules": ["lm_head"],
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.enable_qarl is True
    assert args.train.qarl_quant_cfg == {
        "format": "fp8_e4m3",
        "weight": True,
        "activation": False,
        "dynamic": True,
        "weight_block_size": [128, 128],
    }
    assert args.train.qarl_sync_format == "fp8"
    assert args.train.qarl_calib_data == str(tmp_path / "calib.json")
    assert args.train.qarl_calib_size == 8
    assert args.train.qarl_quant_sequence_length == 32
    assert args.train.qarl_target_modules == ["q_proj", "k_proj"]
    assert args.train.qarl_exclude_modules == ["lm_head"]


def _assert_qarl_calibration_knobs_require_data(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_qarl": True,
                    "qarl_calib_size": 4,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="require qarl_calib_data"):
        parse_args(Arguments)


def _assert_parse_args_rejects_incompatible_low_precision_configs(tmp_path, monkeypatch):
    cases = (
        (
            {"enable_fp8_training": True},
            {"enable_lora": True},
            "enable_fp8_training is a full-weight mode",
        ),
        (
            {"enable_fp8_training": True},
            {"enable_qlora": True},
            "enable_fp8_training is a full-weight mode",
        ),
        (
            {"enable_qarl": True},
            {"enable_lora": True},
            "enable_qarl is a full-weight mode",
        ),
        (
            {"enable_qarl": True},
            {"enable_qlora": True},
            "enable_qarl is a full-weight mode",
        ),
    )
    for train_updates, lora_updates, expected_error in cases:
        config_path = tmp_path / "config.yaml"
        train_config = {
            "init_device": "meta",
            "output_dir": str(tmp_path / "outputs"),
            "use_wandb": False,
        }
        train_config.update(train_updates)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "model_path": "Qwen/Qwen3-8B",
                    },
                    "data": {
                        "datasets": [{"path": "dummy", "type": "tokenized"}],
                    },
                    "train": train_config,
                    "lora": lora_updates,
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

        with pytest.raises(ValueError, match=expected_error):
            parse_args(Arguments)

    _assert_qarl_calibration_knobs_require_data(tmp_path, monkeypatch)
    _assert_qarl_and_fp8_are_mutually_exclusive(tmp_path, monkeypatch)
    _assert_qarl_rejects_mtp_metadata(tmp_path, monkeypatch)
    _assert_qarl_rejects_mamba_config(tmp_path, monkeypatch)


def _assert_qarl_and_fp8_are_mutually_exclusive(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_qarl": True,
                    "enable_fp8_training": True,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="enable_qarl cannot be combined with enable_fp8_training"):
        parse_args(Arguments)


def _assert_qarl_rejects_mtp_metadata(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "foundation": {"text_config": {"num_nextn_predict_layers": 1}},
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_qarl": True,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="MTP/speculative and Mamba"):
        parse_args(Arguments)


def _assert_qarl_rejects_mamba_config(tmp_path, monkeypatch):
    model_dir = tmp_path / "mamba-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mamba", "architectures": ["MambaForCausalLM"]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "config_path": str(model_dir),
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "enable_qarl": True,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    with pytest.raises(ValueError, match="config_json.model_type=mamba"):
        parse_args(Arguments)


def _assert_parse_args_resolves_auto_checkpoint_before_validation(tmp_path, monkeypatch):
    resolved_checkpoint = str(tmp_path / "outputs" / "checkpoints" / "global_step_10")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "load_weights_mode": "skip",
                    "load_checkpoint_path": "auto",
                    "repo_commit": "test-commit",
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])
    monkeypatch.setattr(
        arguments_module,
        "get_checkpoint_path",
        lambda output_dir, is_local_rank0, ckpt_manager: resolved_checkpoint,
    )

    args = parse_args(Arguments)

    assert args.train.load_checkpoint_path == resolved_checkpoint


def _assert_parse_args_load_optimizer_flag(tmp_path, monkeypatch):
    """load_optimizer defaults True (standard resume) and accepts an explicit False
    for a weights-only resume."""
    for yaml_value, expected in ((None, True), (True, True), (False, False)):
        train_cfg = {
            "init_device": "meta",
            "output_dir": str(tmp_path / "outputs"),
            "use_wandb": False,
        }
        if yaml_value is not None:
            train_cfg["load_optimizer"] = yaml_value
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_path": "Qwen/Qwen3-8B"},
                    "data": {"datasets": [{"path": "dummy", "type": "tokenized"}]},
                    "train": train_cfg,
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

        args = parse_args(Arguments)

        assert args.train.load_optimizer is expected
