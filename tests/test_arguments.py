import json
import sys

import pytest
import torch
import yaml

import xorl.arguments as arguments_module
from xorl.arguments import Arguments, parse_args


pytestmark = [pytest.mark.cpu]


def test_parse_args_accepts_signsgd_from_yaml(tmp_path, monkeypatch):
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
                    "optimizer": "signsgd",
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

    assert args.train.optimizer == "signsgd"
    assert args.train.optimizer_kwargs == {}
    assert args.train.load_weights_mode == "grouped"


def test_parse_args_accepts_distsignsgd_from_yaml(tmp_path, monkeypatch):
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
                    "optimizer": "distsignsgd",
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

    assert args.train.optimizer == "distsignsgd"
    assert args.train.optimizer_kwargs == {}


def test_parse_args_accepts_multipack_bin_size_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
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

    assert args.data.sample_packing_method == "multipack"
    assert args.data.sample_packing_sequence_len == 4096
    assert args.data.sample_packing_group_size == 64
    assert args.data.sample_packing_bin_size == 16


def test_parse_args_accepts_model_numeric_alignment_flags_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "router_fp32": False,
                    "lm_head_fp32": False,
                    "activation_native": True,
                    "rope_native": True,
                    "attention_cast_bf16": True,
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
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

    args = parse_args(Arguments)

    assert args.model.router_fp32 is False
    assert args.model.lm_head_fp32 is False
    assert args.model.activation_native is True
    assert args.model.rope_native is True
    assert args.model.attention_cast_bf16 is True


def test_parse_args_wires_muon_kwargs_from_yaml(tmp_path, monkeypatch):
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


def test_parse_args_accepts_legacy_ep_and_moe_checkpoint_aliases(tmp_path, monkeypatch):
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
                    "ep_outside": True,
                    "moe_checkpoint_method": "moe_act",
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

    assert args.train.ep_intranode is False
    assert args.train.gradient_checkpointing_method == "recompute_before_dispatch"
    assert args.train.moe_recomputed is False


def test_parse_args_accepts_fsdp_reduce_dtype(tmp_path, monkeypatch):
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
                    "fsdp_reduce_dtype": "bf16",
                    "skip_param_upcast": True,
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

    assert args.train.fsdp_reduce_dtype == "bf16"
    assert args.train.skip_param_upcast is True


def test_parse_args_accepts_omitted_optional_fp8_module_overrides(tmp_path, monkeypatch):
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


def test_parse_args_accepts_fp8_cfg_alias_and_layer_islands(tmp_path, monkeypatch):
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
    assert args.train.fp8_training_num_first_layers_bf16 == 1
    assert args.train.fp8_training_num_last_layers_bf16 == 2
    assert args.train.fp8_training_allow_blackwell is True
    assert args.train.fp8_training_blackwell_validation_artifact == "artifact.json"


def test_parse_args_accepts_nemo_policy_fp8_cfg_alias(tmp_path, monkeypatch):
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
                    "megatron_cfg": {
                        "fp8_cfg": {
                            "enabled": True,
                            "fp8": "e4m3",
                            "fp8_recipe": "blockwise",
                            "fp8_param": False,
                        }
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

    args = parse_args(Arguments)

    assert args.train.enable_fp8_training is True
    assert args.train.fp8_cfg == {"enabled": True, "fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": False}


def test_parse_args_rejects_vllm_fp8_runtime_knobs(tmp_path, monkeypatch):
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


def test_parse_args_rejects_nemo_modelopt_qarl_configs(tmp_path, monkeypatch):
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


def test_parse_args_fp8_training_defaults_to_fail_fast_fallback(tmp_path, monkeypatch):
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


def test_parse_args_accepts_qarl_quant_cfg_from_yaml(tmp_path, monkeypatch):
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


def test_parse_args_rejects_qarl_calibration_knobs_without_data(tmp_path, monkeypatch):
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


@pytest.mark.parametrize(
    ("train_updates", "lora_updates", "expected_error"),
    [
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
    ],
)
def test_parse_args_rejects_full_weight_low_precision_with_adapters(
    tmp_path, monkeypatch, train_updates, lora_updates, expected_error
):
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


def test_parse_args_rejects_qarl_with_fp8_training(tmp_path, monkeypatch):
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


def test_parse_args_rejects_qarl_with_mtp_model_metadata(tmp_path, monkeypatch):
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


def test_parse_args_rejects_qarl_with_mamba_config_json(tmp_path, monkeypatch):
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


def test_parse_args_resolves_auto_checkpoint_before_skip_validation(tmp_path, monkeypatch):
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


@pytest.mark.parametrize(
    "yaml_value,expected",
    [(None, True), (True, True), (False, False)],
)
def test_parse_args_load_optimizer_flag(tmp_path, monkeypatch, yaml_value, expected):
    """load_optimizer defaults True (standard resume) and accepts an explicit False
    for a weights-only resume."""
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
