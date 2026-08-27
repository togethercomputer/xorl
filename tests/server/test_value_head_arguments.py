"""Validation of the enable_value_head server-argument surface."""

import pytest

from xorl.server.server_arguments import ServerArguments


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _args(**overrides):
    base = {"model_path": "Qwen/Qwen3-8B"}
    base.update(overrides)
    return ServerArguments(**base)


def test_value_head_defaults_off():
    args = _args(enable_lora=True)
    assert args.enable_value_head is False
    assert args.to_config_dict()["lora"]["enable_value_head"] is False


def test_value_head_accepts_plain_lora():
    args = _args(enable_lora=True, enable_value_head=True)
    assert args.to_config_dict()["lora"]["enable_value_head"] is True


def test_value_head_requires_lora():
    with pytest.raises(ValueError, match="plain LoRA"):
        _args(enable_value_head=True)


def test_value_head_rejects_qlora():
    with pytest.raises(ValueError, match="plain LoRA"):
        _args(enable_lora=True, enable_qlora=True, enable_value_head=True)


def test_value_head_rejects_pipeline_parallel():
    with pytest.raises(ValueError, match="pipeline parallelism"):
        _args(enable_lora=True, enable_value_head=True, pipeline_parallel_size=2)


def test_value_head_rejects_fsdp_sharded_lm_head_loss():
    with pytest.raises(ValueError, match="fsdp_sharded_lm_head_loss"):
        _args(enable_lora=True, enable_value_head=True, fsdp_sharded_lm_head_loss=True)


def test_value_head_rejects_explicit_lm_head_target():
    with pytest.raises(ValueError, match="lm_head excluded"):
        _args(
            enable_lora=True,
            enable_value_head=True,
            lora_target_modules=["q_proj", "lm_head"],
        )
