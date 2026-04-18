from types import SimpleNamespace

import torch
import torch.nn as nn

from xorl.trainers.model_builder import (
    build_training_model,
    resolve_training_model_dtype,
    should_skip_generic_param_upcast,
)


class TinyModel(nn.Module):
    _no_split_modules = []

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.proj = nn.Linear(4, 4, bias=False, dtype=dtype)


def test_lora_mixed_precision_keeps_base_bf16_and_skips_generic_upcast(monkeypatch):
    captured = {}

    def fake_build_foundation_model(**kwargs):
        return TinyModel(getattr(torch, kwargs["torch_dtype"]))

    def fake_parallelize(model, **kwargs):
        captured["skip_param_upcast"] = kwargs["skip_param_upcast"]
        captured["base_dtype"] = model.proj.weight.dtype
        captured["lora_a_dtype"] = model.proj.lora_A.dtype
        captured["lora_b_dtype"] = model.proj.lora_B.dtype
        return model

    monkeypatch.setattr("xorl.models.build_foundation_model", fake_build_foundation_model)
    monkeypatch.setattr(
        "xorl.distributed.torch_parallelize.build_parallelize_model",
        fake_parallelize,
    )
    monkeypatch.setattr(
        "xorl.trainers.model_builder.helper.print_device_mem_info",
        lambda *args, **kwargs: None,
    )

    result = build_training_model(
        config_path="unused",
        weights_path="unused",
        torch_dtype="bfloat16",
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_target_modules=["proj"],
        enable_mixed_precision=True,
    )

    assert captured["skip_param_upcast"] is True
    assert captured["base_dtype"] == torch.bfloat16
    assert captured["lora_a_dtype"] == torch.float32
    assert captured["lora_b_dtype"] == torch.float32
    assert result.model.proj.weight.requires_grad is False
    assert result.model.proj.lora_A.requires_grad is True
    assert result.model.proj.lora_B.requires_grad is True


def test_lora_dtype_policy_matches_intended_training_modes():
    assert (
        resolve_training_model_dtype(
            enable_lora=True,
            enable_qlora=False,
            enable_mixed_precision=True,
        )
        == "bfloat16"
    )
    assert (
        resolve_training_model_dtype(
            enable_lora=False,
            enable_qlora=True,
            enable_mixed_precision=True,
        )
        == "bfloat16"
    )
    assert (
        resolve_training_model_dtype(
            enable_lora=False,
            enable_qlora=False,
            enable_mixed_precision=True,
        )
        == "float32"
    )
    assert (
        should_skip_generic_param_upcast(
            enable_lora=True,
            enable_qlora=False,
        )
        is True
    )
    assert (
        should_skip_generic_param_upcast(
            enable_lora=False,
            enable_qlora=True,
        )
        is True
    )
    assert (
        should_skip_generic_param_upcast(
            enable_lora=False,
            enable_qlora=False,
        )
        is False
    )
