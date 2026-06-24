from types import SimpleNamespace

import pytest
import torch.nn as nn
import torch.nn.functional as F

from xorl.models.layers.moe.experts import MoEExperts
from xorl.trainers.model_builder import build_training_model


pytestmark = [pytest.mark.cpu]


class TinyDenseMoEModel(nn.Module):
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.proj = nn.Linear(16, 16)
        self.gate = nn.Linear(16, 2)
        self.experts = MoEExperts(num_experts=2, hidden_dim=16, intermediate_size=32, moe_implementation="triton")
        self.lm_head = nn.Linear(16, 8)


class TinyDenseOnlyModel(nn.Module):
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.proj = nn.Linear(16, 16)
        self.gate = nn.Linear(16, 2)
        self.lm_head = nn.Linear(16, 8)


class TinyQARLCalibrationModel(nn.Module):
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.embed_tokens = nn.Embedding(8, 16)
        self.proj = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 8)

    def forward(self, input_ids):
        hidden = F.silu(self.proj(self.embed_tokens(input_ids)))
        return self.lm_head(hidden)


def test_build_training_model_applies_full_model_fp8_training(monkeypatch):
    captured = {}

    def fake_build_foundation_model(**_kwargs):
        return TinyDenseMoEModel()

    def fake_parallelize(model, **kwargs):
        captured["skip_param_upcast"] = kwargs["skip_param_upcast"]
        captured["proj_type"] = type(model.proj)
        captured["gate_type"] = type(model.gate)
        captured["lm_head_type"] = type(model.lm_head)
        captured["moe_impl"] = model.experts.moe_implementation
        captured["moe_fp8_enabled"] = model.experts.fp8_training_enabled
        captured["moe_fp8_backend"] = model.experts.fp8_training_grouped_backend
        captured["moe_fp8_block_size"] = model.experts.fp8_training_block_size
        captured["proj_smoothquant_alpha"] = model.proj.fp8_smoothquant_alpha
        captured["proj_block_size"] = model.proj.fp8_block_size
        captured["lm_head_smoothquant_alpha"] = model.lm_head.fp8_smoothquant_alpha
        captured["proj_activation_amax_scale"] = model.proj.fp8_activation_amax_scale
        captured["proj_weight_amax_scale"] = model.proj.fp8_weight_amax_scale
        captured["proj_correction_mode"] = model.proj.fp8_correction_mode
        captured["lm_head_output_dtype"] = model.lm_head.fp8_output_dtype
        captured["proj_allow_bf16_fallback"] = model.proj.fp8_allow_bf16_fallback
        captured["gate_allow_bf16_fallback"] = model.gate.fp8_allow_bf16_fallback
        captured["lm_head_allow_bf16_fallback"] = model.lm_head.fp8_allow_bf16_fallback
        return model

    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", fake_build_foundation_model)
    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    result = build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_fp8_training=True,
        fp8_training_block_size=64,
        fp8_training_smoothquant_alpha=0.5,
        fp8_training_lm_head_smoothquant_alpha=0.4,
        fp8_training_activation_amax_scale=0.875,
        fp8_training_weight_amax_scale=1.125,
        fp8_training_correction_mode="activation2",
        fp8_training_module_overrides={
            "proj": {"block_size": 32, "smoothquant_alpha": 0.25, "correction_mode": "full"}
        },
        fp8_training_moe_grouped_backend="triton_grouped",
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )

    assert captured["skip_param_upcast"] is False
    assert captured["proj_type"].__name__ == "FP8Linear"
    assert captured["gate_type"].__name__ == "FP8Linear"
    assert captured["lm_head_type"].__name__ == "FP8Linear"
    assert captured["moe_impl"] == "quack"
    assert captured["moe_fp8_enabled"] is True
    assert captured["moe_fp8_backend"] == "triton_grouped"
    assert captured["moe_fp8_block_size"] == 64
    assert captured["proj_block_size"] == 32
    assert captured["proj_smoothquant_alpha"] == 0.25
    assert captured["lm_head_smoothquant_alpha"] == 0.4
    assert captured["proj_activation_amax_scale"] == 0.875
    assert captured["proj_weight_amax_scale"] == 1.125
    assert captured["proj_correction_mode"] == "full"
    assert captured["lm_head_output_dtype"] == "float32"
    assert captured["proj_allow_bf16_fallback"] is False
    assert captured["gate_allow_bf16_fallback"] is False
    assert captured["lm_head_allow_bf16_fallback"] is False
    assert type(result.model.proj).__name__ == "FP8Linear"
    assert type(result.model.gate).__name__ == "FP8Linear"
    assert type(result.model.lm_head).__name__ == "FP8Linear"
    assert all(param.requires_grad for param in result.model.parameters())


def test_build_training_model_rejects_fp8_with_adapters(monkeypatch):
    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyDenseMoEModel())
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="enable_fp8_training is a full-weight mode"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            enable_fp8_training=True,
            enable_lora=True,
            lora_target_modules=["proj"],
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
        )


def test_build_training_model_applies_dense_qarl_fake_quant(monkeypatch):
    captured = {}

    def fake_parallelize(model, **kwargs):
        captured["skip_param_upcast"] = kwargs["skip_param_upcast"]
        captured["proj_type"] = type(model.proj).__name__
        captured["gate_type"] = type(model.gate).__name__
        captured["lm_head_type"] = type(model.lm_head).__name__
        captured["proj_quantize_activation"] = model.proj.qarl_quantize_activation
        return model

    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyDenseOnlyModel())
    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    result = build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_qarl=True,
        qarl_quant_cfg={"format": "fp8_e4m3", "activation": False},
        qarl_target_modules=["proj", "lm_head"],
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )

    assert captured["skip_param_upcast"] is False
    assert captured["proj_type"] == "QARLLinear"
    assert captured["gate_type"] == "Linear"
    assert captured["lm_head_type"] == "QARLLinear"
    assert captured["proj_quantize_activation"] is False
    assert type(result.model.proj).__name__ == "QARLLinear"
    assert all(param.requires_grad for param in result.model.parameters())


def test_build_training_model_runs_qarl_calibration_before_parallelize(monkeypatch, tmp_path):
    calibration_path = tmp_path / "calib.json"
    calibration_path.write_text('{"input_ids": [[1, 2, 3, 4], [4, 3, 2, 1]]}\n', encoding="utf-8")
    captured = {}

    def fake_parallelize(model, **_kwargs):
        captured["summary"] = model._qarl_calibration_summary
        captured["proj_forward_count"] = int(model.proj.qarl_forward_count.item())
        captured["proj_weight_scale_inv"] = model.proj.qarl_weight_scale_inv.detach().clone()
        return model

    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyQARLCalibrationModel())
    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    result = build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_qarl=True,
        qarl_quant_cfg={"format": "fp8_e4m3", "weight_block_size": [4, 4]},
        qarl_target_modules=["proj", "lm_head"],
        qarl_calib_data=str(calibration_path),
        qarl_calib_size=1,
        qarl_quant_sequence_length=3,
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )

    assert captured["summary"]["calibration_batches"] == 1
    assert captured["summary"]["calibration_samples"] == 1
    assert captured["summary"]["forward_counts"]["proj"] == 1
    assert captured["proj_forward_count"] == 1
    assert captured["proj_weight_scale_inv"].shape == (4, 4)
    assert captured["proj_weight_scale_inv"].abs().max().item() > 0
    assert result.model._qarl_calibration_summary["linear_count"] == 2


def test_build_training_model_rejects_qarl_with_adapters_or_fp8(monkeypatch):
    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyDenseOnlyModel())
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="enable_qarl is a full-weight mode"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            enable_qarl=True,
            enable_lora=True,
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
        )

    with pytest.raises(ValueError, match="cannot be combined with enable_fp8_training"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            enable_qarl=True,
            enable_fp8_training=True,
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
        )


def test_build_training_model_rejects_qarl_for_moe(monkeypatch):
    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyDenseMoEModel())
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="dense full-weight models only"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            enable_qarl=True,
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
        )


def test_build_training_model_allows_full_fp8_lm_head_with_tensor_parallel(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "xorl.trainers.model_builder.get_parallel_state",
        lambda: SimpleNamespace(tp_enabled=True, pp_enabled=False),
    )
    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyDenseMoEModel())
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    def fake_parallelize(model, **_kwargs):
        captured["proj_type"] = type(model.proj).__name__
        captured["lm_head_type"] = type(model.lm_head).__name__
        return model

    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)

    build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_fp8_training=True,
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )

    assert captured["proj_type"] == "FP8Linear"
    assert captured["lm_head_type"] == "FP8Linear"


def test_build_training_model_allows_tensor_parallel_when_lm_head_excluded_from_fp8(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "xorl.trainers.model_builder.get_parallel_state",
        lambda: SimpleNamespace(tp_enabled=True, pp_enabled=False),
    )
    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyDenseMoEModel())
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    def fake_parallelize(model, **_kwargs):
        captured["lm_head_type"] = type(model.lm_head).__name__
        captured["proj_type"] = type(model.proj).__name__
        return model

    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)

    build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_fp8_training=True,
        fp8_training_exclude_modules=["lm_head"],
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )

    assert captured["proj_type"] == "FP8Linear"
    assert captured["lm_head_type"] == "Linear"
