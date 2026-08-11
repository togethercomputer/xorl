from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.cli.export_quantized import quantize_weight_to_fp8
from xorl.models.layers.moe.experts import MoEExperts
from xorl.qarl import (
    QARLLinear,
    inject_qarl_into_model,
    normalize_qarl_quant_cfg,
    qarl_activation_quant_override,
    summarize_qarl_model,
)
from xorl.qarl.moe_experts import QARLMoEExperts, convert_moe_experts_to_qarl


pytestmark = pytest.mark.cpu


class TinyDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                nn.ModuleDict({"proj": nn.Linear(4, 4), "skip": nn.Linear(4, 4)}),
                nn.ModuleDict({"proj": nn.Linear(4, 4)}),
            ]
        )
        self.lm_head = nn.Linear(4, 8)


class _TinyW4A4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlin_off = QARLLinear(4, 4, quantize_activation=False, quant_format="nvfp4")
        self.qlin_on = QARLLinear(4, 4, quantize_activation=True, quant_format="nvfp4")
        experts = MoEExperts(num_experts=2, hidden_dim=4, intermediate_size=8, moe_implementation="eager")
        self.experts = convert_moe_experts_to_qarl(experts, quantize_weight=True, quantize_activation=False)
        self.plain = nn.Linear(4, 4)


def _dequantize_block_fp8(quantized: torch.Tensor, scale: torch.Tensor, block_size: tuple[int, int]) -> torch.Tensor:
    block_rows, block_cols = block_size
    rows, cols = quantized.shape
    pad_rows = (block_rows - rows % block_rows) % block_rows
    pad_cols = (block_cols - cols % block_cols) % block_cols
    work = quantized.to(torch.float32)
    if pad_rows or pad_cols:
        padded = torch.zeros(rows + pad_rows, cols + pad_cols, dtype=torch.float32)
        padded[:rows, :cols] = work
    else:
        padded = work
    block_row_count = padded.shape[0] // block_rows
    block_col_count = padded.shape[1] // block_cols
    blocks = padded.reshape(block_row_count, block_rows, block_col_count, block_cols).permute(0, 2, 1, 3)
    dequantized = blocks * scale.unsqueeze(-1).unsqueeze(-1)
    dequantized = dequantized.permute(0, 2, 1, 3).reshape(padded.shape)
    return dequantized[:rows, :cols].contiguous()


def test_qarl_dense_fake_quant_injection_and_configuration_policy():
    base = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5) / 8)
    layer = QARLLinear.from_linear(
        base,
        quant_cfg=normalize_qarl_quant_cfg({"format": "fp8_e4m3", "activation": False, "weight_block_size": [2, 2]}),
    )
    x = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 7

    qarl_out = layer(x)
    exported_weight, exported_scale = quantize_weight_to_fp8(base.weight, weight_block_size=(2, 2))
    dequantized_weight = _dequantize_block_fp8(exported_weight, exported_scale, (2, 2))

    torch.testing.assert_close(qarl_out, F.linear(x, dequantized_weight), rtol=0, atol=0)
    torch.testing.assert_close(layer.qarl_weight_scale_inv, exported_scale)

    restored = QARLLinear.from_linear(
        nn.Linear(5, 3, bias=False),
        quant_cfg=normalize_qarl_quant_cfg({"format": "fp8_e4m3", "activation": False, "weight_block_size": [2, 2]}),
    )
    restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(restored.qarl_weight_scale_inv, layer.qarl_weight_scale_inv)

    _assert_inject_qarl_dense_lifecycle_and_model_admission()
    _assert_normalize_qarl_quant_cfg_rejects_static_or_unknown_recipes()
    _assert_nvfp4_weight_only_forward_and_ste_policy()
    _assert_activation_quant_override_lifecycle()


def _assert_inject_qarl_dense_lifecycle_and_model_admission():
    model = TinyDenseModel()

    changed = inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight": True, "activation": False},
        target_modules=["proj", "lm_head"],
        exclude_modules=["model.layers.1.*"],
    )

    assert changed == 2
    assert isinstance(model.model.layers[0]["proj"], QARLLinear)
    assert isinstance(model.model.layers[0]["skip"], nn.Linear)
    assert isinstance(model.model.layers[1]["proj"], nn.Linear)
    assert isinstance(model.lm_head, QARLLinear)
    assert "model.layers.0.proj.weight" in dict(model.named_parameters())
    assert "lm_head.weight" in dict(model.named_parameters())

    model.lm_head(model.model.layers[0]["proj"](torch.randn(1, 4)))
    summary = summarize_qarl_model(model)
    assert summary["enabled"] is True
    assert summary["linear_count"] == 2
    assert "model.layers.0.proj" in summary["linear_names"]
    assert summary["forward_counts"]["model.layers.0.proj"] == 1
    assert summary["forward_counts"]["lm_head"] == 1

    for config in (
        SimpleNamespace(model_type="tiny", text_config=SimpleNamespace(num_nextn_predict_layers=1)),
        SimpleNamespace(model_type="mamba", architectures=["MambaForCausalLM"]),
    ):
        rejected = TinyDenseModel()
        rejected.config = config
        with pytest.raises(ValueError, match="MTP/speculative and Mamba"):
            inject_qarl_into_model(rejected)


def _assert_normalize_qarl_quant_cfg_rejects_static_or_unknown_recipes():
    assert normalize_qarl_quant_cfg("fp8_default_cfg") == {
        "format": "fp8_e4m3",
        "weight": True,
        "activation": True,
        "dynamic": True,
        "weight_block_size": [128, 128],
    }
    assert normalize_qarl_quant_cfg({"quant_method": "FP8", "weight": False, "weight_block_size": [2, 4]}) == {
        "format": "fp8_e4m3",
        "weight": False,
        "activation": True,
        "dynamic": True,
        "weight_block_size": [2, 4],
    }
    with pytest.raises(ValueError, match="Static/calibrated"):
        normalize_qarl_quant_cfg({"format": "fp8_e4m3", "dynamic": False})
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_qarl_quant_cfg("NVFP4_DEFAULT_CFG")
    with pytest.raises(ValueError, match="weight_block_size"):
        normalize_qarl_quant_cfg({"format": "fp8_e4m3", "weight_block_size": [0, 2]})


def _assert_nvfp4_weight_only_forward_and_ste_policy():
    assert normalize_qarl_quant_cfg("nvfp4") == {
        "format": "nvfp4",
        "weight": True,
        "activation": False,
        "dynamic": True,
        "group_size": 16,
    }
    cfg = normalize_qarl_quant_cfg({"format": "nvfp4", "activation": True})
    assert cfg["activation"] is True and cfg["group_size"] == 16
    for group_size in (32, 15, 0):
        with pytest.raises(ValueError, match="group_size"):
            normalize_qarl_quant_cfg({"format": "nvfp4", "group_size": group_size})

    torch.manual_seed(0)
    linear = nn.Linear(64, 128, bias=True)
    quantized = QARLLinear.from_linear(linear, quant_cfg=normalize_qarl_quant_cfg("nvfp4"))
    assert quantized.qarl_group_size == 16
    assert quantized.qarl_quantize_activation is False

    x = torch.randn(8, 64)
    output = quantized(x)
    reference = F.linear(x, linear.weight, linear.bias)
    assert output.shape == reference.shape
    assert not torch.allclose(output, reference)
    output.sum().backward()
    expected_gradient = torch.ones(8, 128).T @ x
    torch.testing.assert_close(quantized.weight.grad, expected_gradient, rtol=1e-4, atol=1e-4)

    quantized.qarl_quantize_weight = False
    torch.testing.assert_close(quantized(x), reference, rtol=0, atol=0)


def _assert_activation_quant_override_lifecycle():
    model = _TinyW4A4Model()
    modules = [module for module in model.modules() if isinstance(module, (QARLLinear, QARLMoEExperts))]
    assert len(modules) == 3
    prior = {id(module): module.qarl_quantize_activation for module in modules}
    assert [prior[id(module)] for module in modules] == [False, True, False]

    for enabled in (True, False):
        with qarl_activation_quant_override(model, enabled=enabled):
            assert all(module.qarl_quantize_activation is enabled for module in modules)
            assert not hasattr(model.plain, "qarl_quantize_activation")
        assert {id(module): module.qarl_quantize_activation for module in modules} == prior

    with pytest.raises(RuntimeError):
        with qarl_activation_quant_override(model, enabled=True):
            raise RuntimeError("boom")
    assert {id(module): module.qarl_quantize_activation for module in modules} == prior

    with qarl_activation_quant_override(model, enabled=True):
        with qarl_activation_quant_override(model, enabled=False):
            assert model.qlin_off.qarl_quantize_activation is False
        assert model.qlin_off.qarl_quantize_activation is True
    assert model.qlin_off.qarl_quantize_activation is False
