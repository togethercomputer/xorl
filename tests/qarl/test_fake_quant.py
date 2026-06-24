from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.cli.export_quantized import quantize_weight_to_fp8
from xorl.models.layers.moe.experts import MoEExperts
from xorl.qarl import QARLLinear, inject_qarl_into_model, normalize_qarl_quant_cfg, summarize_qarl_model


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


class TinyMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny", num_experts=2)
        self.proj = nn.Linear(4, 4)
        self.experts = MoEExperts(num_experts=2, hidden_dim=4, intermediate_size=8)


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


def test_qarl_linear_fake_quant_uses_ste_and_persists_state():
    base = nn.Linear(4, 3)
    layer = QARLLinear.from_linear(base, quant_cfg=normalize_qarl_quant_cfg("FP8_DEFAULT_CFG"))
    x = torch.randn(2, 4, requires_grad=True)

    y = layer(x).sum()
    y.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.qarl_forward_count.item() == 1
    assert layer.qarl_input_amax.item() > 0
    assert layer.qarl_weight_amax.item() > 0
    assert layer.qarl_input_scale_inv.item() > 0
    assert layer.qarl_weight_scale_inv.shape == (1, 1)
    assert layer.qarl_weight_scale_inv.item() > 0

    state = layer.state_dict()
    restored = QARLLinear(4, 3)
    restored.load_state_dict(state)
    assert restored.qarl_forward_count.item() == 1
    torch.testing.assert_close(restored.qarl_input_amax, layer.qarl_input_amax)
    torch.testing.assert_close(restored.qarl_weight_scale_inv, layer.qarl_weight_scale_inv)
    torch.testing.assert_close(restored.weight, layer.weight)


def test_qarl_weight_fake_quant_matches_folded_block_fp8_export():
    base = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5) / 8)
    layer = QARLLinear.from_linear(
        base,
        quant_cfg=normalize_qarl_quant_cfg(
            {"format": "fp8_e4m3", "activation": False, "weight_block_size": [2, 2]}
        ),
    )
    x = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 7

    qarl_out = layer(x)
    exported_weight, exported_scale = quantize_weight_to_fp8(base.weight, weight_block_size=(2, 2))
    dequantized_weight = _dequantize_block_fp8(exported_weight, exported_scale, (2, 2))

    torch.testing.assert_close(qarl_out, F.linear(x, dequantized_weight), rtol=0, atol=0)
    torch.testing.assert_close(layer.qarl_weight_scale_inv, exported_scale)


def test_inject_qarl_wraps_dense_linears_and_preserves_parameter_names():
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


def test_inject_qarl_rejects_moe_models():
    with pytest.raises(ValueError, match="dense full-weight models only"):
        inject_qarl_into_model(TinyMoEModel())


def test_inject_qarl_rejects_mtp_model_configs():
    model = TinyDenseModel()
    model.config = SimpleNamespace(model_type="tiny", text_config=SimpleNamespace(num_nextn_predict_layers=1))

    with pytest.raises(ValueError, match="MTP/speculative and Mamba"):
        inject_qarl_into_model(model)


def test_inject_qarl_rejects_mamba_model_configs():
    model = TinyDenseModel()
    model.config = SimpleNamespace(model_type="mamba", architectures=["MambaForCausalLM"])

    with pytest.raises(ValueError, match="MTP/speculative and Mamba"):
        inject_qarl_into_model(model)


def test_normalize_qarl_quant_cfg_rejects_static_or_unknown_recipes():
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
