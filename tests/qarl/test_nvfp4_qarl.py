"""CPU tests for NVFP4 weight-only QARL fake-quant (generalized from FP8)."""

import pytest
import torch
import torch.nn as nn

from xorl.qarl.fake_quant import (
    QARLLinear,
    inject_qarl_into_model,
    normalize_qarl_quant_cfg,
)


pytestmark = pytest.mark.cpu


class TestNormalizeNVFP4:
    def test_string_alias(self):
        assert normalize_qarl_quant_cfg("nvfp4") == {
            "format": "nvfp4",
            "weight": True,
            "activation": False,  # weight-only W4 default
            "dynamic": True,
            "group_size": 16,
        }

    def test_dict_defaults_weight_only(self):
        cfg = normalize_qarl_quant_cfg({"format": "nvfp4"})
        assert cfg["format"] == "nvfp4"
        assert cfg["weight"] is True
        assert cfg["activation"] is False
        assert cfg["group_size"] == 16

    def test_dict_activation_with_default_group_size(self):
        cfg = normalize_qarl_quant_cfg({"format": "nvfp4", "group_size": 16, "activation": True})
        assert cfg["group_size"] == 16
        assert cfg["activation"] is True

    def test_rejects_non_16_group_size(self):
        # NVFP4 is a block-16 format; export/serving assume 16, so a non-16 training
        # group_size would silently mismatch the served model -> fail loud.
        with pytest.raises(ValueError, match="group_size"):
            normalize_qarl_quant_cfg({"format": "nvfp4", "group_size": 32})

    def test_rejects_odd_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            normalize_qarl_quant_cfg({"format": "nvfp4", "group_size": 15})

    def test_rejects_zero_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            normalize_qarl_quant_cfg({"format": "nvfp4", "group_size": 0})

    def test_fp8_path_unchanged(self):
        # Regression: the FP8 default path must be untouched by the NVFP4 addition.
        assert normalize_qarl_quant_cfg(None) == {
            "format": "fp8_e4m3",
            "weight": True,
            "activation": True,
            "dynamic": True,
            "weight_block_size": [128, 128],
        }


class TestNVFP4QARLLinear:
    def _make(self, cfg):
        torch.manual_seed(0)
        lin = nn.Linear(64, 128, bias=True)
        return QARLLinear.from_linear(lin, quant_cfg=normalize_qarl_quant_cfg(cfg)), lin

    def test_format_and_group_size(self):
        q, _ = self._make("nvfp4")
        assert q.qarl_format == "nvfp4"
        assert q.qarl_group_size == 16
        assert q.qarl_quantize_activation is False

    def test_forward_is_lossy_vs_plain_linear(self):
        q, lin = self._make("nvfp4")
        x = torch.randn(8, 64)
        out_q = q(x)
        out_ref = torch.nn.functional.linear(x, lin.weight, lin.bias)
        assert out_q.shape == out_ref.shape
        assert not torch.allclose(out_q, out_ref)  # weights were fake-quantized

    def test_ste_identity_gradient_through_weight(self):
        q, _ = self._make("nvfp4")
        x = torch.randn(8, 64)
        q(x).sum().backward()
        # STE: d(sum(x @ w_fq.T))/dw == column-sum of x broadcast over output rows.
        expected = torch.ones(8, 128).T @ x
        assert q.weight.grad is not None
        torch.testing.assert_close(q.weight.grad, expected, rtol=1e-4, atol=1e-4)

    def test_weight_only_leaves_activations_untouched(self):
        # With activation off, toggling weight quant is the only source of change.
        q, lin = self._make("nvfp4")
        x = torch.randn(4, 64)
        q.qarl_quantize_weight = False
        out = q(x)
        ref = torch.nn.functional.linear(x, lin.weight, lin.bias)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


class TestInjectNVFP4Dense:
    def test_inject_wraps_linears(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        n = inject_qarl_into_model(model, quant_cfg=normalize_qarl_quant_cfg("nvfp4"))
        assert n == 2
        wrapped = [m for m in model.modules() if isinstance(m, QARLLinear)]
        assert len(wrapped) == 2
        assert all(m.qarl_format == "nvfp4" for m in wrapped)
        # forward runs end-to-end with fake-quant in the loop
        out = model(torch.randn(8, 64))
        assert out.shape == (8, 64)
