"""CPU tests for NVFP4 weight-only QARL MoE expert fake-quant.

Covers the in-place class-swap (parameter/identity preservation), the STE
fake-quant of the two 3D GKN expert tensors, the __dict__-shadow mechanism that
feeds the inherited forward, the eager backend path end-to-end on CPU, and the
inject_qarl_into_model MoE wiring (nvfp4 allowed, fp8 rejected).
"""

import pytest
import torch
import torch.nn as nn

from xorl.models.layers.moe.experts import MoEExperts
from xorl.qarl.fake_quant import QARLLinear, inject_qarl_into_model, normalize_qarl_quant_cfg
from xorl.qarl.moe_experts import QARLMoEExperts, convert_moe_experts_to_qarl


pytestmark = pytest.mark.cpu


def _make_experts(num_experts=4, hidden_dim=32, intermediate_size=16, impl="eager"):
    torch.manual_seed(0)
    e = MoEExperts(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        moe_implementation=impl,
    )
    with torch.no_grad():
        e.gate_up_proj.normal_()
        e.down_proj.normal_()
    return e


class TestClassSwap:
    def test_swap_preserves_identity_and_attrs(self):
        e = _make_experts()
        gup, down = e.gate_up_proj, e.down_proj
        out = convert_moe_experts_to_qarl(e, group_size=16)
        assert out is e
        assert isinstance(e, QARLMoEExperts)
        assert isinstance(e, MoEExperts)
        # Same Parameter objects (optimizer state / FSDP2 / DCP keys preserved).
        assert e.gate_up_proj is gup
        assert e.down_proj is down
        assert e.qarl_group_size == 16
        assert e.qarl_format == "nvfp4"
        assert e.moe_implementation == "eager"  # backend setting preserved

    def test_idempotent(self):
        e = convert_moe_experts_to_qarl(_make_experts())
        assert convert_moe_experts_to_qarl(e) is e

    def test_rejects_non_experts(self):
        with pytest.raises(TypeError):
            convert_moe_experts_to_qarl(nn.Linear(8, 8))


class TestFakeQuantAndShadow:
    def test_fake_quant_weights_lossy_with_ste(self):
        e = convert_moe_experts_to_qarl(_make_experts())
        gup_fq, down_fq = e._qarl_fake_quant_weights()
        assert gup_fq.shape == e.gate_up_proj.shape
        assert down_fq.shape == e.down_proj.shape
        assert not torch.equal(gup_fq.detach(), e.gate_up_proj.detach())
        assert not torch.equal(down_fq.detach(), e.down_proj.detach())
        # STE: d(sum(fq))/d(param) == ones.
        gup_fq.sum().backward()
        down_fq.sum().backward()
        torch.testing.assert_close(e.gate_up_proj.grad, torch.ones_like(e.gate_up_proj), rtol=0, atol=0)
        torch.testing.assert_close(e.down_proj.grad, torch.ones_like(e.down_proj), rtol=0, atol=0)

    def test_shadow_swaps_then_restores(self):
        e = convert_moe_experts_to_qarl(_make_experts())
        real_gup, real_down = e.gate_up_proj, e.down_proj
        a = torch.zeros_like(real_gup)
        b = torch.ones_like(real_down)
        with e._qarl_shadow_weights(a, b):
            assert e.gate_up_proj is a  # __dict__ shadow takes precedence
            assert e.down_proj is b
        assert e.gate_up_proj is real_gup  # real Parameters restored
        assert e.down_proj is real_down


class TestEagerForwardCPU:
    def test_forward_lossy_and_grad(self):
        e = convert_moe_experts_to_qarl(_make_experts(), group_size=16)
        x = torch.randn(5, 32)
        out = e(x, expert_idx=0)
        assert out.shape == (5, 32)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert e.gate_up_proj.grad is not None
        assert e.down_proj.grad is not None

    def test_differs_from_unquantized(self):
        ref = _make_experts()
        x = torch.randn(5, 32)
        out_ref = ref(x, expert_idx=1)
        q = convert_moe_experts_to_qarl(_make_experts())  # same seed -> same init
        out_q = q(x, expert_idx=1)
        assert not torch.allclose(out_q, out_ref)

    def test_disable_weight_quant_is_passthrough(self):
        ref = _make_experts()
        x = torch.randn(5, 32)
        out_ref = ref(x, expert_idx=2)
        q = convert_moe_experts_to_qarl(_make_experts(), quantize_weight=False)
        out_q = q(x, expert_idx=2)
        torch.testing.assert_close(out_q, out_ref, rtol=0, atol=0)


class TestInjectMoE:
    def _model_with_experts(self):
        torch.manual_seed(0)
        m = nn.Module()
        m.attn = nn.Linear(32, 32)
        m.experts = _make_experts()
        return m

    def test_inject_nvfp4_converts_experts_and_linear(self):
        m = self._model_with_experts()
        n = inject_qarl_into_model(m, quant_cfg=normalize_qarl_quant_cfg("nvfp4"))
        assert isinstance(m.experts, QARLMoEExperts)
        assert isinstance(m.attn, QARLLinear)
        assert n == 2  # 1 Linear + 1 MoE expert container
        assert m._qarl_config["moe_expert_modules"] == ["experts"]

    def test_inject_fp8_rejects_moe(self):
        m = self._model_with_experts()
        with pytest.raises(ValueError, match="nvfp4"):
            inject_qarl_into_model(m, quant_cfg=normalize_qarl_quant_cfg("fp8"))

    def test_target_modules_honored_for_experts(self):
        """target_modules now gates the expert pass too: targeting only the Linear must
        leave the experts unquantized (previously experts were always converted)."""
        m = self._model_with_experts()
        n = inject_qarl_into_model(m, quant_cfg=normalize_qarl_quant_cfg("nvfp4"), target_modules=["attn"])
        assert isinstance(m.attn, QARLLinear)
        assert not isinstance(m.experts, QARLMoEExperts)  # not targeted -> untouched
        assert n == 1
        assert m._qarl_config["moe_expert_modules"] == []

    def test_target_modules_can_select_experts(self):
        """Targeting `experts` converts the expert container and leaves the Linear alone."""
        m = self._model_with_experts()
        n = inject_qarl_into_model(m, quant_cfg=normalize_qarl_quant_cfg("nvfp4"), target_modules=["experts"])
        assert isinstance(m.experts, QARLMoEExperts)
        assert not isinstance(m.attn, QARLLinear)
        assert n == 1
        assert m._qarl_config["moe_expert_modules"] == ["experts"]
