"""CPU tests for NVFP4 weight-only QARL MoE expert fake-quant.

Covers the in-place class-swap, the eager backend path end-to-end on CPU, and
the inject_qarl_into_model MoE wiring (nvfp4 allowed, fp8 rejected). The exact
3D fake-quant arithmetic and STE are owned by the quantization-op suite.
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
    def _assert_conversion_identity_and_admission_policy(self):
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

        self._assert_idempotent()
        self._assert_rejects_non_experts()

    def _assert_idempotent(self):
        e = convert_moe_experts_to_qarl(_make_experts())
        assert convert_moe_experts_to_qarl(e) is e

    def _assert_rejects_non_experts(self):
        with pytest.raises(TypeError):
            convert_moe_experts_to_qarl(nn.Linear(8, 8))


class TestEagerForwardCPU:
    def _assert_eager_execution_and_weight_quantization_policy(self):
        ref = _make_experts()
        x = torch.randn(5, 32)
        out_ref = ref(x, expert_idx=1)
        q = convert_moe_experts_to_qarl(_make_experts(), group_size=16)
        gate_up, down = q.gate_up_proj, q.down_proj
        out_q = q(x, expert_idx=1)
        assert out_q.shape == (5, 32)
        assert torch.isfinite(out_q).all()
        assert not torch.allclose(out_q, out_ref)
        assert q.gate_up_proj is gate_up
        assert q.down_proj is down
        out_q.sum().backward()
        assert q.gate_up_proj.grad is not None
        assert q.down_proj.grad is not None

        self._assert_disabled_weight_quant_is_passthrough()

    def _assert_disabled_weight_quant_is_passthrough(self):
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

    def _assert_injection_selection_and_admission_policy(self):
        m = self._model_with_experts()
        n = inject_qarl_into_model(m, quant_cfg=normalize_qarl_quant_cfg("nvfp4"))
        assert isinstance(m.experts, QARLMoEExperts)
        assert isinstance(m.attn, QARLLinear)
        assert n == 2  # 1 Linear + 1 MoE expert container
        assert m._qarl_config["moe_expert_modules"] == ["experts"]

        self._assert_fp8_rejects_moe()
        self._assert_target_modules_select_independently()

    def _assert_fp8_rejects_moe(self):
        m = self._model_with_experts()
        with pytest.raises(ValueError, match="nvfp4"):
            inject_qarl_into_model(m, quant_cfg=normalize_qarl_quant_cfg("fp8"))

    def _assert_target_modules_select_independently(self):
        cases = [
            ("attn", True, False, []),
            ("experts", False, True, ["experts"]),
        ]
        for target, linear_wrapped, experts_wrapped, expert_modules in cases:
            m = self._model_with_experts()
            n = inject_qarl_into_model(
                m,
                quant_cfg=normalize_qarl_quant_cfg("nvfp4"),
                target_modules=[target],
            )
            assert isinstance(m.attn, QARLLinear) is linear_wrapped
            assert isinstance(m.experts, QARLMoEExperts) is experts_wrapped
            assert n == 1
            assert m._qarl_config["moe_expert_modules"] == expert_modules


def test_nvfp4_moe_expert_conversion_execution_and_injection_policy():
    TestClassSwap()._assert_conversion_identity_and_admission_policy()
    TestEagerForwardCPU()._assert_eager_execution_and_weight_quantization_policy()
    TestInjectMoE()._assert_injection_selection_and_admission_policy()
    _assert_activation_quant_shadow_backend_lifecycle()


def _assert_activation_quant_shadow_backend_lifecycle():
    experts = convert_moe_experts_to_qarl(
        _make_experts(impl="triton"),
        quantize_weight=True,
        quantize_activation=True,
    )
    assert isinstance(experts, QARLMoEExperts)
    assert experts.moe_implementation == "triton"

    with pytest.raises(RuntimeError):
        with experts._qarl_shadow_moe_impl():
            assert experts.moe_implementation == "triton_w4a4"
            raise RuntimeError("boom")
    assert experts.moe_implementation == "triton"

    for implementation, quantize_activation in (("triton", False), ("eager", True)):
        experts = convert_moe_experts_to_qarl(
            _make_experts(impl=implementation),
            quantize_weight=True,
            quantize_activation=quantize_activation,
        )
        with experts._qarl_shadow_moe_impl():
            assert experts.moe_implementation == implementation
        assert experts.moe_implementation == implementation
