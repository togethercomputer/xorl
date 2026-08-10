"""CPU tests for the 100% W4A4 MoE-expert down-GEMM-input fake-quant wiring.

The triton EP group GEMM (where the down input is NVFP4-fake-quantized) is
CUDA-only, so it cannot run on CPU. These tests cover the CPU-testable pieces:

  * ``fake_quantize_activation_nvfp4`` round-trips to dequantized NVFP4 with an STE
    (identity) gradient — this is the exact op the triton kernel calls on the down
    input.
  * ``QARLMoEExperts._qarl_shadow_moe_impl`` selects ``triton_w4a4`` only when
    activation quant is on AND the base backend is ``triton``, and restores the
    original ``moe_implementation`` on exit (no-op otherwise).
  * ``EP_EXPERT_COMPUTE`` registers ``triton_w4a4`` when the triton backend imports
    (guarded — skipped if triton/CUDA is unavailable on the CPU box).
"""

import pytest
import torch

from xorl.ops.quantize.nvfp4_fake_quant import (
    _nvfp4_quantize_blocks,
    fake_quantize_activation_nvfp4,
)
from xorl.qarl.moe_experts import QARLMoEExperts, convert_moe_experts_to_qarl


pytestmark = pytest.mark.cpu


def _make_experts(num_experts=4, hidden_dim=32, intermediate_size=16, impl="eager"):
    from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415

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


class TestActivationFakeQuantSTE:
    def test_forward_equals_dequant_nvfp4(self):
        # The down-input quant op: forward must equal the dequantized NVFP4 of x.
        torch.manual_seed(1)
        x = torch.randn(7, 64)  # last dim divisible by block size 16
        xq = fake_quantize_activation_nvfp4(x, 16)
        # Reference dequant via the shared RTN core (per-row blocks of 16).
        x2d = x.reshape(-1, 64).float()
        w_dq, *_ = _nvfp4_quantize_blocks(x2d, 16)
        ref = w_dq.reshape(x.shape).to(x.dtype)
        torch.testing.assert_close(xq, ref, rtol=0, atol=0)
        # Lossy (NVFP4 rounding actually changed the values).
        assert not torch.equal(xq, x)

    def test_ste_gradient_is_identity(self):
        torch.manual_seed(2)
        x = torch.randn(5, 32, requires_grad=True)
        w = torch.randn(5, 32)
        # Scalar of the (x*w).sum() form -> d/dx = w straight through the quant.
        (fake_quantize_activation_nvfp4(x, 16) * w).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()
        torch.testing.assert_close(x.grad, w, rtol=0, atol=0)

    def test_ste_pure_identity_on_sum(self):
        # d(sum(quant(x)))/dx == ones (textbook STE).
        x = torch.randn(3, 48, requires_grad=True)
        fake_quantize_activation_nvfp4(x, 16).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


class TestShadowMoeImpl:
    def _qarl_triton_experts(self):
        e = convert_moe_experts_to_qarl(
            _make_experts(impl="triton"),
            quantize_weight=True,
            quantize_activation=True,
        )
        # convert() preserves the backend; assert the precondition for the shadow.
        assert e.moe_implementation == "triton"
        return e

    def test_shadow_selects_triton_w4a4_and_restores(self):
        e = self._qarl_triton_experts()
        assert isinstance(e, QARLMoEExperts)
        with e._qarl_shadow_moe_impl():
            assert e.moe_implementation == "triton_w4a4"
        assert e.moe_implementation == "triton"

    def test_shadow_restores_on_exception(self):
        e = self._qarl_triton_experts()
        with pytest.raises(RuntimeError):
            with e._qarl_shadow_moe_impl():
                assert e.moe_implementation == "triton_w4a4"
                raise RuntimeError("boom")
        assert e.moe_implementation == "triton"

    def test_no_shadow_when_activation_quant_off(self):
        e = convert_moe_experts_to_qarl(
            _make_experts(impl="triton"),
            quantize_weight=True,
            quantize_activation=False,
        )
        with e._qarl_shadow_moe_impl():
            assert e.moe_implementation == "triton"  # unchanged
        assert e.moe_implementation == "triton"

    def test_no_shadow_for_non_triton_backend(self):
        # eager (and other backends) have no w4a4 variant: gate/up quant still applies
        # (in forward), down stays bf16 — pre-existing partial-W4A4 behaviour.
        e = convert_moe_experts_to_qarl(
            _make_experts(impl="eager"),
            quantize_weight=True,
            quantize_activation=True,
        )
        with e._qarl_shadow_moe_impl():
            assert e.moe_implementation == "eager"  # unchanged
        assert e.moe_implementation == "eager"


def test_ep_expert_compute_registers_triton_w4a4():
    # The triton backend import needs CUDA on most boxes; skip cleanly if unavailable,
    # mirroring how the repo guards triton/CUDA-only registrations.
    try:
        from xorl.models.layers.moe.backend import EP_EXPERT_COMPUTE  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import guard
        pytest.skip(f"moe backend import failed (triton/CUDA unavailable): {exc}")
    if "triton" not in EP_EXPERT_COMPUTE:
        pytest.skip("triton EP backend unavailable (no triton/CUDA) — no w4a4 variant to register")
    assert "triton_w4a4" in EP_EXPERT_COMPUTE
    assert callable(EP_EXPERT_COMPUTE["triton_w4a4"])
