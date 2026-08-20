"""Tests for the NVFP4 STE fake-quant op (weight-only + 3D MoE experts).

Validates the pure-PyTorch NVFP4 round-to-nearest forward against an independent
reference and the straight-through backward through ``F.linear`` and the 3D
expert helpers. Pure PyTorch — runs on CPU.
"""

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.quantize.nvfp4_fake_quant import (
    _E2M1_ABS,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    _fake_quantize_3d_experts,
    _fake_quantize_3d_fused_gate_up,
    fake_quantize_activation_nvfp4,
    fake_quantize_nvfp4,
)


pytestmark = pytest.mark.cpu


def _ref_fake_quant(w: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Independent pure-PyTorch NVFP4 RTN reference (no kernels)."""
    wf = w.float()
    M, K = wf.shape
    g = wf.abs().amax() / (FP4_E2M1_MAX * FP8_E4M3_MAX)
    blocks = wf.reshape(-1, block_size)
    bamax = blocks.abs().amax(dim=1, keepdim=True)
    bscale = (bamax / FP4_E2M1_MAX / g.clamp_min(1e-30)).clamp(max=FP8_E4M3_MAX)
    bscale = bscale.to(torch.float8_e4m3fn).float()
    eff = bscale * g
    eff_safe = torch.where(eff > 0, eff, torch.ones_like(eff))
    grid = torch.tensor(_E2M1_ABS, device=w.device)
    mids = (grid[1:] + grid[:-1]) * 0.5
    x = blocks / eff_safe
    idx = torch.bucketize(x.abs().clamp(max=FP4_E2M1_MAX), mids)
    q = torch.where(x < 0, -grid[idx], grid[idx])
    return (q * eff).reshape(M, K).to(w.dtype)


class TestForward:
    def test_2d_quantization_reference_dispatch_and_ste(self):
        for dtype in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            w = torch.randn(64, 32, dtype=dtype)
            out = fake_quantize_nvfp4(w, block_size=16)
            ref = _ref_fake_quant(w, block_size=16)
            assert out.shape == w.shape
            assert out.dtype == w.dtype
            torch.testing.assert_close(out, ref, rtol=0, atol=0)

        torch.manual_seed(0)
        x = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(128, 256, dtype=torch.float32, requires_grad=True)
        w_fq = fake_quantize_nvfp4(w, block_size=16)
        y = F.linear(x, w_fq)
        y.sum().backward()
        expected = torch.ones(8, 128).T @ x
        assert w.grad is not None
        torch.testing.assert_close(w.grad, expected, rtol=1e-4, atol=1e-4)

        _assert_activation_wrapper_preserves_leading_dimensions()
        TestContract()._assert_input_shape_admission()
        TestMoEExperts3D()._assert_projection_shapes_and_ste()


def _assert_activation_wrapper_preserves_leading_dimensions():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 32, requires_grad=True)
    upstream = torch.randn_like(x)

    actual = fake_quantize_activation_nvfp4(x, block_size=16)
    flattened = fake_quantize_nvfp4(x.detach().reshape(-1, 32), block_size=16).reshape_as(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, flattened, rtol=0, atol=0)
    (actual * upstream).sum().backward()
    torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)


class TestContract:
    def _assert_input_shape_admission(self):
        w = torch.randn(4, 16, 16, dtype=torch.float32)
        with pytest.raises(AssertionError):
            fake_quantize_nvfp4(w)

        w = torch.randn(16, 24, dtype=torch.float32)
        with pytest.raises(AssertionError, match="in_features"):
            fake_quantize_nvfp4(w, block_size=16)


class TestMoEExperts3D:
    """The 3D GKN helpers used by the MoE expert fake-quant wrap."""

    def _assert_projection_shapes_and_ste(self):
        torch.manual_seed(0)
        E, I, H = 4, 64, 128
        w = torch.randn(E, I, H, dtype=torch.float32, requires_grad=True)
        w_fq = _fake_quantize_3d_experts(w, block_size=16)
        assert w_fq.shape == w.shape
        assert not torch.equal(w_fq.detach(), w.detach())
        g = torch.randn_like(w_fq)
        w_fq.backward(g)
        torch.testing.assert_close(w.grad, g, rtol=0, atol=0)

        w = torch.randn(E, H, 2 * I, dtype=torch.float32, requires_grad=True)
        w_fq = _fake_quantize_3d_fused_gate_up(w, intermediate_size=I, block_size=16)
        assert w_fq.shape == w.shape
        assert not torch.equal(w_fq.detach(), w.detach())
        g = torch.randn_like(w_fq)
        w_fq.backward(g)
        torch.testing.assert_close(w.grad, g, rtol=0, atol=0)

        self._assert_experts_quantized_independently()
        self._assert_fused_gate_up_uses_per_half_global_scale()

    def _assert_experts_quantized_independently(self):
        """Scaling one expert's weights must not change another expert's dequant."""
        torch.manual_seed(0)
        E, I, H = 3, 32, 64
        w = torch.randn(E, I, H, dtype=torch.float32)
        base = _fake_quantize_3d_experts(w, block_size=16).detach().clone()
        w2 = w.clone()
        w2[0] *= 100.0  # blow up expert 0's global amax
        out2 = _fake_quantize_3d_experts(w2, block_size=16).detach()
        torch.testing.assert_close(out2[1:], base[1:], rtol=0, atol=0)
        assert not torch.equal(out2[0], base[0])

    def _assert_fused_gate_up_uses_per_half_global_scale(self):
        """gate and up of an expert are quantized with INDEPENDENT global scales.

        Per-half weight_scale_2 (REVERT of PR #399's shared scale): the strictly-matched
        shared ``max(gate, up)`` scale made QAT unstable (grad blow-ups); per-half trains
        smoothly. Export/serve still fuse to one shared weight_scale_2, but the per-row fp8
        block scales absorb most of the gate/up amax gap (the difference is second-order).
        """
        torch.manual_seed(0)
        E, H, I = 3, 64, 32
        w = torch.randn(E, H, 2 * I, dtype=torch.float32)
        # Give gate and up very different dynamic ranges so per-half scales clearly differ.
        w[:, :, :I] *= 0.01  # gate small
        w[:, :, I:] *= 100.0  # up large
        _, meta = _fake_quantize_3d_fused_gate_up(w, intermediate_size=I, block_size=16, return_metadata=True)
        ws2 = meta["weight_scale_2"]  # [E, 2] = (gate, up) per expert
        # Each half uses its OWN global amax / (FP4_MAX*FP8_MAX).
        gate_expected = w[:, :, :I].float().abs().amax(dim=(1, 2)) / (FP4_E2M1_MAX * FP8_E4M3_MAX)
        up_expected = w[:, :, I:].float().abs().amax(dim=(1, 2)) / (FP4_E2M1_MAX * FP8_E4M3_MAX)
        torch.testing.assert_close(ws2[:, 0], gate_expected, rtol=0, atol=0)
        torch.testing.assert_close(ws2[:, 1], up_expected, rtol=0, atol=0)
        # ... and they are NOT shared: up (×100) >> gate (×0.01).
        assert (ws2[:, 1] > ws2[:, 0] * 10).all(), "per-half scales must differ for differently-scaled halves"
