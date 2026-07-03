"""Tests for the NVFP4 STE fake-quant op (weight-only + 3D MoE experts).

Validates the pure-PyTorch NVFP4 round-to-nearest forward (matches an independent
reference; values lie on the E2M1 grid) and the straight-through-identity backward,
including through an ``F.linear`` and the 3D expert helpers. Pure PyTorch — runs on CPU.
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
    fake_quantize,
    fake_quantize_nvfp4,
    is_supported_format,
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
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("shape", [(256, 256), (128, 512), (320, 64)])
    def test_forward_matches_reference(self, dtype, shape):
        torch.manual_seed(0)
        w = torch.randn(*shape, dtype=dtype)
        out = fake_quantize_nvfp4(w, block_size=16)
        ref = _ref_fake_quant(w, block_size=16)
        assert out.shape == w.shape
        assert out.dtype == w.dtype
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_values_lie_on_scaled_grid(self):
        """Each dequantized element is a grid value times its block's scale."""
        torch.manual_seed(0)
        w = torch.randn(64, 16, dtype=torch.float32)
        out = fake_quantize_nvfp4(w, block_size=16)
        grid = torch.tensor(_E2M1_ABS)
        allowed = torch.cat([grid, -grid]).unique()
        for r in range(out.shape[0]):
            row = out[r]
            scale = row.abs().max() / FP4_E2M1_MAX
            if scale == 0:
                continue
            codes = row / scale
            nearest = torch.min((codes[:, None] - allowed[None, :]).abs(), dim=1).values
            assert nearest.max() < 1e-2, "dequantized values must lie on the E2M1 grid"

    def test_roundtrip_error_small(self):
        """FP4 fake-quant of a Gaussian weight should be a low-error approximation."""
        torch.manual_seed(0)
        w = torch.randn(256, 256, dtype=torch.float32)
        out = fake_quantize_nvfp4(w, block_size=16)
        rel_err = (out - w).norm() / w.norm()
        assert rel_err < 0.15, f"relative fake-quant error too high: {rel_err:.4f}"

    def test_is_lossy(self):
        """Fake quant must actually change the weights (not a no-op identity)."""
        torch.manual_seed(0)
        w = torch.randn(128, 128, dtype=torch.float32)
        out = fake_quantize_nvfp4(w)
        assert not torch.equal(out, w)


class TestBackwardSTE:
    def test_identity_gradient(self):
        """Backward is the straight-through identity: w.grad == grad_out."""
        torch.manual_seed(0)
        w = torch.randn(256, 256, dtype=torch.float32, requires_grad=True)
        out = fake_quantize_nvfp4(w, block_size=16)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        assert w.grad is not None
        torch.testing.assert_close(w.grad, grad_out, rtol=0, atol=0)

    def test_gradient_through_linear(self):
        """STE flows the full upstream gradient through an F.linear weight."""
        torch.manual_seed(0)
        x = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(128, 256, dtype=torch.float32, requires_grad=True)
        w_fq = fake_quantize_nvfp4(w, block_size=16)
        y = F.linear(x, w_fq)
        y.sum().backward()
        expected = torch.ones(8, 128).T @ x
        assert w.grad is not None
        torch.testing.assert_close(w.grad, expected, rtol=1e-4, atol=1e-4)


class TestContract:
    def test_rejects_non_2d(self):
        w = torch.randn(4, 16, 16, dtype=torch.float32)
        with pytest.raises(AssertionError):
            fake_quantize_nvfp4(w)

    def test_rejects_not_divisible_by_block(self):
        w = torch.randn(17, 17, dtype=torch.float32)  # 289 not divisible by 16
        with pytest.raises(AssertionError):
            fake_quantize_nvfp4(w, block_size=16)


class TestRegistry:
    def test_nvfp4_supported(self):
        assert is_supported_format("nvfp4")
        assert not is_supported_format("int4")

    def test_dispatch_matches_direct(self):
        torch.manual_seed(0)
        w = torch.randn(128, 128, dtype=torch.bfloat16)
        torch.testing.assert_close(fake_quantize(w, "nvfp4", 16), fake_quantize_nvfp4(w, 16), rtol=0, atol=0)

    def test_unsupported_format_raises(self):
        w = torch.randn(128, 128, dtype=torch.float32)
        with pytest.raises(ValueError):
            fake_quantize(w, "int4", 16)


class TestMoEExperts3D:
    """The 3D GKN helpers used by the MoE expert fake-quant wrap."""

    def test_down_proj_ste_and_shape(self):
        """down_proj [E, K=I, N=H]: STE identity backward, shape preserved, lossy."""
        torch.manual_seed(0)
        E, I, H = 4, 64, 128
        w = torch.randn(E, I, H, dtype=torch.float32, requires_grad=True)
        w_fq = _fake_quantize_3d_experts(w, block_size=16)
        assert w_fq.shape == w.shape
        assert not torch.equal(w_fq.detach(), w.detach())
        g = torch.randn_like(w_fq)
        w_fq.backward(g)
        torch.testing.assert_close(w.grad, g, rtol=0, atol=0)

    def test_experts_quantized_independently(self):
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

    def test_fused_gate_up_ste_and_shape(self):
        """gate_up_proj [E, H, 2I]: STE identity backward, shape preserved, lossy."""
        torch.manual_seed(0)
        E, H, I = 4, 128, 64
        w = torch.randn(E, H, 2 * I, dtype=torch.float32, requires_grad=True)
        w_fq = _fake_quantize_3d_fused_gate_up(w, intermediate_size=I, block_size=16)
        assert w_fq.shape == w.shape
        assert not torch.equal(w_fq.detach(), w.detach())
        g = torch.randn_like(w_fq)
        w_fq.backward(g)
        torch.testing.assert_close(w.grad, g, rtol=0, atol=0)

    def test_fused_gate_up_metadata_shapes(self):
        """return_metadata exposes GKN-layout block_scales + per-(expert,half) scale."""
        torch.manual_seed(0)
        E, H, I, bs = 2, 64, 48, 16
        w = torch.randn(E, H, 2 * I, dtype=torch.float32)
        _, meta = _fake_quantize_3d_fused_gate_up(w, intermediate_size=I, block_size=bs, return_metadata=True)
        assert meta["weight_scale_2"].shape == (E, 2)
        assert meta["block_scales"].shape == (E, 2, H // bs, I)
        assert meta["codes"].shape == (E, 2, I, H)

    def test_fused_gate_up_uses_per_half_global_scale(self):
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

    def test_2d_requires_in_features_divisible_by_block(self):
        """K (in_features) not divisible by block_size must fail loud, not silently
        group elements across output rows (M*K-divisible-but-K-not is the trap)."""
        # M*K = 16*24 = 384 is divisible by 16, but K=24 is NOT -> must raise.
        w = torch.randn(16, 24, dtype=torch.float32)
        with pytest.raises(AssertionError, match="in_features"):
            fake_quantize_nvfp4(w, block_size=16)
        # K divisible by block_size is fine.
        ok = fake_quantize_nvfp4(torch.randn(16, 32), block_size=16)
        assert ok.shape == (16, 32)
