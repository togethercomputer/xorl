"""Tests for pre-quantized GNK → GKN weight loading correctness.

HuggingFace/modelopt checkpoints store expert weights in GNK format:
  [N, K] per expert (N=out_features, K=in_features)

Our internal format is GKN:
  [K, N] per expert (K=in_features, N=out_features)

The loading code transposes quantized data: fp8/packed.T and scales.T.
This only works if 2D block quantization is transposition-equivariant:
  quant(W.T).T == quant(W)

These tests verify that property for both block_fp8 and nvfp4 formats.
"""

import pytest
import torch
import triton

from xorl.ops.quantize import (
    block_fp8_quantize_gkn,
    block_fp8_dequantize_gkn,
    nvfp4_quantize,
    nvfp4_dequantize,
)
from xorl.ops.quantize.nvfp4_gkn_quantize import nvfp4_quantize_gkn, nvfp4_dequantize_gkn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# =========================================================================
# Block FP8 — GNK → GKN
# =========================================================================

class TestBlockFP8PrequantGNKtoGKN:
    """Verify block_fp8 quantized data can be correctly transposed from GNK to GKN."""

    def test_transpose_matches_direct_quantization(self):
        """quant(W.T).T should be bit-identical to quant(W) for 2D block FP8.

        Tile (i, j) of [N, K] contains the same elements as tile (j, i) of [K, N],
        so scales and quantized values should be identical after transpose.
        """
        K, N = 512, 256
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Direct GKN quantization
        fp8_direct, scales_direct = block_fp8_quantize_gkn(W_gkn)

        # Simulate HF checkpoint: quantize in GNK [N, K], then transpose
        W_gnk = W_gkn.T.contiguous()
        fp8_gnk, scales_gnk = block_fp8_quantize_gkn(W_gnk)
        fp8_transposed = fp8_gnk.T.contiguous()
        scales_transposed = scales_gnk.T.contiguous()

        assert torch.equal(fp8_direct, fp8_transposed), "FP8 values differ after transpose"
        assert torch.equal(scales_direct, scales_transposed), "Scales differ after transpose"

    def test_gnk_to_gkn_roundtrip(self):
        """Pre-quantized GNK weight → transpose → dequantize → low error vs original."""
        K, N = 512, 256
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Simulate HF checkpoint: quantize W as [N, K]
        W_gnk = W_gkn.T.contiguous()
        fp8_gnk, scales_gnk = block_fp8_quantize_gkn(W_gnk)

        # Loading logic: transpose to GKN
        fp8_loaded = fp8_gnk.T.contiguous()
        scales_loaded = scales_gnk.float().T.contiguous()

        # Dequantize in GKN format
        W_recovered = block_fp8_dequantize_gkn(fp8_loaded, scales_loaded)

        rel_err = (W_gkn - W_recovered).abs().mean() / W_gkn.abs().mean()
        assert rel_err < 0.03, f"Roundtrip rel error {rel_err:.4f} > 0.03"

    def test_gnk_to_gkn_matches_direct_dequant(self):
        """Transposed-path dequant should exactly match direct-path dequant."""
        K, N = 512, 256
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Direct path: quantize GKN → dequantize GKN
        fp8_direct, scales_direct = block_fp8_quantize_gkn(W_gkn)
        W_direct = block_fp8_dequantize_gkn(fp8_direct, scales_direct)

        # Transposed path: quantize GNK → transpose → dequantize GKN
        fp8_gnk, scales_gnk = block_fp8_quantize_gkn(W_gkn.T.contiguous())
        fp8_loaded = fp8_gnk.T.contiguous()
        scales_loaded = scales_gnk.T.contiguous()
        W_transposed = block_fp8_dequantize_gkn(fp8_loaded, scales_loaded)

        assert torch.equal(W_direct, W_transposed), "Direct and transposed paths diverge"

    def test_non_square_shapes(self):
        """Test with typical MoE projection shapes (non-square, large)."""
        for K, N in [(256, 512), (512, 256), (384, 256)]:
            W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

            fp8_gnk, scales_gnk = block_fp8_quantize_gkn(W_gkn.T.contiguous())
            fp8_loaded = fp8_gnk.T.contiguous()
            scales_loaded = scales_gnk.T.contiguous()

            assert fp8_loaded.shape == (K, N)
            assert scales_loaded.shape == (triton.cdiv(K, 128), triton.cdiv(N, 128))

            W_recovered = block_fp8_dequantize_gkn(fp8_loaded, scales_loaded)
            rel_err = (W_gkn - W_recovered).abs().mean() / W_gkn.abs().mean()
            assert rel_err < 0.03, f"Shape ({K},{N}): rel error {rel_err:.4f}"

    def test_multiple_experts_stacked(self):
        """Simulate loading G experts: stack transposed weights into [G, K, N]."""
        G, K, N = 4, 256, 512
        experts_gkn = torch.randn(G, K, N, device="cuda", dtype=torch.float32)

        fp8_list, scales_list = [], []
        for i in range(G):
            w_gnk = experts_gkn[i].T.contiguous()
            fp8_gnk, scales_gnk = block_fp8_quantize_gkn(w_gnk)
            fp8_list.append(fp8_gnk.T.contiguous())
            scales_list.append(scales_gnk.T.contiguous())

        fp8_stacked = torch.stack(fp8_list)
        scales_stacked = torch.stack(scales_list)

        assert fp8_stacked.shape == (G, K, N)
        assert scales_stacked.shape == (G, triton.cdiv(K, 128), triton.cdiv(N, 128))

        # Dequantize each expert and verify
        for i in range(G):
            W_rec = block_fp8_dequantize_gkn(fp8_stacked[i], scales_stacked[i])
            rel_err = (experts_gkn[i] - W_rec).abs().mean() / experts_gkn[i].abs().mean()
            assert rel_err < 0.03, f"Expert {i}: rel error {rel_err:.4f}"


# =========================================================================
# NVFP4 — GNK → GKN
# =========================================================================

class TestNVFP4PrequantGNKtoGKN:
    """Verify nvfp4 quantized data can be correctly transposed from GNK to GKN.

    HF/modelopt checkpoint format (per expert, GNK):
        weight:        [N, K//2]         uint8  (packed FP4)
        weight_scale:  [N, K//block_size] fp8    (per-block scales)
        weight_scale_2: [1]               fp32   (global scale)

    Loading logic (from _load_prequantized_experts):
        packed_gkn = packed_gnk.T.contiguous()
        block_scales_gkn = (block_scales_gnk.float() * global_scale.float()).T.contiguous()
        global_scale_loaded = 1.0
    """

    def test_gnk_to_gkn_roundtrip(self):
        """Pre-quantized GNK → transpose + absorb → dequantize → low error."""
        K, N = 256, 256
        block_size = 16
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Simulate HF checkpoint: quantize W as flattened [N, K] (1D quantize)
        W_gnk = W_gkn.T.contiguous()
        packed_flat, scales_flat, global_scale = nvfp4_quantize(W_gnk, block_size)

        # Reshape to HF per-expert format
        packed_gnk = packed_flat.reshape(N, K // 2)          # [N, K//2]
        scales_gnk = scales_flat.reshape(N, K // block_size)  # [N, K//bs]

        # Apply loading logic: transpose + absorb global_scale
        packed_gkn = packed_gnk.T.contiguous()                # [K//2, N]
        scales_gkn = (scales_gnk.float() * global_scale.float()).T.contiguous()  # [K//bs, N]
        gs_loaded = torch.ones(1, dtype=torch.float32, device=W_gkn.device)

        # Dequantize in GKN format
        W_recovered = nvfp4_dequantize_gkn(packed_gkn, scales_gkn, gs_loaded, K, N, block_size)

        rel_err = (W_gkn - W_recovered.float()).abs().mean() / W_gkn.abs().mean()
        assert rel_err < 0.15, f"Roundtrip rel error {rel_err:.4f} > 0.15"

    def test_transposed_matches_direct_quantization(self):
        """1D quant on GNK reshaped+transposed should match GKN quant output."""
        K, N = 256, 256
        block_size = 16
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Path A: direct GKN quantization
        packed_direct, scales_direct, gs_direct = nvfp4_quantize_gkn(W_gkn, block_size)

        # Path B: 1D quantize on W_gnk, reshape, transpose
        W_gnk = W_gkn.T.contiguous()
        packed_flat, scales_flat, gs_1d = nvfp4_quantize(W_gnk, block_size)
        packed_gnk = packed_flat.reshape(N, K // 2)
        packed_transposed = packed_gnk.T.contiguous()

        # Packed bytes should match (same elements, same packing order within blocks)
        assert torch.equal(packed_direct, packed_transposed), (
            "Packed data differs between direct GKN and transposed GNK"
        )

    def test_transposed_dequant_matches_direct(self):
        """Both dequantization paths should produce the same reconstructed weight."""
        K, N = 256, 256
        block_size = 16
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Path A: direct GKN → dequant GKN
        packed_a, scales_a, gs_a = nvfp4_quantize_gkn(W_gkn, block_size)
        W_a = nvfp4_dequantize_gkn(packed_a, scales_a, gs_a, K, N, block_size)

        # Path B: 1D quant GNK → reshape → transpose → absorb → dequant GKN
        W_gnk = W_gkn.T.contiguous()
        packed_flat, scales_flat, gs_b = nvfp4_quantize(W_gnk, block_size)
        packed_gnk = packed_flat.reshape(N, K // 2)
        scales_gnk = scales_flat.reshape(N, K // block_size)
        packed_gkn = packed_gnk.T.contiguous()
        scales_gkn = (scales_gnk.float() * gs_b.float()).T.contiguous()
        gs_loaded = torch.ones(1, dtype=torch.float32, device=W_gkn.device)
        W_b = nvfp4_dequantize_gkn(packed_gkn, scales_gkn, gs_loaded, K, N, block_size)

        # Both paths should agree (may differ slightly due to fp8 scale precision)
        max_diff = (W_a.float() - W_b.float()).abs().max().item()
        assert max_diff < 1e-3, f"Max diff between paths: {max_diff}"

    def test_non_square_shapes(self):
        """Test with gate_proj (K=hidden, N=intermediate) and down_proj (K=intermediate, N=hidden)."""
        block_size = 16
        for K, N in [(256, 512), (512, 256)]:
            W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)
            W_gnk = W_gkn.T.contiguous()

            packed_flat, scales_flat, gs = nvfp4_quantize(W_gnk, block_size)
            packed_gnk = packed_flat.reshape(N, K // 2)
            scales_gnk = scales_flat.reshape(N, K // block_size)

            packed_gkn = packed_gnk.T.contiguous()
            scales_gkn = (scales_gnk.float() * gs.float()).T.contiguous()
            gs_loaded = torch.ones(1, dtype=torch.float32, device=W_gkn.device)

            assert packed_gkn.shape == (K // 2, N)
            assert scales_gkn.shape == (K // block_size, N)

            W_recovered = nvfp4_dequantize_gkn(packed_gkn, scales_gkn, gs_loaded, K, N, block_size)
            rel_err = (W_gkn - W_recovered.float()).abs().mean() / W_gkn.abs().mean()
            assert rel_err < 0.15, f"Shape ({K},{N}): rel error {rel_err:.4f}"

    def test_multiple_experts_stacked(self):
        """Simulate loading G experts with global_scale absorption."""
        G, K, N = 4, 256, 512
        block_size = 16
        experts_gkn = torch.randn(G, K, N, device="cuda", dtype=torch.float32)

        packed_list, scales_list = [], []
        for i in range(G):
            w_gnk = experts_gkn[i].T.contiguous()
            packed_flat, scales_flat, gs = nvfp4_quantize(w_gnk, block_size)
            packed_gnk = packed_flat.reshape(N, K // 2)
            scales_gnk = scales_flat.reshape(N, K // block_size)

            packed_gkn = packed_gnk.T.contiguous()
            scales_gkn = (scales_gnk.float() * gs.float()).T.contiguous()
            packed_list.append(packed_gkn)
            scales_list.append(scales_gkn)

        packed_stacked = torch.stack(packed_list)
        scales_stacked = torch.stack(scales_list)

        assert packed_stacked.shape == (G, K // 2, N)
        assert scales_stacked.shape == (G, K // block_size, N)

        gs_loaded = torch.ones(1, dtype=torch.float32, device=experts_gkn.device)
        for i in range(G):
            W_rec = nvfp4_dequantize_gkn(
                packed_stacked[i], scales_stacked[i], gs_loaded, K, N, block_size
            )
            rel_err = (experts_gkn[i] - W_rec.float()).abs().mean() / experts_gkn[i].abs().mean()
            assert rel_err < 0.15, f"Expert {i}: rel error {rel_err:.4f}"

    def test_global_scale_absorption_preserves_precision(self):
        """Absorbed scales (float32) should match non-absorbed scales (fp8 * global_scale)."""
        K, N = 256, 256
        block_size = 16
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        packed, scales_fp8, gs = nvfp4_quantize_gkn(W_gkn, block_size)

        # Non-absorbed: fp8 scales + float32 global_scale
        W_non_absorbed = nvfp4_dequantize_gkn(packed, scales_fp8, gs, K, N, block_size)

        # Absorbed: float32 scales (fp8→f32 * gs), global_scale=1.0
        absorbed_scales = scales_fp8.float() * gs.float()
        gs_one = torch.ones(1, dtype=torch.float32, device=W_gkn.device)
        W_absorbed = nvfp4_dequantize_gkn(packed, absorbed_scales, gs_one, K, N, block_size)

        # Should be identical — absorption is just reordering the multiplication
        assert torch.equal(W_non_absorbed, W_absorbed), (
            f"Max diff: {(W_non_absorbed.float() - W_absorbed.float()).abs().max().item()}"
        )
