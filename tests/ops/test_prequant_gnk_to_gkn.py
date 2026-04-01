"""Tests for pre-quantized GNK -> GKN weight loading correctness.

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
    block_fp8_dequantize_gkn,
    block_fp8_quantize_gkn,
    nvfp4_quantize,
)
from xorl.ops.quantize.nvfp4_gkn_quantize import nvfp4_dequantize_gkn, nvfp4_quantize_gkn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# =========================================================================
# Block FP8 -- GNK -> GKN
# =========================================================================


class TestBlockFP8PrequantGNKtoGKN:
    """Verify block_fp8 quantized data can be correctly transposed from GNK to GKN."""

    def test_block_fp8_transpose_roundtrip_and_stacking(self):
        """Transpose matches direct, roundtrip accuracy, path equivalence, non-square shapes, multi-expert stacking."""
        K, N = 512, 256
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # --- Transpose matches direct quantization ---
        fp8_direct, scales_direct = block_fp8_quantize_gkn(W_gkn)
        W_gnk = W_gkn.T.contiguous()
        fp8_gnk, scales_gnk = block_fp8_quantize_gkn(W_gnk)
        fp8_transposed = fp8_gnk.T.contiguous()
        scales_transposed = scales_gnk.T.contiguous()
        assert torch.equal(fp8_direct, fp8_transposed), "FP8 values differ after transpose"
        assert torch.equal(scales_direct, scales_transposed), "Scales differ after transpose"

        # --- GNK->GKN roundtrip accuracy ---
        fp8_loaded = fp8_gnk.T.contiguous()
        scales_loaded = scales_gnk.float().T.contiguous()
        W_recovered = block_fp8_dequantize_gkn(fp8_loaded, scales_loaded)
        rel_err = (W_gkn - W_recovered).abs().mean() / W_gkn.abs().mean()
        assert rel_err < 0.03, f"Roundtrip rel error {rel_err:.4f} > 0.03"

        # --- Direct path == transposed path dequantization ---
        W_direct = block_fp8_dequantize_gkn(fp8_direct, scales_direct)
        fp8_gnk2, scales_gnk2 = block_fp8_quantize_gkn(W_gkn.T.contiguous())
        W_trans = block_fp8_dequantize_gkn(fp8_gnk2.T.contiguous(), scales_gnk2.T.contiguous())
        assert torch.equal(W_direct, W_trans), "Direct and transposed paths diverge"

        # --- Non-square shapes ---
        for K_ns, N_ns in [(256, 512), (512, 256), (384, 256)]:
            W_ns = torch.randn(K_ns, N_ns, device="cuda", dtype=torch.float32)
            fp8_ns, s_ns = block_fp8_quantize_gkn(W_ns.T.contiguous())
            fp8_l = fp8_ns.T.contiguous()
            s_l = s_ns.T.contiguous()
            assert fp8_l.shape == (K_ns, N_ns)
            assert s_l.shape == (triton.cdiv(K_ns, 128), triton.cdiv(N_ns, 128))
            W_rec = block_fp8_dequantize_gkn(fp8_l, s_l)
            re = (W_ns - W_rec).abs().mean() / W_ns.abs().mean()
            assert re < 0.03, f"Shape ({K_ns},{N_ns}): rel error {re:.4f}"

        # --- Multiple experts stacked ---
        G = 4
        experts_gkn = torch.randn(G, K, N, device="cuda", dtype=torch.float32)
        fp8_list, scales_list = [], []
        for i in range(G):
            w_gnk = experts_gkn[i].T.contiguous()
            fp8_e, s_e = block_fp8_quantize_gkn(w_gnk)
            fp8_list.append(fp8_e.T.contiguous())
            scales_list.append(s_e.T.contiguous())
        fp8_stacked = torch.stack(fp8_list)
        scales_stacked = torch.stack(scales_list)
        assert fp8_stacked.shape == (G, K, N)
        assert scales_stacked.shape == (G, triton.cdiv(K, 128), triton.cdiv(N, 128))
        for i in range(G):
            W_rec = block_fp8_dequantize_gkn(fp8_stacked[i], scales_stacked[i])
            re = (experts_gkn[i] - W_rec).abs().mean() / experts_gkn[i].abs().mean()
            assert re < 0.03, f"Expert {i}: rel error {re:.4f}"


# =========================================================================
# NVFP4 -- GNK -> GKN
# =========================================================================


class TestNVFP4PrequantGNKtoGKN:
    """Verify nvfp4 quantized data can be correctly transposed from GNK to GKN."""

    def test_nvfp4_transpose_roundtrip_and_stacking(self):
        """Roundtrip, direct match, dequant path agreement, non-square, multi-expert stacking, global scale absorption."""
        K, N = 256, 256
        block_size = 16
        W_gkn = torch.randn(K, N, device="cuda", dtype=torch.float32)
        W_gnk = W_gkn.T.contiguous()

        # --- GNK -> GKN roundtrip ---
        packed_flat, scales_flat, global_scale = nvfp4_quantize(W_gnk, block_size)
        packed_gnk = packed_flat.reshape(N, K // 2)
        scales_gnk = scales_flat.reshape(N, K // block_size)
        packed_gkn = packed_gnk.T.contiguous()
        scales_gkn = (scales_gnk.float() * global_scale.float()).T.contiguous()
        gs_loaded = torch.ones(1, dtype=torch.float32, device=W_gkn.device)
        W_recovered = nvfp4_dequantize_gkn(packed_gkn, scales_gkn, gs_loaded, K, N, block_size)
        rel_err = (W_gkn - W_recovered.float()).abs().mean() / W_gkn.abs().mean()
        assert rel_err < 0.15, f"Roundtrip rel error {rel_err:.4f} > 0.15"

        # --- Transposed matches direct quantization ---
        packed_direct, scales_direct, gs_direct = nvfp4_quantize_gkn(W_gkn, block_size)
        packed_flat2, _, _ = nvfp4_quantize(W_gnk, block_size)
        packed_gnk2 = packed_flat2.reshape(N, K // 2)
        packed_transposed = packed_gnk2.T.contiguous()
        assert torch.equal(packed_direct, packed_transposed), "Packed data differs"

        # --- Transposed dequant matches direct dequant ---
        W_a = nvfp4_dequantize_gkn(packed_direct, scales_direct, gs_direct, K, N, block_size)

        packed_flat_b, scales_flat_b, gs_b = nvfp4_quantize(W_gnk, block_size)
        packed_gnk_b = packed_flat_b.reshape(N, K // 2)
        scales_gnk_b = scales_flat_b.reshape(N, K // block_size)
        packed_gkn_b = packed_gnk_b.T.contiguous()
        scales_gkn_b = (scales_gnk_b.float() * gs_b.float()).T.contiguous()
        gs_one = torch.ones(1, dtype=torch.float32, device=W_gkn.device)
        W_b = nvfp4_dequantize_gkn(packed_gkn_b, scales_gkn_b, gs_one, K, N, block_size)
        max_diff = (W_a.float() - W_b.float()).abs().max().item()
        assert max_diff < 1e-3, f"Max diff between paths: {max_diff}"

        # --- Non-square shapes ---
        for K_ns, N_ns in [(256, 512), (512, 256)]:
            W_ns = torch.randn(K_ns, N_ns, device="cuda", dtype=torch.float32)
            W_ns_gnk = W_ns.T.contiguous()
            pf, sf, gs = nvfp4_quantize(W_ns_gnk, block_size)
            p_gnk = pf.reshape(N_ns, K_ns // 2)
            s_gnk = sf.reshape(N_ns, K_ns // block_size)
            p_gkn = p_gnk.T.contiguous()
            s_gkn = (s_gnk.float() * gs.float()).T.contiguous()
            gs_l = torch.ones(1, dtype=torch.float32, device=W_ns.device)
            assert p_gkn.shape == (K_ns // 2, N_ns)
            assert s_gkn.shape == (K_ns // block_size, N_ns)
            W_rec = nvfp4_dequantize_gkn(p_gkn, s_gkn, gs_l, K_ns, N_ns, block_size)
            re = (W_ns - W_rec.float()).abs().mean() / W_ns.abs().mean()
            assert re < 0.15, f"Shape ({K_ns},{N_ns}): rel error {re:.4f}"

        # --- Multiple experts stacked ---
        G, K_e, N_e = 4, 256, 512
        experts_gkn = torch.randn(G, K_e, N_e, device="cuda", dtype=torch.float32)
        packed_list, scales_list = [], []
        for i in range(G):
            w_gnk = experts_gkn[i].T.contiguous()
            pf, sf, gs = nvfp4_quantize(w_gnk, block_size)
            p_gnk = pf.reshape(N_e, K_e // 2)
            s_gnk = sf.reshape(N_e, K_e // block_size)
            packed_list.append(p_gnk.T.contiguous())
            scales_list.append((s_gnk.float() * gs.float()).T.contiguous())
        packed_stacked = torch.stack(packed_list)
        scales_stacked = torch.stack(scales_list)
        assert packed_stacked.shape == (G, K_e // 2, N_e)
        assert scales_stacked.shape == (G, K_e // block_size, N_e)
        gs_l = torch.ones(1, dtype=torch.float32, device=experts_gkn.device)
        for i in range(G):
            W_rec = nvfp4_dequantize_gkn(packed_stacked[i], scales_stacked[i], gs_l, K_e, N_e, block_size)
            re = (experts_gkn[i] - W_rec.float()).abs().mean() / experts_gkn[i].abs().mean()
            assert re < 0.15, f"Expert {i}: rel error {re:.4f}"

        # --- Global scale absorption preserves precision ---
        packed_abs, scales_fp8_abs, gs_abs = nvfp4_quantize_gkn(W_gkn, block_size)
        W_non_absorbed = nvfp4_dequantize_gkn(packed_abs, scales_fp8_abs, gs_abs, K, N, block_size)
        absorbed_scales = scales_fp8_abs.float() * gs_abs.float()
        gs_one_abs = torch.ones(1, dtype=torch.float32, device=W_gkn.device)
        W_absorbed = nvfp4_dequantize_gkn(packed_abs, absorbed_scales, gs_one_abs, K, N, block_size)
        assert torch.equal(W_non_absorbed, W_absorbed), (
            f"Max diff: {(W_non_absorbed.float() - W_absorbed.float()).abs().max().item()}"
        )
