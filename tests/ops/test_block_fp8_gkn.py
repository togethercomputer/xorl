"""Tests for block_fp8 GKN (2D weight) quantization kernels.

Correctness tests and bandwidth benchmarks for block_fp8_quantize_gkn
and block_fp8_dequantize_gkn.
"""

import pytest
import torch
import triton

from xorl.ops.quantize import block_fp8_dequantize_gkn, block_fp8_quantize_gkn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class TestBlockFP8QuantizeGKN:
    """Correctness tests for block_fp8_quantize_gkn: shapes, accuracy, edge cases."""

    def test_quantize_shapes_accuracy_and_edge_cases(self):
        """Output shapes, roundtrip accuracy (f32/bf16), non-divisible shapes, zeros, scale ranges, large matrix."""
        # Basic output shapes
        K, N = 512, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        assert y.shape == (K, N)
        assert y.dtype == torch.float8_e4m3fn
        assert s.shape == (triton.cdiv(K, 128), triton.cdiv(N, 128))
        assert s.dtype == torch.float32

        # Roundtrip accuracy (f32)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)
        rel_err = (x - x_deq).abs().mean() / x.abs().mean()
        assert rel_err < 0.03, f"f32 roundtrip rel error {rel_err:.4f} too high"

        # Roundtrip accuracy (bf16 input)
        K2, N2 = 256, 256
        x_bf = torch.randn(K2, N2, device="cuda", dtype=torch.bfloat16)
        y_bf, s_bf = block_fp8_quantize_gkn(x_bf.float(), block_size=128)
        x_deq_bf = block_fp8_dequantize_gkn(y_bf, s_bf, block_size=128)
        rel_err_bf = (x_bf.float() - x_deq_bf).abs().mean() / x_bf.float().abs().mean()
        assert rel_err_bf < 0.03

        # Non-divisible shapes (one dim)
        K3, N3 = 384, 256
        x3 = torch.randn(K3, N3, device="cuda", dtype=torch.float32)
        y3, s3 = block_fp8_quantize_gkn(x3, block_size=128)
        x3_deq = block_fp8_dequantize_gkn(y3, s3, block_size=128)
        assert (x3 - x3_deq).abs().mean() / x3.abs().mean() < 0.03

        # Non-divisible both dims
        K4, N4 = 300, 200
        x4 = torch.randn(K4, N4, device="cuda", dtype=torch.float32)
        y4, s4 = block_fp8_quantize_gkn(x4, block_size=128)
        assert y4.shape == (K4, N4)
        assert s4.shape == (triton.cdiv(K4, 128), triton.cdiv(N4, 128))
        x4_deq = block_fp8_dequantize_gkn(y4, s4, block_size=128)
        assert (x4 - x4_deq).abs().mean() / x4.abs().mean() < 0.03

        # Zero block
        x_z = torch.zeros(128, 128, device="cuda", dtype=torch.float32)
        y_z, s_z = block_fp8_quantize_gkn(x_z, block_size=128)
        x_z_deq = block_fp8_dequantize_gkn(y_z, s_z, block_size=128)
        assert (x_z_deq == 0).all()

        # Scale value ranges
        x5 = torch.randn(256, 256, device="cuda", dtype=torch.float32) * 3.0
        _, s5 = block_fp8_quantize_gkn(x5, block_size=128)
        assert (s5 > 0).all()
        assert s5.max().item() < 1.0
        assert s5.min().item() > 1e-12

        # Large matrix
        K6, N6 = 4096, 4096
        x6 = torch.randn(K6, N6, device="cuda", dtype=torch.float32)
        y6, s6 = block_fp8_quantize_gkn(x6, block_size=128)
        x6_deq = block_fp8_dequantize_gkn(y6, s6, block_size=128)
        assert (x6 - x6_deq).abs().mean() / x6.abs().mean() < 0.03


class TestBlockFP8DequantizeGKN:
    """Correctness tests for block_fp8_dequantize_gkn: shape, dtype, input requirements."""

    def test_dequantize_output_and_requirements(self):
        """Output shape/dtype, contiguity requirement, 2D requirement."""
        K, N = 256, 512
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)
        assert x_deq.shape == (K, N)
        assert x_deq.dtype == torch.get_default_dtype()

        # Non-contiguous input should fail
        K2, N2 = 256, 256
        x2 = torch.randn(K2, N2, device="cuda", dtype=torch.float32)
        y2, s2 = block_fp8_quantize_gkn(x2, block_size=128)
        with pytest.raises(AssertionError):
            block_fp8_dequantize_gkn(y2.t(), s2, block_size=128)

        # Non-2D input should fail
        with pytest.raises(AssertionError):
            block_fp8_dequantize_gkn(
                torch.zeros(8, 128, 128, device="cuda", dtype=torch.float8_e4m3fn),
                torch.ones(8, 1, 1, device="cuda", dtype=torch.float32),
                block_size=128,
            )


# ---------------------------------------------------------------------------
# Bandwidth benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBlockFP8GKNBandwidth:
    """Bandwidth tests targeting >2000 GB/s on H100."""

    def test_quantize_dequantize_gkn_bandwidth(self):
        """Measure quantize and dequantize bandwidth."""
        K, N = 4096, 4096
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # --- Quantize bandwidth ---
        for _ in range(10):
            block_fp8_quantize_gkn(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            block_fp8_quantize_gkn(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 100
        total_bytes = K * N * (4 + 1)
        bw_quant = total_bytes / (elapsed_ms * 1e-3) / 1e9
        print(f"\nblock_fp8_quantize_gkn: {bw_quant:.0f} GB/s ({elapsed_ms * 1000:.1f} us)")
        assert bw_quant > 2000, f"Quant bandwidth {bw_quant:.0f} GB/s < 2000 GB/s"

        # --- Dequantize bandwidth ---
        y, s = block_fp8_quantize_gkn(x)
        for _ in range(10):
            block_fp8_dequantize_gkn(y, s)
        torch.cuda.synchronize()
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()
        for _ in range(100):
            block_fp8_dequantize_gkn(y, s)
        end2.record()
        torch.cuda.synchronize()
        elapsed_ms2 = start2.elapsed_time(end2) / 100
        total_bytes2 = K * N * (1 + 4)
        bw_dequant = total_bytes2 / (elapsed_ms2 * 1e-3) / 1e9
        print(f"\nblock_fp8_dequantize_gkn: {bw_dequant:.0f} GB/s ({elapsed_ms2 * 1000:.1f} us)")
        assert bw_dequant > 2000, f"Dequant bandwidth {bw_dequant:.0f} GB/s < 2000 GB/s"
