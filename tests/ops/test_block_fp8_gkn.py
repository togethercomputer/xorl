"""Tests for block_fp8 GKN (2D weight) quantization kernels.

Correctness tests and bandwidth benchmarks for block_fp8_quantize_gkn
and block_fp8_dequantize_gkn.
"""

import pytest
import torch
import triton

from xorl.ops.quantize import block_fp8_quantize_gkn, block_fp8_dequantize_gkn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class TestBlockFP8QuantizeGKN:
    """Correctness tests for block_fp8_quantize_gkn."""

    def test_output_shapes(self):
        """Quantized weight and scales should have correct shapes."""
        K, N = 512, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        assert y.shape == (K, N)
        assert y.dtype == torch.float8_e4m3fn
        assert s.shape == (triton.cdiv(K, 128), triton.cdiv(N, 128))
        assert s.dtype == torch.float32

    def test_roundtrip_accuracy(self):
        """Quant -> dequant roundtrip should have low relative error."""
        K, N = 512, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)

        rel_err = (x - x_deq).abs().mean() / x.abs().mean()
        assert rel_err < 0.03, f"Roundtrip relative error {rel_err:.4f} too high (expected < 0.03)"

    def test_roundtrip_bf16_input(self):
        """Should work with bf16 input (internally casts to float32)."""
        K, N = 256, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        y, s = block_fp8_quantize_gkn(x.float(), block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)

        rel_err = (x.float() - x_deq).abs().mean() / x.float().abs().mean()
        assert rel_err < 0.03, f"Roundtrip relative error {rel_err:.4f} too high"

    def test_non_divisible_shapes(self):
        """Should handle shapes not perfectly divisible by block_size via masking."""
        K, N = 384, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)

        rel_err = (x - x_deq).abs().mean() / x.abs().mean()
        assert rel_err < 0.03

    def test_non_divisible_both_dims(self):
        """Both K and N not multiples of 128."""
        K, N = 300, 200
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        assert y.shape == (K, N)
        assert s.shape == (triton.cdiv(K, 128), triton.cdiv(N, 128))

        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)
        rel_err = (x - x_deq).abs().mean() / x.abs().mean()
        assert rel_err < 0.03

    def test_zero_block(self):
        """A block of all zeros should produce zero quantized values."""
        K, N = 128, 128
        x = torch.zeros(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)
        assert (x_deq == 0).all()

    def test_scale_values_reasonable(self):
        """Per-block scales should be positive and reasonable."""
        K, N = 256, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32) * 3.0
        _, s = block_fp8_quantize_gkn(x, block_size=128)
        assert (s > 0).all(), "All scales should be positive"
        assert s.max().item() < 1.0, f"Scale too large: {s.max().item()}"
        assert s.min().item() > 1e-12, f"Scale too small: {s.min().item()}"

    def test_large_matrix(self):
        """Test with a large realistic weight shape."""
        K, N = 4096, 4096
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)

        rel_err = (x - x_deq).abs().mean() / x.abs().mean()
        assert rel_err < 0.03


class TestBlockFP8DequantizeGKN:
    """Correctness tests for block_fp8_dequantize_gkn."""

    def test_output_shape_and_dtype(self):
        """Dequantized output should have correct shape and default dtype."""
        K, N = 256, 512
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        x_deq = block_fp8_dequantize_gkn(y, s, block_size=128)
        assert x_deq.shape == (K, N)
        assert x_deq.dtype == torch.get_default_dtype()

    def test_requires_contiguous(self):
        """Should assert on non-contiguous input."""
        K, N = 256, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x, block_size=128)
        with pytest.raises(AssertionError):
            block_fp8_dequantize_gkn(y.t(), s, block_size=128)

    def test_requires_2d(self):
        """Should assert on non-2D input."""
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

    def test_quantize_gkn_bandwidth(self):
        K, N = 4096, 4096
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        # Warmup (also triggers autotuning)
        for _ in range(10):
            block_fp8_quantize_gkn(x)
        torch.cuda.synchronize()
        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            block_fp8_quantize_gkn(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 100
        total_bytes = K * N * (4 + 1)  # read f32 + write fp8
        bandwidth_gbs = total_bytes / (elapsed_ms * 1e-3) / 1e9
        print(f"\nblock_fp8_quantize_gkn: {bandwidth_gbs:.0f} GB/s ({elapsed_ms*1000:.1f} us)")
        assert bandwidth_gbs > 2000, f"Bandwidth {bandwidth_gbs:.0f} GB/s < 2000 GB/s"

    def test_dequantize_gkn_bandwidth(self):
        K, N = 4096, 4096
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quantize_gkn(x)
        # Warmup
        for _ in range(10):
            block_fp8_dequantize_gkn(y, s)
        torch.cuda.synchronize()
        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            block_fp8_dequantize_gkn(y, s)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 100
        total_bytes = K * N * (1 + 4)  # read fp8 + write f32
        bandwidth_gbs = total_bytes / (elapsed_ms * 1e-3) / 1e9
        print(f"\nblock_fp8_dequantize_gkn: {bandwidth_gbs:.0f} GB/s ({elapsed_ms*1000:.1f} us)")
        assert bandwidth_gbs > 2000, f"Bandwidth {bandwidth_gbs:.0f} GB/s < 2000 GB/s"
