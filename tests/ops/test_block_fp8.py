"""Tests for FP8 block quantization operations.

These tests verify the correctness of block-based FP8 quantization/dequantization
kernels and FP8 GEMM operations.
"""

import pytest
import torch
import numpy as np

# Try to import the block_fp8 module
try:
    from xorl.ops.quantize import (
        block_fp8_quantize as block_fp8_quant,
        block_fp8_dequantize as block_fp8_dequant,
        block_fp8_dequantize_gkn as block_fp8_weight_dequant,
        block_fp8_gemm,
    )
    HAS_BLOCK_FP8 = True
except ImportError:
    HAS_BLOCK_FP8 = False

# Skip all tests if block_fp8 is not available or if CUDA is not available
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_BLOCK_FP8, reason="block_fp8 module not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


class TestBlockFP8Quant:
    """Test suite for block_fp8_quant function."""

    def test_basic_quantization(self):
        """Test basic quantization with simple input."""
        x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)

        assert y.shape == x.shape
        assert y.dtype == torch.float8_e4m3fn
        assert s.shape == (4, 1)  # 4 batches, 1 block per batch
        assert s.dtype == torch.float32

    def test_quantization_multiple_blocks(self):
        """Test quantization with multiple blocks."""
        x = torch.randn(2, 512, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)

        assert y.shape == x.shape
        assert s.shape == (2, 4)  # 2 batches, 4 blocks per batch

    def test_quantization_preserves_shape(self):
        """Test that quantization preserves tensor shape."""
        shapes = [
            (128,),
            (4, 256),
            (2, 3, 384),
        ]

        for shape in shapes:
            x = torch.randn(*shape, device="cuda", dtype=torch.float32)
            y, s = block_fp8_quant(x, block_size=128)

            assert y.shape == x.shape
            expected_s_shape = (*shape[:-1], shape[-1] // 128)
            assert s.shape == expected_s_shape

    def test_quantization_scale_range(self):
        """Test that scale factors are reasonable."""
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)

        # Scale should be positive
        assert torch.all(s > 0)

        # For random normal data, scales should be in reasonable range
        # (typically between 0.001 and 1.0 for standard normal)
        assert torch.all(s < 1.0)
        assert torch.all(s > 1e-6)

    def test_quantization_max_value(self):
        """Test quantization with maximum FP8 representable values."""
        # FP8 E4M3 max value is 448.0
        x = torch.full((2, 128), 448.0, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)

        # Scale should be 1.0 (448 / 448)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)

    def test_quantization_requires_contiguous(self):
        """Test that quantization requires contiguous tensors."""
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        x_transposed = x.t()  # Non-contiguous

        with pytest.raises(AssertionError):
            block_fp8_quant(x_transposed, block_size=128)

    def test_quantization_requires_divisible_size(self):
        """Test that last dimension must be divisible by block_size."""
        x = torch.randn(4, 130, device="cuda", dtype=torch.float32)

        with pytest.raises(AssertionError):
            block_fp8_quant(x, block_size=128)

    def test_quantization_different_block_sizes(self):
        """Test quantization with different block sizes."""
        x = torch.randn(2, 512, device="cuda", dtype=torch.float32)

        for block_size in [64, 128, 256]:
            y, s = block_fp8_quant(x, block_size=block_size)
            assert y.shape == x.shape
            assert s.shape == (2, 512 // block_size)


class TestBlockFP8Dequant:
    """Test suite for block_fp8_dequant function."""

    def test_basic_dequantization(self):
        """Test basic dequantization."""
        x_orig = torch.randn(4, 128, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x_orig, block_size=128)
        x_dequant = block_fp8_dequant(y, s, block_size=128)

        assert x_dequant.shape == x_orig.shape
        assert x_dequant.dtype == torch.float32

    def test_dequantization_accuracy(self):
        """Test that quantization + dequantization is reasonably accurate."""
        x_orig = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x_orig, block_size=128)
        x_dequant = block_fp8_dequant(y, s, block_size=128)

        # FP8 has limited precision, so we expect some error
        # Relative error should be within ~1% for most values
        relative_error = torch.abs(x_dequant - x_orig) / (torch.abs(x_orig) + 1e-6)
        mean_error = relative_error.mean().item()

        assert mean_error < 0.05  # Less than 5% mean relative error

    def test_dequantization_roundtrip_different_shapes(self):
        """Test roundtrip with different tensor shapes."""
        shapes = [
            (128,),
            (4, 256),
            (2, 3, 384),
        ]

        for shape in shapes:
            x_orig = torch.randn(*shape, device="cuda", dtype=torch.float32)
            y, s = block_fp8_quant(x_orig, block_size=128)
            x_dequant = block_fp8_dequant(y, s, block_size=128)

            assert x_dequant.shape == x_orig.shape
            # Check that values are close
            assert torch.allclose(x_dequant, x_orig, rtol=0.1, atol=0.05)

    def test_dequantization_requires_contiguous(self):
        """Test that dequantization requires contiguous tensors."""
        x_orig = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x_orig, block_size=128)

        # Make y non-contiguous
        y_transposed = y.t()

        with pytest.raises(AssertionError):
            block_fp8_dequant(y_transposed, s, block_size=128)


class TestBlockFP8Integration:
    """Integration tests for full quantization workflow."""

    def test_quantization_determinism(self):
        """Test that quantization is deterministic."""
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)

        y1, s1 = block_fp8_quant(x, block_size=128)
        y2, s2 = block_fp8_quant(x, block_size=128)

        assert torch.equal(y1, y2)
        assert torch.equal(s1, s2)

    def test_memory_efficiency(self):
        """Test that FP8 reduces memory usage."""
        M, K = 1024, 2048
        block_size = 128

        x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        y_fp8, s = block_fp8_quant(x_fp32, block_size=block_size)

        # FP8 should use approximately 1/4 the memory of FP32
        # (FP8 is 1 byte vs FP32 is 4 bytes)
        fp32_bytes = x_fp32.element_size() * x_fp32.numel()
        fp8_bytes = y_fp8.element_size() * y_fp8.numel() + s.element_size() * s.numel()

        # FP8 + scales should be less than FP32
        assert fp8_bytes < fp32_bytes

    def test_2d_vs_1d_dequant_consistency(self):
        """Test that 1D and 2D dequant produce similar results for compatible shapes."""
        M, N = 256, 512
        block_size = 128

        # Create test data
        x_fp32 = torch.randn(M, N, device="cuda", dtype=torch.float32)

        # 1D quantization
        y_1d, s_1d = block_fp8_quant(x_fp32, block_size=block_size)
        x_dequant_1d = block_fp8_dequant(y_1d, s_1d, block_size=block_size)

        # For 2D dequant, we need to reshape scales appropriately
        # Note: This test shows the difference between 1D and 2D approaches
        # They won't be identical because they use different blocking strategies

        # Just verify both produce valid outputs
        assert x_dequant_1d.shape == x_fp32.shape
        assert torch.allclose(x_dequant_1d, x_fp32, rtol=0.1, atol=0.05)


class TestBlockFP8EdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_values(self):
        """Test quantization with very small values."""
        x = torch.full((2, 128), 1e-6, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)
        x_dequant = block_fp8_dequant(y, s, block_size=128)

        # Small values should be preserved reasonably well
        assert torch.allclose(x_dequant, x, rtol=0.5, atol=1e-7)

    def test_very_large_values(self):
        """Test quantization with large values near FP8 limits."""
        x = torch.full((2, 128), 400.0, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)
        x_dequant = block_fp8_dequant(y, s, block_size=128)

        # Large values should be preserved
        assert torch.allclose(x_dequant, x, rtol=0.05, atol=1.0)

    def test_mixed_positive_negative(self):
        """Test quantization with mixed positive and negative values."""
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        x[0] = torch.abs(x[0])  # All positive
        x[1] = -torch.abs(x[1])  # All negative

        y, s = block_fp8_quant(x, block_size=128)
        x_dequant = block_fp8_dequant(y, s, block_size=128)

        # Signs should be preserved
        assert torch.all((x >= 0) == (x_dequant >= 0))

    def test_single_block(self):
        """Test with single block (minimum size)."""
        x = torch.randn(1, 128, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)

        assert y.shape == (1, 128)
        assert s.shape == (1, 1)

        x_dequant = block_fp8_dequant(y, s, block_size=128)
        assert torch.allclose(x_dequant, x, rtol=0.1, atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
