"""Tests for FP8 block quantization operations.

These tests verify the correctness of block-based FP8 quantization/dequantization
kernels and FP8 GEMM operations.
"""

import pytest
import torch


# Try to import the block_fp8 module
try:
    from xorl.ops.quantize import (
        block_fp8_dequantize as block_fp8_dequant,
    )
    from xorl.ops.quantize import (
        block_fp8_dequantize_gkn as block_fp8_weight_dequant,
    )
    from xorl.ops.quantize import (
        block_fp8_gemm,
    )
    from xorl.ops.quantize import (
        block_fp8_quantize as block_fp8_quant,
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


class TestBlockFP8Quantization:
    """Comprehensive tests for block_fp8 quantization and dequantization."""

    def test_quantization_shapes_dtypes_and_scales(self):
        """Quantization output shapes, dtypes, scale ranges, block sizes, and max value handling."""
        # Basic quantization shape and dtype
        x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x, block_size=128)
        assert y.shape == x.shape
        assert y.dtype == torch.float8_e4m3fn
        assert s.shape == (4, 1)
        assert s.dtype == torch.float32

        # Multiple blocks
        x2 = torch.randn(2, 512, device="cuda", dtype=torch.float32)
        y2, s2 = block_fp8_quant(x2, block_size=128)
        assert y2.shape == x2.shape
        assert s2.shape == (2, 4)

        # Shape preservation across various dimensions
        for shape in [(128,), (4, 256), (2, 3, 384)]:
            x_s = torch.randn(*shape, device="cuda", dtype=torch.float32)
            y_s, s_s = block_fp8_quant(x_s, block_size=128)
            assert y_s.shape == x_s.shape
            expected_s_shape = (*shape[:-1], shape[-1] // 128)
            assert s_s.shape == expected_s_shape

        # Scale range for random normal data
        x3 = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        _, s3 = block_fp8_quant(x3, block_size=128)
        assert torch.all(s3 > 0)
        assert torch.all(s3 < 1.0)
        assert torch.all(s3 > 1e-6)

        # Max FP8 value (448.0) -> scale should be 1.0
        x_max = torch.full((2, 128), 448.0, device="cuda", dtype=torch.float32)
        _, s_max = block_fp8_quant(x_max, block_size=128)
        assert torch.allclose(s_max, torch.ones_like(s_max), atol=1e-5)

        # Different block sizes
        x4 = torch.randn(2, 512, device="cuda", dtype=torch.float32)
        for bs in [64, 128, 256]:
            y4, s4 = block_fp8_quant(x4, block_size=bs)
            assert y4.shape == x4.shape
            assert s4.shape == (2, 512 // bs)

    def test_quantization_input_requirements(self):
        """Contiguity and divisibility requirements."""
        # Non-contiguous tensor should fail
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        with pytest.raises(AssertionError):
            block_fp8_quant(x.t(), block_size=128)

        # Non-divisible size should fail
        x2 = torch.randn(4, 130, device="cuda", dtype=torch.float32)
        with pytest.raises(AssertionError):
            block_fp8_quant(x2, block_size=128)


class TestBlockFP8Dequantization:
    """Comprehensive tests for dequantization accuracy and roundtrip."""

    def test_dequantization_accuracy_and_roundtrip(self):
        """Dequantization accuracy, shape preservation, and roundtrip across shapes."""
        # Basic accuracy
        x_orig = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        y, s = block_fp8_quant(x_orig, block_size=128)
        x_dequant = block_fp8_dequant(y, s, block_size=128)
        assert x_dequant.shape == x_orig.shape
        assert x_dequant.dtype == torch.float32
        relative_error = torch.abs(x_dequant - x_orig) / (torch.abs(x_orig) + 1e-6)
        assert relative_error.mean().item() < 0.05

        # Roundtrip across shapes
        for shape in [(128,), (4, 256), (2, 3, 384)]:
            x_s = torch.randn(*shape, device="cuda", dtype=torch.float32)
            y_s, s_s = block_fp8_quant(x_s, block_size=128)
            x_d = block_fp8_dequant(y_s, s_s, block_size=128)
            assert x_d.shape == x_s.shape
            assert torch.allclose(x_d, x_s, rtol=0.1, atol=0.05)

        # Non-contiguous dequant should fail
        y2, s2 = block_fp8_quant(torch.randn(4, 256, device="cuda", dtype=torch.float32), block_size=128)
        with pytest.raises(AssertionError):
            block_fp8_dequant(y2.t(), s2, block_size=128)


class TestBlockFP8Integration:
    """Integration tests: determinism, memory efficiency, edge cases."""

    def test_determinism_memory_and_edge_cases(self):
        """Determinism, memory efficiency, edge cases (small/large/mixed values, single block)."""
        # Determinism
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        y1, s1 = block_fp8_quant(x, block_size=128)
        y2, s2 = block_fp8_quant(x, block_size=128)
        assert torch.equal(y1, y2)
        assert torch.equal(s1, s2)

        # Memory efficiency
        M, K = 1024, 2048
        x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        y_fp8, s = block_fp8_quant(x_fp32, block_size=128)
        fp32_bytes = x_fp32.element_size() * x_fp32.numel()
        fp8_bytes = y_fp8.element_size() * y_fp8.numel() + s.element_size() * s.numel()
        assert fp8_bytes < fp32_bytes

        # Very small values
        x_small = torch.full((2, 128), 1e-6, device="cuda", dtype=torch.float32)
        y_sm, s_sm = block_fp8_quant(x_small, block_size=128)
        x_d_sm = block_fp8_dequant(y_sm, s_sm, block_size=128)
        assert torch.allclose(x_d_sm, x_small, rtol=0.5, atol=1e-7)

        # Very large values
        x_large = torch.full((2, 128), 400.0, device="cuda", dtype=torch.float32)
        y_lg, s_lg = block_fp8_quant(x_large, block_size=128)
        x_d_lg = block_fp8_dequant(y_lg, s_lg, block_size=128)
        assert torch.allclose(x_d_lg, x_large, rtol=0.05, atol=1.0)

        # Mixed positive/negative - signs preserved
        x_mixed = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        x_mixed[0] = torch.abs(x_mixed[0])
        x_mixed[1] = -torch.abs(x_mixed[1])
        y_m, s_m = block_fp8_quant(x_mixed, block_size=128)
        x_d_m = block_fp8_dequant(y_m, s_m, block_size=128)
        assert torch.all((x_mixed >= 0) == (x_d_m >= 0))

        # Single block (minimum size)
        x_single = torch.randn(1, 128, device="cuda", dtype=torch.float32)
        y_sg, s_sg = block_fp8_quant(x_single, block_size=128)
        assert y_sg.shape == (1, 128)
        assert s_sg.shape == (1, 1)
        x_d_sg = block_fp8_dequant(y_sg, s_sg, block_size=128)
        assert torch.allclose(x_d_sg, x_single, rtol=0.1, atol=0.05)

        # 1D/2D consistency
        x_2d = torch.randn(256, 512, device="cuda", dtype=torch.float32)
        y_1d, s_1d = block_fp8_quant(x_2d, block_size=128)
        x_d_1d = block_fp8_dequant(y_1d, s_1d, block_size=128)
        assert x_d_1d.shape == x_2d.shape
        assert torch.allclose(x_d_1d, x_2d, rtol=0.1, atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
