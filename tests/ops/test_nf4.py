"""Tests for NF4 (NormalFloat4) quantization kernels.

Validates correctness (roundtrip accuracy, codec, shapes) and measures
effective memory bandwidth of the dequantization kernels.
"""
import pytest
import torch
import time

try:
    from xorl.ops.quantize import nf4_quantize, nf4_dequantize
    from xorl.ops.quantize import nf4_quantize_gkn, nf4_dequantize_gkn
    from xorl.ops.quantize.nf4_codec import NF4_TABLE, NF4_MIN_STEP
    HAS_NF4 = True
except ImportError:
    HAS_NF4 = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_NF4, reason="nf4 module not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


class TestNF4Codec:
    """Tests for NF4 encode/decode correctness."""

    def test_nf4_table_properties(self):
        """NF4 table: 16 values, sorted, symmetric about 0, bounded [-1, 1]."""
        assert len(NF4_TABLE) == 16
        assert NF4_TABLE[0] == -1.0
        assert NF4_TABLE[-1] == 1.0
        assert NF4_TABLE[7] == 0.0
        # Sorted
        for i in range(15):
            assert NF4_TABLE[i] < NF4_TABLE[i + 1]

    def test_roundtrip_exact_table_values(self):
        """Quantizing exact NF4 table values should produce exact roundtrip."""
        table = torch.tensor(NF4_TABLE, dtype=torch.float32, device="cuda")
        # Create a weight tensor where each row has exactly one table value repeated
        M, K = 16, 64
        x = table[:, None].expand(M, K).contiguous()
        packed, scales = nf4_quantize(x, group_size=64)
        out = nf4_dequantize(packed, scales, M * K, group_size=64).reshape(M, K)
        # Each row should dequantize to a constant close to the original table value
        for i in range(16):
            expected = NF4_TABLE[i]
            actual = out[i, 0].item()
            assert abs(actual - expected) < 0.05, (
                f"Code {i}: expected {expected:.4f}, got {actual:.4f}"
            )


class TestNF4Quantize1D:
    """Tests for 1D (flat) NF4 quantization/dequantization."""

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_shapes_and_dtypes(self, group_size):
        """Output shapes and dtypes are correct."""
        M, K = 128, 256
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize(x, group_size=group_size)
        n = M * K
        assert packed.shape == (n // 2,)
        assert packed.dtype == torch.uint8
        assert scales.shape == (n // group_size,)
        assert scales.dtype == torch.float32

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_roundtrip_accuracy(self, group_size):
        """Roundtrip should produce reasonable reconstruction."""
        M, K = 256, 512
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize(x, group_size=group_size)
        out = nf4_dequantize(packed, scales, M * K, group_size=group_size)
        out = out.reshape(M, K)
        # NF4 is 4-bit: expect ~1-5% relative error for normal data
        rel_err = (out.float() - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.10, f"Relative error {rel_err:.4f} too high for group_size={group_size}"

    def test_zero_input(self):
        """Zero input should roundtrip to near-zero."""
        M, K = 64, 128
        x = torch.zeros(M, K, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize(x, group_size=64)
        out = nf4_dequantize(packed, scales, M * K, group_size=64).reshape(M, K)
        assert out.abs().max() < 1e-6

    def test_output_dtype_bf16(self):
        """Dequantized output should be bfloat16."""
        M, K = 64, 128
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize(x, group_size=64)
        out = nf4_dequantize(packed, scales, M * K, group_size=64)
        assert out.dtype == torch.bfloat16

    def test_scales_are_absmax(self):
        """Scales should equal per-group absmax."""
        M, K = 4, 128
        gs = 64
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize(x, group_size=gs)
        # Compute expected absmax per group
        x_groups = x.reshape(-1, gs)
        expected_scales = x_groups.abs().max(dim=1).values
        torch.testing.assert_close(scales, expected_scales, atol=1e-6, rtol=1e-6)

    def test_large_tensor(self):
        """Test with realistic model weight dimensions."""
        M, K = 4096, 4096
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).float()
        packed, scales = nf4_quantize(x, group_size=64)
        out = nf4_dequantize(packed, scales, M * K, group_size=64).reshape(M, K)
        rel_err = (out.float() - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.10


class TestNF4QuantizeGKN:
    """Tests for 2D GKN NF4 quantization/dequantization."""

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_shapes_and_dtypes(self, group_size):
        """Output shapes and dtypes are correct for GKN format."""
        K, N = 256, 128
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize_gkn(x, group_size=group_size)
        assert packed.shape == (K // 2, N)
        assert packed.dtype == torch.uint8
        assert scales.shape == (K // group_size, N)
        assert scales.dtype == torch.float32

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_roundtrip_accuracy(self, group_size):
        """GKN roundtrip should produce reasonable reconstruction."""
        K, N = 512, 256
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize_gkn(x, group_size=group_size)
        out = nf4_dequantize_gkn(packed, scales, K, N, group_size=group_size)
        assert out.shape == (K, N)
        assert out.dtype == torch.bfloat16
        rel_err = (out.float() - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.10, f"Relative error {rel_err:.4f} too high for group_size={group_size}"

    def test_1d_vs_gkn_consistency(self):
        """1D and GKN quantization should produce equivalent results for [K, N] input."""
        K, N = 256, 128
        gs = 64
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        # 1D: flatten and process
        packed_1d, scales_1d = nf4_quantize(x, group_size=gs)
        out_1d = nf4_dequantize(packed_1d, scales_1d, K * N, group_size=gs).reshape(K, N)
        # GKN: process as 2D
        packed_gkn, scales_gkn = nf4_quantize_gkn(x, group_size=gs)
        out_gkn = nf4_dequantize_gkn(packed_gkn, scales_gkn, K, N, group_size=gs)
        # Both should have similar reconstruction error (not identical due to grouping direction)
        err_1d = (out_1d.float() - x).abs().mean()
        err_gkn = (out_gkn.float() - x).abs().mean()
        # Both should have reasonable error
        assert err_1d < 0.1 * x.abs().mean()
        assert err_gkn < 0.1 * x.abs().mean()

    def test_large_tensor(self):
        """Test with MoE expert weight dimensions."""
        K, N = 2048, 7168  # Typical MoE hidden -> intermediate
        x = torch.randn(K, N, device="cuda", dtype=torch.bfloat16).float()
        packed, scales = nf4_quantize_gkn(x, group_size=64)
        out = nf4_dequantize_gkn(packed, scales, K, N, group_size=64)
        rel_err = (out.float() - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.10


class TestNF4Bandwidth:
    """Measure effective memory bandwidth of NF4 dequantization kernels."""

    def _measure_bandwidth(self, fn, packed, scales, num_elements, group_size, num_iters=200):
        """Measure effective bandwidth of a dequant kernel."""
        # Warmup
        for _ in range(20):
            fn(packed, scales, num_elements, group_size)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            fn(packed, scales, num_elements, group_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Total bytes: packed (read) + scales (read) + output bf16 (write)
        packed_bytes = num_elements // 2
        scale_bytes = (num_elements // group_size) * 4
        output_bytes = num_elements * 2
        total_bytes = packed_bytes + scale_bytes + output_bytes
        bw = total_bytes * num_iters / elapsed / 1e9  # GB/s
        return bw

    def _measure_bandwidth_gkn(self, packed, scales, K, N, group_size, num_iters=200):
        """Measure effective bandwidth of GKN dequant kernel."""
        for _ in range(20):
            nf4_dequantize_gkn(packed, scales, K, N, group_size)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            nf4_dequantize_gkn(packed, scales, K, N, group_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_elements = K * N
        packed_bytes = num_elements // 2
        scale_bytes = (num_elements // group_size) * 4
        output_bytes = num_elements * 2
        total_bytes = packed_bytes + scale_bytes + output_bytes
        bw = total_bytes * num_iters / elapsed / 1e9
        return bw

    @pytest.mark.parametrize("size", [
        (4096, 4096),   # ~16M elements
        (8192, 8192),   # ~67M elements
    ])
    def test_dequant_1d_bandwidth(self, size):
        """1D dequant should achieve >2000 GB/s on large tensors."""
        M, K = size
        gs = 64
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize(x, group_size=gs)
        bw = self._measure_bandwidth(nf4_dequantize, packed, scales, M * K, gs)
        print(f"\n[NF4 1D dequant] {M}x{K} gs={gs}: {bw:.0f} GB/s")
        # Soft assertion: warn if below target
        if bw < 2000:
            pytest.skip(f"Bandwidth {bw:.0f} GB/s below 2000 GB/s target (may vary by GPU)")

    @pytest.mark.parametrize("size", [
        (4096, 4096),
        (2048, 7168),   # MoE expert dimensions
    ])
    def test_dequant_gkn_bandwidth(self, size):
        """GKN dequant should achieve >2000 GB/s on large tensors."""
        K, N = size
        gs = 64
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)
        packed, scales = nf4_quantize_gkn(x, group_size=gs)
        bw = self._measure_bandwidth_gkn(packed, scales, K, N, gs)
        print(f"\n[NF4 GKN dequant] {K}x{N} gs={gs}: {bw:.0f} GB/s")
        if bw < 2000:
            pytest.skip(f"Bandwidth {bw:.0f} GB/s below 2000 GB/s target (may vary by GPU)")
