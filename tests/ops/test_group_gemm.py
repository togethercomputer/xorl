"""Tests for xorl.ops.group_gemm module.

These tests compare the optimized Triton kernels with naive PyTorch implementations
to ensure correctness across various configurations.
"""

import pytest
import torch


# Grouped GEMM requires CUDA.  Skip unsupported hosts before entering the test
# body, but let failures importing XORL's own kernel module surface normally.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def naive_group_gemm_same_nk(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> torch.Tensor:
    """Naive PyTorch implementation of grouped GEMM with same N, K."""
    G = b.shape[0]
    N = b.shape[1] if transpose_b else b.shape[2]

    total_M = a.shape[1] if transpose_a else a.shape[0]
    output = torch.zeros(total_M, N, dtype=a.dtype, device=a.device)

    start_idx = 0
    for g in range(G):
        end_idx = cumsum_M[g].item()
        group_size = end_idx - start_idx
        if group_size == 0:
            continue
        if transpose_a:
            a_group = a[:, start_idx:end_idx].t()
        else:
            a_group = a[start_idx:end_idx, :]
        if transpose_b:
            b_group = b[g].t()
        else:
            b_group = b[g]
        output[start_idx:end_idx, :] = torch.matmul(a_group, b_group)
        start_idx = end_idx

    return output


def naive_group_gemm_same_mn(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_K: torch.Tensor,
    M: int,
    N: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> torch.Tensor:
    """Naive PyTorch implementation of grouped GEMM with same M, N."""
    G = cumsum_K.shape[0]
    output = torch.zeros(G, M, N, dtype=a.dtype, device=a.device)

    start_idx = 0
    for g in range(G):
        end_idx = cumsum_K[g].item()
        group_size = end_idx - start_idx
        if group_size == 0:
            continue
        if transpose_a:
            a_group = a[:, start_idx:end_idx].t()
            a_group = a_group.t()
        else:
            a_group = a[start_idx:end_idx, :]
            a_group = a_group.t()
        if transpose_b:
            b_group = b[:, start_idx:end_idx].t()
        else:
            b_group = b[start_idx:end_idx, :]
        output[g] = torch.matmul(a_group, b_group)
        start_idx = end_idx

    return output


class TestGroupGemmSameNK:
    """Numerical and input-policy contracts for ``group_gemm_same_nk``."""

    def test_same_nk_and_same_mn_comprehensive(self):
        """Aligned/unaligned numerics, transpose-B, and input rejection."""
        from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk  # noqa: PLC0415

        # --- Basic forward ---
        G, K, N = 4, 128, 256
        group_sizes = [10, 20, 15, 25]
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[: i + 1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)

        a = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b = torch.randn(G, K, N, dtype=torch.bfloat16).cuda()

        output_kernel = group_gemm_same_nk(a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False)
        output_naive = naive_group_gemm_same_nk(a, b, cumsum_M, transpose_a=False, transpose_b=False)
        assert output_kernel.shape == output_naive.shape
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)

        # --- Unequal group sizes ---
        G2, K2, N2 = 8, 70, 130
        gs2 = [5, 100, 2, 50, 30, 8, 45, 20]
        total_M2 = sum(gs2)
        cumsum_M2 = torch.tensor([sum(gs2[: i + 1]) for i in range(G2)], dtype=torch.int32).cuda()
        a2 = torch.randn(total_M2, K2, dtype=torch.float16).cuda()
        b2 = torch.randn(G2, K2, N2, dtype=torch.float16).cuda()
        out2 = group_gemm_same_nk(a2, b2, cumsum_M2, max(gs2))
        naive2 = naive_group_gemm_same_nk(a2, b2, cumsum_M2)
        assert torch.allclose(out2.float(), naive2.float(), rtol=1e-2, atol=1e-2)

        # --- Transpose B ---
        a3 = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b3 = torch.randn(G, N, K, dtype=torch.bfloat16).cuda()
        out3 = group_gemm_same_nk(a3, b3, cumsum_M, max_M, transpose_a=False, transpose_b=True)
        naive3 = naive_group_gemm_same_nk(a3, b3, cumsum_M, transpose_a=False, transpose_b=True)
        assert torch.allclose(out3.float(), naive3.float(), rtol=1e-2, atol=1e-2)

        # Input-policy guards use the same admitted shape.
        a_nc = torch.randn(total_M2, K2 * 2, dtype=torch.float16).cuda()[:, ::2]
        with pytest.raises(AssertionError, match="Not implemented: Noncontiguous input"):
            group_gemm_same_nk(a_nc, b2, cumsum_M2, max(gs2))

        with pytest.raises(AssertionError, match="a.device.*b.device"):
            group_gemm_same_nk(a2, b2.cpu(), cumsum_M2, max(gs2))

        TestGroupGemmSameMN()._assert_same_mn_comprehensive()


class TestGroupGemmSameMN:
    """Test suite for group_gemm_same_mn: basic, unequal groups, zero-K, single group."""

    def _assert_same_mn_comprehensive(self):
        """Basic forward, unequal K dims, zero-K group, single group."""
        from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn  # noqa: PLC0415

        # --- Basic forward ---
        G, M, N = 4, 128, 256
        group_Ks = [64, 128, 96, 112]
        total_K = sum(group_Ks)
        cumsum_K = torch.tensor([sum(group_Ks[: i + 1]) for i in range(G)], dtype=torch.int32).cuda()

        a = torch.randn(total_K, M, dtype=torch.bfloat16).cuda()
        b = torch.randn(total_K, N, dtype=torch.bfloat16).cuda()
        c = torch.empty(G, M, N, dtype=torch.bfloat16).cuda()
        group_gemm_same_mn(a, b, c, cumsum_K, max(group_Ks), transpose_a=True)
        naive = naive_group_gemm_same_mn(a, b, cumsum_K, M, N)
        assert c.shape == naive.shape
        assert torch.allclose(c.float(), naive.float(), rtol=1e-2, atol=1e-2)

        # --- Unequal K dims ---
        G2, M2, N2 = 8, 64, 128
        gKs2 = [10, 200, 5, 100, 50, 15, 90, 30]
        total_K2 = sum(gKs2)
        cumsum_K2 = torch.tensor([sum(gKs2[: i + 1]) for i in range(G2)], dtype=torch.int32).cuda()
        a2 = torch.randn(total_K2, M2, dtype=torch.float16).cuda()
        b2 = torch.randn(total_K2, N2, dtype=torch.float16).cuda()
        c2 = torch.empty(G2, M2, N2, dtype=torch.float16).cuda()
        group_gemm_same_mn(a2, b2, c2, cumsum_K2, max(gKs2), transpose_a=True)
        naive2 = naive_group_gemm_same_mn(a2, b2, cumsum_K2, M2, N2)
        assert torch.allclose(c2.float(), naive2.float(), rtol=1e-2, atol=1e-2)

        # --- Zero-K group ---
        G3, M3, N3 = 4, 64, 128
        gKs3 = [64, 0, 96, 32]
        total_K3 = sum(gKs3)
        cumsum_K3 = torch.tensor([sum(gKs3[: i + 1]) for i in range(G3)], dtype=torch.int32).cuda()
        a3 = torch.randn(total_K3, M3, dtype=torch.bfloat16).cuda()
        b3 = torch.randn(total_K3, N3, dtype=torch.bfloat16).cuda()
        c3 = torch.empty(G3, M3, N3, dtype=torch.bfloat16).cuda()
        group_gemm_same_mn(a3, b3, c3, cumsum_K3, max(gKs3), transpose_a=True)
        naive3 = naive_group_gemm_same_mn(a3, b3, cumsum_K3, M3, N3)
        assert torch.all(c3[1] == 0)
        assert torch.allclose(c3.float(), naive3.float(), rtol=1e-2, atol=1e-2)
