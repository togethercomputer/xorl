"""Tests for xorl.ops.group_gemm module.

These tests compare the optimized Triton kernels with naive PyTorch implementations
to ensure correctness across various configurations.
"""

import pytest
import torch
from typing import List

# Mark all tests as GPU since group_gemm requires CUDA
pytestmark = pytest.mark.gpu


def naive_group_gemm_same_nk(
    a: torch.Tensor, b: torch.Tensor, cumsum_M: torch.Tensor,
    transpose_a: bool = False, transpose_b: bool = False,
) -> torch.Tensor:
    """Naive PyTorch implementation of grouped GEMM with same N, K."""
    G = b.shape[0]
    if transpose_b:
        N, K = b.shape[1], b.shape[2]
    else:
        K, N = b.shape[1], b.shape[2]

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
    a: torch.Tensor, b: torch.Tensor, cumsum_K: torch.Tensor,
    M: int, N: int, transpose_a: bool = False, transpose_b: bool = False,
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
    """Test suite for group_gemm_same_nk: basic, unequal groups, transpose_b, large dims, single group."""

    def test_same_nk_comprehensive(self):
        """Basic forward, unequal groups, transpose_b, large dims, single group."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")

        # --- Basic forward ---
        G, K, N = 4, 128, 256
        group_sizes = [10, 20, 15, 25]
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)

        a = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b = torch.randn(G, K, N, dtype=torch.bfloat16).cuda()

        output_kernel = group_gemm_same_nk(a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False)
        output_naive = naive_group_gemm_same_nk(a, b, cumsum_M, transpose_a=False, transpose_b=False)
        assert output_kernel.shape == output_naive.shape
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)

        # --- Unequal group sizes ---
        G2, K2, N2 = 8, 64, 128
        gs2 = [5, 100, 2, 50, 30, 8, 45, 20]
        total_M2 = sum(gs2)
        cumsum_M2 = torch.tensor([sum(gs2[:i+1]) for i in range(G2)], dtype=torch.int32).cuda()
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

        # --- Large dimensions ---
        G4, K4, N4 = 8, 4096, 14336
        gs4 = [512, 480, 520, 490, 510, 505, 495, 488]
        total_M4 = sum(gs4)
        cumsum_M4 = torch.tensor([sum(gs4[:i+1]) for i in range(G4)], dtype=torch.int32).cuda()
        a4 = torch.randn(total_M4, K4, dtype=torch.bfloat16).cuda()
        b4 = torch.randn(G4, K4, N4, dtype=torch.bfloat16).cuda()
        out4 = group_gemm_same_nk(a4, b4, cumsum_M4, max(gs4))
        naive4 = naive_group_gemm_same_nk(a4, b4, cumsum_M4)
        assert torch.allclose(out4.float(), naive4.float(), rtol=5e-2, atol=5e-2)

        # --- Single group ---
        M_sg, K_sg, N_sg = 64, 128, 256
        cumsum_sg = torch.tensor([M_sg], dtype=torch.int32).cuda()
        a_sg = torch.randn(M_sg, K_sg, dtype=torch.float16).cuda()
        b_sg = torch.randn(1, K_sg, N_sg, dtype=torch.float16).cuda()
        out_sg = group_gemm_same_nk(a_sg, b_sg, cumsum_sg, M_sg)
        expected_sg = torch.matmul(a_sg, b_sg[0])
        assert torch.allclose(out_sg.float(), expected_sg.float(), rtol=1e-2, atol=1e-2)


class TestGroupGemmSameMN:
    """Test suite for group_gemm_same_mn: basic, unequal groups, zero-K, single group."""

    def test_same_mn_comprehensive(self):
        """Basic forward, unequal K dims, zero-K group, single group."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn
        except ImportError:
            pytest.skip("group_gemm not available")

        # --- Basic forward ---
        G, M, N = 4, 128, 256
        group_Ks = [64, 128, 96, 112]
        total_K = sum(group_Ks)
        cumsum_K = torch.tensor([sum(group_Ks[:i+1]) for i in range(G)], dtype=torch.int32).cuda()

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
        cumsum_K2 = torch.tensor([sum(gKs2[:i+1]) for i in range(G2)], dtype=torch.int32).cuda()
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
        cumsum_K3 = torch.tensor([sum(gKs3[:i+1]) for i in range(G3)], dtype=torch.int32).cuda()
        a3 = torch.randn(total_K3, M3, dtype=torch.bfloat16).cuda()
        b3 = torch.randn(total_K3, N3, dtype=torch.bfloat16).cuda()
        c3 = torch.empty(G3, M3, N3, dtype=torch.bfloat16).cuda()
        group_gemm_same_mn(a3, b3, c3, cumsum_K3, max(gKs3), transpose_a=True)
        naive3 = naive_group_gemm_same_mn(a3, b3, cumsum_K3, M3, N3)
        assert torch.all(c3[1] == 0)
        assert torch.allclose(c3.float(), naive3.float(), rtol=1e-2, atol=1e-2)

        # --- Single group ---
        K_sg, M_sg, N_sg = 256, 64, 128
        cumsum_sg = torch.tensor([K_sg], dtype=torch.int32).cuda()
        a_sg = torch.randn(K_sg, M_sg, dtype=torch.float16).cuda()
        b_sg = torch.randn(K_sg, N_sg, dtype=torch.float16).cuda()
        c_sg = torch.empty(1, M_sg, N_sg, dtype=torch.float16).cuda()
        group_gemm_same_mn(a_sg, b_sg, c_sg, cumsum_sg, K_sg, transpose_a=True)
        expected_sg = torch.matmul(a_sg.t(), b_sg).unsqueeze(0)
        assert torch.allclose(c_sg.float(), expected_sg.float(), rtol=1e-2, atol=1e-2)


class TestGroupGemmProperties:
    """Test mathematical properties and edge cases: dtype support, contiguity, device consistency."""

    def test_properties(self):
        """Dtype support (float16/bfloat16), contiguity requirement, device consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")

        G, K, N = 2, 64, 128
        group_sizes = [32, 32]
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)

        # Dtype support
        for dtype in [torch.float16, torch.bfloat16]:
            a = torch.randn(total_M, K, dtype=dtype).cuda()
            b = torch.randn(G, K, N, dtype=dtype).cuda()
            output = group_gemm_same_nk(a, b, cumsum_M, max_M)
            assert output.dtype == dtype
            assert output.shape == (total_M, N)

        # Contiguity requirement
        a_nc = torch.randn(total_M, K * 2, dtype=torch.bfloat16).cuda()[:, ::2]
        b_ok = torch.randn(G, K, N, dtype=torch.bfloat16).cuda()
        with pytest.raises(AssertionError, match="Not implemented: Noncontiguous input"):
            group_gemm_same_nk(a_nc, b_ok, cumsum_M, max_M)

        # Device consistency
        a_gpu = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b_cpu = torch.randn(G, K, N, dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="a.device.*b.device"):
            group_gemm_same_nk(a_gpu, b_cpu, cumsum_M, max_M)
