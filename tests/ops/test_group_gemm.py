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
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> torch.Tensor:
    """
    Naive PyTorch implementation of grouped GEMM with same N, K.
    
    Args:
        a: Input tensor (total_M, K) or (K, total_M) if transposed
        b: Weight tensors (G, K, N) or (G, N, K) if transposed
        cumsum_M: Cumulative sum of M dimensions for each group
        transpose_a: Whether to transpose a
        transpose_b: Whether to transpose b
    
    Returns:
        Output tensor (total_M, N)
    """
    G = b.shape[0]
    if transpose_b:
        N, K = b.shape[1], b.shape[2]
    else:
        K, N = b.shape[1], b.shape[2]
    
    total_M = a.shape[1] if transpose_a else a.shape[0]
    output = torch.zeros(total_M, N, dtype=a.dtype, device=a.device)
    
    # Process each group
    start_idx = 0
    for g in range(G):
        end_idx = cumsum_M[g].item()
        group_size = end_idx - start_idx
        
        if group_size == 0:
            continue
        
        # Extract group's input
        if transpose_a:
            a_group = a[:, start_idx:end_idx].t()  # (group_size, K)
        else:
            a_group = a[start_idx:end_idx, :]  # (group_size, K)
        
        # Extract group's weight
        if transpose_b:
            b_group = b[g].t()  # (K, N)
        else:
            b_group = b[g]  # (K, N)
        
        # Compute matmul for this group
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
    """
    Naive PyTorch implementation of grouped GEMM with same M, N.
    
    Args:
        a: Input tensor (total_K, M) or (M, total_K) if transposed
        b: Weight tensors (total_K, N) or (N, total_K) if transposed
        cumsum_K: Cumulative sum of K dimensions for each group
        M: Output M dimension
        N: Output N dimension
        transpose_a: Whether to transpose a
        transpose_b: Whether to transpose b
    
    Returns:
        Output tensor (G, M, N)
    """
    G = cumsum_K.shape[0]
    output = torch.zeros(G, M, N, dtype=a.dtype, device=a.device)
    
    # Process each group
    start_idx = 0
    for g in range(G):
        end_idx = cumsum_K[g].item()
        group_size = end_idx - start_idx
        
        if group_size == 0:
            # Zero K dimension
            continue
        
        # Extract group's input
        if transpose_a:
            a_group = a[:, start_idx:end_idx].t()  # (group_size, M) -> need (M, group_size)
            a_group = a_group.t()
        else:
            a_group = a[start_idx:end_idx, :]  # (group_size, M)
            a_group = a_group.t()  # (M, group_size)
        
        # Extract group's weight
        if transpose_b:
            b_group = b[:, start_idx:end_idx].t()  # (N, group_size) -> (group_size, N)
        else:
            b_group = b[start_idx:end_idx, :]  # (group_size, N)
        
        # Compute matmul for this group: (M, K) @ (K, N) = (M, N)
        output[g] = torch.matmul(a_group, b_group)
        
        start_idx = end_idx
    
    return output


class TestGroupGemmSameNK:
    """Test suite for group_gemm_same_nk comparing with naive implementation."""
    
    def test_basic_forward(self):
        """Test basic forward pass against naive implementation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, K, N = 4, 128, 256
        group_sizes = [10, 20, 15, 25]
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)
        
        # Create inputs
        a = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b = torch.randn(G, K, N, dtype=torch.bfloat16).cuda()
        
        # Compute with kernel
        output_kernel = group_gemm_same_nk(
            a, b, cumsum_M, max_M,
            transpose_a=False,
            transpose_b=False,
        )
        
        # Compute with naive implementation
        output_naive = naive_group_gemm_same_nk(
            a, b, cumsum_M,
            transpose_a=False,
            transpose_b=False,
        )
        
        # Compare results
        assert output_kernel.shape == output_naive.shape
        # Use higher tolerance for bfloat16
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_unequal_group_sizes(self):
        """Test with very unequal group sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, K, N = 8, 64, 128
        group_sizes = [5, 100, 2, 50, 30, 8, 45, 20]  # Very unequal
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)
        
        a = torch.randn(total_M, K, dtype=torch.float16).cuda()
        b = torch.randn(G, K, N, dtype=torch.float16).cuda()
        
        output_kernel = group_gemm_same_nk(a, b, cumsum_M, max_M)
        output_naive = naive_group_gemm_same_nk(a, b, cumsum_M)
        
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_with_transpose_b(self):
        """Test with transposed B matrix."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, K, N = 4, 128, 256
        group_sizes = [10, 20, 15, 25]
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)
        
        a = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b = torch.randn(G, N, K, dtype=torch.bfloat16).cuda()  # Transposed shape
        
        output_kernel = group_gemm_same_nk(
            a, b, cumsum_M, max_M,
            transpose_a=False,
            transpose_b=True,
        )
        output_naive = naive_group_gemm_same_nk(
            a, b, cumsum_M,
            transpose_a=False,
            transpose_b=True,
        )
        
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_large_dimensions(self):
        """Test with large dimensions typical for LLMs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, K, N = 8, 4096, 14336  # Typical LLM dimensions
        group_sizes = [512, 480, 520, 490, 510, 505, 495, 488]
        total_M = sum(group_sizes)
        cumsum_M = torch.tensor([sum(group_sizes[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_M = max(group_sizes)
        
        a = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b = torch.randn(G, K, N, dtype=torch.bfloat16).cuda()
        
        output_kernel = group_gemm_same_nk(a, b, cumsum_M, max_M)
        output_naive = naive_group_gemm_same_nk(a, b, cumsum_M)
        
        # For large matrices, allow slightly higher tolerance
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=5e-2, atol=5e-2)
    
    def test_single_group(self):
        """Test with single group (should behave like regular matmul)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, M, K, N = 1, 64, 128, 256
        cumsum_M = torch.tensor([M], dtype=torch.int32).cuda()
        max_M = M
        
        a = torch.randn(M, K, dtype=torch.float16).cuda()
        b = torch.randn(G, K, N, dtype=torch.float16).cuda()
        
        output_kernel = group_gemm_same_nk(a, b, cumsum_M, max_M)
        output_expected = torch.matmul(a, b[0])
        
        assert torch.allclose(output_kernel.float(), output_expected.float(), rtol=1e-2, atol=1e-2)


class TestGroupGemmSameMN:
    """Test suite for group_gemm_same_mn comparing with naive implementation."""
    
    def test_basic_forward(self):
        """Test basic forward pass against naive implementation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, M, N = 4, 128, 256
        group_Ks = [64, 128, 96, 112]
        total_K = sum(group_Ks)
        cumsum_K = torch.tensor([sum(group_Ks[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_K = max(group_Ks)
        
        # Create inputs
        a = torch.randn(total_K, M, dtype=torch.bfloat16).cuda()
        b = torch.randn(total_K, N, dtype=torch.bfloat16).cuda()
        c = torch.empty(G, M, N, dtype=torch.bfloat16).cuda()
        
        # Compute with kernel — transpose_a=True because a is (total_K, M) layout
        group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=True)

        # Compute with naive implementation (handles transpose explicitly)
        output_naive = naive_group_gemm_same_mn(a, b, cumsum_K, M, N)

        # Compare results
        assert c.shape == output_naive.shape
        assert torch.allclose(c.float(), output_naive.float(), rtol=1e-2, atol=1e-2)

    def test_unequal_group_sizes(self):
        """Test with very unequal K dimensions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn
        except ImportError:
            pytest.skip("group_gemm not available")

        G, M, N = 8, 64, 128
        group_Ks = [10, 200, 5, 100, 50, 15, 90, 30]  # Very unequal
        total_K = sum(group_Ks)
        cumsum_K = torch.tensor([sum(group_Ks[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_K = max(group_Ks)

        a = torch.randn(total_K, M, dtype=torch.float16).cuda()
        b = torch.randn(total_K, N, dtype=torch.float16).cuda()
        c = torch.empty(G, M, N, dtype=torch.float16).cuda()

        group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=True)
        output_naive = naive_group_gemm_same_mn(a, b, cumsum_K, M, N)

        assert torch.allclose(c.float(), output_naive.float(), rtol=1e-2, atol=1e-2)

    def test_zero_k_dimension(self):
        """Test handling of zero K dimension for some groups."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn
        except ImportError:
            pytest.skip("group_gemm not available")

        G, M, N = 4, 64, 128
        group_Ks = [64, 0, 96, 32]  # Second group has K=0
        total_K = sum(group_Ks)
        cumsum_K = torch.tensor([sum(group_Ks[:i+1]) for i in range(G)], dtype=torch.int32).cuda()
        max_K = max(group_Ks)

        a = torch.randn(total_K, M, dtype=torch.bfloat16).cuda()
        b = torch.randn(total_K, N, dtype=torch.bfloat16).cuda()
        c = torch.empty(G, M, N, dtype=torch.bfloat16).cuda()

        group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=True)
        output_naive = naive_group_gemm_same_mn(a, b, cumsum_K, M, N)

        # Check that second group is zero
        assert torch.all(c[1] == 0)
        assert torch.allclose(c.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_single_group(self):
        """Test with single group."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn
        except ImportError:
            pytest.skip("group_gemm not available")
        
        G, M, N, K = 1, 64, 128, 256
        cumsum_K = torch.tensor([K], dtype=torch.int32).cuda()
        max_K = K
        
        a = torch.randn(K, M, dtype=torch.float16).cuda()
        b = torch.randn(K, N, dtype=torch.float16).cuda()
        c = torch.empty(G, M, N, dtype=torch.float16).cuda()
        
        group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=True)

        # Expected result: a^T @ b = (M, K) @ (K, N) = (M, N)
        output_expected = torch.matmul(a.t(), b).unsqueeze(0)
        
        assert torch.allclose(c.float(), output_expected.float(), rtol=1e-2, atol=1e-2)


class TestGroupGemmProperties:
    """Test mathematical properties and edge cases."""
    
    def test_dtype_support(self):
        """Test that only supported dtypes work."""
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
        
        # float16 and bfloat16 should work
        for dtype in [torch.float16, torch.bfloat16]:
            a = torch.randn(total_M, K, dtype=dtype).cuda()
            b = torch.randn(G, K, N, dtype=dtype).cuda()
            
            output = group_gemm_same_nk(a, b, cumsum_M, max_M)
            assert output.dtype == dtype
            assert output.shape == (total_M, N)
    
    def test_contiguity_requirement(self):
        """Test that inputs must be contiguous."""
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
        
        # Create non-contiguous tensors
        a = torch.randn(total_M, K * 2, dtype=torch.bfloat16).cuda()[:, ::2]  # Non-contiguous
        b = torch.randn(G, K, N, dtype=torch.bfloat16).cuda()
        
        # Should raise assertion error
        with pytest.raises(AssertionError, match="Not implemented: Noncontiguous input"):
            group_gemm_same_nk(a, b, cumsum_M, max_M)
    
    def test_device_consistency(self):
        """Test that all tensors must be on same device."""
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
        
        a = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()
        b = torch.randn(G, K, N, dtype=torch.bfloat16)  # CPU tensor
        
        # Should raise assertion error
        with pytest.raises(AssertionError, match="a.device.*b.device"):
            group_gemm_same_nk(a, b, cumsum_M, max_M)


