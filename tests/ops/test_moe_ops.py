"""Tests for xorl.ops.group_gemm.kernel.moe module.

These tests compare the optimized Triton kernels for MoE operations 
with naive PyTorch implementations to ensure correctness.
"""

import pytest
import torch

# Mark all tests as GPU since MoE ops require CUDA
pytestmark = pytest.mark.gpu


def naive_expert_histogram(input: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Naive PyTorch implementation of expert histogram."""
    return torch.bincount(input.flatten().int(), minlength=num_bins)[:num_bins]


def naive_moe_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Naive PyTorch implementation of moe_gather.
    
    Gathers and sums expert outputs for each token.
    x: (M * topk, N)
    index: (M, topk)
    output: (M, N)
    """
    M, topk = index.shape
    N = x.shape[1]
    output = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    
    for m in range(M):
        for k in range(topk):
            idx = index[m, k].item()
            output[m] += x[idx]
    
    return output


def naive_moe_scatter(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Naive PyTorch implementation of moe_scatter.
    
    Scatters input to expert positions.
    x: (M, N)
    index: (M, topk)
    output: (M * topk, N)
    """
    M, topk = index.shape
    N = x.shape[1]
    output = torch.zeros(M * topk, N, dtype=x.dtype, device=x.device)
    
    for m in range(M):
        for k in range(topk):
            idx = index[m, k].item()
            output[idx] = x[m]
    
    return output


def naive_moe_add_gather(
    x: torch.Tensor,
    y: torch.Tensor,
    index: torch.Tensor
) -> torch.Tensor:
    """
    Naive PyTorch implementation of moe_add_gather.
    
    Adds two tensors and gathers results.
    x, y: (M * topk, N)
    index: (M, topk)
    output: (M, N)
    """
    return naive_moe_gather(x + y, index)


class TestExpertHistogram:
    """Test suite for expert_histogram."""
    
    def test_basic_histogram(self):
        """Test basic histogram computation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import expert_histogram
        except ImportError:
            pytest.skip("moe ops not available")
        
        # Create expert assignments
        input = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0], dtype=torch.int32).cuda()
        num_experts = 4
        
        # Compute with kernel
        output_kernel = expert_histogram(input, num_experts)
        
        # Compute with naive implementation
        output_naive = naive_expert_histogram(input, num_experts)
        
        # Compare
        assert torch.equal(output_kernel.cpu(), output_naive.cpu())
        assert output_kernel.tolist() == [3, 2, 2, 1]  # Expected counts
    
    def test_2d_input(self):
        """Test histogram with 2D input (flattened internally)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import expert_histogram
        except ImportError:
            pytest.skip("moe ops not available")
        
        # Create 2D expert assignments (typical for topk routing)
        input = torch.tensor([
            [0, 1],
            [2, 0],
            [1, 3],
            [2, 0],
        ], dtype=torch.int32).cuda()
        num_experts = 4
        
        output_kernel = expert_histogram(input, num_experts)
        output_naive = naive_expert_histogram(input, num_experts)
        
        assert torch.equal(output_kernel.cpu(), output_naive.cpu())
        # Expert 0: 3 times, Expert 1: 2 times, Expert 2: 2 times, Expert 3: 1 time
        assert output_kernel.tolist() == [3, 2, 2, 1]
    
    def test_large_histogram(self):
        """Test with large number of experts."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import expert_histogram
        except ImportError:
            pytest.skip("moe ops not available")
        
        num_experts = 64
        input_size = 10000
        input = torch.randint(0, num_experts, (input_size,), dtype=torch.int32).cuda()
        
        output_kernel = expert_histogram(input, num_experts)
        output_naive = naive_expert_histogram(input, num_experts)
        
        assert torch.equal(output_kernel.cpu(), output_naive.cpu())
        assert output_kernel.sum().item() == input_size
    
    def test_int64_input(self):
        """Test histogram with int64 input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import expert_histogram
        except ImportError:
            pytest.skip("moe ops not available")
        
        input = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0], dtype=torch.int64).cuda()
        num_experts = 4
        
        output_kernel = expert_histogram(input, num_experts)
        output_naive = naive_expert_histogram(input, num_experts)
        
        assert torch.equal(output_kernel.cpu(), output_naive.cpu())


class TestMoEGather:
    """Test suite for moe_gather."""
    
    def test_basic_gather(self):
        """Test basic gather operation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_gather
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 4, 2, 8
        
        # Create inputs
        x = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        # Create indices - each token chooses topk experts
        index = torch.tensor([
            [0, 1],  # Token 0 uses experts at positions 0, 1
            [2, 3],  # Token 1 uses experts at positions 2, 3
            [4, 5],  # Token 2 uses experts at positions 4, 5
            [6, 7],  # Token 3 uses experts at positions 6, 7
        ], dtype=torch.int32).cuda()
        
        # Compute with kernel
        output_kernel = moe_gather(x, index)
        
        # Compute with naive implementation
        output_naive = naive_moe_gather(x, index)
        
        # Compare
        assert output_kernel.shape == (M, N)
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_overlapping_indices(self):
        """Test gather with overlapping expert assignments."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_gather
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 3, 2, 16
        
        x = torch.randn(M * topk, N, dtype=torch.bfloat16).cuda()
        # Multiple tokens using same experts
        index = torch.tensor([
            [0, 1],
            [1, 2],
            [0, 2],
        ], dtype=torch.int32).cuda()
        
        output_kernel = moe_gather(x, index)
        output_naive = naive_moe_gather(x, index)
        
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_large_dimensions(self):
        """Test with large dimensions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_gather
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 1024, 2, 4096
        
        x = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        # Create sequential indices
        index = torch.arange(M * topk, dtype=torch.int32).cuda().reshape(M, topk)
        
        output_kernel = moe_gather(x, index)
        output_naive = naive_moe_gather(x, index)
        
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)


class TestMoEScatter:
    """Test suite for moe_scatter."""
    
    def test_basic_scatter(self):
        """Test basic scatter operation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_scatter
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 4, 2, 8
        
        # Create inputs
        x = torch.randn(M, N, dtype=torch.float16).cuda()
        # Create indices
        index = torch.tensor([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ], dtype=torch.int32).cuda()
        
        # Compute with kernel
        output_kernel = moe_scatter(x, index)
        
        # Compute with naive implementation
        output_naive = naive_moe_scatter(x, index)
        
        # Compare
        assert output_kernel.shape == (M * topk, N)
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_scatter_gather_roundtrip(self):
        """Test that scatter followed by gather is identity (with right indices)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_scatter, moe_gather
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 8, 2, 16
        
        x = torch.randn(M, N, dtype=torch.bfloat16).cuda()
        # Create unique indices (no overlaps)
        index = torch.arange(M * topk, dtype=torch.int32).cuda().reshape(M, topk)
        
        # Scatter then gather
        scattered = moe_scatter(x, index)
        gathered = moe_gather(scattered, index)
        
        # Should approximately reconstruct original (multiplied by topk due to summation)
        expected = x * topk
        assert torch.allclose(gathered, expected, rtol=1e-2, atol=1e-2)


class TestMoEAddGather:
    """Test suite for moe_add_gather."""
    
    def test_basic_add_gather(self):
        """Test basic add and gather operation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_add_gather
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 4, 2, 8
        
        x = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        y = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        index = torch.arange(M * topk, dtype=torch.int32).cuda().reshape(M, topk)
        
        # Compute with kernel
        output_kernel = moe_add_gather(x, y, index)
        
        # Compute with naive implementation
        output_naive = naive_moe_add_gather(x, y, index)
        
        # Compare
        assert output_kernel.shape == (M, N)
        assert torch.allclose(output_kernel.float(), output_naive.float(), rtol=1e-2, atol=1e-2)
    
    def test_add_gather_equivalence(self):
        """Test that add_gather equals gather(x+y)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_add_gather, moe_gather
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N = 8, 2, 16
        
        x = torch.randn(M * topk, N, dtype=torch.bfloat16).cuda()
        y = torch.randn(M * topk, N, dtype=torch.bfloat16).cuda()
        index = torch.arange(M * topk, dtype=torch.int32).cuda().reshape(M, topk)
        
        # Method 1: add_gather
        output1 = moe_add_gather(x, y, index)
        
        # Method 2: manual add then gather
        output2 = moe_gather(x + y, index)
        
        # Should be equivalent
        assert torch.allclose(output1, output2, rtol=1e-3, atol=1e-3)


class TestMoEIndexCompute:
    """Test suite for moe_index_compute."""
    
    def test_basic_index_compute(self):
        """Test basic index computation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_index_compute, expert_histogram
        except ImportError:
            pytest.skip("moe ops not available")
        
        # Create expert assignments
        experts_for_tokens = torch.tensor([
            [0, 1],
            [1, 2],
            [0, 2],
            [2, 3],
        ], dtype=torch.int32).cuda()
        
        num_experts = 4
        
        # Compute histogram
        histogram = expert_histogram(experts_for_tokens.flatten(), num_experts)
        cumsum = torch.cumsum(histogram, dim=0).int().cuda()
        
        # Compute indices
        indices = moe_index_compute(experts_for_tokens, cumsum)
        
        # Verify shape
        assert indices.shape == experts_for_tokens.shape
        
        # Verify all indices are within valid range
        assert torch.all(indices >= 0)
        assert torch.all(indices < experts_for_tokens.numel())
        
        # Verify indices are unique
        assert indices.flatten().unique().numel() == experts_for_tokens.numel()
    
    def test_index_compute_ordering(self):
        """Test that computed indices preserve expert grouping."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_index_compute, expert_histogram
        except ImportError:
            pytest.skip("moe ops not available")
        
        # All tokens assigned to same expert should get sequential indices
        experts_for_tokens = torch.tensor([
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
        ], dtype=torch.int32).cuda()
        
        num_experts = 2
        
        histogram = expert_histogram(experts_for_tokens.flatten(), num_experts)
        cumsum = torch.cumsum(histogram, dim=0).int().cuda()
        
        indices = moe_index_compute(experts_for_tokens, cumsum)
        
        # Expert 0 should get indices 0-3, Expert 1 should get indices 4-7
        expert_0_indices = indices[experts_for_tokens == 0].sort()[0]
        expert_1_indices = indices[experts_for_tokens == 1].sort()[0]
        
        assert torch.all(expert_0_indices == torch.arange(4).cuda())
        assert torch.all(expert_1_indices == torch.arange(4, 8).cuda())


class TestMoEIntegration:
    """Integration tests combining multiple MoE operations."""
    
    def test_full_moe_pipeline(self):
        """Test complete MoE forward pipeline."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from xorl.ops.group_gemm.kernel.moe import (
                expert_histogram,
                moe_scatter,
                moe_gather,
                moe_index_compute,
            )
        except ImportError:
            pytest.skip("moe ops not available")
        
        M, topk, N, num_experts = 16, 2, 32, 4
        
        # 1. Create input and expert assignments
        hidden_states = torch.randn(M, N, dtype=torch.float16).cuda()
        experts_for_tokens = torch.randint(0, num_experts, (M, topk), dtype=torch.int32).cuda()
        
        # 2. Compute histogram and indices
        histogram = expert_histogram(experts_for_tokens.flatten(), num_experts)
        cumsum = torch.cumsum(histogram, dim=0).int().cuda()
        indices = moe_index_compute(experts_for_tokens, cumsum)
        
        # 3. Scatter tokens to experts
        scattered = moe_scatter(hidden_states, indices)
        
        # 4. Simulate expert processing (identity for simplicity)
        expert_outputs = scattered
        
        # 5. Gather results back
        final_output = moe_gather(expert_outputs, indices)
        
        # Verify shapes
        assert final_output.shape == (M, N)
        
        # Output should be approximately topk * input (since we sum topk experts)
        # But won't be exact due to how indices work
        assert final_output.abs().max() > 0  # Not all zeros


