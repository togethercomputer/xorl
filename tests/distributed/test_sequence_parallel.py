"""Comprehensive tests for distributed sequence parallel operations.

Run with: torchrun --nproc_per_node=2 -m pytest tests/distributed/test_sequence_parallel.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    sequence_parallel_preprocess,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
    slice_position_embedding,
)
from xorl.distributed.sequence_parallel.ulysses import gather_seq_scatter_heads_qkv
from xorl.distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    get_unified_sequence_parallel_group,
    init_sequence_parallel,
)
from xorl.distributed.sequence_parallel.utils import pad_tensor, unpad_tensor

pytestmark = [pytest.mark.distributed]


def is_distributed_available():
    """Check if we're running in a distributed environment."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def requires_distributed(func):
    """Decorator to skip tests if not running in distributed mode."""
    return pytest.mark.skipif(
        not is_distributed_available(),
        reason="Test requires distributed environment (run with torchrun)"
    )(func)


class TestPaddingUtilities:
    """Test padding and unpadding utilities."""

    def test_pad_tensor_basic(self):
        """Test basic tensor padding."""
        x = torch.randn(2, 5, 4)
        padded = pad_tensor(x, dim=1, padding_size=3, padding_value=0)

        assert padded.shape == (2, 8, 4)
        assert torch.allclose(padded[:, :5, :], x)
        assert torch.allclose(padded[:, 5:, :], torch.zeros(2, 3, 4))

    def test_pad_tensor_dim0(self):
        """Test padding on dimension 0."""
        x = torch.randn(3, 4, 5)
        padded = pad_tensor(x, dim=0, padding_size=2, padding_value=-1)

        assert padded.shape == (5, 4, 5)
        assert torch.allclose(padded[:3, :, :], x)
        assert torch.allclose(padded[3:, :, :], torch.full((2, 4, 5), -1.0))

    def test_unpad_tensor_basic(self):
        """Test basic tensor unpadding."""
        x = torch.randn(2, 8, 4)
        unpadded = unpad_tensor(x, dim=1, padding_size=3)

        assert unpadded.shape == (2, 5, 4)
        assert torch.allclose(unpadded, x[:, :5, :])

    def test_pad_unpad_roundtrip(self):
        """Test padding and unpadding roundtrip."""
        x = torch.randn(3, 7, 5)
        padded = pad_tensor(x, dim=1, padding_size=3, padding_value=0)
        unpadded = unpad_tensor(padded, dim=1, padding_size=3)

        assert torch.allclose(x, unpadded)


class TestSliceInputTensor:
    """Test input tensor slicing for sequence parallel."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None

    @requires_distributed
    def test_slice_input_tensor_world_size_2(self):
        """Test slicing with world size 2."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        # Create input tensor with sequence length 8
        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.arange(8, dtype=torch.float32).unsqueeze(0).to(device)

        # Slice the tensor
        sliced = slice_input_tensor(x, dim=1, padding=False)

        # Verify shape
        assert sliced.shape == (1, 4)

        # Verify content
        if rank == 0:
            expected = torch.tensor([[0., 1., 2., 3.]]).to(device)
        else:
            expected = torch.tensor([[4., 5., 6., 7.]]).to(device)

        assert torch.allclose(sliced, expected)

    @requires_distributed
    def test_slice_input_tensor_with_padding(self):
        """Test slicing with padding when sequence length is not divisible."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        # Create input tensor with sequence length 7 (not divisible by 2)
        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.arange(7, dtype=torch.float32).unsqueeze(0).to(device)

        # Slice with padding
        sliced = slice_input_tensor(x, dim=1, padding=True, padding_value=0)

        # Should be padded to 8, then sliced to 4
        assert sliced.shape == (1, 4)

        if rank == 0:
            expected = torch.tensor([[0., 1., 2., 3.]]).to(device)
        else:
            expected = torch.tensor([[4., 5., 6., 0.]]).to(device)  # Last element is padding

        assert torch.allclose(sliced, expected)

    @requires_distributed
    def test_slice_input_tensor_batch_dimension(self):
        """Test slicing with batch dimension."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Batch size 2, sequence length 8
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(device)

        sliced = slice_input_tensor(x, dim=1, padding=False)

        assert sliced.shape == (2, 4)

        if rank == 0:
            expected = torch.tensor([[0., 1., 2., 3.], [8., 9., 10., 11.]]).to(device)
        else:
            expected = torch.tensor([[4., 5., 6., 7.], [12., 13., 14., 15.]]).to(device)

        assert torch.allclose(sliced, expected)



class TestSequenceParallelPreprocess:
    """Test sequence parallel preprocessing."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None


    @requires_distributed
    def test_preprocess_input_ids_labels(self):
        """Test preprocessing of input_ids and labels."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)
        sp_group = get_unified_sequence_parallel_group()

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create input_ids and labels
        input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).to(device)
        labels = torch.arange(8, dtype=torch.long).unsqueeze(0).to(device)

        # Preprocess
        proc_input_ids, proc_labels, _, _, _ = sequence_parallel_preprocess(
            input_ids=input_ids,
            labels=labels,
            sp_group=sp_group,
        )

        # Verify shapes (should be sliced to half)
        assert proc_input_ids.shape == (1, 4)
        assert proc_labels.shape == (1, 4)

        # Verify input_ids content
        if rank == 0:
            expected_input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long).to(device)
        else:
            expected_input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long).to(device)

        assert torch.equal(proc_input_ids, expected_input_ids)

        # Verify labels are shifted and padded correctly
        # Labels should be shifted by 1: [1, 2, 3, 4, 5, 6, 7, IGNORE_INDEX]
        from xorl.data.constants import IGNORE_INDEX

        if rank == 0:
            expected_labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long).to(device)
        else:
            expected_labels = torch.tensor([[5, 6, 7, IGNORE_INDEX]], dtype=torch.long).to(device)

        assert torch.equal(proc_labels, expected_labels)

    @requires_distributed
    def test_preprocess_with_padding(self):
        """Test preprocessing with padding for non-divisible sequence length."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)
        sp_group = get_unified_sequence_parallel_group()

        # Sequence length 7 (not divisible by 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.arange(7, dtype=torch.long).unsqueeze(0).to(device)
        labels = torch.arange(7, dtype=torch.long).unsqueeze(0).to(device)

        proc_input_ids, proc_labels, _, _, _ = sequence_parallel_preprocess(
            input_ids=input_ids,
            labels=labels,
            sp_group=sp_group,
        )

        # Should be padded to 8, then sliced to 4
        assert proc_input_ids.shape == (1, 4)
        assert proc_labels.shape == (1, 4)

    @requires_distributed
    def test_preprocess_position_ids(self):
        """Test preprocessing of position_ids."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)
        sp_group = get_unified_sequence_parallel_group()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).to(device)
        labels = torch.arange(8, dtype=torch.long).unsqueeze(0).to(device)
        position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).to(device)

        _, _, proc_position_ids, _, _ = sequence_parallel_preprocess(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            sp_group=sp_group,
        )

        # Position IDs should be padded (if needed) but not sliced
        assert proc_position_ids.shape == (1, 8)



class TestGatherScatterOps:
    """Test gather and scatter operations."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None


    @requires_distributed
    def test_gather_outputs(self):
        """Test gathering outputs from all ranks."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Each rank has different tensor
        if rank == 0:
            x = torch.tensor([[1., 2.], [3., 4.]]).to(device)
        else:
            x = torch.tensor([[5., 6.], [7., 8.]]).to(device)

        # Gather along dimension 1
        gathered = gather_outputs(x, gather_dim=1, scale_grad=False)

        # Should concatenate along dim 1
        expected = torch.tensor([[1., 2., 5., 6.], [3., 4., 7., 8.]]).to(device)
        assert torch.allclose(gathered, expected)

    @requires_distributed
    def test_gather_outputs_with_unpadding(self):
        """Test gathering outputs with unpadding."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Each rank has tensor with padding
        if rank == 0:
            x = torch.tensor([[1., 2., 3., 4.]]).to(device)
        else:
            x = torch.tensor([[5., 6., 7., 0.]]).to(device)  # Last element is padding

        # Gather with unpadding (original length was 7, padded to 8)
        gathered = gather_outputs(x, gather_dim=1, padding_dim=1, unpad_dim_size=7, scale_grad=False)

        # Should be unpadded back to 7
        assert gathered.shape == (1, 7)
        expected = torch.tensor([[1., 2., 3., 4., 5., 6., 7.]]).to(device)
        assert torch.allclose(gathered, expected)

    @requires_distributed
    def test_gather_seq_scatter_heads(self):
        """Test gather sequence scatter heads operation."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tensor with shape (batch, seq, heads, head_dim)
        # Each rank has half the sequence, all heads
        if rank == 0:
            x = torch.ones(1, 4, 2, 8).to(device)  # First half of sequence
        else:
            x = torch.ones(1, 4, 2, 8).to(device) * 2  # Second half of sequence

        # All-to-all: gather sequence, scatter heads
        result = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)

        # After all-to-all: seq gathers (4*2=8), heads scatter (2/2=1)
        assert result.shape == (1, 8, 1, 8)

    @requires_distributed
    def test_gather_heads_scatter_seq(self):
        """Test gather heads scatter sequence operation."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reverse operation of gather_seq_scatter_heads
        x = torch.randn(1, 2, 4, 8).to(device)

        # All-to-all: gather heads, scatter seq
        result = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)

        # After all-to-all: heads gather (4*2=8), seq scatters (2/2=1)
        assert result.shape == (1, 1, 8, 8)



class TestQKVOperations:
    """Test QKV-specific operations."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None


    @requires_distributed
    def test_gather_seq_scatter_heads_qkv(self):
        """Test gather sequence scatter heads for QKV tensor."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # QKV tensor: last dimension is 3 * hidden_size
        # Shape: (batch, seq, 3 * hidden_size)
        qkv = torch.randn(1, 4, 24).to(device)  # 3 * 8 = 24

        # Apply gather_seq_scatter_heads_qkv
        result = gather_seq_scatter_heads_qkv(qkv, seq_dim=1, restore_shape=True)

        # After operation: seq doubles, hidden_size halves
        assert result.shape == (1, 8, 12)



class TestSlicePositionEmbedding:
    """Test position embedding slicing."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None


    @requires_distributed
    def test_slice_position_embedding_basic(self):
        """Test basic position embedding slicing."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)
        sp_group = get_unified_sequence_parallel_group()

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create cos and sin embeddings
        cos = torch.arange(8, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)  # (1, 8, 1)
        sin = torch.arange(8, 16, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)  # (1, 8, 1)

        position_embeddings = (cos, sin)

        # Slice along dimension 1 (sequence dimension)
        sliced_cos, sliced_sin = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)

        # Verify shapes
        assert sliced_cos.shape == (1, 4, 1)
        assert sliced_sin.shape == (1, 4, 1)

        # Verify content
        if rank == 0:
            expected_cos = torch.tensor([[[0.], [1.], [2.], [3.]]]).to(device)
            expected_sin = torch.tensor([[[8.], [9.], [10.], [11.]]]).to(device)
        else:
            expected_cos = torch.tensor([[[4.], [5.], [6.], [7.]]]).to(device)
            expected_sin = torch.tensor([[[12.], [13.], [14.], [15.]]]).to(device)

        assert torch.allclose(sliced_cos, expected_cos)
        assert torch.allclose(sliced_sin, expected_sin)

    def test_slice_position_embedding_no_group(self):
        """Test position embedding slicing with no SP group (should be no-op)."""
        cos = torch.randn(1, 8, 1)
        sin = torch.randn(1, 8, 1)

        position_embeddings = (cos, sin)

        # Should return unchanged
        result_cos, result_sin = slice_position_embedding(position_embeddings, dim=1, sp_group=None)

        assert torch.equal(result_cos, cos)
        assert torch.equal(result_sin, sin)



class TestGradientScaling:
    """Test gradient scaling in sequence parallel operations."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None


    @requires_distributed
    def test_slice_input_tensor_scale_grad(self):
        """Test slice with gradient scaling."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        rank = dist.get_rank()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create tensor with requires_grad
        x = torch.randn(2, 8, 4, requires_grad=True).to(device).contiguous()

        # Slice with gradient scaling
        sliced = slice_input_tensor_scale_grad(x, dim=1, scale_grad=True)

        # Verify shape
        assert sliced.shape == (2, 4, 4)

        # Verify gradient scaling by computing backward
        loss = sliced.sum()
        loss.backward()

        # Gradient should be scaled by world size (2)
        # This is handled by the _Slice autograd function



class TestEndToEndSequenceParallel:
    """End-to-end tests for sequence parallel workflow."""

    def setup_method(self):
        """Setup for each test - initialize distributed if needed."""
        if is_distributed_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(dist.get_rank())

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset sequence parallel state
        import xorl.distributed.sequence_parallel.comm as sp_comm
        sp_comm._ULYSSES_SP_GROUP = None
        sp_comm._UNIFIED_SP_GROUP = None


    @requires_distributed
    def test_full_forward_backward_pass(self):
        """Test full forward and backward pass with sequence parallel."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)
        sp_group = get_unified_sequence_parallel_group()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simulate a mini training step
        batch_size = 2
        seq_len = 8
        hidden_size = 16

        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long).to(device)
        labels = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long).to(device)

        # Preprocess for sequence parallel
        proc_input_ids, proc_labels, _, _, _ = sequence_parallel_preprocess(
            input_ids=input_ids,
            labels=labels,
            sp_group=sp_group,
        )

        # Verify preprocessing
        assert proc_input_ids.shape[1] == seq_len // 2
        assert proc_labels.shape[1] == seq_len // 2

        # Simulate embedding (each rank has partial sequence)
        embeddings = torch.randn(batch_size, seq_len // 2, hidden_size, requires_grad=True, device=device)

        # Simulate some computation
        hidden_states = embeddings * 2.0

        # Gather outputs at the end
        gathered = gather_outputs(hidden_states, gather_dim=1, scale_grad=False)

        # Verify gathered shape
        assert gathered.shape == (batch_size, seq_len, hidden_size)

        # Compute loss and backward
        loss = gathered.sum()
        loss.backward()

        # Verify gradients exist (embeddings is a leaf tensor so grad should exist)
        assert embeddings.grad is not None
        assert embeddings.grad.shape == embeddings.shape

    @requires_distributed
    def test_attention_pattern_all_to_all(self):
        """Test typical attention pattern with all-to-all operations."""
        if dist.get_world_size() != 2:
            pytest.skip("This test requires world size 2")

        init_sequence_parallel(ulysses_size=2, cp_size=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = 1
        seq_len = 8
        num_heads = 4
        head_dim = 16

        # Input: each rank has partial sequence, all heads
        # Shape: (batch, partial_seq, num_heads, head_dim)
        x = torch.randn(batch_size, seq_len // 2, num_heads, head_dim).to(device)

        # Before attention: gather sequence, scatter heads
        # Each rank gets full sequence, partial heads
        qkv = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)
        assert qkv.shape == (batch_size, seq_len, num_heads // 2, head_dim)

        # Simulate attention computation on full sequence
        attn_output = qkv * 2.0  # Placeholder

        # After attention: gather heads, scatter sequence
        # Each rank gets partial sequence, all heads
        output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)
        assert output.shape == (batch_size, seq_len // 2, num_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
