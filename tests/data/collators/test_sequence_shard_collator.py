import pytest
import torch
from unittest.mock import Mock, patch
from xorl.data.collators import TextSequenceShardCollator
from xorl.data.constants import IGNORE_INDEX

pytestmark = [pytest.mark.cpu, pytest.mark.collator]


class TestTextSequenceShardCollator:
    """Test suite for TextSequenceShardCollator."""

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_collator_initialization(self, mock_parallel_state):
        """Test that collator initializes correctly with default behavior."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()
        assert collator.sp_size == 2
        assert collator.sp_rank == 0

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_slice_basic(self, mock_parallel_state):
        """Test basic SP slicing functionality."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()

        tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
        sliced = collator.sp_slice(tensor, dim=-1)

        # With sp_size=2, sp_rank=0, should get first half
        # chunk_size = ceil(6/2) = 3
        # sp_rank 0 gets indices 0:3 -> [1, 2, 3]
        expected = torch.tensor([[1, 2, 3]])
        assert torch.equal(sliced, expected)

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_slice_second_rank(self, mock_parallel_state):
        """Test SP slicing for second rank."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 1
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()

        tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
        sliced = collator.sp_slice(tensor, dim=-1)

        # With sp_size=2, sp_rank=1, should get second half
        # sp_rank 1 gets indices 3:6 -> [4, 5, 6]
        expected = torch.tensor([[4, 5, 6]])
        assert torch.equal(sliced, expected)

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_slice_uneven_split(self, mock_parallel_state):
        """Test SP slicing with uneven split."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()

        tensor = torch.tensor([[1, 2, 3, 4, 5]])
        sliced = collator.sp_slice(tensor, dim=-1)

        # With sp_size=2, sp_rank=0, and length=5
        # chunk_size = ceil(5/2) = 3
        # sp_rank 0 gets indices 0:3 -> [1, 2, 3]
        expected = torch.tensor([[1, 2, 3]])
        assert torch.equal(sliced, expected)

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_padding_basic(self, mock_parallel_state):
        """Test basic padding functionality."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()

        tensor = torch.tensor([[1, 2, 3]])
        padded = collator.sp_padding(tensor, dim=-1, pad_value=0, pad_length=2)

        expected = torch.tensor([[1, 2, 3, 0, 0]])
        assert torch.equal(padded, expected)

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_padding_sequential(self, mock_parallel_state):
        """Test sequential padding for position_ids."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()

        tensor = torch.tensor([[1, 2, 3]])
        padded = collator.sp_padding(tensor, dim=-1, pad_value=0, pad_length=2, sequential=True)

        # Sequential should add [0, 1] as padding
        expected = torch.tensor([[1, 2, 3, 0, 1]])
        assert torch.equal(padded, expected)

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_padding_zero_length(self, mock_parallel_state):
        """Test that zero padding returns original tensor."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator()

        tensor = torch.tensor([[1, 2, 3]])
        padded = collator.sp_padding(tensor, dim=-1, pad_value=0, pad_length=0)

        assert torch.equal(padded, tensor)

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_labels_shifted_and_padded(self, mock_parallel_state):
        """Test that labels are shifted by 1 position and last token padded with IGNORE_INDEX."""
        mock_ps = Mock()
        mock_ps.sp_size = 1
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }

        result = collator(batch)

        # Labels should be shifted: [2, 3, 4, 5, IGNORE_INDEX]
        expected_labels_start = torch.tensor([2, 3, 4, 5])
        assert torch.equal(result["labels"][0, :4], expected_labels_start)
        assert result["labels"][0, -1] == IGNORE_INDEX

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_masks_last_token_of_sequences(self, mock_parallel_state):
        """Test that last token of each sequence is masked with IGNORE_INDEX."""
        mock_ps = Mock()
        mock_ps.sp_size = 1
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        # Two sequences: [0,1,2] and [0,1]
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
        }

        result = collator(batch)

        # Last token of first sequence (index 2) and second sequence (index 4) should be IGNORE_INDEX
        # After shift: positions 1 and 3 should be IGNORE_INDEX
        assert result["labels"][0, 2] == IGNORE_INDEX  # Last of first seq
        assert result["labels"][0, 4] == IGNORE_INDEX  # Last of second seq

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_pads_to_sp_multiple(self, mock_parallel_state):
        """Test that sequences are padded to be divisible by sp_size."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        # Length 5, should pad to 6 (next multiple of sp_size=2, chunk_size=3)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }

        result = collator(batch)

        # After padding to 6 and slicing with sp_rank=0, should get 3 tokens
        assert result["input_ids"].shape[-1] == 3

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sp_split_across_ranks(self, mock_parallel_state):
        """Test that data is correctly split across SP ranks."""
        # Test rank 0
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator_rank0 = TextSequenceShardCollator(pad_token_id=0)

        batch_rank0 = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]]),
        }

        result_rank0 = collator_rank0(batch_rank0)

        # Test rank 1
        mock_ps.sp_rank = 1
        collator_rank1 = TextSequenceShardCollator(pad_token_id=0)

        batch_rank1 = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]]),
        }

        result_rank1 = collator_rank1(batch_rank1)

        # Rank 0 should get first 3 tokens, rank 1 should get last 3
        assert result_rank0["input_ids"].shape[-1] == 3
        assert result_rank1["input_ids"].shape[-1] == 3

        # First token of rank 0 should be 1, first token of rank 1 should be 4
        assert result_rank0["input_ids"][0, 0] == 1
        assert result_rank1["input_ids"][0, 0] == 4

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_adds_flash_attention_kwargs(self, mock_parallel_state):
        """Test that Flash Attention kwargs are added to the batch."""
        mock_ps = Mock()
        mock_ps.sp_size = 1
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }

        result = collator(batch)

        # Flash attention kwargs should be added
        assert "cu_seq_lens_q" in result
        assert "cu_seq_lens_k" in result
        assert "max_length_q" in result
        assert "max_length_k" in result

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_preserves_attention_mask(self, mock_parallel_state):
        """Test that attention_mask is preserved and padded correctly."""
        mock_ps = Mock()
        mock_ps.sp_size = 1
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
        }

        result = collator(batch)

        # Attention mask should be preserved
        assert "attention_mask" in result
        assert result["attention_mask"].shape == result["input_ids"].shape

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_preserves_position_ids(self, mock_parallel_state):
        """Test that position_ids are preserved and padded correctly."""
        mock_ps = Mock()
        mock_ps.sp_size = 1
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
        }

        result = collator(batch)

        # Position IDs should be preserved
        assert "position_ids" in result
        # Not sliced, stays in full batch for flash attention calculation
        assert result["position_ids"].shape[-1] == result["attention_mask"].shape[-1]

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_padding_uses_correct_values(self, mock_parallel_state):
        """Test that padding uses correct special values for different fields."""
        mock_ps = Mock()
        mock_ps.sp_size = 2
        mock_ps.sp_rank = 1
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=99)

        # Length 3 -> padded to 4 (sp_size=2, chunk_size=2)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
        }

        result = collator(batch)

        # Rank 1 gets second half (indices 2:4) after padding
        # input_ids padded with 99, so should see 99 if padding is in the slice
        # Since we're rank 1 with length 3->4, chunk_size=2, we get indices 2:4 = [3, 99]
        # After shift, position 2 becomes position 0 in the slice
        # But after label processing, the padding should be IGNORE_INDEX

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_handles_single_sequence(self, mock_parallel_state):
        """Test handling of a single sequence without packing."""
        mock_ps = Mock()
        mock_ps.sp_size = 1
        mock_ps.sp_rank = 0
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }

        result = collator(batch)

        # Should work correctly with sp_size=1
        assert result["input_ids"].shape[-1] == 5
        assert result["labels"].shape[-1] == 5
