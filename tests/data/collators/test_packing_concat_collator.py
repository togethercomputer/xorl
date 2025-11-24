import pytest
import torch
from unittest.mock import Mock, patch
from xorl.data.collators import PackingConcatCollator, add_flash_attention_kwargs_from_position_ids

pytestmark = [pytest.mark.cpu, pytest.mark.collator]


class TestAddFlashAttentionKwargs:
    """Test suite for add_flash_attention_kwargs_from_position_ids function."""

    def test_adds_all_required_kwargs(self):
        """Test that all required flash attention kwargs are added to batch."""
        # Create a batch with position_ids
        # Position IDs: [0, 1, 2, 0, 1] represents two sequences: [0,1,2] and [0,1]
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
        }

        cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k = add_flash_attention_kwargs_from_position_ids(batch)

        # Check that new keys are added
        assert "cu_seq_lens_q" in batch
        assert "cu_seq_lens_k" in batch
        assert "max_length_q" in batch
        assert "max_length_k" in batch

    def test_cu_seq_lens_correctness(self):
        """Test that cumulative sequence lengths are calculated correctly."""
        batch = {
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
        }

        cu_seq_lens_q, _, _, _ = add_flash_attention_kwargs_from_position_ids(batch)

        # Two sequences: [0,1,2] (length 3) and [0,1,2,3] (length 4)
        # cu_seqlens should be [0, 3, 7]
        expected_cu_seq_lens = torch.tensor([0, 3, 7], dtype=torch.int32)
        assert torch.equal(cu_seq_lens_q, expected_cu_seq_lens)

    def test_max_length_correctness(self):
        """Test that max_length is calculated correctly."""
        batch = {
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
        }

        _, _, max_length_q, max_length_k = add_flash_attention_kwargs_from_position_ids(batch)

        # Maximum sequence length is 4
        assert max_length_q == 4
        assert max_length_k == 4

    def test_single_sequence(self):
        """Test with a single sequence."""
        batch = {
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }

        cu_seq_lens_q, _, max_length_q, _ = add_flash_attention_kwargs_from_position_ids(batch)

        expected_cu_seq_lens = torch.tensor([0, 5], dtype=torch.int32)
        assert torch.equal(cu_seq_lens_q, expected_cu_seq_lens)
        assert max_length_q == 5


class TestPackingConcatCollator:
    """Test suite for PackingConcatCollator."""

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_basic_concatenation(self, mock_parallel_state, sample_packed_features):
        """Test that features are concatenated correctly."""
        # Mock parallel state to disable SP
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator()
        batch = collator(sample_packed_features)

        # Check shapes
        assert batch["input_ids"].shape == (1, 10)  # (batch_size=1, total_seq_len=10)
        assert batch["attention_mask"].shape == (1, 10)
        assert batch["labels"].shape == (1, 10)
        assert batch["position_ids"].shape == (1, 10)

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_concatenation_values(self, mock_parallel_state, sample_packed_features):
        """Test that values are concatenated in the correct order."""
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator()
        batch = collator(sample_packed_features)

        expected_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        expected_position_ids = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]])

        assert torch.equal(batch["input_ids"], expected_input_ids)
        assert torch.equal(batch["position_ids"], expected_position_ids)

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_adds_position_ids_if_missing(self, mock_parallel_state, sample_features):
        """Test that position_ids are generated if not provided."""
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator()
        batch = collator(sample_features)

        # Position IDs should be generated as sequential indices per feature
        # Each feature gets its own sequence: [0,1,2,3,4] + [0,1,2,3,4]
        assert "position_ids" in batch
        expected_position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
        assert torch.equal(batch["position_ids"], expected_position_ids)

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_flash_attention_kwargs_added_when_sp_disabled(self, mock_parallel_state, sample_packed_features):
        """Test that Flash Attention kwargs are added when SP is disabled."""
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator()
        batch = collator(sample_packed_features)

        assert "cu_seq_lens_q" in batch
        assert "cu_seq_lens_k" in batch
        assert "max_length_q" in batch
        assert "max_length_k" in batch

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_flash_attention_kwargs_not_added_when_sp_enabled(self, mock_parallel_state, sample_packed_features):
        """Test that Flash Attention kwargs are not added when SP is enabled."""
        mock_ps = Mock()
        mock_ps.sp_enabled = True
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator()
        batch = collator(sample_packed_features)

        # These should NOT be in the batch when SP is enabled
        assert "cu_seq_lens_q" not in batch
        assert "cu_seq_lens_k" not in batch
        assert "max_length_q" not in batch
        assert "max_length_k" not in batch

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_single_feature(self, mock_parallel_state):
        """Test with a single feature."""
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
                "position_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            }
        ]

        collator = PackingConcatCollator()
        batch = collator(features)

        assert batch["input_ids"].shape == (1, 3)
        assert batch["position_ids"].shape == (1, 3)

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_handles_extra_fields(self, mock_parallel_state):
        """Test that extra fields are handled with default_collate."""
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
                "position_ids": torch.tensor([0, 1, 2], dtype=torch.long),
                "extra_field": torch.tensor([100], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5, 6], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([4, 5, 6], dtype=torch.long),
                "position_ids": torch.tensor([0, 1, 2], dtype=torch.long),
                "extra_field": torch.tensor([200], dtype=torch.long),
            }
        ]

        collator = PackingConcatCollator()
        batch = collator(features)

        # Extra field should be collated as a batch
        assert "extra_field" in batch
        assert batch["extra_field"].shape == (2, 1)

    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_multiple_sequences_per_sample(self, mock_parallel_state):
        """Test with multiple sequences packed into samples."""
        mock_ps = Mock()
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3, 10, 11], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3, 10, 11], dtype=torch.long),
                "position_ids": torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([4, 5, 6, 7], dtype=torch.long),
                "position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            }
        ]

        collator = PackingConcatCollator()
        batch = collator(features)

        # Total length should be 5 + 4 = 9
        assert batch["input_ids"].shape == (1, 9)

        # Position IDs should reset for each sequence
        # [0,1,2,0,1] + [0,1,2,3] = [0,1,2,0,1,0,1,2,3]
        expected_position_ids = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
        assert torch.equal(batch["position_ids"], expected_position_ids)
