from unittest.mock import Mock, patch

import pytest
import torch

from xorl.data.collators import TextSequenceShardCollator
from xorl.data.constants import IGNORE_INDEX


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


def _make_mock_ps(cp_size=2, cp_rank=0, ringattn_size=1):
    mock_ps = Mock()
    mock_ps.cp_size = cp_size
    mock_ps.cp_rank = cp_rank
    mock_ps.ringattn_size = ringattn_size
    return mock_ps


class TestSPSliceAndPadding:
    """Tests for sp_slice and sp_padding utility methods."""

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_sp_slice_across_ranks_and_uneven(self, mock_parallel_state):
        """Covers initialization, basic slicing rank 0/1, and uneven split."""
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=0)
        collator = TextSequenceShardCollator()
        assert collator.cp_size == 2 and collator.cp_rank == 0

        tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
        assert torch.equal(collator.sp_slice(tensor, dim=-1), torch.tensor([[1, 2, 3]]))

        # Rank 1
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=1)
        collator1 = TextSequenceShardCollator()
        assert torch.equal(collator1.sp_slice(tensor, dim=-1), torch.tensor([[4, 5, 6]]))

        # Uneven split (length 5, cp_size=2 -> chunk_size=3 for rank 0)
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=0)
        collator0 = TextSequenceShardCollator()
        assert torch.equal(collator0.sp_slice(torch.tensor([[1, 2, 3, 4, 5]]), dim=-1), torch.tensor([[1, 2, 3]]))

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_sp_padding_basic_sequential_zero(self, mock_parallel_state):
        """Covers basic padding, sequential padding, and zero-length padding."""
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=0)
        collator = TextSequenceShardCollator()
        tensor = torch.tensor([[1, 2, 3]])

        assert torch.equal(
            collator.sp_padding(tensor, dim=-1, pad_value=0, pad_length=2), torch.tensor([[1, 2, 3, 0, 0]])
        )
        assert torch.equal(
            collator.sp_padding(tensor, dim=-1, pad_value=0, pad_length=2, sequential=True),
            torch.tensor([[1, 2, 3, 0, 1]]),
        )
        assert torch.equal(collator.sp_padding(tensor, dim=-1, pad_value=0, pad_length=0), tensor)


class TestCollatorCall:
    """Tests for the full collator __call__ method."""

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_preshifted_labels_and_packed_sequences(self, mock_parallel_state):
        """Covers pre-shifted labels pass-through and packed sequence boundary masking with cp_size=1."""
        mock_parallel_state.return_value = _make_mock_ps(cp_size=1, cp_rank=0)
        collator = TextSequenceShardCollator(pad_token_id=0)

        # Pre-shifted labels pass through
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }
        result = collator(batch)
        assert torch.equal(result["labels"], torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]))

        # Packed sequence boundary masking
        packed_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, IGNORE_INDEX, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
        }
        packed_result = collator(packed_batch)
        assert packed_result["labels"][0, 2] == IGNORE_INDEX
        assert packed_result["labels"][0, 4] == IGNORE_INDEX

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_sp_splitting_padding_and_flash_attn_kwargs(self, mock_parallel_state):
        """Covers SP padding to multiple, splitting across ranks, flash attention kwargs,
        attention_mask/position_ids preservation, and padding values."""
        # SP splitting with cp_size=2
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=0)
        collator0 = TextSequenceShardCollator(pad_token_id=0)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]]),
        }
        r0 = collator0(batch)
        assert r0["input_ids"].shape[-1] == 3
        assert r0["input_ids"][0, 0] == 1

        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=1)
        collator1 = TextSequenceShardCollator(pad_token_id=0)
        batch_r1 = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]]),
        }
        r1 = collator1(batch_r1)
        assert r1["input_ids"].shape[-1] == 3
        assert r1["input_ids"][0, 0] == 4

        # Padding to SP multiple (length 5 -> padded to 6, then split to 3)
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=0)
        collator_pad = TextSequenceShardCollator(pad_token_id=0)
        batch5 = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }
        r_pad = collator_pad(batch5)
        assert r_pad["input_ids"].shape[-1] == 3

        # Flash attention kwargs added (cp_size=1)
        mock_parallel_state.return_value = _make_mock_ps(cp_size=1, cp_rank=0)
        collator_fa = TextSequenceShardCollator(pad_token_id=0)
        batch_fa = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }
        r_fa = collator_fa(batch_fa)
        assert all(k in r_fa for k in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"])

        # Attention mask and position_ids preserved
        assert "attention_mask" in r_fa and r_fa["attention_mask"].shape == r_fa["input_ids"].shape
        assert "position_ids" in r_fa and r_fa["position_ids"].shape[-1] == r_fa["attention_mask"].shape[-1]

        # Single sequence handling
        batch_single = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }
        r_single = collator_fa(batch_single)
        assert r_single["input_ids"].shape[-1] == 5
        assert r_single["labels"].shape[-1] == 5
