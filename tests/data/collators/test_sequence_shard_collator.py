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
    def test_shift_check_skips_supervised_packed_document_tails(self, mock_parallel_state):
        """A target at a packed tail must not be compared with the next document's first input."""
        mock_parallel_state.return_value = _make_mock_ps(cp_size=8, cp_rank=0)
        collator = TextSequenceShardCollator(pad_token_id=0)
        result = collator(
            {
                "input_ids": torch.tensor([[10, 11, 20, 21, 30, 31, 40, 41]]),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
                "labels": torch.tensor([[IGNORE_INDEX, 91, IGNORE_INDEX, 92, IGNORE_INDEX, 93, IGNORE_INDEX, 94]]),
                "position_ids": torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]]),
            }
        )
        assert result["input_ids"].shape == (1, 1)

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_shift_check_still_rejects_same_document_mismatch(self, mock_parallel_state):
        mock_parallel_state.return_value = _make_mock_ps(cp_size=1, cp_rank=0)
        collator = TextSequenceShardCollator(pad_token_id=0)
        with pytest.raises(AssertionError, match="first comparable non-ignore label"):
            collator(
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                    "labels": torch.tensor([[99, IGNORE_INDEX, IGNORE_INDEX]]),
                    "position_ids": torch.tensor([[0, 1, 2]]),
                }
            )

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

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_teacher_hidden_states_are_sharded_with_token_fields(self, mock_parallel_state):
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=1)
        collator = TextSequenceShardCollator(pad_token_id=0)
        teacher_hidden_states = torch.tensor(
            [
                [
                    [0.25, 0.5],
                    [1.25, 1.5],
                    [2.25, 2.5],
                    [3.25, 3.5],
                    [4.25, 4.5],
                ]
            ]
        )
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
            "teacher_hidden_states": teacher_hidden_states,
        }

        result = collator(batch)

        assert result["input_ids"].shape[-1] == 3
        assert result["teacher_hidden_states"].shape == (1, 3, 2)
        torch.testing.assert_close(
            result["teacher_hidden_states"],
            torch.tensor([[[3.25, 3.5], [4.25, 4.5], [0.0, 0.0]]]),
        )

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_logprob_temperatures_follow_cp_slice_with_identity_padding(self, mock_parallel_state):
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=1)
        collator = TextSequenceShardCollator(pad_token_id=0)
        result = collator(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
                "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
                "position_ids": torch.arange(5).unsqueeze(0),
                "logprob_temperatures": torch.tensor([[0.7, 0.8, 0.9, 1.2, 1.3]]),
            }
        )

        assert result["logprob_temperatures"].dtype is torch.float32
        torch.testing.assert_close(
            result["logprob_temperatures"],
            torch.tensor([[1.2, 1.3, 1.0]]),
        )

    @pytest.mark.parametrize("cp_rank", [0, 15])
    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_drgrpo_side_channels_follow_cp16_target_shard(self, mock_parallel_state, cp_rank):
        """Canonical and reference logprobs must follow the same padded CP slice as labels."""
        cp_size = 16
        seq_len = 4099
        chunk_size = 257
        mock_parallel_state.return_value = _make_mock_ps(cp_size=cp_size, cp_rank=cp_rank)
        collator = TextSequenceShardCollator(pad_token_id=0)

        input_ids = torch.arange(seq_len).unsqueeze(0)
        target_tokens = torch.cat([torch.arange(1, seq_len), torch.tensor([IGNORE_INDEX])]).unsqueeze(0)
        old_logprobs = torch.arange(seq_len, dtype=torch.float32).add(0.25).unsqueeze(0)
        advantages = torch.arange(seq_len, dtype=torch.float32).add(0.5).unsqueeze(0)
        ref_logprobs = torch.arange(seq_len, dtype=torch.float32).neg().sub(0.75).unsqueeze(0)
        result = collator(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": target_tokens.clone(),
                "target_tokens": target_tokens.clone(),
                "position_ids": torch.arange(seq_len).unsqueeze(0),
                "old_logprobs": old_logprobs,
                "advantages": advantages,
                "ref_logprobs": ref_logprobs,
            }
        )

        start = cp_rank * chunk_size
        end = min(start + chunk_size, seq_len)
        valid_length = end - start
        assert result["labels"].shape == (1, chunk_size)
        for field in ("target_tokens", "old_logprobs", "advantages", "ref_logprobs"):
            assert result[field].shape == result["labels"].shape

        torch.testing.assert_close(result["target_tokens"][0, :valid_length], target_tokens[0, start:end])
        torch.testing.assert_close(result["old_logprobs"][0, :valid_length], old_logprobs[0, start:end])
        torch.testing.assert_close(result["advantages"][0, :valid_length], advantages[0, start:end])
        torch.testing.assert_close(result["ref_logprobs"][0, :valid_length], ref_logprobs[0, start:end])
        if valid_length < chunk_size:
            assert torch.equal(
                result["target_tokens"][0, valid_length:],
                torch.full((chunk_size - valid_length,), IGNORE_INDEX),
            )
            for field in ("old_logprobs", "advantages", "ref_logprobs"):
                assert torch.equal(result[field][0, valid_length:], torch.zeros(chunk_size - valid_length))
