from unittest.mock import Mock, patch

import pytest
import torch

from xorl.data.collators import TextSequenceShardCollator
from xorl.data.collators.sequence_shard_collator import zigzag_reorder_packed_sequence
from xorl.data.constants import IGNORE_INDEX


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


def _make_mock_ps(cp_size=2, cp_rank=0, ringattn_size=1):
    mock_ps = Mock()
    mock_ps.cp_size = cp_size
    mock_ps.cp_rank = cp_rank
    mock_ps.ringattn_size = ringattn_size
    return mock_ps


class TestCollatorCall:
    """Tests for the full collator __call__ method."""

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def _assert_preshifted_labels_and_packed_sequences(self, mock_parallel_state):
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
        self._assert_preshifted_labels_and_packed_sequences()

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
        assert torch.equal(r0["input_ids"], torch.tensor([[1, 2, 3]]))
        assert torch.equal(r0["labels"], torch.tensor([[2, 3, 4]]))

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
        assert torch.equal(r1["input_ids"], torch.tensor([[4, 5, 6]]))
        assert torch.equal(r1["labels"], torch.tensor([[5, 6, IGNORE_INDEX]]))

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
        assert torch.equal(r_pad["input_ids"], torch.tensor([[1, 2, 3]]))

        # The last rank observes both constant and sequential padding through
        # the production collator, rather than through its private primitives.
        mock_parallel_state.return_value = _make_mock_ps(cp_size=2, cp_rank=1)
        collator_pad_last = TextSequenceShardCollator(pad_token_id=0)
        batch5_last = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        }
        r_pad_last = collator_pad_last(batch5_last)
        assert torch.equal(r_pad_last["input_ids"], torch.tensor([[4, 5, 0]]))
        assert torch.equal(r_pad_last["labels"], torch.tensor([[5, IGNORE_INDEX, IGNORE_INDEX]]))
        assert torch.equal(r_pad_last["position_ids"], torch.tensor([[0, 1, 2, 3, 4, 0]]))

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

        self._assert_token_side_channels_follow_sequence_shards()
        _assert_zigzag_reorder_policy()

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def _assert_token_side_channels_follow_sequence_shards(self, mock_parallel_state):
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

        self._assert_drgrpo_side_channels_follow_cp16_target_shards()

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def _assert_drgrpo_side_channels_follow_cp16_target_shards(self, mock_parallel_state):
        """Canonical and reference logprobs must follow the same padded CP slice as labels."""
        cp_size = 16
        seq_len = 4099
        chunk_size = 257
        input_ids = torch.arange(seq_len).unsqueeze(0)
        target_tokens = torch.cat([torch.arange(1, seq_len), torch.tensor([IGNORE_INDEX])]).unsqueeze(0)
        old_logprobs = torch.arange(seq_len, dtype=torch.float32).add(0.25).unsqueeze(0)
        advantages = torch.arange(seq_len, dtype=torch.float32).add(0.5).unsqueeze(0)
        ref_logprobs = torch.arange(seq_len, dtype=torch.float32).neg().sub(0.75).unsqueeze(0)

        for cp_rank in (0, 15):
            mock_parallel_state.return_value = _make_mock_ps(cp_size=cp_size, cp_rank=cp_rank)
            collator = TextSequenceShardCollator(pad_token_id=0)
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


def _assert_zigzag_reorder_policy() -> None:
    ringattn_size = 2
    tensor = torch.arange(40).unsqueeze(0)
    position_ids = torch.arange(40).unsqueeze(0)
    reordered = zigzag_reorder_packed_sequence(tensor, position_ids, ringattn_size, dim=-1)
    expected = torch.cat(
        (
            torch.arange(0, 10),
            torch.arange(30, 40),
            torch.arange(10, 20),
            torch.arange(20, 30),
        )
    ).unsqueeze(0)
    assert torch.equal(reordered, expected)

    doc_len = 20
    position_ids = torch.cat((torch.arange(doc_len), torch.arange(doc_len))).unsqueeze(0)
    reordered = zigzag_reorder_packed_sequence(torch.arange(2 * doc_len).unsqueeze(0), position_ids, ringattn_size)
    expected = torch.cat(
        (
            torch.arange(0, 5),
            torch.arange(15, 20),
            torch.arange(20, 25),
            torch.arange(35, 40),
            torch.arange(5, 10),
            torch.arange(10, 15),
            torch.arange(25, 30),
            torch.arange(30, 35),
        )
    ).unsqueeze(0)
    assert torch.equal(reordered, expected)
    reordered_position_ids = zigzag_reorder_packed_sequence(position_ids, position_ids, ringattn_size)
    assert (reordered_position_ids[0, :doc_len] == 0).nonzero(as_tuple=False).view(-1).numel() == 2

    for ringattn_size in (2, 4, 8):
        sequence_length = 8 * 2 * ringattn_size
        tensor = torch.arange(sequence_length).unsqueeze(0)
        reordered = zigzag_reorder_packed_sequence(tensor, tensor, ringattn_size)
        assert reordered.shape == tensor.shape
        assert torch.equal(reordered.sort().values, tensor)
        rank_width = sequence_length // ringattn_size
        for rank in range(ringattn_size):
            rank_slice = reordered[0, rank * rank_width : (rank + 1) * rank_width]
            half = rank_width // 2
            assert rank_slice[:half].max() < rank_slice[half:].min()

    tensor = torch.arange(20).unsqueeze(0)
    assert zigzag_reorder_packed_sequence(tensor, tensor, 1) is tensor
    with pytest.raises(ValueError, match="not divisible"):
        zigzag_reorder_packed_sequence(torch.arange(15).unsqueeze(0), torch.arange(15).unsqueeze(0), 2)
