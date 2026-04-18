from unittest.mock import Mock, patch

import pytest
import torch

from xorl.data.collators import PackingConcatCollator, add_flash_attention_kwargs_from_position_ids


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


class TestAddFlashAttentionKwargs:
    """Tests for add_flash_attention_kwargs_from_position_ids function."""

    def test_kwargs_and_correctness(self):
        """Covers all required kwargs added, cu_seq_lens correctness, max_length correctness, and single sequence."""
        # Multiple sequences: [0,1,2] and [0,1,2,3]
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
        }
        cu_q, cu_k, max_q, max_k = add_flash_attention_kwargs_from_position_ids(batch)

        assert all(k in batch for k in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"])
        assert torch.equal(cu_q, torch.tensor([0, 3, 7], dtype=torch.int32))
        assert max_q == 4 and max_k == 4

        # Single sequence
        batch2 = {"position_ids": torch.tensor([[0, 1, 2, 3, 4]])}
        cu_q2, _, max_q2, _ = add_flash_attention_kwargs_from_position_ids(batch2)
        assert torch.equal(cu_q2, torch.tensor([0, 5], dtype=torch.int32))
        assert max_q2 == 5


class TestPackingConcatCollator:
    """Tests for PackingConcatCollator."""

    @patch("xorl.data.collators.packing_concat_collator.get_parallel_state")
    def test_concatenation_and_flash_attn(self, mock_parallel_state, sample_packed_features, sample_features):
        """Covers basic concatenation shapes, value correctness, position_ids generation,
        flash attn kwargs with SP disabled/enabled, and single feature handling."""
        mock_ps = Mock()
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator(pad_to_multiple_of=1)

        # Basic concatenation
        batch = collator(sample_packed_features)
        assert batch["input_ids"].shape == (1, 10)
        assert batch["attention_mask"].shape == (1, 10)
        assert batch["labels"].shape == (1, 10)
        assert batch["position_ids"].shape == (1, 10)
        assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
        assert torch.equal(batch["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]]))

        # Flash attn kwargs present when SP disabled
        assert all(k in batch for k in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"])

        # Position IDs generated if missing
        batch_gen = collator(sample_features)
        assert "position_ids" in batch_gen
        assert torch.equal(batch_gen["position_ids"], torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]))

        # Flash attn kwargs NOT present when SP enabled
        mock_ps.cp_enabled = True
        batch_sp = collator(sample_packed_features)
        assert all(k not in batch_sp for k in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"])

        # Single feature
        mock_ps.cp_enabled = False
        single = [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
                "position_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            }
        ]
        batch_single = collator(single)
        assert batch_single["input_ids"].shape == (1, 3)

    @patch("xorl.data.collators.packing_concat_collator.get_parallel_state")
    def test_extra_fields_and_multiple_seqs(self, mock_parallel_state):
        """Covers extra field handling and multiple packed sequences per sample."""
        mock_ps = Mock()
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator = PackingConcatCollator(pad_to_multiple_of=1)

        # Extra fields collated as batch
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
            },
        ]
        batch = collator(features)
        assert "extra_field" in batch and batch["extra_field"].shape == (2, 1)

        # Multiple sequences per sample
        features2 = [
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
            },
        ]
        batch2 = collator(features2)
        assert batch2["input_ids"].shape == (1, 9)
        assert torch.equal(batch2["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]]))
