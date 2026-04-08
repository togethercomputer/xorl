"""
Integration tests for distributed data loading.

These tests verify that the data loader correctly distributes data across multiple
processes/ranks in a distributed training setting, and that the data alignment is correct.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import Dataset

from tests.conftest import FakeTextDataset, SimpleCollator
from xorl.data.collators import CollatePipeline, TextSequenceShardCollator
from xorl.data.constants import IGNORE_INDEX
from xorl.data.data_loader import (
    DataLoaderBuilder,
    MicroBatchCollator,
)


pytestmark = [pytest.mark.gpu, pytest.mark.dataloader, pytest.mark.distributed]


class TestDistributedDataAlignment:
    """Tests for data partitioning, micro-batch splitting, SP sharding, batch size, and drop_last."""

    @patch("xorl.data.data_loader.get_parallel_state")
    @patch("xorl.data.data_loader.StatefulDistributedSampler")
    def test_partitioning_micro_batches_sp_batch_size_and_drop_last(
        self, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Covers data partitioning across ranks, micro-batch splitting, SP pipeline setup,
        batch size calculation, and drop_last behavior."""
        # --- Data partitioning across 4 DP ranks ---
        dp_size = 4
        datasets_per_rank = []
        for rank in range(dp_size):
            mock_ps = Mock()
            mock_ps.dp_size = dp_size
            mock_ps.dp_rank = rank
            mock_ps.cp_size = 1
            mock_ps.cp_enabled = False
            mock_parallel_state.return_value = mock_ps

            mock_sampler = Mock()
            rank_indices = list(range(rank, len(fake_text_dataset), dp_size))
            mock_sampler.__iter__ = Mock(return_value=iter(rank_indices))
            mock_sampler.__len__ = Mock(return_value=len(rank_indices))
            mock_sampler_cls.return_value = mock_sampler

            builder = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2)
            builder.build(verbose=False)
            assert mock_sampler_cls.call_args[1]["rank"] == rank
            assert mock_sampler_cls.call_args[1]["num_replicas"] == dp_size
            datasets_per_rank.append(rank_indices)

        all_indices = [idx for indices in datasets_per_rank for idx in indices]
        assert len(set(all_indices)) == len(all_indices)  # No overlap
        assert len(set(all_indices)) >= len(fake_text_dataset) - dp_size

        # --- Micro-batch splitting correctness ---
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=4, gradient_accumulation_steps=3)
        dataloader = builder.build(verbose=False)
        micro_batches = next(iter(dataloader))
        assert isinstance(micro_batches, list) and len(micro_batches) == 3
        for mb in micro_batches:
            assert isinstance(mb, dict)
            assert "input_ids" in mb and "labels" in mb
            assert len(mb["input_ids"].shape) == 2

        # --- SP pipeline verification ---
        mock_ps.cp_size = 2
        mock_ps.cp_enabled = True
        mock_parallel_state.return_value = mock_ps
        builder_sp = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=1)

        mock_sampler = Mock()
        mock_sampler.__iter__ = Mock(return_value=iter(range(len(fake_text_dataset))))
        mock_sampler.__len__ = Mock(return_value=len(fake_text_dataset))
        mock_sampler_cls.return_value = mock_sampler

        dataloader_sp = builder_sp.build(verbose=False)

        internal = builder_sp.collate_fn.internal_collator
        assert isinstance(internal, CollatePipeline)
        assert any(isinstance(c, TextSequenceShardCollator) for c in internal.data_collators)

        # --- Batch size calculation ---
        mock_ps.dp_size = 2
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps
        assert 4 * 3 == 12  # micro_batch_size * gradient_accumulation_steps

        # --- Drop last behavior ---
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps

        small_dataset = FakeTextDataset(num_samples=10, seq_len=64)

        # Reset mock sampler for the small dataset
        mock_sampler = Mock()
        mock_sampler.__iter__ = Mock(return_value=iter(range(len(small_dataset))))
        mock_sampler.__len__ = Mock(return_value=len(small_dataset))
        mock_sampler_cls.return_value = mock_sampler

        builder_dl = DataLoaderBuilder(
            dataset=small_dataset, micro_batch_size=2, gradient_accumulation_steps=2, drop_last=True
        )
        dl = builder_dl.build(verbose=False)
        batches = list(dl)
        assert len(batches) == 2


class TestMicroBatchCollatorAndSequenceSharding:
    """Tests for micro-batch data order, gradient accumulation alignment, error handling,
    and sequence sharding across SP ranks."""

    def test_order_preservation_ga_alignment_and_error(self):
        """Covers data order within micro-batches, gradient accumulation alignment, and incorrect batch size error."""

        collator = MicroBatchCollator(
            micro_batch_size=2, gradient_accumulation_steps=3, internal_collator=SimpleCollator()
        )
        features = [{"input_ids": torch.tensor([i]), "value": torch.tensor([i * 100])} for i in range(6)]
        micro_batches = collator(features)
        assert torch.equal(micro_batches[0]["value"], torch.tensor([[0], [100]]))
        assert torch.equal(micro_batches[1]["value"], torch.tensor([[200], [300]]))
        assert torch.equal(micro_batches[2]["value"], torch.tensor([[400], [500]]))

        collator2 = MicroBatchCollator(
            micro_batch_size=3, gradient_accumulation_steps=2, internal_collator=SimpleCollator()
        )
        features2 = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i])} for i in range(6)]
        mbs = collator2(features2)
        assert len(mbs) == 2
        for mb in mbs:
            assert mb["input_ids"].shape[0] == 3

        collator3 = MicroBatchCollator(
            micro_batch_size=4, gradient_accumulation_steps=2, internal_collator=SimpleCollator()
        )
        features3 = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i])} for i in range(7)]
        with pytest.raises(ValueError, match="Expected 8 samples"):
            collator3(features3)

    @patch("xorl.data.collators.sequence_shard_collator.get_parallel_state")
    def test_sequence_sharding_and_padding(self, mock_parallel_state):
        """Covers correct chunk sizes across SP ranks and padding for non-divisible lengths."""

        cp_size = 4
        seq_len = 128
        input_ids = torch.arange(seq_len).unsqueeze(0)
        labels = torch.cat([torch.arange(1, seq_len), torch.tensor([IGNORE_INDEX])]).unsqueeze(0)

        sharded = []
        for cp_rank in range(cp_size):
            mock_ps = Mock()
            mock_ps.cp_size = cp_size
            mock_ps.cp_rank = cp_rank
            mock_ps.ringattn_size = 1
            mock_parallel_state.return_value = mock_ps
            collator = TextSequenceShardCollator(pad_token_id=0)
            batch = {
                "input_ids": input_ids.clone(),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                "labels": labels.clone(),
                "position_ids": torch.arange(seq_len).unsqueeze(0),
            }
            result = collator(batch)
            sharded.append(result["input_ids"])

        expected_chunk = seq_len // cp_size
        for rank, s in enumerate(sharded):
            assert s.shape[1] == expected_chunk, f"Rank {rank} has incorrect chunk size"

        # Padding alignment for non-divisible length
        seq_len_odd = 130
        mock_ps = Mock()
        mock_ps.cp_size = cp_size
        mock_ps.cp_rank = 0
        mock_ps.ringattn_size = 1
        mock_parallel_state.return_value = mock_ps
        collator = TextSequenceShardCollator(pad_token_id=0)
        batch = {
            "input_ids": torch.arange(seq_len_odd).unsqueeze(0),
            "attention_mask": torch.ones(1, seq_len_odd, dtype=torch.long),
            "labels": torch.cat([torch.arange(1, seq_len_odd), torch.tensor([IGNORE_INDEX])]).unsqueeze(0),
            "position_ids": torch.arange(seq_len_odd).unsqueeze(0),
        }
        result = collator(batch)
        total_padded = result["input_ids"].shape[1] * cp_size
        assert total_padded >= seq_len_odd
        assert total_padded < seq_len_odd + cp_size


class TestEndToEndAndPackedSequences:
    """End-to-end integration tests for distributed data loading and packed sequences."""

    @patch("xorl.data.data_loader.get_parallel_state")
    @patch("xorl.data.collators.packing_concat_collator.get_parallel_state")
    def test_pipeline_epoch_consistency_and_packed_sequences(
        self, mock_ps_collator, mock_ps_loader, fake_packed_dataset, fake_text_dataset
    ):
        """Covers complete pipeline output, epoch consistency, packed sequence flattening,
        multi-DP, SP with packing, and variable lengths."""
        mock_ps = Mock()
        mock_ps.dp_size = 2
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps

        # Complete pipeline output
        builder = DataLoaderBuilder(
            dataset=fake_packed_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=0,
            prefetch_factor=None,
            seed=42,
        )
        dataloader = builder.build(verbose=False)

        batch_count = 0
        for micro_batches in dataloader:
            assert isinstance(micro_batches, list) and len(micro_batches) == 2
            for mb in micro_batches:
                for field in ["input_ids", "labels", "position_ids", "attention_mask"]:
                    assert field in mb
                assert "cu_seq_lens_q" in mb
                bs, sl = mb["input_ids"].shape
                assert mb["labels"].shape == (bs, sl)
                assert mb["position_ids"].shape == (bs, sl)
                assert mb["attention_mask"].shape == (bs, sl)
            batch_count += 1
            if batch_count >= 3:
                break
        assert batch_count == 3

        # Epoch consistency
        mock_ps.dp_size = 1
        mock_ps_loader.return_value = mock_ps
        mock_ps_collator.return_value = mock_ps
        builder2 = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=4,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
            seed=42,
        )
        dl2 = builder2.build(verbose=False)

        epoch1 = [next(iter(dl2))[0]["input_ids"].clone() for _ in range(3)]
        dl2.set_epoch(1)
        epoch2 = [next(iter(dl2))[0]["input_ids"].clone() for _ in range(3)]
        assert len(epoch1) == len(epoch2)

        # --- Packed sequence tests ---

        class PackedDataset(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return [
                    {
                        "input_ids": torch.tensor([idx * 10, idx * 10 + 1, idx * 10 + 2]),
                        "labels": torch.tensor([idx * 10, idx * 10 + 1, idx * 10 + 2]),
                        "position_ids": torch.tensor([0, 1, 2]),
                        "attention_mask": torch.ones(3, dtype=torch.long),
                    },
                    {
                        "input_ids": torch.tensor([idx * 10 + 3, idx * 10 + 4]),
                        "labels": torch.tensor([idx * 10 + 3, idx * 10 + 4]),
                        "position_ids": torch.tensor([0, 1]),
                        "attention_mask": torch.ones(2, dtype=torch.long),
                    },
                ]

        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps

        builder3 = DataLoaderBuilder(
            dataset=PackedDataset(),
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )
        dl3 = builder3.build(verbose=False)
        mbs = next(iter(dl3))
        assert len(mbs) == 1
        assert mbs[0]["input_ids"].shape[0] == 1
        assert mbs[0]["input_ids"].shape[1] == 6
        assert mbs[0]["attention_mask"].shape == (1, 6)

        # Multiple DP ranks
        class SimplePacked(Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, idx):
                return [
                    {
                        "input_ids": torch.tensor([idx, idx + 1]),
                        "labels": torch.tensor([idx, idx + 1]),
                        "position_ids": torch.tensor([0, 1]),
                        "attention_mask": torch.ones(2, dtype=torch.long),
                    }
                ]

        for dp_rank in [0, 1]:
            mock_ps.dp_size = 2
            mock_ps.dp_rank = dp_rank
            mock_ps_collator.return_value = mock_ps
            mock_ps_loader.return_value = mock_ps
            builder4 = DataLoaderBuilder(
                dataset=SimplePacked(),
                micro_batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                prefetch_factor=None,
                pad_to_multiple_of=1,
            )
            dl4 = builder4.build(verbose=False)
            mbs = next(iter(dl4))
            assert len(mbs) == 1
            assert mbs[0]["input_ids"].shape == (1, 2)

        # Variable lengths
        class VariablePackedDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                if idx % 2 == 0:
                    return [
                        {
                            "input_ids": torch.tensor([idx, idx + 1]),
                            "labels": torch.tensor([idx, idx + 1]),
                            "position_ids": torch.tensor([0, 1]),
                            "attention_mask": torch.ones(2, dtype=torch.long),
                        }
                    ]
                else:
                    return [
                        {
                            "input_ids": torch.tensor([idx]),
                            "labels": torch.tensor([idx]),
                            "position_ids": torch.tensor([0]),
                            "attention_mask": torch.ones(1, dtype=torch.long),
                        },
                        {
                            "input_ids": torch.tensor([idx + 1, idx + 2]),
                            "labels": torch.tensor([idx + 1, idx + 2]),
                            "position_ids": torch.tensor([0, 1]),
                            "attention_mask": torch.ones(2, dtype=torch.long),
                        },
                        {
                            "input_ids": torch.tensor([idx + 3]),
                            "labels": torch.tensor([idx + 3]),
                            "position_ids": torch.tensor([0]),
                            "attention_mask": torch.ones(1, dtype=torch.long),
                        },
                    ]

        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps
        builder5 = DataLoaderBuilder(
            dataset=VariablePackedDataset(),
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )
        dl5 = builder5.build(verbose=False)
        mbs = next(iter(dl5))
        assert len(mbs) == 1
        assert mbs[0]["input_ids"].shape == (1, 4)
        assert mbs[0]["attention_mask"].shape == (1, 4)
