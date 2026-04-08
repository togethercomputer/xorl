from typing import Any, Dict, Sequence
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import Dataset

from xorl.data.collators import CollatePipeline, DataCollator, FlattenCollator, PackingConcatCollator
from xorl.data.data_loader import (
    DataLoaderBuilder,
    DistributedDataloader,
    MicroBatchCollator,
)


pytestmark = [pytest.mark.cpu, pytest.mark.dataloader]


class SimpleCollator(DataCollator):
    """Simple collator for testing that stacks tensors."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in features[0].keys():
            result[key] = torch.stack([f[key] for f in features])
        return result


class TestMicroBatchCollatorAndDistributedDataloader:
    """Tests for MicroBatchCollator splitting, DistributedDataloader.set_epoch, and DataLoaderBuilder."""

    def test_micro_batch_splitting_set_epoch_and_builder(self):
        """Covers micro-batch splitting, edge cases, set_epoch delegation, batch size, sampler, SP collator, and custom collators."""
        internal_collator = SimpleCollator()

        # Multiple micro-batches with correct data
        collator = MicroBatchCollator(
            micro_batch_size=2, gradient_accumulation_steps=2, internal_collator=internal_collator
        )
        features = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i * 10])} for i in range(4)]
        result = collator(features)
        assert len(result) == 2
        assert torch.equal(result[0]["input_ids"], torch.tensor([[0], [1]]))
        assert torch.equal(result[0]["labels"], torch.tensor([[0], [10]]))
        assert torch.equal(result[1]["input_ids"], torch.tensor([[2], [3]]))
        assert torch.equal(result[1]["labels"], torch.tensor([[20], [30]]))

        # Single micro-batch
        collator_single = MicroBatchCollator(
            micro_batch_size=4, gradient_accumulation_steps=1, internal_collator=internal_collator
        )
        features = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i])} for i in range(4)]
        result = collator_single(features)
        assert len(result) == 1
        assert result[0]["input_ids"].shape[0] == 4

        # Uneven division raises ValueError
        collator_uneven = MicroBatchCollator(
            micro_batch_size=3, gradient_accumulation_steps=2, internal_collator=internal_collator
        )
        features = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i])} for i in range(5)]
        with pytest.raises(ValueError, match="Expected 6 samples"):
            collator_uneven(features)

        # --- set_epoch delegation ---
        # Sampler has set_epoch
        dl = DistributedDataloader.__new__(DistributedDataloader)
        dl.sampler = Mock()
        dl.sampler.set_epoch = Mock()
        dl.dataset = Mock()
        dl.set_epoch(5)
        dl.sampler.set_epoch.assert_called_once_with(5)

        # No sampler, dataset has set_epoch
        dl2 = DistributedDataloader.__new__(DistributedDataloader)
        dl2.sampler = None
        dl2.dataset = Mock()
        dl2.dataset.set_epoch = Mock()
        dl2.set_epoch(5)
        dl2.dataset.set_epoch.assert_called_once_with(5)

        # No sampler, no set_epoch on dataset -- should not raise
        dl3 = DistributedDataloader.__new__(DistributedDataloader)
        dl3.sampler = None
        dl3.dataset = Mock(spec=[])
        dl3.set_epoch(5)

    @patch("xorl.data.data_loader.get_parallel_state")
    @patch("xorl.data.data_loader.StatefulDistributedSampler")
    @patch("xorl.data.data_loader.DistributedDataloader")
    def test_builder_batch_size_sampler_sp_and_custom_collators(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Covers batch size, sampler params, SP collator, single/multiple custom collators."""
        mock_ps = Mock()
        mock_ps.dp_size = 2
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps

        DataLoaderBuilder(
            dataset=fake_text_dataset, micro_batch_size=4, gradient_accumulation_steps=2, num_workers=4, seed=42
        ).build(verbose=False)
        assert mock_dataloader_cls.call_args[1]["batch_size"] == 8

        # Sampler params
        mock_ps.dp_size = 4
        mock_ps.dp_rank = 1
        mock_parallel_state.return_value = mock_ps
        DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2, seed=123).build(
            verbose=False
        )
        call_kwargs = mock_sampler_cls.call_args[1]
        assert call_kwargs["num_replicas"] == 4
        assert call_kwargs["rank"] == 1
        assert call_kwargs["shuffle"] is True
        assert call_kwargs["seed"] == 123

        # SP collator added when cp_enabled
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 2
        mock_ps.cp_enabled = True
        mock_parallel_state.return_value = mock_ps
        DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2).build(
            verbose=False
        )
        collate_fn = mock_dataloader_cls.call_args[1]["collate_fn"]
        assert isinstance(collate_fn, MicroBatchCollator)

        assert isinstance(collate_fn.internal_collator, CollatePipeline)
        assert len(collate_fn.internal_collator.data_collators) == 5

        # Single custom collator
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_parallel_state.return_value = mock_ps
        custom = SimpleCollator()
        builder = DataLoaderBuilder(
            dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2, use_default_collators=False
        )
        builder.add_collator(custom)
        builder.build(verbose=False)
        micro_batch_collator = mock_dataloader_cls.call_args[1]["collate_fn"]
        assert isinstance(micro_batch_collator, MicroBatchCollator)
        assert micro_batch_collator.internal_collator is custom

        # Multiple custom collators
        builder2 = DataLoaderBuilder(
            dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2, use_default_collators=False
        )
        builder2.add_collator(SimpleCollator())
        builder2.add_collator(SimpleCollator())
        builder2.build(verbose=False)
        assert isinstance(mock_dataloader_cls.call_args[1]["collate_fn"].internal_collator, CollatePipeline)
        assert len(mock_dataloader_cls.call_args[1]["collate_fn"].internal_collator.data_collators) == 2


class TestDataLoaderBuilderPipelineAndIntegration:
    """Tests for pipeline manipulation and integration with packed sequences."""

    @patch("xorl.data.data_loader.get_parallel_state")
    @patch("xorl.data.collators.packing_concat_collator.get_parallel_state")
    def test_pipeline_manipulation_and_packed_integration(self, mock_ps_collator, mock_ps_loader, fake_text_dataset):
        """Covers add/insert/remove, defaults flag, invalid position, empty build,
        training loop, nested lists, packed dataset, flatten+packing, and variable seqs."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.cp_size = 1
        mock_ps.cp_enabled = False
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps

        # Add to end (default)
        builder = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2)
        custom = SimpleCollator()
        builder.add_collator(custom, position="end")
        pipeline = builder.get_collator_pipeline()
        assert len(pipeline) == 5
        assert pipeline[-1] is custom

        # Add to start
        builder2 = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2)
        custom2 = SimpleCollator()
        builder2.add_collator(custom2, position="start")
        assert builder2.get_collator_pipeline()[0] is custom2

        # Insert at index
        builder3 = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2)
        builder3.add_collator(SimpleCollator())
        builder3.add_collator(SimpleCollator())
        inserted = SimpleCollator()
        builder3.insert_collator(inserted, index=1)
        assert len(builder3.get_collator_pipeline()) == 7
        assert builder3.get_collator_pipeline()[1] is inserted

        # Remove collator
        builder4 = DataLoaderBuilder(dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2)
        builder4.add_collator(SimpleCollator())
        assert len(builder4.get_collator_pipeline()) == 5
        builder4.remove_collator(0)
        assert len(builder4.get_collator_pipeline()) == 4

        # use_default_collators flag
        with_defaults = DataLoaderBuilder(
            dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2, use_default_collators=True
        )
        assert len(with_defaults.get_collator_pipeline()) > 0
        without_defaults = DataLoaderBuilder(
            dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2, use_default_collators=False
        )
        assert len(without_defaults.get_collator_pipeline()) == 0

        # Invalid position raises ValueError
        with pytest.raises(ValueError, match="Invalid position: middle"):
            builder.add_collator(SimpleCollator(), position="middle")

        # Empty pipeline build raises ValueError
        empty_builder = DataLoaderBuilder(
            dataset=fake_text_dataset, micro_batch_size=2, gradient_accumulation_steps=2, use_default_collators=False
        )
        with pytest.raises(ValueError, match="Collator pipeline is empty"):
            empty_builder.build(verbose=False)

        # --- Integration: training loop ---
        class SimpleTokenizedDataset(Dataset):
            def __init__(self, num_samples=32, seq_len=10):
                self.num_samples = num_samples
                self.seq_len = seq_len

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.arange(idx * 10, idx * 10 + self.seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "labels": torch.arange(idx * 10, idx * 10 + self.seq_len, dtype=torch.long),
                    "position_ids": torch.arange(self.seq_len, dtype=torch.long),
                }

        dataset = SimpleTokenizedDataset(num_samples=16, seq_len=5)
        int_builder = DataLoaderBuilder(
            dataset=dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )
        dataloader = int_builder.build(verbose=False)

        for step, micro_batches in enumerate(dataloader):
            assert isinstance(micro_batches, list)
            assert len(micro_batches) == 2
            for mb in micro_batches:
                assert isinstance(mb, dict)
                assert "input_ids" in mb
                assert "position_ids" in mb
                assert mb["input_ids"].shape[0] == 1
                assert mb["input_ids"].shape[1] == 8
            if step == 0:
                break

        # Nested list features handled correctly
        internal_collator = PackingConcatCollator()
        micro_batch_collator = MicroBatchCollator(
            micro_batch_size=2, gradient_accumulation_steps=2, internal_collator=internal_collator
        )
        correct_features = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.ones(2),
                "labels": torch.tensor([1, 2]),
                "position_ids": torch.tensor([0, 1]),
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "attention_mask": torch.ones(2),
                "labels": torch.tensor([3, 4]),
                "position_ids": torch.tensor([0, 1]),
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "attention_mask": torch.ones(2),
                "labels": torch.tensor([5, 6]),
                "position_ids": torch.tensor([0, 1]),
            },
            {
                "input_ids": torch.tensor([7, 8]),
                "attention_mask": torch.ones(2),
                "labels": torch.tensor([7, 8]),
                "position_ids": torch.tensor([0, 1]),
            },
        ]
        result = micro_batch_collator(correct_features)
        assert len(result) == 2

        # --- Packed sequences ---
        class PackedSequencesDataset(Dataset):
            def __init__(self, num_samples=32, num_seqs_per_sample=2, seq_len=10):
                self.num_samples = num_samples
                self.num_seqs_per_sample = num_seqs_per_sample
                self.seq_len = seq_len

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                sequences = []
                for seq_idx in range(self.num_seqs_per_sample):
                    offset = (idx * self.num_seqs_per_sample + seq_idx) * self.seq_len
                    sequences.append(
                        {
                            "input_ids": torch.arange(offset, offset + self.seq_len, dtype=torch.long),
                            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                            "labels": torch.arange(offset, offset + self.seq_len, dtype=torch.long),
                            "position_ids": torch.arange(self.seq_len, dtype=torch.long),
                        }
                    )
                return sequences

        packed_dataset = PackedSequencesDataset(num_samples=16, num_seqs_per_sample=2, seq_len=5)
        sample = packed_dataset[0]
        assert isinstance(sample, list) and len(sample) == 2
        assert isinstance(sample[0], dict)

        packed_builder = DataLoaderBuilder(
            dataset=packed_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )
        packed_dl = packed_builder.build(verbose=False)
        batch = next(iter(packed_dl))
        assert isinstance(batch, list) and len(batch) == 2
        for mb in batch:
            assert isinstance(mb, dict)
            assert mb["input_ids"].shape == (1, 16)

        # Flatten + Packing collator

        flatten_collator = FlattenCollator()
        packing_collator = PackingConcatCollator(pad_to_multiple_of=1)
        features = [
            [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.ones(3),
                    "labels": torch.tensor([1, 2, 3]),
                    "position_ids": torch.tensor([0, 1, 2]),
                },
                {
                    "input_ids": torch.tensor([4, 5]),
                    "attention_mask": torch.ones(2),
                    "labels": torch.tensor([4, 5]),
                    "position_ids": torch.tensor([0, 1]),
                },
            ],
            [
                {
                    "input_ids": torch.tensor([6, 7]),
                    "attention_mask": torch.ones(2),
                    "labels": torch.tensor([6, 7]),
                    "position_ids": torch.tensor([0, 1]),
                },
                {
                    "input_ids": torch.tensor([8, 9, 10]),
                    "attention_mask": torch.ones(3),
                    "labels": torch.tensor([8, 9, 10]),
                    "position_ids": torch.tensor([0, 1, 2]),
                },
            ],
        ]
        result = packing_collator(flatten_collator(features))
        assert result["input_ids"].shape == (1, 10)
        assert torch.equal(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))

        # Variable sequences per sample
        class VariablePackedDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                if idx == 0:
                    return [
                        {
                            "input_ids": torch.tensor([1, 2]),
                            "attention_mask": torch.ones(2),
                            "labels": torch.tensor([1, 2]),
                            "position_ids": torch.tensor([0, 1]),
                        }
                    ]
                elif idx == 1:
                    return [
                        {
                            "input_ids": torch.tensor([3, 4, 5]),
                            "attention_mask": torch.ones(3),
                            "labels": torch.tensor([3, 4, 5]),
                            "position_ids": torch.tensor([0, 1, 2]),
                        },
                        {
                            "input_ids": torch.tensor([6]),
                            "attention_mask": torch.ones(1),
                            "labels": torch.tensor([6]),
                            "position_ids": torch.tensor([0]),
                        },
                    ]
                elif idx == 2:
                    return [
                        {
                            "input_ids": torch.tensor([7, 8]),
                            "attention_mask": torch.ones(2),
                            "labels": torch.tensor([7, 8]),
                            "position_ids": torch.tensor([0, 1]),
                        },
                        {
                            "input_ids": torch.tensor([9]),
                            "attention_mask": torch.ones(1),
                            "labels": torch.tensor([9]),
                            "position_ids": torch.tensor([0]),
                        },
                        {
                            "input_ids": torch.tensor([10, 11]),
                            "attention_mask": torch.ones(2),
                            "labels": torch.tensor([10, 11]),
                            "position_ids": torch.tensor([0, 1]),
                        },
                    ]
                else:
                    return [
                        {
                            "input_ids": torch.tensor([12, 13, 14]),
                            "attention_mask": torch.ones(3),
                            "labels": torch.tensor([12, 13, 14]),
                            "position_ids": torch.tensor([0, 1, 2]),
                        }
                    ]

        var_builder = DataLoaderBuilder(
            dataset=VariablePackedDataset(),
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )
        var_dl = var_builder.build(verbose=False)
        var_batch = next(iter(var_dl))
        assert len(var_batch) == 1
        assert var_batch[0]["input_ids"].shape == (1, 4)
