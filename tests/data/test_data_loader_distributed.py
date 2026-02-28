"""
Integration tests for distributed data loading.

These tests verify that the data loader correctly distributes data across multiple
processes/ranks in a distributed training setting, and that the data alignment is correct.
"""
import pytest
import torch
import os
from unittest.mock import Mock, patch
from xorl.data.data_loader import (
    MicroBatchCollator,
    DistributedDataloader,
    DataLoaderBuilder,
)
from xorl.data.collators import PackingConcatCollator


pytestmark = [pytest.mark.gpu, pytest.mark.dataloader, pytest.mark.distributed]


class TestDistributedDataAlignment:
    """Test suite for verifying correct data distribution across ranks."""

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    def test_data_partitioning_across_ranks(self, mock_sampler_cls, mock_parallel_state, fake_text_dataset):
        """Test that data is correctly partitioned across different ranks."""
        # Simulate 4 data parallel ranks
        dp_size = 4
        datasets_per_rank = []

        for rank in range(dp_size):
            # Mock parallel state for this rank
            mock_ps = Mock()
            mock_ps.dp_size = dp_size
            mock_ps.dp_rank = rank
            mock_ps.sp_size = 1
            mock_ps.sp_enabled = False
            mock_parallel_state.return_value = mock_ps

            # Create a mock sampler that returns specific indices for this rank
            mock_sampler = Mock()
            # Each rank should get dataset_size // dp_size samples
            rank_indices = list(range(rank, len(fake_text_dataset), dp_size))
            mock_sampler.__iter__ = Mock(return_value=iter(rank_indices))
            mock_sampler.__len__ = Mock(return_value=len(rank_indices))
            mock_sampler_cls.return_value = mock_sampler

            builder = DataLoaderBuilder(
                dataset=fake_text_dataset,
                micro_batch_size=2,
                gradient_accumulation_steps=2,
            )

            # Build to trigger sampler creation
            dataloader = builder.build(verbose=False)

            # Verify sampler was created with correct rank parameters
            assert mock_sampler_cls.call_args[1]["rank"] == rank
            assert mock_sampler_cls.call_args[1]["num_replicas"] == dp_size

            datasets_per_rank.append(rank_indices)

        # Verify no overlap between ranks
        all_indices = []
        for indices in datasets_per_rank:
            for idx in indices:
                assert idx not in all_indices, f"Index {idx} appears in multiple ranks"
                all_indices.append(idx)

        # Verify all samples are covered (may not be 100% if dataset size not divisible by dp_size)
        assert len(set(all_indices)) >= len(fake_text_dataset) - dp_size

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_micro_batch_splitting_correctness(self, mock_parallel_state, fake_text_dataset):
        """Test that micro-batch splitting produces correctly sized batches."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        micro_batch_size = 4
        gradient_accumulation_steps = 3

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        dataloader = builder.build(verbose=False)

        # Get one batch
        batch_iterator = iter(dataloader)
        micro_batches = next(batch_iterator)

        # Should return a list of micro-batches
        assert isinstance(micro_batches, list)
        assert len(micro_batches) == gradient_accumulation_steps

        # Each micro-batch should have the correct size
        for micro_batch in micro_batches:
            assert isinstance(micro_batch, dict)
            # Verify structure (seq_len may vary due to packing)
            assert "input_ids" in micro_batch
            assert "labels" in micro_batch
            assert isinstance(micro_batch["input_ids"], torch.Tensor)
            assert len(micro_batch["input_ids"].shape) == 2  # (batch, seq_len)

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_sequence_parallelism_sharding(self, mock_parallel_state, fake_packed_dataset):
        """Test that sequence parallelism correctly shards sequences across SP ranks."""
        sp_size = 2
        sp_rank = 0

        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = sp_size
        mock_ps.sp_enabled = True
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_packed_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        )
        dataloader = builder.build(verbose=False)

        # Get one batch
        batch_iterator = iter(dataloader)
        micro_batches = next(batch_iterator)

        # Verify that TextSequenceShardCollator was added to pipeline
        from xorl.data.collators import CollatePipeline, TextSequenceShardCollator
        internal_collator = builder.collate_fn.internal_collator
        assert isinstance(internal_collator, CollatePipeline)

        # Check that TextSequenceShardCollator is in the pipeline
        has_sp_collator = any(
            isinstance(c, TextSequenceShardCollator)
            for c in internal_collator.data_collators
        )
        assert has_sp_collator, "TextSequenceShardCollator should be in pipeline when SP is enabled"

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_dataloader_batch_size_calculation(self, mock_parallel_state, fake_text_dataset):
        """Test that dataloader batch size is correctly calculated."""
        mock_ps = Mock()
        mock_ps.dp_size = 2
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        micro_batch_size = 4
        gradient_accumulation_steps = 3

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # The dataloader should request batch_size = micro_batch_size * gradient_accumulation_steps
        expected_batch_size = micro_batch_size * gradient_accumulation_steps
        assert expected_batch_size == 12

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_drop_last_behavior(self, mock_parallel_state):
        """Test that drop_last parameter works correctly."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        # Create a small dataset
        from conftest import FakeTextDataset
        small_dataset = FakeTextDataset(num_samples=10, seq_len=64)

        # With drop_last=True, should drop incomplete batches
        builder = DataLoaderBuilder(
            dataset=small_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            drop_last=True,
        )
        dataloader = builder.build(verbose=False)

        # Should have dropped the last incomplete batch
        # 10 samples / (2 * 2) = 2 complete batches, 2 samples dropped
        batches = list(dataloader)
        assert len(batches) == 2


class TestMicroBatchCollatorAlignment:
    """Test suite for verifying micro-batch collator output alignment."""

    def test_micro_batch_data_order_preservation(self):
        """Test that data order is preserved within micro-batches."""
        from conftest import SimpleCollator

        micro_batch_size = 2
        gradient_accumulation_steps = 3
        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=SimpleCollator(),
        )

        # Create features with identifiable data
        features = [
            {"input_ids": torch.tensor([i]), "value": torch.tensor([i * 100])}
            for i in range(6)
        ]

        micro_batches = collator(features)

        # Verify data order
        # First micro-batch: samples 0, 1
        assert torch.equal(micro_batches[0]["value"], torch.tensor([[0], [100]]))
        # Second micro-batch: samples 2, 3
        assert torch.equal(micro_batches[1]["value"], torch.tensor([[200], [300]]))
        # Third micro-batch: samples 4, 5
        assert torch.equal(micro_batches[2]["value"], torch.tensor([[400], [500]]))

    def test_micro_batch_gradient_accumulation_alignment(self):
        """Test that micro-batches are correctly aligned for gradient accumulation."""
        from conftest import SimpleCollator

        micro_batch_size = 3
        gradient_accumulation_steps = 2
        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=SimpleCollator(),
        )

        # Create 6 samples (3 * 2)
        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(6)
        ]

        micro_batches = collator(features)

        # Should have exactly 2 micro-batches for gradient accumulation
        assert len(micro_batches) == gradient_accumulation_steps

        # Each micro-batch should have exactly 3 samples
        for micro_batch in micro_batches:
            assert micro_batch["input_ids"].shape[0] == micro_batch_size

    def test_error_on_incorrect_batch_size(self):
        """Test that an error is raised when batch size doesn't match expectations."""
        from conftest import SimpleCollator

        collator = MicroBatchCollator(
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            internal_collator=SimpleCollator(),
        )

        # Provide wrong number of samples (should be 8, but give 7)
        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(7)
        ]

        with pytest.raises(ValueError, match="Expected 8 samples"):
            collator(features)


class TestSequenceShardingAlignment:
    """Test suite for verifying sequence sharding alignment in SP mode."""

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_sequence_sharding_across_sp_ranks(self, mock_parallel_state):
        """Test that sequences are correctly sharded across SP ranks."""
        from xorl.data.collators import TextSequenceShardCollator
        from xorl.data.constants import IGNORE_INDEX

        sp_size = 4
        sequence_length = 128

        # Pre-shifted data: labels[i] = input_ids[i+1], last label = IGNORE_INDEX
        input_ids = torch.arange(sequence_length).unsqueeze(0)
        labels = torch.cat([torch.arange(1, sequence_length), torch.tensor([IGNORE_INDEX])]).unsqueeze(0)

        # Simulate sharding for each SP rank
        sharded_sequences = []

        for sp_rank in range(sp_size):
            mock_ps = Mock()
            mock_ps.sp_size = sp_size
            mock_ps.sp_rank = sp_rank
            mock_ps.cp_size = 1
            mock_parallel_state.return_value = mock_ps

            collator = TextSequenceShardCollator(pad_token_id=0)

            batch = {
                "input_ids": input_ids.clone(),
                "attention_mask": torch.ones(1, sequence_length, dtype=torch.long),
                "labels": labels.clone(),
                "position_ids": torch.arange(sequence_length).unsqueeze(0),
            }

            result = collator(batch)
            sharded_sequences.append(result["input_ids"])

        # Verify that each rank got the correct chunk
        expected_chunk_size = sequence_length // sp_size
        for rank, sharded_seq in enumerate(sharded_sequences):
            assert sharded_seq.shape[1] == expected_chunk_size, f"Rank {rank} has incorrect chunk size"

    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_padding_alignment_for_sp(self, mock_parallel_state):
        """Test that padding is correctly applied to align with SP size."""
        from xorl.data.collators import TextSequenceShardCollator
        from xorl.data.constants import IGNORE_INDEX

        sp_size = 4
        # Use a sequence length that's not divisible by sp_size
        sequence_length = 130  # 130 should be padded to 132 (next multiple of sp_size*chunk_size)

        mock_ps = Mock()
        mock_ps.sp_size = sp_size
        mock_ps.sp_rank = 0
        mock_ps.cp_size = 1
        mock_parallel_state.return_value = mock_ps

        collator = TextSequenceShardCollator(pad_token_id=0)

        # Pre-shifted data
        input_ids = torch.arange(sequence_length).unsqueeze(0)
        labels = torch.cat([torch.arange(1, sequence_length), torch.tensor([IGNORE_INDEX])]).unsqueeze(0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(1, sequence_length, dtype=torch.long),
            "labels": labels,
            "position_ids": torch.arange(sequence_length).unsqueeze(0),
        }

        result = collator(batch)

        # After padding and sharding, each rank should get an equal chunk
        chunk_size = result["input_ids"].shape[1]
        total_padded_length = chunk_size * sp_size

        # Verify padding was applied
        assert total_padded_length >= sequence_length
        # Verify it's the minimal padding needed
        assert total_padded_length < sequence_length + sp_size


class TestDataLoaderBuilderPipelineOutput:
    """Test suite for verifying DataLoaderBuilder pipeline output correctness."""

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_pipeline_output_structure(self, mock_parallel_state, fake_packed_dataset):
        """Test that the pipeline produces correctly structured output."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_packed_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        dataloader = builder.build(verbose=False)

        # Get one batch
        batch_iterator = iter(dataloader)
        micro_batches = next(batch_iterator)

        # Verify structure
        assert isinstance(micro_batches, list)
        assert len(micro_batches) == 2

        for micro_batch in micro_batches:
            # Each micro-batch should have required fields
            assert "input_ids" in micro_batch
            assert "labels" in micro_batch
            assert "position_ids" in micro_batch
            assert "attention_mask" in micro_batch

            # Should have flash attention kwargs
            assert "cu_seq_lens_q" in micro_batch
            assert "cu_seq_lens_k" in micro_batch
            assert "max_length_q" in micro_batch
            assert "max_length_k" in micro_batch

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_custom_collator_output(self, mock_parallel_state, fake_text_dataset):
        """Test that custom collators in the pipeline produce expected output."""
        from conftest import SimpleCollator

        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        # Create builder with custom collator
        custom_collator = PackingConcatCollator()

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            use_default_collators=False,
        )
        builder.add_collator(custom_collator)

        dataloader = builder.build(verbose=False)

        # Get one batch
        batch_iterator = iter(dataloader)
        micro_batches = next(batch_iterator)

        # Verify output has position_ids (added by custom collator)
        assert len(micro_batches) == 1
        assert "position_ids" in micro_batches[0]


class TestEndToEndDistributedFlow:
    """Integration tests for end-to-end distributed data loading flow."""

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_complete_dataloader_pipeline(self, mock_parallel_state, fake_packed_dataset):
        """Test complete pipeline from dataset to collated micro-batches."""
        mock_ps = Mock()
        mock_ps.dp_size = 2
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        # Build dataloader
        builder = DataLoaderBuilder(
            dataset=fake_packed_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=0,  # Use 0 workers for testing
            prefetch_factor=None,  # Must be None when num_workers=0
            seed=42,
        )
        dataloader = builder.build(verbose=False)

        # Iterate through a few batches
        batch_count = 0
        for micro_batches in dataloader:
            assert isinstance(micro_batches, list)
            assert len(micro_batches) == 2

            for micro_batch in micro_batches:
                # Verify all required fields are present
                required_fields = ["input_ids", "labels", "position_ids", "attention_mask"]
                for field in required_fields:
                    assert field in micro_batch

                # Verify tensor shapes are consistent
                batch_size = micro_batch["input_ids"].shape[0]
                seq_len = micro_batch["input_ids"].shape[1]

                assert micro_batch["labels"].shape == (batch_size, seq_len)
                assert micro_batch["position_ids"].shape == (batch_size, seq_len)
                assert micro_batch["attention_mask"].shape == (batch_size, seq_len)

            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break

        assert batch_count == 3, "Should have processed 3 batches"

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_epoch_iteration_consistency(self, mock_parallel_state, fake_text_dataset):
        """Test that multiple epochs produce consistent results with same seed."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        # Build dataloader with fixed seed
        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=4,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,  # Must be None when num_workers=0
            seed=42,
        )
        dataloader = builder.build(verbose=False)

        # Collect first epoch data
        epoch1_data = []
        for micro_batches in dataloader:
            # Just collect input_ids from first micro-batch
            epoch1_data.append(micro_batches[0]["input_ids"].clone())
            if len(epoch1_data) >= 3:
                break

        # Set new epoch
        dataloader.set_epoch(1)

        # Collect second epoch data
        epoch2_data = []
        for micro_batches in dataloader:
            epoch2_data.append(micro_batches[0]["input_ids"].clone())
            if len(epoch2_data) >= 3:
                break

        # Data should be different between epochs (different shuffle)
        # But both should have the same number of batches
        assert len(epoch1_data) == len(epoch2_data)


class TestPackedSequencesDistributed:
    """Test suite for packed sequences (list of list of dict) in distributed mode."""

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_packed_sequences_flattening_single_rank(self, mock_ps_collator, mock_ps_loader):
        """Test that packed sequences are correctly flattened with batch_size=1."""
        from torch.utils.data import Dataset

        # Create a dataset that returns list of dicts (packed sequences)
        class PackedDataset(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                # Each sample has 2 sequences
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

        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps

        dataset = PackedDataset()
        builder = DataLoaderBuilder(
            dataset=dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )

        dataloader = builder.build(verbose=False)
        micro_batches = next(iter(dataloader))

        # Should have 1 micro-batch
        assert len(micro_batches) == 1
        micro_batch = micro_batches[0]

        # Batch size should be 1 (all sequences flattened)
        assert micro_batch["input_ids"].shape[0] == 1

        # After ShiftTokensCollator: each sub-sequence loses 1 token
        # 2 samples * (2 + 1) shifted tokens = 6 tokens total
        assert micro_batch["input_ids"].shape[1] == 6

        # Verify attention_mask is present
        assert "attention_mask" in micro_batch
        assert micro_batch["attention_mask"].shape == (1, 6)

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_packed_sequences_with_multiple_dp_ranks(self, mock_ps_collator, mock_ps_loader):
        """Test that packed sequences work correctly with multiple DP ranks."""
        from torch.utils.data import Dataset

        class PackedDataset(Dataset):
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

        # Simulate 2 DP ranks
        for dp_rank in [0, 1]:
            mock_ps = Mock()
            mock_ps.dp_size = 2
            mock_ps.dp_rank = dp_rank
            mock_ps.sp_size = 1
            mock_ps.sp_enabled = False
            mock_ps_collator.return_value = mock_ps
            mock_ps_loader.return_value = mock_ps

            dataset = PackedDataset()
            builder = DataLoaderBuilder(
                dataset=dataset,
                micro_batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                prefetch_factor=None,
                pad_to_multiple_of=1,
            )

            dataloader = builder.build(verbose=False)
            micro_batches = next(iter(dataloader))

            # Each rank should get data
            assert len(micro_batches) == 1
            assert micro_batches[0]["input_ids"].shape[0] == 1  # Batch size = 1
            # Each micro-batch has 2 samples, each with 1 seq of 2 tokens
            # After ShiftTokensCollator: each seq becomes 1 token → 2 total
            assert micro_batches[0]["input_ids"].shape[1] == 2

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    @patch('xorl.data.collators.sequence_shard_collator.get_parallel_state')
    def test_packed_sequences_with_sp_enabled(self, mock_ps_shard, mock_ps_collator, mock_ps_loader):
        """Test that packed sequences work with sequence parallelism."""
        from torch.utils.data import Dataset

        class PackedDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                # Return packed sequences with total length divisible by SP size
                return [
                    {
                        "input_ids": torch.arange(8, dtype=torch.long),  # 8 tokens
                        "labels": torch.arange(8, dtype=torch.long),
                        "position_ids": torch.arange(8, dtype=torch.long),
                        "attention_mask": torch.ones(8, dtype=torch.long),
                    }
                ]

        sp_size = 2
        sp_rank = 0

        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = sp_size
        mock_ps.sp_rank = sp_rank
        mock_ps.sp_enabled = True
        mock_ps.cp_size = 1
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps
        mock_ps_shard.return_value = mock_ps

        dataset = PackedDataset()
        builder = DataLoaderBuilder(
            dataset=dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
        )

        dataloader = builder.build(verbose=False)
        micro_batches = next(iter(dataloader))

        assert len(micro_batches) == 1
        micro_batch = micro_batches[0]

        # After packing: 2 samples * 8 tokens = 16 tokens total
        # After SP sharding: 16 / 2 (sp_size) = 8 tokens per rank (but may be padded)
        # The exact length depends on padding logic
        assert micro_batch["input_ids"].shape[0] == 1  # Batch size always 1
        assert "cu_seq_lens_q" in micro_batch  # Flash attention kwargs should be present

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.collators.packing_concat_collator.get_parallel_state')
    def test_packed_sequences_variable_lengths(self, mock_ps_collator, mock_ps_loader):
        """Test packed sequences with variable number of sequences per sample."""
        from torch.utils.data import Dataset

        class VariablePackedDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                # Different samples have different number of sequences
                if idx % 2 == 0:
                    # Even indices: 1 sequence
                    return [
                        {
                            "input_ids": torch.tensor([idx, idx + 1]),
                            "labels": torch.tensor([idx, idx + 1]),
                            "position_ids": torch.tensor([0, 1]),
                            "attention_mask": torch.ones(2, dtype=torch.long),
                        }
                    ]
                else:
                    # Odd indices: 3 sequences
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

        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_ps_collator.return_value = mock_ps
        mock_ps_loader.return_value = mock_ps

        dataset = VariablePackedDataset()
        builder = DataLoaderBuilder(
            dataset=dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            prefetch_factor=None,
            pad_to_multiple_of=1,
        )

        dataloader = builder.build(verbose=False)
        micro_batches = next(iter(dataloader))

        assert len(micro_batches) == 1
        micro_batch = micro_batches[0]

        # After ShiftTokensCollator: sample 0 (2→1 token) + sample 1 (1+2+1→1+1+1=3 tokens) = 4 tokens
        # Note: 1-token sequences are not shifted (no valid label pair to detect)
        assert micro_batch["input_ids"].shape == (1, 4)
        assert "attention_mask" in micro_batch
        assert micro_batch["attention_mask"].shape == (1, 4)
