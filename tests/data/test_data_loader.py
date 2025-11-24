import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from xorl.data.data_loader import (
    MicroBatchCollator,
    DistributedDataloader,
    DataLoaderBuilder,
)
from xorl.data.collators import DataCollator
from typing import Dict, Sequence, Any

pytestmark = [pytest.mark.cpu, pytest.mark.dataloader]


class SimpleCollator(DataCollator):
    """Simple collator for testing that stacks tensors."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in features[0].keys():
            result[key] = torch.stack([f[key] for f in features])
        return result


class TestMicroBatchCollator:
    """Test suite for MicroBatchCollator."""

    def test_splits_into_correct_number_of_micro_batches(self):
        """Test that the collator splits data into correct number of micro-batches."""
        micro_batch_size = 2
        gradient_accumulation_steps = 3
        internal_collator = SimpleCollator()

        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=internal_collator,
        )

        # Create 6 samples (2 * 3)
        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(6)
        ]

        result = collator(features)

        # Should return 3 micro-batches
        assert len(result) == 3

    def test_each_micro_batch_has_correct_size(self):
        """Test that each micro-batch has the correct size."""
        micro_batch_size = 2
        gradient_accumulation_steps = 3
        internal_collator = SimpleCollator()

        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=internal_collator,
        )

        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(6)
        ]

        result = collator(features)

        # Each micro-batch should have batch size of 2
        for micro_batch in result:
            assert micro_batch["input_ids"].shape[0] == 2

    def test_micro_batches_contain_correct_data(self):
        """Test that micro-batches contain the correct sequential data."""
        micro_batch_size = 2
        gradient_accumulation_steps = 2
        internal_collator = SimpleCollator()

        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=internal_collator,
        )

        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i * 10])}
            for i in range(4)
        ]

        result = collator(features)

        # First micro-batch should have samples 0, 1
        assert torch.equal(result[0]["input_ids"], torch.tensor([[0], [1]]))
        assert torch.equal(result[0]["labels"], torch.tensor([[0], [10]]))

        # Second micro-batch should have samples 2, 3
        assert torch.equal(result[1]["input_ids"], torch.tensor([[2], [3]]))
        assert torch.equal(result[1]["labels"], torch.tensor([[20], [30]]))

    def test_handles_single_micro_batch(self):
        """Test handling when gradient_accumulation_steps = 1."""
        micro_batch_size = 4
        gradient_accumulation_steps = 1
        internal_collator = SimpleCollator()

        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=internal_collator,
        )

        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(4)
        ]

        result = collator(features)

        # Should return 1 micro-batch
        assert len(result) == 1
        assert result[0]["input_ids"].shape[0] == 4

    def test_internal_collator_is_called(self):
        """Test that the internal collator is called for each micro-batch."""
        micro_batch_size = 2
        gradient_accumulation_steps = 2
        internal_collator = Mock(spec=DataCollator)
        internal_collator.side_effect = lambda x: {
            "input_ids": torch.stack([f["input_ids"] for f in x])
        }

        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=internal_collator,
        )

        features = [
            {"input_ids": torch.tensor([i])} for i in range(4)
        ]

        result = collator(features)

        # Internal collator should be called twice (once per micro-batch)
        assert internal_collator.call_count == 2

    def test_with_uneven_division(self):
        """Test that error is raised when features don't match expected length."""
        micro_batch_size = 3
        gradient_accumulation_steps = 2
        internal_collator = SimpleCollator()

        collator = MicroBatchCollator(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            internal_collator=internal_collator,
        )

        # 5 samples: expected is 6 (3 * 2), so this should raise an error
        features = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(5)
        ]

        with pytest.raises(ValueError, match="Expected 6 samples"):
            collator(features)


class TestDistributedDataloader:
    """Test suite for DistributedDataloader."""

    def test_set_epoch_calls_sampler_set_epoch(self):
        """Test that set_epoch calls sampler.set_epoch if available."""
        dataloader = DistributedDataloader.__new__(DistributedDataloader)
        dataloader.sampler = Mock()
        dataloader.sampler.set_epoch = Mock()
        dataloader.dataset = Mock()

        dataloader.set_epoch(5)

        dataloader.sampler.set_epoch.assert_called_once_with(5)

    def test_set_epoch_calls_dataset_set_epoch_if_no_sampler(self):
        """Test that set_epoch calls dataset.set_epoch if sampler doesn't have it."""
        dataloader = DistributedDataloader.__new__(DistributedDataloader)
        dataloader.sampler = None
        dataloader.dataset = Mock()
        dataloader.dataset.set_epoch = Mock()

        dataloader.set_epoch(5)

        dataloader.dataset.set_epoch.assert_called_once_with(5)

    def test_set_epoch_handles_missing_methods(self):
        """Test that set_epoch doesn't crash if methods are missing."""
        dataloader = DistributedDataloader.__new__(DistributedDataloader)
        dataloader.sampler = None
        dataloader.dataset = Mock(spec=[])  # No set_epoch method

        # Should not raise an error
        dataloader.set_epoch(5)


class TestDataLoaderBuilder:
    """Test suite for DataLoaderBuilder class."""

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_creates_dataloader_with_correct_batch_size(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that dataloader is created with correct batch size."""
        mock_ps = Mock()
        mock_ps.dp_size = 2
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            num_workers=4,
            seed=42,
        ).build(verbose=False)

        # Dataloader should be created with batch_size = micro_batch_size * gradient_accumulation_steps
        assert mock_dataloader_cls.call_args[1]["batch_size"] == 8

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_creates_sampler_with_correct_params(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that sampler is created with correct parameters."""
        mock_ps = Mock()
        mock_ps.dp_size = 4
        mock_ps.dp_rank = 1
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            seed=123,
        ).build(verbose=False)

        # Check sampler was created with correct args
        mock_sampler_cls.assert_called_once()
        call_kwargs = mock_sampler_cls.call_args[1]
        assert call_kwargs["num_replicas"] == 4
        assert call_kwargs["rank"] == 1
        assert call_kwargs["shuffle"] is True
        assert call_kwargs["seed"] == 123

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_uses_default_collate_fn_when_none_provided(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that default collate_fn is used when none is provided."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        ).build(verbose=False)

        # Collate function should be a MicroBatchCollator
        call_kwargs = mock_dataloader_cls.call_args[1]
        assert isinstance(call_kwargs["collate_fn"], MicroBatchCollator)

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_adds_sequence_shard_collator_when_sp_enabled(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that TextSequenceShardCollator is added when SP is enabled."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 2
        mock_ps.sp_enabled = True
        mock_parallel_state.return_value = mock_ps

        DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        ).build(verbose=False)

        # The collate_fn should be MicroBatchCollator wrapping CollatePipeline with SP collator
        call_kwargs = mock_dataloader_cls.call_args[1]
        collate_fn = call_kwargs["collate_fn"]
        assert isinstance(collate_fn, MicroBatchCollator)

        # Check that internal collator is CollatePipeline with 4 collators
        # (Flatten + ToTensor + PackingConcat + TextSequenceShard)
        from xorl.data.collators import CollatePipeline
        assert isinstance(collate_fn.internal_collator, CollatePipeline)
        assert len(collate_fn.internal_collator.data_collators) == 4

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_uses_custom_collator_when_added(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that custom collator is used when added to pipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        custom_collator = SimpleCollator()

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=False,
        )
        builder.add_collator(custom_collator)
        builder.build(verbose=False)

        # The MicroBatchCollator should wrap our custom collator
        call_kwargs = mock_dataloader_cls.call_args[1]
        micro_batch_collator = call_kwargs["collate_fn"]
        assert isinstance(micro_batch_collator, MicroBatchCollator)
        assert micro_batch_collator.internal_collator is custom_collator

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_uses_multiple_collators_when_added(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that multiple collators are wrapped in CollatePipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        collator1 = SimpleCollator()
        collator2 = SimpleCollator()

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=False,
        )
        builder.add_collator(collator1)
        builder.add_collator(collator2)
        builder.build(verbose=False)

        # Should create CollatePipeline with both collators
        call_kwargs = mock_dataloader_cls.call_args[1]
        micro_batch_collator = call_kwargs["collate_fn"]
        from xorl.data.collators import CollatePipeline
        assert isinstance(micro_batch_collator.internal_collator, CollatePipeline)
        assert len(micro_batch_collator.internal_collator.data_collators) == 2

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_passes_dataloader_kwargs_correctly(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that dataloader kwargs are passed correctly."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=16,
            drop_last=False,
            pin_memory=False,
            prefetch_factor=4,
        ).build(verbose=False)

        call_kwargs = mock_dataloader_cls.call_args[1]
        assert call_kwargs["num_workers"] == 16
        assert call_kwargs["drop_last"] is False
        assert call_kwargs["pin_memory"] is False
        assert call_kwargs["prefetch_factor"] == 4


class TestDataLoaderBuilderPipeline:
    """Test suite for DataLoaderBuilder pipeline manipulation methods."""

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_add_collator_to_end(self, mock_parallel_state, fake_text_dataset):
        """Test adding a collator to the end of the pipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        
        custom_collator = SimpleCollator()
        builder.add_collator(custom_collator, position="end")

        pipeline = builder.get_collator_pipeline()
        assert len(pipeline) == 4  # 3 default (Flatten + ToTensor + PackingConcat) + 1 custom
        assert pipeline[-1] is custom_collator

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_add_collator_to_start(self, mock_parallel_state, fake_text_dataset):
        """Test adding a collator to the start of the pipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        
        custom_collator = SimpleCollator()
        builder.add_collator(custom_collator, position="start")

        pipeline = builder.get_collator_pipeline()
        assert len(pipeline) == 4  # 3 default (Flatten + ToTensor + PackingConcat) + 1 custom
        assert pipeline[0] is custom_collator

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_insert_collator(self, mock_parallel_state, fake_text_dataset):
        """Test inserting a collator at a specific position."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        
        # Add two more collators
        builder.add_collator(SimpleCollator())
        builder.add_collator(SimpleCollator())

        # Insert in the middle
        custom_collator = SimpleCollator()
        builder.insert_collator(custom_collator, index=1)

        pipeline = builder.get_collator_pipeline()
        assert len(pipeline) == 6  # 3 default + 2 added + 1 inserted
        assert pipeline[1] is custom_collator

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_remove_collator(self, mock_parallel_state, fake_text_dataset):
        """Test removing a collator from the pipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        
        # Add a collator (now 4 total: 3 default + 1 added)
        builder.add_collator(SimpleCollator())
        assert len(builder.get_collator_pipeline()) == 4

        # Remove the first collator (now 3 remaining)
        builder.remove_collator(0)
        assert len(builder.get_collator_pipeline()) == 3

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_method_chaining(self, mock_parallel_state, fake_text_dataset):
        """Test that pipeline methods support method chaining."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = (DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        .add_collator(SimpleCollator(), position="end")
        .add_collator(SimpleCollator(), position="start")
        .insert_collator(SimpleCollator(), index=1))

        pipeline = builder.get_collator_pipeline()
        assert len(pipeline) == 6  # 3 default + 3 added

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_get_collator_pipeline_returns_copy(self, mock_parallel_state, fake_text_dataset):
        """Test that get_collator_pipeline returns a copy, not the original list."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        
        pipeline1 = builder.get_collator_pipeline()
        pipeline2 = builder.get_collator_pipeline()
        
        # Modify one copy
        pipeline1.append(SimpleCollator())

        # Should not affect the other copy or the builder (default is 3 collators)
        assert len(pipeline2) == 3
        assert len(builder.get_collator_pipeline()) == 3

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.StatefulDistributedSampler')
    @patch('xorl.data.data_loader.DistributedDataloader')
    def test_collate_fn_accessible_after_build(
        self, mock_dataloader_cls, mock_sampler_cls, mock_parallel_state, fake_text_dataset
    ):
        """Test that collate_fn and sampler are accessible after building."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        
        # Before build, collate_fn and sampler should be None
        assert builder.collate_fn is None
        assert builder.sampler is None
        
        # Build the dataloader
        dataloader = builder.build(verbose=False)
        
        # After build, collate_fn and sampler should be accessible
        assert builder.collate_fn is not None
        assert isinstance(builder.collate_fn, MicroBatchCollator)
        assert builder.sampler is not None
        
        # Verify the collate_fn has the correct parameters
        assert builder.collate_fn.micro_batch_size == 2
        assert builder.collate_fn.gradient_accumulation_steps == 2

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_use_default_collators_flag(self, mock_parallel_state, fake_text_dataset):
        """Test that use_default_collators flag controls initial pipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        # With use_default_collators=True (default)
        builder_with_defaults = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=True,
        )
        assert len(builder_with_defaults.get_collator_pipeline()) > 0

        # With use_default_collators=False
        builder_empty = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=False,
        )
        assert len(builder_empty.get_collator_pipeline()) == 0

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_add_collator_invalid_position(self, mock_parallel_state, fake_text_dataset):
        """Test that add_collator raises error for invalid position."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )

        with pytest.raises(ValueError, match="Invalid position: middle"):
            builder.add_collator(SimpleCollator(), position="middle")

    @patch('xorl.data.data_loader.get_parallel_state')
    def test_build_with_empty_collator_list_raises_error(self, mock_parallel_state, fake_text_dataset):
        """Test that building with empty collator list raises error."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=False,
        )

        with pytest.raises(ValueError, match="Collator pipeline is empty"):
            builder.build(verbose=False)

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.logger')
    def test_print_pipeline_with_collators(self, mock_logger, mock_parallel_state, fake_text_dataset):
        """Test that print_pipeline logs correct information."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=4,
            gradient_accumulation_steps=3,
        )
        builder.add_collator(SimpleCollator())

        builder.print_pipeline()

        # Verify logging was called
        assert mock_logger.info_rank0.call_count > 0

        # Check that it logged the micro-batch info
        logged_text = ' '.join([str(call[0][0]) for call in mock_logger.info_rank0.call_args_list])
        assert 'micro_batch_size: 4' in logged_text
        assert 'gradient_accumulation_steps: 3' in logged_text
        assert 'dataloader_batch_size: 12' in logged_text

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.logger')
    def test_print_pipeline_empty(self, mock_logger, mock_parallel_state, fake_text_dataset):
        """Test print_pipeline with empty pipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=False,
        )

        builder.print_pipeline()

        # Should log empty pipeline message
        logged_text = ' '.join([str(call[0][0]) for call in mock_logger.info_rank0.call_args_list])
        assert '(empty pipeline)' in logged_text or 'empty' in logged_text.lower()

    @patch('xorl.data.data_loader.get_parallel_state')
    @patch('xorl.data.data_loader.logger')
    def test_print_pipeline_with_collate_pipeline(self, mock_logger, mock_parallel_state, fake_text_dataset):
        """Test print_pipeline with nested CollatePipeline."""
        mock_ps = Mock()
        mock_ps.dp_size = 1
        mock_ps.dp_rank = 0
        mock_ps.sp_size = 1
        mock_ps.sp_enabled = False
        mock_parallel_state.return_value = mock_ps

        from xorl.data.collators import CollatePipeline

        nested_pipeline = CollatePipeline([SimpleCollator(), SimpleCollator()])

        builder = DataLoaderBuilder(
            dataset=fake_text_dataset,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            use_default_collators=False,
        )
        builder.add_collator(nested_pipeline)

        builder.print_pipeline()

        # Should detect nested pipeline
        assert mock_logger.info_rank0.call_count > 0
