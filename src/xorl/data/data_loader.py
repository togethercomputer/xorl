from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .collators import (
    CollatePipeline,
    DataCollator,
    FlattenCollator,
    PackingConcatCollator,
    TextSequenceShardCollator,
)


if TYPE_CHECKING:
    from torch.utils.data import Dataset


logger = logging.get_logger(__name__)


class MicroBatchCollator(DataCollator):
    """
    Collator that splits a large batch into multiple micro-batches for gradient accumulation.
    
    This collator receives a batch of size (micro_batch_size * gradient_accumulation_steps)
    and splits it into gradient_accumulation_steps separate micro-batches, each of size
    micro_batch_size. Each micro-batch is then processed by the internal_collator.
    
    Args:
        micro_batch_size: Size of each micro-batch
        gradient_accumulation_steps: Number of micro-batches to create
        internal_collator: The collator to apply to each micro-batch
    """
    
    def __init__(
        self,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        internal_collator: DataCollator,
    ):
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.internal_collator = internal_collator
    
    def __call__(
        self, features: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split features into micro-batches and collate each one.
        
        Args:
            features: List of samples (length = micro_batch_size * gradient_accumulation_steps)
        
        Returns:
            List of collated micro-batches (length = gradient_accumulation_steps)
        """
        expected_length = self.micro_batch_size * self.gradient_accumulation_steps
        if len(features) != expected_length:
            raise ValueError(
                f"Expected {expected_length} samples "
                f"(micro_batch_size={self.micro_batch_size} * gradient_accumulation_steps={self.gradient_accumulation_steps}), "
                f"but got {len(features)} samples. "
                f"This usually means the dataset length is not divisible by the dataloader batch size. "
                f"Consider setting drop_last=True in build_dataloader."
            )
        
        micro_batches = []
        for i in range(0, len(features), self.micro_batch_size):
            micro_batch_features = features[i : i + self.micro_batch_size]
            collated_micro_batch = self.internal_collator(micro_batch_features)
            micro_batches.append(collated_micro_batch)
        
        assert len(micro_batches) == self.gradient_accumulation_steps, (
            f"Internal error: Expected {self.gradient_accumulation_steps} micro-batches, "
            f"but got {len(micro_batches)}"
        )
        
        return micro_batches


class DistributedDataloader(StatefulDataLoader):
    dataset: "Dataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class DataLoaderBuilder:
    """
    Builder class for constructing distributed dataloaders with gradient accumulation support.
    
    This class provides a flexible interface for building dataloaders, allowing customization
    of collators, samplers, and other components through method chaining or inheritance.
    
    Example:
        # Basic usage
        builder = DataLoaderBuilder(dataset, micro_batch_size=2, gradient_accumulation_steps=4)
        dataloader = builder.build()
        
        # Add custom collators to the pipeline
        builder = DataLoaderBuilder(dataset, micro_batch_size=2, gradient_accumulation_steps=4)
        builder.add_collator(MyCustomCollator(), position="end")
        builder.print_pipeline()  # Visualize the pipeline
        dataloader = builder.build()
        
        # Method chaining
        dataloader = (DataLoaderBuilder(dataset, micro_batch_size=2, gradient_accumulation_steps=4)
                      .add_collator(MyPreprocessor(), position="start")
                      .add_collator(MyPostprocessor(), position="end")
                      .build())
        
        # Custom builder subclass
        class CustomBuilder(DataLoaderBuilder):
            def _build_default_collator_list(self):
                return [MyCustomCollator()]
        
        builder = CustomBuilder(dataset, micro_batch_size=2, gradient_accumulation_steps=4)
        dataloader = builder.build()
    """
    
    def __init__(
        self,
        dataset: "Dataset",
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        use_default_collators: bool = True,
        num_workers: int = 1,
        drop_last: bool = True,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 2,
        seed: int = 0,
        pad_to_multiple_of: int = 128,
    ):
        """
        Initialize the DataLoaderBuilder.

        Args:
            dataset: The dataset to load data from
            micro_batch_size: Size of each micro-batch for gradient accumulation
            gradient_accumulation_steps: Number of micro-batches to accumulate
            use_default_collators: If True, initializes with default collator pipeline.
                                   If False, starts with empty pipeline (use add_collator to customize)
            num_workers: Number of worker processes for data loading
            drop_last: Whether to drop the last incomplete batch
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker. Must be None when num_workers=0.
            seed: Random seed for sampler
            pad_to_multiple_of: Pad packed sequences to a multiple of this value
        """
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        # Set prefetch_factor to None when num_workers=0 to avoid PyTorch error
        self.prefetch_factor = None if num_workers == 0 else prefetch_factor
        self.seed = seed
        self.pad_to_multiple_of = pad_to_multiple_of
        self.parallel_state = get_parallel_state()
        
        # Initialize collator pipeline
        if use_default_collators:
            self._collator_list: List[DataCollator] = self._build_default_collator_list()
        else:
            self._collator_list: List[DataCollator] = []
        
        # Final collator and sampler will be set when build() is called
        self.collate_fn = None
        self.sampler = None
    
    def _build_default_collator_list(self) -> List[DataCollator]:
        """Build the default collator pipeline."""
        from .collators import ToTensorCollator

        collators = [
            ToTensorCollator(),                                 # 1. Convert to tensors
            FlattenCollator(),                                  # 2. Flatten list of lists to flat list
            PackingConcatCollator(pad_to_multiple_of=self.pad_to_multiple_of)  # 3. Concatenate sequences for packing
        ]

        if self.parallel_state.sp_enabled:
            collators.append(TextSequenceShardCollator())  # 4. Shard sequences (if SP enabled)
        
        return collators
    
    def add_collator(self, collator: DataCollator, position: str = "end") -> "DataLoaderBuilder":
        """
        Add a collator to the pipeline.
        
        Args:
            collator: The collator to add
            position: Where to add ('start' or 'end')
        
        Returns:
            Self for method chaining
        """
        if position == "start":
            self._collator_list.insert(0, collator)
        elif position == "end":
            self._collator_list.append(collator)
        else:
            raise ValueError(f"Invalid position: {position}. Must be 'start' or 'end'")
        return self
    
    def insert_collator(self, collator: DataCollator, index: int) -> "DataLoaderBuilder":
        """
        Insert a collator at a specific position in the pipeline.
        
        Args:
            collator: The collator to insert
            index: The position to insert at
        
        Returns:
            Self for method chaining
        """
        self._collator_list.insert(index, collator)
        return self
    
    def remove_collator(self, index: int) -> "DataLoaderBuilder":
        """
        Remove a collator from the pipeline.
        
        Args:
            index: The index of the collator to remove
        
        Returns:
            Self for method chaining
        """
        self._collator_list.pop(index)
        return self
    
    def get_collator_pipeline(self) -> List[DataCollator]:
        """
        Get the current list of collators in the pipeline.
        
        Returns:
            List of collators
        """
        return self._collator_list.copy()
    
    def print_pipeline(self) -> None:
        """
        Print the current collator pipeline in a readable format.
        
        This shows the order and types of all collators that will be applied
        to each micro-batch during data loading.
        """
        logger.info_rank0("=" * 60)
        logger.info_rank0("Collator Pipeline:")
        logger.info_rank0("=" * 60)
        
        if not self._collator_list:
            logger.info_rank0("  (empty pipeline)")
        else:
            for i, collator in enumerate(self._collator_list, 1):
                collator_name = collator.__class__.__name__
                logger.info_rank0(f"  {i}. {collator_name}")
                
                # If it's a CollatePipeline, show its internal collators
                if isinstance(collator, CollatePipeline):
                    for j, sub_collator in enumerate(collator.data_collators, 1):
                        sub_collator_name = sub_collator.__class__.__name__
                        logger.info_rank0(f"     {i}.{j} {sub_collator_name}")
        
        logger.info_rank0("-" * 60)
        logger.info_rank0(f"Micro-batch splitting:")
        logger.info_rank0(f"  micro_batch_size: {self.micro_batch_size}")
        logger.info_rank0(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info_rank0(f"  dataloader_batch_size: {self.micro_batch_size * self.gradient_accumulation_steps}")
        logger.info_rank0("=" * 60)
    
    def build_collator(self) -> MicroBatchCollator:
        """
        Build the final collator that wraps the pipeline with micro-batch splitting.
        
        Override this method to customize the entire collator construction logic.
        
        Returns:
            A MicroBatchCollator that wraps the collator pipeline
        """
        # Create the internal pipeline from the collator list
        if len(self._collator_list) == 0:
            raise ValueError("Collator pipeline is empty. Add at least one collator.")
        elif len(self._collator_list) == 1:
            internal_collator = self._collator_list[0]
        else:
            internal_collator = CollatePipeline(self._collator_list)
        
        # Wrap with MicroBatchCollator for gradient accumulation
        return MicroBatchCollator(
            micro_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            internal_collator=internal_collator,
        )
    
    def build_sampler(self) -> StatefulDistributedSampler:
        """
        Build the distributed sampler.
        
        Override this method to customize the sampler (e.g., change shuffle behavior).
        
        Returns:
            A StatefulDistributedSampler instance
        """
        return StatefulDistributedSampler(
            self.dataset,
            num_replicas=self.parallel_state.dp_size,
            rank=self.parallel_state.dp_rank,
            shuffle=True,
            seed=self.seed,
        )
    
    def build(self, verbose: bool = True) -> "DistributedDataloader":
        """
        Build and return the final dataloader.
        
        Args:
            verbose: If True, prints the collator pipeline configuration
        
        Returns:
            A DistributedDataloader configured with all components
        """
        if verbose:
            self.print_pipeline()
        
        dataloader_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        
        logger.info_rank0(
            f"Building DataLoader with dp_size={self.parallel_state.dp_size}, "
            f"sp_size={self.parallel_state.sp_size}"
        )

        # Validate dataset structure - each item can be either:
        # 1. A dict (old structure for backward compatibility)
        # 2. A list of dicts (new structure for packed sequences)
        if len(self.dataset) > 0:
            first_item = self.dataset[0]
            if isinstance(first_item, list):
                # New structure: list of dicts
                assert len(first_item) > 0, "Dataset item is an empty list"
                assert isinstance(first_item[0], dict), (
                    f"Dataset items must be lists of dictionaries, but got list of {type(first_item[0]).__name__}. "
                    f"Each element in the list should be a dict with keys like 'input_ids', 'labels', etc."
                )
            elif isinstance(first_item, dict):
                # Old structure: single dict (backward compatibility)
                pass
            else:
                raise AssertionError(
                    f"Dataset items must be either dict or list of dicts, but got {type(first_item).__name__}. "
                    f"Each dataset item should be a dict (e.g., {{'input_ids': ..., 'labels': ...}}) "
                    f"or a list of dicts (e.g., [{{'input_ids': ..., 'labels': ...}}, ...])."
                )

        # Build and store the final collator and sampler
        self.collate_fn = self.build_collator()
        self.sampler = self.build_sampler()

        dataloader = DistributedDataloader(
            self.dataset,
            batch_size=dataloader_batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
        )
        
        return dataloader
