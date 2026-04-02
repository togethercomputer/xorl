"""Data handling specific to SFT."""

from typing import Literal

import torch.distributed as dist
from datasets import (
    Dataset as HFDataset,
)
from datasets import (
    DatasetDict as HFDatasetDict,
)
from datasets import (
    IterableDataset as HFIterableDataset,
)
from datasets import (
    IterableDatasetDict as HFIterableDatasetDict,
)
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...arguments import Arguments, DatasetConfig
from ...data.prepare.utils import retry_on_request_exceptions
from ...utils import logging
from .hash import generate_dataset_hash_from_config
from .packing import PackingDataset, process_datasets_for_packing
from .shared import (
    create_train_validation_split,
    datasets_with_name_generator,
    load_dataset_with_config,
    load_preprocessed_dataset,
    merge_datasets,
    save_preprocessed_dataset,
    try_load_from_hub,
)


logger = logging.get_logger(__name__)


def _create_dummy_dataset(seq_len: int, num_samples: int = 4096, seed: int = 42, vocab_size: int = 151936) -> HFDataset:
    """Create a dummy tokenized dataset for benchmarking.

    All ranks call this independently with the same fixed seed, producing
    identical data without any disk I/O or rank synchronization.  Builds
    the Arrow table directly from numpy and includes ``position_ids`` /
    ``length`` so we can skip the slow ``process_datasets_for_packing``
    step entirely.

    Token format: each sample is ``[1, 2, 3, ..., length-1, 0]`` where
    0 is the EOD marker. All ``num_samples`` examples are intentionally
    identical so local benchmarks can show clean loss convergence.
    """
    import numpy as np
    import pyarrow as pa

    EOD = 0
    VOCAB_SIZE = vocab_size
    MAX_SAMPLE_LEN = max(seq_len, 1)
    # Lengths are k*16+1 so that after ShiftTokensCollator drops 1 token,
    # effective length k*16 is divisible by 2*ringattn_size for zigzag ring
    # attention. This keeps the repeated sample valid for the benchmark path.
    LENGTH_ALIGN = 16
    sample_len = max(1, ((MAX_SAMPLE_LEN - 1) // LENGTH_ALIGN) * LENGTH_ALIGN + 1)
    lengths = np.full((num_samples,), sample_len, dtype=np.int64)

    sample_tokens = (np.arange(sample_len, dtype=np.int64) + 1) % VOCAB_SIZE
    sample_tokens[-1] = EOD
    flat_tokens = np.tile(sample_tokens.astype(np.int32), num_samples)
    offsets = np.arange(0, sample_len * (num_samples + 1), sample_len, dtype=np.int64)
    tokens = pa.ListArray.from_arrays(offsets, flat_tokens)

    pos_flat = np.tile(np.arange(sample_len, dtype=np.int32), num_samples)
    position_ids = pa.ListArray.from_arrays(offsets, pos_flat)

    pa_lengths = pa.array(lengths.tolist(), type=pa.int64())

    table = pa.table(
        {
            "input_ids": tokens,
            "labels": tokens,
            "position_ids": position_ids,
            "length": pa_lengths,
        }
    )
    return HFDataset(table)


@retry_on_request_exceptions(max_retries=3, delay=5)
def prepare_datasets(
    args: Arguments,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None = None,
) -> tuple[Dataset, Dataset]:
    """Prepare training and evaluation datasets based on configuration. Note that this dataset is Torch Dataset, not Hugging Face Dataset."""

    # Fast path: dummy dataset for benchmarking
    datasets_configs = args.data.datasets
    if datasets_configs and all(dc.path == "dummy" for dc in datasets_configs):
        seq_len = datasets_configs[0].max_seq_len or args.data.sample_packing_sequence_len
        logger.info_rank0(f"Creating dummy dataset: {4096} samples x {seq_len} tokens")
        dataset = _create_dummy_dataset(seq_len=seq_len, seed=args.train.seed, vocab_size=len(tokenizer))

        if args.data.sample_packing_method and args.data.sample_packing_method != "none":
            train_dataset = PackingDataset(args, tokenizer, dataset, split="train")
        else:
            train_dataset = dataset

        return train_dataset, None

    def _load_datasets():
        # Load training dataset
        train_dataset, eval_dataset = _load_and_prepare_datasets(tokenizer, args, split="train", processor=processor)

        # Override with test dataset if available
        if args.data.test_datasets:
            _, eval_dataset = _load_and_prepare_datasets(tokenizer, args, split="test", processor=processor)

        # Apply sample packing if configured
        if args.data.sample_packing_method and args.data.sample_packing_method != "none":
            train_dataset = PackingDataset(args, tokenizer, train_dataset, split="train")
            if eval_dataset:
                eval_dataset = PackingDataset(args, tokenizer, eval_dataset, split="test")
            else:
                eval_dataset = None

        return train_dataset, eval_dataset

    # Check if we're in a distributed setting
    is_distributed = dist.is_available() and dist.is_initialized()
    is_rank_zero = not is_distributed or dist.get_rank() == 0

    if is_distributed and is_rank_zero:
        logger.info("Rank 0: Loading and processing datasets...")

    # Only rank 0 loads and processes datasets
    if is_rank_zero:
        train_dataset, eval_dataset = _load_datasets()

    # Synchronize all ranks - rank 0 has finished processing
    if is_distributed:
        dist.barrier()
        logger.info(f"Rank {dist.get_rank()}: Dataset processing synchronized")

    # Non-rank-0 processes load from the prepared path
    if is_distributed and not is_rank_zero:
        logger.info(f"Rank {dist.get_rank()}: Loading prepared datasets from disk...")
        train_dataset, eval_dataset = _load_datasets()

    return train_dataset, eval_dataset


def _load_tokenized_prepared_datasets(
    tokenizer: PreTrainedTokenizer,
    args: Arguments,
    split: Literal["train", "test"] = "train",
    processor: ProcessorMixin | None = None,
) -> HFDataset | HFDatasetDict:
    """Load or create tokenized and prepared datasets for training or testing.

    Args:
        tokenizer: Tokenizer for processing text.
        cfg: Configuration object.
        split: Dataset split to load ('train' or 'test').
        processor: Optional processor for multimodal datasets.

    Returns:
        Dataset or DatasetDict.
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    is_rank_zero = not is_distributed or dist.get_rank() == 0

    # Select correct dataset configuration based on split
    datasets_configs = args.data.datasets if split == "train" else args.data.test_datasets

    # Generate dataset hash for caching
    dataset_hash = generate_dataset_hash_from_config(args, datasets_configs, tokenizer.name_or_path)

    # Try loading from hub if push_dataset_to_hub is configured
    dataset = None
    if args.data.push_dataset_to_hub:
        dataset = try_load_from_hub(args, dataset_hash, split)

    # If not found on hub, try loading from disk
    if dataset is None:
        dataset = load_preprocessed_dataset(args, dataset_hash)

    # If not found on disk or skipping prepared dataset, load and process raw datasets
    # Only rank 0 should do the raw dataset loading and processing
    if dataset is None and is_rank_zero:
        logger.info_rank0(f"Rank 0: No prepared dataset found, loading raw datasets for {split} split...")
        dataset = _load_raw_datasets(
            args,
            datasets_configs,
            tokenizer,
            split,
            processor,
        )
    elif dataset is None and not is_rank_zero:
        # Non-rank-0 processes should not reach here if synchronization is correct
        # They should wait for rank 0 to finish and then load the prepared dataset
        raise RuntimeError(
            f"Rank {dist.get_rank()}: Attempted to load raw datasets on non-rank-0 process. "
            "This indicates a synchronization issue."
        )

    return dataset


def _load_raw_datasets(
    args: Arguments,
    datasets_configs: list,
    tokenizer: PreTrainedTokenizer,
    split: str,
    processor: ProcessorMixin | None = None,
) -> HFDataset:
    """Load, process, merge, and save raw datasets. Only rank 0 should call this."""
    is_distributed = dist.is_available() and dist.is_initialized()
    is_rank_zero = not is_distributed or dist.get_rank() == 0

    # This function should only be called by rank 0
    if is_distributed and not is_rank_zero:
        logger.warning(f"Rank {dist.get_rank()}: _load_raw_datasets should only be called by rank 0!")

    logger.info_rank0("Loading raw datasets...")
    if not args.data.is_preprocess and not args.data.skip_prepare_dataset:
        logger.warning_rank0(
            "Processing datasets during training can lead to VRAM instability. Please "
            "pre-process your dataset using `axolotl preprocess path/to/config.yml`."
        )

    # Load and process individual datasets
    datasets = []
    for dataset_config in datasets_with_name_generator(datasets_configs):
        dataset = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            args=args,
            tokenizer=tokenizer,
            split=split,
            seed=args.train.seed,
            processor=processor,
        )
        datasets.append(dataset)

    # Merge datasets
    dataset = merge_datasets(datasets, args)

    # sample packing
    dataset, _ = process_datasets_for_packing(args, dataset, None)

    # Save the prepared dataset
    if not args.data.skip_prepare_dataset:
        dataset_hash = generate_dataset_hash_from_config(args, datasets_configs, tokenizer.name_or_path)
        save_preprocessed_dataset(args, dataset, dataset_hash, split)

    return dataset


def _load_and_process_single_dataset(
    dataset_config: DatasetConfig,
    args: Arguments,
    tokenizer: PreTrainedTokenizer,
    split: str,
    seed: int,
    processor: ProcessorMixin | None = None,
) -> HFDataset | HFIterableDataset:
    """Load and process a single dataset based on the passed config."""
    # Load the dataset
    dataset = load_dataset_with_config(
        dataset_config,
        args.data.hf_use_auth_token,
        streaming=False,
        num_proc=args.data.dataset_num_proc,
    )

    # Parse dataset type
    assert dataset_config.type == "tokenized", "Only tokenized datasets are supported"

    # Select the appropriate split
    if isinstance(dataset, (HFDatasetDict, HFIterableDatasetDict)):
        if dataset_config.split and dataset_config.split in dataset:
            dataset = dataset[dataset_config.split]
        elif split in dataset:
            dataset = dataset[split]
        else:
            raise ValueError(
                f"no {split} split found for dataset {dataset_config.path}, you may specify a split with 'split: ...'"
            )

    # Apply sharding if configured
    if dataset_config.shards:
        shards_idx = dataset_config.shards_idx or 0
        dataset = dataset.shuffle(seed=seed).shard(num_shards=dataset_config.shards, index=shards_idx)

    # Select columns if configured
    if args.data.select_columns:
        dataset = dataset.select_columns(args.data.select_columns)

    if not ("input_ids" in dataset.features and "labels" in dataset.features):
        raise ValueError("Dataset is not tokenized. Please use a tokenized dataset.")

    # Filter out samples that exceed max_seq_len
    if dataset_config.max_seq_len is not None:
        max_len = dataset_config.max_seq_len
        prior_len = len(dataset)
        dataset = dataset.filter(
            lambda example: len(example["input_ids"]) <= max_len,
            num_proc=args.data.dataset_num_proc,
            desc=f"Filtering samples > {max_len} tokens",
        )
        dropped = prior_len - len(dataset)
        if dropped > 0:
            logger.info_rank0(
                f"Filtered {dropped}/{prior_len} samples exceeding max_seq_len={max_len} "
                f"from {dataset_config.name or dataset_config.path}"
            )

    # Add activations_path field if configured (for distillation training)
    if dataset_config.activations_path:
        logger.info_rank0(f"Adding activations_path '{dataset_config.activations_path}' to dataset")

        def add_activations_path(example):
            example["activations_path"] = dataset_config.activations_path
            return example

        dataset = dataset.map(
            add_activations_path,
            desc=f"Adding activations_path to {dataset_config.name or 'dataset'}",
            num_proc=args.data.dataset_num_proc,
        )

    return dataset


def _handle_train_dataset_split(dataset: HFDataset, args: Arguments) -> tuple[HFDataset, HFDataset | None]:
    """Handle processing for train split, including validation set creation."""
    val_set_size = (
        int(args.data.val_set_size)
        if args.data.val_set_size is not None and args.data.val_set_size > 1
        else float(args.data.val_set_size)
        if args.data.val_set_size is not None
        else 0
    )

    if val_set_size:
        # Create train/validation split
        train_dataset, eval_dataset = create_train_validation_split(dataset, args, val_set_size)
        return train_dataset, eval_dataset
    else:
        return dataset, None


def _apply_dataset_sharding(dataset: HFDataset, args: Arguments) -> HFDataset:
    """Apply dataset sharding if configured.

    Args:
        dataset: Dataset to shard.
        args: Arguments object containing shard settings.

    Returns:
        Sharded dataset or original dataset if no sharding configured.
    """
    if args.data.dataset_shard_num and args.data.dataset_shard_idx is not None:
        logger.info_rank0(f"Using index #{args.data.dataset_shard_idx} of {args.data.dataset_shard_num} shards")
        dataset = dataset.shard(
            num_shards=args.data.dataset_shard_num,
            index=args.data.dataset_shard_idx,
        )
    return dataset


def _load_and_prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    args: Arguments,
    split: Literal["train", "test"] = "train",
    processor: ProcessorMixin | None = None,
) -> tuple[HFDataset | None, HFDataset | None]:
    """Load and prepare datasets with optional validation split and sharding.

    Args:
        tokenizer: Tokenizer for processing text.
        cfg: Configuration object.
        split: Dataset split to load ('train' or 'test').
        processor: Optional processor for multimodal datasets.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    # Load the base dataset
    dataset = _load_tokenized_prepared_datasets(
        tokenizer,
        args,
        split=split,
        processor=processor,
    )

    # Apply dataset sharding
    dataset = _apply_dataset_sharding(dataset, args)

    # Handle train/validation split
    if split == "train":
        return _handle_train_dataset_split(dataset, args)
    else:
        return dataset, None
