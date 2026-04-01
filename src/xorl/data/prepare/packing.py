"""
Multipack Batch Sampler - An efficient batch sampler for packing variable-length sequences
into fixed-capacity batches to optimize memory usage and training throughput.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numba
import numpy as np
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ...arguments import Arguments
from ...utils import logging
from .hash import generate_dataset_hash_from_config, generate_packing_hash
from .shared import get_prepared_dataset_path


LOG = logging.get_logger(__name__)


@numba.njit
def ffd_check(sequence_lengths: np.ndarray, bin_capacity: int, num_bins: int) -> bool:
    """First-fit-decreasing bin packing algorithm check.

    Checks if sequences with the given lengths could fit in the specified number of
    bins.

    Args:
        sequence_lengths: Array of sequence lengths.
        bin_capacity: Maximum capacity of each bin.
        num_bins: Number of bins available.

    Returns:
        `True` if all sequences can be packed, `False` otherwise.
    """
    # Sort sequence lengths in descending order for optimal packing
    sequence_lengths = np.sort(sequence_lengths)[::-1]
    # Initialize all bins with full capacity
    bins = np.full((num_bins,), bin_capacity, dtype=sequence_lengths.dtype)

    # Try to place each sequence in the first bin it fits
    for size in sequence_lengths:
        not_found = True
        for idx in range(num_bins):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        # If no bin could fit this sequence, packing failed
        if not_found:
            return False

    return True


@numba.njit
def pack_group(
    sequence_lengths: np.ndarray,
    group_offset: int,
    bin_capacity: int,
    max_bins: int,
    bin_size: int,
    safe_mode: bool = True,
) -> list[list[int]]:
    """Pack a group of sequences into bins using First-Fit Decreasing algorithm.

    Args:
        sequence_lengths: Array of sequence lengths.
        group_offset: Offset to apply to indices when returning results.
        bin_capacity: Maximum capacity of each bin.
        max_bins: Maximum number of bins to use.
        bin_size: Maximum number of sequences per bin.
        safe_mode: If True, use a more conservative packing approach.

    Returns:
        List of bins, where each bin contains indices of sequences assigned to it.
    """
    bins_remaining_space: list = []  # Tracks remaining capacity in each bin
    bins_assigned_sequences: list = []  # Tracks sequence indices assigned to each bin

    for seq_id, size in enumerate(sequence_lengths):
        global_idx = seq_id + group_offset

        # Try to place sequence in existing bins
        add_new_bin = True
        for bin_idx, _ in enumerate(bins_remaining_space):
            if bins_remaining_space[bin_idx] >= size and len(bins_assigned_sequences[bin_idx]) < bin_size:
                bins_remaining_space[bin_idx] -= size
                bins_assigned_sequences[bin_idx].append(global_idx)
                add_new_bin = False
                break

        # Create a new bin if needed and if we haven't reached the limit
        if add_new_bin:
            if len(bins_remaining_space) >= max_bins and safe_mode:
                # In safe mode, skip items that would exceed max_bins
                continue
            bins_remaining_space.append(bin_capacity - size)
            bins_assigned_sequences.append([global_idx])

            # Safety check to avoid infinite bins
            if len(bins_remaining_space) > len(sequence_lengths):
                break

    return bins_assigned_sequences


def _process_group(
    args: tuple[np.ndarray, int, int, int, int, bool],
) -> list[list[int]]:
    """Standalone function for multiprocessing."""
    group_lengths, start_idx, bin_capacity, max_bins, bin_size, safe_mode = args
    return pack_group(group_lengths, start_idx, bin_capacity, max_bins, bin_size, safe_mode)


def pack_parallel(
    sequence_lengths: np.ndarray,
    bin_capacity: int,
    group_size: int,
    bin_size: int,
    num_processes: int | None = None,
    safe_mode: bool = True,
    mp_start_method: str | None = "fork",
) -> list[list[int]]:
    """Pack sequences into bins using parallel processing.

    Args:
        sequence_lengths: Array of sequence lengths.
        bin_capacity: Maximum capacity of each bin as total number of tokens.
        group_size: Number of sequences to process in each group.
        bin_size: Maximum number of bins to use.
        num_processes: Number of parallel processes to use.
        safe_mode: If True, use a more conservative packing approach.
        mp_start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver').
                         'spawn' is often safer with Numba/PyTorch.
                         Set to None to use system default.
    Returns:
        List of bins, where each bin contains indices of sequences assigned to it.
    """
    num_items = len(sequence_lengths)
    if num_processes is None:
        num_processes = max(1, min(num_items // group_size, cpu_count(), 16))

    # Create tasks for parallel processing
    tasks = []
    for i in range(0, num_items, group_size):
        group_lengths = sequence_lengths[i : i + group_size]
        max_bins = len(group_lengths)  # Allow as many bins as items in the group
        tasks.append((group_lengths, i, bin_capacity, max_bins, bin_size, safe_mode))

    # Process groups in parallel
    all_bins = []

    mp_ctx = None
    if mp_start_method:
        try:
            mp_ctx = get_context(mp_start_method)
        except ValueError:
            LOG.warning(
                f"Failed to get multiprocessing context '{mp_start_method}'. "
                f"Falling back to default. Available: {get_context().get_all_start_methods()}"
            )
            mp_ctx = None  # Fallback to default context if specified one is not available

    if num_processes == 1:
        LOG.debug("Using single process for pack_parallel, running sequentially.")
        for task_args in tasks:
            group_bins = _process_group(task_args)
            all_bins.extend(group_bins)
    else:
        # Use ProcessPoolExecutor only if num_processes > 1
        # Pass mp_context if available
        with ProcessPoolExecutor(max_workers=num_processes, mp_context=mp_ctx) as executor:
            for group_bins in executor.map(_process_group, tasks):
                all_bins.extend(group_bins)

    return all_bins


@numba.njit
def allocate_sequentially(
    sequence_lengths: np.ndarray, rank: int, bin_capacity: int, num_ranks: int
) -> tuple[list[list[int]], int, int]:
    """Sequential allocator that preserves example order.

    Args:
        sequence_lengths: The lengths of all examples.
        rank: The current rank (for distributed training).
        bin_capacity: The capacity of each bin (maximum sequence length).
        num_ranks: Number of ranks (processes / GPUs).

    Returns:
        rank_batches: List of batches for the current rank.
        total_tokens_used: Number of actual example tokens.
        total_token_slots: Maximum theoretical number of example tokens (number of bins
            * bin capacity).
    """
    result = []
    total_used = 0

    # First, do sequential packing into bins
    all_bins = []
    current_bin = [0 for i in range(0)]  # numba hint
    remaining_capacity = bin_capacity

    for idx, size in enumerate(sequence_lengths):
        if size <= remaining_capacity:
            # Example fits in current bin
            current_bin.append(idx)
            remaining_capacity -= size
            total_used += size
        else:
            # Example doesn't fit, start a new bin
            if current_bin:  # Add non-empty bin to all_bins
                all_bins.append(current_bin)
            current_bin = [idx]
            remaining_capacity = bin_capacity - size
            total_used += size

    # Add the last bin if not empty
    if current_bin:
        all_bins.append(current_bin)

    # Assign bins to ranks - each rank gets every n-th bin
    for bin_idx in range(rank, len(all_bins), num_ranks):
        result.append(all_bins[bin_idx])

    return result, total_used, len(all_bins) * bin_capacity


def add_position_ids(sample):
    """
    Handle both single-example and batched data.
    - single example: sample['input_ids'] is a list[int]
    - batched data: sample['input_ids'] is a list[list[int]]
    """
    # Return sample unchanged if "input_ids" is not present, or is empty
    if "input_ids" not in sample or not sample["input_ids"]:
        return sample

    input_ids = sample["input_ids"]

    # If first element is an int, it’s a single example
    # If first element is a list, it’s a batch
    if isinstance(input_ids[0], int):
        # ---- SINGLE EXAMPLE ----
        seq_len = len(input_ids)
        # Position IDs for a single example
        # As a list
        sample["position_ids"] = list(range(seq_len))
        sample["length"] = seq_len

    else:
        # ---- BATCHED EXAMPLES ----
        # input_ids is a list of lists
        position_ids_batch = []
        lengths_batch = []
        for seq in input_ids:
            seq_len = len(seq)
            position_ids_batch.append(list(range(seq_len)))
            lengths_batch.append(seq_len)

        # Now store them back
        sample["position_ids"] = position_ids_batch
        sample["length"] = lengths_batch

    return sample


def drop_no_trainable_tokens(sample: Dict[str, Any]):
    """
    Drop samples if all labels are -100 (i.e., zero trainable tokens).
    Works for both single-example or batched input.
    """
    labels = sample["labels"]
    if not labels:
        return True

    # Check if single example or batch
    # If first element is an int, we assume a single example
    # If it's a list, we assume we're dealing with a batch
    if isinstance(labels[0], int):
        # Single example: return a single bool
        return bool(np.any(np.array(labels) != -100))

    # Batched: 'labels' is a list of lists
    # Return a list of booleans, one per sub-list
    results = [bool(np.any(np.array(row_labels) != -100)) for row_labels in labels]
    return results


# drop samples with no trainable tokens
def filter_dataset_with_logging(dataset: HFDataset, filter_func: Callable, dataset_name: str, num_proc: int):
    """Filter dataset and log dropped samples."""
    try:
        prior_len = len(dataset)
    except TypeError:
        # handle iterable datasets case
        prior_len = None

    filtered_dataset = dataset.filter(
        filter_func,
        num_proc=num_proc,
        desc="Drop Samples with Zero Trainable Tokens",
    )

    if prior_len:
        dropped = prior_len - len(filtered_dataset)
        if dropped:
            LOG.warning(f"Dropped {dropped} samples with no trainable tokens from {dataset_name} dataset")

    return filtered_dataset


def process_datasets_for_packing(args: Arguments, train_dataset: HFDataset, eval_dataset: HFDataset | None = None):
    # drop samples with no trainable tokens
    train_dataset = filter_dataset_with_logging(
        train_dataset, drop_no_trainable_tokens, "train", args.data.dataset_num_proc
    )
    # drop samples with no trainable tokens from eval_dataset if it exists
    if eval_dataset:
        eval_dataset = filter_dataset_with_logging(
            eval_dataset, drop_no_trainable_tokens, "eval", args.data.dataset_num_proc
        )

    # add position_ids to train_dataset
    train_dataset = train_dataset.map(
        add_position_ids,
        num_proc=args.data.dataset_num_proc,
        desc="Add position_id column (Sample Packing)",
    )

    # add position_ids to eval_dataset
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            add_position_ids,
            num_proc=args.data.dataset_num_proc,
            desc="Add position_id column (Sample Packing)",
        )

    return train_dataset, eval_dataset


class PackingDataset(Dataset):
    def __init__(
        self, args: Arguments, tokenizer: PreTrainedTokenizer, dataset: HFDataset, split: str = "train"
    ) -> None:
        self.dataset: HFDataset = dataset
        self.args: Arguments = args

        datasets_configs = args.data.datasets if split == "train" else args.data.test_datasets
        self.dataset_hash: str = generate_dataset_hash_from_config(args, datasets_configs, tokenizer.name_or_path)
        self.prepared_dataset_path: str | None = get_prepared_dataset_path(args, self.dataset_hash)

        # Get sequence lengths from the dataset
        assert "length" in self.dataset.features, "Length column not found in dataset"
        self.sequence_lengths: np.ndarray = np.array(dataset["length"])

        # Determine bin capacity from args
        self.bin_capacity: int = args.data.sample_packing_sequence_len

        # Validate: all samples must fit within bin_capacity.
        # Truncation is a data processing concern and should be handled upstream.
        max_len = int(self.sequence_lengths.max())
        if max_len > self.bin_capacity:
            num_oversized = int(np.sum(self.sequence_lengths > self.bin_capacity))
            raise ValueError(
                f"{num_oversized}/{len(self.sequence_lengths)} samples exceed "
                f"sample_packing_sequence_len ({self.bin_capacity}), max is {max_len}. "
                f"Truncate your data before packing."
            )

        # Per-document alignment for ring attention zigzag:
        # each doc must be divisible by 2*ringattn_size*ulysses_size = 2*cp_size.
        ringattn_size = args.train.ringattn_parallel_size
        ulysses_size = args.train.ulysses_parallel_size
        cp_size = ringattn_size * ulysses_size
        self.doc_align: int = 2 * cp_size if ringattn_size > 1 else 1

        # Try to load cached bins, otherwise compute them
        self.bins: List[List[int]] = self._load_or_compute_bins()

        LOG.info(f"PackingDataset created with {len(self.bins)} bins from {len(dataset)} samples")

    def _get_bins_cache_path(self) -> Path | None:
        """Get the path where bins should be cached as HF dataset."""
        if self.prepared_dataset_path is None:
            return None

        # Include packing args hash in the cache path
        packing_hash = generate_packing_hash(
            self.args.data.sample_packing_method,
            self.args.data.sample_packing_sequence_len,
            self.args.data.sample_packing_group_size,
            self.args.data.sample_packing_mp_start_method,
            doc_align=self.doc_align,
        )
        return Path(self.prepared_dataset_path) / f"packing_bins_{packing_hash}"

    def _load_cached_bins(self) -> Optional[List[List[int]]]:
        """Try to load cached bins from HF dataset."""
        cache_path = self._get_bins_cache_path()
        if cache_path is None or not cache_path.exists():
            return None

        try:
            # Load the HF dataset
            bins_dataset = HFDataset.load_from_disk(str(cache_path))

            # The cache hash is already validated by the cache_path name itself
            # (cache_path includes dataset_hash which is computed from all relevant parameters)
            # So if the file exists and loads successfully, it's valid

            # Extract bins from the dataset
            bins = bins_dataset["bins"]
            LOG.info(f"Loaded {len(bins)} cached bins from {cache_path}")
            return bins

        except Exception as e:
            LOG.warning(f"Failed to load cached bins: {e}")
            return None

    def _save_bins_cache(self, bins: List[List[int]]) -> None:
        """Save computed bins as HF dataset for future use."""
        cache_path = self._get_bins_cache_path()
        if cache_path is None:
            return

        # Ensure the directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a HF dataset from the bins
        bins_dataset = HFDataset.from_dict({"bins": bins})

        # Add metadata to the dataset info
        bins_dataset.info.description = (
            f"Packing bins for dataset fingerprint: {self.dataset_hash}"
            f" | packing method: {self.args.data.sample_packing_method}"
            f" | bin capacity: {self.bin_capacity}"
        )

        # Save the dataset
        num_workers = max(self.args.data.dataset_num_proc // 8, 1)
        bins_dataset.save_to_disk(str(cache_path), num_proc=num_workers)

        LOG.info(f"Saved bins cache to {cache_path}")

    def _load_or_compute_bins(self) -> List[List[int]]:
        """Load cached bins or compute new ones with distributed coordination."""
        # Get rank from environment variable (set by torchrun/distributed launcher)
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Try to load from cache first
        cached_bins = self._load_cached_bins()
        if cached_bins is not None:
            return cached_bins

        cache_path = self._get_bins_cache_path()

        # Use rank 0 to compute and save, others wait and load
        if rank == 0:
            # Rank 0 computes new bins
            LOG.info("Computing new bins...")
            bins = self._compute_bins()

            # Save to cache for future use
            if cache_path is not None:
                self._save_bins_cache(bins)
        else:
            # Other ranks wait for rank 0 to finish
            if cache_path is not None:
                LOG.info(f"Rank {rank} waiting for rank 0 to compute bins...")
                # Wait for cache file to be created
                max_wait_time = 3600  # 1 hour max wait
                wait_interval = 5  # Check every 5 seconds
                waited = 0

                while not cache_path.exists() and waited < max_wait_time:
                    time.sleep(wait_interval)
                    waited += wait_interval

                if cache_path.exists():
                    # Wait a bit more to ensure rank 0 finished writing
                    # (HF datasets creates the directory first, then writes files)
                    time.sleep(2)

                    # Try to load with retries
                    bins = None
                    max_retries = 30
                    for attempt in range(max_retries):
                        bins = self._load_cached_bins()
                        if bins is not None:
                            break
                        if attempt < max_retries - 1:
                            LOG.info(f"Rank {rank} retry {attempt + 1}/{max_retries} loading bins...")
                            time.sleep(5)

                    if bins is None:
                        raise RuntimeError(
                            f"Rank {rank} failed to load bins cached by rank 0 after {max_retries} retries. "
                            f"Cache path: {cache_path}"
                        )
                    LOG.info(f"Rank {rank} loaded {len(bins)} bins from cache")
                else:
                    raise RuntimeError(f"Rank {rank} timed out waiting for rank 0 to compute bins")
            else:
                # No cache path, fall back to computing on all ranks
                LOG.warning(f"No cache path available, rank {rank} computing bins independently")
                bins = self._compute_bins()

        return bins

    def _compute_bins(self) -> List[List[int]]:
        """Compute bins based on the packing method specified in args."""
        if self.args.data.sample_packing_method == "sequential":
            return self._compute_sequential_bins()
        else:  # default to multipack
            return self._compute_multipack_bins()

    def _compute_sequential_bins(self) -> List[List[int]]:
        """Compute bins using sequential packing."""
        # Use the sequential allocator from the existing code
        rank: int = 0  # For single-process case
        num_ranks: int = 1  # For single-process case

        bins, _, _ = allocate_sequentially(self.sequence_lengths, rank, self.bin_capacity, num_ranks)
        return bins

    def _compute_multipack_bins(self) -> List[List[int]]:
        """Compute bins using multipack algorithm."""
        group_size: int = self.args.data.sample_packing_group_size or 100000
        bin_size: int = self.bin_capacity  # Maximum sequences per bin
        mp_start_method: str = self.args.data.sample_packing_mp_start_method or "fork"

        # Use the parallel packing function
        bins: List[List[int]] = pack_parallel(
            sequence_lengths=self.sequence_lengths,
            bin_capacity=self.bin_capacity,
            group_size=group_size,
            bin_size=bin_size,
            num_processes=1,  # Use single process for simplicity
            safe_mode=True,
            mp_start_method=mp_start_method,
        )
        return bins

    def __len__(self) -> int:
        """Return the number of bins (packed samples)."""
        return len(self.bins)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a packed sample containing multiple sequences concatenated together."""
        if index >= len(self.bins):
            raise IndexError(f"Index {index} out of range for {len(self.bins)} bins")

        bin_indices: List[int] = self.bins[index]

        # Get the individual samples for this bin
        samples: List[Dict[str, Any]] = [self.dataset[idx] for idx in bin_indices]

        return samples
