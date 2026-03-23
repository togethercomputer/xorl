"""Tests for xorl.data.prepare.packing module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from datasets import Dataset as HFDataset

from xorl.data.prepare.packing import (
    ffd_check,
    pack_group,
    allocate_sequentially,
    add_position_ids,
    drop_no_trainable_tokens,
    filter_dataset_with_logging,
    process_datasets_for_packing,
    PackingDataset,
    pack_parallel,
)


pytestmark = pytest.mark.cpu


def test_ffd_check():
    """FFD feasibility check: fit, don't fit, edge cases."""
    # Sequences that fit
    assert ffd_check(np.array([10, 20, 30, 40]), 50, 3) is True
    assert ffd_check(np.array([30]), 50, 1) is True
    assert ffd_check(np.array([25, 25]), 50, 1) is True  # exact fit

    # Sequences that don't fit
    assert ffd_check(np.array([40, 40, 40, 40]), 50, 2) is False
    assert ffd_check(np.array([60]), 50, 1) is False

    # Empty
    assert ffd_check(np.array([]), 50, 1) is True


def test_pack_group():
    """Pack sequences into bins: capacity, bin_size limit, safe/non-safe mode, offset."""
    # Basic packing respects capacity
    seq = np.array([10, 20, 30, 40])
    bins = pack_group(seq, 0, 50, 10, 10, safe_mode=True)
    all_indices = [idx for b in bins for idx in b]
    assert len(all_indices) == 4
    for b in bins:
        assert sum(seq[idx] for idx in b) <= 50

    # Respects bin_size (max sequences per bin)
    bins = pack_group(np.array([5, 5, 5, 5, 5, 5]), 0, 100, 10, 3, safe_mode=True)
    for b in bins:
        assert len(b) <= 3

    # Safe mode skips when exceeding max_bins
    bins = pack_group(np.array([40, 40, 40, 40]), 0, 50, 2, 1, safe_mode=True)
    assert len(bins) <= 2

    # Non-safe mode packs all sequences
    bins = pack_group(np.array([40, 40, 40, 40]), 0, 50, 2, 1, safe_mode=False)
    assert len([idx for b in bins for idx in b]) == 4

    # Group offset applied to indices
    bins = pack_group(np.array([10, 20]), 100, 50, 10, 10, safe_mode=True)
    assert min(idx for b in bins for idx in b) >= 100


def test_allocate_sequentially():
    """Sequential allocation: distributes to ranks, no overlap, full coverage."""
    seq = np.array([10, 20, 30, 40, 50, 60])

    # Distributes batches respecting capacity
    r0_batches, tok0, slots0 = allocate_sequentially(seq, 0, 100, 2)
    assert len(r0_batches) > 0
    for batch in r0_batches:
        assert sum(seq[idx] for idx in batch) <= 100
    assert tok0 > 0 and slots0 > 0 and tok0 <= slots0

    # Different ranks get non-overlapping batches that cover all sequences
    r1_batches, _, _ = allocate_sequentially(seq, 1, 100, 2)
    r0_idx = {idx for b in r0_batches for idx in b}
    r1_idx = {idx for b in r1_batches for idx in b}
    assert len(r0_idx & r1_idx) == 0

    all_idx = set()
    for rank in range(2):
        batches, _, _ = allocate_sequentially(np.array([10, 20, 30, 40]), rank, 100, 2)
        for b in batches:
            all_idx.update(b)
    assert all_idx == {0, 1, 2, 3}


def test_add_position_ids():
    """Add position_ids: single, batched, missing/empty input_ids, preserves fields."""
    # Single sample
    result = add_position_ids({"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]})
    assert result["position_ids"] == [0, 1, 2, 3, 4]
    assert result["length"] == 5

    # Batched
    result = add_position_ids({"input_ids": [[1, 2, 3], [4, 5, 6, 7]], "labels": [[1, 2, 3], [4, 5, 6, 7]]})
    assert result["position_ids"] == [[0, 1, 2], [0, 1, 2, 3]]
    assert result["length"] == [3, 4]

    # Missing or empty input_ids -> unchanged
    s1 = {"labels": [1, 2, 3]}
    assert "position_ids" not in add_position_ids(s1)
    s2 = {"input_ids": []}
    assert "position_ids" not in add_position_ids(s2)

    # Preserves other fields
    result = add_position_ids({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]})
    assert result["attention_mask"] == [1, 1, 1]


def test_drop_no_trainable_tokens():
    """Drop samples with no trainable tokens, handle batched, raise on missing labels."""
    assert drop_no_trainable_tokens({"labels": [1, 2, -100, 3]}) == True
    assert bool(drop_no_trainable_tokens({"labels": [-100, -100, -100]})) == False

    # Batched
    result = drop_no_trainable_tokens({"labels": [[1, 2, 3], [-100, -100, -100], [1, -100, 2]]})
    assert [bool(r) for r in result] == [True, False, True]

    # Missing labels
    with pytest.raises(KeyError):
        drop_no_trainable_tokens({"input_ids": [1, 2, 3]})


def test_filter_dataset_with_logging():
    """Filter dataset and verify correct samples are kept."""
    dataset = HFDataset.from_dict({
        "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "labels": [[1, 2, 3], [-100, -100, -100], [7, 8, 9]],
    })
    filtered = filter_dataset_with_logging(dataset, lambda x: x["labels"][0] != -100, "test", num_proc=1)
    assert len(filtered) == 2
    assert filtered[0]["labels"] == [1, 2, 3]
    assert filtered[1]["labels"] == [7, 8, 9]


def test_process_datasets_for_packing():
    """Process train+eval datasets: adds position_ids/length, handles None eval."""
    args = Mock()
    args.data.dataset_num_proc = 1

    train = HFDataset.from_dict({"input_ids": [[1, 2, 3], [4, 5, 6]], "labels": [[1, 2, 3], [4, 5, 6]]})
    eval_ds = HFDataset.from_dict({"input_ids": [[7, 8, 9]], "labels": [[7, 8, 9]]})

    # With eval
    p_train, p_eval = process_datasets_for_packing(args, train, eval_ds)
    assert "position_ids" in p_train.column_names and "length" in p_train.column_names
    assert p_eval is not None and "position_ids" in p_eval.column_names

    # Without eval
    _, p_eval_none = process_datasets_for_packing(args, train, None)
    assert p_eval_none is None


def test_packing_dataset():
    """PackingDataset: init, bins, getitem, cache, missing length column."""
    args = Mock()
    args.data.sample_packing_method = "sequential"
    args.data.sample_packing_sequence_len = 100
    args.data.sample_packing_group_size = 10
    args.data.sample_packing_bin_size = 5
    args.data.sample_packing_num_processes = 1
    args.data.sample_packing_safe_mode = True
    args.data.sample_packing_mp_start_method = None
    args.data.dataset_prepared_path = None
    args.data.datasets = []
    args.data.test_datasets = []
    args.data.dataset_num_proc = 1

    dataset = HFDataset.from_dict({
        "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        "labels": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        "length": [3, 3, 3, 3],
    })
    tokenizer = Mock()
    tokenizer.name_or_path = "test-tokenizer"

    with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
        # Init + bins
        pds = PackingDataset(args, tokenizer, dataset, split="train")
        assert pds.dataset == dataset
        assert len(pds.sequence_lengths) == 4
        assert len(pds.bins) > 0
        assert len(pds) == len(pds.bins)

        # getitem returns packed sample
        sample = pds[0]
        assert isinstance(sample, list) and len(sample) > 0 and "input_ids" in sample[0]

        # Multipack method also works
        args.data.sample_packing_method = "multipack"
        pds2 = PackingDataset(args, tokenizer, dataset, split="train")
        assert len(pds2.bins) > 0

    # Missing length column raises
    bad_dataset = HFDataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]})
    with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="h"):
        with pytest.raises(AssertionError):
            PackingDataset(args, tokenizer, bad_dataset, split="train")

    # Cached bins
    with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="h"):
        with patch("xorl.data.prepare.packing.PackingDataset._load_cached_bins", return_value=[[0, 1], [2, 3]]):
            args.data.sample_packing_method = "sequential"
            pds3 = PackingDataset(args, tokenizer, dataset, split="train")
            assert pds3.bins == [[0, 1], [2, 3]]


def test_pack_parallel():
    """pack_parallel: single process and auto num_processes."""
    seq = np.array([10, 20, 30, 40, 50, 60, 70, 80])

    bins = pack_parallel(seq, 100, 3, 10, num_processes=1, safe_mode=True, mp_start_method=None)
    assert len(bins) > 0

    bins2 = pack_parallel(seq, 100, 2, 10, num_processes=None, safe_mode=True, mp_start_method=None)
    assert len(bins2) > 0
