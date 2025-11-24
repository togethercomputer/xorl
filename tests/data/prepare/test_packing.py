"""Tests for xorl.data.prepare.packing module."""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
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
)


pytestmark = pytest.mark.cpu


class TestFfdCheck:
    """Tests for ffd_check function."""

    def test_sequences_fit_in_bins(self):
        """Should return True when sequences fit in bins."""
        sequence_lengths = np.array([10, 20, 30, 40])
        bin_capacity = 50
        num_bins = 3

        result = ffd_check(sequence_lengths, bin_capacity, num_bins)

        assert result is True

    def test_sequences_do_not_fit(self):
        """Should return False when sequences don't fit."""
        sequence_lengths = np.array([40, 40, 40, 40])
        bin_capacity = 50
        num_bins = 2

        result = ffd_check(sequence_lengths, bin_capacity, num_bins)

        assert result is False

    def test_single_sequence_fits(self):
        """Should handle single sequence."""
        sequence_lengths = np.array([30])
        bin_capacity = 50
        num_bins = 1

        result = ffd_check(sequence_lengths, bin_capacity, num_bins)

        assert result is True

    def test_single_sequence_exceeds_capacity(self):
        """Should return False when single sequence exceeds bin capacity."""
        sequence_lengths = np.array([60])
        bin_capacity = 50
        num_bins = 1

        result = ffd_check(sequence_lengths, bin_capacity, num_bins)

        assert result is False

    def test_exact_fit(self):
        """Should return True for exact fit."""
        sequence_lengths = np.array([25, 25])
        bin_capacity = 50
        num_bins = 1

        result = ffd_check(sequence_lengths, bin_capacity, num_bins)

        assert result is True

    def test_empty_sequences(self):
        """Should handle empty sequences."""
        sequence_lengths = np.array([])
        bin_capacity = 50
        num_bins = 1

        result = ffd_check(sequence_lengths, bin_capacity, num_bins)

        assert result is True


class TestPackGroup:
    """Tests for pack_group function."""

    def test_packs_sequences_into_bins(self):
        """Should pack sequences into bins using FFD."""
        sequence_lengths = np.array([10, 20, 30, 40])
        group_offset = 0
        bin_capacity = 50
        max_bins = 10
        bin_size = 10

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=True)

        # Check that all sequences are packed
        all_indices = [idx for bin in bins for idx in bin]
        assert len(all_indices) == 4

        # Check bin capacities
        for bin in bins:
            total_length = sum(sequence_lengths[idx - group_offset] for idx in bin)
            assert total_length <= bin_capacity

    def test_respects_bin_size_limit(self):
        """Should not exceed bin_size (max sequences per bin)."""
        sequence_lengths = np.array([5, 5, 5, 5, 5, 5])
        group_offset = 0
        bin_capacity = 100
        max_bins = 10
        bin_size = 3

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=True)

        # Each bin should have at most bin_size sequences
        for bin in bins:
            assert len(bin) <= bin_size

    def test_safe_mode_skips_when_exceeding_max_bins(self):
        """Safe mode should skip sequences when max_bins would be exceeded."""
        sequence_lengths = np.array([40, 40, 40, 40])
        group_offset = 0
        bin_capacity = 50
        max_bins = 2
        bin_size = 1

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=True)

        # Should only pack 2 bins in safe mode
        assert len(bins) <= max_bins

    def test_non_safe_mode_allows_more_bins(self):
        """Non-safe mode should pack all sequences even if exceeding max_bins."""
        sequence_lengths = np.array([40, 40, 40, 40])
        group_offset = 0
        bin_capacity = 50
        max_bins = 2
        bin_size = 1

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=False)

        # All sequences should be packed
        all_indices = [idx for bin in bins for idx in bin]
        assert len(all_indices) == 4

    def test_applies_group_offset(self):
        """Should apply group_offset to returned indices."""
        sequence_lengths = np.array([10, 20])
        group_offset = 100
        bin_capacity = 50
        max_bins = 10
        bin_size = 10

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=True)

        all_indices = [idx for bin in bins for idx in bin]
        assert min(all_indices) >= group_offset


class TestAllocateSequentially:
    """Tests for allocate_sequentially function."""

    def test_distributes_batches_to_ranks(self):
        """Should distribute batches across ranks."""
        sequence_lengths = np.array([10, 20, 30, 40, 50, 60])
        rank = 0
        bin_capacity = 100
        num_ranks = 2

        rank_batches, total_tokens, total_slots = allocate_sequentially(
            sequence_lengths, rank, bin_capacity, num_ranks
        )

        # Rank 0 should get some batches
        assert len(rank_batches) > 0

        # All batches should fit in bin_capacity
        for batch in rank_batches:
            total_length = sum(sequence_lengths[idx] for idx in batch)
            assert total_length <= bin_capacity

    def test_different_ranks_get_different_batches(self):
        """Different ranks should get different batches."""
        sequence_lengths = np.array([10, 20, 30, 40, 50, 60])
        bin_capacity = 100
        num_ranks = 2

        rank0_batches, _, _ = allocate_sequentially(sequence_lengths, 0, bin_capacity, num_ranks)
        rank1_batches, _, _ = allocate_sequentially(sequence_lengths, 1, bin_capacity, num_ranks)

        # Extract indices
        rank0_indices = {idx for batch in rank0_batches for idx in batch}
        rank1_indices = {idx for batch in rank1_batches for idx in batch}

        # Ranks should get non-overlapping data
        assert len(rank0_indices & rank1_indices) == 0

    def test_covers_all_sequences(self):
        """All ranks together should cover all sequences."""
        sequence_lengths = np.array([10, 20, 30, 40])
        bin_capacity = 100
        num_ranks = 2

        all_indices = set()
        for rank in range(num_ranks):
            rank_batches, _, _ = allocate_sequentially(sequence_lengths, rank, bin_capacity, num_ranks)
            for batch in rank_batches:
                all_indices.update(batch)

        assert all_indices == {0, 1, 2, 3}

    def test_returns_token_statistics(self):
        """Should return total tokens used and total slots."""
        sequence_lengths = np.array([10, 20, 30])
        rank = 0
        bin_capacity = 100
        num_ranks = 1

        _, total_tokens, total_slots = allocate_sequentially(sequence_lengths, rank, bin_capacity, num_ranks)

        assert total_tokens > 0
        assert total_slots > 0
        assert total_tokens <= total_slots


class TestAddPositionIds:
    """Tests for add_position_ids function."""

    def test_adds_position_ids_to_single_sample(self):
        """Should add position_ids to single sample."""
        sample = {
            "input_ids": [1, 2, 3, 4, 5],
            "labels": [1, 2, 3, 4, 5],
        }

        result = add_position_ids(sample)

        assert "position_ids" in result
        assert "length" in result
        assert result["position_ids"] == [0, 1, 2, 3, 4]
        assert result["length"] == 5

    def test_adds_position_ids_to_batched_samples(self):
        """Should add position_ids to batched samples."""
        sample = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7]],
            "labels": [[1, 2, 3], [4, 5, 6, 7]],
        }

        result = add_position_ids(sample)

        assert "position_ids" in result
        assert "length" in result
        assert result["position_ids"] == [[0, 1, 2], [0, 1, 2, 3]]
        assert result["length"] == [3, 4]

    def test_returns_unchanged_if_input_ids_missing(self):
        """Should return unchanged if input_ids missing."""
        sample = {"labels": [1, 2, 3]}

        result = add_position_ids(sample)

        assert result == sample
        assert "position_ids" not in result

    def test_returns_unchanged_if_input_ids_empty(self):
        """Should return unchanged if input_ids empty."""
        sample = {"input_ids": []}

        result = add_position_ids(sample)

        assert result == sample
        assert "position_ids" not in result

    def test_preserves_other_fields(self):
        """Should preserve other fields in sample."""
        sample = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [1, 2, 3],
        }

        result = add_position_ids(sample)

        assert result["attention_mask"] == [1, 1, 1]
        assert result["labels"] == [1, 2, 3]


class TestDropNoTrainableTokens:
    """Tests for drop_no_trainable_tokens function."""

    def test_keeps_sample_with_trainable_tokens(self):
        """Should return True for samples with trainable tokens."""
        sample = {"labels": [1, 2, -100, 3]}

        result = drop_no_trainable_tokens(sample)

        assert result == True

    def test_drops_sample_with_all_ignore_labels(self):
        """Should return False for samples with all labels = -100."""
        sample = {"labels": [-100, -100, -100]}

        result = drop_no_trainable_tokens(sample)

        assert bool(result) == False

    def test_handles_batched_samples(self):
        """Should handle batched samples."""
        sample = {
            "labels": [
                [1, 2, 3],           # Has trainable tokens
                [-100, -100, -100],  # No trainable tokens
                [1, -100, 2]         # Has trainable tokens
            ]
        }

        result = drop_no_trainable_tokens(sample)

        assert [bool(r) for r in result] == [True, False, True]

    def test_raises_on_missing_labels(self):
        """Should raise KeyError if labels field missing."""
        sample = {"input_ids": [1, 2, 3]}

        with pytest.raises(KeyError):
            drop_no_trainable_tokens(sample)


class TestFilterDatasetWithLogging:
    """Tests for filter_dataset_with_logging function."""

    def test_filters_dataset(self):
        """Should filter dataset using provided function."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "labels": [[1, 2, 3], [-100, -100, -100], [7, 8, 9]],
        })

        filtered = filter_dataset_with_logging(
            dataset,
            lambda x: x["labels"][0] != -100,
            "test_dataset",
            num_proc=1
        )

        assert len(filtered) == 2

    def test_logs_dropped_samples(self):
        """Should filter dataset and handle logging."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "labels": [[1, 2, 3], [-100, -100, -100], [7, 8, 9]],
        })

        result = filter_dataset_with_logging(
            dataset,
            lambda x: x["labels"][0] != -100,
            "test_dataset",
            num_proc=1
        )

        # Check that filtering worked correctly - dropped the sample with all -100 labels
        assert len(result) == 2
        # Verify the correct samples remain
        assert result[0]["labels"] == [1, 2, 3]
        assert result[1]["labels"] == [7, 8, 9]


class TestProcessDatasetsForPacking:
    """Tests for process_datasets_for_packing function."""

    def test_processes_train_dataset(self):
        """Should process train dataset by adding position_ids and filtering."""
        args = Mock()
        args.data.dataset_num_proc = 1

        train_dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "labels": [[1, 2, 3], [4, 5, 6]],
        })

        processed_train, _ = process_datasets_for_packing(args, train_dataset, None)

        assert "position_ids" in processed_train.column_names
        assert "length" in processed_train.column_names

    def test_processes_eval_dataset(self):
        """Should process eval dataset if provided."""
        args = Mock()
        args.data.dataset_num_proc = 1

        train_dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]],
        })

        eval_dataset = HFDataset.from_dict({
            "input_ids": [[4, 5, 6]],
            "labels": [[4, 5, 6]],
        })

        processed_train, processed_eval = process_datasets_for_packing(args, train_dataset, eval_dataset)

        assert processed_eval is not None
        assert "position_ids" in processed_eval.column_names

    def test_handles_none_eval_dataset(self):
        """Should handle None eval_dataset."""
        args = Mock()
        args.data.dataset_num_proc = 1

        train_dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]],
        })

        processed_train, processed_eval = process_datasets_for_packing(args, train_dataset, None)

        assert processed_eval is None


class TestPackingDataset:
    """Tests for PackingDataset class."""

    @pytest.fixture
    def mock_args(self):
        """Provides mock Arguments."""
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
        return args

    @pytest.fixture
    def mock_dataset(self):
        """Provides mock HuggingFace dataset."""
        return HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            "labels": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            "length": [3, 3, 3, 3],
        })

    @pytest.fixture
    def mock_tokenizer(self):
        """Provides mock tokenizer."""
        tokenizer = Mock()
        tokenizer.name_or_path = "test-tokenizer"
        return tokenizer

    def test_initializes_with_dataset(self, mock_args, mock_dataset, mock_tokenizer):
        """Should initialize PackingDataset."""
        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            assert packing_ds.dataset == mock_dataset
            assert packing_ds.dataset_hash == "test_hash"
            assert len(packing_ds.sequence_lengths) == 4

    def test_raises_if_length_column_missing(self, mock_args, mock_tokenizer):
        """Should raise AssertionError if length column missing."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]],
        })

        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            with pytest.raises(AssertionError):
                PackingDataset(mock_args, mock_tokenizer, dataset, split="train")

    def test_computes_bins_for_sequential_method(self, mock_args, mock_dataset, mock_tokenizer):
        """Should compute bins using sequential method."""
        mock_args.data.sample_packing_method = "sequential"

        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            assert len(packing_ds.bins) > 0

    def test_computes_bins_for_multipack_method(self, mock_args, mock_dataset, mock_tokenizer):
        """Should compute bins using multipack method."""
        mock_args.data.sample_packing_method = "multipack"

        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            assert len(packing_ds.bins) > 0

    def test_getitem_returns_packed_sample(self, mock_args, mock_dataset, mock_tokenizer):
        """Should return packed sample from __getitem__."""
        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            sample = packing_ds[0]

            assert isinstance(sample, list)
            assert len(sample) > 0
            assert "input_ids" in sample[0]

    def test_len_returns_number_of_bins(self, mock_args, mock_dataset, mock_tokenizer):
        """Should return number of bins from __len__."""
        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            assert len(packing_ds) == len(packing_ds.bins)

    @patch("xorl.data.prepare.packing.PackingDataset._load_cached_bins")
    def test_uses_cached_bins_if_available(self, mock_load_cached, mock_args, mock_dataset, mock_tokenizer):
        """Should use cached bins if available."""
        mock_load_cached.return_value = [[0, 1], [2, 3]]

        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            mock_load_cached.assert_called_once()
            assert packing_ds.bins == [[0, 1], [2, 3]]

    def test_getitem_raises_index_error(self, mock_args, mock_dataset, mock_tokenizer):
        """Should raise IndexError for out of bounds index."""
        with patch("xorl.data.prepare.packing.generate_dataset_hash_from_config", return_value="test_hash"):
            packing_ds = PackingDataset(mock_args, mock_tokenizer, mock_dataset, split="train")

            with pytest.raises(IndexError):
                _ = packing_ds[len(packing_ds.bins) + 1]

    def test_empty_labels_handling(self):
        """Should handle empty labels in drop_no_trainable_tokens."""
        sample = {"labels": []}
        result = drop_no_trainable_tokens(sample)
        assert result == True


class TestPackGroupEdgeCases:
    """Tests for pack_group edge cases."""

    def test_pack_group_with_non_safe_mode(self):
        """Should allow more bins in non-safe mode."""
        sequence_lengths = np.array([40, 40, 40, 40, 40])
        group_offset = 0
        bin_capacity = 50
        max_bins = 2
        bin_size = 10

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=False)

        # In non-safe mode, should create more bins even if exceeds max_bins
        assert len(bins) >= 2

    def test_pack_group_with_group_offset(self):
        """Should apply group offset correctly."""
        sequence_lengths = np.array([10, 20])
        group_offset = 100
        bin_capacity = 50
        max_bins = 10
        bin_size = 10

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=True)

        # Check that indices include offset
        all_indices = [idx for bin in bins for idx in bin]
        assert all(idx >= 100 for idx in all_indices)

    def test_pack_group_safety_check(self):
        """Should not create infinite bins."""
        # Create a case where each sequence needs its own bin
        sequence_lengths = np.array([50] * 20)
        group_offset = 0
        bin_capacity = 50
        max_bins = 100
        bin_size = 1

        bins = pack_group(sequence_lengths, group_offset, bin_capacity, max_bins, bin_size, safe_mode=True)

        # Should not exceed sequence length (safety check)
        assert len(bins) <= len(sequence_lengths)


class TestPackParallel:
    """Tests for pack_parallel function."""

    def test_pack_parallel_single_process(self):
        """Should handle single process packing."""
        from xorl.data.prepare.packing import pack_parallel

        sequence_lengths = np.array([10, 20, 30, 40, 50])
        bin_capacity = 100
        group_size = 3
        bin_size = 10

        bins = pack_parallel(
            sequence_lengths,
            bin_capacity,
            group_size,
            bin_size,
            num_processes=1,
            safe_mode=True,
            mp_start_method=None
        )

        assert len(bins) > 0

    def test_pack_parallel_with_auto_num_processes(self):
        """Should automatically determine number of processes."""
        from xorl.data.prepare.packing import pack_parallel

        sequence_lengths = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        bin_capacity = 100
        group_size = 2
        bin_size = 10

        # Let it automatically determine num_processes
        bins = pack_parallel(
            sequence_lengths,
            bin_capacity,
            group_size,
            bin_size,
            num_processes=None,  # Auto-determine
            safe_mode=True,
            mp_start_method=None
        )

        assert len(bins) > 0


class TestAddPositionIdsEdgeCases:
    """Tests for add_position_ids edge cases."""

    def test_missing_input_ids(self):
        """Should return unchanged if input_ids missing."""
        sample = {"labels": [1, 2, 3]}
        result = add_position_ids(sample)
        assert result == sample
        assert "position_ids" not in result

    def test_empty_input_ids(self):
        """Should return unchanged if input_ids empty."""
        sample = {"input_ids": []}
        result = add_position_ids(sample)
        assert result == sample
