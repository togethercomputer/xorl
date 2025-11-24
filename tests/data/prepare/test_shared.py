"""Tests for xorl.data.prepare.shared module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from datasets import Dataset as HFDataset, DatasetDict

from xorl.data.prepare.shared import (
    EXTENSIONS_TO_DATASET_TYPES,
    get_dataset_type,
    datasets_with_name_generator,
    load_dataset_with_config,
    _check_if_hub_dataset,
    _get_remote_filesystem,
    get_prepared_dataset_path,
    create_train_validation_split,
    merge_datasets,
)
from xorl.arguments import DatasetConfig


pytestmark = pytest.mark.cpu


class TestExtensionsToDatasetTypes:
    """Tests for EXTENSIONS_TO_DATASET_TYPES constant."""

    def test_has_expected_extensions(self):
        """Should have expected file extension mappings."""
        assert ".parquet" in EXTENSIONS_TO_DATASET_TYPES
        assert ".arrow" in EXTENSIONS_TO_DATASET_TYPES
        assert ".csv" in EXTENSIONS_TO_DATASET_TYPES
        assert ".txt" in EXTENSIONS_TO_DATASET_TYPES


class TestGetDatasetType:
    """Tests for get_dataset_type function."""

    def test_infers_parquet_type(self):
        """Should infer parquet type from extension."""
        config = DatasetConfig(
            path="data.parquet",
            type="tokenized",
            ds_type=None,
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = get_dataset_type(config)
        assert result == "parquet"

    def test_infers_arrow_type(self):
        """Should infer arrow type from extension."""
        config = DatasetConfig(
            path="data.arrow",
            type="tokenized",
            ds_type=None,
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = get_dataset_type(config)
        assert result == "arrow"

    def test_defaults_to_json(self):
        """Should default to json for unknown extensions."""
        config = DatasetConfig(
            path="data.unknown",
            type="tokenized",
            ds_type=None,
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = get_dataset_type(config)
        assert result == "json"

    def test_handles_path_without_extension(self):
        """Should default to json for paths without extension."""
        config = DatasetConfig(
            path="my_dataset",
            type="tokenized",
            ds_type=None,
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = get_dataset_type(config)
        assert result == "json"


class TestDatasetsWithNameGenerator:
    """Tests for datasets_with_name_generator function."""

    def test_yields_single_config_unchanged(self):
        """Should yield single config unchanged."""
        config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = list(datasets_with_name_generator([config]))

        assert len(result) == 1
        assert result[0].path == "dataset1"

    def test_expands_multiple_names(self):
        """Should expand config with multiple names."""
        config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=["config1", "config2"],
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = list(datasets_with_name_generator([config]))

        assert len(result) == 2
        assert result[0].name == "config1"
        assert result[1].name == "config2"

    def test_expands_preprocess_shards(self):
        """Should expand config with preprocess_shards."""
        config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=3,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = list(datasets_with_name_generator([config]))

        assert len(result) == 3
        assert result[0].shards_idx == 0
        assert result[1].shards_idx == 1
        assert result[2].shards_idx == 2

    def test_does_not_expand_if_shards_set(self):
        """Should not expand preprocess_shards if shards already set."""
        config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=4,
            shards_idx=None,
            preprocess_shards=3,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = list(datasets_with_name_generator([config]))

        assert len(result) == 1

    def test_handles_multiple_configs(self):
        """Should handle multiple configs."""
        config1 = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        config2 = DatasetConfig(
            path="dataset2",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=2,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = list(datasets_with_name_generator([config1, config2]))

        assert len(result) == 3  # 1 + 2


class TestCheckIfHubDataset:
    """Tests for _check_if_hub_dataset function."""

    @patch("xorl.data.prepare.shared.snapshot_download")
    def test_returns_true_for_valid_hub_dataset(self, mock_snapshot_download):
        """Should return True for valid HuggingFace Hub dataset."""
        mock_snapshot_download.return_value = "/path/to/dataset"

        config = DatasetConfig(
            path="username/dataset",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = _check_if_hub_dataset(config, use_auth_token=False)

        assert result is True

    @patch("xorl.data.prepare.shared.snapshot_download")
    def test_returns_false_for_invalid_hub_dataset(self, mock_snapshot_download):
        """Should return False if dataset not found on Hub."""
        from huggingface_hub.errors import RepositoryNotFoundError

        mock_snapshot_download.side_effect = RepositoryNotFoundError("not found")

        config = DatasetConfig(
            path="invalid/dataset",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        result = _check_if_hub_dataset(config, use_auth_token=False)

        assert result is False


class TestGetRemoteFilesystem:
    """Tests for _get_remote_filesystem function."""

    def test_returns_none_for_non_remote_path(self):
        """Should return None for non-remote paths."""
        fs, opts = _get_remote_filesystem("local/path")

        assert fs is None
        assert opts == {}

    def test_returns_filesystem_for_s3_path(self):
        """Should return filesystem for s3:// paths."""
        pytest.importorskip("s3fs")
        with patch("s3fs.S3FileSystem") as mock_s3fs_class:
            mock_fs = Mock()
            mock_s3fs_class.return_value = mock_fs

            fs, opts = _get_remote_filesystem("s3://bucket/path")

            assert fs == mock_fs
            assert opts == {"anon": False}
            mock_s3fs_class.assert_called_once_with(anon=False)


class TestGetPreparedDatasetPath:
    """Tests for get_prepared_dataset_path function."""

    def test_uses_custom_path_when_provided(self):
        """Should use custom dataset_prepared_path when provided."""
        args = Mock()
        args.data.dataset_prepared_path = "/custom/path"

        result = get_prepared_dataset_path(args, "test_hash")

        assert "/custom/path" in str(result)
        assert "test_hash" in str(result)

    def test_uses_default_path_when_none(self):
        """Should use default path when dataset_prepared_path is None."""
        args = Mock()
        args.data.dataset_prepared_path = None

        result = get_prepared_dataset_path(args, "test_hash")

        assert "last_prepared_dataset" in str(result)
        assert "test_hash" in str(result)


class TestCreateTrainValidationSplit:
    """Tests for create_train_validation_split function."""

    def test_splits_dataset_with_absolute_val_size(self):
        """Should split dataset with absolute validation size."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            "labels": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        })

        args = Mock()
        args.train.seed = 42

        train_ds, eval_ds = create_train_validation_split(dataset, args, val_set_size=2)

        assert len(train_ds) == 8
        assert len(eval_ds) == 2

    def test_splits_dataset_with_fractional_val_size(self):
        """Should split dataset with fractional validation size."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            "labels": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        })

        args = Mock()
        args.train.seed = 42

        train_ds, eval_ds = create_train_validation_split(dataset, args, val_set_size=0.2)

        assert len(train_ds) == 8
        assert len(eval_ds) == 2

    def test_uses_seed_for_reproducibility(self):
        """Should produce consistent splits with same seed."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            "labels": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        })

        args = Mock()
        args.train.seed = 42

        train_ds1, eval_ds1 = create_train_validation_split(dataset, args, val_set_size=0.2)
        train_ds2, eval_ds2 = create_train_validation_split(dataset, args, val_set_size=0.2)

        assert train_ds1._fingerprint == train_ds2._fingerprint
        assert eval_ds1._fingerprint == eval_ds2._fingerprint


class TestMergeDatasets:
    """Tests for merge_datasets function."""

    def test_returns_single_dataset_unchanged(self):
        """Should return single dataset unchanged."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]],
        })

        args = Mock()
        args.data.shuffle_merged_datasets = False
        args.train.seed = 42

        result = merge_datasets([dataset], args)

        assert len(result) == 1

    def test_merges_multiple_datasets(self):
        """Should merge multiple datasets."""
        dataset1 = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]],
        })

        dataset2 = HFDataset.from_dict({
            "input_ids": [[4, 5, 6]],
            "labels": [[4, 5, 6]],
        })

        args = Mock()
        args.data.shuffle_merged_datasets = False
        args.data.shuffle_before_merging_datasets = False
        args.train.seed = 42

        result = merge_datasets([dataset1, dataset2], args)

        assert len(result) == 2

    def test_shuffles_merged_dataset_when_enabled(self):
        """Should shuffle merged dataset when flag is True."""
        dataset1 = HFDataset.from_dict({
            "input_ids": [[1], [2], [3]],
            "labels": [[1], [2], [3]],
        })

        dataset2 = HFDataset.from_dict({
            "input_ids": [[4], [5], [6]],
            "labels": [[4], [5], [6]],
        })

        args = Mock()
        args.data.shuffle_merged_datasets = True
        args.data.shuffle_before_merging_datasets = False
        args.train.seed = 42

        result = merge_datasets([dataset1, dataset2], args)

        assert len(result) == 6

    def test_shuffles_before_merging_when_enabled(self):
        """Should shuffle individual datasets before merging."""
        dataset1 = HFDataset.from_dict({
            "input_ids": [[1], [2], [3]],
            "labels": [[1], [2], [3]],
        })

        dataset2 = HFDataset.from_dict({
            "input_ids": [[4], [5], [6]],
            "labels": [[4], [5], [6]],
        })

        args = Mock()
        args.data.shuffle_merged_datasets = False
        args.data.shuffle_before_merging_datasets = True
        args.train.seed = 42

        result = merge_datasets([dataset1, dataset2], args)

        assert len(result) == 6

    def test_handles_empty_dataset_list(self):
        """Should handle empty dataset list."""
        args = Mock()
        args.data.shuffle_merged_datasets = False
        args.train.seed = 42

        with pytest.raises(ValueError):
            merge_datasets([], args)


class TestLoadDatasetWithConfig:
    """Tests for load_dataset_with_config function."""

    def test_loads_from_local_file(self, tmp_path):
        """Should load from local file."""
        # Create a temporary JSON file
        data_file = tmp_path / "data.json"
        data_file.write_text('{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}')

        config = DatasetConfig(
            path=str(data_file),
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        dataset = load_dataset_with_config(config, use_auth_token=False, streaming=False)

        assert dataset is not None

    def test_loads_from_local_directory(self, tmp_path):
        """Should load from local directory."""
        # Create a temporary directory with dataset
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Create a simple dataset and save it
        test_dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]],
        })
        test_dataset.save_to_disk(str(dataset_dir))

        config = DatasetConfig(
            path=str(dataset_dir),
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        dataset = load_dataset_with_config(config, use_auth_token=False, streaming=False)

        assert dataset is not None
        assert len(dataset) == 1

    @patch("xorl.data.prepare.shared._check_if_hub_dataset")
    @patch("xorl.data.prepare.shared._load_from_hub")
    def test_loads_from_hub_when_available(self, mock_load_hub, mock_check_hub):
        """Should load from HuggingFace Hub when dataset exists there."""
        mock_check_hub.return_value = True
        mock_load_hub.return_value = HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})

        config = DatasetConfig(
            path="username/dataset",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        dataset = load_dataset_with_config(config, use_auth_token=False, streaming=False)

        assert dataset is not None
        mock_load_hub.assert_called_once()

    def test_raises_when_no_valid_source(self):
        """Should raise ValueError when no valid source found."""
        config = DatasetConfig(
            path="nonexistent/dataset",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None,
            data_files=None
        )

        with patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=False):
            with pytest.raises(ValueError, match="The dataset could not be loaded"):
                load_dataset_with_config(config, use_auth_token=False, streaming=False)

    @patch("xorl.data.prepare.shared._check_if_hub_dataset")
    @patch("xorl.data.prepare.shared._load_from_url")
    def test_loads_from_https_url(self, mock_load_url, mock_check_hub):
        """Should load from HTTPS URL."""
        mock_check_hub.return_value = False
        mock_load_url.return_value = HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})

        config = DatasetConfig(
            path="https://example.com/dataset.json",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None
        )

        dataset = load_dataset_with_config(config, use_auth_token=False, streaming=False)

        assert dataset is not None
        mock_load_url.assert_called_once()

    @patch("xorl.data.prepare.shared._check_if_hub_dataset")
    @patch("xorl.data.prepare.shared.hf_hub_download")
    @patch("xorl.data.prepare.shared.load_dataset")
    def test_loads_from_data_files_string(self, mock_load_dataset, mock_hub_download, mock_check_hub):
        """Should load from data_files as string."""
        mock_check_hub.return_value = False
        mock_hub_download.return_value = "/tmp/file.json"
        mock_load_dataset.return_value = HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})

        config = DatasetConfig(
            path="username/dataset",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None,
            data_files="data.json",
            ds_type="json"
        )

        dataset = load_dataset_with_config(config, use_auth_token=False, streaming=False)

        assert dataset is not None
        mock_hub_download.assert_called_once()

    @patch("xorl.data.prepare.shared._check_if_hub_dataset")
    @patch("xorl.data.prepare.shared.hf_hub_download")
    @patch("xorl.data.prepare.shared.load_dataset")
    def test_loads_from_data_files_list(self, mock_load_dataset, mock_hub_download, mock_check_hub):
        """Should load from data_files as list."""
        mock_check_hub.return_value = False
        mock_hub_download.side_effect = ["/tmp/file1.json", "/tmp/file2.json"]
        mock_load_dataset.return_value = HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})

        config = DatasetConfig(
            path="username/dataset",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split=None,
            revision=None,
            trust_remote_code=False,
            sequence_len=None,
            data_files=["data1.json", "data2.json"],
            ds_type="json"
        )

        dataset = load_dataset_with_config(config, use_auth_token=False, streaming=False)

        assert dataset is not None
        assert mock_hub_download.call_count == 2


class TestGetDatasetType:
    """Tests for get_dataset_type function."""

    def test_returns_ds_type_when_specified(self):
        """Should return ds_type when explicitly specified."""
        config = DatasetConfig(
            path="dataset.parquet",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            sequence_len=None,
            ds_type="arrow"
        )

        result = get_dataset_type(config)
        assert result == "arrow"

    def test_infers_type_from_extension(self):
        """Should infer type from file extension."""
        test_cases = [
            ("data.parquet", "parquet"),
            ("data.arrow", "arrow"),
            ("data.csv", "csv"),
            ("data.txt", "text"),
            ("data.json", "json"),
            ("data.unknown", "json"),
        ]

        for path, expected_type in test_cases:
            config = DatasetConfig(
                path=path,
                type="tokenized",
                shards=None,
                shards_idx=None,
                preprocess_shards=None,
                name=None,
                split="train",
                revision=None,
                trust_remote_code=False,
                sequence_len=None
            )
            result = get_dataset_type(config)
            assert result == expected_type


class TestSaveAndLoadPreprocessedDataset:
    """Tests for save/load preprocessed dataset functions."""

    def test_save_and_load_preprocessed_dataset(self, tmp_path):
        """Should save and load preprocessed dataset."""
        from xorl.data.prepare.shared import save_preprocessed_dataset, load_preprocessed_dataset

        args = Mock()
        args.data.dataset_prepared_path = str(tmp_path)
        args.data.dataset_num_proc = 1
        args.data.num_dataset_shards_to_save = None
        args.data.push_dataset_to_hub = None
        args.data.skip_prepare_dataset = False
        args.data.is_preprocess = False

        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "labels": [[1, 2, 3], [4, 5, 6]],
        })

        dataset_hash = "test_hash_123"

        # Save the dataset
        save_preprocessed_dataset(args, dataset, dataset_hash, split="train")

        # Load it back
        loaded = load_preprocessed_dataset(args, dataset_hash)

        assert loaded is not None
        assert len(loaded) == 2

    def test_load_preprocessed_returns_none_when_not_found(self, tmp_path):
        """Should return None when preprocessed dataset not found."""
        from xorl.data.prepare.shared import load_preprocessed_dataset

        args = Mock()
        args.data.dataset_prepared_path = str(tmp_path)
        args.data.skip_prepare_dataset = False
        args.data.is_preprocess = False

        result = load_preprocessed_dataset(args, "nonexistent_hash")

        assert result is None


class TestMergeDatasetsSingleWithShuffle:
    """Tests for merge_datasets with single dataset and shuffle enabled."""

    def test_single_dataset_with_shuffle_enabled(self):
        """Should shuffle single dataset when shuffle_merged_datasets is True."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1], [2], [3], [4], [5]],
            "labels": [[1], [2], [3], [4], [5]],
        })

        args = Mock()
        args.data.shuffle_merged_datasets = True
        args.train.seed = 42

        result = merge_datasets([dataset], args)

        assert len(result) == 5
