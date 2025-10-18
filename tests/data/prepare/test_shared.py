"""Tests for xorl.data.prepare.shared module."""

from unittest.mock import Mock, patch
import pytest
from datasets import Dataset as HFDataset

from xorl.data.prepare.shared import (
    get_dataset_type,
    datasets_with_name_generator,
    load_dataset_with_config,
    _check_if_hub_dataset,
    _get_remote_filesystem,
    create_train_validation_split,
    merge_datasets,
)
from xorl.arguments import DatasetConfig


pytestmark = pytest.mark.cpu


def _make_config(**overrides):
    """Helper to create DatasetConfig with sensible defaults."""
    defaults = dict(
        path="dataset1",
        type="tokenized",
        shards=None,
        shards_idx=None,
        preprocess_shards=None,
        name=None,
        split="train",
        revision=None,
        trust_remote_code=False,
        max_seq_len=None,
    )
    defaults.update(overrides)
    return DatasetConfig(**defaults)


class TestDatasetsWithNameGeneratorAndDatasetType:
    """Tests for datasets_with_name_generator and get_dataset_type."""

    def test_expansion_and_passthrough_behaviors(self):
        """Covers name expansion, preprocess_shards expansion, shards-blocks expansion,
        and get_dataset_type inference from extension and explicit ds_type."""
        # Multiple names expansion
        config_names = _make_config(path="ds1", name=["c1", "c2"])
        result = list(datasets_with_name_generator([config_names]))
        assert len(result) == 2
        assert result[0].name == "c1"
        assert result[1].name == "c2"

        # Preprocess_shards expansion
        config_shards = _make_config(path="ds1", preprocess_shards=3)
        result = list(datasets_with_name_generator([config_shards]))
        assert len(result) == 3
        assert [r.shards_idx for r in result] == [0, 1, 2]

        # Preprocess_shards NOT expanded when shards already set
        config_no_expand = _make_config(path="ds1", shards=4, preprocess_shards=3)
        result = list(datasets_with_name_generator([config_no_expand]))
        assert len(result) == 1

        # get_dataset_type: explicit ds_type overrides extension
        config_explicit = _make_config(path="data.parquet", ds_type="arrow")
        assert get_dataset_type(config_explicit) == "arrow"

        # get_dataset_type: infer from extension
        extension_map = [
            ("data.parquet", "parquet"),
            ("data.arrow", "arrow"),
            ("data.csv", "csv"),
            ("data.txt", "text"),
            ("data.json", "json"),
            ("data.unknown", "json"),
        ]
        for path, expected_type in extension_map:
            assert get_dataset_type(_make_config(path=path)) == expected_type


class TestHubAndRemoteDetection:
    """Tests for _check_if_hub_dataset and _get_remote_filesystem."""

    @patch("xorl.data.prepare.shared.snapshot_download")
    def test_hub_detection_and_remote_filesystem(self, mock_snapshot_download):
        """Covers hub dataset detection (valid/invalid) and remote filesystem resolution."""
        # Valid hub dataset
        mock_snapshot_download.return_value = "/path/to/dataset"
        assert _check_if_hub_dataset(_make_config(path="user/ds"), use_auth_token=False) is True

        # Invalid hub dataset
        from huggingface_hub.errors import RepositoryNotFoundError
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_snapshot_download.side_effect = RepositoryNotFoundError("not found", response=mock_response)
        assert _check_if_hub_dataset(_make_config(path="invalid/ds"), use_auth_token=False) is False

        # Non-remote path
        fs, opts = _get_remote_filesystem("local/path")
        assert fs is None
        assert opts == {}

        # S3 path
        pytest.importorskip("s3fs")
        with patch("s3fs.S3FileSystem") as mock_s3fs_class:
            mock_fs = Mock()
            mock_s3fs_class.return_value = mock_fs
            fs, opts = _get_remote_filesystem("s3://bucket/path")
            assert fs == mock_fs
            assert opts == {"anon": False}


class TestSplitAndMerge:
    """Tests for create_train_validation_split and merge_datasets."""

    def test_split_and_merge_operations(self):
        """Covers absolute/fractional split, merge with shuffle variants, and empty merge error."""
        dataset = HFDataset.from_dict({
            "input_ids": [[i] for i in range(10)],
            "labels": [[i] for i in range(10)],
        })
        args = Mock()
        args.train.seed = 42

        # Absolute val_set_size
        train_ds, eval_ds = create_train_validation_split(dataset, args, val_set_size=2)
        assert len(train_ds) == 8
        assert len(eval_ds) == 2

        # Fractional val_set_size
        train_ds, eval_ds = create_train_validation_split(dataset, args, val_set_size=0.2)
        assert len(train_ds) == 8
        assert len(eval_ds) == 2

        # Merge without shuffle
        ds1 = HFDataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]})
        ds2 = HFDataset.from_dict({"input_ids": [[4, 5, 6]], "labels": [[4, 5, 6]]})
        args.data.shuffle_merged_datasets = False
        args.data.shuffle_before_merging_datasets = False
        assert len(merge_datasets([ds1, ds2], args)) == 2

        # Merge with shuffle_merged_datasets
        ds1 = HFDataset.from_dict({"input_ids": [[i] for i in range(3)], "labels": [[i] for i in range(3)]})
        ds2 = HFDataset.from_dict({"input_ids": [[i] for i in range(3)], "labels": [[i] for i in range(3)]})
        args.data.shuffle_merged_datasets = True
        args.data.shuffle_before_merging_datasets = False
        assert len(merge_datasets([ds1, ds2], args)) == 6

        # Merge with shuffle_before_merging
        args.data.shuffle_merged_datasets = False
        args.data.shuffle_before_merging_datasets = True
        assert len(merge_datasets([ds1, ds2], args)) == 6

        # Empty dataset list raises ValueError
        args.data.shuffle_merged_datasets = False
        with pytest.raises(ValueError):
            merge_datasets([], args)


class TestLoadDatasetWithConfig:
    """Tests for load_dataset_with_config function."""

    def test_local_and_hub_loading(self, tmp_path):
        """Covers loading from local file, local directory, hub, URL, data_files, and error on missing."""
        # Local JSON file
        data_file = tmp_path / "data.json"
        data_file.write_text('{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}')
        config = _make_config(path=str(data_file), split=None)
        assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None

        # Local directory (saved HF dataset)
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        HFDataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]}).save_to_disk(str(dataset_dir))
        config = _make_config(path=str(dataset_dir), split=None)
        ds = load_dataset_with_config(config, use_auth_token=False, streaming=False)
        assert ds is not None and len(ds) == 1

        # Hub dataset
        with patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=True), \
             patch("xorl.data.prepare.shared._load_from_hub",
                   return_value=HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})) as mock_hub:
            config = _make_config(path="username/dataset", split=None)
            assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
            mock_hub.assert_called_once()

        # HTTPS URL
        with patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=False), \
             patch("xorl.data.prepare.shared._load_from_url",
                   return_value=HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})) as mock_url:
            config = _make_config(path="https://example.com/dataset.json", split=None)
            assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
            mock_url.assert_called_once()

        # No valid source raises ValueError
        with patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=False):
            config = _make_config(path="nonexistent/dataset", split=None, data_files=None)
            with pytest.raises(ValueError, match="The dataset could not be loaded"):
                load_dataset_with_config(config, use_auth_token=False, streaming=False)

    @patch("xorl.data.prepare.shared._check_if_hub_dataset")
    @patch("xorl.data.prepare.shared.hf_hub_download")
    @patch("xorl.data.prepare.shared.load_dataset")
    def test_data_files_string_and_list(self, mock_load_dataset, mock_hub_download, mock_check_hub):
        """Covers loading from data_files as string and as list."""
        mock_check_hub.return_value = False
        mock_load_dataset.return_value = HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]})

        # data_files as string
        mock_hub_download.return_value = "/tmp/file.json"
        config = _make_config(path="user/ds", split=None, data_files="data.json", ds_type="json")
        assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
        mock_hub_download.assert_called_once()

        # data_files as list
        mock_hub_download.reset_mock()
        mock_hub_download.side_effect = ["/tmp/file1.json", "/tmp/file2.json"]
        config = _make_config(path="user/ds", split=None, data_files=["d1.json", "d2.json"], ds_type="json")
        assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
        assert mock_hub_download.call_count == 2


class TestSaveAndLoadPreprocessedDataset:
    """Tests for save/load preprocessed dataset functions."""

    def test_save_load_and_missing(self, tmp_path):
        """Covers save+load round-trip and load returning None when not found."""
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

        save_preprocessed_dataset(args, dataset, "test_hash_123", split="train")
        loaded = load_preprocessed_dataset(args, "test_hash_123")
        assert loaded is not None and len(loaded) == 2

        assert load_preprocessed_dataset(args, "nonexistent_hash") is None
