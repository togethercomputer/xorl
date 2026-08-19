"""Tests for xorl.data.prepare.shared module."""

from unittest.mock import Mock, patch

import pytest
import requests
from datasets import Dataset as HFDataset
from huggingface_hub.errors import HfHubHTTPError

from xorl.arguments import DatasetConfig
from xorl.data.prepare.hash import generate_dataset_hash_from_config, generate_split_fingerprints
from xorl.data.prepare.shared import (
    create_train_validation_split,
    datasets_with_name_generator,
    get_dataset_type,
    load_dataset_with_config,
    load_preprocessed_dataset,
    merge_datasets,
    save_preprocessed_dataset,
)
from xorl.data.prepare.utils import retry_on_request_exceptions


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

    def test_dataset_preparation_lifecycle(self, tmp_path, monkeypatch):
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

        TestSplitAndMerge()._assert_split_and_merge_operations()
        load_root = tmp_path / "load"
        load_root.mkdir()
        with monkeypatch.context() as load_patch:
            TestLoadDatasetWithConfig()._assert_local_and_hub_loading(load_root, load_patch)
        save_root = tmp_path / "save"
        save_root.mkdir()
        TestSaveAndLoadPreprocessedDataset()._assert_save_load_and_missing(save_root)
        _assert_dataset_hash_and_split_fingerprint_policy()
        _assert_request_retry_policy(monkeypatch)


def _assert_request_retry_policy(monkeypatch):
    sleeps = []
    monkeypatch.setattr("xorl.data.prepare.utils.time.sleep", sleeps.append)

    transient = Mock(
        side_effect=[requests.exceptions.ReadTimeout("timeout"), requests.exceptions.ReadTimeout("timeout"), "success"]
    )
    wrapped = retry_on_request_exceptions(max_retries=3, delay=0.01)(transient)
    assert wrapped() == "success"
    assert transient.call_count == 3
    assert sleeps == [0.01, 0.02]

    response = Mock(status_code=500, headers={})
    hub_transient = Mock(side_effect=[HfHubHTTPError("HF error", response=response), "success"])
    wrapped = retry_on_request_exceptions(max_retries=3, delay=0.01)(hub_transient)
    sleeps.clear()
    assert wrapped() == "success"
    assert sleeps == [0.01]

    persistent = Mock(side_effect=requests.exceptions.ReadTimeout("persistent timeout"))
    wrapped = retry_on_request_exceptions(max_retries=2, delay=0.01)(persistent)
    sleeps.clear()
    with pytest.raises(requests.exceptions.ReadTimeout):
        wrapped()
    assert persistent.call_count == 2
    assert sleeps == [0.01]

    unrelated = Mock(side_effect=ValueError("not a request exception"))
    wrapped = retry_on_request_exceptions(max_retries=3, delay=0.01)(unrelated)
    sleeps.clear()
    with pytest.raises(ValueError):
        wrapped()
    assert unrelated.call_count == 1
    assert sleeps == []


def _assert_dataset_hash_and_split_fingerprint_policy():
    dataset = Mock(spec=HFDataset)
    dataset._fingerprint = "base_fingerprint"
    train, evaluation = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
    train_again, evaluation_again = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
    assert train != evaluation
    assert (train, evaluation) == (train_again, evaluation_again)
    assert len(train) == len(evaluation) == 32

    dataset_two = Mock(spec=HFDataset)
    dataset_two._fingerprint = "fingerprint2"
    assert train != generate_split_fingerprints(dataset, val_set_size=200, seed=42)[0]
    assert train != generate_split_fingerprints(dataset, val_set_size=100, seed=99)[0]
    assert train != generate_split_fingerprints(dataset_two, val_set_size=100, seed=42)[0]
    fractional_train, fractional_evaluation = generate_split_fingerprints(dataset, val_set_size=0.1, seed=42)
    assert fractional_train != fractional_evaluation

    args = Mock()
    args.data.select_columns = None
    config = _make_config()
    dataset_hash = generate_dataset_hash_from_config(args, [config], "gpt2")
    assert dataset_hash == generate_dataset_hash_from_config(args, [config], "gpt2")
    assert len(dataset_hash) == 32

    args_with_columns = Mock()
    args_with_columns.data.select_columns = ["col1", "col2"]
    config_two = _make_config(path="dataset2")
    assert dataset_hash != generate_dataset_hash_from_config(args, [config], "llama")
    assert dataset_hash != generate_dataset_hash_from_config(args, [config_two], "gpt2")
    assert dataset_hash != generate_dataset_hash_from_config(args_with_columns, [config], "gpt2")
    assert dataset_hash != generate_dataset_hash_from_config(args, [config, config_two], "gpt2")
    assert generate_dataset_hash_from_config(args, [config, config_two], "gpt2") == generate_dataset_hash_from_config(
        args, [config_two, config], "gpt2"
    )


class TestSplitAndMerge:
    """Tests for create_train_validation_split and merge_datasets."""

    def _assert_split_and_merge_operations(self):
        """Covers absolute/fractional split, merge with shuffle variants, and empty merge error."""
        dataset = HFDataset.from_dict(
            {
                "input_ids": [[i] for i in range(10)],
                "labels": [[i] for i in range(10)],
            }
        )
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

        # The shuffle modes must change ordering, not merely preserve row count.
        ds1 = HFDataset.from_dict({"input_ids": [[i] for i in range(3)], "labels": [[i] for i in range(3)]})
        ds2 = HFDataset.from_dict({"input_ids": [[i] for i in range(3, 6)], "labels": [[i] for i in range(3, 6)]})
        args.data.shuffle_merged_datasets = False
        args.data.shuffle_before_merging_datasets = False
        merged = merge_datasets([ds1, ds2], args)
        assert [row[0] for row in merged["input_ids"]] == list(range(6))

        args.data.shuffle_merged_datasets = True
        shuffled_values = [row[0] for row in merge_datasets([ds1, ds2], args)["input_ids"]]
        assert sorted(shuffled_values) == list(range(6))
        assert shuffled_values != list(range(6))

        args.data.shuffle_merged_datasets = False
        args.data.shuffle_before_merging_datasets = True
        individually_shuffled = [row[0] for row in merge_datasets([ds1, ds2], args)["input_ids"]]
        assert set(individually_shuffled[:3]) == set(range(3))
        assert set(individually_shuffled[3:]) == set(range(3, 6))
        assert individually_shuffled != list(range(6))

        # Empty dataset list raises ValueError
        args.data.shuffle_merged_datasets = False
        with pytest.raises(ValueError):
            merge_datasets([], args)


class TestLoadDatasetWithConfig:
    """Tests for load_dataset_with_config function."""

    def _assert_local_and_hub_loading(self, tmp_path, monkeypatch):
        import datasets.config

        hf_cache = str(tmp_path / "hf_cache")
        monkeypatch.setenv("HF_DATASETS_CACHE", hf_cache)
        monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", hf_cache)
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
        with (
            patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=True),
            patch(
                "xorl.data.prepare.shared._load_from_hub",
                return_value=HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]}),
            ) as mock_hub,
        ):
            config = _make_config(path="username/dataset", split=None)
            assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
            mock_hub.assert_called_once()

        # HTTPS URL
        with (
            patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=False),
            patch(
                "xorl.data.prepare.shared._load_from_url",
                return_value=HFDataset.from_dict({"input_ids": [[1]], "labels": [[1]]}),
            ) as mock_url,
        ):
            config = _make_config(path="https://example.com/dataset.json", split=None)
            assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
            mock_url.assert_called_once()

        # No valid source raises ValueError
        with patch("xorl.data.prepare.shared._check_if_hub_dataset", return_value=False):
            config = _make_config(path="nonexistent/dataset", split=None, data_files=None)
            with pytest.raises(ValueError, match="The dataset could not be loaded"):
                load_dataset_with_config(config, use_auth_token=False, streaming=False)

        self._assert_data_files_string_and_list()

    @patch("xorl.data.prepare.shared._check_if_hub_dataset")
    @patch("xorl.data.prepare.shared.hf_hub_download")
    @patch("xorl.data.prepare.shared.load_dataset")
    def _assert_data_files_string_and_list(self, mock_load_dataset, mock_hub_download, mock_check_hub):
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
        mock_hub_download.side_effect = ["/tmp/file1.parquet", "/tmp/file2.parquet"]
        config = _make_config(path="user/ds", split=None, data_files=["d1.parquet", "d2.parquet"], ds_type="parquet")
        assert load_dataset_with_config(config, use_auth_token=False, streaming=False) is not None
        assert mock_hub_download.call_count == 2


class TestSaveAndLoadPreprocessedDataset:
    """Tests for save/load preprocessed dataset functions."""

    def _assert_save_load_and_missing(self, tmp_path):
        """Covers save+load round-trip and load returning None when not found."""

        args = Mock()
        args.data.dataset_prepared_path = str(tmp_path)
        args.data.dataset_num_proc = 1
        args.data.num_dataset_shards_to_save = None
        args.data.push_dataset_to_hub = None
        args.data.skip_prepare_dataset = False
        args.data.is_preprocess = False

        dataset = HFDataset.from_dict(
            {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "labels": [[1, 2, 3], [4, 5, 6]],
            }
        )

        save_preprocessed_dataset(args, dataset, "test_hash_123", split="train")
        loaded = load_preprocessed_dataset(args, "test_hash_123")
        assert loaded is not None and len(loaded) == 2

        assert load_preprocessed_dataset(args, "nonexistent_hash") is None
