"""Tests for xorl.data.prepare.hash module."""

from unittest.mock import Mock, patch
import pytest
from datasets import Dataset

from xorl.data.prepare.hash import (
    generate_split_fingerprints,
    generate_packing_hash,
    generate_dataset_hash_from_config,
)
from xorl.arguments import DatasetConfig


pytestmark = pytest.mark.cpu


class TestGenerateSplitFingerprints:
    """Tests for generate_split_fingerprints function."""

    def test_generates_different_fingerprints_for_train_and_test(self):
        """Should generate different fingerprints for train and test splits."""
        dataset = Mock(spec=Dataset)
        dataset._fingerprint = "base_fingerprint"

        train_fp, test_fp = generate_split_fingerprints(dataset, val_set_size=100, seed=42)

        assert train_fp != test_fp
        assert isinstance(train_fp, str)
        assert isinstance(test_fp, str)
        assert len(train_fp) == 32  # MD5 hash length
        assert len(test_fp) == 32

    def test_consistent_fingerprints_with_same_inputs(self):
        """Should generate consistent fingerprints with same inputs."""
        dataset = Mock(spec=Dataset)
        dataset._fingerprint = "base_fingerprint"

        train_fp1, test_fp1 = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
        train_fp2, test_fp2 = generate_split_fingerprints(dataset, val_set_size=100, seed=42)

        assert train_fp1 == train_fp2
        assert test_fp1 == test_fp2

    def test_different_fingerprints_with_different_val_set_size(self):
        """Should generate different fingerprints with different val_set_size."""
        dataset = Mock(spec=Dataset)
        dataset._fingerprint = "base_fingerprint"

        train_fp1, _ = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
        train_fp2, _ = generate_split_fingerprints(dataset, val_set_size=200, seed=42)

        assert train_fp1 != train_fp2

    def test_different_fingerprints_with_different_seed(self):
        """Should generate different fingerprints with different seed."""
        dataset = Mock(spec=Dataset)
        dataset._fingerprint = "base_fingerprint"

        train_fp1, _ = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
        train_fp2, _ = generate_split_fingerprints(dataset, val_set_size=100, seed=99)

        assert train_fp1 != train_fp2

    def test_different_fingerprints_with_different_dataset(self):
        """Should generate different fingerprints with different dataset fingerprint."""
        dataset1 = Mock(spec=Dataset)
        dataset1._fingerprint = "fingerprint1"

        dataset2 = Mock(spec=Dataset)
        dataset2._fingerprint = "fingerprint2"

        train_fp1, _ = generate_split_fingerprints(dataset1, val_set_size=100, seed=42)
        train_fp2, _ = generate_split_fingerprints(dataset2, val_set_size=100, seed=42)

        assert train_fp1 != train_fp2

    def test_handles_float_val_set_size(self):
        """Should handle float val_set_size."""
        dataset = Mock(spec=Dataset)
        dataset._fingerprint = "base_fingerprint"

        train_fp, test_fp = generate_split_fingerprints(dataset, val_set_size=0.1, seed=42)

        assert train_fp != test_fp
        assert len(train_fp) == 32
        assert len(test_fp) == 32


class TestGeneratePackingHash:
    """Tests for generate_packing_hash function."""

    def test_generates_consistent_hash(self):
        """Should generate consistent hash for same inputs."""
        hash1 = generate_packing_hash(
            sample_packing_method="multipack",
            sample_packing_sequence_len=2048,
            sample_packing_group_size=100,
            sample_packing_mp_start_method="fork"
        )
        hash2 = generate_packing_hash(
            sample_packing_method="multipack",
            sample_packing_sequence_len=2048,
            sample_packing_group_size=100,
            sample_packing_mp_start_method="fork"
        )

        assert hash1 == hash2

    def test_includes_all_parameters(self):
        """Should include all packing parameters in hash."""
        result = generate_packing_hash(
            sample_packing_method="multipack",
            sample_packing_sequence_len=2048,
            sample_packing_group_size=100,
            sample_packing_mp_start_method="fork"
        )

        assert "multipack" in result
        assert "2048" in result
        assert "100" in result
        assert "fork" in result

    def test_different_hash_for_different_method(self):
        """Should generate different hash for different packing method."""
        hash1 = generate_packing_hash("multipack", 2048, 100, "fork")
        hash2 = generate_packing_hash("sequential", 2048, 100, "fork")

        assert hash1 != hash2

    def test_different_hash_for_different_sequence_len(self):
        """Should generate different hash for different sequence length."""
        hash1 = generate_packing_hash("multipack", 2048, 100, "fork")
        hash2 = generate_packing_hash("multipack", 4096, 100, "fork")

        assert hash1 != hash2

    def test_different_hash_for_different_group_size(self):
        """Should generate different hash for different group size."""
        hash1 = generate_packing_hash("multipack", 2048, 100, "fork")
        hash2 = generate_packing_hash("multipack", 2048, 200, "fork")

        assert hash1 != hash2

    def test_different_hash_for_different_start_method(self):
        """Should generate different hash for different start method."""
        hash1 = generate_packing_hash("multipack", 2048, 100, "fork")
        hash2 = generate_packing_hash("multipack", 2048, 100, "spawn")

        assert hash1 != hash2

    def test_format_is_underscore_separated(self):
        """Should format hash as underscore-separated string."""
        result = generate_packing_hash("multipack", 2048, 100, "fork")
        parts = result.split("_")

        assert len(parts) == 4
        assert parts[0] == "multipack"
        assert parts[1] == "2048"
        assert parts[2] == "100"
        assert parts[3] == "fork"


class TestGenerateDatasetHashFromConfig:
    """Tests for generate_dataset_hash_from_config function."""

    def test_generates_consistent_hash(self):
        """Should generate consistent hash for same configuration."""
        args = Mock()
        args.data.select_columns = None

        dataset_config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        hash1 = generate_dataset_hash_from_config(args, [dataset_config], "gpt2")
        hash2 = generate_dataset_hash_from_config(args, [dataset_config], "gpt2")

        assert hash1 == hash2

    def test_hash_is_md5_format(self):
        """Should return MD5 hash format."""
        args = Mock()
        args.data.select_columns = None

        dataset_config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        result = generate_dataset_hash_from_config(args, [dataset_config], "gpt2")

        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_hash_for_different_tokenizer(self):
        """Should generate different hash for different tokenizer."""
        args = Mock()
        args.data.select_columns = None

        dataset_config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        hash1 = generate_dataset_hash_from_config(args, [dataset_config], "gpt2")
        hash2 = generate_dataset_hash_from_config(args, [dataset_config], "llama")

        assert hash1 != hash2

    def test_different_hash_for_different_dataset_path(self):
        """Should generate different hash for different dataset path."""
        args = Mock()
        args.data.select_columns = None

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
            max_seq_len=None
        )

        config2 = DatasetConfig(
            path="dataset2",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        hash1 = generate_dataset_hash_from_config(args, [config1], "gpt2")
        hash2 = generate_dataset_hash_from_config(args, [config2], "gpt2")

        assert hash1 != hash2

    def test_different_hash_for_different_select_columns(self):
        """Should generate different hash for different select_columns."""
        args1 = Mock()
        args1.data.select_columns = ["col1", "col2"]

        args2 = Mock()
        args2.data.select_columns = ["col1"]

        dataset_config = DatasetConfig(
            path="dataset1",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        hash1 = generate_dataset_hash_from_config(args1, [dataset_config], "gpt2")
        hash2 = generate_dataset_hash_from_config(args2, [dataset_config], "gpt2")

        assert hash1 != hash2

    def test_handles_multiple_datasets(self):
        """Should handle multiple dataset configurations."""
        args = Mock()
        args.data.select_columns = None

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
            max_seq_len=None
        )

        config2 = DatasetConfig(
            path="dataset2",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        hash1 = generate_dataset_hash_from_config(args, [config1], "gpt2")
        hash2 = generate_dataset_hash_from_config(args, [config1, config2], "gpt2")

        assert hash1 != hash2

    def test_consistent_hash_regardless_of_dataset_order(self):
        """Should generate consistent hash regardless of dataset order (sorted)."""
        args = Mock()
        args.data.select_columns = None

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
            max_seq_len=None
        )

        config2 = DatasetConfig(
            path="dataset2",
            type="tokenized",
            shards=None,
            shards_idx=None,
            preprocess_shards=None,
            name=None,
            split="train",
            revision=None,
            trust_remote_code=False,
            max_seq_len=None
        )

        hash1 = generate_dataset_hash_from_config(args, [config1, config2], "gpt2")
        hash2 = generate_dataset_hash_from_config(args, [config2, config1], "gpt2")

        # Should be same because configs are sorted by string representation
        assert hash1 == hash2
