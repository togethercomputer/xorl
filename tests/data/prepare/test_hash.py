"""Tests for xorl.data.prepare.hash module."""

from unittest.mock import Mock

import pytest
from datasets import Dataset

from xorl.arguments import DatasetConfig
from xorl.data.prepare.hash import (
    generate_dataset_hash_from_config,
    generate_packing_hash,
    generate_split_fingerprints,
)


pytestmark = pytest.mark.cpu


def _make_config(path="dataset1"):
    return DatasetConfig(
        path=path,
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


class TestGenerateSplitFingerprints:
    """Tests for generate_split_fingerprints function."""

    def test_fingerprint_properties(self):
        """Covers train/test difference, consistency, different inputs produce different results, and float val_set_size."""
        dataset = Mock(spec=Dataset)
        dataset._fingerprint = "base_fingerprint"

        # Train and test are different, consistent, and correct length
        t1, e1 = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
        t2, e2 = generate_split_fingerprints(dataset, val_set_size=100, seed=42)
        assert t1 != e1 and len(t1) == 32 and len(e1) == 32
        assert t1 == t2 and e1 == e2

        # Different val_set_size, seed, and dataset produce different fingerprints
        dataset2 = Mock(spec=Dataset)
        dataset2._fingerprint = "fingerprint2"
        diff_val, _ = generate_split_fingerprints(dataset, val_set_size=200, seed=42)
        diff_seed, _ = generate_split_fingerprints(dataset, val_set_size=100, seed=99)
        diff_ds, _ = generate_split_fingerprints(dataset2, val_set_size=100, seed=42)
        assert t1 != diff_val and t1 != diff_seed and t1 != diff_ds

        # Float val_set_size
        tf, ef = generate_split_fingerprints(dataset, val_set_size=0.1, seed=42)
        assert tf != ef and len(tf) == 32


class TestGeneratePackingHash:
    """Tests for generate_packing_hash function."""

    def test_packing_hash_format_consistency_and_uniqueness(self):
        """Covers consistent hash, format/content, and different params produce different hashes."""
        h1 = generate_packing_hash("multipack", 2048, 100, "fork")
        h2 = generate_packing_hash("multipack", 2048, 100, "fork")
        assert h1 == h2

        parts = h1.split("_")
        assert len(parts) == 4
        assert parts == ["multipack", "2048", "100", "fork"]

        assert h1 != generate_packing_hash("sequential", 2048, 100, "fork")
        assert h1 != generate_packing_hash("multipack", 4096, 100, "fork")
        assert h1 != generate_packing_hash("multipack", 2048, 200, "fork")
        assert h1 != generate_packing_hash("multipack", 2048, 100, "spawn")


class TestGenerateDatasetHashFromConfig:
    """Tests for generate_dataset_hash_from_config function."""

    def test_hash_consistency_uniqueness_and_order_independence(self):
        """Covers consistent MD5 hash, different configs produce different hashes,
        multiple datasets, and order independence."""
        args = Mock()
        args.data.select_columns = None
        config = _make_config()

        # Consistent and valid MD5
        h1 = generate_dataset_hash_from_config(args, [config], "gpt2")
        h2 = generate_dataset_hash_from_config(args, [config], "gpt2")
        assert h1 == h2 and len(h1) == 32
        assert all(c in "0123456789abcdef" for c in h1)

        # Different tokenizer, path, and select_columns produce different hashes
        args_cols = Mock()
        args_cols.data.select_columns = ["col1", "col2"]
        assert h1 != generate_dataset_hash_from_config(args, [config], "llama")
        assert h1 != generate_dataset_hash_from_config(args, [_make_config("dataset2")], "gpt2")
        assert h1 != generate_dataset_hash_from_config(args_cols, [config], "gpt2")

        # Multiple datasets change hash
        config2 = _make_config("dataset2")
        assert h1 != generate_dataset_hash_from_config(args, [config, config2], "gpt2")

        # Order independence
        h_ab = generate_dataset_hash_from_config(args, [config, config2], "gpt2")
        h_ba = generate_dataset_hash_from_config(args, [config2, config], "gpt2")
        assert h_ab == h_ba
