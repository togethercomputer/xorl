from typing import List

from ...arguments import Arguments, DatasetConfig
from .utils import md5
from datasets import Dataset


def generate_split_fingerprints(
    dataset: Dataset, val_set_size: int | float, seed: int
) -> tuple[str, str]:
    """Generate consistent fingerprints for train/test splits."""
    fingerprint = dataset._fingerprint

    train_hash_input = f"{fingerprint}|{val_set_size}|train|{seed}"
    test_hash_input = f"{fingerprint}|{val_set_size}|test|{seed}"

    train_fingerprint = md5(train_hash_input)
    test_fingerprint = md5(test_hash_input)

    return train_fingerprint, test_fingerprint


def generate_packing_hash(
    sample_packing_method: str,
    sample_packing_sequence_len: int,
    sample_packing_group_size: int,
    sample_packing_mp_start_method: str,
) -> str:
    """Generate a hash to uniquely identify a packing configuration."""
    packing_str = f"{sample_packing_method}_{sample_packing_sequence_len}_{sample_packing_group_size}_{sample_packing_mp_start_method}"
    return packing_str

def generate_dataset_hash_from_config(
    args: Arguments, args_datasets: List[DatasetConfig], tokenizer_name: str
) -> str:
    """Generate a hash to uniquely identify a dataset configuration for SFT.

    Args:
        args: Arguments object.
        tokenizer_name: Name of the tokenizer being used.

    Returns:
        MD5 hash string representing the configuration.
    """
    config_str = (
        f"{'|'.join(sorted([f'{d.path}:{d.type}:{d.shards}:{d.shards_idx}:{d.preprocess_shards}:{d.name}:{d.split}:{d.revision}:{d.trust_remote_code}:{d.max_seq_len}:{d.activations_path}' for d in args_datasets]))}"
        f"|{tokenizer_name}"
        f"|{args.data.select_columns}"
    )
    return str(md5(config_str))
