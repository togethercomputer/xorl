from typing import Any, Dict, Sequence

import pytest
import torch
from torch.utils.data import Dataset


class SimpleCollator:
    """Simple collator for testing that stacks tensors."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in features[0].keys():
            result[key] = torch.stack([f[key] for f in features])
        return result


class FakeTextDataset(Dataset):
    """
    A fake text dataset for testing purposes.

    Returns samples with input_ids, attention_mask, and labels.
    """

    def __init__(self, num_samples: int = 100, seq_len: int = 128, vocab_size: int = 1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Create deterministic but varied data based on index
        torch.manual_seed(idx)

        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class FakePackedDataset(Dataset):
    """
    A fake dataset that returns packed sequences with position_ids.

    Simulates packing multiple sequences together with position IDs.
    """

    def __init__(
        self,
        num_samples: int = 100,
        min_seq_len: int = 64,
        max_seq_len: int = 256,
        vocab_size: int = 1000,
        num_sequences_per_sample: int = 3,
    ):
        self.num_samples = num_samples
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_sequences_per_sample = num_sequences_per_sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(idx)

        # Generate multiple sequences and pack them
        sequences = []
        position_ids = []
        current_pos = 0

        for _ in range(self.num_sequences_per_sample):
            seq_len = torch.randint(self.min_seq_len, self.max_seq_len, (1,)).item()
            seq = torch.randint(1, self.vocab_size, (seq_len,), dtype=torch.long)
            sequences.append(seq)

            # Position IDs start from 0 for each sequence
            pos_ids = torch.arange(seq_len, dtype=torch.long)
            position_ids.append(pos_ids)

        # Concatenate all sequences
        input_ids = torch.cat(sequences)
        position_ids = torch.cat(position_ids)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
        }


@pytest.fixture
def fake_text_dataset():
    """Provides a fake text dataset."""
    return FakeTextDataset(num_samples=100, seq_len=128, vocab_size=1000)


@pytest.fixture
def fake_packed_dataset():
    """Provides a fake packed dataset with position IDs."""
    return FakePackedDataset(
        num_samples=100, min_seq_len=64, max_seq_len=256, vocab_size=1000, num_sequences_per_sample=3
    )


@pytest.fixture
def sample_features():
    """Provides sample features for testing collators."""
    return [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([6, 7, 8, 9, 10], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([6, 7, 8, 9, 10], dtype=torch.long),
        },
    ]


@pytest.fixture
def sample_packed_features():
    """Provides sample packed features with position IDs."""
    return [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([6, 7, 8, 9, 10], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([6, 7, 8, 9, 10], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        },
    ]
