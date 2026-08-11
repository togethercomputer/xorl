import pytest
import torch
from torch.utils.data import Dataset


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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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


@pytest.fixture
def fake_text_dataset():
    """Provides a fake text dataset."""
    return FakeTextDataset(num_samples=100, seq_len=128, vocab_size=1000)


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
