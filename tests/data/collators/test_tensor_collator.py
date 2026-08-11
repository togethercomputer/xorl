"""Tests for ToTensorCollator."""

import numpy as np
import pytest
import torch

from xorl.data.collators import ToTensorCollator


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


def test_to_tensor_collator_preserves_its_pipeline_shapes_and_types():
    collator = ToTensorCollator()
    labels = torch.tensor([4, 5, 6])

    [sample] = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": labels,
                "embeddings": np.array([0.5, 1.5, 2.5], dtype=np.float32),
                "text": ["source text"],
                "length": 3,
            }
        ]
    )
    assert torch.equal(sample["input_ids"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert sample["labels"] is labels
    assert sample["embeddings"].dtype == torch.float32
    assert sample["text"] == ["source text"]
    assert sample["length"].shape == ()

    batched = collator({"input_ids": [[1, 2], [3, 4]], "attention_mask": [[1, 1], [1, 0]]})
    assert batched["input_ids"].shape == (2, 2)
    assert batched["attention_mask"].dtype == torch.long

    nested = collator(
        [
            [{"input_ids": [1, 2], "labels": [2, 3]}],
            [{"input_ids": [4], "labels": [5]}],
        ]
    )
    assert nested[0][0]["input_ids"].shape == (2,)
    assert nested[1][0]["labels"].dtype == torch.long
    assert collator([]) == {}
