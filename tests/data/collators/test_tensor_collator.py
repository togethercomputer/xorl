"""Tests for ToTensorCollator."""

import numpy as np
import pytest
import torch

from xorl.data.collators import ToTensorCollator


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


class TestToTensorCollator:
    """Test suite for ToTensorCollator."""

    def test_type_conversion_and_passthrough(self):
        """Covers list->tensor, numpy->tensor, tensor passthrough, mixed inputs, and boolean lists."""
        collator = ToTensorCollator()

        # Lists to tensors
        result = collator(
            [{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}, {"input_ids": [7, 8, 9], "labels": [10, 11, 12]}]
        )
        assert isinstance(result, list) and len(result) == 2
        assert isinstance(result[0]["input_ids"], torch.Tensor) and result[0]["input_ids"].shape == (3,)
        assert torch.equal(result[0]["input_ids"], torch.tensor([1, 2, 3]))
        assert torch.equal(result[1]["input_ids"], torch.tensor([7, 8, 9]))

        # Numpy arrays to tensors
        result_np = collator([{"input_ids": np.array([1, 2, 3]), "labels": np.array([4, 5, 6])}])
        assert isinstance(result_np[0]["input_ids"], torch.Tensor)

        # Already tensors pass through
        t = torch.tensor([1, 2, 3])
        result_t = collator([{"input_ids": t, "labels": torch.tensor([4, 5, 6])}])
        assert isinstance(result_t[0]["input_ids"], torch.Tensor)
        assert torch.equal(result_t[0]["input_ids"], t)

        # Mixed list and tensor
        result_mix = collator([{"input_ids": [1, 2, 3], "labels": torch.tensor([4, 5, 6])}])
        assert isinstance(result_mix[0]["input_ids"], torch.Tensor)
        assert isinstance(result_mix[0]["labels"], torch.Tensor)

        # Boolean lists
        result_bool = collator([{"input_ids": [1, 2, 3], "mask": [True, False, True]}])
        assert isinstance(result_bool[0]["mask"], torch.Tensor)
        assert result_bool[0]["mask"].dtype == torch.bool

    def test_scalars_empty_and_dtype_inference(self):
        """Covers scalar fields, empty features, dtype inference for known fields, numpy dtypes, and batch_size=1."""
        collator = ToTensorCollator()

        # Scalar fields
        result = collator([{"input_ids": [1, 2, 3], "length": 3, "score": 0.95}])
        assert isinstance(result[0]["length"], torch.Tensor) and result[0]["length"].shape == ()

        # Empty features
        assert collator([]) == {}

        # Dtype inference
        result_dtype = collator(
            [
                {
                    "input_ids": [1, 2, 3],
                    "labels": [4, 5, 6],
                    "position_ids": [0, 1, 2],
                    "attention_mask": [1, 1, 1],
                    "other_field": [1.0, 2.0, 3.0],
                }
            ]
        )
        assert result_dtype[0]["input_ids"].dtype == torch.long
        assert result_dtype[0]["labels"].dtype == torch.long
        assert result_dtype[0]["position_ids"].dtype == torch.long
        assert result_dtype[0]["attention_mask"].dtype == torch.long
        assert result_dtype[0]["other_field"].dtype in [torch.float32, torch.float64]

        # Numpy dtype preservation
        result_np = collator(
            [
                {
                    "input_ids": np.array([1, 2, 3], dtype=np.int32),
                    "embeddings": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                }
            ]
        )
        assert result_np[0]["embeddings"].dtype == torch.float32

        # Batch size one
        result_one = collator([{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}])
        assert result_one[0]["input_ids"].shape == (3,)

    def test_2d_lists_and_string_handling(self):
        """Covers 2D numeric lists, 2D string lists, single strings, and mixed numeric/string fields."""
        collator = ToTensorCollator()

        # 2D numeric lists
        result = collator([{"position_ids": [[0, 1], [2, 3]]}])
        assert isinstance(result[0]["position_ids"], torch.Tensor) and result[0]["position_ids"].shape == (2, 2)

        # 2D string lists kept as-is
        result_str2d = collator([{"text": [["hello", "world"], ["foo", "bar"]]}])
        assert isinstance(result_str2d[0]["text"], list)

        # Single string kept as-is
        result_str = collator([{"input_ids": [1, 2, 3], "text": "hello world"}])
        assert isinstance(result_str[0]["text"], str) and result_str[0]["text"] == "hello world"

        # Mixed numeric and string fields
        result_mixed = collator([{"input_ids": [1, 2, 3], "text": ["hello"], "labels": [4, 5, 6], "source": ["web"]}])
        assert isinstance(result_mixed[0]["input_ids"], torch.Tensor)
        assert isinstance(result_mixed[0]["text"], list)

    def test_different_lengths_and_packed_sequences(self):
        """Covers different length fallback and packed sequence handling."""
        collator = ToTensorCollator()

        # Different lengths fallback
        result = collator(
            [
                {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
                {"input_ids": [7, 8, 9, 10, 11], "labels": [12, 13, 14, 15, 16]},
            ]
        )
        assert isinstance(result, list)
        assert result[0]["input_ids"].shape == (3,)
        assert result[1]["input_ids"].shape == (5,)

        # Packed sequences (same length, should be stacked)
        packed = [
            {
                "input_ids": [1, 2, 3, 101, 4, 5, 6, 102],
                "labels": [-100, -100, 7, -100, -100, 8, 9, -100],
                "position_ids": [0, 1, 2, 3, 0, 1, 2, 3],
                "length": 8,
            },
            {
                "input_ids": [10, 11, 12, 101, 13, 14, 15, 16],
                "labels": [-100, 15, 16, -100, 17, 18, 19, 20],
                "position_ids": [0, 1, 2, 0, 1, 2, 3, 4],
                "length": 8,
            },
        ]
        result_packed = collator(packed)
        assert result_packed[0]["input_ids"].shape == (8,)
        assert result_packed[0]["position_ids"].shape == (8,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
