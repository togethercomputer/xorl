import pytest
import torch
from xorl.data.collators import CollatePipeline, DataCollator
from typing import Dict, Sequence, Any

pytestmark = [pytest.mark.cpu, pytest.mark.collator]


class MockCollator1(DataCollator):
    """Mock collator that adds a constant to input_ids."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in features[0].keys():
            if key == "input_ids":
                result[key] = torch.cat([f[key] for f in features]) + 1
            else:
                result[key] = torch.cat([f[key] for f in features])
        return result


class MockCollator2(DataCollator):
    """Mock collator that multiplies input_ids by 2."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if isinstance(features, dict):
            # Already collated
            features["input_ids"] = features["input_ids"] * 2
            return features
        else:
            # Not yet collated
            result = {}
            for key in features[0].keys():
                if key == "input_ids":
                    result[key] = torch.cat([f[key] for f in features]) * 2
                else:
                    result[key] = torch.cat([f[key] for f in features])
            return result


class MockCollator3(DataCollator):
    """Mock collator that adds a new field."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if isinstance(features, dict):
            features["new_field"] = torch.tensor([100])
            return features
        else:
            result = {}
            for key in features[0].keys():
                result[key] = torch.cat([f[key] for f in features])
            result["new_field"] = torch.tensor([100])
            return result


class TestCollatePipeline:
    """Test suite for CollatePipeline."""

    def test_single_collator(self, sample_features):
        """Test pipeline with a single collator."""
        collator = MockCollator1()
        pipeline = CollatePipeline(collator)

        result = pipeline(sample_features)

        # MockCollator1 adds 1 to input_ids
        expected_input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 1
        assert torch.equal(result["input_ids"], expected_input_ids)

    def test_multiple_collators_in_sequence(self, sample_features):
        """Test pipeline with multiple collators applied in sequence."""
        collator1 = MockCollator1()  # adds 1
        collator2 = MockCollator2()  # multiplies by 2

        pipeline = CollatePipeline([collator1, collator2])

        result = pipeline(sample_features)

        # First adds 1, then multiplies by 2
        expected_input_ids = (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 1) * 2
        assert torch.equal(result["input_ids"], expected_input_ids)

    def test_collators_add_new_fields(self, sample_features):
        """Test that collators can add new fields to the batch."""
        collator1 = MockCollator1()
        collator2 = MockCollator3()  # adds new_field

        pipeline = CollatePipeline([collator1, collator2])

        result = pipeline(sample_features)

        assert "new_field" in result
        assert torch.equal(result["new_field"], torch.tensor([100]))

    def test_empty_collator_list(self, sample_features):
        """Test pipeline with empty collator list."""
        pipeline = CollatePipeline([])

        result = pipeline(sample_features)

        # Should return original features unchanged
        assert result == sample_features

    def test_single_collator_as_list(self, sample_features):
        """Test that single collator can be wrapped in a list."""
        collator = MockCollator1()
        pipeline1 = CollatePipeline(collator)
        pipeline2 = CollatePipeline([collator])

        result1 = pipeline1(sample_features)
        result2 = pipeline2(sample_features)

        assert torch.equal(result1["input_ids"], result2["input_ids"])

    def test_preserves_all_keys(self, sample_features):
        """Test that all keys are preserved through the pipeline."""
        collator = MockCollator1()
        pipeline = CollatePipeline([collator])

        result = pipeline(sample_features)

        # Check all original keys are present
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_three_collators_chain(self, sample_features):
        """Test chaining three collators together."""
        pipeline = CollatePipeline([MockCollator1(), MockCollator2(), MockCollator3()])

        result = pipeline(sample_features)

        # (input_ids + 1) * 2
        expected_input_ids = (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 1) * 2
        assert torch.equal(result["input_ids"], expected_input_ids)
        assert "new_field" in result

    def test_with_tuple_of_collators(self, sample_features):
        """Test that tuples of collators work as well as lists."""
        collators_tuple = (MockCollator1(), MockCollator2())
        pipeline = CollatePipeline(collators_tuple)

        result = pipeline(sample_features)

        expected_input_ids = (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 1) * 2
        assert torch.equal(result["input_ids"], expected_input_ids)
