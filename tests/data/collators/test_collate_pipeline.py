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
            features["input_ids"] = features["input_ids"] * 2
            return features
        else:
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

    def test_single_and_sequential_collators(self, sample_features):
        """Covers single collator, multiple collators in sequence, three-collator chain,
        and adding new fields through pipeline."""
        base_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Single collator
        pipeline1 = CollatePipeline(MockCollator1())
        r1 = pipeline1(sample_features)
        assert torch.equal(r1["input_ids"], base_ids + 1)

        # Two collators: add 1 then multiply by 2
        pipeline2 = CollatePipeline([MockCollator1(), MockCollator2()])
        r2 = pipeline2(sample_features)
        assert torch.equal(r2["input_ids"], (base_ids + 1) * 2)

        # Three collators: add 1, multiply by 2, add new_field
        pipeline3 = CollatePipeline([MockCollator1(), MockCollator2(), MockCollator3()])
        r3 = pipeline3(sample_features)
        assert torch.equal(r3["input_ids"], (base_ids + 1) * 2)
        assert "new_field" in r3 and torch.equal(r3["new_field"], torch.tensor([100]))

    def test_empty_list_tuple_and_key_preservation(self, sample_features):
        """Covers empty collator list, single collator as list vs direct, tuple of collators,
        and key preservation through pipeline."""
        # Empty list returns original features
        assert CollatePipeline([])(sample_features) == sample_features

        # Single collator direct vs wrapped in list produces same result
        c = MockCollator1()
        r_direct = CollatePipeline(c)(sample_features)
        r_list = CollatePipeline([c])(sample_features)
        assert torch.equal(r_direct["input_ids"], r_list["input_ids"])

        # Tuple of collators works
        r_tuple = CollatePipeline((MockCollator1(), MockCollator2()))(sample_features)
        expected = (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 1) * 2
        assert torch.equal(r_tuple["input_ids"], expected)

        # All keys preserved
        r_keys = CollatePipeline([MockCollator1()])(sample_features)
        assert all(k in r_keys for k in ["input_ids", "attention_mask", "labels"])
