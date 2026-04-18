"""Tests for xorl.distributed.utils module."""

import pytest
import torch.nn as nn

from xorl.distributed.utils import (
    check_all_fqn_match,
    check_any_fqn_match,
    check_fqn_match,
    get_module_from_path,
    set_module_from_path,
)


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


class SimpleModel(nn.Module):
    """Simple test model for path-based operations."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.nested = nn.Sequential(nn.Linear(30, 40), nn.ReLU(), nn.Linear(40, 50))


class TestModulePaths:
    """Test set_module_from_path and get_module_from_path together."""

    def test_set_get_roundtrip_and_nested(self):
        """Set/get roundtrip at top-level and nested levels; get nonexistent raises."""
        model = SimpleModel()

        # Top-level set/get roundtrip
        new_layer = nn.Linear(10, 99)
        set_module_from_path(model, "layer1", new_layer)
        assert get_module_from_path(model, "layer1") is new_layer

        # Nested set/get roundtrip
        nested_layer = nn.Linear(30, 100)
        set_module_from_path(model, "nested.0", nested_layer)
        assert get_module_from_path(model, "nested.0") is nested_layer
        assert model.nested[0].out_features == 100

        # Deeply nested via ModuleDict
        model.deep = nn.ModuleDict({"a": nn.Sequential(nn.Linear(5, 10))})
        deep_layer = nn.Linear(5, 20)
        set_module_from_path(model, "deep.a.0", deep_layer)
        assert get_module_from_path(model, "deep.a.0") is deep_layer
        assert model.deep["a"][0].out_features == 20

        # Nonexistent paths raise
        with pytest.raises(AttributeError):
            get_module_from_path(model, "nonexistent")
        with pytest.raises((AttributeError, TypeError)):
            get_module_from_path(model, "layer1.nonexistent")


class TestCheckFqnMatch:
    """Test check_fqn_match: exact, wildcard, partial, and input validation."""

    def test_matching_and_wildcards(self):
        """Exact match, wildcard positions, partial match fails, input validation."""
        # Exact match
        assert check_fqn_match("layer1.weight", "layer1.weight") is not None

        # No match
        assert check_fqn_match("layer1.weight", "layer2.weight") is None

        # Partial match fails
        assert check_fqn_match("layer1", "layer1.weight") is None

        # Wildcard at beginning
        assert check_fqn_match("*.weight", "layer1.weight") is not None

        # Wildcard at end
        assert check_fqn_match("layer1.*", "layer1.weight") is not None

        # Multiple wildcards
        assert check_fqn_match("*.layer*.weight", "model.layer1.weight") is not None

        # Input validation
        with pytest.raises(AssertionError, match="fqn_pattern must be a str"):
            check_fqn_match(["not_a_str"], "fqn")
        with pytest.raises(AssertionError, match="fqn must be a str"):
            check_fqn_match("pattern", ["not_a_str"])


class TestCheckAllFqnMatch:
    """Test check_all_fqn_match: wildcards, failures, edge cases, validation."""

    def test_all_fqn_matching(self):
        """Wildcard match, number mismatch, length mismatch, no match, multi-wildcard, empty, validation."""
        # Matching with wildcards (same number)
        assert check_all_fqn_match(["layer*.weight", "layer*.bias"], ["layer1.weight", "layer1.bias"]) is True

        # Different wildcard numbers -> fail
        assert check_all_fqn_match(["layer*.weight", "layer*.bias"], ["layer1.weight", "layer2.bias"]) is False

        # Length mismatch -> fail
        assert check_all_fqn_match(["layer1.weight", "layer2.bias"], ["layer1.weight"]) is False

        # No match -> fail
        assert check_all_fqn_match(["layer1.weight", "layer2.bias"], ["layer3.weight", "layer4.bias"]) is False

        # Multiple wildcards same number
        assert (
            check_all_fqn_match(
                ["block*.layer*.weight", "block*.layer*.bias"], ["block1.layer1.weight", "block1.layer1.bias"]
            )
            is True
        )

        # Empty lists
        assert check_all_fqn_match([], []) is True

        # Input validation
        with pytest.raises(AssertionError, match="path_patterns must be a list"):
            check_all_fqn_match("not_a_list", ["key1"])
        with pytest.raises(AssertionError, match="path_keys must be a list or tuple"):
            check_all_fqn_match(["pattern1"], "not_a_list")


class TestCheckAnyFqnMatch:
    """Test check_any_fqn_match: exact, wildcard, no match, return_idx, prefix, validation."""

    def test_any_fqn_matching(self):
        """Exact match, wildcard, no match, return_idx, prefix, first-match-wins, validation."""
        patterns = ["layer1.weight", "layer2.bias", "layer3.weight"]

        # Exact match
        assert check_any_fqn_match(patterns, "layer1.weight") is True

        # No match
        assert check_any_fqn_match(patterns, "layer99.weight") is False

        # Wildcard match
        assert check_any_fqn_match(["layer*.weight", "layer*.bias"], "layer5.weight") is True

        # return_idx
        assert check_any_fqn_match(patterns, "layer2.bias", return_idx=True) == 1
        assert check_any_fqn_match(patterns, "layer99.weight", return_idx=True) == -1

        # First match wins
        assert check_any_fqn_match(["layer*.weight", "layer1.weight"], "layer1.weight", return_idx=True) == 0

        # With prefix
        assert check_any_fqn_match(["weight", "bias"], "model.layer1.weight", prefix="model.layer1") is True
        assert check_any_fqn_match(["weight", "bias"], "model.layer1.weight", prefix="model.layer2") is False

        # Input validation
        with pytest.raises(AssertionError, match="path_patterns must be a list"):
            check_any_fqn_match("not_a_list", "key")
        with pytest.raises(AssertionError, match="path_key must be a str"):
            check_any_fqn_match(["pattern"], ["not_a_str"])
