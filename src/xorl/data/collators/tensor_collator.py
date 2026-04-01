"""
Tensor conversion collator for converting lists/arrays to tensors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .base_collator import DataCollator


@dataclass
class ToTensorCollator(DataCollator):
    """
    Converts lists/numpy arrays to torch tensors without batching.

    The collator handles:
    - Numeric lists → tensors
    - Numpy arrays → tensors
    - String lists → kept as-is
    - Already-tensors → kept as-is

    Returns list of dicts with converted tensors, preserving the structure.
    Downstream collators (like PackingConcatCollator) handle batching/concatenation.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert lists/arrays to tensors without batching.

        Args:
            features: List of dicts, where each dict contains lists/arrays or tensors,
                      OR a single dict if already batched by a previous collator,
                      OR list of list of dicts (nested structure from packed datasets)

        Returns:
            List of dicts with tensors converted, OR
            List of list of dicts with tensors converted (for nested structure)
        """
        if not features:
            return {}

        # Handle case where features is already a batched dict (from PackingConcatCollator)
        if isinstance(features, dict):
            # Just ensure all values are tensors
            converted = {}
            for key, value in features.items():
                converted[key] = self._convert_value(value, key)
            return converted

        # Handle nested structure: list of list of dicts (from packed datasets)
        if isinstance(features[0], list):
            # Convert tensors within nested structure but preserve the nesting
            converted_nested = []
            for sample_seqs in features:  # Iterate over batch
                converted_sample = []
                for seq in sample_seqs:  # Iterate over sequences in sample
                    converted_seq = {}
                    for key, value in seq.items():
                        converted_seq[key] = self._convert_value(value, key)
                    converted_sample.append(converted_seq)
                converted_nested.append(converted_sample)
            return converted_nested

        # Convert lists/arrays to tensors (flat list of dicts)
        converted_features = []

        for feature in features:
            converted_feature = {}

            for key, value in feature.items():
                converted_feature[key] = self._convert_value(value, key)

            converted_features.append(converted_feature)

        # Return list of dicts (not batched)
        # PackingConcatCollator will handle the concatenation/batching
        return converted_features

    def _convert_value(self, value: Any, key: str) -> Any:
        """
        Recursively convert a value to tensor if appropriate.

        Args:
            value: The value to convert
            key: The field name (used for dtype inference)

        Returns:
            Converted value (tensor) or original value
        """
        # Already a tensor - keep as-is
        if isinstance(value, torch.Tensor):
            return value

        # Numpy array - convert to tensor
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)

        # List - try to convert recursively
        if isinstance(value, list):
            if not value:
                # Empty list - keep as-is
                return value

            # Try to convert to tensor if it's all numeric
            try:
                # Check if this is a flat numeric list or nested numeric list
                if self._is_numeric_list(value):
                    # Try to convert directly to tensor
                    return torch.tensor(value, dtype=self._infer_dtype(key))
            except (ValueError, TypeError):
                # Not a uniform numeric list, keep as-is or process recursively
                pass

            # If it's a list of dicts, keep as-is (handled by outer collator)
            if isinstance(value[0], dict):
                return value

            # If it's a list of other types (strings, mixed types), keep as-is
            return value

        # Dict - recursively convert values
        if isinstance(value, dict):
            return {k: self._convert_value(v, k) for k, v in value.items()}

        # Scalar numeric value - convert to tensor
        if isinstance(value, (int, float, bool, np.number)):
            return torch.tensor(value)

        # Other types - keep as-is
        return value

    def _is_numeric_list(self, lst: list) -> bool:
        """
        Check if a list (possibly nested) contains only numeric values.

        Args:
            lst: The list to check

        Returns:
            True if all elements are numeric (int, float, bool), False otherwise
        """
        if not lst:
            return False

        first = lst[0]

        # Check if it's a nested list
        if isinstance(first, list):
            # All elements should be lists and all should be numeric
            return all(isinstance(item, list) and self._is_numeric_list(item) for item in lst)

        # Flat list - check if all are numeric
        return all(isinstance(item, (int, float, bool, np.number)) for item in lst)

    def _infer_dtype(self, key: str) -> torch.dtype:
        """
        Infer appropriate dtype based on field name.

        Args:
            key: Field name

        Returns:
            PyTorch dtype
        """
        # These fields should be long (int64) for model compatibility
        if key in ["input_ids", "labels", "attention_mask", "position_ids"]:
            return torch.long

        # Let PyTorch infer for other fields
        return None  # Will use default inference
