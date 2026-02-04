"""
Shift Tokens Collator - Handles token shifting for causal language modeling.

In causal LM, input_ids[i] predicts labels[i+1]. This collator shifts the data
so that input_ids and labels are aligned for loss computation:
- input_ids = original_tokens[:-1]  (drop last token)
- labels = original_tokens[1:]      (drop first token)

This shifting is required when:
1. Dataset provides unshifted data (HF format): len(input_ids) == len(labels) and labels == input_ids
2. The loss function does NOT perform internal shifting (uses shifted data directly)

This shifting is NOT required when:
1. Dataset already provides shifted data (xorl_client API format): input_ids and target_tokens are already shifted
2. The loss function performs internal shifting (e.g., HuggingFace's CrossEntropyLoss)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import torch

from .base_collator import DataCollator

logger = logging.getLogger(__name__)


@dataclass
class ShiftTokensCollator(DataCollator):
    """
    Collator that shifts tokens for causal language modeling.

    When data is not pre-shifted (HF format where input_ids == labels), this collator:
    - Drops the last token from input_ids
    - Drops the first token from labels (and other per-token fields)

    The collator can auto-detect if shifting is needed based on whether the data
    looks like it's already shifted (xorl_client API format) or not (HF format).

    Args:
        auto_detect: If True (default), automatically detect whether shifting is needed.
                     If False, always apply shifting.
        shift_fields: List of fields to shift along with labels. Default: ["labels"].
                      Other common fields: ["weights", "advantages", "logprobs"].

    Example:
        >>> collator = ShiftTokensCollator()
        >>> # HF format (unshifted): input_ids == labels
        >>> sample = {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]}
        >>> shifted = collator([sample])[0]
        >>> # shifted["input_ids"] == [1, 2, 3, 4]  (dropped last)
        >>> # shifted["labels"] == [2, 3, 4, 5]     (dropped first)
    """

    auto_detect: bool = True
    shift_fields: List[str] = None

    def __post_init__(self):
        if self.shift_fields is None:
            self.shift_fields = ["labels", "weights", "advantages", "logprobs"]

    def _needs_shifting(self, sample: Dict[str, Any]) -> bool:
        """
        Detect if a sample needs shifting.

        Shifting is needed when:
        1. input_ids and labels have the same length, AND
        2. Data doesn't look like it's already shifted (xorl_client format)

        xorl_client format indicators:
        - Has "target_tokens" field (xorl_client API always uses this name)
        - labels[i] == input_ids[i+1] for the first non-IGNORE label

        HF format indicators:
        - labels[i] == input_ids[i] for the first non-IGNORE label (unshifted)
        """
        input_ids = sample.get("input_ids")
        labels = sample.get("labels")

        if input_ids is None or labels is None:
            return False

        # Convert to lists for comparison if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        # If lengths differ, data is already shifted or incompatible
        if len(input_ids) != len(labels):
            return False

        # If "target_tokens" field exists, assume xorl_client format (already shifted)
        if "target_tokens" in sample:
            logger.debug("Detected xorl_client format (target_tokens field present), skipping shift")
            return False

        # Find the first non-IGNORE_INDEX label and check if it matches input_ids[i] or input_ids[i+1]
        IGNORE_INDEX = -100
        for i in range(len(labels) - 1):
            if labels[i] != IGNORE_INDEX:
                # Found first valid label
                if labels[i] == input_ids[i + 1]:
                    # labels[i] == input_ids[i+1] means data is already shifted
                    logger.debug("Data appears to be pre-shifted, skipping shift")
                    return False
                elif labels[i] == input_ids[i]:
                    # labels[i] == input_ids[i] means data is NOT shifted (HF format)
                    logger.debug("Data appears to need shifting (HF format)")
                    return True
                else:
                    # labels[i] doesn't match either - unclear, default to no shift
                    logger.debug("Data format unclear, defaulting to no shift")
                    return False

        # No valid labels found - data has no trainable tokens, no shift needed
        logger.debug("No valid labels found, no shift needed")
        return False

    def _shift_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply shifting to a single sample.

        - input_ids: drop last token
        - labels and other shift_fields: drop first token
        - position_ids: regenerate based on new length (if present)
        """
        shifted = {}

        for key, value in sample.items():
            if key == "input_ids":
                # Drop last token from input_ids
                if isinstance(value, torch.Tensor):
                    shifted[key] = value[:-1]
                elif isinstance(value, list):
                    shifted[key] = value[:-1]
                else:
                    shifted[key] = value
            elif key in self.shift_fields:
                # Drop first token from labels and related fields
                if isinstance(value, torch.Tensor):
                    shifted[key] = value[1:]
                elif isinstance(value, list):
                    shifted[key] = value[1:]
                else:
                    shifted[key] = value
            elif key == "position_ids":
                # Regenerate position_ids based on new input_ids length
                input_ids = sample.get("input_ids")
                if input_ids is not None:
                    new_len = len(input_ids) - 1
                    if isinstance(value, torch.Tensor):
                        shifted[key] = torch.arange(new_len, dtype=value.dtype)
                    else:
                        shifted[key] = list(range(new_len))
                else:
                    shifted[key] = value
            else:
                # Keep other fields unchanged
                shifted[key] = value

        return shifted

    def __call__(
        self, features: Union[Sequence[Dict[str, Any]], List[List[Dict[str, Any]]]]
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Apply shifting to a batch of samples.

        Handles both flat lists of dicts and nested lists (from packed datasets).

        Args:
            features: Either a list of dicts (flat) or list of list of dicts (nested)

        Returns:
            Shifted features in the same structure as input
        """
        if not features:
            return features

        # Handle nested structure: list of list of dicts
        if isinstance(features[0], list):
            shifted_nested = []
            for sample_seqs in features:
                shifted_sample = []
                for seq in sample_seqs:
                    if self.auto_detect:
                        if self._needs_shifting(seq):
                            shifted_sample.append(self._shift_sample(seq))
                        else:
                            shifted_sample.append(seq)
                    else:
                        shifted_sample.append(self._shift_sample(seq))
                shifted_nested.append(shifted_sample)
            return shifted_nested

        # Handle flat structure: list of dicts
        shifted_flat = []
        for sample in features:
            if self.auto_detect:
                if self._needs_shifting(sample):
                    shifted_flat.append(self._shift_sample(sample))
                else:
                    shifted_flat.append(sample)
            else:
                shifted_flat.append(self._shift_sample(sample))

        return shifted_flat
