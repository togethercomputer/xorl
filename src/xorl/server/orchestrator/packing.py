"""
Data Processor - Sample Packing Strategies.

This module provides flexible sample packing strategies for converting raw training
samples (datum) into structured micro-batches for distributed training.

## Input Format (Datum)
Each datum is a dictionary with:
    - input_ids: List[int] - Token IDs (REQUIRED)
    - labels: List[int] - Target labels (optional, for training)
    - position_ids: List[int] - Position IDs (optional, auto-generated if missing)
    - other fields are preserved but not used in packing

Example:
    datum = {
        "input_ids": [1, 2, 3, 4, 5],
        "labels": [2, 3, 4, 5, 6],
    }

## Output Format (MicroBatch)
Each micro-batch is a dictionary with:
    - input_ids: List[List[int]] - Packed sequences (one per sample)
    - labels: List[List[int]] - Packed labels (if present in input)
    - position_ids: List[List[int]] - Position IDs for each sequence
    - request_id: str - Request ID to track which request this batch came from
    - batch_id: int - Batch index (for tracking multiple batches from same request)

Example:
    micro_batch = {
        "input_ids": [[1, 2, 3], [4, 5]],
        "labels": [[2, 3, 4], [5, 6]],
        "position_ids": [[0, 1, 2], [0, 1]],
        "request_id": "req-12345",
        "batch_id": 0,
    }

## Usage
    # Using the class-based API (recommended for flexibility)
    packer = SequentialPacker(enable_packing=True)
    micro_batches = packer.pack(datum_list, max_seq_len=2048)

    # Using the function-based API (backward compatible)
    micro_batches = pack_samples(datum_list, max_seq_len=2048, enable_packing=True)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict, Union

import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.distributed.parallel_state import get_parallel_state
from xorl.utils.seqlen_pos_transform_utils import pos2culen


logger = logging.getLogger(__name__)


def apply_weights_to_labels(
    labels: List[int],
    weights: Optional[List[float]],
    sample_idx: int,
) -> List[int]:
    """
    Apply weights mask to labels by setting labels to IGNORE_INDEX where weights=0.

    This converts the xorl_client/tinker `weights` field into the xorl label masking convention:
    - weights=0.0 -> labels=-100 (IGNORE_INDEX, don't compute loss)
    - weights=1.0 -> labels unchanged (compute loss)

    Args:
        labels: List of label token IDs
        weights: Optional list of weights (0.0 or 1.0). If None, returns labels unchanged.
        sample_idx: Sample index for error messages

    Returns:
        Labels with IGNORE_INDEX applied where weights=0

    Raises:
        ValueError: If weights contains values other than 0.0 or 1.0
        ValueError: If weights length doesn't match labels length
    """
    if weights is None:
        return labels

    if len(weights) != len(labels):
        raise ValueError(
            f"Sample {sample_idx}: weights length ({len(weights)}) doesn't match labels length ({len(labels)})"
        )

    # Validate weights are only 0.0 or 1.0
    for i, w in enumerate(weights):
        if w not in (0, 0.0, 1, 1.0):
            raise ValueError(
                f"Sample {sample_idx}: weights[{i}]={w} is not supported. "
                f"Only 0.0 and 1.0 are allowed. "
                f"Weighted loss with arbitrary weights is not currently supported."
            )

    # Apply mask: set labels to IGNORE_INDEX where weights=0
    masked_labels = [IGNORE_INDEX if w == 0 or w == 0.0 else label for label, w in zip(labels, weights)]

    return masked_labels


def apply_advantages_to_labels(
    labels: List[int],
    advantages: Optional[List[float]],
    sample_idx: int,
) -> List[int]:
    """
    Apply advantages mask to labels by setting labels to IGNORE_INDEX where advantages=0.

    For RL training (importance_sampling, policy_loss), advantages=0.0 indicates
    prompt/non-action tokens where we don't want to compute loss. This ensures that
    valid_tokens count (labels != IGNORE_INDEX) matches the action token count.

    Args:
        labels: List of label token IDs
        advantages: Optional list of advantages. If None, returns labels unchanged.
        sample_idx: Sample index for error messages

    Returns:
        Labels with IGNORE_INDEX applied where advantages=0

    Raises:
        ValueError: If advantages length doesn't match labels length
    """
    if advantages is None:
        return labels

    if len(advantages) != len(labels):
        raise ValueError(
            f"Sample {sample_idx}: advantages length ({len(advantages)}) doesn't match labels length ({len(labels)})"
        )

    # Use torch for faster processing with long sequences (e.g., 128k tokens)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

    # Apply mask: set labels to IGNORE_INDEX where advantages=0
    labels_tensor[advantages_tensor == 0.0] = IGNORE_INDEX

    return labels_tensor.tolist()


def unpack_per_token_outputs(
    packed_output: Union[torch.Tensor, List[float]],
    position_ids: Union[torch.Tensor, List[int]],
) -> List[List[float]]:
    """
    Unpack a packed tensor back to per-sample lists using position_ids boundaries.

    Position IDs reset to 0 at each sample boundary, which allows us to identify
    where one sample ends and the next begins.

    Note: packed_output may be shorter than position_ids due to label shifting in
    the loss function. In causal LM, labels are shifted by 1 (labels[..., 1:] is
    aligned with hidden_states[..., :-1]), so per-token outputs have length
    (seq_len - 1) per sample. This function handles this case automatically.

    Args:
        packed_output: Packed tensor of shape [1, total_packed_len] or [total_packed_len],
                       or a list of floats. Contains per-token values (logprobs or losses).
        position_ids: Position IDs tensor of shape [1, total_packed_len] or [total_packed_len],
                      or a list of ints. Position IDs reset to 0 at sample boundaries.

    Returns:
        List of lists, where each inner list contains the per-token values for one sample.
        If the output was shifted (seq_len - 1), each sample's output will have one fewer
        token than its position_ids span.

    Example:
        >>> # Two samples packed: sample1 has 5 tokens, sample2 has 3 tokens
        >>> position_ids = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])  # 8 tokens total
        >>> # Due to shifting, output has 4 + 2 = 6 tokens (each sample loses 1)
        >>> packed_output = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> result = unpack_per_token_outputs(packed_output, position_ids)
        >>> # result = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6]]
    """

    # Convert to tensors if needed
    if isinstance(packed_output, list):
        packed_output = torch.tensor(packed_output)
    if isinstance(position_ids, list):
        position_ids = torch.tensor(position_ids)

    # Flatten tensors
    packed_output = packed_output.flatten()
    position_ids = position_ids.flatten()

    # Get cumulative sequence lengths from position_ids
    # cu_seqlens: [0, len1, len1+len2, ..., total_len]
    cu_seqlens = pos2culen(position_ids)

    # Determine if output is shifted (seq_len - 1 per sample)
    output_len = packed_output.size(0)
    pos_len = position_ids.size(0)
    num_samples = len(cu_seqlens) - 1

    # Calculate expected output length if shifted
    # Each sample loses 1 token, so shifted_len = pos_len - num_samples
    expected_shifted_len = pos_len - num_samples
    is_shifted = output_len == expected_shifted_len

    if output_len != pos_len and not is_shifted:
        logger.warning(
            f"Unexpected output length: output_len={output_len}, pos_len={pos_len}, "
            f"expected_shifted_len={expected_shifted_len}. Using non-shifted unpacking."
        )

    results = []
    output_offset = 0  # Track position in the packed output

    for i in range(num_samples):
        start_pos = cu_seqlens[i].item()
        end_pos = cu_seqlens[i + 1].item()
        sample_len = end_pos - start_pos

        if is_shifted:
            # Output has (seq_len - 1) tokens per sample
            sample_output_len = sample_len - 1
        else:
            # Output has same length as position_ids
            sample_output_len = sample_len

        # Extract this sample's output
        sample_output = packed_output[output_offset : output_offset + sample_output_len]
        results.append(sample_output.cpu().tolist())

        output_offset += sample_output_len

    return results


# ============================================================================
# Type Definitions for Input/Output Contracts
# ============================================================================


class Datum(TypedDict, total=False):
    """
    Input format for a single training sample.

    Required fields:
        - input_ids: List[int]

    Optional fields:
        - labels: List[int]
        - position_ids: List[int]
    """

    input_ids: List[int]  # Required
    labels: List[int]  # Optional
    position_ids: List[int]  # Optional


class MicroBatch(TypedDict, total=False):
    """
    Output format for a packed micro-batch.

    Contains packed sequences with identity tracking.

    Fields:
        - input_ids: Packed input token sequences
        - labels: Packed label sequences (if present)
        - position_ids: Position IDs for each sequence
        - request_id: Request ID to track source request
        - batch_id: Batch index (multiple batches per request)
    """

    input_ids: List[List[int]]
    labels: List[List[int]]
    position_ids: List[List[int]]
    request_id: str
    batch_id: int


# ============================================================================
# Abstract Base Class for Data Processors
# ============================================================================


class Packer(ABC):
    """
    Abstract base class for data packing strategies.

    Subclasses should implement the `pack()` method to define their own
    packing algorithm (sequential, sorted by length, optimal bin packing, etc.)

    Example:
        class MyCustomPacker(Packer):
            def pack(self, datum_list, max_seq_len):
                # Your custom packing logic here
                return micro_batches
    """

    @abstractmethod
    def pack(
        self,
        datum_list: List[Dict[str, Any]],
        max_seq_len: int,
        request_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Pack datum into micro-batches.

        Args:
            datum_list: List of datum (see Datum TypedDict for format)
            max_seq_len: Maximum sequence length per micro-batch
            request_id: Request ID to track source (optional)

        Returns:
            List of micro-batches (see MicroBatch TypedDict for format)
        """
        pass

    def get_name(self) -> str:
        """Return the name of this packing strategy."""
        return self.__class__.__name__


# ============================================================================
# Sequential Packer Implementation
# ============================================================================


class SequentialPacker(Packer):
    """
    Sequential greedy packing strategy.

    Packs samples sequentially into micro-batches using a greedy first-fit
    algorithm. Samples are added to the current batch until it reaches capacity,
    then a new batch is started.

    Args:
        enable_packing: If False, each sample becomes its own micro-batch (no packing)
        log_stats: If True, log detailed packing statistics after packing

    Example:
        >>> packer = SequentialPacker(enable_packing=True)
        >>> datum_list = [
        ...     {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]},
        ...     {"input_ids": [10, 20], "labels": [20, 30]},
        ... ]
        >>> batches = packer.pack(datum_list, max_seq_len=10)
        >>> len(batches)
        1
        >>> batches[0]["num_samples"]
        2
    """

    def __init__(
        self,
        enable_packing: bool = True,
        log_stats: bool = True,
        pad_to_multiple_of: int = 128,
    ):
        self.enable_packing = enable_packing
        self.log_stats = log_stats
        self.pad_to_multiple_of = pad_to_multiple_of

    def pack(
        self,
        datum_list: List[Dict[str, Any]],
        max_seq_len: int,
        request_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Pack samples sequentially into micro-batches with concatenation.

        When packing is enabled, samples are CONCATENATED into a single sequence
        per micro-batch with shape [1, total_packed_length]. Position IDs are
        generated to mark sample boundaries (reset to 0 for each sample).

        This format is compatible with TextSequenceShardCollator for sequence
        parallelism.

        Args:
            datum_list: List of datum dictionaries (see Datum TypedDict)
            max_seq_len: Maximum sequence length per micro-batch
            request_id: Request ID to track source request

        Returns:
            List of micro-batch dictionaries with concatenated sequences
        """
        if not datum_list:
            logger.warning("Empty datum_list provided")
            return []

        logger.debug(
            f"[{self.get_name()}] Packing {len(datum_list)} samples with "
            f"max_seq_len={max_seq_len}, packing={'enabled' if self.enable_packing else 'disabled'}, "
            f"request_id={request_id}"
        )

        # If packing disabled, create one batch per sample (not concatenated)
        if not self.enable_packing:
            return self._pack_without_packing(datum_list, request_id)

        # Pack samples sequentially with CONCATENATION
        # Output format: [1, total_packed_length] with position_ids marking boundaries
        micro_batches = []
        current_batch = self._create_empty_packed_batch(request_id, batch_id=0)
        current_tokens = 0

        # Track skipped samples for better error messages
        skipped_samples = []

        for sample_idx, datum in enumerate(datum_list):
            # Handle nested structure to check for input_ids
            if "input_ids" in datum:
                input_ids = datum["input_ids"]
            elif "model_input" in datum and "input_ids" in datum["model_input"]:
                input_ids = datum["model_input"]["input_ids"]
            else:
                reason = "missing 'input_ids' field"
                skipped_samples.append((sample_idx, reason))
                logger.warning(f"Sample {sample_idx} {reason}, skipping")
                continue

            if not isinstance(input_ids, list):
                input_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)

            seq_len = len(input_ids)

            # Skip samples that exceed max_seq_len
            if seq_len > max_seq_len:
                reason = f"has {seq_len} tokens, exceeding max_seq_len {max_seq_len}"
                skipped_samples.append((sample_idx, reason))
                logger.warning(f"Sample {sample_idx} {reason}. Skipping.")
                continue

            # Skip empty samples
            if seq_len == 0:
                reason = "has empty input_ids (0 tokens)"
                skipped_samples.append((sample_idx, reason))
                logger.warning(f"Sample {sample_idx} {reason}, skipping")
                continue

            # Check if sample fits in current batch
            if current_tokens + seq_len > max_seq_len:
                # Current batch is full, finalize and start a new one
                if current_batch["_num_samples"] > 0:
                    self._finalize_packed_batch(current_batch)
                    micro_batches.append(current_batch)
                current_batch = self._create_empty_packed_batch(request_id, batch_id=len(micro_batches))
                current_tokens = 0

            # Add sample to current batch (concatenate)
            self._add_sample_to_packed_batch(current_batch, datum, sample_idx)
            current_tokens += seq_len

        # Add the last batch if it has samples
        if current_batch["_num_samples"] > 0:
            self._finalize_packed_batch(current_batch)
            micro_batches.append(current_batch)

        # If all samples were skipped, raise a descriptive error
        if not micro_batches and skipped_samples:
            self._raise_all_samples_skipped_error(len(datum_list), skipped_samples, max_seq_len)

        # Log statistics
        if self.log_stats:
            self._log_packing_stats(datum_list, micro_batches, max_seq_len)

        return micro_batches

    def _create_empty_batch(self, request_id: str, batch_id: int) -> Dict[str, Any]:
        """Create an empty micro-batch structure (list of lists format)."""
        return {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "request_id": request_id,
            "batch_id": batch_id,
        }

    def _create_empty_packed_batch(self, request_id: str, batch_id: int) -> Dict[str, Any]:
        """
        Create an empty packed micro-batch structure.

        Packed batches concatenate all samples into single flat lists.
        After finalization, the batch will have shape [1, total_len].
        """
        return {
            "input_ids": [],  # Will be flat list, then wrapped as [[...]]
            "labels": [],
            "position_ids": [],
            "request_id": request_id,
            "batch_id": batch_id,
            "_num_samples": 0,  # Internal counter, removed in finalization
        }

    def _add_sample_to_packed_batch(
        self,
        batch: Dict[str, Any],
        datum: Dict[str, Any],
        sample_idx: int,
    ) -> None:
        """
        Add a sample to a packed micro-batch by CONCATENATING to flat lists.

        Position IDs reset to 0 for each new sample to mark boundaries.

        Token Shifting Logic:
        --------------------
        Detects whether input and target tokens are already shifted (xorl_client API format)
        or need to be shifted (HF format):

        - xorl_client API (already shifted): input_tokens = tokens[:-1], target_tokens = tokens[1:]
          Both have the same length, len(input_ids) == len(target_tokens)

        - HF format (not shifted): input_ids = full sequence, labels = full sequence
          The loss function expects to shift them: input[:-1] predicts labels[1:]

        When NOT shifted, we shift here:
        - input_ids becomes input_ids[:-1] (drop last token)
        - labels becomes labels[1:] (drop first token, which is the prompt start)
        - weights becomes weights[1:] (align with shifted labels)

        The batch stores a '_shifted' flag indicating whether shifting was applied,
        which is used later when unpacking per-token outputs.
        """
        # Handle nested datum structure: flatten model_input and loss_fn_inputs
        flattened_datum = {}
        if "model_input" in datum:
            flattened_datum.update(datum["model_input"])
        if "loss_fn_inputs" in datum:
            flattened_datum.update(datum["loss_fn_inputs"])
        for key, value in datum.items():
            if key not in ["model_input", "loss_fn_inputs"]:
                flattened_datum[key] = value

        if not flattened_datum:
            flattened_datum = datum

        # Extract input_ids
        input_ids = flattened_datum.get("input_ids")
        if input_ids is None:
            raise ValueError(f"Sample {sample_idx} missing 'input_ids'")
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)

        # Extract labels/target_tokens
        if "labels" in flattened_datum:
            labels = flattened_datum["labels"]
            if not isinstance(labels, list):
                labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        elif "target_tokens" in flattened_datum:
            labels = flattened_datum["target_tokens"]
            if not isinstance(labels, list):
                labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        else:
            # No labels - use IGNORE_INDEX for this sample's tokens
            labels = [IGNORE_INDEX] * len(input_ids)

        # Extract weights if present
        weights = flattened_datum.get("weights")
        if weights is not None:
            if not isinstance(weights, list):
                weights = weights.tolist() if hasattr(weights, "tolist") else list(weights)

        # Extract advantages if present (for RL losses like importance_sampling)
        advantages = flattened_datum.get("advantages")
        if advantages is not None:
            if not isinstance(advantages, list):
                advantages = advantages.tolist() if hasattr(advantages, "tolist") else list(advantages)

        # Detect if tokens are already shifted (xorl_client API format)
        # xorl_client format: len(input_ids) == len(target_tokens) and target_tokens field exists
        # HF format: len(input_ids) == len(labels) but need shifting
        is_already_shifted = "target_tokens" in flattened_datum and len(input_ids) == len(labels)

        if not is_already_shifted and len(input_ids) == len(labels):
            # HF format: shift tokens here
            # input_ids[:-1] predicts labels[1:]
            logger.debug(
                f"Sample {sample_idx}: Applying token shifting (HF format detected). "
                f"Original len={len(input_ids)}, shifted len={len(input_ids) - 1}"
            )
            input_ids = input_ids[:-1]
            labels = labels[1:]
            if weights is not None:
                weights = weights[1:]
            if advantages is not None:
                advantages = advantages[1:]
            batch["_shifted"] = True
        else:
            # xorl_client format: already shifted, use as-is
            logger.debug(
                f"Sample {sample_idx}: No shifting needed (xorl_client format detected). "
                f"len(input_ids)={len(input_ids)}, len(labels)={len(labels)}"
            )
            if "_shifted" not in batch:
                batch["_shifted"] = False

        seq_len = len(input_ids)

        # Concatenate input_ids to flat list
        batch["input_ids"].extend(input_ids)

        # Generate position_ids that reset for each sample (marks boundaries)
        position_ids = list(range(seq_len))
        batch["position_ids"].extend(position_ids)

        # Apply weights mask to labels if weights field is present
        # weights=0 -> labels=-100 (IGNORE_INDEX), weights=1 -> labels unchanged
        if weights is not None:
            labels = apply_weights_to_labels(labels, weights, sample_idx)

        # Apply advantages mask to labels if advantages field is present
        # For RL losses, advantages=0 indicates prompt tokens where we don't compute loss
        # advantages=0 -> labels=-100 (IGNORE_INDEX)
        if advantages is not None:
            labels = apply_advantages_to_labels(labels, advantages, sample_idx)

        batch["labels"].extend(labels)

        # Handle other sequence fields (logprobs, advantages, etc.)
        for key, value in flattened_datum.items():
            if key not in ["input_ids", "position_ids", "labels", "weights"]:
                if key not in batch:
                    batch[key] = []
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if isinstance(value, list) and len(value) == seq_len:
                    # Sequence field - concatenate
                    batch[key].extend(value)
                elif not isinstance(value, list) or len(value) != seq_len:
                    # Scalar or non-sequence field - store once per batch
                    # (will be overwritten by last sample, which is OK for metadata)
                    batch[key] = value

        batch["_num_samples"] += 1
        logger.debug(
            f"Added sample {sample_idx} (len={seq_len}) to packed batch {batch['batch_id']}, "
            f"total_len={len(batch['input_ids'])}"
        )

    def _finalize_packed_batch(self, batch: Dict[str, Any]) -> None:
        """
        Finalize a packed batch by wrapping flat lists as [[...]].

        Converts from flat lists to shape [1, total_len] format.

        For non-SP cases (cp_size==1), also adds Flash Attention kwargs and
        masks boundary tokens to prevent attention bleed across packed sequences.
        """
        # Wrap flat lists as single-element lists for batch_size=1
        batch["input_ids"] = [batch["input_ids"]]
        batch["labels"] = [batch["labels"]]
        batch["position_ids"] = [batch["position_ids"]]

        # Store num_samples for reference and remove internal counter
        num_samples = batch.pop("_num_samples")
        batch["num_samples"] = num_samples

        # Wrap any other sequence fields that were concatenated
        for key, value in batch.items():
            if key not in ["input_ids", "labels", "position_ids", "request_id", "batch_id", "num_samples"]:
                if isinstance(value, list) and len(value) == len(batch["input_ids"][0]):
                    batch[key] = [value]

        # Pad to multiple of pad_to_multiple_of (same as PackingConcatCollator in CLI path)
        if self.pad_to_multiple_of > 1:
            seq_len = len(batch["input_ids"][0])
            pad_length = (self.pad_to_multiple_of - seq_len % self.pad_to_multiple_of) % self.pad_to_multiple_of

            if pad_length > 0:
                batch["input_ids"][0].extend([0] * pad_length)
                batch["labels"][0].extend([IGNORE_INDEX] * pad_length)
                # Pad position_ids with sequential values [0, 1, ...] (creates one padding "sequence")
                batch["position_ids"][0].extend(list(range(pad_length)))

                # Pad other sequence fields that were wrapped as [[...]]
                for key, value in batch.items():
                    if key in (
                        "input_ids",
                        "labels",
                        "position_ids",
                        "request_id",
                        "batch_id",
                        "num_samples",
                        "_shifted",
                    ):
                        continue
                    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                        if len(value[0]) == seq_len:
                            value[0].extend([0] * pad_length)

        # For non-SP cases, pre-compute Flash Attention kwargs from position_ids.
        # (SP cases handle this in TextSequenceShardCollator after SP padding.)
        ps = get_parallel_state()
        if ps.cp_size == 1 and num_samples > 1:
            position_ids_tensor = torch.tensor(batch["position_ids"], dtype=torch.long)
            cu_seqlens = pos2culen(position_ids_tensor)

            batch["cu_seq_lens_q"] = cu_seqlens.tolist()
            batch["cu_seq_lens_k"] = cu_seqlens.tolist()

            segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            max_length = int(segment_lengths.max().item())
            batch["max_length_q"] = max_length
            batch["max_length_k"] = max_length

        logger.debug(
            f"Finalized packed batch {batch['batch_id']}: "
            f"total_len={len(batch['input_ids'][0])}, num_samples={num_samples}"
        )

    def _add_sample_to_batch(
        self,
        batch: Dict[str, Any],
        datum: Dict[str, Any],
        sample_idx: int,
    ) -> None:
        """Add a sample to a micro-batch."""
        # Handle nested datum structure: flatten model_input and loss_fn_inputs
        flattened_datum = {}
        if "model_input" in datum:
            flattened_datum.update(datum["model_input"])
        if "loss_fn_inputs" in datum:
            flattened_datum.update(datum["loss_fn_inputs"])
        # Also copy any top-level fields
        for key, value in datum.items():
            if key not in ["model_input", "loss_fn_inputs"]:
                flattened_datum[key] = value

        # If no nested structure, use datum as-is
        if not flattened_datum:
            flattened_datum = datum

        # Extract input_ids
        input_ids = flattened_datum.get("input_ids")
        if input_ids is None:
            raise ValueError(f"Sample {sample_idx} missing 'input_ids'")
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)

        seq_len = len(input_ids)
        batch["input_ids"].append(input_ids)

        # Extract or generate position_ids
        if "position_ids" in flattened_datum:
            position_ids = flattened_datum["position_ids"]
            if not isinstance(position_ids, list):
                position_ids = position_ids.tolist() if hasattr(position_ids, "tolist") else list(position_ids)
        else:
            # Auto-generate position_ids: [0, 1, 2, ..., seq_len-1]
            position_ids = list(range(seq_len))

        batch["position_ids"].append(position_ids)

        # Extract labels if present (or use target_tokens for RL)
        if "labels" in flattened_datum:
            labels = flattened_datum["labels"]
            if not isinstance(labels, list):
                labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        elif "target_tokens" in flattened_datum:
            # For RL datums, use target_tokens as labels
            labels = flattened_datum["target_tokens"]
            if not isinstance(labels, list):
                labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        else:
            # No labels for this sample
            labels = []

        # Apply weights mask to labels if weights field is present
        # weights=0 -> labels=-100 (IGNORE_INDEX), weights=1 -> labels unchanged
        if labels:  # Only apply if we have labels
            weights = flattened_datum.get("weights")
            if weights is not None:
                if not isinstance(weights, list):
                    weights = weights.tolist() if hasattr(weights, "tolist") else list(weights)
                labels = apply_weights_to_labels(labels, weights, sample_idx)

            # Apply advantages mask to labels if advantages field is present
            # For RL losses, advantages=0 indicates prompt tokens where we don't compute loss
            # advantages=0 -> labels=-100 (IGNORE_INDEX)
            advantages = flattened_datum.get("advantages")
            if advantages is not None:
                if not isinstance(advantages, list):
                    advantages = advantages.tolist() if hasattr(advantages, "tolist") else list(advantages)
                labels = apply_advantages_to_labels(labels, advantages, sample_idx)

        batch["labels"].append(labels)

        # Preserve all other fields from loss_fn_inputs (logprobs, advantages, target_tokens, etc.)
        # Note: Keep target_tokens separate even if we used it as labels
        # Note: Exclude weights as it has been applied to labels
        for key, value in flattened_datum.items():
            if key not in ["input_ids", "position_ids", "labels", "weights"]:
                # Initialize list for this field if not present
                if key not in batch:
                    batch[key] = []

                # Append value
                if not isinstance(value, list):
                    value = (
                        value.tolist()
                        if hasattr(value, "tolist")
                        else list(value)
                        if hasattr(value, "__iter__") and not isinstance(value, str)
                        else value
                    )
                batch[key].append(value)

    def _pack_without_packing(
        self,
        datum_list: List[Dict[str, Any]],
        request_id: str,
    ) -> List[Dict[str, Any]]:
        """Create one micro-batch per sample (no packing)."""
        logger.debug("Packing disabled, creating one batch per sample")

        micro_batches = []
        skipped_samples = []

        for batch_id, datum in enumerate(datum_list):
            # Handle nested structure to check for input_ids
            has_input_ids = ("input_ids" in datum) or ("model_input" in datum and "input_ids" in datum["model_input"])
            if not has_input_ids:
                reason = "missing 'input_ids' field"
                skipped_samples.append((batch_id, reason))
                logger.warning(f"Sample {batch_id} {reason}, skipping")
                continue

            batch = self._create_empty_batch(request_id, batch_id)
            self._add_sample_to_batch(batch, datum, batch_id)
            micro_batches.append(batch)

        # If all samples were skipped, raise a descriptive error
        if not micro_batches and skipped_samples:
            self._raise_all_samples_skipped_error(len(datum_list), skipped_samples)

        return micro_batches

    def _log_packing_stats(
        self,
        datum_list: List[Dict[str, Any]],
        micro_batches: List[Dict[str, Any]],
        max_seq_len: int,
    ) -> None:
        """Log packing statistics."""
        if not micro_batches:
            logger.warning("No batches created")
            return

        total_samples = len(datum_list)
        total_batches = len(micro_batches)

        # Calculate statistics from actual data
        # Handle both packed format (list of one list) and unpacked format (list of lists)
        samples_per_batch = []
        tokens_per_batch = []

        for batch in micro_batches:
            input_ids = batch["input_ids"]
            if "num_samples" in batch:
                # Packed format: single concatenated sequence
                samples_per_batch.append(batch["num_samples"])
                tokens_per_batch.append(len(input_ids[0]) if input_ids else 0)
            else:
                # Unpacked format: list of separate sequences
                samples_per_batch.append(len(input_ids))
                tokens_per_batch.append(sum(len(seq) for seq in input_ids))

        total_tokens = sum(tokens_per_batch)
        total_capacity = total_batches * max_seq_len
        utilization = (total_tokens / total_capacity * 100) if total_capacity > 0 else 0

        is_packed = any("num_samples" in batch for batch in micro_batches)

        logger.debug("=" * 70)
        logger.debug(f"[{self.get_name()}] Packing Statistics:")
        logger.debug("-" * 70)
        logger.debug(
            f"  Packing mode:         {'CONCATENATED (batch_size=1)' if is_packed else 'BATCHED (separate sequences)'}"
        )
        logger.debug(f"  Total samples:        {total_samples}")
        logger.debug(f"  Total micro-batches:  {total_batches}")
        logger.debug(f"  Total tokens:         {total_tokens}")
        logger.debug(f"  Capacity per batch:   {max_seq_len}")
        logger.debug(f"  Total capacity:       {total_capacity}")
        logger.debug(f"  Utilization:          {utilization:.1f}%")
        logger.debug("-" * 70)
        logger.debug("  Samples per batch:")
        logger.debug(f"    Min:  {min(samples_per_batch)}")
        logger.debug(f"    Max:  {max(samples_per_batch)}")
        logger.debug(f"    Avg:  {sum(samples_per_batch) / len(samples_per_batch):.1f}")
        logger.debug("-" * 70)
        logger.debug("  Tokens per batch:")
        logger.debug(f"    Min:  {min(tokens_per_batch)}")
        logger.debug(f"    Max:  {max(tokens_per_batch)}")
        logger.debug(f"    Avg:  {sum(tokens_per_batch) / len(tokens_per_batch):.1f}")
        logger.debug("=" * 70)
        logger.info(
            f"[{self.get_name()}] Packed {total_samples} samples into {total_batches} batches ({utilization:.1f}% utilization, {total_tokens} tokens)"
        )

    def _raise_all_samples_skipped_error(
        self,
        total_samples: int,
        skipped_samples: List[tuple],
        max_seq_len: Optional[int] = None,
    ) -> None:
        """
        Raise a descriptive ValueError when all samples were skipped during packing.

        Args:
            total_samples: Total number of samples submitted
            skipped_samples: List of (sample_idx, reason) tuples
            max_seq_len: Maximum sequence length configured (only included if length was checked)
        """
        # Group skipped samples by reason for a cleaner message
        reasons_count = {}
        for idx, reason in skipped_samples:
            if reason not in reasons_count:
                reasons_count[reason] = []
            reasons_count[reason].append(idx)

        # Build detailed error message
        error_lines = [
            f"All {total_samples} samples were skipped - no valid batches could be created.",
        ]

        if max_seq_len is not None:
            error_lines.append(f"Server max_seq_len is {max_seq_len} tokens.")

        error_lines.append("")
        error_lines.append("Skipped samples by reason:")

        for reason, indices in reasons_count.items():
            if len(indices) <= 5:
                indices_str = ", ".join(str(i) for i in indices)
            else:
                indices_str = f"{', '.join(str(i) for i in indices[:5])}, ... ({len(indices)} total)"
            error_lines.append(f"  - {reason}: sample(s) {indices_str}")

        error_lines.append("")
        error_lines.append("Please check your input data and ensure:")
        error_lines.append("  1. Each sample has an 'input_ids' field (list of token IDs)")
        if max_seq_len is not None:
            error_lines.append(f"  2. Sequence lengths do not exceed {max_seq_len} tokens")
            error_lines.append("  3. Samples are not empty (at least 1 token)")

        error_msg = "\n".join(error_lines)
        raise ValueError(error_msg)


# ============================================================================
# Function-based API (backward compatible)
# ============================================================================


def pack_samples(
    datum_list: List[Dict[str, Any]],
    max_seq_len: int = 2048,
    enable_packing: bool = True,
    request_id: str = "",
    pad_to_multiple_of: int = 128,
) -> List[Dict[str, Any]]:
    """
    Pack samples into micro-batches using sequential greedy algorithm.

    This is a convenience function that uses SequentialPacker internally.
    For more control and flexibility, use the class-based API directly.

    Args:
        datum_list: List of datum dictionaries, each containing:
            - input_ids: List[int] - Token IDs (required)
            - labels: List[int] - Target labels (optional, for training)
            - position_ids: List[int] - Position IDs (optional, auto-generated if missing)
        max_seq_len: Maximum sequence length per micro-batch
        enable_packing: If False, each sample becomes its own micro-batch
        request_id: Request ID to track source request
        pad_to_multiple_of: Pad sequence length to be divisible by this value.
            Should account for both hardware alignment (e.g., 128) and sequence
            parallel size (cp_size). Use math.lcm(base_pad, cp_size) to combine.

    Returns:
        List of micro-batch dictionaries (see MicroBatch TypedDict for format)

    Example:
        >>> datum_list = [
        ...     {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]},
        ...     {"input_ids": [10, 20], "labels": [20, 30]},
        ... ]
        >>> batches = pack_samples(datum_list, max_seq_len=10, request_id="req-123")
        >>> # Both samples fit in one batch (4 + 2 = 6 tokens < 10)
        >>> len(batches)
        1
        >>> len(batches[0]["input_ids"])
        2
    """
    packer = SequentialPacker(enable_packing=enable_packing, log_stats=True, pad_to_multiple_of=pad_to_multiple_of)
    return packer.pack(datum_list, max_seq_len, request_id)


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_micro_batches(micro_batches: List[Dict[str, Any]]) -> bool:
    """
    Validate micro-batch structure.

    Supports both packed format (batch_size=1, concatenated) and
    unpacked format (batch_size>1, separate sequences).

    Checks:
    - All required fields are present
    - Lengths are consistent (input_ids, labels, position_ids)
    - batch_id and request_id are set
    - At least one valid token exists (not all labels are -100)

    Args:
        micro_batches: List of micro-batch dictionaries

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["input_ids", "labels", "position_ids", "request_id", "batch_id"]
    total_valid_tokens = 0

    for batch_idx, batch in enumerate(micro_batches):
        # Check required fields
        for field in required_fields:
            if field not in batch:
                logger.error(f"Batch {batch_idx} missing required field: {field}")
                return False

        # Check that input_ids is a list
        if not isinstance(batch["input_ids"], list):
            logger.error(f"Batch {batch_idx}: input_ids should be a list")
            return False

        num_sequences = len(batch["input_ids"])
        if num_sequences == 0:
            logger.error(f"Batch {batch_idx}: input_ids is empty")
            return False

        # Check if this is packed format (list of one list) or unpacked (list of multiple lists)
        is_packed = "num_samples" in batch

        # Check lengths are consistent
        if len(batch["position_ids"]) != num_sequences:
            logger.error(
                f"Batch {batch_idx}: position_ids length mismatch ({len(batch['position_ids'])} != {num_sequences})"
            )
            return False

        if len(batch["labels"]) != num_sequences:
            logger.error(f"Batch {batch_idx}: labels length mismatch ({len(batch['labels'])} != {num_sequences})")
            return False

        # Check each sequence has matching lengths
        for i, input_seq in enumerate(batch["input_ids"]):
            if not isinstance(input_seq, list):
                logger.error(f"Batch {batch_idx}, sequence {i}: input_ids should be a list of lists")
                return False

            pos_seq = batch["position_ids"][i]
            if len(input_seq) != len(pos_seq):
                logger.error(
                    f"Batch {batch_idx}, sequence {i}: input_ids length ({len(input_seq)}) "
                    f"!= position_ids length ({len(pos_seq)})"
                )
                return False

            labels_seq = batch["labels"][i]
            if len(input_seq) != len(labels_seq):
                logger.error(
                    f"Batch {batch_idx}, sequence {i}: input_ids length ({len(input_seq)}) "
                    f"!= labels length ({len(labels_seq)})"
                )
                return False

        # Check batch_id is an integer
        if not isinstance(batch["batch_id"], int):
            logger.error(f"Batch {batch_idx}: batch_id should be an integer")
            return False

        # Count valid tokens (labels that are not IGNORE_INDEX)
        for seq_idx, labels in enumerate(batch["labels"]):
            if labels:  # Check if labels list is not empty
                valid_count = sum(1 for label in labels if label != IGNORE_INDEX)
                total_valid_tokens += valid_count

    # Check that we have at least one valid token across all batches
    if total_valid_tokens == 0:
        logger.error(
            "All labels across all batches are -100 (IGNORE_INDEX). "
            "No valid tokens to train on. This will cause NaN loss from division by zero. "
            "Check your data preparation: ensure labels are properly set and not all masked."
        )
        return False

    # Log format info
    is_packed = any("num_samples" in batch for batch in micro_batches)
    logger.debug(
        f"Validated {len(micro_batches)} micro-batches successfully "
        f"({total_valid_tokens} valid tokens, format={'packed' if is_packed else 'unpacked'})"
    )
    return True
