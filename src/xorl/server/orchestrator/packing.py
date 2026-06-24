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

import heapq
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict, Union

import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.distributed.parallel_state import get_parallel_state
from xorl.utils.seqlen_pos_transform_utils import pos2culen


logger = logging.getLogger(__name__)


OPD_TOKEN_ALIGNED_FIELDS = (
    "teacher_ids",
    "teacher_cache_indices",
    "teacher_weights",
    "hidden_match_weights",
    "teacher_hidden_states",
    # Metrics-only KL-split diagnostics (region ids / sample correctness).
    "opd_region_ids",
    "opd_sample_ok",
)

# Packing strategies (see SequentialPacker for semantics).
PACKING_STRATEGIES = ("sequential", "best_fit", "balanced_dp")
# How to treat samples whose raw length exceeds max_seq_len.
ON_OVERSIZED_MODES = ("error", "skip", "truncate")


class _MeasuredSample(NamedTuple):
    """A validated, length-measured sample awaiting bin assignment.

    ``orig_idx`` is the index into the original ``datum_list`` (used to keep
    per-datum side arrays such as ``routed_experts`` aligned after reordering).
    ``seq_len`` is the *raw* ``input_ids`` length, matching the capacity
    accounting the legacy greedy packer used (the HF-shift drops one token at
    build time, but capacity is checked pre-shift to preserve behavior).
    """

    orig_idx: int
    datum: Dict[str, Any]
    seq_len: int


@dataclass
class PackingResult:
    """Output of the bin-assignment phase, before micro-batches are built."""

    bins: List[List[_MeasuredSample]]
    # Original datum indices in batch-flattened (build) order. Lets callers
    # reorder per-datum side arrays (routed_experts/logits) to match the order
    # samples actually appear in the emitted micro-batches.
    datum_order: List[int]
    dropped: List[Tuple[int, str]]


def shift_opd_token_aligned_fields(
    flattened_datum: Dict[str, Any],
    original_seq_len: int,
    shifted_seq_len: int,
    sample_idx: int,
) -> None:
    """Trim OPD per-token fields when HF-style causal shifting trims input_ids.

    OPD compares the student's hidden state at each retained input position with
    the teacher hidden state for the same context position. Therefore fields that
    index or carry teacher context activations drop the final token when
    input_ids becomes input_ids[:-1].
    """
    for key in OPD_TOKEN_ALIGNED_FIELDS:
        if key not in flattened_datum:
            continue
        value = flattened_datum[key]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            continue
        if len(value) == shifted_seq_len:
            flattened_datum[key] = value
            continue
        if len(value) != original_seq_len:
            raise ValueError(
                f"Sample {sample_idx}: {key} length ({len(value)}) must match either shifted length "
                f"({shifted_seq_len}) or original length ({original_seq_len})"
            )
        flattened_datum[key] = value[:-1]


def _resolve_teacher_cache_base(t_base: Any, t_cache: List[int]) -> int:
    """Resolve the per-sample OPRD re-base anchor.

    The client sends ``teacher_cache_base`` as a 1-element list (the API schema's
    loss_fn_inputs InputType has no scalar form); accept a bare scalar too.
    Legacy clients omit it — fall back to ``min()`` inference, which is only
    correct for unmasked contiguous slices (masked variants 0-fill, so min()==0;
    see docs/notes/oprd_warm_cache_indices_rebase_bug.md).
    """
    if t_base is not None:
        if hasattr(t_base, "tolist"):
            t_base = t_base.tolist()
        if isinstance(t_base, (list, tuple)):
            t_base = t_base[0] if t_base else None
    if t_base is not None:
        return int(t_base)
    return min(t_cache) if t_cache else 0


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
    Configurable bin-packing strategy for server training micro-batches.

    The packer runs in three phases:

    1. **Measure / filter** — extract each sample's raw ``input_ids`` length and
       apply the oversized policy (``on_oversized``). The legacy "silently drop
       samples longer than ``max_seq_len``" behavior is now opt-in
       (``on_oversized="skip"``); the default raises a descriptive error so a
       misconfigured ``pack_len`` fails loud instead of dropping training data.
    2. **Assign** — partition the surviving samples into bins (rows) according to
       ``strategy`` (see below). No bin exceeds ``max_seq_len``.
    3. **Build** — concatenate each bin into a packed ``[1, total_len]`` micro-batch
       (token shifting, position-id resets, OPD field expansion, padding). This
       phase is shared by every strategy, so only *which sample lands in which
       row* changes — never the per-token contents — keeping numerics stable.

    Strategies:
        - ``"sequential"`` (default): greedy first-fit in arrival order. Bit-for-bit
          identical to the historical packer.
        - ``"best_fit"``: sort longest-first, then best-fit-decreasing. Raises
          capacity utilization and reduces row count for the same ``max_seq_len``.
        - ``"balanced_dp"``: longest-processing-time partition into ``N = k·dp_size``
          balanced bins so the dispatcher needs zero dummy batches and every rank
          does equal real work. Requires ``dp_size`` (the number of distinct batch
          slices the dispatcher will produce). NOTE: forcing rows to be small enough
          for ``N == dp_size`` only improves throughput when the batch is large
          enough that rows stay at/above the GEMM knee — see the redesign spec.

    Args:
        enable_packing: If False, each sample becomes its own micro-batch (no packing).
        log_stats: If True, log packing statistics after packing.
        pad_to_multiple_of: Pad each packed row's length to a multiple of this.
        strategy: One of ``PACKING_STRATEGIES``.
        on_oversized: One of ``ON_OVERSIZED_MODES`` — how to treat samples longer
            than ``max_seq_len``. ``"error"`` (default) raises; ``"skip"`` drops with
            a warning (legacy behavior); ``"truncate"`` clips the sample to
            ``max_seq_len`` (and trims token-aligned fields).
        dp_size: Number of distinct batch slices the dispatcher produces
            (world_size // (cp_size·pp_size)). Used by ``"balanced_dp"``.

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
        strategy: str = "sequential",
        on_oversized: str = "error",
        dp_size: int = 1,
    ):
        if strategy not in PACKING_STRATEGIES:
            raise ValueError(f"Unknown packing strategy {strategy!r}; expected one of {PACKING_STRATEGIES}")
        if on_oversized not in ON_OVERSIZED_MODES:
            raise ValueError(f"Unknown on_oversized mode {on_oversized!r}; expected one of {ON_OVERSIZED_MODES}")
        self.enable_packing = enable_packing
        self.log_stats = log_stats
        self.pad_to_multiple_of = pad_to_multiple_of
        self.strategy = strategy
        self.on_oversized = on_oversized
        self.dp_size = max(1, int(dp_size))
        # Set after each pack(): original datum indices in batch-flattened order.
        self.last_datum_order: List[int] = []

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
        self.last_datum_order = []
        if not datum_list:
            logger.warning("Empty datum_list provided")
            return []

        logger.debug(
            f"[{self.get_name()}] Packing {len(datum_list)} samples with "
            f"max_seq_len={max_seq_len}, strategy={self.strategy}, on_oversized={self.on_oversized}, "
            f"dp_size={self.dp_size}, packing={'enabled' if self.enable_packing else 'disabled'}, "
            f"request_id={request_id}"
        )

        # If packing disabled, create one batch per sample (not concatenated)
        if not self.enable_packing:
            self.last_datum_order = [i for i, d in enumerate(datum_list) if self._has_input_ids(d)]
            return self._pack_without_packing(datum_list, request_id)

        # Phase 1: measure + filter (apply oversized policy).
        measured, dropped = self._measure_and_filter(datum_list, max_seq_len)

        # If everything was filtered out, raise a descriptive error.
        if not measured:
            self._raise_all_samples_skipped_error(len(datum_list), dropped, max_seq_len)

        # Phase 2: assign surviving samples to bins per the configured strategy.
        bins = self._assign_bins(measured, max_seq_len)

        # Phase 3: build a packed micro-batch per (non-empty) bin.
        micro_batches: List[Dict[str, Any]] = []
        datum_order: List[int] = []
        for bin_samples in bins:
            if not bin_samples:
                continue
            current_batch = self._create_empty_packed_batch(request_id, batch_id=len(micro_batches))
            for sample in bin_samples:
                self._add_sample_to_packed_batch(current_batch, sample.datum, sample.orig_idx)
                datum_order.append(sample.orig_idx)
            self._finalize_packed_batch(current_batch)
            micro_batches.append(current_batch)

        self.last_datum_order = datum_order

        # Log statistics
        if self.log_stats:
            self._log_packing_stats(datum_list, micro_batches, max_seq_len)

        return micro_batches

    @staticmethod
    def _has_input_ids(datum: Dict[str, Any]) -> bool:
        return ("input_ids" in datum) or (
            isinstance(datum.get("model_input"), dict) and "input_ids" in datum["model_input"]
        )

    @staticmethod
    def _extract_input_ids(datum: Dict[str, Any]) -> Optional[List[int]]:
        """Extract raw input_ids from a (possibly nested) datum, or None if absent."""
        if "input_ids" in datum:
            input_ids = datum["input_ids"]
        elif isinstance(datum.get("model_input"), dict) and "input_ids" in datum["model_input"]:
            input_ids = datum["model_input"]["input_ids"]
        else:
            return None
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        return input_ids

    def _measure_and_filter(
        self,
        datum_list: List[Dict[str, Any]],
        max_seq_len: int,
    ) -> Tuple[List[_MeasuredSample], List[Tuple[int, str]]]:
        """Validate samples, apply the oversized policy, and measure raw lengths."""
        measured: List[_MeasuredSample] = []
        dropped: List[Tuple[int, str]] = []

        for sample_idx, datum in enumerate(datum_list):
            input_ids = self._extract_input_ids(datum)
            if input_ids is None:
                reason = "missing 'input_ids' field"
                dropped.append((sample_idx, reason))
                logger.warning(f"Sample {sample_idx} {reason}, skipping")
                continue

            seq_len = len(input_ids)

            if seq_len == 0:
                reason = "has empty input_ids (0 tokens)"
                dropped.append((sample_idx, reason))
                logger.warning(f"Sample {sample_idx} {reason}, skipping")
                continue

            if seq_len > max_seq_len:
                if self.on_oversized == "error":
                    raise ValueError(
                        f"Sample {sample_idx} has {seq_len} tokens, exceeding max_seq_len {max_seq_len}. "
                        f"Refusing to silently drop training data. Increase sample_packing_sequence_len, "
                        f"or set on_oversized='truncate'/'skip' to allow clipping/dropping."
                    )
                if self.on_oversized == "skip":
                    reason = f"has {seq_len} tokens, exceeding max_seq_len {max_seq_len}"
                    dropped.append((sample_idx, reason))
                    logger.warning(f"Sample {sample_idx} {reason}. Skipping.")
                    continue
                # truncate
                datum = self._truncate_datum(datum, seq_len, max_seq_len, sample_idx)
                seq_len = max_seq_len
                logger.warning(
                    f"Sample {sample_idx} truncated from original length to max_seq_len {max_seq_len}."
                )

            measured.append(_MeasuredSample(orig_idx=sample_idx, datum=datum, seq_len=seq_len))

        return measured, dropped

    @staticmethod
    def _truncate_datum(
        datum: Dict[str, Any],
        original_len: int,
        max_seq_len: int,
        sample_idx: int,
    ) -> Dict[str, Any]:
        """Return a shallow copy of ``datum`` with token-aligned fields clipped to ``max_seq_len``.

        Any top-level (or nested model_input/loss_fn_inputs) list/array field whose
        length equals the raw input length is treated as token-aligned and clipped.
        Scalar metadata is preserved untouched.
        """

        def _clip(value: Any) -> Any:
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list) and len(value) == original_len:
                return value[:max_seq_len]
            return value

        def _clip_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
            return {k: _clip(v) for k, v in mapping.items()}

        truncated = dict(datum)
        if isinstance(datum.get("model_input"), dict):
            truncated["model_input"] = _clip_mapping(datum["model_input"])
        if isinstance(datum.get("loss_fn_inputs"), dict):
            truncated["loss_fn_inputs"] = _clip_mapping(datum["loss_fn_inputs"])
        for key, value in datum.items():
            if key in ("model_input", "loss_fn_inputs"):
                continue
            truncated[key] = _clip(value)
        return truncated

    def _assign_bins(
        self,
        measured: List[_MeasuredSample],
        max_seq_len: int,
    ) -> List[List[_MeasuredSample]]:
        """Partition measured samples into bins (rows) per the configured strategy."""
        if self.strategy == "sequential":
            return self._assign_sequential(measured, max_seq_len)
        if self.strategy == "best_fit":
            return self._assign_best_fit(measured, max_seq_len)
        if self.strategy == "balanced_dp":
            return self._assign_balanced_dp(measured, max_seq_len)
        raise ValueError(f"Unknown packing strategy {self.strategy!r}")

    @staticmethod
    def _assign_sequential(
        measured: List[_MeasuredSample],
        max_seq_len: int,
    ) -> List[List[_MeasuredSample]]:
        """Greedy first-fit in arrival order — bit-for-bit the legacy packer."""
        bins: List[List[_MeasuredSample]] = []
        current: List[_MeasuredSample] = []
        current_tokens = 0
        for sample in measured:
            if current and current_tokens + sample.seq_len > max_seq_len:
                bins.append(current)
                current = []
                current_tokens = 0
            current.append(sample)
            current_tokens += sample.seq_len
        if current:
            bins.append(current)
        return bins

    @staticmethod
    def _assign_best_fit(
        measured: List[_MeasuredSample],
        max_seq_len: int,
    ) -> List[List[_MeasuredSample]]:
        """Best-fit-decreasing: sort longest-first, place each in the tightest fitting bin.

        Minimizes row count / maximizes utilization for a fixed ``max_seq_len``.
        Deterministic: stable longest-first order, and samples within a bin are
        restored to original arrival order.
        """
        order = sorted(measured, key=lambda m: (-m.seq_len, m.orig_idx))
        bin_items: List[List[_MeasuredSample]] = []
        bin_loads: List[int] = []
        for sample in order:
            best_idx = -1
            best_load = -1
            for bidx, load in enumerate(bin_loads):
                if load + sample.seq_len <= max_seq_len and load > best_load:
                    best_idx = bidx
                    best_load = load
            if best_idx < 0:
                bin_items.append([sample])
                bin_loads.append(sample.seq_len)
            else:
                bin_items[best_idx].append(sample)
                bin_loads[best_idx] += sample.seq_len
        return [sorted(items, key=lambda m: m.orig_idx) for items in bin_items]

    def _assign_balanced_dp(
        self,
        measured: List[_MeasuredSample],
        max_seq_len: int,
    ) -> List[List[_MeasuredSample]]:
        """LPT partition into ``N = k·dp_size`` balanced bins (zero dispatcher dummies).

        Picks the smallest ``k ≥ 1`` such that (a) ``N`` non-empty bins exist
        (``N ≤ num_samples``), (b) capacity allows ``N`` bins, and (c) a valid
        longest-processing-time packing keeps every bin ≤ ``max_seq_len``. If
        ``num_samples < dp_size`` we cannot create ``dp_size`` non-empty bins, so
        we fall back to best-fit (the dispatcher still pads with dummies) and warn.
        """
        dp_size = self.dp_size
        n_samples = len(measured)
        total_tokens = sum(m.seq_len for m in measured)

        if dp_size <= 1:
            # No DP fan-out to balance against; best-fit gives the fullest rows.
            return self._assign_best_fit(measured, max_seq_len)

        if n_samples < dp_size:
            logger.warning(
                f"balanced_dp: only {n_samples} samples for dp_size={dp_size}; cannot fill every rank "
                f"with real work — falling back to best_fit (dispatcher will pad with dummy batches)."
            )
            return self._assign_best_fit(measured, max_seq_len)

        # Smallest k whose capacity can hold all tokens; never more bins than samples.
        k_min = max(1, math.ceil(total_tokens / (dp_size * max_seq_len)))
        max_k = n_samples // dp_size
        for k in range(k_min, max_k + 1):
            n_bins = k * dp_size
            packed = self._lpt_into_bins(measured, n_bins, max_seq_len)
            if packed is not None:
                return packed

        # Capacity could not be satisfied with any clean multiple of dp_size
        # (would only happen with adversarial length distributions). Fall back to
        # best-fit so we still emit valid, capacity-respecting rows.
        logger.warning(
            "balanced_dp: could not find a k·dp_size partition within max_seq_len; falling back to best_fit."
        )
        return self._assign_best_fit(measured, max_seq_len)

    @staticmethod
    def _lpt_into_bins(
        measured: List[_MeasuredSample],
        n_bins: int,
        max_seq_len: int,
    ) -> Optional[List[List[_MeasuredSample]]]:
        """Longest-processing-time partition into exactly ``n_bins`` bins.

        Returns ``None`` if the emptiest bin cannot hold the next-largest sample
        (i.e. ``n_bins`` is too few for ``max_seq_len``). On success, every bin is
        non-empty (guaranteed when ``n_bins ≤ len(measured)``) and ≤ ``max_seq_len``.
        Deterministic via a (load, bin_idx) min-heap and stable longest-first order.
        """
        order = sorted(measured, key=lambda m: (-m.seq_len, m.orig_idx))
        bin_items: List[List[_MeasuredSample]] = [[] for _ in range(n_bins)]
        # heap of (load, bin_idx): always assign to the currently-emptiest bin.
        heap: List[Tuple[int, int]] = [(0, i) for i in range(n_bins)]
        heapq.heapify(heap)
        for sample in order:
            load, bidx = heapq.heappop(heap)
            if load + sample.seq_len > max_seq_len:
                # Emptiest bin can't fit the largest remaining sample => need more bins.
                return None
            bin_items[bidx].append(sample)
            heapq.heappush(heap, (load + sample.seq_len, bidx))
        # Restore original arrival order within each bin for deterministic builds.
        return [sorted(items, key=lambda m: m.orig_idx) for items in bin_items]

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
            "_r3_sample_lengths": [],
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
            shift_opd_token_aligned_fields(
                flattened_datum,
                original_seq_len=len(input_ids),
                shifted_seq_len=len(input_ids) - 1,
                sample_idx=sample_idx,
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
        batch["_r3_sample_lengths"].append(seq_len)

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

        # OPRD trainer-side teacher forward: teacher_input_ids/teacher_kept_indices form
        # a SEPARATE (teacher-length) sequence and teacher_cache_indices is a per-student-
        # token remap INTO the teacher's kept rows. Under packing these need CUMULATIVE
        # offsets that the generic concat path (below) cannot give:
        #   * teacher_input_ids / teacher_kept_indices have len != student seq_len, so the
        #     generic path would OVERWRITE them, keeping only the LAST packed sample;
        #   * teacher_cache_indices (len == seq_len) would be concatenated WITHOUT the
        #     cumulative kept-row offset, so later samples' remaps would alias sample 0.
        # Handle them explicitly with the right offsets (kept tensor and teacher seq are
        # concatenated in sample order in the trainer), then exclude from the generic loop.
        t_in = flattened_datum.get("teacher_input_ids")
        if t_in is not None:
            if hasattr(t_in, "tolist"):
                t_in = t_in.tolist()
            t_kept = flattened_datum.get("teacher_kept_indices")
            if t_kept is None:
                raise ValueError(f"Sample {sample_idx}: teacher_input_ids requires teacher_kept_indices for OPRD packing")
            if hasattr(t_kept, "tolist"):
                t_kept = t_kept.tolist()
            cum_teacher = batch.get("_oprd_cum_teacher_len", 0)
            cum_kept = batch.get("_oprd_cum_num_kept", 0)
            # Raw teacher token ids concatenated; kept indices offset into the
            # concatenated teacher sequence; per-sample teacher position_ids reset to 0
            # so the trainer can build block-diagonal (per-sample) teacher attention.
            batch.setdefault("teacher_input_ids", []).extend(int(x) for x in t_in)
            batch.setdefault("teacher_kept_indices", []).extend(int(k) + cum_teacher for k in t_kept)
            batch.setdefault("teacher_position_ids", []).extend(range(len(t_in)))
            # teacher_cache_indices (already trimmed to the shifted student length by
            # shift_opd_token_aligned_fields above) carries GLOBAL teacher-cache rows
            # with TWO consumers that need different views (see
            # docs/notes/oprd_warm_cache_indices_rebase_bug.md):
            #   * the KL hidden-fetch indexes the GLOBAL teacher cache — it needs the
            #     rows passed through UNCHANGED;
            #   * the trainer-forward OPRD gather indexes the PER-MICRO-BATCH kept-row
            #     tensor — it needs each sample re-based to local [0, num_kept_sample)
            #     plus the cumulative kept-row offset.
            # Emit BOTH: teacher_cache_indices stays global, and
            # teacher_cache_local_indices carries the re-based view. The re-base uses
            # the client's explicit per-sample base (teacher_cache_base = the sample's
            # first cache row); min()-inference is the legacy fallback and is WRONG for
            # the masked client variants (their masked positions are 0-filled, so
            # min()==0 — the ARITH-014 step-0 OOB).
            t_cache = flattened_datum.get("teacher_cache_indices")
            if t_cache is not None:
                if hasattr(t_cache, "tolist"):
                    t_cache = t_cache.tolist()
                base = _resolve_teacher_cache_base(flattened_datum.get("teacher_cache_base"), t_cache)
                batch.setdefault("teacher_cache_indices", []).extend(int(c) for c in t_cache)
                # max(..., 0) clamps the masked variants' 0-filled positions (global 0
                # < base) to local row 0 — in range and never gathered (the trainer's
                # flat_valid filters masked positions before indexing).
                batch.setdefault("teacher_cache_local_indices", []).extend(
                    max(int(c) - base, 0) + cum_kept for c in t_cache
                )
                # Handled here (OPRD dual-view path); drop from flattened_datum so the
                # generic loop below does NOT also concatenate it. For NON-OPRD runs this
                # whole block is skipped (no teacher_input_ids) and teacher_cache_indices
                # flows through the generic loop unchanged — exactly as before OPRD.
                flattened_datum.pop("teacher_cache_indices", None)
            batch["_oprd_cum_teacher_len"] = cum_teacher + len(t_in)
            batch["_oprd_cum_num_kept"] = cum_kept + len(t_kept)

        # OPD convenience fields: allow sample-level teacher metadata and expand
        # it to token-aligned sequence fields before generic field preservation.
        teacher_id = flattened_datum.get("teacher_id")
        if teacher_id is not None and "teacher_ids" not in flattened_datum:
            batch.setdefault("teacher_ids", []).extend([int(teacher_id)] * seq_len)

        teacher_weight = flattened_datum.get("teacher_weight")
        if teacher_weight is not None and "teacher_weights" not in flattened_datum:
            batch.setdefault("teacher_weights", []).extend([float(teacher_weight)] * seq_len)

        # Handle other sequence fields (logprobs, advantages, etc.). The OPRD teacher
        # fields are handled explicitly above with cumulative offsets — exclude them here.
        for key, value in flattened_datum.items():
            if key not in [
                "input_ids",
                "position_ids",
                "labels",
                "weights",
                "teacher_id",
                "teacher_weight",
                "teacher_input_ids",
                "teacher_kept_indices",
                "teacher_position_ids",
                "teacher_cache_base",
            ]:
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
        # Drop OPRD internal accumulators (ints) so they never reach the model forward
        # as stray kwargs; the offsets they tracked are already baked into the fields.
        batch.pop("_oprd_cum_teacher_len", None)
        batch.pop("_oprd_cum_num_kept", None)

        # Wrap any other sequence fields that were concatenated
        for key, value in batch.items():
            if key not in [
                "input_ids",
                "labels",
                "position_ids",
                "request_id",
                "batch_id",
                "num_samples",
                "_r3_sample_lengths",
            ]:
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
                        "_r3_sample_lengths",
                    ):
                        continue
                    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                        if len(value[0]) == seq_len:
                            if key == "teacher_hidden_states":
                                if not value[0] or not isinstance(value[0][0], list):
                                    raise ValueError("teacher_hidden_states must be a sequence of hidden vectors")
                                hidden_dim = len(value[0][0])
                                value[0].extend([[0.0] * hidden_dim for _ in range(pad_length)])
                            else:
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
        """Add a sample to a non-packed micro-batch.

        This mirrors the packed path's token-shifting semantics:
        - HF-format datums (`input_ids` + full-length `labels`) are shifted to
          next-token prediction with `input_ids[:-1]` and `labels[1:]`.
        - xorl_client/RL datums (`input_ids` + `target_tokens`) are already
          shifted and are used as-is.
        """
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

        # Extract or generate position_ids
        if "position_ids" in flattened_datum:
            position_ids = flattened_datum["position_ids"]
            if not isinstance(position_ids, list):
                position_ids = position_ids.tolist() if hasattr(position_ids, "tolist") else list(position_ids)
        else:
            position_ids = list(range(len(input_ids)))

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

        weights = flattened_datum.get("weights")
        if weights is not None and not isinstance(weights, list):
            weights = weights.tolist() if hasattr(weights, "tolist") else list(weights)

        advantages = flattened_datum.get("advantages")
        if advantages is not None and not isinstance(advantages, list):
            advantages = advantages.tolist() if hasattr(advantages, "tolist") else list(advantages)

        # Detect if tokens are already shifted (xorl_client API format)
        is_already_shifted = "target_tokens" in flattened_datum and len(input_ids) == len(labels)
        if labels and not is_already_shifted and len(input_ids) == len(labels):
            logger.warning(
                "Sample %s has labels with the same length as input_ids; treating it as HF-format data "
                "and shifting for next-token prediction. Use target_tokens for already shifted targets.",
                sample_idx,
            )
            input_ids = input_ids[:-1]
            position_ids = position_ids[:-1]
            labels = labels[1:]
            if weights is not None:
                weights = weights[1:]
            if advantages is not None:
                advantages = advantages[1:]

        if advantages is not None:
            flattened_datum["advantages"] = advantages

        seq_len = len(input_ids)
        batch["input_ids"].append(input_ids)
        batch["position_ids"].append(position_ids)

        # Apply weights mask to labels if weights field is present
        # weights=0 -> labels=-100 (IGNORE_INDEX), weights=1 -> labels unchanged
        if labels:  # Only apply if we have labels
            if weights is not None:
                labels = apply_weights_to_labels(labels, weights, sample_idx)

            # Apply advantages mask to labels if advantages field is present
            # For RL losses, advantages=0 indicates prompt tokens where we don't compute loss
            # advantages=0 -> labels=-100 (IGNORE_INDEX)
            if advantages is not None:
                labels = apply_advantages_to_labels(labels, advantages, sample_idx)

        batch["labels"].append(labels)

        # OPD convenience fields: expand sample-level metadata to token-aligned
        # sequences so downstream tensor conversion/sharding can treat them like
        # labels and logprobs.
        teacher_id = flattened_datum.get("teacher_id")
        if teacher_id is not None and "teacher_ids" not in flattened_datum:
            batch.setdefault("teacher_ids", []).append([int(teacher_id)] * seq_len)

        teacher_weight = flattened_datum.get("teacher_weight")
        if teacher_weight is not None and "teacher_weights" not in flattened_datum:
            batch.setdefault("teacher_weights", []).append([float(teacher_weight)] * seq_len)

        # OPRD dual-view cache indices (mirrors the packed path): teacher_cache_indices
        # keeps the GLOBAL cache rows for the KL hidden-fetch; the trainer-forward
        # kept-row gather gets a per-sample-local view (single sample → no cumulative
        # offset). See the packed-path comment / oprd_warm_cache_indices_rebase_bug.md.
        if flattened_datum.get("teacher_input_ids") is not None:
            t_cache = flattened_datum.get("teacher_cache_indices")
            if t_cache is not None:
                if hasattr(t_cache, "tolist"):
                    t_cache = t_cache.tolist()
                base = _resolve_teacher_cache_base(flattened_datum.get("teacher_cache_base"), t_cache)
                batch.setdefault("teacher_cache_local_indices", []).append([max(int(c) - base, 0) for c in t_cache])

        # Preserve all other fields from loss_fn_inputs (logprobs, advantages, target_tokens, etc.)
        # Note: Keep target_tokens separate even if we used it as labels
        # Note: Exclude weights as it has been applied to labels
        for key, value in flattened_datum.items():
            if key not in [
                "input_ids",
                "position_ids",
                "labels",
                "weights",
                "teacher_id",
                "teacher_weight",
                "teacher_cache_base",
            ]:
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
    strategy: str = "sequential",
    on_oversized: str = "error",
    dp_size: int = 1,
    return_datum_order: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[int]]]:
    """
    Pack samples into micro-batches.

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
        strategy: Packing strategy (see ``PACKING_STRATEGIES``).
        on_oversized: How to treat samples longer than ``max_seq_len`` (see
            ``ON_OVERSIZED_MODES``). Defaults to ``"error"`` (no silent drops).
        dp_size: Distinct dispatcher batch slices, used by ``"balanced_dp"``.
        return_datum_order: If True, also return the original datum indices in
            batch-flattened order so callers can realign per-datum side arrays.

    Returns:
        List of micro-batch dictionaries (see MicroBatch TypedDict for format).
        If ``return_datum_order`` is True, returns ``(micro_batches, datum_order)``.

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
    packer = SequentialPacker(
        enable_packing=enable_packing,
        log_stats=True,
        pad_to_multiple_of=pad_to_multiple_of,
        strategy=strategy,
        on_oversized=on_oversized,
        dp_size=dp_size,
    )
    batches = packer.pack(datum_list, max_seq_len, request_id)
    if return_datum_order:
        return batches, packer.last_datum_order
    return batches


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
