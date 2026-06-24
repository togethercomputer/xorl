"""
Batch processing utilities for the distributed model worker.

Standalone functions for converting, validating, and sharding batches.
Extracted from RunnerDispatcher to keep the worker class focused
on communication and command dispatch.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

import torch

from xorl.distributed.parallel_state import get_parallel_state


logger = logging.getLogger(__name__)


def ep_duplicate_batches_enabled() -> bool:
    """Whether EP groups receive one duplicated batch slice (legacy dispatch).

    Legacy dispatch keyed the batch slice on the ep_fsdp coordinate, so all
    ep_size ranks of an EP group computed the same packed batch — ep_size-times
    redundant compute. Per-rank-distinct slices are correct: the MoE all-to-all
    routes per-rank-distinct tokens (the local-training path always runs this
    way) and the OPD full-vocab KL is rank-local; loss normalization by global
    valid tokens makes both regimes produce identical gradients. The duplication
    is kept only as a rollback switch: XORL_SERVER_EP_DUPLICATE_BATCHES=1.
    """
    return os.getenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "0").strip().lower() in {"1", "true", "yes"}


def batch_slice_rank_and_size(
    rank: int,
    world_size: int,
    parallel_state: Any,
    cp_size: int,
    pp_size: int,
) -> tuple[int, int]:
    """Return (slice_rank, slice_count) for request batch dispatch.

    Every rank gets a distinct batch slice; ranks that shard the same sequence
    (CP/SP) or the same pipeline stage share one. With EP enabled the slice is
    derived from the stage-local rank, so ranks within an EP group also see
    distinct data. Set XORL_SERVER_EP_DUPLICATE_BATCHES=1 to restore the legacy
    one-duplicated-slice-per-EP-group dispatch (see ep_duplicate_batches_enabled).
    """
    if getattr(parallel_state, "ep_enabled", False):
        ranks_per_pp_stage = max(1, world_size // max(1, pp_size))
        local_stage_rank = rank % ranks_per_pp_stage

        if ep_duplicate_batches_enabled():
            ep_size = max(1, int(getattr(parallel_state, "ep_size", 1)))
            ep_fsdp_size = max(1, int(getattr(parallel_state, "dp_shard_in_ep_size", 1)))
            ep_mesh = getattr(parallel_state, "ep_fsdp_device_mesh", None)
            if ep_mesh is not None:
                try:
                    ep_fsdp_rank = int(ep_mesh.get_local_rank("ep_fsdp"))
                    return min(ep_fsdp_rank, ep_fsdp_size - 1), ep_fsdp_size
                except Exception as exc:
                    logger.debug(
                        "Rank %s: could not read ep_fsdp local rank from EP mesh (%s); falling back to rank arithmetic",
                        rank,
                        exc,
                    )
            ep_fsdp_rank = min(local_stage_rank // ep_size, ep_fsdp_size - 1)
            return ep_fsdp_rank, ep_fsdp_size

        denom = max(1, cp_size)
        slice_count = max(1, ranks_per_pp_stage // denom)
        return min(local_stage_rank // denom, slice_count - 1), slice_count

    dp_size = max(1, world_size // max(1, cp_size * pp_size))
    dp_rank = min(rank // max(1, cp_size * pp_size), dp_size - 1)
    return dp_rank, dp_size


def _pad_teacher_hidden_states(value: list[Any]) -> torch.Tensor:
    max_len = max(len(seq) for seq in value)
    hidden_dim = None

    normalized = []
    for seq in value:
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        normalized_seq = []
        for hidden in seq:
            if hasattr(hidden, "tolist"):
                hidden = hidden.tolist()
            normalized_seq.append(hidden)
            if hidden_dim is None and isinstance(hidden, list):
                hidden_dim = len(hidden)
        normalized.append(normalized_seq)

    if hidden_dim is None:
        raise ValueError("teacher_hidden_states must be a nested sequence of hidden vectors")

    pad_vector = [0.0] * hidden_dim
    padded = [seq + [pad_vector] * (max_len - len(seq)) for seq in normalized]
    return torch.tensor(padded, dtype=torch.float)


def convert_batch_to_tensors(batch: Dict[str, Any], rank: int = 0) -> Dict[str, Any]:
    """
    Convert batch data from lists to torch tensors, with padding if needed.

    Args:
        batch: Batch dictionary with list data
        rank: Worker rank for logging

    Returns:
        Batch dictionary with torch tensors
    """
    converted_batch = {}

    # Fields that should be float tensors (probabilities, advantages, etc.)
    float_fields = {
        "logprobs",
        "advantages",
        "old_logprobs",
        "values",
        "returns",
        "teacher_weights",
        "hidden_match_weights",
        "teacher_hidden_states",
    }
    long_fields = {"teacher_ids", "teacher_cache_indices", "teacher_cache_local_indices"}
    # Fields that must be int32 (flash attention requires cu_seqlens as int32)
    int32_fields = {"cu_seq_lens_q", "cu_seq_lens_k"}

    for key, value in batch.items():
        if isinstance(value, list):
            try:
                # Determine dtype based on field name
                if key in float_fields:
                    dtype = torch.float
                elif key in int32_fields:
                    dtype = torch.int32
                elif key in long_fields:
                    dtype = torch.long
                else:
                    dtype = torch.long

                # Convert list to torch tensor
                tensor = torch.tensor(value, dtype=dtype)
                converted_batch[key] = tensor
                logger.debug(
                    f"Rank {rank}: Converted {key}: {type(value)} -> torch.Tensor{tuple(tensor.shape)} dtype={dtype}"
                )
            except Exception as e:
                # If conversion failed (likely due to ragged sequences), try padding
                if isinstance(value[0], list):
                    # This is a list of sequences - pad them
                    try:
                        if key == "teacher_hidden_states":
                            tensor = _pad_teacher_hidden_states(value)
                            max_len = tensor.shape[1]
                            dtype = torch.float
                        else:
                            max_len = max(len(seq) for seq in value)
                            pad_value = (
                                -100 if key in ("labels", "target_tokens") else 0
                            )  # Use -100 for labels/target_tokens (IGNORE_INDEX)
                            padded = []
                            for seq in value:
                                padded_seq = seq + [pad_value] * (max_len - len(seq))
                                padded.append(padded_seq)
                            # Determine dtype for padded sequences
                            dtype = torch.float if key in float_fields else torch.long
                            tensor = torch.tensor(padded, dtype=dtype)
                        converted_batch[key] = tensor
                        logger.debug(
                            f"Rank {rank}: Padded and converted {key}: {len(value)} sequences, max_len={max_len}, dtype={dtype}"
                        )
                    except Exception as e2:
                        logger.warning(f"Rank {rank}: Failed to convert {key} even after padding: {e2}, keeping as-is")
                        converted_batch[key] = value
                else:
                    logger.warning(f"Rank {rank}: Failed to convert {key} to tensor: {e}, keeping as-is")
                    converted_batch[key] = value
        else:
            # Keep non-list values as-is (e.g., request_id, batch_id)
            converted_batch[key] = value

    return converted_batch


def validate_batch_shapes(batch: Dict[str, Any], rank: int = 0, batch_idx: int = 0) -> bool:
    """
    Validate that all sequence tensors in a batch have consistent shapes.

    Args:
        batch: Batch dictionary with tensors
        rank: Worker rank for logging
        batch_idx: Index of this batch for logging

    Returns:
        True if shapes are consistent, False otherwise
    """
    seq_fields = ["input_ids", "labels", "position_ids", "attention_mask"]
    shapes = {}

    for key in seq_fields:
        if key in batch:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                shapes[key] = tuple(value.shape)
            elif isinstance(value, list):
                # Estimate shape from list
                if value and isinstance(value[0], list):
                    shapes[key] = (len(value), len(value[0]))
                else:
                    shapes[key] = (len(value),)

    # Check that all have the same sequence length (last dimension)
    seq_lengths = {}
    for key, shape in shapes.items():
        seq_len = shape[-1] if shape else 0
        seq_lengths[key] = seq_len

    unique_lengths = set(seq_lengths.values())
    if len(unique_lengths) > 1:
        logger.error(
            f"Rank {rank}: Batch {batch_idx} has INCONSISTENT sequence lengths: {seq_lengths}. Full shapes: {shapes}"
        )
        return False

    logger.debug(f"Rank {rank}: Batch {batch_idx} shapes validated: {shapes}")
    return True


def simple_sequence_shard(batch: Dict[str, Any], rank: int = 0) -> Dict[str, Any]:
    """
    Simple sequence sharding for non-packed batches (batch_size > 1).

    Unlike TextSequenceShardCollator which is designed for packed sequences
    (batch_size=1 with concatenated samples), this method handles batched
    sequences where each row is a separate sample.

    For batched data with shape [batch_size, seq_len]:
    1. Pad seq_len to be divisible by cp_size
    2. Slice each tensor to get [batch_size, seq_len // cp_size]

    Args:
        batch: Batch dictionary with tensors of shape [batch_size, seq_len]
        rank: Worker rank for logging

    Returns:
        Sharded batch dictionary
    """
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp_size
    cp_rank = parallel_state.cp_rank

    # Get sequence length from input_ids
    input_ids = batch.get("input_ids")
    if input_ids is None:
        return batch

    # Ensure tensor format
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch["input_ids"] = input_ids

    seq_len = input_ids.size(-1)

    # Calculate padding needed to make seq_len divisible by cp_size
    cp_chunk_size = (seq_len + cp_size - 1) // cp_size
    pad_len = cp_chunk_size * cp_size - seq_len

    # Helper to pad and slice tensors
    def pad_and_slice(tensor, pad_value=0, seq_dim=-1):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.long)

        if seq_dim < 0:
            seq_dim = tensor.ndim + seq_dim

        # Pad if needed
        if pad_len > 0:
            pad_shape = list(tensor.shape)
            pad_shape[seq_dim] = pad_len
            pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad_tensor], dim=seq_dim)

        # Slice for this cp_rank
        start_idx = cp_rank * cp_chunk_size
        return tensor.narrow(seq_dim, start_idx, cp_chunk_size)

    # Apply to all sequence tensors
    sharded_batch = {}
    for key, value in batch.items():
        if key == "_original_position_ids":
            # Keep original position_ids unsharded for unpacking per-token outputs
            sharded_batch[key] = value
        elif key == "input_ids":
            sharded_batch[key] = pad_and_slice(value, pad_value=0)
        elif key == "labels":
            sharded_batch[key] = pad_and_slice(value, pad_value=-100)  # IGNORE_INDEX
        elif key == "attention_mask":
            sharded_batch[key] = pad_and_slice(value, pad_value=0)
        elif key == "position_ids":
            # CRITICAL: position_ids should only be PADDED, NOT sliced
            # It needs to remain full-length for cu_seq_lens calculation
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.long)
            # Only pad, don't slice
            if pad_len > 0:
                pad_shape = list(value.shape)
                pad_shape[-1] = pad_len
                pad_tensor = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
                sharded_batch[key] = torch.cat([value, pad_tensor], dim=-1)
            else:
                sharded_batch[key] = value
        elif key == "teacher_hidden_states":
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float)
            elif not torch.is_floating_point(value):
                value = value.float()
            if value.dim() == 2:
                value = value.unsqueeze(0)
            sharded_batch[key] = pad_and_slice(value, pad_value=0.0, seq_dim=1)
        elif isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(-1) == seq_len:
            # Other tensors with matching sequence length
            # Use appropriate pad value based on field type
            if key == "target_tokens":
                pad_val = -100  # IGNORE_INDEX
            else:
                pad_val = 0
            sharded_batch[key] = pad_and_slice(value, pad_value=pad_val)
        else:
            # Non-sequence tensors (e.g., scalar values, metadata)
            sharded_batch[key] = value

    logger.debug(
        f"Rank {rank}: Simple sequence shard: {seq_len} -> {cp_chunk_size} (cp_rank={cp_rank}, cp_size={cp_size})"
    )

    return sharded_batch


def apply_sequence_sharding(
    batch: Dict[str, Any],
    rank: int = 0,
    sequence_shard_collator: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Apply appropriate sequence sharding based on batch format.

    - For packed batches (batch_size=1): Use sequence_shard_collator (TextSequenceShardCollator)
    - For non-packed batches (batch_size>1): Use simple sequence slicing

    Args:
        batch: Batch dictionary
        rank: Worker rank for logging
        sequence_shard_collator: Callable collator for packed sequence sharding

    Returns:
        Sharded batch dictionary
    """
    # Get batch size from input_ids
    input_ids = batch.get("input_ids")
    if input_ids is None:
        return batch

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch["input_ids"] = input_ids

    # Ensure 2D shape
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        batch["input_ids"] = input_ids

    # Ensure labels is tensor with correct shape
    if "labels" in batch and not isinstance(batch["labels"], torch.Tensor):
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
    if "labels" in batch and batch["labels"].dim() == 1:
        batch["labels"] = batch["labels"].unsqueeze(0)

    # Ensure position_ids is tensor with correct shape
    if "position_ids" in batch and not isinstance(batch["position_ids"], torch.Tensor):
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
    if "position_ids" in batch and batch["position_ids"].dim() == 1:
        batch["position_ids"] = batch["position_ids"].unsqueeze(0)

    # Generate attention_mask if not present (all 1s - attend to all tokens)
    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
        logger.debug(f"Rank {rank}: Generated attention_mask with shape {batch['attention_mask'].shape}")
    elif not isinstance(batch["attention_mask"], torch.Tensor):
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
    if batch["attention_mask"].dim() == 1:
        batch["attention_mask"] = batch["attention_mask"].unsqueeze(0)

    batch_size = input_ids.size(0)

    if batch_size == 1 and sequence_shard_collator is not None:
        # Packed batch - use full collator with cu_seqlens handling
        logger.debug(f"Rank {rank}: Using TextSequenceShardCollator for packed batch")
        return sequence_shard_collator(batch)
    else:
        # Non-packed batch - use simple sequence slicing
        logger.debug(f"Rank {rank}: Using simple sequence sharding for batch_size={batch_size}")
        return simple_sequence_shard(batch, rank=rank)
