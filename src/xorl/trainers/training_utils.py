"""Shared training utilities used by both Trainer and ModelRunner.

Extracts duplicated logic (gradient sync, gradient clipping, valid token
counting, LoRA merge, PP forward-backward) into reusable free functions.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.data.constants import IGNORE_INDEX
from xorl.lora.modules.base import LoraModule
from xorl.lora.utils import maybe_merge_lora as _merge
from xorl.qlora.utils import maybe_requant_qlora
from xorl.utils.device import get_device_type


def sync_sp_gradients(model: torch.nn.Module, sp_grad_sync_group) -> None:
    """All-reduce gradients for ring/Ulysses dims not folded into FSDP.

    SP ranks hold complementary (non-overlapping) parts of the same sequence,
    so their gradient contributions must be summed, not averaged.

    cp_fsdp_mode="all":           group is None → no-op
    cp_fsdp_mode="ulysses_only":  group is ring group
    cp_fsdp_mode="none":          group is unified SP group
    """
    if sp_grad_sync_group is not None:
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=sp_grad_sync_group)


def clip_gradients(
    model: torch.nn.Module,
    max_grad_norm: float,
    pp_enabled: bool = False,
    pp_group=None,
) -> float:
    """Clip gradients and return grad_norm. Handles PP all-reduce.

    Args:
        model: The model (may have FSDP's clip_grad_norm_).
        max_grad_norm: Maximum gradient norm for clipping.
        pp_enabled: Whether pipeline parallelism is active.
        pp_group: Process group for PP all-reduce of grad norms.

    Returns:
        Scalar grad_norm value.
    """
    if hasattr(model, "clip_grad_norm_"):
        _gn = model.clip_grad_norm_(max_grad_norm)
        grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor().item()
        elif hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()

    if pp_enabled and pp_group is not None:
        grad_norm_tensor = torch.tensor([grad_norm], device=get_device_type())
        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.MAX, group=pp_group)
        grad_norm = grad_norm_tensor.item()

    return grad_norm


def count_valid_tokens(
    micro_batches: List[Dict[str, Any]],
    group=None,
) -> torch.Tensor:
    """Count valid (non-IGNORE_INDEX) tokens and all-reduce across group.

    Supports both "labels" and "target_tokens" keys for compatibility
    with Trainer and ModelRunner respectively.
    """
    global_valid_tokens = torch.tensor(0, device=get_device_type())
    for mb in micro_batches:
        labels = mb.get("labels", mb.get("target_tokens"))
        if labels is not None:
            global_valid_tokens += (labels != IGNORE_INDEX).sum()
    dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM, group=group)
    return global_valid_tokens


def reset_lora_optimizer_states(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """ReLoRA-style optimizer reset after LoRA merge.

    Clears optimizer states (momentum, variance, step counter) for LoRA
    parameters. After merge, LoRA params are re-initialized (kaiming A, zero B),
    so old optimizer states are stale and must be discarded. Adam will rebuild
    its running averages from scratch for the fresh LoRA parameters.

    Args:
        model: Model with LoRA modules.
        optimizer: The optimizer whose states to reset.

    Returns:
        Number of parameters whose optimizer states were cleared.
    """

    # Collect LoRA parameter ids
    lora_param_ids = set()
    for module in model.modules():
        if isinstance(module, LoraModule):
            for p in module.get_lora_parameters():
                lora_param_ids.add(id(p))

    count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in lora_param_ids:
                continue
            if p in optimizer.state:
                del optimizer.state[p]
                count += 1

    return count


def maybe_merge_lora(
    model: torch.nn.Module,
    enable_lora: bool,
    enable_qlora: bool,
    merge_interval: int,
    global_step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    reset_optimizer: bool = False,
) -> None:
    """Periodic LoRA merge at merge_lora_interval.

    Args:
        optimizer: If provided with reset_optimizer=True, performs ReLoRA-style
            partial optimizer state reset after merge (prune 99% by magnitude).
        reset_optimizer: Whether to reset optimizer states after merge.
    """
    if merge_interval <= 0 or global_step % merge_interval != 0:
        return
    if enable_qlora:
        maybe_requant_qlora(model)
    elif enable_lora:
        _merge(model)

    if reset_optimizer and optimizer is not None:
        count = reset_lora_optimizer_states(model, optimizer)
        if count > 0:
            logging.getLogger(__name__).info(f"ReLoRA optimizer reset: pruned states for {count} LoRA parameters")


def negotiate_pp_seq_len(micro_batches: List[Dict[str, Any]], pp_group) -> int:
    """All-reduce max sequence length across all PP ranks for this step.

    All PP ranks must call this together.  Returns the global max seq_len
    so every rank pads to the same target, keeping P2P buffer shapes consistent.
    """
    local_max = max(mb["input_ids"].shape[-1] for mb in micro_batches)
    t = torch.tensor([local_max], device=get_device_type(), dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pp_group)
    return int(t.item())


@torch.compile
def pp_loss_fn(pred, labels):
    """Compiled PP cross-entropy loss (raw CE sum, unnormalized).

    Returns CE_sum over all non-ignored tokens.  Callers are responsible
    for dividing gradients by global_valid_tokens after the backward
    (either immediately or deferred to optim_step).
    """
    return F.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def pad_micro_batches_for_pp(
    micro_batches: List[Dict[str, Any]],
    sample_packing_sequence_len: int,
    sp_size: int = 1,
    pad_to_multiple_of: int = 1,
) -> None:
    """Pad all micro-batches to a fixed sequence length for pipeline parallelism.

    PP stages allocate fixed-size P2P communication buffers on the first step
    and reuse them across all subsequent steps.  Variable packed sequence
    lengths would cause send/recv shape mismatches.  This pads every
    micro-batch to ``sample_packing_sequence_len / sp_size`` (rounded up
    to ``pad_to_multiple_of``).

    cu_seq_lens are extended by growing the last real document (NOT by
    adding a separate all-zero "padding document") to avoid FA3 varlen
    backward NaN from degenerate inputs and stale max_length_q/k.
    """

    if sample_packing_sequence_len <= 0:
        return

    # Target sharded length (after SP split)
    target_sharded = sample_packing_sequence_len // sp_size if sp_size > 1 else sample_packing_sequence_len
    if pad_to_multiple_of > 1 and target_sharded % pad_to_multiple_of != 0:
        target_sharded = ((target_sharded + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    _PAD_VALUES = {"input_ids": 0, "labels": IGNORE_INDEX, "attention_mask": 0}
    full_target = target_sharded * sp_size if sp_size > 1 else target_sharded

    for mb in micro_batches:
        ids_len = mb["input_ids"].shape[-1]
        if ids_len < target_sharded:
            pad_tokens = target_sharded - ids_len

            for key in ("input_ids", "labels", "attention_mask"):
                if key in mb and isinstance(mb[key], torch.Tensor):
                    mb[key] = F.pad(mb[key], (0, pad_tokens), value=_PAD_VALUES.get(key, 0))

            if "position_ids" in mb and isinstance(mb["position_ids"], torch.Tensor):
                scale = mb["position_ids"].shape[-1] // ids_len if ids_len > 0 else 1
                mb["position_ids"] = F.pad(mb["position_ids"], (0, pad_tokens * scale), value=0)

        for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
            if key in mb and isinstance(mb[key], torch.Tensor):
                if mb[key][-1] < full_target:
                    mb[key] = mb[key].clone()
                    mb[key][-1] = full_target

        for ml_key, cu_key in (("max_length_q", "cu_seq_lens_q"), ("max_length_k", "cu_seq_lens_k")):
            if cu_key in mb and isinstance(mb[cu_key], torch.Tensor):
                new_max = mb[cu_key].diff().max().item()
                if ml_key in mb:
                    mb[ml_key] = max(mb[ml_key], new_max)
                else:
                    mb[ml_key] = new_max


def forward_backward_pp(
    model_parts: List[torch.nn.Module],
    pp_schedule,
    micro_batches: List[Dict[str, Any]],
    has_first_stage: bool,
    has_last_stage: bool,
    pp_group,
) -> float:
    """Pipeline parallel forward-backward step.

    Shared between Trainer and ModelRunner.  Returns raw CE_sum (unnormalized);
    callers normalize gradients by global_valid_tokens after this returns.

    Returns:
        raw_total_loss scalar (broadcast from last stage via MAX all-reduce).
    """
    device = get_device_type()

    input_ids = torch.cat([mb["input_ids"].to(device, non_blocking=True) for mb in micro_batches], dim=0)
    labels = torch.cat([mb["labels"].to(device, non_blocking=True) for mb in micro_batches], dim=0)

    # Per-microbatch metadata for PP forward (position_ids, flash-attn kwargs)
    _PP_FA_KEYS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
    pp_metadata_list = []
    for mb in micro_batches:
        md = {}
        if "position_ids" in mb:
            md["position_ids"] = mb["position_ids"]
        for key in _PP_FA_KEYS:
            if key in mb:
                md[key] = mb[key]
        pp_metadata_list.append(md)

    for model_part in model_parts:
        model_part._pp_batch_metadata = deque(pp_metadata_list)

    targets = labels if has_last_stage else None
    losses = [] if has_last_stage else None

    if has_first_stage:
        pp_schedule.step(input_ids, target=targets, losses=losses)
    else:
        pp_schedule.step(target=targets, losses=losses)

    # Broadcast loss from last stage via MAX
    if has_last_stage:
        total_loss = torch.sum(torch.stack(losses)).item()
        loss_tensor = torch.tensor([total_loss], device=device)
    else:
        loss_tensor = torch.tensor([-1.0], device=device)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.MAX, group=pp_group)

    del input_ids, labels
    return loss_tensor.item()
