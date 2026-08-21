"""Shared training utilities used by both Trainer and ModelRunner.

Extracts duplicated logic (gradient sync, gradient clipping, valid token
counting, LoRA merge, PP forward-backward) into reusable free functions.
"""

import logging
import os
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
from xorl.utils.dist_utils import all_reduce_metadata_tensor


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - torch 2.10+ always provides DTensor here
    DTensor = None


def sync_sp_gradients(
    model: torch.nn.Module,
    sp_grad_sync_group,
    *,
    skip_dtensor_grads: bool = False,
    excluded_parameter_ids: frozenset[int] = frozenset(),
) -> None:
    """All-reduce gradients for ring/Ulysses dims not folded into FSDP.

    SP ranks hold complementary (non-overlapping) parts of the same sequence,
    so their gradient contributions must be summed, not averaged.

    cp_fsdp_mode="all":           group is None → no-op
    cp_fsdp_mode="ulysses_only":  group is ring group
    cp_fsdp_mode="none":          group is unified SP group

    When DistSignSGD is active, FSDP-managed grads perform the exact SP sum
    inside the custom reduce-scatter hook before `sign()`. In that case, the
    later external SP sync should only touch non-FSDP grads.
    """
    if sp_grad_sync_group is not None:
        for p in model.parameters():
            if id(p) in excluded_parameter_ids:
                continue
            if p.grad is None:
                continue
            if skip_dtensor_grads and DTensor is not None and isinstance(p.grad, DTensor):
                continue
            grad = p.grad.to_local() if DTensor is not None and isinstance(p.grad, DTensor) else p.grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=sp_grad_sync_group)


def sync_lm_head_tp_gradient(
    model: torch.nn.Module,
    lm_head_tp_replica_group,
    *,
    excluded_parameter_ids: frozenset[int] = frozenset(),
) -> None:
    """Sum the lm-head-TP weight gradient over its replica dim (cp_replica x DP).

    With lm-head-only TP the lm_head is FSDP-sharded over a dedicated 2-D mesh
    (Shard(0) over the vocab/lm_head_tp dim, replicated over cp_replica x DP). The
    vocab-parallel CE reads lm_head.weight directly, so FSDP's reduce-scatter hook
    never fires and the replica ranks are left holding *partial* gradients for the
    same vocab rows. Sum them here. This is complementary to ``sync_sp_gradients``:
    lm-head TP requires cp_fsdp_mode='all' (sp_grad_sync_group is None), so that
    pass is a no-op for the lm_head and there is no double reduction.
    """
    if lm_head_tp_replica_group is None:
        return
    if dist.get_world_size(lm_head_tp_replica_group) <= 1:
        return
    for module in model.modules():
        if not getattr(module, "_xorl_fsdp_sharded_lm_head_loss", False):
            continue
        for p in module.parameters(recurse=False):
            if id(p) in excluded_parameter_ids:
                continue
            if p.grad is None:
                continue
            grad = p.grad.to_local() if DTensor is not None and isinstance(p.grad, DTensor) else p.grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=lm_head_tp_replica_group)


def _group_root_rank(group) -> int:
    if hasattr(dist, "get_global_rank"):
        return int(dist.get_global_rank(group, 0))
    return int(dist.get_process_group_ranks(group)[0])


def sync_lm_head_tp_parameters(
    model: torch.nn.Module,
    lm_head_tp_replica_group,
    lm_head_tp_group=None,
) -> None:
    """Broadcast lm-head-TP parameter shards over the replica dim after load.

    The lm-head-only TP mesh shards vocab rows over ``lm_head_tp`` and replicates
    each local vocab shard over DP/cp_replica. Gradients are summed over that
    replica dim before optimizer step, so replicas stay identical once they start
    identical. This post-load sync makes that invariant explicit for DCP loads
    and model-only replay loads.
    """
    with torch.no_grad():
        # Exact GLM-5.2 A is the one parameter intentionally ignored by the
        # vocab-sharding FSDP unit. Its custom VJP TP-sums dA, so the masters
        # remain identical after they begin from identical bytes. Establish
        # that invariant explicitly after initialization and every restore.
        if lm_head_tp_group is not None and dist.get_world_size(lm_head_tp_group) > 1:
            exact_src = _group_root_rank(lm_head_tp_group)
            for module in model.modules():
                if not (
                    getattr(module, "_glm52_exact_tp16_lm_head", False)
                    or getattr(module, "_dsv4_exact_tp8_lm_head", False)
                ):
                    continue
                parameter = getattr(module, "lora_A", None)
                if parameter is None or (DTensor is not None and isinstance(parameter, DTensor)):
                    raise RuntimeError("The exact GLM-5.2 lm-head lora_A must remain a plain replicated Parameter")
                dist.broadcast(parameter.data, src=exact_src, group=lm_head_tp_group)

        if lm_head_tp_replica_group is None or dist.get_world_size(lm_head_tp_replica_group) <= 1:
            return
        src = _group_root_rank(lm_head_tp_replica_group)
        for module in model.modules():
            if not getattr(module, "_xorl_fsdp_sharded_lm_head_loss", False):
                continue
            for p in module.parameters(recurse=False):
                param = p.data.to_local() if DTensor is not None and isinstance(p.data, DTensor) else p.data
                dist.broadcast(param, src=src, group=lm_head_tp_replica_group)


def clip_gradients(
    model: "torch.nn.Module | List[torch.nn.Module]",
    max_grad_norm: float | None,
    pp_enabled: bool = False,
    pp_group=None,
) -> float:
    """Clip gradients and return grad_norm. Handles PP all-reduce.

    Args:
        model: The model (may have FSDP's clip_grad_norm_), or a list of PP
            virtual-stage model parts (clipped jointly with one local norm).
        max_grad_norm: Maximum gradient norm for clipping.
        pp_enabled: Whether pipeline parallelism is active.
        pp_group: Process group for PP all-reduce of grad norms.

    Returns:
        Scalar grad_norm value.
    """
    if max_grad_norm is None:
        raise ValueError(
            "max_grad_norm must be configured for single-model optimizer steps; "
            "use a finite value <= 0 to disable clipping"
        )
    if max_grad_norm <= 0:
        return 0.0

    models = model if isinstance(model, (list, tuple)) else [model]
    if len(models) == 1 and hasattr(models[0], "clip_grad_norm_"):
        _gn = models[0].clip_grad_norm_(max_grad_norm)
        grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
    else:
        params = [p for m in models for p in m.parameters()]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor().item()
        elif hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()

    if pp_enabled and pp_group is not None:
        grad_norm_tensor = torch.tensor([grad_norm], device=get_device_type())
        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.MAX, group=pp_group)
        grad_norm = grad_norm_tensor.item()

    return grad_norm


def get_effective_grad_clip_value(max_grad_norm: float | None, *, use_distsignsgd: bool) -> float:
    """Return the clipping threshold to use for the current optimizer path.

    Non-positive ``max_grad_norm`` disables local-training gradient clipping.

    DistSignSGD turns gradients into sign-vote accumulators before the training
    loop reaches grad clipping. Clipping those sign votes changes the update
    scale by orders of magnitude, so we pass float("inf") to disable clipping
    and let the downstream `clip_gradients` call return the unclipped L2 norm
    purely for observability.

    Note for log readers: under DistSignSGD the value reported as "grad_norm"
    is really the L2 norm of accumulated sign votes (think `vote_l2_norm`),
    not a true gradient magnitude — its scale tracks `sqrt(num_params)` and
    voter agreement, not the underlying loss landscape.
    """
    if max_grad_norm is None:
        raise ValueError(
            "max_grad_norm must be configured for single-model optimizer steps; "
            "use a finite value <= 0 to disable clipping"
        )
    if use_distsignsgd:
        return float("inf")
    return max_grad_norm


def get_distsign_grad_scale_factor(active_voter_total: int) -> float:
    """Return the scale factor that converts accumulated sign votes to a mean.

    `active_voter_total` is the total number of (microbatch, rank) pairs that
    actually cast a sign vote — i.e. ranks whose microbatch had at least one
    valid token. Ranks with zero valid tokens contribute sign(0) = 0, not a
    ±1 vote, so multiplying `active_microbatches * dp_size` would over-count
    abstainers and bias the per-step update toward zero on uneven token
    distributions.
    """
    if active_voter_total <= 0:
        return 1.0
    return 1.0 / float(active_voter_total)


def count_valid_tokens(
    micro_batches: List[Dict[str, Any]],
    group=None,
) -> torch.Tensor:
    """Count valid (non-IGNORE_INDEX) tokens and all-reduce across group.

    Supports both "labels" and "target_tokens" keys for compatibility
    with Trainer and ModelRunner respectively. When both exist, prefer
    target_tokens because RL losses use it as the actual selected-token mask.
    """
    global_valid_tokens = torch.tensor(0, device="cpu", dtype=torch.int64)
    for mb in micro_batches:
        labels = mb.get("target_tokens", mb.get("labels"))
        if labels is not None:
            global_valid_tokens += (labels != IGNORE_INDEX).sum().to(device="cpu", dtype=torch.int64)
    return all_reduce_metadata_tensor(
        global_valid_tokens,
        op=dist.ReduceOp.SUM,
        group=group,
        device=get_device_type(),
    )


def count_active_microbatches(
    micro_batches: List[Dict[str, Any]],
    group=None,
) -> tuple[int, int]:
    """Return ``(active_microbatches, active_voter_total)`` for sign-vote aggregation.

    A single batched all-reduce (op=SUM) is issued for the whole accumulation step:

    - ``active_microbatches``: number of micro-batches in which *any* rank in
      ``group`` had at least one valid token.
    - ``active_voter_total``: sum over micro-batches of the number of ranks
      with valid tokens. This equals the number of (micro-batch, rank) pairs
      that contribute a real ±1 sign vote (ranks with zero valid tokens emit
      sign(0) = 0 and abstain).

    Callers should use ``active_voter_total`` as the divisor when normalizing
    accumulated sign votes; using ``active_microbatches * dp_size`` would
    over-count abstainers when token distribution is uneven.
    """
    if not micro_batches:
        return 0, 0

    flags = torch.zeros(len(micro_batches), device="cpu", dtype=torch.int64)
    for i, mb in enumerate(micro_batches):
        labels = mb.get("target_tokens", mb.get("labels"))
        if labels is None:
            continue
        flags[i] = int((labels != IGNORE_INDEX).any().item())
    flags = all_reduce_metadata_tensor(
        flags,
        op=dist.ReduceOp.SUM,
        group=group,
        device="cpu",
    )
    active_voter_total = int(flags.sum().item())
    active_microbatches = int((flags > 0).sum().item())
    return active_microbatches, active_voter_total


def scale_model_gradients(model_or_models, scale: float) -> None:
    """Scale gradients in-place while preserving DTensor metadata."""
    if scale == 1.0:
        return

    modules = model_or_models if isinstance(model_or_models, (list, tuple)) else [model_or_models]
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param.grad is not None:
                param.grad.mul_(scale)


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


def _pp_ce_sum(pred, labels):
    """Raw PP cross-entropy sum over all non-ignored tokens (unnormalized).

    Callers are responsible for dividing gradients by global_valid_tokens
    after the backward (either immediately or deferred to optim_step).
    """
    return F.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


_pp_ce_sum_compiled = torch.compile(_pp_ce_sum)


def _pp_ce_chunk_tokens() -> int:
    raw_value = os.environ.get("XORL_PP_CE_CHUNK_TOKENS", "0").strip()
    try:
        return max(0, int(raw_value))
    except ValueError as exc:
        raise ValueError("XORL_PP_CE_CHUNK_TOKENS must be an integer") from exc


def _pp_ce_sum_chunked(pred, labels):
    """Raw PP cross-entropy sum computed in token chunks to bound CE temporaries."""
    chunk_tokens = _pp_ce_chunk_tokens()
    if chunk_tokens <= 0:
        return _pp_ce_sum(pred, labels)

    pred_flat = pred.flatten(0, 1)
    labels_flat = labels.flatten(0, 1)
    loss = torch.zeros((), dtype=torch.float32, device=pred.device)
    for start in range(0, pred_flat.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, pred_flat.shape[0])
        loss = loss + F.cross_entropy(
            pred_flat[start:end].float(),
            labels_flat[start:end],
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
    return loss


def _pp_quack_linear_ce_sum(hidden, labels, *, lm_head, num_chunks: int = 8):
    """Fused linear+CE sum for PP, taking HIDDEN states (not logits).

    The last PP stage returns hidden instead of materializing the full
    [mbs, seq, vocab] logits (8GB+ at 248k vocab -> OOM). This applies the
    lm_head and cross-entropy in a single chunked kernel that never holds the
    full logits, matching the unnormalized reduction='sum' convention of
    ``_pp_ce_sum``. lm_head.weight is kept all-gathered by FSDP (norm+lm_head
    share a reshard_after_forward=False unit whose norm runs in the stage
    forward), so the schedule's autograd.backward(loss) flows grads to both
    hidden (pipeline) and lm_head.weight (its FSDP unit reduce-scatters them).
    """
    from xorl.models.module_utils import get_lm_head_weight  # noqa: PLC0415
    from xorl.objectives.causallm_loss import _chunk_size_from_num_chunks  # noqa: PLC0415
    from xorl.ops._vendored.quack.linear_cross_entropy import chunked_linear_cross_entropy  # noqa: PLC0415

    weight = get_lm_head_weight(lm_head, fsdp_sharded_loss=False)
    h = hidden.reshape(-1, hidden.shape[-1])
    lbl = labels.reshape(-1)
    chunk_size = _chunk_size_from_num_chunks(h.shape[0], num_chunks)
    return chunked_linear_cross_entropy(
        h, weight, lbl, chunk_size=chunk_size, ignore_index=IGNORE_INDEX, reduction="sum"
    )


def _pp_lm_head_ce_sum(
    hidden,
    labels,
    *,
    lm_head,
    ce_mode: str,
    tp_group=None,
    lm_head_fp32: bool = False,
    num_chunks: int = 8,
    logprob_temperature: float | torch.Tensor = 1.0,
):
    """Apply the ordinary per-token CE dispatcher to a PP terminal hidden state."""

    from xorl.models.module_utils import get_lm_head_weight  # noqa: PLC0415
    from xorl.ops.loss.per_token_ce import compute_per_token_ce  # noqa: PLC0415

    weight = get_lm_head_weight(
        lm_head,
        fsdp_sharded_loss=bool(getattr(lm_head, "_xorl_fsdp_sharded_lm_head_loss", False)),
    )
    per_token_ce = compute_per_token_ce(
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        ce_mode=ce_mode,
        num_chunks=num_chunks,
        tp_group=tp_group,
        lm_head_fp32=lm_head_fp32,
        lm_head=lm_head,
        logprob_temperature=logprob_temperature,
    )
    return per_token_ce.sum()


def make_pp_loss_fn(
    ce_mode: str = "compiled",
    lm_head=None,
    *,
    tp_group=None,
    lm_head_fp32: bool = False,
    num_chunks: int = 8,
    loss_owner=None,
    logprob_temperature: float = 1.0,
):
    """Return the PP cross-entropy loss variant selected by ``ce_mode``.

    'compiled' (default) returns the torch.compile'd CE sum; 'eager'
    returns the uncompiled baseline (useful for debugging or when compile
    regresses). 'quack_linear' returns a fused linear+CE that consumes the
    last stage's HIDDEN states (the stage must return hidden, not logits) and
    requires the last-stage ``lm_head`` module — avoiding the full 248k-vocab
    logits materialization (OOM) on the last stage.
    """
    exact_head = bool(
        lm_head is not None
        and (getattr(lm_head, "_glm52_exact_tp16_lm_head", False) or getattr(lm_head, "_dsv4_exact_tp8_lm_head", False))
    )
    if ce_mode == "batch_invariant" or exact_head:
        # Every rank constructs the schedule, but only the terminal stage calls
        # the loss. Defer the missing-head error so headless PP stages remain
        # independent of the output projection.
        def _lm_head_loss(hidden, labels):
            if lm_head is None:
                raise ValueError(f"ce_mode={ce_mode!r} under PP requires the terminal-stage lm_head module")
            scalar_temperature = float(getattr(loss_owner, "_pp_loss_scalar_temperature", logprob_temperature))
            temperature = scalar_temperature
            temperature_queue = getattr(loss_owner, "_pp_loss_temperatures", None)
            if temperature_queue is not None:
                if not temperature_queue:
                    raise RuntimeError("PP loss-temperature metadata was exhausted before the terminal loss")
                queued_temperature = temperature_queue.popleft()
                if queued_temperature is not None:
                    if not isinstance(queued_temperature, torch.Tensor):
                        raise TypeError("PP logprob_temperatures entries must be tensors")
                    if tuple(queued_temperature.shape) != tuple(labels.shape):
                        raise ValueError(
                            "PP logprob_temperatures must match terminal labels shape: "
                            f"temperature={tuple(queued_temperature.shape)}, labels={tuple(labels.shape)}"
                        )
                    if scalar_temperature != 1.0:
                        raise ValueError(
                            "A per-token logprob_temperatures tensor cannot be combined with a non-unit scalar "
                            "logprob_temperature"
                        )
                    temperature = (
                        queued_temperature.to(
                            device=hidden.device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                        .reshape(-1)
                        .contiguous()
                    )
            return _pp_lm_head_ce_sum(
                hidden,
                labels,
                lm_head=lm_head,
                ce_mode=ce_mode,
                tp_group=tp_group,
                lm_head_fp32=lm_head_fp32,
                num_chunks=num_chunks,
                logprob_temperature=temperature,
            )

        return _lm_head_loss
    if ce_mode == "eager":
        if _pp_ce_chunk_tokens() > 0:
            return _pp_ce_sum_chunked
        return _pp_ce_sum
    if ce_mode == "compiled":
        return _pp_ce_sum_compiled
    if ce_mode == "quack_linear":
        # The schedule constructs a loss_fn on EVERY rank (for _has_backward) but
        # only CALLS it on the last stage. Defer the lm_head check into the
        # closure so non-last stages (which pass lm_head=None and never call it)
        # don't fail at construction.
        def _quack_loss(hidden, labels):
            if lm_head is None:
                raise ValueError(
                    "ce_mode='quack_linear' under PP requires the last-stage lm_head module "
                    "(pass lm_head=model.lm_head on the last stage)."
                )
            return _pp_quack_linear_ce_sum(hidden, labels, lm_head=lm_head)

        return _quack_loss
    raise ValueError(
        f"Unknown ce_mode: {ce_mode!r} (expected 'eager', 'compiled', 'quack_linear', or 'batch_invariant')"
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

    _PAD_VALUES = {
        "input_ids": 0,
        "labels": IGNORE_INDEX,
        "attention_mask": 0,
        # Exact sampling-transform metadata must pad with the mathematical
        # identity so PP's fixed communication shape cannot change scoring.
        "logprob_temperatures": 1.0,
        "logprob_top_ks": 1 << 30,
        "logprob_top_ps": 1.0,
        "logprob_min_ps": 0.0,
        "_cp_logical_row_indices": -1,
        "_cp_request_ids": -1,
        "_cp_request_positions": 0,
        "_cp_live_mask": False,
    }
    full_target = target_sharded * sp_size if sp_size > 1 else target_sharded

    for mb in micro_batches:
        ids_len = mb["input_ids"].shape[-1]
        if ids_len < target_sharded:
            pad_tokens = target_sharded - ids_len

            for key, pad_value in _PAD_VALUES.items():
                if key in mb and isinstance(mb[key], torch.Tensor):
                    mb[key] = F.pad(mb[key], (0, pad_tokens), value=pad_value)

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


def align_dsv4_pp_storage_rows(
    micro_batches: List[Dict[str, Any]],
    *,
    cp_size: int,
    bucket_size: int = 1,
    minimum_storage_rows: int = 0,
    pad_to_multiple_of: int = 1,
) -> int:
    """Align compact-hyperconnection PP storage before stage buffers exist.

    Exact DSV4 compacts live rows for decoder compute, but its physical PP
    wire has storage-row capacity. ``compute_rows`` spans several owner planes,
    so PP-only length negotiation can leave a shorter rank unable to carry the
    compact prefix. One world MAX is the deadlock-free transitive superset of
    those overlapping PP/FSDP/EP/SP planes. Existing padding then grows local
    storage side channels and full-domain position/FA metadata together without
    changing ``_r3_sample_lengths`` or the live-row compute plan.
    """
    cp_size = int(cp_size)
    if cp_size < 1:
        raise ValueError(f"Exact DSV4 PP storage alignment requires cp_size >= 1, got {cp_size}")

    local_count = len(micro_batches)
    local_storage_rows = max(
        max((int(mb["input_ids"].shape[-1]) for mb in micro_batches), default=0),
        int(minimum_storage_rows),
    )
    negotiation = torch.tensor(
        [local_storage_rows, local_count, -local_count],
        dtype=torch.int64,
        device=get_device_type(),
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(negotiation, op=dist.ReduceOp.MAX)
    storage_rows, max_count, negative_min_count = (int(value) for value in negotiation.tolist())
    min_count = -negative_min_count
    if min_count != max_count:
        raise ValueError(
            f"Exact DSV4 PP ranks disagree on the number of microbatches: minimum={min_count}, maximum={max_count}"
        )
    if max_count == 0:
        raise ValueError("Exact DSV4 PP storage alignment requires at least one microbatch on every rank")
    if bucket_size > 1:
        storage_rows = ((storage_rows + bucket_size - 1) // bucket_size) * bucket_size
    if pad_to_multiple_of > 1:
        storage_rows = ((storage_rows + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    pad_micro_batches_for_pp(
        micro_batches,
        sample_packing_sequence_len=storage_rows * cp_size,
        sp_size=cp_size,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return storage_rows


_PP_FA_KEYS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
_PP_EXACT_ROW_KEYS = (
    "_cp_logical_row_indices",
    "_cp_request_ids",
    "_cp_request_positions",
    "_cp_live_mask",
    "_r3_sample_lengths",
    "sampler_prefill_lengths",
    "num_samples",
)


def _set_pp_batch_metadata(
    model_parts: List[torch.nn.Module],
    micro_batches: List[Dict[str, Any]],
    *,
    logprob_temperature: float = 1.0,
    index_share_mode: str | None = None,
) -> None:
    """Queue per-microbatch metadata (position_ids, flash-attn kwargs) on each part.

    Each part gets its own dict copies: _pp_forward pops keys from the entry,
    so sharing dicts across virtual stages would corrupt later stages' metadata.
    Exact DSV4 also needs the original storage-order token IDs on every stage:
    live hidden states occupy a compact prefix of the storage-capacity PP wire,
    while hash-routed MoE layers must reproduce token ownership from the
    original IDs.
    """
    pp_metadata_list = []
    for mb in micro_batches:
        md = {}
        if "input_ids" in mb:
            # Some decoder programs use token IDs inside later layers (DSV4
            # hash routing).  Every PP rank owns the same microbatch metadata,
            # so carry the original IDs out-of-band rather than putting them
            # on the differentiable activation wire.
            md["_pp_input_ids"] = mb["input_ids"]
        if "position_ids" in mb:
            md["position_ids"] = mb["position_ids"]
        if "input_ids" in mb:
            md["_pp_original_input_ids"] = mb["input_ids"]
        for key in _PP_FA_KEYS:
            if key in mb:
                md[key] = mb[key]
        for key in _PP_EXACT_ROW_KEYS:
            if key in mb:
                md[key] = mb[key]
        pp_metadata_list.append(md)

    for model_part in model_parts:
        requires_index_share = bool(getattr(model_part, "_pp_requires_index_share_mode", False))
        model_part._pp_batch_metadata = deque(
            {
                **md,
                **(
                    {"index_share_mode": index_share_mode}
                    if requires_index_share and index_share_mode is not None
                    else {}
                ),
            }
            for md in pp_metadata_list
        )
        model_part._pp_loss_temperatures = deque(mb.get("logprob_temperatures") for mb in micro_batches)
        model_part._pp_loss_scalar_temperature = float(logprob_temperature)


def _release_pp_index_share_contexts(model_parts: List[torch.nn.Module]) -> None:
    """Close contexts only after a schedule finishes all of its backwards."""

    for model_part in model_parts:
        release = getattr(model_part, "release_index_share_context", None)
        if callable(release):
            release()


def forward_backward_pp(
    model_parts: List[torch.nn.Module],
    pp_schedule,
    micro_batches: List[Dict[str, Any]],
    has_first_stage: bool,
    has_last_stage: bool,
    pp_group,
    logprob_temperature: float = 1.0,
    schedule_targets: torch.Tensor | None = None,
) -> float:
    """Pipeline parallel forward-backward step.

    Shared between Trainer and ModelRunner.  Returns raw CE_sum (unnormalized);
    callers normalize gradients by global_valid_tokens after this returns.

    Returns:
        raw_total_loss scalar (broadcast from the terminal stage via SUM all-reduce).
    """
    device = get_device_type()

    input_ids = torch.cat([mb["input_ids"].to(device, non_blocking=True) for mb in micro_batches], dim=0)
    labels = None
    if schedule_targets is None:
        label_tensors = [mb.get("labels", mb.get("target_tokens")) for mb in micro_batches]
        if any(label is None for label in label_tensors):
            raise ValueError("PP cross-entropy schedule requires labels or target_tokens in every microbatch")
        labels = torch.cat([label.to(device, non_blocking=True) for label in label_tensors], dim=0)

    _set_pp_batch_metadata(
        model_parts,
        micro_batches,
        logprob_temperature=logprob_temperature,
        index_share_mode="training_with_backward",
    )

    targets = (labels if schedule_targets is None else schedule_targets) if has_last_stage else None
    losses = [] if has_last_stage else None

    # return_outputs=False: the merged last-stage output is unused for training and
    # costs an O(n_microbatches x seq x vocab) allocation (37 GiB at m=16/8k/151k).
    try:
        if has_first_stage:
            pp_schedule.step(input_ids, target=targets, losses=losses, return_outputs=False)
        else:
            pp_schedule.step(target=targets, losses=losses, return_outputs=False)
    finally:
        # TRAINING_WITH_BACKWARD contexts must remain open through checkpoint
        # recomputation.  ``step`` is the first boundary at which every
        # scheduled backward is known to have completed (or aborted).
        _release_pp_index_share_contexts(model_parts)

    # Exactly one physical/virtual stage in each PP group owns the terminal
    # objective. SUM carries signed RL objectives correctly; the historical
    # MAX-with--1 sentinel silently corrupted sufficiently negative losses.
    if has_last_stage:
        total_loss = torch.sum(torch.stack(losses)).item()
        loss_tensor = torch.tensor([total_loss], device=device)
    else:
        loss_tensor = torch.tensor([0.0], device=device)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM, group=pp_group)

    del input_ids, labels
    return loss_tensor.item()


def forward_only_pp(
    model_parts: List[torch.nn.Module],
    pp_schedule,
    micro_batches: List[Dict[str, Any]],
    has_first_stage: bool,
    has_last_stage: bool,
) -> Optional[List[torch.Tensor]]:
    """Run the PP schedule forward-only (eval / reference logprobs).

    The schedule must be built with ``loss_fn=None`` and with meta input/output
    args on every stage (shape inference would otherwise consume the metadata
    queue). The last stage must return HIDDEN states (``_pp_lm_head_in_loss``);
    the caller applies lm_head + loss outside the schedule.

    Returns:
        Per-microbatch last-stage hidden states on ranks holding the last
        stage; None elsewhere.
    """
    device = get_device_type()
    input_ids = torch.cat([mb["input_ids"].to(device, non_blocking=True) for mb in micro_batches], dim=0)

    _set_pp_batch_metadata(model_parts, micro_batches, index_share_mode="forward_only")
    for model_part in model_parts:
        model_part._pp_forward_only = True
    try:
        if has_first_stage:
            output = pp_schedule.step(input_ids)
        else:
            output = pp_schedule.step()
    finally:
        for model_part in model_parts:
            model_part._pp_forward_only = False
        # Successful forward-only GLM invocations close themselves.  This
        # also owns cleanup if one stage forward raises partway through.
        _release_pp_index_share_contexts(model_parts)

    if not has_last_stage or output is None:
        return None
    return list(output.chunk(len(micro_batches), dim=0))
