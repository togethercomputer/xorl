"""Shared machinery for the exact TP-sharded LM-head value programs.

The GLM-5.2 and DSV4 exact heads run the same program skeleton — local
base+LoRA logits through the literal serving SGEMMs, a rank-order vocabulary
all-gather, native or filtered selected scoring, and a straight-through
surrogate VJP over rank-local training rows.  They pin different byte
contracts (FP32 vs BF16 logit buffers, ``head_v2`` vs plain base GEMM, rank-r
vs rank-1 factors, equal vs ragged row ownership).  This module owns the
skeleton; each family injects its contract-bearing pieces as closures,
constants, and an :class:`ExactHeadRowPlan`.

Nothing here computes a family's value bytes on its own: the base projection,
the temperature dtype boundary, and the native selected-score kernel always
come from the family module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from xorl.ops.exact_sampling_transforms import (
    EXACT_FILTER_ROW_CHUNK,
    exact_sampling_identity_rows,
    exact_sampling_support,
    selected_logprob_reference_grad,
    selected_logprob_reference_grad_partitioned,
)


SamplingTransforms = tuple[Tensor | None, Tensor | None, Tensor | None]


# ---------------------------------------------------------------------------
# Serving-kernel plumbing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def single_adapter_lora_batch_info(device_index: int, rows: int, rank: int = 1, scaling: float = 1.0) -> Any:
    """One active exact adapter, in the metadata format used by serving."""

    from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415

    device = torch.device("cuda", device_index)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.full((1,), rank, dtype=torch.int32, device=device),
        scalings=torch.full((1,), scaling, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )


def exact_lora_local_logits(
    hidden_2d: Tensor,
    effective_a: Tensor,
    effective_b: Tensor,
    *,
    base_logits: Tensor,
    rank: int,
    scaling: float = 1.0,
) -> Tensor:
    """Run the literal serving A/B SGEMMs with the fused add into ``base_logits``.

    The caller computes ``base_logits`` through its family's pinned base
    projection (FP32 ``head_v2`` for GLM-5.2, BF16 ``F.linear`` for DSV4); this
    helper owns only the shared LoRA-kernel choreography and its invariants.
    """

    try:
        from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
        from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError("Pinned exact LoRA SGEMM kernels are required") from exc

    rows = hidden_2d.shape[0]
    batch_info = single_adapter_lora_batch_info(hidden_2d.device.index, rows, rank, scaling)
    lora_a_output = sgemm_lora_a_fwd(hidden_2d, effective_a.unsqueeze(0), batch_info)
    if lora_a_output.dtype is not torch.bfloat16 or tuple(lora_a_output.shape) != (rows, rank):
        raise RuntimeError("The pinned exact A SGEMM did not produce the required BF16 rank store")
    output = sgemm_lora_b_fwd(
        lora_a_output,
        effective_b.unsqueeze(0),
        batch_info,
        base_output=base_logits,
    )
    if output.data_ptr() != base_logits.data_ptr():
        raise RuntimeError("The pinned exact B SGEMM did not perform the required in-place base+delta store")
    if output.dtype is not base_logits.dtype or not output.is_contiguous():
        raise RuntimeError("The pinned exact B SGEMM did not preserve the contiguous base-logit buffer")
    return output


# ---------------------------------------------------------------------------
# Rank-order collectives
# ---------------------------------------------------------------------------


def rank_order_vocab_from_stacked(
    stacked_logits: Tensor,
    *,
    expected_world_size: int,
    expected_local_vocab_size: int,
    expected_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Apply serving's ``[rank,row,vocab] -> [row,rank*vocab]`` order."""

    if stacked_logits.ndim != 3 or stacked_logits.shape[0] != expected_world_size:
        raise ValueError(
            "Rank-stacked LM-head logits must be [world, rows, local_vocab], got "
            f"{tuple(stacked_logits.shape)} for world={expected_world_size}"
        )
    if stacked_logits.shape[-1] != expected_local_vocab_size:
        raise ValueError(
            f"Rank-stacked local vocabulary width must be {expected_local_vocab_size}, got {stacked_logits.shape[-1]}"
        )
    if stacked_logits.dtype is not expected_dtype:
        raise TypeError(f"Rank-stacked LM-head logits must be {expected_dtype}, got {stacked_logits.dtype}")
    if not stacked_logits.is_contiguous():
        raise ValueError("Rank-stacked LM-head logits must be contiguous in collective rank order")
    rows = stacked_logits.shape[1]
    return stacked_logits.permute(1, 0, 2).reshape(rows, expected_world_size * expected_local_vocab_size)


def rank_order_vocab_all_gather(
    local_logits: Tensor,
    group: dist.ProcessGroup,
    *,
    expected_world_size: int,
    expected_local_vocab_size: int,
    expected_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Match serving's concat-style ``all_gather(..., dim=-1)`` byte order."""

    if not dist.is_initialized():
        raise RuntimeError("Rank-order LM-head all-gather requires initialized torch.distributed")
    if dist.get_world_size(group) != expected_world_size:
        raise RuntimeError(
            f"LM-head all-gather expected world size {expected_world_size}, got {dist.get_world_size(group)}"
        )
    if local_logits.ndim != 2 or local_logits.shape[1] != expected_local_vocab_size:
        raise ValueError(
            f"Local LM-head logits must be [rows, {expected_local_vocab_size}], got {tuple(local_logits.shape)}"
        )
    if local_logits.dtype is not expected_dtype or not local_logits.is_contiguous():
        raise ValueError(f"Local LM-head logits must be contiguous {expected_dtype} before the rank-order gather")

    rows = local_logits.shape[0]
    gathered = torch.empty(
        (expected_world_size * rows, expected_local_vocab_size),
        dtype=local_logits.dtype,
        device=local_logits.device,
    )
    dist.all_gather_into_tensor(gathered, local_logits, group=group)
    return rank_order_vocab_from_stacked(
        gathered.view(expected_world_size, rows, expected_local_vocab_size),
        expected_world_size=expected_world_size,
        expected_local_vocab_size=expected_local_vocab_size,
        expected_dtype=expected_dtype,
    )


def rank_order_row_all_gather(value: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Gather equal-shaped local row blocks in process-group rank order."""

    if not dist.is_initialized():
        raise RuntimeError("Rank-order LM-head row gathering requires initialized torch.distributed")
    if value.ndim == 0:
        raise ValueError("Rank-order LM-head row gathering requires at least one dimension")
    if not value.is_contiguous():
        raise ValueError("Rank-order LM-head row gathering requires a contiguous tensor")
    world_size = dist.get_world_size(group)
    gathered = torch.empty(
        (world_size * value.shape[0], *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(gathered, value, group=group)
    return gathered


def rank_order_row_counts(
    local_rows: int,
    device: torch.device,
    group: dist.ProcessGroup,
    *,
    world_size: int,
    program: str,
) -> tuple[int, ...]:
    """Exchange per-rank row counts in process-group rank order."""

    local = torch.tensor([local_rows], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    counts = tuple(int(value) for value in gathered.cpu().tolist())
    if any(value < 0 for value in counts):
        raise RuntimeError(f"{program} received negative row counts: {counts}")
    return counts


def rank_order_variable_row_all_gather(
    value: Tensor,
    group: dist.ProcessGroup,
    *,
    world_size: int,
    row_counts: tuple[int, ...],
    padded_rows: int,
) -> Tensor:
    """Gather ragged local row blocks (padded to ``padded_rows``) in rank order."""

    if value.ndim == 0 or not value.is_contiguous():
        raise ValueError("Variable-row all-gather requires a contiguous non-scalar tensor")
    if value.shape[0] > padded_rows or len(row_counts) != world_size:
        raise ValueError(f"Invalid variable-row geometry: local={value.shape[0]}, counts={row_counts}")
    if value.shape[0] < padded_rows:
        padding = value.new_zeros((padded_rows - value.shape[0], *value.shape[1:]))
        value = torch.cat((value, padding), dim=0)
    gathered = torch.empty(
        (world_size * padded_rows, *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(gathered, value, group=group)
    pieces = [
        gathered[rank * padded_rows : rank * padded_rows + count] for rank, count in enumerate(row_counts) if count
    ]
    return torch.cat(pieces, dim=0) if pieces else gathered[:0]


def require_equal_nonzero_row_count(value: Tensor, group: dist.ProcessGroup, *, program: str) -> None:
    """Fail before payload collectives when source-row shapes diverge."""

    counts = rank_order_row_counts(
        value.shape[0],
        value.device,
        group,
        world_size=dist.get_world_size(group),
        program=program,
    )
    if any(count <= 0 for count in counts):
        raise ValueError(f"{program} requires at least one source row on every rank")
    if len(set(counts)) > 1:
        raise ValueError(f"{program} requires equal source-row counts across the group, got {list(counts)}")


def all_reduce_sum_fp32(value: Tensor, group: dist.ProcessGroup) -> Tensor:
    """In-place logical-owner sum used by the hidden/factor surrogate gradients."""

    if value.dtype is not torch.float32 or not value.is_contiguous():
        raise ValueError("Exact LM-head surrogate reductions require contiguous FP32 tensors")
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def check_exact_head_tp_group(
    *,
    program: str,
    world_size: int,
    group_rank: int,
    global_rank: int,
    group_ranks: tuple[int, ...],
    backend: str,
    expected_world_size: int,
    expected_ranks: Sequence[int],
    shard_rank: int,
    source_ordinal: int | None = None,
) -> None:
    """Validate one family's TP group geometry against its declared contract.

    Pure checks over already-queried values, so family modules keep owning
    their ``torch.distributed`` calls (and their tests' patch points).
    """

    expected_ranks = tuple(int(rank) for rank in expected_ranks)
    if world_size != expected_world_size:
        raise RuntimeError(f"{program} requires TP{expected_world_size}, got TP{world_size}")
    if group_ranks != expected_ranks:
        raise RuntimeError(
            f"{program} gather order must match its expected TP{expected_world_size} group; "
            f"expected {expected_ranks}, got {group_ranks}"
        )
    if (
        group_rank != shard_rank
        or (source_ordinal is not None and group_rank != source_ordinal)
        or global_rank != group_ranks[shard_rank]
    ):
        raise RuntimeError(
            f"{program} shard/group rank mismatch: "
            f"shard_rank={shard_rank}, group_rank={group_rank}, global_rank={global_rank}"
        )
    if backend != "nccl":
        raise RuntimeError(f"{program} production group must use NCCL, got {backend}")


# ---------------------------------------------------------------------------
# The one exact-head autograd boundary
# ---------------------------------------------------------------------------


class ExactHeadRowPlan:
    """How this rank's training rows map onto the head's replicated row set.

    ``gather`` assembles every contributor's rows in rank order (an identity
    for already-replicated rows); ``narrow_local`` returns this rank's slice
    of a row-aligned result.

    The plan is stored on the autograd context from forward to backward, so
    the closures must capture only plain metadata (the process-group handle,
    ints, tuples) — never tensors, which would silently extend activation
    lifetimes past the saved-tensor accounting.
    """

    __slots__ = ("gather", "narrow_local")

    def __init__(
        self,
        gather: Callable[[Tensor], Tensor],
        narrow_local: Callable[[Tensor], Tensor],
    ) -> None:
        self.gather = gather
        self.narrow_local = narrow_local


REPLICATED_ROW_PLAN = ExactHeadRowPlan(lambda value: value, lambda value: value)


class ExactLmHeadFunction(torch.autograd.Function):
    """Own the literal exact-head forward; delegate only the VJP to the family.

    The forward gathers this rank's rows per the row plan, runs the family
    component's exact value program (native or sampling-filtered), and returns
    the caller's local rows.  The backward gathers the downstream rank-local
    gradients, applies the family's declared straight-through surrogate VJP to
    the identical global row block on every vocabulary rank, and returns only
    the caller's hidden-state slice.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        local_weight: Tensor,
        lora_a: Tensor,
        local_lora_b: Tensor,
        token_ids: Tensor,
        temperature: Tensor | None,
        sampling_transforms: SamplingTransforms,
        row_plan: ExactHeadRowPlan,
        component,
    ) -> Tensor:
        effective_a = lora_a.to(torch.bfloat16).contiguous()
        effective_b = local_lora_b.to(torch.bfloat16).contiguous()
        gathered_hidden = row_plan.gather(hidden_states)
        gathered_token_ids = row_plan.gather(token_ids)
        gathered_temperature = None if temperature is None else row_plan.gather(temperature)
        gathered_transforms = tuple(None if value is None else row_plan.gather(value) for value in sampling_transforms)
        has_sampling_filter = gathered_transforms[0] is not None
        if has_sampling_filter:
            gathered_logprob = component._exact_forward_value_filtered(
                gathered_hidden,
                local_weight,
                effective_a,
                effective_b,
                gathered_token_ids,
                gathered_temperature,
                gathered_transforms,
            )
        else:
            gathered_logprob = component._exact_forward_value(
                gathered_hidden,
                local_weight,
                effective_a,
                effective_b,
                gathered_token_ids,
                gathered_temperature,
            )
        local_logprob = row_plan.narrow_local(gathered_logprob).contiguous()
        # The output dtype is part of each family's pinned value program:
        # GLM-5.2 scores FP32 logits, DSV4 is BF16 end-to-end (BF16 gather,
        # BF16 temperature store, batch-invariant BF16 log_softmax).
        expected_dtype = getattr(component, "logprob_dtype", torch.float32)
        if local_logprob.dtype is not expected_dtype or tuple(local_logprob.shape) != tuple(token_ids.shape):
            raise RuntimeError(
                f"{type(component).__name__} returned an invalid selected-logprob tensor: "
                f"dtype={local_logprob.dtype}, shape={tuple(local_logprob.shape)}, "
                f"expected {expected_dtype} {tuple(token_ids.shape)}"
            )
        ctx.set_materialize_grads(False)
        ctx.component = component
        ctx.row_plan = row_plan
        ctx.has_sampling_filter = has_sampling_filter
        # local_weight and the masters are also version-counter sentinels.
        ctx.save_for_backward(
            gathered_hidden.detach(),
            local_weight,
            effective_a,
            effective_b,
            gathered_token_ids,
            gathered_temperature
            if gathered_temperature is not None
            else torch.empty((0,), dtype=torch.float32, device=hidden_states.device),
            lora_a,
            local_lora_b,
            *gathered_transforms,
        )
        return local_logprob

    @staticmethod
    def backward(ctx, grad_local_logprob: Tensor | None):
        (
            gathered_hidden,
            local_weight,
            effective_a,
            effective_b,
            gathered_token_ids,
            stored_temperature,
            _lora_a_master,
            _local_lora_b_master,
            top_ks,
            top_ps,
            min_ps,
        ) = ctx.saved_tensors
        if grad_local_logprob is None:
            return (None,) * 9
        # Revalidate before the first collective so a TP group that was
        # rebound or destroyed between forward and backward surfaces as the
        # deterministic contract error, not an NCCL hang.
        validate_tp_group = getattr(ctx.component, "_validate_tp_group", None)
        if callable(validate_tp_group):
            validate_tp_group()
        temperature = None if stored_temperature.numel() == 0 else stored_temperature
        gathered_grad_logprob = ctx.row_plan.gather(grad_local_logprob.float().contiguous())
        vjp = ctx.component._surrogate_vjp_filtered if ctx.has_sampling_filter else ctx.component._surrogate_vjp
        args = (
            gathered_hidden,
            local_weight,
            effective_a,
            effective_b,
            gathered_token_ids,
            gathered_grad_logprob,
            temperature,
        )
        if ctx.has_sampling_filter:
            args = (*args, (top_ks, top_ps, min_ps))
        grad_hidden, grad_a, grad_b = vjp(
            *args,
            needs_input_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]),
        )
        if grad_hidden is not None:
            grad_hidden = ctx.row_plan.narrow_local(grad_hidden).contiguous()
        return (grad_hidden, None, grad_a, grad_b, None, None, None, None, None)


# ---------------------------------------------------------------------------
# The shared surrogate-VJP plumbing
# ---------------------------------------------------------------------------


def surrogate_local_grad_logits(
    hidden_2d: Tensor,
    token_ids_1d: Tensor,
    grad_logprob_1d: Tensor,
    temperature_1d: Tensor | None,
    *,
    reference_full_logits_fn: Callable[[Tensor], Tensor],
    local_vocab_slice: slice,
) -> Tensor:
    """Local-shard reference dlogits for the unfiltered surrogate VJP."""

    with torch.no_grad(), torch.autocast(device_type=hidden_2d.device.type, enabled=False):
        full_reference_logits = reference_full_logits_fn(hidden_2d)
    full_grad_logits = selected_logprob_reference_grad(
        full_reference_logits,
        token_ids_1d,
        grad_logprob_1d,
        temperature_1d,
    )
    return full_grad_logits[:, local_vocab_slice].contiguous()


def filtered_surrogate_local_grad_logits(
    hidden_2d: Tensor,
    token_ids_1d: Tensor,
    grad_logprob_1d: Tensor,
    temperature_1d: Tensor | None,
    sampling_transforms: SamplingTransforms,
    *,
    exact_score_logits_fn: Callable[[Tensor, Tensor | None], Tensor],
    reference_full_logits_fn: Callable[[Tensor], Tensor],
    local_vocab_slice: slice,
    local_vocab_size: int,
    row_chunk: int = EXACT_FILTER_ROW_CHUNK,
) -> Tensor:
    """Local-shard reference dlogits over the exact support, in bounded chunks.

    Per chunk: the family recreates its literal value logits (post-gather,
    post-temperature) to derive the support, recreates its differentiable
    reference logits, and the shared partitioned reference grad is sliced to
    the local vocabulary columns.  The dense support mask is transient and
    never saved on an autograd context.
    """

    top_ks, top_ps, min_ps = sampling_transforms
    if top_ks is None or top_ps is None or min_ps is None:
        raise ValueError("filtered exact scoring requires complete row metadata")
    rows = hidden_2d.shape[0]
    local_grad_logits = torch.empty(
        (rows, local_vocab_size),
        dtype=torch.float32,
        device=hidden_2d.device,
    )
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        hidden_chunk = hidden_2d[start:end]
        temperature_chunk = None if temperature_1d is None else temperature_1d[start:end]
        chunk_transforms = (top_ks[start:end], top_ps[start:end], min_ps[start:end])
        with torch.no_grad(), torch.autocast(device_type=hidden_2d.device.type, enabled=False):
            exact_score_logits = exact_score_logits_fn(hidden_chunk, temperature_chunk)
            support = exact_sampling_support(exact_score_logits, *chunk_transforms)
            identity_rows = exact_sampling_identity_rows(
                *chunk_transforms,
                vocab_size=exact_score_logits.shape[1],
            )
            full_reference_logits = reference_full_logits_fn(hidden_chunk)
        full_grad_logits = selected_logprob_reference_grad_partitioned(
            full_reference_logits,
            token_ids_1d[start:end],
            grad_logprob_1d[start:end],
            temperature_chunk,
            support,
            identity_rows,
        )
        local_grad_logits[start:end] = full_grad_logits[:, local_vocab_slice]
    return local_grad_logits


__all__ = [
    "REPLICATED_ROW_PLAN",
    "ExactHeadRowPlan",
    "ExactLmHeadFunction",
    "all_reduce_sum_fp32",
    "check_exact_head_tp_group",
    "exact_lora_local_logits",
    "filtered_surrogate_local_grad_logits",
    "rank_order_row_all_gather",
    "rank_order_row_counts",
    "rank_order_variable_row_all_gather",
    "rank_order_vocab_all_gather",
    "rank_order_vocab_from_stacked",
    "require_equal_nonzero_row_count",
    "single_adapter_lora_batch_info",
    "surrogate_local_grad_logits",
]
