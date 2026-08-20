"""One chunked LM-head scoring core for sampling-transform replay.

This is the single implementation of "chunked projection -> temperature ->
pinned support program -> selected logprob + VJP" shared by the scoring
lanes.  The forward runs bounded row chunks through an injected logits
function, applies the replay contract from
:mod:`xorl.ops.exact_sampling_transforms`, and saves only per-row scalars —
never a ``[tokens, vocab]`` activation.  The backward recomputes each chunk's
logits and applies the closed-form cross-entropy gradient on the same
support.

The core is policy-injected: ``ce_mode='bi_fused'`` passes the batch-invariant
kernels so its forward bytes keep matching serving, while the generic modes
(``eager``, ``compiled``, ``quack_linear``, ``fused_quack``) pass a plain GEMM
via :func:`sampling_transform_per_token_ce`.  Ordinary vocabulary-sharded body
TP is handled inside the core: each chunk's local logits are gathered in rank
order (loss rows are replicated across a body-TP group), the hidden gradient
is summed across vocabulary ranks, and each weight-shard gradient stays local.

A replayed token outside its row's current support scores ``-inf`` logprob
(``+inf`` CE) with zero gradient — a transform-carrying request is never
silently scored against unfiltered support.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from xorl.ops.exact_sampling_transforms import (
    EXACT_FILTER_ROW_CHUNK,
    NativeSelectedScore,
    exact_sampling_support,
    score_with_sampling_transforms,
    validate_temperature_rows,
)


LogitsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TemperatureFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ChunkedScoringPolicy:
    """How one lane computes chunk logits and native selected scores.

    ``logits_fn(hidden_chunk, prepared_weight)`` must return FP32
    ``[rows, local_vocab]`` logits BEFORE temperature.  ``temperature_fn``
    applies the lane's declared per-row FP32 temperature boundary to those
    logits.  ``native_selected_score`` scores identity (unfiltered) rows with
    the lane's native kernel; ``None`` selects the plain
    ``selected - logsumexp`` reduction.  ``prepare_weight`` runs once per
    forward/backward pass (e.g. one FP32 master cast) so ``logits_fn`` never
    re-casts per chunk.  ``grad_operand_dtype`` is the dtype of the recomputed
    ``dlogits`` operand in the two gradient GEMMs.
    """

    __slots__ = (
        "grad_operand_dtype",
        "logits_fn",
        "native_selected_score",
        "prepare_weight",
        "row_chunk",
        "temperature_fn",
    )

    def __init__(
        self,
        *,
        logits_fn: LogitsFn,
        temperature_fn: TemperatureFn,
        native_selected_score: NativeSelectedScore | None,
        prepare_weight: Callable[[torch.Tensor], torch.Tensor] = lambda weight: weight,
        grad_operand_dtype: torch.dtype | None = None,
        row_chunk: int = EXACT_FILTER_ROW_CHUNK,
    ) -> None:
        if row_chunk < 1:
            raise ValueError("row_chunk must be >= 1")
        self.logits_fn = logits_fn
        self.temperature_fn = temperature_fn
        self.native_selected_score = native_selected_score
        self.prepare_weight = prepare_weight
        self.grad_operand_dtype = grad_operand_dtype
        self.row_chunk = row_chunk


class _TpVocabLayout:
    """Rank-order vocabulary-shard geometry for one body-TP scoring call."""

    __slots__ = ("group", "local_vocab", "vocab_offset", "vocab_sizes")

    def __init__(self, group: dist.ProcessGroup, vocab_sizes: tuple[int, ...], rank: int) -> None:
        self.group = group
        self.vocab_sizes = vocab_sizes
        self.vocab_offset = sum(vocab_sizes[:rank])
        self.local_vocab = vocab_sizes[rank]


def _resolve_tp_layout(
    tp_group: dist.ProcessGroup | None,
    rows: int,
    local_vocab: int,
    device: torch.device,
) -> _TpVocabLayout | None:
    if tp_group is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("vocabulary-parallel transform scoring requires initialized torch.distributed")
    world_size = dist.get_world_size(tp_group)
    local = torch.tensor([rows, local_vocab], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size * 2, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=tp_group)
    layout = gathered.view(world_size, 2).cpu()
    row_counts = layout[:, 0].tolist()
    if any(count != rows for count in row_counts):
        raise ValueError(
            f"vocabulary-parallel transform scoring requires replicated loss rows across TP, got {row_counts}"
        )
    vocab_sizes = tuple(int(size) for size in layout[:, 1].tolist())
    if any(size <= 0 for size in vocab_sizes):
        raise ValueError(f"vocabulary-parallel transform scoring requires non-empty shards, got {vocab_sizes}")
    return _TpVocabLayout(tp_group, vocab_sizes, dist.get_rank(tp_group))


def gather_vocab_shards(
    local_logits: torch.Tensor,
    *,
    vocab_sizes: tuple[int, ...],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather possibly ragged vocabulary shards in process-group rank order."""

    max_vocab = max(vocab_sizes)
    padded = local_logits.new_zeros((local_logits.shape[0], max_vocab))
    padded[:, : local_logits.shape[1]].copy_(local_logits)
    world_size = dist.get_world_size(group)
    gathered = local_logits.new_empty((world_size * local_logits.shape[0], max_vocab))
    dist.all_gather_into_tensor(gathered, padded.contiguous(), group=group)
    rank_major = gathered.view(world_size, local_logits.shape[0], max_vocab)
    return torch.cat(
        [rank_major[rank, :, :vocab_size] for rank, vocab_size in enumerate(vocab_sizes)],
        dim=1,
    ).contiguous()


def _mm_accumulate_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply with an FP32-accumulated FP32 result."""

    if a.dtype is torch.float32 and b.dtype is torch.float32:
        return torch.mm(a, b)
    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


def _plain_selected_score(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lse = torch.logsumexp(logits, dim=-1)
    selected = logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return selected - lse, lse, selected


def _chunk_transformed_logits(
    hidden_chunk: torch.Tensor,
    prepared_weight: torch.Tensor,
    temperature_chunk: torch.Tensor | None,
    policy: ChunkedScoringPolicy,
    tp: _TpVocabLayout | None,
) -> torch.Tensor:
    logits = policy.logits_fn(hidden_chunk, prepared_weight)
    if logits.dtype is not torch.float32:
        raise TypeError(f"transform scoring requires FP32 chunk logits, got {logits.dtype}")
    if tp is not None:
        logits = gather_vocab_shards(logits, vocab_sizes=tp.vocab_sizes, group=tp.group)
    if temperature_chunk is not None:
        logits = policy.temperature_fn(logits, temperature_chunk)
    return logits


class _ChunkedTransformScoredCE(torch.autograd.Function):
    """Row-chunked selected-token CE on the pinned replay support."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels_safe: torch.Tensor,
        valid_mask: torch.Tensor,
        temperature_rows: torch.Tensor | None,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
        min_ps: torch.Tensor | None,
        policy: ChunkedScoringPolicy,
        tp: _TpVocabLayout | None,
    ) -> torch.Tensor:
        prepared_weight = policy.prepare_weight(weight)
        native_score = policy.native_selected_score or _plain_selected_score
        logprob_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for start in range(0, hidden.shape[0], policy.row_chunk):
            end = min(start + policy.row_chunk, hidden.shape[0])
            logits = _chunk_transformed_logits(
                hidden[start:end],
                prepared_weight,
                None if temperature_rows is None else temperature_rows[start:end],
                policy,
                tp,
            )
            ids = labels_safe[start:end]
            if top_ks is None:
                logprob, lse, _ = native_score(logits, ids)
            else:
                logprob, lse, _ = score_with_sampling_transforms(
                    logits,
                    ids,
                    top_ks[start:end],
                    top_ps[start:end],
                    min_ps[start:end],
                    native_score,
                )
            logprob_chunks.append(logprob)
            lse_chunks.append(lse)

        empty_float = torch.empty((0,), dtype=torch.float32, device=hidden.device)
        empty_long = torch.empty((0,), dtype=torch.int64, device=hidden.device)
        logprob = torch.cat(logprob_chunks) if logprob_chunks else empty_float
        lse = torch.cat(lse_chunks) if lse_chunks else empty_float.clone()

        ctx.policy = policy
        ctx.tp = tp
        ctx.has_temperature = temperature_rows is not None
        ctx.has_sampling_filter = top_ks is not None
        ctx.save_for_backward(
            hidden,
            weight,
            labels_safe,
            valid_mask,
            temperature_rows if temperature_rows is not None else empty_float,
            top_ks if top_ks is not None else empty_long,
            top_ps if top_ps is not None else empty_float,
            min_ps if min_ps is not None else empty_float,
            lse,
        )
        return torch.where(valid_mask, -logprob, torch.zeros_like(logprob))

    @staticmethod
    def backward(ctx, grad_ce: torch.Tensor):
        (
            hidden,
            weight,
            labels,
            valid_mask,
            stored_temperature,
            stored_top_ks,
            stored_top_ps,
            stored_min_ps,
            lse_all,
        ) = ctx.saved_tensors
        policy: ChunkedScoringPolicy = ctx.policy
        tp: _TpVocabLayout | None = ctx.tp
        temperature_rows = stored_temperature if ctx.has_temperature else None
        need_hidden = ctx.needs_input_grad[0]
        need_weight = ctx.needs_input_grad[1]
        if tp is None:
            if not need_hidden and not need_weight:
                return (None,) * 10
            compute_hidden = need_hidden
        else:
            # Every vocabulary rank must join the hidden-gradient sum, so the
            # decision to compute it has to be collective.
            need_hidden_flag = torch.tensor(int(need_hidden), dtype=torch.int64, device=hidden.device)
            dist.all_reduce(need_hidden_flag, op=dist.ReduceOp.MAX, group=tp.group)
            compute_hidden = bool(need_hidden_flag.item())

        operand_dtype = policy.grad_operand_dtype or hidden.dtype
        prepared_weight = policy.prepare_weight(weight)
        grad_weight_operand = prepared_weight if prepared_weight.dtype is operand_dtype else weight.to(operand_dtype)
        vocab_offset = 0 if tp is None else tp.vocab_offset
        local_vocab = weight.shape[0]
        grad_hidden = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device) if compute_hidden else None
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_weight else None
        grad_all = (grad_ce * valid_mask).float()

        for start in range(0, hidden.shape[0], policy.row_chunk):
            end = min(start + policy.row_chunk, hidden.shape[0])
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            temperature_chunk = None if temperature_rows is None else temperature_rows[start:end]
            logits = _chunk_transformed_logits(
                hidden_chunk,
                prepared_weight,
                temperature_chunk,
                policy,
                tp,
            )
            if ctx.has_sampling_filter:
                support = exact_sampling_support(
                    logits,
                    stored_top_ks[start:end],
                    stored_top_ps[start:end],
                    stored_min_ps[start:end],
                )
                selected_support = support.gather(1, labels_chunk.unsqueeze(1)).squeeze(1)
                local_support = support[:, vocab_offset : vocab_offset + local_vocab]
            else:
                selected_support = torch.ones_like(labels_chunk, dtype=torch.bool)
                local_support = None

            g = grad_all[start:end] * selected_support
            local_logits = logits[:, vocab_offset : vocab_offset + local_vocab]
            grad_logits = (local_logits - lse_all[start:end].unsqueeze(1)).exp()
            if local_support is not None:
                grad_logits *= local_support
            grad_logits *= g.unsqueeze(1)
            target_in_shard = (
                selected_support & (labels_chunk >= vocab_offset) & (labels_chunk < vocab_offset + local_vocab)
            )
            rows = torch.arange(labels_chunk.shape[0], device=labels_chunk.device)
            grad_logits[rows[target_in_shard], labels_chunk[target_in_shard] - vocab_offset] -= g[target_in_shard]
            if temperature_chunk is not None:
                grad_logits *= (1.0 / temperature_chunk).unsqueeze(1)
            grad_logits_op = grad_logits.to(operand_dtype)

            if compute_hidden:
                grad_hidden[start:end] += _mm_accumulate_fp32(grad_logits_op, grad_weight_operand)
            if need_weight:
                grad_weight += _mm_accumulate_fp32(grad_logits_op.t(), hidden_chunk.to(operand_dtype))

        if tp is not None and compute_hidden and grad_hidden.numel():
            # The hidden rows are replicated across vocabulary ranks, so their
            # gradient is the sum of every shard's contribution.  The weight
            # shard gradient stays local by construction.
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=tp.group)

        return (
            grad_hidden.to(hidden.dtype) if need_hidden else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_transform_scored_ce(
    hidden_states_flat: torch.Tensor,
    local_weight: torch.Tensor,
    labels_flat: torch.Tensor,
    *,
    ignore_index: int,
    temperature_rows: torch.Tensor | None,
    top_ks: torch.Tensor | None,
    top_ps: torch.Tensor | None,
    min_ps: torch.Tensor | None,
    policy: ChunkedScoringPolicy,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Per-token CE (``-log p(labels)``; 0 at ignored rows) through ``policy``."""

    if hidden_states_flat.ndim != 2 or local_weight.ndim != 2:
        raise ValueError("transform scoring requires two-dimensional hidden states and weight")
    if labels_flat.ndim != 1 or labels_flat.shape[0] != hidden_states_flat.shape[0]:
        raise ValueError("transform scoring requires one label per hidden row")
    rows = hidden_states_flat.shape[0]
    temperature_rows = validate_temperature_rows(
        temperature_rows,
        rows=rows,
        device=hidden_states_flat.device,
    )
    if (top_ks is None, top_ps is None, min_ps is None).count(True) not in (0, 3):
        raise ValueError("transform scoring requires all or none of top-k/top-p/min-p row metadata")
    valid_mask = labels_flat != ignore_index
    labels_safe = torch.where(valid_mask, labels_flat, torch.zeros_like(labels_flat))
    tp = _resolve_tp_layout(tp_group, rows, local_weight.shape[0], hidden_states_flat.device)
    return _ChunkedTransformScoredCE.apply(
        hidden_states_flat,
        local_weight,
        labels_safe,
        valid_mask,
        temperature_rows,
        top_ks,
        top_ps,
        min_ps,
        policy,
        tp,
    )


def _generic_scoring_policy(lm_head_fp32: bool) -> ChunkedScoringPolicy:
    """The stock-numerics plain-GEMM policy used by the generic ce_modes."""

    if lm_head_fp32:

        def logits_fn(hidden_chunk: torch.Tensor, prepared_weight: torch.Tensor) -> torch.Tensor:
            return hidden_chunk.float() @ prepared_weight.t()

        prepare_weight = lambda weight: weight.float()  # noqa: E731
        grad_operand_dtype = torch.float32
    else:

        def logits_fn(hidden_chunk: torch.Tensor, prepared_weight: torch.Tensor) -> torch.Tensor:
            return (hidden_chunk @ prepared_weight.t()).float()

        prepare_weight = lambda weight: weight  # noqa: E731
        grad_operand_dtype = None

    return ChunkedScoringPolicy(
        logits_fn=logits_fn,
        temperature_fn=lambda logits, temperature: logits / temperature.unsqueeze(1),
        native_selected_score=None,
        prepare_weight=prepare_weight,
        grad_operand_dtype=grad_operand_dtype,
    )


def sampling_transform_per_token_ce(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    *,
    ignore_index: int,
    temperature_rows: torch.Tensor | None,
    top_ks: torch.Tensor | None,
    top_ps: torch.Tensor | None,
    min_ps: torch.Tensor | None,
    lm_head_fp32: bool = False,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Generic-mode entry: score replay transforms from the raw lm-head weight.

    Serves every generic ``ce_mode`` (they share stock GEMM numerics) with and
    without ordinary body TP, honoring the ``lm_head_fp32`` convention of the
    unfiltered paths.
    """

    local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
    return chunked_transform_scored_ce(
        hidden_states_flat,
        local_weight,
        labels_flat,
        ignore_index=ignore_index,
        temperature_rows=temperature_rows,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        policy=_generic_scoring_policy(lm_head_fp32),
        tp_group=tp_group,
    )


__all__ = [
    "ChunkedScoringPolicy",
    "chunked_transform_scored_ce",
    "gather_vocab_shards",
    "sampling_transform_per_token_ce",
]
