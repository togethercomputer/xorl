"""Trainable wrapper for the batch-invariant fused LM-head logprob contract.

Forward scores per-token cross-entropy through
:func:`xorl.ops.batch_invariant_ops.bi_lm_head_selected_logprob` — the K3
lm-head contract vendored identically in SGLang, so trainer and serving
logprobs are bitwise identical from bit-exact hidden states. The bf16 weight
stays resident (no fp32 lm-head copy). Per-row temperature materializes the
same FP32 ``z * (1/T)`` tensor that serving samples and scores; the scalar-one
call keeps the original non-materialized path.

Backward is the closed-form CE gradient computed against the saved forward
``lse`` with chunked cuBLAS recompute (stock-numerics class, like the other
fused CE backwards — the contract governs the forward bits only).

Per-row temperature and top-k/top-p/min-p replay route through the shared
chunked scoring core in :mod:`xorl.ops.loss.sampling_transform_ce`, injected
with the batch-invariant kernels so the forward values stay contract-exact;
the scalar-one no-filter call keeps the original non-materialized path.
"""

import torch
import torch.distributed as dist

from xorl.ops.batch_invariant_ops import (
    BI_LM_HEAD_VOCAB_CHUNK,
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob,
    bi_lm_head_selected_logprob_from_logits,
)
from xorl.ops.bi_families_v2 import (
    exact_temperature_scale_fp32_logits,
    families_v2_enabled,
    head_v2_full_logits_with_lse,
    head_v2_selected_logprob,
    head_v2_selected_logprob_from_logits,
)
from xorl.ops.exact_sampling_transforms import (
    EXACT_FILTER_ROW_CHUNK,
    TOP_K_ALL,
    exact_sampling_support,
    normalize_temperature_rows,
    score_with_sampling_transforms,
    validate_sampling_transform_rows,
)
from xorl.ops.loss.sampling_transform_ce import (
    ChunkedScoringPolicy,
    chunked_transform_scored_ce,
    gather_vocab_shards,
)


_TEMPERATURE_MATERIALIZE_ROW_CHUNK = EXACT_FILTER_ROW_CHUNK
_TP_LOCAL_ROW_CHUNK = 8


def _bi_scoring_policy(vocab_chunk: int) -> ChunkedScoringPolicy:
    """The batch-invariant lane's injection into the shared scoring core."""

    use_v2 = families_v2_enabled()
    if use_v2:

        def logits_fn(hidden_chunk, weight):
            logits, _ = head_v2_full_logits_with_lse(hidden_chunk, weight, temperature=None)
            return logits

        def native_selected_score(logits, token_ids):
            return head_v2_selected_logprob_from_logits(logits, token_ids, temperature=None)

    else:

        def logits_fn(hidden_chunk, weight):
            return bi_lm_head_full_logits(hidden_chunk, weight, vocab_chunk=vocab_chunk)

        def native_selected_score(logits, token_ids):
            return bi_lm_head_selected_logprob_from_logits(
                logits,
                token_ids,
                temperature=None,
                vocab_chunk=vocab_chunk,
            )

    return ChunkedScoringPolicy(
        logits_fn=logits_fn,
        temperature_fn=exact_temperature_scale_fp32_logits,
        native_selected_score=native_selected_score,
        row_chunk=_TEMPERATURE_MATERIALIZE_ROW_CHUNK,
        backward_vocab_chunk=vocab_chunk,
    )


def _tp_collective_layout(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    has_temperature: bool,
    has_sampling_filter: bool,
    use_v2: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], bool, bool]:
    """Exchange the small shape/program header before TP payload collectives."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("ce_mode='bi_fused' tensor parallelism requires initialized torch.distributed")
    world_size = dist.get_world_size(group)
    flags = int(has_temperature) | (int(has_sampling_filter) << 1) | (int(use_v2) << 2)
    local = torch.tensor(
        [hidden.shape[0], weight.shape[0], hidden.shape[1], flags],
        dtype=torch.int64,
        device=hidden.device,
    )
    gathered = torch.empty(world_size * local.numel(), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    layout = gathered.view(world_size, local.numel())

    hidden_sizes = layout[:, 2]
    program_flags = layout[:, 3]
    if bool((hidden_sizes != hidden_sizes[0]).any().item()):
        raise ValueError(f"bi_fused TP hidden widths differ across ranks: {hidden_sizes.cpu().tolist()}")
    head_families = program_flags >> 2
    if bool((head_families != head_families[0]).any().item()):
        raise ValueError(f"bi_fused TP ranks must use the same head-family program, got {head_families.cpu().tolist()}")
    vocab_sizes = tuple(int(value) for value in layout[:, 1].cpu().tolist())
    if any(size <= 0 for size in vocab_sizes):
        raise ValueError(f"bi_fused TP requires a non-empty vocabulary shard on every rank, got {vocab_sizes}")
    row_counts = tuple(int(value) for value in layout[:, 0].cpu().tolist())
    group_has_temperature = bool(((program_flags & 1) != 0).any().item())
    group_has_sampling_filter = bool(((program_flags & 2) != 0).any().item())
    return row_counts, vocab_sizes, group_has_temperature, group_has_sampling_filter


def _tp_broadcast_source_rows(
    value: torch.Tensor,
    *,
    source_rank: int,
    start: int,
    rows: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Broadcast one real source-owner chunk without a full row all-gather."""

    rank = dist.get_rank(group)
    if rank == source_rank:
        chunk = value.narrow(0, start, rows).contiguous()
    else:
        chunk = value.new_empty((rows, *value.shape[1:]))
    global_source_rank = dist.get_global_rank(group, source_rank)
    dist.broadcast(chunk, src=global_source_rank, group=group)
    return chunk


def _tp_exact_full_logits(
    hidden: torch.Tensor,
    local_weight: torch.Tensor,
    *,
    vocab_sizes: tuple[int, ...],
    group: dist.ProcessGroup,
    use_v2: bool,
    vocab_chunk: int,
) -> torch.Tensor:
    if use_v2:
        local_logits, _ = head_v2_full_logits_with_lse(hidden, local_weight, temperature=None)
    else:
        local_logits = bi_lm_head_full_logits(hidden, local_weight, vocab_chunk=vocab_chunk)
    return gather_vocab_shards(local_logits, vocab_sizes=vocab_sizes, group=group)


def _tp_score_full_logits(
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: torch.Tensor | None,
    sampling_transforms: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    *,
    use_v2: bool,
    vocab_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    transformed_logits = (
        full_logits if temperature is None else exact_temperature_scale_fp32_logits(full_logits, temperature)
    )
    top_ks, top_ps, min_ps = sampling_transforms
    if use_v2:
        native_score = lambda logits, token_ids: head_v2_selected_logprob_from_logits(  # noqa: E731
            logits,
            token_ids,
            temperature=None,
        )
    else:
        native_score = lambda logits, token_ids: bi_lm_head_selected_logprob_from_logits(  # noqa: E731
            logits,
            token_ids,
            temperature=None,
            vocab_chunk=vocab_chunk,
        )
    if top_ks is None:
        logprob, lse, _ = native_score(transformed_logits, labels)
    else:
        logprob, lse, _ = score_with_sampling_transforms(
            transformed_logits,
            labels,
            top_ks,
            top_ps,
            min_ps,
            native_score,
        )
    return logprob, lse


class _BiFusedVocabParallelPerTokenCE(torch.autograd.Function):
    """Exact TP forward over distinct row owners with a local-shard CE VJP."""

    @staticmethod
    def forward(
        ctx,
        local_hidden,
        local_weight,
        local_labels,
        local_valid,
        local_temperature,
        local_top_ks,
        local_top_ps,
        local_min_ps,
        tp_group,
        vocab_chunk,
    ):
        use_v2 = families_v2_enabled()
        has_temperature = local_temperature is not None
        has_sampling_filter = local_top_ks is not None
        row_counts, vocab_sizes, has_temperature, has_sampling_filter = _tp_collective_layout(
            local_hidden,
            local_weight,
            group=tp_group,
            has_temperature=has_temperature,
            has_sampling_filter=has_sampling_filter,
            use_v2=use_v2,
        )
        if has_temperature and local_temperature is None:
            local_temperature = torch.ones(
                local_hidden.shape[0],
                dtype=torch.float32,
                device=local_hidden.device,
            )
        if has_sampling_filter and local_top_ks is None:
            local_top_ks = torch.full(
                (local_hidden.shape[0],),
                TOP_K_ALL,
                dtype=torch.int64,
                device=local_hidden.device,
            )
            local_top_ps = torch.ones(local_hidden.shape[0], dtype=torch.float32, device=local_hidden.device)
            local_min_ps = torch.zeros(local_hidden.shape[0], dtype=torch.float32, device=local_hidden.device)
        rank = dist.get_rank(tp_group)
        local_ce_chunks: list[torch.Tensor] = []
        lse_slots: list[torch.Tensor] = []

        for source_rank, source_rows in enumerate(row_counts):
            for start in range(0, source_rows, _TP_LOCAL_ROW_CHUNK):
                rows = min(_TP_LOCAL_ROW_CHUNK, source_rows - start)
                hidden = _tp_broadcast_source_rows(
                    local_hidden,
                    source_rank=source_rank,
                    start=start,
                    rows=rows,
                    group=tp_group,
                )
                labels = _tp_broadcast_source_rows(
                    local_labels,
                    source_rank=source_rank,
                    start=start,
                    rows=rows,
                    group=tp_group,
                )
                valid = _tp_broadcast_source_rows(
                    local_valid,
                    source_rank=source_rank,
                    start=start,
                    rows=rows,
                    group=tp_group,
                )
                temperature = None
                if has_temperature:
                    temperature = _tp_broadcast_source_rows(
                        local_temperature,
                        source_rank=source_rank,
                        start=start,
                        rows=rows,
                        group=tp_group,
                    )
                sampling_transforms = (None, None, None)
                if has_sampling_filter:
                    top_ks = _tp_broadcast_source_rows(
                        local_top_ks,
                        source_rank=source_rank,
                        start=start,
                        rows=rows,
                        group=tp_group,
                    )
                    top_ps = _tp_broadcast_source_rows(
                        local_top_ps,
                        source_rank=source_rank,
                        start=start,
                        rows=rows,
                        group=tp_group,
                    )
                    min_ps = _tp_broadcast_source_rows(
                        local_min_ps,
                        source_rank=source_rank,
                        start=start,
                        rows=rows,
                        group=tp_group,
                    )
                    sampling_transforms = (top_ks, top_ps, min_ps)

                full_logits = _tp_exact_full_logits(
                    hidden,
                    local_weight,
                    vocab_sizes=vocab_sizes,
                    group=tp_group,
                    use_v2=use_v2,
                    vocab_chunk=vocab_chunk,
                )
                logprob, lse = _tp_score_full_logits(
                    full_logits,
                    labels,
                    temperature,
                    sampling_transforms,
                    use_v2=use_v2,
                    vocab_chunk=vocab_chunk,
                )
                if rank == source_rank:
                    local_ce_chunks.append(torch.where(valid, -logprob, torch.zeros_like(logprob)))
                lse_slots.append(lse)

        empty_float = torch.empty((0,), dtype=torch.float32, device=local_hidden.device)
        empty_long = torch.empty((0,), dtype=torch.int64, device=local_hidden.device)
        lse_tensor = (
            torch.cat(lse_slots) if lse_slots else torch.empty((0,), dtype=torch.float32, device=local_hidden.device)
        )
        ctx.tp_group = tp_group
        ctx.vocab_chunk = int(vocab_chunk)
        ctx.use_v2 = use_v2
        ctx.has_temperature = has_temperature
        ctx.has_sampling_filter = has_sampling_filter
        ctx.row_counts = row_counts
        ctx.vocab_sizes = vocab_sizes
        ctx.save_for_backward(
            local_hidden,
            local_weight,
            local_labels,
            local_valid,
            local_temperature if has_temperature else empty_float,
            local_top_ks if has_sampling_filter else empty_long,
            local_top_ps if has_sampling_filter else empty_float,
            local_min_ps if has_sampling_filter else empty_float,
            lse_tensor,
        )
        if local_ce_chunks:
            return torch.cat(local_ce_chunks)
        return empty_float

    @staticmethod
    def backward(ctx, grad_local_ce):
        (
            local_hidden,
            local_weight,
            local_labels,
            local_valid,
            stored_temperature,
            stored_top_ks,
            stored_top_ps,
            stored_min_ps,
            lse_slots,
        ) = ctx.saved_tensors
        if grad_local_ce is None:
            grad_local_ce = torch.zeros(
                local_hidden.shape[0],
                dtype=torch.float32,
                device=local_hidden.device,
            )

        group = ctx.tp_group
        row_counts = ctx.row_counts
        vocab_sizes = ctx.vocab_sizes
        rank = dist.get_rank(group)
        vocab_offset = sum(vocab_sizes[:rank])
        local_vocab = vocab_sizes[rank]
        need_hidden = ctx.needs_input_grad[0]
        need_weight = ctx.needs_input_grad[1]
        need_hidden_tensor = torch.tensor(int(need_hidden), dtype=torch.int64, device=local_hidden.device)
        dist.all_reduce(need_hidden_tensor, op=dist.ReduceOp.MAX, group=group)
        compute_hidden = bool(need_hidden_tensor.item())
        grad_hidden = torch.zeros_like(local_hidden) if need_hidden else None
        grad_weight = (
            torch.zeros(local_weight.shape, dtype=torch.float32, device=local_weight.device) if need_weight else None
        )
        grad_local_ce = grad_local_ce.float().contiguous()
        lse_offset = 0

        for source_rank, source_rows in enumerate(row_counts):
            for start in range(0, source_rows, _TP_LOCAL_ROW_CHUNK):
                rows_in_chunk = min(_TP_LOCAL_ROW_CHUNK, source_rows - start)
                hidden = _tp_broadcast_source_rows(
                    local_hidden,
                    source_rank=source_rank,
                    start=start,
                    rows=rows_in_chunk,
                    group=group,
                )
                labels = _tp_broadcast_source_rows(
                    local_labels,
                    source_rank=source_rank,
                    start=start,
                    rows=rows_in_chunk,
                    group=group,
                )
                valid = _tp_broadcast_source_rows(
                    local_valid,
                    source_rank=source_rank,
                    start=start,
                    rows=rows_in_chunk,
                    group=group,
                )
                grad_ce = _tp_broadcast_source_rows(
                    grad_local_ce,
                    source_rank=source_rank,
                    start=start,
                    rows=rows_in_chunk,
                    group=group,
                )
                temperature = None
                if ctx.has_temperature:
                    temperature = _tp_broadcast_source_rows(
                        stored_temperature,
                        source_rank=source_rank,
                        start=start,
                        rows=rows_in_chunk,
                        group=group,
                    )

                local_support = None
                selected_support = torch.ones_like(valid)
                if ctx.has_sampling_filter:
                    top_ks = _tp_broadcast_source_rows(
                        stored_top_ks,
                        source_rank=source_rank,
                        start=start,
                        rows=rows_in_chunk,
                        group=group,
                    )
                    top_ps = _tp_broadcast_source_rows(
                        stored_top_ps,
                        source_rank=source_rank,
                        start=start,
                        rows=rows_in_chunk,
                        group=group,
                    )
                    min_ps = _tp_broadcast_source_rows(
                        stored_min_ps,
                        source_rank=source_rank,
                        start=start,
                        rows=rows_in_chunk,
                        group=group,
                    )
                    exact_logits = _tp_exact_full_logits(
                        hidden,
                        local_weight,
                        vocab_sizes=vocab_sizes,
                        group=group,
                        use_v2=ctx.use_v2,
                        vocab_chunk=ctx.vocab_chunk,
                    )
                    transformed_logits = (
                        exact_logits
                        if temperature is None
                        else exact_temperature_scale_fp32_logits(exact_logits, temperature)
                    )
                    support = exact_sampling_support(transformed_logits, top_ks, top_ps, min_ps)
                    selected_support = support.gather(1, labels.unsqueeze(1)).squeeze(1)
                    local_support = support[:, vocab_offset : vocab_offset + local_vocab]

                g = (grad_ce * valid * selected_support).float()
                local_logits = torch.mm(hidden, local_weight.t(), out_dtype=torch.float32)
                inv_temperature = None
                if temperature is not None:
                    inv_temperature = (1.0 / temperature).unsqueeze(1)
                    local_logits *= inv_temperature
                lse = lse_slots.narrow(0, lse_offset, rows_in_chunk)
                lse_offset += rows_in_chunk
                grad_logits = local_logits.sub_(lse.unsqueeze(1)).exp_()
                grad_logits *= g.unsqueeze(1)
                if local_support is not None:
                    grad_logits *= local_support
                target_in_shard = (
                    selected_support & valid & (labels >= vocab_offset) & (labels < vocab_offset + local_vocab)
                )
                rows = torch.arange(labels.shape[0], device=labels.device)
                grad_logits[rows[target_in_shard], labels[target_in_shard] - vocab_offset] -= g[target_in_shard]
                if inv_temperature is not None:
                    grad_logits *= inv_temperature
                grad_logits_bf16 = grad_logits.to(local_hidden.dtype)

                if compute_hidden:
                    grad_hidden_chunk = torch.mm(
                        grad_logits_bf16,
                        local_weight,
                        out_dtype=torch.float32,
                    )
                    dist.all_reduce(grad_hidden_chunk, op=dist.ReduceOp.SUM, group=group)
                    if need_hidden and rank == source_rank:
                        grad_hidden.narrow(0, start, rows_in_chunk).copy_(grad_hidden_chunk.to(grad_hidden.dtype))
                if need_weight:
                    grad_weight.add_(torch.mm(grad_logits_bf16.t(), hidden, out_dtype=torch.float32))

        return (
            grad_hidden,
            grad_weight.to(local_weight.dtype) if need_weight else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _BiFusedLmHeadPerTokenCE(torch.autograd.Function):
    """The identity (no-temperature, no-filter) fast path: fused kernels only.

    Transform-carrying calls never reach this Function — they route through
    the shared chunked scoring core with the batch-invariant policy.
    """

    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        labels_safe,
        valid_mask,
        vocab_chunk,
    ):
        if families_v2_enabled():
            # head v2 (families-v2 migration): same GEMM K-chain, epilogue-stats
            # online LSE; logits never materialize; backward only consumes lse.
            logprob, lse, _ = head_v2_selected_logprob(hidden, weight, labels_safe, temperature=None)
        else:
            logprob, lse, _ = bi_lm_head_selected_logprob(
                hidden, weight, labels_safe, temperature=None, vocab_chunk=vocab_chunk
            )
        ctx.save_for_backward(hidden, weight, labels_safe, valid_mask, lse)
        ctx.vocab_chunk = vocab_chunk
        return torch.where(valid_mask, -logprob, torch.zeros_like(logprob))

    @staticmethod
    def backward(ctx, grad_ce):
        hidden, weight, labels, valid_mask, lse = ctx.saved_tensors
        vocab_chunk = ctx.vocab_chunk
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        need_h = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]

        g = (grad_ce * valid_mask).float()
        g_col = g.unsqueeze(1)
        lse_col = lse.unsqueeze(1)
        grad_h = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device) if need_h else None
        grad_w = torch.empty_like(weight) if need_w else None
        rows = torch.arange(n_tokens, device=hidden.device)

        for col_start in range(0, vocab, vocab_chunk):
            col_end = min(col_start + vocab_chunk, vocab)
            w_c = weight[col_start:col_end]
            # bf16 tensor-core GEMM, fp32 accumulate + fp32 out (no fp32 copies)
            logits_c = torch.mm(hidden, w_c.t(), out_dtype=torch.float32)
            grad_z = logits_c.sub_(lse_col).exp_().mul_(g_col)
            in_chunk = (labels >= col_start) & (labels < col_end)
            grad_z[rows[in_chunk], labels[in_chunk] - col_start] -= g[in_chunk]
            grad_z16 = grad_z.to(hidden.dtype)
            if need_h:
                torch.addmm(grad_h, grad_z16, w_c, out_dtype=torch.float32, out=grad_h)
            if need_w:
                grad_w[col_start:col_end] = torch.mm(grad_z16.t(), hidden, out_dtype=torch.float32).to(weight.dtype)

        return (
            grad_h.to(hidden.dtype) if need_h else None,
            grad_w if need_w else None,
            None,
            None,
            None,
        )


def bi_fused_per_token_ce(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    temperature: float | torch.Tensor = 1.0,
    top_ks: torch.Tensor | None = None,
    top_ps: torch.Tensor | None = None,
    min_ps: torch.Tensor | None = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> torch.Tensor:
    """Per-token CE (``-log p(labels)``; 0 at ignored positions) through the
    batch-invariant lm-head contract. Requires CUDA bf16 hidden/weight; the
    fp32-class numerics come from the contract itself, so ``lm_head_fp32`` is
    implied rather than materialized. Per-row temperature scores the same
    materialized FP32 ``z * (1/T)`` tensor as serving."""
    if not hidden_states.is_cuda:
        raise ValueError("ce_mode='bi_fused' requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("ce_mode='bi_fused' requires bf16 hidden states and lm-head weight")
    temp_row = normalize_temperature_rows(
        temperature,
        rows=hidden_states.shape[0],
        device=hidden_states.device,
    )
    top_ks, top_ps, min_ps = validate_sampling_transform_rows(
        top_ks,
        top_ps,
        min_ps,
        rows=hidden_states.shape[0],
        device=hidden_states.device,
    )
    if temp_row is not None or top_ks is not None:
        return chunked_transform_scored_ce(
            hidden_states,
            weight,
            labels,
            ignore_index=ignore_index,
            temperature_rows=temp_row,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            policy=_bi_scoring_policy(vocab_chunk),
        )
    valid_mask = labels != ignore_index
    labels_safe = torch.where(valid_mask, labels, torch.zeros_like(labels))
    return _BiFusedLmHeadPerTokenCE.apply(
        hidden_states,
        weight,
        labels_safe,
        valid_mask,
        vocab_chunk,
    )


def bi_fused_vocab_parallel_per_token_ce(
    hidden_states: torch.Tensor,
    local_weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    ignore_index: int = -100,
    temperature: float | torch.Tensor = 1.0,
    top_ks: torch.Tensor | None = None,
    top_ps: torch.Tensor | None = None,
    min_ps: torch.Tensor | None = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> torch.Tensor:
    """Run the batch-invariant LM head over rank-local vocab and token shards.

    LM-head-only TP is composed from DP/CP row owners, so ranks can own
    different (or zero) token rows while also owning disjoint vocab rows.  The
    forward broadcasts bounded chunks from each row owner, computes the exact
    local FP32 logits, gathers vocabulary shards in group-rank order, and
    scores through the same from-logits tail as serving.  Only the source
    owner's rows are returned.  The custom backward reduces hidden gradients
    over vocab ranks and leaves each weight-shard gradient local for the
    existing replica sync.
    """

    if not hidden_states.is_cuda or not local_weight.is_cuda:
        raise ValueError("ce_mode='bi_fused' tensor parallelism requires CUDA tensors")
    if hidden_states.ndim != 2 or local_weight.ndim != 2:
        raise ValueError("ce_mode='bi_fused' tensor parallelism requires two-dimensional hidden and weight")
    if hidden_states.shape[1] != local_weight.shape[1]:
        raise ValueError("ce_mode='bi_fused' tensor parallelism received mismatched hidden dimensions")
    if hidden_states.dtype is not torch.bfloat16 or local_weight.dtype is not torch.bfloat16:
        raise ValueError("ce_mode='bi_fused' tensor parallelism requires bf16 hidden states and lm-head weight")
    if labels.ndim != 1 or labels.shape[0] != hidden_states.shape[0]:
        raise ValueError("ce_mode='bi_fused' tensor parallelism requires one label per local hidden row")
    hidden_states = hidden_states.contiguous()
    local_weight = local_weight.contiguous()
    labels = labels.contiguous()
    valid_mask = labels != ignore_index
    labels_safe = torch.where(valid_mask, labels, torch.zeros_like(labels))

    temp_row = normalize_temperature_rows(
        temperature,
        rows=hidden_states.shape[0],
        device=hidden_states.device,
    )
    top_ks, top_ps, min_ps = validate_sampling_transform_rows(
        top_ks,
        top_ps,
        min_ps,
        rows=hidden_states.shape[0],
        device=hidden_states.device,
    )

    return _BiFusedVocabParallelPerTokenCE.apply(
        hidden_states,
        local_weight,
        labels_safe,
        valid_mask,
        temp_row,
        top_ks,
        top_ps,
        min_ps,
        tp_group,
        vocab_chunk,
    )
