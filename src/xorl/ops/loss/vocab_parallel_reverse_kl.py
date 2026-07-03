"""Vocab-parallel full-vocab reverse-KL for OPD distillation.

When the lm_head weight is row-sharded along the vocab dimension across a
process group (each rank holds vocab_size/world rows), we compute the
full-vocab reverse-KL KL(p_student || p_teacher) WITHOUT gathering the full
lm_head or the full logits onto any rank. This is the OPD analogue of
``vocab_parallel_cross_entropy``: it removes the full-lm_head FSDP all-gather
(the 1-node OPD memory blocker) and shrinks each rank's logits by `world`x, so
each rank can materialize its [N, V/world] logits in a single fast matmul.

Algorithm (per token, both student `s` and teacher `t` logits):
    1. local_logits = hidden @ local_weight.T            # [N, V/world]
    2. global_max = all_reduce(local_max, MAX)           # numerically-stable shift
    3. local_sumexp = sum(exp(local_logits - global_max))
       global_sumexp = all_reduce(local_sumexp, SUM)     # => Z (per student/teacher)
    4. Reverse-KL decomposes (Σ_v p_s = 1) as
           KL = Σ_v p_s(v)·(s_v - t_v) - logZ_s + logZ_t,
       and  Σ_v p_s(v)·(s_v - t_v) = A / Z_s   with
           A = Σ_v exp(s_v - s_global_max)·(s_v - t_v).
       A is a plain vocab sum, so A_local + all_reduce(SUM) gives the global A.
    5. KL = A/Z_s - (s_global_max + log Z_s) + (t_global_max + log Z_t).

Backward: d KL / d s_v = p_s(v)·(log p_s(v) - log p_t(v) - KL), recomputed from
the saved tiny [N,1] normalizers + the local shards; grad_hidden is summed
across ranks (all_reduce SUM), grad_weight is purely local (this rank's shard).
The teacher is frozen (no grad). Functional collectives keep it compile-traceable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


def _resolve_group(group):
    """funcol requires a concrete group; map None to the default process group."""
    return group if group is not None else c10d._get_default_group()


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """Force functional collective outputs to complete before saving/using them."""
    if isinstance(t, funcol.AsyncCollectiveTensor):
        return t.wait()
    return t


def _vp_reverse_kl_forward(
    student_hidden: torch.Tensor,  # [N, H]
    student_weight_local: torch.Tensor,  # [V/world, H]
    teacher_hidden: torch.Tensor,  # [N, H]
    teacher_weight_local: torch.Tensor,  # [V/world, H]
    labels: torch.Tensor,  # [N]
    ignore_index: int,
    group,
):
    """Returns (kl[N], s_max[N,1], s_sumexp[N,1], t_max[N,1], t_sumexp[N,1])."""
    group = _resolve_group(group)
    s_logits = (student_hidden @ student_weight_local.t()).float()  # [N, V/world]
    t_logits = (teacher_hidden @ teacher_weight_local.t()).float()

    # Combined MAX all-reduce for the student + teacher normalizer shift.
    local_max = torch.cat(
        [s_logits.max(dim=-1, keepdim=True).values, t_logits.max(dim=-1, keepdim=True).values],
        dim=-1,
    )  # [N, 2]
    global_max = _materialize(funcol.all_reduce(local_max, reduceOp=c10d.ReduceOp.MAX.name, group=group))
    s_max = global_max[:, 0:1]
    t_max = global_max[:, 1:2]

    s_exp = (s_logits - s_max).exp()  # [N, V/world]
    t_exp = (t_logits - t_max).exp()
    s_local_sumexp = s_exp.sum(dim=-1, keepdim=True)
    t_local_sumexp = t_exp.sum(dim=-1, keepdim=True)
    # A_local = Σ_local exp(s - s_max)·(s - t)
    a_local = (s_exp * (s_logits - t_logits)).sum(dim=-1, keepdim=True)

    # Single SUM all-reduce for [s_sumexp, t_sumexp, A].
    local_sums = torch.cat([s_local_sumexp, t_local_sumexp, a_local], dim=-1)  # [N, 3]
    global_sums = _materialize(funcol.all_reduce(local_sums, reduceOp=c10d.ReduceOp.SUM.name, group=group))
    s_sumexp = global_sums[:, 0:1]
    t_sumexp = global_sums[:, 1:2]
    a_global = global_sums[:, 2:3]

    s_logz = s_sumexp.log() + s_max
    t_logz = t_sumexp.log() + t_max
    kl = (a_global / s_sumexp - s_logz + t_logz).squeeze(-1)  # [N]
    valid = labels != ignore_index
    kl = kl * valid.to(kl.dtype)
    return kl, s_max, s_sumexp, t_max, t_sumexp


def _vp_reverse_kl_backward(
    grad_output: torch.Tensor,  # [N]
    student_hidden: torch.Tensor,  # [N, H]
    student_weight_local: torch.Tensor,  # [V/world, H]
    teacher_hidden: torch.Tensor,
    teacher_weight_local: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    s_max: torch.Tensor,  # [N,1] global
    s_sumexp: torch.Tensor,  # [N,1] global
    t_max: torch.Tensor,
    t_sumexp: torch.Tensor,
    kl: torch.Tensor,  # [N] global (masked)
    group,
    needs_hidden_grad: bool,
    needs_weight_grad: bool,
):
    group = _resolve_group(group)
    s_logits = (student_hidden @ student_weight_local.t()).float()
    t_logits = (teacher_hidden @ teacher_weight_local.t()).float()

    s_logz = s_sumexp.log() + s_max
    t_logz = t_sumexp.log() + t_max
    s_log_probs = s_logits - s_logz  # [N, V/world]
    t_log_probs = t_logits - t_logz
    s_probs = (s_logits - s_max).exp() / s_sumexp  # softmax, global-normalized

    valid = (labels != ignore_index).to(dtype=torch.float32)
    scale = (grad_output.to(torch.float32) * valid).unsqueeze(1)  # [N,1]

    # d KL / d s_logits = p_s · (log p_s - log p_t - KL)
    grad_logits = s_probs * (s_log_probs - t_log_probs - kl.unsqueeze(1))
    grad_logits = grad_logits * scale  # [N, V/world]

    grad_hidden = None
    if needs_hidden_grad:
        # sum over ranks of local grad_logits @ local_weight
        grad_hidden = grad_logits.to(student_weight_local.dtype) @ student_weight_local  # [N,H]
        grad_hidden = _materialize(funcol.all_reduce(grad_hidden, reduceOp=c10d.ReduceOp.SUM.name, group=group))
        grad_hidden = grad_hidden.to(student_hidden.dtype)

    grad_weight = None
    if needs_weight_grad:
        grad_weight = grad_logits.to(student_hidden.dtype).t() @ student_hidden  # [V/world, H], local only
        grad_weight = grad_weight.to(student_weight_local.dtype)

    return grad_hidden, grad_weight


class _VocabParallelReverseKL(torch.autograd.Function):
    """Full-vocab reverse-KL over a vocab-sharded lm_head, no full gather."""

    @staticmethod
    def forward(
        ctx,
        student_hidden: torch.Tensor,
        student_weight_local: torch.Tensor,
        teacher_hidden: torch.Tensor,
        teacher_weight_local: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        group,
    ) -> torch.Tensor:
        kl, s_max, s_sumexp, t_max, t_sumexp = _vp_reverse_kl_forward(
            student_hidden,
            student_weight_local,
            teacher_hidden,
            teacher_weight_local,
            labels,
            ignore_index,
            group,
        )
        ctx.save_for_backward(
            student_hidden,
            student_weight_local,
            teacher_hidden,
            teacher_weight_local,
            labels,
            s_max,
            s_sumexp,
            t_max,
            t_sumexp,
            kl,
        )
        ctx.ignore_index = ignore_index
        ctx.group = group
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            student_hidden,
            student_weight_local,
            teacher_hidden,
            teacher_weight_local,
            labels,
            s_max,
            s_sumexp,
            t_max,
            t_sumexp,
            kl,
        ) = ctx.saved_tensors
        grad_hidden, grad_weight = _vp_reverse_kl_backward(
            grad_output,
            student_hidden,
            student_weight_local,
            teacher_hidden,
            teacher_weight_local,
            labels,
            ctx.ignore_index,
            s_max,
            s_sumexp,
            t_max,
            t_sumexp,
            kl,
            ctx.group,
            needs_hidden_grad=ctx.needs_input_grad[0],
            needs_weight_grad=ctx.needs_input_grad[1],
        )
        return grad_hidden, grad_weight, None, None, None, None, None


def _gather_counts(n_local: int, *, device: torch.device, group, world: int) -> torch.Tensor:
    local_count = torch.tensor([n_local], dtype=torch.long, device=device)
    counts = torch.empty(world, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(counts, local_count, group=group)
    return counts


def _padded_all_gather_2d(local_tensor: torch.Tensor, counts: torch.Tensor, group, world: int) -> torch.Tensor:
    if local_tensor.ndim != 2:
        raise ValueError(f"expected rank-2 local hidden tensor, got shape {tuple(local_tensor.shape)}")
    max_count = int(counts.max().item()) if counts.numel() else 0
    hdim = local_tensor.shape[-1]
    padded = local_tensor.new_zeros((max_count, hdim))
    n_local = local_tensor.shape[0]
    if n_local:
        padded[:n_local].copy_(local_tensor.contiguous())
    gathered = local_tensor.new_empty((world * max_count, hdim))
    dist.all_gather_into_tensor(gathered, padded.contiguous(), group=group)
    pieces = []
    counts_list = [int(x) for x in counts.detach().cpu().tolist()]
    for rank, n_rows in enumerate(counts_list):
        if n_rows:
            start = rank * max_count
            pieces.append(gathered[start : start + n_rows])
    if not pieces:
        return local_tensor.new_empty((0, hdim))
    return torch.cat(pieces, dim=0)


class _GatherHiddenPaddedLocalGrad(torch.autograd.Function):
    """Gather uneven token slices and return grad ONLY for this rank's slice.

    Integration glue for using the vocab-parallel KL under FSDP, where the lm-head
    shard group ALSO data-shards the tokens: each rank holds its own [n_local, H]
    hidden states, but the vocab-parallel KL needs every rank to see ALL tokens
    (each rank then owns its vocab shard). We gather the (cheap) activations rather
    than the (large) weight. Because each token is owned by exactly one rank, the
    backward returns grad only for the local slice — so the model's transformer
    backward on this rank receives exactly its tokens' grad, with no cross-rank
    double-counting. Local token counts may differ; forward pads to the group max
    for the collective and removes padding before returning the full token set.
    """

    @staticmethod
    def forward(ctx, local_hidden: torch.Tensor, counts: torch.Tensor, group, world: int, rank: int) -> torch.Tensor:
        group = _resolve_group(group)
        counts_cpu = counts.detach().cpu()
        offsets = torch.cat([counts_cpu.new_zeros(1), counts_cpu.cumsum(0)])
        ctx.lo = int(offsets[rank].item())
        ctx.hi = int(offsets[rank + 1].item())
        return _padded_all_gather_2d(local_hidden, counts, group, world)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        return grad_full[ctx.lo : ctx.hi].contiguous(), None, None, None, None


def _gather_hidden_nograd(local_hidden: torch.Tensor, counts: torch.Tensor, group, world: int) -> torch.Tensor:
    return _padded_all_gather_2d(local_hidden, counts, group, world)


def _gather_labels_nograd(
    local_labels: torch.Tensor,
    counts: torch.Tensor,
    *,
    ignore_index: int,
    group,
    world: int,
) -> torch.Tensor:
    if local_labels.ndim != 1:
        raise ValueError(f"expected rank-1 local labels, got shape {tuple(local_labels.shape)}")
    max_count = int(counts.max().item()) if counts.numel() else 0
    padded = torch.full((max_count,), ignore_index, dtype=local_labels.dtype, device=local_labels.device)
    n_local = local_labels.shape[0]
    if n_local:
        padded[:n_local].copy_(local_labels.contiguous())
    gathered = torch.empty(world * max_count, dtype=local_labels.dtype, device=local_labels.device)
    dist.all_gather_into_tensor(gathered, padded.contiguous(), group=group)
    pieces = []
    counts_list = [int(x) for x in counts.detach().cpu().tolist()]
    for rank, n_rows in enumerate(counts_list):
        if n_rows:
            start = rank * max_count
            pieces.append(gathered[start : start + n_rows])
    if not pieces:
        return local_labels.new_empty((0,))
    return torch.cat(pieces, dim=0)


def vocab_parallel_reverse_kl_gathered(
    local_student_hidden: torch.Tensor,
    student_weight_local: torch.Tensor,
    local_teacher_hidden: torch.Tensor,
    teacher_weight_local: torch.Tensor,
    labels_full: Optional[torch.Tensor] = None,
    local_labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Vocab-parallel reverse-KL when the group ALSO data-shards tokens.

    Each rank passes its own ``local_*_hidden`` ([n_local, H]) + its vocab shard of
    the lm-head; this gathers all tokens' activations (cheap), runs the
    vocab-parallel KL over the local vocab shard, and returns the full per-token KL
    [N] (replicated). Student-hidden grad flows back only to the local slice.
    ``labels_full`` is the all-token label vector when the caller already has it.
    Otherwise pass ``local_labels`` and this helper gathers uneven local label slices
    with the same padding layout as the hidden states.
    """
    g = _resolve_group(group)
    world = dist.get_world_size(g)
    rank = dist.get_rank(g)
    if local_student_hidden.shape[0] != local_teacher_hidden.shape[0]:
        raise ValueError(
            "local student/teacher hidden token counts must match, got "
            f"{local_student_hidden.shape[0]} and {local_teacher_hidden.shape[0]}"
        )
    if local_labels is not None and local_labels.shape[0] != local_student_hidden.shape[0]:
        raise ValueError(
            "local_labels must have the same token count as local_student_hidden, got "
            f"{local_labels.shape[0]} and {local_student_hidden.shape[0]}"
        )
    counts = _gather_counts(local_student_hidden.shape[0], device=local_student_hidden.device, group=g, world=world)
    full_student = _GatherHiddenPaddedLocalGrad.apply(local_student_hidden, counts, g, world, rank)
    full_teacher = _gather_hidden_nograd(local_teacher_hidden, counts, g, world)
    if labels_full is None:
        if local_labels is None:
            raise ValueError("vocab_parallel_reverse_kl_gathered requires labels_full or local_labels")
        labels_full = _gather_labels_nograd(local_labels, counts, ignore_index=ignore_index, group=g, world=world)
    expected_tokens = int(counts.sum().item())
    if labels_full.shape[0] != expected_tokens:
        raise ValueError(f"labels_full has {labels_full.shape[0]} rows, expected {expected_tokens}")
    return vocab_parallel_reverse_kl_function(
        full_student,
        student_weight_local,
        full_teacher,
        teacher_weight_local,
        labels_full,
        ignore_index,
        g,
    )


def vocab_parallel_reverse_kl_function(
    student_hidden_states: torch.Tensor,
    student_weight_local: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight_local: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Per-token reverse-KL [N] over a vocab-sharded lm_head.

    ``student_weight_local`` / ``teacher_weight_local`` are this rank's
    contiguous vocab-row shard ([V/world, H]); the shards across the group must
    partition the full vocab. ``group`` is the vocab-shard process group
    (e.g. the FSDP shard group); None => the default group.
    """
    return _VocabParallelReverseKL.apply(
        student_hidden_states,
        student_weight_local,
        teacher_hidden_states,
        teacher_weight_local,
        labels,
        ignore_index,
        group,
    )
