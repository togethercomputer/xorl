"""Chunked fused selected-token log-probability for the LM head.

Baseline per-chunk path
-----------------------
The chunked baseline (:mod:`vocab_parallel_cross_entropy`) computes, per chunk of
``N`` token rows, the output-head matmul ``h @ W.T``, a log-softmax, a
selected-token gather, and a tensor-parallel all-reduce — but it materializes the
full ``[N, V_local]`` logits tile in HBM and runs the reduction in several passes.

This module
-----------
Keeps the per-chunk logits tile chunk-sized (never the full ``[N, V_local]``) and
fuses the matmul + log-softmax + selected-token gather using the quack CuTeDSL
cross-entropy kernel on each chunk, while doing the GEMMs through cuBLAS
(``torch.mm``) so the matmul stays at library throughput. Tensor-parallel
semantics are preserved exactly by reducing only per-token scalars:

  * ``lse_r`` — this rank's local log-sum-exp over its vocab shard ``V_r``;
  * ``u_r``   — this rank's selected target logit (0 if ``y_t`` not in ``V_r``).

The host merges them with three ``[N]`` scalar reductions (no ``[N, V_local]``
tensor on the wire):

  * ``m_t  = AllReduce_max(lse_r)``
  * ``lse_t = m_t + log AllReduce_sum(exp(lse_r - m_t))``   (= ``logsumexp_r(lse_r)``)
  * ``s_t  = AllReduce_sum(u_r)``

and ``log p(y_t) = s_t - lse_t``, identical to the full-vocab path. The function
returns per-token cross-entropy ``-log p(y_t)`` so it is a drop-in replacement for
:func:`vocab_parallel_cross_entropy`.

LoRA backward-pass fix
----------------------
Backward recomputes the per-chunk logits (chunked cuBLAS matmul), forms
``grad_z = grad_ell · (softmax(z) - onehot(y)) · (1/T)`` against the *global* lse,
and respects ``ctx.needs_input_grad``: ``grad_h`` only if the hidden states need
grad (and only then the TP all-reduce for ``grad_h``), ``grad_W`` only if the
weight needs grad, ``grad_b`` only if the bias needs grad. With a frozen output
layer (LoRA RL) ``grad_W`` and ``grad_b`` are skipped entirely.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d

from xorl.ops.loss.vocab_parallel_cross_entropy import _get_vocab_shard_offset


# Default per-chunk token count. Bounds the logits tile to [chunk, V_local].
_DEFAULT_CHUNK = 8192


try:
    from xorl.ops.quack.cross_entropy import cross_entropy_bwd_out as _quack_ce_bwd_out
    from xorl.ops.quack.cross_entropy import cross_entropy_fwd_out as _quack_ce_fwd_out

    _HAS_QUACK = True
except Exception:  # pragma: no cover - quack/CuTeDSL optional
    _HAS_QUACK = False


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """Unwrap a functional-collective result into a plain dense tensor.

    ``funcol.all_reduce`` returns an ``AsyncCollectiveTensor`` whose collective is
    completed lazily by the next torch op; force the wait so downstream consumers
    (and saved-for-backward tensors) see materialized data.
    """
    if isinstance(t, funcol.AsyncCollectiveTensor):
        return t.wait()
    return t


def _quack_usable(logits: torch.Tensor) -> bool:
    """The quack CuTeDSL CE kernel needs CUDA + a vocab dim divisible by 8."""
    return (
        _HAS_QUACK and logits.is_cuda and logits.dtype in (torch.float16, torch.bfloat16) and logits.shape[1] % 8 == 0
    )


def _chunk_logits(x_chunk, weight, bias, inv_temp):
    """logits = (x @ W^T + b) / T for one chunk, in the inputs' native dtype (cuBLAS)."""
    logits = x_chunk @ weight.t()
    if bias is not None:
        logits = logits + bias
    if inv_temp != 1.0:
        logits = logits * inv_temp
    return logits


def _chunk_lse(logits_native: torch.Tensor, safe_t: torch.Tensor, ignore_index: int):
    """Per-row log-sum-exp for one chunk, fp32.

    Uses the fused quack CuTeDSL cross-entropy kernel when available (one pass over
    the chunk, no fp32 logits tile), else ``torch.logsumexp`` over an fp32 upcast.
    """
    if _quack_usable(logits_native):
        loss = torch.empty(logits_native.shape[0], device=logits_native.device, dtype=torch.float32)
        lse = torch.empty_like(loss)
        _quack_ce_fwd_out(logits_native, safe_t.to(torch.int32), None, loss, lse, None, ignore_index)
        return lse
    return torch.logsumexp(logits_native.float(), dim=-1)


def _chunked_forward(hidden, weight, bias, labels, vocab_offset, ignore_index, inv_temp, chunk_size):
    """Per-rank forward: returns (lse_r [N], target_logit_r [N]) over this vocab shard.

    The matmul is chunked cuBLAS; the logits tile is only ever [chunk, V_local].
    """
    N = hidden.shape[0]
    V_local = weight.shape[0]
    device = hidden.device
    lse_r = torch.empty(N, device=device, dtype=torch.float32)
    target_logit_r = torch.zeros(N, device=device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        labels_chunk = labels[start:end]
        logits = _chunk_logits(hidden[start:end], weight, bias, inv_temp)

        local_t = labels_chunk - vocab_offset
        valid = labels_chunk != ignore_index
        found = valid & (local_t >= 0) & (local_t < V_local)
        safe_t = local_t.clamp(0, V_local - 1)

        lse_r[start:end] = _chunk_lse(logits, safe_t, ignore_index)
        sel = logits.gather(1, safe_t.unsqueeze(1)).squeeze(1).float()
        target_logit_r[start:end] = sel * found.to(torch.float32)

    return lse_r, target_logit_r


def _chunk_grad_logits(logits, safe_t, found, glse_chunk, dloss, tp_active):
    """grad w.r.t. the (pre-temperature-scaled) logits for one chunk.

    ``dlogits = dloss * (softmax(logits; global_lse) - onehot(target))`` in the
    inputs' native dtype. Uses the fused quack CuTeDSL backward kernel (one pass,
    global lse injected) when available, else a torch fallback. Under TP, rows
    whose target is not in this shard (``~found``) must keep the softmax mass but
    no onehot, so the kernel's onehot subtraction at the clamped index is added
    back. Without TP every valid row is owned (``found == valid``) and ignored
    rows carry ``dloss == 0``, so the add-back is a no-op and is skipped.
    """
    if _quack_usable(logits):
        dlogits = torch.empty_like(logits)
        _quack_ce_bwd_out(logits, safe_t.to(torch.int32), dloss, glse_chunk, dlogits, -1)
        if tp_active:
            not_owned = ~found
            if bool(not_owned.any()):
                rows = not_owned.nonzero(as_tuple=True)[0]
                dlogits[rows, safe_t[rows]] += dloss[rows].to(dlogits.dtype)
        return dlogits
    # torch fallback: single fp32 [chunk, V_local] buffer, in place.
    dlogits = logits.float()
    dlogits.sub_(glse_chunk.unsqueeze(1)).exp_()
    dlogits.scatter_add_(1, safe_t.unsqueeze(1), -found.to(torch.float32).unsqueeze(1))
    dlogits.mul_(dloss.unsqueeze(1))
    return dlogits.to(logits.dtype)


def _chunked_backward(
    grad_out,
    hidden,
    weight,
    bias,
    labels,
    global_lse,
    vocab_offset,
    ignore_index,
    inv_temp,
    chunk_size,
    needs,
    tp_active,
):
    """Per-rank backward. Returns (grad_h, grad_w, grad_b); entries None when not needed."""
    needs_h, needs_w, needs_b = needs
    N, H = hidden.shape
    V_local = weight.shape[0]
    device = hidden.device

    grad_h = torch.zeros((N, H), device=device, dtype=torch.float32) if needs_h else None
    grad_w = torch.zeros((V_local, H), device=device, dtype=torch.float32) if needs_w else None
    grad_b = torch.zeros((V_local,), device=device, dtype=torch.float32) if needs_b else None

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_chunk = hidden[start:end]
        labels_chunk = labels[start:end]
        glse_chunk = global_lse[start:end]
        go_chunk = grad_out[start:end]

        local_t = labels_chunk - vocab_offset
        valid = labels_chunk != ignore_index
        found = valid & (local_t >= 0) & (local_t < V_local)
        safe_t = local_t.clamp(0, V_local - 1)

        logits = _chunk_logits(x_chunk, weight, bias, inv_temp)
        dloss = go_chunk * valid.to(torch.float32) * inv_temp  # 0 for ignored rows
        dlogits_cast = _chunk_grad_logits(logits, safe_t, found, glse_chunk, dloss, tp_active)

        if needs_h:
            grad_h[start:end] = (dlogits_cast @ weight).float()
        if needs_w:
            grad_w += dlogits_cast.t() @ x_chunk
        if needs_b:
            grad_b += dlogits_cast.float().sum(0)

    return grad_h, grad_w, grad_b


class _FusedSelectedLogProb(torch.autograd.Function):
    """Autograd for chunked fused selected-token cross-entropy with vocab-parallel TP.

    Forward returns per-token cross-entropy ``-log p(y_t)``. Backward respects
    ``ctx.needs_input_grad`` and only all-reduces ``grad_h`` when it is requested.
    """

    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, tp_group, vocab_offset, ignore_index, inv_temp, chunk_size):
        lse_r, target_logit_r = _chunked_forward(
            hidden, weight, bias, labels, vocab_offset, ignore_index, inv_temp, chunk_size
        )

        tp_active = tp_group is not None and dist.get_world_size(tp_group) > 1
        if tp_active:
            # global log-sum-exp via logsumexp_r(lse_r) = m + log sum_r exp(lse_r - m),
            # and selected logit via a sum reduction. Three [N] scalar reductions.
            m = _materialize(funcol.all_reduce(lse_r, reduceOp=c10d.ReduceOp.MAX.name, group=tp_group))
            sum_exp = _materialize(
                funcol.all_reduce(torch.exp(lse_r - m), reduceOp=c10d.ReduceOp.SUM.name, group=tp_group)
            )
            global_lse = m + torch.log(sum_exp)
            s = _materialize(funcol.all_reduce(target_logit_r, reduceOp=c10d.ReduceOp.SUM.name, group=tp_group))
        else:
            global_lse, s = lse_r, target_logit_r

        valid = (labels != ignore_index).to(global_lse.dtype)
        per_token_ce = (global_lse - s) * valid

        ctx.save_for_backward(hidden, weight, bias, labels, global_lse)
        ctx.tp_group = tp_group
        ctx.vocab_offset = vocab_offset
        ctx.ignore_index = ignore_index
        ctx.inv_temp = inv_temp
        ctx.chunk_size = chunk_size
        ctx.has_bias = bias is not None
        return per_token_ce

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias, labels, global_lse = ctx.saved_tensors
        needs_h = ctx.needs_input_grad[0]
        needs_w = ctx.needs_input_grad[1]
        needs_b = ctx.needs_input_grad[2] and ctx.has_bias

        if not (needs_h or needs_w or needs_b):
            return None, None, None, None, None, None, None, None, None

        tp_active = ctx.tp_group is not None and dist.get_world_size(ctx.tp_group) > 1
        grad_output = grad_output.contiguous().float()
        grad_h, grad_w, grad_b = _chunked_backward(
            grad_output,
            hidden,
            weight,
            bias,
            labels,
            global_lse,
            ctx.vocab_offset,
            ctx.ignore_index,
            ctx.inv_temp,
            ctx.chunk_size,
            (needs_h, needs_w, needs_b),
            tp_active,
        )

        if needs_h:
            if tp_active:
                grad_h = _materialize(funcol.all_reduce(grad_h, reduceOp=c10d.ReduceOp.SUM.name, group=ctx.tp_group))
            grad_h = grad_h.to(hidden.dtype)
        grad_w = grad_w.to(weight.dtype) if grad_w is not None else None
        grad_b = grad_b.to(bias.dtype) if grad_b is not None else None

        return grad_h, grad_w, grad_b, None, None, None, None, None, None


def fused_selected_logprob_ce(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    temperature: float = 1.0,
    chunk_size: int = _DEFAULT_CHUNK,
) -> torch.Tensor:
    """Per-token cross-entropy ``-log p(y_t)`` via chunked cuBLAS matmul + fused CE.

    Drop-in replacement for :func:`vocab_parallel_cross_entropy` /
    ``F.cross_entropy(reduction="none")`` that bounds the logits tile to
    ``[chunk, V_local]`` and reduces only three ``[N]`` scalar vectors across TP.

    Args:
        hidden_states: ``[BT, H]`` hidden states (replicated across TP ranks).
        weight: ``[V/tp, H]`` local lm_head weight shard.
        labels: ``[BT]`` global target token ids (replicated across TP ranks).
        tp_group: TP process group. ``None`` (or world size 1) => no reduction.
        bias: optional ``[V/tp]`` local lm_head bias shard.
        ignore_index: label value to ignore (default: -100).
        temperature: softmax temperature ``T`` (default: 1.0).
        chunk_size: token rows processed per chunk (bounds logits memory).

    Returns:
        ``[BT]`` per-token cross-entropy (replicated across TP ranks).
    """
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels_1d = labels.reshape(-1)
    if hidden_2d.is_cuda:
        hidden_2d = hidden_2d.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        labels_1d = labels_1d.contiguous()

    local_vocab_size = weight.shape[0]
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        vocab_offset = _get_vocab_shard_offset(local_vocab_size, tp_group, weight.device)
    else:
        vocab_offset = 0

    inv_temp = 1.0 / float(temperature)
    return _FusedSelectedLogProb.apply(
        hidden_2d,
        weight,
        bias,
        labels_1d,
        tp_group,
        vocab_offset,
        ignore_index,
        inv_temp,
        int(chunk_size),
    )
