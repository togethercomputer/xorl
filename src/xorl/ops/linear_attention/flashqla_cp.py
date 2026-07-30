"""FlashQLA Gated Delta Rule under xorl's native Ulysses/sequence-parallel CP.

FlashQLA's fused TileLang kernels are single-GPU; xorl's GDN context-parallelism
(``ops/cp``) is a distributed wrapper that, per rank, computes a compact boundary
state ``hm = [h_e | M]`` (local accumulated state + transition matrix), all-gathers
it, and chains it across ranks with a generic merge kernel to produce each rank's
*corrected* ``initial_state`` (forward) / ``dht`` (backward). The per-rank kernel is
CP-unaware — it only consumes the corrected boundary tensors.

This module reuses that orchestration **unchanged** and substitutes FlashQLA only for
the heavy per-rank interior compute:

* the cheap boundary ``hm``/``dhm`` (computed from the WY representation ``w``/``u``),
  the all-gather + merge, and ``compress_h0``/``expand_h0`` stay on the FLA path
  (``ops/cp/chunk_delta_h.py``); these are O(#chunks) ``K×K`` matrices and are already
  proven correct;
* the expensive forward output (``fused_gdr_fwd``) and backward
  (``fused_gdr_h`` recompute + ``fused_gdr_bwd``) run on FlashQLA, fed the corrected
  ``initial_state`` / ``dht``.

Because FLA's backward pre-process recomputes the transition matrix itself, FlashQLA is
never asked for a transition-matrix gradient (``dmt``) — so no TileLang kernel changes
are needed; this is pure Python orchestration.

The single-card FlashQLA path (no ``cp_context``) and the FLA path are untouched.
Hopper (SM90) only; requires ``tilelang`` with the ``tl_gemm`` builtin + PR #2303.
"""

from __future__ import annotations

import torch

from xorl.ops.linear_attention.flashqla.ops.gated_delta_rule.chunk import robust_kkt_solve
from xorl.ops.linear_attention.flashqla.ops.gated_delta_rule.chunk.hopper import (
    fused_gdr_bwd,
    fused_gdr_fwd,
    fused_gdr_h,
)
from xorl.ops.linear_attention.flashqla.ops.utils import chunk_local_cumsum as flashqla_chunk_local_cumsum
from xorl.ops.linear_attention.modules.l2norm import l2norm_bwd, l2norm_fwd
from xorl.ops.linear_attention.ops.common.chunk_o import chunk_bwd_dv_local
from xorl.ops.linear_attention.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from xorl.ops.linear_attention.ops.cp import FLACPContext
from xorl.ops.linear_attention.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
    compress_h0,
    expand_h0,
)
from xorl.ops.linear_attention.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from xorl.ops.linear_attention.ops.utils import chunk_local_cumsum, solve_tril
from xorl.ops.linear_attention.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _flashqla_cp_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor,
    cp_context: FLACPContext,
):
    # Cumsum the raw gate with each library's own convention (never double-applied on
    # the same tensor): g_fla feeds the FLA boundary pre-process, g_qla the FlashQLA
    # interior. They are mathematically the same chunk-local cumsum.
    g_fla = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    g_qla = flashqla_chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    # FLA WY representation (w, u) — needed to build the cross-rank boundary state.
    A_fla = chunk_scaled_dot_kkt_fwd(k=k, g=g_fla, beta=beta, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    A_fla = solve_tril(A=A_fla, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_fla, g=g_fla, cu_seqlens=cu_seqlens)

    # Cross-rank boundary: all_gather local hm + merge -> corrected per-rank initial state.
    initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
        k=k,
        w=w,
        u=u,
        g=g_fla,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        context=cp_context,
    )

    # FlashQLA heavy interior: outputs computed from the corrected boundary state.
    A_qla = robust_kkt_solve(k=k, b=beta, cu_seqlens=cu_seqlens)
    o, _, _ = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A_qla,
        g=g_qla,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        output_h=False,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=None,
        raw_cu_seqlens=None,
    )

    # Compress the boundary state for backward storage (only the leading partial seq
    # carries a non-zero cross-rank state; interior seqs start fresh).
    initial_state = compress_h0(initial_state, context=cp_context)
    return g_fla, g_qla, A_fla, A_qla, o, initial_state


def _flashqla_cp_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_fla: torch.Tensor,
    g_qla: torch.Tensor,
    beta: torch.Tensor,
    A_fla: torch.Tensor,
    A_qla: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    cp_context: FLACPContext,
):
    # Restore the per-rank boundary state used in the forward.
    initial_state = expand_h0(initial_state, context=cp_context)

    # Recompute the per-chunk states with the corrected boundary state (matches fwd).
    h, _, _ = fused_gdr_h(
        k=k,
        v=v,
        a=A_qla,
        g=g_qla,
        b=beta,
        initial_state=initial_state,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cu_seqlens,
    )

    # Cross-rank boundary gradient: all_gather local dhm + reverse-merge -> corrected dht.
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_fla, g=g_fla, cu_seqlens=cu_seqlens)
    dv_local = chunk_bwd_dv_local(q=q, k=k, g=g_fla, do=do, scale=scale, cu_seqlens=cu_seqlens)
    dht, _ = chunk_gated_delta_rule_bwd_dhu_pre_process(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv_local,
        g=g_fla,
        scale=scale,
        cu_seqlens=cu_seqlens,
        dht=None,
        initial_state=initial_state,
        context=cp_context,
    )

    # FlashQLA heavy interior backward, fed the corrected boundary gradient.
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=A_qla,
        g=g_qla,
        b=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dg = flashqla_chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleCPFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor,
        use_qk_l2norm_in_kernel: bool,
        cp_context: FLACPContext,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g_fla, g_qla, A_fla, A_qla, o, initial_state = _flashqla_cp_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_fla, g_qla, beta, A_fla, A_qla, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context.copy_for_backward()
        return o.to(q.dtype), None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, _dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, g_fla, g_qla, beta, A_fla, A_qla, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, _dh0 = _flashqla_cp_bwd(
            q=q,
            k=k,
            v=v,
            g_fla=g_fla,
            g_qla=g_qla,
            beta=beta,
            A_fla=A_fla,
            A_qla=A_qla,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        # grads align with forward inputs: q, k, v, g, beta, scale, cu_seqlens,
        # use_qk_l2norm_in_kernel, cp_context
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g_fla), db.to(beta), None, None, None, None


@torch.compiler.disable
def flashqla_chunk_gated_delta_rule_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    **kwargs,
):
    """FlashQLA GDN chunk kernel driven by xorl's native CP.

    Mirrors :func:`xorl.ops.linear_attention.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`
    but runs the FlashQLA interior. Requires an active ``cp_context`` (Ulysses CP) with
    ``cu_seqlens``; ``initial_state`` / ``output_final_state`` are unsupported under CP
    (as in the FLA CP path). ``q``/``k``/``v`` must already be repeated to ``num_v_heads``
    (done by :class:`GatedDeltaNet`) and have head dim 128.
    """
    assert cp_context is not None and cp_context.group is not None, "flashqla CP path requires a cp_context"
    assert initial_state is None, "Initial state is not supported for CP"
    assert output_final_state is False, "Output final state is not supported for CP"
    assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
    cu_seqlens = cp_context.cu_seqlens

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleCPFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        cp_context,
    )
    return o, final_state
