"""Shared-prefix attention on the FA3 stack.

Computes full causal attention over a shared-prefix-repacked sequence (see
:mod:`xorl.ops.shared_prefix.repack`) without recomputing the shared prefix once
per response. Per group the repacked layout is a shared prefix block
``[p_0..p_{P-2}]`` (KV only) followed by one decoded block ``[p_{P-1}, r_0, ...]``
per member. Attention decomposes into standard flash calls + a log-sum-exp merge:

1. **prefix self-attention** — the shared prefix attends causally to itself
   (propagates prefix hidden states through layers; its query outputs are unused
   by the loss but are valid hidden states for the next layer).
2. **prefix cross-attention** — each decoded query attends, non-causally, to its
   group's shared prefix K/V (only for groups whose prefix is non-empty).
3. **decoded self-attention** — each decoded block attends causally to itself
   (its duplicated boundary token + responses).

(2) and (3) merge per decoded token. The result equals standard causal attention
over the un-deduplicated ``N*(P+R)`` layout. Correct gradients use the split-KV
trick: each part's backward is fed the *global merged* out/lse so the recomputed
softmax uses the global normalizer.

Composes with Ulysses SP unchanged: the SP strategy's all-to-all re-gathers the
full repacked sequence before this runs, so the (full-coordinate) context applies.
No new CUDA: reuses ``flash_attn_interface`` and the ring-attention LSE merge.
"""

from typing import Optional, Tuple

import torch
from flash_attn_interface import _flash_attn_backward, _flash_attn_forward, flash_attn_varlen_func

from xorl.distributed.sequence_parallel.ring_attention import _merge_attn_outputs
from xorl.ops.shared_prefix.repack import SharedPrefixContext


def _fwd(q, k, v, scale, causal, cu_q, cu_k, max_q, max_k):
    out, lse = _flash_attn_forward(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
    )[:2]
    return out, lse


def _bwd(grad_out, q, k, v, out, lse, scale, causal, cu_q, cu_k, max_q, max_k):
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=scale,
        is_causal=causal,
        dq=dq,
        dk=dk,
        dv=dv,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
    )
    return dq, dk, dv


class SharedPrefixDecodedAttn(torch.autograd.Function):
    """Decoded-token attention = (prefix cross-attn) merged with (decoded self-attn).

    Cross-attn is applied only to decoded tokens whose group has a non-empty
    prefix (``cross_local_idx``); the rest use decoded self-attn alone (e.g. P==1
    groups). Backward feeds the global merged out/lse into each part's flash
    backward and sums the two ``dq`` contributions.
    """

    @staticmethod
    def forward(
        ctx,
        q_dec,
        k_shared,
        v_shared,
        k_dec,
        v_dec,
        cross_local_idx,
        cu_cross_q,
        cu_shared,
        max_cross_q,
        max_shared,
        cu_dec,
        max_dec,
        scale,
    ):
        out_self, lse_self = _fwd(q_dec, k_dec, v_dec, scale, True, cu_dec, cu_dec, max_dec, max_dec)

        if cross_local_idx.numel() > 0:
            q_cross = q_dec.index_select(0, cross_local_idx)
            out_cross, lse_cross = _fwd(
                q_cross, k_shared, v_shared, scale, False, cu_cross_q, cu_shared, max_cross_q, max_shared
            )
            out_sub, lse_sub = _merge_attn_outputs(
                out_cross,
                lse_cross,
                out_self.index_select(0, cross_local_idx),
                lse_self.index_select(1, cross_local_idx),
                is_varlen=True,
            )
            out = out_self.index_copy(0, cross_local_idx, out_sub.to(out_self.dtype))
            lse_global = lse_self.index_copy(1, cross_local_idx, lse_sub)
        else:
            out = out_self
            lse_global = lse_self

        out = out.to(q_dec.dtype)
        ctx.save_for_backward(
            q_dec, k_shared, v_shared, k_dec, v_dec, out, lse_global, cross_local_idx, cu_cross_q, cu_shared, cu_dec
        )
        ctx.scale, ctx.max_cross_q, ctx.max_shared, ctx.max_dec = scale, max_cross_q, max_shared, max_dec
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (q_dec, k_shared, v_shared, k_dec, v_dec, out, lse, cross_local_idx, cu_cross_q, cu_shared, cu_dec) = (
            ctx.saved_tensors
        )
        grad_out = grad_out.contiguous()
        scale = ctx.scale

        dq, dk_dec, dv_dec = _bwd(
            grad_out, q_dec, k_dec, v_dec, out, lse, scale, True, cu_dec, cu_dec, ctx.max_dec, ctx.max_dec
        )
        if cross_local_idx.numel() > 0:
            q_cross = q_dec.index_select(0, cross_local_idx)
            dq_cross, dk_shared, dv_shared = _bwd(
                grad_out.index_select(0, cross_local_idx),
                q_cross,
                k_shared,
                v_shared,
                out.index_select(0, cross_local_idx),
                lse.index_select(1, cross_local_idx),
                scale,
                False,
                cu_cross_q,
                cu_shared,
                ctx.max_cross_q,
                ctx.max_shared,
            )
            dq = dq.index_add(0, cross_local_idx, dq_cross)
        else:
            dk_shared, dv_shared = torch.zeros_like(k_shared), torch.zeros_like(v_shared)

        return (dq, dk_shared, dv_shared, dk_dec, dv_dec, None, None, None, None, None, None, None, None)


def shared_prefix_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    dropout: float = 0.0,
    shared_prefix_context: SharedPrefixContext,
    **_kwargs,
) -> Tuple[torch.Tensor, None]:
    """Full causal attention over a shared-prefix-repacked packed sequence.

    Args:
        query/key/value: ``[1, T_rep, H, D]`` or ``[T_rep, H, D]`` (B=1).
        scaling: softmax scale (``1/sqrt(head_dim)``).
        shared_prefix_context: from ``shared_prefix_repack_batch``.

    Returns:
        ``(attn_output, None)`` matching the query's rank.
    """
    if sliding_window is not None:
        raise NotImplementedError("shared-prefix attention does not support sliding windows")
    if dropout:
        raise NotImplementedError("shared-prefix attention does not support attention dropout")

    ctx = shared_prefix_context
    had_batch = query.dim() == 4
    q = query.squeeze(0) if had_batch else query  # [T_rep, Hq, D]
    k = key.squeeze(0) if had_batch else key
    v = value.squeeze(0) if had_batch else value
    if scaling is None:
        scaling = q.size(-1) ** -0.5

    out = q.new_zeros((q.size(0), q.size(1), q.size(2)))

    # 1) prefix self-attention (causal), over the shared prefix blocks (P>=2 groups).
    if ctx.shared_idx.numel() > 0:
        out_prefix = flash_attn_varlen_func(
            q.index_select(0, ctx.shared_idx),
            k.index_select(0, ctx.shared_idx),
            v.index_select(0, ctx.shared_idx),
            cu_seqlens_q=ctx.cu_shared,
            cu_seqlens_k=ctx.cu_shared,
            max_seqlen_q=ctx.max_shared,
            max_seqlen_k=ctx.max_shared,
            softmax_scale=scaling,
            causal=True,
        )
        out = out.index_copy(0, ctx.shared_idx, out_prefix)

    # 2)+3) decoded cross-attention to the shared prefix, merged with decoded self-attention.
    out_dec = SharedPrefixDecodedAttn.apply(
        q.index_select(0, ctx.dec_idx),
        k.index_select(0, ctx.shared_idx),
        v.index_select(0, ctx.shared_idx),
        k.index_select(0, ctx.dec_idx),
        v.index_select(0, ctx.dec_idx),
        ctx.cross_local_idx,
        ctx.cu_cross_q,
        ctx.cu_shared,
        ctx.max_cross_q,
        ctx.max_shared,
        ctx.cu_dec,
        ctx.max_dec,
        scaling,
    )
    out = out.index_copy(0, ctx.dec_idx, out_dec)

    return (out.unsqueeze(0) if had_batch else out), None
