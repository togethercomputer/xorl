"""Sink-aware Flash Attention 3 for GPT-OSS.

Wraps FA3's forward with an autograd function that fuses the learned
per-head sink into the softmax via post-hoc multiplication by
``sigmoid(lse - sink)``.  Mathematically equivalent to eager's concat-sink
-then-softmax, just using FA3's fused kernel for the underlying attention.

Needed because the installed ``flash_attn_interface`` build does not yet
expose a native ``sinks=`` kwarg; when that kernel support lands, this file
can be replaced by a direct passthrough.

A single autograd function handles both batched and packed-varlen inputs:
the only layout difference the sink math cares about is the rank of the
``lse`` tensor (``[B, Hq, S]`` batched vs ``[Hq, T]`` varlen), which broadcasts
uniformly once we keep *head* on axis ``-2`` and *sequence* on axis ``-1``.

Backward decomposition
----------------------
Let ``m = sigmoid(lse - sink)``.  Then ``o = o_flash * m``, and gradients
decompose into four paths:

    (1) main:      dq, dk, dv via FA backward with ``dout' = dO * m``
    (2) dsink:     -sum(g_r * m * (1-m))          where g_r = (dO * o_flash).sum(-1)
    (3) dq extra:  scale * g_ell * attention(Q, K, K)    (extra FA fwd)
    (4) dk extra:  scale * P^T (g_ell * Q)               (FA bwd, dv slot)

where ``g_ell = g_r * m * (1-m)``.  The derivation uses
``dlse/dq = scale * PK`` and ``dlse/dk = scale * P^T (scalar * Q)``.
"""

import math
from typing import Optional, Tuple

import torch
from flash_attn_interface import (
    _flash_attn_backward,
    flash_attn_func,
    flash_attn_varlen_func,
)


def _fa3_forward(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    softcap,
    deterministic,
    return_attn_probs,
):
    common = dict(
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    if cu_seqlens_q is not None:
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **common,
        )
    return flash_attn_func(q, k, v, **common)


def _fa3_backward(
    dout,
    q,
    k,
    v,
    out,
    lse,
    dq,
    dk,
    dv,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    softcap,
    deterministic,
):
    kwargs = dict(
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        deterministic=deterministic,
    )
    if cu_seqlens_q is not None:
        kwargs.update(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    _flash_attn_backward(dout, q, k, v, out, lse, **kwargs)


class FlashAttnWithSinkFA3(torch.autograd.Function):
    """FA3 attention with a learned per-head sink in the softmax denominator.

    Handles batched (4D q/k/v, 3D lse) and packed-varlen (3D q/k/v, 2D lse)
    uniformly by requiring that in both layouts ``lse`` has head on axis -2
    and sequence on axis -1, and ``out`` has sequence, head, dim as the last
    three axes.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sink,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        window_size,
        softmax_scale,
        softcap,
        deterministic,
    ):
        out, lse = _fa3_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        # Broadcast sink over lse: lse is [..., Hq, S] (head on -2, seq on -1).
        sink_f = sink.float().view(*([1] * (lse.ndim - 2)), -1, 1)
        m = torch.sigmoid(lse - sink_f)
        # Align m to out's layout ([..., S, Hq, D]): move seq next to head-dim slot.
        m_for_out = m.transpose(-2, -1).unsqueeze(-1).to(out.dtype)
        out_final = out * m_for_out

        saved = [q, k, v, sink, out, lse]
        if cu_seqlens_q is not None:
            saved.extend([cu_seqlens_q, cu_seqlens_k])
        ctx.save_for_backward(*saved)
        ctx.is_varlen = cu_seqlens_q is not None
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return out_final

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        if ctx.is_varlen:
            q, k, v, sink, raw_out, lse, cu_seqlens_q, cu_seqlens_k = saved
        else:
            q, k, v, sink, raw_out, lse = saved
            cu_seqlens_q = cu_seqlens_k = None

        fa_kwargs = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
        )
        scale = ctx.softmax_scale

        sink_f = sink.float().view(*([1] * (lse.ndim - 2)), -1, 1)
        m = torch.sigmoid(lse - sink_f)  # [..., Hq, S]
        m_seq_head = m.transpose(-2, -1)  # [..., S, Hq]
        m_for_out = m_seq_head.unsqueeze(-1)  # [..., S, Hq, 1]

        # fp32 accumulation is critical for dsink precision
        g_r = (grad_output.float() * raw_out.float()).sum(dim=-1)  # [..., S, Hq]
        g_ell = g_r * m_seq_head * (1.0 - m_seq_head)  # fp32

        # --- path 1: main grad via FA bwd with dout' = dO * m ---
        dout_main = (grad_output * m_for_out.to(grad_output.dtype)).contiguous()
        dq_main = torch.empty_like(q)
        dk_main = torch.empty_like(k)
        dv = torch.empty_like(v)
        _fa3_backward(dout_main, q, k, v, raw_out, lse, dq_main, dk_main, dv, **fa_kwargs)

        # --- path 2: dsink = -sum over all non-head axes of g_ell ---
        dsink = -g_ell.flatten(0, -2).sum(dim=0).to(sink.dtype)  # [Hq]

        # --- path 3: dq_extra = scale * g_ell * attention(Q, K, K) ---
        mu_k = _fa3_forward(q, k, k, return_attn_probs=False, **fa_kwargs)
        dq_extra = (scale * g_ell.unsqueeze(-1) * mu_k.float()).to(q.dtype)

        # --- path 4: dk_extra via FA bwd dv slot, dout' = g_ell * Q ---
        x = (g_ell.unsqueeze(-1).to(q.dtype) * q).contiguous()
        dq_dummy = torch.empty_like(q)
        dk_dummy = torch.empty_like(k)
        dk_extra = torch.empty_like(k)
        _fa3_backward(x, q, k, k, raw_out, lse, dq_dummy, dk_dummy, dk_extra, **fa_kwargs)
        dk_extra = scale * dk_extra

        dq = dq_main + dq_extra
        dk = dk_main + dk_extra
        # Nones correspond to: cu_q, cu_k, max_q, max_k, causal, ws, scale, sc, det
        return dq, dk, dv, dsink, None, None, None, None, None, None, None, None, None


def flash_attn_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Batched FA3 attention with a per-head learned sink.

    Args:
        q: ``[B, Sq, Hq, D]``
        k, v: ``[B, Sk, Hkv, D]`` (GQA supported)
        sink: ``[Hq]`` fp32 learned per-head logit
    """
    return FlashAttnWithSinkFA3.apply(
        q,
        k,
        v,
        sink,
        None,
        None,
        None,
        None,
        causal,
        window_size,
        softmax_scale,
        softcap,
        deterministic,
    )


def flash_attn_varlen_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Varlen / packed-sequence FA3 attention with a per-head learned sink.

    Args:
        q: ``[total_tokens, Hq, D]``
        k, v: ``[total_tokens, Hkv, D]``
        sink: ``[Hq]`` fp32 learned per-head logit
        cu_seqlens_q, cu_seqlens_k: int32 prefix sums, shape ``[batch+1]``
    """
    return FlashAttnWithSinkFA3.apply(
        q,
        k,
        v,
        sink,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        window_size,
        softmax_scale,
        softcap,
        deterministic,
    )
