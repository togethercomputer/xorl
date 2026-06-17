"""Dynamo-opaque torch.library custom-op wrappers around Flash-Attention-4 (CuTeDSL).

FA4 (``flash_attn.cute``) ships its public ``flash_attn_func`` /
``flash_attn_varlen_func`` as plain ``torch.autograd.Function`` subclasses whose
``forward``/``backward`` dispatch CuTeDSL JIT kernels. The CuTeDSL dispatch layer
(cutlass_dsl/tvm_ffi_provider.py) builds a *fresh* Python wrapper (a new
``__code__`` object) on every call. Dynamo cannot trace it, and worse, guards the
wrapper by ``___check_obj_id(<wrapper>.__code__, ...)`` which never matches across
steps, so ``torch.compile`` either graph-breaks or recompiles every step.

The proper fix (mirroring how FA3 / ``flash_attn_interface.py`` achieves
compile-compatibility) is to register the heavy fwd/bwd work as
``torch.library`` custom ops with ``register_fake`` meta functions, then attach a
``torch.autograd.Function`` on top via ``register_autograd``. Dynamo/AOTAutograd
treat a registered custom op as a single opaque node — no tracing into the
per-call CuTeDSL wrapper — so there is ZERO graph break in fwd OR bwd.

This module is FA4-only; FA3 is unaffected (separate path in
``flash_attention.py``).
"""

from typing import Optional, Tuple

import torch

from .....utils import logging


logger = logging.get_logger(__name__)


# Lower-level FA4 CuTe entry points. These bypass FA4's own
# ``FlashAttnFunc``/``FlashAttnVarlenFunc`` autograd wrappers (which Dynamo cannot
# trace) and call the raw fwd/bwd that launch the CuTeDSL kernels. We wrap *these*
# in registered custom ops and supply our own autograd glue.
try:
    from flash_attn.cute.interface import _flash_attn_bwd as _fa4_cute_bwd
    from flash_attn.cute.interface import _flash_attn_fwd as _fa4_cute_fwd

    FA4_CUSTOM_OP_AVAILABLE = True
except ImportError:  # pragma: no cover - overlay not present
    _fa4_cute_fwd = None
    _fa4_cute_bwd = None
    FA4_CUSTOM_OP_AVAILABLE = False


# ----------------------------------------------------------------------------- #
# Forward custom op                                                             #
# ----------------------------------------------------------------------------- #
@torch.library.custom_op("xorl_fa4::fwd", mutates_args=(), device_types="cuda")
def _fa4_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the FA4 CuTe forward. Always materializes lse (the backward needs it)."""
    out, lse = _fa4_cute_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        return_lse=True,
    )
    return out, lse


@torch.library.register_fake("xorl_fa4::fwd")
def _fa4_fwd_op_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symbolic shapes only — matches the real FA4 fwd output layout.

    out layout follows ``q``'s ``(..., num_head, head_dim)`` with head_dim from
    ``v`` (head_dim_v). lse layout:
      - varlen: ``(num_head, total_q)``
      - batched: ``(batch, num_head, seqlen_q)``
    """
    num_head, _head_dim = q.shape[-2], q.shape[-1]
    head_dim_v = v.shape[-1]
    if cu_seqlens_q is not None:
        # varlen: q is (total_q, num_head, head_dim)
        total_q = q.shape[0]
        out = torch.empty((total_q, num_head, head_dim_v), dtype=q.dtype, device=q.device)
        lse = torch.empty((num_head, total_q), dtype=torch.float32, device=q.device)
    else:
        # batched: q is (batch, seqlen_q, num_head, head_dim)
        batch, seqlen_q = q.shape[0], q.shape[1]
        out = torch.empty((batch, seqlen_q, num_head, head_dim_v), dtype=q.dtype, device=q.device)
        lse = torch.empty((batch, num_head, seqlen_q), dtype=torch.float32, device=q.device)
    return out, lse


# ----------------------------------------------------------------------------- #
# Backward custom op                                                            #
# ----------------------------------------------------------------------------- #
@torch.library.custom_op("xorl_fa4::bwd", mutates_args=(), device_types="cuda")
def _fa4_bwd_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the FA4 CuTe backward, returning (dq, dk, dv)."""
    dq, dk, dv = _fa4_cute_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale,
        causal,
        softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        deterministic=deterministic,
    )
    return dq, dk, dv


@torch.library.register_fake("xorl_fa4::bwd")
def _fa4_bwd_op_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dq/dk/dv have the same shape/dtype as q/k/v respectively."""
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv


# ----------------------------------------------------------------------------- #
# Autograd glue                                                                 #
# ----------------------------------------------------------------------------- #
def _fa4_fwd_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right
    ctx.softcap = softcap


def _fa4_fwd_backward(ctx, dout, dlse):
    # dlse is ignored: lse is not part of the loss graph in our usage (we only
    # consume `out`), so no gradient flows back through it. This matches FA3, which
    # also drops the lse gradient unless return_lse is explicitly requested.
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    dq, dk, dv = torch.ops.xorl_fa4.bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size_left,
        ctx.window_size_right,
        ctx.softcap,
        False,  # deterministic
    )
    # One grad per fwd input; only q/k/v get gradients.
    return dq, dk, dv, None, None, None, None, None, None, None, None, None


if FA4_CUSTOM_OP_AVAILABLE:
    torch.library.register_autograd(
        "xorl_fa4::fwd",
        _fa4_fwd_backward,
        setup_context=_fa4_fwd_setup_context,
    )


# ----------------------------------------------------------------------------- #
# Public, compile-safe entry points                                             #
# ----------------------------------------------------------------------------- #
def fa4_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compile-safe FA4 varlen attention. Returns just ``out`` (drops lse)."""
    out, _ = torch.ops.xorl_fa4.fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
    )
    return out


def fa4_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compile-safe FA4 batched attention. Returns just ``out`` (drops lse)."""
    out, _ = torch.ops.xorl_fa4.fwd(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
    )
    return out
