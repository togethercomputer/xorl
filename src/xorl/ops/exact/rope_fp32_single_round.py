from __future__ import annotations

# Class-B RoPE numerics, vendored from the matching sampler implementation so the trainer
# takes no runtime dependency on SGLang.
#
# Two RoPE numerics classes exist across the trainer/sampler pair:
#   Class A -- per-op bf16 rounding (8 rounding points). The trainer's
#              ``_naive_apply_rotary_pos_emb``, SGLang's eager ``apply_rotary_emb`` and its
#              ``bi_fused_native`` triton kernel all agree bitwise here.
#   Class B -- one fp32 chain with a single final round. SGLang's stock fused CUDA rope
#              (``jit_kernel/csrc/elementwise/rope.cuh``) and the ``torch.compile`` RoPE path
#              SGLang's RL lane selects agree bitwise here.
#
# ``_rotary_emb`` is the body of SGLang's ``rotary_embedding/utils.py::apply_rotary_emb``,
# vendored verbatim. What selects Class B is not the function but its input: callers hand it
# the fp32 cos/sin table serving builds, rather than the bf16 one the trainer's eager path
# rounds first. Compiled, it is bitwise equal to the stock CUDA kernel across every variant and
# shape the target models use -- that equality is a gate, not an assumption; see
# ``gate_phase1.py``. (The ``.to(x.dtype)`` below is load-bearing for the output dtype and is
# round-tripped away by inductor for the arithmetic; dropping it yields fp32 outputs.)
#
# A hand-written triton reimplementation of the CUDA kernel was tried first and rejected: it
# lands within one bf16 ULP on ~7e-4 of elements but is not bitwise equal, and no FMA
# association (plain, ``fma(x,cos,-(y*sin))``, ``fma(-y,sin,x*cos)``, or fully uncontracted
# ``mul.rn``) closes the gap. Bitwise is the requirement, so the compiled fp32 expression is
# what ships.
import logging
import os

import torch


logger = logging.getLogger(__name__)

# torch.compile is load-bearing here, not an optimisation: the EAGER form of this same
# expression is Class A. Dynamo's default recompile budget is 8, far under what a real model
# needs (separate q and k shapes x rollout lengths x forward and backward), and on exhausting
# it dynamo silently falls back to eager -- reverting those layers to Class A with no error and
# no log line. That is indistinguishable from a numerics bug downstream, so pin the budget and
# make exhaustion loud.
_ROPE_SINGLE_ROUND_RECOMPILE_LIMIT = 2048
_ROPE_SINGLE_ROUND_ACCUMULATED_LIMIT = 8192


def _pin_compile_budget() -> None:
    cfg = torch._dynamo.config
    cfg.recompile_limit = max(getattr(cfg, "recompile_limit", 0), _ROPE_SINGLE_ROUND_RECOMPILE_LIMIT)
    cfg.accumulated_recompile_limit = max(
        getattr(cfg, "accumulated_recompile_limit", 0), _ROPE_SINGLE_ROUND_ACCUMULATED_LIMIT
    )
    # A silent eager fallback breaks the zero-K3 contract, so fail instead. Opt out with
    # XORL_ROPE_SINGLE_ROUND_ALLOW_FALLBACK=1 for throughput experiments outside the contract.
    if os.environ.get("XORL_ROPE_SINGLE_ROUND_ALLOW_FALLBACK") != "1" and hasattr(cfg, "fail_on_recompile_limit_hit"):
        cfg.fail_on_recompile_limit_hit = True
    logger.info(
        "rope Class B: recompile_limit=%s accumulated=%s fail_on_limit=%s",
        cfg.recompile_limit,
        cfg.accumulated_recompile_limit,
        getattr(cfg, "fail_on_recompile_limit_hit", None),
    )


_pin_compile_budget()


def _rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool) -> torch.Tensor:
    """SGLang's ``apply_rotary_emb``, vendored.

    x: ``[num_tokens, num_heads, rotary_dim]``; cos/sin: ``[num_tokens, rotary_dim // 2]`` fp32.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


_rotary_emb_compiled = torch.compile(dynamic=True)(_rotary_emb)


class _SingleRoundRoPE(torch.autograd.Function):
    """Class-B rope with a hand-written backward.

    RoPE is a block rotation, so its adjoint is the same rotation driven by a negated sin.
    Supplying it explicitly keeps the backward one fused pass and lands it closer to the fp64
    adjoint than differentiating the forward expression.
    """

    @staticmethod
    def forward(ctx, q_rot, k_rot, cos, sin, is_neox):
        ctx.save_for_backward(cos, sin)
        ctx.is_neox = is_neox
        return (
            _rotary_emb_compiled(q_rot, cos, sin, is_neox),
            _rotary_emb_compiled(k_rot, cos, sin, is_neox),
        )

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        neg_sin = -sin
        return (
            _rotary_emb_compiled(grad_q.contiguous(), cos, neg_sin, ctx.is_neox),
            _rotary_emb_compiled(grad_k.contiguous(), cos, neg_sin, ctx.is_neox),
            None,
            None,
            None,
        )


def build_single_round_cos_sin(cos, sin, *, doubled: bool = True):
    """Flatten the trainer's cos/sin into the fp32 ``[num_tokens, rotary_dim // 2]`` serving layout.

    With ``doubled`` (the default) cos/sin arrive in the ``[..., rotary_dim]`` layout the shared
    ``RotaryEmbedding`` emits, where only the leading half carries distinct frequencies. GPT-OSS
    emits them already halved, at ``[..., rotary_dim // 2]``.
    """
    half = cos.shape[-1] // 2 if doubled else cos.shape[-1]
    return cos[..., :half].reshape(-1, half).float(), sin[..., :half].reshape(-1, half).float()


def single_round_apply_rotary_pos_emb(q, k, cos, sin, *, interleaved: bool = False, doubled: bool = True):
    """Class-B RoPE over ``[B, S, H, D]`` q/k.

    ``interleaved`` selects GPT-J pairing; the default is the half-split Neox pairing. Partial
    rotary rotates the leading rotary_dim features and passes the tail through untouched.
    """
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise RuntimeError(
            "Class-B RoPE requires fp32 cos/sin at the apply boundary; "
            f"got cos={cos.dtype}, sin={sin.dtype}. A mixed-precision wrapper "
            "likely downcast the table in transit."
        )
    cos_f, sin_f = build_single_round_cos_sin(cos, sin, doubled=doubled)
    rotary_dim = 2 * cos_f.shape[-1]
    batch, seq_len = q.shape[0], q.shape[1]
    num_tokens = batch * seq_len

    q_rot = q[..., :rotary_dim].reshape(num_tokens, -1, rotary_dim)
    k_rot = k[..., :rotary_dim].reshape(num_tokens, -1, rotary_dim)
    q_out, k_out = _SingleRoundRoPE.apply(q_rot, k_rot, cos_f, sin_f, not interleaved)
    q_out = q_out.view(batch, seq_len, -1, rotary_dim)
    k_out = k_out.view(batch, seq_len, -1, rotary_dim)

    if q.shape[-1] > rotary_dim:
        q_out = torch.cat((q_out, q[..., rotary_dim:]), dim=-1)
        k_out = torch.cat((k_out, k[..., rotary_dim:]), dim=-1)
    return q_out, k_out


__all__ = [
    "build_single_round_cos_sin",
    "single_round_apply_rotary_pos_emb",
]
