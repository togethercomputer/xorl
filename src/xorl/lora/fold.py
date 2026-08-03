"""Canonical LoRA fold — the arithmetic contract of the merged-forward K3 lane.

One pinned op order for materializing merged weights ``W' = W + scaling * (A @ B)``,
shared by every consumer that must agree bitwise:

- the trainer's merged forward (``XORL_LORA_MERGED_FORWARD=1``): the adapted
  module folds its delta and runs the BASE contract kernels on ``W'``;
- the weight-sync merged extraction (same flag): the engine receives exactly
  the bytes the trainer trains with;
- the serving-side fold-on-receipt mirror (sglang ``SGLANG_LORA_FOLD_CANONICAL=1``).

Pinned order (do not reorder — the bits are the contract):
  1. upcast the low-rank factors to fp32 (contiguous),
  2. ``delta = bmm(A_gkn, B_gkn) * float(scaling)`` in the GKN orientation
     ``[E, in, r] @ [E, r, out]`` (dense: ``B @ A`` in ``[out, r] @ [r, in]``),
  3. ``W' = (W.to(fp32) + delta).to(W.dtype)`` — fp32 accumulate, cast ONCE
     (the PR #164 / ZORL fp32-master doctrine; never bf16(W) + bf16(delta)).

Cross-venv note: step 2 measured bit-identical between torch 2.12.1+cu132 and
torch 2.9.1+cu128 on H100 for the r<=64 shapes of this lane. The fold-parity
gate should be rerun on every environment change.
"""

import os

import torch


_MERGED_FORWARD_ENV = "XORL_LORA_MERGED_FORWARD"
_MERGED_CACHE_ENV = "XORL_LORA_MERGED_FORWARD_CACHE"


def lora_merged_forward_enabled() -> bool:
    """Opt-in flag for the merged-forward LoRA K3 contract lane."""
    return os.environ.get(_MERGED_FORWARD_ENV, "0") == "1"


def lora_merged_cache_enabled() -> bool:
    """Cache folded weights per module, keyed on adapter-param versions
    (default on). ``XORL_LORA_MERGED_FORWARD_CACHE=0`` refolds on every call
    (bounded memory, slower)."""
    return os.environ.get(_MERGED_CACHE_ENV, "1") == "1"


def canonical_lora_delta_gkn(
    lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float, num_experts: int | None = None
) -> torch.Tensor:
    """fp32 delta ``[E, K, N]`` from GKN factors ``A [E_or_1, K, r]``,
    ``B [E_or_1, r, N]`` (shared factors expanded; expand is stride-0, no copy)."""
    E = num_experts or max(lora_A.shape[0], lora_B.shape[0])
    A32 = lora_A.to(torch.float32).expand(E, -1, -1)
    B32 = lora_B.to(torch.float32).expand(E, -1, -1)
    return torch.bmm(A32, B32) * float(scaling)


def canonical_lora_fold_gkn(
    base_gkn: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Merged GKN weight ``[E, K, N]`` in ``base_gkn.dtype`` (cast once)."""
    delta = canonical_lora_delta_gkn(lora_A, lora_B, scaling, num_experts=base_gkn.shape[0])
    return (base_gkn.to(torch.float32) + delta).to(base_gkn.dtype)


def canonical_lora_delta_linear(lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    """fp32 delta ``[out, in]`` from dense factors ``A [r, in]``, ``B [out, r]``."""
    return (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * float(scaling)


def canonical_lora_fold_linear(
    weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Merged dense weight ``[out, in]`` in ``weight.dtype`` (cast once)."""
    delta = canonical_lora_delta_linear(lora_A, lora_B, scaling)
    return (weight.to(torch.float32) + delta).to(weight.dtype)


def _factor_grad_dtype(factor: torch.Tensor) -> torch.dtype:
    """Return the dtype autograd expects for a LoRA-factor gradient.

    FSDP mixed precision exposes a low-precision forward view while retaining
    the unsharded parameter's gradient dtype on ``grad_dtype``.  Custom
    autograd must honor that metadata rather than the transient view dtype or
    the engine rejects the returned gradient before FSDP can reduce it.
    The active-rank factors passed here are slices, so resolve their leaf base
    before reading ``grad_dtype`` (PyTorch rejects reading that property from a
    non-leaf tensor). Plain tensors use their own dtype.
    """
    base = factor
    while not base.is_leaf and getattr(base, "_base", None) is not None:
        base = base._base
    try:
        return getattr(base, "grad_dtype", None) or base.dtype
    except RuntimeError:
        # A non-view non-leaf has no leaf metadata to recover.
        return factor.dtype


class FoldedLoraWeightGKN(torch.autograd.Function):
    """Straight-through folded expert weight.

    forward: returns the (cached, detached) folded weight ``W' [E, K, N]``.
    backward: exact chain rule through ``W' = W + scaling * (A @ B)``:
      ``dA = scaling * dW' @ B^T`` (summed over experts for a shared A),
      ``dB = scaling * A^T @ dW'`` (summed for a shared B), fp32 math, grads
    cast to the factor dtypes. The base weight is frozen (no grad)."""

    @staticmethod
    def forward(ctx, folded, lora_A, lora_B, scaling):
        ctx.scaling = float(scaling)
        ctx.save_for_backward(lora_A, lora_B)
        return folded

    @staticmethod
    def backward(ctx, grad_w):
        lora_A, lora_B = ctx.saved_tensors
        E = grad_w.shape[0]
        A32 = lora_A.to(torch.float32).expand(E, -1, -1)
        B32 = lora_B.to(torch.float32).expand(E, -1, -1)
        gw32 = grad_w.to(torch.float32)
        grad_A = grad_B = None
        if ctx.needs_input_grad[1]:
            grad_A = torch.bmm(gw32, B32.transpose(1, 2)) * ctx.scaling
            if lora_A.shape[0] == 1:
                grad_A = grad_A.sum(dim=0, keepdim=True)
            grad_A = grad_A.to(_factor_grad_dtype(lora_A))
        if ctx.needs_input_grad[2]:
            grad_B = torch.bmm(A32.transpose(1, 2), gw32) * ctx.scaling
            if lora_B.shape[0] == 1:
                grad_B = grad_B.sum(dim=0, keepdim=True)
            grad_B = grad_B.to(_factor_grad_dtype(lora_B))
        return None, grad_A, grad_B, None


class FoldedLoraWeightGateUpGKN(torch.autograd.Function):
    """Straight-through folded FUSED gate_up expert weight ``[E, K, 2I]``
    (gate cols first), folded from per-projection factors. Backward slices the
    incoming weight grad into the gate/up halves and applies the
    :class:`FoldedLoraWeightGKN` chain rule per projection."""

    @staticmethod
    def forward(ctx, folded_gate_up, gate_A, gate_B, up_A, up_B, scaling, intermediate_size):
        ctx.scaling = float(scaling)
        ctx.intermediate_size = int(intermediate_size)
        ctx.save_for_backward(gate_A, gate_B, up_A, up_B)
        return folded_gate_up

    @staticmethod
    def backward(ctx, grad_w):
        gate_A, gate_B, up_A, up_B = ctx.saved_tensors
        inter = ctx.intermediate_size
        E = grad_w.shape[0]
        grads = [None]
        for A, B, gw in (
            (gate_A, gate_B, grad_w[..., :inter]),
            (up_A, up_B, grad_w[..., inter:]),
        ):
            A32 = A.to(torch.float32).expand(E, -1, -1)
            B32 = B.to(torch.float32).expand(E, -1, -1)
            gw32 = gw.to(torch.float32)
            grad_A = torch.bmm(gw32, B32.transpose(1, 2)) * ctx.scaling
            grad_B = torch.bmm(A32.transpose(1, 2), gw32) * ctx.scaling
            if A.shape[0] == 1:
                grad_A = grad_A.sum(dim=0, keepdim=True)
            if B.shape[0] == 1:
                grad_B = grad_B.sum(dim=0, keepdim=True)
            grads.extend([grad_A.to(_factor_grad_dtype(A)), grad_B.to(_factor_grad_dtype(B))])
        grads.extend([None, None])
        return tuple(grads)


class FoldedLoraWeightLinear(torch.autograd.Function):
    """Straight-through folded dense weight ``W' = W + scaling * (B @ A)``.

    forward returns the (cached, detached) ``W' [out, in]``; backward:
    ``dA = scaling * B^T @ dW'``, ``dB = scaling * dW' @ A^T`` in fp32."""

    @staticmethod
    def forward(ctx, folded, lora_A, lora_B, scaling):
        ctx.scaling = float(scaling)
        ctx.save_for_backward(lora_A, lora_B)
        return folded

    @staticmethod
    def backward(ctx, grad_w):
        lora_A, lora_B = ctx.saved_tensors
        gw32 = grad_w.to(torch.float32)
        grad_A = grad_B = None
        if ctx.needs_input_grad[1]:
            grad_A = (lora_B.to(torch.float32).t() @ gw32 * ctx.scaling).to(_factor_grad_dtype(lora_A))
        if ctx.needs_input_grad[2]:
            grad_B = (gw32 @ lora_A.to(torch.float32).t() * ctx.scaling).to(_factor_grad_dtype(lora_B))
        return None, grad_A, grad_B, None


def _params_version_key(*tensors: torch.Tensor) -> tuple:
    return tuple(t._version for t in tensors)
