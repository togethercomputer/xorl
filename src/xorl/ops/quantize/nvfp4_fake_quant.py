"""NVFP4 fake quantization (STE shadow) for Quantization-Aware Training (QAT).

Fake quantization simulates the NVFP4 inference rounding in the forward pass so
the network *sees* the quantization error, while gradients pass straight through
(straight-through estimator, STE). For QAT we never store a packed weight — we
round and immediately use the dequantized value — so this is **pure PyTorch
round-to-nearest**, NOT the inference packing kernels in ``nvfp4_quantize`` /
``nvfp4_gkn_quantize`` (those are Triton, GPU-only, and return packed uint8).
This module is CPU-testable and is the single source of the RTN math shared by
QARL training fake-quant and the NVFP4 exporter.

NVFP4 two-level scaling (matches the inference format):
  * a single **global FP32 scale** derived from the tensor's global amax, and
  * a **per-group (of 16) FP8 (E4M3) scale** derived from each block's amax,
followed by round-to-nearest onto the FP4 E2M1 grid. Because each block's FP8
scale is ``block_amax * FP8_MAX / global_amax <= FP8_MAX`` by construction, only
the single global-max block ever reaches the FP8 ceiling — there is no mass
saturation (the failure mode the packing kernel exhibits on outlier-heavy MLP
weights).

The STE is the textbook ``w + (w_q - w).detach()`` identity — no custom
autograd.Function and no non-differentiable kernels involved.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX


# The rounding math generalizes across NVFP4 group sizes; this guard exists so
# callers fail loudly on an unsupported quant_format rather than silently no-op.
_SUPPORTED_FORMATS = frozenset({"nvfp4"})

# FP4 E2M1 representable magnitudes (codes 0..7); index == E2M1 magnitude code.
_E2M1_ABS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def is_supported_format(quant_format: str) -> bool:
    """Whether ``quant_format`` is supported by the fake quantizer (nvfp4)."""
    return quant_format in _SUPPORTED_FORMATS


def _round_to_e2m1(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Round ``x`` to the nearest signed FP4 E2M1 value.

    Returns ``(values, codes)`` where ``codes`` is the 4-bit NVFP4 code
    (``sign<<3 | magnitude_index``), matching ``fp4_codec``.
    """
    grid = torch.tensor(_E2M1_ABS, device=x.device, dtype=x.dtype)
    mids = (grid[1:] + grid[:-1]) * 0.5  # 7 midpoints for nearest-rounding
    neg = x < 0
    mag = x.abs().clamp(max=FP4_E2M1_MAX)
    idx = torch.bucketize(mag, mids)  # 0..7
    values = torch.where(neg, -grid[idx], grid[idx])
    codes = (neg.to(torch.uint8) << 3) | idx.to(torch.uint8)
    return values, codes


def _nvfp4_block_quantize_per_slice(
    wf3d: Tensor,
    block_size: int,
    global_scale_per_slice: Tensor | None = None,
):
    """Batched per-slice NVFP4 round-to-nearest on a 3D fp32 weight.

    Each ``[M, K]`` slice of ``wf3d [B, M, K]`` is quantized **independently**:
    its own global amax (or supplied ``global_scale_per_slice[i]``) and its own
    per-1×``block_size`` block scales (blocks along the **last (K) dim**).

    Returns ``(w_dq, block_scales_fp8, global_scale, codes)``:
      * ``w_dq``           fp32  ``[B, M, K]``
      * ``block_scales_fp8`` float8_e4m3fn ``[B, M, K // block_size]``
      * ``global_scale``   fp32 ``[B]``
      * ``codes``          uint8 ``[B, M, K]``
    """
    assert wf3d.dim() == 3, f"expected 3D, got shape {tuple(wf3d.shape)}"
    B, M, K = wf3d.shape
    assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"

    if global_scale_per_slice is None:
        global_amax = wf3d.abs().amax(dim=(1, 2))  # [B]
        global_scale = (global_amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)).reshape(B)
    else:
        global_scale = global_scale_per_slice.to(wf3d.device, torch.float32).reshape(B)

    blocks = wf3d.reshape(B, -1, block_size)  # [B, n_blocks, bs]
    block_amax = blocks.abs().amax(dim=2, keepdim=True)  # [B, n_blocks, 1]

    tiny = torch.finfo(torch.float32).tiny
    safe_gs = global_scale.clamp_min(tiny).reshape(B, 1, 1)
    block_scales_fp8 = (block_amax / FP4_E2M1_MAX / safe_gs).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    effective = block_scales_fp8.float() * global_scale.reshape(B, 1, 1)  # [B, n_blocks, 1]
    safe_eff = torch.where(effective > 0, effective, torch.ones_like(effective))

    values, codes = _round_to_e2m1(blocks / safe_eff)
    w_dq = (values * effective).reshape(B, M, K)
    block_scales_fp8 = block_scales_fp8.reshape(B, M, K // block_size)
    codes = codes.reshape(B, M, K)
    return w_dq, block_scales_fp8, global_scale, codes


def _nvfp4_quantize_blocks(wf: Tensor, block_size: int, global_scale: Tensor | None = None):
    """Core NVFP4 round-to-nearest on a 2D fp32 weight.

    Returns ``(w_dq, block_scales_fp8, global_scale, codes)``:
      * ``w_dq``           fp32 dequantized weight, same shape as ``wf``
      * ``block_scales_fp8`` float8_e4m3fn, one per block of ``block_size``
      * ``global_scale``   fp32 scalar (shape ``[1]``)
      * ``codes``          uint8 FP4 codes (sign<<3 | mag), same shape as ``wf``

    ``global_scale`` may be supplied to force a shared per-tensor scale across a
    group of weights that the inference stack fuses into a single GEMM (e.g.
    q/k/v -> qkv_proj, gate/up -> gate_up_proj). The fused NVFP4 kernel uses ONE
    ``weight_scale_2`` per fused tensor, so the members must share it; otherwise
    the kernel mis-scales all but one projection. When ``None`` (default,
    used by training STE) it is derived from this tensor's own global amax.
    """
    M, K = wf.shape
    if global_scale is None:
        global_amax = wf.abs().amax()
        global_scale = (global_amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)).reshape(1)
    else:
        global_scale = global_scale.to(wf.device, torch.float32).reshape(1)

    blocks = wf.reshape(-1, block_size)
    block_amax = blocks.abs().amax(dim=1, keepdim=True)

    tiny = torch.finfo(torch.float32).tiny
    safe_gs = global_scale.clamp_min(tiny)
    # Per-block scale relative to the global scale, rounded to FP8 E4M3 (the
    # representation real NVFP4 inference uses). <= FP8_MAX by construction.
    block_scales_fp8 = (block_amax / FP4_E2M1_MAX / safe_gs).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    effective = block_scales_fp8.float() * global_scale  # [nblk, 1]
    safe_eff = torch.where(effective > 0, effective, torch.ones_like(effective))

    values, codes = _round_to_e2m1(blocks / safe_eff)
    w_dq = (values * effective).reshape(M, K)
    return w_dq, block_scales_fp8.reshape(-1), global_scale, codes.reshape(M, K)


def fake_quantize_nvfp4(w: Tensor, block_size: int = 16) -> Tensor:
    """Fake-quantize ``w`` to NVFP4 with a straight-through-estimator backward.

    Forward returns ``dequant(round_nvfp4(w))`` in ``w.dtype``; the gradient
    w.r.t. ``w`` is the identity. ``w`` must be 2D with the **in_features** (K, the
    last dim) divisible by ``block_size`` — NVFP4 blocks run along K within a single
    output row, so a ``K`` that is not a multiple of ``block_size`` would group
    elements from two different output channels into one block (silently wrong, and
    unservable). The 3D expert helpers enforce the same K-divisibility.
    """
    assert w.dim() == 2, f"fake_quantize_nvfp4 expects a 2D weight, got shape {tuple(w.shape)}"
    M, K = w.shape
    assert K % block_size == 0, f"in_features K ({K}) must be divisible by block_size ({block_size})"
    w_dq, *_ = _nvfp4_quantize_blocks(w.float(), block_size)
    w_dq = w_dq.to(w.dtype)
    # STE: forward is the rounded value; backward is identity w.r.t. w.
    return w + (w_dq - w).detach()


def fake_quantize_activation_nvfp4(x: Tensor, block_size: int = 16) -> Tensor:
    """Fake-quantize an **activation** tensor to NVFP4 with an STE backward.

    Same per-group-of-``block_size`` E2M1 round-to-nearest as weights, applied along the
    **last (feature) dim**, with a per-tensor dynamic global scale recomputed from ``x``
    each call (matches modelopt's dynamic NVFP4 activation quant). ``x`` may be any shape
    whose last dim is divisible by ``block_size``; the leading dims are flattened to rows
    so each block of 16 stays within a single token's feature vector.
    """
    K = x.shape[-1]
    assert K % block_size == 0, f"activation last dim ({K}) must be divisible by block_size ({block_size})"
    x2d = x.reshape(-1, K)
    xq = fake_quantize_nvfp4(x2d, block_size)  # STE: w + (w_q - w).detach()
    return xq.reshape(x.shape)


def _fake_quantize_3d_fused_gate_up(
    w: Tensor,
    intermediate_size: int,
    block_size: int = 16,
    return_metadata: bool = False,
):
    """Fake-quantize a fused MoE ``gate_up_proj`` tensor with an STE backward.

    ``w`` has shape ``[E, H, 2*I]`` (GKN: experts × hidden × 2·intermediate).
    Split each expert along **dim 2** into gate ``[H, I]`` and up ``[H, I]`` halves.
    Each ``(expert, half)`` is quantized **independently** (its own global amax →
    its own ``weight_scale_2``), and blocks of ``block_size`` are taken along the
    K (contraction) dim = ``H`` (axis 1 of the 3D tensor; the last dim of the HF
    per-expert ``[out=I, in=H]`` layout).

    Per-half (not shared) global scale — REVERT of PR #399. A single shared
    ``max(gate, up)`` scale (to strictly match the sglang fused-w13 serve kernel, which
    reads ``w13_weight_scale_2[:, 0]``) made the QAT optimization **unstable**: coarsening
    the smaller-amax half's quant gives a rough loss landscape with grad blow-ups (the v2
    early-step spike and the v3 late-training divergence). Per-half trains smoothly (the
    v1 run reached the best GPQA, 0.646). The residual train/serve difference — export and
    sglang still apply one shared ``weight_scale_2`` per fused expert — is second-order:
    the per-row fp8 block scales absorb most of the gate/up amax gap.

    When ``return_metadata`` is true, also returns a dict with:
      * ``block_scales`` fp8  ``[E, 2, H//block_size, I]`` (GKN layout, K-blocks first)
      * ``weight_scale_2`` fp32 ``[E, 2]`` (per-(expert, half) global scale)
      * ``codes`` uint8 ``[E, 2, I, H]`` (HF per-half layout; for export packing)
    """
    assert w.dim() == 3, f"expected [E, H, 2I], got shape {tuple(w.shape)}"
    E, H, two_I = w.shape
    assert two_I == 2 * intermediate_size, (
        f"gate_up_proj last dim must be 2*intermediate_size ({2 * intermediate_size}), got {two_I}"
    )
    assert H % block_size == 0, f"H ({H}) must be divisible by block_size ({block_size})"
    I = intermediate_size

    wf = w.float()
    # GKN [E, H, I] per half → HF [E, I, H] (K=H is now last dim → blocks along last).
    gate_hf = wf[..., :I].transpose(1, 2).contiguous()  # [E, I, H]
    up_hf = wf[..., I:].transpose(1, 2).contiguous()  # [E, I, H]
    # Stack halves into the batch dim: [2E, I, H], so each (expert, half) is one slice,
    # quantized against its OWN global amax (per-half weight_scale_2). The shared-scale
    # variant (PR #399) was reverted for training stability — see the docstring.
    stacked = torch.stack([gate_hf, up_hf], dim=1).reshape(2 * E, I, H)
    dq_hf, bs_fp8, gs, codes = _nvfp4_block_quantize_per_slice(stacked, block_size)

    # Back to per-half tensors in HF layout.
    dq_hf = dq_hf.reshape(E, 2, I, H)
    bs_fp8 = bs_fp8.reshape(E, 2, I, H // block_size)
    gs = gs.reshape(E, 2)
    codes = codes.reshape(E, 2, I, H)

    # Rebuild GKN-layout dequant: per expert concat gate/up along dim 2 (the 2I axis).
    dq_gkn = torch.cat(
        [dq_hf[:, 0].transpose(1, 2), dq_hf[:, 1].transpose(1, 2)],
        dim=2,
    ).contiguous()  # [E, H, 2I]

    w_fq = w + (dq_gkn.to(w.dtype) - w).detach()

    if not return_metadata:
        return w_fq
    # Tests expect block_scales in GKN order (K_blocks, N): [E, 2, H/bs, I].
    bs_gkn = bs_fp8.transpose(2, 3).contiguous()  # [E, 2, H/bs, I]
    return w_fq, {
        "block_scales": bs_gkn,
        "weight_scale_2": gs,
        "codes": codes,  # HF layout per half (E, 2, I, H), for export packing
    }


def _fake_quantize_3d_experts(
    w: Tensor,
    block_size: int = 16,
    return_metadata: bool = False,
):
    """Fake-quantize a 3D MoE expert tensor (e.g. ``down_proj``) with STE backward.

    ``w`` has shape ``[E, K, N]`` in GKN — for ``down_proj`` ``K=intermediate``,
    ``N=hidden``. Each expert quantized **independently**; blocks of ``block_size``
    along **K** (axis 1 of the 3D tensor; the last dim of HF ``[out=N, in=K]``).

    When ``return_metadata`` is true, also returns a dict with:
      * ``block_scales`` fp8 ``[E, K//block_size, N]`` (GKN layout)
      * ``weight_scale_2`` fp32 ``[E]``
      * ``codes`` uint8 ``[E, N, K]`` (HF per-expert layout)
    """
    assert w.dim() == 3, f"expected [E, K, N], got shape {tuple(w.shape)}"
    E, K, N = w.shape
    assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"

    wf = w.float()
    # GKN [E, K, N] → HF [E, N, K] so K is last for blocking.
    hf = wf.transpose(1, 2).contiguous()  # [E, N, K]
    dq_hf, bs_fp8, gs, codes = _nvfp4_block_quantize_per_slice(hf, block_size)
    # Back to GKN.
    dq_gkn = dq_hf.transpose(1, 2).contiguous()  # [E, K, N]
    w_fq = w + (dq_gkn.to(w.dtype) - w).detach()

    if not return_metadata:
        return w_fq
    # bs_fp8 is [E, N, K/bs] HF; transpose to GKN [E, K/bs, N].
    bs_gkn = bs_fp8.transpose(1, 2).contiguous()  # [E, K/bs, N]
    return w_fq, {
        "block_scales": bs_gkn,
        "weight_scale_2": gs,
        "codes": codes,  # HF layout per expert (E, N, K)
    }


def fake_quantize_activation_nvfp4_static(x: Tensor, global_scale: float, block_size: int = 16) -> Tensor:
    """Fake-quantize activations to NVFP4 using a STATIC (calibrated) global scale.

    Unlike :func:`fake_quantize_activation_nvfp4` (which recomputes the global scale from
    ``x`` each call), this uses a **fixed** ``global_scale`` — the modelopt ``input_scale``
    = ``amax / (FP4_MAX * FP8_MAX)`` calibrated during training (running statistics).
    Per-group-of-``block_size`` FP8 block scales are still computed dynamically from ``x``.
    This matches modelopt/TRT-LLM static-input-scale NVFP4 activation quant (W4A4 inference).
    STE backward.
    """
    K = x.shape[-1]
    assert K % block_size == 0, f"activation last dim ({K}) must be divisible by block_size ({block_size})"
    orig_shape = x.shape
    x2d = x.reshape(-1, K).float()
    tiny = torch.finfo(torch.float32).tiny
    gs = torch.tensor(max(float(global_scale), tiny), device=x.device, dtype=torch.float32)
    blocks = x2d.reshape(-1, block_size)
    block_amax = blocks.abs().amax(dim=1, keepdim=True)
    block_scales_fp8 = (block_amax / FP4_E2M1_MAX / gs).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    effective = block_scales_fp8.float() * gs
    safe_eff = torch.where(effective > 0, effective, torch.ones_like(effective))
    values, _ = _round_to_e2m1(blocks / safe_eff)
    xq = (values * effective).reshape(x2d.shape).reshape(orig_shape).to(x.dtype)
    return x + (xq - x).detach()  # STE
