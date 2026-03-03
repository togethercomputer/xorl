"""NVFP4 quantization/dequantization for G,K,N format via 2D Triton kernels.

Input weights are in (G,K,N) format: [in_features, out_features] per expert.
Groups are formed along the K (first) dimension — the contraction dimension
for matmul x @ W — matching HF-format grouping quality.

Packed output: [K//2, N] — pairs of consecutive K-elements per column.
Block scales: [K//block_size, N] — one fp8 scale per block per column.
Global scale: scalar float32.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

from .fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX, _fp4_encode, _fp4_encode_stochastic, _fp4_decode


@triton.jit
def _nvfp4_quantize_gkn_kernel(
    X, Out, Amax, GlobalAmax,
    K, N, clip_ratio, seed,
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
    STOCHASTIC: tl.constexpr,
):
    """Single-pass NVFP4 quant for [K, N] with groups along K.

    2D grid: (K // BLOCK_SIZE, ceil(N / TILE_N))
    Codes depend only on per-block amax, not global scale.
    Loads even/odd K rows separately to avoid Triton 2D slicing.
    """
    pid_group = tl.program_id(0)
    pid_n = tl.program_id(1)
    base_k = pid_group * BLOCK_SIZE
    base_n = pid_n * TILE_N
    half_bs: tl.constexpr = BLOCK_SIZE // 2

    k_pair = tl.arange(0, half_bs)
    n_offs = tl.arange(0, TILE_N)
    n_mask = (base_n + n_offs) < N

    # Load even K rows: x[base_k + 2*i, base_n + j]
    even_addrs = X + (base_k + 2 * k_pair[:, None]) * N + (base_n + n_offs[None, :])
    even_mask = ((base_k + 2 * k_pair[:, None]) < K) & n_mask[None, :]
    x_even = tl.load(even_addrs, mask=even_mask, other=0.0).to(tl.float32)  # [BS//2, TN]

    # Load odd K rows: x[base_k + 2*i + 1, base_n + j]
    odd_addrs = X + (base_k + 2 * k_pair[:, None] + 1) * N + (base_n + n_offs[None, :])
    odd_mask = ((base_k + 2 * k_pair[:, None] + 1) < K) & n_mask[None, :]
    x_odd = tl.load(odd_addrs, mask=odd_mask, other=0.0).to(tl.float32)  # [BS//2, TN]

    # Per-column absmax across all BLOCK_SIZE rows
    amax_val = tl.maximum(tl.max(tl.abs(x_even), axis=0), tl.max(tl.abs(x_odd), axis=0))  # [TN]
    amax_val = tl.maximum(amax_val, 1e-12)

    # Apply clip_ratio: reduce effective range to clip outliers
    clipped_amax = amax_val * clip_ratio
    clipped_amax = tl.maximum(clipped_amax, 1e-12)

    # Encode FP4 (codes don't depend on global scale)
    inv_amax = 6.0 / clipped_amax[None, :]  # [1, TN]
    scaled_even = x_even * inv_amax  # [BS//2, TN]
    scaled_odd = x_odd * inv_amax  # [BS//2, TN]
    # Clamp for clipped outliers
    scaled_even = tl.minimum(tl.maximum(scaled_even, -6.0), 6.0)
    scaled_odd = tl.minimum(tl.maximum(scaled_odd, -6.0), 6.0)

    # FP4 encode (element-wise, works directly on 2D)
    if STOCHASTIC:
        # Generate 2D noise for even and odd rows
        even_seed_offs = (base_k + 2 * k_pair[:, None]) * N + (base_n + n_offs[None, :])
        odd_seed_offs = (base_k + 2 * k_pair[:, None] + 1) * N + (base_n + n_offs[None, :])
        noise_even = tl.rand(seed, even_seed_offs)
        noise_odd = tl.rand(seed, odd_seed_offs)
        codes_even = _fp4_encode_stochastic(scaled_even, noise_even)
        codes_odd = _fp4_encode_stochastic(scaled_odd, noise_odd)
    else:
        codes_even = _fp4_encode(scaled_even)
        codes_odd = _fp4_encode(scaled_odd)

    # Store per-column amax: Amax[pid_group, base_n + n]
    amax_addrs = Amax + pid_group * N + (base_n + n_offs)
    tl.store(amax_addrs, amax_val, mask=n_mask)

    # Update global amax (atomic across all programs)
    tl.atomic_max(GlobalAmax, tl.max(amax_val))

    # Pack: lo nibble = even K, hi nibble = odd K
    packed = ((codes_even & 0xF) | ((codes_odd & 0xF) << 4)).to(tl.uint8)  # [BS//2, TN]

    # 2D store
    out_base_k = pid_group * half_bs
    out_addrs = Out + (out_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    out_mask = ((out_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    tl.store(out_addrs, packed, mask=out_mask)


@triton.jit
def _nvfp4_scale_convert_gkn_kernel(
    Amax, ScaleOut, GlobalAmaxInOut,
    n_scales, FP4_MAX_INV, FP8_MAX,
    BLOCK: tl.constexpr,
):
    """Convert raw amax to fp8 block_scales (operates on flat amax array)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_scales
    amax = tl.load(Amax + offs, mask=mask, other=1.0)
    gamax = tl.maximum(tl.load(GlobalAmaxInOut), 1e-12)
    tl.store(ScaleOut + offs, (amax / gamax * FP8_MAX).to(tl.float8e4nv), mask=mask)
    if pid == 0:
        tl.store(GlobalAmaxInOut, gamax * FP4_MAX_INV / FP8_MAX)


@triton.jit
def _nvfp4_dequantize_gkn_kernel(
    Packed, Scale, GlobalScale, Out,
    K, N,
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Dequantize NVFP4 [K//2, N] packed → [K, N] bf16 output directly.

    2D grid: (K // BLOCK_SIZE, ceil(N / TILE_N))
    Writes even/odd K rows directly to output, no Python interleave needed.
    """
    pid_group = tl.program_id(0)
    pid_n = tl.program_id(1)
    base_k = pid_group * BLOCK_SIZE
    base_n = pid_n * TILE_N
    half_bs: tl.constexpr = BLOCK_SIZE // 2

    k_pair = tl.arange(0, half_bs)
    n_offs = tl.arange(0, TILE_N)
    n_mask = (base_n + n_offs) < N

    # Load global scale
    gs = tl.load(GlobalScale).to(tl.float32)

    # Load per-column block scales: Scale[pid_group, base_n + n]
    scale_addrs = Scale + pid_group * N + (base_n + n_offs)
    scale = tl.load(scale_addrs, mask=n_mask, other=1.0).to(tl.float32) * gs  # [TN]

    # Load packed
    pack_base_k = pid_group * half_bs
    pack_addrs = Packed + (pack_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    pack_mask = ((pack_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    packed = tl.load(pack_addrs, mask=pack_mask, other=0).to(tl.int32)  # [BS//2, TN]

    # Unpack and decode FP4 (element-wise, works directly on 2D)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    val_lo = _fp4_decode(lo)
    val_hi = _fp4_decode(hi)

    # Apply scale → bf16
    scale_row = scale[None, :]  # [1, TN]
    val_even = (val_lo * scale_row).to(tl.bfloat16)  # [BS//2, TN]
    val_odd = (val_hi * scale_row).to(tl.bfloat16)  # [BS//2, TN]

    # Write directly to even K rows: Out[base_k + 2*i, base_n + j]
    even_addrs = Out + (base_k + 2 * k_pair[:, None]) * N + (base_n + n_offs[None, :])
    even_mask = ((base_k + 2 * k_pair[:, None]) < K) & n_mask[None, :]
    tl.store(even_addrs, val_even, mask=even_mask)

    # Write directly to odd K rows: Out[base_k + 2*i + 1, base_n + j]
    odd_addrs = Out + (base_k + 2 * k_pair[:, None] + 1) * N + (base_n + n_offs[None, :])
    odd_mask = ((base_k + 2 * k_pair[:, None] + 1) < K) & n_mask[None, :]
    tl.store(odd_addrs, val_odd, mask=odd_mask)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def nvfp4_quantize_gkn(x: Tensor, block_size: int = 16,
                       global_amax: Optional[Tensor] = None,
                       clip_ratio: float = 1.0,
                       stochastic_rounding: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize a [K, N] weight tensor in G,K,N format to NVFP4.

    Groups are formed along the K (first) dimension.

    Args:
        x: Weight tensor of shape [K, N]
        block_size: Number of K-elements per block.
        global_amax: Optional EMA-tracked amax to override computed global amax.
        clip_ratio: Scale clipping ratio (0.0-1.0). Lower = more clipping.
        stochastic_rounding: If True, use stochastic rounding in FP4 encode.

    Returns:
        packed: [K//2, N] uint8
        block_scales: [K//block_size, N] float8_e4m3fn
        global_scale: (1,) float32
    """
    assert x.dim() == 2
    K, N = x.shape
    assert K % block_size == 0 and block_size % 2 == 0

    num_blocks = K // block_size
    packed = torch.empty(K // 2, N, dtype=torch.uint8, device=x.device)
    amax = torch.empty(num_blocks, N, dtype=torch.float32, device=x.device)
    global_amax_buf = torch.zeros(1, dtype=torch.float32, device=x.device)

    TILE_N = min(128, _next_pow2(N))
    grid = (num_blocks, (N + TILE_N - 1) // TILE_N)

    seed = 0 if not stochastic_rounding else torch.randint(0, 2**31, (1,)).item()
    _nvfp4_quantize_gkn_kernel[grid](
        x.contiguous(), packed, amax, global_amax_buf,
        K, N, clip_ratio, seed,
        block_size, TILE_N, stochastic_rounding,
        num_warps=4,
    )

    # Override global amax with EMA-tracked value if provided
    if global_amax is not None:
        global_amax_buf.copy_(global_amax)

    # Convert raw amax to fp8 block_scales (flat operation)
    n_scales = num_blocks * N
    block_scales = torch.empty(n_scales, dtype=torch.float8_e4m3fn, device=x.device)
    SC_BLOCK = 1024
    _nvfp4_scale_convert_gkn_kernel[((n_scales + SC_BLOCK - 1) // SC_BLOCK,)](
        amax.reshape(-1), block_scales, global_amax_buf, n_scales,
        1.0 / FP4_E2M1_MAX, FP8_E4M3_MAX, SC_BLOCK,
        num_warps=4,
    )
    return packed, block_scales.reshape(num_blocks, N), global_amax_buf


def nvfp4_dequantize_gkn(packed: Tensor, block_scales: Tensor, global_scale: Tensor,
                          K: int, N: int, block_size: int = 16) -> Tensor:
    """Dequantize NVFP4 packed [K//2, N] back to [K, N] in G,K,N format.

    Args:
        packed: [K//2, N] uint8
        block_scales: [K//block_size, N] float8_e4m3fn
        global_scale: (1,) float32
        K: Original first dimension (in_features)
        N: Original second dimension (out_features)
        block_size: Block size used during quantization

    Returns:
        Tensor of shape [K, N] in bfloat16
    """
    num_blocks = K // block_size
    result = torch.empty(K, N, dtype=torch.bfloat16, device=packed.device)

    TILE_N = min(256, _next_pow2(N))
    grid = (num_blocks, (N + TILE_N - 1) // TILE_N)

    gs = global_scale if global_scale.numel() == 1 else global_scale.reshape(1)

    _nvfp4_dequantize_gkn_kernel[grid](
        packed.contiguous(), block_scales.contiguous(), gs, result,
        K, N,
        block_size, TILE_N,
        num_warps=1,
    )
    return result
