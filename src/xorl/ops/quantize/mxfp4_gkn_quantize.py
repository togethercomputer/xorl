"""MXFP4 quantization/dequantization for G,K,N format via 2D Triton kernels.

Input weights are in (G,K,N) format: [in_features, out_features] per expert.
Groups are formed along the K (first) dimension — the contraction dimension
for matmul x @ W — matching HF-format grouping quality.

Packed output: [K//2, N] — pairs of consecutive K-elements per column.
Scales output: [K//block_size, N] — one E8M0 scale per block per column.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from .fp4_codec import _fp4_decode, _fp4_encode


@triton.jit
def _mxfp4_quantize_gkn_kernel(
    X,
    Out,
    Scale,
    K,
    N,
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Quantize [K, N] weight with MXFP4 groups along K dimension.

    2D grid: (K // BLOCK_SIZE, ceil(N / TILE_N))
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
    amax = tl.maximum(tl.max(tl.abs(x_even), axis=0), tl.max(tl.abs(x_odd), axis=0))  # [TN]
    scale = tl.maximum(amax, 1e-12) / 6.0
    # Bitwise E8M0 rounding: multiply by sqrt(2) then zero mantissa bits
    scale = ((scale * 1.4142135623730951).to(tl.int32, bitcast=True) & 0x7F800000).to(tl.float32, bitcast=True)

    # Store scales
    scale_addrs = Scale + pid_group * N + (base_n + n_offs)
    tl.store(scale_addrs, scale, mask=n_mask)

    # Scale and encode even/odd separately
    inv_scale = 1.0 / scale[None, :]  # [1, TN]
    scaled_even = x_even * inv_scale  # [BS//2, TN]
    scaled_odd = x_odd * inv_scale  # [BS//2, TN]

    # FP4 encode (element-wise, works directly on 2D)
    codes_even = _fp4_encode(scaled_even)
    codes_odd = _fp4_encode(scaled_odd)

    # Pack: lo nibble = even K, hi nibble = odd K
    packed = ((codes_even & 0xF) | ((codes_odd & 0xF) << 4)).to(tl.uint8)  # [BS//2, TN]

    # 2D store
    out_base_k = pid_group * half_bs
    out_addrs = Out + (out_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    out_mask = ((out_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    tl.store(out_addrs, packed, mask=out_mask)


@triton.jit
def _mxfp4_dequantize_gkn_kernel(
    Packed,
    Scale,
    Out,
    K,
    N,
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Dequantize MXFP4 [K//2, N] packed → [K, N] bf16 output directly.

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

    # Load scales
    scale_addrs = Scale + pid_group * N + (base_n + n_offs)
    scale = tl.load(scale_addrs, mask=n_mask, other=1.0).to(tl.float32)  # [TN]

    # Load packed
    pack_base_k = pid_group * half_bs
    pack_addrs = Packed + (pack_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    pack_mask = ((pack_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    packed = tl.load(pack_addrs, mask=pack_mask, other=0).to(tl.int32)  # [BS//2, TN]

    # Unpack and decode FP4 (element-wise, works directly on 2D)
    lo = packed & 0xF  # even K [BS//2, TN]
    hi = (packed >> 4) & 0xF  # odd K [BS//2, TN]
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


def mxfp4_quantize_gkn(x: Tensor, block_size: int = 32) -> Tuple[Tensor, Tensor]:
    """Quantize a [K, N] weight tensor in G,K,N format to MXFP4.

    Groups are formed along the K (first) dimension.

    Args:
        x: Weight tensor of shape [K, N]
        block_size: Number of K-elements per block.

    Returns:
        packed: [K//2, N] uint8 — pairs of K-elements packed per column
        scales: [K//block_size, N] float16 — one E8M0 scale per block per column
    """
    assert x.dim() == 2
    K, N = x.shape
    assert K % block_size == 0 and block_size % 2 == 0

    num_blocks = K // block_size
    packed = torch.empty(K // 2, N, dtype=torch.uint8, device=x.device)
    # E8M0 scale stored as float32 in kernel, converted to fp16 for storage
    scales = torch.empty(num_blocks, N, dtype=torch.float32, device=x.device)

    TILE_N = min(128, _next_pow2(N))
    grid = (num_blocks, (N + TILE_N - 1) // TILE_N)

    _mxfp4_quantize_gkn_kernel[grid](
        x.contiguous(),
        packed,
        scales,
        K,
        N,
        block_size,
        TILE_N,
        num_warps=4,
    )
    return packed, scales.to(torch.float16)


def mxfp4_dequantize_gkn(packed: Tensor, scales: Tensor, K: int, N: int, block_size: int = 32) -> Tensor:
    """Dequantize MXFP4 packed [K//2, N] back to [K, N] in G,K,N format.

    Args:
        packed: [K//2, N] uint8
        scales: [K//block_size, N] float16
        K: Original first dimension (in_features)
        N: Original second dimension (out_features)
        block_size: Block size used during quantization

    Returns:
        Tensor of shape [K, N] in bfloat16
    """
    num_blocks = K // block_size
    result = torch.empty(K, N, dtype=torch.bfloat16, device=packed.device)

    TILE_N = min(128, _next_pow2(N))
    grid = (num_blocks, (N + TILE_N - 1) // TILE_N)

    _mxfp4_dequantize_gkn_kernel[grid](
        packed.contiguous(),
        scales.float().contiguous(),
        result,
        K,
        N,
        block_size,
        TILE_N,
        num_warps=2,
    )
    return result
