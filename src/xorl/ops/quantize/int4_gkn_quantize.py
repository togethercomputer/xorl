"""INT4 quantization/dequantization for G,K,N format via 2D Triton kernels.

Input weights are in (G,K,N) format: [in_features, out_features] per expert.
Groups are formed along the K (first) dimension — the contraction dimension
for matmul x @ W — matching HF-format grouping quality.

Packed output: [K//2, N] — pairs of consecutive K-elements per column.
Scales output: [K//group_size, N] — one scale per group per column.
"""

from typing import Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _int4_quantize_gkn_kernel(
    X, Out, Scale,
    K, N,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Quantize [K, N] weight with groups along K dimension.

    2D grid: (K // GROUP_SIZE, ceil(N / TILE_N))
    Each program processes [GROUP_SIZE, TILE_N] elements.
    Loads even/odd K rows separately to avoid Triton 2D slicing.
    """
    pid_group = tl.program_id(0)
    pid_n = tl.program_id(1)
    base_k = pid_group * GROUP_SIZE
    base_n = pid_n * TILE_N
    half_gs: tl.constexpr = GROUP_SIZE // 2

    k_pair = tl.arange(0, half_gs)
    n_offs = tl.arange(0, TILE_N)
    n_mask = (base_n + n_offs) < N

    # Load even K rows: x[base_k + 2*i, base_n + j]
    even_addrs = X + (base_k + 2 * k_pair[:, None]) * N + (base_n + n_offs[None, :])
    even_mask = ((base_k + 2 * k_pair[:, None]) < K) & n_mask[None, :]
    x_even = tl.load(even_addrs, mask=even_mask, other=0.0).to(tl.float32)  # [GS//2, TN]

    # Load odd K rows: x[base_k + 2*i + 1, base_n + j]
    odd_addrs = X + (base_k + 2 * k_pair[:, None] + 1) * N + (base_n + n_offs[None, :])
    odd_mask = ((base_k + 2 * k_pair[:, None] + 1) < K) & n_mask[None, :]
    x_odd = tl.load(odd_addrs, mask=odd_mask, other=0.0).to(tl.float32)  # [GS//2, TN]

    # Per-column absmax across all GROUP_SIZE rows
    amax = tl.maximum(tl.max(tl.abs(x_even), axis=0), tl.max(tl.abs(x_odd), axis=0))  # [TN]
    scale = tl.maximum(amax, 1e-12) * (2.0 / 15.0)

    # Store scales: Scale[pid_group, base_n + n]
    scale_addrs = Scale + pid_group * N + (base_n + n_offs)
    tl.store(scale_addrs, scale, mask=n_mask)

    # Quantize even and odd rows separately
    inv_scale = 1.0 / scale[None, :]  # [1, TN]
    q_even = tl.maximum(0, tl.minimum(15, (x_even * inv_scale + 8.5).to(tl.int32)))  # [GS//2, TN]
    q_odd = tl.maximum(0, tl.minimum(15, (x_odd * inv_scale + 8.5).to(tl.int32)))  # [GS//2, TN]

    # Pack: lo nibble = even K, hi nibble = odd K
    packed = ((q_even & 0xF) | ((q_odd & 0xF) << 4)).to(tl.uint8)  # [GS//2, TN]

    # 2D store: Out[base_k//2 + i, base_n + n]
    out_base_k = pid_group * half_gs
    out_addrs = Out + (out_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    out_mask = ((out_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    tl.store(out_addrs, packed, mask=out_mask)


@triton.jit
def _int4_dequantize_gkn_kernel(
    Packed, Scale, Out,
    K, N,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Dequantize [K//2, N] packed → [K, N] bf16 output directly.

    2D grid: (K // GROUP_SIZE, ceil(N / TILE_N))
    Writes even/odd K rows directly to output, no Python interleave needed.
    """
    pid_group = tl.program_id(0)
    pid_n = tl.program_id(1)
    base_k = pid_group * GROUP_SIZE
    base_n = pid_n * TILE_N
    half_gs: tl.constexpr = GROUP_SIZE // 2

    k_pair = tl.arange(0, half_gs)
    n_offs = tl.arange(0, TILE_N)
    n_mask = (base_n + n_offs) < N

    # Load scales: Scale[pid_group, base_n + n]
    scale_addrs = Scale + pid_group * N + (base_n + n_offs)
    scale = tl.load(scale_addrs, mask=n_mask, other=1.0).to(tl.float32)  # [TN]

    # Load packed: Packed[base_k//2 + i, base_n + n]
    pack_base_k = pid_group * half_gs
    pack_addrs = Packed + (pack_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    pack_mask = ((pack_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    packed = tl.load(pack_addrs, mask=pack_mask, other=0).to(tl.int32)  # [GS//2, TN]

    # Unpack lo/hi nibbles
    lo = (packed & 0xF).to(tl.float32)   # even K values [GS//2, TN]
    hi = ((packed >> 4) & 0xF).to(tl.float32)  # odd K values [GS//2, TN]

    # Dequantize: (nibble - 8) * scale → bf16
    scale_row = scale[None, :]  # [1, TN]
    val_even = ((lo - 8.0) * scale_row).to(tl.bfloat16)  # [GS//2, TN]
    val_odd = ((hi - 8.0) * scale_row).to(tl.bfloat16)  # [GS//2, TN]

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


def int4_quantize_gkn(x: Tensor, group_size: int = -1) -> Tuple[Tensor, Tensor]:
    """Quantize a [K, N] weight tensor in G,K,N format to INT4.

    Groups are formed along the K (first) dimension.

    Args:
        x: Weight tensor of shape [K, N]
        group_size: Number of K-elements per group. -1 means entire K dimension.

    Returns:
        packed: [K//2, N] uint8 — pairs of K-elements packed per column
        scales: [K//group_size, N] float32 — one scale per group per column
    """
    assert x.dim() == 2
    K, N = x.shape
    gs = K if group_size == -1 else group_size
    assert K % gs == 0 and gs % 2 == 0

    num_groups = K // gs
    packed = torch.empty(K // 2, N, dtype=torch.uint8, device=x.device)
    scales = torch.empty(num_groups, N, dtype=torch.float32, device=x.device)

    # TILE_N: process multiple columns per program for efficiency
    TILE_N = min(64, _next_pow2(N))
    grid = (num_groups, (N + TILE_N - 1) // TILE_N)

    _int4_quantize_gkn_kernel[grid](
        x.contiguous(), packed, scales,
        K, N,
        gs, TILE_N,
        num_warps=4,
    )
    return packed, scales


def int4_dequantize_gkn(packed: Tensor, scales: Tensor, K: int, N: int,
                         group_size: int = -1,
                         out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Dequantize INT4 packed [K//2, N] back to [K, N] in G,K,N format.

    Args:
        packed: [K//2, N] uint8
        scales: [K//group_size, N] float32
        K: Original first dimension (in_features)
        N: Original second dimension (out_features)
        group_size: Group size used during quantization
        out_dtype: Output dtype (default bfloat16)

    Returns:
        Tensor of shape [K, N]
    """
    gs = K if group_size == -1 else group_size
    num_groups = K // gs

    # Output buffer: [K, N] bf16 — kernel writes directly to even/odd rows
    result = torch.empty(K, N, dtype=torch.bfloat16, device=packed.device)

    TILE_N = min(128, _next_pow2(N))
    grid = (num_groups, (N + TILE_N - 1) // TILE_N)

    scales_f32 = scales.float() if scales.dtype != torch.float32 else scales

    _int4_dequantize_gkn_kernel[grid](
        packed.contiguous(), scales_f32.contiguous(), result,
        K, N,
        gs, TILE_N,
        num_warps=4,
    )

    if out_dtype != torch.bfloat16:
        result = result.to(out_dtype)
    return result
