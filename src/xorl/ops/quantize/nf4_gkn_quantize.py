"""NF4 quantization/dequantization for G,K,N format via 2D Triton kernels.

Input weights are in (G,K,N) format: [in_features, out_features] per expert.
Groups are formed along the K (first) dimension -- the contraction dimension
for matmul x @ W -- matching HF-format grouping quality.

Packed output: [K//2, N] -- pairs of consecutive K-elements per column.
Scales output: [K//group_size, N] -- one float32 scale per group per column.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from .nf4_codec import _nf4_decode, _nf4_encode, get_nf4_lut


@triton.jit
def _nf4_quantize_gkn_kernel(
    X,
    Out,
    Scale,
    K,
    N,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """NF4 quantize [K, N] with groups along K.

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
    x_even = tl.load(even_addrs, mask=even_mask, other=0.0).to(tl.float32)

    # Load odd K rows: x[base_k + 2*i + 1, base_n + j]
    odd_addrs = X + (base_k + 2 * k_pair[:, None] + 1) * N + (base_n + n_offs[None, :])
    odd_mask = ((base_k + 2 * k_pair[:, None] + 1) < K) & n_mask[None, :]
    x_odd = tl.load(odd_addrs, mask=odd_mask, other=0.0).to(tl.float32)

    # Per-column absmax across all GROUP_SIZE rows
    amax = tl.maximum(tl.max(tl.abs(x_even), axis=0), tl.max(tl.abs(x_odd), axis=0))
    scale = tl.maximum(amax, 1e-12)

    # Store scales: Scale[pid_group, base_n + n]
    scale_addrs = Scale + pid_group * N + (base_n + n_offs)
    tl.store(scale_addrs, scale, mask=n_mask)

    # Normalize to [-1, 1] and encode
    inv_scale = 1.0 / scale[None, :]
    norm_even = tl.minimum(tl.maximum(x_even * inv_scale, -1.0), 1.0)
    norm_odd = tl.minimum(tl.maximum(x_odd * inv_scale, -1.0), 1.0)

    codes_even = _nf4_encode(norm_even)
    codes_odd = _nf4_encode(norm_odd)

    # Pack: lo nibble = even K, hi nibble = odd K
    packed = ((codes_even & 0xF) | ((codes_odd & 0xF) << 4)).to(tl.uint8)

    # 2D store: Out[base_k//2 + i, base_n + n]
    out_base_k = pid_group * half_gs
    out_addrs = Out + (out_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    out_mask = ((out_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    tl.store(out_addrs, packed, mask=out_mask)


@triton.jit
def _nf4_dequantize_gkn_kernel(
    Packed,
    Scale,
    LUT,
    Out,
    K,
    N,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Dequantize NF4 [K//2, N] packed -> [K, N] bf16 output directly.

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
    scale = tl.load(scale_addrs, mask=n_mask, other=1.0).to(tl.float32)

    # Load packed: Packed[base_k//2 + i, base_n + n]
    pack_base_k = pid_group * half_gs
    pack_addrs = Packed + (pack_base_k + k_pair[:, None]) * N + (base_n + n_offs[None, :])
    pack_mask = ((pack_base_k + k_pair[:, None]) < (K // 2)) & n_mask[None, :]
    packed = tl.load(pack_addrs, mask=pack_mask, other=0).to(tl.int32)

    # Unpack and decode NF4 via LUT gather
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    val_lo = _nf4_decode(lo, LUT)
    val_hi = _nf4_decode(hi, LUT)

    # Apply scale -> bf16
    scale_row = scale[None, :]
    val_even = (val_lo * scale_row).to(tl.bfloat16)
    val_odd = (val_hi * scale_row).to(tl.bfloat16)

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


def nf4_quantize_gkn(x: Tensor, group_size: int = 64) -> Tuple[Tensor, Tensor]:
    """Quantize a [K, N] weight tensor in G,K,N format to NF4.

    Groups are formed along the K (first) dimension.

    Args:
        x: Weight tensor of shape [K, N]
        group_size: Number of K-elements per group.

    Returns:
        packed: [K//2, N] uint8
        scales: [K//group_size, N] float32
    """
    assert x.dim() == 2
    K, N = x.shape
    assert K % group_size == 0 and group_size % 2 == 0

    num_groups = K // group_size
    packed = torch.empty(K // 2, N, dtype=torch.uint8, device=x.device)
    scales = torch.empty(num_groups, N, dtype=torch.float32, device=x.device)

    TILE_N = min(128, _next_pow2(N))
    grid = (num_groups, (N + TILE_N - 1) // TILE_N)

    _nf4_quantize_gkn_kernel[grid](
        x.contiguous(),
        packed,
        scales,
        K,
        N,
        group_size,
        TILE_N,
        num_warps=4,
    )
    return packed, scales


def nf4_dequantize_gkn(packed: Tensor, scales: Tensor, K: int, N: int, group_size: int = 64) -> Tensor:
    """Dequantize NF4 packed [K//2, N] back to [K, N] in G,K,N format.

    Args:
        packed: [K//2, N] uint8
        scales: [K//group_size, N] float32
        K: Original first dimension (in_features)
        N: Original second dimension (out_features)
        group_size: Group size used during quantization

    Returns:
        Tensor of shape [K, N] in bfloat16
    """
    num_groups = K // group_size
    result = torch.empty(K, N, dtype=torch.bfloat16, device=packed.device)

    TILE_N = min(256, _next_pow2(N))
    grid = (num_groups, (N + TILE_N - 1) // TILE_N)

    scales_f32 = scales.float() if scales.dtype != torch.float32 else scales
    lut = get_nf4_lut(packed.device)

    _nf4_dequantize_gkn_kernel[grid](
        packed.contiguous(),
        scales_f32.contiguous(),
        lut,
        result,
        K,
        N,
        group_size,
        TILE_N,
        num_warps=4,
    )
    return result
