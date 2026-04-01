"""NF4 quantization/dequantization via Triton kernels (1D flat format).

NF4 uses per-group absmax scaling with a 16-level non-uniform codebook
optimized for normally distributed weights. Simpler than NVFP4: one scale
per group (no global_scale + block_scale two-level system).

Storage: packed uint8 (2 nf4 codes per byte) + float32 scales per group.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from .nf4_codec import _nf4_decode, _nf4_encode, get_nf4_lut


@triton.jit
def _nf4_quantize_kernel(
    X,
    Out,
    Scale,
    total_elems,
    GROUP_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    """Single-pass NF4 quantize: normalize by per-group absmax, encode to 4-bit codes."""
    pid = tl.program_id(0)
    base = pid * TILE_ELEMS
    half_tile: tl.constexpr = TILE_ELEMS // 2
    groups_per_tile: tl.constexpr = TILE_ELEMS // GROUP_SIZE
    base_group = pid * groups_per_tile
    offs = tl.arange(0, TILE_ELEMS)
    mask = (base + offs) < total_elems
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    # Per-group absmax
    x_2d = tl.reshape(x, (groups_per_tile, GROUP_SIZE))
    amax = tl.max(tl.abs(x_2d), axis=1)
    scale = tl.maximum(amax, 1e-12)
    # Store scales
    scale_offs = tl.arange(0, groups_per_tile)
    tl.store(Scale + base_group + scale_offs, scale, mask=(base_group + scale_offs) < (total_elems // GROUP_SIZE))
    # Normalize to [-1, 1] and encode
    inv_scale_2d = 1.0 / tl.expand_dims(scale, axis=1)
    norm_2d = x_2d * inv_scale_2d
    norm_2d = tl.minimum(tl.maximum(norm_2d, -1.0), 1.0)
    norm = tl.reshape(norm_2d, (TILE_ELEMS,))
    codes = _nf4_encode(norm)
    # Pack: lo nibble = even index, hi nibble = odd index
    is_odd = (offs & 1).to(tl.int32)
    shifted = (codes & 0xF) << (is_odd * 4)
    packed = tl.sum(tl.reshape(shifted, (half_tile, 2)), axis=1).to(tl.uint8)
    out_base = pid * half_tile
    out_offs = tl.arange(0, half_tile)
    tl.store(Out + out_base + out_offs, packed, mask=(out_base + out_offs) < (total_elems // 2))


@triton.jit
def _nf4_dequantize_kernel(
    Packed,
    Scale,
    LUT,
    Out,
    total_elems,
    GROUP_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    """NF4 dequantize: unpack codes, decode via LUT gather, scale by per-group absmax.

    Output trick: pack two bf16 values into one int32 store for coalesced writes.
    """
    pid = tl.program_id(0)
    half_tile: tl.constexpr = TILE_ELEMS // 2
    groups_per_tile: tl.constexpr = TILE_ELEMS // GROUP_SIZE
    half_gs: tl.constexpr = GROUP_SIZE // 2
    base_group = pid * groups_per_tile
    # Load scales
    scale_offs = tl.arange(0, groups_per_tile)
    scale = tl.load(
        Scale + base_group + scale_offs, mask=(base_group + scale_offs) < (total_elems // GROUP_SIZE), other=1.0
    ).to(tl.float32)
    # Load packed bytes
    pack_base = pid * half_tile
    pack_offs = tl.arange(0, half_tile)
    pack_mask = (pack_base + pack_offs) < (total_elems // 2)
    packed = tl.load(Packed + pack_base + pack_offs, mask=pack_mask, other=0).to(tl.int32)
    # Unpack
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    # Decode NF4 via LUT gather (16-element LUT, L1-cached)
    val_lo = _nf4_decode(lo, LUT)
    val_hi = _nf4_decode(hi, LUT)
    # Apply per-group scale
    scale_2d = tl.expand_dims(scale, axis=1)
    lo_2d = tl.reshape(val_lo, (groups_per_tile, half_gs))
    hi_2d = tl.reshape(val_hi, (groups_per_tile, half_gs))
    val_lo = tl.reshape(lo_2d * scale_2d, (half_tile,))
    val_hi = tl.reshape(hi_2d * scale_2d, (half_tile,))
    # Pack two bf16 into int32 for coalesced write
    lo_i16 = val_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    hi_i16 = val_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    word = (lo_i16 & 0xFFFF) | (hi_i16 << 16)
    tl.store(Out + pack_base + pack_offs, word, mask=pack_mask)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _get_config(group_size, total_elems, is_dequant=False):
    """Tile size and warp count for NF4 kernels."""
    if is_dequant:
        te, nw = 8192, 4  # Large tiles for bandwidth
    else:
        te, nw = 4096, 4
    while te % group_size != 0:
        te *= 2
    te = min(te, _next_pow2(total_elems))
    return te, nw


def nf4_quantize(x: Tensor, group_size: int = 64) -> Tuple[Tensor, Tensor]:
    """Quantize a 2D tensor to NF4 (flat output).

    Args:
        x: Weight tensor of shape [M, K]
        group_size: Number of elements per group for absmax scaling.

    Returns:
        packed: (M * K // 2,) uint8
        scales: (M * K // group_size,) float32
    """
    assert x.dim() == 2
    M, K = x.shape
    n = M * K
    assert n % group_size == 0 and group_size % 2 == 0
    x_flat = x.reshape(-1).contiguous()
    packed = torch.empty(n // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(n // group_size, dtype=torch.float32, device=x.device)
    te, nw = _get_config(group_size, n)
    grid = (n + te - 1) // te
    _nf4_quantize_kernel[(grid,)](
        x_flat,
        packed,
        scales,
        n,
        group_size,
        te,
        num_warps=nw,
    )
    return packed, scales


def nf4_dequantize(packed: Tensor, scales: Tensor, num_elements: int, group_size: int = 64) -> Tensor:
    """Dequantize NF4 packed data back to bfloat16 (flat output).

    Args:
        packed: Flat uint8 packed data
        scales: Flat float32 per-group scales
        num_elements: Total number of original elements
        group_size: Group size used during quantization

    Returns:
        Flat bfloat16 tensor of num_elements values
    """
    assert num_elements % group_size == 0
    packed_flat = packed.reshape(-1)
    scales_flat = scales.float().reshape(-1) if scales.dtype != torch.float32 else scales.reshape(-1)
    out_i32 = torch.empty(num_elements // 2, dtype=torch.int32, device=packed.device)
    lut = get_nf4_lut(packed.device)
    te, nw = _get_config(group_size, num_elements, is_dequant=True)
    grid = (num_elements + te - 1) // te
    _nf4_dequantize_kernel[(grid,)](
        packed_flat,
        scales_flat,
        lut,
        out_i32,
        num_elements,
        group_size,
        te,
        num_warps=nw,
    )
    return out_i32.view(torch.bfloat16)
