"""MXFP4 quantization/dequantization via Triton kernels."""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from .fp4_codec import _fp4_decode, _fp4_encode


_QUANT_CONFIGS = {
    16: (1024, 1),
    32: (1024, 1),
    64: (1024, 1),
    128: (1024, 1),
    256: (2048, 2),
}

_DEQUANT_CONFIGS = {
    16: (4096, 4),
    32: (4096, 4),
    64: (2048, 1),
    128: (2048, 1),
    256: (2048, 1),
}


@triton.jit
def _mxfp4_quantize_kernel(
    X,
    Out,
    Scale,
    total_elems,
    BLOCK_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * TILE_ELEMS
    half_tile: tl.constexpr = TILE_ELEMS // 2
    blocks_per_tile: tl.constexpr = TILE_ELEMS // BLOCK_SIZE
    base_block = pid * blocks_per_tile
    offs = tl.arange(0, TILE_ELEMS)
    mask = (base + offs) < total_elems
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    x_2d = tl.reshape(x, (blocks_per_tile, BLOCK_SIZE))
    amax = tl.max(tl.abs(x_2d), axis=1)
    scale = tl.maximum(amax, 1e-12) / 6.0
    # Bitwise E8M0 rounding: multiply by sqrt(2) then zero mantissa bits
    scale = ((scale * 1.4142135623730951).to(tl.int32, bitcast=True) & 0x7F800000).to(tl.float32, bitcast=True)
    scale_offs = tl.arange(0, blocks_per_tile)
    tl.store(Scale + base_block + scale_offs, scale, mask=(base_block + scale_offs) < (total_elems // BLOCK_SIZE))
    inv_scale_2d = 1.0 / tl.expand_dims(scale, axis=1)
    scaled_2d = x_2d * inv_scale_2d
    scaled = tl.reshape(scaled_2d, (TILE_ELEMS,))
    codes = _fp4_encode(scaled)
    is_odd = (offs & 1).to(tl.int32)
    shifted = (codes & 0xF) << (is_odd * 4)
    packed = tl.sum(tl.reshape(shifted, (half_tile, 2)), axis=1).to(tl.uint8)
    out_base = pid * half_tile
    out_offs = tl.arange(0, half_tile)
    tl.store(Out + out_base + out_offs, packed, mask=(out_base + out_offs) < (total_elems // 2))


@triton.jit
def _mxfp4_dequantize_kernel(
    Packed,
    Scale,
    Out,
    total_elems,
    BLOCK_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    half_tile: tl.constexpr = TILE_ELEMS // 2
    blocks_per_tile: tl.constexpr = TILE_ELEMS // BLOCK_SIZE
    pairs_per_block: tl.constexpr = BLOCK_SIZE // 2
    base_block = pid * blocks_per_tile
    scale_offs = tl.arange(0, blocks_per_tile)
    scale_mask = (base_block + scale_offs) < (total_elems // BLOCK_SIZE)
    scale_vec = tl.load(Scale + base_block + scale_offs, mask=scale_mask, other=1.0).to(tl.float32)
    pack_base = pid * half_tile
    pack_offs = tl.arange(0, half_tile)
    pack_mask = (pack_base + pack_offs) < (total_elems // 2)
    packed = tl.load(Packed + pack_base + pack_offs, mask=pack_mask, other=0).to(tl.int32)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    val_lo = _fp4_decode(lo)
    val_hi = _fp4_decode(hi)
    scale_2d = tl.expand_dims(scale_vec, axis=1)
    lo_2d = tl.reshape(val_lo, (blocks_per_tile, pairs_per_block))
    hi_2d = tl.reshape(val_hi, (blocks_per_tile, pairs_per_block))
    val_lo = tl.reshape(lo_2d * scale_2d, (half_tile,))
    val_hi = tl.reshape(hi_2d * scale_2d, (half_tile,))
    lo_i16 = val_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    hi_i16 = val_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    word = (lo_i16 & 0xFFFF) | (hi_i16 << 16)
    tl.store(Out + pack_base + pack_offs, word, mask=pack_mask)


def _get_config(block_size, total_elems, is_dequant=False):
    configs = _DEQUANT_CONFIGS if is_dequant else _QUANT_CONFIGS
    te, nw = configs.get(block_size, (1024, 2))
    while te % block_size != 0:
        te *= 2
    te = min(te, total_elems)
    return te, nw


def mxfp4_quantize(x: Tensor, block_size: int = 32) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 2
    M, K = x.shape
    n = M * K
    assert n % block_size == 0 and block_size % 2 == 0
    x_flat = x.reshape(-1).contiguous()
    packed = torch.empty(n // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(n // block_size, dtype=torch.float16, device=x.device)
    te, nw = _get_config(block_size, n)
    grid = (n + te - 1) // te
    _mxfp4_quantize_kernel[(grid,)](x_flat, packed, scales, n, block_size, te, num_warps=nw)
    return packed, scales


def mxfp4_dequantize(packed: Tensor, scales: Tensor, num_elements: int, block_size: int = 32) -> Tensor:
    assert num_elements % block_size == 0
    packed_flat = packed.reshape(-1)
    out_i32 = torch.empty(num_elements // 2, dtype=torch.int32, device=packed_flat.device)
    te, nw = _get_config(block_size, num_elements, is_dequant=True)
    grid = (num_elements + te - 1) // te
    _mxfp4_dequantize_kernel[(grid,)](
        packed_flat,
        scales,
        out_i32,
        num_elements,
        block_size,
        te,
        num_warps=nw,
    )
    return out_i32.view(torch.bfloat16)
