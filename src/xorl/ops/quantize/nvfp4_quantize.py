"""NVFP4 quantization/dequantization via Triton kernels."""
from typing import Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

from .fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX, _fp4_encode, _fp4_decode


_QUANT_CONFIGS = {
    16: (8192, 4),
    32: (8192, 4),
}

_DEQUANT_CONFIGS = {
    16: (4096, 4),
    32: (4096, 4),
}


@triton.jit
def _nvfp4_quantize_kernel(
    X, Out, Amax, GlobalAmax,
    total_elems,
    BLOCK_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    """Single-pass NVFP4 quant: codes depend only on per-block amax, not global scale."""
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
    amax = tl.maximum(amax, 1e-12)
    # Compute codes before stores to overlap compute with memory
    inv_amax_2d = (6.0 / tl.expand_dims(amax, axis=1))
    scaled_2d = x_2d * inv_amax_2d
    scaled_2d = tl.minimum(tl.maximum(scaled_2d, -6.0), 6.0)
    scaled = tl.reshape(scaled_2d, (TILE_ELEMS,))
    codes = _fp4_encode(scaled)
    scale_offs = tl.arange(0, blocks_per_tile)
    tl.store(Amax + base_block + scale_offs, amax,
             mask=(base_block + scale_offs) < (total_elems // BLOCK_SIZE))
    tl.atomic_max(GlobalAmax, tl.max(amax))
    is_odd = (offs & 1).to(tl.int32)
    shifted = (codes & 0xF) << (is_odd * 4)
    packed = tl.sum(tl.reshape(shifted, (half_tile, 2)), axis=1).to(tl.uint8)
    out_base = pid * half_tile
    out_offs = tl.arange(0, half_tile)
    tl.store(Out + out_base + out_offs, packed,
             mask=(out_base + out_offs) < (total_elems // 2))


@triton.jit
def _nvfp4_dequantize_kernel(
    Packed, Scale, GlobalScale, Out,
    total_elems,
    BLOCK_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    half_tile: tl.constexpr = TILE_ELEMS // 2
    blocks_per_tile: tl.constexpr = TILE_ELEMS // BLOCK_SIZE
    pairs_per_block: tl.constexpr = BLOCK_SIZE // 2
    base_block = pid * blocks_per_tile
    gs = tl.load(GlobalScale).to(tl.float32)
    scale_offs = tl.arange(0, blocks_per_tile)
    scale_mask = (base_block + scale_offs) < (total_elems // BLOCK_SIZE)
    scale_vec = tl.load(Scale + base_block + scale_offs,
                        mask=scale_mask, other=1.0).to(tl.float32) * gs
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


@triton.jit
def _nvfp4_scale_convert_kernel(Amax, ScaleOut, GlobalAmaxInOut, n_blocks,
                                FP4_MAX_INV, FP8_MAX,
                                BLOCK: tl.constexpr):
    """Convert raw amax to fp8 block_scales and update GlobalAmaxInOut to global_scale."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_blocks
    amax = tl.load(Amax + offs, mask=mask, other=1.0)
    gamax = tl.maximum(tl.load(GlobalAmaxInOut), 1e-12)
    tl.store(ScaleOut + offs, (amax / gamax * FP8_MAX).to(tl.float8e4nv), mask=mask)
    if pid == 0:
        tl.store(GlobalAmaxInOut, gamax * FP4_MAX_INV / FP8_MAX)


def nvfp4_quantize(x: Tensor, block_size: int = 16,
                   global_amax: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    assert x.dim() == 2
    M, K = x.shape
    n = M * K
    assert n % block_size == 0 and block_size % 2 == 0
    x_flat = x.reshape(-1).contiguous()
    n_scales = n // block_size
    packed = torch.empty(n // 2, dtype=torch.uint8, device=x.device)
    amax = torch.empty(n_scales, dtype=torch.float32, device=x.device)
    global_amax_buf = torch.zeros(1, dtype=torch.float32, device=x.device)
    te, nw = _QUANT_CONFIGS.get(block_size, (2048, 1))
    te = min(te, n)
    grid = (n + te - 1) // te
    _nvfp4_quantize_kernel[(grid,)](
        x_flat, packed, amax, global_amax_buf, n,
        block_size, te, num_warps=nw,
    )
    # Override global amax with EMA-tracked value if provided
    if global_amax is not None:
        global_amax_buf.copy_(global_amax)
    block_scales = torch.empty(n_scales, dtype=torch.float8_e4m3fn, device=x.device)
    SC_BLOCK = 1024
    _nvfp4_scale_convert_kernel[((n_scales + SC_BLOCK - 1) // SC_BLOCK,)](
        amax, block_scales, global_amax_buf, n_scales,
        1.0 / FP4_E2M1_MAX, FP8_E4M3_MAX, SC_BLOCK, num_warps=4,
    )
    return packed, block_scales, global_amax_buf


def nvfp4_dequantize(packed: Tensor, block_scales: Tensor, global_scale: Tensor,
                     num_elements: int, block_size: int = 16) -> Tensor:
    assert num_elements % block_size == 0
    packed_flat = packed.reshape(-1)
    out_i32 = torch.empty(num_elements // 2, dtype=torch.int32, device=packed_flat.device)
    te, nw = _DEQUANT_CONFIGS.get(block_size, (2048, 2))
    te = min(te, num_elements)
    grid = (num_elements + te - 1) // te
    gs = global_scale if global_scale.numel() == 1 else global_scale.reshape(1)
    _nvfp4_dequantize_kernel[(grid,)](
        packed_flat, block_scales, gs, out_i32, num_elements,
        block_size, te, num_warps=nw,
    )
    return out_i32.view(torch.bfloat16)
