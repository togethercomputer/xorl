"""INT4 quantization/dequantization via Triton kernels."""
from typing import Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _int4_quantize_kernel(
    X, Out, Scale,
    total_elems,
    GROUP_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * TILE_ELEMS
    half_tile: tl.constexpr = TILE_ELEMS // 2
    groups_per_tile: tl.constexpr = TILE_ELEMS // GROUP_SIZE
    base_group = pid * groups_per_tile
    offs = tl.arange(0, TILE_ELEMS)
    mask = (base + offs) < total_elems
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    x_2d = tl.reshape(x, (groups_per_tile, GROUP_SIZE))
    amax = tl.max(tl.abs(x_2d), axis=1)
    scale = tl.maximum(amax, 1e-12) * (2.0 / 15.0)
    scale_offs = tl.arange(0, groups_per_tile)
    tl.store(Scale + base_group + scale_offs, scale,
             mask=(base_group + scale_offs) < (total_elems // GROUP_SIZE))
    inv_scale_2d = 1.0 / tl.expand_dims(scale, axis=1)
    q_2d = tl.maximum(0, tl.minimum(15, (x_2d * inv_scale_2d + 8.5).to(tl.int32)))
    q = tl.reshape(q_2d, (TILE_ELEMS,))
    is_odd = (offs & 1).to(tl.int32)
    shifted = (q & 0xF) << (is_odd * 4)
    packed = tl.sum(tl.reshape(shifted, (half_tile, 2)), axis=1).to(tl.uint8)
    out_base = pid * half_tile
    out_offs = tl.arange(0, half_tile)
    tl.store(Out + out_base + out_offs, packed,
             mask=(out_base + out_offs) < (total_elems // 2))


@triton.jit
def _int4_dequantize_kernel(
    Packed, Scale, Out,
    total_elems,
    GROUP_SIZE: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    half_tile: tl.constexpr = TILE_ELEMS // 2
    groups_per_tile: tl.constexpr = TILE_ELEMS // GROUP_SIZE
    half_gs: tl.constexpr = GROUP_SIZE // 2
    base_group = pid * groups_per_tile
    pack_base = pid * half_tile
    pack_offs = tl.arange(0, half_tile)
    pack_mask = (pack_base + pack_offs) < (total_elems // 2)
    packed = tl.load(Packed + pack_base + pack_offs, mask=pack_mask, other=0).to(tl.int32)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    scale_offs = tl.arange(0, groups_per_tile)
    scale = tl.load(Scale + base_group + scale_offs,
                    mask=(base_group + scale_offs) < (total_elems // GROUP_SIZE),
                    other=1.0).to(tl.float32)
    scale_2d = tl.expand_dims(scale, axis=1)
    lo_2d = tl.reshape(lo.to(tl.float32), (groups_per_tile, half_gs))
    hi_2d = tl.reshape(hi.to(tl.float32), (groups_per_tile, half_gs))
    val_lo = tl.reshape((lo_2d - 8.0) * scale_2d, (half_tile,))
    val_hi = tl.reshape((hi_2d - 8.0) * scale_2d, (half_tile,))
    lo_i16 = val_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    hi_i16 = val_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    word = (lo_i16 & 0xFFFF) | (hi_i16 << 16)
    tl.store(Out + pack_base + pack_offs, word, mask=pack_mask)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _get_config(group_size, total_elems, is_dequant=False):
    if group_size >= 4096:
        te = _next_pow2(group_size)
        nw = 8
    elif is_dequant:
        te, nw = 2048, 1
    else:
        te, nw = 2048, 1
    while te % group_size != 0:
        te *= 2
    te = min(te, _next_pow2(total_elems))
    return te, nw


def int4_quantize(x: Tensor, group_size: int = -1) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 2
    M, K = x.shape
    gs = K if group_size == -1 else group_size
    assert K % gs == 0 and gs % 2 == 0
    n = M * K
    packed = torch.empty(n // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(n // gs, dtype=torch.float32, device=x.device)
    x_flat = x.reshape(-1).contiguous()
    te, nw = _get_config(gs, n)
    grid = (n + te - 1) // te
    _int4_quantize_kernel[(grid,)](x_flat, packed, scales, n, gs, te, num_warps=nw)
    num_groups = K // gs
    return packed.reshape(M, K // 2), scales.reshape(M, num_groups)


def int4_dequantize(packed: Tensor, scales: Tensor, M: int, K: int,
                    group_size: int = -1, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    gs = K if group_size == -1 else group_size
    n = M * K
    packed_flat = packed.reshape(-1)
    scales_flat = scales.float().reshape(-1) if scales.dtype != torch.float32 else scales.reshape(-1)
    out_i32 = torch.empty(n // 2, dtype=torch.int32, device=packed.device)
    te, nw = _get_config(gs, n, is_dequant=True)
    grid = (n + te - 1) // te
    _int4_dequantize_kernel[(grid,)](packed_flat, scales_flat, out_i32, n, gs, te, num_warps=nw)
    out = out_i32.view(torch.bfloat16)
    if out_dtype != torch.bfloat16:
        out = out.to(out_dtype)
    return out.reshape(M, K)
