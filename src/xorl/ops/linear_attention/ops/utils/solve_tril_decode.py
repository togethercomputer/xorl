# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.

"""Decode-shaped 64x64 triangular solve (recompute-decode budget).

Bitwise-identical re-schedule of ``solve_tril`` (BT=64 only): the monolithic
``merge_16x16_to_64x64_inverse_kernel`` runs at 255 regs/thread -> 12.5%
occupancy, which is what makes it pathological at the bs x 64 recompute-decode
shape. Split into a diagonal forward-substitution kernel (4x parallelism, small
register footprint, fp32 scratch) and a merge kernel (the nine ieee-precision
dots + all stores, including the upper-triangle zero blocks, so the
``zeros_like`` fill disappears). Reduction orders are unchanged: the
substitution reproduces the pinned num_warps=2 ``tl.sum`` reduction
(``_sum_rows_16_fla_tree``) and the merge keeps the exact dot expression tree;
the fp32 diagonal inverses cross kernels through lossless fp32 scratch.
"""

import torch
import triton
import triton.language as tl

from xorl.ops.linear_attention.ops.utils.index import prepare_chunk_indices
from xorl.ops.linear_attention.utils import input_guard


# The forward-substitution reduction tree must match the pinned solve_tril
# kernels (num_warps=2, see SOLVE_TRIL_NUM_WARPS). The diag kernel spells that
# tree out explicitly (see _sum_rows_16_fla_tree), while the launch geometry
# remains pinned below.
DIAG_HEAD_GROUP = 16
DIAG_NUM_WARPS = 2
DIAG_NUM_STAGES = 1
MERGE_NUM_WARPS = 2
MERGE_NUM_STAGES = 1


@triton.jit
def _no_contract_f32(x):
    """Optimization barrier: forces the product feeding the reduction tree to be
    rounded as a plain mul, forbidding config-dependent mul+add FMA contraction
    (the pinned tl.sum's first add sits behind a warp shuffle, so the reference
    kernel never contracts there; measured: without this barrier some launch
    configs produce rare 1-ulp bf16 tails)."""
    return tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sum_rows_16_fla_tree(b_a, b_A):
    """Compute sum_r b_a[:, r, None] * b_A[:, r, :] over the 16 rows in the exact
    association order of the pinned num_warps=2 ``tl.sum`` in solve_tril.

    That reduce's PTX (per warp-half of 8 rows, per column) is
    ``v_i = fma(a_i, b_i, round(a_{i+4} * b_{i+4}))`` for i in 0..3 — the lane's
    own product is CONTRACTED into the first xor-4 butterfly add — followed by
    plain adds for the xor-2 / xor-1 levels and the cross-warp add:
    ``((v_0 + v_2) + (v_1 + v_3))_half0 + (same)_half1``. Spelled out with
    tl.fma + explicit adds so the tree no longer depends on this kernel's
    launch config; the mul feeding the fma is barriered against re-contraction.
    """
    G: tl.constexpr = b_A.shape[0]
    # rows r = h*8 + q*4 + i, axes [G, h, q, i, c]; split q: lo = rows i, hi = rows i+4
    tA = tl.permute(tl.reshape(b_A, [G, 2, 2, 4, 16]), [0, 1, 3, 4, 2])
    A_lo, A_hi = tl.split(tA)
    ta = tl.permute(tl.reshape(b_a, [G, 2, 2, 4]), [0, 1, 3, 2])
    a_lo, a_hi = tl.split(ta)
    v = tl.fma(a_lo[:, :, :, None], A_lo, _no_contract_f32(a_hi[:, :, :, None] * A_hi))
    # xor-2 level: pairs (i, i+2); axes [G, h, j_hi, j_lo, c] with i = j_hi*2 + j_lo
    t = tl.permute(tl.reshape(v, [G, 2, 2, 2, 16]), [0, 1, 3, 4, 2])
    lo, hi = tl.split(t)
    t = lo + hi
    # xor-1 level: pairs (0, 1)
    t = tl.permute(t, [0, 1, 3, 2])
    lo, hi = tl.split(t)
    t = lo + hi
    # cross-warp: halves (rows 0-7) + (rows 8-15)
    t = tl.permute(t, [0, 2, 1])
    lo, hi = tl.split(t)
    return lo + hi


@triton.jit(do_not_specialize=["T"])
def solve_tril_64x64_diag_inv_grouped_kernel(
    A,
    Di,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tq, i_bhg = tl.program_id(0), tl.program_id(1)
    i_t, i_q = i_tq // 4, i_tq % 4
    NHG: tl.constexpr = H // G
    i_b, i_h0 = i_bhg // NHG, (i_bhg % NHG) * G
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_g = tl.arange(0, G)
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    off = i_q * 16
    row = i_t * BT + off + o_i
    A += (bos * H + i_h0) * BT + off

    p_A = A + row[None, :, None] * H * BT + (o_g * BT)[:, None, None] + o_i[None, None, :]
    b_A = tl.load(p_A, mask=(row < T)[None, :, None], other=0.0)
    b_A = -tl.where(m_A[None, :, :], b_A, 0)

    for i in range(2, min(16, T - i_t * BT - off)):
        b_a = -tl.load(A + (i_t * BT + off + i) * H * BT + (o_g * BT)[:, None] + o_i[None, :])
        b_a = tl.where((o_i < i)[None, :], b_a, 0.0)
        b_a += _sum_rows_16_fla_tree(b_a, b_A)
        b_A = tl.where((o_i == i)[None, :, None], b_a[:, None, :], b_A)
    b_A += m_I[None, :, :]

    Di += (bos * H + i_h0) * 16
    p_Di = Di + row[None, :, None] * H * 16 + (o_g * 16)[:, None, None] + o_i[None, None, :]
    tl.store(p_Di, b_A, mask=(row < T)[None, :, None])


@triton.jit(do_not_specialize=["T"])
def solve_tril_64x64_merge_inv_kernel(
    A,
    Di,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT
    Di += (bos * H + i_h) * 16

    p_D_11 = tl.make_block_ptr(Di, (T, 16), (H * 16, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_D_22 = tl.make_block_ptr(Di, (T, 16), (H * 16, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_D_33 = tl.make_block_ptr(Di, (T, 16), (H * 16, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_D_44 = tl.make_block_ptr(Di, (T, 16), (H * 16, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_D_11, boundary_check=(0, 1))
    b_Ai_22 = tl.load(p_D_22, boundary_check=(0, 1))
    b_Ai_33 = tl.load(p_D_33, boundary_check=(0, 1))
    b_Ai_44 = tl.load(p_D_44, boundary_check=(0, 1))

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    # Upper-triangle zero blocks (replaces the wrapper-level zeros_like fill).
    b_z = tl.zeros([16, 16], dtype=tl.float32).to(p_Ai_11.dtype.element_ty)
    p_Ai_12 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 16), (16, 16), (1, 0))
    p_Ai_13 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 32), (16, 16), (1, 0))
    p_Ai_14 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 48), (16, 16), (1, 0))
    p_Ai_23 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 32), (16, 16), (1, 0))
    p_Ai_24 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 48), (16, 16), (1, 0))
    p_Ai_34 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 48), (16, 16), (1, 0))
    tl.store(p_Ai_12, b_z, boundary_check=(0, 1))
    tl.store(p_Ai_13, b_z, boundary_check=(0, 1))
    tl.store(p_Ai_14, b_z, boundary_check=(0, 1))
    tl.store(p_Ai_23, b_z, boundary_check=(0, 1))
    tl.store(p_Ai_24, b_z, boundary_check=(0, 1))
    tl.store(p_Ai_34, b_z, boundary_check=(0, 1))


@input_guard
def solve_tril_decode(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Compute the inverse of I + A for strictly lower-triangular 64x64 chunk A.

    Drop-in replacement for :func:`solve_tril` at BT=64, bitwise-identical to the
    pinned (num_warps=2) kernel; scheduled for small-T decode-recompute shapes.
    """
    assert A.shape[-1] == 64, "solve_tril_decode only supports BT=64"
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.empty_like(A, dtype=output_dtype)
    Di = torch.empty(B, T, H, 16, dtype=torch.float32, device=A.device)
    G = DIAG_HEAD_GROUP
    while H % G:
        G //= 2
    solve_tril_64x64_diag_inv_grouped_kernel[NT * 4, B * (H // G)](
        A=A,
        Di=Di,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        G=G,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=DIAG_NUM_WARPS,
        num_stages=DIAG_NUM_STAGES,
    )
    solve_tril_64x64_merge_inv_kernel[NT, B * H](
        A=A,
        Di=Di,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
        IS_VARLEN=cu_seqlens is not None,
        num_warps=MERGE_NUM_WARPS,
        num_stages=MERGE_NUM_STAGES,
    )
    return Ai
