# ruff: noqa
# Vendored from https://github.com/tile-ai/tilelang/blob/4ff81c7d40803d269569e157e847623e84553f78/examples/deepseek_v32/sparse_mla_bwd.py
# with local fixes for GLM-5 sentinel indices and tilelang 0.1.9 shared-memory aliasing.
import os

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(out_idx=[-1])
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    B,
    S_kv,
    D,
    D_tail,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, B, threads=threads) as (bx, by, bz):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE produces NaN dq/dkv on tilelang 0.1.9
        # (aliases acc_dkv_shared with buffers still in use by the dq gemm path).
    },
)
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
    compute_dq=True,
    compute_dkv=True,
    split_store=2,
    block_H_cap=64,
):
    assert is_causal == True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32
    assert compute_dq or compute_dkv
    assert block_size % split_store == 0, f"block_size={block_size} must be divisible by split_store={split_store}"

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    assert indices_dtype == T.int32
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(block_H_cap, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, B, kv_group * NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            if compute_dq:
                dQ_shared = T.alloc_shared([block_H, D], dtype)
                dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            if compute_dq:
                acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
                acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
            if compute_dkv:
                acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
                acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
                acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
                acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)
            safe_indices = T.alloc_fragment([BS], indices_dtype)

            # max_kv_i = s_i

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            if compute_dq:
                T.clear(acc_dq)
                T.clear(acc_dq_tail)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    # Changed here for thd
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] != -1
                    safe_indices[bi_i] = T.if_then_else(
                        mask[bi_i],
                        Indices[by, s_i, bz // NH, i_i * BS + bi_i],
                        0,
                    )

                # Compute attention scores
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, safe_indices[bi_i], bz // NH, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[by, safe_indices[bi_i], bz // NH, D + d_i]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)
                if compute_dq:
                    T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                if compute_dkv:
                    T.gemm(
                        dP_shared_cast,
                        Q_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                    T.clear(acc_dkv_tail)
                    T.gemm(
                        dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol
                    )

                    for s in range(split_store):
                        for bi_i, d_i in T.Parallel(BS, D):
                            if bi_i < BS // split_store:
                                acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                        for bi_i, d_i in T.Parallel(BS, D_tail):
                            if bi_i < BS // split_store:
                                acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                        for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                            T.atomic_addx4(
                                dKV[
                                    by,
                                    safe_indices[bi_i + s * (BS // split_store)],
                                    bz // NH,
                                    d_i * 4,
                                ],
                                acc_dkv_shared[bi_i, d_i * 4],
                            )

                        # Atomically update dKV, dKV_tail tensors
                        for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                            T.atomic_addx4(
                                dKV[
                                    by,
                                    safe_indices[bi_i + s * (BS // split_store)],
                                    bz // NH,
                                    D + d_i * 4,
                                ],
                                acc_dkv_tail_shared[bi_i, d_i * 4],
                            )

            if compute_dq:
                # Store the accumulated dQ
                T.copy(acc_dq, dQ_shared)
                T.copy(acc_dq_tail, dQ_tail_shared)

                T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
                T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


# ---------------------------------------------------------------------------
# Deterministic dKV path (ported from DSv4: writes per-CTA partials, no
# atomics; a non-atomic segment_sum reduces them). 2.5–2.6x faster than the
# atomic path at S=32k topk=640 H=64 on DSv4 (213 ms → 84 ms). For GLM-5
# the partial buffer is [B, S, NH, topk, D+D_tail] bf16 (~9.7 GB at the
# production shape B=1, S=2048, NH=2, topk=2048, D+D_tail=576).
# ---------------------------------------------------------------------------
@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd_partial(
    B,
    S,
    S_kv,
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    block_size=64,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
    block_H_cap=32,
):
    """Same math as `bwd` but writes per-CTA partial dKV deterministically
    (no atomic_addx4). Output `PartialDKV` shape: [B, S, NH, topk, D+D_tail]
    bf16. Each CTA owns a unique `[bz, i_i*BS:(i_i+1)*BS, :D+D_tail]` slice.

    The partial's [..., :D] holds the combined `dP^T @ Q[..., :D] + P^T @ dO`
    contribution; [..., D:] holds the `dP^T @ Q[..., D:]` contribution. The
    downstream `segment_sum_indirect_accumulate` reduces by kv_pos.
    """
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert kv_group == 1, "deterministic path currently assumes kv_group=1 (GLM-5)"

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(block_H_cap, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size

    partial_shape = [B, S, NH, topk, D + D_tail]

    @T.prim_func
    def sparse_mla_bwd_partial_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        PartialDKV: T.Tensor(partial_shape, dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")
            safe_indices = T.alloc_fragment([BS], indices_dtype)

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            # bf16 staging buffers for the deterministic partial writes.
            # DSv4 measured ~3% speedup vs direct fragment-to-global because
            # the shared-mem path coalesces the bulk write better.
            acc_dkv_shared = T.alloc_shared([BS, D], dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS, D_tail], dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)
            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            for i_i in T.Pipelined(topk // BS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, 0, i_i * BS + bi_i] != -1
                    safe_indices[bi_i] = T.if_then_else(mask[bi_i], Indices[by, s_i, 0, i_i * BS + bi_i], 0)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, safe_indices[bi_i], 0, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[by, safe_indices[bi_i], 0, D + d_i]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )
                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )
                T.copy(acc_dp, dP_shared_cast)

                # dQ += dP @ KV, dQ_tail += dP @ KV_tail
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                # acc_dkv[:D]    = dP^T @ Q[:D] + P^T @ dO
                # acc_dkv_tail[D:] = dP^T @ Q[D:]
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                # Deterministic write: fp32 fragments -> bf16 staging shared -> bf16 global.
                T.copy(acc_dkv, acc_dkv_shared)
                T.copy(
                    acc_dkv_shared,
                    PartialDKV[by, s_i, bz, i_i * BS : (i_i + 1) * BS, :D],
                )
                T.copy(acc_dkv_tail, acc_dkv_tail_shared)
                T.copy(
                    acc_dkv_tail_shared,
                    PartialDKV[by, s_i, bz, i_i * BS : (i_i + 1) * BS, D : D + D_tail],
                )

            # Store dQ and dQ_tail (fragment -> global, implicit fp32->bf16 cast)
            T.copy(acc_dq, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            T.copy(acc_dq_tail, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_partial_kernel


@tilelang.jit  # no out_idx: dKV is mutable in/out
def segment_sum_indirect_accumulate(S_kv, D_total, threads=None, dtype=T.bfloat16, accum_dtype=T.float32):
    """Non-atomic segment_sum that ACCUMULATES into an existing dKV buffer.

    Reads previous dKV[bx, :], adds the new segment sum, writes back. No
    atomics needed because each CTA owns a unique kv_pos. Enables chunked
    execution: call once per chunk with the same dKV; first call accumulates
    onto zeros (caller initialises).

    Inputs:
        Partial:  [N, D_total] bf16 — flat (a chunk's partials reshaped)
        SortPerm: [N] int64 — sort permutation by kv index (within the chunk)
        Offsets:  [S_kv+1] int32 — CSR-style segment boundaries
    In/out:
        dKV: [S_kv, D_total] fp32 — accumulator
    """
    if threads is None:
        # `T.Parallel(D_total)` requires `threads | D_total`. The GLM-5
        # partial is D + D_tail wide (576 = 512 + 64 for the production
        # MLA shape) so 128 doesn't fit. Pick the largest valid multiple
        # of 32 below 256 that divides D_total.
        threads = 128
        for cand in (256, 192, 128, 96, 64, 32):
            if D_total % cand == 0:
                threads = cand
                break
    N = T.dynamic("N")

    @T.prim_func
    def kernel(
        Partial: T.Tensor([N, D_total], dtype),
        SortPerm: T.Tensor([N], T.int64),
        Offsets: T.Tensor([S_kv + 1], T.int32),
        dKV: T.Tensor([S_kv, D_total], accum_dtype),
    ):
        with T.Kernel(S_kv, threads=threads) as bx:
            acc = T.alloc_fragment([D_total], accum_dtype)
            for d in T.Parallel(D_total):
                acc[d] = dKV[bx, d]

            start = Offsets[bx]
            end = Offsets[bx + 1]
            for i in T.serial(start, end):
                for d in T.Parallel(D_total):
                    acc[d] = acc[d] + T.cast(Partial[SortPerm[i], d], accum_dtype)
            for d in T.Parallel(D_total):
                dKV[bx, d] = acc[d]

    return kernel


def _build_invert_index_chunk_glm5(topk_idxs_chunk, S_kv, NH):
    """Build sort_perm + offsets for one chunk of a single batch.

    `topk_idxs_chunk` shape: [S_chunk, kv_group, topk] int32 with kv_group=1.
    Returns:
        sort_perm [S_chunk * NH * topk] int64
        offsets   [S_kv + 1] int32 (CSR segment boundaries; sentinel slots
                  map to S_kv and are skipped via the offsets[:S_kv] slice).
    """
    # Squeeze kv_group=1 and replicate across NH so each (s, bz, k) slot of
    # the partial buffer aligns with the same kv index.
    flat_idx = topk_idxs_chunk.squeeze(1).unsqueeze(1).expand(-1, NH, -1).reshape(-1).long()
    # Route sentinel (-1) entries to a phantom slot at S_kv so they're
    # dropped from per-kv_pos segments.
    flat_idx_clamped = torch.where(flat_idx >= 0, flat_idx, torch.full_like(flat_idx, S_kv))
    sort_perm = flat_idx_clamped.argsort()
    counts = torch.bincount(flat_idx_clamped, minlength=S_kv + 1)[:S_kv]
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=flat_idx.device), torch.cumsum(counts, 0)]).int()
    return sort_perm, offsets


def _choose_chunk_size(B, S, NH, topk, D_total, max_partial_gb=24.0):
    """Pick S_chunk so the partial buffer fits in `max_partial_gb` GB.

    Partial bytes = B * S_chunk * NH * topk * D_total * 2 (bf16).
    Returns S_chunk that divides S (or the largest divisor below the cap).
    """
    max_bytes = int(max_partial_gb * 1024**3)
    bytes_per_query = B * NH * topk * D_total * 2
    s_chunk_cap = max(1, max_bytes // bytes_per_query)
    if s_chunk_cap >= S:
        return S
    for c in (s_chunk_cap, s_chunk_cap // 2 * 2):
        if c > 0 and S % c == 0:
            return c
    c = 1
    while c * 2 <= s_chunk_cap and S % (c * 2) == 0:
        c *= 2
    return c


def sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True, return_kernel=False, delta=None):
    q = q.unsqueeze(0)
    kv = kv.unsqueeze(0)
    o = o.unsqueeze(0)
    do = do.unsqueeze(0)
    indices = indices.unsqueeze(0)
    lse = lse.unsqueeze(0)

    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert kv.shape[0] == B
    # dim should be assigned
    D = 512

    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    assert indices.shape == (B, S, kv_group, topk)
    assert lse.shape == (B, S, H)

    preprocess_kernel = preprocess(B, S, H, D)
    postprocess_kernel = postprocess(B, S_kv, D, D_tail, kv_group)
    # Combined kernel at the DSv4-tuned layout (block_H=32, block_size=64,
    # threads=256) is 2.05x faster than the original split path (29.0 vs
    # 60.0 ms at S=2048, topk=2048, H=64, D=512+64 on H100). The earlier
    # iter-1 default (block_H=64, block_size=32, threads=512) was 38.6 ms;
    # DSv4 found that halving block_H + doubling block_size lets the Q/dO
    # footprint shrink enough to fit a larger inner KV gemm, doubles the
    # CTA grid for better SM occupancy, and avoids smem aliasing. dkv_max
    # vs the iter-1 reference is 0.0078 — well within BF16 atomic-add noise.
    # Set XORL_GLM5_SPLIT_SPARSE_MLA_BWD=1 to restore the old split path.
    # Set XORL_GLM5_DETERMINISTIC_DKV=1 to use the atomic-free dKV path
    # (per-CTA partials + segment_sum; further 2.5x bwd speedup at a memory
    # cost of ~10 GB scratch per call at production shape).
    split_bwd = os.environ.get("XORL_GLM5_SPLIT_SPARSE_MLA_BWD", "0") != "0"
    use_deterministic = os.environ.get("XORL_GLM5_DETERMINISTIC_DKV", "0") != "0"

    if delta is None:
        delta = preprocess_kernel(o, do)

    if use_deterministic and not split_bwd:
        # Deterministic dKV path: bwd_partial writes per-CTA partials into a
        # [B, S, NH, topk, D+D_tail] bf16 buffer (no atomics), then
        # segment_sum_indirect_accumulate reduces them. The kernel is the
        # same math as `bwd` so dq is identical; dkv differs only by
        # reduction-order rounding (deterministic sum vs atomic-add).
        assert kv_group == 1, "deterministic dKV path requires kv_group=1"
        D_total = D + D_tail
        padded_H = max(tilelang.math.next_power_of_2(H // kv_group), 16)
        block_H = min(32, padded_H)
        NH = padded_H // block_H
        max_partial_gb = float(os.environ.get("XORL_GLM5_DETERMINISTIC_DKV_MAX_GB", "24"))
        s_chunk = _choose_chunk_size(B, S, NH, topk, D_total, max_partial_gb)

        dkv_fp32 = torch.zeros(B, S_kv, kv_group, D_total, device=kv.device, dtype=torch.float32)
        dq = torch.empty_like(q)

        seg_kernel = segment_sum_indirect_accumulate(S_kv, D_total)
        partial_dkv = torch.empty(B, s_chunk, NH, topk, D_total, device=kv.device, dtype=torch.bfloat16)
        bwd_partial_kernel = bwd_partial(
            B,
            s_chunk,
            S_kv,
            H,
            D,
            D_tail,
            topk,
            kv_group,
            sm_scale,
            block_size=64,
            threads=256,
            block_H_cap=32,
        )

        for chunk_start in range(0, S, s_chunk):
            chunk_end = chunk_start + s_chunk  # divides S by construction
            q_chunk = q[:, chunk_start:chunk_end].contiguous()
            do_chunk = do[:, chunk_start:chunk_end].contiguous()
            idx_chunk = indices[:, chunk_start:chunk_end].contiguous()
            lse_chunk = lse[:, chunk_start:chunk_end].contiguous()
            delta_chunk = delta[:, chunk_start:chunk_end].contiguous()

            dq_chunk = bwd_partial_kernel(
                q_chunk,
                kv,
                do_chunk,
                idx_chunk,
                lse_chunk,
                delta_chunk,
                partial_dkv,
            )
            dq[:, chunk_start:chunk_end].copy_(dq_chunk)

            for b in range(B):
                sort_perm, offsets = _build_invert_index_chunk_glm5(idx_chunk[b], S_kv, NH)
                seg_kernel(
                    partial_dkv[b].reshape(-1, D_total),
                    sort_perm,
                    offsets,
                    dkv_fp32[b, :, 0],  # kv_group=1 squeeze
                )

        dkv = postprocess_kernel(dkv_fp32)
    elif split_bwd:
        bwd_dq_kernel = bwd(
            B,
            S,
            S_kv,
            H,
            D,
            D_tail,
            topk,
            kv_group,
            sm_scale,
            is_casual,
            threads=256,
            compute_dq=True,
            compute_dkv=False,
        )
        bwd_dkv_kernel = bwd(
            B,
            S,
            S_kv,
            H,
            D,
            D_tail,
            topk,
            kv_group,
            sm_scale,
            is_casual,
            threads=512,
            compute_dq=False,
            compute_dkv=True,
        )
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        dq = bwd_dq_kernel(q, kv, do, indices, lse, delta, dkv)
        bwd_dkv_kernel(q, kv, do, indices, lse, delta, dkv)
        dkv = postprocess_kernel(dkv)
    else:
        bwd_kernel = bwd(
            B,
            S,
            S_kv,
            H,
            D,
            D_tail,
            topk,
            kv_group,
            sm_scale,
            is_casual,
            block_size=64,
            threads=256,
            block_H_cap=32,
        )
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        dq = bwd_kernel(q, kv, do, indices, lse, delta, dkv)
        dkv = postprocess_kernel(dkv)

    dq = dq.squeeze(0)
    dkv = dkv.squeeze(0)

    return dq, dkv
