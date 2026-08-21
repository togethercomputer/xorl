# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - attn_sink: gradient computation for learnable per-head scalar
#   - Single-head KV: kv shape [B, S_kv, D] (no kv_group, no D/D_tail split)
#   - Index shape: [B, S, topk] (no kv_group dim)
#   - Outputs: dQ [B, S, H, D], dKV [B, S_kv, D], dAttnSink [H]
#
# Two backward paths are available:
#   - `bwd` (default): atomic_addx4 dKV writes. Memory: O(S_kv * D) for dKV.
#   - `bwd_partial` + `segment_sum_indirect`: deterministic dKV via per-CTA
#     partials + non-atomic segment_sum reduction. 2.5-2.6x faster at S=32k
#     but requires O(B*S*NH*topk*D) bf16 scratch (~21 GB at production shape).
#     Enabled via env var XORL_DSV4_DETERMINISTIC_DKV=1.
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
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    dkv_shape = [B, S_kv, D]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), B, threads=threads) as (bx, by):
            for bn_i, d_i in T.Parallel(block_N, D):
                if bx * block_N + bn_i < S_kv:
                    dKV_out[by, bx * block_N + bn_i, d_i] = dKV[by, bx * block_N + bn_i, d_i]

    return postprocess_kernel


# NOTE: TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE produces NaN dq/dkv on
# tilelang 0.1.9 — the pass aliases acc_dkv_shared with buffers still live
# in the dq path, corrupting the dq accumulator across i_i iterations.
# Do not re-enable. The shared-memory budget at block_H=32 already fits
# without the merge (and is ~1.7x faster than the merge-on, block_H=64 path).
@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    block_size=64,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    # block_H=32 (was 64) halves the per-CTA Q/dO shared-mem footprint, lets
    # the kernel fit in the H100 228KB SMEM budget without aggressive merge,
    # and doubles the CTA grid for better SM occupancy.
    block_H = min(32, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mqa_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            # No dQ_shared: write acc_dq fragment -> global directly with
            # implicit fp32->bf16 cast (matches the fwd kernel's acc_o store).

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] >= 0 and Indices[by, s_i, i_i * BS + bi_i] < S_kv

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        by,
                        T.if_then_else(
                            Indices[by, s_i, i_i * BS + bi_i] >= 0 and Indices[by, s_i, i_i * BS + bi_i] < S_kv,
                            Indices[by, s_i, i_i * BS + bi_i],
                            0,
                        ),
                        d_i,
                    ]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                # P = exp2(scores * sm_scale_log2e - LSE)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                # dP = P * (dO @ KV^T - Delta)
                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                # dQ += dP @ KV
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                # dKV += dP^T @ Q + P^T @ dO
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                # Atomic store dKV with split to reduce register pressure
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        if mask[bi_i + s * (BS // split_store)]:
                            T.atomic_addx4(
                                dKV[
                                    by,
                                    Indices[by, s_i, i_i * BS + bi_i + s * (BS // split_store)],
                                    d_i * 4,
                                ],
                                acc_dkv_shared[bi_i, d_i * 4],
                            )

            # Store dQ (fragment -> global, implicit fp32->bf16 cast)
            T.copy(acc_dq, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])

            # dAttnSink[h] = -sum_{b,s}( Delta[b,s,h] * p_sink[b,s,h] )
            # where p_sink = exp(attn_sink[h]) / Z = exp2(attn_sink[h]*log2e - LSE)
            # attn_sink is a pre-scaled logit, so only convert to log2 base (no sm_scale)
            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i]
                    * T.exp2(AttnSink[bz * block_H + h_i] * 1.44269504 - Lse[by, s_i, bz * block_H + h_i]),
                )

    return sparse_mqa_bwd_kernel


# ---------------------------------------------------------------------------
# Deterministic dKV path: writes per-CTA partials (no atomics), then
# a non-atomic segment_sum reduces them. 2.5-2.6x faster than the atomic
# path at S=32k topk=640 (212ms -> 80ms). Memory cost: B*S*NH*topk*D bf16
# scratch buffer (~21 GB at production shape).
# ---------------------------------------------------------------------------
@tilelang.jit(
    out_idx=[-3],
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
    topk,
    sm_scale=None,
    block_size=64,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Same math as `bwd` but writes per-CTA partial dKV deterministically
    (no atomic_addx4). Output PartialDKV shape: [B, S, NH, topk, D] bf16.

    NH = padded_H // block_H so each head-group CTA gets its own slot.
    """
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(32, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size

    partial_shape = [B, S, NH, topk, D]

    @T.prim_func
    def sparse_mqa_bwd_partial_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        PartialDKV: T.Tensor(partial_shape, dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            # bf16 staging buffer for the deterministic partial write.
            # Tried removing it (write fragment->global directly like the fwd does
            # for acc_o), but staging is ~3% faster (84 vs 87 ms at S=32k topk=640)
            # because the shared-mem path coalesces the bulk 64 KB write better
            # than per-thread fragment scatter.
            acc_dkv_shared = T.alloc_shared([BS, D], dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)
            T.clear(acc_dq)

            for i_i in T.Pipelined(topk // BS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] >= 0 and Indices[by, s_i, i_i * BS + bi_i] < S_kv

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        by,
                        T.if_then_else(
                            Indices[by, s_i, i_i * BS + bi_i] >= 0 and Indices[by, s_i, i_i * BS + bi_i] < S_kv,
                            Indices[by, s_i, i_i * BS + bi_i],
                            0,
                        ),
                        d_i,
                    ]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

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

                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                # Deterministic write: fp32 acc_dkv -> bf16 staging shared -> bf16 global.
                # Each CTA's [bz, i_i*BS:(i_i+1)*BS, :D] slice is uniquely owned.
                T.copy(acc_dkv, acc_dkv_shared)
                T.copy(
                    acc_dkv_shared,
                    PartialDKV[by, s_i, bz, i_i * BS : (i_i + 1) * BS, :D],
                )

            T.copy(acc_dq, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])

            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i]
                    * T.exp2(AttnSink[bz * block_H + h_i] * 1.44269504 - Lse[by, s_i, bz * block_H + h_i]),
                )

    return sparse_mqa_bwd_partial_kernel


@tilelang.jit  # no out_idx: dKV is mutable in/out
def segment_sum_indirect_accumulate(S_kv, D, threads=128, dtype=T.bfloat16, accum_dtype=T.float32):
    """Non-atomic segment_sum that ACCUMULATES into an existing dKV buffer.

    Reads previous dKV[bx], adds the new segment sum, writes back. No atomics
    needed because each CTA owns a unique bx. Enables chunked execution: call
    once per chunk with the same dKV buffer; first call accumulates onto
    zeros (caller initialises).

    Inputs:
        Partial:  [N, D] bf16, unsorted (a chunk's partials, flattened)
        SortPerm: [N] int64, sort permutation by KV index (within the chunk)
        Offsets:  [S_kv+1] int32, segment boundaries
    In/out:
        dKV: [S_kv, D] fp32 — accumulator
    """
    N = T.dynamic("N")

    @T.prim_func
    def kernel(
        Partial: T.Tensor([N, D], dtype),
        SortPerm: T.Tensor([N], T.int64),
        Offsets: T.Tensor([S_kv + 1], T.int32),
        dKV: T.Tensor([S_kv, D], accum_dtype),
    ):
        with T.Kernel(S_kv, threads=threads) as bx:
            acc = T.alloc_fragment([D], accum_dtype)
            for d in T.Parallel(D):
                acc[d] = dKV[bx, d]

            start = Offsets[bx]
            end = Offsets[bx + 1]
            for i in T.serial(start, end):
                for d in T.Parallel(D):
                    acc[d] = acc[d] + T.cast(Partial[SortPerm[i], d], accum_dtype)
            for d in T.Parallel(D):
                dKV[bx, d] = acc[d]

    return kernel


def _build_invert_index_chunk(topk_idxs_chunk, S_kv, NH):
    """Build sort_perm + offsets for one chunk of a single batch.

    topk_idxs_chunk: [S_chunk, topk] int32
    Returns: (sort_perm [S_chunk*NH*topk] int64, offsets [S_kv+1] int32)
    """
    flat_idx = topk_idxs_chunk.unsqueeze(1).expand(-1, NH, -1).reshape(-1).long()
    flat_idx_clamped = torch.where(flat_idx >= 0, flat_idx, torch.full_like(flat_idx, S_kv))
    sort_perm = flat_idx_clamped.argsort()
    counts = torch.bincount(flat_idx_clamped, minlength=S_kv + 1)[:S_kv]
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=flat_idx.device), torch.cumsum(counts, 0)]).int()
    return sort_perm, offsets


def _choose_chunk_size(B, S, NH, topk, D, max_partial_gb=24.0):
    """Pick S_chunk so the partial buffer fits in `max_partial_gb` GB.

    Partial bytes = B * S_chunk * NH * topk * D * 2 (bf16). We assume B==1
    in practice (it's typical for training). For B > 1 we shrink further.
    Returns S_chunk that divides S (or the largest divisor below the cap).
    """
    max_bytes = int(max_partial_gb * 1024**3)
    bytes_per_query = B * NH * topk * D * 2
    s_chunk_cap = max(1, max_bytes // bytes_per_query)
    if s_chunk_cap >= S:
        return S
    # Find largest divisor of S below s_chunk_cap.
    for c in (s_chunk_cap, s_chunk_cap // 2 * 2):
        if c > 0 and S % c == 0:
            return c
    # Fall back to nearest power-of-2 divisor.
    c = 1
    while c * 2 <= s_chunk_cap and S % (c * 2) == 0:
        c *= 2
    return c


def sparse_mqa_bwd_interface(q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=None):
    """Backward interface for V4 sparse MQA attention.

    Two paths:
      - Default (atomic): uses `bwd` + atomic_addx4 dKV writes.
      - Deterministic (env XORL_DSV4_DETERMINISTIC_DKV=1): uses `bwd_partial` +
        `segment_sum_indirect`. 2.5-2.6x faster at S=32k but allocates a
        B*S*NH*topk*D bf16 scratch buffer (~21 GB at production shape).

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        attn_sink: [H] fp32
        o:         [B, S, H, D] bf16 (forward output)
        do:        [B, S, H, D] bf16 (grad of output)
        topk_idxs: [B, S, topk] int32
        lse:       [B, S, H] fp32 (log-sum-exp from forward)
        sm_scale:  float or None

    Returns:
        dq:         [B, S, H, D] bf16
        dkv:        [B, S_kv, D] bf16
        d_attn_sink: [H] fp32
    """
    assert q.is_contiguous() and kv.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]

    # Pad topk to next multiple of block_size (kernel requires divisibility)
    block_size = 64
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = torch.full((B, S, padded_topk - topk), -1, device=topk_idxs.device, dtype=topk_idxs.dtype)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk

    preprocess_kernel = preprocess(B, S, H, D)
    delta = preprocess_kernel(o, do)

    use_deterministic = os.environ.get("XORL_DSV4_DETERMINISTIC_DKV", "0") == "1"

    if use_deterministic:
        # Stage 0: figure out NH and chunk size for the partial buffer.
        import tilelang as _tl

        _padded_H = max(_tl.math.next_power_of_2(H), 16)
        NH = _padded_H // min(32, _padded_H)
        max_partial_gb = float(os.environ.get("XORL_DSV4_DETERMINISTIC_DKV_MAX_GB", "24"))
        s_chunk = _choose_chunk_size(B, S, NH, topk, D, max_partial_gb)

        # Outputs
        dq = torch.empty_like(q)
        dkv_fp32 = torch.zeros(B, S_kv, D, device=kv.device, dtype=torch.float32)
        d_attn_sink = torch.zeros_like(attn_sink)

        # Stage 2 kernel (compiled once, dKV accumulator)
        seg_kernel = segment_sum_indirect_accumulate(S_kv, D)

        # Pre-allocate one partial buffer reused across chunks
        partial_dkv = torch.empty(B, s_chunk, NH, topk, D, device=kv.device, dtype=torch.bfloat16)

        # Chunked stage-1 + stage-2.
        # Per-chunk Stage-1 kernel is compiled once (tilelang caches by shape).
        bwd_partial_kernel = bwd_partial(B, s_chunk, S_kv, H, D, topk, sm_scale)

        for chunk_start in range(0, S, s_chunk):
            chunk_end = chunk_start + s_chunk  # guaranteed to fit since s_chunk divides S
            q_chunk = q[:, chunk_start:chunk_end].contiguous()
            do_chunk = do[:, chunk_start:chunk_end].contiguous()
            idx_chunk = topk_idxs[:, chunk_start:chunk_end].contiguous()
            lse_chunk = lse[:, chunk_start:chunk_end].contiguous()
            delta_chunk = delta[:, chunk_start:chunk_end].contiguous()

            # Stage 1 (chunk): writes partial_dkv and dq[chunk]
            dq_chunk = bwd_partial_kernel(
                q_chunk,
                kv,
                do_chunk,
                attn_sink,
                idx_chunk,
                lse_chunk,
                delta_chunk,
                partial_dkv,
                d_attn_sink,
            )
            dq[:, chunk_start:chunk_end].copy_(dq_chunk)

            # Stage 2 (chunk, per batch): segment_sum accumulates into dkv_fp32
            for b in range(B):
                sort_perm, offsets = _build_invert_index_chunk(idx_chunk[b], S_kv, NH)
                seg_kernel(partial_dkv[b].reshape(-1, D), sort_perm, offsets, dkv_fp32[b])

        # Stage 3: fp32 -> bf16
        postprocess_kernel = postprocess(B, S_kv, D)
        dkv = postprocess_kernel(dkv_fp32)
    else:
        bwd_kernel = bwd(B, S, S_kv, H, D, topk, sm_scale)
        postprocess_kernel = postprocess(B, S_kv, D)
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        d_attn_sink = torch.zeros_like(attn_sink)
        dq = bwd_kernel(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
        dkv = postprocess_kernel(dkv)

    return dq, dkv, d_attn_sink
