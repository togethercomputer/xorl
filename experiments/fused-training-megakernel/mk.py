"""Host-side program builder + loader for the fused training megakernel.

A Program is a list of waves; each wave is a list of instructions; each instruction is
(op, ntiles, args). Buffers (torch CUDA tensors) are registered into a pointer table and
referenced by index. `run()` launches the single cooperative kernel over the stream.
"""

import os
import struct

import torch
from torch.utils.cpp_extension import load


_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirrors `enum Op` in megakernel.cu — keep in sync.
OP_NOP = 0
OP_FILL_F32 = 1
OP_AXPY_F32 = 2
OP_GEMM = 3
OP_RMSNORM_FWD = 4
OP_RMSNORM_BWD = 5
OP_SWIGLU_FWD = 6
OP_SWIGLU_BWD = 7
OP_QKNORM_ROPE_FWD = 8
OP_QKNORM_ROPE_BWD = 9
OP_EMBED_FWD = 10
OP_EMBED_BWD = 11
OP_CE_FWD = 12
OP_CE_BWD = 13
OP_ATTN_FWD = 14
OP_ATTN_DPRE = 15
OP_ATTN_DKV = 16
OP_ATTN_DQ = 17
OP_CVT_F32BF16 = 18
OP_ATTN_FWD_SPLIT = 19
OP_ATTN_COMBINE = 20
OP_ATTN_FWD_WG = 21  # wgmma attention (D=64, S%128==0): 128-row tiles, qt-outer
OP_ATTN_DKV_WG = 22  # args as OP_ATTN_DKV + trailing q-chunk count C
OP_ATTN_DQ_WG = 23  # args as OP_ATTN_DQ (incl. kv-chunk count C)
OP_RMSNORM_BWD_DX = 24  # dx-only half of env-gated split RMSNorm backward
OP_RMSNORM_BWD_DW = 25  # dw-only cold sink half of env-gated split RMSNorm backward
OP_RMSNORM_BWD_DX_R4 = 26  # dx-only four-row fold for H256 long-S shapes
OP_INV_VALID = 27  # one-tile valid-label count, writes reciprocal for CE
OP_RMSNORM_BWD_DX_FMA = 28  # H256/S128 RMSNorm dx arithmetic route
OP_SWIGLU_BWD_2W = 29  # opt-in two-warps-per-row SwiGLU backward route
OP_QKV_V_BWD = 30  # V-head fp32 workspace -> bf16 raw-grad pass-through
OP_RMSNORM_BWD_DX_H256 = 31  # H==256 fixed-width dx-only route (env probe)

GEMM_BM, GEMM_BN = 64, 128  # keep in sync with ops.cuh
FILL_CHUNK = 16384  # elements per fill/cvt work item (MK_CHUNK in ops.cuh)


def chunk_tiles(n):
    return (n + FILL_CHUNK - 1) // FILL_CHUNK


def gemm_tiles(M, N):
    return ((M + GEMM_BM - 1) // GEMM_BM) * ((N + GEMM_BN - 1) // GEMM_BN)


def wgmma_ok(M, N, K, flags):
    """Gemms with wgmma-friendly shapes route to the 128x64 warpgroup path.

    History: with the INTER (no-swizzle) smem layout, NN/TN routing measured SLOWER
    in-model twice (round 3, P6) and only NT routed. The SW128 layout (P4b) doubled
    the wgmma path's throughput (probe: pipe_probe.py) and flipped NN decisively at
    small (-285us); occupancy — not majors — is now the binding constraint, so NN
    routes when the instr exposes enough 128x64 tiles (MK_WGMMA_NN_MIN override; default
    16 for M=512 nano NN gemms after the later route retunes, 64 elsewhere). TN (dW,
    fp32 split-K) routes via MK_WGMMA_TN. bit10 (Drow epilogue) is implemented on both
    paths and falls under the NN tile gate like its siblings."""
    if flags & 32:
        return False
    if M % 128 or N % 64 or K % 64:
        return False
    if flags & 1:  # TN (dW split-K pattern): short-S sinks still prefer WMMA, but
        # after the later route retunes, S2048+ has enough K to amortize the WGMMA route.
        # H512/S1024 also wins because its K=1024 dW shapes have no skinny 256-wide side;
        # keep H256/S1024 and shorter shapes on WMMA. MK_WGMMA_TN force-overrides this gate.
        tn_env = os.environ.get("MK_WGMMA_TN")
        if tn_env is not None:
            return bool(int(tn_env))
        return K >= 2048 or (K == 1024 and min(M, N) >= 512)
    if not (flags & 2):  # NN (dX pattern): tile-gated
        if not int(os.environ.get("MK_WGMMA_NN", "1")):
            return False
        nn_min_env = os.environ.get("MK_WGMMA_NN_MIN")
        if nn_min_env is not None:
            nn_min = int(nn_min_env)
        elif M == 128 and N == 256:
            nn_min = 4
        elif M == 256 and N == 256:
            nn_min = 8
        elif M == 512:
            nn_min = 16
        elif M == 1024 and N == 256:
            nn_min = 32
        else:
            nn_min = 64
        return gemm_tiles_wgmma(M, N) >= nn_min
    return True  # NT


def wgmma_n128_ok(M, N, K, flags):
    """NT gemms eligible for the m64n128 tile (flags bit12): 64 accs/thread double
    the mma work per sync and halve B-traffic per FLOP at the full 255-reg df
    budget. Excludes split-K/acc/f32-out/qkrope/Drow (epilogues not implemented
    at 128 cols; CE partials bit11 and residual bit16 ARE supported).
    MK_WGMMA_N128: 0=off, 1=all eligible, 2=lm_head(bit11)-only. The default is
    shape-gated: short-row GEMMs cannot amortize the larger tile, while M>=1024 can."""
    mode_env = os.environ.get("MK_WGMMA_N128")
    if mode_env is None:
        mode = 0 if M < 256 else (2 if M < 1024 else 1)
    else:
        mode = int(mode_env)
    if mode == 0:
        return False
    if flags & (1 | 4 | 8 | 32 | 256 | 1024):
        return False
    if not (flags & 2):  # NN (dX): MN-major 128-wide B; tile-gated like the m64n64
        # NN route (halved tile count needs co-scheduling headroom). MK_WGMMA_N128_NN
        # gates it separately; threshold in n128 tiles.
        if not int(os.environ.get("MK_WGMMA_N128_NN", "1")):
            return False
        if gemm_tiles_wgmma_n128(M, N) < int(os.environ.get("MK_WGMMA_N128_NN_MIN", "32")):
            return False
    if mode == 2 and not (flags & 2048):
        return False
    return M % 128 == 0 and N % 128 == 0 and K % 64 == 0


def gemm_tiles_wgmma_n128(M, N):
    return (M // 128) * (N // 128)


def wgmma_n256_direct_ok(M, N, K, flags):
    """Qwen giant-vocab lm_head-only direct m64n256 route.

    The staged 128x256 tile wins standalone but needs 160KB smem, which does not fit the
    current cooperative launch. The 100KB direct-store variant only looked promising for
    the qwen high-K/high-V shape, so the default is an exact shape gate. Set
    MK_WGMMA_N256_DIRECT=0 to force the old n128 route, or =1 to force all eligible
    lm_head CE shapes for probing.
    """
    mode_env = os.environ.get("MK_WGMMA_N256_DIRECT")
    mode = -1 if mode_env is None else int(mode_env)
    if mode == 0:
        return False
    if flags != (2 | 2048):
        return False
    if M % 128 or N % 128 or K % 64:
        return False
    if mode == 1:
        return True
    return (M, N, K) == (1024, 151936, 2560)


def gemm_tiles_wgmma_n256_direct(M, N):
    return (M // 128) * ((N + 255) // 256)


def _default_split_k_target(K):
    env = os.environ.get("MK_DW_TARGET_TILES")
    if env is not None:
        return int(env)
    return 96 if K == 1024 else 64


def wgmma_split_k(M, N, K, target_tiles=0):
    """K-slices for a wgmma split-K gemm (64-aligned chunks)."""
    if target_tiles == 0:
        target_tiles = _default_split_k_target(K)
    mnt = gemm_tiles_wgmma(M, N)
    return max(1, min(target_tiles // max(mnt, 1), K // 64))


def gemm_tiles_wgmma(M, N):
    return (M // 128) * (N // 64)


def gemm_split_k(M, N, K, target_tiles=0):
    """K-slice count that lifts a small-M*N GEMM to ~target_tiles work items."""
    if target_tiles == 0:
        target_tiles = _default_split_k_target(K)
    mn = gemm_tiles(M, N)
    sk = max(1, min(target_tiles // mn, (K + 31) // 32))
    return sk


MAX_ARGS = 23
INSTR_INTS = 3 + MAX_ARGS  # op, tile_off, ntiles, args[23]


def load_ext(
    verbose=False,
    swiglu_bwd_2w=None,
    swiglu_cache_sig=None,
    drow_direct_store=None,
    attn_exp2_approx=None,
    lmhead_exp2_approx=None,
    ce_bwd_exp2_approx=None,
    idle_ns=None,
    attn_dkv_float2_atomic=None,
    attn_dq_float2_store=None,
    attn_dkv_row_bcast=None,
):
    # MK_OCC2=1 builds the 256-thread executors with __launch_bounds__(256, 2):
    # 2 blocks/SM (128-reg ceiling, ptxas spills the fat op paths). Motivated by the
    # P4b nsys counters — in-kernel SM issue 19%, warps-in-flight 12%, DRAM <10%:
    # the interpreter is LATENCY-bound at 1 block/SM, not bandwidth-bound. Separate
    # extension name per value: the torch build cache is name-keyed.
    occ2 = int(os.environ.get("MK_OCC2", "0"))
    regcopy = int(os.environ.get("MK_WS_REGCOPY", "0"))
    attnpipe = int(os.environ.get("MK_ATTN_PIPE", "0"))
    attn_dkv_row_bcast_env = os.environ.get("MK_ATTN_DKV_ROW_BCAST")
    if attn_dkv_row_bcast_env is not None:
        attn_dkv_row_bcast = int(attn_dkv_row_bcast_env)
    else:
        attn_dkv_row_bcast = int(bool(attn_dkv_row_bcast))
    # Direct register-layout atomics for ATTN_DKV_WG avoid staging dK/dV through smem.
    # MK_ATTN_DKV_DIRECT_ATOMIC=0 restores the old coalesced smem-drain epilogue.
    attn_dkv_direct_atomic = int(os.environ.get("MK_ATTN_DKV_DIRECT_ATOMIC", "1"))
    # MK_ATTN_DKV_FLOAT2_ATOMIC=0 restores the scalar direct-atomic epilogue for A/B.
    attn_dkv_float2_atomic_env = os.environ.get("MK_ATTN_DKV_FLOAT2_ATOMIC")
    if attn_dkv_float2_atomic_env is not None:
        attn_dkv_float2_atomic = int(attn_dkv_float2_atomic_env)
    else:
        attn_dkv_float2_atomic = int(bool(attn_dkv_float2_atomic))
    attn_dkv_float2_atomic = int(bool(attn_dkv_float2_atomic and attn_dkv_direct_atomic))
    attn_dq_float2_store_env = os.environ.get("MK_ATTN_DQ_FLOAT2_STORE")
    if attn_dq_float2_store_env is not None:
        attn_dq_float2_store = int(attn_dq_float2_store_env)
    else:
        attn_dq_float2_store = int(bool(attn_dq_float2_store))
    drow_direct_store_env = os.environ.get("MK_DROW_DIRECT_STORE")
    if drow_direct_store_env is not None:
        drow_direct_store = int(drow_direct_store_env)
    else:
        drow_direct_store = int(bool(drow_direct_store))
    # MK_ATTN_FAST_LOG=0 restores precise logf for WGMMA fwd LSE.
    attn_fast_log = int(os.environ.get("MK_ATTN_FAST_LOG", "1"))
    attn_exp2_approx_env = os.environ.get("MK_ATTN_EXP2_APPROX")
    if attn_exp2_approx_env is not None:
        attn_exp2_approx = int(attn_exp2_approx_env)
    else:
        attn_exp2_approx = int(bool(attn_exp2_approx))
    lmhead_exp2_approx = int(
        os.environ.get("MK_LMHEAD_EXP2_APPROX", int(bool(lmhead_exp2_approx)))
    )
    ce_bwd_exp2_approx = int(
        os.environ.get("MK_CE_BWD_EXP2_APPROX", int(bool(ce_bwd_exp2_approx)))
    )
    idle_ns_env = os.environ.get("MK_IDLE_NS")
    if idle_ns_env is not None:
        idle_ns = int(idle_ns_env)
    else:
        idle_ns = 256 if idle_ns is None else int(idle_ns)
    # D=64 qknorm-bwd fast path; MK_QKBWD_D64_CACHE=0 keeps the old generic loop for
    # A/B and bisects. Separate extension name because torch's cache is name-keyed.
    qkbc = int(os.environ.get("MK_QKBWD_D64_CACHE", "1"))
    # SWIGLU_BWD derivative algebra: fmaf(-sg, sig, sig+sg) avoids the explicit
    # (1-sig) dependency. MK_SWIGLU_FMA_DERIV=0 restores the old form for A/B.
    swiglu_fma_deriv = int(os.environ.get("MK_SWIGLU_FMA_DERIV", "1"))
    # H512/S1024 small uses a two-warps-per-row SwiGLU backward body by default.
    # MK_SWIGLU_BWD_2W=0/1 force-overrides the model's shape default for A/B.
    swiglu_bwd_2w_env = os.environ.get("MK_SWIGLU_BWD_2W")
    if swiglu_bwd_2w_env is not None:
        swiglu_bwd_2w = int(swiglu_bwd_2w_env)
    else:
        swiglu_bwd_2w = int(bool(swiglu_bwd_2w))
    swiglu_cache_sig = int(
        os.environ.get("MK_SWIGLU_CACHE_SIG", int(bool(swiglu_cache_sig)))
    )
    return load(
        name="xorl_megakernel" + ("_occ2" if occ2 else "") + ("_wsrc" if regcopy else "")
        + ("_apipe" if attnpipe else "") + ("_adkva" if attn_dkv_direct_atomic else "")
        + ("_adkvbc" if attn_dkv_row_bcast else "")
        + ("_adkvf2" if attn_dkv_float2_atomic else "")
        + ("_adqf2" if attn_dq_float2_store else "")
        + ("_drowst" if drow_direct_store else "")
        + ("_aflog" if attn_fast_log else "") + ("_aex2" if attn_exp2_approx else "")
        + ("_lex2" if lmhead_exp2_approx else "")
        + ("_ceb2" if ce_bwd_exp2_approx else "")
        + (f"_idle{idle_ns}" if idle_ns != 256 else "")
        + ("_qkbc" if qkbc else "")
        + ("_swfma" if swiglu_fma_deriv else "") + ("_swb2w" if swiglu_bwd_2w else "")
        + ("_swcsig" if swiglu_cache_sig else ""),
        sources=[os.path.join(_DIR, "megakernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            # wgmma + setmaxnreg need the 90a feature set. Explicit -gencode, NOT
            # -arch=sm_90a: CUDA 13.1's -arch=sm_90a also runs a compute_90 PTX embed
            # pass that rejects the megakernel_ws setmaxnreg asm (see mkv3-p2 notes).
            "-gencode=arch=compute_90a,code=sm_90a",
            "-I/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ]
        + (["-DMK_OCC2"] if occ2 else [])
        + (["-DMK_WS_REGCOPY"] if regcopy else [])
        + (["-DMK_ATTN_PIPE"] if attnpipe else [])
        + (["-DMK_ATTN_DKV_ROW_BCAST"] if attn_dkv_row_bcast else [])
        + (["-DMK_ATTN_DKV_DIRECT_ATOMIC"] if attn_dkv_direct_atomic else [])
        + (["-DMK_ATTN_DKV_FLOAT2_ATOMIC"] if attn_dkv_float2_atomic else [])
        + (["-DMK_ATTN_DQ_FLOAT2_STORE"] if attn_dq_float2_store else [])
        + (["-DMK_DROW_DIRECT_STORE"] if drow_direct_store else [])
        + (["-DMK_ATTN_FAST_LOG"] if attn_fast_log else [])
        + (["-DMK_ATTN_EXP2_APPROX"] if attn_exp2_approx else [])
        + (["-DMK_LMHEAD_EXP2_APPROX"] if lmhead_exp2_approx else [])
        + (["-DMK_CE_BWD_EXP2_APPROX"] if ce_bwd_exp2_approx else [])
        + ([f"-DMK_IDLE_NS={idle_ns}"] if idle_ns != 256 else [])
        + (["-DMK_QKBWD_D64_CACHE"] if qkbc else [])
        + (["-DMK_SWIGLU_FMA_DERIV"] if swiglu_fma_deriv else [])
        + (["-DMK_SWIGLU_BWD_2W"] if swiglu_bwd_2w else [])
        + (["-DMK_SWIGLU_CACHE_SIG"] if swiglu_cache_sig else []),
        verbose=verbose,
    )


def f2i(x: float) -> int:
    """Reinterpret a float's bits as int32 (for scalar args)."""
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


def _access_sets(op, args):
    """(read_arg_positions, write_arg_positions) for dependency analysis.

    Writes are treated as read+write (covers accumulation/atomics). Positions index
    into `args`; each named position must hold a buffer-table id.
    """
    if op == OP_NOP:
        return [], []
    if op == OP_FILL_F32:
        return [], [0]
    if op == OP_AXPY_F32:
        return [1], [0]
    if op == OP_GEMM:
        flags = args[6]
        r, w = [0, 1], [2]
        if flags & 16:
            r.append(7)
        if flags & 256:  # fused qk-norm+rope epilogue
            r += [9, 10, 13, 14]
            w += [11, 12, 15]
        if flags & 1024:  # fused Drow epilogue (dOatt gemm)
            r.append(9)
            w.append(10)
        if flags & 8192:  # fused ssq partials (rmsnorm variance pass skip)
            w.append(9)
        return r, w
    if op == OP_RMSNORM_FWD:
        r = [0, 1]
        if len(args) > 8 and args[8]:  # ssq partials from the producing gemm
            r.append(7)
        return r, [2, 3]
    if op == OP_RMSNORM_BWD:
        return [0, 1, 2, 5], [3, 4]
    if op in (OP_RMSNORM_BWD_DX, OP_RMSNORM_BWD_DX_R4, OP_RMSNORM_BWD_DX_FMA,
              OP_RMSNORM_BWD_DX_H256):
        return [0, 1, 2, 5], [3]
    if op == OP_RMSNORM_BWD_DW:
        return [0, 2, 5], [4]
    if op == OP_SWIGLU_FWD:
        return [0], ([1, 4] if len(args) > 4 else [1])
    if op in (OP_SWIGLU_BWD, OP_SWIGLU_BWD_2W):
        return ([0, 1, 6] if len(args) > 6 else [0, 1]), [2]
    if op == OP_QKNORM_ROPE_FWD:
        return [0, 2, 3, 6, 7], [1, 4, 5]
    if op == OP_QKNORM_ROPE_BWD:
        return [0, 1, 3, 4, 7, 8, 9, 10], [2, 5, 6]
    if op == OP_QKV_V_BWD:
        return [0], [1]
    if op == OP_EMBED_FWD:
        return [0, 1], [2]
    if op == OP_EMBED_BWD:
        return [0, 1], [2]
    if op == OP_INV_VALID:
        return [0], [1]
    if op == OP_CE_FWD:
        r = [0, 1, 4]
        if len(args) > 6:  # fused lm_head-epilogue lse partials
            r.append(6)
        return r, [2, 3]
    if op == OP_CE_BWD:
        return [1, 2, 3], [0]
    if op in (OP_ATTN_FWD, OP_ATTN_FWD_WG):
        # banded WG fwd chunks (12-arg form) write flash-decoding partials at
        # 9..11 instead of O/LSE; keeping 1/2 marked written is conservative-safe
        # (per-band slots make it non-serializing)
        return [0], ([1, 2, 9, 10, 11] if len(args) > 9 else [1, 2])
    if op == OP_ATTN_DPRE:
        return [0, 1], [2]
    if op in (OP_ATTN_DKV, OP_ATTN_DQ, OP_ATTN_DKV_WG, OP_ATTN_DQ_WG):
        return [0, 1, 2, 3], [4]
    if op == OP_CVT_F32BF16:
        return [0], [1]
    if op == OP_ATTN_FWD_SPLIT:
        return [0], [1, 2, 3]
    if op == OP_ATTN_COMBINE:
        return [0, 1, 2], [3, 4]
    raise ValueError(f"no access signature for op {op}")


REGION_ROWS = 128  # producer progress granularity for df2 region watermarks

ROWOP_R = 8    # swiglu/qknorm rows per tile (ops.cuh MK_ROW_R)
ROWOP_R2 = 16  # rmsnorm rows per tile, 2 rows/warp interleaved (ops.cuh MK_ROW_R2)
ROWOP_R4 = 32  # dx-only RMSNorm long-S fold, 4 rows/warp interleaved
SWIGLU_BWD_2W_R = 4  # two warps per row, four rows per 8-warp block
# measured split (v3 P4b): interleave pays only where per-row MLP is starved
# (rmsnorm's H<=512 rows = 2 load iterations); swiglu's 6-iteration rows are
# already saturated and qknorm's per-warp task chain doubles serially (+142us).

# batched row ops: tile = ROWOP_R rows (all other row-tiled ops remain 1 row/tile)
_ROW_TILE_R = {
    OP_RMSNORM_FWD: ROWOP_R2,
    OP_RMSNORM_BWD: ROWOP_R2,
    OP_RMSNORM_BWD_DX: ROWOP_R2,
    OP_RMSNORM_BWD_DX_FMA: ROWOP_R2,
    OP_RMSNORM_BWD_DX_H256: ROWOP_R2,
    OP_RMSNORM_BWD_DW: ROWOP_R2,
    OP_RMSNORM_BWD_DX_R4: ROWOP_R4,
    OP_SWIGLU_FWD: ROWOP_R,
    OP_SWIGLU_BWD: ROWOP_R,
    OP_SWIGLU_BWD_2W: SWIGLU_BWD_2W_R,
    OP_QKNORM_ROPE_BWD: ROWOP_R,
    OP_QKV_V_BWD: ROWOP_R,
}


def rowop_tiles(S, R=ROWOP_R):
    return (S + R - 1) // R

# write positions whose output is row-linear in the instr's m-major tile order
# (tile t covers rows [t*rows_per_tile, ...) — the requirement for region gating)
_ROW_WRITE_POS = {
    OP_RMSNORM_FWD: (2, 3),
    OP_RMSNORM_BWD: (3,),  # dx only; dw is a cross-row atomic scatter
    OP_RMSNORM_BWD_DX: (3,),
    OP_RMSNORM_BWD_DX_R4: (3,),
    OP_RMSNORM_BWD_DX_FMA: (3,),
    OP_RMSNORM_BWD_DX_H256: (3,),
    OP_SWIGLU_FWD: (1, 4),
    OP_SWIGLU_BWD: (2,),
    OP_SWIGLU_BWD_2W: (2,),
    OP_QKNORM_ROPE_FWD: (1, 4, 5),
    OP_QKNORM_ROPE_BWD: (2,),
    OP_QKV_V_BWD: (1,),
    OP_CE_FWD: (2,),  # lse; loss is a scalar atomic
    OP_CE_BWD: (0,),
    OP_EMBED_FWD: (2,),
}

# read positions that consume a buffer row-linearly (consumer tile = row)
_ROW_READ_POS = {
    OP_RMSNORM_FWD: (0,),
    OP_RMSNORM_BWD: (0, 2, 5),
    OP_RMSNORM_BWD_DX: (0, 2, 5),
    OP_RMSNORM_BWD_DX_FMA: (0, 2, 5),
    OP_RMSNORM_BWD_DX_H256: (0, 2, 5),
    OP_RMSNORM_BWD_DW: (0, 2, 5),
    OP_RMSNORM_BWD_DX_R4: (0, 2, 5),
    OP_SWIGLU_FWD: (0,),
    OP_SWIGLU_BWD: (0, 1, 6),
    OP_SWIGLU_BWD_2W: (0, 1, 6),
    OP_QKNORM_ROPE_FWD: (0,),
    OP_QKNORM_ROPE_BWD: (0, 1, 7, 8),
    OP_QKV_V_BWD: (0,),
    OP_CE_FWD: (0, 2, 6),
    OP_CE_BWD: (0, 2),
    OP_EMBED_BWD: (1,),
}


def _gemm_row_info(args):
    """(rows, tiles_per_region) for a gemm's OUTPUT, or None if not 128-row-band linear."""
    flags, M, N = args[6], args[3], args[4]
    if M % REGION_ROWS:
        return None
    sk = args[8] if flags & 32 else 1
    if flags & 128:
        return M, (N // 64) * sk
    return M, ((N + GEMM_BN - 1) // GEMM_BN) * 2 * sk


def _producer_row_info(op, ntiles, args, root, root_of):
    """(rows, band_tiles) if `root` is written row-linearly by this instr."""
    if op == OP_GEMM:
        if root_of(args[2]) == root:
            return _gemm_row_info(args)
        if args[6] & 256 and root_of(args[15]) == root:  # fused qkrope: qkvr rows = C rows
            return _gemm_row_info(args)
        return None
    if op == OP_ATTN_FWD:  # qt-outer tile order: O completes in row order
        S, nq = args[3], args[4]
        if root_of(args[1]) == root and S % REGION_ROWS == 0:
            return S, nq * (REGION_ROWS // 32)
        return None
    if op == OP_ATTN_FWD_WG:  # same qt-outer order, 128-row tiles: band = nq
        S, nq = args[3], args[4]
        if len(args) > 8 and args[8]:  # banded chunk: not row-linear over full S
            return None
        if root_of(args[1]) == root and S % REGION_ROWS == 0:
            return S, nq * (REGION_ROWS // 128)
        return None
    for pos in _ROW_WRITE_POS.get(op, ()):
        if pos >= len(args):
            continue
        if root_of(args[pos]) == root:
            R = _ROW_TILE_R.get(op, 1)
            rows = ntiles * R  # exact only when S % R == 0; else falls through to ungated
            return (rows, REGION_ROWS // R) if rows % REGION_ROWS == 0 else None
    return None


def _consumer_gate_k(op, ntiles, args, pos, prod_rows):
    """Consumer tiles enabled per completed producer region, or None if not gateable."""
    if op == OP_GEMM:
        flags = args[6]
        ok = (pos == 0 and not (flags & 1)) or (pos == 7 and (flags & 16))
        if not ok:
            return None  # B operand / transposed A: tile needs ALL producer rows
        info = _gemm_row_info(args)  # consumer's own M-band structure
        return info[1] if info is not None and args[3] == prod_rows else None
    if op == OP_ATTN_FWD and pos == 0:
        # qt-outer + causal: tile t needs qkvr rows < (t/nq + 1)*32 — a row prefix
        S, nq = args[3], args[4]
        if S == prod_rows and S % REGION_ROWS == 0:
            return nq * (REGION_ROWS // 32)
        return None
    if op == OP_ATTN_FWD_WG and pos == 0:
        # qt-outer + causal, 128-row tiles: tile t needs qkvr rows < (t/nq + 1)*128
        S, nq = args[3], args[4]
        if len(args) > 8 and args[8]:  # banded chunk: tile->row prefix mapping differs
            return None
        if S == prod_rows and S % REGION_ROWS == 0:
            return nq * (REGION_ROWS // 128)
        return None
    R = _ROW_TILE_R.get(op, 1)
    if pos in _ROW_READ_POS.get(op, ()) and ntiles * R == prod_rows:
        return REGION_ROWS // R
    return None


class Program:
    def __init__(self):
        self.bufs = []
        self._buf_ids = {}
        self._buf_meta = []  # (root_ptr, slot) per buffer-table entry
        self.waves = [[]]  # list of waves; each wave = list of (op, ntiles, args)
        self.default_cold_cap = 16

    def buf(self, t: torch.Tensor, slot=None) -> int:
        """Register a CUDA tensor; returns its buffer-table index.

        `slot` declares a named disjoint region of the tensor (e.g. the q vs kv halves
        of the packed dqkv buffer): entries of the SAME tensor with two different
        non-None slots are treated as non-conflicting by the dependency analysis.
        """
        assert t.is_cuda and t.is_contiguous()
        key = (t.data_ptr(), slot)
        if key not in self._buf_ids:
            self._buf_ids[key] = len(self.bufs)
            self.bufs.append(t)
            self._buf_meta.append((t.data_ptr(), slot))
        return self._buf_ids[key]

    def instr(self, op: int, ntiles: int, args):
        assert len(args) <= MAX_ARGS and ntiles >= 1
        self.waves[-1].append((op, ntiles, list(args)))

    def wave(self):
        """Close the current wave (a grid.sync boundary; ignored in dataflow mode)."""
        if self.waves[-1]:
            self.waves.append([])

    def _build_deps(self, flat):
        """RAW/WAR/WAW dependency DAG from per-op access signatures."""
        history = {}  # root_ptr -> list of (instr_idx, is_write, slot)
        deps = [set() for _ in flat]
        for idx, (op, _ntiles, args) in enumerate(flat):
            r_pos, w_pos = _access_sets(op, args)
            accesses = [(args[p], False) for p in r_pos] + [(args[p], True) for p in w_pos]
            for buf_id, is_write in accesses:
                root, slot = self._buf_meta[buf_id]
                for prior_idx, prior_write, prior_slot in history.get(root, ()):
                    if not (is_write or prior_write):
                        continue  # read-read never conflicts
                    if slot is not None and prior_slot is not None and slot != prior_slot:
                        continue  # declared-disjoint regions
                    if prior_idx != idx:
                        deps[idx].add(prior_idx)
            for buf_id, is_write in accesses:
                root, _ = self._buf_meta[buf_id]
                history.setdefault(root, []).append((idx, is_write, self._buf_meta[buf_id][1]))
        return deps

    def _build_gates(self, flat, deps):
        """Region-watermark gating for df2: pick ≤1 gated in-edge per consumer.

        A RAW edge P→C on buffer X is gateable when P writes X row-linearly in m-major
        128-row bands and C's tiles consume X row-linearly (rowop input, gemm A operand
        non-transposed, or gemm residual) — then C may claim tiles as P's row regions
        complete instead of waiting for P's last tile. The gated edge leaves C's pending
        count; every other dependency stays instruction-granular. Safe because any
        interleaved writer Q of X orders after P instruction-granularly, and the
        producer publishes an unbounded watermark on full completion.
        """
        n = len(flat)
        root_of = lambda b: self._buf_meta[b][0]  # noqa: E731
        # per-read-access dependency contributions (same conflict rules as _build_deps)
        history = {}
        read_contrib = [dict() for _ in range(n)]  # pos -> set of prior writers
        for idx, (op, _ntiles, args) in enumerate(flat):
            r_pos, w_pos = _access_sets(op, args)
            for pos in r_pos:
                root, slot = self._buf_meta[args[pos]]
                s = set()
                for prior_idx, prior_write, prior_slot in history.get(root, ()):
                    if not prior_write:
                        continue
                    if slot is not None and prior_slot is not None and slot != prior_slot:
                        continue
                    if prior_idx != idx:
                        s.add(prior_idx)
                read_contrib[idx][pos] = s
            accesses = [(args[p], False) for p in r_pos] + [(args[p], True) for p in w_pos]
            for buf_id, is_write in accesses:
                root, _ = self._buf_meta[buf_id]
                history.setdefault(root, []).append((idx, is_write, self._buf_meta[buf_id][1]))

        deps2 = [set(d) for d in deps]
        gated_in = [0] * n
        prod_info = {}  # producer idx -> (rows, band_tiles)
        gates = [[] for _ in range(n)]  # producer idx -> [(consumer, k)]
        for idx, (op, ntiles, args) in enumerate(flat):
            best = None  # (producer idx, k, rows, band_tiles)
            candidates = {max(s) for s in read_contrib[idx].values() if s}
            for P in sorted(candidates, reverse=True):
                if P not in deps[idx]:
                    continue
                op_p, ntiles_p, args_p = flat[P]
                # EVERY read position touching P must be row-linear against P's output
                # (min k across them bounds the claim); any non-gateable one keeps the
                # edge instruction-granular.
                k = None
                ok = True
                rows_p = band_p = 0
                for pos, writers in read_contrib[idx].items():
                    if P not in writers:
                        continue
                    root = self._buf_meta[args[pos]][0]
                    info = _producer_row_info(op_p, ntiles_p, args_p, root, root_of)
                    kp = None if info is None else _consumer_gate_k(op, ntiles, args, pos, info[0])
                    if kp is None:
                        ok = False
                        break
                    rows_p, band_p = info
                    k = kp if k is None else min(k, kp)
                if not ok or k is None:
                    continue
                # our writes must not touch anything P touches (WAR/WAW stay granular)
                r_p, w_p = _access_sets(op_p, args_p)
                p_roots = {root_of(args_p[q]) for q in r_p + w_p}
                _, w_pos = _access_sets(op, args)
                if any(root_of(args[q]) in p_roots for q in w_pos):
                    continue
                best = (P, k, rows_p, band_p)
                break  # latest qualifying producer wins
            if best is None:
                continue
            P, k, rows_p, band_p = best
            gated_in[idx] = 1
            deps2[idx].discard(P)
            gates[P].append((idx, k))
            prod_info[P] = (rows_p, band_p)

        band = [0] * n
        region_off, region_cnt0 = [0], []
        gate_off, gate_cons, gate_k = [0], [], []
        for i, (op, ntiles, args) in enumerate(flat):
            if i in prod_info:
                rows, bt = prod_info[i]
                band[i] = bt
                nr = (rows + REGION_ROWS - 1) // REGION_ROWS
                for r in range(nr):
                    region_cnt0.append(min(ntiles - r * bt, bt))
            region_off.append(len(region_cnt0))
            for cons, k in gates[i]:
                gate_cons.append(cons)
                gate_k.append(k)
            gate_off.append(len(gate_cons))
        return deps2, gated_in, band, region_off, region_cnt0, gate_off, gate_cons, gate_k

    def finalize(self, device="cuda"):
        if not self.waves[-1]:
            self.waves.pop()
        flat = [ins for wave in self.waves for ins in wave]

        # wave-mode arrays
        instrs, wave_start, wave_tiles = [], [0], []
        for wave in self.waves:
            off = 0
            for op, ntiles, args in wave:
                row = [op, off, ntiles] + args + [0] * (MAX_ARGS - len(args))
                instrs.extend(row)
                off += ntiles
            wave_tiles.append(off)
            wave_start.append(wave_start[-1] + len(wave))
        self._instrs = torch.tensor(instrs, dtype=torch.int32, device=device)
        self._wave_start = torch.tensor(wave_start, dtype=torch.int32, device=device)
        self._wave_tiles = torch.tensor(wave_tiles, dtype=torch.int32, device=device)
        self._buftab = torch.tensor([t.data_ptr() for t in self.bufs], dtype=torch.int64, device=device)

        # dataflow-mode arrays: same instruction order, dependency DAG instead of waves
        deps = self._build_deps(flat)
        n = len(flat)
        dep_cnt = [len(d) for d in deps]
        dependents = [[] for _ in range(n)]  # forward edges
        for i, d in enumerate(deps):
            for j in d:
                dependents[j].append(i)
        adj_off, adj = [0], []
        for i in range(n):
            adj.extend(sorted(dependents[i]))
            adj_off.append(len(adj))
        # claim quantum: 132 (the true block count). 264 was better pre-P6 (its bigger
        # batches amortized the expensive claim path); the hot/cold rings + smem-staged
        # dispatch made claims cheap enough that finer batches' tail balance wins
        # (-27/-237us vs 264). MK_CLAIM overrides for sweeps.
        cq = int(os.environ.get("MK_CLAIM", "132"))
        claim = [max(1, min(8, (ntiles + cq - 1) // cq)) for _, ntiles, _ in flat]
        # rowop claim floor: measured NEGATIVE at small (floor 2: +275us, floor 4:
        # +838us on idle GPU 0) — bigger rowop claims serialize rows per block and
        # lose the tail balance (the Stream-K physics, again). Default 1 = no-op;
        # MK_ROWOP_CLAIM re-runs the experiment.
        rc = int(os.environ.get("MK_ROWOP_CLAIM", "1"))
        _rowops = (OP_RMSNORM_FWD, OP_RMSNORM_BWD, OP_RMSNORM_BWD_DX,
                   OP_RMSNORM_BWD_DX_FMA, OP_RMSNORM_BWD_DX_H256,
                   OP_RMSNORM_BWD_DW, OP_RMSNORM_BWD_DX_R4,
                   OP_SWIGLU_FWD, OP_SWIGLU_BWD, OP_SWIGLU_BWD_2W,
                   OP_QKNORM_ROPE_FWD, OP_QKNORM_ROPE_BWD, OP_QKV_V_BWD)
        claim = [max(c, rc) if op in _rowops else c
                 for c, (op, ntiles, _) in zip(claim, flat)]
        self.n_instr = n
        self._dep_cnt = torch.tensor(dep_cnt, dtype=torch.int32, device=device)
        self._adj_off = torch.tensor(adj_off, dtype=torch.int32, device=device)
        self._adj = torch.tensor(adj if adj else [0], dtype=torch.int32, device=device)
        self._claim = torch.tensor(claim, dtype=torch.int32, device=device)
        # criticality class for the df hot/cold ready rings (v3 P6): COLD (0) when
        # nothing in the step depends on the instr (dW/sink gemms, embed_bwd) or it is
        # a fill; HOT (1) otherwise. Idle blocks drain hot first, so chain consumers
        # start within ~a claim batch instead of behind sticky off-path tile claims.
        crit = [
            0 if (adj_off[i + 1] == adj_off[i] or flat[i][0] == OP_FILL_F32) else 1
            for i in range(n)
        ]
        if os.environ.get("MK_ALLHOT"):  # bisect knob: single-ring behavior
            crit = [1] * n
        self._crit = torch.tensor(crit, dtype=torch.int32, device=device)
        # state: pending[n] | cursor[n] | done[n] | ready_hot[n] | ready_cold[n] | ctrl[8]
        self._state = torch.empty(5 * n + 8, dtype=torch.int32, device=device)
        # ws executor state: cursor[n*pad] | done[n*pad] | pending[n*pad] | ready[n]
        # | ctrl[4*pad], pad = ints per entry. Allocated at pad=32 (one 128B line per
        # instr) so any runtime pad fits, but the DEFAULT is pad=1: the 128B stride
        # measured consistently +10..+30us SLOWER in-model (ring scans touch 32x more
        # L2 lines; the wspec-probe false-sharing win does not transfer). lookahead 2 =
        # eager slot-B pre-claim (measured best); 1 = commit-late (A/B attribution).
        # lookahead DEFAULT MOVED 2 -> 1 (v3 P6): with the batched rowops' much shorter
        # tile batches, lookahead=2's eager slot-B pre-claim HANGS intermittently at the
        # small config (~1 in 2-6 rounds of 20 steps; old ops never hit it — the race
        # predates P6 but its window was cadence-narrow). la=1 stressed clean 160 steps.
        # Race is in the pre-claim path (unidentified); revisit in a dedicated ws round.
        self._ws_pad = 1
        self._ws_lookahead = int(os.environ.get("MK_WS_LOOKAHEAD", "1"))
        # + 2n rings (hot + cold, v3 P4b port) + 6*pad ctrl
        self._state_ws = torch.empty(3 * n * 32 + 2 * n + 6 * 32, dtype=torch.int32, device=device)
        self.critical_path = self._critical_path(deps, flat)

        # df2 arrays: region-watermark gating (dep DAG minus gated edges + gate CSR)
        deps2, gated_in, band, region_off, region_cnt0, gate_off, gate_cons, gate_k = (
            self._build_gates(flat, deps)
        )
        dep_cnt2 = [len(d) for d in deps2]
        dependents2 = [[] for _ in range(n)]
        for i, d in enumerate(deps2):
            for j in d:
                dependents2[j].append(i)
        adj_off2, adj2 = [0], []
        for i in range(n):
            adj2.extend(sorted(dependents2[i]))
            adj_off2.append(len(adj2))
        t32 = lambda v: torch.tensor(v if v else [0], dtype=torch.int32, device=device)  # noqa: E731
        self._dep_cnt2 = torch.tensor(dep_cnt2, dtype=torch.int32, device=device)
        self._adj_off2 = torch.tensor(adj_off2, dtype=torch.int32, device=device)
        self._adj2 = t32(adj2)
        self._gated_in = torch.tensor(gated_in, dtype=torch.int32, device=device)
        self._band = torch.tensor(band, dtype=torch.int32, device=device)
        self._region_off = torch.tensor(region_off, dtype=torch.int32, device=device)
        self._region_cnt0 = t32(region_cnt0)
        self._gate_off = torch.tensor(gate_off, dtype=torch.int32, device=device)
        self._gate_cons = t32(gate_cons)
        self._gate_k = t32(gate_k)
        # ring capacity: every gated consumer may be parked + re-pushed once per
        # producer region publish plus the final publish
        ring_cap = n
        for i in range(n):
            nr = region_off[i + 1] - region_off[i]
            ring_cap += (nr + 1) * (gate_off[i + 1] - gate_off[i])
        self._ring_cap = ring_cap
        # state2: pending|cursor|done|queued|watermark|frontier (n each)
        #         | ready[ring_cap] | ctrl[4] | rcnt[R]
        self._state2 = torch.empty(6 * n + ring_cap + 4 + len(region_cnt0), dtype=torch.int32, device=device)
        self.n_gated = sum(gated_in)
        return self

    def _critical_path(self, deps, flat):
        depth = [0] * len(flat)
        for i, d in enumerate(deps):  # instruction order is a topological order
            depth[i] = 1 + max((depth[j] for j in d), default=0)
        return max(depth, default=0)

    def run(self, ext, smem_bytes=None, wave_clk=None, mode="df", bind_bufs=None):
        if smem_bytes is None:
            # Default ops fit in 100KB. The measured-negative MK_ATTN_PIPE artifact
            # needs 112KB, so only that opt-in build takes the larger carveout.
            smem_bytes = (120 if int(os.environ.get("MK_ATTN_PIPE", "0")) else 100) * 1024
        if mode == "waves":
            ext.run(self._instrs, self._wave_start, self._wave_tiles, self._buftab, smem_bytes, wave_clk)
        elif mode == "ws":
            ext.run_ws(
                self._instrs,
                self._dep_cnt,
                self._adj_off,
                self._adj,
                self._claim,
                self._crit,
                self._state_ws,
                self._ws_pad,
                self._ws_lookahead,
                self._buftab,
                smem_bytes,
                wave_clk,
            )
        elif mode == "df2":
            ext.run_df2(
                self._instrs,
                self._dep_cnt2,
                self._adj_off2,
                self._adj2,
                self._claim,
                self._gated_in,
                self._band,
                self._region_off,
                self._region_cnt0,
                self._gate_off,
                self._gate_cons,
                self._gate_k,
                self._ring_cap,
                self._state2,
                self._buftab,
                smem_bytes,
                wave_clk,
            )
        else:
            if bind_bufs:
                (bind0, tensor0), (bind1, tensor1) = bind_bufs
                bind_args = (bind0, tensor0.data_ptr(), bind1, tensor1.data_ptr())
            else:
                bind_args = (-1, 0, -1, 0)
            ext.run_df(
                self._instrs,
                self._dep_cnt,
                self._adj_off,
                self._adj,
                self._claim,
                self._crit,
                # MK_COLD_CAP overrides. Default is shape-set by the model builder:
                # cap short shapes where cold dW work is net-contentious; leave long S
                # uncapped where the cap delays useful tail work.
                int(os.environ.get("MK_COLD_CAP", str(self.default_cold_cap))),
                self._state,
                self._buftab,
                smem_bytes,
                wave_clk,
                *bind_args,
            )


if __name__ == "__main__":
    # Skeleton smoke test: scheduling correctness + grid.sync cost.
    import time

    torch.cuda.set_device(0)
    ext = load_ext(verbose=True)

    # correctness: fill two buffers, axpy them together across many tiles/waves
    n = 1 << 20
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    y = torch.empty(n, device="cuda", dtype=torch.float32)
    p = Program()
    bx, by = p.buf(x), p.buf(y)
    ntiles = (n + 4095) // 4096
    p.instr(OP_FILL_F32, ntiles, [bx, n, f2i(3.0)])
    p.instr(OP_FILL_F32, ntiles, [by, n, f2i(1.5)])
    p.wave()
    p.instr(OP_AXPY_F32, ntiles, [by, bx, n, f2i(2.0)])
    p.wave()
    p.finalize().run(ext)
    torch.cuda.synchronize()
    assert torch.all(y == 7.5), y.unique()
    print(f"scheduling OK (nblocks={ext.nblocks()})")

    # warp-specialized executor on the same tiny program (hang canary)
    for lookahead in (1, 2):
        y.zero_()
        p._ws_lookahead = lookahead
        p.run(ext, mode="ws")
        torch.cuda.synchronize()
        assert torch.all(y == 7.5), y.unique()
    print("ws scheduling OK")

    # grid.sync overhead: 2000 waves of trivial work
    q = Program()
    bz = q.buf(x)
    for _ in range(2000):
        q.instr(OP_FILL_F32, 1, [bz, 4096, f2i(0.0)])
        q.wave()
    q.finalize()
    q.run(ext)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        q.run(ext)
    torch.cuda.synchronize()
    per_wave_us = (time.time() - t0) / 10 / 2000 * 1e6
    print(f"grid.sync + dispatch overhead: {per_wave_us:.2f} us/wave")
