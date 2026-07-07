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
OP_ATTN_FWD_WG128 = 32  # D=128 wgmma attention fwd: 64-row tiles, redundant-S + split-D halves
OP_ATTN_DKV_WG128 = 33  # D=128 wgmma attention dK/dV (same pattern)
OP_ATTN_DQ_WG128 = 34  # D=128 wgmma attention dQ (same pattern)
OP_EMBED_ZERO_ROWS = 35  # sparse grad:emb clear for previous/current token rows
OP_COPY_I32 = 36  # small state copy, currently current tokens -> previous tokens
OP_SWIGLU_BWD_4W = 37  # opt-in four-warps-per-row SwiGLU backward route
OP_SKR_REDUCE = 38  # sum split-K fp32 partial slabs (round-12 SKR head-dX route)

GEMM_BM, GEMM_BN = 64, 128  # keep in sync with ops.cuh
GEMM_SKR_FLAG = 1 << 15  # with bit5: plain per-slice slab stores + OP_SKR_REDUCE
GEMM_N256_STAGE3_FLAG = 1 << 25
GEMM_N256_NMAJOR_FLAG = 1 << 26
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
    shape-gated: S256 lm_head rechecked slower at 128 cols, S512+ keeps the larger
    tile, and M>=1024 enables all eligible routes."""
    mode_env = os.environ.get("MK_WGMMA_N128")
    if mode_env is None:
        mode = 0 if M < 512 else (2 if M < 1024 else 1)
    else:
        mode = int(mode_env)
    if mode == 0:
        return False
    if flags & (1 | 4 | 8 | 32 | 256 | 1024):
        return False
    # Post-small-4W retune: H512/S1024's remaining general NT n128 rows are
    # slower than the normal m64n64 WGMMA path. Env mode 1 remains a force-on
    # override, and mode 2 remains the lm-head-only probe.
    if mode_env is None and (flags & 2) and (M, N, K) in {
        (1024, 3072, 512),
        (1024, 512, 512),
        (1024, 512, 1536),
    }:
        return False
    if not (flags & 2):  # NN (dX): MN-major 128-wide B; tile-gated like the m64n64
        # NN route (halved tile count needs co-scheduling headroom). MK_WGMMA_N128_NN
        # gates it separately; threshold in n128 tiles.
        n128_nn_env = os.environ.get("MK_WGMMA_N128_NN")
        if n128_nn_env is not None and not int(n128_nn_env):
            return False
        # Post-small-4W retune: H512/S1024's MLP dX NN bf16 rows below run faster
        # on the normal m64n64 WGMMA path despite doubled tile count. Keep env =1
        # as a force-on override for A/B.
        if n128_nn_env is None and (M, N, K) in {
            (1024, 1536, 512),
            (1024, 512, 1024),
            (1024, 512, 3072),
        }:
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
    # long-S gauntlet head (H256/V8192): the 0928Z gauntlet gated this to M>=3072
    # (S3072 -59us, S4096 -39, S8192 -255; S<=2048/small regressed then). ALL
    # THREE stale rejections FLIPPED after the evening structural promotions
    # (SKR head-dX at small; mbar ring/commit batching at H256 S>=1024) — the
    # resweep law: small -33.5/-22.8us 40/40, s2048 -25.0/-20.9 40/40 both
    # orders (mkv3-p4b-small-lmhead-n256-postskr-*, -postskr-resweep1{,-rev}-*).
    # s1024 is dispersion-edged: probe -10.9/-11.9 (40/40+39/40) but rechecks
    # bounce -4.5..+8.4 (mkv3-p4b-s1024-n256-tiebreak-*) — kept on the probe
    # majority; revisit if a later resweep flips it consistently. s128-512
    # remain unmeasured-post; their heads are sub-wave anyway.
    return (M, N, K) in {(1024, 151936, 2560),
                         (1024, 16384, 512),  # small lm_head fwd (post-SKR)
                         (1024, 8192, 256), (2048, 8192, 256),
                         (3072, 8192, 256), (4096, 8192, 256), (8192, 8192, 256)}


def wgmma_n256_nt_bf16_ok(M, N, K, flags):
    """Qwen NT bf16 forward m64n256 route.

    This reuses the direct-store n256 kernel without CE partials, plus residual/SSQ
    support for the qwen RMSNorm producers. Keep it exact-gated: the direct epilogue
    is useful for qwen's large NT forwards, while broader direct-store use has
    regressed in standalone probes. Set
    MK_WGMMA_N256_NT_BF16=0 to restore the n128 route, or =1 to force all structurally
    eligible NT bf16 probes.
    """
    mode_env = os.environ.get("MK_WGMMA_N256_NT_BF16")
    mode = -1 if mode_env is None else int(mode_env)
    if mode == 0:
        return False
    if flags not in (2, 2 | 16):
        return False
    if M % 128 or N % 256 or K % 64:
        return False
    if mode == 1:
        return True
    return (M, N, K) in {
        (1024, 19456, 2560),  # qwen4b-l1 wgu
        (1024, 6144, 2560),   # qwen4b-l1 wqkv
        (1024, 2560, 9728),   # qwen4b-l1 wd
        (1024, 2560, 4096),   # qwen4b-l1 wo
    }


def wgmma_n256_nn_bf16_ok(M, N, K, flags):
    """Qwen NN bf16 dX m64n256 route.

    A small-shape probe of `GEMMNN 1024x512x3072` lost decisively, so the
    default tries the exact qwen4b-l1 on-path MLP dX rows plus the qkv dX row.
    Set MK_WGMMA_N256_NN_BF16=0 to force the old n128 route, or =1 to force all
    structurally eligible NN bf16 routes for probing. For an isolated A/B
    against the prior MLP dX commit, set MK_WGMMA_N256_QKVDX_BF16=0 to disable
    only the qkv dX route.
    """
    mode_env = os.environ.get("MK_WGMMA_N256_NN_BF16")
    mode = -1 if mode_env is None else int(mode_env)
    if mode == 0:
        return False
    if flags != 0:
        return False
    if M % 128 or N % 256 or K % 64:
        return False
    if mode == 1:
        return True
    if (M, N, K) in {
        (1024, 9728, 2560),    # qwen4b-l1 wd dX
        (1024, 2560, 19456),   # qwen4b-l1 wgu dX
        (8192, 768, 256),      # H256/D64/S8192 wd dX
        (8192, 256, 1536),     # H256/D64/S8192 wgu dX
        (8192, 256, 512),      # H256/D64/S8192 wo dX
    }:
        return True
    qkvdx_env = os.environ.get("MK_WGMMA_N256_QKVDX_BF16")
    qkvdx_on = True if qkvdx_env is None else bool(int(qkvdx_env))
    return qkvdx_on and (M, N, K) == (1024, 2560, 6144)  # qwen4b-l1 wqkv dX


def wgmma_n256_nn_bf16_drow_ok(M, N, K, flags):
    """Qwen fused dOatt/Drow m64n256 route.

    Exact-gated because the n256 Drow epilogue is specialized for qwen D=128:
    each 256-column tile covers two complete attention heads and publishes two
    drow reductions per row. Set MK_WGMMA_N256_DROW_BF16=0 to restore the
    existing WGMMA Drow route.
    """
    mode_env = os.environ.get("MK_WGMMA_N256_DROW_BF16")
    mode = -1 if mode_env is None else int(mode_env)
    if mode == 0:
        return False
    if flags != 1024:
        return False
    if M % 128 or N % 256 or K % 64:
        return False
    if mode == 1:
        return True
    return (M, N, K) == (1024, 4096, 2560)  # qwen4b-l1 dOatt + Drow, D=128


def gemm_tiles_wgmma_n256_direct(M, N):
    return (M // 128) * ((N + 255) // 256)


_QWEN_N256_STAGE3_SHAPES = {
    (1024, 151936, 2560),   # lm-head fwd
    (1024, 2560, 151936),   # lm-head dX
    (1024, 19456, 2560),    # wgu fwd
    (1024, 6144, 2560),     # wqkv fwd
    (1024, 2560, 9728),     # wd fwd
    (1024, 2560, 4096),     # wo fwd
    (1024, 4096, 2560),     # wo dX + Drow
    (1024, 9728, 2560),     # wd dX
    (1024, 2560, 19456),    # wgu dX
    (1024, 2560, 6144),     # wqkv dX
    (151936, 2560, 1024),   # wlm dW
    (2560, 9728, 1024),     # wd dW
    (19456, 2560, 1024),    # wgu dW
    (2560, 4096, 1024),     # wo dW
    (6144, 2560, 1024),     # wqkv dW
}


def wgmma_n256_stage3_flag(M, N, K):
    """3-stage ring flag for exact qwen n256 direct routes.

    The smem footprint is 144KB before the 1KB alignment pad, so this is only valid
    under the qwen 148KB launch carveout. The model builder owns the env gate and
    matching smem request; this helper only guards exact n256 shapes.
    """
    if M % 128 or K % 64:
        return 0
    return GEMM_N256_STAGE3_FLAG if (M, N, K) in _QWEN_N256_STAGE3_SHAPES else 0


def wgmma_n256_nmajor_flag(M, N, K):
    """Opt-in n-major tile order for exact qwen n256 direct routes.

    This preserves the same tile count and math but groups all M bands for a 256-column
    B tile together. It is a probe for B-cache reuse and the future cluster-2 B-multicast
    scheduler/body, so it remains env-gated by the model builder.
    """
    if M % 128 or K % 64:
        return 0
    return GEMM_N256_NMAJOR_FLAG if (M, N, K) in _QWEN_N256_STAGE3_SHAPES else 0


def wgmma_n256_head_dx_ok(M, N, K, flags):
    """Qwen giant-vocab head-dX-only direct m64n256 NN fp32 route.

    This reuses bit14 after the qwen lm-head forward path, but only for no-split
    WGMMA NN fp32 head-dX shapes. The default is exact because broader MLP/qkv dX n128
    expansions have mixed history. Set MK_HEAD_DX_N256_F32=0 to force the current n128
    route, or =1 to force all structurally eligible head-dX shapes for probing.
    """
    mode_env = os.environ.get("MK_HEAD_DX_N256_F32")
    mode = -1 if mode_env is None else int(mode_env)
    if mode == 0:
        return False
    if flags != (8 | 128):
        return False
    if M % 128 or N % 256 or K % 64:
        return False
    if mode == 1:
        return True
    # long-S gauntlet head-dX companions of the lm-head gate above
    return (M, N, K) in {(1024, 2560, 151936),
                         (3072, 256, 8192), (4096, 256, 8192), (8192, 256, 8192)}


def wgmma_n256_dw_tn_ok(M, N, K, flags):
    """Qwen dW-only direct m64n256 TN fp32 route.

    The standalone TN probe showed the wider tile is a clear win for qwen dW, but this
    path is exact-gated because other dW shapes have different scheduler tradeoffs. Set
    MK_DW_N256_TN_F32=0 to restore the current n64 no-atomic route, or =1 to force all
    structurally eligible TN dW fp32 shapes for probing.
    """
    mode_env = os.environ.get("MK_DW_N256_TN_F32")
    mode = -1 if mode_env is None else int(mode_env)
    if mode == 0:
        return False
    if flags != (1 | 8 | 128):
        return False
    if M % 128 or N % 256 or K % 64:
        return False
    if mode == 1:
        return True
    return (M, N, K) in {
        (151936, 2560, 1024),
        (2560, 9728, 1024),
        (19456, 2560, 1024),
        (2560, 4096, 1024),
        (6144, 2560, 1024),
    }


def gemm_n256_tma_eligible(args, tn_default=False):
    """Per-instruction gate for the round-4 TMA feed (MK_GEMM_N256_TMA).

    Only the n256 NN/TN fp32-body ring rows (bit14, not bit2) qualify. The TN
    dW rows measured order-mixed/neutral STANDALONE (n256_tma_ring_probe.py)
    but -294.6/-333.8us 16/16 both orders IN-MODEL on top of the NN promotion
    (in-model-only composition win: the sinks stop burning issue slots the
    co-scheduled chain needs), so the model builder passes tn_default=True for
    exact qwen; MK_GEMM_N256_TMA_TN overrides both ways. Geometry must be
    tile-exact (the tensormap boxes have no tail handling).
    """
    flags = args[6]
    if not (flags & 16384) or (flags & 2):
        return False
    if flags & (4 | 16 | 32 | 256 | 2048 | 8192):
        return False
    if flags & 1:
        tn_env = os.environ.get("MK_GEMM_N256_TMA_TN")
        tn_on = bool(int(tn_env)) if tn_env is not None else bool(tn_default)
        if not tn_on:
            return False
    M, N, K = args[3], args[4], args[5]
    if M % 128 or N % 256 or K % 64:
        return False
    return True


def gemm_n256_nt_tma_eligible(args):
    """NT TMA gate for the qwen giant-vocab lm-head fwd row.

    The direct n256 NT body already handles the final 128-column vocab tail via
    cp.async. The TMA path is therefore instruction-scoped here and tile-scoped
    in the kernel: args[20..22] are injected only for the exact row, while the
    device path still requires a full 256-column tile before issuing TMA.
    """
    flags = args[6]
    if not (flags & 128) or not (flags & 16384):
        return False
    if not (flags & 2) or not (flags & 2048):
        return False
    if flags & (1 | 4 | 16 | 32 | 256 | 8192):
        return False
    M, N, K = args[3], args[4], args[5]
    if (M, N, K) != (1024, 151936, 2560):
        return False
    return M % 128 == 0 and K % 64 == 0


def gemm_d64_tma_eligible(args):
    """Per-instruction gate for the D64 ring TMA feed (MK_GEMM_D64_TMA).

    Any wgmma-routed mbarrier-ring row that is NOT an n256 row qualifies:
    the m64n64 body (all four storage majors, incl. split-K dW rows — the
    feed coordinates are k-offset based) and the m64n128 body (bit12; A is
    K-major only there). bit14 rows have their own MK_GEMM_N256_TMA gate;
    the two ports share args[20..22] because the row sets are disjoint.
    Geometry must be tile-exact (the tensormap boxes have no tail handling).
    Standalone (d64_tma_ring_probe.py, both orders): every class wins —
    TN long-K dW rows -16.5..-19.6%, NT/NN short-K rows -1..-4.6%.
    MK_GEMM_D64_TMA_TN=0 excludes the TN (a_t) rows for A/B probing.
    """
    flags = args[6]
    if not (flags & 128) or (flags & 16384):
        return False
    M, N, K = args[3], args[4], args[5]
    if M % 128 or K % 64:
        return False
    if flags & 4096:  # m64n128 body: A K-major only
        if (flags & 1) or N % 128:
            return False
    elif N % 64:
        return False
    if (flags & 1) and not int(os.environ.get("MK_GEMM_D64_TMA_TN", "1")):
        return False
    # Probe-only subset filters (same-binary A/B bisection; the compiled path
    # keys off the injected args, so patch-set changes need no rebuild).
    cls_env = os.environ.get("MK_GEMM_D64_TMA_CLASS")
    if cls_env is not None:
        if ("n128" if flags & 4096 else "n64") not in cls_env.split(","):
            return False
    maj_env = os.environ.get("MK_GEMM_D64_TMA_MAJ")
    if maj_env is not None:
        maj = "TN" if flags & 1 else ("NT" if flags & 2 else "NN")
        if maj not in maj_env.split(","):
            return False
    kmin_env = os.environ.get("MK_GEMM_D64_TMA_KMIN")
    if kmin_env is not None:  # min k-iterations per tile slice (split-K aware)
        sk = args[8] if (flags & 32) and len(args) > 8 else 1
        kchunk = -(-K // (sk * 64)) * 64
        if min(K, kchunk) // 64 < int(kmin_env):
            return False
    return True


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
    swiglu_bwd_4w=False,
    swiglu_cache_sig=None,
    drow_direct_store=None,
    attn_exp2_approx=None,
    attn_exp2_prebias=None,
    lmhead_exp2_approx=None,
    ce_bwd_exp2_approx=None,
    ce_bwd_label_fixup=None,
    idle_ns=None,
    attn_fast_log=None,
    attn_dkv_float2_atomic=None,
    attn_dq_float2_store=None,
    attn_dq_rs_feed=None,
    attn_dkv_row_bcast=None,
    attn_combine_unroll=None,
    gemm_mbar_ring=None,
    gemm_n256_nt_mbar=None,
    gemm_n256_tma=None,
    gemm_n256_nt_tma=None,
    gemm_d64_tma=None,
    gemm_direct_bf16_epilogue=None,
    head_dx_skr=0,
    pdf_producer=0,
    pdf_d64_feed=None,
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
    # D64 dQ register-A dS feed: skips the dS smem store/fence/sync and feeds
    # the computed bf16 dS pairs directly to WGMMA as an RS A operand.
    attn_dq_rs_feed_env = os.environ.get("MK_ATTN_DQ_RS_FEED")
    if attn_dq_rs_feed_env is not None:
        attn_dq_rs_feed = int(attn_dq_rs_feed_env)
    else:
        attn_dq_rs_feed = int(bool(attn_dq_rs_feed))
    drow_direct_store_env = os.environ.get("MK_DROW_DIRECT_STORE")
    if drow_direct_store_env is not None:
        drow_direct_store = int(drow_direct_store_env)
    else:
        drow_direct_store = int(bool(drow_direct_store))
    # MK_ATTN_FAST_LOG=0 restores precise logf for WGMMA fwd LSE.
    attn_fast_log_env = os.environ.get("MK_ATTN_FAST_LOG")
    if attn_fast_log_env is not None:
        attn_fast_log = int(attn_fast_log_env)
    else:
        attn_fast_log = 1 if attn_fast_log is None else int(bool(attn_fast_log))
    attn_exp2_approx_env = os.environ.get("MK_ATTN_EXP2_APPROX")
    if attn_exp2_approx_env is not None:
        attn_exp2_approx = int(attn_exp2_approx_env)
    else:
        attn_exp2_approx = int(bool(attn_exp2_approx))
    # D64 attention-bwd exact-shape prebias: with the exp2 approximation enabled,
    # hoist log2(e) row bias out of the per-element expression. Env overrides keep
    # the candidate force-on/off for A/B and bisect.
    attn_exp2_prebias_env = os.environ.get("MK_ATTN_EXP2_PREBIAS")
    if attn_exp2_prebias_env is not None:
        attn_exp2_prebias = int(attn_exp2_prebias_env)
    else:
        attn_exp2_prebias = 0 if attn_exp2_prebias is None else int(bool(attn_exp2_prebias))
    attn_exp2_prebias = int(bool(attn_exp2_approx and attn_exp2_prebias))
    attn_combine_unroll = int(
        os.environ.get("MK_ATTN_COMBINE_UNROLL", int(bool(attn_combine_unroll)))
    )
    lmhead_exp2_approx = int(
        os.environ.get("MK_LMHEAD_EXP2_APPROX", int(bool(lmhead_exp2_approx)))
    )
    ce_bwd_exp2_approx = int(
        os.environ.get("MK_CE_BWD_EXP2_APPROX", int(bool(ce_bwd_exp2_approx)))
    )
    ce_bwd_label_fixup = int(
        os.environ.get("MK_CE_BWD_LABEL_FIXUP", int(bool(ce_bwd_label_fixup)))
    )
    idle_ns_env = os.environ.get("MK_IDLE_NS")
    if idle_ns_env is not None:
        idle_ns = int(idle_ns_env)
    else:
        idle_ns = 256 if idle_ns is None else int(idle_ns)
    # Round-12 SKR head-dX route (splitK + separate reduce): compiles the per-slice
    # slab stores + OP_SKR_REDUCE. Model shape defaults flow in via the kwarg;
    # MK_HEAD_DX_SKR=<slices> force-overrides for A/B (0 restores the old route).
    head_dx_skr = int(os.environ.get("MK_HEAD_DX_SKR", int(head_dx_skr)))
    # 240/24 producer-df study, phase 1: entry register ceiling on megakernel_df
    # (MK_DF_MAXNREG=240 etc.; 0/unset = plain 255). Probe-only knob.
    df_maxnreg = int(os.environ.get("MK_DF_MAXNREG", "0"))
    # 240/24 producer-df study, step A: MK_DF_PRODUCER=1 builds the 384-thread
    # megakernel_df executor shell (WG0+WG1 = consumers at setmaxnreg 240, WG2
    # parked at 24 until step B's mailbox producer; entry __maxnreg__ 168; df
    # launches at 384 threads host-side via the same define). Mutually exclusive
    # with MK_DF_MAXNREG (the variant owns the entry ceiling) and MK_OCC2
    # (__launch_bounds__(256,2) vs 384 threads).
    df_producer = int(os.environ.get("MK_DF_PRODUCER", "0"))
    assert not (df_producer and df_maxnreg), (
        "MK_DF_PRODUCER and MK_DF_MAXNREG are mutually exclusive"
    )
    assert not (df_producer and occ2), (
        "MK_DF_PRODUCER and MK_OCC2 are mutually exclusive"
    )
    # Attribution knob: plain 256-thread df with the executor-loop __syncthreads
    # swapped for the producer variant's bar.sync 1,256 (isolates barrier cost
    # from WG2-residency/setmaxnreg cost). Redundant under MK_DF_PRODUCER.
    df_named_bar = int(os.environ.get("MK_DF_NAMED_BAR", "0"))
    assert not (df_named_bar and df_producer), (
        "MK_DF_NAMED_BAR is redundant under MK_DF_PRODUCER"
    )
    # Producer-df register-point executor (megakernel_pdf): 384 threads at entry
    # __maxnreg__(168), consumers setmaxnreg.inc->MK_PDF_REGS (default 240), WG2
    # dec->MK_PDF_DEC (phase 1: exits; phase 2 MK_PDF_PRODUCER=1: pure TMA
    # producer for n256-TMA rows). Compiled only when MK_PDF=1; the reg points
    # are part of the extension name (name-keyed cache). Pool feasibility:
    # 8*regs + 4*dec <= 2048 (240/24 exact, 232/<=40, 224/<=56).
    pdf = int(os.environ.get("MK_PDF", "0"))
    pdf_regs = int(os.environ.get("MK_PDF_REGS", "240"))
    pdf_dec = int(os.environ.get("MK_PDF_DEC", "24"))
    pdf_producer = int(os.environ.get("MK_PDF_PRODUCER", int(bool(pdf_producer))))
    pdf_d64_feed = int(os.environ.get(
        "MK_PDF_D64_FEED",
        "0" if pdf_d64_feed is None else str(int(bool(pdf_d64_feed))),
    ))
    if pdf_producer:
        pdf = 1
    assert not pdf or 8 * pdf_regs + 4 * pdf_dec <= 2048, "pdf register pool infeasible"
    # D=64 qknorm-bwd fast path; MK_QKBWD_D64_CACHE=0 keeps the old generic loop for
    # A/B and bisects. Separate extension name because torch's cache is name-keyed.
    qkbc = int(os.environ.get("MK_QKBWD_D64_CACHE", "1"))
    # D=128 qwen qknorm-bwd fast path; MK_QKBWD_D128_CACHE=0 keeps the old
    # generic D!=64 loop for A/B and bisects. The kernel body is runtime-scoped
    # to the measured qwen attention shape.
    qkbc128 = int(os.environ.get("MK_QKBWD_D128_CACHE", "1"))
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
    # Qwen4B-L1 uses a four-warps-per-row SwiGLU backward by default. The env
    # override keeps the old 2W route available for A/B and bisects.
    swiglu_bwd_4w = int(os.environ.get("MK_SWIGLU_BWD_4W", int(bool(swiglu_bwd_4w))))
    swiglu_cache_sig = int(
        os.environ.get("MK_SWIGLU_CACHE_SIG", int(bool(swiglu_cache_sig)))
    )
    gemm_mbar_ring_env = os.environ.get("MK_GEMM_MBAR_RING")
    if gemm_mbar_ring_env is not None:
        gemm_mbar_ring = int(gemm_mbar_ring_env)
    else:
        gemm_mbar_ring = int(bool(gemm_mbar_ring))
    gemm_n256_nt_mbar_env = os.environ.get("MK_GEMM_N256_NT_MBAR")
    if gemm_n256_nt_mbar_env is not None:
        gemm_n256_nt_mbar = int(gemm_n256_nt_mbar_env)
    else:
        gemm_n256_nt_mbar = (
            gemm_mbar_ring
            if gemm_n256_nt_mbar is None
            else int(bool(gemm_n256_nt_mbar))
        )
    gemm_n256_nt_mbar = int(bool(gemm_mbar_ring and gemm_n256_nt_mbar))
    # GEMM round-4 TMA feed for the n256 NN/TN mbarrier-ring bodies. Requires
    # the ring; the per-instruction gate is the tmap args injected by
    # Program._inject_gemm_tmaps (this only compiles the device path).
    gemm_n256_tma_env = os.environ.get("MK_GEMM_N256_TMA")
    if gemm_n256_tma_env is not None:
        gemm_n256_tma = int(gemm_n256_tma_env)
    else:
        gemm_n256_tma = 0 if gemm_n256_tma is None else int(bool(gemm_n256_tma))
    gemm_n256_tma = int(bool(gemm_mbar_ring and gemm_n256_tma))
    # Exact-qwen n256 direct NT lm-head feed. It shares the n256 tensormap table
    # but has its own device define because the B map is [N,K] K-contiguous and
    # the final vocab tail must fall back to cp.async.
    gemm_n256_nt_tma_env = os.environ.get("MK_GEMM_N256_NT_TMA")
    if gemm_n256_nt_tma_env is not None:
        gemm_n256_nt_tma = int(gemm_n256_nt_tma_env)
    else:
        gemm_n256_nt_tma = (
            0 if gemm_n256_nt_tma is None else int(bool(gemm_n256_nt_tma))
        )
    gemm_n256_nt_tma = int(bool(gemm_mbar_ring and gemm_n256_nt_tma))
    # D64 ring TMA feed for the m64n64/m64n128 mbarrier-ring bodies. Requires
    # the ring; the per-instruction gate is the tmap args injected by
    # Program._inject_gemm_tmaps (this only compiles the device path).
    gemm_d64_tma_env = os.environ.get("MK_GEMM_D64_TMA")
    if gemm_d64_tma_env is not None:
        gemm_d64_tma = int(gemm_d64_tma_env)
    else:
        gemm_d64_tma = 0 if gemm_d64_tma is None else int(bool(gemm_d64_tma))
    gemm_d64_tma = int(bool(gemm_mbar_ring and gemm_d64_tma))
    pdf_d64_feed = int(bool(pdf_producer and gemm_d64_tma and pdf_d64_feed))
    gemm_direct_bf16_epilogue = int(
        os.environ.get(
            "MK_GEMM_DIRECT_BF16_EPILOGUE",
            int(bool(gemm_direct_bf16_epilogue)),
        )
    )
    return load(
        name="xorl_megakernel" + ("_occ2" if occ2 else "") + ("_wsrc" if regcopy else "")
        + ("_apipe" if attnpipe else "") + ("_adkva" if attn_dkv_direct_atomic else "")
        + ("_adkvbc" if attn_dkv_row_bcast else "")
        + ("_adkvf2" if attn_dkv_float2_atomic else "")
        + ("_adqf2" if attn_dq_float2_store else "")
        + ("_adqrs" if attn_dq_rs_feed else "")
        + ("_drowst" if drow_direct_store else "")
        + ("_aflog" if attn_fast_log else "") + ("_aex2" if attn_exp2_approx else "")
        + ("pb" if attn_exp2_prebias else "")
        + ("_acur" if attn_combine_unroll else "")
        + ("_lex2" if lmhead_exp2_approx else "")
        + ("_ceb2" if ce_bwd_exp2_approx else "")
        + ("_cefix" if ce_bwd_label_fixup else "")
        + (f"_idle{idle_ns}" if idle_ns != 256 else "")
        + (f"_dfnr{df_maxnreg}" if df_maxnreg else "")
        + ("_dfprod" if df_producer else "")
        + ("_dfnbar" if df_named_bar else "")
        + ("_qkbc" if qkbc else "")
        + ("_qkbc128" if qkbc128 else "")
        + ("_swfma" if swiglu_fma_deriv else "") + ("_swb2w" if swiglu_bwd_2w else "")
        + ("_swb4w" if swiglu_bwd_4w else "")
        + ("_swcsig" if swiglu_cache_sig else "")
        + ("_gmbar" if gemm_mbar_ring else "")
        + ("_n256ntold" if gemm_mbar_ring and not gemm_n256_nt_mbar else "")
        + ("_gtma" if gemm_n256_tma else "")
        + ("_nttma" if gemm_n256_nt_tma else "")
        + ("_d64tma" if gemm_d64_tma else "")
        + ("_gdbf16" if gemm_direct_bf16_epilogue else "")
        + ("_hdskr" if head_dx_skr else "")
        + (f"_pdf{pdf_regs}" if pdf else "")
        + (f"d{pdf_dec}" if pdf and pdf_dec != 24 else "")
        + ("p" if pdf_producer else "")
        + ("_pd64f" if pdf_d64_feed else ""),
        sources=[os.path.join(_DIR, "megakernel.cu")],
        extra_ldflags=["-lcuda"],
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
        + (["-DMK_ATTN_DQ_RS_FEED"] if attn_dq_rs_feed else [])
        + (["-DMK_DROW_DIRECT_STORE"] if drow_direct_store else [])
        + (["-DMK_ATTN_FAST_LOG"] if attn_fast_log else [])
        + (["-DMK_ATTN_EXP2_APPROX"] if attn_exp2_approx else [])
        + (["-DMK_ATTN_EXP2_PREBIAS"] if attn_exp2_prebias else [])
        + (["-DMK_ATTN_COMBINE_UNROLL"] if attn_combine_unroll else [])
        + (["-DMK_LMHEAD_EXP2_APPROX"] if lmhead_exp2_approx else [])
        + (["-DMK_CE_BWD_EXP2_APPROX"] if ce_bwd_exp2_approx else [])
        + (["-DMK_CE_BWD_LABEL_FIXUP"] if ce_bwd_label_fixup else [])
        + ([f"-DMK_IDLE_NS={idle_ns}"] if idle_ns != 256 else [])
        + ([f"-DMK_DF_MAXNREG={df_maxnreg}"] if df_maxnreg else [])
        + (["-DMK_DF_PRODUCER"] if df_producer else [])
        + (["-DMK_DF_NAMED_BAR"] if df_named_bar else [])
        + (["-DMK_QKBWD_D64_CACHE"] if qkbc else [])
        + (["-DMK_QKBWD_D128_CACHE"] if qkbc128 else [])
        + (["-DMK_SWIGLU_FMA_DERIV"] if swiglu_fma_deriv else [])
        + (["-DMK_SWIGLU_BWD_2W"] if swiglu_bwd_2w else [])
        + (["-DMK_SWIGLU_BWD_4W"] if swiglu_bwd_4w else [])
        + (["-DMK_SWIGLU_CACHE_SIG"] if swiglu_cache_sig else [])
        + (["-DMK_GEMM_MBAR_RING"] if gemm_mbar_ring else [])
        + (["-DMK_GEMM_N256_NT_MBAR"] if gemm_n256_nt_mbar else [])
        + (["-DMK_GEMM_N256_TMA"] if gemm_n256_tma else [])
        + (["-DMK_GEMM_N256_NT_TMA"] if gemm_n256_nt_tma else [])
        + (["-DMK_GEMM_D64_TMA"] if gemm_d64_tma else [])
        + (["-DMK_GEMM_DIRECT_BF16_EPILOGUE"] if gemm_direct_bf16_epilogue else [])
        + (["-DMK_HEAD_DX_SKR"] if head_dx_skr else [])
        + (["-DMK_PDF", f"-DMK_PDF_REGS={pdf_regs}", f"-DMK_PDF_DEC={pdf_dec}"] if pdf else [])
        + (["-DMK_PDF_PRODUCER"] if pdf_producer else [])
        + (["-DMK_PDF_D64_FEED"] if pdf_d64_feed else []),
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
    if op == OP_SKR_REDUCE:
        return [0], [1]
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
    if op in (OP_SWIGLU_BWD, OP_SWIGLU_BWD_2W, OP_SWIGLU_BWD_4W):
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
    if op == OP_EMBED_ZERO_ROWS:
        return [0, 1], [2]
    if op == OP_COPY_I32:
        return [0], [1]
    if op == OP_INV_VALID:
        return [0], [1]
    if op == OP_CE_FWD:
        r = [0, 1, 4]
        if len(args) > 6:  # fused lm_head-epilogue lse partials
            r.append(6)
        return r, [2, 3]
    if op == OP_CE_BWD:
        return [1, 2, 3], [0]
    if op == OP_ATTN_FWD_WG128:
        return [0], [1, 2]
    if op in (OP_ATTN_DKV_WG128, OP_ATTN_DQ_WG128):
        return [0, 1, 2, 3], [4]
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
SWIGLU_BWD_4W_R = 2  # four warps per row, two rows per 8-warp block
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
    OP_SWIGLU_BWD_4W: SWIGLU_BWD_4W_R,
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
    OP_SWIGLU_BWD_4W: (2,),
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
    OP_SWIGLU_BWD_4W: (0, 1, 6),
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
    if flags & GEMM_N256_NMAJOR_FLAG:
        return None
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
        # set to the loaded extension by the model builder when the n256 TMA
        # feed is enabled; finalize() then encodes per-instruction tensormaps.
        self.gemm_n256_tma_ext = None
        self.gemm_n256_tma_enabled = False
        self.gemm_n256_nt_tma_enabled = False
        self.gemm_n256_tma_tn_default = False
        self.hot_embed_bwd_default = False
        self.hot_qwen_wgu_dw_default = False
        # D64 ring TMA feed (m64n64/m64n128 mbarrier-ring bodies): same
        # contract as gemm_n256_tma_ext; the two ports share one tmap table.
        self.gemm_d64_tma_ext = None

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

    def _inject_gemm_tmaps(self):
        """Encode CUtensorMaps for TMA-eligible ring GEMMs (round-4 ports).

        Covers the n256 NN/TN rows (gemm_n256_tma_ext), the exact qwen n256
        direct NT lm-head probe row, and the D64 m64n64/m64n128 ring rows
        (gemm_d64_tma_ext) — disjoint row sets (bit14/storage-major flags) that
        share one table and the args[20..22] contract. Builds one GPU uint8
        table of 128B tensormap rows (deduped by encode key), registers it in
        the buffer table, and patches each eligible instruction's
        args[20..22] = (1 + table buf id, A row, B row). The table is
        read-only and never written by any op, so it adds no dependency
        edges; tokens/labels rebinding never touches these rows (bf16 operand
        guard).
        """
        n256_ext = self.gemm_n256_tma_ext
        d64_ext = self.gemm_d64_tma_ext
        ext = n256_ext if n256_ext is not None else d64_ext
        if ext is None:
            return
        rows, row_ids, patches = [], {}, []

        def tmap_row(ptr, inner, outer, stride_bytes, box_inner, box_outer):
            key = (ptr, inner, outer, stride_bytes, box_inner, box_outer)
            if key not in row_ids:
                row_ids[key] = len(rows)
                rows.append(
                    ext.encode_tmap_2d(ptr, inner, outer, stride_bytes, box_inner,
                                       box_outer))
            return row_ids[key]

        for wave in self.waves:
            for op, _ntiles, args in wave:
                if op != OP_GEMM or len(args) <= 6:
                    continue
                flags = args[6]
                is_n256 = (self.gemm_n256_tma_enabled and n256_ext is not None
                            and gemm_n256_tma_eligible(
                                args, self.gemm_n256_tma_tn_default))
                is_n256_nt = (self.gemm_n256_nt_tma_enabled and n256_ext is not None
                               and gemm_n256_nt_tma_eligible(args))
                is_d64 = (not (is_n256 or is_n256_nt) and d64_ext is not None
                          and gemm_d64_tma_eligible(args))
                if not (is_n256 or is_n256_nt or is_d64):
                    continue
                ta, tb = self.bufs[args[0]], self.bufs[args[1]]
                if ta.dtype != torch.bfloat16 or tb.dtype != torch.bfloat16:
                    continue
                M, N, K = args[3], args[4], args[5]
                if is_n256_nt:
                    ra = tmap_row(ta.data_ptr(), K, M, K * 2, 64, 128)  # A[M,K]
                    rb = tmap_row(tb.data_ptr(), K, N, K * 2, 64, 64)   # B[N,K]
                elif is_n256:
                    if flags & 1:  # TN: A[K,M] M-contig, two {64m,64k} boxes
                        ra = tmap_row(ta.data_ptr(), M, K, M * 2, 64, 64)
                    else:  # NN: A[M,K] K-contig, one {64k,128m} box
                        ra = tmap_row(ta.data_ptr(), K, M, K * 2, 64, 128)
                    rb = tmap_row(tb.data_ptr(), N, K, N * 2, 64, 64)  # B[K,N] N-contig
                elif flags & 4096:  # D64 m64n128 body: A always [M,K] K-contig
                    ra = tmap_row(ta.data_ptr(), K, M, K * 2, 64, 128)
                    if flags & 2:  # NT B[N,K] K-contig, one {64k,128n} box
                        rb = tmap_row(tb.data_ptr(), K, N, K * 2, 64, 128)
                    else:  # NN B[K,N] N-contig, two {64n,64k} MN boxes
                        rb = tmap_row(tb.data_ptr(), N, K, N * 2, 64, 64)
                else:  # D64 m64n64 body: all four storage majors
                    if flags & 1:  # A[K,M] M-contig, two {64m,64k} MN boxes
                        ra = tmap_row(ta.data_ptr(), M, K, M * 2, 64, 64)
                    else:  # A[M,K] K-contig, one {64k,128m} box
                        ra = tmap_row(ta.data_ptr(), K, M, K * 2, 64, 128)
                    if flags & 2:  # B[N,K] K-contig, one {64k,64n} box
                        rb = tmap_row(tb.data_ptr(), K, N, K * 2, 64, 64)
                    else:  # B[K,N] N-contig, one {64n,64k} MN box
                        rb = tmap_row(tb.data_ptr(), N, K, N * 2, 64, 64)
                patches.append((args, ra, rb))
        if not patches:
            return
        table = torch.cat(rows).cuda()
        assert table.data_ptr() % 128 == 0
        table_buf = self.buf(table)
        for args, ra, rb in patches:
            while len(args) < MAX_ARGS:
                args.append(0)
            args[20] = table_buf + 1
            args[21] = ra
            args[22] = rb

    def finalize(self, device="cuda"):
        if not self.waves[-1]:
            self.waves.pop()
        self._inject_gemm_tmaps()
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
                   OP_SWIGLU_BWD_4W,
                   OP_QKNORM_ROPE_FWD, OP_QKNORM_ROPE_BWD, OP_QKV_V_BWD)
        claim = [max(c, rc) if op in _rowops else c
                 for c, (op, ntiles, _) in zip(claim, flat)]
        # Unbanded causal D=128 WGMMA attention tiles are triangle-imbalanced:
        # claim batching runs the longest tiles serially on one block. Rechecked
        # after the qwen sparse-embed/default stack, claim1 won both construction
        # orders; MK_ATTN_D128_CLAIM1=0 restores the old ntiles/132 batching.
        if int(os.environ.get("MK_ATTN_D128_CLAIM1", "1")):
            _d128_attn = (OP_ATTN_FWD_WG128, OP_ATTN_DKV_WG128, OP_ATTN_DQ_WG128)
            claim = [1 if op in _d128_attn else c
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
        hot_embed_bwd_env = os.environ.get("MK_HOT_EMBED_BWD")
        hot_embed_bwd = (
            self.hot_embed_bwd_default
            if hot_embed_bwd_env is None
            else bool(int(hot_embed_bwd_env))
        )
        hot_qwen_wgu_dw_env = os.environ.get("MK_HOT_QWEN_WGU_DW")
        hot_qwen_wgu_dw = (
            self.hot_qwen_wgu_dw_default
            if hot_qwen_wgu_dw_env is None
            else bool(int(hot_qwen_wgu_dw_env))
        )

        def hot_qwen_wgu_dw_leaf(i, op, args):
            if not hot_qwen_wgu_dw or op != OP_GEMM:
                return False
            flags = args[6]
            return (
                adj_off[i + 1] == adj_off[i]
                and args[3] == 19456 and args[4] == 2560 and args[5] == 1024
                and (flags & 1) and not (flags & 2) and (flags & 128)
            )

        crit = [
            1 if (
                (hot_embed_bwd and flat[i][0] == OP_EMBED_BWD)
                or hot_qwen_wgu_dw_leaf(i, flat[i][0], flat[i][2])
            )
            else 0 if (adj_off[i + 1] == adj_off[i] or flat[i][0] == OP_FILL_F32)
            else 1
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
        elif mode == "pdf":
            # producer-df register-point executor: identical protocol and state
            # layout to df, so it shares self._state and the df bind fast path.
            if bind_bufs:
                (bind0, tensor0), (bind1, tensor1) = bind_bufs
                bind_args = (bind0, tensor0.data_ptr(), bind1, tensor1.data_ptr())
            else:
                bind_args = (-1, 0, -1, 0)
            ext.run_pdf(
                self._instrs,
                self._dep_cnt,
                self._adj_off,
                self._adj,
                self._claim,
                self._crit,
                int(os.environ.get("MK_COLD_CAP", str(self.default_cold_cap))),
                self._state,
                self._buftab,
                smem_bytes,
                wave_clk,
                *bind_args,
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
