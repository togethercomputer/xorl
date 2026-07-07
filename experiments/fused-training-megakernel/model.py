"""Qwen3-architecture training megakernel: builds the ONE-kernel fwd+bwd program.

MKQwen3 owns bf16 parameters, fp32 gradients, all saved activations, and the instruction
stream for a complete forward+backward (embedding -> L decoder layers -> final norm ->
lm_head -> CE -> full backward -> embedding grad). `step(tokens, labels)` runs the single
cooperative kernel launch and returns the loss buffer. The optimizer stays outside.
"""

import os
from dataclasses import dataclass

import mk
import torch


@dataclass
class Cfg:
    H: int = 256  # hidden
    L: int = 4  # layers
    nq: int = 4  # query heads
    nkv: int = 2  # kv heads
    D: int = 64  # head dim
    I: int = 768  # mlp intermediate
    V: int = 8192  # vocab
    S: int = 512  # sequence length (fixed)
    eps: float = 1e-6
    rope_theta: float = 1e6


def rope_tables(cfg, dev):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.D, 2, device=dev).float() / cfg.D))
    freqs = torch.outer(torch.arange(cfg.S, device=dev).float(), inv)
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def attn_bands(n_qt128, stages_of, T):
    """Contiguous 128-row-tile bands with chunk count C = ceil(stages/T).

    Returns [(tile_off, width, C)]; consecutive tiles with equal C share a band.
    Used by the attention bwd banding (promoted) and the fwd split banding.
    """
    out, j = [], 0
    while j < n_qt128:
        Cj = max(1, -(-stages_of(j) // T))
        k = j
        while k < n_qt128 and max(1, -(-stages_of(k) // T)) == Cj:
            k += 1
        out.append((j, k - j, Cj))
        j = k
    return out


# ---- measured per-shape tuning (v3 P4b program) ---------------------------------------
# Every EXACT-SHAPE tuned constant lives here; formula gates (drow direct store, the
# exp2 gates, dkv float2 — derivable from shape math) stay inline in __init__. Keys
# deliberately differ per knob: each keeps the gate dimensionality it was measured
# under, so these are exact relocations of the previously scattered expressions
# (routes byte-identical; verified by results/route_snapshot.py gauntlet equality in
# the knob-consol worktree). Env overrides are unchanged and noted per knob at the
# use sites. Measurement logs: NOTES.md P4b sections.

# cached-sigmoid SwiGLU fwd + two-warp bwd; keys (H, S, I). One set drives BOTH
# MK_SWIGLU_CACHE_SIG and MK_SWIGLU_BWD_2W defaults (independently overridable).
# Small H512/S1024 moved to cache-off 4W, with separate 2W fallback below.
_SWIGLU_CACHED_2W = {
    (256, 1024, 768), (256, 2048, 768),
    (256, 3072, 768), (256, 4096, 768), (256, 8192, 768),
}
_QWEN_L1_SWIGLU_BWD_2W = {
    (2560, 1024, 9728, 151936, 32, 8, 128, 1),  # H,S,I,V,nq,nkv,D,L
}
_QWEN_L1_SWIGLU_BWD_4W = {
    (2560, 1024, 9728, 151936, 32, 8, 128, 1),  # H,S,I,V,nq,nkv,D,L
    # L=2 promoted on the l2 support battery: -74.9/-75.1us, 16-17/24 both
    # orders (repeat windows; mkv3-p4b-l2followup-confirm-20260707T0120Z.log).
    (2560, 1024, 9728, 151936, 32, 8, 128, 2),
}
_SMALL_SWIGLU_BWD_4W = {
    (512, 1024, 1536, 16384, 8, 4, 64, 8),  # H,S,I,V,nq,nkv,D,L
}
_SMALL_SWIGLU_BWD_2W = _SMALL_SWIGLU_BWD_4W
_H256_IDLE32_S = (2048, 3072, 4096, 8192)  # H==256: scheduler idle poll 32ns (else 256)
_H256_DQ_FLOAT2_S = (3072, 8192)           # H==256: attention-dQ float2 direct store
_H256_D64_DKV_ROW_BCAST_S = ()             # H==256/D==64: attention-dKV row scalar shuffles
_H256_RMS_DX_H256_S = (512, 1024, 8192)    # H==256: fixed-width RMS bwd-dx opcode
_H256_D64_DROW_ZERO_SKIP_S = (256, 512)    # H==256/D==64: direct-store drow overwrites
_ATTN_BWD_BAND_T = {2048: 12, 3072: 20, 4096: 29, 8192: 40}  # H==256/D==64; 0 elsewhere
# (3072: 16->20 post-dq-p-pack, resweep flip: -12.8/-5.4 38/40+32/40 both orders)
_ATTN_FWD_BAND_T = {2048: 16, 3072: 32, 4096: 22, 8192: 64}  # H==256/D==64; 0 elsewhere
_ATTN_BAND_DQ_FIRST_S = ()  # H==256/D==64: dq-first band emission (else lpt)
_H256_D64_QKBWD_SPLIT_V_S = (3072, 4096, 8192)  # H==256/D==64: split qkrope v-bwd
_H256_D64_QKV_V_BWD_KV_SLOT_S = (3072, 8192)  # H==256/D==64: qkv-v reads only KV rows
_H256_D64_QKROPE_N128_S = (2048, 3072, 4096, 8192)  # H==256/D==64: qkv fwd fused-qkrope n128
# route. 2048 joined post pdf-d64-feed (d4758a3): the pre-wave order-mixed
# boundary flipped — 120-rep both orders -13.7 (119/120) / -11.1 (114/120)
# on a dedicated k8s H100 (resweep law; mkv3-p4b-resweep2-s2048-*).
_H256_D64_COMBINE_UNROLL_S = (4096, 8192)  # H==256/D==64: scalarized attention combine weights
# H==256 uniform attention chunks (Ckv, Cq) when bands are off; other shapes use the
# formula fallback (Ckv = 1 once nq*(S/128) >= 64 else 2, Cq = 1) at the use site.
_H256_ATTN_CHUNKS = {512: (2, 2), 1024: (2, 2), 2048: (2, 2)}
# non-WGMMA D=128 fallback DQ chunks; exact attention-shape gate from qwen4b-l1.
_D128_GENERIC_DQ_C1 = {(2560, 1024, 32, 8)}  # (H, S, nq, nkv); D==128
_ATTN_D128_FWD_MB_FLAG = 1 << 24
_ATTN_D128_DQ_RS_FLAG = 1 << 24
_D128_FWD_MBAR = {(2560, 1024, 9728, 151936, 32, 8, 128, 1)}  # H,S,I,V,nq,nkv,D,L
# qwen4b L=1 and L=2: the L2 tuple was promoted as the arm-A GEMM cluster
# (stage3+nmajor+dqRS+ring+TMA together): -1530.45/-1509.79us, 12/12 both
# construction orders (mkv3-p4b-qwenl2-gemmcluster-*-20260706T2350Z.log).
_D128_DQ_ROWSPLIT = {(2560, 1024, 9728, 151936, 32, 8, 128, 1),
                     (2560, 1024, 9728, 151936, 32, 8, 128, 2)}  # H,S,I,V,nq,nkv,D,L
# L=1 original; L=2 promoted 20260707: routing head-dX off the splitK route
# onto no-split n256+stage3+nmajor+ring+TMA measured -2152.2/-2149.6us,
# 12/12 both construction orders, loss diff exactly 0, n_instr 79->78
# (mkv3-p4b-qwenl2-headdx-n256-*-20260707T0225Z.log).
_QWEN_L1_HEAD_DX_N128_F32 = {(2560, 1024, 151936, 32, 8, 128, 1),
                             (2560, 1024, 151936, 32, 8, 128, 2)}  # H,S,V,nq,nkv,D,L
_QWEN_L1_DW_NO_ATOMIC_SK1 = {  # M,N,K for qwen4b-l1 dW GEMMs that split-K computes as sk=1
    (151936, 2560, 1024),  # wlm
    (2560, 9728, 1024),    # wd
    (19456, 2560, 1024),   # wgu
    (2560, 4096, 1024),    # wo
    (6144, 2560, 1024),    # wqkv
}
_GENERIC_SHORT_DW_NO_ATOMIC_SK1_CFGS = {
    # 20260707 current-head gate: removing generic dW sk=1 zero-fill+atomics
    # wins both construction orders only for the H256/L4 short-S family.
    # Keep S1024+, deep (L12), and small (H512) on the split-K atomic route.
    (256, 4, 4, 2, 64, 768, 8192, 128),
    (256, 4, 4, 2, 64, 768, 8192, 256),
    (256, 4, 4, 2, 64, 768, 8192, 512),
}
# dlogits @ Wlm split-K tile targets: {H: {S: target}}, 192 elsewhere.
_HEAD_DX_TARGET = {256: {128: 32, 256: 64, 1024: 64, 512: 96, 2048: 96, 3072: 96},
                   512: {1024: 96}}
# Round-12 SKR head-dX (splitK + separate reduce): K-sliced n128 tiles write plain
# fp32 partial slabs; OP_SKR_REDUCE sums them into dXN_f32. Value = slice count.
# Measured: small skr=2 -115/-108us 40/40 both orders (skr=3 -98, skr=4 -77/-59);
# nano skr=4 -10.5/-14.3 (37/40+40/40) and deep skr=4 -32.9/-31.1 (38/40+40/40),
# replacing their zero-fill+atomic splitK head route (skr=6 no better); s3072
# skr=2 -118.6/-120.6 40/40 and s4096 skr=2 -90.0/-75.6 40/40+39/40 at 0abc08f
# (their n256 direct head-dX was 24/32 CTAs, parallelism-starved; K/CTAs 170/128
# matches nvjet's >=90 split gate; skr=4 inferior -100.4 — one wave is enough);
# s128 NO-GO +36.1 0/40 and s256 NO-GO +11.9 2/40 (the n128 tile shape collapses
# at M<=256); s1024 order-flipped at 0abc08f (-6.3 69/80 then +10.0 4/80; won
# both orders pre-qkbwd at 5c6e234 — resweep-law casualty, stays atomic); s2048
# NO-GO (+18 skr=4, +3 skr=2) — its no-atomic direct route stands.
# MK_HEAD_DX_SKR force-overrides (0 = old route).
_HEAD_DX_SKR = {
    (512, 1024, 1536, 16384, 8, 4, 64, 8): 2,  # H,S,I,V,nq,nkv,D,L (small)
    (256, 512, 768, 8192, 4, 2, 64, 4): 4,     # nano
    (256, 512, 768, 8192, 4, 2, 64, 12): 4,    # deep
    (256, 3072, 768, 8192, 4, 2, 64, 4): 2,    # s3072
    (256, 4096, 768, 8192, 4, 2, 64, 4): 2,    # s4096
}
# Producer-df executor mode (megakernel_pdf): 384thr, consumers in a
# setmaxnreg-region at 240 regs, WG2 = pure-TMA producer feeding the n256-TMA
# rows via the MkPdfFeed smem mailbox. Measured at head e8837a5 (k8s paired
# same-binary df-vs-pdf, both orders, parity clean + GPU-5 iclk profile):
# qwen4b-l1 -1056/-1127us 12/12 (lm-head dX span 2453->1680, -32%),
# qwen4b-l2 -1511/-1370us 12/12 (on top of the l2 gemm-cluster promotion),
# s3072 -20.7/-21.7us 40/40+39/40, s8192 -171/-162us 16/16 (head binary;
# pre-rebase binary read +63 — re-certify after any big codegen shift);
# NO-GO small +142/+147us 0/40 (the 240-region tax, short-S stays df).
# Flat-cap control MK_DF_MAXNREG=240 at qwen is +30/+44us WORSE — the
# region-compiled image + producer, not the ceiling, carry the win.
# MK_MODE force-overrides (df|pdf|ws|df2|waves).
_PDF_MODE = {
    (2560, 1024, 9728, 151936, 32, 8, 128, 1),  # H,S,I,V,nq,nkv,D,L (qwen4b-l1)
    (2560, 1024, 9728, 151936, 32, 8, 128, 2),  # qwen4b-l2
    (256, 2048, 768, 8192, 4, 2, 64, 4),        # s2048
    (256, 3072, 768, 8192, 4, 2, 64, 4),        # s3072
    (256, 4096, 768, 8192, 4, 2, 64, 4),        # s4096
    (256, 8192, 768, 8192, 4, 2, 64, 4),        # s8192
}


def _cold_cap(c):
    """Hot/cold ring cold-work cap (v3 P6/P4b retunes; 0 = uncapped)."""
    if c.H == 256 and c.L == 4 and c.S in (128, 512):
        return 0
    if c.S >= 2048:
        return 0
    if c.L == 1 and c.H >= 1024 and c.V >= 32768:
        return 0
    if c.H == 256 and c.S == 1024:
        return 64
    if c.S >= 1024:
        return 48
    return 16


class MKQwen3:
    def __init__(self, cfg: Cfg, dev="cuda", seed=0):
        self.cfg = cfg
        self.dev = dev
        torch.manual_seed(seed)
        c = cfg
        QD = (c.nq + 2 * c.nkv) * c.D
        bf, f32 = torch.bfloat16, torch.float32
        swiglu_cache_sig_env = os.environ.get("MK_SWIGLU_CACHE_SIG")
        # S8192 joined post-band: the pre-band recheck rejected it (+47.5us), but
        # the banded-attention scheduling regime flipped it to -112/-128us (16/16
        # both construction orders; mkv3-p4b-postband-knob-recheck-20260705T185629Z).
        self.swiglu_cache_sig_default = (c.H, c.S, c.I) in _SWIGLU_CACHED_2W
        self.swiglu_cache_sig_enabled = (
            self.swiglu_cache_sig_default
            if swiglu_cache_sig_env is None
            else bool(int(swiglu_cache_sig_env))
        )

        def P(*shape, std=0.02):
            return (torch.randn(*shape, device=dev) * std).to(bf).contiguous()

        # ---- parameters (bf16) and grads (fp32) ----
        self.params, self.grads = {}, {}

        def par(name, *shape, std=0.02, ones=False):
            t = torch.ones(*shape, device=dev, dtype=bf) if ones else P(*shape, std=std)
            self.params[name] = t
            self.grads[name] = torch.zeros(*shape, device=dev, dtype=f32)
            return t

        par("emb", c.V, c.H)
        for l in range(c.L):
            par(f"w1.{l}", c.H, ones=True)
            par(f"wqkv.{l}", QD, c.H)
            par(f"qn.{l}", c.D, ones=True)
            par(f"kn.{l}", c.D, ones=True)
            par(f"wo.{l}", c.H, c.nq * c.D)
            par(f"w2.{l}", c.H, ones=True)
            par(f"wgu.{l}", 2 * c.I, c.H)
            par(f"wd.{l}", c.H, c.I)
        par("wf", c.H, ones=True)
        par("wlm", c.V, c.H)

        # ---- activations saved for backward ----
        A = {}
        A["X"] = torch.empty(c.L + 1, c.S, c.H, device=dev, dtype=bf)  # residual stream
        for l in range(c.L):
            A[f"rstd1.{l}"] = torch.empty(c.S, device=dev, dtype=f32)
            A[f"xn1.{l}"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
            A[f"qkvraw.{l}"] = torch.empty(c.S, QD, device=dev, dtype=bf)
            A[f"qkvr.{l}"] = torch.empty(c.S, QD, device=dev, dtype=bf)
            A[f"rq.{l}"] = torch.empty(c.S, c.nq, device=dev, dtype=f32)
            A[f"rk.{l}"] = torch.empty(c.S, c.nkv, device=dev, dtype=f32)
            A[f"oatt.{l}"] = torch.empty(c.S, c.nq * c.D, device=dev, dtype=bf)
            A[f"lse.{l}"] = torch.empty(c.nq, c.S, device=dev, dtype=f32)
            A[f"x2.{l}"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
            A[f"rstd2.{l}"] = torch.empty(c.S, device=dev, dtype=f32)
            A[f"xn2.{l}"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
            A[f"gu.{l}"] = torch.empty(c.S, 2 * c.I, device=dev, dtype=bf)
            A[f"hs.{l}"] = torch.empty(c.S, c.I, device=dev, dtype=bf)
            if self.swiglu_cache_sig_enabled:
                A[f"swsig.{l}"] = torch.empty(c.S, c.I, device=dev, dtype=bf)
        if c.H % 64 == 0:  # ssq partials for the bit13 fusion (tiny: S x H/64 fp32)
            for l in range(c.L):
                A[f"x2ssq.{l}"] = torch.empty(c.S, c.H // 64, device=dev, dtype=f32)
            for l in range(1, c.L + 1):
                A[f"Xssq.{l}"] = torch.empty(c.S, c.H // 64, device=dev, dtype=f32)
        A["rstdf"] = torch.empty(c.S, device=dev, dtype=f32)
        A["xnf"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
        A["logits"] = torch.empty(c.S, c.V, device=dev, dtype=bf)
        A["lse_ce"] = torch.empty(c.S, device=dev, dtype=f32)
        self.acts = A

        # ---- io + backward workspace ----
        self.tokens = torch.zeros(c.S, device=dev, dtype=torch.int32)
        self.prev_tokens = torch.zeros(c.S, device=dev, dtype=torch.int32)
        self.labels = torch.full((c.S,), -100, device=dev, dtype=torch.int32)
        self.inv_valid = torch.zeros(1, device=dev, dtype=f32)
        self.loss = torch.zeros(1, device=dev, dtype=f32)
        W = {}
        W["dX"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
        W["dXN"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
        W["dQKVraw"] = torch.empty(c.S, QD, device=dev, dtype=bf)
        W["dOatt"] = torch.empty(c.S, c.nq * c.D, device=dev, dtype=bf)
        W["dGU"] = torch.empty(c.S, 2 * c.I, device=dev, dtype=bf)
        W["dHs"] = torch.empty(c.S, c.I, device=dev, dtype=bf)
        # per layer (like dQKV_f32): Drow is atomically accumulated by the fused dOatt
        # gemm epilogue, so its zero-fill must be a dependency root, not chained
        for l in range(c.L):
            W[f"drow.{l}"] = torch.empty(c.nq, c.S, device=dev, dtype=f32)
        # split-KV attention fwd partials (flash-decoding style combine). Measured
        # neutral at nano and NEGATIVE at small (combine hop + partial traffic outweigh
        # the chain-latency saving) -> routing disabled; ops retained for future tuning.
        self.attn_C = 1
        if self.attn_C > 1:
            W["opart"] = torch.empty(self.attn_C, c.S, c.nq * c.D, device=dev, dtype=bf)
            W["mpart"] = torch.empty(self.attn_C, c.nq, c.S, device=dev, dtype=f32)
            W["lpart"] = torch.empty(self.attn_C, c.nq, c.S, device=dev, dtype=f32)
        # fp32 atomic-accumulation workspaces (attention bwd splits; big split-K gemms).
        # dQKV_f32 is per layer so its zero-fill runs up front with no inter-layer
        # dependency (a shared one chains layer N's fill behind layer N+1's convert).
        for l in range(c.L):
            W[f"dQKV_f32.{l}"] = torch.empty(c.S, QD, device=dev, dtype=f32)
        W["dXN_f32"] = torch.empty(c.S, c.H, device=dev, dtype=f32)
        # round-12 SKR head-dX (see _HEAD_DX_SKR): per-K-slice fp32 partial slabs for
        # the split head-dX gemm; OP_SKR_REDUCE sums them into dXN_f32
        skr_env = os.environ.get("MK_HEAD_DX_SKR")
        skr_key = (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L)
        self.head_dx_skr = (
            _HEAD_DX_SKR.get(skr_key, 0) if skr_env is None else int(skr_env)
        )
        if not (c.S % 128 == 0 and c.H % 128 == 0 and c.V % 64 == 0):
            self.head_dx_skr = 0
        if self.head_dx_skr > 1:
            W["headdx_skr"] = torch.empty(self.head_dx_skr, c.S, c.H, device=dev, dtype=f32)
        # per-layer fp32 workspaces for the parallelism-starved bwd dX gemms
        # (dx_split_k routing; see gemm_dx below); allocated only for the shapes the
        # tile gate will actually split
        self.dx_split_k = True
        if self.dx_split_k:
            for l in range(c.L):
                if mk.gemm_tiles(c.S, c.H) < 32:
                    W[f"dXN2_f32.{l}"] = torch.empty(c.S, c.H, device=dev, dtype=f32)
                    W[f"dXN1_f32.{l}"] = torch.empty(c.S, c.H, device=dev, dtype=f32)
                if mk.gemm_tiles(c.S, c.I) < 32:
                    W[f"dHs_f32.{l}"] = torch.empty(c.S, c.I, device=dev, dtype=f32)
        # per-row (max, sumexp) partials from the lm_head gemm epilogue (bit11)
        if c.V % 64 == 0:
            W["lse_parts"] = torch.empty(c.S, c.V // 64, 2, device=dev, dtype=f32)
        # fwd split-band partials (MK_ATTN_FWD_BAND = target stages per chunk, 0 =
        # off): straggler q-tiles run as flash-decoding kv chunks writing
        # locally-normalized partials; a range-limited OP_ATTN_COMBINE merges them.
        default_attn_fwd_band_T = (
            _ATTN_FWD_BAND_T.get(c.S, 0) if c.H == 256 and c.D == 64 else 0
        )
        self.attn_fwd_band_T = int(os.environ.get("MK_ATTN_FWD_BAND", str(default_attn_fwd_band_T)))
        self.attn_fwd_bands = None
        if self.attn_fwd_band_T > 0 and c.D == 64 and c.S % 128 == 0:
            fb = attn_bands(c.S // 128, lambda i: i * 2 + 2, self.attn_fwd_band_T)
            if max(cb for _, _, cb in fb) > 1:
                self.attn_fwd_bands = fb
                cmax = max(cb for _, _, cb in fb)
                W["fopart"] = torch.empty(cmax, c.S, c.nq * c.D, device=dev, dtype=bf)
                W["fmpart"] = torch.empty(cmax, c.nq, c.S, device=dev, dtype=f32)
                W["flpart"] = torch.empty(cmax, c.nq, c.S, device=dev, dtype=f32)
        self.ws = W

        self.cos, self.sin = rope_tables(c, dev)
        self.swiglu_bwd_2w_default = (
            self.swiglu_cache_sig_default
            or (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _QWEN_L1_SWIGLU_BWD_2W
            or (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _SMALL_SWIGLU_BWD_2W
        )
        self.swiglu_bwd_4w_default = (
            (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _QWEN_L1_SWIGLU_BWD_4W
            or (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _SMALL_SWIGLU_BWD_4W
        )
        self.drow_direct_store_default = c.D == 64 and c.S < 2048
        drow_direct_store_env = os.environ.get("MK_DROW_DIRECT_STORE")
        self.drow_direct_store_enabled = (
            self.drow_direct_store_default
            if drow_direct_store_env is None
            else bool(int(drow_direct_store_env))
        )
        self.drow_direct_store_overwrites = (
            self.drow_direct_store_enabled and c.D == 64 and c.S < 2048
        )
        drow_zero_fill_env = os.environ.get("MK_DROW_ZERO_FILL")
        self.drow_zero_fill_default = not (
            self.drow_direct_store_overwrites
            and c.H == 256 and c.L == 4 and c.S in _H256_D64_DROW_ZERO_SKIP_S
        )
        self.drow_zero_fill_enabled = (
            self.drow_zero_fill_default
            if drow_zero_fill_env is None
            else bool(int(drow_zero_fill_env)) or not self.drow_direct_store_overwrites
        )
        self.attn_exp2_approx_default = c.D == 64 and c.S >= 512 and c.S % 128 == 0
        self.lmhead_exp2_approx_default = c.V >= 8192 and c.V % 64 == 0 and c.S >= 256
        self.ce_bwd_exp2_approx_default = c.S >= 1024 and c.V >= 8192 and c.V % 8 == 0
        # S2048/S8192 joined post-band; the full long-S H256 bucket now wins 32ns
        # polling after the banding/row-batching scheduling changes. Exact small
        # joined at 64ns post-SKR/n256 (32/64/128 all beat 256 in 10 paired runs;
        # 64 = plateau center, -37.3 40/40 / -6.4 34/40 both orders —
        # mkv3-p4b-small-idle{,-postskr,64-check,128-rev}-*.log).
        # nano/deep joined at 64ns post-SKR (nano -5.7/-5.6, 33/40 + 107/120;
        # deep -28.0 40/40 / -31.4 38/40; idle32 equivalent — plateau, 64 kept
        # for consistency with small; mkv3-p4b-nano-deep-idle-postskr-*,
        # mkv3-p4b-idle-rev-dfnr240-*.log).
        if c.H == 256 and c.S in _H256_IDLE32_S:
            self.idle_ns_default = 32
        elif (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) == (512, 1024, 1536, 16384, 8, 4, 64, 8):
            self.idle_ns_default = 64
        elif (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D) == (256, 512, 768, 8192, 4, 2, 64) and c.L in (4, 12):
            self.idle_ns_default = 64
        else:
            self.idle_ns_default = 256
        self.attn_dkv_float2_atomic_default = c.D == 64 and c.S % 128 == 0
        self.attn_dkv_row_bcast_default = (
            c.H == 256 and c.D == 64 and c.S in _H256_D64_DKV_ROW_BCAST_S
        )
        self.attn_dq_float2_store_default = c.H == 256 and c.S in _H256_DQ_FLOAT2_S
        self.attn_fast_log_default = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            != (512, 8, 1024, 8, 4, 64, 1536, 16384)
        )
        exact_qwen4b_l1 = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            == (2560, 1, 1024, 32, 8, 128, 9728, 151936)
        )
        exact_qwen4b_l2 = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            == (2560, 2, 1024, 32, 8, 128, 9728, 151936)
        )
        exact_s8192 = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            == (256, 4, 8192, 4, 2, 64, 768, 8192)
        )
        exact_s4096 = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            == (256, 4, 4096, 4, 2, 64, 768, 8192)
        )
        exact_s3072 = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            == (256, 4, 3072, 4, 2, 64, 768, 8192)
        )
        exact_small_h512_s1024 = (
            (c.H, c.L, c.S, c.nq, c.nkv, c.D, c.I, c.V)
            == (512, 8, 1024, 8, 4, 64, 1536, 16384)
        )
        # exact_qwen4b_l2 added on the l2 support battery: 4 negative windows
        # (-52.6/-12.9/-13.6/-73.1us; mkv3-p4b-qwenl2-peel-support + followup).
        self.ce_bwd_label_fixup_default = (
            exact_qwen4b_l1 or exact_s8192 or exact_small_h512_s1024
            or exact_qwen4b_l2
        )
        self.gemm_mbar_ring_default = (
            c.D == 64 and c.S >= 1024 and c.S % 128 == 0
        ) or exact_qwen4b_l1 or exact_qwen4b_l2
        self.gemm_n256_nt_mbar_default = (
            self.gemm_mbar_ring_default
            and not (exact_qwen4b_l1 or exact_qwen4b_l2)
        )
        # GEMM round-4 TMA feed on the n256 NN ring rows (elected-thread
        # cp.async.bulk.tensor + expect_tx from a global tensormap table).
        # Promoted exact-qwen: -340.40/-335.25us, 16/16 both construction
        # orders, parity clean (mkv3-p4b-qwen-n256tma-*-20260706T2145Z.log).
        # H256/D64 boundary: S3072 was briefly promoted (its only eligible
        # row was the long-K head-dX) but REVERTED after fe15e24 moved that
        # row to the SKR route — post-SKR windows read TMA-off faster by
        # -16.3/-20.2us at 40/40 both orders (resweep law). S4096
        # (+6.1/+11.9, <=7/40) and S8192 (+27.3/+33.6, 2/16 — 12 of 13 rows
        # short-K, fence/expect_tx does not amortize) are NO-GO;
        # see results/operator-gap/s8192-n256tma-nogo.md.
        # MK_GEMM_N256_TMA=0 restores the per-thread cp.async ring feed;
        # =1 force-enables for probing; TN rows stay off (order-mixed
        # standalone) behind MK_GEMM_N256_TMA_TN except at exact qwen.
        self.gemm_n256_tma_default = exact_qwen4b_l1 or exact_qwen4b_l2
        # Qwen4B-L1 lm-head fwd NT row: TMA/pdf producer feed for full
        # 256-column vocab tiles, with the final 128-column tail on the existing
        # cp.async direct body. L2 stayed mixed in the local order-reversal
        # window, so keep the default l1-only; MK_GEMM_N256_NT_TMA=0/1 guards A/B.
        self.gemm_n256_nt_tma_default = exact_qwen4b_l1
        # L2 terminal EMBED_BWD sits as an off-path cold leaf while lm-head dW drains;
        # making only that leaf hot recovers ~0.14-0.17ms at exact qwen4b-l2 with
        # qwen4b-l1 neutral and generic shapes order-mixed. MK_HOT_EMBED_BWD=0/1
        # remains the explicit override.
        self.hot_embed_bwd_default = exact_qwen4b_l2
        # L2 layer-0 WGU dW leaf (#64) becomes the next terminal cold sink after
        # EMBED_BWD is hot. Marking only that exact leaf hot wins locally in both
        # construction orders; MK_HOT_QWEN_WGU_DW=0 restores the cold-leaf route.
        self.hot_qwen_wgu_dw_default = exact_qwen4b_l2
        # D64 ring TMA feed (round-4 port to the m64n64/m64n128 mbarrier-ring
        # bodies, all majors): standalone every class wins (d64_tma_ring_probe
        # both orders; TN long-K dW -16.5..-19.6%, NT/NN -1..-4.6%). Default ON
        # for the exact long-S H256/D64 bench shapes only. S2048 was widened
        # after the current-head gate at 9550a7b (-16.34/-8.40us, 80/80 and
        # 74/80 wins; S1024/H512 small stayed no-go). MK_GEMM_D64_TMA=0/1
        # overrides, MK_GEMM_D64_TMA_TN=0 excludes the TN dW rows for A/B.
        exact_long_d64 = (
            (c.H, c.L, c.nq, c.nkv, c.D, c.I, c.V) == (256, 4, 4, 2, 64, 768, 8192)
            and c.S in (2048, 3072, 4096, 8192)
        )
        self.gemm_d64_tma_default = exact_long_d64
        self.gemm_direct_bf16_epilogue_default = c.D == 64 and (
            c.S == 128 or (c.H, c.L, c.S, c.nq, c.nkv, c.I) == (512, 8, 1024, 8, 4, 1536)
        )
        self.attn_combine_unroll_default = (
            c.H == 256
            and c.L == 4
            and c.S in _H256_D64_COMBINE_UNROLL_S
            and c.nq == 4
            and c.nkv == 2
            and c.D == 64
            and c.I == 768
            and c.V == 8192
        )
        # D64 attention-bwd exp2 prebias: exact S4096 was promoted first; after
        # the S8192 dQ register-feed landing, exact S8192 also wins both orders.
        # Keep the gate exact because small stayed neutral/negative.
        self.attn_exp2_prebias_default = exact_s4096 or exact_s8192
        # D64 dQ register-A dS feed: exact S4096/S8192 remove the dS smem round
        # trip and won both construction orders after the pdf+d64feed/scalar
        # store gates. S3072 stayed too small in reverse-order confirmation.
        # MK_ATTN_DQ_RS_FEED=0/1 remains the explicit A/B override.
        self.attn_dq_rs_feed_default = exact_s4096 or exact_s8192
        # D64 dQ fp32-P pack: exact S8192 and S3072 win by removing an extra
        # bf16(P)->fp32(P) round before the final bf16 dS pack on the RS-feed
        # path. S4096 was neutral, so keep this exact-shape gated;
        # MK_ATTN_DQ_FP32_P=0/1 guards A/B.
        self.attn_dq_fp32_p_default = exact_s3072 or exact_s8192
        # D64 dQ C>1 bulk-reduce drain (cp.reduce.async.bulk.add.f32): exact
        # S4096 wins -32.7..-41.4us at 40/40 both orders plus reruns; S8192 is
        # REFUTED (+39..+55us — the drain's mandatory wait_group 0 puts the TMA
        # round trip on the critical C=4 tail bands; the replaced fp32 atomics
        # are fire-and-forget). MK_ATTN_DQ_BULK_RED=0/1 guards A/B. Flag-on
        # builds force-inline op_attn_dq_wg and hard-audit UBLKRED F32 SASS per
        # kernel clone (ptxas 13.1 otherwise silently assembles the asm as
        # ADD.U64 in all but one cloned entry).
        # See results/operator-gap/dq-bulkreduce-drain-b1d36305.md.
        self.attn_dq_bulk_red_default = exact_s4096
        # Producer-df default mode (per-shape executor routing; see _PDF_MODE).
        # The pdf executor + WG2 producer compile only for gated shapes.
        self.default_mode = (
            "pdf"
            if (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _PDF_MODE
            else "df"
        )
        mode_env = os.environ.get("MK_MODE")
        if mode_env:
            self.default_mode = mode_env
        # D64-TMA producer feed on the pdf executor. The opt-in probe split by
        # shape: S2048, S3072, and S4096 won in both orders, while S8192
        # regressed decisively. Keep the default exact and let MK_PDF_D64_FEED
        # force A/Bs.
        self.pdf_d64_feed_default = (
            self.default_mode == "pdf"
            and exact_long_d64
            and c.S in (2048, 3072, 4096)
        )
        # Attention producer-feed (attn-pdf-feed lane, Phase 0/1): WG2 issues
        # the streamed attention stage cp.async fills (dq K/V at =1, + dkv
        # Q/dO at =2) into the existing wga_off64 layout. Default OFF
        # everywhere until an exact-S8192 both-order win is certified;
        # MK_ATTN_PDF_FEED=1/2 force-enables on pdf-mode shapes for probing.
        self.attn_pdf_feed_default = 0
        self.ext = mk.load_ext(
            swiglu_bwd_2w=self.swiglu_bwd_2w_default,
            swiglu_bwd_4w=self.swiglu_bwd_4w_default,
            swiglu_cache_sig=self.swiglu_cache_sig_enabled,
            drow_direct_store=self.drow_direct_store_enabled,
            attn_exp2_approx=self.attn_exp2_approx_default,
            attn_exp2_prebias=self.attn_exp2_prebias_default,
            lmhead_exp2_approx=self.lmhead_exp2_approx_default,
            ce_bwd_exp2_approx=self.ce_bwd_exp2_approx_default,
            ce_bwd_label_fixup=self.ce_bwd_label_fixup_default,
            idle_ns=self.idle_ns_default,
            attn_fast_log=self.attn_fast_log_default,
            attn_dkv_float2_atomic=self.attn_dkv_float2_atomic_default,
            attn_dkv_row_bcast=self.attn_dkv_row_bcast_default,
            attn_dq_float2_store=self.attn_dq_float2_store_default,
            attn_dq_fp32_p=self.attn_dq_fp32_p_default,
            attn_dq_rs_feed=self.attn_dq_rs_feed_default,
            attn_dq_bulk_red=self.attn_dq_bulk_red_default,
            attn_combine_unroll=self.attn_combine_unroll_default,
            gemm_mbar_ring=self.gemm_mbar_ring_default,
            gemm_n256_nt_mbar=self.gemm_n256_nt_mbar_default,
            gemm_n256_tma=self.gemm_n256_tma_default,
            gemm_n256_nt_tma=self.gemm_n256_nt_tma_default,
            gemm_d64_tma=self.gemm_d64_tma_default,
            gemm_direct_bf16_epilogue=self.gemm_direct_bf16_epilogue_default,
            head_dx_skr=self.head_dx_skr,
            pdf_producer=self.default_mode == "pdf",
            pdf_d64_feed=self.pdf_d64_feed_default,
            attn_pdf_feed=self.attn_pdf_feed_default,
        )
        # D=128 WGMMA attention route (default ON for D==128, S%64==0; the opgap
        # FA4-C trio spec's fallback replacement): MK_ATTN_D128_WG=0 restores the
        # generic WMMA ops. qwen4b-l1 in-model: -1586/-1846us, 12/12 both orders.
        # Its DKV op needs a 112KB smem struct and ws mode offsets ops by 256B of
        # control smem, so the route takes the 120KB carveout (MK_ATTN_PIPE
        # precedent, measured neutral).
        d128_env = os.environ.get("MK_ATTN_D128_WG")
        self.attn_d128_wg_enabled = (
            (c.D == 128 and c.S % 64 == 0)
            if d128_env is None
            else (bool(int(d128_env)) and c.D == 128 and c.S % 64 == 0)
        )
        dq_rs_env = os.environ.get("MK_ATTN_D128_DQ_RS")
        # dq RS-feed stays L1-ONLY: the l2 peel measured RS-off faster in 4
        # consecutive windows (-165.9/-63.9/-85.1/-52.8us, 10-16/24 wins;
        # mkv3-p4b-qwenl2-{peel-support,followup-confirm} logs) — the l2
        # 2-layer dq structure does not profit from the row-split. stage3/
        # nmajor/smem keep the full _D128_DQ_ROWSPLIT (l1+l2).
        dq_rs_default = (
            (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _D128_DQ_ROWSPLIT
            and c.L == 1
        )
        self.attn_d128_dq_rowsplit_enabled = (
            self.attn_d128_wg_enabled
            and c.S % 128 == 0
            and (dq_rs_default if dq_rs_env is None else bool(int(dq_rs_env)))
        )
        fwd_mb_env = os.environ.get("MK_ATTN_D128_FWD_MB")
        fwd_mb_default = (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _D128_FWD_MBAR
        self.attn_d128_fwd_mbar_enabled = (
            self.attn_d128_wg_enabled
            and c.S % 128 == 0
            and (fwd_mb_default if fwd_mb_env is None else bool(int(fwd_mb_env)))
        )
        n256_stage3_env = os.environ.get("MK_WGMMA_N256_STAGE3")
        n256_stage3_default = (c.H, c.S, c.I, c.V, c.nq, c.nkv, c.D, c.L) in _D128_DQ_ROWSPLIT
        self.n256_stage3_enabled = (
            n256_stage3_default if n256_stage3_env is None else
            (bool(int(n256_stage3_env)) and n256_stage3_default)
        )
        n256_nmajor_env = os.environ.get("MK_WGMMA_N256_NMAJOR")
        n256_nmajor_default = n256_stage3_default
        self.n256_nmajor_enabled = (
            n256_nmajor_default if n256_nmajor_env is None else
            (bool(int(n256_nmajor_env)) and n256_nmajor_default)
        )
        if self.attn_d128_dq_rowsplit_enabled or self.n256_stage3_enabled:
            self._smem_bytes = 148 * 1024
        else:
            self._smem_bytes = 120 * 1024 if self.attn_d128_wg_enabled else None
        self.in_kernel_inv_valid = bool(int(os.environ.get("MK_INV_VALID_IN_KERNEL", "1")))
        self.bind_inputs = bool(int(os.environ.get("MK_BIND_INPUTS", "1")))
        self._inputs_bound_external = False
        self._build_program()

    # ------------------------------------------------------------------ #
    def _build_program(self):
        c, A, W = self.cfg, self.acts, self.ws
        QD = (c.nq + 2 * c.nkv) * c.D
        scale = mk.f2i(c.D**-0.5)
        eps = mk.f2i(c.eps)
        # wgmma attention ops (v3 P5): fwd 3.3x/5.6x, dkv 2.1x/3.3x, dq 2.2x/4.9x vs
        # the WMMA ops at nano/small (results/mkv3-p5-attnprobe.md). D=128 or ragged S
        # falls back to the WMMA path.
        wg_attn = c.D == 64 and c.S % 128 == 0
        p = mk.Program()
        p.default_cold_cap = _cold_cap(c)
        p.hot_embed_bwd_default = self.hot_embed_bwd_default
        p.hot_qwen_wgu_dw_default = self.hot_qwen_wgu_dw_default
        # Program-side arm of MK_GEMM_N256_TMA: mirrors load_ext's resolution
        # (ring required) so the compiled path and the injected tmap args agree.
        _ring_env = os.environ.get("MK_GEMM_MBAR_RING")
        _ring_on = (bool(int(_ring_env)) if _ring_env is not None
                    else self.gemm_mbar_ring_default)
        _tma_env = os.environ.get("MK_GEMM_N256_TMA")
        _tma_on = (bool(int(_tma_env)) if _tma_env is not None
                   else self.gemm_n256_tma_default)
        _nt_tma_env = os.environ.get("MK_GEMM_N256_NT_TMA")
        _nt_tma_on = (bool(int(_nt_tma_env)) if _nt_tma_env is not None
                      else self.gemm_n256_nt_tma_default)
        if _ring_on and (_tma_on or _nt_tma_on):
            p.gemm_n256_tma_ext = self.ext
        if _ring_on and _tma_on:
            p.gemm_n256_tma_enabled = True
            # TN dW rows: promoted for exact qwen on the R4 resweep evidence
            # (-294.6/-333.8us, 16/16 both orders; see mk.gemm_n256_tma_eligible).
            p.gemm_n256_tma_tn_default = self.gemm_n256_tma_default
        if _ring_on and _nt_tma_on:
            p.gemm_n256_nt_tma_enabled = True
        # Program-side arm of MK_GEMM_D64_TMA (same ring requirement).
        _d64tma_env = os.environ.get("MK_GEMM_D64_TMA")
        _d64tma_on = (bool(int(_d64tma_env)) if _d64tma_env is not None
                      else self.gemm_d64_tma_default)
        if _ring_on and _d64tma_on:
            p.gemm_d64_tma_ext = self.ext
        B = p.buf
        dw_no_atomic_env = os.environ.get("MK_DW_NO_ATOMIC_SK1")

        def dw_no_atomic_sk1_enabled(M, N, K, wgmma):
            if dw_no_atomic_env is None:
                short_default = (
                    c.H, c.L, c.nq, c.nkv, c.D, c.I, c.V, c.S
                ) in _GENERIC_SHORT_DW_NO_ATOMIC_SK1_CFGS
                return short_default or (
                    wgmma and (M, N, K) in _QWEN_L1_DW_NO_ATOMIC_SK1
                )
            return bool(int(dw_no_atomic_env))

        def dw_direct_store_overwrites(M, N, K):
            flags = 1 | 4 | 8
            if mk.wgmma_ok(M, N, K, flags):
                return dw_no_atomic_sk1_enabled(M, N, K, True) and mk.wgmma_split_k(M, N, K) == 1
            return dw_no_atomic_sk1_enabled(M, N, K, False) and mk.gemm_split_k(M, N, K) == 1

        def n256_stage3_flag(M, N, K):
            return mk.wgmma_n256_stage3_flag(M, N, K) if self.n256_stage3_enabled else 0

        def n256_nmajor_flag(M, N, K):
            return mk.wgmma_n256_nmajor_flag(M, N, K) if self.n256_nmajor_enabled else 0

        def gemm(a, b, out, M, N, K, flags, res=0, ssq=0, ssq_nparts=0):
            # ssq/ssq_nparts (bit13, v3 P4b r4): the wgmma epilogues emit per-64-col
            # sum-of-squares partials of the bf16-rounded output — free math on values
            # already in registers — so the consuming rmsnorm_fwd skips its variance
            # pass. Returns True when fused (the WMMA fallback has no such epilogue;
            # the rmsnorm call site must then use the classic path).
            if flags & 8 and flags & 4:  # fp32 accumulating dW: split-K for occupancy
                if mk.wgmma_ok(M, N, K, flags):
                    sk = mk.wgmma_split_k(M, N, K)
                    no_atomic_sk1 = dw_no_atomic_sk1_enabled(M, N, K, True)
                    if no_atomic_sk1 and sk == 1:
                        f = ((flags | 128) & ~(4 | 32))
                        if mk.wgmma_n256_dw_tn_ok(M, N, K, f):
                            stage3 = n256_stage3_flag(M, N, K)
                            nmajor = n256_nmajor_flag(M, N, K)
                            p.instr(
                                mk.OP_GEMM,
                                mk.gemm_tiles_wgmma_n256_direct(M, N),
                                [a, b, out, M, N, K, f | 16384 | stage3 | nmajor, res],
                            )
                            return False
                        p.instr(
                            mk.OP_GEMM,
                            mk.gemm_tiles_wgmma(M, N),
                            [a, b, out, M, N, K, f, res],
                        )
                        return False
                    p.instr(
                        mk.OP_GEMM,
                        mk.gemm_tiles_wgmma(M, N) * sk,
                        [a, b, out, M, N, K, ((flags | 32 | 128) & ~4), res, sk],
                    )
                else:
                    sk = mk.gemm_split_k(M, N, K)
                    no_atomic_sk1 = dw_no_atomic_sk1_enabled(M, N, K, False)
                    if no_atomic_sk1 and sk == 1:
                        p.instr(
                            mk.OP_GEMM,
                            mk.gemm_tiles(M, N),
                            [a, b, out, M, N, K, (flags & ~(4 | 32)), res],
                        )
                        return False
                    p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N) * sk, [a, b, out, M, N, K, (flags | 32) & ~4, res, sk])
                return False
            ssq_fuse_env = os.environ.get("MK_SSQ_FUSE")
            ssq_fuse_default = not (c.H == 256 and c.D == 64 and c.S in (2048, 3072))
            ssq_fuse = ssq_fuse_default if ssq_fuse_env is None else bool(int(ssq_fuse_env))
            do_ssq = ssq_nparts > 0 and ssq_fuse
            if mk.wgmma_n256_nn_bf16_ok(M, N, K, flags):
                stage3 = n256_stage3_flag(M, N, K)
                nmajor = n256_nmajor_flag(M, N, K)
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma_n256_direct(M, N),
                    [a, b, out, M, N, K, flags | 128 | 16384 | stage3 | nmajor, res],
                )
                return False
            if mk.wgmma_n256_nt_bf16_ok(M, N, K, flags):
                stage3 = n256_stage3_flag(M, N, K)
                nmajor = n256_nmajor_flag(M, N, K)
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma_n256_direct(M, N),
                    [a, b, out, M, N, K,
                     flags | 128 | 16384 | stage3 | nmajor | (8192 if do_ssq else 0),
                     res, 0, ssq, ssq_nparts] if do_ssq
                    else [a, b, out, M, N, K, flags | 128 | 16384 | stage3 | nmajor, res],
                )
                return do_ssq
            if mk.wgmma_n128_ok(M, N, K, flags):  # m64n128 NT tile (P4b r3)
                f = flags | 128 | 4096 | (8192 if do_ssq else 0)
                p.instr(mk.OP_GEMM, mk.gemm_tiles_wgmma_n128(M, N),
                        [a, b, out, M, N, K, f, res, 0, ssq, ssq_nparts] if do_ssq
                        else [a, b, out, M, N, K, f, res])
                return do_ssq
            if mk.wgmma_ok(M, N, K, flags):  # Hopper warpgroup path
                f = flags | 128 | (8192 if do_ssq else 0)
                p.instr(mk.OP_GEMM, mk.gemm_tiles_wgmma(M, N),
                        [a, b, out, M, N, K, f, res, 0, ssq, ssq_nparts] if do_ssq
                        else [a, b, out, M, N, K, f, res])
                return do_ssq
            p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N), [a, b, out, M, N, K, flags, res])
            return False

        def fill_zero(t):
            n = t.numel()
            p.instr(mk.OP_FILL_F32, mk.chunk_tiles(n), [B(t), n, mk.f2i(0.0)])

        # Split RMSNorm backward's on-path dx from its cold dw atomic drain. This adds
        # one sink instruction per norm but lets dX consumers run before dw finishes.
        split_rms_bwd = bool(int(os.environ.get("MK_RMS_BWD_SPLIT_DW", "1")))

        rms_dx_r4_env = os.environ.get("MK_RMS_DX_R4")
        if rms_dx_r4_env is None:
            # Post-dW/current-head retune: keep the long-H256 fold, but H512/S1024
            # small moved back to the normal two-row dx op.
            rms_dx_r4 = c.H == 256 and c.S == 2048
        else:
            rms_dx_r4 = bool(int(rms_dx_r4_env))

        rms_dx_fma_route_env = os.environ.get("MK_RMS_DX_FMA_ROUTE")
        if rms_dx_fma_route_env is None:
            rms_dx_fma_route = c.H == 256 and c.S == 128
        else:
            rms_dx_fma_route = bool(int(rms_dx_fma_route_env)) and c.H == 256 and c.S == 128
        rms_dx_h256_route_env = os.environ.get("MK_RMS_DX_H256")
        if rms_dx_h256_route_env is None:
            rms_dx_h256_route = c.H == 256 and c.S in _H256_RMS_DX_H256_S
        else:
            rms_dx_h256_route = bool(int(rms_dx_h256_route_env)) and c.H == 256
        swiglu_bwd_2w_env = os.environ.get("MK_SWIGLU_BWD_2W")
        if swiglu_bwd_2w_env is None:
            swiglu_bwd_2w = self.swiglu_bwd_2w_default
        else:
            swiglu_bwd_2w = bool(int(swiglu_bwd_2w_env))
        swiglu_bwd_4w_env = os.environ.get("MK_SWIGLU_BWD_4W")
        if swiglu_bwd_4w_env is None:
            swiglu_bwd_4w = self.swiglu_bwd_4w_default
        else:
            swiglu_bwd_4w = bool(int(swiglu_bwd_4w_env))
        swiglu_cache_sig = self.swiglu_cache_sig_enabled
        qkbwd_split_v_env = os.environ.get("MK_QKBWD_SPLIT_V")
        if qkbwd_split_v_env is None:
            qkbwd_split_v = c.H == 256 and c.D == 64 and c.S in _H256_D64_QKBWD_SPLIT_V_S
        else:
            qkbwd_split_v = bool(int(qkbwd_split_v_env))
        qkv_v_bwd_kv_slot_env = os.environ.get("MK_QKV_V_BWD_KV_SLOT")
        if qkv_v_bwd_kv_slot_env is None:
            qkv_v_bwd_kv_slot = (
                c.H == 256 and c.D == 64 and c.S in _H256_D64_QKV_V_BWD_KV_SLOT_S
            )
        else:
            qkv_v_bwd_kv_slot = bool(int(qkv_v_bwd_kv_slot_env))

        def head_dx_target_tiles():
            env = os.environ.get("MK_HEAD_DX_TARGET_TILES")
            if env is not None:
                return int(env)
            # per-shape retuned targets in _HEAD_DX_TARGET; 192 elsewhere
            return _HEAD_DX_TARGET.get(c.H, {}).get(c.S, 192)

        def rmsnorm_bwd(args):
            if split_rms_bwd:
                if rms_dx_h256_route:
                    p.instr(mk.OP_RMSNORM_BWD_DX_H256, mk.rowop_tiles(args[-1], mk.ROWOP_R2), args)
                elif rms_dx_r4:
                    p.instr(mk.OP_RMSNORM_BWD_DX_R4, mk.rowop_tiles(args[-1], mk.ROWOP_R4), args)
                elif rms_dx_fma_route:
                    p.instr(mk.OP_RMSNORM_BWD_DX_FMA, mk.rowop_tiles(args[-1], mk.ROWOP_R2), args)
                else:
                    p.instr(mk.OP_RMSNORM_BWD_DX, mk.rowop_tiles(args[-1], mk.ROWOP_R2), args)
                p.instr(mk.OP_RMSNORM_BWD_DW, mk.rowop_tiles(args[-1], mk.ROWOP_R2), args)
            else:
                p.instr(mk.OP_RMSNORM_BWD, mk.rowop_tiles(args[-1], mk.ROWOP_R2), args)

        def gemm_dx(a, b, out_bf, out_f32, M, N, K):
            """On-path NN dX gemm; parallelism-starved shapes (< 32 MN tiles) route via
            split-K fp32 atomics into a pre-zeroed workspace and the rowop consumer
            reads it directly (dy_f32, no CVT hop).

            History (v3 P6): under the single FIFO ready ring this was NEGATIVE at any
            shape/target (claim contention turned the span saving into consumer wait,
            +120/+460us step). The hot/cold criticality rings flipped it: gated re-run
            measured nano -26us (its dXN gemms are 16-tile), small +82us where the
            plain gemms already have >= 64 tiles — hence the tile gate."""
            if self.dx_split_k and mk.gemm_tiles(M, N) < 32:
                sk = mk.gemm_split_k(M, N, K, target_tiles=128)
                p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N) * sk, [a, b, out_f32(), M, N, K, 8 | 32, 0, sk])
                return out_f32(), 1
            gemm(a, b, out_bf, M, N, K, 0)  # via the helper: participates in routing flips
            return out_bf, 0

        X = [B(A["X"][l]) for l in range(c.L + 1)]
        X_fused = {}  # X[l] ssq partials available (down-gemm bit13 fused)
        n_qt = (c.S + 31) // 32
        tokens_buf = B(self.tokens)
        labels_buf = B(self.labels)
        self._tokens_buf = tokens_buf
        self._labels_buf = labels_buf
        prev_tokens_buf = B(self.prev_tokens)
        skip_direct_dw_fill = bool(int(os.environ.get("MK_DW_DIRECT_SKIP_FILL", "1")))
        sparse_embed_zero_env = os.environ.get("MK_EMB_SPARSE_ZERO")
        # L==1 original; L==2 promoted on the l2 support battery
        # (-219.39/-218.38us, 11/12 both orders,
        # mkv3-p4b-qwenl2-peel-support-20260707T0046Z.log).
        sparse_embed_zero_default = c.L in (1, 2) and c.H >= 1024 and c.V >= 32768
        sparse_embed_zero = (
            sparse_embed_zero_default
            if sparse_embed_zero_env is None
            else bool(int(sparse_embed_zero_env))
        )
        direct_store_grads = set()
        if skip_direct_dw_fill:
            if dw_direct_store_overwrites(c.V, c.H, c.S):
                direct_store_grads.add("wlm")
            for lz in range(c.L):
                if dw_direct_store_overwrites(QD, c.H, c.S):
                    direct_store_grads.add(f"wqkv.{lz}")
                if dw_direct_store_overwrites(c.H, c.nq * c.D, c.S):
                    direct_store_grads.add(f"wo.{lz}")
                if dw_direct_store_overwrites(2 * c.I, c.H, c.S):
                    direct_store_grads.add(f"wgu.{lz}")
                if dw_direct_store_overwrites(c.H, c.I, c.S):
                    direct_store_grads.add(f"wd.{lz}")

        # ---- wave 0: embedding gather + zero fp32 grad / loss / dX streams ----
        p.instr(mk.OP_EMBED_FWD, c.S, [tokens_buf, B(self.params["emb"]), X[0], c.H])
        if sparse_embed_zero:
            p.instr(
                mk.OP_EMBED_ZERO_ROWS,
                c.S,
                [prev_tokens_buf, tokens_buf, B(self.grads["emb"]), c.H],
            )
        if self.in_kernel_inv_valid:
            p.instr(mk.OP_INV_VALID, 1, [labels_buf, B(self.inv_valid), c.S])
        for name, g in self.grads.items():
            if name == "emb" and sparse_embed_zero:
                continue
            if name in direct_store_grads:
                continue
            fill_zero(g)
        fill_zero(self.loss)
        for lz in range(c.L):
            fill_zero(W[f"dQKV_f32.{lz}"])
            if self.drow_zero_fill_enabled:
                fill_zero(W[f"drow.{lz}"])
            for nm in (f"dXN2_f32.{lz}", f"dXN1_f32.{lz}", f"dHs_f32.{lz}"):
                if nm in W:
                    fill_zero(W[nm])
        # dX is bf16; zero via fp32 fill over half the elements (bf16 pair = one f32 zero)
        p.instr(mk.OP_FILL_F32, mk.chunk_tiles(W["dX"].numel() // 2), [B(W["dX"]), W["dX"].numel() // 2, mk.f2i(0.0)])
        p.wave()

        # ---- forward layers ----
        for l in range(c.L):
            pr = lambda n: B(self.params[f"{n}.{l}"])  # noqa: E731
            a = lambda n: B(A[f"{n}.{l}"])  # noqa: E731
            p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(c.S, mk.ROWOP_R2),
                    [X[l], pr("w1"), a("xn1"), a("rstd1"), c.H, eps, c.S]
                    + ([B(A[f"Xssq.{l}"]), c.H // 64] if X_fused.get(l) else []))
            p.wave()
            if c.D == 64 and mk.wgmma_ok(c.S, QD, c.H, 2):
                qkrope_n128_env = os.environ.get("MK_WGMMA_N128_QKROPE")
                qkrope_n128_default = c.H == 256 and c.S in _H256_D64_QKROPE_N128_S
                qkrope_n128 = qkrope_n128_default if qkrope_n128_env is None else bool(int(qkrope_n128_env))
                qkrope_flags = 2 | 128 | 256
                qkrope_tiles = mk.gemm_tiles_wgmma(c.S, QD)
                if qkrope_n128 and c.S % 128 == 0 and QD % 128 == 0 and c.H % 64 == 0:
                    qkrope_flags |= 4096
                    qkrope_tiles = mk.gemm_tiles_wgmma_n128(c.S, QD)
                # qk-norm + rope fused into the qkv gemm epilogue.
                p.instr(
                    mk.OP_GEMM,
                    qkrope_tiles,
                    [
                        a("xn1"),
                        pr("wqkv"),
                        a("qkvraw"),
                        c.S,
                        QD,
                        c.H,
                        qkrope_flags,
                        0,
                        0,
                        pr("qn"),
                        pr("kn"),
                        a("rq"),
                        a("rk"),
                        B(self.cos),
                        B(self.sin),
                        a("qkvr"),
                        c.nq,
                        c.nkv,
                        c.D,
                        eps,
                    ],
                )
                p.wave()
            else:
                gemm(a("xn1"), pr("wqkv"), a("qkvraw"), c.S, QD, c.H, 2)
                p.wave()
                p.instr(
                    mk.OP_QKNORM_ROPE_FWD,
                    c.S,
                    [
                        a("qkvraw"),
                        a("qkvr"),
                        pr("qn"),
                        pr("kn"),
                        a("rq"),
                        a("rk"),
                        B(self.cos),
                        B(self.sin),
                        c.nq,
                        c.nkv,
                        c.D,
                        eps,
                    ],
                )
                p.wave()
            if wg_attn and self.attn_fwd_bands is not None:
                # fwd split banding: straggler q-tiles run as C flash-decoding kv
                # chunks writing locally-normalized partials; a range-limited
                # combine merges each split band. C=1 bands keep the direct O/LSE
                # epilogue. Per-band slots on O/LSE/parts keep bands and combines
                # concurrent (row-disjoint); downstream readers register slot=None
                # and so conflict with every writer. Longest chunks emit first.
                for bi, (off, w, Cb) in enumerate(
                        sorted(self.attn_fwd_bands, key=lambda e: -e[2])):
                    if Cb == 1:
                        p.instr(
                            mk.OP_ATTN_FWD_WG,
                            c.nq * w,
                            [a("qkvr"),
                             p.buf(A[f"oatt.{l}"], slot=f"fo{bi}"),
                             p.buf(A[f"lse.{l}"], slot=f"fl{bi}"),
                             c.S, c.nq, c.nkv, c.D, scale, 1 | (off << 8)],
                        )
                    else:
                        p.instr(
                            mk.OP_ATTN_FWD_WG,
                            c.nq * w * Cb,
                            [a("qkvr"),
                             p.buf(A[f"oatt.{l}"], slot=f"fo{bi}"),
                             p.buf(A[f"lse.{l}"], slot=f"fl{bi}"),
                             c.S, c.nq, c.nkv, c.D, scale, Cb | (off << 8),
                             p.buf(W["fopart"], slot=f"fp{bi}"),
                             p.buf(W["fmpart"], slot=f"fp{bi}"),
                             p.buf(W["flpart"], slot=f"fp{bi}")],
                        )
                # combine rows are batched R=8 per tile (MK_ATTN_COMBINE_R=1 for
                # the old one-row tiles): at long S the one-row tiling made the
                # combine claim-overhead-bound (2-4k tiny tiles per instruction).
                comb_R = max(1, int(os.environ.get("MK_ATTN_COMBINE_R", "8")))
                for bi, (off, w, Cb) in enumerate(
                        sorted(self.attn_fwd_bands, key=lambda e: -e[2])):
                    if Cb > 1:
                        p.instr(
                            mk.OP_ATTN_COMBINE,
                            (w * 128 + comb_R - 1) // comb_R,
                            [p.buf(W["fopart"], slot=f"fp{bi}"),
                             p.buf(W["fmpart"], slot=f"fp{bi}"),
                             p.buf(W["flpart"], slot=f"fp{bi}"),
                             p.buf(A[f"oatt.{l}"], slot=f"foc{bi}"),
                             p.buf(A[f"lse.{l}"], slot=f"flc{bi}"),
                             c.S, c.nq, c.D, Cb, off * 128, comb_R],
                        )
                p.wave()
            elif wg_attn:
                p.instr(
                    mk.OP_ATTN_FWD_WG,
                    c.nq * (c.S // 128),
                    [a("qkvr"), a("oatt"), a("lse"), c.S, c.nq, c.nkv, c.D, scale],
                )
                p.wave()
            elif self.attn_d128_wg_enabled:
                # D=128 WGMMA fwd: 64-row q tiles, redundant-S both-WG softmax +
                # split-D output halves.
                fwd_tiles = c.nq * (c.S // (128 if self.attn_d128_fwd_mbar_enabled else 64))
                fwd_D = c.D | (_ATTN_D128_FWD_MB_FLAG if self.attn_d128_fwd_mbar_enabled else 0)
                p.instr(
                    mk.OP_ATTN_FWD_WG128,
                    fwd_tiles,
                    [a("qkvr"), a("oatt"), a("lse"), c.S, c.nq, c.nkv, fwd_D, scale],
                )
                p.wave()
            elif self.attn_C > 1:
                Ca = self.attn_C
                p.instr(
                    mk.OP_ATTN_FWD_SPLIT,
                    c.nq * n_qt * Ca,
                    [a("qkvr"), B(W["opart"]), B(W["mpart"]), B(W["lpart"]), c.S, c.nq, c.nkv, c.D, scale, Ca],
                )
                p.wave()
                p.instr(
                    mk.OP_ATTN_COMBINE,
                    c.S,
                    [B(W["opart"]), B(W["mpart"]), B(W["lpart"]), a("oatt"), a("lse"), c.S, c.nq, c.D, Ca],
                )
                p.wave()
            else:
                p.instr(mk.OP_ATTN_FWD, c.nq * n_qt, [a("qkvr"), a("oatt"), a("lse"), c.S, c.nq, c.nkv, c.D, scale])
                p.wave()
            x2_fused = gemm(a("oatt"), pr("wo"), a("x2"), c.S, c.H, c.nq * c.D, 2 | 16, X[l],
                            ssq=(B(A[f"x2ssq.{l}"]) if c.H % 64 == 0 else 0),
                            ssq_nparts=(c.H // 64 if c.H % 64 == 0 else 0))
            p.wave()
            p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(c.S, mk.ROWOP_R2),
                    [a("x2"), pr("w2"), a("xn2"), a("rstd2"), c.H, eps, c.S]
                    + ([B(A[f"x2ssq.{l}"]), c.H // 64] if x2_fused else []))
            p.wave()
            # Paired-column swiglu fusion (gate/up in one tile, two k-loops) was TRIED
            # AND REMOVED: halving the gu tiles doubles per-tile serial span (nano +88us,
            # small +297us) AND the pass-loop restructure blew the interpreter to
            # 255 regs -> 1 block/SM. Revisit only on top of tile-granular deps.
            gemm(a("xn2"), pr("wgu"), a("gu"), c.S, 2 * c.I, c.H, 2)
            p.wave()
            swiglu_fwd_args = [a("gu"), a("hs"), c.S, c.I]
            if swiglu_cache_sig:
                swiglu_fwd_args.append(a("swsig"))
            p.instr(mk.OP_SWIGLU_FWD, mk.rowop_tiles(c.S), swiglu_fwd_args)
            p.wave()
            X_fused[l + 1] = gemm(a("hs"), pr("wd"), X[l + 1], c.S, c.H, c.I, 2 | 16, a("x2"),
                                  ssq=(B(A[f"Xssq.{l + 1}"]) if c.H % 64 == 0 else 0),
                                  ssq_nparts=(c.H // 64 if c.H % 64 == 0 else 0))
            p.wave()

        # ---- head + loss ----
        p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(c.S, mk.ROWOP_R2),
                [X[c.L], B(self.params["wf"]), B(A["xnf"]), B(A["rstdf"]), c.H, eps, c.S]
                + ([B(A[f"Xssq.{c.L}"]), c.H // 64] if X_fused.get(c.L) else []))
        p.wave()
        # bit11 (lse partials in the lm_head epilogue): A/B measured NEUTRAL within
        # noise (on: 1865/9275, off: 1861/9317). Kept ON — cheapens the CE hop ~5x
        # for free and the partials become useful once tile-granular deps land.
        self.fuse_ce = True
        if self.fuse_ce and mk.wgmma_ok(c.S, c.V, c.H, 2):
            # lm_head gemm with per-row lse partials in the epilogue (bit11): CE fwd
            # reduces V/64 (max, sumexp) pairs instead of rescanning the V-wide row
            n256d = mk.wgmma_n256_direct_ok(c.S, c.V, c.H, 2 | 2048)
            n128 = (not n256d) and mk.wgmma_n128_ok(c.S, c.V, c.H, 2 | 2048)
            n256_stage3_bits = n256_stage3_flag(c.S, c.V, c.H) if n256d else 0
            n256_nmajor_bits = n256_nmajor_flag(c.S, c.V, c.H) if n256d else 0
            p.instr(
                mk.OP_GEMM,
                (mk.gemm_tiles_wgmma_n256_direct(c.S, c.V) if n256d else
                 mk.gemm_tiles_wgmma_n128(c.S, c.V) if n128 else mk.gemm_tiles_wgmma(c.S, c.V)),
                [B(A["xnf"]), B(self.params["wlm"]), B(A["logits"]), c.S, c.V, c.H,
                 2 | 128 | 2048 | n256_stage3_bits | n256_nmajor_bits |
                 (16384 if n256d else 4096 if n128 else 0), 0, 0,
                 B(W["lse_parts"]), c.V // 64],
            )
            p.wave()
            p.instr(
                mk.OP_CE_FWD,
                c.S,
                [B(A["logits"]), labels_buf, B(A["lse_ce"]), B(self.loss), B(self.inv_valid), c.V,
                 B(W["lse_parts"]), c.V // 64],
            )
        else:
            gemm(B(A["xnf"]), B(self.params["wlm"]), B(A["logits"]), c.S, c.V, c.H, 2)
            p.wave()
            p.instr(
                mk.OP_CE_FWD, c.S, [B(A["logits"]), labels_buf, B(A["lse_ce"]), B(self.loss), B(self.inv_valid), c.V]
            )
        p.wave()

        # ---- backward ----
        p.instr(mk.OP_CE_BWD, c.S, [B(A["logits"]), labels_buf, B(A["lse_ce"]), B(self.inv_valid), c.V])
        p.wave()
        # dXN = dlogits @ Wlm has K=V (huge) but few output tiles: split-K into the fp32
        # workspace (atomic accumulate), then convert. H256/S2048 currently chooses
        # sk=1, where a normal fp32-output WGMMA avoids one zero-fill plus atomics.
        head_dx_no_atomic_sk1_env = os.environ.get("MK_HEAD_DX_NO_ATOMIC_SK1")
        head_dx_n128_f32_env = os.environ.get("MK_HEAD_DX_N128_F32")
        head_dx_n128_split_env = os.environ.get("MK_HEAD_DX_N128_SPLIT")
        if head_dx_n128_f32_env is None:
            head_dx_n128_f32 = (
                (c.H == 512 and c.S == 1024 and c.V % 64 == 0)
                or (c.H, c.S, c.V, c.nq, c.nkv, c.D, c.L) in _QWEN_L1_HEAD_DX_N128_F32
            )
        else:
            head_dx_n128_f32 = bool(int(head_dx_n128_f32_env))
        if head_dx_no_atomic_sk1_env is None:
            head_dx_no_atomic_sk1 = (c.H == 256 and c.S >= 2048) or head_dx_n128_f32
        else:
            head_dx_no_atomic_sk1 = bool(int(head_dx_no_atomic_sk1_env))
        if head_dx_n128_split_env is None:
            head_dx_n128_split = c.H == 256 and c.S == 512 and c.V % 64 == 0
        else:
            head_dx_n128_split = bool(int(head_dx_n128_split_env))
        if mk.wgmma_ok(c.S, c.H, c.V, 0):
            sk_head = mk.wgmma_split_k(
                c.S, c.H, c.V, target_tiles=head_dx_target_tiles()
            )
            head_dx_tiles = mk.gemm_tiles_wgmma(c.S, c.H)
            head_dx_flags = 8 | 128
            head_dx_args = [B(A["logits"]), B(self.params["wlm"]), B(W["dXN_f32"]), c.S, c.H, c.V]
        else:
            sk_head = mk.gemm_split_k(
                c.S, c.H, c.V, target_tiles=head_dx_target_tiles()
            )
            head_dx_tiles = mk.gemm_tiles(c.S, c.H)
            head_dx_flags = 8
            head_dx_args = [B(A["logits"]), B(self.params["wlm"]), B(W["dXN_f32"]), c.S, c.H, c.V]
        if self.head_dx_skr > 1 and (head_dx_flags & 128):
            # Round-12 SKR (splitK + separate reduce): K-sliced n128 tiles write
            # per-slice fp32 partial slabs (plain stores, no zero-fill, no atomics);
            # a grid-stride reduce hop sums the slices into dXN_f32 for the existing
            # dy_f32 consumer. Gate: _HEAD_DX_SKR / MK_HEAD_DX_SKR.
            skr = min(self.head_dx_skr, c.V // 64)
            kchunk = ((c.V + skr * 64 - 1) // (skr * 64)) * 64
            active = (c.V + kchunk - 1) // kchunk  # trailing slices can be empty
            p.instr(
                mk.OP_GEMM,
                mk.gemm_tiles_wgmma_n128(c.S, c.H) * skr,
                [B(A["logits"]), B(self.params["wlm"]), B(W["headdx_skr"]), c.S, c.H,
                 c.V, head_dx_flags | 32 | 4096 | mk.GEMM_SKR_FLAG, 0, skr],
            )
            p.wave()  # wave-mode barrier; df derives the slab dep automatically
            p.instr(
                mk.OP_SKR_REDUCE,
                (c.S * c.H + 4095) // 4096,
                [B(W["headdx_skr"]), B(W["dXN_f32"]), c.S * c.H, active],
            )
        elif head_dx_no_atomic_sk1 and sk_head == 1:
            if mk.wgmma_n256_head_dx_ok(c.S, c.H, c.V, head_dx_flags):
                stage3 = n256_stage3_flag(c.S, c.H, c.V)
                nmajor = n256_nmajor_flag(c.S, c.H, c.V)
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma_n256_direct(c.S, c.H),
                    head_dx_args + [head_dx_flags | 16384 | stage3 | nmajor, 0],
                )
            elif head_dx_n128_f32 and c.S % 128 == 0 and c.H % 128 == 0 and c.V % 64 == 0:
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma_n128(c.S, c.H),
                    head_dx_args + [head_dx_flags | 4096, 0],
                )
            else:
                p.instr(
                    mk.OP_GEMM,
                    head_dx_tiles,
                    head_dx_args + [head_dx_flags, 0],
                )
        else:
            fill_zero(W["dXN_f32"])
            p.wave()
            if (
                head_dx_n128_split
                and (head_dx_flags & 128)
                and c.S % 128 == 0
                and c.H % 128 == 0
                and c.V % 64 == 0
            ):
                n128_tiles = mk.gemm_tiles_wgmma_n128(c.S, c.H)
                n128_target = int(
                    os.environ.get(
                        "MK_HEAD_DX_N128_SPLIT_TARGET",
                        48 if c.H == 256 and c.S == 512 else head_dx_target_tiles(),
                    )
                )
                sk_n128 = max(1, min(n128_target // max(n128_tiles, 1), c.V // 64))
                p.instr(
                    mk.OP_GEMM,
                    n128_tiles * sk_n128,
                    head_dx_args + [head_dx_flags | 32 | 4096, 0, sk_n128],
                )
            else:
                p.instr(
                    mk.OP_GEMM,
                    head_dx_tiles * sk_head,
                    head_dx_args + [head_dx_flags | 32, 0, sk_head],
                )
        gemm(B(A["logits"]), B(A["xnf"]), B(self.grads["wlm"]), c.V, c.H, c.S, 1 | 4 | 8)
        p.wave()
        # final-norm bwd reads the split-K fp32 workspace directly (dy_f32; no CVT hop)
        rmsnorm_bwd([X[c.L], B(self.params["wf"]), B(W["dXN_f32"]), B(W["dX"]), B(self.grads["wf"]), B(A["rstdf"]), c.H, 1, c.S])
        p.wave()

        for l in reversed(range(c.L)):
            pr = lambda n: B(self.params[f"{n}.{l}"])  # noqa: E731
            gr = lambda n: B(self.grads[f"{n}.{l}"])  # noqa: E731
            a = lambda n: B(A[f"{n}.{l}"])  # noqa: E731
            # down proj: dHs = dX @ Wd ; dWd += dX^T Hs
            dhs, dhs_f32 = gemm_dx(B(W["dX"]), pr("wd"), B(W["dHs"]), lambda: B(W[f"dHs_f32.{l}"]), c.S, c.I, c.H)
            gemm(B(W["dX"]), a("hs"), gr("wd"), c.H, c.I, c.S, 1 | 4 | 8)
            p.wave()
            if swiglu_bwd_4w:
                swiglu_bwd_args = [a("gu"), dhs, B(W["dGU"]), c.S, c.I, dhs_f32]
                if swiglu_cache_sig:
                    swiglu_bwd_args.append(a("swsig"))
                p.instr(
                    mk.OP_SWIGLU_BWD_4W,
                    mk.rowop_tiles(c.S, mk.SWIGLU_BWD_4W_R),
                    swiglu_bwd_args,
                )
            elif swiglu_bwd_2w:
                swiglu_bwd_args = [a("gu"), dhs, B(W["dGU"]), c.S, c.I, dhs_f32]
                if swiglu_cache_sig:
                    swiglu_bwd_args.append(a("swsig"))
                p.instr(
                    mk.OP_SWIGLU_BWD_2W,
                    mk.rowop_tiles(c.S, mk.SWIGLU_BWD_2W_R),
                    swiglu_bwd_args,
                )
            else:
                swiglu_bwd_args = [a("gu"), dhs, B(W["dGU"]), c.S, c.I, dhs_f32]
                if swiglu_cache_sig:
                    swiglu_bwd_args.append(a("swsig"))
                p.instr(mk.OP_SWIGLU_BWD, mk.rowop_tiles(c.S), swiglu_bwd_args)
            p.wave()
            dxn, dxn_f32 = gemm_dx(B(W["dGU"]), pr("wgu"), B(W["dXN"]), lambda: B(W[f"dXN2_f32.{l}"]), c.S, c.H, 2 * c.I)
            gemm(B(W["dGU"]), a("xn2"), gr("wgu"), 2 * c.I, c.H, c.S, 1 | 4 | 8)
            p.wave()
            rmsnorm_bwd([a("x2"), pr("w2"), dxn, B(W["dX"]), gr("w2"), a("rstd2"), c.H, dxn_f32, c.S])
            p.wave()
            # o proj: dOatt = dX @ Wo with the Drow reduction fused into the epilogue
            # (flags bit10; replaces OP_ATTN_DPRE — one chain hop less per layer);
            # dWo += dX^T Oatt
            drow_flags = 1024
            # Historical env name kept for compatibility: unset now means "route Drow
            # through WGMMA whenever the GEMM gate accepts the shape"; =0 restores WMMA.
            drow_wg_default = "1"
            drow_wg = bool(int(os.environ.get("MK_DROW_WG_LONGONLY", drow_wg_default)))
            if drow_wg and mk.wgmma_n256_nn_bf16_drow_ok(c.S, c.nq * c.D, c.H, drow_flags):
                stage3 = n256_stage3_flag(c.S, c.nq * c.D, c.H)
                nmajor = n256_nmajor_flag(c.S, c.nq * c.D, c.H)
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma_n256_direct(c.S, c.nq * c.D),
                    [B(W["dX"]), pr("wo"), B(W["dOatt"]), c.S, c.nq * c.D, c.H,
                     drow_flags | 128 | 16384 | stage3 | nmajor,
                     0, 0, a("oatt"), B(W[f"drow.{l}"]), c.D],
                )
            elif drow_wg and mk.wgmma_ok(c.S, c.nq * c.D, c.H, drow_flags):
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma(c.S, c.nq * c.D),
                    [B(W["dX"]), pr("wo"), B(W["dOatt"]), c.S, c.nq * c.D, c.H,
                     drow_flags | 128, 0, 0, a("oatt"), B(W[f"drow.{l}"]), c.D],
                )
            else:
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles(c.S, c.nq * c.D),
                    [B(W["dX"]), pr("wo"), B(W["dOatt"]), c.S, c.nq * c.D, c.H,
                     drow_flags, 0, 0, a("oatt"), B(W[f"drow.{l}"]), c.D],
                )
            gemm(B(W["dX"]), a("oatt"), gr("wo"), c.H, c.nq * c.D, c.S, 1 | 4 | 8)
            p.wave()
            # attention bwd: dkv splits the GQA loop (one group member per tile), dq
            # chunks its kv loop, then one convert drains the fp32 workspace to bf16.
            # DQ's C=1 kernel path has one writer per q slice and stores directly;
            # DQ C>1 and DKV use atomics. Alias slots keep disjoint q/kv writes
            # parallel in the dependency analysis.
            G = c.nq // c.nkv
            dkv_args = lambda: [  # noqa: E731
                a("qkvr"),
                B(W["dOatt"]),
                a("lse"),
                B(W[f"drow.{l}"]),
                p.buf(W[f"dQKV_f32.{l}"], slot="kv"),
                c.S,
                c.nq,
                c.nkv,
                c.D,
                scale,
            ]
            dq_args = lambda: [  # noqa: E731
                a("qkvr"),
                B(W["dOatt"]),
                a("lse"),
                B(W[f"drow.{l}"]),
                p.buf(W[f"dQKV_f32.{l}"], slot="q"),
                c.S,
                c.nq,
                c.nkv,
                c.D,
                scale,
            ]
            if wg_attn:
                # P4b retune after SW128/NN routing: dQ usually wants one kv chunk.
                # S2048/H256 is the exception after the cold-dW retune: paired dQ+dKV
                # C=2 trims the attention-bwd critical path, while small/S4096 regress.
                # S1024/H256 later joined the same dQ C=2 bucket after the head-dX retune.
                # H256/S512 wants dQ chunking, but DKV moved back to C=2 after the
                # fast-log/current-head resweep; C=3 over-splits the DKV path now.
                # dKV otherwise wants C=1 once nq * (S/128) already exposes >=64 chunks,
                # else C=2 keeps enough tail parallelism (nano/S1024-H256).
                n_qt128 = c.S // 128
                if c.H == 256 and c.S in _H256_ATTN_CHUNKS:
                    default_Ckv, default_Cq = _H256_ATTN_CHUNKS[c.S]
                else:
                    # dKV wants C=1 once nq * (S/128) already exposes >=64 chunks,
                    # else C=2 keeps enough tail parallelism; dQ wants one kv chunk.
                    default_Ckv = 1 if c.nq * n_qt128 >= 64 else 2
                    default_Cq = 1
                Ckv = max(1, int(os.environ.get("MK_ATTN_DKV_C", str(default_Ckv))))
                Cq = max(1, int(os.environ.get("MK_ATTN_DQ_C", str(default_Cq))))
                # Banded chunking (MK_ATTN_BAND = target stages per chunk, 0 = off):
                # at C=1 long-S the bwd ops are STRAGGLER-BOUND — makespan equals the
                # longest causal tile's serial stage chain (measured 2.7us/stage dkv,
                # 1.9us/stage dq), while uniform C>1 pays fill/atomic overhead on ALL
                # tiles (the measured S4096 no-go). Bands split only the long tiles:
                # per-tile chunks = ceil(stages/T), consecutive equal-C tiles grouped
                # into one instruction with its kv/q-tile offset+width packed into the
                # C arg (C | off<<8 | width<<16). Per-band ws slots keep the bands'
                # disjoint row ranges parallel in the dependency analysis.
                # Measured defaults (in-model paired A/B, both construction orders,
                # 40/40 or 16/16 wins each): S2048 retuned to T=12 after idle32
                # composition (-30/-37us vs T=16), T=32 degenerates to uniform C=1
                # and loses +41/+48,
                # S3072 T=16 (-37/-42us), S4096 retuned to T=29 after fwd-band
                # composition (-60us vs T=32), S8192 retuned to T=40 after idle32/
                # cached-SwiGLU composition (-88/-110us vs T=32). Standalone +
                # in-model logs in the attn-band worktree and shared results/.
                default_band_T = (
                    _ATTN_BWD_BAND_T.get(c.S, 0) if c.H == 256 and c.D == 64 else 0
                )
                band_T = int(os.environ.get("MK_ATTN_BAND", str(default_band_T)))
                if band_T > 0:
                    # Post-pdf S8192 recheck flipped back to LPT; keep dq-first
                    # only for shapes that re-earn it in both construction orders.
                    default_band_order = (
                        "dq_first"
                        if c.H == 256 and c.D == 64 and c.S in _ATTN_BAND_DQ_FIRST_S
                        else "lpt"
                    )
                    band_order = os.environ.get("MK_ATTN_BAND_ORDER", default_band_order)

                    def bands(stages_of):
                        out, j = [], 0
                        while j < n_qt128:
                            Cj = max(1, -(-stages_of(j) // band_T))
                            k = j
                            while k < n_qt128 and max(1, -(-stages_of(k) // band_T)) == Cj:
                                k += 1
                            out.append((j, k - j, Cj))
                            j = k
                        return out
                    # (chunk_stages, seq, kind, off, width, chunks, op, ntiles, args)
                    # LPT order matches the pre-order-probe promoted route exactly. The
                    # dq_first probe tests whether the post-band DQ wait path is
                    # sensitive to same-wave emission order.
                    emit = []
                    for bi, (off, w, Cb) in enumerate(bands(lambda j: (c.S - j * 128) // 64)):
                        st = -(-((c.S - off * 128) // 64) // Cb)
                        emit.append((st, len(emit), "dkv", off, w, Cb, mk.OP_ATTN_DKV_WG, c.nkv * w * G * Cb,
                                     dkv_args()[:4]
                                     + [p.buf(W[f"dQKV_f32.{l}"], slot=f"kv{bi}")]
                                     + dkv_args()[5:] + [Cb | (off << 8) | (w << 16)]))
                    for bi, (off, w, Cb) in enumerate(bands(lambda i: i * 2 + 2)):
                        st = -(-((off + w - 1) * 2 + 2) // Cb)
                        emit.append((st, len(emit), "dq", off, w, Cb, mk.OP_ATTN_DQ_WG, c.nq * w * Cb,
                                     dq_args()[:4]
                                     + [p.buf(W[f"dQKV_f32.{l}"], slot=f"q{bi}")]
                                     + dq_args()[5:] + [Cb | (off << 8) | (w << 16)]))
                    if band_order == "dq_first":
                        ordered_emit = sorted(
                            emit,
                            key=lambda e: (0 if e[2] == "dq" else 1, -e[0], -e[5], -e[3], e[1]),
                        )
                    elif band_order == "lpt":
                        ordered_emit = sorted(emit, key=lambda e: (-e[0], e[1]))
                    else:
                        raise ValueError(f"unknown MK_ATTN_BAND_ORDER={band_order!r}")
                    for _, _, _, _, _, _, op_id, nt, ag in ordered_emit:
                        p.instr(op_id, nt, ag)
                else:
                    p.instr(mk.OP_ATTN_DKV_WG, c.nkv * n_qt128 * G * Ckv, dkv_args() + [Ckv])
                    p.instr(mk.OP_ATTN_DQ_WG, c.nq * n_qt128 * Cq, dq_args() + [Cq])
            elif self.attn_d128_wg_enabled:
                # D=128 WGMMA bwd: 64-row kv/q tiles, redundant-S + split-D-half
                # accumulators, C=1.
                n_t64 = c.S // 64
                p.instr(mk.OP_ATTN_DKV_WG128, c.nkv * n_t64 * G, dkv_args() + [1])
                if self.attn_d128_dq_rowsplit_enabled:
                    p.instr(
                        mk.OP_ATTN_DQ_WG128,
                        c.nq * (c.S // 128),
                        dq_args() + [1 | _ATTN_D128_DQ_RS_FLAG],
                    )
                else:
                    p.instr(mk.OP_ATTN_DQ_WG128, c.nq * n_t64, dq_args() + [1])
            else:
                generic_dq_c_env = os.environ.get("MK_ATTN_DQ_C")
                if generic_dq_c_env is None:
                    if c.D == 128 and (c.H, c.S, c.nq, c.nkv) in _D128_GENERIC_DQ_C1:
                        Cq = 1
                    else:
                        Cq = 4 if n_qt >= 8 else 2
                else:
                    Cq = max(1, int(generic_dq_c_env))
                p.instr(mk.OP_ATTN_DKV, c.nkv * n_qt * G, dkv_args())
                p.instr(mk.OP_ATTN_DQ, c.nq * n_qt * Cq, dq_args() + [Cq])
            p.wave()
            # qk-norm+rope bwd reads the attention-bwd fp32 atomic workspace DIRECTLY
            # (dy_f32) — the former per-layer CVT chain hop is gone (v3 P1).
            qkvraw_bwd = p.buf(W["dQKVraw"], slot="qk") if qkbwd_split_v else B(W["dQKVraw"])
            if qkbwd_split_v:
                qkv_v_bwd_src = (
                    p.buf(W[f"dQKV_f32.{l}"], slot="kv*")
                    if qkv_v_bwd_kv_slot
                    else B(W[f"dQKV_f32.{l}"])
                )
                p.instr(
                    mk.OP_QKV_V_BWD,
                    mk.rowop_tiles(c.S),
                    [
                        qkv_v_bwd_src,
                        p.buf(W["dQKVraw"], slot="v"),
                        c.nq,
                        c.nkv,
                        c.D,
                        c.S,
                    ],
                )
            p.instr(
                mk.OP_QKNORM_ROPE_BWD,
                mk.rowop_tiles(c.S),
                [
                    a("qkvraw"),
                    B(W[f"dQKV_f32.{l}"]),
                    qkvraw_bwd,
                    pr("qn"),
                    pr("kn"),
                    gr("qn"),
                    gr("kn"),
                    a("rq"),
                    a("rk"),
                    B(self.cos),
                    B(self.sin),
                    c.nq,
                    c.nkv,
                    c.D,
                    1,  # dy_f32
                    c.S,
                    int(qkbwd_split_v),
                ],
            )
            p.wave()
            dxn1, dxn1_f32 = gemm_dx(B(W["dQKVraw"]), pr("wqkv"), B(W["dXN"]), lambda: B(W[f"dXN1_f32.{l}"]), c.S, c.H, QD)
            gemm(B(W["dQKVraw"]), a("xn1"), gr("wqkv"), QD, c.H, c.S, 1 | 4 | 8)
            p.wave()
            rmsnorm_bwd([X[l], pr("w1"), dxn1, B(W["dX"]), gr("w1"), a("rstd1"), c.H, dxn1_f32, c.S])
            p.wave()

        p.instr(mk.OP_EMBED_BWD, c.S, [B(self.tokens), B(W["dX"]), B(self.grads["emb"]), c.H])
        if sparse_embed_zero:
            p.instr(mk.OP_COPY_I32, mk.chunk_tiles(c.S), [tokens_buf, prev_tokens_buf, c.S])
        p.wave()

        self.prog = p.finalize(self.dev)
        self.n_waves = len(p.waves)

    # ------------------------------------------------------------------ #
    def step(self, tokens: torch.Tensor, labels: torch.Tensor, mode=None) -> torch.Tensor:
        """One fused fwd+bwd. Returns the (device) loss scalar; grads are in self.grads.

        mode=None resolves to the shape's default executor (_PDF_MODE routing);
        pass an explicit mode ("df"/"pdf"/"ws"/...) to force one.
        """
        if mode is None:
            mode = self.default_mode
        bind_inputs = (
            self.bind_inputs
            and mode in ("df", "pdf")  # pdf shares df's state layout and bind path
            and tokens.is_cuda
            and labels.is_cuda
            and tokens.device == self.tokens.device
            and labels.device == self.labels.device
            and tokens.is_contiguous()
            and labels.is_contiguous()
            and tokens.dtype == torch.int32
            and labels.dtype == torch.int32
            and tuple(tokens.shape) == (self.cfg.S,)
            and tuple(labels.shape) == (self.cfg.S,)
        )
        if bind_inputs:
            if not self.in_kernel_inv_valid:
                self.inv_valid.copy_(1.0 / (labels >= 0).sum().clamp(min=1).float().reshape(1))
            self.prog.run(
                self.ext,
                smem_bytes=self._smem_bytes,
                mode=mode,
                bind_bufs=((self._tokens_buf, tokens), (self._labels_buf, labels)),
            )
            stream = torch.cuda.current_stream(tokens.device)
            tokens.record_stream(stream)
            labels.record_stream(stream)
            self._inputs_bound_external = True
        else:
            # Alternate executors and non-canonical inputs use the original internal
            # buffers. Restore their buftab entries in case a prior df bind changed them.
            if self._inputs_bound_external:
                self.prog._buftab[self._tokens_buf] = self.tokens.data_ptr()
                self.prog._buftab[self._labels_buf] = self.labels.data_ptr()
                self._inputs_bound_external = False
            self.tokens.copy_(tokens)
            self.labels.copy_(labels)
            if not self.in_kernel_inv_valid:
                self.inv_valid.copy_(1.0 / (labels >= 0).sum().clamp(min=1).float().reshape(1))
            self.prog.run(self.ext, smem_bytes=self._smem_bytes, mode=mode)
        return self.loss
