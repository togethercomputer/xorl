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


class MKQwen3:
    def __init__(self, cfg: Cfg, dev="cuda", seed=0):
        self.cfg = cfg
        self.dev = dev
        torch.manual_seed(seed)
        c = cfg
        QD = (c.nq + 2 * c.nkv) * c.D
        bf, f32 = torch.bfloat16, torch.float32

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
        A["rstdf"] = torch.empty(c.S, device=dev, dtype=f32)
        A["xnf"] = torch.empty(c.S, c.H, device=dev, dtype=bf)
        A["logits"] = torch.empty(c.S, c.V, device=dev, dtype=bf)
        A["lse_ce"] = torch.empty(c.S, device=dev, dtype=f32)
        self.acts = A

        # ---- io + backward workspace ----
        self.tokens = torch.zeros(c.S, device=dev, dtype=torch.int32)
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
        self.ws = W

        self.cos, self.sin = rope_tables(c, dev)
        self.ext = mk.load_ext()
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
        p.default_cold_cap = 0 if c.S >= 2048 else (33 if c.S >= 1024 else 16)
        B = p.buf

        def gemm(a, b, out, M, N, K, flags, res=0):
            if flags & 8 and flags & 4:  # fp32 accumulating dW: split-K for occupancy
                if mk.wgmma_ok(M, N, K, flags):
                    sk = mk.wgmma_split_k(M, N, K)
                    p.instr(
                        mk.OP_GEMM,
                        mk.gemm_tiles_wgmma(M, N) * sk,
                        [a, b, out, M, N, K, ((flags | 32 | 128) & ~4), res, sk],
                    )
                else:
                    sk = mk.gemm_split_k(M, N, K)
                    p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N) * sk, [a, b, out, M, N, K, (flags | 32) & ~4, res, sk])
            elif mk.wgmma_n128_ok(M, N, K, flags):  # m64n128 NT tile (P4b r3)
                p.instr(mk.OP_GEMM, mk.gemm_tiles_wgmma_n128(M, N), [a, b, out, M, N, K, flags | 128 | 4096, res])
            elif mk.wgmma_ok(M, N, K, flags):  # Hopper warpgroup path
                p.instr(mk.OP_GEMM, mk.gemm_tiles_wgmma(M, N), [a, b, out, M, N, K, flags | 128, res])
            else:
                p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N), [a, b, out, M, N, K, flags, res])

        def fill_zero(t):
            n = t.numel()
            p.instr(mk.OP_FILL_F32, mk.chunk_tiles(n), [B(t), n, mk.f2i(0.0)])

        # Split RMSNorm backward's on-path dx from its cold dw atomic drain. This adds
        # one sink instruction per norm but lets dX consumers run before dw finishes.
        split_rms_bwd = bool(int(os.environ.get("MK_RMS_BWD_SPLIT_DW", "1")))

        rms_dx_r4_env = os.environ.get("MK_RMS_DX_R4")
        if rms_dx_r4_env is None:
            # A/B: S2048 wins repeatably; S1024/S3072/S4096 are neutral and H512 regresses.
            rms_dx_r4 = c.H == 256 and c.S == 2048
        else:
            rms_dx_r4 = bool(int(rms_dx_r4_env))

        def rmsnorm_bwd(args):
            if split_rms_bwd:
                if rms_dx_r4:
                    p.instr(mk.OP_RMSNORM_BWD_DX_R4, mk.rowop_tiles(args[-1], mk.ROWOP_R4), args)
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
        n_qt = (c.S + 31) // 32

        # ---- wave 0: embedding gather + zero every fp32 grad / loss / dX stream ----
        p.instr(mk.OP_EMBED_FWD, c.S, [B(self.tokens), B(self.params["emb"]), X[0], c.H])
        for g in self.grads.values():
            fill_zero(g)
        fill_zero(self.loss)
        for lz in range(c.L):
            fill_zero(W[f"dQKV_f32.{lz}"])
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
            p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(c.S, mk.ROWOP_R2), [X[l], pr("w1"), a("xn1"), a("rstd1"), c.H, eps, c.S])
            p.wave()
            if c.D == 64 and mk.wgmma_ok(c.S, QD, c.H, 2):
                # qk-norm + rope fused into the qkv gemm epilogue (one head per tile)
                p.instr(
                    mk.OP_GEMM,
                    mk.gemm_tiles_wgmma(c.S, QD),
                    [
                        a("xn1"),
                        pr("wqkv"),
                        a("qkvraw"),
                        c.S,
                        QD,
                        c.H,
                        2 | 128 | 256,
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
            if wg_attn:
                p.instr(
                    mk.OP_ATTN_FWD_WG,
                    c.nq * (c.S // 128),
                    [a("qkvr"), a("oatt"), a("lse"), c.S, c.nq, c.nkv, c.D, scale],
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
            gemm(a("oatt"), pr("wo"), a("x2"), c.S, c.H, c.nq * c.D, 2 | 16, X[l])
            p.wave()
            p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(c.S, mk.ROWOP_R2), [a("x2"), pr("w2"), a("xn2"), a("rstd2"), c.H, eps, c.S])
            p.wave()
            # Paired-column swiglu fusion (gate/up in one tile, two k-loops) was TRIED
            # AND REMOVED: halving the gu tiles doubles per-tile serial span (nano +88us,
            # small +297us) AND the pass-loop restructure blew the interpreter to
            # 255 regs -> 1 block/SM. Revisit only on top of tile-granular deps.
            gemm(a("xn2"), pr("wgu"), a("gu"), c.S, 2 * c.I, c.H, 2)
            p.wave()
            p.instr(mk.OP_SWIGLU_FWD, mk.rowop_tiles(c.S), [a("gu"), a("hs"), c.S, c.I])
            p.wave()
            gemm(a("hs"), pr("wd"), X[l + 1], c.S, c.H, c.I, 2 | 16, a("x2"))
            p.wave()

        # ---- head + loss ----
        p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(c.S, mk.ROWOP_R2), [X[c.L], B(self.params["wf"]), B(A["xnf"]), B(A["rstdf"]), c.H, eps, c.S])
        p.wave()
        # bit11 (lse partials in the lm_head epilogue): A/B measured NEUTRAL within
        # noise (on: 1865/9275, off: 1861/9317). Kept ON — cheapens the CE hop ~5x
        # for free and the partials become useful once tile-granular deps land.
        self.fuse_ce = True
        if self.fuse_ce and mk.wgmma_ok(c.S, c.V, c.H, 2):
            # lm_head gemm with per-row lse partials in the epilogue (bit11): CE fwd
            # reduces V/64 (max, sumexp) pairs instead of rescanning the V-wide row
            n128 = mk.wgmma_n128_ok(c.S, c.V, c.H, 2 | 2048)
            p.instr(
                mk.OP_GEMM,
                mk.gemm_tiles_wgmma_n128(c.S, c.V) if n128 else mk.gemm_tiles_wgmma(c.S, c.V),
                [B(A["xnf"]), B(self.params["wlm"]), B(A["logits"]), c.S, c.V, c.H,
                 2 | 128 | 2048 | (4096 if n128 else 0), 0, 0, B(W["lse_parts"]), c.V // 64],
            )
            p.wave()
            p.instr(
                mk.OP_CE_FWD,
                c.S,
                [B(A["logits"]), B(self.labels), B(A["lse_ce"]), B(self.loss), B(self.inv_valid), c.V,
                 B(W["lse_parts"]), c.V // 64],
            )
        else:
            gemm(B(A["xnf"]), B(self.params["wlm"]), B(A["logits"]), c.S, c.V, c.H, 2)
            p.wave()
            p.instr(
                mk.OP_CE_FWD, c.S, [B(A["logits"]), B(self.labels), B(A["lse_ce"]), B(self.loss), B(self.inv_valid), c.V]
            )
        p.wave()

        # ---- backward ----
        p.instr(mk.OP_CE_BWD, c.S, [B(A["logits"]), B(self.labels), B(A["lse_ce"]), B(self.inv_valid), c.V])
        p.wave()
        # dXN = dlogits @ Wlm has K=V (huge) but few output tiles: split-K into the fp32
        # workspace (atomic accumulate), then convert. dWlm splits via the gemm helper.
        fill_zero(W["dXN_f32"])
        p.wave()
        if mk.wgmma_ok(c.S, c.H, c.V, 0):
            sk_head = mk.wgmma_split_k(
                c.S, c.H, c.V, target_tiles=int(os.environ.get("MK_HEAD_DX_TARGET_TILES", "192"))
            )
            p.instr(
                mk.OP_GEMM,
                mk.gemm_tiles_wgmma(c.S, c.H) * sk_head,
                [B(A["logits"]), B(self.params["wlm"]), B(W["dXN_f32"]), c.S, c.H, c.V, 32 | 8 | 128, 0, sk_head],
            )
        else:
            sk_head = mk.gemm_split_k(
                c.S, c.H, c.V, target_tiles=int(os.environ.get("MK_HEAD_DX_TARGET_TILES", "192"))
            )
            p.instr(
                mk.OP_GEMM,
                mk.gemm_tiles(c.S, c.H) * sk_head,
                [B(A["logits"]), B(self.params["wlm"]), B(W["dXN_f32"]), c.S, c.H, c.V, 32 | 8, 0, sk_head],
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
            p.instr(mk.OP_SWIGLU_BWD, mk.rowop_tiles(c.S), [a("gu"), dhs, B(W["dGU"]), c.S, c.I, dhs_f32])
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
            drow_wg_default = "1" if c.S >= 2048 else "0"
            drow_wg = bool(int(os.environ.get("MK_DROW_WG_LONGONLY", drow_wg_default)))
            if drow_wg and mk.wgmma_ok(c.S, c.nq * c.D, c.H, drow_flags):
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
                # P4b retune after SW128/NN routing: dQ now wants one kv chunk.
                # dKV wants C=1 once nq * (S/128) already exposes >=64 chunks (small),
                # otherwise C=2 keeps enough tail parallelism (nano/S1024-H256).
                n_qt128 = c.S // 128
                default_Ckv = 1 if c.nq * n_qt128 >= 64 else 2
                Ckv = max(1, int(os.environ.get("MK_ATTN_DKV_C", str(default_Ckv))))
                Cq = max(1, int(os.environ.get("MK_ATTN_DQ_C", "1")))
                p.instr(mk.OP_ATTN_DKV_WG, c.nkv * n_qt128 * G * Ckv, dkv_args() + [Ckv])
                p.instr(mk.OP_ATTN_DQ_WG, c.nq * n_qt128 * Cq, dq_args() + [Cq])
            else:
                Cq = 4 if n_qt >= 8 else 2
                p.instr(mk.OP_ATTN_DKV, c.nkv * n_qt * G, dkv_args())
                p.instr(mk.OP_ATTN_DQ, c.nq * n_qt * Cq, dq_args() + [Cq])
            p.wave()
            # qk-norm+rope bwd reads the attention-bwd fp32 atomic workspace DIRECTLY
            # (dy_f32) — the former per-layer CVT chain hop is gone (v3 P1).
            p.instr(
                mk.OP_QKNORM_ROPE_BWD,
                mk.rowop_tiles(c.S),
                [
                    a("qkvraw"),
                    B(W[f"dQKV_f32.{l}"]),
                    B(W["dQKVraw"]),
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
                ],
            )
            p.wave()
            dxn1, dxn1_f32 = gemm_dx(B(W["dQKVraw"]), pr("wqkv"), B(W["dXN"]), lambda: B(W[f"dXN1_f32.{l}"]), c.S, c.H, QD)
            gemm(B(W["dQKVraw"]), a("xn1"), gr("wqkv"), QD, c.H, c.S, 1 | 4 | 8)
            p.wave()
            rmsnorm_bwd([X[l], pr("w1"), dxn1, B(W["dX"]), gr("w1"), a("rstd1"), c.H, dxn1_f32, c.S])
            p.wave()

        p.instr(mk.OP_EMBED_BWD, c.S, [B(self.tokens), B(W["dX"]), B(self.grads["emb"]), c.H])
        p.wave()

        self.prog = p.finalize(self.dev)
        self.n_waves = len(p.waves)

    # ------------------------------------------------------------------ #
    def step(self, tokens: torch.Tensor, labels: torch.Tensor, mode="df") -> torch.Tensor:
        """One fused fwd+bwd. Returns the (device) loss scalar; grads are in self.grads."""
        self.tokens.copy_(tokens)
        self.labels.copy_(labels)
        self.inv_valid.copy_(1.0 / (labels >= 0).sum().clamp(min=1).float().reshape(1))
        self.prog.run(self.ext, mode=mode)
        return self.loss
