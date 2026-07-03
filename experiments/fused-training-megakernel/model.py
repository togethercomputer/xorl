"""Qwen3-architecture training megakernel: builds the ONE-kernel fwd+bwd program.

MKQwen3 owns bf16 parameters, fp32 gradients, all saved activations, and the instruction
stream for a complete forward+backward (embedding -> L decoder layers -> final norm ->
lm_head -> CE -> full backward -> embedding grad). `step(tokens, labels)` runs the single
cooperative kernel launch and returns the loss buffer. The optimizer stays outside.
"""

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
        W["dQKVr"] = torch.empty(c.S, QD, device=dev, dtype=bf)
        W["dQKVraw"] = torch.empty(c.S, QD, device=dev, dtype=bf)
        W["dOatt"] = torch.empty(c.S, c.nq * c.D, device=dev, dtype=bf)
        W["dGU"] = torch.empty(c.S, 2 * c.I, device=dev, dtype=bf)
        W["dHs"] = torch.empty(c.S, c.I, device=dev, dtype=bf)
        W["drow"] = torch.empty(c.nq, c.S, device=dev, dtype=f32)
        # fp32 atomic-accumulation workspaces (attention bwd splits; big split-K gemms).
        # dQKV_f32 is per layer so its zero-fill runs up front with no inter-layer
        # dependency (a shared one chains layer N's fill behind layer N+1's convert).
        for l in range(c.L):
            W[f"dQKV_f32.{l}"] = torch.empty(c.S, QD, device=dev, dtype=f32)
        W["dXN_f32"] = torch.empty(c.S, c.H, device=dev, dtype=f32)
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
        p = mk.Program()
        B = p.buf

        def gemm(a, b, out, M, N, K, flags, res=0):
            if flags & 8 and flags & 4:  # fp32 accumulating dW: split-K for occupancy
                sk = mk.gemm_split_k(M, N, K)
                p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N) * sk, [a, b, out, M, N, K, (flags | 32) & ~4, res, sk])
            elif mk.wgmma_ok(M, N, K, flags):  # Hopper warpgroup path
                p.instr(mk.OP_GEMM, mk.gemm_tiles_wgmma(M, N), [a, b, out, M, N, K, flags | 128, res])
            else:
                p.instr(mk.OP_GEMM, mk.gemm_tiles(M, N), [a, b, out, M, N, K, flags, res])

        def fill_zero(t):
            n = t.numel()
            p.instr(mk.OP_FILL_F32, mk.chunk_tiles(n), [B(t), n, mk.f2i(0.0)])

        X = [B(A["X"][l]) for l in range(c.L + 1)]
        n_qt = (c.S + 31) // 32

        # ---- wave 0: embedding gather + zero every fp32 grad / loss / dX stream ----
        p.instr(mk.OP_EMBED_FWD, c.S, [B(self.tokens), B(self.params["emb"]), X[0], c.H])
        for g in self.grads.values():
            fill_zero(g)
        fill_zero(self.loss)
        for lz in range(c.L):
            fill_zero(W[f"dQKV_f32.{lz}"])
        # dX is bf16; zero via fp32 fill over half the elements (bf16 pair = one f32 zero)
        p.instr(mk.OP_FILL_F32, mk.chunk_tiles(W["dX"].numel() // 2), [B(W["dX"]), W["dX"].numel() // 2, mk.f2i(0.0)])
        p.wave()

        # ---- forward layers ----
        for l in range(c.L):
            pr = lambda n: B(self.params[f"{n}.{l}"])  # noqa: E731
            a = lambda n: B(A[f"{n}.{l}"])  # noqa: E731
            p.instr(mk.OP_RMSNORM_FWD, c.S, [X[l], pr("w1"), a("xn1"), a("rstd1"), c.H, eps])
            p.wave()
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
            p.instr(mk.OP_ATTN_FWD, c.nq * n_qt, [a("qkvr"), a("oatt"), a("lse"), c.S, c.nq, c.nkv, c.D, scale])
            p.wave()
            gemm(a("oatt"), pr("wo"), a("x2"), c.S, c.H, c.nq * c.D, 2 | 16, X[l])
            p.wave()
            p.instr(mk.OP_RMSNORM_FWD, c.S, [a("x2"), pr("w2"), a("xn2"), a("rstd2"), c.H, eps])
            p.wave()
            gemm(a("xn2"), pr("wgu"), a("gu"), c.S, 2 * c.I, c.H, 2)
            p.wave()
            p.instr(mk.OP_SWIGLU_FWD, c.S, [a("gu"), a("hs"), c.S, c.I])
            p.wave()
            gemm(a("hs"), pr("wd"), X[l + 1], c.S, c.H, c.I, 2 | 16, a("x2"))
            p.wave()

        # ---- head + loss ----
        p.instr(mk.OP_RMSNORM_FWD, c.S, [X[c.L], B(self.params["wf"]), B(A["xnf"]), B(A["rstdf"]), c.H, eps])
        p.wave()
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
        sk_head = mk.gemm_split_k(c.S, c.H, c.V)
        p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles(c.S, c.H) * sk_head,
            [B(A["logits"]), B(self.params["wlm"]), B(W["dXN_f32"]), c.S, c.H, c.V, 32 | 8, 0, sk_head],
        )
        gemm(B(A["logits"]), B(A["xnf"]), B(self.grads["wlm"]), c.V, c.H, c.S, 1 | 4 | 8)
        p.wave()
        p.instr(mk.OP_CVT_F32BF16, mk.chunk_tiles(c.S * c.H), [B(W["dXN_f32"]), B(W["dXN"]), c.S * c.H])
        p.wave()
        p.instr(
            mk.OP_RMSNORM_BWD,
            c.S,
            [X[c.L], B(self.params["wf"]), B(W["dXN"]), B(W["dX"]), B(self.grads["wf"]), B(A["rstdf"]), c.H],
        )
        p.wave()

        for l in reversed(range(c.L)):
            pr = lambda n: B(self.params[f"{n}.{l}"])  # noqa: E731
            gr = lambda n: B(self.grads[f"{n}.{l}"])  # noqa: E731
            a = lambda n: B(A[f"{n}.{l}"])  # noqa: E731
            # down proj: dHs = dX @ Wd ; dWd += dX^T Hs
            gemm(B(W["dX"]), pr("wd"), B(W["dHs"]), c.S, c.I, c.H, 0)
            gemm(B(W["dX"]), a("hs"), gr("wd"), c.H, c.I, c.S, 1 | 4 | 8)
            p.wave()
            p.instr(mk.OP_SWIGLU_BWD, c.S, [a("gu"), B(W["dHs"]), B(W["dGU"]), c.S, c.I])
            p.wave()
            gemm(B(W["dGU"]), pr("wgu"), B(W["dXN"]), c.S, c.H, 2 * c.I, 0)
            gemm(B(W["dGU"]), a("xn2"), gr("wgu"), 2 * c.I, c.H, c.S, 1 | 4 | 8)
            p.wave()
            p.instr(mk.OP_RMSNORM_BWD, c.S, [a("x2"), pr("w2"), B(W["dXN"]), B(W["dX"]), gr("w2"), a("rstd2"), c.H])
            p.wave()
            # o proj: dOatt = dX @ Wo ; dWo += dX^T Oatt
            gemm(B(W["dX"]), pr("wo"), B(W["dOatt"]), c.S, c.nq * c.D, c.H, 0)
            gemm(B(W["dX"]), a("oatt"), gr("wo"), c.H, c.nq * c.D, c.S, 1 | 4 | 8)
            p.wave()
            p.instr(mk.OP_ATTN_DPRE, c.S, [B(W["dOatt"]), a("oatt"), B(W["drow"]), c.S, c.nq, c.D])
            p.wave()
            # attention bwd: dkv splits the GQA loop (one group member per tile), dq
            # chunks its kv loop; both accumulate into the fp32 workspace with atomics
            # (pre-zeroed), then one convert drains it to bf16. Alias slots keep the
            # disjoint q/kv writes parallel in the dependency analysis.
            G = c.nq // c.nkv
            Cq = 4 if n_qt >= 8 else 2
            p.instr(
                mk.OP_ATTN_DKV,
                c.nkv * n_qt * G,
                [
                    a("qkvr"),
                    B(W["dOatt"]),
                    a("lse"),
                    B(W["drow"]),
                    p.buf(W[f"dQKV_f32.{l}"], slot="kv"),
                    c.S,
                    c.nq,
                    c.nkv,
                    c.D,
                    scale,
                ],
            )
            p.instr(
                mk.OP_ATTN_DQ,
                c.nq * n_qt * Cq,
                [
                    a("qkvr"),
                    B(W["dOatt"]),
                    a("lse"),
                    B(W["drow"]),
                    p.buf(W[f"dQKV_f32.{l}"], slot="q"),
                    c.S,
                    c.nq,
                    c.nkv,
                    c.D,
                    scale,
                    Cq,
                ],
            )
            p.wave()
            QD_ = (c.nq + 2 * c.nkv) * c.D
            p.instr(mk.OP_CVT_F32BF16, mk.chunk_tiles(c.S * QD_), [B(W[f"dQKV_f32.{l}"]), B(W["dQKVr"]), c.S * QD_])
            p.wave()
            p.instr(
                mk.OP_QKNORM_ROPE_BWD,
                c.S,
                [
                    a("qkvraw"),
                    B(W["dQKVr"]),
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
                ],
            )
            p.wave()
            gemm(B(W["dQKVraw"]), pr("wqkv"), B(W["dXN"]), c.S, c.H, QD, 0)
            gemm(B(W["dQKVraw"]), a("xn1"), gr("wqkv"), QD, c.H, c.S, 1 | 4 | 8)
            p.wave()
            p.instr(mk.OP_RMSNORM_BWD, c.S, [X[l], pr("w1"), B(W["dXN"]), B(W["dX"]), gr("w1"), a("rstd1"), c.H])
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
