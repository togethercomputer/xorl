"""Per-op unit tests for the megakernel device library, vs PyTorch references.

Run: CUDA_VISIBLE_DEVICES=<idle> <fa4-venv>/bin/python test_ops.py
"""

import os

import mk
import torch


torch.manual_seed(0)
DEV = "cuda"
EXT = None


def run1(build, smem_bytes=None):
    """Build a one-off program via `build(p)` and run it."""
    p = mk.Program()
    build(p)
    p.finalize().run(EXT, smem_bytes=smem_bytes)
    torch.cuda.synchronize()


def check(name, got, ref, atol, rtol=2e-2):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs().max().item()
    denom = ref.abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"  {name:34s} max_abs_err={err:.3e} (ref_max={denom:.3e}) {'OK' if ok else 'FAIL'}")
    assert ok, name
    return err


def gemm_tiles(M, N):
    return mk.gemm_tiles(M, N)


def test_gemm():
    print("gemm:")
    M, N, K = 200, 136, 96  # deliberately non-multiples of tile sizes
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    Wt = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)  # [N,K] used as B^T
    At = torch.randn(K, M, device=DEV, dtype=torch.bfloat16)  # [K,M] used as A^T
    Res = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)

    # plain NN, bf16 out
    C = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    run1(lambda p: p.instr(mk.OP_GEMM, gemm_tiles(M, N), [p.buf(A), p.buf(B), p.buf(C), M, N, K, 0, 0]))
    check("NN bf16", C, A.float() @ B.float(), atol=0.35)

    # B^T (Linear fwd) + residual
    C2 = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    run1(lambda p: p.instr(mk.OP_GEMM, gemm_tiles(M, N), [p.buf(A), p.buf(Wt), p.buf(C2), M, N, K, 2 | 16, p.buf(Res)]))
    check("NT + residual", C2, A.float() @ Wt.float().T + Res.float(), atol=0.35)

    # Direct m64n256 NT bf16 route: used by exact-gated qwen forwards.
    M2, N2, K2 = 128, 256, 64
    A2 = torch.randn(M2, K2, device=DEV, dtype=torch.bfloat16)
    Wt2 = torch.randn(N2, K2, device=DEV, dtype=torch.bfloat16)
    C2n = torch.empty(M2, N2, device=DEV, dtype=torch.bfloat16)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M2, N2),
            [p.buf(A2), p.buf(Wt2), p.buf(C2n), M2, N2, K2, 2 | 128 | 16384, 0],
        )
    )
    check("NT n256 bf16", C2n, A2.float() @ Wt2.float().T, atol=0.35)

    M3, N3, K3 = 256, 384, 64  # includes a 128-column n256 tail and >1 M tile
    A3 = torch.randn(M3, K3, device=DEV, dtype=torch.bfloat16)
    Wt3 = torch.randn(N3, K3, device=DEV, dtype=torch.bfloat16)
    C3s = torch.empty(M3, N3, device=DEV, dtype=torch.bfloat16)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, N3),
            [
                p.buf(A3),
                p.buf(Wt3),
                p.buf(C3s),
                M3,
                N3,
                K3,
                2 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("NT n256 stage3 bf16", C3s, A3.float() @ Wt3.float().T, atol=0.35)

    C3sn = torch.empty(M3, N3, device=DEV, dtype=torch.bfloat16)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, N3),
            [
                p.buf(A3),
                p.buf(Wt3),
                p.buf(C3sn),
                M3,
                N3,
                K3,
                2 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG | mk.GEMM_N256_NMAJOR_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("NT n256 nmajor bf16", C3sn, A3.float() @ Wt3.float().T, atol=0.35)

    B3 = torch.randn(K3, 256, device=DEV, dtype=torch.bfloat16)
    C3nn = torch.empty(M3, 256, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(A3),
                p.buf(B3),
                p.buf(C3nn),
                M3,
                256,
                K3,
                8 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("NN n256 stage3 fp32", C3nn, A3.float() @ B3.float(), atol=0.35)

    C3nnn = torch.empty(M3, 256, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(A3),
                p.buf(B3),
                p.buf(C3nnn),
                M3,
                256,
                K3,
                8 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG | mk.GEMM_N256_NMAJOR_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("NN n256 nmajor fp32", C3nnn, A3.float() @ B3.float(), atol=0.35)

    C3nnb = torch.empty(M3, 256, device=DEV, dtype=torch.bfloat16)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(A3),
                p.buf(B3),
                p.buf(C3nnb),
                M3,
                256,
                K3,
                128 | 16384 | mk.GEMM_N256_STAGE3_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("NN n256 stage3 bf16", C3nnb, A3.float() @ B3.float(), atol=0.35)

    C3nnbn = torch.empty(M3, 256, device=DEV, dtype=torch.bfloat16)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(A3),
                p.buf(B3),
                p.buf(C3nnbn),
                M3,
                256,
                K3,
                128 | 16384 | mk.GEMM_N256_STAGE3_FLAG | mk.GEMM_N256_NMAJOR_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("NN n256 nmajor bf16", C3nnbn, A3.float() @ B3.float(), atol=0.35)

    O3 = torch.randn(M3, 256, device=DEV, dtype=torch.bfloat16)
    C3drow = torch.empty(M3, 256, device=DEV, dtype=torch.bfloat16)
    Drow3 = torch.zeros(2, M3, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(A3),
                p.buf(B3),
                p.buf(C3drow),
                M3,
                256,
                K3,
                1024 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG | mk.GEMM_N256_NMAJOR_FLAG,
                0,
                0,
                p.buf(O3),
                p.buf(Drow3),
                128,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    ref3d = (A3.float() @ B3.float()).to(torch.bfloat16).float()
    check("NN n256 drow bf16", C3drow, ref3d, atol=0.35)
    ref_drow3 = (ref3d.view(M3, 2, 128) * O3.float().view(M3, 2, 128)).sum(-1).T
    check("NN n256 drow", Drow3, ref_drow3, atol=0.45, rtol=3e-2)

    At3 = torch.randn(K3, M3, device=DEV, dtype=torch.bfloat16)
    C3tn = torch.empty(M3, 256, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(At3),
                p.buf(B3),
                p.buf(C3tn),
                M3,
                256,
                K3,
                1 | 8 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("TN n256 stage3 fp32", C3tn, At3.float().T @ B3.float(), atol=0.35)

    C3tnn = torch.empty(M3, 256, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M3, 256),
            [
                p.buf(At3),
                p.buf(B3),
                p.buf(C3tnn),
                M3,
                256,
                K3,
                1 | 8 | 128 | 16384 | mk.GEMM_N256_STAGE3_FLAG | mk.GEMM_N256_NMAJOR_FLAG,
                0,
            ],
        ),
        smem_bytes=148 * 1024,
    )
    check("TN n256 nmajor fp32", C3tnn, At3.float().T @ B3.float(), atol=0.35)

    Res2 = torch.randn(M2, N2, device=DEV, dtype=torch.bfloat16)
    C2r = torch.empty(M2, N2, device=DEV, dtype=torch.bfloat16)
    parts2 = torch.empty(M2, N2 // 64, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n256_direct(M2, N2),
            [
                p.buf(A2),
                p.buf(Wt2),
                p.buf(C2r),
                M2,
                N2,
                K2,
                2 | 16 | 128 | 16384 | 8192,
                p.buf(Res2),
                0,
                p.buf(parts2),
                N2 // 64,
            ],
        )
    )
    ref2r = (A2.float() @ Wt2.float().T + Res2.float()).to(torch.bfloat16).float()
    check("NT n256 residual", C2r, ref2r, atol=0.35)
    check("NT n256 ssq", parts2, ref2r.view(M2, N2 // 64, 64).pow(2).sum(-1), atol=0.4, rtol=3e-2)

    Mq, Nq, Kq, Dq, nq, nkv, eps = 128, 256, 64, 64, 2, 1, 1e-6
    Aq = torch.randn(Mq, Kq, device=DEV, dtype=torch.bfloat16)
    Wq = torch.randn(Nq, Kq, device=DEV, dtype=torch.bfloat16)
    Cq = torch.empty(Mq, Nq, device=DEV, dtype=torch.bfloat16)
    QKVR = torch.empty(Mq, Nq, device=DEV, dtype=torch.bfloat16)
    qw = torch.randn(Dq, device=DEV, dtype=torch.bfloat16)
    kw = torch.randn(Dq, device=DEV, dtype=torch.bfloat16)
    rq = torch.empty(Mq, nq, device=DEV, dtype=torch.float32)
    rk = torch.empty(Mq, nkv, device=DEV, dtype=torch.float32)
    cos = torch.randn(Mq, Dq // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(Mq, Dq // 2, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_GEMM,
            mk.gemm_tiles_wgmma_n128(Mq, Nq),
            [
                p.buf(Aq),
                p.buf(Wq),
                p.buf(Cq),
                Mq,
                Nq,
                Kq,
                2 | 128 | 256 | 4096,
                0,
                0,
                p.buf(qw),
                p.buf(kw),
                p.buf(rq),
                p.buf(rk),
                p.buf(cos),
                p.buf(sin),
                p.buf(QKVR),
                nq,
                nkv,
                Dq,
                mk.f2i(eps),
            ],
        )
    )
    raw = Aq.float() @ Wq.float().T
    ref_heads = raw.view(Mq, nq + 2 * nkv, Dq)
    ref_q = ref_heads[:, :nq]
    ref_k = ref_heads[:, nq:nq + nkv]
    ref_v = ref_heads[:, nq + nkv:]
    ref_rq = torch.rsqrt(ref_q.pow(2).mean(-1) + eps)
    ref_rk = torch.rsqrt(ref_k.pow(2).mean(-1) + eps)

    def rope(x, rstd, w):
        y = x * rstd[..., None] * w.float()
        a, b = y[..., : Dq // 2], y[..., Dq // 2:]
        cc, ss = cos[:, None, :], sin[:, None, :]
        return torch.cat([a * cc - b * ss, b * cc + a * ss], dim=-1)

    ref_qkvr = torch.cat([rope(ref_q, ref_rq, qw), rope(ref_k, ref_rk, kw), ref_v], dim=1).reshape(Mq, Nq)
    check("NT n128 qkrope raw", Cq, raw, atol=0.35)
    check("NT n128 qkrope out", QKVR, ref_qkvr, atol=0.35)
    check("NT n128 qkrope rq", rq, ref_rq, atol=1e-4)
    check("NT n128 qkrope rk", rk, ref_rk, atol=1e-4)

    # A^T (dW shape), fp32 out, accumulate
    C3 = torch.ones(M, N, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_GEMM, gemm_tiles(M, N), [p.buf(At), p.buf(B), p.buf(C3), M, N, K, 1 | 4 | 8, 0]))
    check("TN fp32 accum", C3, At.float().T @ B.float() + 1.0, atol=0.35)

    # Round-12 SKR pair (only compiled when the env/gate builds -DMK_HEAD_DX_SKR):
    # K-sliced n128 NN gemm -> per-slice fp32 partial slabs + OP_SKR_REDUCE. The
    # slab prefill checks every element is overwritten by the plain-store epilogue.
    if int(os.environ.get("MK_HEAD_DX_SKR", "0")):
        M4, N4, K4, skr = 256, 128, 512, 2
        A4 = torch.randn(M4, K4, device=DEV, dtype=torch.bfloat16)
        B4 = torch.randn(K4, N4, device=DEV, dtype=torch.bfloat16)
        ws = torch.full((skr, M4, N4), 7.0, device=DEV, dtype=torch.float32)
        C4 = torch.empty(M4, N4, device=DEV, dtype=torch.float32)

        def build_skr(p):
            p.instr(
                mk.OP_GEMM,
                mk.gemm_tiles_wgmma_n128(M4, N4) * skr,
                [p.buf(A4), p.buf(B4), p.buf(ws), M4, N4, K4,
                 8 | 32 | 128 | 4096 | mk.GEMM_SKR_FLAG, 0, skr],
            )
            p.wave()
            p.instr(
                mk.OP_SKR_REDUCE,
                (M4 * N4 + 4095) // 4096,
                [p.buf(ws), p.buf(C4), M4 * N4, skr],
            )

        run1(build_skr)
        check("NN n128 SKR f32", C4, A4.float() @ B4.float(), atol=0.35)


def test_rmsnorm():
    print("rmsnorm:")
    S, H, eps = 100, 192, 1e-6
    x = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    rstd = torch.empty(S, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(S), [p.buf(x), p.buf(w), p.buf(y), p.buf(rstd), H, mk.f2i(eps), S]))
    xf = x.float()
    ref_rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    check("fwd", y, xf * ref_rstd * w.float(), atol=0.05)
    check("rstd", rstd, ref_rstd.squeeze(-1), atol=1e-4)

    dy = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)
    dx = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)  # pre-existing (residual stream)
    dx0 = dx.clone()
    dw = torch.zeros(H, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_RMSNORM_BWD, mk.rowop_tiles(S), [p.buf(x), p.buf(w), p.buf(dy), p.buf(dx), p.buf(dw), p.buf(rstd), H, 0, S]))
    xr = xf.detach().requires_grad_(True)
    wr = w.float().detach().requires_grad_(True)
    yr = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps) * wr
    yr.backward(dy.float())
    check("bwd dx (+= into stream)", dx, dx0.float() + xr.grad, atol=0.06)
    check("bwd dw", dw, wr.grad, atol=0.25)


def test_swiglu():
    print("swiglu:")
    S, I = 64, 160
    gu = torch.randn(S, 2 * I, device=DEV, dtype=torch.bfloat16)
    h = torch.empty(S, I, device=DEV, dtype=torch.bfloat16)
    run1(lambda p: p.instr(mk.OP_SWIGLU_FWD, mk.rowop_tiles(S), [p.buf(gu), p.buf(h), S, I]))
    g, u = gu.float().chunk(2, dim=-1)
    check("fwd", h, torch.nn.functional.silu(g) * u, atol=0.05)

    dh = torch.randn(S, I, device=DEV, dtype=torch.bfloat16)
    dgu = torch.empty(S, 2 * I, device=DEV, dtype=torch.bfloat16)
    run1(lambda p: p.instr(mk.OP_SWIGLU_BWD, mk.rowop_tiles(S), [p.buf(gu), p.buf(dh), p.buf(dgu), S, I]))
    gur = gu.float().detach().requires_grad_(True)
    gr, ur = gur.chunk(2, dim=-1)
    (torch.nn.functional.silu(gr) * ur).backward(dh.float())
    check("bwd", dgu, gur.grad, atol=0.06)


def _rope_tables(S, D, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=DEV).float() / D))
    t = torch.arange(S, device=DEV).float()
    freqs = torch.outer(t, inv)  # [S, D/2]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _qknorm_rope_ref(qkv, qw, kw, cos, sin, nq, nkv, D, eps):
    S = qkv.shape[0]
    x = qkv.float().view(S, nq + 2 * nkv, D)
    q, k, v = x[:, :nq], x[:, nq : nq + nkv], x[:, nq + nkv :]

    def norm(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    def rope(t):
        a, b = t[..., : D // 2], t[..., D // 2 :]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

    return torch.cat([rope(norm(q, qw)), rope(norm(k, kw)), v], dim=1).view(S, -1)


def test_qknorm_rope():
    print("qknorm_rope:")
    S, nq, nkv, D, eps = 96, 4, 2, 64, 1e-6
    stride = (nq + 2 * nkv) * D
    qkv = torch.randn(S, stride, device=DEV, dtype=torch.bfloat16)
    qw = torch.randn(D, device=DEV, dtype=torch.bfloat16)
    kw = torch.randn(D, device=DEV, dtype=torch.bfloat16)
    cos, sin = _rope_tables(S, D)
    out = torch.empty_like(qkv)
    rq = torch.empty(S, nq, device=DEV, dtype=torch.float32)
    rk = torch.empty(S, nkv, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_QKNORM_ROPE_FWD,
            S,
            [
                p.buf(qkv),
                p.buf(out),
                p.buf(qw),
                p.buf(kw),
                p.buf(rq),
                p.buf(rk),
                p.buf(cos),
                p.buf(sin),
                nq,
                nkv,
                D,
                mk.f2i(eps),
            ],
        )
    )
    ref = _qknorm_rope_ref(qkv, qw, kw, cos, sin, nq, nkv, D, eps)
    check("fwd", out, ref, atol=0.06)

    dout = torch.randn_like(qkv)
    din = torch.empty_like(qkv)
    dqw = torch.zeros(D, device=DEV, dtype=torch.float32)
    dkw = torch.zeros(D, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_QKNORM_ROPE_BWD,
            mk.rowop_tiles(S),
            [
                p.buf(qkv),
                p.buf(dout),
                p.buf(din),
                p.buf(qw),
                p.buf(kw),
                p.buf(dqw),
                p.buf(dkw),
                p.buf(rq),
                p.buf(rk),
                p.buf(cos),
                p.buf(sin),
                nq,
                nkv,
                D,
                0,  # dy_f32
                S,
            ],
        )
    )
    qkvr = qkv.float().detach().requires_grad_(True)
    qwr = qw.float().detach().requires_grad_(True)
    kwr = kw.float().detach().requires_grad_(True)
    # rebuild the reference on differentiable inputs
    S_ = qkv.shape[0]
    x = qkvr.view(S_, nq + 2 * nkv, D)
    q, k, v = x[:, :nq], x[:, nq : nq + nkv], x[:, nq + nkv :]
    nrm = lambda t, w: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w  # noqa: E731

    def rope(t):
        a, b = t[..., : D // 2], t[..., D // 2 :]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

    refb = torch.cat([rope(nrm(q, qwr)), rope(nrm(k, kwr)), v], dim=1).view(S_, -1)
    refb.backward(dout.float())
    check("bwd dqkv", din, qkvr.grad, atol=0.08)
    check("bwd dqw", dqw, qwr.grad, atol=0.3)
    check("bwd dkw", dkw, kwr.grad, atol=0.3)


def test_embed():
    print("embed:")
    S, V, H = 128, 512, 96
    tok = torch.randint(0, V, (S,), device=DEV, dtype=torch.int32)
    emb = torch.randn(V, H, device=DEV, dtype=torch.bfloat16)
    x = torch.empty(S, H, device=DEV, dtype=torch.bfloat16)
    run1(lambda p: p.instr(mk.OP_EMBED_FWD, S, [p.buf(tok), p.buf(emb), p.buf(x), H]))
    check("fwd", x, emb[tok.long()], atol=0)

    dx = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)
    de = torch.zeros(V, H, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_EMBED_BWD, S, [p.buf(tok), p.buf(dx), p.buf(de), H]))
    ref = torch.zeros(V, H, device=DEV, dtype=torch.float32)
    ref.index_add_(0, tok.long(), dx.float())
    check("bwd", de, ref, atol=1e-3)


def test_ce():
    print("ce:")
    S, V = 96, 1000
    logits = (torch.randn(S, V, device=DEV) * 3).to(torch.bfloat16)
    labels = torch.randint(0, V, (S,), device=DEV, dtype=torch.int32)
    labels[5] = -100
    labels[70] = -100
    nvalid = int((labels >= 0).sum())
    inv = torch.tensor([1.0 / nvalid], device=DEV, dtype=torch.float32)
    lse = torch.empty(S, device=DEV, dtype=torch.float32)
    loss = torch.zeros(1, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_CE_FWD, S, [p.buf(logits), p.buf(labels), p.buf(lse), p.buf(loss), p.buf(inv), V]))
    zf = logits.float().detach().requires_grad_(True)
    ref_loss = torch.nn.functional.cross_entropy(zf, labels.long(), ignore_index=-100)
    check("fwd loss", loss[0], ref_loss, atol=2e-3)
    check("fwd lse", lse, torch.logsumexp(logits.float(), -1), atol=2e-3)

    dz = logits.clone()
    run1(lambda p: p.instr(mk.OP_CE_BWD, S, [p.buf(dz), p.buf(labels), p.buf(lse), p.buf(inv), V]))
    ref_loss.backward()
    check("bwd dlogits", dz, zf.grad, atol=2e-3)


def _attn_ref(qkv, nq, nkv, D, S):
    x = qkv.float().view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)  # [nq, S, D]
    k = x[:, nq : nq + nkv].permute(1, 0, 2)
    v = x[:, nq + nkv :].permute(1, 0, 2)
    k = k.repeat_interleave(nq // nkv, dim=0)
    v = v.repeat_interleave(nq // nkv, dim=0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.permute(1, 0, 2).reshape(S, nq * D)  # [S, nq*D]


def test_attention():
    print("attention:")
    S, nq, nkv, D = 200, 4, 2, 64  # S deliberately not a multiple of 32
    scale = D**-0.5
    stride = (nq + 2 * nkv) * D
    qkv = (torch.randn(S, stride, device=DEV) * 0.5).to(torch.bfloat16)
    O = torch.empty(S, nq * D, device=DEV, dtype=torch.bfloat16)
    LSE = torch.empty(nq, S, device=DEV, dtype=torch.float32)
    n_qt = (S + 31) // 32
    run1(lambda p: p.instr(mk.OP_ATTN_FWD, nq * n_qt, [p.buf(qkv), p.buf(O), p.buf(LSE), S, nq, nkv, D, mk.f2i(scale)]))
    check("fwd O", O, _attn_ref(qkv, nq, nkv, D, S), atol=0.06)

    dO = (torch.randn(S, nq * D, device=DEV) * 0.5).to(torch.bfloat16)
    Drow = torch.empty(nq, S, device=DEV, dtype=torch.float32)
    dqkv = torch.zeros_like(qkv)
    ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)  # fp32 atomic workspace
    G, Cq = nq // nkv, 2
    run1(lambda p: p.instr(mk.OP_ATTN_DPRE, S, [p.buf(dO), p.buf(O), p.buf(Drow), S, nq, D]))
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_DKV,
            nkv * n_qt * G,
            [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(ws), S, nq, nkv, D, mk.f2i(scale)],
        )
    )
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_DQ,
            nq * n_qt * Cq,
            [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(ws), S, nq, nkv, D, mk.f2i(scale), Cq],
        )
    )
    run1(lambda p: p.instr(mk.OP_CVT_F32BF16, (S * stride + 4095) // 4096, [p.buf(ws), p.buf(dqkv), S * stride]))

    qkvr = qkv.float().detach().requires_grad_(True)
    x = qkvr.view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)
    k = x[:, nq : nq + nkv].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    v = x[:, nq + nkv :].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o.permute(1, 0, 2).reshape(S, nq * D).backward(dO.float())
    check("bwd dqkv", dqkv, qkvr.grad, atol=0.08)

    # wgmma (Hopper) attention ops: the D=64, S%128==0 model-shape fast path
    Sw, nqw, nkvw = 512, 4, 2
    stw = (nqw + 2 * nkvw) * 64
    scw = mk.f2i(64**-0.5)
    qkvw = (torch.randn(Sw, stw, device=DEV) * 0.5).to(torch.bfloat16)
    Ow = torch.empty(Sw, nqw * 64, device=DEV, dtype=torch.bfloat16)
    LSEw = torch.empty(nqw, Sw, device=DEV, dtype=torch.float32)
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_FWD_WG, nqw * (Sw // 128), [p.buf(qkvw), p.buf(Ow), p.buf(LSEw), Sw, nqw, nkvw, 64, scw]
        )
    )
    check("fwd O (wgmma)", Ow, _attn_ref(qkvw, nqw, nkvw, 64, Sw), atol=0.06)

    dOw = (torch.randn(Sw, nqw * 64, device=DEV) * 0.5).to(torch.bfloat16)
    Droww = torch.empty(nqw, Sw, device=DEV, dtype=torch.float32)
    dqkvw = torch.zeros_like(qkvw)
    wsw = torch.zeros(Sw, stw, device=DEV, dtype=torch.float32)
    Gw, Ckvw, Cqw = nqw // nkvw, 2, 4
    run1(lambda p: p.instr(mk.OP_ATTN_DPRE, Sw, [p.buf(dOw), p.buf(Ow), p.buf(Droww), Sw, nqw, 64]))
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_DKV_WG,
            nkvw * (Sw // 128) * Gw * Ckvw,
            [p.buf(qkvw), p.buf(dOw), p.buf(LSEw), p.buf(Droww), p.buf(wsw), Sw, nqw, nkvw, 64, scw, Ckvw],
        )
    )
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_DQ_WG,
            nqw * (Sw // 128) * Cqw,
            [p.buf(qkvw), p.buf(dOw), p.buf(LSEw), p.buf(Droww), p.buf(wsw), Sw, nqw, nkvw, 64, scw, Cqw],
        )
    )
    run1(lambda p: p.instr(mk.OP_CVT_F32BF16, (Sw * stw + 4095) // 4096, [p.buf(wsw), p.buf(dqkvw), Sw * stw]))
    qkvwr = qkvw.float().detach().requires_grad_(True)
    xw = qkvwr.view(Sw, nqw + 2 * nkvw, 64)
    qw = xw[:, :nqw].permute(1, 0, 2)
    kw = xw[:, nqw : nqw + nkvw].permute(1, 0, 2).repeat_interleave(nqw // nkvw, dim=0)
    vw = xw[:, nqw + nkvw :].permute(1, 0, 2).repeat_interleave(nqw // nkvw, dim=0)
    ow = torch.nn.functional.scaled_dot_product_attention(qw, kw, vw, is_causal=True)
    ow.permute(1, 0, 2).reshape(Sw, nqw * 64).backward(dOw.float())
    check("bwd dqkv (wgmma)", dqkvw, qkvwr.grad, atol=0.08)

    # D=128 variant (real Qwen3 head_dim)
    S2, nq2, nkv2, D2 = 96, 2, 1, 128
    stride2 = (nq2 + 2 * nkv2) * D2
    qkv2 = (torch.randn(S2, stride2, device=DEV) * 0.5).to(torch.bfloat16)
    O2 = torch.empty(S2, nq2 * D2, device=DEV, dtype=torch.bfloat16)
    LSE2 = torch.empty(nq2, S2, device=DEV, dtype=torch.float32)
    nqt2 = (S2 + 31) // 32
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_FWD, nq2 * nqt2, [p.buf(qkv2), p.buf(O2), p.buf(LSE2), S2, nq2, nkv2, D2, mk.f2i(D2**-0.5)]
        )
    )
    check("fwd O (D=128)", O2, _attn_ref(qkv2, nq2, nkv2, D2, S2), atol=0.06)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    EXT = mk.load_ext()
    test_gemm()
    test_rmsnorm()
    test_swiglu()
    test_qknorm_rope()
    test_embed()
    test_ce()
    test_attention()
    print("ALL OP TESTS PASSED")
