"""Per-op unit tests for the megakernel device library, vs PyTorch references.

Run: CUDA_VISIBLE_DEVICES=<idle> <fa4-venv>/bin/python test_ops.py
"""

import mk
import torch


torch.manual_seed(0)
DEV = "cuda"
EXT = None


def run1(build):
    """Build a one-off program via `build(p)` and run it."""
    p = mk.Program()
    build(p)
    p.finalize().run(EXT)
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

    # A^T (dW shape), fp32 out, accumulate
    C3 = torch.ones(M, N, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_GEMM, gemm_tiles(M, N), [p.buf(At), p.buf(B), p.buf(C3), M, N, K, 1 | 4 | 8, 0]))
    check("TN fp32 accum", C3, At.float().T @ B.float() + 1.0, atol=0.35)


def test_rmsnorm():
    print("rmsnorm:")
    S, H, eps = 100, 192, 1e-6
    x = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    rstd = torch.empty(S, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_RMSNORM_FWD, S, [p.buf(x), p.buf(w), p.buf(y), p.buf(rstd), H, mk.f2i(eps)]))
    xf = x.float()
    ref_rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    check("fwd", y, xf * ref_rstd * w.float(), atol=0.05)
    check("rstd", rstd, ref_rstd.squeeze(-1), atol=1e-4)

    dy = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)
    dx = torch.randn(S, H, device=DEV, dtype=torch.bfloat16)  # pre-existing (residual stream)
    dx0 = dx.clone()
    dw = torch.zeros(H, device=DEV, dtype=torch.float32)
    run1(lambda p: p.instr(mk.OP_RMSNORM_BWD, S, [p.buf(x), p.buf(w), p.buf(dy), p.buf(dx), p.buf(dw), p.buf(rstd), H]))
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
    run1(lambda p: p.instr(mk.OP_SWIGLU_FWD, S, [p.buf(gu), p.buf(h), S, I]))
    g, u = gu.float().chunk(2, dim=-1)
    check("fwd", h, torch.nn.functional.silu(g) * u, atol=0.05)

    dh = torch.randn(S, I, device=DEV, dtype=torch.bfloat16)
    dgu = torch.empty(S, 2 * I, device=DEV, dtype=torch.bfloat16)
    run1(lambda p: p.instr(mk.OP_SWIGLU_BWD, S, [p.buf(gu), p.buf(dh), p.buf(dgu), S, I]))
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
            S,
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
    run1(lambda p: p.instr(mk.OP_ATTN_DPRE, S, [p.buf(dO), p.buf(O), p.buf(Drow), S, nq, D]))
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_DKV,
            nkv * n_qt,
            [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(dqkv), S, nq, nkv, D, mk.f2i(scale)],
        )
    )
    run1(
        lambda p: p.instr(
            mk.OP_ATTN_DQ,
            nq * n_qt,
            [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(dqkv), S, nq, nkv, D, mk.f2i(scale)],
        )
    )

    qkvr = qkv.float().detach().requires_grad_(True)
    x = qkvr.view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)
    k = x[:, nq : nq + nkv].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    v = x[:, nq + nkv :].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o.permute(1, 0, 2).reshape(S, nq * D).backward(dO.float())
    check("bwd dqkv", dqkv, qkvr.grad, atol=0.08)

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
