"""Phase 5 probe: wgmma causal GQA attention (fwd + FA2 two-pass bwd) vs the current ops.

Milestones:
  A  descriptor/view validation (single-tile gemms, all four major combos)
  B  full fwd parity vs SDPA (O) + torch logsumexp (LSE), 4 shape combos
  C  fwd timing vs current OP_ATTN_FWD (mk.Program harness), median-of-50 cuda events
  D  bwd parity: dKV then dQ vs torch autograd (torch-fed LSE/Drow, fp32 workspace)
  E  bwd timing vs current OP_ATTN_DKV / OP_ATTN_DQ

Run: CUDA_VISIBLE_DEVICES=<idle> <fa4-venv>/bin/python attention_probe.py [A B C D E]
"""

import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load

import mk

_DIR = os.path.dirname(os.path.abspath(__file__))
CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"
DEV = "cuda"
D = 64


def load_probe(verbose=False):
    return load(
        name="xorl_attn_probe",
        sources=[os.path.join(_DIR, "attention_probe.cu")],
        extra_cuda_cflags=[
            "-O3",
            # explicit -gencode, NOT -arch=sm_90a (CUDA 13.1 -arch also emits compute_90
            # PTX that silently drops 90a features — established in mkv3-p2)
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-I{CUTE_INC}",
            "--expt-relaxed-constexpr",
        ],
        verbose=verbose,
    )


def check(name, got, ref, atol, rtol=2e-2):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"  {name:44s} max_abs_err={err:.3e} (ref_max={ref.abs().max().item():.3e}) "
          f"{'OK' if ok else 'FAIL'}")
    assert ok, name
    return err


# ---- references ---------------------------------------------------------------------


def qkv_split(qkv, nq, nkv, S):
    x = qkv.float().view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)  # [nq, S, D]
    k = x[:, nq : nq + nkv].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    v = x[:, nq + nkv :].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    return q, k, v


def sdpa_ref(qkv, nq, nkv, S):
    q, k, v = qkv_split(qkv, nq, nkv, S)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.permute(1, 0, 2).reshape(S, nq * D)


def lse_ref(qkv, nq, nkv, S, scale):
    q, k, _ = qkv_split(qkv, nq, nkv, S)
    sc = q @ k.transpose(-1, -2) * scale  # [nq, S, S] fp32
    mask = torch.ones(S, S, device=DEV, dtype=torch.bool).tril()
    sc.masked_fill_(~mask, float("-inf"))
    return torch.logsumexp(sc, -1)  # [nq, S]


def grad_ref(qkv, dO, nq, nkv, S):
    qkvr = qkv.float().detach().requires_grad_(True)
    x = qkvr.view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)
    k = x[:, nq : nq + nkv].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    v = x[:, nq + nkv :].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o.permute(1, 0, 2).reshape(S, nq * D).backward(dO.float())
    return qkvr.grad  # [S, stride]


def make_inputs(S, nq, nkv, seed=0):
    torch.manual_seed(seed)
    stride = (nq + 2 * nkv) * D
    qkv = (torch.randn(S, stride, device=DEV) * 0.5).to(torch.bfloat16)
    dO = (torch.randn(S, nq * D, device=DEV) * 0.5).to(torch.bfloat16)
    return qkv, dO


SHAPES = [(512, 4, 2), (512, 8, 4), (1024, 4, 2), (1024, 8, 4)]
TIMED = [(512, 4, 2), (1024, 8, 4)]  # nano / small attention shapes


# ---- milestones ---------------------------------------------------------------------


def milestone_a(ext):
    print("A: descriptor/view validation (64x64x64 single tiles)")
    torch.manual_seed(1)
    A = torch.randn(64, 64, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(64, 64, device=DEV, dtype=torch.bfloat16)
    refs = {
        0: ("K/K   C=A@B^T ", A.float() @ B.float().T),
        1: ("K/MN  C=A@B   ", A.float() @ B.float()),
        2: ("MN/K  C=A^T@B^T", A.float().T @ B.float().T),
        3: ("MN/MN C=A^T@B ", A.float().T @ B.float()),
    }
    for mode, (name, ref) in refs.items():
        C = torch.zeros(64, 64, device=DEV, dtype=torch.float32)
        ext.probe_views(A, B, C, mode)
        torch.cuda.synchronize()
        check(name, C, ref, atol=0.05)


def milestone_b(ext):
    print("B: fwd parity (O vs SDPA, LSE vs logsumexp)")
    for S, nq, nkv in SHAPES:
        qkv, _ = make_inputs(S, nq, nkv)
        scale = D**-0.5
        O = torch.empty(S, nq * D, device=DEV, dtype=torch.bfloat16)
        LSE = torch.empty(nq, S, device=DEV, dtype=torch.float32)
        ext.attn_fwd(qkv, O, LSE, S, nq, nkv, scale)
        torch.cuda.synchronize()
        check(f"O   S={S} nq={nq} nkv={nkv}", O, sdpa_ref(qkv, nq, nkv, S), atol=2.5e-2)
        check(f"LSE S={S} nq={nq} nkv={nkv}", LSE, lse_ref(qkv, nq, nkv, S, scale),
              atol=1e-3, rtol=1e-4)


def time_launch(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)  # us
    return statistics.median(ts)


def current_op_runner(EXT, build):
    p = mk.Program()
    build(p)
    p.finalize()
    return lambda: p.run(EXT)


def milestone_c(ext, EXT):
    print("C: fwd timing (median of 50, cuda events, kernel-only)")
    results = []
    for S, nq, nkv in TIMED:
        qkv, _ = make_inputs(S, nq, nkv)
        scale = D**-0.5
        O = torch.empty(S, nq * D, device=DEV, dtype=torch.bfloat16)
        LSE = torch.empty(nq, S, device=DEV, dtype=torch.float32)
        mine = time_launch(lambda: ext.attn_fwd(qkv, O, LSE, S, nq, nkv, scale))

        O2 = torch.empty_like(O)
        LSE2 = torch.empty_like(LSE)
        n_qt = S // 32
        cur = time_launch(current_op_runner(EXT, lambda p: p.instr(
            mk.OP_ATTN_FWD, nq * n_qt,
            [p.buf(qkv), p.buf(O2), p.buf(LSE2), S, nq, nkv, D, mk.f2i(scale)])))
        # cross-check outputs agree (guards against timing a broken config)
        err = (O.float() - O2.float()).abs().max().item()
        print(f"  S={S:5d} nq={nq}: probe {mine:7.1f}us  current {cur:7.1f}us  "
              f"speedup {cur / mine:4.2f}x  (cross-err {err:.2e})")
        results.append((S, nq, mine, cur))
    return results


def milestone_d(ext):
    print("D: bwd parity (torch-fed LSE/Drow, fp32 atomic workspace)")
    for S, nq, nkv in SHAPES:
        qkv, dO = make_inputs(S, nq, nkv)
        stride = (nq + 2 * nkv) * D
        scale = D**-0.5
        LSE = lse_ref(qkv, nq, nkv, S, scale).contiguous()
        O_ref = sdpa_ref(qkv, nq, nkv, S)
        Drow = (dO.float() * O_ref).view(S, nq, D).sum(-1).T.contiguous()  # [nq, S]
        ref = grad_ref(qkv, dO, nq, nkv, S)

        for C in (1, 2, 4):
            ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)
            ext.attn_dkv(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C)
            torch.cuda.synchronize()
            check(f"dK S={S} nq={nq} C={C}", ws[:, nq * D:(nq + nkv) * D],
                  ref[:, nq * D:(nq + nkv) * D], atol=0.06)
            check(f"dV S={S} nq={nq} C={C}", ws[:, (nq + nkv) * D:],
                  ref[:, (nq + nkv) * D:], atol=0.06)

            ext.attn_dq(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C)
            torch.cuda.synchronize()
            check(f"dQ S={S} nq={nq} C={C}", ws[:, :nq * D], ref[:, :nq * D], atol=0.06)


def milestone_e(ext, EXT):
    print("E: bwd timing (median of 50, cuda events)")
    results = []
    for S, nq, nkv in TIMED:
        qkv, dO = make_inputs(S, nq, nkv)
        stride = (nq + 2 * nkv) * D
        scale = D**-0.5
        LSE = lse_ref(qkv, nq, nkv, S, scale).contiguous()
        Drow = (dO.float() * sdpa_ref(qkv, nq, nkv, S)).view(S, nq, D).sum(-1).T.contiguous()
        ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)

        dkv_by_c, dq_by_c = {}, {}
        for C in (1, 2, 4):
            dkv_by_c[C] = time_launch(
                lambda: ext.attn_dkv(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C))
            dq_by_c[C] = time_launch(
                lambda: ext.attn_dq(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C))
        best_ckv = min(dkv_by_c, key=dkv_by_c.get)
        best_cq = min(dq_by_c, key=dq_by_c.get)
        mine_dkv, mine_dq = dkv_by_c[best_ckv], dq_by_c[best_cq]
        print(f"  S={S:5d} nq={nq}: dKV by C {dkv_by_c} -> C={best_ckv} | "
              f"dQ by C {dq_by_c} -> C={best_cq}")

        n_qt = S // 32
        G = nq // nkv
        Cq = 4 if n_qt >= 8 else 2  # what model.py uses at these shapes
        ws2 = torch.zeros_like(ws)
        cur_dkv = time_launch(current_op_runner(EXT, lambda p: p.instr(
            mk.OP_ATTN_DKV, nkv * n_qt * G,
            [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(ws2),
             S, nq, nkv, D, mk.f2i(scale)])))
        cur_dq = time_launch(current_op_runner(EXT, lambda p: p.instr(
            mk.OP_ATTN_DQ, nq * n_qt * Cq,
            [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(ws2),
             S, nq, nkv, D, mk.f2i(scale), Cq])))
        print(f"  S={S:5d} nq={nq}: dKV probe {mine_dkv:7.1f}us vs {cur_dkv:7.1f}us "
              f"({cur_dkv / mine_dkv:4.2f}x) | dQ probe {mine_dq:7.1f}us vs {cur_dq:7.1f}us "
              f"({cur_dq / mine_dq:4.2f}x)")
        results.append((S, nq, mine_dkv, cur_dkv, mine_dq, cur_dq))
    return results


if __name__ == "__main__":
    torch.cuda.set_device(0)
    stages = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "D", "E"]
    ext = load_probe(verbose=False)
    EXT = mk.load_ext() if ("C" in stages or "E" in stages) else None
    if "A" in stages:
        milestone_a(ext)
    if "B" in stages:
        milestone_b(ext)
    if "C" in stages:
        milestone_c(ext, EXT)
    if "D" in stages:
        milestone_d(ext)
    if "E" in stages:
        milestone_e(ext, EXT)
    print("PROBE DONE")
