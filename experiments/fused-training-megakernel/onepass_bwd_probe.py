"""One-pass D64 attention-bwd standalone feasibility probe (session b1d36305).

Compares the FA4-structure one-pass backward (S/dP computed once; dV, dK, dQ in one
pass; dQ drained per stage) against the two-pass structure at current-op feature
level, standalone (plain launches, no megakernel).

Stages:
  P  parity: dkv2/dq2 + all 4 one-pass variants vs torch autograd, S=2048/4096
  T  timing: paired same-GPU runs at s4096/s8192, C sweep, best-vs-best table

Run: CUDA_VISIBLE_DEVICES=<idle> TORCH_EXTENSIONS_DIR=... \
     .venv-fa4/bin/python onepass_bwd_probe.py [P T]
"""

import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load

_DIR = os.path.dirname(os.path.abspath(__file__))
CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"
DEV = "cuda"
D = 64


def load_probe(verbose=False):
    return load(
        name="xorl_onepass_probe",
        sources=[os.path.join(_DIR, "onepass_bwd_probe.cu")],
        extra_cuda_cflags=[
            "-O3",
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
    print(f"  {name:52s} max_abs_err={err:.3e} (ref_max={ref.abs().max().item():.3e}) "
          f"{'OK' if ok else 'FAIL'}")
    assert ok, name
    return err


def make_inputs(S, nq, nkv, seed=0):
    torch.manual_seed(seed)
    stride = (nq + 2 * nkv) * D
    qkv = (torch.randn(S, stride, device=DEV) * 0.5).to(torch.bfloat16)
    dO = (torch.randn(S, nq * D, device=DEV) * 0.5).to(torch.bfloat16)
    return qkv, dO


def qkv_split(qkv, nq, nkv, S):
    x = qkv.float().view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)
    k = x[:, nq: nq + nkv].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    v = x[:, nq + nkv:].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    return q, k, v


def lse_drow_ref(qkv, dO, nq, nkv, S, scale):
    """Per-head loop (memory-lean at S=8192): LSE [nq,S], Drow [nq,S] fp32."""
    q, k, v = qkv_split(qkv, nq, nkv, S)
    dOf = dO.float().view(S, nq, D).permute(1, 0, 2)
    LSE = torch.empty(nq, S, device=DEV, dtype=torch.float32)
    Drow = torch.empty(nq, S, device=DEV, dtype=torch.float32)
    mask = torch.ones(S, S, device=DEV, dtype=torch.bool).tril()
    for h in range(nq):
        sc = (q[h] @ k[h].T) * scale
        sc.masked_fill_(~mask, float("-inf"))
        LSE[h] = torch.logsumexp(sc, -1)
        o = torch.softmax(sc, -1) @ v[h]
        Drow[h] = (dOf[h] * o).sum(-1)
        del sc, o
    del mask
    torch.cuda.empty_cache()
    return LSE, Drow


def grad_ref(qkv, dO, nq, nkv, S):
    qkvr = qkv.float().detach().requires_grad_(True)
    x = qkvr.view(S, nq + 2 * nkv, D)
    q = x[:, :nq].permute(1, 0, 2)
    k = x[:, nq: nq + nkv].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    v = x[:, nq + nkv:].permute(1, 0, 2).repeat_interleave(nq // nkv, dim=0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o.permute(1, 0, 2).reshape(S, nq * D).backward(dO.float())
    return qkvr.grad


def unscramble_map():
    """Inverse of the DRAIN==1 slab order: block[4096] -> natural [64,64] fp32.

    src index (p*128 + wtid)*2 + j maps to r = (wtid>>5)*16 + ((wtid&31)>>2) + 8*(p&1),
    c = (p>>1)*8 + (wtid&3)*2 + j.
    """
    src = torch.arange(4096)
    p = src // 256
    wtid = (src % 256) // 2
    j = src % 2
    w = wtid >> 5
    ln = wtid & 31
    r = w * 16 + (ln >> 2) + 8 * (p & 1)
    c = (p >> 1) * 8 + (ln & 3) * 2 + j
    dst = r * 64 + c
    return dst.to(DEV)


def dqa_to_natural(dqa, nq, S, drain):
    """dqa [nq*S*D] fp32 -> [S, nq*D] natural (matching grad_ref q columns)."""
    x = dqa.view(nq, S, D)
    if drain == 1:
        dst = unscramble_map().view(1, 1, 4096).expand(nq, S // 64, 4096)
        blocks = x.reshape(nq, S // 64, 4096)
        nat = torch.zeros_like(blocks)
        nat.scatter_(2, dst, blocks)
        x = nat.view(nq, S, D)
    return x.permute(1, 0, 2).reshape(S, nq * D)


def stage_parity(ext):
    print("P: parity vs torch autograd (dkv2/dq2 + 4 one-pass variants)")
    for S, nq, nkv in [(2048, 4, 2), (4096, 4, 2)]:
        qkv, dO = make_inputs(S, nq, nkv)
        stride = (nq + 2 * nkv) * D
        scale = D ** -0.5
        LSE, Drow = lse_drow_ref(qkv, dO, nq, nkv, S, scale)
        ref = grad_ref(qkv, dO, nq, nkv, S)

        for C in (1, 2):
            ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)
            ext.attn_dkv2(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C)
            ext.attn_dq2(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C)
            torch.cuda.synchronize()
            check(f"2pass dK S={S} C={C}", ws[:, nq * D:(nq + nkv) * D],
                  ref[:, nq * D:(nq + nkv) * D], atol=0.06)
            check(f"2pass dV S={S} C={C}", ws[:, (nq + nkv) * D:],
                  ref[:, (nq + nkv) * D:], atol=0.06)
            check(f"2pass dQ S={S} C={C}", ws[:, :nq * D], ref[:, :nq * D], atol=0.06)

            for dq_rs in (1, 0):
                for drain in (0, 1):
                    ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)
                    dqa = torch.zeros(nq * S * D, device=DEV, dtype=torch.float32)
                    ext.attn_onepass(qkv, dO, LSE, Drow, ws, dqa, S, nq, nkv, scale, C,
                                     dq_rs, drain)
                    torch.cuda.synchronize()
                    tag = f"1pass rs={dq_rs} drain={drain} S={S} C={C}"
                    check(f"{tag} dK", ws[:, nq * D:(nq + nkv) * D],
                          ref[:, nq * D:(nq + nkv) * D], atol=0.06)
                    check(f"{tag} dV", ws[:, (nq + nkv) * D:],
                          ref[:, (nq + nkv) * D:], atol=0.06)
                    dq = dqa_to_natural(dqa, nq, S, drain)
                    check(f"{tag} dQ", dq, ref[:, :nq * D], atol=0.06)


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


def stage_timing(ext):
    print("T: timing (median of 50, cuda events; per-kernel best C)")
    print("   note: ws/dqa NOT re-zeroed per iter (steady-state accumulate; timing only)")
    flops_note = []
    for S, nq, nkv in [(4096, 4, 2), (8192, 4, 2)]:
        qkv, dO = make_inputs(S, nq, nkv)
        stride = (nq + 2 * nkv) * D
        scale = D ** -0.5
        LSE, Drow = lse_drow_ref(qkv, dO, nq, nkv, S, scale)
        ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)
        dqa = torch.zeros(nq * S * D, device=DEV, dtype=torch.float32)

        res = {}
        for C in (1, 2, 4):
            res[("dkv2", C)] = time_launch(
                lambda: ext.attn_dkv2(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C))
            res[("dq2", C)] = time_launch(
                lambda: ext.attn_dq2(qkv, dO, LSE, Drow, ws, S, nq, nkv, scale, C))
            for dq_rs in (1, 0):
                for drain in (0, 1):
                    res[(f"1p rs{dq_rs} dr{drain}", C)] = time_launch(
                        lambda: ext.attn_onepass(qkv, dO, LSE, Drow, ws, dqa, S, nq,
                                                 nkv, scale, C, dq_rs, drain))

        names = ["dkv2", "dq2", "1p rs1 dr0", "1p rs1 dr1", "1p rs0 dr0", "1p rs0 dr1"]
        best = {}
        for n in names:
            by_c = {C: res[(n, C)] for C in (1, 2, 4)}
            bc = min(by_c, key=by_c.get)
            best[n] = (bc, by_c[bc])
            pretty = " ".join(f"C{C}={by_c[C]:7.1f}" for C in (1, 2, 4))
            print(f"  S={S}: {n:11s} {pretty} -> best C={bc} {by_c[bc]:7.1f}us")

        two = best["dkv2"][1] + best["dq2"][1]
        gflop = 5 * S * S * D * nq / 1e9  # causal bwd FLOPs convention (concept map §0)
        for n in names[2:]:
            one = best[n][1]
            print(f"  S={S}: TOTAL two-pass {two:7.1f}us vs {n} {one:7.1f}us "
                  f"-> {two / one:4.2f}x  ({gflop / one * 1e3:5.0f} vs "
                  f"{gflop / two * 1e3:5.0f} TF/s)")
        flops_note.append((S, two, {n: best[n] for n in names}))
    return flops_note


if __name__ == "__main__":
    torch.cuda.set_device(0)
    stages = [a.upper() for a in sys.argv[1:]] or ["P", "T"]
    print(f"device={torch.cuda.get_device_name(0)}")
    ext = load_probe(verbose=False)
    if "P" in stages:
        stage_parity(ext)
    if "T" in stages:
        stage_timing(ext)
    print("ONEPASS_PROBE_DONE")
