"""mkv3 Phase 2 probe: warp-specialized paged-smem persistent interpreter vs flat.

Serial chain of 256 wgmma GEMM instructions (C_i = C_{i-1} @ B^T, 128x64x64 bf16 NT,
one tile per instr, distinct C buffers). Measures us/hop for:
  flat      milestone A: flat 256-thread persistent kernel (harness re-baseline)
  ws        milestone B: 384 threads, WG0 producer + 2 consumer warpgroups, 2 smem
            pages, volatile-flag handoff
  mbar      milestone C: same, mbarrier handoff
  ws_reg /  milestone D: + setmaxnreg.dec 40 (WG0) / .inc 224 (consumers)
  mbar_reg
  --nn      milestone E: NN storage variant (B stored [K,N], MN-major descriptor)

Parity gate before any timing: max-abs-err of C_255 vs a torch fp32-matmul chain
(bf16 rounding between hops, matching the kernel's bf16 output buffers) < 0.15.

Run: cd experiments/fused-training-megakernel && \
     timeout 180 env CUDA_VISIBLE_DEVICES=3 ../../.venv-fa4/bin/python wspec_probe.py
"""

import argparse
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

MODES = {"flat": 0, "ws": 1, "mbar": 2, "ws_reg": 3, "mbar_reg": 4}


def build(verbose=False):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"  # setmaxnreg/wgmma need sm_90a
    ext = load(
        name="xorl_wspec_probe",
        sources=[os.path.join(HERE, "wspec_probe.cu")],
        # CUDA 13.1: -arch=sm_90a silently emits compute_90 PTX (setmaxnreg then fails
        # to assemble); the explicit -gencode spelling is required.
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-I{CUTE_INC}",
                           "--expt-relaxed-constexpr"],
        verbose=verbose,
    )
    ext.init()
    return ext


def make_chain(n, nn, seed=0):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(64, 64, device="cuda", dtype=torch.float32))
    # near-orthogonal, slightly contractive: 256-deep chain stays O(1), never underflows
    B = (0.995 * q).to(torch.bfloat16).contiguous()  # NT: stored [N,K]; NN: stored [K,N]
    A0 = torch.randn(128, 64, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    Cs = [torch.full((128, 64), float("nan"), device="cuda", dtype=torch.bfloat16)
          for _ in range(n)]
    ptrs = torch.tensor([c.data_ptr() for c in Cs], device="cuda", dtype=torch.int64)
    done = torch.zeros(n * 32, device="cuda", dtype=torch.int32)
    ctrl = torch.zeros(8, device="cuda", dtype=torch.int32)
    return A0, B, Cs, ptrs, done, ctrl


def parity(ext, mode, nn, n, nblocks, seed=0):
    """Gate: run the chain once, compare checkpoints against the torch reference."""
    A0, B, Cs, ptrs, done, ctrl = make_chain(n, nn, seed)
    ext.run(MODES[mode], int(nn), A0, B, ptrs, done, ctrl, n, nblocks)
    torch.cuda.synchronize()
    x = A0.float()
    Bf = B.float() if nn else B.float().t()
    checks = sorted({0, 1, min(15, n - 1), n - 1})
    errs = {}
    for i in range(n):
        x = (x @ Bf).to(torch.bfloat16).float()
        if i in checks:
            errs[i] = (Cs[i].float() - x).abs().max().item()
    return errs


def timeit(ext, mode, nn, n, nblocks, iters=20, warmup=5, seed=0):
    A0, B, Cs, ptrs, done, ctrl = make_chain(n, nn, seed)
    args = (MODES[mode], int(nn), A0, B, ptrs, done, ctrl, n, nblocks)
    for _ in range(warmup):
        ext.run(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        ext.run(*args)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    ms = ts[len(ts) // 2]
    return ms * 1e3, ms * 1e3 / (n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="flat")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--nblocks", type=int, default=132)
    ap.add_argument("--nn", action="store_true", help="NN storage variant (milestone E)")
    ap.add_argument("--bringup", type=int, default=16, help="short-chain parity first (0=skip)")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = build()
    print(f"ext built. device={torch.cuda.get_device_name(0)} nblocks={args.nblocks}")

    for mode in args.modes.split(","):
        assert mode in MODES, f"unknown mode {mode}"
        tag = f"{mode}{'-nn' if args.nn else ''}"
        if args.bringup:
            errs = parity(ext, mode, args.nn, args.bringup, args.nblocks)
            print(f"[{tag}] bringup n={args.bringup} parity: "
                  + " ".join(f"C{i}={e:.4f}" for i, e in errs.items()), flush=True)
            assert max(errs.values()) < 0.15, f"{tag} bringup parity FAILED"
        errs = parity(ext, mode, args.nn, args.n, args.nblocks)
        emax = max(errs.values())
        print(f"[{tag}] n={args.n} parity: " + " ".join(f"C{i}={e:.4f}" for i, e in errs.items())
              + f"  -> {'OK' if emax < 0.15 else 'FAIL'}", flush=True)
        assert emax < 0.15, f"{tag} parity gate FAILED ({emax:.4f})"
        if not args.no_timing:
            tot, hop = timeit(ext, mode, args.nn, args.n, args.nblocks, iters=args.iters)
            print(f"[{tag}] n={args.n} nblocks={args.nblocks}: {tot:9.1f} us total  "
                  f"{hop:6.3f} us/hop", flush=True)
    print("WSPEC PROBE DONE")


if __name__ == "__main__":
    main()
