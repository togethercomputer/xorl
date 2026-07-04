"""Per-instruction attribution for the dataflow executor.

Consumes the %globaltimer stamps the df kernel already emits (iclk[2*i] = start of the
tile-0 batch, iclk[2*i+1] = completion of the last tile), walks the REALIZED critical
path through the host dependency DAG, and splits every hop into

  wait  (producer end -> consumer start: scheduling gap — ring publish/discovery/claim)
  span  (consumer start -> consumer end: prologue + math + epilogue + tile spread)

This is the meter for the v3 structural round: fusion shrinks the number of hops,
tile-granular deps shrink `wait`, warp specialization shrinks the fixed part of `span`.

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python profile_df.py [nano|small|both]
"""

import sys

import mk
import torch
from model import Cfg, MKQwen3

OP_NAMES = {v: k[3:] for k, v in vars(mk).items() if k.startswith("OP_")}


def gemm_label(args):
    M, N, K, flags = args[3], args[4], args[5], args[6]
    tags = []
    if flags & 128:
        tags.append("wg")
    if flags & 32:
        tags.append("splitK")
    if flags & 256:
        tags.append("+qkrope")
    layout = ("N", "T")[bool(flags & 1)] + ("N", "T")[bool(flags & 2)]
    return f"GEMM{layout} {M}x{N}x{K}{'.' + '.'.join(tags) if tags else ''}"


def instr_label(op, args):
    if op == mk.OP_GEMM:
        return gemm_label(args)
    return OP_NAMES.get(op, f"op{op}")


def profile(m, runs=5, mode="df"):
    prog = m.prog
    n = prog.n_instr
    flat = [ins for wave in prog.waves for ins in wave]
    deps = prog._build_deps(flat)

    tokens = torch.randint(0, m.cfg.V, (m.cfg.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100
    for _ in range(5):  # warmup / jit settle
        m.step(tokens, labels, mode=mode)
    torch.cuda.synchronize()

    best = None
    for _ in range(runs):
        iclk = torch.zeros(2 * n, dtype=torch.int64, device="cuda")
        prog.run(m.ext, wave_clk=iclk, mode=mode)
        torch.cuda.synchronize()
        clk = iclk.cpu()
        starts, ends = clk[0::2].numpy(), clk[1::2].numpy()
        total = ends.max() - starts.min()
        if best is None or total < best[0]:
            best = (total, starts, ends)
    total, starts, ends = best
    t0 = starts.min()

    # realized critical path: from the last-ending instr, walk back through the dep
    # whose end is latest (instr-granular deps guarantee start_i >= end_j for j in deps)
    path = []
    cur = int(ends.argmax())
    while True:
        pred, pred_end = -1, -1
        for j in deps[cur]:
            if ends[j] > pred_end:
                pred, pred_end = j, ends[j]
        wait = starts[cur] - (pred_end if pred >= 0 else t0)
        path.append((cur, wait, ends[cur] - starts[cur]))
        if pred < 0:
            break
        cur = pred
    path.reverse()

    covered = sum(w + s for _, w, s in path) + (0 if not path else starts[path[0][0]] - t0)
    print(f"step total {total / 1e3:9.1f} us   chain hops {len(path)}   "
          f"attribution {covered / total * 100:5.1f}%")
    W = sum(w for _, w, _ in path if w > 0)
    O = -sum(w for _, w, _ in path if w < 0)  # negative wait = region-gated overlap (df2)
    E = sum(s for _, _, s in path)
    print(f"  on-path wait {W / 1e3:8.1f} us ({W / total * 100:4.1f}%)   "
          f"on-path span {E / 1e3:8.1f} us ({E / total * 100:4.1f}%)   "
          f"overlap {O / 1e3:8.1f} us")

    # per-label aggregation along the path
    agg = {}
    for i, w, s in path:
        op, _, args = flat[i]
        lbl = instr_label(op, args)
        c, tw, ts = agg.get(lbl, (0, 0, 0))
        agg[lbl] = (c + 1, tw + w, ts + s)
    print(f"  {'on-path op':44s} {'cnt':>4s} {'wait us':>9s} {'span us':>9s} {'total us':>9s}")
    for lbl, (c, tw, ts) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
        print(f"  {lbl:44s} {c:4d} {tw / 1e3:9.1f} {ts / 1e3:9.1f} {(tw + ts) / 1e3:9.1f}")

    # worst individual hops
    print("  worst hops (wait+span):")
    for i, w, s in sorted(path, key=lambda x: -(x[1] + x[2]))[:12]:
        op, ntiles, args = flat[i]
        print(f"    #{i:3d} {instr_label(op, args):40s} tiles={ntiles:4d} "
              f"wait={w / 1e3:7.1f} span={s / 1e3:7.1f} us")

    # off-path volume: total span by label across ALL instrs (overlap work supply)
    vol = {}
    onpath = {i for i, _, _ in path}
    for i, (op, ntiles, args) in enumerate(flat):
        if i in onpath:
            continue
        lbl = instr_label(op, args)
        c, ts = vol.get(lbl, (0, 0))
        vol[lbl] = (c + 1, ts + (ends[i] - starts[i]))
    print("  off-path span by op (top 8; overlapped work, not additive):")
    for lbl, (c, ts) in sorted(vol.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"    {lbl:44s} {c:4d} {ts / 1e3:9.1f} us")
    return total, path, flat


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    mode = sys.argv[2] if len(sys.argv) > 2 else "df"
    torch.cuda.set_device(0)
    if which in ("nano", "both"):
        print(f"=== nano ({mode}) ===")
        m = MKQwen3(Cfg(), seed=0)
        print(f"n_instr={m.prog.n_instr} critical_path={m.prog.critical_path} gated={m.prog.n_gated}")
        profile(m, mode=mode)
    if which in ("small", "both"):
        print(f"=== small ({mode}) ===")
        m = MKQwen3(Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024), seed=0)
        print(f"n_instr={m.prog.n_instr} critical_path={m.prog.critical_path} gated={m.prog.n_gated}")
        profile(m, mode=mode)
