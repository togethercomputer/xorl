"""Per-instruction attribution for the dataflow executor.

Consumes the %globaltimer stamps the df kernel already emits (iclk[2*i] = start of the
tile-0 batch, iclk[2*i+1] = completion of the last tile), walks the REALIZED critical
path through the host dependency DAG, and splits every hop into

  wait  (producer end -> consumer start: scheduling gap — ring publish/discovery/claim)
  span  (consumer start -> consumer end: prologue + math + epilogue + tile spread)

This is the meter for the v3 structural round: fusion shrinks the number of hops,
tile-granular deps shrink `wait`, warp specialization shrinks the fixed part of `span`.

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python profile_df.py [nano|small|both] [mode]

mode defaults to "auto" = the shape's certified executor route (model.default_mode);
pass df/pdf/df2/ws explicitly only for executor A/Bs — a decomposition taken under a
non-certified mode does not explain the certified step.
Run selection is the MEDIAN of `runs` by step total (MK_PROFILE_SELECT=min restores the
old fastest-run pick; the min-run decomposition understates wait vs the certified
median-of-50 score step).
"""

import os
import subprocess
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


def anchor_stamp():
    """Commit hash (+dirty) of this tree, for stamping decompositions."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "."],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{head}{'+dirty' if dirty else ''}" if head else "unknown"
    except Exception:
        return "unknown"


def profile(m, runs=5, mode=None):
    # mode=None -> the shape's certified route; a profile under any other mode
    # decomposes a schedule the score runs never execute.
    forced = mode is not None
    if mode is None:
        mode = getattr(m, "default_mode", "df")
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

    trials = []
    for _ in range(runs):
        iclk = torch.zeros(2 * n, dtype=torch.int64, device="cuda")
        prog.run(m.ext, smem_bytes=getattr(m, "_smem_bytes", None), wave_clk=iclk, mode=mode)
        torch.cuda.synchronize()
        clk = iclk.cpu()
        starts, ends = clk[0::2].numpy(), clk[1::2].numpy()
        trials.append((ends.max() - starts.min(), starts, ends))
    trials.sort(key=lambda t: t[0])
    select = os.environ.get("MK_PROFILE_SELECT", "median")
    picked = trials[0] if select == "min" else trials[len(trials) // 2]
    total, starts, ends = picked
    t0 = starts.min()
    print(f"anchor {anchor_stamp()}   mode={mode}{' (forced)' if forced else ' (certified default)'}   "
          f"select={select} of {runs} runs   min {trials[0][0] / 1e3:.1f} / "
          f"median {trials[len(trials) // 2][0] / 1e3:.1f} us")

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
    print(f"  READ RULE: the chain telescopes the step by construction — interpreter/"
          f"route/dispatch cost is folded INSIDE every wait/span cell (~22% at "
          f"exclusive-run calibration), so bare-kernel standalone times are NOT "
          f"comparable to these rows; compare only against the certified score step.")
    W = sum(w for _, w, _ in path if w > 0)
    O = -sum(w for _, w, _ in path if w < 0)  # negative wait = region-gated overlap (df2)
    E = sum(s for _, _, s in path)
    print(f"  on-path wait {W / 1e3:8.1f} us ({W / total * 100:4.1f}%)   "
          f"on-path span {E / 1e3:8.1f} us ({E / total * 100:4.1f}%)   "
          f"overlap {O / 1e3:8.1f} us")

    # per-label aggregation: on-path (wait/span) AND off-path span in one table.
    # on-path total is the LOWER bound of a label's step contribution; onpath+offpath
    # is the occupancy UPPER bound (off-path is overlapped, not additive — but it is
    # not free either: it competes for issue slots and moves the realized path).
    onpath = {i for i, _, _ in path}
    agg = {}
    for i, w, s in path:
        op, _, args = flat[i]
        lbl = instr_label(op, args)
        c, tw, ts, oc, os_ = agg.get(lbl, (0, 0, 0, 0, 0))
        agg[lbl] = (c + 1, tw + w, ts + s, oc, os_)
    for i, (op, ntiles, args) in enumerate(flat):
        if i in onpath:
            continue
        lbl = instr_label(op, args)
        c, tw, ts, oc, os_ = agg.get(lbl, (0, 0, 0, 0, 0))
        agg[lbl] = (c, tw, ts, oc + 1, os_ + (ends[i] - starts[i]))
    print(f"  {'op (on+off path)':44s} {'cnt':>4s} {'wait us':>9s} {'span us':>9s} "
          f"{'onpath us':>9s} {'offp cnt':>8s} {'offp us':>9s} {'ub us':>9s}")
    for lbl, (c, tw, ts, oc, os_) in sorted(
            agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2] + kv[1][4])):
        print(f"  {lbl:44s} {c:4d} {tw / 1e3:9.1f} {ts / 1e3:9.1f} {(tw + ts) / 1e3:9.1f} "
              f"{oc:8d} {os_ / 1e3:9.1f} {(tw + ts + os_) / 1e3:9.1f}")
    offtot = sum(os_ for _, _, _, _, os_ in agg.values())
    print(f"  {'TOTALS':44s} {'':4s} {W / 1e3:9.1f} {E / 1e3:9.1f} "
          f"{(W + E) / 1e3:9.1f} {'':8s} {offtot / 1e3:9.1f}   "
          f"(quote a bucket as 'onpath..ub us @ anchor/mode', never onpath alone)")

    # worst individual hops
    print("  worst hops (wait+span):")
    for i, w, s in sorted(path, key=lambda x: -(x[1] + x[2]))[:12]:
        op, ntiles, args = flat[i]
        print(f"    #{i:3d} {instr_label(op, args):40s} tiles={ntiles:4d} "
              f"wait={w / 1e3:7.1f} span={s / 1e3:7.1f} us")
    return total, path, flat


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    mode_arg = sys.argv[2] if len(sys.argv) > 2 else "auto"
    mode = None if mode_arg == "auto" else mode_arg  # None -> certified default_mode
    torch.cuda.set_device(0)
    if which in ("nano", "both"):
        m = MKQwen3(Cfg(), seed=0)
        print(f"=== nano ({mode or m.default_mode}) ===")
        print(f"n_instr={m.prog.n_instr} critical_path={m.prog.critical_path} gated={m.prog.n_gated}")
        profile(m, mode=mode)
    if which in ("small", "both"):
        m = MKQwen3(Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024), seed=0)
        print(f"=== small ({mode or m.default_mode}) ===")
        print(f"n_instr={m.prog.n_instr} critical_path={m.prog.critical_path} gated={m.prog.n_gated}")
        profile(m, mode=mode)
