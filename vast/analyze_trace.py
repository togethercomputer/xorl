#!/usr/bin/env python3
"""Analyze a torch.profiler Chrome trace for exposed (non-overlapped) communication
and kernel-launch-overhead bubbles.

Usage:
    python vast/analyze_trace.py <trace.pt.trace.json.gz> [--start S] [--end S] [--bin MS]

What it reports (for the auto-detected — or user-given — steady-state window):
  * compute / comm / any-GPU busy %  and  GPU-idle %
  * EXPOSED comm  = communication time NOT overlapped by any compute (the real cost
    of TP all-reduce / DP sync that sits on the critical path), broken down per collective
  * launch-overhead bubbles = small (<=30us) inter-kernel GPU idle gaps; many tiny
    (<2-5us) compute kernels is the launch-bound signature (fix: fusion / CUDA graphs)

Stream classification is by kernel name: anything matching 'nccl' (or rccl) is comm,
everything else on a GPU stream is compute. No assumptions about stream ids, so it works
for any rank / parallelism layout.

The full trace is parsed once (slow: ~1-2 min for a ~200MB gz); pass --cache to pickle the
slimmed GPU/runtime events next to the trace for fast re-runs.
"""
import gzip, json, sys, os, pickle, argparse, re
from collections import defaultdict, Counter

COMM_RE = re.compile(r'nccl|rccl', re.I)
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

def load_slim(path, use_cache):
    cache = path + ".slim.pkl"
    if use_cache and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pickle.load(open(cache, "rb"))
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        ev = json.load(fh)["traceEvents"]
    slim = [{k: e.get(k) for k in ("cat","name","pid","tid","ts","dur")}
            for e in ev if e.get("ph") == "X" and e.get("cat") in (GPU_CATS | {"cuda_runtime"})]
    if use_cache:
        pickle.dump(slim, open(cache, "wb"))
    return slim

def union(intervals):
    iv = sorted(intervals); m = []
    for a, b in iv:
        if m and a <= m[-1][1]: m[-1] = (m[-1][0], max(m[-1][1], b))
        else: m.append((a, b))
    return m

def total(m): return sum(b - a for a, b in m)

def subtract(A, B):
    res = []
    for a, b in A:
        cur = a
        for x, y in B:
            if y <= cur: continue
            if x >= b: break
            if x > cur: res.append((cur, min(x, b)))
            cur = max(cur, y)
            if cur >= b: break
        if cur < b: res.append((cur, b))
    return res

def is_comm(e): return bool(COMM_RE.search(e["name"] or ""))

def auto_window(gpu, t0, t1, binw):
    """Densest contiguous run of high-busy bins = steady-state training region."""
    import math
    nb = int((t1 - t0) // binw) + 1
    busy = [0.0] * nb
    for e in gpu:
        busy[int((e["ts"] - t0) // binw)] += e["dur"]
    frac = [b / binw for b in busy]
    hi = [i for i, f in enumerate(frac) if f > 0.6]
    if not hi:  # fall back to whole span
        return t0, t1
    # longest contiguous (gap<=1 bin) run of high bins
    best = cur = [hi[0]]
    for i in hi[1:]:
        if i - cur[-1] <= 2: cur.append(i)
        else:
            if cur[-1] - cur[0] > best[-1] - best[0]: best = cur
            cur = [i]
    if cur[-1] - cur[0] > best[-1] - best[0]: best = cur
    return t0 + best[0] * binw, t0 + (best[-1] + 1) * binw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--start", type=float, help="window start (s, relative to first GPU event)")
    ap.add_argument("--end", type=float, help="window end (s, relative)")
    ap.add_argument("--bin", type=float, default=50.0, help="histogram bin in ms")
    ap.add_argument("--cache", action="store_true", help="pickle slim events for fast re-runs")
    ap.add_argument("--hist", action="store_true", help="print busy/comm time histogram")
    a = ap.parse_args()

    slim = load_slim(a.trace, a.cache)
    gpu = [e for e in slim if e["cat"] in GPU_CATS]
    if not gpu:
        print("no GPU kernel events found"); return
    t0 = min(e["ts"] for e in gpu); t1 = max(e["ts"] + e["dur"] for e in gpu)
    binw = a.bin * 1000
    print("GPU span: %.3f s   (%d kernels, %d comm)" %
          ((t1 - t0) / 1e6, len(gpu), sum(is_comm(e) for e in gpu)))

    if a.hist:
        nb = int((t1 - t0) // binw) + 1
        busy = [0.0] * nb; comm = [0.0] * nb
        for e in gpu:
            i = int((e["ts"] - t0) // binw); busy[i] += e["dur"]
            if is_comm(e): comm[i] += e["dur"]
        for i in range(nb):
            if busy[i] / binw > 0.01:
                print("%6.2fs busy=%5.1f%% comm=%5.1f%% %s" %
                      (i * binw / 1e6, busy[i]/binw*100, comm[i]/binw*100, "#"*int(busy[i]/binw*40)))

    if a.start is not None and a.end is not None:
        W0, W1 = t0 + a.start*1e6, t0 + a.end*1e6
    else:
        W0, W1 = auto_window(gpu, t0, t1, binw)
        print("[auto] steady-state window: %.3f .. %.3f s" % ((W0-t0)/1e6, (W1-t0)/1e6))
    win = W1 - W0

    def clip(e):
        x = max(e["ts"], W0); y = min(e["ts"]+e["dur"], W1); return (x, y) if y > x else None
    inwin = [c for c in (clip(e) for e in gpu) if c]
    comp_iv = [clip(e) for e in gpu if not is_comm(e)]
    comm_iv = [clip(e) for e in gpu if is_comm(e)]
    comp = union([c for c in comp_iv if c]); comm = union([c for c in comm_iv if c])
    allk = union(inwin)
    idle = subtract([(W0, W1)], allk)
    exposed = subtract(comm, comp)

    print("\n=== STEADY-STATE WINDOW  %.3f s ===" % (win/1e6))
    print("compute busy : %7.1f ms (%5.1f%%)" % (total(comp)/1e3, total(comp)/win*100))
    print("comm busy    : %7.1f ms (%5.1f%%)" % (total(comm)/1e3, total(comm)/win*100))
    print("any-GPU busy : %7.1f ms (%5.1f%%)" % (total(allk)/1e3, total(allk)/win*100))
    print("GPU idle     : %7.1f ms (%5.1f%%)" % (total(idle)/1e3, total(idle)/win*100))
    print("EXPOSED comm : %7.1f ms (%5.1f%% of window, %4.0f%% of comm) <- non-overlapped" %
          (total(exposed)/1e3, total(exposed)/win*100, total(exposed)/max(total(comm),1)*100))

    # per-collective exposed
    bycoll = defaultdict(list)
    for e in gpu:
        if is_comm(e) and clip(e):
            key = re.sub(r'_RING.*|_TREE.*|_\d+$', '', e["name"].split("(")[0])
            bycoll[key].append(e)
    print("\nper-collective (in window):")
    for k, es in sorted(bycoll.items(), key=lambda kv: -sum(x["dur"] for x in kv[1])):
        u = union([clip(e) for e in es if clip(e)]); ex = subtract(u, comp)
        print("  %-44s n=%-4d busy=%6.1fms exposed=%6.1fms (%3.0f%%)" %
              (k[:44], len(es), total(u)/1e3, total(ex)/1e3, total(ex)/max(total(u),1)*100))

    # launch-overhead bubbles
    small = [g for g in idle if (g[1]-g[0]) <= 30]
    big = [g for g in idle if (g[1]-g[0]) > 30]
    durs = sorted((e["dur"] for e in gpu if not is_comm(e) and clip(e)))
    print("\nlaunch-overhead bubbles (GPU-idle gaps):")
    print("  small <=30us : %6.2f ms over %4d gaps (mean %.1fus) <- launch-bound" %
          (sum(b-a for a,b in small)/1e3, len(small), sum(b-a for a,b in small)/max(len(small),1)))
    print("  big   >30us  : %6.2f ms over %4d gaps" % (sum(b-a for a,b in big)/1e3, len(big)))
    if big:
        big.sort(key=lambda g: -(g[1]-g[0]))
        print("    top:", ["%.0fus@%.3fs" % ((b-a), (a-t0)/1e6) for a, b in big[:6]])
    if durs:
        print("  compute kernels: %d, median %.1fus, <5us: %d, <2us: %d (tiny=launch-bound)" %
              (len(durs), durs[len(durs)//2], sum(d<5 for d in durs), sum(d<2 for d in durs)))

if __name__ == "__main__":
    main()
