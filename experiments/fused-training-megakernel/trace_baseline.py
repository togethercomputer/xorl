"""nsys tracing + gap analysis of the compile+CUDAGraph baseline.

Two modes:
  trace:   builds the baseline exactly as bench.py does, then replays the whole-step
           graph N times inside a cudaProfilerStart/Stop range (each replay followed by
           a synchronize, so replays separate cleanly in the timeline).
             nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
               -o <rep> --force-overwrite true python trace_baseline.py trace nano
  analyze: reads the exported sqlite and reports, per replay: kernel count, wall
           (first start -> last end), active (union of kernel intervals), gap =
           wall - active; plus the top kernels by total time.
             python trace_baseline.py analyze <rep>.sqlite

The gap total is the boundary tax a megakernel structurally avoids; active is the
baseline's own floor. Plan Phase 0 stop rule keys off these numbers.
"""

import sqlite3
import sys

REPLAYS = 20


def build_and_trace(cfg_name):
    import torch
    from bench import TorchQwen3
    from model import Cfg, MKQwen3

    cfg = Cfg() if cfg_name == "nano" else Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)
    torch.cuda.set_device(0)
    mk_model = MKQwen3(cfg, seed=0)  # only for identical param init
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.V, (cfg.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100

    tmg = torch.compile(TorchQwen3(cfg, mk_model.params).cuda())
    for _ in range(5):
        for p in tmg.parameters():
            p.grad = None
        tmg(tokens, labels).backward()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for p in tmg.parameters():
                p.grad.zero_()
            tmg(tokens, labels).backward()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for p in tmg.parameters():
            p.grad.zero_()
        tmg(tokens, labels).backward()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(REPLAYS):
        graph.replay()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("traced", REPLAYS, "replays")


def analyze(sqlite_path):
    db = sqlite3.connect(sqlite_path)
    rows = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id ORDER BY k.start"
    ).fetchall()
    if not rows:
        print("no kernels in trace")
        return
    # split into replays at big gaps (the inter-replay synchronize)
    replays, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][1] > 200_000:  # >200us gap = replay boundary
            replays.append(cur)
            cur = []
        cur.append(r)
    replays.append(cur)
    if len(replays) < REPLAYS // 2 and len(rows) % REPLAYS == 0:
        # back-to-back replays merged (inter-replay gap under threshold): every replay
        # launches the identical kernel sequence, so equal-count chunking is exact
        per = len(rows) // REPLAYS
        replays = [rows[i * per : (i + 1) * per] for i in range(REPLAYS)]
    print(f"{len(rows)} kernels in {len(replays)} replay clusters "
          f"(expect {REPLAYS}; first cluster may be partial)")

    stats = []
    for rep in replays:
        wall = rep[-1][1] - rep[0][0]
        # union of intervals (graph may overlap kernels across streams)
        active, ce = 0, rep[0][0]
        for s, e, _ in rep:
            if s > ce:
                active += e - s
                ce = e
            elif e > ce:
                active += e - ce
                ce = e
        stats.append((len(rep), wall, active, wall - active))
    stats.sort(key=lambda x: x[1])
    n, wall, active, gap = stats[len(stats) // 2]
    print(f"median replay: {n} kernels  wall {wall / 1e3:8.1f} us  "
          f"active {active / 1e3:8.1f} us ({active / wall * 100:4.1f}%)  "
          f"gap {gap / 1e3:8.1f} us ({gap / wall * 100:4.1f}%)")
    print(f"per-kernel boundary tax: {gap / n / 1e3:.2f} us/kernel")

    agg = {}
    total_active = 0
    for s, e, name in replays[len(replays) // 2]:
        c, t = agg.get(name, (0, 0))
        agg[name] = (c + 1, t + e - s)
        total_active += e - s
    print(f"sum-of-durations (no union) {total_active / 1e3:.1f} us; top kernels of median replay:")
    for name, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {c:4d}x {t / 1e3:8.1f} us  {name[:90]}")


if __name__ == "__main__":
    if sys.argv[1] == "trace":
        build_and_trace(sys.argv[2])
    else:
        analyze(sys.argv[2])
