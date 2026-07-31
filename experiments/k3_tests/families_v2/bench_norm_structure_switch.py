#!/usr/bin/env python
"""Re-fit the v2 norm structure switch (``_v2_norm_use_split``) from measurement.

Times both realizations at every (H, M) cell with the other one forced off, then
scores the shipped rule against a per-cell oracle. Also re-checks bit-neutrality,
which is what licenses moving the switch at all.

  PYTHONPATH=src python experiments/k3_tests/families_v2/bench_norm_structure_switch.py [--write]

Result frozen at results/norm_structure_switch_h100.json (H100 80GB, 132 SMs,
torch 2.12.1+cu132, triton 3.7.1).
"""

import argparse
import json
from pathlib import Path

import torch
import triton

from xorl.ops import bi_families_v2 as v2


EPS = 1e-6
# hidden sizes this fleet trains, then deep-tile probes past them
SHIPPED_H = (2048, 3840, 4096, 5120, 7168, 8192)
DEEP_H = (12288, 16384, 32768)
ROWS = (1, 8, 16, 32, 64, 256, 512, 2048)
LEGACY_SPLIT_M = 256  # the replaced rule: split iff rows > 256
REPORT_PATH = Path(__file__).resolve().with_name("results") / "norm_structure_switch.json"


def bench(fn, iters=200, reps=11):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        best = us if best is None else min(best, us)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write",
        action="store_true",
        help=f"write the report to the fixed repository path {REPORT_PATH}",
    )
    args = ap.parse_args()

    def realization(x, w, residual, zero_centered, force):
        # Selecting by name is the only honest way to time one realization.
        entry = v2._rms_norm_v2_split if force == "split" else v2._rms_norm_v2_fused
        return lambda: entry(x, w, EPS, residual, zero_centered)

    cells = []
    for h in SHIPPED_H + DEEP_H:
        n_tiles = triton.cdiv(h, v2.V2_NORM_TILE)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16).contiguous()
        for rows in ROWS:
            x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            fused = bench(realization(x, w, r, False, "fused"))
            split = bench(realization(x, w, r, False, "split"))
            oracle = "split" if split < fused else "fused"
            picked = "split" if v2._v2_norm_use_split(rows, n_tiles) else "fused"
            legacy = "split" if rows > LEGACY_SPLIT_M else "fused"  # the rule this replaced
            cells.append(
                {
                    "H": h,
                    "M": rows,
                    "n_tiles": n_tiles,
                    "fused_us": round(fused, 3),
                    "split_us": round(split, 3),
                    "oracle": oracle,
                    "picked": picked,
                    "regret_x": 1.0 if picked == oracle else round(max(fused, split) / min(fused, split), 3),
                    "legacy_picked": legacy,
                    "legacy_regret_x": 1.0 if legacy == oracle else round(max(fused, split) / min(fused, split), 3),
                }
            )
            print(json.dumps(cells[-1]), flush=True)
            del x, r
            torch.cuda.empty_cache()

    # Bit-neutrality: the premise that lets the switch move on speed alone.
    compared, mismatched = 0, []
    for h in SHIPPED_H + DEEP_H:
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16).contiguous()
        qw = w[:128].contiguous()
        for rows in ROWS:
            x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            zx = x[:, :128].contiguous()
            for args_ in ((x, w, EPS, r, False), (x, w, EPS, None, False), (zx, qw, EPS, None, True)):
                a = v2._rms_norm_v2_fused(*args_)
                b = v2._rms_norm_v2_split(*args_)
                a = a if isinstance(a, tuple) else (a,)
                b = b if isinstance(b, tuple) else (b,)
                for ta, tb in zip(a, b, strict=True):
                    compared += 1
                    if not torch.equal(ta, tb):
                        mismatched.append({"H": h, "M": rows})
            del x, r
            torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(0)
    report = {
        "env": {
            "gpu": props.name,
            "sms": props.multi_processor_count,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "V2_NORM_TILE": v2.V2_NORM_TILE,
            "V2_NORM_SPLIT_MIN_TILES": v2.V2_NORM_SPLIT_MIN_TILES,
        },
        "cells": cells,
        "misclassified": [c for c in cells if c["regret_x"] > 1.0],
        "worst_regret_x": max(c["regret_x"] for c in cells),
        "legacy_misclassified_count": sum(1 for c in cells if c["legacy_regret_x"] > 1.0),
        "legacy_worst_regret_x": max(c["legacy_regret_x"] for c in cells),
        "cell_count": len(cells),
        "bit_neutrality": {"compared": compared, "mismatched": mismatched},
    }
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=1))
    if args.write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
