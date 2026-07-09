#!/usr/bin/env python3
"""Validate explicit qwen4b-l1 sidecar API support."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
sys.path.insert(0, str(MKDIR))

from model import Cfg  # noqa: E402
import qwen_nt_sidecar_api_0ba235d as api  # noqa: E402


CFG = Cfg(H=2560, L=1, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--loss-atol", type=float, default=5.0e-3)
    parser.add_argument("--grad-rtol", type=float, default=5.0e-2)
    args = parser.parse_args()

    api.CFG = CFG
    api.torch.cuda.set_device(args.device)
    api.torch.manual_seed(0)

    orders = [
        api.run_order("default_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
        api.run_order("split_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
    ]
    ok = all(
        item["parity"]["pass"]
        and item["guard"]["pass"]
        and item["api_equivalence"]["pass"]
        and item["split_wins"] == item["reps"]
        and item["delta_us"] < 0.0
        for item in orders
    )
    summary = {
        "claim": "qwen_nt_sidecar_l1api_0ba235d",
        "sha": api.git_sha(),
        "cfg": CFG.__dict__,
        "pass": ok,
        "orders": orders,
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
