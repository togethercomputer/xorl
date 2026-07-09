#!/usr/bin/env python3
"""Validate qwen4b-l1 graph replay through the opt-in sidecar policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
RESULTS = ROOT / "results"
sys.path.insert(0, str(MKDIR))
sys.path.insert(0, str(RESULTS))

from model import Cfg  # noqa: E402
import qwen_nt_sidecar_api_0ba235d as api  # noqa: E402
import qwen_nt_sidecar_graph_policy_0ba235d as gpol  # noqa: E402


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
    gpol.CFG = CFG
    gpol.torch.cuda.set_device(args.device)
    gpol.torch.manual_seed(0)

    orders = [
        gpol.run_order("default_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
        gpol.run_order("policy_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
    ]
    ok = all(
        item["boundary_step_guard"]["pass"]
        and item["boundary_graph_guard"]["pass"]
        and item["graph_capture"]["pass"]
        and item["graph_capture"]["default_has_graph"]
        and item["graph_capture"]["policy_has_graph"]
        and item["policy_equivalence"]["pass"]
        and item["parity"]["pass"]
        and item["policy_wins"] == item["reps"]
        and item["delta_us"] < 0.0
        for item in orders
    )
    summary = {
        "claim": "qwen_nt_sidecar_l1_graph_policy_0ba235d",
        "sha": gpol.git_sha(),
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
