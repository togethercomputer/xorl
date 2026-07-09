#!/usr/bin/env python3
"""Validate CUDA graph replay for the qwen NT sidecar API."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
RESULTS = ROOT / "results"
sys.path.insert(0, str(MKDIR))
sys.path.insert(0, str(RESULTS))

from qwen_nt_sidecar_api_0ba235d import (  # noqa: E402
    CFG,
    build_default,
    build_split,
    compare_grads,
    make_tokens,
    route_summary,
)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def time_call(fn: Callable[[], object]) -> float:
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3


def timing_stats(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def check_graph_parity(
    default,
    split,
    default_replay: Callable[[], object],
    split_replay: Callable[[], object],
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    default_replay()
    split_replay()
    torch.cuda.synchronize()
    default_loss = float(default.loss.item())
    split_loss = float(split.loss.item())
    grad = compare_grads(default, split)
    loss_diff = split_loss - default_loss
    ok = abs(loss_diff) < loss_atol and float(grad["worst_grad_rel"]) < grad_rtol
    return {
        "pass": ok,
        "loss_atol": loss_atol,
        "grad_rtol": grad_rtol,
        "default_loss": default_loss,
        "split_loss": split_loss,
        "loss_diff": loss_diff,
        "grad": grad,
    }


def run_order(order: str, reps: int, warmup: int, loss_atol: float, grad_rtol: float) -> dict[str, object]:
    tokens, labels = make_tokens()
    if order == "default_first":
        default = build_default()
        split = build_split()
    elif order == "split_first":
        split = build_split()
        default = build_default()
    else:
        raise ValueError(f"bad order {order!r}")

    routes = {
        "default": route_summary(default),
        "split": route_summary(split),
    }
    print("ROUTE_JSON " + json.dumps({"order": order, **routes}, sort_keys=True), flush=True)
    if int(routes["default"]["cutpoint_count"]) != 0:
        raise RuntimeError("default route unexpectedly has a sidecar cutpoint")
    if not bool(routes["split"]["api_available"]):
        raise RuntimeError("split route did not expose the sidecar API")

    default_replay = default.make_graphed_step(tokens, labels, warmup=warmup)
    split_replay = split.make_graphed_qwen_nt_sidecar_step(tokens, labels, warmup=warmup)
    graph_capture = {
        "pass": True,
        "default_has_graph": hasattr(default_replay, "graph"),
        "split_has_graph": hasattr(split_replay, "graph"),
    }
    print("GRAPH_CAPTURE_JSON " + json.dumps({"order": order, **graph_capture}, sort_keys=True), flush=True)

    parity = check_graph_parity(default, split, default_replay, split_replay, loss_atol, grad_rtol)
    print("GRAPH_PARITY_JSON " + json.dumps({"order": order, **parity}, sort_keys=True), flush=True)
    if not parity["pass"]:
        raise RuntimeError(f"graph parity failed before timing: {parity}")

    for _ in range(warmup):
        default_replay()
        split_replay()
    torch.cuda.synchronize()

    default_times: list[float] = []
    split_times: list[float] = []
    split_wins = 0
    for _ in range(reps):
        td = time_call(default_replay)
        ts = time_call(split_replay)
        default_times.append(td)
        split_times.append(ts)
        split_wins += int(ts < td)

    default_stats = timing_stats(default_times)
    split_stats = timing_stats(split_times)
    result = {
        "order": order,
        "reps": reps,
        "warmup": warmup,
        "graph_capture": graph_capture,
        "parity": parity,
        "default": default_stats,
        "split": split_stats,
        "delta_us": split_stats["median_us"] - default_stats["median_us"],
        "split_wins": split_wins,
        "default_times_us": default_times,
        "split_times_us": split_times,
        "routes": routes,
    }
    print("GRAPH_TIMING_JSON " + json.dumps(result, sort_keys=True), flush=True)
    del default_replay, split_replay, default, split
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--loss-atol", type=float, default=5.0e-3)
    parser.add_argument("--grad-rtol", type=float, default=5.0e-2)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    orders = [
        run_order("default_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
        run_order("split_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
    ]
    ok = all(
        item["graph_capture"]["pass"]
        and item["parity"]["pass"]
        and item["split_wins"] == item["reps"]
        and item["delta_us"] < 0.0
        for item in orders
    )
    summary = {
        "claim": "qwen_nt_sidecar_graph_0ba235d",
        "sha": git_sha(),
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
