#!/usr/bin/env python3
"""Validate graph replay through the opt-in qwen NT sidecar step policy."""

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

from model import MKQwen3  # noqa: E402
from qwen_nt_sidecar_api_0ba235d import (  # noqa: E402
    BOUNDARY_ENV,
    CFG,
    compare_grads,
    grad_stats,
    make_tokens,
    max_grad_stat_delta,
    restore_env,
    with_env,
)
from qwen_nt_sidecar_api_0ba235d import (
    route_summary as api_route_summary,
)


POLICY_ENV = "MK_QWEN_NT_SIDECAR_STEP"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_default() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: None, POLICY_ENV: None})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def build_boundary_no_policy() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: "1", POLICY_ENV: None})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def build_policy() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: "1", POLICY_ENV: "1"})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def route_summary(model: MKQwen3) -> dict[str, object]:
    summary = api_route_summary(model)
    summary["policy_requested"] = bool(getattr(model, "qwen_nt_sidecar_step_requested", False))
    return summary


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


def check_boundary_step_guard(
    boundary: MKQwen3,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    try:
        boundary.step(tokens, labels)
    except RuntimeError as exc:
        return {
            "pass": "sidecar boundary step requires" in str(exc),
            "error": str(exc),
        }
    raise RuntimeError("boundary sidecar model accepted plain step()")


def check_boundary_graph_guard(
    boundary: MKQwen3,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    try:
        boundary.make_graphed_step(tokens, labels)
    except RuntimeError as exc:
        return {
            "pass": "boundary graph capture requires" in str(exc),
            "error": str(exc),
        }
    raise RuntimeError("boundary sidecar model accepted make_graphed_step() without policy")


def check_policy_graph_equivalence(
    policy: MKQwen3,
    policy_replay: Callable[[], object],
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_atol: float = 1.0e-5,
    stat_atol: float = 5.0e-2,
    stat_rtol: float = 1.0e-4,
) -> dict[str, object]:
    policy_replay()
    torch.cuda.synchronize()
    graph_loss = float(policy.loss.item())
    graph_stats = grad_stats(policy)

    policy.step_qwen_nt_sidecar(tokens, labels)
    torch.cuda.synchronize()
    explicit_loss = float(policy.loss.item())
    explicit_stats = grad_stats(policy)

    stat_delta = max_grad_stat_delta(graph_stats, explicit_stats)
    loss_diff = explicit_loss - graph_loss
    stat_ok = float(stat_delta["max_abs_delta"]) <= stat_atol or float(stat_delta["max_rel_delta"]) <= stat_rtol
    ok = abs(loss_diff) <= loss_atol and stat_ok
    return {
        "pass": ok,
        "loss_atol": loss_atol,
        "stat_atol": stat_atol,
        "stat_rtol": stat_rtol,
        "graph_loss": graph_loss,
        "explicit_loss": explicit_loss,
        "loss_diff": loss_diff,
        "grad_stat_delta": stat_delta,
    }


def check_graph_parity(
    default: MKQwen3,
    policy: MKQwen3,
    default_replay: Callable[[], object],
    policy_replay: Callable[[], object],
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    default_replay()
    policy_replay()
    torch.cuda.synchronize()
    default_loss = float(default.loss.item())
    policy_loss = float(policy.loss.item())
    grad = compare_grads(default, policy)
    loss_diff = policy_loss - default_loss
    ok = abs(loss_diff) < loss_atol and float(grad["worst_grad_rel"]) < grad_rtol
    return {
        "pass": ok,
        "loss_atol": loss_atol,
        "grad_rtol": grad_rtol,
        "default_loss": default_loss,
        "policy_loss": policy_loss,
        "loss_diff": loss_diff,
        "grad": grad,
    }


def run_order(
    order: str,
    reps: int,
    warmup: int,
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    tokens, labels = make_tokens()
    if order == "default_first":
        default = build_default()
        boundary = build_boundary_no_policy()
        policy = build_policy()
    elif order == "policy_first":
        policy = build_policy()
        boundary = build_boundary_no_policy()
        default = build_default()
    else:
        raise ValueError(f"bad order {order!r}")

    routes = {
        "default": route_summary(default),
        "boundary": route_summary(boundary),
        "policy": route_summary(policy),
    }
    print("ROUTE_JSON " + json.dumps({"order": order, **routes}, sort_keys=True), flush=True)
    if routes["default"]["api_available"] or routes["default"]["policy_requested"]:
        raise RuntimeError("default route unexpectedly has qwen NT sidecar policy state")
    if not routes["boundary"]["api_available"] or routes["boundary"]["policy_requested"]:
        raise RuntimeError("boundary route did not expose API without policy")
    if not routes["policy"]["api_available"] or not routes["policy"]["policy_requested"]:
        raise RuntimeError("policy route did not expose the sidecar graph policy")

    boundary_step_guard = check_boundary_step_guard(boundary, tokens, labels)
    print(
        "BOUNDARY_STEP_GUARD_JSON " + json.dumps({"order": order, **boundary_step_guard}, sort_keys=True),
        flush=True,
    )
    if not boundary_step_guard["pass"]:
        raise RuntimeError(f"boundary step guard failed: {boundary_step_guard}")

    boundary_graph_guard = check_boundary_graph_guard(boundary, tokens, labels)
    print(
        "BOUNDARY_GRAPH_GUARD_JSON " + json.dumps({"order": order, **boundary_graph_guard}, sort_keys=True),
        flush=True,
    )
    if not boundary_graph_guard["pass"]:
        raise RuntimeError(f"boundary graph guard failed: {boundary_graph_guard}")

    default_replay = default.make_graphed_step(tokens, labels, warmup=warmup)
    policy_replay = policy.make_graphed_step(tokens, labels, warmup=warmup)
    graph_capture = {
        "pass": True,
        "default_has_graph": hasattr(default_replay, "graph"),
        "policy_has_graph": hasattr(policy_replay, "graph"),
    }
    print("GRAPH_CAPTURE_JSON " + json.dumps({"order": order, **graph_capture}, sort_keys=True), flush=True)

    policy_equivalence = check_policy_graph_equivalence(policy, policy_replay, tokens, labels)
    print(
        "POLICY_GRAPH_EQUIV_JSON " + json.dumps({"order": order, **policy_equivalence}, sort_keys=True),
        flush=True,
    )
    if not policy_equivalence["pass"]:
        raise RuntimeError(f"policy graph/API equivalence failed: {policy_equivalence}")

    parity = check_graph_parity(
        default,
        policy,
        default_replay,
        policy_replay,
        loss_atol,
        grad_rtol,
    )
    print("GRAPH_PARITY_JSON " + json.dumps({"order": order, **parity}, sort_keys=True), flush=True)
    if not parity["pass"]:
        raise RuntimeError(f"graph parity failed before timing: {parity}")

    for _ in range(warmup):
        default_replay()
        policy_replay()
    torch.cuda.synchronize()

    default_times: list[float] = []
    policy_times: list[float] = []
    policy_wins = 0
    for _ in range(reps):
        td = time_call(default_replay)
        tp = time_call(policy_replay)
        default_times.append(td)
        policy_times.append(tp)
        policy_wins += int(tp < td)

    default_stats = timing_stats(default_times)
    policy_stats = timing_stats(policy_times)
    result = {
        "order": order,
        "reps": reps,
        "warmup": warmup,
        "boundary_step_guard": boundary_step_guard,
        "boundary_graph_guard": boundary_graph_guard,
        "graph_capture": graph_capture,
        "policy_equivalence": policy_equivalence,
        "parity": parity,
        "default": default_stats,
        "policy": policy_stats,
        "delta_us": policy_stats["median_us"] - default_stats["median_us"],
        "policy_wins": policy_wins,
        "default_times_us": default_times,
        "policy_times_us": policy_times,
        "routes": routes,
    }
    print("GRAPH_POLICY_TIMING_JSON " + json.dumps(result, sort_keys=True), flush=True)
    del default_replay, policy_replay, default, boundary, policy
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
        run_order("policy_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
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
        "claim": "qwen_nt_sidecar_graph_policy_0ba235d",
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
