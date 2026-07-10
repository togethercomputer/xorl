#!/usr/bin/env python3
"""Validate no-env qwen NT sidecar defaults for exact qwen4b L1/L2."""

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

from model import Cfg, MKQwen3  # noqa: E402
from qwen_nt_sidecar_api_0ba235d import (  # noqa: E402
    BOUNDARY_ENV,
    compare_grads,
    grad_stats,
    max_grad_stat_delta,
    restore_env,
    with_env,
)
from qwen_nt_sidecar_api_0ba235d import (
    route_summary as api_route_summary,
)
from qwen_nt_sidecar_graph_policy_0ba235d import POLICY_ENV  # noqa: E402


CFG_L1 = Cfg(H=2560, L=1, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
CFG_L2 = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_tokens(cfg: Cfg) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.V, (cfg.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100
    return tokens, labels


def build_promoted(cfg: Cfg) -> MKQwen3:
    old = with_env({BOUNDARY_ENV: None, POLICY_ENV: None})
    try:
        return MKQwen3(cfg, seed=0)
    finally:
        restore_env(old)


def build_forced_old(cfg: Cfg) -> MKQwen3:
    old = with_env({BOUNDARY_ENV: "0", POLICY_ENV: None})
    try:
        return MKQwen3(cfg, seed=0)
    finally:
        restore_env(old)


def build_policy_off_guard(cfg: Cfg) -> MKQwen3:
    old = with_env({BOUNDARY_ENV: None, POLICY_ENV: "0"})
    try:
        return MKQwen3(cfg, seed=0)
    finally:
        restore_env(old)


def route_summary(model: MKQwen3) -> dict[str, object]:
    summary = api_route_summary(model)
    summary["policy_requested"] = bool(getattr(model, "qwen_nt_sidecar_step_requested", False))
    return summary


def check_policy_off_guards(
    model: MKQwen3,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    step_error = ""
    graph_error = ""
    try:
        model.step(tokens, labels)
    except RuntimeError as exc:
        step_error = str(exc)
    try:
        model.make_graphed_step(tokens, labels)
    except RuntimeError as exc:
        graph_error = str(exc)
    return {
        "pass": ("sidecar boundary step requires" in step_error and "boundary graph capture requires" in graph_error),
        "step_error": step_error,
        "graph_error": graph_error,
    }


def max_grad_stat_delta_for_models(a: MKQwen3, b: MKQwen3) -> dict[str, object]:
    return max_grad_stat_delta(grad_stats(a), grad_stats(b))


def check_step_equivalence(
    promoted: MKQwen3,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_atol: float = 1.0e-5,
    stat_atol: float = 5.0e-2,
    stat_rtol: float = 1.0e-4,
) -> dict[str, object]:
    promoted.step(tokens, labels)
    torch.cuda.synchronize()
    step_loss = float(promoted.loss.item())
    step_stats = grad_stats(promoted)

    promoted.step_qwen_nt_sidecar(tokens, labels)
    torch.cuda.synchronize()
    explicit_loss = float(promoted.loss.item())
    explicit_stats = grad_stats(promoted)

    stat_delta = max_grad_stat_delta(step_stats, explicit_stats)
    loss_diff = explicit_loss - step_loss
    stat_ok = float(stat_delta["max_abs_delta"]) <= stat_atol or float(stat_delta["max_rel_delta"]) <= stat_rtol
    return {
        "pass": abs(loss_diff) <= loss_atol and stat_ok,
        "loss_atol": loss_atol,
        "stat_atol": stat_atol,
        "stat_rtol": stat_rtol,
        "step_loss": step_loss,
        "explicit_loss": explicit_loss,
        "loss_diff": loss_diff,
        "grad_stat_delta": stat_delta,
    }


def check_step_parity(
    old: MKQwen3,
    promoted: MKQwen3,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    old.step(tokens, labels)
    promoted.step(tokens, labels)
    torch.cuda.synchronize()
    old_loss = float(old.loss.item())
    promoted_loss = float(promoted.loss.item())
    grad = compare_grads(old, promoted)
    loss_diff = promoted_loss - old_loss
    return {
        "pass": abs(loss_diff) < loss_atol and float(grad["worst_grad_rel"]) < grad_rtol,
        "loss_atol": loss_atol,
        "grad_rtol": grad_rtol,
        "old_loss": old_loss,
        "promoted_loss": promoted_loss,
        "loss_diff": loss_diff,
        "grad": grad,
    }


def check_graph_parity(
    old: MKQwen3,
    promoted: MKQwen3,
    old_replay: Callable[[], object],
    promoted_replay: Callable[[], object],
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    old_replay()
    promoted_replay()
    torch.cuda.synchronize()
    old_loss = float(old.loss.item())
    promoted_loss = float(promoted.loss.item())
    grad = compare_grads(old, promoted)
    loss_diff = promoted_loss - old_loss
    return {
        "pass": abs(loss_diff) < loss_atol and float(grad["worst_grad_rel"]) < grad_rtol,
        "loss_atol": loss_atol,
        "grad_rtol": grad_rtol,
        "old_loss": old_loss,
        "promoted_loss": promoted_loss,
        "loss_diff": loss_diff,
        "grad": grad,
    }


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


def time_pair(
    old_fn: Callable[[], object],
    promoted_fn: Callable[[], object],
    reps: int,
) -> dict[str, object]:
    old_times: list[float] = []
    promoted_times: list[float] = []
    promoted_wins = 0
    for _ in range(reps):
        old_t = time_call(old_fn)
        promoted_t = time_call(promoted_fn)
        old_times.append(old_t)
        promoted_times.append(promoted_t)
        promoted_wins += int(promoted_t < old_t)
    old_stats = timing_stats(old_times)
    promoted_stats = timing_stats(promoted_times)
    return {
        "old": old_stats,
        "promoted": promoted_stats,
        "delta_us": promoted_stats["median_us"] - old_stats["median_us"],
        "promoted_wins": promoted_wins,
        "old_times_us": old_times,
        "promoted_times_us": promoted_times,
    }


def run_case(
    shape: str,
    cfg: Cfg,
    order: str,
    reps: int,
    warmup: int,
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    tokens, labels = make_tokens(cfg)
    if order == "old_first":
        old = build_forced_old(cfg)
        promoted = build_promoted(cfg)
        guard = build_policy_off_guard(cfg)
    elif order == "promoted_first":
        promoted = build_promoted(cfg)
        guard = build_policy_off_guard(cfg)
        old = build_forced_old(cfg)
    else:
        raise ValueError(f"bad order {order!r}")

    routes = {
        "old": route_summary(old),
        "promoted": route_summary(promoted),
        "policy_off_guard": route_summary(guard),
    }
    print(
        "ROUTE_JSON " + json.dumps({"shape": shape, "order": order, **routes}, sort_keys=True),
        flush=True,
    )
    if routes["old"]["api_available"] or routes["old"]["policy_requested"]:
        raise RuntimeError(f"{shape} forced-old route unexpectedly has sidecar policy")
    if not routes["promoted"]["api_available"] or not routes["promoted"]["policy_requested"]:
        raise RuntimeError(f"{shape} promoted default did not enable sidecar policy")
    if not routes["policy_off_guard"]["api_available"] or routes["policy_off_guard"]["policy_requested"]:
        raise RuntimeError(f"{shape} policy-off guard route has wrong availability")

    guard_result = check_policy_off_guards(guard, tokens, labels)
    print(
        "POLICY_OFF_GUARD_JSON " + json.dumps({"shape": shape, "order": order, **guard_result}, sort_keys=True),
        flush=True,
    )
    if not guard_result["pass"]:
        raise RuntimeError(f"{shape} policy-off guard failed: {guard_result}")

    step_equivalence = check_step_equivalence(promoted, tokens, labels)
    print(
        "STEP_EQUIV_JSON " + json.dumps({"shape": shape, "order": order, **step_equivalence}, sort_keys=True),
        flush=True,
    )
    if not step_equivalence["pass"]:
        raise RuntimeError(f"{shape} default step/API equivalence failed: {step_equivalence}")

    step_parity = check_step_parity(old, promoted, tokens, labels, loss_atol, grad_rtol)
    print(
        "STEP_PARITY_JSON " + json.dumps({"shape": shape, "order": order, **step_parity}, sort_keys=True),
        flush=True,
    )
    if not step_parity["pass"]:
        raise RuntimeError(f"{shape} step parity failed: {step_parity}")

    for _ in range(warmup):
        old.step(tokens, labels)
        promoted.step(tokens, labels)
    torch.cuda.synchronize()
    step_timing = time_pair(
        lambda: old.step(tokens, labels),
        lambda: promoted.step(tokens, labels),
        reps,
    )
    print(
        "STEP_TIMING_JSON " + json.dumps({"shape": shape, "order": order, **step_timing}, sort_keys=True),
        flush=True,
    )

    old_replay = old.make_graphed_step(tokens, labels, warmup=warmup)
    promoted_replay = promoted.make_graphed_step(tokens, labels, warmup=warmup)
    graph_capture = {
        "pass": True,
        "old_has_graph": hasattr(old_replay, "graph"),
        "promoted_has_graph": hasattr(promoted_replay, "graph"),
    }
    print(
        "GRAPH_CAPTURE_JSON " + json.dumps({"shape": shape, "order": order, **graph_capture}, sort_keys=True),
        flush=True,
    )

    graph_parity = check_graph_parity(
        old,
        promoted,
        old_replay,
        promoted_replay,
        loss_atol,
        grad_rtol,
    )
    print(
        "GRAPH_PARITY_JSON " + json.dumps({"shape": shape, "order": order, **graph_parity}, sort_keys=True),
        flush=True,
    )
    if not graph_parity["pass"]:
        raise RuntimeError(f"{shape} graph parity failed: {graph_parity}")

    for _ in range(warmup):
        old_replay()
        promoted_replay()
    torch.cuda.synchronize()
    graph_timing = time_pair(old_replay, promoted_replay, reps)
    print(
        "GRAPH_TIMING_JSON " + json.dumps({"shape": shape, "order": order, **graph_timing}, sort_keys=True),
        flush=True,
    )

    result = {
        "shape": shape,
        "order": order,
        "reps": reps,
        "warmup": warmup,
        "routes": routes,
        "policy_off_guard": guard_result,
        "step_equivalence": step_equivalence,
        "step_parity": step_parity,
        "step_timing": step_timing,
        "graph_capture": graph_capture,
        "graph_parity": graph_parity,
        "graph_timing": graph_timing,
    }
    del old_replay, promoted_replay, old, promoted, guard
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
    cases = []
    for shape, cfg in (("qwen4b-l1", CFG_L1), ("qwen4b-l2", CFG_L2)):
        cases.append(run_case(shape, cfg, "old_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol))
        cases.append(run_case(shape, cfg, "promoted_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol))
    ok = all(
        item["policy_off_guard"]["pass"]
        and item["step_equivalence"]["pass"]
        and item["step_parity"]["pass"]
        and item["step_timing"]["promoted_wins"] == item["reps"]
        and item["step_timing"]["delta_us"] < 0.0
        and item["graph_capture"]["pass"]
        and item["graph_capture"]["old_has_graph"]
        and item["graph_capture"]["promoted_has_graph"]
        and item["graph_parity"]["pass"]
        and item["graph_timing"]["promoted_wins"] == item["reps"]
        and item["graph_timing"]["delta_us"] < 0.0
        for item in cases
    )
    summary = {
        "claim": "qwen_nt_sidecar_default_policy_0ba235d",
        "sha": git_sha(),
        "pass": ok,
        "cases": cases,
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
