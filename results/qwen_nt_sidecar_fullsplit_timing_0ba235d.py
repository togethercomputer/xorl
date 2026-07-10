#!/usr/bin/env python3
"""Timing baseline for the qwen NT full sidecar split protocol."""

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
sys.path.insert(0, str(MKDIR))

from model import Cfg, MKQwen3  # noqa: E402


CFG = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
BOUNDARY_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def with_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in updates}
    for k, v in updates.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def restore_env(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def build_default() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: None})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def build_split() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: "1"})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def make_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    tokens = torch.randint(0, CFG.V, (CFG.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100
    return tokens, labels


def route_summary(model: MKQwen3) -> dict[str, object]:
    subs = None
    if getattr(model, "qwen_nt_sidecar_cutpoints", []):
        subs = model.prog.qwen_nt_sidecar_pdf_subprograms()
    cutpoints = getattr(model, "qwen_nt_sidecar_cutpoints", [])
    return {
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": int(model._smem_bytes),
        "default_mode": str(getattr(model, "default_mode", "")),
        "cutpoint_count": len(cutpoints),
        "boundary_row_count": len(getattr(model, "qwen_nt_sidecar_boundary_rows", [])),
        "prefix_n_instr": None if subs is None else int(subs["prefix"]["n_instr"]),
        "post_n_instr": None if subs is None else int(subs["post"]["n_instr"]),
        "sidecar_tile_range": None if not cutpoints else [0, int(cutpoints[0]["ntiles"])],
    }


def split_step(
    model: MKQwen3,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    subprograms: dict[str, object],
) -> torch.Tensor:
    if model._inputs_bound_external:
        model.prog._buftab[model._tokens_buf] = model.tokens.data_ptr()
        model.prog._buftab[model._labels_buf] = model.labels.data_ptr()
        model._inputs_bound_external = False
    model.tokens.copy_(tokens)
    model.labels.copy_(labels)
    if not model.in_kernel_inv_valid:
        model.inv_valid.copy_(1.0 / (labels >= 0).sum().clamp(min=1).float().reshape(1))
    model.prog.run_qwen_nt_sidecar_prefix(model.ext, model._smem_bytes, subprograms)
    model.prog.run_qwen_nt_lmhead_sidecar(model.ext, model._smem_bytes)
    model.prog.run_qwen_nt_sidecar_post(model.ext, model._smem_bytes, subprograms)
    return model.loss


def compare_grads(default: MKQwen3, split: MKQwen3) -> dict[str, object]:
    worst_rel = 0.0
    worst_abs = 0.0
    worst_name = ""
    per_grad: dict[str, dict[str, float]] = {}
    for name, ref_grad in default.grads.items():
        ga = ref_grad.float()
        gb = split.grads[name].float()
        abs_diff = float((ga - gb).abs().max().item())
        denom = float(ga.abs().max().item())
        rel = abs_diff / denom if denom >= 1.0e-8 else 0.0
        per_grad[name] = {"abs": abs_diff, "rel": rel}
        if rel > worst_rel:
            worst_rel = rel
            worst_abs = abs_diff
            worst_name = name
    return {
        "worst_grad_rel": worst_rel,
        "worst_grad_abs": worst_abs,
        "worst_grad_name": worst_name,
        "per_grad": per_grad,
    }


def check_parity(
    default: MKQwen3,
    split: MKQwen3,
    split_subs: dict[str, object],
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_atol: float,
    grad_rtol: float,
) -> dict[str, object]:
    default.step(tokens, labels)
    split_step(split, tokens, labels, split_subs)
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

    split_subs = split.prog.qwen_nt_sidecar_pdf_subprograms()
    routes = {
        "default": route_summary(default),
        "split": route_summary(split),
    }
    print("ROUTE_JSON " + json.dumps({"order": order, **routes}, sort_keys=True), flush=True)
    if int(routes["default"]["cutpoint_count"]) != 0:
        raise RuntimeError("default route unexpectedly has a sidecar cutpoint")
    if int(routes["split"]["cutpoint_count"]) != 1:
        raise RuntimeError("split route did not produce exactly one sidecar cutpoint")

    parity = check_parity(default, split, split_subs, tokens, labels, loss_atol, grad_rtol)
    print("PARITY_JSON " + json.dumps({"order": order, **parity}, sort_keys=True), flush=True)
    if not parity["pass"]:
        raise RuntimeError(f"parity failed before timing: {parity}")

    for _ in range(warmup):
        default.step(tokens, labels)
        split_step(split, tokens, labels, split_subs)
    torch.cuda.synchronize()

    default_times: list[float] = []
    split_times: list[float] = []
    split_wins = 0
    for _ in range(reps):
        td = time_call(lambda: default.step(tokens, labels))
        ts = time_call(lambda: split_step(split, tokens, labels, split_subs))
        default_times.append(td)
        split_times.append(ts)
        split_wins += int(ts < td)

    default_stats = timing_stats(default_times)
    split_stats = timing_stats(split_times)
    result = {
        "order": order,
        "reps": reps,
        "warmup": warmup,
        "default": default_stats,
        "split": split_stats,
        "delta_us": split_stats["median_us"] - default_stats["median_us"],
        "split_wins": split_wins,
        "default_times_us": default_times,
        "split_times_us": split_times,
        "routes": routes,
        "parity": parity,
    }
    print("TIMING_JSON " + json.dumps(result, sort_keys=True), flush=True)
    del default, split
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
    results = [
        run_order("default_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
        run_order("split_first", args.reps, args.warmup, args.loss_atol, args.grad_rtol),
    ]
    summary = {
        "pass": all(r["parity"]["pass"] for r in results),
        "claim": "qwen4b-l2 full sidecar split timing baseline; no promotion",
        "sha": git_sha(),
        "cfg": {
            "H": CFG.H,
            "L": CFG.L,
            "nq": CFG.nq,
            "nkv": CFG.nkv,
            "D": CFG.D,
            "I": CFG.I,
            "V": CFG.V,
            "S": CFG.S,
        },
        "orders": results,
    }
    print("QWEN_NT_SIDECAR_FULLSPLIT_TIMING " + json.dumps(summary, sort_keys=True), flush=True)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
