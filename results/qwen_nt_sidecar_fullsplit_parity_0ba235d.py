#!/usr/bin/env python3
"""Qwen NT sidecar full split parity proof."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
sys.path.insert(0, str(MKDIR))

import mk  # noqa: E402
from model import Cfg, MKQwen3  # noqa: E402


CFG = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
BOUNDARY_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY"


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


def route_summary(model: MKQwen3) -> dict[str, object]:
    return {
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": int(model._smem_bytes),
        "cutpoint_count": len(getattr(model, "qwen_nt_sidecar_cutpoints", [])),
        "boundary_row_count": len(getattr(model, "qwen_nt_sidecar_boundary_rows", [])),
    }


def split_step(model: MKQwen3, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if model._inputs_bound_external:
        model.prog._buftab[model._tokens_buf] = model.tokens.data_ptr()
        model.prog._buftab[model._labels_buf] = model.labels.data_ptr()
        model._inputs_bound_external = False
    model.tokens.copy_(tokens)
    model.labels.copy_(labels)
    if not model.in_kernel_inv_valid:
        model.inv_valid.copy_(1.0 / (labels >= 0).sum().clamp(min=1).float().reshape(1))
    subs = model.prog.qwen_nt_sidecar_pdf_subprograms()
    model.prog.run_qwen_nt_sidecar_prefix(model.ext, model._smem_bytes, subs)
    model.prog.run_qwen_nt_lmhead_sidecar(model.ext, model._smem_bytes)
    model.prog.run_qwen_nt_sidecar_post(model.ext, model._smem_bytes, subs)
    return model.loss


def clone_grads_cpu(model: MKQwen3) -> dict[str, torch.Tensor]:
    return {name: grad.detach().cpu().clone() for name, grad in model.grads.items()}


def compare_grads(ref: dict[str, torch.Tensor], split: MKQwen3) -> dict[str, object]:
    worst_rel = 0.0
    worst_abs = 0.0
    worst_name = ""
    per_grad: dict[str, dict[str, float]] = {}
    for name, ref_grad in ref.items():
        split_grad = split.grads[name].detach().cpu()
        ga = ref_grad.float()
        gb = split_grad.float()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--loss-atol", type=float, default=5.0e-3)
    parser.add_argument("--grad-rtol", type=float, default=5.0e-2)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1)
    tokens = torch.randint(0, CFG.V, (CFG.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100

    default = build_default()
    default_route = route_summary(default)
    default.step(tokens, labels)
    torch.cuda.synchronize()
    default_loss = float(default.loss.item())
    ref_grads = clone_grads_cpu(default)
    del default
    torch.cuda.empty_cache()

    split = build_split()
    split_route = route_summary(split)
    split_error = None
    try:
        split_step(split, tokens, labels)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        split_error = repr(exc)
    split_loss = float(split.loss.item()) if split_error is None else None
    grad_cmp = compare_grads(ref_grads, split) if split_error is None else {
        "worst_grad_rel": float("inf"),
        "worst_grad_abs": float("inf"),
        "worst_grad_name": "",
        "per_grad": {},
    }
    loss_diff = None if split_loss is None else split_loss - default_loss
    pass_gate = (
        split_error is None
        and loss_diff is not None
        and abs(loss_diff) < args.loss_atol
        and float(grad_cmp["worst_grad_rel"]) < args.grad_rtol
    )
    summary = {
        "pass": pass_gate,
        "claim": "full sidecar split parity; no timing",
        "split_error": split_error,
        "loss_atol": args.loss_atol,
        "grad_rtol": args.grad_rtol,
        "default_loss": default_loss,
        "split_loss": split_loss,
        "loss_diff": loss_diff,
        "grad": grad_cmp,
        "default_route": default_route,
        "split_route": split_route,
        "split": {
            "prefix_indices": list(range(37)),
            "sidecar_tile_range": [0, 4748],
            "post_indices": list(range(38, 78)),
        },
    }
    print("QWEN_NT_SIDECAR_FULLSPLIT_PARITY " + json.dumps(summary, sort_keys=True))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
