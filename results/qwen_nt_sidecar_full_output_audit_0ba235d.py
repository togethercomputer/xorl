#!/usr/bin/env python3
"""Audit qwen NT sidecar full-output finiteness before post subprogram."""

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


def build_split() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: "1"})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def bad_location_2d(bad: torch.Tensor) -> list[int] | None:
    if not bool(bad.any().item()):
        return None
    row_mask = bad.any(dim=1)
    row = int(torch.argmax(row_mask.to(torch.int32)).item())
    col = int(torch.argmax(bad[row].to(torch.int32)).item())
    return [row, col]


def tensor_stats(tensor: torch.Tensor, sentinel: float) -> dict[str, object]:
    view = tensor.detach()
    finite = torch.isfinite(view)
    sentinel_mask = view.float() == sentinel
    bad = (~finite) | sentinel_mask
    total = int(view.numel())
    stats: dict[str, object] = {
        "shape": list(view.shape),
        "dtype": str(view.dtype),
        "numel": total,
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(view).sum().item()),
        "inf_count": int(torch.isinf(view).sum().item()),
        "sentinel_count": int(sentinel_mask.sum().item()),
        "bad_count": int(bad.sum().item()),
    }
    if view.ndim == 2:
        loc = bad_location_2d(bad)
        stats["first_bad_2d"] = loc
        if loc is not None:
            row, col = loc
            stats["first_bad_value"] = float(view[row, col].float().item())
            stats["first_bad_tile"] = [row // 256, col // 128]
    elif stats["bad_count"]:
        flat_bad = bad.reshape(-1)
        idx = int(torch.argmax(flat_bad.to(torch.int32)).item())
        stats["first_bad_flat"] = idx
        stats["first_bad_value"] = float(view.reshape(-1)[idx].float().item())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1)
    tokens = torch.randint(0, CFG.V, (CFG.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100

    model = build_split()
    prefix_error = None
    sidecar_error = None
    try:
        subs = model.prog.qwen_nt_sidecar_pdf_subprograms()
        cutpoint = subs["cutpoint"]
        logits = model.prog.bufs[int(cutpoint["output_bufs"]["logits"])]
        lse_parts = model.prog.bufs[int(cutpoint["output_bufs"]["lse_parts"])]
        model.tokens.copy_(tokens)
        model.labels.copy_(labels)
        if not model.in_kernel_inv_valid:
            model.inv_valid.copy_(1.0 / (labels >= 0).sum().clamp(min=1).float().reshape(1))
        try:
            model.prog.run_qwen_nt_sidecar_prefix(model.ext, model._smem_bytes, subs)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            prefix_error = repr(exc)

        sentinel = -7.0
        if prefix_error is None:
            with torch.no_grad():
                logits.fill_(sentinel)
                lse_parts.fill_(sentinel)
                torch.cuda.synchronize()
            try:
                model.prog.run_qwen_nt_lmhead_sidecar(model.ext, model._smem_bytes)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                sidecar_error = repr(exc)

        logits_stats = tensor_stats(logits, sentinel) if sidecar_error is None else {}
        lse_stats = tensor_stats(lse_parts, sentinel) if sidecar_error is None else {}
        summary = {
            "pass": (
                prefix_error is None
                and sidecar_error is None
                and logits_stats.get("bad_count") == 0
                and lse_stats.get("bad_count") == 0
            ),
            "claim": "prefix plus full sidecar output audit; no post subprogram",
            "prefix_error": prefix_error,
            "sidecar_error": sidecar_error,
            "sidecar_tile_range": [0, int(cutpoint["ntiles"])],
            "logits": logits_stats,
            "lse_parts": lse_stats,
            "post_not_run": True,
        }
    finally:
        del model
        torch.cuda.empty_cache()

    print("QWEN_NT_SIDECAR_FULL_OUTPUT_AUDIT " + json.dumps(summary, sort_keys=True))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
