#!/usr/bin/env python3
"""Compile/import proof for the qwen NT lm-head sidecar boundary row."""

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


def build() -> MKQwen3:
    old = with_env({BOUNDARY_ENV: "1"})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    model = build()
    try:
        name = Path(model.ext.__file__).name
        flat = [ins for wave in model.prog.waves for ins in wave]
        head_rows = [
            {"idx": i, "op": int(op), "ntiles": int(ntiles), "flags": int(row_args[6])}
            for i, (op, ntiles, row_args) in enumerate(flat)
            if op in (mk.OP_GEMM, mk.OP_QWEN_NT_SIDECAR_BOUNDARY)
            and int(row_args[3]) == 1024
            and int(row_args[4]) == 151936
            and int(row_args[5]) == 2560
            and (int(row_args[6]) & 2)
        ]
        has_export = hasattr(model.ext, "run_qwen_nt_lmhead_sidecar")
        export_callable = callable(getattr(model.ext, "run_qwen_nt_lmhead_sidecar", None))
        summary = {
            "pass": (
                has_export
                and export_callable
                and "_ntscbnd" in name
                and len(model.qwen_nt_sidecar_boundary_rows) == 1
                and len(model.qwen_nt_sidecar_cutpoints) == 1
                and head_rows == [{
                    "idx": 37,
                    "op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
                    "ntiles": 4748,
                    "flags": 234899586,
                }]
                and model.qwen_nt_sidecar_split_plan["main_row_replaced_by_boundary"]
            ),
            "name": name,
            "so": model.ext.__file__,
            "has_export": has_export,
            "export_callable": export_callable,
            "n_instr": int(model.prog.n_instr),
            "critical_path": int(model.prog.critical_path),
            "gated": int(model.prog.n_gated),
            "smem_bytes": model._smem_bytes,
            "head_rows": head_rows,
            "boundary_rows": model.qwen_nt_sidecar_boundary_rows,
            "cutpoint_count": len(model.qwen_nt_sidecar_cutpoints),
            "cutpoint": model.qwen_nt_sidecar_cutpoints[0],
            "split_plan": model.qwen_nt_sidecar_split_plan,
        }
    finally:
        del model
        torch.cuda.empty_cache()

    print("QWEN_NT_SIDECAR_BOUNDARY_BUILD_SUMMARY " + json.dumps(summary, sort_keys=True))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
