#!/usr/bin/env python3
"""CUDA launch smoke for the qwen NT lm-head sidecar boundary image."""

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


def expect_raises(fn, exc_type: type[BaseException]) -> bool:
    try:
        fn()
    except exc_type:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)

    model = build()
    launch_error = None
    try:
        flat = [ins for wave in model.prog.waves for ins in wave]
        cutpoints = list(model.qwen_nt_sidecar_cutpoints)
        boundary_rows = list(model.qwen_nt_sidecar_boundary_rows)
        cutpoint = cutpoints[0] if len(cutpoints) == 1 else None
        instr_index = int(cutpoint["instr_index"]) if cutpoint else -1
        op, ntiles, row_args = flat[instr_index] if instr_index >= 0 else (-1, -1, [])

        c_tensor = model.prog.bufs[int(row_args[2])] if row_args else None
        sentinel = -7.0
        write_summary = {
            "supported": bool(torch.is_tensor(c_tensor) and c_tensor.ndim == 2),
            "sentinel": sentinel,
            "overwritten_count": 0,
            "sentinel_count": 0,
            "finite_count": 0,
            "nan_count": 0,
            "tile_numel": 0,
            "max_finite_abs_delta_from_sentinel": 0.0,
        }

        if write_summary["supported"]:
            with torch.no_grad():
                tile = c_tensor[:256, :128]
                tile.fill_(sentinel)
                torch.cuda.synchronize()

        empty_range_guard = expect_raises(
            lambda: model.prog.run_qwen_nt_lmhead_sidecar(
                model.ext,
                model._smem_bytes,
                tile_start=0,
                tile_stop=0,
            ),
            ValueError,
        )
        out_of_range_guard = expect_raises(
            lambda: model.prog.run_qwen_nt_lmhead_sidecar(
                model.ext,
                model._smem_bytes,
                tile_start=0,
                tile_stop=4750,
            ),
            ValueError,
        )

        try:
            model.prog.run_qwen_nt_lmhead_sidecar(
                model.ext,
                model._smem_bytes,
                tile_start=0,
                tile_stop=1,
            )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            launch_error = repr(exc)

        if write_summary["supported"] and launch_error is None:
            with torch.no_grad():
                after = c_tensor[:256, :128].float()
                finite = torch.isfinite(after)
                sentinel_mask = after == sentinel
                delta = torch.abs(after - sentinel)
                finite_delta = delta[finite]
                write_summary.update(
                    {
                        "overwritten_count": int((~sentinel_mask).sum().item()),
                        "sentinel_count": int(sentinel_mask.sum().item()),
                        "finite_count": int(finite.sum().item()),
                        "nan_count": int(torch.isnan(after).sum().item()),
                        "tile_numel": int(after.numel()),
                        "max_finite_abs_delta_from_sentinel": (
                            float(finite_delta.max().item()) if int(finite_delta.numel()) > 0 else 0.0
                        ),
                    }
                )

        has_export = hasattr(model.ext, "run_qwen_nt_lmhead_sidecar")
        export_callable = callable(getattr(model.ext, "run_qwen_nt_lmhead_sidecar", None))
        row_summary = {
            "idx": instr_index,
            "op": int(op),
            "ntiles": int(ntiles),
            "flags": int(row_args[6]) if row_args else -1,
            "shape": {
                "M": int(row_args[3]) if row_args else -1,
                "N": int(row_args[4]) if row_args else -1,
                "K": int(row_args[5]) if row_args else -1,
            },
        }
        write_ok = (
            write_summary["supported"]
            and write_summary["overwritten_count"] == write_summary["tile_numel"]
            and write_summary["sentinel_count"] == 0
        )
        summary = {
            "pass": (
                launch_error is None
                and has_export
                and export_callable
                and empty_range_guard
                and out_of_range_guard
                and len(cutpoints) == 1
                and len(boundary_rows) == 1
                and row_summary
                == {
                    "idx": 37,
                    "op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
                    "ntiles": 4748,
                    "flags": 234899586,
                    "shape": {"M": 1024, "N": 151936, "K": 2560},
                }
                and write_ok
            ),
            "claim": "launch-and-one-tile-write-smoke-only; no parity or timing",
            "launch_error": launch_error,
            "launch_range": [0, 1],
            "empty_range_guard": empty_range_guard,
            "out_of_range_guard": out_of_range_guard,
            "so": model.ext.__file__,
            "has_export": has_export,
            "export_callable": export_callable,
            "n_instr": int(model.prog.n_instr),
            "critical_path": int(model.prog.critical_path),
            "gated": int(model.prog.n_gated),
            "smem_bytes": int(model._smem_bytes),
            "row": row_summary,
            "boundary_rows": boundary_rows,
            "cutpoint": cutpoint,
            "write": write_summary,
            "remaining_gap": (
                "pre-sidecar dependency closure and post-sidecar resume are not "
                "orchestrated here; this proof only launches the sidecar kernel "
                "against initialized model buffers and verifies tile 0 is overwritten; "
                "finite math is not expected until the producer closure runs first"
            ),
        }
    finally:
        del model
        torch.cuda.empty_cache()

    print("QWEN_NT_SIDECAR_BOUNDARY_LAUNCH_SMOKE " + json.dumps(summary, sort_keys=True))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
