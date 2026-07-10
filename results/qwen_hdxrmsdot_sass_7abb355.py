#!/usr/bin/env python3
"""SASS/resource gate for qwen head-DX -> first-RMS dot partials."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
sys.path.insert(0, str(MKDIR))

import mk  # noqa: E402
from model import Cfg, MKQwen3  # noqa: E402


CFG = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
PROBE_ENV = "MK_QWEN_HEADDX_RMS_DOT_PARTIALS"
UNSET_FOR_COMPARISON = (
    "MK_MODE",
    "MK_QWEN_NT_SIDECAR_STEP",
    "MK_RMS_DX_H2560",
    "MK_GEMM_N256_HEAD_DX_EXACT",
    "MK_GEMM_N256_HEAD_DX_PDFONLY",
    "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY",
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


def with_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return old


def restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def build_model(enabled: bool) -> MKQwen3:
    updates = dict.fromkeys(UNSET_FOR_COMPARISON)
    updates[PROBE_ENV] = "1" if enabled else "0"
    old = with_env(updates)
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def run_text(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def function_resusage(text: str, needle: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if f"Function {needle}:" in line or f"Function : {needle}" in line or needle in line:
            for nxt in lines[idx + 1 : idx + 12]:
                if "REG:" in nxt:
                    return nxt.strip()
    return ""


def parse_resusage(line: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in ("REG", "STACK", "SHARED", "LOCAL"):
        match = re.search(rf"{key}:([0-9]+)", line)
        if match:
            out[key.lower()] = int(match.group(1))
    return out


def sass_counts(text: str) -> dict[str, int]:
    needles = [
        "HGMMA",
        "WARPGROUP.DEPBAR",
        "WARPGROUP.ARRIVE",
        "DEPBAR",
        "CALL",
        "LDG",
        "STG",
        "LDSM",
        "LDL",
        "STL",
        "FFMA",
        "FMUL",
        "FADD",
        "F2F",
        "MUFU",
        "R2UR",
    ]
    return {needle: sum(1 for line in text.splitlines() if needle in line) for needle in needles}


def route_summary(model: MKQwen3, enabled: bool) -> dict[str, Any]:
    flat = [ins for wave in model.prog.waves for ins in wave]
    head_rows = []
    rms_dx_rows = []
    for idx, (op, ntiles, args) in enumerate(flat):
        if (
            op == mk.OP_GEMM
            and int(args[3]) == 1024
            and int(args[4]) == 2560
            and int(args[5]) == 151936
            and (int(args[6]) & 8)
        ):
            flags = int(args[6])
            head_rows.append(
                {
                    "idx": idx,
                    "op": int(op),
                    "ntiles": int(ntiles),
                    "flags": flags,
                    "has_rmsdot_flag": bool(flags & mk.GEMM_HEADDX_RMSDOT_FLAG),
                    "arg9_partials": int(args[9]) if len(args) > 9 else 0,
                    "arg10_nparts": int(args[10]) if len(args) > 10 else 0,
                    "arg11_x": int(args[11]) if len(args) > 11 else 0,
                    "arg12_wf": int(args[12]) if len(args) > 12 else 0,
                    "arg20_tmap_table_plus1": int(args[20]) if len(args) > 20 else 0,
                }
            )
        if op == mk.OP_RMSNORM_BWD_DX and len(args) > 8 and int(args[6]) == 2560:
            rms_dx_rows.append(
                {
                    "idx": idx,
                    "op": int(op),
                    "ntiles": int(ntiles),
                    "dy_f32": int(args[7]),
                    "S": int(args[8]),
                    "has_partials": len(args) > 10 and bool(args[9]),
                    "arg9_partials": int(args[9]) if len(args) > 9 else 0,
                    "arg10_nparts": int(args[10]) if len(args) > 10 else 0,
                }
            )
    name = Path(model.ext.__file__).name
    return {
        "enabled": enabled,
        "name": name,
        "so": model.ext.__file__,
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": int(model._smem_bytes),
        "default_mode": model.default_mode,
        "sidecar_available": bool(model.qwen_nt_sidecar_step_available()),
        "sidecar_boundary_rows": len(model.qwen_nt_sidecar_boundary_rows),
        "head_dx_rows": head_rows,
        "rms_dx_rows": rms_dx_rows,
        "has_ntscbnd_suffix": "_ntscbnd" in name,
        "has_hdx_exact_pdf_suffix": "_hdxexpdf" in name,
        "has_hdxrmsdot_suffix": "_hdxrmsdot" in name,
    }


def dump_sass(label: str, model: MKQwen3, out_dir: Path, stamp: str) -> dict[str, Any]:
    so = model.ext.__file__
    sha = git_sha()
    prefix = f"qwen-hdxrmsdot-{label}-{sha}-{stamp}"
    res_text = run_text(["cuobjdump", "-res-usage", so])
    pdf_sass = run_text(["cuobjdump", "-sass", "-fun", "megakernel_pdf", so])
    full_sass = run_text(["cuobjdump", "-sass", so])
    res_path = out_dir / f"{prefix}-resusage.txt"
    pdf_path = out_dir / f"{prefix}-megakernel_pdf.sass"
    full_path = out_dir / f"{prefix}-full.sass"
    res_path.write_text(res_text)
    pdf_path.write_text(pdf_sass)
    full_path.write_text(full_sass)
    pdf_res = function_resusage(res_text, "megakernel_pdf")
    return {
        "resusage": str(res_path),
        "megakernel_pdf_sass": str(pdf_path),
        "full_sass": str(full_path),
        "megakernel_pdf_resusage": pdf_res,
        "megakernel_pdf_parsed_resusage": parse_resusage(pdf_res),
        "head_dx_exact_resusage": function_resusage(
            res_text,
            "op_gemm_wgmma_n256_head_dx_exact_impl",
        ),
        "rms_dotparts_resusage": function_resusage(
            res_text,
            "op_rmsnorm_bwd_dx_h2560_dotparts",
        ),
        "megakernel_pdf_counts": sass_counts(pdf_sass),
        "full_counts": sass_counts(full_sass),
        "full_contains_hdxrmsdot_symbol": "op_rmsnorm_bwd_dx_h2560_dotparts" in full_sass,
        "full_contains_head_dx_exact_symbol": "op_gemm_wgmma_n256_head_dx_exact_impl" in full_sass,
    }


def audit(enabled: bool, out_dir: Path, stamp: str) -> dict[str, Any]:
    label = "candidate" if enabled else "default"
    model = build_model(enabled)
    try:
        return {
            "label": label,
            "route": route_summary(model, enabled),
            "sass": dump_sass(label, model, out_dir, stamp),
        }
    finally:
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/operator-gap")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default = audit(False, out_dir, args.stamp)
    candidate = audit(True, out_dir, args.stamp)

    d_route = default["route"]
    c_route = candidate["route"]
    d_pdf_res = default["sass"]["megakernel_pdf_parsed_resusage"]
    c_pdf_res = candidate["sass"]["megakernel_pdf_parsed_resusage"]
    d_head = d_route["head_dx_rows"]
    c_head = c_route["head_dx_rows"]
    d_part_rms = [row for row in d_route["rms_dx_rows"] if row["has_partials"]]
    c_part_rms = [row for row in c_route["rms_dx_rows"] if row["has_partials"]]

    route_page_pass = (
        d_route["n_instr"] == c_route["n_instr"] == 78
        and d_route["critical_path"] == c_route["critical_path"] == 44
        and d_route["gated"] == c_route["gated"] == 14
        and d_route["smem_bytes"] == c_route["smem_bytes"] == 151552
        and d_route["sidecar_available"]
        and c_route["sidecar_available"]
    )
    suffix_pass = (
        d_route["has_ntscbnd_suffix"]
        and c_route["has_ntscbnd_suffix"]
        and d_route["has_hdx_exact_pdf_suffix"]
        and c_route["has_hdx_exact_pdf_suffix"]
        and not d_route["has_hdxrmsdot_suffix"]
        and c_route["has_hdxrmsdot_suffix"]
    )
    arg_pass = (
        len(d_head) == 1
        and len(c_head) == 1
        and not d_head[0]["has_rmsdot_flag"]
        and c_head[0]["has_rmsdot_flag"]
        and c_head[0]["arg10_nparts"] == 10
        and not d_part_rms
        and len(c_part_rms) == 1
        and c_part_rms[0]["arg9_partials"] == c_head[0]["arg9_partials"]
        and c_part_rms[0]["arg10_nparts"] == 10
        and c_part_rms[0]["ntiles"] == d_route["rms_dx_rows"][0]["ntiles"]
    )
    resource_pass = (
        d_pdf_res.get("local", 1) == 0
        and c_pdf_res.get("local", 1) == 0
        and c_pdf_res.get("stack", 10**9) <= d_pdf_res.get("stack", -1)
    )
    sass_changed = (
        default["sass"]["megakernel_pdf_counts"] != candidate["sass"]["megakernel_pdf_counts"]
        or default["sass"]["full_counts"] != candidate["sass"]["full_counts"]
    )
    summary = {
        "claim": "qwen head-DX RMS-dot partials SASS/resource gate; no parity/timing",
        "sha": git_sha(),
        "pass": bool(route_page_pass and suffix_pass and arg_pass and resource_pass and sass_changed),
        "route_page_pass": bool(route_page_pass),
        "suffix_pass": bool(suffix_pass),
        "arg_pass": bool(arg_pass),
        "resource_pass": bool(resource_pass),
        "sass_changed": bool(sass_changed),
        "rows": [default, candidate],
    }
    print("QWEN_HDXRMSDOT_SASS_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
