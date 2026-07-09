#!/usr/bin/env python3
"""CUDA build and focused SASS helper for qwen NT lm-head sidecar scaffold."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
sys.path.insert(0, str(MKDIR))

import mk  # noqa: E402
from model import Cfg, MKQwen3  # noqa: E402


CFG = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
SIDECAR_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR"


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
    old = with_env({SIDECAR_ENV: "1"})
    try:
        return MKQwen3(CFG, seed=0)
    finally:
        restore_env(old)


def run_text(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def function_resusage(text: str, needle: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if f"Function {needle}:" in line or f"Function : {needle}" in line:
            for nxt in lines[idx + 1: idx + 8]:
                if "REG:" in nxt:
                    return nxt.strip()
    return ""


def parse_resusage(line: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in ("REG", "STACK", "SHARED", "LOCAL"):
        m = re.search(rf"{key}:([0-9]+)", line)
        if m:
            out[key.lower()] = int(m.group(1))
    return out


def sass_counts(text: str) -> dict[str, object]:
    counts = {
        "HGMMA": 0,
        "HGMMA.64x128": 0,
        "WARPGROUP.ARRIVE": 0,
        "WARPGROUP.DEPBAR": 0,
        "WARPGROUP.DEPBAR.LE 0x0": 0,
        "CALL": 0,
        "TMA": 0,
    }
    hist: dict[int, int] = {}
    cur = 0
    max_run = 0
    for line in text.splitlines():
        if "HGMMA" in line:
            counts["HGMMA"] += 1
            cur += 1
            max_run = max(max_run, cur)
            if "64x128" in line:
                counts["HGMMA.64x128"] += 1
            continue
        if "WARPGROUP.ARRIVE" in line:
            counts["WARPGROUP.ARRIVE"] += 1
        if "WARPGROUP.DEPBAR" in line:
            counts["WARPGROUP.DEPBAR"] += 1
            if "LE" in line and "0x0" in line:
                counts["WARPGROUP.DEPBAR.LE 0x0"] += 1
            hist[cur] = hist.get(cur, 0) + 1
            cur = 0
        if re.search(r"\bCALL", line):
            counts["CALL"] += 1
        if "UTMA" in line or "TMA." in line:
            counts["TMA"] += 1
    counts["max_hgmma_between_depbar"] = max_run
    counts["segment_hgmma_hist"] = {str(k): v for k, v in sorted(hist.items())}
    return counts


def summarize_model(model: MKQwen3) -> dict[str, object]:
    flat = [ins for wave in model.prog.waves for ins in wave]
    head_rows: list[tuple[int, int]] = []
    for op, ntiles, args in flat:
        if op != mk.OP_GEMM:
            continue
        if int(args[3]) == 1024 and int(args[4]) == 151936 and int(args[5]) == 2560:
            if int(args[6]) & 2:
                head_rows.append((int(ntiles), int(args[6])))
    name = Path(model.ext.__file__).name
    return {
        "name": name,
        "so": model.ext.__file__,
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": model._smem_bytes,
        "head_rows": head_rows,
        "has_ntsc_suffix": "_ntsc" in name,
    }


def dump_function(model: MKQwen3, symbol: str, out_dir: Path) -> dict[str, object]:
    so = model.ext.__file__
    res_path = out_dir / f"qwen-nt-sidecar-{symbol}-resusage-0ba235d.txt"
    sass_path = out_dir / f"qwen-nt-sidecar-{symbol}.sass"
    res = run_text(["cuobjdump", "-res-usage", so])
    sass = run_text(["cuobjdump", "-sass", "-fun", symbol, so])
    res_path.write_text(res)
    sass_path.write_text(sass)
    res_line = function_resusage(res, symbol)
    row = {"symbol": symbol, **sass_counts(sass)}
    row["resusage"] = str(res_path)
    row["sass"] = str(sass_path)
    row["function_resusage"] = res_line
    row["parsed_resusage"] = parse_resusage(res_line)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build()
    try:
        model_row = summarize_model(model)
        sidecar = dump_function(model, "qwen_nt_lmhead_sidecar", out_dir)
        pdf = dump_function(model, "megakernel_pdf", out_dir)
    finally:
        del model
        torch.cuda.empty_cache()

    summary = {
        "pass": True,
        "model": model_row,
        "functions": [sidecar, pdf],
    }
    print("QWEN_NT_SIDECAR_BUILD_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    assert model_row["n_instr"] == 78
    assert model_row["critical_path"] == 44
    assert model_row["gated"] == 14
    assert model_row["smem_bytes"] == 151552
    assert model_row["has_ntsc_suffix"]
    assert sidecar["HGMMA.64x128"] > 0
    assert sidecar["max_hgmma_between_depbar"] > 1
    assert sidecar["parsed_resusage"].get("local", 1) == 0


if __name__ == "__main__":
    main()
