#!/usr/bin/env python3
"""Host route contract for the dirty-frontier qwen NT sidecar manual port."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
sys.path.insert(0, str(MKDIR))

import mk  # noqa: E402


class FakeExt:
    __file__ = "/tmp/fake-qwen-nt-sidecar-dirty-manual.so"

    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = dict(kwargs)


def install_host_stubs() -> None:
    def fake_load(**kwargs):
        return FakeExt(kwargs)

    mk.load = fake_load
    mk._audit_bulkred_sass = lambda _path: None
    mk.Program._inject_gemm_tmaps = lambda self: None

    def fake_buf(self: mk.Program, t, slot=None) -> int:
        key = (id(t), slot)
        if key not in self._buf_ids:
            self._buf_ids[key] = len(self.bufs)
            self.bufs.append(t)
            self._buf_meta.append((id(t), slot))
        return self._buf_ids[key]

    mk.Program.buf = fake_buf


install_host_stubs()

from model import Cfg, MKQwen3  # noqa: E402


CFG_L1 = Cfg(H=2560, L=1, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
CFG_L2 = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
CFG_SMALL = Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)

BOUNDARY_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY"
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


def with_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in updates}
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


def build(cfg: Cfg, updates: dict[str, str | None]) -> MKQwen3:
    old = with_env(updates)
    try:
        return MKQwen3(cfg, dev="meta", seed=0)
    finally:
        restore_env(old)


def route_summary(tag: str, model: MKQwen3) -> dict[str, Any]:
    flat = [ins for wave in model.prog.waves for ins in wave]
    target_gemm_rows: list[dict[str, int]] = []
    boundary_rows: list[dict[str, int]] = []
    for flat_index, (op, ntiles, args) in enumerate(flat):
        if len(args) < 11:
            continue
        row = {
            "flat_index": int(flat_index),
            "op": int(op),
            "ntiles": int(ntiles),
            "flags": int(args[6]),
            "M": int(args[3]),
            "N": int(args[4]),
            "K": int(args[5]),
        }
        is_head = row["M"] == 1024 and row["N"] == 151936 and row["K"] == 2560
        if is_head and op == mk.OP_GEMM and (row["flags"] & 2):
            target_gemm_rows.append(row)
        if is_head and op == mk.OP_QWEN_NT_SIDECAR_BOUNDARY:
            boundary_rows.append(row)

    cutpoints = list(getattr(model, "qwen_nt_sidecar_cutpoints", []))
    plan = getattr(model, "qwen_nt_sidecar_split_plan", None)
    subprograms = None
    if model.qwen_nt_sidecar_step_available():
        subprograms = model.qwen_nt_sidecar_pdf_subprograms()
    cflags = list(model.ext.kwargs["extra_cuda_cflags"])
    name = str(model.ext.kwargs["name"])
    return {
        "tag": tag,
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": model._smem_bytes,
        "name": name,
        "has_ntscbnd_suffix": "_ntscbnd" in name,
        "has_sidecar_define": "-DMK_GEMM_N256_NT_SUPERTILE_SIDECAR" in cflags,
        "has_boundary_define": "-DMK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY" in cflags,
        "has_pdfonly_define": "-DMK_GEMM_N256_NT_SUPERTILE_PDFONLY" in cflags,
        "has_reg_define": "-DMK_GEMM_N256_NT_SUPERTILE_REG_EPI" in cflags,
        "has_pdf_producer_define": "-DMK_PDF_PRODUCER" in cflags,
        "step_requested": bool(model.qwen_nt_sidecar_step_requested),
        "api_available": bool(model.qwen_nt_sidecar_step_available()),
        "cutpoint_count": len(cutpoints),
        "boundary_row_count": len(getattr(model, "qwen_nt_sidecar_boundary_rows", [])),
        "target_gemm_rows": target_gemm_rows,
        "boundary_rows": boundary_rows,
        "split_plan_kind": None if plan is None else plan.get("kind"),
        "split_plan_valid": None if plan is None else bool(plan.get("valid_topological_split")),
        "main_row_replaced_by_boundary": None if plan is None else bool(plan.get("main_row_replaced_by_boundary")),
        "prefix_n_instr": None if subprograms is None else int(subprograms["prefix"]["n_instr"]),
        "post_n_instr": None if subprograms is None else int(subprograms["post"]["n_instr"]),
        "sidecar_tile_range": None if not cutpoints else [0, int(cutpoints[0]["ntiles"])],
    }


def emit(tag: str, cfg: Cfg, updates: dict[str, str | None]) -> dict[str, Any]:
    row = route_summary(tag, build(cfg, updates))
    print("QWEN_NT_DIRTY_SIDECAR_ROUTE_JSON " + json.dumps(row, sort_keys=True), flush=True)
    return row


def expect_sidecar(row: dict[str, Any], expected: dict[str, int], errors: list[str]) -> None:
    tag = row["tag"]
    for key, value in expected.items():
        if row[key] != value:
            errors.append(f"{tag} expected {key}={value}, got {row[key]!r}")
    for key in ("has_ntscbnd_suffix", "has_sidecar_define", "has_boundary_define", "api_available", "step_requested"):
        if not row[key]:
            errors.append(f"{tag} missing {key}")
    if row["cutpoint_count"] != 1 or row["boundary_row_count"] != 1:
        errors.append(f"{tag} expected exactly one cutpoint/boundary row")
    if row["target_gemm_rows"]:
        errors.append(f"{tag} still has direct target GEMM rows")
    if len(row["boundary_rows"]) != 1:
        errors.append(f"{tag} did not expose one target boundary row")
    if row["split_plan_kind"] != "qwen_nt_lmhead_sidecar_split_plan":
        errors.append(f"{tag} bad split plan kind {row['split_plan_kind']!r}")
    if row["split_plan_valid"] is not True or row["main_row_replaced_by_boundary"] is not True:
        errors.append(f"{tag} invalid split plan")
    if row["sidecar_tile_range"] != [0, 4748]:
        errors.append(f"{tag} bad sidecar tile range {row['sidecar_tile_range']!r}")
    for key in ("has_pdfonly_define", "has_reg_define", "has_pdf_producer_define"):
        if not row[key]:
            errors.append(f"{tag} missing prerequisite {key}")


def expect_no_sidecar(
    row: dict[str, Any],
    errors: list[str],
    *,
    allow_step_requested: bool = False,
) -> None:
    tag = row["tag"]
    for key in ("has_ntscbnd_suffix", "has_sidecar_define", "has_boundary_define", "api_available"):
        if row[key]:
            errors.append(f"{tag} unexpectedly has {key}")
    if row["step_requested"] and not allow_step_requested:
        errors.append(f"{tag} unexpectedly has step_requested")
    if row["cutpoint_count"] or row["boundary_row_count"] or row["boundary_rows"]:
        errors.append(f"{tag} unexpectedly produced sidecar metadata")
    if row["split_plan_kind"] is not None:
        errors.append(f"{tag} unexpectedly produced split plan")


def main() -> None:
    rows = [
        emit("l1_default", CFG_L1, {BOUNDARY_ENV: None, POLICY_ENV: None}),
        emit("l2_default", CFG_L2, {BOUNDARY_ENV: None, POLICY_ENV: None}),
        emit("l1_forced_old", CFG_L1, {BOUNDARY_ENV: "0", POLICY_ENV: None}),
        emit("l2_forced_old", CFG_L2, {BOUNDARY_ENV: "0", POLICY_ENV: None}),
        emit("l2_policy_off", CFG_L2, {BOUNDARY_ENV: None, POLICY_ENV: "0"}),
        emit("small_forced_boundary", CFG_SMALL, {BOUNDARY_ENV: "1", POLICY_ENV: None}),
        emit("l2_supertile_off", CFG_L2, {BOUNDARY_ENV: "1", "MK_GEMM_N256_NT_SUPERTILE": "0"}),
        emit("l2_pdfprod_off", CFG_L2, {BOUNDARY_ENV: "1", "MK_PDF_PRODUCER": "0"}),
        emit("l2_pdfonly_off", CFG_L2, {BOUNDARY_ENV: "1", "MK_GEMM_N256_NT_SUPERTILE_PDFONLY": "0"}),
        emit("l2_reg_off", CFG_L2, {BOUNDARY_ENV: "1", "MK_GEMM_N256_NT_SUPERTILE_REG_EPI": "0"}),
    ]
    by_tag = {row["tag"]: row for row in rows}
    errors: list[str] = []
    expect_sidecar(
        by_tag["l1_default"],
        {
            "n_instr": 47,
            "critical_path": 26,
            "gated": 9,
            "prefix_n_instr": 22,
            "post_n_instr": 24,
        },
        errors,
    )
    expect_sidecar(
        by_tag["l2_default"],
        {
            "n_instr": 78,
            "critical_path": 44,
            "gated": 14,
            "prefix_n_instr": 37,
            "post_n_instr": 40,
        },
        errors,
    )
    for tag in (
        "l1_forced_old",
        "l2_forced_old",
        "small_forced_boundary",
        "l2_supertile_off",
        "l2_reg_off",
    ):
        expect_no_sidecar(by_tag[tag], errors)
    for tag in ("l2_pdfprod_off", "l2_pdfonly_off"):
        expect_no_sidecar(by_tag[tag], errors, allow_step_requested=True)
        if not by_tag[tag]["step_requested"]:
            errors.append(f"{tag} should keep step_requested so step() raises unavailable-route guard")
    policy_off = by_tag["l2_policy_off"]
    if not policy_off["api_available"] or not policy_off["has_ntscbnd_suffix"]:
        errors.append("policy-off should keep the sidecar boundary route available")
    if policy_off["step_requested"]:
        errors.append("policy-off unexpectedly requested sidecar step")

    summary = {
        "claim": "qwen_nt_sidecar_dirty_manual_route_be55098",
        "sha": git_sha(),
        "pass": not errors,
        "errors": errors,
        "rows": rows,
    }
    print("QWEN_NT_DIRTY_SIDECAR_ROUTE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
