#!/usr/bin/env python3
"""Host route proof for the qwen NT lm-head sidecar split plan."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("MKAB_TREE", Path(__file__).resolve().parents[1]))
MKDIR = ROOT / "experiments" / "fused-training-megakernel"
sys.path.insert(0, str(MKDIR))

import mk  # noqa: E402


class FakeExt:
    __file__ = "/tmp/fake-qwen-nt-sidecar-splitplan.so"

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

SIDECAR_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR"
CUTPOINT_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_CUTPOINT"
SPLIT_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_SPLIT_PLAN"
SIDECAR_DEFINE = "-DMK_GEMM_N256_NT_SUPERTILE_SIDECAR"


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


def build(cfg: Cfg, updates: dict[str, str | None]) -> MKQwen3:
    old = with_env(updates)
    try:
        return MKQwen3(cfg, dev="meta", seed=0)
    finally:
        restore_env(old)


def route_summary(model: MKQwen3) -> dict[str, object]:
    flat = [ins for wave in model.prog.waves for ins in wave]
    n256 = nmajor = stage3 = tma = ntst = 0
    head_flags: list[int] = []
    head_ntiles: list[int] = []
    for op, ntiles, args in flat:
        if op != mk.OP_GEMM:
            continue
        flags = int(args[6])
        if flags & (1 << 14):
            n256 += 1
            stage3 += int(bool(flags & mk.GEMM_N256_STAGE3_FLAG))
            nmajor += int(bool(flags & mk.GEMM_N256_NMAJOR_FLAG))
            ntst += int(bool(flags & mk.GEMM_N256_NT_SUPERTILE_FLAG))
            tma += int(len(args) > 20 and int(args[20]) > 0)
        if int(args[3]) == 1024 and int(args[4]) == 151936 and int(args[5]) == 2560:
            if flags & 2:
                head_flags.append(flags)
                head_ntiles.append(int(ntiles))

    cflags = list(model.ext.kwargs["extra_cuda_cflags"])
    name = str(model.ext.kwargs["name"])
    cutpoints = list(getattr(model, "qwen_nt_sidecar_cutpoints", []))
    plan = getattr(model, "qwen_nt_sidecar_split_plan", None)
    return {
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": model._smem_bytes,
        "n256": n256,
        "stage3": stage3,
        "nmajor": nmajor,
        "ntst": ntst,
        "tma": tma,
        "head_flags": head_flags,
        "head_ntiles": head_ntiles,
        "name": name,
        "has_ntsc_suffix": "_ntsc" in name,
        "has_ntsc_define": SIDECAR_DEFINE in cflags,
        "has_nt_pdfonly_define": "-DMK_GEMM_N256_NT_SUPERTILE_PDFONLY" in cflags,
        "has_nt_reg_define": "-DMK_GEMM_N256_NT_SUPERTILE_REG_EPI" in cflags,
        "has_pdf_producer_define": "-DMK_PDF_PRODUCER" in cflags,
        "cutpoint_count": len(cutpoints),
        "cutpoints": cutpoints,
        "split_plan": plan,
    }


def emit(tag: str, cfg: Cfg, updates: dict[str, str | None]) -> dict[str, object]:
    payload = {"tag": tag, **route_summary(build(cfg, updates))}
    print(
        "QWEN_NT_SIDECAR_SPLITPLAN_ROUTE_JSON " + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return payload


def same_fields(
    lhs: dict[str, object],
    rhs: dict[str, object],
    fields: tuple[str, ...],
    label: str,
    errors: list[str],
) -> None:
    for field in fields:
        if lhs[field] != rhs[field]:
            errors.append(f"{label} changed {field}: {lhs[field]!r} != {rhs[field]!r}")


def check_split_plan(row: dict[str, object], errors: list[str]) -> None:
    plan = row.get("split_plan")
    if not isinstance(plan, dict):
        errors.append("split-plan route did not expose a plan dict")
        return
    expected_scalars = {
        "kind": "qwen_nt_lmhead_sidecar_split_plan",
        "runnable_now": False,
        "valid_topological_split": True,
        "cutpoint_instr": 37,
        "cutpoint_kind": "qwen_nt_lmhead",
        "cutpoint_symbol": "qwen_nt_lmhead_sidecar",
        "cutpoint_shape": {"M": 1024, "N": 151936, "K": 2560},
        "cutpoint_flags": 234899586,
        "cutpoint_ntiles": 4748,
        "direct_rejoin_dependents": [38, 39, 40, 41],
        "direct_rejoin_ops": [mk.OP_CE_FWD, mk.OP_CE_BWD, mk.OP_GEMM, mk.OP_GEMM],
        "pre_sidecar_instruction_window": [0, 37],
    }
    for key, val in expected_scalars.items():
        if plan.get(key) != val:
            errors.append(f"split plan {key} mismatch: {plan.get(key)!r} != {val!r}")
    if plan.get("violations") != []:
        errors.append(f"split plan has violations: {plan.get('violations')!r}")

    pre = plan.get("pre_sidecar_required_closure")
    post = plan.get("post_sidecar_closure")
    if pre != [0] + list(range(18, 37)):
        errors.append(f"bad pre-sidecar closure: {pre!r}")
    if plan.get("pre_sidecar_independent_before_cutpoint") != list(range(1, 18)):
        errors.append(f"bad pre-sidecar independent rows: {plan.get('pre_sidecar_independent_before_cutpoint')!r}")
    if post != list(range(38, 77)):
        errors.append(f"bad post-sidecar closure: {post!r}")
    if plan.get("independent_after_cutpoint") != [77]:
        errors.append(f"bad post-sidecar independent rows: {plan.get('independent_after_cutpoint')!r}")
    for row_idx in ["38", "39", "40", "41"]:
        deps = plan.get("direct_rejoin_original_deps", {}).get(row_idx)
        if not isinstance(deps, list) or 37 not in deps:
            errors.append(f"rejoin row {row_idx} does not depend on cutpoint: {deps!r}")

    sidecar = plan.get("sidecar_external_producer", {})
    launch = sidecar.get("launch", {}) if isinstance(sidecar, dict) else {}
    if launch != {
        "instrs": "Program._instrs",
        "ins": 37,
        "t0": 0,
        "t1": 4748,
        "bufs": "Program._buftab",
        "threads": 384,
    }:
        errors.append(f"wrong sidecar launch in split plan: {launch!r}")
    if sidecar.get("reads") is None or sidecar.get("writes") is None:
        errors.append(f"missing sidecar read/write buffers: {sidecar!r}")
    edges = plan.get("external_edges")
    if not isinstance(edges, list) or len(edges) != 4:
        errors.append(f"expected four external rejoin edges, got {edges!r}")
    else:
        consumers = [edge.get("consumer") for edge in edges]
        if consumers != [38, 39, 40, 41]:
            errors.append(f"wrong external edge consumers: {consumers!r}")


def main() -> None:
    l2_default = emit("l2_default", CFG_L2, {})
    l2_sidecar = emit("l2_sidecar_forced_no_cutpoint", CFG_L2, {SIDECAR_ENV: "1"})
    l2_cutpoint = emit("l2_cutpoint_forced_no_split", CFG_L2, {CUTPOINT_ENV: "1"})
    l2_split = emit("l2_split_plan_forced", CFG_L2, {SPLIT_ENV: "1"})
    l1_split = emit("l1_split_plan_forced_negative", CFG_L1, {SPLIT_ENV: "1"})
    small_split = emit("small_split_plan_forced_negative", CFG_SMALL, {SPLIT_ENV: "1"})
    l2_super_off = emit(
        "l2_supertile_off_split_plan_forced",
        CFG_L2,
        {
            SPLIT_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE": "0",
        },
    )
    l2_pdfprod_off = emit(
        "l2_pdfprod_off_split_plan_forced",
        CFG_L2,
        {
            SPLIT_ENV: "1",
            "MK_PDF_PRODUCER": "0",
        },
    )
    l2_pdfonly_off = emit(
        "l2_pdfonly_off_split_plan_forced",
        CFG_L2,
        {
            SPLIT_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE_PDFONLY": "0",
        },
    )
    l2_reg_off = emit(
        "l2_reg_off_split_plan_forced",
        CFG_L2,
        {
            SPLIT_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE_REG_EPI": "0",
        },
    )

    errors: list[str] = []
    invariant_fields = (
        "n_instr",
        "critical_path",
        "gated",
        "smem_bytes",
        "head_flags",
        "head_ntiles",
        "n256",
        "stage3",
        "nmajor",
        "ntst",
        "tma",
    )
    same_fields(l2_default, l2_sidecar, invariant_fields, "l2 sidecar", errors)
    same_fields(l2_default, l2_cutpoint, invariant_fields, "l2 cutpoint", errors)
    same_fields(l2_default, l2_split, invariant_fields, "l2 split plan", errors)

    if l2_default["cutpoint_count"] != 0 or l2_default["split_plan"] is not None:
        errors.append("default route unexpectedly exposed qwen sidecar metadata")
    if l2_sidecar["cutpoint_count"] != 0 or l2_sidecar["split_plan"] is not None:
        errors.append("sidecar-only route unexpectedly exposed split metadata")
    if l2_cutpoint["cutpoint_count"] != 1 or l2_cutpoint["split_plan"] is not None:
        errors.append("cutpoint-only route did not expose exactly one cutpoint only")
    if l2_split["cutpoint_count"] != 1:
        errors.append("split-plan route did not imply exactly one cutpoint")
    if not l2_split["has_ntsc_suffix"] or not l2_split["has_ntsc_define"]:
        errors.append("split-plan route did not build the sidecar export image")
    if not (
        l2_split["has_nt_pdfonly_define"] and l2_split["has_nt_reg_define"] and l2_split["has_pdf_producer_define"]
    ):
        errors.append("split-plan route missing prerequisite qwen pdf/reg defines")
    check_split_plan(l2_split, errors)

    for tag, row in (
        ("l1 negative", l1_split),
        ("small negative", small_split),
        ("supertile off", l2_super_off),
        ("pdf producer off", l2_pdfprod_off),
        ("pdfonly off", l2_pdfonly_off),
        ("reg epilogue off", l2_reg_off),
    ):
        if row["cutpoint_count"] != 0 or row["split_plan"] is not None:
            errors.append(f"{tag} unexpectedly exposed split/cutpoint metadata")
        if row["has_ntsc_suffix"] or row["has_ntsc_define"]:
            errors.append(f"{tag} unexpectedly enabled sidecar image")

    plan = l2_split.get("split_plan") or {}
    summary = {
        "pass": not errors,
        "errors": errors,
        "default_route": {
            "n_instr": l2_default["n_instr"],
            "critical_path": l2_default["critical_path"],
            "gated": l2_default["gated"],
            "smem_bytes": l2_default["smem_bytes"],
        },
        "cutpoint_count": l2_split["cutpoint_count"],
        "split_plan_present": isinstance(plan, dict),
        "pre_closure_count": (len(plan.get("pre_sidecar_required_closure", [])) if isinstance(plan, dict) else 0),
        "post_closure_count": (len(plan.get("post_sidecar_closure", [])) if isinstance(plan, dict) else 0),
        "independent_after_cutpoint": (plan.get("independent_after_cutpoint", []) if isinstance(plan, dict) else []),
    }
    print("QWEN_NT_SIDECAR_SPLITPLAN_SUMMARY " + json.dumps(summary, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
