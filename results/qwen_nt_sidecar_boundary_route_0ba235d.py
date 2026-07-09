#!/usr/bin/env python3
"""Host route proof for the qwen NT lm-head sidecar boundary row."""

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
    __file__ = "/tmp/fake-qwen-nt-sidecar-boundary.so"

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

SPLIT_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_SPLIT_PLAN"
BOUNDARY_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY"
SIDECAR_DEFINE = "-DMK_GEMM_N256_NT_SUPERTILE_SIDECAR"
BOUNDARY_DEFINE = "-DMK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY"


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
    n256_like = nmajor = stage3 = ntst = 0
    head_rows: list[dict[str, object]] = []
    boundary_rows = list(getattr(model, "qwen_nt_sidecar_boundary_rows", []))
    for idx, (op, ntiles, args) in enumerate(flat):
        if op not in (mk.OP_GEMM, mk.OP_QWEN_NT_SIDECAR_BOUNDARY):
            continue
        flags = int(args[6])
        if flags & (1 << 14):
            n256_like += 1
            stage3 += int(bool(flags & mk.GEMM_N256_STAGE3_FLAG))
            nmajor += int(bool(flags & mk.GEMM_N256_NMAJOR_FLAG))
            ntst += int(bool(flags & mk.GEMM_N256_NT_SUPERTILE_FLAG))
        if int(args[3]) == 1024 and int(args[4]) == 151936 and int(args[5]) == 2560:
            if flags & 2:
                head_rows.append({
                    "idx": idx,
                    "op": int(op),
                    "flags": flags,
                    "ntiles": int(ntiles),
                })

    cflags = list(model.ext.kwargs["extra_cuda_cflags"])
    name = str(model.ext.kwargs["name"])
    cutpoints = list(getattr(model, "qwen_nt_sidecar_cutpoints", []))
    plan = getattr(model, "qwen_nt_sidecar_split_plan", None)
    return {
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": model._smem_bytes,
        "n256_like": n256_like,
        "stage3": stage3,
        "nmajor": nmajor,
        "ntst": ntst,
        "head_rows": head_rows,
        "boundary_rows": boundary_rows,
        "boundary_row_count": len(boundary_rows),
        "name": name,
        "has_ntsc_suffix": "_ntsc" in name,
        "has_ntscbnd_suffix": "_ntscbnd" in name,
        "has_ntsc_define": SIDECAR_DEFINE in cflags,
        "has_boundary_define": BOUNDARY_DEFINE in cflags,
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
        "QWEN_NT_SIDECAR_BOUNDARY_ROUTE_JSON "
        + json.dumps(payload, sort_keys=True),
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


def check_boundary(row: dict[str, object], errors: list[str]) -> None:
    head_rows = row.get("head_rows")
    if head_rows != [{
        "idx": 37,
        "op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
        "flags": 234899586,
        "ntiles": 4748,
    }]:
        errors.append(f"wrong typed head row: {head_rows!r}")
    boundary_rows = row.get("boundary_rows")
    if boundary_rows != [{
        "flags": 234899586,
        "instr_index": 37,
        "ntiles": 4748,
        "op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
        "replaces_op": mk.OP_GEMM,
        "shape": {"M": 1024, "N": 151936, "K": 2560},
        "symbol": "qwen_nt_lmhead_sidecar",
    }]:
        errors.append(f"wrong boundary row metadata: {boundary_rows!r}")

    cps = row.get("cutpoints")
    if not isinstance(cps, list) or len(cps) != 1:
        errors.append("boundary route did not expose exactly one cutpoint")
        return
    cp = cps[0]
    expected_cp = {
        "instr_index": 37,
        "op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
        "original_op": mk.OP_GEMM,
        "boundary_op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
        "ntiles": 4748,
        "flags": 234899586,
        "producer_deps": [36],
        "direct_dependents": [38, 39, 40, 41],
        "direct_dependent_ops": [mk.OP_CE_FWD, mk.OP_CE_BWD, mk.OP_GEMM, mk.OP_GEMM],
    }
    for key, val in expected_cp.items():
        if cp.get(key) != val:
            errors.append(f"cutpoint {key} mismatch: {cp.get(key)!r} != {val!r}")

    plan = row.get("split_plan")
    if not isinstance(plan, dict):
        errors.append("boundary route did not expose a split plan")
        return
    expected_plan = {
        "cutpoint_op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
        "cutpoint_original_op": mk.OP_GEMM,
        "cutpoint_boundary_op": mk.OP_QWEN_NT_SIDECAR_BOUNDARY,
        "main_row_replaced_by_boundary": True,
        "valid_topological_split": True,
        "direct_rejoin_dependents": [38, 39, 40, 41],
        "pre_sidecar_required_closure": [0] + list(range(18, 37)),
        "pre_sidecar_independent_before_cutpoint": list(range(1, 18)),
        "post_sidecar_closure": list(range(38, 77)),
        "independent_after_cutpoint": [77],
    }
    for key, val in expected_plan.items():
        if plan.get(key) != val:
            errors.append(f"split plan {key} mismatch: {plan.get(key)!r} != {val!r}")
    if plan.get("violations") != []:
        errors.append(f"split plan has violations: {plan.get('violations')!r}")


def main() -> None:
    l2_default = emit("l2_default", CFG_L2, {})
    l2_split = emit("l2_split_plan_forced_no_boundary", CFG_L2, {SPLIT_ENV: "1"})
    l2_boundary = emit("l2_boundary_forced", CFG_L2, {BOUNDARY_ENV: "1"})
    l1_boundary = emit("l1_boundary_forced_negative", CFG_L1, {BOUNDARY_ENV: "1"})
    small_boundary = emit("small_boundary_forced_negative", CFG_SMALL, {BOUNDARY_ENV: "1"})
    l2_super_off = emit("l2_supertile_off_boundary_forced", CFG_L2, {
        BOUNDARY_ENV: "1",
        "MK_GEMM_N256_NT_SUPERTILE": "0",
    })
    l2_pdfprod_off = emit("l2_pdfprod_off_boundary_forced", CFG_L2, {
        BOUNDARY_ENV: "1",
        "MK_PDF_PRODUCER": "0",
    })
    l2_pdfonly_off = emit("l2_pdfonly_off_boundary_forced", CFG_L2, {
        BOUNDARY_ENV: "1",
        "MK_GEMM_N256_NT_SUPERTILE_PDFONLY": "0",
    })
    l2_reg_off = emit("l2_reg_off_boundary_forced", CFG_L2, {
        BOUNDARY_ENV: "1",
        "MK_GEMM_N256_NT_SUPERTILE_REG_EPI": "0",
    })

    errors: list[str] = []
    invariant_fields = (
        "n_instr",
        "critical_path",
        "gated",
        "smem_bytes",
        "n256_like",
        "stage3",
        "nmajor",
        "ntst",
    )
    same_fields(l2_default, l2_split, invariant_fields, "l2 split", errors)
    same_fields(l2_default, l2_boundary, invariant_fields, "l2 boundary", errors)
    if l2_default["head_rows"] != [{
        "idx": 37,
        "op": mk.OP_GEMM,
        "flags": 234899586,
        "ntiles": 4748,
    }]:
        errors.append(f"default head row changed: {l2_default['head_rows']!r}")
    if l2_split["head_rows"] != l2_default["head_rows"]:
        errors.append("split-plan-only route should not type the main row")
    if l2_split["boundary_row_count"] != 0 or l2_split["has_boundary_define"]:
        errors.append("split-plan-only route unexpectedly enabled boundary")
    if not (
        l2_boundary["has_ntscbnd_suffix"]
        and l2_boundary["has_ntsc_define"]
        and l2_boundary["has_boundary_define"]
    ):
        errors.append("boundary route did not build the sidecar boundary image")
    check_boundary(l2_boundary, errors)

    for tag, row in (
        ("l1 negative", l1_boundary),
        ("small negative", small_boundary),
        ("supertile off", l2_super_off),
        ("pdf producer off", l2_pdfprod_off),
        ("pdfonly off", l2_pdfonly_off),
        ("reg epilogue off", l2_reg_off),
    ):
        if (
            row["boundary_row_count"] != 0
            or row["cutpoint_count"] != 0
            or row["split_plan"] is not None
        ):
            errors.append(f"{tag} unexpectedly exposed boundary metadata")
        if row["has_boundary_define"] or row["has_ntscbnd_suffix"]:
            errors.append(f"{tag} unexpectedly enabled boundary image")

    plan = l2_boundary.get("split_plan") or {}
    summary = {
        "pass": not errors,
        "errors": errors,
        "default_route": {
            "n_instr": l2_default["n_instr"],
            "critical_path": l2_default["critical_path"],
            "gated": l2_default["gated"],
            "smem_bytes": l2_default["smem_bytes"],
        },
        "boundary_row_count": l2_boundary["boundary_row_count"],
        "cutpoint_count": l2_boundary["cutpoint_count"],
        "main_row_replaced_by_boundary": (
            plan.get("main_row_replaced_by_boundary") if isinstance(plan, dict) else None
        ),
        "direct_rejoin_dependents": (
            plan.get("direct_rejoin_dependents", []) if isinstance(plan, dict) else []
        ),
    }
    print("QWEN_NT_SIDECAR_BOUNDARY_SUMMARY " + json.dumps(summary, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
