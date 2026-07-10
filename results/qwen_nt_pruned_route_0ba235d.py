#!/usr/bin/env python3
"""Host route/load proof for the qwen NT pruned production-image SASS lane."""

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
    __file__ = "/tmp/fake-qwen-nt-pruned.so"

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
TOP_ENV = "MK_GEMM_N256_NT_SUPERTILE_TOPEMBED"
PRUNED_ENV = "MK_GEMM_N256_NT_SUPERTILE_PRUNED"
TOP_DEFINE = "-DMK_GEMM_N256_NT_SUPERTILE_TOPEMBED"
PRUNED_DEFINE = "-DMK_GEMM_N256_NT_SUPERTILE_PRUNED"


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
        "has_nttop_suffix": "_nttop" in name,
        "has_ntpr_suffix": "_ntpr" in name,
        "has_nttop_define": TOP_DEFINE in cflags,
        "has_ntpr_define": PRUNED_DEFINE in cflags,
        "has_nt_pdfonly_define": "-DMK_GEMM_N256_NT_SUPERTILE_PDFONLY" in cflags,
        "has_nt_reg_define": "-DMK_GEMM_N256_NT_SUPERTILE_REG_EPI" in cflags,
        "has_pdf_producer_define": "-DMK_PDF_PRODUCER" in cflags,
    }


def emit(tag: str, cfg: Cfg, updates: dict[str, str | None]) -> dict[str, object]:
    payload = {"tag": tag, **route_summary(build(cfg, updates))}
    print("QWEN_NT_PRUNED_ROUTE_JSON " + json.dumps(payload, sort_keys=True), flush=True)
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


def main() -> None:
    l2_default = emit("l2_default", CFG_L2, {})
    l2_top = emit("l2_topembed_forced", CFG_L2, {TOP_ENV: "1"})
    l2_pruned = emit("l2_pruned_forced", CFG_L2, {PRUNED_ENV: "1"})
    l2_pruned_top0 = emit("l2_pruned_forced_topembed_env0", CFG_L2, {TOP_ENV: "0", PRUNED_ENV: "1"})
    l1_pruned = emit("l1_pruned_forced_negative", CFG_L1, {PRUNED_ENV: "1"})
    small_pruned = emit("small_pruned_forced_negative", CFG_SMALL, {PRUNED_ENV: "1"})
    l2_super_off = emit(
        "l2_supertile_off_pruned_forced",
        CFG_L2,
        {
            PRUNED_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE": "0",
        },
    )
    l2_pdfprod_off = emit(
        "l2_pdfprod_off_pruned_forced",
        CFG_L2,
        {
            PRUNED_ENV: "1",
            "MK_PDF_PRODUCER": "0",
        },
    )
    l2_pdfonly_off = emit(
        "l2_pdfonly_off_pruned_forced",
        CFG_L2,
        {
            PRUNED_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE_PDFONLY": "0",
        },
    )
    l2_reg_off = emit(
        "l2_reg_off_pruned_forced",
        CFG_L2,
        {
            PRUNED_ENV: "1",
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
    same_fields(l2_default, l2_top, invariant_fields, "l2 topembed", errors)
    same_fields(l2_default, l2_pruned, invariant_fields, "l2 pruned", errors)
    same_fields(l2_pruned, l2_pruned_top0, invariant_fields, "l2 pruned top0", errors)

    if l2_pruned["n_instr"] != 78 or l2_pruned["critical_path"] != 44:
        errors.append("unexpected qwen4b-l2 route shape")
    if l2_pruned["gated"] != 14 or l2_pruned["smem_bytes"] != 151552:
        errors.append("unexpected qwen4b-l2 gated/smem route")
    if l2_default["has_nttop_suffix"] or l2_default["has_nttop_define"]:
        errors.append("default route unexpectedly carries topembed flag")
    if l2_default["has_ntpr_suffix"] or l2_default["has_ntpr_define"]:
        errors.append("default route unexpectedly carries pruned flag")
    if not l2_top["has_nttop_suffix"] or not l2_top["has_nttop_define"]:
        errors.append("forced l2 route did not carry topembed suffix/define")
    if l2_top["has_ntpr_suffix"] or l2_top["has_ntpr_define"]:
        errors.append("topembed-only route unexpectedly carries pruned flag")
    for tag, row in (("pruned", l2_pruned), ("pruned top0", l2_pruned_top0)):
        if not (
            row["has_nttop_suffix"] and row["has_nttop_define"] and row["has_ntpr_suffix"] and row["has_ntpr_define"]
        ):
            errors.append(f"{tag} did not carry topembed+pruned suffix/define")
        if not (row["has_nt_pdfonly_define"] and row["has_nt_reg_define"] and row["has_pdf_producer_define"]):
            errors.append(f"{tag} missing prerequisite qwen pdf/reg defines")

    for tag, row in (
        ("l1 negative", l1_pruned),
        ("small negative", small_pruned),
        ("supertile off", l2_super_off),
        ("pdf producer off", l2_pdfprod_off),
        ("pdfonly off", l2_pdfonly_off),
        ("reg epilogue off", l2_reg_off),
    ):
        if row["has_ntpr_suffix"] or row["has_ntpr_define"]:
            errors.append(f"{tag} unexpectedly enabled pruned mode")
        if row["has_nttop_suffix"] or row["has_nttop_define"]:
            errors.append(f"{tag} unexpectedly enabled topembed through pruned mode")

    summary = {
        "pass": not errors,
        "errors": errors,
        "rows": [
            l2_default,
            l2_top,
            l2_pruned,
            l2_pruned_top0,
            l1_pruned,
            small_pruned,
            l2_super_off,
            l2_pdfprod_off,
            l2_pdfonly_off,
            l2_reg_off,
        ],
    }
    print("QWEN_NT_PRUNED_ROUTE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
