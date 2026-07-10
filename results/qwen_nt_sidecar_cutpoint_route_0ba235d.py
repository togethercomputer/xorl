#!/usr/bin/env python3
"""Host route proof for the qwen NT lm-head sidecar cutpoint contract."""

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
    __file__ = "/tmp/fake-qwen-nt-sidecar-cutpoint.so"

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
SIDECAR_DEFINE = "-DMK_GEMM_N256_NT_SUPERTILE_SIDECAR"
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
    cutpoints = list(getattr(model, "qwen_nt_sidecar_cutpoints", []))
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
        "has_ntsc_suffix": "_ntsc" in name,
        "has_nttop_define": TOP_DEFINE in cflags,
        "has_ntpr_define": PRUNED_DEFINE in cflags,
        "has_ntsc_define": SIDECAR_DEFINE in cflags,
        "has_nt_pdfonly_define": "-DMK_GEMM_N256_NT_SUPERTILE_PDFONLY" in cflags,
        "has_nt_reg_define": "-DMK_GEMM_N256_NT_SUPERTILE_REG_EPI" in cflags,
        "has_pdf_producer_define": "-DMK_PDF_PRODUCER" in cflags,
        "cutpoint_count": len(cutpoints),
        "cutpoints": cutpoints,
    }


def emit(tag: str, cfg: Cfg, updates: dict[str, str | None]) -> dict[str, object]:
    payload = {"tag": tag, **route_summary(build(cfg, updates))}
    print(
        "QWEN_NT_SIDECAR_CUTPOINT_ROUTE_JSON " + json.dumps(payload, sort_keys=True),
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


def check_cutpoint(row: dict[str, object], errors: list[str]) -> None:
    cps = row["cutpoints"]
    if not isinstance(cps, list) or len(cps) != 1:
        errors.append("cutpoint route did not expose exactly one contract")
        return
    cp = cps[0]
    expected = {
        "kind": "qwen_nt_lmhead",
        "symbol": "qwen_nt_lmhead_sidecar",
        "instr_index": 37,
        "op": mk.OP_GEMM,
        "ntiles": 4748,
        "tile_start": 0,
        "tile_stop": 4748,
        "shape": {"M": 1024, "N": 151936, "K": 2560},
        "flags": 234899586,
        "producer_deps": [36],
        "producer_dep_ops": [mk.OP_RMSNORM_FWD],
        "direct_dependents": [38, 39, 40, 41],
        "direct_dependent_ops": [mk.OP_CE_FWD, mk.OP_CE_BWD, mk.OP_GEMM, mk.OP_GEMM],
        "ce_fwd_dependents": [38],
        "ce_bwd_dependents": [39],
    }
    for key, val in expected.items():
        if cp.get(key) != val:
            errors.append(f"cutpoint {key} mismatch: {cp.get(key)!r} != {val!r}")
    if cp.get("read_arg_positions") != [0, 1]:
        errors.append(f"cutpoint read positions wrong: {cp.get('read_arg_positions')!r}")
    if cp.get("write_arg_positions") != [2, 9]:
        errors.append(f"cutpoint write positions wrong: {cp.get('write_arg_positions')!r}")
    if cp.get("input_bufs", {}).keys() != {"xnf", "wlm"}:
        errors.append(f"cutpoint input buffer keys wrong: {cp.get('input_bufs')!r}")
    if cp.get("output_bufs", {}).keys() != {"logits", "lse_parts"}:
        errors.append(f"cutpoint output buffer keys wrong: {cp.get('output_bufs')!r}")
    launch = cp.get("sidecar_launch", {})
    if launch != {
        "instrs": "Program._instrs",
        "ins": 37,
        "t0": 0,
        "t1": 4748,
        "bufs": "Program._buftab",
        "threads": 384,
    }:
        errors.append(f"cutpoint launch contract wrong: {launch!r}")


def main() -> None:
    l2_default = emit("l2_default", CFG_L2, {})
    l2_sidecar = emit("l2_sidecar_forced_no_cutpoint", CFG_L2, {SIDECAR_ENV: "1"})
    l2_cutpoint = emit("l2_cutpoint_forced", CFG_L2, {CUTPOINT_ENV: "1"})
    l1_cutpoint = emit("l1_cutpoint_forced_negative", CFG_L1, {CUTPOINT_ENV: "1"})
    small_cutpoint = emit("small_cutpoint_forced_negative", CFG_SMALL, {CUTPOINT_ENV: "1"})
    l2_super_off = emit(
        "l2_supertile_off_cutpoint_forced",
        CFG_L2,
        {
            CUTPOINT_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE": "0",
        },
    )
    l2_pdfprod_off = emit(
        "l2_pdfprod_off_cutpoint_forced",
        CFG_L2,
        {
            CUTPOINT_ENV: "1",
            "MK_PDF_PRODUCER": "0",
        },
    )
    l2_pdfonly_off = emit(
        "l2_pdfonly_off_cutpoint_forced",
        CFG_L2,
        {
            CUTPOINT_ENV: "1",
            "MK_GEMM_N256_NT_SUPERTILE_PDFONLY": "0",
        },
    )
    l2_reg_off = emit(
        "l2_reg_off_cutpoint_forced",
        CFG_L2,
        {
            CUTPOINT_ENV: "1",
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
    same_fields(l2_default, l2_cutpoint, invariant_fields, "l2 cutpoint", errors)
    same_fields(l2_default, l2_sidecar, invariant_fields, "l2 sidecar", errors)

    if l2_default["cutpoint_count"] != 0 or l2_sidecar["cutpoint_count"] != 0:
        errors.append("default/sidecar-only route unexpectedly has a cutpoint")
    if l2_cutpoint["cutpoint_count"] != 1:
        errors.append("cutpoint route missing contract")
    if l2_default["has_ntsc_suffix"] or l2_default["has_ntsc_define"]:
        errors.append("default route unexpectedly carries sidecar flag")
    if not l2_cutpoint["has_ntsc_suffix"] or not l2_cutpoint["has_ntsc_define"]:
        errors.append("cutpoint route did not imply compiled sidecar image")
    if l2_cutpoint["has_nttop_suffix"] or l2_cutpoint["has_nttop_define"]:
        errors.append("cutpoint unexpectedly enabled topembed path")
    if l2_cutpoint["has_ntpr_suffix"] or l2_cutpoint["has_ntpr_define"]:
        errors.append("cutpoint unexpectedly enabled pruned path")
    check_cutpoint(l2_cutpoint, errors)

    for tag, row in (
        ("l1 negative", l1_cutpoint),
        ("small negative", small_cutpoint),
        ("supertile off", l2_super_off),
        ("pdf producer off", l2_pdfprod_off),
        ("pdfonly off", l2_pdfonly_off),
        ("reg epilogue off", l2_reg_off),
    ):
        if row["cutpoint_count"]:
            errors.append(f"{tag} unexpectedly exposed a cutpoint")
        if row["has_ntsc_suffix"] or row["has_ntsc_define"]:
            errors.append(f"{tag} unexpectedly enabled sidecar image")

    summary = {
        "pass": not errors,
        "errors": errors,
        "rows": [
            l2_default,
            l2_sidecar,
            l2_cutpoint,
            l1_cutpoint,
            small_cutpoint,
            l2_super_off,
            l2_pdfprod_off,
            l2_pdfonly_off,
            l2_reg_off,
        ],
    }
    print(
        "QWEN_NT_SIDECAR_CUTPOINT_ROUTE_SUMMARY " + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
