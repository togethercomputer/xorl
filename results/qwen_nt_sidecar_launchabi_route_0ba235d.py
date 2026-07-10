#!/usr/bin/env python3
"""Host proof for the qwen NT lm-head sidecar launch ABI wrapper."""

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
    __file__ = "/tmp/fake-qwen-nt-sidecar-launchabi.so"

    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = dict(kwargs)
        self.sidecar_calls: list[dict[str, object]] = []

    def run_qwen_nt_lmhead_sidecar(self, instrs, ins, t0, t1, bufs, smem_bytes) -> None:
        self.sidecar_calls.append(
            {
                "instrs_device": str(getattr(instrs, "device", "")),
                "bufs_device": str(getattr(bufs, "device", "")),
                "ins": int(ins),
                "t0": int(t0),
                "t1": int(t1),
                "smem_bytes": int(smem_bytes),
            }
        )


class NoExportExt:
    pass


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


CFG_L2 = Cfg(H=2560, L=2, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)
SIDECAR_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR"
CUTPOINT_ENV = "MK_GEMM_N256_NT_SUPERTILE_SIDECAR_CUTPOINT"
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


def build(updates: dict[str, str | None]) -> MKQwen3:
    old = with_env(updates)
    try:
        return MKQwen3(CFG_L2, dev="meta", seed=0)
    finally:
        restore_env(old)


def expect_raises(fn, fragment: str, errors: list[str], label: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - proof wants the exact guard text.
        if fragment not in str(exc):
            errors.append(f"{label} raised wrong error: {exc!r}")
    else:
        errors.append(f"{label} did not raise")


def main() -> None:
    errors: list[str] = []
    default_model = build({})
    sidecar_only = build({SIDECAR_ENV: "1"})
    cutpoint_model = build({CUTPOINT_ENV: "1"})

    cutpoints = cutpoint_model.qwen_nt_sidecar_cutpoints
    if len(cutpoints) != 1:
        errors.append(f"cutpoint model exposed {len(cutpoints)} cutpoints")
    else:
        cp = cutpoints[0]
        if cp["instr_index"] != 37 or cp["ntiles"] != 4748:
            errors.append(f"unexpected cutpoint identity: {cp!r}")

    cutpoint_model.prog.run_qwen_nt_lmhead_sidecar(
        cutpoint_model.ext,
        cutpoint_model._smem_bytes,
        tile_start=0,
        tile_stop=1,
    )
    cutpoint_model.prog.run_qwen_nt_lmhead_sidecar(
        cutpoint_model.ext,
        cutpoint_model._smem_bytes,
    )
    calls = cutpoint_model.ext.sidecar_calls
    expected_calls = [
        {
            "instrs_device": "meta",
            "bufs_device": "meta",
            "ins": 37,
            "t0": 0,
            "t1": 1,
            "smem_bytes": 151552,
        },
        {
            "instrs_device": "meta",
            "bufs_device": "meta",
            "ins": 37,
            "t0": 0,
            "t1": 4748,
            "smem_bytes": 151552,
        },
    ]
    if calls != expected_calls:
        errors.append(f"wrong sidecar call ABI: {calls!r}")

    expect_raises(
        lambda: default_model.prog.run_qwen_nt_lmhead_sidecar(default_model.ext, default_model._smem_bytes),
        "expected exactly one qwen NT sidecar cutpoint",
        errors,
        "default no-cutpoint guard",
    )
    expect_raises(
        lambda: sidecar_only.prog.run_qwen_nt_lmhead_sidecar(sidecar_only.ext, sidecar_only._smem_bytes),
        "expected exactly one qwen NT sidecar cutpoint",
        errors,
        "sidecar-only no-cutpoint guard",
    )
    expect_raises(
        lambda: cutpoint_model.prog.run_qwen_nt_lmhead_sidecar(NoExportExt(), cutpoint_model._smem_bytes),
        "extension was not built with qwen NT sidecar export",
        errors,
        "missing export guard",
    )
    expect_raises(
        lambda: cutpoint_model.prog.run_qwen_nt_lmhead_sidecar(
            cutpoint_model.ext, cutpoint_model._smem_bytes, tile_start=10, tile_stop=10
        ),
        "invalid qwen NT sidecar tile range",
        errors,
        "empty range guard",
    )

    cflags = list(cutpoint_model.ext.kwargs["extra_cuda_cflags"])
    name = str(cutpoint_model.ext.kwargs["name"])
    if "_ntsc" not in name or SIDECAR_DEFINE not in cflags:
        errors.append("cutpoint model did not build the sidecar-export image")

    summary = {
        "pass": not errors,
        "errors": errors,
        "default_cutpoints": len(default_model.qwen_nt_sidecar_cutpoints),
        "sidecar_only_cutpoints": len(sidecar_only.qwen_nt_sidecar_cutpoints),
        "cutpoint_count": len(cutpoints),
        "calls": calls,
        "name": name,
        "has_ntsc_define": SIDECAR_DEFINE in cflags,
    }
    print("QWEN_NT_SIDECAR_LAUNCHABI_SUMMARY " + json.dumps(summary, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
