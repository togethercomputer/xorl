#!/usr/bin/env python3
"""Host proof for qwen NT sidecar PDF subprogram split tensors."""

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
    __file__ = "/tmp/fake-qwen-nt-sidecar-subprogram.so"

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
        return MKQwen3(CFG, dev="meta", seed=0)
    finally:
        restore_env(old)


def sub_summary(sub: dict[str, object]) -> dict[str, object]:
    return {
        "indices": sub["indices"],
        "n_instr": sub["n_instr"],
        "critical_path": sub["critical_path"],
        "dropped_deps": sub["dropped_deps"],
        "local_deps_by_index": sub["local_deps_by_index"],
        "dep_cnt_head": sub["dep_cnt_list"][:8],
        "dep_cnt_tail": sub["dep_cnt_list"][-8:],
        "adj_edges": len(sub["adj_list"]),
        "instr_numel": int(sub["instrs"].numel()),
        "state_numel": int(sub["state"].numel()),
    }


def main() -> None:
    model = build()
    subs = model.prog.qwen_nt_sidecar_pdf_subprograms()
    prefix = sub_summary(subs["prefix"])
    post = sub_summary(subs["post"])
    expected_post_dropped = {
        38: [2, 12, 37],
        39: [2, 37],
        40: [37],
        41: [36, 37],
    }
    errors: list[str] = []
    if subs["cutpoint"]["instr_index"] != 37:
        errors.append(f"cutpoint mismatch: {subs['cutpoint']!r}")
    if subs["prefix_indices"] != list(range(37)):
        errors.append(f"prefix indices mismatch: {subs['prefix_indices']!r}")
    if subs["post_indices"] != list(range(38, 78)):
        errors.append(f"post indices mismatch: {subs['post_indices']!r}")
    if prefix["dropped_deps"] != {}:
        errors.append(f"prefix dropped deps: {prefix['dropped_deps']!r}")
    post_dropped = post["dropped_deps"]
    bad_external = {row: deps for row, deps in post_dropped.items() if any(dep >= 38 for dep in deps)}
    if bad_external:
        errors.append(f"post dropped non-external deps: {bad_external!r}")
    for row, deps in expected_post_dropped.items():
        if post_dropped.get(row) != deps:
            errors.append(f"post dropped deps for row {row}: {post_dropped.get(row)!r}")
    for row, deps in {39: [38], 40: [39], 41: [39]}.items():
        if post["local_deps_by_index"].get(row) != deps:
            errors.append(f"post local deps for row {row}: {post['local_deps_by_index'].get(row)!r} != {deps!r}")
    if prefix["n_instr"] != 37 or post["n_instr"] != 40:
        errors.append(f"wrong subprogram sizes: {prefix['n_instr']} / {post['n_instr']}")
    if prefix["instr_numel"] != 37 * mk.INSTR_INTS:
        errors.append("prefix instr tensor shape mismatch")
    if post["instr_numel"] != 40 * mk.INSTR_INTS:
        errors.append("post instr tensor shape mismatch")

    summary = {
        "pass": not errors,
        "errors": errors,
        "n_instr": int(model.prog.n_instr),
        "critical_path": int(model.prog.critical_path),
        "gated": int(model.prog.n_gated),
        "smem_bytes": int(model._smem_bytes),
        "cutpoint": subs["cutpoint"],
        "prefix": prefix,
        "post": post,
    }
    print("QWEN_NT_SIDECAR_SUBPROGRAM_ROUTE " + json.dumps(summary, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
