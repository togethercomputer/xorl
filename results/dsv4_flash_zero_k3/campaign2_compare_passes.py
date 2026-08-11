#!/usr/bin/env python3
"""First divergent component between two sampler dump passes (same process).

Usage: campaign2_compare_passes.py --root <dump_dir> [--proc <name>]
Compares every pair of passes in pass order per process dir; for the first
pair that differs, walks the dump keys in insertion order and prints the
first key whose tensors are not byte-identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_pass(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=True)
    d.pop("__metadata__", None)
    return d


def tensors_equal(a, b) -> bool:
    if torch.is_tensor(a) != torch.is_tensor(b):
        return False
    if torch.is_tensor(a):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        av = a.contiguous().flatten().view(torch.uint8)
        bv = b.contiguous().flatten().view(torch.uint8)
        return bool(torch.equal(av, bv))
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--proc", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    proc_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if args.proc:
        proc_dirs = [p for p in proc_dirs if p.name == args.proc]
    for proc in proc_dirs:
        passes = sorted(proc.glob("Pass*.pt"))
        print(f"[{proc.name}] {len(passes)} passes")
        if len(passes) < 2:
            continue
        base = load_pass(passes[0])
        for other_path in passes[1:]:
            other = load_pass(other_path)
            diverged = None
            for key in base:
                if key not in other:
                    continue
                if not tensors_equal(base[key], other[key]):
                    diverged = key
                    break
            if diverged is None:
                print(f"  {passes[0].name} vs {other_path.name}: identical on shared keys")
            else:
                a, b = base[diverged], other[diverged]
                n_bad = None
                if torch.is_tensor(a) and a.shape == b.shape and a.dtype.is_floating_point:
                    mismatch = a != b
                    n_bad = int(mismatch.sum())
                    rows = torch.unique(mismatch.reshape(mismatch.shape[0], -1).any(dim=-1).nonzero())[:8]
                    print(
                        f"  {passes[0].name} vs {other_path.name}: FIRST DIVERGENCE at {diverged} "
                        f"shape={tuple(a.shape)} n_mismatch={n_bad} first_rows={rows.tolist()}"
                    )
                else:
                    print(f"  {passes[0].name} vs {other_path.name}: FIRST DIVERGENCE at {diverged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
