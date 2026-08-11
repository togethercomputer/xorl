#!/usr/bin/env python3
"""Locate the first divergent layer-0..2 component between engines.

Trainer side: dumps/trainer_ruler_rep0.rank*.pt (component capture from the
decode-cache replay). Sampler side: dumps/sampler_base/<proc>/Pass*.pt from
the tensor-dump forward hook. Run with --list first to see sampler names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

RD = Path(__file__).resolve().parent


def load_trainer(rep: int = 0, rank: int = 0) -> dict:
    d = torch.load(RD / f"dumps/trainer_ruler_rep{rep}.rank{rank}.pt", map_location="cpu", weights_only=False)
    d.pop("__metadata__", None)
    d.pop("labels", None)
    return d


def sampler_passes(root: Path):
    for proc_dir in sorted(root.iterdir()):
        if not proc_dir.is_dir():
            continue
        for pass_file in sorted(proc_dir.glob("Pass*.pt")):
            yield proc_dir.name, pass_file


def describe(t: torch.Tensor) -> str:
    if t.dtype.is_floating_point:
        f = t.float()
        return f"{tuple(t.shape)} {t.dtype} mean={f.mean():.6g} absmax={f.abs().max():.6g}"
    return f"{tuple(t.shape)} {t.dtype}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list sampler pass files and tensor names")
    parser.add_argument("--pass-file", type=Path, help="specific sampler Pass file to compare")
    parser.add_argument("--prompt-len", type=int, default=10)
    args = parser.parse_args()

    root = RD / "dumps/sampler_base"
    if args.list:
        for proc, pass_file in sampler_passes(root):
            data = torch.load(pass_file, map_location="cpu", weights_only=False)
            ids = data.get("model.forward_batch_info.input_ids")
            n = ids.numel() if isinstance(ids, torch.Tensor) else "?"
            print(f"{proc}/{pass_file.name}: {len(data)} tensors, input_ids={n}")
        return

    if not args.pass_file:
        # Pick the prefill pass of the real request: input_ids length == prompt len.
        candidates = []
        for proc, pass_file in sampler_passes(root):
            data = torch.load(pass_file, map_location="cpu", weights_only=False)
            for key, value in data.items():
                if key.endswith("forward_batch_info.input_ids") and isinstance(value, torch.Tensor):
                    if value.numel() == args.prompt_len:
                        candidates.append((proc, pass_file))
                    break
        if not candidates:
            raise SystemExit("no sampler pass with the prompt length found; use --list")
        proc, pass_file = candidates[0]
        print(f"using {proc}/{pass_file.name}")
    else:
        pass_file = args.pass_file

    sampler = torch.load(pass_file, map_location="cpu", weights_only=False)
    trainer = load_trainer()

    print("=== sampler tensor names (layer 0-2 subset) ===")
    for name in sorted(sampler):
        if any(f"layers.{i}." in name for i in (0, 1, 2)) or "embed" in name:
            print(" ", name, describe(sampler[name]) if isinstance(sampler[name], torch.Tensor) else type(sampler[name]))

    print("=== trainer components (layer 0, prefill) ===")
    for name in sorted(trainer):
        if name.startswith("model.layers.0.") and "occurrence" not in name:
            print(" ", name, describe(trainer[name]))


if __name__ == "__main__":
    main()
