#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Locate the first divergent layer/component at decision 39 across engines.

Serving side: campaign2/dumps_dec39/<rank0 proc>/Pass*.pt — the 40th
one-token decode pass of repetition 0 is decision 39.
Trainer side: campaign2/dumps_dec39_trainer/components.rank0.pt — occurrence
NNNNN suffix 40 (prefill is occurrence 0; decode k is occurrence k+1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


RD = Path(__file__).resolve().parent


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=int, default=39)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    serving_root = RD / "campaign2/dumps_dec39"
    proc = sorted(p for p in serving_root.iterdir() if p.name.startswith(f"TP{args.rank}_"))[0]

    # Identify decision k's decode pass by its absolute position: the forward
    # that PRODUCES decision k consumes the token at position prompt_len+k-1.
    import json

    trace = json.load(open(RD / "campaign2/trace_dec39_dump.json"))
    cap = trace["captures"][0]
    prompt_len = len(cap["prompt_ids"])
    want_position = prompt_len + args.decision - 1
    want_token = cap["full_ids"][want_position]

    serving = None
    pass_name = None
    n_passes = 0
    for pass_file in sorted(proc.glob("Pass*.pt")):
        n_passes += 1
        data = torch.load(pass_file, map_location="cpu", weights_only=True)
        ids = pos = None
        for key, value in data.items():
            if key.endswith("forward_batch_info.input_ids") and torch.is_tensor(value):
                ids = value
            if key.endswith("forward_batch_info.positions") and torch.is_tensor(value):
                pos = value
        if (
            torch.is_tensor(ids)
            and ids.numel() == 1
            and torch.is_tensor(pos)
            and int(pos.flatten()[0]) == want_position
            and int(ids.flatten()[0]) == want_token
        ):
            serving = data
            pass_name = pass_file.name
            break
    print(
        f"[serving] scanned {n_passes} passes; decision {args.decision} (pos {want_position}, tok {want_token}) -> {pass_name}"
    )
    if serving is None:
        raise SystemExit("no matching decode pass found")

    trainer = torch.load(
        RD / f"campaign2/dumps_dec39_trainer/components.rank{args.rank}.pt",
        map_location="cpu",
        weights_only=True,
    )
    # Trainer occurrence 0 is the prefill, which yields decision 0; decode
    # occurrence k (k >= 1) consumes position prompt_len+k-1 and yields
    # decision k.
    occ = args.decision
    suffix = f".occurrence{occ:05d}"

    def tkey(layer: int, name: str) -> torch.Tensor | None:
        return trainer.get(f"model.layers.{layer}.{name}{suffix}")

    PAIRS = [
        ("input.hidden_states", "layer_input"),
        ("self_attn", "attention"),
        ("mlp", "moe_native_combined"),
        ("mlp.gate", "moe_native_gathered_routing"),
    ]
    for layer in range(43):
        row = [f"layer {layer:2d}"]
        diverged = None
        for s_name, t_name in PAIRS:
            s = serving.get(f"model.layers.{layer}.{s_name}")
            t = tkey(layer, t_name)
            if s is None or t is None or not torch.is_tensor(s) or not torch.is_tensor(t):
                row.append(f"{t_name}: n/a")
                continue
            s2, t2 = s.reshape(-1), t.reshape(-1)
            if s2.numel() != t2.numel():
                row.append(f"{t_name}: shape {tuple(s.shape)} vs {tuple(t.shape)}")
                continue
            ok = bytes_equal(s2, t2)
            row.append(f"{t_name}: {'OK' if ok else 'DIFF'}")
            if not ok and diverged is None:
                diverged = (s_name, t_name, s2, t2)
        print(" | ".join(row))
        if diverged:
            s_name, t_name, s2, t2 = diverged
            mism = (s2 != t2).nonzero().flatten()
            print(
                f"  FIRST DIVERGENCE layer {layer} {s_name} vs {t_name}: "
                f"n_mismatch={mism.numel()} first_idx={mism[:5].tolist()} "
                f"serving={s2[mism[:3]].float().tolist()} trainer={t2[mism[:3]].float().tolist()}"
            )
            return 0
    print("no divergence found in compared components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
