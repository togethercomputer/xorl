#!/usr/bin/env python3
"""Compare the offline-recomputed logprob row against serving's top-k anchors."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

RD = Path(__file__).resolve().parent
SNAP = Path(
    "/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
    "60d8d70770c6776ff598c94bb586a859a38244f1"
)


def main() -> int:
    idx = json.load(open(SNAP / "model.safetensors.index.json"))
    with safe_open(SNAP / idx["weight_map"]["head.weight"], framework="pt", device="cuda") as f:
        weight = f.get_tensor("head.weight")
    trainer = torch.load(
        RD / "campaign2/dumps_dec39_trainer2/components.rank0.pt",
        map_location="cpu",
        weights_only=False,
    )
    hidden = trainer["model.layers.-1.final_norm.occurrence00039"].reshape(1, -1).cuda()
    logits = F.linear(hidden, weight)
    lp = torch.log_softmax(logits, dim=-1)[0]
    serving_topk = {
        294: -0.000823974609375,
        16: -7.75,
        339: -9.125,
        603: -9.625,
        271: -10.5,
        6810: -10.625,
        3189: -10.75,
        1099: -11.0,
    }
    for tok, sv in serving_topk.items():
        mine = float(lp[tok])
        verdict = "MATCH" if mine == sv else "DIFF"
        print(f"token {tok}: mine {mine!r} serving {sv!r} {verdict}")

    l64 = logits.double()[0]
    lse64 = torch.logsumexp(l64, dim=-1)
    lp64 = float(l64[294] - lse64)
    print("f64 logprob(294):", lp64)
    boundary = (-0.000823974609375 + -0.000827789306640625) / 2
    print("bf16 boundary:", boundary, "| f64 distance from boundary:", lp64 - boundary)
    for name, v in [("serving", -0.000823974609375), ("trainer", -0.000827789306640625)]:
        print(name, hex(struct.unpack("<I", struct.pack("<f", v))[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
