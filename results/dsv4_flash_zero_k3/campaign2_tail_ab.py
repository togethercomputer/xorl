#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Offline A/B of the DSV4 logprob tail at decision 39.

Inputs: the byte-equal final-norm hidden (trainer dump, pseudo-layer -1) and
the checkpoint head weight. Reproduces the trainer tail (per-rank BF16
F.linear over 8 vocab shards, rank-order concat, BF16 log_softmax) and then
evaluates variants to find which GEMM program reproduces the serving wire
value instead.

wire: serving -0.000823974609375 (ba580000) / trainer -0.000827789306640625
(ba590000), selected token 294.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


RD = Path(__file__).resolve().parent
SNAP = Path(
    "/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
)
WIRE_SERVING = -0.000823974609375
WIRE_TRAINER = -0.000827789306640625
TOKEN = 294


def tag(v: float) -> str:
    if v == WIRE_SERVING:
        return "== SERVING"
    if v == WIRE_TRAINER:
        return "== TRAINER"
    return "neither"


def main() -> int:
    import json

    idx = json.load(open(SNAP / "model.safetensors.index.json"))
    shard = idx["weight_map"]["head.weight"]
    with safe_open(SNAP / shard, framework="pt", device="cuda") as f:
        weight = f.get_tensor("head.weight")
    print("head.weight", tuple(weight.shape), weight.dtype)

    trainer = torch.load(
        RD / "campaign2/dumps_dec39_trainer2/components.rank0.pt",
        map_location="cpu",
        weights_only=True,
    )
    hidden = trainer["model.layers.-1.final_norm.occurrence00039"].reshape(1, -1).cuda()
    print("hidden", tuple(hidden.shape), hidden.dtype)
    if weight.dtype is not torch.bfloat16:
        weight = weight.to(torch.bfloat16)

    vocab = weight.shape[0]
    shard_rows = vocab // 8

    def lp_of(logits: torch.Tensor) -> float:
        lp = torch.log_softmax(logits.reshape(1, -1), dim=-1)
        return float(lp[0, TOKEN])

    # t1: trainer program — 8 per-shard BF16 F.linear, rank-order concat.
    parts = [F.linear(hidden, weight[r * shard_rows : (r + 1) * shard_rows]) for r in range(8)]
    t1 = torch.cat(parts, dim=-1)
    print("t1 per-shard F.linear + concat:", repr(lp_of(t1)), tag(lp_of(t1)))

    # v1: one full-vocab BF16 F.linear.
    v1 = F.linear(hidden, weight)
    print("v1 full F.linear:", repr(lp_of(v1)), tag(lp_of(v1)))
    print("  v1 vs t1 logits equal:", bool(torch.equal(v1, t1)))

    # v2: M=8 rows (serving decode gathers 8 DP rows) then take row 0.
    h8 = hidden.expand(8, -1).contiguous()
    v2 = F.linear(h8, weight)[0:1]
    print("v2 full F.linear M=8:", repr(lp_of(v2)), tag(lp_of(v2)))

    # v3: per-shard with M=8.
    parts8 = [F.linear(h8, weight[r * shard_rows : (r + 1) * shard_rows]) for r in range(8)]
    v3 = torch.cat(parts8, dim=-1)[0:1]
    print("v3 per-shard M=8:", repr(lp_of(v3)), tag(lp_of(v3)))

    # v4: fp32 GEMM then bf16 cast (lm_head_fp32-style) per shard.
    parts32 = [F.linear(hidden.float(), weight[r * shard_rows : (r + 1) * shard_rows].float()) for r in range(8)]
    v4 = torch.cat(parts32, dim=-1).to(torch.bfloat16)
    print("v4 per-shard fp32->bf16:", repr(lp_of(v4)), tag(lp_of(v4)))

    # v5: batch-invariant matmul (serving BI interposition) per shard.
    try:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            matmul_persistent,
        )

        partsbi = [matmul_persistent(hidden, weight[r * shard_rows : (r + 1) * shard_rows].t()) for r in range(8)]
        v5 = torch.cat(partsbi, dim=-1)
        print("v5 per-shard matmul_persistent:", repr(lp_of(v5)), tag(lp_of(v5)))
        v6 = matmul_persistent(hidden, weight.t())
        print("v6 full matmul_persistent:", repr(lp_of(v6)), tag(lp_of(v6)))
    except Exception as exc:  # noqa: BLE001
        print("BI matmul unavailable:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
