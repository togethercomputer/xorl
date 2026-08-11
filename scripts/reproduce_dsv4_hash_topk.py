#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Replay one captured DSV4 hash-topk boundary without loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=129280)
    args = parser.parse_args()

    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    prefix = f"model.layers.{args.layer}."
    logits = capture[prefix + "router_logits"][: args.rows].cuda().contiguous()
    input_ids = capture[prefix + "moe_native_gathered_ids"][: args.rows].cuda().contiguous()
    expected_ids = capture[prefix + "router_selected_experts"][: args.rows].cuda().to(torch.int32).contiguous()
    expected_weights = capture[prefix + "router_routing_weights"][: args.rows].cuda().to(torch.float32).contiguous()
    topk = expected_ids.shape[1]
    table = torch.zeros(args.vocab_size, topk, dtype=torch.int32, device="cuda")
    table[input_ids] = expected_ids

    from sglang.kernels.ops.attention.dsv4 import hash_topk

    weights, ids = hash_topk(
        router_logits=logits,
        input_ids=input_ids,
        tid2eid=table,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        use_pdl=False,
    )
    torch.cuda.synchronize()
    result = {
        "rows": args.rows,
        "ids_byte_equal": torch.equal(ids.view(torch.uint8), expected_ids.view(torch.uint8)),
        "weights_byte_equal": torch.equal(weights.view(torch.uint8), expected_weights.view(torch.uint8)),
        "max_weight_abs_diff": float((weights - expected_weights).abs().max()),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
