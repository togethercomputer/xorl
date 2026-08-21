#!/usr/bin/env python3
"""Manually certify combined versus split GLM-5 sparse-MLA backward speed."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch


def _make_inputs(sequence, kv_sequence, heads, rank, tail, topk):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    query = torch.randn((sequence, heads, rank + tail), device="cuda", dtype=torch.bfloat16, generator=generator)
    kv = torch.randn((kv_sequence, 1, rank + tail), device="cuda", dtype=torch.bfloat16, generator=generator)
    relative = torch.arange(topk, device="cuda", dtype=torch.int64)
    query_positions = torch.arange(kv_sequence - sequence, kv_sequence, device="cuda", dtype=torch.int64)
    indices = query_positions.unsqueeze(1) - (topk - 1 - relative).unsqueeze(0)
    indices = indices.clamp(min=-1, max=kv_sequence - 1).to(torch.int32).unsqueeze(1)
    return query, kv, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=0.15)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise SystemExit("This certification requires an H100-class CUDA device")
    try:
        import tilelang  # noqa: F401, PLC0415
    except ImportError as exc:
        raise SystemExit("This certification requires TileLang") from exc

    from xorl.ops.families.glm5.sparse_mla import SparseMLA  # noqa: PLC0415

    sequence, kv_sequence, heads, rank, tail, topk = 2048, 32768, 64, 512, 64, 2048
    scale = (rank + tail) ** -0.5
    query, kv, indices = _make_inputs(sequence, kv_sequence, heads, rank, tail, topk)
    generator = torch.Generator(device="cuda").manual_seed(7)
    grad_output = torch.randn((sequence, heads, rank), device="cuda", dtype=torch.bfloat16, generator=generator)

    def time_backward() -> float:
        local_query = query.detach().clone().requires_grad_(True)
        local_kv = kv.detach().clone().requires_grad_(True)
        output, _ = SparseMLA.apply(local_query, local_kv, indices, scale)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output.backward(grad_output)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    previous = os.environ.get("XORL_GLM5_SPLIT_SPARSE_MLA_BWD")
    try:
        timings = {}
        for label, mode in (("combined", "0"), ("split", "1")):
            os.environ["XORL_GLM5_SPLIT_SPARSE_MLA_BWD"] = mode
            for _ in range(args.warmup):
                time_backward()
            timings[label] = [time_backward() for _ in range(args.trials)]
    finally:
        if previous is None:
            os.environ.pop("XORL_GLM5_SPLIT_SPARSE_MLA_BWD", None)
        else:
            os.environ["XORL_GLM5_SPLIT_SPARSE_MLA_BWD"] = previous

    combined_ms = statistics.median(timings["combined"])
    split_ms = statistics.median(timings["split"])
    speedup = 1.0 - combined_ms / split_ms
    result = {
        "combined_ms": combined_ms,
        "split_ms": split_ms,
        "speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "timings_ms": timings,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if speedup < args.minimum_speedup:
        raise SystemExit("Combined sparse-MLA backward did not meet the requested speedup")


if __name__ == "__main__":
    main()
