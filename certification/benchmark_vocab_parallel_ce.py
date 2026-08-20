#!/usr/bin/env python3
"""Benchmark eager and compiled vocab-parallel CE outside the pytest suite.

Run with:
    PYTHONPATH=src torchrun --nproc_per_node=2 certification/benchmark_vocab_parallel_ce.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


def _benchmark_once(hidden, weight, labels, *, compiled, iterations, backward):
    for _ in range(5):
        loss = vocab_parallel_cross_entropy(hidden, weight, labels, dist.group.WORLD, use_compile=compiled)
        if backward:
            loss.sum().backward()
            hidden.grad = None
            weight.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(iterations):
        loss = vocab_parallel_cross_entropy(hidden, weight, labels, dist.group.WORLD, use_compile=compiled)
        if backward:
            loss.sum().backward()
            hidden.grad = None
            weight.grad = None
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iterations * 1000
    peak_memory = torch.cuda.max_memory_allocated()
    return {
        "milliseconds": elapsed_ms,
        "peak_activation_mb": (peak_memory - memory_before) / 1024**2,
        "peak_total_mb": peak_memory / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    torch.manual_seed(42)
    tokens, hidden_size, vocabulary = 4096, 4096, 152064
    local_vocabulary = vocabulary // world_size
    hidden = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(local_vocabulary, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, vocabulary, (tokens,), device="cuda")

    results = {}
    for backward in (False, True):
        phase = "forward_backward" if backward else "forward"
        results[phase] = {}
        for compiled in (False, True):
            mode = "compiled" if compiled else "eager"
            results[phase][mode] = _benchmark_once(
                hidden,
                weight,
                labels,
                compiled=compiled,
                iterations=args.iterations,
                backward=backward,
            )
        dist.barrier()

    if rank == 0:
        print(json.dumps(results, indent=2, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
