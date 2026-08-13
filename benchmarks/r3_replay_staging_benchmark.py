"""Benchmark R3 replay setup plus forward/recompute device access.

Run this script with PYTHONPATH bound to either the baseline or candidate XoRL
source tree.  It deliberately uses the production RoutingReplayHandler and
RoutingReplay implementations while replacing only model discovery and the
distributed parallel-state lookup.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from xorl.models.layers.moe.routing_replay import RoutingReplay
from xorl.server.runner.utils import routing_replay_handler as rrh


def _synchronize() -> None:
    torch.cuda.synchronize()


def _measure_once(
    routed_experts: np.ndarray,
    routed_weights: np.ndarray,
    rows: int,
    layers: int,
) -> dict[str, float | bool]:
    RoutingReplay._instances.clear()
    blocks = [SimpleNamespace(_routing_replay=RoutingReplay()) for _ in range(layers)]
    model = SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=routed_experts.shape[2]))
    handler = rrh.RoutingReplayHandler(model)
    handler.get_moe_blocks = lambda blocks=blocks: blocks
    rrh.get_parallel_state = lambda: SimpleNamespace(cp_enabled=False)
    micro_batches = [
        {
            "input_ids": torch.zeros((1, rows), dtype=torch.long),
            "num_samples": 1,
        }
    ]

    _synchronize()
    total_start = time.perf_counter()
    setup_start = total_start
    assert handler.fill_routing_replay(
        micro_batches,
        [routed_experts],
        [routed_weights],
    )
    _synchronize()
    setup_s = time.perf_counter() - setup_start

    forward_start = time.perf_counter()
    forward = []
    for block in blocks:
        replay = block._routing_replay
        forward.append((replay.pop_forward(), replay.pop_forward_weights()))
    _synchronize()
    forward_s = time.perf_counter() - forward_start

    backward_start = time.perf_counter()
    backward = []
    for block in blocks:
        replay = block._routing_replay
        backward.append((replay.pop_backward(), replay.pop_backward_weights()))
    _synchronize()
    backward_s = time.perf_counter() - backward_start
    total_s = time.perf_counter() - total_start

    sample_rows = torch.tensor([0, rows // 2, rows - 1], device="cuda")
    for layer in (0, layers // 2, layers - 1):
        expected_ids = torch.from_numpy(routed_experts[:, layer]).long().cuda()[sample_rows]
        expected_weights = torch.from_numpy(routed_weights[:, layer]).cuda()[sample_rows]
        torch.testing.assert_close(forward[layer][0][sample_rows], expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(backward[layer][0][sample_rows], expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(forward[layer][1][sample_rows], expected_weights, rtol=0, atol=0)
        torch.testing.assert_close(backward[layer][1][sample_rows], expected_weights, rtol=0, atol=0)

    index_devices = {str(block._routing_replay.top_indices_list[0].device) for block in blocks}
    weight_devices = {str(block._routing_replay.top_weights_list[0].device) for block in blocks}
    index_storages = {
        block._routing_replay.top_indices_list[0].untyped_storage().data_ptr() for block in blocks
    }
    weight_storages = {
        block._routing_replay.top_weights_list[0].untyped_storage().data_ptr() for block in blocks
    }
    result: dict[str, float | bool] = {
        "setup_s": setup_s,
        "forward_s": forward_s,
        "recompute_s": backward_s,
        "forward_plus_recompute_s": forward_s + backward_s,
        "total_s": total_s,
        "device_resident": index_devices == {"cuda:0"} and weight_devices == {"cuda:0"},
        "single_index_backing_storage": len(index_storages) == 1,
        "single_weight_backing_storage": len(weight_storages) == 1,
    }
    metrics = getattr(handler, "last_setup_metrics", {})
    for key, value in metrics.items():
        result[key] = float(value)

    del forward, backward, blocks, handler
    torch.cuda.empty_cache()
    return result


def _median(rows: list[dict[str, float | bool]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--rows", type=int, default=46_000)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    rng = np.random.default_rng(20260813)
    shape = (args.rows, args.layers, args.topk)
    routed_experts = rng.integers(0, 256, size=shape, dtype=np.int32)
    routed_weights = rng.random(shape, dtype=np.float32)
    routed_weights /= routed_weights.sum(axis=2, keepdims=True)

    for _ in range(args.warmups):
        _measure_once(routed_experts, routed_weights, args.rows, args.layers)
    measurements = [
        _measure_once(routed_experts, routed_weights, args.rows, args.layers)
        for _ in range(args.iterations)
    ]

    summary = {
        "schema": "xorl.r3_replay_staging_benchmark.v1",
        "label": args.label,
        "node": __import__("socket").gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "xorl_handler_module": str(Path(rrh.__file__).resolve()),
        "shape": list(shape),
        "host_input_bytes": int(routed_experts.nbytes + routed_weights.nbytes),
        "device_replay_bytes": int(routed_experts.size * 8 + routed_weights.nbytes),
        "iterations": args.iterations,
        "median_setup_s": _median(measurements, "setup_s"),
        "median_forward_s": _median(measurements, "forward_s"),
        "median_recompute_s": _median(measurements, "recompute_s"),
        "median_forward_plus_recompute_s": _median(measurements, "forward_plus_recompute_s"),
        "median_total_s": _median(measurements, "total_s"),
        "all_device_resident": all(bool(row["device_resident"]) for row in measurements),
        "all_single_index_backing_storage": all(
            bool(row["single_index_backing_storage"]) for row in measurements
        ),
        "all_single_weight_backing_storage": all(
            bool(row["single_weight_backing_storage"]) for row in measurements
        ),
        "measurements": measurements,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
