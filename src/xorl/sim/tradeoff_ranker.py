"""Rank calibrated benchmark tradeoffs across parallelism/config choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


try:
    from .benchmark_behavior import load_benchmark_behavior_points, predict_benchmark_behavior
    from .calibration_packs import resolve_calibration_pack
    from .config_fingerprint import load_training_config, resolve_topology
    from .schemas import BenchmarkBehaviorPoint, Topology, TradeoffCandidate, TradeoffReport, to_jsonable
    from .shape_engine import build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import load_benchmark_behavior_points, predict_benchmark_behavior
    from calibration_packs import resolve_calibration_pack
    from config_fingerprint import load_training_config, resolve_topology
    from schemas import BenchmarkBehaviorPoint, Topology, TradeoffCandidate, TradeoffReport, to_jsonable
    from shape_engine import build_shape_ledger


def _config_output_basename(config_path: Path) -> str | None:
    raw_config = load_training_config(config_path)
    output_dir = raw_config.get("train", {}).get("output_dir")
    return Path(output_dir).name if output_dir else None


def _config_topology(config_path: Path, *, world_size: int | None, local_world_size: int | None) -> Topology:
    raw_config = load_training_config(config_path)
    return resolve_topology(raw_config, world_size=world_size, local_world_size=local_world_size)


def _behavior_key(point: BenchmarkBehaviorPoint) -> str:
    if point.label.startswith("best_by_mfu:"):
        return point.label.split(":", 1)[1]
    if ":" in point.label:
        return point.label.split(":", 1)[1]
    return point.label


def _find_matching_config(
    point: BenchmarkBehaviorPoint,
    configs: list[Path],
    *,
    world_size: int | None,
    local_world_size: int | None,
) -> tuple[Path | None, Topology | None]:
    behavior_key = _behavior_key(point)
    output_names = {path: _config_output_basename(path) for path in configs}
    for path, output_name in output_names.items():
        if output_name and output_name == behavior_key:
            return path, _config_topology(path, world_size=world_size, local_world_size=local_world_size)

    for path in configs:
        topology = _config_topology(path, world_size=world_size, local_world_size=local_world_size)
        if (
            point.micro_batch_size == topology.micro_batch_size
            and point.global_batch_size == topology.global_batch_size
        ):
            if point.expert_parallel_size is None or point.expert_parallel_size == topology.expert_parallel_size:
                return path, topology
    return None, None


def _is_promotable(point: BenchmarkBehaviorPoint) -> bool:
    return point.correctness_status == "k3_pass"


def _candidate_sort_key(candidate: TradeoffCandidate) -> tuple[float, float]:
    return (
        candidate.score_tokens_per_sec if candidate.score_tokens_per_sec is not None else float("-inf"),
        candidate.score_tflops_per_gpu if candidate.score_tflops_per_gpu is not None else float("-inf"),
    )


def _fallback_topology(
    point: BenchmarkBehaviorPoint,
    *,
    world_size: int | None,
    local_world_size: int | None,
) -> Topology:
    resolved_world_size = world_size or point.gpu_count or 1
    resolved_local_world_size = local_world_size or (8 if resolved_world_size % 8 == 0 else resolved_world_size)
    global_batch_size = point.global_batch_size or point.micro_batch_size or 1
    micro_batch_size = point.micro_batch_size or 1
    return Topology(
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        node_count=max(resolved_world_size // resolved_local_world_size, 1),
        data_parallel_size=1,
        data_parallel_replicate_size=1,
        data_parallel_shard_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=point.expert_parallel_size or 1,
        ep_fsdp_size=point.ep_fsdp_size,
        ulysses_parallel_size=1,
        ringattn_parallel_size=1,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=max(global_batch_size // micro_batch_size, 1),
        global_batch_size=global_batch_size,
        sample_packing_sequence_len=None,
        num_experts=None,
        top_k=None,
    )


def rank_benchmark_tradeoffs(
    benchmark_dir: str | Path,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
) -> TradeoffReport:
    benchmark_path = resolve_calibration_pack(benchmark_dir)
    configs = sorted((benchmark_path / "configs").glob("*.yaml"))
    behavior_points = load_benchmark_behavior_points(benchmark_path)
    warnings: list[str] = []
    candidates: list[TradeoffCandidate] = []

    for point in behavior_points:
        config_path, topology = _find_matching_config(
            point,
            configs,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        if topology is None:
            warnings.append(f"no matching config found for behavior point {point.label}")
            fallback_topology = _fallback_topology(
                point,
                world_size=world_size,
                local_world_size=local_world_size,
            )
            behavior = predict_benchmark_behavior(
                [point],
                fallback_topology,
                build_shape_ledger(fallback_topology, balanced_routing=True),
            )
        else:
            shape = build_shape_ledger(topology, balanced_routing=True)
            behavior = predict_benchmark_behavior([point], topology, shape)

        notes = list(point.notes)
        if point.correctness_status and point.correctness_status != "k3_pass":
            notes.append(f"not promotable: {point.correctness_status}")
        candidates.append(
            TradeoffCandidate(
                label=point.label,
                config_path=str(config_path) if config_path else None,
                behavior_source=point.source,
                topology=topology,
                behavior=behavior,
                promotable=_is_promotable(point),
                score_tokens_per_sec=behavior.tokens_per_sec,
                score_tflops_per_gpu=behavior.tflops_per_gpu,
                notes=notes,
            )
        )

    candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
    best_raw = candidates[0] if candidates else None
    promotable = [candidate for candidate in candidates if candidate.promotable]
    best_promotable = promotable[0] if promotable else None
    if best_raw is not None and not best_raw.promotable:
        warnings.append(f"best raw candidate {best_raw.label} is not correctness-promotable")
    if best_promotable is None:
        warnings.append("no correctness-promotable candidate found")

    return TradeoffReport(
        benchmark_dir=str(benchmark_path),
        status="ok" if candidates else "no_behavior_points",
        candidate_count=len(candidates),
        best_raw=best_raw,
        best_promotable=best_promotable,
        candidates=candidates,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", help="Path or built-in calibration-pack name")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-world-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = rank_benchmark_tradeoffs(
        args.benchmark_dir,
        world_size=args.world_size,
        local_world_size=args.local_world_size,
    )
    rendered = json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
