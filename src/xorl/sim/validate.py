"""Run the portable simulator, calibration-pack, and analytical golden gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analytical_ledgers import activation_ledger, communication_ledger, flops_ledger
from .benchmark_behavior import load_benchmark_behavior_points
from .calibration_packs import (
    list_calibration_packs,
    load_calibration_pack,
    validate_calibration_pack,
)
from .config_fingerprint import load_training_config
from .model_metadata import resolve_model_metadata
from .predict import build_report
from .schemas import to_jsonable
from .tradeoff_ranker import rank_benchmark_tradeoffs


def _check(name: str, expected: Any, actual: Any, *, tolerance: float | None = None) -> dict[str, Any]:
    if tolerance is None:
        passed = expected == actual
    else:
        passed = expected is not None and actual is not None and abs(float(expected) - float(actual)) <= tolerance
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "expected": expected,
        "actual": actual,
    }


def validate_simulator_pack(name: str) -> dict[str, Any]:
    pack = load_calibration_pack(name)
    manifest = pack.manifest
    golden = manifest.get("golden", {})
    balanced_routing = bool(manifest.get("balanced_routing", False))
    sanitation = validate_calibration_pack(pack.path)
    points = load_benchmark_behavior_points(pack.path)
    raw_config = load_training_config(pack.default_config)
    report = build_report(
        pack.default_config,
        world_size=None,
        local_world_size=None,
        balanced_routing=balanced_routing,
        num_experts=None,
        top_k=None,
        benchmark_dir=pack.path,
    )
    topology = report.fingerprint.topology
    metadata = resolve_model_metadata(raw_config)
    train = raw_config.get("train", {})
    flops = flops_ledger(metadata, topology)
    activations = activation_ledger(metadata, topology, train)
    communication = communication_ledger(metadata, topology, train)
    tradeoffs = rank_benchmark_tradeoffs(pack.path)

    default_tps = report.benchmark_behavior.tokens_per_sec if report.benchmark_behavior is not None else None
    best_raw_tps = tradeoffs.best_raw.score_tokens_per_sec if tradeoffs.best_raw is not None else None
    best_promotable_tps = (
        tradeoffs.best_promotable.score_tokens_per_sec if tradeoffs.best_promotable is not None else None
    )
    checks = [
        _check("pack_sanitation", "pass", sanitation["status"]),
        _check("behavior_point_count", golden.get("behavior_point_count"), len(points)),
        _check("world_size", golden.get("world_size"), topology.world_size),
        _check("global_batch_size", golden.get("global_batch_size"), topology.global_batch_size),
        _check("default_tokens_per_sec", golden.get("default_tokens_per_sec"), default_tps, tolerance=0.001),
        _check("best_raw_tokens_per_sec", golden.get("best_raw_tokens_per_sec"), best_raw_tps, tolerance=0.001),
        _check(
            "best_promotable_tokens_per_sec",
            golden.get("best_promotable_tokens_per_sec"),
            best_promotable_tps,
            tolerance=0.001 if golden.get("best_promotable_tokens_per_sec") is not None else None,
        ),
        _check(
            "analytic_peak_floor_gb",
            golden.get("analytic_peak_floor_gb"),
            report.memory.analytic_peak_floor_gb,
            tolerance=0.001,
        ),
        _check("flops_ledger", "exact_analytic", flops.get("status")),
        _check("activation_ledger", "exact_analytic_lower_bound", activations.get("status")),
        _check("communication_ledger", "exact_analytic_bytes", communication.get("status")),
        _check("simulator_support", True, report.support.support_status.startswith("supported_")),
    ]
    return {
        "name": pack.name,
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "manifest": manifest,
        "checks": checks,
        "sanitation": sanitation,
        "prediction_report": to_jsonable(report),
        "tradeoff_report": to_jsonable(tradeoffs),
        "analytical": {
            "flops": flops,
            "activations": activations,
            "communication": communication,
        },
    }


def validate_simulator(pack_names: list[str] | None = None) -> dict[str, Any]:
    names = pack_names or list_calibration_packs()
    packs = [validate_simulator_pack(name) for name in names]
    check_count = sum(len(pack["checks"]) + len(pack["sanitation"]["checks"]) for pack in packs)
    failed_check_count = sum(
        check["status"] != "pass" for pack in packs for check in [*pack["checks"], *pack["sanitation"]["checks"]]
    )
    return {
        "schema_version": 1,
        "status": "pass" if packs and all(pack["status"] == "pass" for pack in packs) else "fail",
        "pack_count": len(packs),
        "check_count": check_count,
        "failed_check_count": failed_check_count,
        "packs": packs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", action="append", default=None, help="Validate only this built-in pack; repeatable")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-fail-on-error", action="store_true")
    args = parser.parse_args()

    payload = validate_simulator(args.pack)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if payload["status"] != "pass" and not args.no_fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
