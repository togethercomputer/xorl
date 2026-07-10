"""Portable comparison helpers for measured kernel variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelVariantMeasurement:
    family: str
    variant: str
    workload: str
    latency_ms: float
    correctness_status: str
    tokens: int | None = None
    peak_memory_gb: float | None = None
    source: str | None = None
    notes: list[str] | None = None

    @property
    def promotable(self) -> bool:
        return self.correctness_status in {"pass", "k3_pass", "validated"}


def _measurement(value: KernelVariantMeasurement | dict[str, Any]) -> KernelVariantMeasurement:
    return value if isinstance(value, KernelVariantMeasurement) else KernelVariantMeasurement(**value)


def rank_kernel_variants(
    measurements: list[KernelVariantMeasurement | dict[str, Any]],
    *,
    require_correctness: bool = True,
) -> dict[str, Any]:
    """Rank like-for-like measured variants without loading infrastructure-specific artifacts."""

    rows = [_measurement(value) for value in measurements]
    if not rows:
        return {"status": "no_measurements", "best": None, "measurements": []}
    families = {row.family for row in rows}
    workloads = {row.workload for row in rows}
    if len(families) != 1 or len(workloads) != 1:
        raise ValueError("kernel variants must share one family and one workload")
    if any(row.latency_ms <= 0 for row in rows):
        raise ValueError("kernel-variant latency must be positive")

    eligible = [row for row in rows if row.promotable or not require_correctness]
    ranked = sorted(rows, key=lambda row: row.latency_ms)
    best = min(eligible, key=lambda row: row.latency_ms) if eligible else None
    baseline = max(row.latency_ms for row in rows)
    rendered = []
    for row in ranked:
        payload = asdict(row)
        payload["promotable"] = row.promotable
        payload["speedup_vs_slowest"] = round(baseline / row.latency_ms, 6)
        rendered.append(payload)
    return {
        "status": "ok" if best is not None else "no_correctness_promotable_variant",
        "family": next(iter(families)),
        "workload": next(iter(workloads)),
        "require_correctness": require_correctness,
        "best": asdict(best) if best is not None else None,
        "measurements": rendered,
    }


def compare_kernel_variants(
    baseline: KernelVariantMeasurement | dict[str, Any],
    candidate: KernelVariantMeasurement | dict[str, Any],
) -> dict[str, Any]:
    base = _measurement(baseline)
    other = _measurement(candidate)
    if (base.family, base.workload) != (other.family, other.workload):
        raise ValueError("kernel variants must share one family and one workload")
    return {
        "family": base.family,
        "workload": base.workload,
        "baseline": base.variant,
        "candidate": other.variant,
        "latency_delta_ms": round(other.latency_ms - base.latency_ms, 6),
        "latency_delta_percent": round((other.latency_ms / base.latency_ms - 1.0) * 100.0, 6),
        "speedup": round(base.latency_ms / other.latency_ms, 6),
        "peak_memory_delta_gb": (
            round(other.peak_memory_gb - base.peak_memory_gb, 6)
            if base.peak_memory_gb is not None and other.peak_memory_gb is not None
            else None
        ),
        "candidate_promotable": other.promotable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="JSON list of kernel-variant measurements")
    parser.add_argument("--allow-ungated", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.measurements.read_text(encoding="utf-8"))
    report = rank_kernel_variants(rows, require_correctness=not args.allow_ungated)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
