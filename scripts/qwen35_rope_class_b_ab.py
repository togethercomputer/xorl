#!/usr/bin/env python3
"""Audit the focused Qwen3.5/3.6 Class-A versus Class-B promotion gate.

The existing Qwen lifecycle tools collect decision-time token IDs and raw
float32 logprob bytes.  This script deliberately does not launch servers or
rescore sampler output.  It checks one compact JSON record after collection:

* two fresh four-decision lifecycles for each arm (A/A repeatability);
* independent trainer replay of every sampler byte in both A and B;
* one fresh 64-decision Class-B confirmation with the accepted four-decision
  trace as an exact prefix; and
* warm, unprofiled A/B timing samples from one session and workload.

Run ``python scripts/qwen35_rope_class_b_ab.py record.json``.  A zero exit is
an evidence audit, not a launcher or a default-selection decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit_exact_run(run: dict[str, Any], *, decisions: int, rope_class: str) -> None:
    name = run.get("name", "<unnamed>")
    _require(run.get("rope_class") == rope_class, f"{name}: expected Class {rope_class}")
    _require(run.get("decision_count") == decisions, f"{name}: expected {decisions} decisions")
    ids = run.get("sampled_ids", [])
    sampler = run.get("sampler_logprob_f32_hex", [])
    trainer = run.get("trainer_logprob_f32_hex", [])
    _require(len(ids) == len(sampler) == len(trainer) == decisions, f"{name}: incomplete decision rows")
    _require(sampler == trainer, f"{name}: trainer/sampler raw float32 bytes differ")
    _require(float(run.get("k3", float("nan"))) == 0.0, f"{name}: K3 is not exactly zero")
    expected_class_b = rope_class == "B"
    _require(
        run.get("resolved_trainer_rope_class_b") is expected_class_b,
        f"{name}: trainer resolved the wrong RoPE arm",
    )
    _require(
        run.get("resolved_sampler_rope_class_b") is expected_class_b,
        f"{name}: sampler resolved the wrong RoPE arm",
    )


def _audit_repeat(first: dict[str, Any], second: dict[str, Any]) -> None:
    _require(
        first.get("lifecycle_id") != second.get("lifecycle_id"),
        f"{first['rope_class']}: A/A evidence must use fresh sampler lifecycles",
    )
    _require(first["sampled_ids"] == second["sampled_ids"], f"{first['rope_class']}: sampled IDs are not repeatable")
    _require(
        first["sampler_logprob_f32_hex"] == second["sampler_logprob_f32_hex"],
        f"{first['rope_class']}: sampler denominator bytes are not repeatable",
    )


def audit(record: dict[str, Any]) -> dict[str, Any]:
    _require(record.get("schema_version") == 1, "unsupported schema_version")
    contract = record.get("contract", {})
    for key in ("xorl_commit", "sglang_commit", "weights_id", "prompt_id", "topology_id"):
        _require(bool(contract.get(key)), f"contract.{key} is required")

    four = record.get("four_decision", [])
    _require(len(four) == 4, "four_decision must contain A1, A2, B1, and B2")
    by_class = {arm: [run for run in four if run.get("rope_class") == arm] for arm in ("A", "B")}
    for arm, runs in by_class.items():
        _require(len(runs) == 2, f"Class {arm}: expected two four-decision lifecycles")
        for run in runs:
            _audit_exact_run(run, decisions=4, rope_class=arm)
        _audit_repeat(runs[0], runs[1])

    confirmation = record.get("class_b_confirmation", {})
    _audit_exact_run(confirmation, decisions=64, rope_class="B")
    _require(
        confirmation["sampled_ids"][:4] == by_class["B"][0]["sampled_ids"],
        "Class B 64-decision IDs do not preserve the accepted four-decision prefix",
    )
    _require(
        confirmation["sampler_logprob_f32_hex"][:4] == by_class["B"][0]["sampler_logprob_f32_hex"],
        "Class B 64-decision bytes do not preserve the accepted four-decision prefix",
    )

    timing = record.get("timing", [])
    _require(len(timing) >= 4, "timing requires warm samples in both A/B orders")
    session_ids = {sample.get("session_id") for sample in timing}
    workload_ids = {sample.get("workload_id") for sample in timing}
    orders = {sample.get("pair_order") for sample in timing}
    _require(len(session_ids) == 1 and None not in session_ids, "timing arms must share one campaign session")
    _require(len(workload_ids) == 1 and None not in workload_ids, "timing arms must share one workload")
    _require(orders == {"AB", "BA"}, "timing must include both A/B and B/A orders")
    medians: dict[str, float] = {}
    for arm in ("A", "B"):
        samples = [sample for sample in timing if sample.get("rope_class") == arm]
        _require(len(samples) >= 2, f"Class {arm}: need at least two timing samples")
        _require(
            {sample.get("pair_order") for sample in samples} == {"AB", "BA"},
            f"Class {arm}: timing samples do not cover both orders",
        )
        _require(all(sample.get("warm") is True for sample in samples), f"Class {arm}: timing includes a cold sample")
        seconds = [float(sample["elapsed_s"]) for sample in samples]
        _require(all(value > 0 for value in seconds), f"Class {arm}: timing must be positive")
        medians[arm] = statistics.median(seconds)

    return {
        "four_decision_exact": True,
        "class_b_64_exact": True,
        "class_a_median_s": medians["A"],
        "class_b_median_s": medians["B"],
        "class_b_speedup": medians["A"] / medians["B"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    summary = audit(json.loads(args.record.read_text()))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
