import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[2] / "scripts" / "qwen35_rope_class_b_ab.py"
_SPEC = importlib.util.spec_from_file_location("qwen35_rope_class_b_ab", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _run(name, arm, decisions, offset=0):
    values = list(range(offset, offset + decisions))
    return {
        "name": name,
        "rope_class": arm,
        "decision_count": decisions,
        "lifecycle_id": name,
        "sampled_ids": values,
        "sampler_logprob_f32_hex": [f"{value:08x}" for value in values],
        "trainer_logprob_f32_hex": [f"{value:08x}" for value in values],
        "k3": 0.0,
        "resolved_trainer_rope_class_b": arm == "B",
        "resolved_sampler_rope_class_b": arm == "B",
    }


def _record():
    return {
        "schema_version": 1,
        "contract": {
            "xorl_commit": "x",
            "sglang_commit": "s",
            "weights_id": "w",
            "prompt_id": "p",
            "topology_id": "t",
        },
        "four_decision": [
            _run("a1", "A", 4),
            _run("a2", "A", 4),
            _run("b1", "B", 4, 10),
            _run("b2", "B", 4, 10),
        ],
        "class_b_confirmation": _run("b64", "B", 64, 10),
        "timing": [
            {"rope_class": "A", "pair_order": "AB", "session_id": "s", "workload_id": "w", "warm": True, "elapsed_s": 2.0},
            {"rope_class": "B", "pair_order": "AB", "session_id": "s", "workload_id": "w", "warm": True, "elapsed_s": 1.0},
            {"rope_class": "B", "pair_order": "BA", "session_id": "s", "workload_id": "w", "warm": True, "elapsed_s": 1.2},
            {"rope_class": "A", "pair_order": "BA", "session_id": "s", "workload_id": "w", "warm": True, "elapsed_s": 2.2},
        ],
    }


def test_audit_accepts_exact_repeatable_gate():
    summary = _MODULE.audit(_record())
    assert summary["four_decision_exact"]
    assert summary["class_b_64_exact"]
    assert summary["class_b_speedup"] == pytest.approx(2.1 / 1.1)


def test_audit_rejects_trainer_sampler_byte_mismatch():
    record = _record()
    record["four_decision"][2]["trainer_logprob_f32_hex"][1] = "ffffffff"
    with pytest.raises(ValueError, match="trainer/sampler raw float32 bytes differ"):
        _MODULE.audit(record)
