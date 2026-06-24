"""Unit tests for the local-only phase/memory finalize helpers used on the
exception path of `Trainer.train_step`.

These exercise the pure summarization logic without instantiating a Trainer or
requiring CUDA / distributed init.
"""

import pytest

from xorl.trainers.trainer import (
    _STEP_PHASE_TIMING_ORDER,
    _order_step_phases,
    _summarize_memory_stats_local,
    _summarize_phase_times_local,
)


pytestmark = [pytest.mark.cpu]


def test_order_step_phases_known_only_follows_canonical_order():
    keys = ["reduce_metrics", "model_forward", "backward"]
    ordered = _order_step_phases(keys)
    # Canonical order: model_forward < backward < reduce_metrics
    assert ordered == ["model_forward", "backward", "reduce_metrics"]


def test_order_step_phases_unknown_keys_appended_sorted():
    keys = ["zz_custom", "aa_custom", "model_forward"]
    ordered = _order_step_phases(keys)
    assert ordered == ["model_forward", "aa_custom", "zz_custom"]


def test_order_step_phases_empty():
    assert _order_step_phases([]) == []


def test_order_step_phases_covers_every_canonical_phase():
    # Sanity: every canonical name round-trips when present.
    ordered = _order_step_phases(list(_STEP_PHASE_TIMING_ORDER))
    assert ordered == list(_STEP_PHASE_TIMING_ORDER)


def test_summarize_phase_times_local_empty_returns_empty_dict():
    assert _summarize_phase_times_local({}) == {}


def test_summarize_phase_times_local_all_aggregates_equal_local():
    summary = _summarize_phase_times_local({"model_forward": 0.25, "backward": 0.5})
    assert summary["model_forward"] == {"local": 0.25, "mean": 0.25, "max": 0.25, "min": 0.25}
    assert summary["backward"] == {"local": 0.5, "mean": 0.5, "max": 0.5, "min": 0.5}


def test_summarize_phase_times_local_orders_by_canonical_then_unknown():
    summary = _summarize_phase_times_local(
        {
            "reduce_metrics": 0.1,
            "zz_custom": 0.2,
            "model_forward": 0.3,
            "aa_custom": 0.4,
        }
    )
    # model_forward < reduce_metrics (canonical), then unknowns sorted.
    assert list(summary.keys()) == ["model_forward", "reduce_metrics", "aa_custom", "zz_custom"]


def test_summarize_phase_times_local_coerces_to_float():
    summary = _summarize_phase_times_local({"backward": 1})  # int
    assert isinstance(summary["backward"]["local"], float)
    assert summary["backward"]["local"] == 1.0


def test_summarize_memory_stats_local_empty_returns_empty_dict():
    assert _summarize_memory_stats_local({}) == {}


def test_summarize_memory_stats_local_all_aggregates_equal_local():
    summary = _summarize_memory_stats_local(
        {
            "model_forward": {"delta_allocated_gb": 1.5, "after_allocated_gb": 10.0},
        }
    )
    assert summary["model_forward"]["delta_allocated_gb"] == {
        "local": 1.5,
        "mean": 1.5,
        "max": 1.5,
        "min": 1.5,
    }
    assert summary["model_forward"]["after_allocated_gb"] == {
        "local": 10.0,
        "mean": 10.0,
        "max": 10.0,
        "min": 10.0,
    }


def test_summarize_memory_stats_local_orders_phases_canonically():
    summary = _summarize_memory_stats_local(
        {
            "reduce_metrics": {"delta_allocated_gb": 1.0},
            "model_forward": {"delta_allocated_gb": 2.0},
            "custom_phase": {"delta_allocated_gb": 3.0},
        }
    )
    assert list(summary.keys()) == ["model_forward", "reduce_metrics", "custom_phase"]
