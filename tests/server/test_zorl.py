"""Unit tests for ZORL config normalization and generation planning."""

from __future__ import annotations

import pytest
import torch

from xorl.server.zorl import (
    ZORLSessionState,
    build_zorl_candidate_lora_state_dict,
    build_zorl_update_from_rewards,
    filter_zorl_materialized_candidates,
    normalize_zorl_materialization,
    normalize_zorl_runtime_config,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_normalize_zorl_runtime_config_uses_aliases_and_defaults():
    config = normalize_zorl_runtime_config({"sigma": 0.125, "num_pairs": 3, "refresh_interval": 5, "seed": 17})

    assert config is not None
    assert config["enabled"] is True
    assert config["b_sigma"] == pytest.approx(0.125)
    assert config["num_perturbation_pairs"] == 3
    assert config["a_refresh_interval"] == 5
    assert config["antithetic_sampling"] is True
    assert config["a_init"] == "gaussian_jl"
    assert config["seed"] == 17


def test_normalize_zorl_runtime_config_returns_none_when_disabled():
    assert normalize_zorl_runtime_config({"enabled": False, "b_sigma": 0.1}) is None


def test_normalize_zorl_runtime_config_ignores_unknown_keys():
    """The Pydantic ZORL request models use ``extra="allow"`` for forward-compat,
    so the normalizer must drop unknown keys rather than raise — otherwise the
    same input is accepted at the API boundary and rejected later, surfacing
    two different error paths for one mistake."""
    config = normalize_zorl_runtime_config({"sigma": 0.5, "num_pairs": 2, "future_field_we_dont_know_about": 42})
    assert config is not None
    assert config["b_sigma"] == pytest.approx(0.5)
    assert config["num_perturbation_pairs"] == 2
    assert "future_field_we_dont_know_about" not in config


def test_zorl_session_state_tracks_active_generation_and_refreshes_families():
    session_spec = {
        "zorl_config": {
            "enabled": True,
            "b_sigma": 0.05,
            "num_perturbation_pairs": 2,
            "a_refresh_interval": 2,
            "antithetic_sampling": True,
            "a_init": "gaussian_jl",
            "seed": 123,
        }
    }

    state = ZORLSessionState.from_session_spec("policy-a", session_spec)

    assert state is not None
    plan0 = state.begin_generation()
    with pytest.raises(ValueError, match="still active"):
        state.begin_generation()

    assert state.snapshot()["active_generation_id"] == plan0.generation_id
    completed0 = state.complete_generation(plan0.generation_id)
    assert completed0.generation_id == plan0.generation_id

    plan1 = state.begin_generation()
    state.complete_generation(plan1.generation_id)
    plan2 = state.begin_generation()

    assert plan0.generation == 0
    assert plan0.family.family_index == 0
    assert [candidate.direction for candidate in plan0.candidates] == [
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    assert {candidate.family_id for candidate in plan0.candidates} == {plan0.family.family_id}

    assert plan1.family.family_id == plan0.family.family_id
    assert plan2.family.family_index == 1
    assert plan2.family.family_id != plan0.family.family_id
    assert plan2.family.a_seed != plan0.family.a_seed
    assert state.generation == 3


def test_zorl_session_state_loaded_parent_skips_initial_refresh():
    session_spec = {
        "zorl_config": {
            "enabled": True,
            "b_sigma": 0.05,
            "num_perturbation_pairs": 2,
            "a_refresh_interval": 2,
            "antithetic_sampling": True,
            "a_init": "gaussian_jl",
            "seed": 123,
        }
    }

    state = ZORLSessionState.from_session_spec("policy-loaded", session_spec)

    assert state is not None
    family = state.seed_loaded_parent_family()
    assert family.family_index == 0
    assert family.a_seed == -1

    plan0 = state.begin_generation()
    assert plan0.family_refreshed is False
    assert plan0.family.family_id == family.family_id
    state.complete_generation(plan0.generation_id)

    plan1 = state.begin_generation()
    assert plan1.family_refreshed is False
    state.complete_generation(plan1.generation_id)

    plan2 = state.begin_generation()
    assert plan2.family_refreshed is True
    assert plan2.family.family_index == 1


def test_zorl_pair_shards_filter_identical_global_plan_without_splitting_pairs():
    session_spec = {
        "zorl_config": {
            "enabled": True,
            "b_sigma": 0.05,
            "num_perturbation_pairs": 5,
            "a_refresh_interval": 0,
            "antithetic_sampling": True,
            "a_init": "gaussian_jl",
            "seed": 123,
        }
    }
    unsharded_state = ZORLSessionState.from_session_spec("policy-a", session_spec)
    sharded_state = ZORLSessionState.from_session_spec("policy-a", session_spec)

    assert unsharded_state is not None
    assert sharded_state is not None
    full_plan = unsharded_state.begin_generation()
    sharded_plan = sharded_state.begin_generation()

    assert [candidate.candidate_id for candidate in sharded_plan.candidates] == [
        candidate.candidate_id for candidate in full_plan.candidates
    ]

    seen_pairs: set[int] = set()
    seen_candidates: set[str] = set()
    for shard_index in range(3):
        materialization = normalize_zorl_materialization(
            {"mode": "pair_shard", "shard_index": shard_index, "num_shards": 3},
            num_pairs=5,
        )
        local_candidates = filter_zorl_materialized_candidates(
            sharded_plan.candidates,
            materialization,
            num_pairs=5,
        )
        local_by_pair: dict[int, set[str]] = {}
        for candidate in local_candidates:
            local_by_pair.setdefault(candidate.perturbation_index, set()).add(candidate.direction)
            assert candidate.candidate_id not in seen_candidates
            seen_candidates.add(candidate.candidate_id)

        for pair_index, directions in local_by_pair.items():
            assert directions == {"positive", "negative"}
            assert pair_index not in seen_pairs
            seen_pairs.add(pair_index)

    assert seen_pairs == set(range(5))
    assert seen_candidates == {candidate.candidate_id for candidate in full_plan.candidates}


def test_build_zorl_candidate_lora_state_dict_only_perturbs_lora_b():
    lora_params = {
        "layer.lora_A": torch.nn.Parameter(torch.full((2, 3), 0.5, dtype=torch.float32)),
        "layer.lora_B": torch.nn.Parameter(torch.zeros((3, 2), dtype=torch.float32)),
    }

    positive = build_zorl_candidate_lora_state_dict(
        lora_params,
        b_seed=123,
        direction="positive",
        b_sigma=0.25,
    )
    negative = build_zorl_candidate_lora_state_dict(
        lora_params,
        b_seed=123,
        direction="negative",
        b_sigma=0.25,
    )

    assert torch.allclose(positive["layer.lora_A"], lora_params["layer.lora_A"].data)
    assert torch.allclose(negative["layer.lora_A"], lora_params["layer.lora_A"].data)
    assert torch.allclose(positive["layer.lora_B"], -negative["layer.lora_B"])


def test_build_zorl_update_from_rewards_averages_weighted_b_noises():
    lora_params = {
        "layer.lora_A": torch.nn.Parameter(torch.full((2, 3), 0.5, dtype=torch.float32)),
        "layer.lora_B": torch.nn.Parameter(torch.zeros((3, 2), dtype=torch.float32)),
    }

    update, update_norm = build_zorl_update_from_rewards(
        lora_params,
        pair_seeds_and_scores=[
            (11, 1.0),
            (17, -0.5),
        ],
    )

    assert set(update) == {"layer.lora_B"}
    assert update["layer.lora_B"].shape == lora_params["layer.lora_B"].shape
    assert update_norm > 0.0
