from argparse import Namespace

import pytest

from scripts.materialize_grouped_load import _derive_parallel_sizes


pytestmark = pytest.mark.cpu


def _args(**overrides):
    args = {
        "tp_size": 1,
        "pp_size": 1,
        "ulysses_size": 1,
        "ringattn_size": 1,
        "dp_replicate_size": None,
        "dp_shard_size": None,
        "ep_size": None,
    }
    args.update(overrides)
    return Namespace(**args)


def test_default_ep_size_uses_ranks_per_pipeline_stage():
    dp_size, dp_replicate_size, dp_shard_size, ep_size = _derive_parallel_sizes(_args(pp_size=2), world_size=8)

    assert dp_size == 4
    assert dp_replicate_size == 1
    assert dp_shard_size == 4
    assert ep_size == 4


def test_default_dp_shard_size_respects_dp_replicate_size():
    dp_size, dp_replicate_size, dp_shard_size, ep_size = _derive_parallel_sizes(
        _args(pp_size=2, dp_replicate_size=2),
        world_size=8,
    )

    assert dp_size == 4
    assert dp_replicate_size == 2
    assert dp_shard_size == 2
    assert ep_size == 4


def test_default_dp_replicate_size_respects_dp_shard_size():
    dp_size, dp_replicate_size, dp_shard_size, ep_size = _derive_parallel_sizes(
        _args(pp_size=2, dp_shard_size=2),
        world_size=8,
    )

    assert dp_size == 4
    assert dp_replicate_size == 2
    assert dp_shard_size == 2
    assert ep_size == 4


def test_explicit_ep_size_must_fit_pipeline_stage():
    with pytest.raises(RuntimeError, match="ep_size must fit within each pipeline stage"):
        _derive_parallel_sizes(_args(pp_size=2, ep_size=8), world_size=8)


def test_default_dp_shard_size_requires_divisible_dp_replicate_size():
    with pytest.raises(RuntimeError, match="data parallel size must be a multiple of dp_replicate_size"):
        _derive_parallel_sizes(_args(dp_replicate_size=3), world_size=8)
