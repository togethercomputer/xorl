from types import SimpleNamespace

import pytest
import torch

from xorl.distributed.torch_parallelize import (
    _expert_mixed_precision_policy,
    _resolve_fsdp_reduce_dtype,
    _sequence_parallel_fully_folded_into_fsdp,
)


def test_singleton_expert_mp_policy_uses_bf16_reduce_dtype() -> None:
    policy = _expert_mixed_precision_policy(ep_fsdp_mesh_size=1)

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.bfloat16


def test_sharded_expert_mp_policy_keeps_fp32_reduce_dtype() -> None:
    policy = _expert_mixed_precision_policy(ep_fsdp_mesh_size=2)

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32


def test_sharded_expert_mp_policy_can_use_bf16_reduce_dtype() -> None:
    policy = _expert_mixed_precision_policy(ep_fsdp_mesh_size=2, reduce_dtype=torch.bfloat16)

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.bfloat16


def test_resolve_fsdp_reduce_dtype() -> None:
    assert _resolve_fsdp_reduce_dtype("fp32") is torch.float32
    assert _resolve_fsdp_reduce_dtype("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="Unsupported fsdp_reduce_dtype"):
        _resolve_fsdp_reduce_dtype("fp16")


@pytest.mark.parametrize(
    ("ulysses_enabled", "ringattn_enabled", "cp_fsdp_mode", "expected"),
    [
        (True, False, "all", True),
        (True, False, "ulysses_only", True),
        (True, False, "ring_only", False),
        (False, True, "all", True),
        (False, True, "ring_only", True),
        (False, True, "ulysses_only", False),
        (True, True, "all", True),
        (True, True, "ulysses_only", False),
        (True, True, "ring_only", False),
        (True, True, "none", False),
    ],
)
def test_sequence_parallel_fully_folded_into_fsdp(
    ulysses_enabled: bool,
    ringattn_enabled: bool,
    cp_fsdp_mode: str,
    expected: bool,
) -> None:
    state = SimpleNamespace(
        ulysses_enabled=ulysses_enabled,
        ringattn_enabled=ringattn_enabled,
        cp_fsdp_mode=cp_fsdp_mode,
    )

    assert _sequence_parallel_fully_folded_into_fsdp(state) is expected
