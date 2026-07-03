from types import SimpleNamespace

import pytest
import torch

from xorl.distributed.torch_parallelize import (
    _configure_manual_fsdp_prefetch,
    _coerce_optional_bool_config,
    _expert_mixed_precision_policy,
    _resolve_fsdp_reduce_dtype,
    _sequence_parallel_fully_folded_into_fsdp,
)


class _FakeBlock:
    def __init__(self, name: str) -> None:
        self.name = name
        self._fsdp_modules = [f"{name}.attn", f"{name}.gate", f"{name}.experts"]
        self.forward_prefetch = None
        self.backward_prefetch = None

    def set_modules_to_forward_prefetch(self, modules) -> None:
        self.forward_prefetch = list(modules)

    def set_modules_to_backward_prefetch(self, modules) -> None:
        self.backward_prefetch = list(modules)


def _fake_blocks():
    return [_FakeBlock("block0"), _FakeBlock("block1"), _FakeBlock("block2")]


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
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("false", False),
        ("YES", True),
        ("off", False),
        (None, None),
    ],
)
def test_coerce_optional_bool_config(value, expected) -> None:
    assert _coerce_optional_bool_config(value, name="flag") is expected


def test_coerce_optional_bool_config_rejects_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="flag must be a boolean value"):
        _coerce_optional_bool_config("definitely", name="flag")


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


def test_manual_fsdp_prefetch_can_enable_backward_without_forward() -> None:
    blocks = _fake_blocks()

    _configure_manual_fsdp_prefetch(
        blocks,
        need_manual_prefetch=True,
        enable_forward_prefetch=False,
        enable_backward_prefetch=True,
    )

    assert [block.forward_prefetch for block in blocks] == [None, None, None]
    assert blocks[0].backward_prefetch is None
    assert blocks[1].backward_prefetch == ["block0.experts", "block0.gate", "block0.attn"]
    assert blocks[2].backward_prefetch == ["block1.experts", "block1.gate", "block1.attn"]


def test_manual_fsdp_prefetch_can_enable_forward_without_backward() -> None:
    blocks = _fake_blocks()

    _configure_manual_fsdp_prefetch(
        blocks,
        need_manual_prefetch=True,
        enable_forward_prefetch=True,
        enable_backward_prefetch=False,
    )

    assert blocks[0].forward_prefetch == ["block1.experts", "block1.gate", "block1.attn"]
    assert blocks[1].forward_prefetch == ["block2.experts", "block2.gate", "block2.attn"]
    assert blocks[2].forward_prefetch is None
    assert [block.backward_prefetch for block in blocks] == [None, None, None]


def test_manual_fsdp_prefetch_configures_both_directions_by_default() -> None:
    blocks = _fake_blocks()

    _configure_manual_fsdp_prefetch(
        blocks,
        need_manual_prefetch=True,
        enable_forward_prefetch=True,
        enable_backward_prefetch=True,
    )

    assert blocks[0].forward_prefetch == ["block1.experts", "block1.gate", "block1.attn"]
    assert blocks[1].forward_prefetch == ["block2.experts", "block2.gate", "block2.attn"]
    assert blocks[2].forward_prefetch is None
    assert blocks[0].backward_prefetch is None
    assert blocks[1].backward_prefetch == ["block0.experts", "block0.gate", "block0.attn"]
    assert blocks[2].backward_prefetch == ["block1.experts", "block1.gate", "block1.attn"]


def test_manual_fsdp_prefetch_noops_when_not_needed() -> None:
    blocks = _fake_blocks()

    _configure_manual_fsdp_prefetch(
        blocks,
        need_manual_prefetch=False,
        enable_forward_prefetch=True,
        enable_backward_prefetch=True,
    )

    assert [block.forward_prefetch for block in blocks] == [None, None, None]
    assert [block.backward_prefetch for block in blocks] == [None, None, None]
