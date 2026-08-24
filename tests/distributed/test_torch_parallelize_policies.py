from types import SimpleNamespace

import pytest
import torch
from torch import nn

import xorl.distributed.torch_parallelize as torch_parallelize
from xorl.distributed.torch_parallelize import (
    _coerce_optional_bool_config,
    _configure_manual_fsdp_prefetch,
    _exact_lm_head_replicated_params,
    _expert_mixed_precision_policy,
    _fsdp_kwargs_for_module,
    _fully_shard_declared_mixed_dtype_unit,
    _resolve_fsdp_reduce_dtype,
    _sequence_parallel_fully_folded_into_fsdp,
    _topmost_modules_matching,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.exact_shared_expert_qlora import Glm52ExactTP16SharedExpertBlockFP8QLoRA
from xorl.ops.block_fp8_native import NativeBlockFP8Linear


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


def test_mixed_precision_ignored_selection_stops_at_topmost_matching_unit() -> None:
    class _Protected(nn.Module):
        pass

    root = nn.Module()
    root.composite = _Protected()
    root.composite.nested = _Protected()
    root.ordinary = nn.Module()
    root.ordinary.protected = _Protected()

    selected = _topmost_modules_matching(root, (_Protected,))

    assert selected == [root.composite, root.ordinary.protected]


def test_explicit_full_precision_module_drops_only_its_fsdp_mp_policy() -> None:
    protected = nn.Module()
    protected.fsdp_requires_full_precision = True
    policy = object()
    original = {"mesh": "mesh", "mp_policy": policy, "reshard_after_forward": True}

    assert _fsdp_kwargs_for_module(original, protected) == {
        "mesh": "mesh",
        "reshard_after_forward": True,
    }
    assert _fsdp_kwargs_for_module(original, nn.Module()) == original
    assert original["mp_policy"] is policy


def test_declared_mixed_dtype_unit_forms_two_sharded_groups_without_renaming(monkeypatch) -> None:
    class _DeclaredComposite(nn.Module):
        fsdp_full_precision_parameter_names = ("A_log", "dt_bias")

        def __init__(self) -> None:
            super().__init__()
            self.A_log = nn.Parameter(torch.zeros(4, dtype=torch.float32))
            self.dt_bias = nn.Parameter(torch.ones(4, dtype=torch.float32))
            self.q_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.o_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    module = _DeclaredComposite()
    parameter_names_before = tuple(dict(module.named_parameters()))
    calls = []

    def fake_fully_shard(target, **kwargs) -> None:
        calls.append((target, kwargs))

    monkeypatch.setattr(torch_parallelize, "fully_shard", fake_fully_shard)
    compute_kwargs = {"mesh": "mesh", "mp_policy": "bf16"}
    full_precision_kwargs = {"mesh": "mesh"}

    representatives = _fully_shard_declared_mixed_dtype_unit(
        module,
        compute_kwargs=compute_kwargs,
        full_precision_kwargs=full_precision_kwargs,
    )

    assert calls == [([module.q_proj, module.o_proj], compute_kwargs), (module, full_precision_kwargs)]
    assert representatives == [module, module.q_proj]
    assert tuple(dict(module.named_parameters())) == parameter_names_before


def test_declared_mixed_dtype_unit_rejects_non_bf16_compute_parameters(monkeypatch) -> None:
    class _BadComposite(nn.Module):
        fsdp_full_precision_parameter_names = ("state",)

        def __init__(self) -> None:
            super().__init__()
            self.state = nn.Parameter(torch.zeros(4, dtype=torch.float32))
            self.proj = nn.Linear(4, 4, bias=False, dtype=torch.float32)

    monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *_args, **_kwargs: None)
    with pytest.raises(TypeError, match="compute parameters must be uniformly BF16"):
        _fully_shard_declared_mixed_dtype_unit(
            _BadComposite(),
            compute_kwargs={},
            full_precision_kwargs={},
        )


def test_dsv4_exact_lm_head_replicates_only_fp32_a() -> None:
    head = nn.Module()
    head.lora_A = nn.Parameter(torch.empty(1, 8, dtype=torch.float32))
    head.lora_B = nn.Parameter(torch.empty(16, 1, dtype=torch.float32))
    head._dsv4_exact_tp8_lm_head = True
    head._dsv4_exact_replicated_parameter_names = ("lora_A",)

    assert _exact_lm_head_replicated_params(head) == {head.lora_A}


def test_exact_shared_expert_is_one_topmost_full_precision_fsdp_unit() -> None:
    root = nn.Module()
    root.shared = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta")

    selected = _topmost_modules_matching(
        root,
        (
            NativeBlockFP8Linear,
            Glm52ExactTP1BlockFP8QLoRALinear,
            Glm52ExactTP16SharedExpertBlockFP8QLoRA,
        ),
    )

    assert selected == [root.shared]
    assert all(
        projection not in selected for projection in (root.shared.gate_proj, root.shared.up_proj, root.shared.down_proj)
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
