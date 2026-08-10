from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from tests.models.test_glm52_qlora import _meta_model, _official_config
from xorl.models.transformers.glm5.checkpoint_handler import Glm5CheckpointHandler
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.native_fp8 import NativeBlockFP8PairBuffer, native_fp8_dense_source_map
from xorl.models.transformers.glm5.qlora import prepare_glm52_block_fp8_qlora


_ATTENTION_PREFIX = "model.layers.0.self_attn"
_ORDINARY_PROJECTIONS = (
    "q_a_proj",
    "kv_a_proj_with_mqa",
    "q_b_proj",
    "o_proj",
)
_ALL_PROJECTIONS = (*_ORDINARY_PROJECTIONS, "kv_b_proj")


@dataclass
class _CheckpointCase:
    model: nn.Module
    full_source_map: dict[str, str]
    source: str
    target: str
    module: Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA
    weight: torch.Tensor
    scale: torch.Tensor


def _exact_attention_config():
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True
    config._glm52_exact_active_lora_attention_component = True
    config._sparse_mla_enabled = True
    config._ep_dispatch = "alltoall"
    return config


def _one_layer_attention_root(source_model: nn.Module) -> nn.Module:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    root.model.layers[0].self_attn = nn.Module()
    source_attention = source_model.model.layers[0].self_attn
    target_attention = root.model.layers[0].self_attn
    for projection in _ALL_PROJECTIONS:
        setattr(target_attention, projection, getattr(source_attention, projection))
    return root


@pytest.fixture(scope="module")
def checkpoint_case() -> _CheckpointCase:
    config = _exact_attention_config()
    source_model = _meta_model(config)
    prepare_glm52_block_fp8_qlora(source_model, config, adapter_rank=1, adapter_alpha=1)
    full_source_map = native_fp8_dense_source_map(source_model)
    model = _one_layer_attention_root(source_model)
    source = target = f"{_ATTENTION_PREFIX}.kv_b_proj"
    module = model.get_submodule(target)
    assert type(module) is Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA

    weight = torch.full(
        (module.out_features, module.in_features),
        0.25,
        dtype=torch.float8_e4m3fn,
    )
    weight[0].fill_(-0.5)
    weight[-1].fill_(0.75)
    scale = (
        torch.arange(module.weight_scale_inv.numel(), dtype=torch.float32)
        .reshape(module.weight_scale_inv.shape)
        .add_(1)
        .div_(1024)
    )
    return _CheckpointCase(model, full_source_map, source, target, module, weight, scale)


def _handler(case: _CheckpointCase) -> Glm5CheckpointHandler:
    handler = Glm5CheckpointHandler(
        num_experts=1,
        checkpoint_has_per_expert=False,
        num_hidden_layers=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        model=case.model,
        load_family="dense",
    )
    assert isinstance(handler._native_weight_buffer, NativeBlockFP8PairBuffer)
    return handler


def test_exact_attention_checkpoint_inventory_routes_only_absorbed_kv_b_through_native_pairs(
    checkpoint_case: _CheckpointCase,
) -> None:
    case = checkpoint_case
    source_map = native_fp8_dense_source_map(case.model)

    assert case.full_source_map[case.source] == case.target
    assert source_map == {case.source: case.target}
    assert case.module._source_fqn == case.source
    assert case.module._source_quant_format == "block_fp8"
    assert case.module._is_prequantized is True
    assert case.module._merge_sources is None
    assert case.module._qlora_expected_skip_keys == {"weight", "weight_scale_inv"}

    for projection in _ORDINARY_PROJECTIONS:
        fqn = f"{_ATTENTION_PREFIX}.{projection}"
        module = case.model.get_submodule(fqn)
        assert type(module) is Glm52ExactTP1BlockFP8QLoRALinear
        assert module._source_fqn == fqn
        assert module._source_quant_format == "block_fp8"
        assert module._is_prequantized is True
        assert module._inline_loaded is False
        assert module._merge_sources is None
        assert module._qlora_expected_skip_keys == {"weight", "weight_scale_inv"}
        assert callable(module._load_prequantized)
        assert fqn not in source_map

    expected_factors = {
        f"{_ATTENTION_PREFIX}.{projection}.lora_{factor}" for projection in _ALL_PROJECTIONS for factor in ("A", "B")
    }
    trainable = {name: parameter for name, parameter in case.model.named_parameters() if parameter.requires_grad}
    assert set(trainable) == expected_factors
    assert len(trainable) == 10
    assert all(parameter.dtype is torch.float32 for parameter in trainable.values())


@pytest.mark.parametrize("arrival_order", (("weight", "scale"), ("scale", "weight")))
def test_exact_absorbed_kv_b_checkpoint_pair_is_order_independent_and_byte_exact(
    checkpoint_case: _CheckpointCase,
    arrival_order: tuple[str, str],
) -> None:
    case = checkpoint_case
    handler = _handler(case)
    values = {
        "weight": (f"{case.source}.weight", case.weight),
        "scale": (f"{case.source}.weight_scale_inv", case.scale),
    }

    first_key, first_tensor = values[arrival_order[0]]
    second_key, second_tensor = values[arrival_order[1]]
    assert handler.on_load_weight(first_key, first_tensor) == []
    emitted = handler.on_load_weight(second_key, second_tensor)
    assert handler.on_load_complete() == []

    assert [name for name, _ in emitted] == [
        f"{case.target}.packed_weight_f32",
        f"{case.target}.weight_scale_inv",
    ]
    packed = emitted[0][1]
    loaded_scale = emitted[1][1]
    assert packed.dtype is torch.float32
    assert torch.equal(packed.view(torch.uint8), case.weight.view(torch.uint8))
    assert loaded_scale.dtype is torch.float32
    assert torch.equal(loaded_scale, case.scale)


@pytest.mark.parametrize("member", ("weight", "scale"))
def test_exact_absorbed_kv_b_checkpoint_pair_rejects_duplicate_members(
    checkpoint_case: _CheckpointCase,
    member: str,
) -> None:
    case = checkpoint_case
    handler = _handler(case)
    key, tensor = (
        (f"{case.source}.weight", case.weight)
        if member == "weight"
        else (f"{case.source}.weight_scale_inv", case.scale)
    )

    assert handler.on_load_weight(key, tensor) == []
    with pytest.raises(ValueError, match="Duplicate native FP8 pair member"):
        handler.on_load_weight(key, tensor)


@pytest.mark.parametrize("member", ("weight", "scale"))
def test_exact_absorbed_kv_b_checkpoint_pair_rejects_missing_members_at_completion(
    checkpoint_case: _CheckpointCase,
    member: str,
) -> None:
    case = checkpoint_case
    handler = _handler(case)
    key, tensor = (
        (f"{case.source}.weight", case.weight)
        if member == "weight"
        else (f"{case.source}.weight_scale_inv", case.scale)
    )

    assert handler.on_load_weight(key, tensor) == []
    with pytest.raises(ValueError, match="Incomplete native FP8 pairs"):
        handler.on_load_complete()


@pytest.mark.parametrize(
    ("bad_member", "bad_tensor", "message"),
    (
        ("weight", torch.zeros(1, dtype=torch.bfloat16), "weight must be float8_e4m3fn"),
        ("scale", torch.zeros(1, dtype=torch.bfloat16), "weight_scale_inv must be FP32"),
    ),
)
def test_exact_absorbed_kv_b_checkpoint_pair_rejects_dtype_mismatches(
    checkpoint_case: _CheckpointCase,
    bad_member: str,
    bad_tensor: torch.Tensor,
    message: str,
) -> None:
    case = checkpoint_case
    handler = _handler(case)
    good_member = "scale" if bad_member == "weight" else "weight"
    good_key, good_tensor = (
        (f"{case.source}.weight_scale_inv", case.scale)
        if good_member == "scale"
        else (f"{case.source}.weight", case.weight)
    )
    bad_key = f"{case.source}.{'weight' if bad_member == 'weight' else 'weight_scale_inv'}"

    assert handler.on_load_weight(good_key, good_tensor) == []
    with pytest.raises((TypeError, ValueError), match=message):
        handler.on_load_weight(bad_key, bad_tensor)


@pytest.mark.parametrize("bad_member", ("weight", "scale"))
def test_exact_absorbed_kv_b_checkpoint_pair_rejects_shape_mismatches(
    checkpoint_case: _CheckpointCase,
    bad_member: str,
) -> None:
    case = checkpoint_case
    handler = _handler(case)
    if bad_member == "weight":
        good_key, good_tensor = f"{case.source}.weight_scale_inv", case.scale
        bad_key, bad_tensor = f"{case.source}.weight", case.weight[:-1]
    else:
        good_key, good_tensor = f"{case.source}.weight", case.weight
        bad_key, bad_tensor = f"{case.source}.weight_scale_inv", case.scale[:-1]

    assert handler.on_load_weight(good_key, good_tensor) == []
    with pytest.raises(ValueError, match="unexpected shape|must be FP32"):
        handler.on_load_weight(bad_key, bad_tensor)
