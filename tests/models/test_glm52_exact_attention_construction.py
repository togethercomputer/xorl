from __future__ import annotations

import pytest
import torch

from tests.models.test_glm52_qlora import _meta_model, _official_config
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.qlora import GLM52_QLORA_FACTOR_COUNT, prepare_glm52_block_fp8_qlora


_ORDINARY_ATTENTION_PROJECTIONS = (
    "q_a_proj",
    "kv_a_proj_with_mqa",
    "q_b_proj",
    "o_proj",
)
_ALL_ATTENTION_PROJECTIONS = (*_ORDINARY_ATTENTION_PROJECTIONS, "kv_b_proj")


def _exact_attention_config():
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True
    config._glm52_exact_active_lora_attention_component = True
    config._sparse_mla_enabled = True
    config._ep_dispatch = "alltoall"
    return config


def _expected_attention_factor_names() -> set[str]:
    return {
        f"model.layers.{layer_idx}.self_attn.{projection}.lora_{factor}"
        for layer_idx in range(78)
        for projection in _ALL_ATTENTION_PROJECTIONS
        for factor in ("A", "B")
    }


def test_glm52_exact_attention_component_preserves_complete_canonical_inventory() -> None:
    config = _exact_attention_config()
    model = _meta_model(config)

    inventory = prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    expected_attention_factors = _expected_attention_factor_names()
    actual_attention_factors = {factor.name for factor in inventory.factors if factor.role.startswith("attention.")}
    assert actual_attention_factors == expected_attention_factors
    assert len(actual_attention_factors) == 78 * 5 * 2 == 780
    assert len(inventory.factors) == GLM52_QLORA_FACTOR_COUNT == 1700
    assert len(inventory.factor_names) == GLM52_QLORA_FACTOR_COUNT

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(trainable) == inventory.factor_names
    assert all(parameter.dtype is torch.float32 for parameter in trainable.values())
    assert len({id(parameter) for parameter in trainable.values()}) == GLM52_QLORA_FACTOR_COUNT

    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        prefix = f"model.layers.{layer_idx}.self_attn"
        for projection in _ORDINARY_ATTENTION_PROJECTIONS:
            module = getattr(attention, projection)
            assert type(module) is Glm52ExactTP1BlockFP8QLoRALinear
            assert module._source_fqn == f"{prefix}.{projection}"
        assert type(attention.kv_b_proj) is Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA
        assert attention.kv_b_proj._source_fqn == f"{prefix}.kv_b_proj"

        layer_attention_factors = {name for name in trainable if name.startswith(f"{prefix}.")}
        assert layer_attention_factors == {
            f"{prefix}.{projection}.lora_{factor}" for projection in _ALL_ATTENTION_PROJECTIONS for factor in ("A", "B")
        }

    exact_dense_roots = {name for name, module in model.named_modules() if isinstance(module, Glm52ExactTP1DenseMLP)}
    assert exact_dense_roots == {f"model.layers.{layer_idx}.mlp" for layer_idx in range(3)}


@pytest.mark.parametrize(("rank", "alpha"), ((16, 16), (1, 2), (2, 1)))
def test_glm52_exact_attention_component_rejects_non_rank1_alpha1_before_mutation(
    rank: int,
    alpha: int,
) -> None:
    config = _exact_attention_config()
    model = _meta_model(config)

    with pytest.raises(ValueError, match="requires adapter_rank=1 and adapter_alpha=1"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=rank, adapter_alpha=alpha)

    assert not any("lora_" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"_glm52_exact_active_lora_dense_component": False}, "exact active-LoRA dense component"),
        ({"_ep_dispatch": "deepep"}, "ep_dispatch='alltoall'"),
        ({"_sparse_mla_enabled": False}, "requires sparse_mla_enabled=true"),
    ),
)
def test_glm52_exact_attention_component_rejects_incomplete_execution_contract_before_mutation(
    override: dict[str, object],
    message: str,
) -> None:
    config = _exact_attention_config()
    for name, value in override.items():
        setattr(config, name, value)
    model = _meta_model(config)

    with pytest.raises(ValueError, match=message):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    assert not any("lora_" in name for name, _ in model.named_parameters())
