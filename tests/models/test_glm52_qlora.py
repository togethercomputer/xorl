from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from torch import nn

from xorl.lora.modules.linear import LoraLinear
from xorl.models.exact_contract import set_glm52_exact_active_lora
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM
from xorl.models.transformers.glm5.qlora import (
    GLM52_FROZEN_NATIVE_INDEXER_PROJECTION_COUNT,
    GLM52_QLORA_FACTOR_COUNT,
    GLM52_QLORA_ORDINARY_TARGET_COUNT,
    GLM52_QLORA_QUANTIZED_LINEAR_COUNT,
    GLM52_QLORA_ROUTED_BANK_COUNT,
    prepare_glm52_block_fp8_qlora,
)
from xorl.models.transformers.glm5.support import validate_glm5_training_mode
from xorl.ops.exact.block_fp8_native import NativeBlockFP8Linear
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts


def _indexer_schedule() -> tuple[str, ...]:
    return tuple("full" if layer_idx < 3 or (layer_idx - 2) % 4 == 0 else "shared" for layer_idx in range(78))


def _official_config() -> Glm5Config:
    indexer_exclusions = [
        f"model.layers.{layer_idx}.self_attn.indexers_proj"
        for layer_idx, indexer_type in enumerate(_indexer_schedule())
        if indexer_type == "full"
    ]
    config = Glm5Config(
        indexer_types=_indexer_schedule(),
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=("dense",) * 3 + ("sparse",) * 75,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "model.embed_tokens",
                "lm_head",
                "model.layers.78.shared_head.norm",
                *indexer_exclusions,
            ],
        },
        _moe_implementation="triton",
    )
    config._ep_dispatch = "deepep"
    config._glm52_block_fp8_qlora = True
    config._glm52_exact_contract = False
    return config


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _meta_model(config: Glm5Config | None = None) -> Glm5ForCausalLM:
    config = _official_config() if config is None else config
    with _default_dtype(torch.bfloat16), torch.device("meta"):
        return Glm5ForCausalLM(config)


def test_glm52_full_block_fp8_qlora_inventory_is_exact_and_fail_closed() -> None:
    model = _meta_model()

    inventory = prepare_glm52_block_fp8_qlora(
        model,
        model.config,
        adapter_rank=16,
        adapter_alpha=16,
    )

    quantized = {name for name, module in model.named_modules() if isinstance(module, BlockFP8QLoRALinear)}
    routed_banks = {name for name, module in model.named_modules() if isinstance(module, BlockFP8QLoRAMoeExperts)}
    ordinary_heads = {name for name, module in model.named_modules() if isinstance(module, LoraLinear)}
    native_indexers = {name for name, module in model.named_modules() if isinstance(module, NativeBlockFP8Linear)}
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert len(quantized) == GLM52_QLORA_QUANTIZED_LINEAR_COUNT == 624
    assert ordinary_heads == {"lm_head"}
    assert len(quantized | ordinary_heads) == GLM52_QLORA_ORDINARY_TARGET_COUNT == 625
    assert len(routed_banks) == GLM52_QLORA_ROUTED_BANK_COUNT == 75
    assert len(native_indexers) == GLM52_FROZEN_NATIVE_INDEXER_PROJECTION_COUNT == 42
    assert native_indexers == {
        f"model.layers.{layer_idx}.self_attn.indexer.{projection}"
        for layer_idx, indexer_type in enumerate(_indexer_schedule())
        if indexer_type == "full"
        for projection in ("wq_b", "wk")
    }
    assert len(inventory.targets) == 700
    assert len(inventory.factors) == GLM52_QLORA_FACTOR_COUNT == 1700
    assert trainable == inventory.factor_names
    assert all(factor.dtype is torch.float32 for factor in inventory.factors)

    assert inventory.role_counts == {
        "attention.q_a_proj": 78,
        "attention.q_b_proj": 78,
        "attention.kv_a_proj_with_mqa": 78,
        "attention.kv_b_proj": 78,
        "attention.o_proj": 78,
        "dense_mlp.gate_proj": 3,
        "dense_mlp.up_proj": 3,
        "dense_mlp.down_proj": 3,
        "routed_expert": 75,
        "shared_expert.gate_proj": 75,
        "shared_expert.up_proj": 75,
        "shared_expert.down_proj": 75,
        "output.lm_head": 1,
    }

    kv_a = model.model.layers[0].self_attn.kv_a_proj_with_mqa
    assert kv_a.weight_block_scales.shape == (5, 192)  # ceil(576 / 128), 4 * ceil(6144 / 128)
    assert model.lm_head.weight.dtype is torch.bfloat16
    assert model.model.layers[0].self_attn.kv_b_proj.adapter_gradient_producer_family == "direct_output_projection"
    assert model.model.layers[0].self_attn.q_a_proj._qlora_expected_skip_keys == {
        "weight",
        "weight_scale_inv",
    }
    assert model.model.layers[3].mlp.experts.hybrid_shared is True
    assert model.model.layers[3].mlp.experts.moe_implementation == "triton"
    assert model.model.layers[3].mlp.experts.ep_dispatch == "deepep"
    assert not model.model.layers[3].mlp.gate.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in model.model.layers[0].self_attn.indexer.wq_b.parameters())
    assert type(model.model.layers[0].self_attn.indexer.weights_proj) is nn.Linear
    assert model.model.layers[0].self_attn.indexer.weights_proj.weight.dtype is torch.bfloat16


def test_glm52_exact_dense_component_preserves_logical_inventory_with_three_physical_fused_roots() -> None:
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True
    config._ep_dispatch = "alltoall"
    model = _meta_model(config)

    inventory = prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    exact_roots = {name for name, module in model.named_modules() if isinstance(module, Glm52ExactTP1DenseMLP)}
    assert exact_roots == {f"model.layers.{layer_idx}.mlp" for layer_idx in range(3)}
    assert sum(isinstance(module, BlockFP8QLoRALinear) for module in model.modules()) == 618
    assert len(inventory.targets) == 700
    assert len(inventory.factors) == GLM52_QLORA_FACTOR_COUNT == 1700
    assert len(inventory.factor_names) == GLM52_QLORA_FACTOR_COUNT
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(trainable) == inventory.factor_names
    assert len({id(parameter) for parameter in trainable.values()}) == GLM52_QLORA_FACTOR_COUNT
    for layer_idx in range(3):
        prefix = f"model.layers.{layer_idx}.mlp"
        assert {name for name in trainable if name.startswith(f"{prefix}.")} == {
            f"{prefix}.gate_proj.lora_A",
            f"{prefix}.gate_proj.lora_B",
            f"{prefix}.up_proj.lora_A",
            f"{prefix}.up_proj.lora_B",
            f"{prefix}.down_proj.lora_A",
            f"{prefix}.down_proj.lora_B",
        }
        root = model.get_submodule(prefix)
        assert root._exact_gate_source_fqn == f"{prefix}.gate_proj"
        assert root._exact_up_source_fqn == f"{prefix}.up_proj"
        assert root.down_proj._source_fqn == f"{prefix}.down_proj"


@pytest.mark.parametrize(("rank", "alpha"), ((0, 1), (1, 0), (-2, 1)))
def test_glm52_exact_dense_component_rejects_nonpositive_rank_or_alpha_before_mutation(rank: int, alpha: int) -> None:
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True
    config._ep_dispatch = "alltoall"
    model = _meta_model(config)

    with pytest.raises(ValueError, match="must be positive"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=rank, adapter_alpha=alpha)

    assert not any("lora_" in name for name, _ in model.named_parameters())


def test_glm52_routed_banks_select_only_the_ep_local_checkpoint_slice(monkeypatch) -> None:
    class _EP16State:
        ep_enabled = True
        ep_size = 16
        ep_rank = 7

    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.get_parallel_state", lambda: _EP16State())
    model = _meta_model()

    prepare_glm52_block_fp8_qlora(model, model.config, adapter_rank=16, adapter_alpha=16)

    routed_banks = [module for module in model.modules() if isinstance(module, BlockFP8QLoRAMoeExperts)]
    assert len(routed_banks) == GLM52_QLORA_ROUTED_BANK_COUNT
    assert all(module.num_experts == 256 for module in routed_banks)
    assert all(module.num_local_experts == 16 for module in routed_banks)
    assert all(module.expert_offset == 112 for module in routed_banks)


def test_glm52_qlora_rejects_missing_or_wrong_shape_target_before_adapterization() -> None:
    model = _meta_model()
    model.model.layers[12].self_attn.q_a_proj = nn.Linear(6144, 1024, bias=False, device="meta")

    with pytest.raises(ValueError, match=r"model\.layers\.12\.self_attn\.q_a_proj has shape"):
        prepare_glm52_block_fp8_qlora(model, model.config, adapter_rank=16, adapter_alpha=16)

    assert not any("lora_" in name for name, _ in model.named_parameters())


def test_glm52_qlora_rejects_missing_official_indexer_exclusion_before_adapterization() -> None:
    config = _official_config()
    config.quantization_config["modules_to_not_convert"].remove("model.layers.0.self_attn.indexers_proj")
    model = _meta_model(config)

    with pytest.raises(ValueError, match="official BF16 DSA head projections"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=16, adapter_alpha=16)

    assert not any("lora_" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"_moe_implementation": "eager"}, "moe_implementation='triton'"),
        ({"_ep_dispatch": "alltoall"}, "ep_dispatch='deepep'"),
        ({"_glm52_exact_contract": True}, "cannot use the scoring-only exact contract"),
        ({"_glm52_block_fp8_qlora": False}, "block_fp8_qlora_training=true"),
    ],
)
def test_glm52_qlora_rejects_unsupported_construction_modes(override: dict, message: str) -> None:
    config = _official_config()
    for name, value in override.items():
        setattr(config, name, value)
    model = _meta_model(config)

    with pytest.raises(ValueError, match=message):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=16, adapter_alpha=16)


def test_glm5_training_mode_admits_only_explicit_product_tuple() -> None:
    config = _official_config()
    validate_glm5_training_mode(
        config,
        enable_qlora=True,
        freeze_router=True,
        merge_qkv=True,
        block_fp8_qlora_training=True,
        quant_format="block_fp8",
        quant_group_size=128,
        moe_implementation="triton",
        ep_dispatch="deepep",
        moe_hybrid_shared_lora=True,
    )

    with pytest.raises(ValueError, match="quant_format='nvfp4'"):
        validate_glm5_training_mode(
            config,
            enable_qlora=True,
            freeze_router=True,
            merge_qkv=True,
            block_fp8_qlora_training=True,
            quant_format="nvfp4",
            quant_group_size=128,
            moe_implementation="triton",
            ep_dispatch="deepep",
            moe_hybrid_shared_lora=True,
        )


def test_glm5_training_mode_uses_alltoall_only_for_complete_exact_active_lora() -> None:
    config = _official_config()
    set_glm52_exact_active_lora(config, enabled=True)

    validate_glm5_training_mode(
        config,
        enable_qlora=True,
        freeze_router=True,
        merge_qkv=True,
        block_fp8_qlora_training=True,
        quant_format="block_fp8",
        quant_group_size=128,
        moe_implementation="triton",
        ep_dispatch="alltoall",
        moe_hybrid_shared_lora=True,
    )

    with pytest.raises(ValueError, match="ep_dispatch='deepep'.*requires 'alltoall'"):
        validate_glm5_training_mode(
            config,
            enable_qlora=True,
            freeze_router=True,
            merge_qkv=True,
            block_fp8_qlora_training=True,
            quant_format="block_fp8",
            quant_group_size=128,
            moe_implementation="triton",
            ep_dispatch="deepep",
            moe_hybrid_shared_lora=True,
        )


def test_block_fp8_qlora_scale_storage_covers_partial_edge_tiles() -> None:
    module = BlockFP8QLoRALinear(6144, 576, r=4, lora_alpha=4, device=torch.device("meta"))

    assert module.weight_block_scales.shape == (5, 192)
