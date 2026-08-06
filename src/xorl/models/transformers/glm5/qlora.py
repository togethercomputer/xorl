"""Deterministic full-target block-FP8 QLoRA construction for GLM-5.2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.lora.modules.linear import LoraLinear
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.transformers.glm5.native_fp8 import (
    replace_glm52_native_fp8_modules,
    validate_glm52_native_fp8_config,
)
from xorl.ops.block_fp8_native import NativeBlockFP8Linear
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts


GLM52_QLORA_ORDINARY_TARGET_COUNT = 625
GLM52_QLORA_QUANTIZED_LINEAR_COUNT = 624
GLM52_QLORA_ROUTED_BANK_COUNT = 75
GLM52_QLORA_FACTOR_COUNT = 1700
GLM52_FROZEN_NATIVE_INDEXER_PROJECTION_COUNT = 42

_ATTENTION_TARGETS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)
_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
_EXPERT_FACTORS = (
    "gate_proj_lora_A",
    "gate_proj_lora_B",
    "up_proj_lora_A",
    "up_proj_lora_B",
    "down_proj_lora_A",
    "down_proj_lora_B",
)


@dataclass(frozen=True)
class Glm52AdapterTarget:
    """One logical adapted module before distributed materialization."""

    name: str
    role: str
    kind: str
    in_features: int
    out_features: int


@dataclass(frozen=True)
class Glm52AdapterFactor:
    """One logical trainable factor tensor in the constructed GLM adapter."""

    name: str
    target_name: str
    role: str
    factor: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class Glm52AdapterInventory:
    """Ordinary in-memory target/factor inventory used by startup assertions."""

    targets: tuple[Glm52AdapterTarget, ...]
    factors: tuple[Glm52AdapterFactor, ...]

    @property
    def target_names(self) -> frozenset[str]:
        return frozenset(target.name for target in self.targets)

    @property
    def factor_names(self) -> frozenset[str]:
        return frozenset(factor.name for factor in self.factors)

    @property
    def role_counts(self) -> dict[str, int]:
        return dict(Counter(target.role for target in self.targets))


def _official_indexer_schedule() -> tuple[str, ...]:
    return tuple("full" if layer_idx < 3 or (layer_idx - 2) % 4 == 0 else "shared" for layer_idx in range(78))


def _validate_official_config(config) -> dict:
    expected = {
        "model_type": "xorl_glm5",
        "vocab_size": 154880,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "first_k_dense_replace": 3,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    mismatches = [
        f"{name}={getattr(config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    if tuple(getattr(config, "indexer_types", ()) or ()) != _official_indexer_schedule():
        mismatches.append("indexer_types does not match the official 78-layer selector schedule")
    if tuple(getattr(config, "mlp_layer_types", ()) or ()) != ("dense",) * 3 + ("sparse",) * 75:
        mismatches.append("mlp_layer_types does not match the official 3 dense + 75 sparse schedule")
    if mismatches:
        raise ValueError("GLM-5.2 block-FP8 QLoRA supports only the official model geometry: " + ", ".join(mismatches))

    if not getattr(config, "_glm52_block_fp8_qlora", False):
        raise ValueError("GLM-5.2 block-FP8 QLoRA construction requires block_fp8_qlora_training=true")
    if getattr(config, "_glm52_exact_contract", False):
        raise ValueError("GLM-5.2 block-FP8 QLoRA is a training lane and cannot use the scoring-only exact contract")
    if getattr(config, "_moe_implementation", None) != "triton":
        raise ValueError("GLM-5.2 block-FP8 QLoRA requires moe_implementation='triton'")
    if getattr(config, "_ep_dispatch", None) != "deepep":
        raise ValueError("GLM-5.2 block-FP8 QLoRA requires ep_dispatch='deepep'")

    quantization_config = validate_glm52_native_fp8_config(config.quantization_config)
    excluded = quantization_config["modules_to_not_convert"]
    if not any("lm_head" == item or "lm_head".startswith(f"{item}.") for item in excluded):
        raise ValueError("The official BF16 lm_head must appear in quantization_config.modules_to_not_convert")
    expected_indexer_exclusions = {
        f"model.layers.{layer_idx}.self_attn.indexers_proj"
        for layer_idx, indexer_type in enumerate(_official_indexer_schedule())
        if indexer_type == "full"
    }
    missing_indexer_exclusions = expected_indexer_exclusions - set(excluded)
    if missing_indexer_exclusions:
        raise ValueError(
            "The official BF16 DSA head projections are missing from "
            "quantization_config.modules_to_not_convert: "
            f"{sorted(missing_indexer_exclusions)}"
        )
    return quantization_config


def _set_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def _linear_shape(module: nn.Module) -> tuple[int, int]:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected an unadapted nn.Linear, got {type(module).__name__}")
    return int(module.in_features), int(module.out_features)


def _expected_targets(model: nn.Module, config) -> tuple[Glm52AdapterTarget, ...]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != 78:
        raise ValueError(f"GLM-5.2 block-FP8 QLoRA requires exactly 78 decoder layers, got {len(layers or ())}")

    expected_shapes = {
        "q_a_proj": (config.hidden_size, config.q_lora_rank),
        "q_b_proj": (config.q_lora_rank, config.num_attention_heads * config.qk_head_dim),
        "kv_a_proj_with_mqa": (config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim),
        "kv_b_proj": (
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
        ),
        "o_proj": (config.num_attention_heads * config.v_head_dim, config.hidden_size),
    }
    targets: list[Glm52AdapterTarget] = []
    for layer_idx, layer in enumerate(layers):
        for projection in _ATTENTION_TARGETS:
            name = f"model.layers.{layer_idx}.self_attn.{projection}"
            module = model.get_submodule(name)
            actual_shape = _linear_shape(module)
            if actual_shape != expected_shapes[projection]:
                raise ValueError(
                    f"GLM-5.2 QLoRA target {name} has shape {actual_shape}, expected {expected_shapes[projection]}"
                )
            targets.append(Glm52AdapterTarget(name, f"attention.{projection}", "block_fp8_linear", *actual_shape))

        if layer_idx < 3:
            mlp_prefix = f"model.layers.{layer_idx}.mlp"
            intermediate_size = config.intermediate_size
            role_prefix = "dense_mlp"
        else:
            mlp_prefix = f"model.layers.{layer_idx}.mlp.shared_experts"
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            role_prefix = "shared_expert"
            experts_name = f"model.layers.{layer_idx}.mlp.experts"
            experts = model.get_submodule(experts_name)
            if not isinstance(experts, MoEExperts):
                raise TypeError(f"GLM-5.2 QLoRA target {experts_name} must be an unadapted MoEExperts bank")
            if (
                experts.num_experts != config.n_routed_experts
                or int(experts.gate_up_proj.shape[1]) != config.hidden_size
                or experts.intermediate_size != config.moe_intermediate_size
            ):
                raise ValueError(f"GLM-5.2 QLoRA routed bank {experts_name} has an unexpected expert geometry")
            targets.append(
                Glm52AdapterTarget(
                    experts_name,
                    "routed_expert",
                    "block_fp8_routed_bank",
                    config.hidden_size,
                    config.moe_intermediate_size,
                )
            )

        mlp_shapes = {
            "gate_proj": (config.hidden_size, intermediate_size),
            "up_proj": (config.hidden_size, intermediate_size),
            "down_proj": (intermediate_size, config.hidden_size),
        }
        for projection in _MLP_TARGETS:
            name = f"{mlp_prefix}.{projection}"
            module = model.get_submodule(name)
            actual_shape = _linear_shape(module)
            if actual_shape != mlp_shapes[projection]:
                raise ValueError(
                    f"GLM-5.2 QLoRA target {name} has shape {actual_shape}, expected {mlp_shapes[projection]}"
                )
            targets.append(Glm52AdapterTarget(name, f"{role_prefix}.{projection}", "block_fp8_linear", *actual_shape))

    lm_head = model.get_submodule("lm_head")
    lm_head_shape = _linear_shape(lm_head)
    expected_head_shape = (config.hidden_size, config.vocab_size)
    if lm_head_shape != expected_head_shape:
        raise ValueError(f"GLM-5.2 QLoRA lm_head has shape {lm_head_shape}, expected {expected_head_shape}")
    if lm_head.weight.dtype is not torch.bfloat16:
        raise TypeError(f"GLM-5.2 QLoRA lm_head base must remain BF16, got {lm_head.weight.dtype}")
    targets.append(Glm52AdapterTarget("lm_head", "output.lm_head", "bf16_linear", *lm_head_shape))
    return tuple(targets)


def _is_config_excluded(name: str, excluded: list[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in excluded)


def _replace_dense_target(
    model: nn.Module, target: Glm52AdapterTarget, *, adapter_rank: int, adapter_alpha: int
) -> None:
    original = model.get_submodule(target.name)
    replacement = BlockFP8QLoRALinear(
        in_features=target.in_features,
        out_features=target.out_features,
        r=adapter_rank,
        lora_alpha=adapter_alpha,
        bias=original.bias is not None,
        device=original.weight.device,
    )
    replacement._is_prequantized = True
    replacement._source_quant_format = "block_fp8"
    replacement._source_fqn = target.name
    replacement._merge_sources = None
    replacement._qlora_expected_skip_keys = {"weight", "weight_scale_inv"}
    if target.name.endswith(".kv_b_proj"):
        replacement.adapter_gradient_producer_family = "direct_output_projection"
    _set_submodule(model, target.name, replacement)


def _replace_routed_target(
    model: nn.Module, target: Glm52AdapterTarget, *, adapter_rank: int, adapter_alpha: int
) -> None:
    original = model.get_submodule(target.name)
    parallel_state = get_parallel_state()
    ep_size = parallel_state.ep_size if parallel_state.ep_enabled else 1
    if original.num_experts % ep_size != 0:
        raise ValueError(f"GLM-5.2 routed expert count {original.num_experts} is not divisible by EP size {ep_size}")
    num_local_experts = original.num_experts // ep_size
    expert_offset = parallel_state.ep_rank * num_local_experts if parallel_state.ep_enabled else 0
    replacement = BlockFP8QLoRAMoeExperts.from_module(
        original,
        r=adapter_rank,
        lora_alpha=adapter_alpha,
        quant_format="block_fp8",
        quant_group_size=128,
        num_local_experts=num_local_experts,
        expert_offset=expert_offset,
        hybrid_shared=True,
    )
    replacement._source_fqn = target.name
    replacement._source_quant_format = "block_fp8"
    _set_submodule(model, target.name, replacement)


def _build_factor_inventory(
    model: nn.Module, targets: tuple[Glm52AdapterTarget, ...]
) -> tuple[Glm52AdapterFactor, ...]:
    factors: list[Glm52AdapterFactor] = []
    for target in targets:
        module = model.get_submodule(target.name)
        factor_names = _EXPERT_FACTORS if target.kind == "block_fp8_routed_bank" else ("lora_A", "lora_B")
        for factor_name in factor_names:
            parameter = getattr(module, factor_name)
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(f"GLM-5.2 QLoRA factor {target.name}.{factor_name} is not an nn.Parameter")
            if parameter.dtype is not torch.float32:
                raise TypeError(f"GLM-5.2 QLoRA factor {target.name}.{factor_name} must be FP32")
            factors.append(
                Glm52AdapterFactor(
                    name=f"{target.name}.{factor_name}",
                    target_name=target.name,
                    role=target.role,
                    factor=factor_name,
                    shape=tuple(parameter.shape),
                    dtype=parameter.dtype,
                )
            )
    return tuple(factors)


def _validate_constructed_model(model: nn.Module, inventory: Glm52AdapterInventory) -> None:
    expected_quantized = {target.name for target in inventory.targets if target.kind == "block_fp8_linear"}
    expected_heads = {target.name for target in inventory.targets if target.kind == "bf16_linear"}
    expected_banks = {target.name for target in inventory.targets if target.kind == "block_fp8_routed_bank"}
    actual_quantized = {name for name, module in model.named_modules() if isinstance(module, BlockFP8QLoRALinear)}
    actual_heads = {
        name
        for name, module in model.named_modules()
        if isinstance(module, LoraLinear) and not isinstance(module, BlockFP8QLoRALinear)
    }
    actual_banks = {name for name, module in model.named_modules() if isinstance(module, BlockFP8QLoRAMoeExperts)}
    if (actual_quantized, actual_heads, actual_banks) != (expected_quantized, expected_heads, expected_banks):
        raise RuntimeError(
            "GLM-5.2 QLoRA constructed target set mismatch: "
            f"quantized_missing={sorted(expected_quantized - actual_quantized)} "
            f"quantized_extra={sorted(actual_quantized - expected_quantized)} "
            f"head_missing={sorted(expected_heads - actual_heads)} head_extra={sorted(actual_heads - expected_heads)} "
            f"bank_missing={sorted(expected_banks - actual_banks)} bank_extra={sorted(actual_banks - expected_banks)}"
        )

    expected_native_indexers = {
        f"model.layers.{layer_idx}.self_attn.indexer.{projection}"
        for layer_idx, indexer_type in enumerate(_official_indexer_schedule())
        if indexer_type == "full"
        for projection in ("wq_b", "wk")
    }
    actual_native_indexers = {
        name for name, module in model.named_modules() if isinstance(module, NativeBlockFP8Linear)
    }
    if actual_native_indexers != expected_native_indexers:
        raise RuntimeError(
            "GLM-5.2 QLoRA frozen native indexer set mismatch: "
            f"missing={sorted(expected_native_indexers - actual_native_indexers)} "
            f"extra={sorted(actual_native_indexers - expected_native_indexers)}"
        )
    if len(actual_native_indexers) != GLM52_FROZEN_NATIVE_INDEXER_PROJECTION_COUNT:
        raise RuntimeError(
            "GLM-5.2 QLoRA requires exactly "
            f"{GLM52_FROZEN_NATIVE_INDEXER_PROJECTION_COUNT} frozen native indexer projections"
        )
    for layer_idx, indexer_type in enumerate(_official_indexer_schedule()):
        if indexer_type != "full":
            continue
        weights_proj = model.get_submodule(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj")
        if type(weights_proj) is not nn.Linear or weights_proj.weight.dtype is not torch.bfloat16:
            raise RuntimeError(
                f"GLM-5.2 QLoRA indexer weights_proj for layer {layer_idx} must remain an ordinary BF16 linear"
            )

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable != inventory.factor_names:
        raise RuntimeError(
            "GLM-5.2 QLoRA trainable factor set mismatch: "
            f"missing={sorted(inventory.factor_names - trainable)} extra={sorted(trainable - inventory.factor_names)}"
        )

    if len(expected_quantized) != GLM52_QLORA_QUANTIZED_LINEAR_COUNT:
        raise RuntimeError(f"GLM-5.2 QLoRA requires 624 quantized dense targets, got {len(expected_quantized)}")
    if len(expected_heads | expected_quantized) != GLM52_QLORA_ORDINARY_TARGET_COUNT:
        raise RuntimeError("GLM-5.2 QLoRA requires exactly 625 ordinary adapted linear modules")
    if len(expected_banks) != GLM52_QLORA_ROUTED_BANK_COUNT:
        raise RuntimeError(f"GLM-5.2 QLoRA requires 75 routed expert banks, got {len(expected_banks)}")
    if len(inventory.factors) != GLM52_QLORA_FACTOR_COUNT:
        raise RuntimeError(f"GLM-5.2 QLoRA requires exactly 1,700 logical factor tensors, got {len(inventory.factors)}")


def prepare_glm52_block_fp8_qlora(
    model: nn.Module,
    config,
    *,
    adapter_rank: int,
    adapter_alpha: int,
) -> Glm52AdapterInventory:
    """Construct the complete official GLM-5.2 training adapter fail-closed.

    This is a deterministic build pass, not a runtime manifest. It deliberately
    leaves the pre-existing exact scoring lane untouched.
    """

    if adapter_rank <= 0 or adapter_alpha <= 0:
        raise ValueError("GLM-5.2 QLoRA adapter_rank and adapter_alpha must be positive")
    quantization_config = _validate_official_config(config)
    targets = _expected_targets(model, config)
    excluded = quantization_config["modules_to_not_convert"]
    accidentally_excluded = sorted(
        target.name for target in targets if target.kind != "bf16_linear" and _is_config_excluded(target.name, excluded)
    )
    if accidentally_excluded:
        raise ValueError(f"GLM-5.2 QLoRA quantized targets are excluded by checkpoint config: {accidentally_excluded}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for target in targets:
        if target.kind == "block_fp8_linear":
            _replace_dense_target(model, target, adapter_rank=adapter_rank, adapter_alpha=adapter_alpha)
        elif target.kind == "block_fp8_routed_bank":
            _replace_routed_target(model, target, adapter_rank=adapter_rank, adapter_alpha=adapter_alpha)
        elif target.kind == "bf16_linear":
            _set_submodule(
                model,
                target.name,
                LoraLinear.from_module(
                    model.get_submodule(target.name),
                    r=adapter_rank,
                    lora_alpha=adapter_alpha,
                ),
            )
        else:
            raise AssertionError(f"Unknown GLM-5.2 QLoRA target kind: {target.kind}")

    # The DSA selector remains frozen and executes under no_grad, so retain its
    # checkpoint-native FP8 q/k projections without exposing the scoring-only
    # native module to any differentiable policy-bearing path.
    replace_glm52_native_fp8_modules(model, quantization_config)

    inventory = Glm52AdapterInventory(targets=targets, factors=_build_factor_inventory(model, targets))
    _validate_constructed_model(model, inventory)
    model._glm52_adapter_inventory = inventory
    return inventory


__all__ = [
    "GLM52_FROZEN_NATIVE_INDEXER_PROJECTION_COUNT",
    "GLM52_QLORA_FACTOR_COUNT",
    "GLM52_QLORA_ORDINARY_TARGET_COUNT",
    "GLM52_QLORA_QUANTIZED_LINEAR_COUNT",
    "GLM52_QLORA_ROUTED_BANK_COUNT",
    "Glm52AdapterFactor",
    "Glm52AdapterInventory",
    "Glm52AdapterTarget",
    "prepare_glm52_block_fp8_qlora",
]
