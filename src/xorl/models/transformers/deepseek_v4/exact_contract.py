"""Fail-closed DeepSeek-V4-Flash exact active-LoRA admission.

This module deliberately contains no numerical kernels.  It freezes the first
RCA lane and derives the complete logical adapter surface before model
wrapping, sharding, or checkpoint loading.  Kernel engagement and exactness
remain separate runtime gates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT = 345
DSV4_FLASH_ROUTED_BANK_COUNT = 43
DSV4_FLASH_TARGET_ENTITY_COUNT = (
    DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT + DSV4_FLASH_ROUTED_BANK_COUNT
)
DSV4_FLASH_LOGICAL_FACTOR_COUNT = 948
DSV4_FLASH_REQUIRED_TARGET_MODULES = frozenset(
    {
        "down_proj",
        "gate_proj",
        "lm_head",
        "up_proj",
        "wkv",
        "wo_a",
        "wo_b",
        "wq_a",
        "wq_b",
    }
)

DSV4_FLASH_COMPRESS_RATIOS = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
)


@dataclass(frozen=True)
class Dsv4FlashAdapterTarget:
    """One trainer-logical adapted projection or routed projection bank."""

    name: str
    role: str
    kind: str
    in_features: int
    out_features: int
    num_experts: int = 1


@dataclass(frozen=True)
class Dsv4FlashAdapterFactor:
    """One FP32 logical factor master in the initial training lane."""

    name: str
    target_name: str
    role: str
    factor: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class Dsv4FlashAdapterInventory:
    """Complete logical target and factor inventory."""

    targets: tuple[Dsv4FlashAdapterTarget, ...]
    factors: tuple[Dsv4FlashAdapterFactor, ...]

    @property
    def target_names(self) -> frozenset[str]:
        return frozenset(target.name for target in self.targets)

    @property
    def factor_names(self) -> frozenset[str]:
        return frozenset(factor.name for factor in self.factors)

    @property
    def role_counts(self) -> dict[str, int]:
        return dict(Counter(target.role for target in self.targets))


@dataclass(frozen=True)
class Dsv4FlashTrainingTopology:
    """First byte-proxy candidate; not a promotion-qualified topology."""

    world_size: int
    dp_size: int
    dp_replicate_size: int
    dp_shard_size: int
    tp_size: int
    pp_size: int
    ep_size: int
    cp_size: int
    ringattn_size: int
    ulysses_size: int
    lm_head_tp_size: int


DSV4_FLASH_RCA_TRAINING_TOPOLOGY = Dsv4FlashTrainingTopology(
    world_size=8,
    dp_size=8,
    dp_replicate_size=1,
    dp_shard_size=8,
    tp_size=1,
    pp_size=1,
    ep_size=8,
    cp_size=1,
    ringattn_size=1,
    ulysses_size=1,
    lm_head_tp_size=8,
)


def is_dsv4_flash_config(config: Any) -> bool:
    architectures = set(getattr(config, "architectures", None) or ())
    return (
        getattr(config, "model_type", None) == "deepseek_v4"
        and "DeepseekV4ForCausalLM" in architectures
    )


def _validate_mapping(actual: Any, expected: dict[str, Any], *, label: str) -> list[str]:
    if isinstance(actual, Mapping):
        get_value = actual.get
    elif hasattr(actual, "__dict__"):
        # ServerArguments/HF config normalization recursively materializes
        # JSON mappings as namespaces. Preserve strict field/value checks
        # while accepting that representation of the same immutable config.
        def get_value(name: str) -> Any:
            return getattr(actual, name, None)

    else:
        return [f"{label}={actual!r} (requires a mapping)"]
    return [
        f"{label}.{name}={get_value(name)!r} (requires {value!r})"
        for name, value in expected.items()
        if get_value(name) != value
    ]


def validate_dsv4_flash_official_geometry(config: Any) -> None:
    """Require the public DeepSeek-V4-Flash checkpoint geometry exactly."""

    if not is_dsv4_flash_config(config):
        raise ValueError(
            "The DeepSeek-V4-Flash exact contract requires model_type='deepseek_v4' "
            "and architecture DeepseekV4ForCausalLM"
        )

    expected = {
        "vocab_size": 129280,
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "sliding_window": 128,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "num_hash_layers": 3,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "compress_rope_theta": 160000,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "expert_dtype": "fp4",
        "num_nextn_predict_layers": 1,
    }
    mismatches = [
        f"{name}={getattr(config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    if tuple(getattr(config, "compress_ratios", ()) or ()) != DSV4_FLASH_COMPRESS_RATIOS:
        mismatches.append("compress_ratios does not match the official C0/C4/C128 schedule")
    mismatches.extend(
        _validate_mapping(
            getattr(config, "quantization_config", None),
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
            label="quantization_config",
        )
    )
    mismatches.extend(
        _validate_mapping(
            getattr(config, "rope_scaling", None),
            {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            label="rope_scaling",
        )
    )
    if mismatches:
        raise ValueError(
            "The exact DeepSeek-V4-Flash program supports only the official model geometry: "
            + ", ".join(mismatches)
        )


def validate_dsv4_flash_training_topology(parallel_state: Any) -> Dsv4FlashTrainingTopology:
    """Resolve and validate the first shape-preserving WORLD8 RCA proxy."""

    actual = Dsv4FlashTrainingTopology(
        world_size=int(parallel_state.world_size),
        dp_size=int(parallel_state.dp_size),
        dp_replicate_size=int(parallel_state.dp_replicate_size),
        dp_shard_size=int(parallel_state.dp_shard_size),
        tp_size=int(parallel_state.tp_size),
        pp_size=int(parallel_state.pp_size),
        ep_size=int(parallel_state.ep_size),
        cp_size=int(parallel_state.cp_size),
        ringattn_size=int(parallel_state.ringattn_size),
        ulysses_size=int(parallel_state.ulysses_size),
        lm_head_tp_size=int(getattr(parallel_state, "lm_head_tp_size", 1)),
    )
    if actual != DSV4_FLASH_RCA_TRAINING_TOPOLOGY:
        raise ValueError(
            "The DeepSeek-V4-Flash exact RCA lane currently admits only "
            f"{DSV4_FLASH_RCA_TRAINING_TOPOLOGY}; got {actual}. This topology is "
            "a byte-proxy candidate and is not promotion-qualified until full replay agrees."
        )
    return actual


def validate_dsv4_flash_adapter_program(
    *,
    adapter_rank: int,
    adapter_alpha: int,
    target_modules: Any = None,
) -> None:
    mismatches = []
    if (adapter_rank, adapter_alpha) != (1, 1):
        mismatches.append(f"rank/alpha={(adapter_rank, adapter_alpha)!r} (requires (1, 1))")
    if target_modules is not None:
        actual_targets = set(target_modules)
        if actual_targets != DSV4_FLASH_REQUIRED_TARGET_MODULES:
            mismatches.append(
                "target_modules mismatch: "
                f"missing={sorted(DSV4_FLASH_REQUIRED_TARGET_MODULES - actual_targets)}, "
                f"extra={sorted(actual_targets - DSV4_FLASH_REQUIRED_TARGET_MODULES)}"
            )
    if mismatches:
        raise ValueError(
            "The exact DeepSeek-V4-Flash active-LoRA lane requires the complete "
            "rank-1/alpha-1 adapter: " + ", ".join(mismatches)
        )


def _linear_factors(
    target: Dsv4FlashAdapterTarget,
    *,
    rank: int,
    dtype: torch.dtype,
) -> tuple[Dsv4FlashAdapterFactor, Dsv4FlashAdapterFactor]:
    return (
        Dsv4FlashAdapterFactor(
            name=f"{target.name}.lora_A",
            target_name=target.name,
            role=target.role,
            factor="lora_A",
            shape=(rank, target.in_features),
            dtype=dtype,
        ),
        Dsv4FlashAdapterFactor(
            name=f"{target.name}.lora_B",
            target_name=target.name,
            role=target.role,
            factor="lora_B",
            shape=(target.out_features, rank),
            dtype=dtype,
        ),
    )


def _routed_factors(
    target: Dsv4FlashAdapterTarget,
    *,
    rank: int,
    dtype: torch.dtype,
) -> tuple[Dsv4FlashAdapterFactor, ...]:
    rows = []
    for projection, in_features, out_features in (
        ("gate_proj", target.in_features, target.out_features),
        ("up_proj", target.in_features, target.out_features),
        ("down_proj", target.out_features, target.in_features),
    ):
        for factor, shape in (
            ("lora_A", (target.num_experts, in_features, rank)),
            ("lora_B", (target.num_experts, rank, out_features)),
        ):
            rows.append(
                Dsv4FlashAdapterFactor(
                    name=f"{target.name}.{projection}_{factor}",
                    target_name=target.name,
                    role=f"routed_expert.{projection}",
                    factor=f"{projection}_{factor}",
                    shape=shape,
                    dtype=dtype,
                )
            )
    return tuple(rows)


def build_dsv4_flash_adapter_inventory(
    config: Any,
    *,
    adapter_rank: int = 1,
    adapter_alpha: int = 1,
    factor_dtype: torch.dtype = torch.float32,
) -> Dsv4FlashAdapterInventory:
    """Derive all 345 logical targets and 948 factor tensors."""

    validate_dsv4_flash_official_geometry(config)
    validate_dsv4_flash_adapter_program(adapter_rank=adapter_rank, adapter_alpha=adapter_alpha)
    hidden = int(config.hidden_size)
    head_dim = int(config.head_dim)
    q_rank = int(config.q_lora_rank)
    o_rank = int(config.o_lora_rank)
    o_groups = int(config.o_groups)
    num_heads = int(config.num_attention_heads)
    moe_intermediate = int(config.moe_intermediate_size)
    num_experts = int(config.n_routed_experts)

    targets: list[Dsv4FlashAdapterTarget] = []
    factors: list[Dsv4FlashAdapterFactor] = []
    for layer_id in range(int(config.num_hidden_layers)):
        layer = f"model.layers.{layer_id}"
        attn = f"{layer}.self_attn"
        for suffix, in_features, out_features in (
            ("wq_a", hidden, q_rank),
            ("wq_b", q_rank, num_heads * head_dim),
            ("wkv", hidden, head_dim),
            ("wo_a", num_heads * head_dim // o_groups, o_groups * o_rank),
            ("wo_b", o_groups * o_rank, hidden),
        ):
            target = Dsv4FlashAdapterTarget(
                name=f"{attn}.{suffix}",
                role=f"attention.{suffix}",
                kind="native_fp8_linear",
                in_features=in_features,
                out_features=out_features,
            )
            targets.append(target)
            factors.extend(_linear_factors(target, rank=adapter_rank, dtype=factor_dtype))

        shared = f"{layer}.mlp.shared_experts"
        for suffix, in_features, out_features in (
            ("gate_proj", hidden, moe_intermediate),
            ("up_proj", hidden, moe_intermediate),
            ("down_proj", moe_intermediate, hidden),
        ):
            target = Dsv4FlashAdapterTarget(
                name=f"{shared}.{suffix}",
                role=f"shared_expert.{suffix}",
                kind="native_fp8_linear",
                in_features=in_features,
                out_features=out_features,
            )
            targets.append(target)
            factors.extend(_linear_factors(target, rank=adapter_rank, dtype=factor_dtype))

        routed = Dsv4FlashAdapterTarget(
            name=f"{layer}.mlp.experts",
            role="routed_expert.bank",
            kind="native_mxfp4_routed_bank",
            in_features=hidden,
            out_features=moe_intermediate,
            num_experts=num_experts,
        )
        targets.append(routed)
        factors.extend(_routed_factors(routed, rank=adapter_rank, dtype=factor_dtype))

    lm_head = Dsv4FlashAdapterTarget(
        name="lm_head",
        role="output.lm_head",
        kind="native_fp8_linear",
        in_features=hidden,
        out_features=int(config.vocab_size),
    )
    targets.append(lm_head)
    factors.extend(_linear_factors(lm_head, rank=adapter_rank, dtype=factor_dtype))

    inventory = Dsv4FlashAdapterInventory(tuple(targets), tuple(factors))
    if len(inventory.targets) != DSV4_FLASH_TARGET_ENTITY_COUNT:
        raise AssertionError(
            f"Internal DSV4 target inventory error: expected {DSV4_FLASH_TARGET_ENTITY_COUNT}, "
            f"got {len(inventory.targets)}"
        )
    routed_count = sum(target.kind == "native_mxfp4_routed_bank" for target in inventory.targets)
    if routed_count != DSV4_FLASH_ROUTED_BANK_COUNT:
        raise AssertionError(
            f"Internal DSV4 routed-bank inventory error: expected {DSV4_FLASH_ROUTED_BANK_COUNT}, "
            f"got {routed_count}"
        )
    non_routed_count = len(inventory.targets) - routed_count
    if non_routed_count != DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT:
        raise AssertionError(
            "Internal DSV4 non-routed projection inventory error: expected "
            f"{DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT}, got {non_routed_count}"
        )
    if len(inventory.factors) != DSV4_FLASH_LOGICAL_FACTOR_COUNT:
        raise AssertionError(
            f"Internal DSV4 factor inventory error: expected {DSV4_FLASH_LOGICAL_FACTOR_COUNT}, "
            f"got {len(inventory.factors)}"
        )
    if len(inventory.target_names) != len(inventory.targets):
        raise AssertionError("DSV4 target inventory contains duplicate names")
    if len(inventory.factor_names) != len(inventory.factors):
        raise AssertionError("DSV4 factor inventory contains duplicate names")
    return inventory


def bind_dsv4_flash_adapter_inventory(model: Any) -> Dsv4FlashAdapterInventory:
    """Validate the complete live FP32 factor bank and freeze its inventory."""

    inventory = build_dsv4_flash_adapter_inventory(model.config)
    expected = {factor.name: factor for factor in inventory.factors}
    actual = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise RuntimeError(
            "The live DSV4-Flash adapter does not match the complete 948-factor "
            f"inventory: missing={missing[:8]}, extra={extra[:8]}"
        )
    mismatches = []
    for name, spec in expected.items():
        parameter = actual[name]
        if tuple(parameter.shape) != spec.shape:
            mismatches.append(
                f"{name} shape={tuple(parameter.shape)} (requires {spec.shape})"
            )
        if parameter.dtype is not spec.dtype:
            mismatches.append(
                f"{name} dtype={parameter.dtype} (requires {spec.dtype})"
            )
        if not parameter.requires_grad:
            mismatches.append(f"{name} is frozen")
    if mismatches:
        raise RuntimeError(
            "The live DSV4-Flash adapter violates its FP32 trainable-factor "
            "contract: " + ", ".join(mismatches[:16])
        )
    model._dsv4_adapter_inventory = inventory
    model._dsv4_flash_exact_active_lora_component = True
    iter_modules = getattr(model, "modules", None)
    for module in iter_modules() if callable(iter_modules) else ():
        if any(
            "lora_A" in local_name or "lora_B" in local_name
            for local_name, _parameter in module.named_parameters(recurse=False)
        ):
            module._dsv4_flash_exact_active_lora_component = True
    return inventory


__all__ = [
    "DSV4_FLASH_COMPRESS_RATIOS",
    "DSV4_FLASH_LOGICAL_FACTOR_COUNT",
    "DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT",
    "DSV4_FLASH_RCA_TRAINING_TOPOLOGY",
    "DSV4_FLASH_REQUIRED_TARGET_MODULES",
    "DSV4_FLASH_ROUTED_BANK_COUNT",
    "DSV4_FLASH_TARGET_ENTITY_COUNT",
    "Dsv4FlashAdapterFactor",
    "Dsv4FlashAdapterInventory",
    "Dsv4FlashAdapterTarget",
    "Dsv4FlashTrainingTopology",
    "build_dsv4_flash_adapter_inventory",
    "bind_dsv4_flash_adapter_inventory",
    "is_dsv4_flash_config",
    "validate_dsv4_flash_adapter_program",
    "validate_dsv4_flash_official_geometry",
    "validate_dsv4_flash_training_topology",
]
