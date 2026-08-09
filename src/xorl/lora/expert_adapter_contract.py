"""Static execution contract for trainable adapters on routed experts.

The adapter-gradient ownership compiler consumes this model-independent
description instead of recognizing model classes, quantization implementations,
or backend names.  Expert wrappers build the contract from their configured
execution backend and factor layout; model-specific exact lanes may add stricter
guard fields after validating their own geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable

from xorl.distributed.gradient_reduction import GradientReductionDomain


EXPERT_PROJECTION_ROLES = frozenset({"gate_proj", "up_proj", "down_proj"})


def validate_gated_silu_expert_adapter_semantics(module: Any) -> None:
    """Reject expert programs the generic gate/up/down adapter cannot preserve."""

    required_properties = ("gated", "hidden_act", "swiglu_limit", "gate_up_bias", "down_bias")
    missing_properties = [name for name in required_properties if not hasattr(module, name)]
    if missing_properties:
        raise NotImplementedError(
            "The generic expert-LoRA wrapper requires explicit expert semantic properties; "
            f"missing={missing_properties}"
        )
    if (
        not bool(module.gated)
        or module.hidden_act != "silu"
        or float(module.swiglu_limit) != 0.0
        or module.gate_up_bias is not None
        or module.down_bias is not None
    ):
        raise NotImplementedError(
            "The generic expert-LoRA wrapper cannot preserve non-gated, non-SiLU, clamped, "
            "or expert-bias semantics; use a model-specific expert-LoRA implementation"
        )


class ExpertAdapterFactorOwnership(str, Enum):
    """Distributed ownership of one expert-adapter factor."""

    OWNER_SHARDED = "owner_sharded"
    EP_REPLICATED = "ep_replicated"


class ZeroTokenGradientBehavior(str, Enum):
    """How an execution path handles a rank with no local expert tokens."""

    STRUCTURAL_ZERO = "structural_zero"


@dataclass(frozen=True)
class ExpertAdapterBackendContract:
    """LoRA capabilities declared by one expert compute backend."""

    name: str
    producer_family: str
    supports_local: bool
    supports_ep: bool
    supported_dispatch_methods: tuple[str, ...]
    gradient_reduction_domain: GradientReductionDomain
    zero_token_gradient_behavior: ZeroTokenGradientBehavior


@dataclass(frozen=True)
class ExpertAdapterGradientContract:
    """Complete static ownership contract emitted by an expert wrapper."""

    backend: ExpertAdapterBackendContract
    factor_layout: str
    projection_roles: tuple[str, ...]
    factor_ownership: tuple[tuple[str, ExpertAdapterFactorOwnership], ...]
    factor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    supported_quantized_base_formats: tuple[str, ...] = ()
    quantized_base_format: str | None = None
    supports_efsdp_replication: bool = False
    requires_managed_fsdp: bool = False
    guard_fields: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        roles = tuple(dict.fromkeys(self.projection_roles))
        if not roles or roles != self.projection_roles:
            raise ValueError("Expert adapter projection roles must be non-empty and unique")
        unsupported_roles = set(roles) - EXPERT_PROJECTION_ROLES
        if unsupported_roles:
            raise ValueError(f"Unsupported expert adapter projection roles: {sorted(unsupported_roles)}")

        expected_factors = {f"{role}_lora_{suffix}" for role in roles for suffix in ("A", "B")}
        ownership = dict(self.factor_ownership)
        if len(ownership) != len(self.factor_ownership) or set(ownership) != expected_factors:
            raise ValueError(
                "Expert adapter factor ownership must cover exactly the selected projection factors; "
                f"expected={sorted(expected_factors)}, actual={sorted(ownership)}"
            )
        shapes = dict(self.factor_shapes)
        if len(shapes) != len(self.factor_shapes) or set(shapes) != expected_factors:
            raise ValueError(
                "Expert adapter factor shapes must cover exactly the selected projection factors; "
                f"expected={sorted(expected_factors)}, actual={sorted(shapes)}"
            )
        malformed_shapes = {
            name: shape
            for name, shape in shapes.items()
            if len(shape) != 3 or any(not isinstance(size, int) or size <= 0 for size in shape)
        }
        if malformed_shapes:
            raise ValueError(
                f"Expert adapter factors require positive three-dimensional GKN shapes: {malformed_shapes}"
            )

        formats = tuple(dict.fromkeys(self.supported_quantized_base_formats))
        if formats != self.supported_quantized_base_formats:
            raise ValueError("Supported expert quantization formats must be unique")
        if self.quantized_base_format is not None and self.quantized_base_format not in formats:
            raise ValueError(
                f"Expert quantization format {self.quantized_base_format!r} is not declared by the wrapper; "
                f"supported={formats}"
            )

    def ownership_for(self, local_parameter_name: str) -> ExpertAdapterFactorOwnership:
        try:
            return dict(self.factor_ownership)[local_parameter_name]
        except KeyError as error:
            raise ValueError(f"Expert adapter contract does not own parameter {local_parameter_name!r}") from error

    def for_active_rank(self, active_rank: int) -> ExpertAdapterGradientContract:
        """Specialize physical factor capacity to one immutable session rank."""

        if not isinstance(active_rank, int) or active_rank <= 0:
            raise ValueError(f"Expert adapter active rank must be a positive integer, got {active_rank!r}")
        specialized_shapes = []
        for name, shape in self.factor_shapes:
            dimensions = list(shape)
            rank_dim = 2 if name.endswith("_lora_A") else 1
            if active_rank > dimensions[rank_dim]:
                raise ValueError(
                    f"Expert adapter active rank {active_rank} exceeds factor capacity "
                    f"{dimensions[rank_dim]} for {name}"
                )
            dimensions[rank_dim] = active_rank
            specialized_shapes.append((name, tuple(dimensions)))
        return replace(self, factor_shapes=tuple(specialized_shapes))

    def gradient_reduction_by_parameter(self) -> dict[str, GradientReductionDomain]:
        """Resolve the EP reduction authority for every declared factor."""

        return {
            name: (
                self.backend.gradient_reduction_domain
                if ownership is ExpertAdapterFactorOwnership.EP_REPLICATED
                else GradientReductionDomain.NONE
            )
            for name, ownership in self.factor_ownership
        }

    def config_guard_fields(self) -> dict[str, Any]:
        """Return the immutable execution properties included in plan identity."""

        return {
            "expert_contract_version": 1,
            "expert_backend": self.backend.name,
            "expert_producer": self.backend.producer_family,
            "expert_factor_layout": self.factor_layout,
            "expert_projection_roles": ",".join(self.projection_roles),
            "expert_factor_ownership": ";".join(
                f"{name}:{ownership.value}" for name, ownership in self.factor_ownership
            ),
            "expert_factor_shapes": ";".join(
                f"{name}:{','.join(str(size) for size in shape)}" for name, shape in self.factor_shapes
            ),
            "expert_gradient_reduction": self.backend.gradient_reduction_domain.value,
            "expert_zero_token_gradients": self.backend.zero_token_gradient_behavior.value,
            "expert_quant_format": self.quantized_base_format or "none",
            "expert_requires_managed_fsdp": self.requires_managed_fsdp,
            **dict(self.guard_fields),
        }


def gated_expert_factor_ownership(
    projection_roles: Iterable[str],
    *,
    hybrid_shared: bool,
) -> tuple[tuple[str, ExpertAdapterFactorOwnership], ...]:
    """Build the canonical gate/up/down GKN factor ownership declaration."""

    roles = tuple(projection_roles)
    entries: list[tuple[str, ExpertAdapterFactorOwnership]] = []
    for role in roles:
        for suffix in ("A", "B"):
            shared = hybrid_shared and (
                (role in {"gate_proj", "up_proj"} and suffix == "A") or (role == "down_proj" and suffix == "B")
            )
            entries.append(
                (
                    f"{role}_lora_{suffix}",
                    ExpertAdapterFactorOwnership.EP_REPLICATED
                    if shared
                    else ExpertAdapterFactorOwnership.OWNER_SHARDED,
                )
            )
    return tuple(entries)


def gated_expert_factor_shapes(
    projection_roles: Iterable[str],
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    hybrid_shared: bool,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Build the logical GKN shape of each selected gate/up/down factor."""

    dimensions = {
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "rank": rank,
    }
    invalid = {name: value for name, value in dimensions.items() if not isinstance(value, int) or value <= 0}
    if invalid:
        raise ValueError(f"Expert adapter dimensions must be positive integers: {invalid}")

    shared = 1 if hybrid_shared else num_experts
    shapes = {
        "gate_proj_lora_A": (shared, hidden_size, rank),
        "gate_proj_lora_B": (num_experts, rank, intermediate_size),
        "up_proj_lora_A": (shared, hidden_size, rank),
        "up_proj_lora_B": (num_experts, rank, intermediate_size),
        "down_proj_lora_A": (num_experts, intermediate_size, rank),
        "down_proj_lora_B": (shared, rank, hidden_size),
    }
    return tuple(
        (f"{role}_lora_{suffix}", shapes[f"{role}_lora_{suffix}"]) for role in projection_roles for suffix in ("A", "B")
    )


__all__ = [
    "EXPERT_PROJECTION_ROLES",
    "ExpertAdapterBackendContract",
    "ExpertAdapterFactorOwnership",
    "ExpertAdapterGradientContract",
    "ZeroTokenGradientBehavior",
    "gated_expert_factor_ownership",
    "gated_expert_factor_shapes",
    "validate_gated_silu_expert_adapter_semantics",
]
