"""Static ownership plans for adapter gradients.

The compiler in this module is intentionally local and side-effect free.  It
turns explicit execution declarations plus the model's discovered tensor
layouts into one immutable plan.  Runtime capture and transport consume that
plan elsewhere; this module does not execute collectives or mutate gradients.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch
import torch.nn as nn

from xorl.server.runner.adapters.sharded_state import AdapterTensorLayout, wait_for_local_tensor


class AdapterGradientOwnershipError(RuntimeError):
    """The active adapter topology does not have one complete ownership plan."""


class AdapterGradientUniformRejection(AdapterGradientOwnershipError):
    """A command-wide rejection whose inputs are bit-identical on every rank."""


class TopologyFamily(str, Enum):
    DENSE_REPLICATED = "dense_replicated"
    DIRECT_OUTPUT_PROJECTION = "direct_output_projection"
    EP_REPLICATED_SHARED = "ep_replicated_shared"
    OWNER_SHARDED = "owner_sharded"


class ProducerFamily(str, Enum):
    MODULE_MANAGED = "module_managed"
    DIRECT_OUTPUT_PROJECTION = "direct_output_projection"
    FUSED_MANAGED = "fused_managed"


class GradientRepresentation(str, Enum):
    FSDP_COMPLETED_LOCAL_SHARD = "fsdp_completed_local_shard"
    DIRECT_DTENSOR_CONTRIBUTION = "direct_dtensor_contribution"
    REPLICATED_LOCAL_CONTRIBUTION = "replicated_local_contribution"
    OWNER_LOCAL_CONTRIBUTION = "owner_local_contribution"
    FULL_LOGICAL_CONTRIBUTION = "full_logical_contribution"


class GradientScaleState(str, Enum):
    RAW_NUMERATOR = "raw_numerator"
    PRE_NORMALIZED = "pre_normalized"


class GradientPresencePolicy(str, Enum):
    REQUIRED = "required"
    REQUIRED_IF_ACTIVE = "required_if_active"
    AUTHORIZED_ZERO = "authorized_zero"


class ReductionAxis(str, Enum):
    FSDP_SHARD = "fsdp_shard"
    SEQUENCE_PARALLEL = "sequence_parallel"
    OUTPUT_PROJECTION_REPLICA = "output_projection_replica"
    EXPERT_PARALLEL_REPLICA = "expert_parallel_replica"


class ReductionAuthority(str, Enum):
    FSDP = "fsdp"
    ADAPTER_CAPTURE = "adapter_capture"
    GENERIC_SP_SYNC = "generic_sp_sync"
    OUTPUT_PROJECTION_SYNC = "output_projection_sync"
    ADAPTER_FINALIZER = "adapter_finalizer"


class ReductionOperation(str, Enum):
    SUM = "sum"
    IDENTITY = "identity"


@dataclass(frozen=True, order=True)
class ReductionDomainPlan:
    axis: ReductionAxis
    authority: ReductionAuthority
    operation: ReductionOperation
    group_key: str

    def __post_init__(self) -> None:
        if not self.group_key:
            raise AdapterGradientOwnershipError("Reduction domains require a resolved group key")
        if self.operation is ReductionOperation.IDENTITY and self.authority is ReductionAuthority.ADAPTER_FINALIZER:
            raise AdapterGradientOwnershipError("Adapter finalization cannot own an identity reduction")


@dataclass(frozen=True)
class ParameterOwnershipDeclaration:
    topology: TopologyFamily
    producer: ProducerFamily
    representation: GradientRepresentation
    completed_domains: tuple[ReductionDomainPlan, ...]
    pending_domains: tuple[ReductionDomainPlan, ...]
    capture_domains: tuple[ReductionDomainPlan, ...] = ()
    presence: GradientPresencePolicy = GradientPresencePolicy.REQUIRED
    scale_state: GradientScaleState = GradientScaleState.RAW_NUMERATOR
    config_guard_fingerprint: str = ""
    config_guard_fields: tuple[tuple[str, object], ...] = ()
    managed_fsdp_shard: bool = False

    def __post_init__(self) -> None:
        if not self.config_guard_fingerprint:
            raise AdapterGradientOwnershipError("Producer-changing configuration requires a fingerprint")
        if self.config_guard_fields != tuple(sorted(self.config_guard_fields)):
            raise AdapterGradientOwnershipError("Producer-changing configuration fields must be sorted")
        guard_names = tuple(name for name, _value in self.config_guard_fields)
        if len(guard_names) != len(set(guard_names)):
            raise AdapterGradientOwnershipError("Producer-changing configuration fields must be unique")
        if any(type(value) not in {bool, int, str} for _name, value in self.config_guard_fields):
            raise AdapterGradientOwnershipError("Producer-changing configuration fields must be scalar")
        if self.scale_state is not GradientScaleState.RAW_NUMERATOR:
            raise AdapterGradientOwnershipError("Revision one admits only raw gradient numerators")
        if self.topology is TopologyFamily.DIRECT_OUTPUT_PROJECTION:
            if self.producer is not ProducerFamily.DIRECT_OUTPUT_PROJECTION:
                raise AdapterGradientOwnershipError("Direct output projection requires its exact producer")
        completed_axes = _unique_axes(self.completed_domains, "completed")
        capture_axes = _unique_axes(self.capture_domains, "capture")
        pending_axes = _unique_axes(self.pending_domains, "pending")
        overlap = (completed_axes & capture_axes) | (completed_axes & pending_axes) | (capture_axes & pending_axes)
        if overlap:
            raise AdapterGradientOwnershipError(
                f"Reduction axes cannot be both complete and pending: {sorted(axis.value for axis in overlap)}"
            )
        if any(domain.authority is not ReductionAuthority.ADAPTER_FINALIZER for domain in self.pending_domains):
            raise AdapterGradientOwnershipError("Every pending reduction must be owned by adapter finalization")
        if any(domain.authority is not ReductionAuthority.ADAPTER_CAPTURE for domain in self.capture_domains):
            raise AdapterGradientOwnershipError("Every capture reduction must be owned by adapter capture")
        fsdp_domains = tuple(
            domain
            for domain in (*self.completed_domains, *self.capture_domains, *self.pending_domains)
            if domain.axis is ReductionAxis.FSDP_SHARD
        )
        if self.managed_fsdp_shard and not fsdp_domains:
            raise AdapterGradientOwnershipError("Managed FSDP ownership requires one declared FSDP domain")
        if any(domain.authority is ReductionAuthority.FSDP for domain in fsdp_domains) and not self.managed_fsdp_shard:
            raise AdapterGradientOwnershipError("FSDP completion requires public managed-FSDP ownership")


def _unique_axes(domains: tuple[ReductionDomainPlan, ...], label: str) -> set[ReductionAxis]:
    axes = [domain.axis for domain in domains]
    if len(axes) != len(set(axes)):
        raise AdapterGradientOwnershipError(f"A parameter has duplicate {label} reduction axes")
    return set(axes)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return wait_for_local_tensor(to_local() if to_local is not None else tensor)


@dataclass(frozen=True)
class CompiledParameterOwnership:
    fqn: str
    topology: TopologyFamily
    producer: ProducerFamily
    representation: GradientRepresentation
    logical_shape: tuple[int, ...]
    local_logical_offset: tuple[int, ...]
    active_storage_shape: tuple[int, ...]
    completed_domains: tuple[ReductionDomainPlan, ...]
    capture_domains: tuple[ReductionDomainPlan, ...]
    pending_domains: tuple[ReductionDomainPlan, ...]
    presence: GradientPresencePolicy
    scale_state: GradientScaleState
    config_guard_fingerprint: str
    config_guard_fields: tuple[tuple[str, object], ...]
    managed_fsdp_shard: bool
    norm_replica_divisor: int

    def semantic_payload(self) -> dict[str, object]:
        return {
            "fqn": self.fqn,
            "topology": self.topology.value,
            "producer": self.producer.value,
            "representation": self.representation.value,
            "logical_shape": self.logical_shape,
            "completed_domains": [_domain_payload(domain) for domain in self.completed_domains],
            "capture_domains": [_domain_payload(domain) for domain in self.capture_domains],
            "pending_domains": [_domain_payload(domain) for domain in self.pending_domains],
            "presence": self.presence.value,
            "scale_state": self.scale_state.value,
            "config_guard_fingerprint": self.config_guard_fingerprint,
            "config_guard_fields": dict(self.config_guard_fields),
            "managed_fsdp_shard": self.managed_fsdp_shard,
            "norm_replica_divisor": self.norm_replica_divisor,
        }

    @property
    def requires_local_gradient(self) -> bool:
        if self.presence is GradientPresencePolicy.REQUIRED:
            return True
        if self.presence is GradientPresencePolicy.REQUIRED_IF_ACTIVE:
            return all(size > 0 for size in self.active_storage_shape)
        return False


def _domain_payload(domain: ReductionDomainPlan) -> tuple[str, str, str, str]:
    return domain.axis.value, domain.authority.value, domain.operation.value, domain.group_key


@dataclass(frozen=True)
class AuthorityMask:
    axis: ReductionAxis
    authority: ReductionAuthority
    fqns: tuple[str, ...]


@dataclass(frozen=True)
class AdapterGradientOwnershipPlan:
    model_generation: str
    adapter_generation: str
    parameters: tuple[CompiledParameterOwnership, ...]
    authority_masks: tuple[AuthorityMask, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_generation or not self.adapter_generation or not self.parameters:
            raise AdapterGradientOwnershipError("Ownership plans require generations and parameters")
        if tuple(item.fqn for item in self.parameters) != tuple(sorted({item.fqn for item in self.parameters})):
            raise AdapterGradientOwnershipError("Ownership parameters must be canonical and unique")

    def parameter(self, fqn: str) -> CompiledParameterOwnership:
        matches = tuple(item for item in self.parameters if item.fqn == fqn)
        if len(matches) != 1:
            raise AdapterGradientOwnershipError(f"Unknown or duplicate adapter parameter {fqn!r}")
        return matches[0]

    def optimizer_restore_contract(self) -> dict[str, object]:
        """Return direct optimizer-coordinate semantics, excluding identity labels.

        ``model_generation``, ``adapter_generation``, and ``fingerprint`` are
        registration/audit labels.  Restore compatibility is narrower and is
        expressed field-by-field so mutable session metadata cannot become a
        false checkpoint rejection.
        """

        parameters = []
        for item in self.parameters:
            parameters.append(
                {
                    "fqn": item.fqn,
                    "topology": item.topology.value,
                    "producer": item.producer.value,
                    "representation": item.representation.value,
                    "logical_shape": list(item.logical_shape),
                    "completed_domains": [list(_domain_payload(domain)) for domain in item.completed_domains],
                    "capture_domains": [list(_domain_payload(domain)) for domain in item.capture_domains],
                    "pending_domains": [list(_domain_payload(domain)) for domain in item.pending_domains],
                    "presence": item.presence.value,
                    "scale_state": item.scale_state.value,
                    "config_guard_fields": dict(item.config_guard_fields),
                    "managed_fsdp_shard": item.managed_fsdp_shard,
                    "norm_replica_divisor": item.norm_replica_divisor,
                }
            )
        return {
            "schema": "adapter-gradient-optimizer-restore-v1",
            "parameters": parameters,
            "authority_masks": [
                {
                    "axis": mask.axis.value,
                    "authority": mask.authority.value,
                    "fqns": list(mask.fqns),
                }
                for mask in self.authority_masks
            ],
        }


def compile_adapter_gradient_ownership(
    *,
    layouts: Mapping[str, AdapterTensorLayout],
    model_parameters: Mapping[str, nn.Parameter],
    optimizer_parameters: Mapping[str, nn.Parameter],
    declarations: Mapping[str, ParameterOwnershipDeclaration],
    model_generation: str,
    adapter_generation: str,
    tensor_parallel_size: int = 1,
    group_memberships: Mapping[str, tuple[tuple[int, ...], ...]] | None = None,
    rank: int = 0,
) -> AdapterGradientOwnershipPlan:
    """Compile one immutable, fail-closed ownership plan without changing execution."""

    if not model_generation or not adapter_generation:
        raise AdapterGradientOwnershipError("Ownership compilation requires explicit generations")
    if tensor_parallel_size > 1:
        raise AdapterGradientOwnershipError(
            "Adapter-gradient ownership does not admit model tensor parallelism greater than one"
        )
    layout_by_fqn = _canonical_layout_mapping(layouts)
    model_by_fqn = _canonical_parameter_mapping(model_parameters)
    optimizer_by_fqn = _canonical_parameter_mapping(optimizer_parameters)
    declaration_by_fqn = _canonical_declaration_mapping(declarations)
    memberships = {
        key: tuple(frozenset(group) for group in groups) for key, groups in (group_memberships or {}).items()
    }
    universes = (set(layout_by_fqn), set(model_by_fqn), set(optimizer_by_fqn), set(declaration_by_fqn))
    if any(universe != universes[0] for universe in universes[1:]) or not universes[0]:
        raise AdapterGradientOwnershipError("Layouts, parameters, and declarations must cover one exact universe")

    compiled: list[CompiledParameterOwnership] = []
    for fqn in sorted(layout_by_fqn):
        layout = layout_by_fqn[fqn]
        model_parameter = model_by_fqn[fqn]
        optimizer_parameter = optimizer_by_fqn[fqn]
        declaration = declaration_by_fqn[fqn]
        local_model = _local_tensor(model_parameter)
        if tuple(local_model.shape) != layout.local_substrate_shape:
            raise AdapterGradientOwnershipError(f"Model parameter shape differs from layout for {fqn!r}")
        if tuple(optimizer_parameter.shape) != layout.active_storage_shape:
            raise AdapterGradientOwnershipError(f"Optimizer parameter shape differs from layout for {fqn!r}")
        model_is_dtensor = hasattr(model_parameter, "to_local") and hasattr(model_parameter, "placements")
        if model_is_dtensor and not declaration.managed_fsdp_shard:
            raise AdapterGradientOwnershipError(f"DTensor parameter {fqn!r} lacks public managed-FSDP ownership")
        if declaration.topology is TopologyFamily.EP_REPLICATED_SHARED and (
            not layout.needs_ep_gradient_sync
            or not any(domain.axis is ReductionAxis.EXPERT_PARALLEL_REPLICA for domain in declaration.pending_domains)
        ):
            raise AdapterGradientOwnershipError("EP-replicated topology requires an explicit pending EP sum")
        if declaration.topology is TopologyFamily.OWNER_SHARDED and not layout.is_ep_owned:
            raise AdapterGradientOwnershipError("Owner-sharded topology requires explicit EP ownership")
        replica_class = frozenset(layout.replica_ranks)
        if not replica_class or rank not in replica_class or len(replica_class) != layout.replica_count:
            raise AdapterGradientOwnershipError(f"Discovered replica class is invalid for {fqn!r}")
        replica_domains = tuple(
            domain
            for domain in (*declaration.completed_domains, *declaration.capture_domains, *declaration.pending_domains)
            if domain.axis is not ReductionAxis.FSDP_SHARD and domain.operation is ReductionOperation.SUM
        )
        resolved_families: list[tuple[frozenset[int], ...]] = []
        for domain in replica_domains:
            family = memberships.get(domain.group_key)
            if not family:
                raise AdapterGradientOwnershipError(
                    f"Replica-reducing group identity {domain.group_key!r} is missing for {fqn!r}"
                )
            resolved_families.append(family)
        for point in replica_class:
            crossing: list[frozenset[int]] = []
            for family in resolved_families:
                matches = [members for members in family if point in members]
                if len(matches) != 1:
                    raise AdapterGradientOwnershipError(
                        f"Replica-reducing group identity is missing or ambiguous at rank {point} for {fqn!r}"
                    )
                members = matches[0]
                if not members.issubset(replica_class):
                    raise AdapterGradientOwnershipError(
                        f"Replica-reducing group includes ranks outside the replica class for {fqn!r}"
                    )
                crossing.append(members)
            for index, members in enumerate(crossing):
                for other in crossing[index + 1 :]:
                    if members & other != {point}:
                        raise AdapterGradientOwnershipError(
                            f"Replica-reducing groups overlap beyond rank {point} for {fqn!r}"
                        )
        reachable = {rank}
        changed = True
        while changed:
            changed = False
            for family in resolved_families:
                for members in family:
                    if reachable & members and not members.issubset(reachable):
                        reachable.update(members)
                        changed = True
        if reachable != set(replica_class):
            raise AdapterGradientOwnershipError(
                f"Declared replica reductions do not exactly cover the discovered replica class for {fqn!r}"
            )
        compiled.append(
            CompiledParameterOwnership(
                fqn=fqn,
                topology=declaration.topology,
                producer=declaration.producer,
                representation=declaration.representation,
                logical_shape=layout.logical_shape,
                local_logical_offset=layout.active_global_offset,
                active_storage_shape=layout.active_storage_shape,
                completed_domains=tuple(sorted(declaration.completed_domains, key=_domain_payload)),
                capture_domains=tuple(sorted(declaration.capture_domains, key=_domain_payload)),
                pending_domains=tuple(sorted(declaration.pending_domains, key=_domain_payload)),
                presence=declaration.presence,
                scale_state=declaration.scale_state,
                config_guard_fingerprint=declaration.config_guard_fingerprint,
                config_guard_fields=declaration.config_guard_fields,
                managed_fsdp_shard=declaration.managed_fsdp_shard,
                norm_replica_divisor=len(replica_class),
            )
        )

    masks: dict[tuple[ReductionAxis, ReductionAuthority], list[str]] = {}
    for item in compiled:
        for domain in (*item.completed_domains, *item.capture_domains, *item.pending_domains):
            masks.setdefault((domain.axis, domain.authority), []).append(item.fqn)
    authority_masks = tuple(
        AuthorityMask(axis, authority, tuple(sorted(fqns)))
        for (axis, authority), fqns in sorted(masks.items(), key=lambda item: (item[0][0].value, item[0][1].value))
    )
    payload = {
        "model_generation": model_generation,
        "adapter_generation": adapter_generation,
        "parameters": [item.semantic_payload() for item in compiled],
        "authority_masks": [(mask.axis.value, mask.authority.value, mask.fqns) for mask in authority_masks],
        "schema": "adapter-gradient-ownership-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return AdapterGradientOwnershipPlan(
        model_generation,
        adapter_generation,
        tuple(compiled),
        authority_masks,
        fingerprint,
    )


def _canonical_layout_mapping(
    values: Mapping[str, AdapterTensorLayout],
) -> dict[str, AdapterTensorLayout]:
    result: dict[str, AdapterTensorLayout] = {}
    for value in values.values():
        if value.fqn in result:
            raise AdapterGradientOwnershipError(f"Duplicate canonical layout {value.fqn!r}")
        result[value.fqn] = value
    return result


def _canonical_parameter_mapping(values: Mapping[str, nn.Parameter]) -> dict[str, nn.Parameter]:
    result: dict[str, nn.Parameter] = {}
    layout_names = {name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): name for name in values}
    if len(layout_names) != len(values):
        raise AdapterGradientOwnershipError("Parameter mapping has duplicate canonical names")
    for canonical, original in layout_names.items():
        result[canonical] = values[original]
    return result


def _canonical_declaration_mapping(
    values: Mapping[str, ParameterOwnershipDeclaration],
) -> dict[str, ParameterOwnershipDeclaration]:
    result: dict[str, ParameterOwnershipDeclaration] = {}
    for name, declaration in values.items():
        canonical = name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
        if canonical in result:
            raise AdapterGradientOwnershipError(f"Duplicate canonical declaration {canonical!r}")
        result[canonical] = declaration
    return result


__all__ = [
    "AdapterGradientOwnershipError",
    "AdapterGradientUniformRejection",
    "AdapterGradientOwnershipPlan",
    "AuthorityMask",
    "CompiledParameterOwnership",
    "GradientPresencePolicy",
    "GradientRepresentation",
    "GradientScaleState",
    "ParameterOwnershipDeclaration",
    "ProducerFamily",
    "ReductionAuthority",
    "ReductionAxis",
    "ReductionDomainPlan",
    "ReductionOperation",
    "TopologyFamily",
    "compile_adapter_gradient_ownership",
]
