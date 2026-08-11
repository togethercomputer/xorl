"""Analytical and structural tests for generic adapter-gradient ownership."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from xorl.distributed.gradient_reduction import GradientReductionDomain
from xorl.lora.modules.linear import LoraLinear
from xorl.server.runner.adapters import gradient_finalizer as gradient_finalizer_module
from xorl.server.runner.adapters.gradient_finalizer import (
    logical_l2_norm,
    transport_complete_local_gradients,
)
from xorl.server.runner.adapters.gradient_ownership import (
    AdapterGradientOwnershipError,
    GradientRepresentation,
    ParameterOwnershipDeclaration,
    ProducerFamily,
    ReductionAuthority,
    ReductionAxis,
    ReductionDomainPlan,
    ReductionOperation,
    TopologyFamily,
    compile_adapter_gradient_ownership,
)
from xorl.server.runner.adapters.sharded_state import AdapterTensorLayout


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_adapter_gradient_ownership_policy(monkeypatch) -> None:
    layer = LoraLinear(2, 2, r=1, lora_alpha=1, device=torch.device("cpu"))
    compiled = torch.compile(layer, backend="eager", fullgraph=True)

    compiled(torch.ones(1, 2)).sum().backward()

    assert layer.adapter_gradient_producer_family == "module_managed"
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None

    _assert_compiler_topology_fingerprint_and_admission_policy()
    with monkeypatch.context() as transport_patch:
        _assert_residual_transport_is_bucketed_and_leaves_raw_accumulator_immutable(transport_patch)


def _layout(
    fqn: str,
    *,
    ep_owned: bool = False,
    replica_count: int = 1,
    reduction: GradientReductionDomain = GradientReductionDomain.NONE,
) -> AdapterTensorLayout:
    return AdapterTensorLayout(
        fqn=fqn,
        dtype=torch.float32,
        rank_dim=0,
        substrate_shape=(2,),
        logical_shape=(2,),
        local_substrate_shape=(2,),
        local_logical_offset=(0,),
        local_logical_shape=(2,),
        active_local_slices=(slice(0, 2),),
        active_storage_shape=(2,),
        replica_count=replica_count,
        replica_ranks=tuple(range(replica_count)),
        replica_key=(fqn, (2,), (0,), (2,), "float32"),
        placement_signature=("local", ep_owned),
        gradient_reduction=reduction,
    )


def _domain(
    axis: ReductionAxis,
    authority: ReductionAuthority,
    *,
    operation: ReductionOperation = ReductionOperation.SUM,
) -> ReductionDomainPlan:
    return ReductionDomainPlan(axis, authority, operation, f"group:{axis.value}")


def _declaration(topology: TopologyFamily) -> ParameterOwnershipDeclaration:
    completed = (_domain(ReductionAxis.FSDP_SHARD, ReductionAuthority.FSDP),)
    pending: tuple[ReductionDomainPlan, ...] = ()
    producer = ProducerFamily.MODULE_MANAGED
    representation = GradientRepresentation.FSDP_COMPLETED_LOCAL_SHARD
    if topology is TopologyFamily.DIRECT_OUTPUT_PROJECTION:
        producer = ProducerFamily.DIRECT_OUTPUT_PROJECTION
    elif topology is TopologyFamily.EP_REPLICATED_SHARED:
        representation = GradientRepresentation.REPLICATED_LOCAL_CONTRIBUTION
        pending = (_domain(ReductionAxis.EXPERT_PARALLEL_REPLICA, ReductionAuthority.ADAPTER_FINALIZER),)
    elif topology is TopologyFamily.OWNER_SHARDED:
        representation = GradientRepresentation.OWNER_LOCAL_CONTRIBUTION
    return ParameterOwnershipDeclaration(
        topology=topology,
        producer=producer,
        representation=representation,
        completed_domains=completed,
        pending_domains=pending,
        config_guard_fingerprint=f"guard:{topology.value}",
        managed_fsdp_shard=True,
    )


def _four_family_inputs():
    families = (
        TopologyFamily.DENSE_REPLICATED,
        TopologyFamily.DIRECT_OUTPUT_PROJECTION,
        TopologyFamily.EP_REPLICATED_SHARED,
        TopologyFamily.OWNER_SHARDED,
    )
    names = {family: f"adapter.{family.value}" for family in families}
    layouts = {
        names[family]: _layout(
            names[family],
            ep_owned=family in {TopologyFamily.EP_REPLICATED_SHARED, TopologyFamily.OWNER_SHARDED},
            replica_count=2 if family is TopologyFamily.EP_REPLICATED_SHARED else 1,
            reduction=(
                GradientReductionDomain.EP_SUM
                if family is TopologyFamily.EP_REPLICATED_SHARED
                else GradientReductionDomain.NONE
            ),
        )
        for family in families
    }
    model_parameters = {name: nn.Parameter(torch.zeros(2)) for name in names.values()}
    optimizer_parameters = {name: nn.Parameter(torch.zeros(2)) for name in names.values()}
    declarations = {names[family]: _declaration(family) for family in families}
    return layouts, model_parameters, optimizer_parameters, declarations


_EP_GROUP_MEMBERSHIPS = {"group:expert_parallel_replica": ((0, 1),)}


def _assert_compiler_topology_fingerprint_and_admission_policy() -> None:
    layouts, model_parameters, optimizer_parameters, declarations = _four_family_inputs()
    plan = compile_adapter_gradient_ownership(
        layouts=layouts,
        model_parameters=model_parameters,
        optimizer_parameters=optimizer_parameters,
        declarations=declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships=_EP_GROUP_MEMBERSHIPS,
    )

    assert {item.topology for item in plan.parameters} == set(TopologyFamily)
    assert len(plan.fingerprint) == 64
    pending_mask = next(
        mask
        for mask in plan.authority_masks
        if mask.axis is ReductionAxis.EXPERT_PARALLEL_REPLICA and mask.authority is ReductionAuthority.ADAPTER_FINALIZER
    )
    assert pending_mask.fqns == ("adapter.ep_replicated_shared",)

    _assert_fingerprint_excludes_rank_local_tensor_identity_and_geometry()
    _assert_compiler_fails_closed_on_incomplete_or_structurally_false_ownership()
    _assert_compiler_replica_coverage_admits_only_complete_orthogonal_domains()


def _assert_fingerprint_excludes_rank_local_tensor_identity_and_geometry() -> None:
    layouts, model_parameters, optimizer_parameters, declarations = _four_family_inputs()
    first = compile_adapter_gradient_ownership(
        layouts=layouts,
        model_parameters=model_parameters,
        optimizer_parameters=optimizer_parameters,
        declarations=declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships=_EP_GROUP_MEMBERSHIPS,
    )
    replacement_model = {name: nn.Parameter(value.detach().clone()) for name, value in model_parameters.items()}
    replacement_optimizer = {name: nn.Parameter(value.detach().clone()) for name, value in optimizer_parameters.items()}
    second = compile_adapter_gradient_ownership(
        layouts=layouts,
        model_parameters=replacement_model,
        optimizer_parameters=replacement_optimizer,
        declarations=declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships=_EP_GROUP_MEMBERSHIPS,
    )
    assert first.fingerprint == second.fingerprint
    shifted = dict(layouts)
    shifted_name = "adapter.owner_sharded"
    shifted[shifted_name] = replace(
        shifted[shifted_name],
        local_logical_offset=(1,),
        local_logical_shape=(1,),
        active_local_slices=(slice(0, 1),),
        active_storage_shape=(1,),
    )
    shifted_optimizer = dict(optimizer_parameters)
    shifted_optimizer[shifted_name] = nn.Parameter(torch.zeros(1))
    second = compile_adapter_gradient_ownership(
        layouts=shifted,
        model_parameters=model_parameters,
        optimizer_parameters=shifted_optimizer,
        declarations=declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships=_EP_GROUP_MEMBERSHIPS,
    )
    assert first.fingerprint == second.fingerprint


def _assert_compiler_fails_closed_on_incomplete_or_structurally_false_ownership() -> None:
    layouts, model_parameters, optimizer_parameters, declarations = _four_family_inputs()
    missing = dict(declarations)
    missing.pop("adapter.dense_replicated")
    with pytest.raises(AdapterGradientOwnershipError, match="one exact universe"):
        compile_adapter_gradient_ownership(
            layouts=layouts,
            model_parameters=model_parameters,
            optimizer_parameters=optimizer_parameters,
            declarations=missing,
            model_generation="model-generation-1",
            adapter_generation="adapter-generation-1",
            group_memberships=_EP_GROUP_MEMBERSHIPS,
        )

    with pytest.raises(AdapterGradientOwnershipError, match="tensor parallelism greater than one"):
        compile_adapter_gradient_ownership(
            layouts=layouts,
            model_parameters=model_parameters,
            optimizer_parameters=optimizer_parameters,
            declarations=declarations,
            model_generation="model-generation-1",
            adapter_generation="adapter-generation-1",
            tensor_parallel_size=2,
            group_memberships=_EP_GROUP_MEMBERSHIPS,
        )

    with pytest.raises(AdapterGradientOwnershipError, match="managed-FSDP ownership"):
        replace(declarations["adapter.dense_replicated"], managed_fsdp_shard=False)

    false_ep = dict(layouts)
    false_ep["adapter.ep_replicated_shared"] = _layout("adapter.ep_replicated_shared")
    with pytest.raises(AdapterGradientOwnershipError, match="pending EP sum"):
        compile_adapter_gradient_ownership(
            layouts=false_ep,
            model_parameters=model_parameters,
            optimizer_parameters=optimizer_parameters,
            declarations=declarations,
            model_generation="model-generation-1",
            adapter_generation="adapter-generation-1",
            group_memberships=_EP_GROUP_MEMBERSHIPS,
        )

    completed = (_domain(ReductionAxis.SEQUENCE_PARALLEL, ReductionAuthority.GENERIC_SP_SYNC),)
    with pytest.raises(AdapterGradientOwnershipError, match="both complete and pending"):
        replace(
            _declaration(TopologyFamily.DENSE_REPLICATED),
            completed_domains=completed,
            pending_domains=(_domain(ReductionAxis.SEQUENCE_PARALLEL, ReductionAuthority.ADAPTER_FINALIZER),),
        )
    with pytest.raises(AdapterGradientOwnershipError, match="owned by adapter finalization"):
        replace(
            _declaration(TopologyFamily.DENSE_REPLICATED),
            pending_domains=(_domain(ReductionAxis.SEQUENCE_PARALLEL, ReductionAuthority.GENERIC_SP_SYNC),),
        )


def _compile_replica_topology(
    *,
    replica_count: int,
    domains: tuple[ReductionDomainPlan, ...],
    group_memberships: dict[str, tuple[tuple[int, ...], ...]],
):
    name = "adapter.replica_topology"
    parameter = nn.Parameter(torch.zeros(2))
    return compile_adapter_gradient_ownership(
        layouts={name: _layout(name, replica_count=replica_count)},
        model_parameters={name: parameter},
        optimizer_parameters={name: nn.Parameter(torch.zeros(2))},
        declarations={
            name: ParameterOwnershipDeclaration(
                topology=TopologyFamily.DENSE_REPLICATED,
                producer=ProducerFamily.MODULE_MANAGED,
                representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
                completed_domains=(),
                pending_domains=domains,
                config_guard_fingerprint="guard:replica-topology",
            )
        },
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships=group_memberships,
    )


def _assert_compiler_replica_coverage_admits_only_complete_orthogonal_domains() -> None:
    with pytest.raises(AdapterGradientOwnershipError, match="do not exactly cover"):
        _compile_replica_topology(replica_count=2, domains=(), group_memberships={})

    sequence = _domain(ReductionAxis.SEQUENCE_PARALLEL, ReductionAuthority.ADAPTER_FINALIZER)
    with pytest.raises(AdapterGradientOwnershipError, match="group identity.*missing"):
        _compile_replica_topology(replica_count=2, domains=(sequence,), group_memberships={})
    with pytest.raises(AdapterGradientOwnershipError, match="outside the replica class"):
        _compile_replica_topology(
            replica_count=2,
            domains=(sequence,),
            group_memberships={"group:sequence_parallel": ((0, 2), (1,))},
        )
    output = _domain(ReductionAxis.OUTPUT_PROJECTION_REPLICA, ReductionAuthority.ADAPTER_FINALIZER)
    with pytest.raises(AdapterGradientOwnershipError, match="overlap beyond rank"):
        _compile_replica_topology(
            replica_count=4,
            domains=(sequence, output),
            group_memberships={
                "group:sequence_parallel": ((0, 1), (2, 3)),
                "group:output_projection_replica": ((0, 1), (2, 3)),
            },
        )
    with pytest.raises(AdapterGradientOwnershipError, match="do not exactly cover"):
        _compile_replica_topology(
            replica_count=4,
            domains=(sequence,),
            group_memberships={"group:sequence_parallel": ((0, 1), (2, 3))},
        )

    two_rank = _compile_replica_topology(
        replica_count=2,
        domains=(sequence,),
        group_memberships={"group:sequence_parallel": ((0, 1),)},
    )
    assert two_rank.parameters[0].norm_replica_divisor == 2

    four_rank = _compile_replica_topology(
        replica_count=4,
        domains=(sequence, output),
        group_memberships={
            "group:sequence_parallel": ((0, 1), (2, 3)),
            "group:output_projection_replica": ((0, 2), (1, 3)),
        },
    )
    assert four_rank.parameters[0].norm_replica_divisor == 4


def _assert_residual_transport_is_bucketed_and_leaves_raw_accumulator_immutable(monkeypatch) -> None:
    name = "adapter.shared_factor"
    layout = _layout(
        name,
        ep_owned=True,
        replica_count=2,
        reduction=GradientReductionDomain.EP_SUM,
    )
    model_parameter = nn.Parameter(torch.zeros(2))
    optimizer_parameter = nn.Parameter(torch.zeros(2))
    declaration = ParameterOwnershipDeclaration(
        topology=TopologyFamily.EP_REPLICATED_SHARED,
        producer=ProducerFamily.FUSED_MANAGED,
        representation=GradientRepresentation.REPLICATED_LOCAL_CONTRIBUTION,
        completed_domains=(),
        pending_domains=(_domain(ReductionAxis.EXPERT_PARALLEL_REPLICA, ReductionAuthority.ADAPTER_FINALIZER),),
        config_guard_fingerprint="guard:fused-shared",
    )
    plan = compile_adapter_gradient_ownership(
        layouts={name: layout},
        model_parameters={name: model_parameter},
        optimizer_parameters={name: optimizer_parameter},
        declarations={name: declaration},
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships=_EP_GROUP_MEMBERSHIPS,
    )
    raw = torch.tensor([1.0, 2.0])
    monkeypatch.setattr(gradient_finalizer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(gradient_finalizer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gradient_finalizer_module.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        gradient_finalizer_module.dist,
        "all_reduce",
        lambda tensor, op, group=None: tensor.mul_(2),
    )

    gradients, stats = transport_complete_local_gradients(
        plan=plan,
        numerators={name: raw},
        templates={name: optimizer_parameter},
        multiplier=1.0,
        resolve_group=lambda _domain: "ep-group",
        bucket_bytes=4,
    )
    norm = logical_l2_norm(plan, gradients, world_group="world")

    torch.testing.assert_close(raw, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(gradients[name], torch.tensor([2.0, 4.0]))
    assert norm.item() == pytest.approx(math.sqrt(20.0))
    assert stats.collective_count == 2
    assert stats.bucket_count == 2
    assert stats.transported_bytes == 8
    assert stats.largest_bucket_bytes == 4
