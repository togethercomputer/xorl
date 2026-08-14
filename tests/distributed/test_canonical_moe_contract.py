from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from distributed_utils import run_distributed_script

from xorl.distributed.canonical_moe import (
    CANONICAL_MOE_DENSE_MAX_CHUNK_ROWS,
    CANONICAL_MOE_FOLD_VERSION,
    CANONICAL_MOE_REDUCE_VERSION,
    CanonicalMoEGraphMetadata,
    CanonicalMoETransport,
    LocalMoEContribution,
    LogicalRowOwnership,
    OutputDistribution,
    ParallelPlan,
    ParallelRole,
    _resolve_transport_chunk_rows,
    canonical_moe_fold_v1,
    canonical_moe_reduce_cp_sharded_v3,
    canonical_moe_reduce_packed_ep16_v2,
    canonical_moe_reduce_reference,
    canonical_moe_reduce_v1,
    resolve_canonical_moe_transport,
)
from xorl.distributed.parallel_state import init_ep_mesh_matrix


pytestmark = [pytest.mark.distributed]


@pytest.mark.cpu
def test_dense_transport_default_bounds_dp_owned_capacity_without_a_selector():
    assert CANONICAL_MOE_DENSE_MAX_CHUNK_ROWS == 4096
    assert _resolve_transport_chunk_rows(66544, None, CanonicalMoETransport.DENSE_V1) == 4096
    assert _resolve_transport_chunk_rows(128, None, CanonicalMoETransport.DENSE_V1) == 128
    assert _resolve_transport_chunk_rows(66544, 2048, CanonicalMoETransport.DENSE_V1) == 2048
    assert _resolve_transport_chunk_rows(66544, None, CanonicalMoETransport.PACKED_EP16_V2) == 66544


def _explicit_tree(partials: torch.Tensor) -> torch.Tensor:
    current = [partials[index] for index in range(partials.shape[0])]
    while len(current) > 1:
        current = [(current[index] + current[index + 1]).bfloat16() for index in range(0, len(current), 2)]
    return current[0]


@pytest.mark.cpu
@pytest.mark.parametrize("contributors", [2, 4, 8, 16, 32])
def test_reference_is_the_adjacent_bf16_tree(contributors: int):
    assert CANONICAL_MOE_FOLD_VERSION == "canonical_moe_fold_v1"
    rows = contributors + 2
    values = torch.zeros((contributors, rows, 3), dtype=torch.bfloat16)
    adversarial = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 4,
        dtype=torch.bfloat16,
    )
    for ordinal in range(contributors):
        values[ordinal, :, 0] = adversarial[ordinal]
        values[ordinal, :, 1] = ordinal + 1
        values[ordinal, :, 2] = torch.arange(rows)
    metadata = CanonicalMoEGraphMetadata.build(
        torch.arange(rows),
        torch.arange(rows),
        capacity=rows + 3,
    )
    padded = torch.zeros((contributors, metadata.capacity, 3), dtype=torch.bfloat16)
    padded[:, :rows] = values

    result = canonical_moe_reduce_reference(padded, metadata)
    expected = _explicit_tree(padded)
    expected[~metadata.valid_mask] = 0
    assert torch.equal(result, expected)
    assert torch.equal(canonical_moe_fold_v1(padded), _explicit_tree(padded))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("contributors", [8, 16])
def test_shared_fold_replays_in_cuda_graph(contributors: int):
    partials = torch.randn((contributors, 64, 32), device="cuda", dtype=torch.bfloat16)
    canonical_moe_fold_v1(partials)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        folded = canonical_moe_fold_v1(partials)
    first = folded.clone()
    partials[0].add_(8.0)
    graph.replay()

    assert not torch.equal(first, folded)
    assert torch.equal(folded, _explicit_tree(partials))


@pytest.mark.cpu
def test_graph_metadata_has_deterministic_padding_and_capacity_guard():
    metadata = CanonicalMoEGraphMetadata.build(
        torch.tensor([7, 3, 11], dtype=torch.int64),
        torch.tensor([17, 2, 25], dtype=torch.int64),
        capacity=8,
    )
    assert metadata.logical_row_ids.tolist() == [7, 3, 11, -1, -1, -1, -1, -1]
    assert metadata.absolute_positions.tolist() == [17, 2, 25, -1, -1, -1, -1, -1]
    assert metadata.valid_mask.tolist() == [True, True, True, False, False, False, False, False]
    with pytest.raises(ValueError, match="exceeds fixed capacity"):
        CanonicalMoEGraphMetadata.build(torch.arange(3), torch.arange(3), capacity=2)


@pytest.mark.cpu
def test_transport_auto_selects_only_regression_qualified_packed_geometry():
    ep16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=1, contributor_count=16)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.PACKED_EP16_V2
    )

    dp_ep16 = ParallelPlan.glm52_trainer(
        world_size=16,
        pp_size=1,
        dp_size=16,
        contributor_count=16,
        cp_size=1,
    )
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=dp_ep16,
            capacity=16 * 4224,
            local_rows=4224,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )

    ep8 = ParallelPlan.primitive(8)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep8,
            capacity=4224,
            local_rows=528,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=True,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )


@pytest.mark.cpu
def test_internal_resolution_serves_packed_ep16_v2_on_the_admitted_geometry():
    """The exact GLM path has no public transport knob: internal resolution
    serves packed_ep16_v2 on the admitted eager EP16/CP16 geometry because it
    passed the full-model byte-parity regression anchor, and the dense
    executable oracle elsewhere."""
    ep16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=1, contributor_count=16)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.PACKED_EP16_V2
    )
    ep8 = ParallelPlan.primitive(8)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep8,
            capacity=4224,
            local_rows=528,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )


@pytest.mark.cpu
def test_transport_explicit_modes_never_silently_fallback():
    ep16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=1, contributor_count=16)
    assert (
        resolve_canonical_moe_transport(
            "dense_v1",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )
    with pytest.raises(ValueError, match="eager execution"):
        resolve_canonical_moe_transport(
            "cp_sharded_v3",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=True,
            consumer_sharded_output=True,
        )
    with pytest.raises(ValueError, match="consumer-sharded output"):
        resolve_canonical_moe_transport(
            "cp_sharded_v3",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=False,
        )
    with pytest.raises(ValueError, match="EP16/CP16"):
        resolve_canonical_moe_transport(
            "packed_ep16_v2",
            plan=ParallelPlan.primitive(8),
            capacity=4224,
            local_rows=528,
            graph_mode=False,
            consumer_sharded_output=False,
        )


@pytest.mark.cpu
def test_packed_ep16_v2_fails_closed_outside_admitted_mode():
    metadata = CanonicalMoEGraphMetadata.build(torch.arange(16), torch.arange(16), capacity=16)
    contribution = LocalMoEContribution(
        torch.zeros((16, 2), dtype=torch.bfloat16),
        metadata,
        "test:packed_ep16:guards",
    )
    with pytest.raises(ValueError, match="EP16/CP16"):
        canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=ParallelPlan.primitive(8),
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )

    with pytest.raises(ValueError, match="EP16/CP16"):
        canonical_moe_reduce_cp_sharded_v3(
            contribution,
            plan=ParallelPlan.primitive(8),
            group=dist.group.WORLD,
        )
    with pytest.raises(ValueError, match="does not yet admit CUDA graph"):
        canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=ParallelPlan.primitive(16),
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
            graph_mode=True,
        )


@pytest.mark.cpu
def test_exact_parallel_plans_hash_launcher_spelling_and_fail_closed():
    trainer = ParallelPlan.glm52_trainer()
    sampler = ParallelPlan.glm52_sampler(launcher_tp_size=8)
    assert trainer.digest != sampler.digest
    assert sampler.as_dict()["launcher_tp_size"] == 8
    assert trainer.pipeline_layer_ranges == ((0, 78),)
    assert trainer.combine_groups == (tuple(range(16)),)
    assert all(trainer.logical_ordinal(physical_rank) == physical_rank for physical_rank in range(16))
    assert trainer.contract_version == CANONICAL_MOE_REDUCE_VERSION

    dp_trainer = ParallelPlan.glm52_trainer(dp_size=16, contributor_count=16, cp_size=1)
    assert dp_trainer.contributor_count == 16
    assert dp_trainer.cp_size == 1
    assert dp_trainer.ep_size == 16
    assert dp_trainer.combine_groups == trainer.combine_groups
    assert dp_trainer.logical_ordinals_by_group == trainer.logical_ordinals_by_group
    assert dp_trainer.cp_ep_aliases == ()
    assert dp_trainer.digest != trainer.digest

    for dp_size, cp_size in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
        plan = ParallelPlan.glm52_trainer(dp_size=dp_size, cp_size=cp_size)
        assert plan.dp_size * plan.cp_size == plan.ep_size == 16

    with pytest.raises(ValueError, match="exactly cover"):
        ParallelPlan.glm52_trainer(dp_size=2, cp_size=4)
    with pytest.raises(ValueError, match="world size"):
        ParallelPlan.glm52_trainer(world_size=32, dp_size=2, cp_size=8)
    with pytest.raises(ValueError, match="explicit model-derived pipeline ranges"):
        ParallelPlan.glm52_trainer(pp_size=2, dp_size=2, cp_size=8)

    payload = sampler.as_dict()
    payload["world_size"] = 16
    payload["role"] = sampler.role
    with pytest.raises(ValueError, match="partition|world|sampler topology"):
        ParallelPlan(**payload)

    payload = sampler.as_dict()
    payload["role"] = ParallelRole.SAMPLER
    payload["launcher_tp_size"] = 99
    with pytest.raises(ValueError, match="launcher TP"):
        ParallelPlan(**payload)

    payload = sampler.as_dict()
    payload["role"] = ParallelRole.SAMPLER
    payload["logical_ordinals_by_group"] = (tuple(reversed(range(8))),)
    with pytest.raises(ValueError, match="identity logical contributor ordinals"):
        ParallelPlan(**payload)


@pytest.mark.cpu
def test_trainer_identity_is_descriptive_and_runtime_invariants_are_structural():
    trainer = ParallelPlan.glm52_trainer()
    assert trainer.family == "glm52"
    renamed = replace(trainer, family="model-defined-exact-moe")
    assert renamed.family == "model-defined-exact-moe"
    assert renamed.digest != trainer.digest

    with pytest.raises(ValueError, match="model metadata exactly"):
        replace(trainer, model_num_layers=80)

    primitive = ParallelPlan.primitive(8)
    qwen_primitive = replace(primitive, family="qwen3_5_moe")
    assert primitive.digest != qwen_primitive.digest
    payload = qwen_primitive.as_dict()
    round_tripped = ParallelPlan(**payload)
    assert round_tripped.family == "qwen3_5_moe"
    assert round_tripped.digest == qwen_primitive.digest
    assert ParallelPlan(**trainer.as_dict()).digest == trainer.digest
    assert ParallelPlan(**ParallelPlan.glm52_sampler(launcher_tp_size=8).as_dict()).role is ParallelRole.SAMPLER


@pytest.mark.cpu
@pytest.mark.parametrize("dp_size,cp_size", [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)])
def test_logical_row_ownership_covers_every_mixed_factorization(dp_size: int, cp_size: int):
    seen = []
    for dp_rank in range(dp_size):
        expected_context = tuple(range(dp_rank * cp_size, (dp_rank + 1) * cp_size))
        for cp_rank in range(cp_size):
            ownership = LogicalRowOwnership(dp_size, cp_size, dp_rank, cp_rank, 16)
            seen.append(ownership.source_ordinal)
            assert ownership.context_source_ordinals == expected_context
            source = ownership.source_slice(padded_rows=7, local_rows=3)
            assert source == slice(ownership.source_ordinal * 7, ownership.source_ordinal * 7 + 3)
    assert seen == list(range(16))


@pytest.mark.cpu
def test_logical_row_ownership_rejects_only_concrete_shape_mismatch():
    with pytest.raises(ValueError, match="exactly cover"):
        LogicalRowOwnership(2, 4, 0, 0, 16)
    with pytest.raises(ValueError, match="DP rank"):
        LogicalRowOwnership(2, 8, 2, 0, 16)
    assert LogicalRowOwnership.valid_positions(torch.tensor([0, -1, 9])).tolist() == [True, False, True]


@pytest.mark.cpu
def test_world32_pp1_cp_and_ep_groups_alias_as_four_rank_octets():
    main_mesh = torch.arange(32).view(4, 8)
    ep_mesh = init_ep_mesh_matrix(ep_size=8, ep_fsdp_size=4, ep_intranode=True)

    cp_groups = tuple(tuple(int(rank) for rank in row) for row in main_mesh)
    ep_groups = tuple(tuple(int(rank) for rank in ep_mesh[:, column]) for column in range(4))
    expert_fsdp_groups = tuple(tuple(int(rank) for rank in row) for row in ep_mesh)

    expected_octets = tuple(tuple(range(start, start + 8)) for start in range(0, 32, 8))
    assert cp_groups == ep_groups == expected_octets
    assert expert_fsdp_groups[0] == (0, 8, 16, 24)
    assert expert_fsdp_groups[-1] == (7, 15, 23, 31)


@pytest.mark.cpu
def test_world32_pp1_cp16_and_ep16_groups_alias_as_two_rank_groups():
    main_mesh = torch.arange(32).view(2, 16)
    ep_mesh = init_ep_mesh_matrix(ep_size=16, ep_fsdp_size=2, ep_intranode=True)

    cp_groups = tuple(tuple(int(rank) for rank in row) for row in main_mesh)
    ep_groups = tuple(tuple(int(rank) for rank in ep_mesh[:, column]) for column in range(2))
    expert_fsdp_groups = tuple(tuple(int(rank) for rank in row) for row in ep_mesh)

    expected_groups = (tuple(range(16)), tuple(range(16, 32)))
    assert cp_groups == ep_groups == expected_groups
    assert expert_fsdp_groups[0] == (0, 16)
    assert expert_fsdp_groups[-1] == (15, 31)


def _make_partials(world: int, capacity: int) -> torch.Tensor:
    rank = dist.get_rank()
    values = torch.empty((capacity, 4), dtype=torch.bfloat16)
    values[:, 0] = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 2,
        dtype=torch.bfloat16,
    )[rank]
    values[:, 1] = rank + 1
    values[:, 2] = torch.arange(capacity)
    values[:, 3] = torch.arange(capacity) * (rank + 1)
    return values


def _run_distributed_case() -> None:
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    physical_rank = dist.get_rank()
    mapping = tuple(reversed(range(world)))
    plan = ParallelPlan.primitive(world, physical_to_logical=mapping)

    capacity = world + 5
    valid_rows = world + 2
    positions = torch.tensor([(index * 3 + 1) % 29 for index in range(valid_rows)], dtype=torch.int64)
    row_ids = torch.tensor(list(reversed(range(valid_rows))), dtype=torch.int64)
    metadata = CanonicalMoEGraphMetadata.build(row_ids, positions, capacity=capacity)
    local = _make_partials(world, capacity).requires_grad_(True)
    contribution = LocalMoEContribution(
        local,
        metadata,
        local_partial_policy="test:routed_then_shared:bf16",
    )

    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.detach())
    physical_stack = torch.stack(gathered)
    logical_to_physical = [mapping.index(logical) for logical in range(world)]
    logical_stack = physical_stack[logical_to_physical]
    expected = canonical_moe_reduce_reference(logical_stack, metadata)

    replicated = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=2,
    )
    assert torch.equal(replicated.tensor, expected)
    assert torch.count_nonzero(replicated.tensor[~metadata.valid_mask]) == 0

    permutation = torch.tensor(list(reversed(range(capacity))), dtype=torch.long)
    permuted_metadata = CanonicalMoEGraphMetadata(
        logical_row_ids=metadata.logical_row_ids.index_select(0, permutation),
        absolute_positions=metadata.absolute_positions.index_select(0, permutation),
        valid_mask=metadata.valid_mask.index_select(0, permutation),
        capacity=capacity,
        valid_rows=valid_rows,
    )
    permuted = canonical_moe_reduce_v1(
        LocalMoEContribution(
            local.index_select(0, permutation),
            permuted_metadata,
            local_partial_policy="test:routed_then_shared:bf16",
        ),
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=2,
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(permuted.tensor.index_select(0, inverse), replicated.tensor)

    for row in range(valid_rows):
        solo_metadata = CanonicalMoEGraphMetadata.build(
            metadata.logical_row_ids[row : row + 1],
            metadata.absolute_positions[row : row + 1],
            capacity=1,
        )
        solo_row = canonical_moe_reduce_v1(
            LocalMoEContribution(
                local[row : row + 1],
                solo_metadata,
                local_partial_policy="test:routed_then_shared:bf16",
            ),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        assert torch.equal(solo_row.tensor[0], replicated.tensor[row])

    solo = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=capacity,
    )
    assert torch.equal(solo.tensor, replicated.tensor)

    owner_sharded = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        chunk_rows=3,
    )
    owners = metadata.owner_ordinals(world)
    local_logical = mapping[physical_rank]
    expected_owner_mask = metadata.valid_mask & (owners == local_logical)
    assert torch.equal(owner_sharded.owner_mask, expected_owner_mask)
    assert torch.equal(owner_sharded.tensor[expected_owner_mask], expected[expected_owner_mask])
    assert torch.count_nonzero(owner_sharded.tensor[~expected_owner_mask]) == 0

    replicated.tensor.float().sum().backward()
    assert local.grad is not None
    assert bool(torch.all(torch.isfinite(local.grad)))
    assert bool(torch.all(local.grad[metadata.valid_mask] != 0))

    with pytest.raises(TypeError, match="already reduced"):
        canonical_moe_reduce_v1(
            replicated,
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )

    dist.barrier()
    dist.destroy_process_group()


def _packed_ep16_local_partial(capacity: int, payload: int) -> torch.Tensor:
    rank = dist.get_rank()
    rows = torch.arange(capacity, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(payload, dtype=torch.float32).unsqueeze(0)
    signs = -1.0 if rank % 4 == 1 else 1.0
    values = signs * (rows.remainder(29) - 14.0) * (rank + 1) / 32.0 + columns / 128.0
    values[:, 0] = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 2,
        dtype=torch.float32,
    )[rank]
    values[min(17, capacity - 1)] = 0.0  # A valid row with no routed/shared contribution.
    return values.to(torch.bfloat16)


def _assert_packed_ep16_matches_v1(
    *,
    capacity: int,
    valid_rows: int,
    payload: int,
    padded_reset_tail_rows: int = 0,
    requested_chunk_rows: int | None = None,
) -> None:
    plan = ParallelPlan.primitive(16)
    if padded_reset_tail_rows:
        if valid_rows != capacity or capacity <= padded_reset_tail_rows:
            raise ValueError("padded-reset test geometry requires all capacity rows valid and a nonempty prefix")
        positions = torch.cat(
            (
                torch.arange(capacity - padded_reset_tail_rows, dtype=torch.int64),
                torch.zeros(padded_reset_tail_rows, dtype=torch.int64),
            ),
        )
        owner_counts = torch.bincount(positions.remainder(16), minlength=16)
        assert int(owner_counts.max()) > (capacity + 15) // 16
    else:
        positions = torch.arange(valid_rows, dtype=torch.int64)
    if positions.numel() == capacity:
        metadata = CanonicalMoEGraphMetadata(
            logical_row_ids=torch.arange(capacity, dtype=torch.int64),
            absolute_positions=positions,
            valid_mask=torch.ones(capacity, dtype=torch.bool),
            capacity=capacity,
            valid_rows=capacity,
        )
    else:
        metadata = CanonicalMoEGraphMetadata.build(
            torch.arange(valid_rows, dtype=torch.int64),
            positions,
            capacity=capacity,
        )
    local = _packed_ep16_local_partial(capacity, payload)
    # Invalid tail bytes must never escape either transport.
    local[valid_rows:] = torch.arange(capacity - valid_rows, dtype=torch.float32).unsqueeze(1).add(1).bfloat16()
    contribution = LocalMoEContribution(local, metadata, "test:packed_ep16:adversarial_bf16")

    dense = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
    )

    for repeat in range(3):
        packed = canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
            chunk_rows=requested_chunk_rows,
        )
        assert torch.equal(packed.tensor.view(torch.int16), dense.tensor.view(torch.int16))
        assert torch.count_nonzero(packed.tensor[~metadata.valid_mask]) == 0

    dense_owner = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
    )
    packed_owner = canonical_moe_reduce_packed_ep16_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        chunk_rows=requested_chunk_rows,
    )
    assert torch.equal(packed_owner.tensor.view(torch.int16), dense_owner.tensor.view(torch.int16))
    assert torch.equal(packed_owner.owner_mask, dense_owner.owner_mask)

    if capacity == 35:
        dense_leaf = local.clone().requires_grad_(True)
        packed_leaf = local.clone().requires_grad_(True)
        dense_for_backward = canonical_moe_reduce_v1(
            LocalMoEContribution(dense_leaf, metadata, "test:packed_ep16:backward"),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        packed_for_backward = canonical_moe_reduce_packed_ep16_v2(
            LocalMoEContribution(packed_leaf, metadata, "test:packed_ep16:backward"),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        dense_for_backward.tensor.float().sum().backward()
        packed_for_backward.tensor.float().sum().backward()
        assert dense_leaf.grad is not None and packed_leaf.grad is not None
        assert torch.equal(packed_leaf.grad.view(torch.int16), dense_leaf.grad.view(torch.int16))


def _assert_cp_sharded_v3_matches_v1(*, capacity: int, valid_rows: int, payload: int) -> None:
    if capacity % 16:
        raise ValueError("v3 test requires equal padded CP-source capacity")
    plan = ParallelPlan.primitive(16)
    metadata = CanonicalMoEGraphMetadata.build(
        torch.arange(valid_rows, dtype=torch.int64),
        torch.arange(valid_rows, dtype=torch.int64),
        capacity=capacity,
    )
    local = _packed_ep16_local_partial(capacity, payload)
    local[valid_rows:] = torch.arange(capacity - valid_rows, dtype=torch.float32).unsqueeze(1).add(1).bfloat16()
    contribution = LocalMoEContribution(local, metadata, "test:cp_sharded_v3:adversarial_bf16")
    dense = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
    )
    local_capacity = capacity // 16
    rank = dist.get_rank()
    start = rank * local_capacity
    end = start + local_capacity

    for repeat in range(3):
        sharded = canonical_moe_reduce_cp_sharded_v3(
            contribution,
            plan=plan,
            group=dist.group.WORLD,
        )
        assert sharded.distribution is OutputDistribution.CONSUMER_SHARDED
        assert torch.equal(sharded.tensor.view(torch.int16), dense.tensor[start:end].view(torch.int16))

    if capacity == 32:
        dense_leaf = local.clone().requires_grad_(True)
        v3_leaf = local.clone().requires_grad_(True)
        dense_for_backward = canonical_moe_reduce_v1(
            LocalMoEContribution(dense_leaf, metadata, "test:cp_sharded_v3:backward"),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        v3_for_backward = canonical_moe_reduce_cp_sharded_v3(
            LocalMoEContribution(v3_leaf, metadata, "test:cp_sharded_v3:backward"),
            plan=plan,
            group=dist.group.WORLD,
        )
        dense_for_backward.tensor[start:end].float().sum().backward()
        v3_for_backward.tensor.float().sum().backward()
        assert dense_leaf.grad is not None and v3_leaf.grad is not None
        assert torch.equal(v3_leaf.grad.view(torch.int16), dense_leaf.grad.view(torch.int16))


def _run_packed_ep16_case() -> None:
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 16

    # Compact adversarial arithmetic plus both production row geometries used
    # by the 4,099-token trace: a 13-row CP tail and the server's 125-row
    # pad-to-128 tail. Padding resets position IDs to zero, so neither owner
    # histogram fits ceil(capacity / 16).
    _assert_packed_ep16_matches_v1(capacity=35, valid_rows=32, payload=7)
    _assert_packed_ep16_matches_v1(
        capacity=4112,
        valid_rows=4112,
        payload=4,
        padded_reset_tail_rows=13,
        requested_chunk_rows=128,
    )
    _assert_packed_ep16_matches_v1(
        capacity=4224,
        valid_rows=4224,
        payload=4,
        padded_reset_tail_rows=125,
        requested_chunk_rows=128,
    )
    _assert_cp_sharded_v3_matches_v1(capacity=32, valid_rows=29, payload=7)
    _assert_cp_sharded_v3_matches_v1(capacity=4112, valid_rows=4099, payload=4)
    _assert_cp_sharded_v3_matches_v1(capacity=4224, valid_rows=4099, payload=4)

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":

    @pytest.mark.cpu
    @pytest.mark.parametrize("contributors", [2, 4, 8])
    def test_distributed_transport_tree_distribution_chunking_and_backward(contributors: int):
        result = run_distributed_script(__file__, num_gpus=contributors, timeout=180)
        result.assert_success(f"canonical MoE primitive with {contributors} CPU contributors")

    @pytest.mark.cpu
    def test_packed_ep16_v2_matches_dense_v1_bitwise():
        result = run_distributed_script(
            __file__,
            num_gpus=16,
            timeout=240,
            extra_env={"CANONICAL_MOE_TEST_MODE": "packed_ep16"},
        )
        result.assert_success("packed EP16 transport must equal dense v1 byte-for-byte")


if __name__ == "__main__":
    if os.environ.get("CANONICAL_MOE_TEST_MODE") == "packed_ep16":
        _run_packed_ep16_case()
    else:
        _run_distributed_case()
