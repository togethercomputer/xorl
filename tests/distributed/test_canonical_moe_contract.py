from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from distributed_utils import run_distributed_script

from xorl.distributed.canonical_moe import (
    CANONICAL_MOE_CP_SHARDED_TRANSPORT_VERSION,
    CANONICAL_MOE_REDUCE_VERSION,
    CanonicalMoEGraphMetadata,
    CanonicalMoETransport,
    LocalMoEContribution,
    OutputDistribution,
    ParallelPlan,
    ParallelRole,
    canonical_moe_reduce_cp_sharded_v3,
    canonical_moe_reduce_packed_ep16_v2,
    canonical_moe_reduce_reference,
    canonical_moe_reduce_v1,
)
from xorl.distributed.parallel_state import init_ep_mesh_matrix


pytestmark = [pytest.mark.distributed]


def _explicit_tree(partials: torch.Tensor) -> torch.Tensor:
    current = [partials[index] for index in range(partials.shape[0])]
    while len(current) > 1:
        current = [(current[index] + current[index + 1]).bfloat16() for index in range(0, len(current), 2)]
    return current[0]


@pytest.mark.cpu
@pytest.mark.parametrize("contributors", [2, 4, 8, 16])
def test_reference_is_the_adjacent_bf16_tree(contributors: int):
    rows = contributors + 2
    values = torch.zeros((contributors, rows, 3), dtype=torch.bfloat16)
    adversarial = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 2,
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
    trainer_pp1_world16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=2)
    trainer_pp1_world32 = ParallelPlan.glm52_trainer(world_size=32, pp_size=1, dp_size=4)
    trainer_ep16_world32 = ParallelPlan.glm52_trainer(
        world_size=32,
        pp_size=1,
        dp_size=2,
        contributor_count=16,
    )
    trainer_ep16_world16 = ParallelPlan.glm52_trainer(
        world_size=16,
        pp_size=1,
        dp_size=1,
        contributor_count=16,
    )
    sampler = ParallelPlan.glm52_sampler(launcher_tp_size=8)
    assert trainer.digest != sampler.digest
    assert (
        len(
            {
                trainer.digest,
                trainer_pp1_world16.digest,
                trainer_pp1_world32.digest,
                trainer_ep16_world16.digest,
                trainer_ep16_world32.digest,
                sampler.digest,
            }
        )
        == 6
    )
    assert sampler.as_dict()["launcher_tp_size"] == 8
    assert trainer.pipeline_layer_ranges == ((0, 38), (38, 78))
    assert trainer_pp1_world16.pipeline_layer_ranges == ((0, 78),)
    assert trainer_pp1_world16.combine_groups == (tuple(range(8)), tuple(range(8, 16)))
    assert trainer_pp1_world32.pipeline_layer_ranges == ((0, 78),)
    assert trainer_pp1_world32.combine_groups == tuple(
        tuple(range(group_start, group_start + 8)) for group_start in range(0, 32, 8)
    )
    assert all(trainer_pp1_world32.logical_ordinal(physical_rank) == physical_rank % 8 for physical_rank in range(32))
    assert trainer_ep16_world32.combine_groups == (tuple(range(16)), tuple(range(16, 32)))
    assert all(trainer_ep16_world32.logical_ordinal(physical_rank) == physical_rank % 16 for physical_rank in range(32))
    assert trainer_ep16_world16.combine_groups == (tuple(range(16)),)
    assert trainer_ep16_world16.pipeline_layer_ranges == ((0, 78),)
    assert all(trainer_ep16_world16.logical_ordinal(physical_rank) == physical_rank for physical_rank in range(16))
    assert trainer.contract_version == CANONICAL_MOE_REDUCE_VERSION

    with pytest.raises(ValueError, match="Unsupported GLM-5.2 trainer topology"):
        ParallelPlan.glm52_trainer(world_size=32, pp_size=2, dp_size=2)

    payload = sampler.as_dict()
    payload["world_size"] = 16
    payload["role"] = sampler.role
    with pytest.raises(ValueError, match="partition|world|sampler topology"):
        ParallelPlan(**payload)

    payload = sampler.as_dict()
    payload["role"] = ParallelRole.SAMPLER
    payload["launcher_tp_size"] = 99
    with pytest.raises(ValueError, match="launcher-level tp_size must be exactly 8"):
        ParallelPlan(**payload)

    payload = sampler.as_dict()
    payload["role"] = ParallelRole.SAMPLER
    payload["logical_ordinals_by_group"] = (tuple(reversed(range(8))),)
    with pytest.raises(ValueError, match="identity logical contributor ordinals"):
        ParallelPlan(**payload)


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
        engagement_key="distributed_replicated",
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
        engagement_key="distributed_permuted",
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
            engagement_key=f"distributed_solo_row_{row}",
        )
        assert torch.equal(solo_row.tensor[0], replicated.tensor[row])

    solo = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=capacity,
        engagement_key="distributed_solo_chunk",
    )
    assert torch.equal(solo.tensor, replicated.tensor)

    owner_sharded = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        chunk_rows=3,
        engagement_key="distributed_owner_sharded",
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
        engagement_key=f"dense_{capacity}_{payload}",
    )
    assert dense.receipt.transport is CanonicalMoETransport.DENSE_V1

    for repeat in range(3):
        packed = canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
            chunk_rows=requested_chunk_rows,
            engagement_key=f"packed_{capacity}_{payload}_{repeat}",
        )
        assert packed.receipt.transport is CanonicalMoETransport.PACKED_EP16_V2
        assert packed.receipt.chunk_rows == capacity
        assert torch.equal(packed.tensor.view(torch.int16), dense.tensor.view(torch.int16))
        assert torch.count_nonzero(packed.tensor[~metadata.valid_mask]) == 0

    dense_owner = canonical_moe_reduce_v1(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        engagement_key=f"dense_owner_{capacity}_{payload}",
    )
    packed_owner = canonical_moe_reduce_packed_ep16_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        chunk_rows=requested_chunk_rows,
        engagement_key=f"packed_owner_{capacity}_{payload}",
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
        engagement_key=f"dense_v3_oracle_{capacity}_{payload}",
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
            engagement_key=f"cp_sharded_v3_{capacity}_{payload}_{repeat}",
        )
        assert sharded.receipt.contract_version == CANONICAL_MOE_CP_SHARDED_TRANSPORT_VERSION
        assert sharded.receipt.transport is CanonicalMoETransport.CP_SHARDED_V3
        assert sharded.receipt.source_capacity == capacity
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
