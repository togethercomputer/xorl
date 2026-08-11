"""Versioned canonical reduction for additive MoE rank contributions.

The forward contract in this module deliberately separates transport from
floating-point arithmetic. Raw BF16 partials travel to one logical row owner,
which evaluates an explicit adjacent-pairwise tree in logical-contributor
order. Any later replication copies the completed owner bytes.

The canonical arithmetic here is a DELIBERATE independent implementation of
the serving side's canonical MoE fold (a shared implementation must not
self-confirm the trainer/sampler pair). The two implementations are
K3-guarded: any divergence between them reads as nonzero behavior_k3 in the
first training steps and in the qualification replays.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import torch
import torch.distributed as dist


CANONICAL_MOE_REDUCE_VERSION = "canonical_moe_reduce_v1"
CANONICAL_MOE_PACKED_EP16_TRANSPORT_VERSION = "packed_ep16_v2"
CANONICAL_MOE_CP_SHARDED_TRANSPORT_VERSION = "cp_sharded_v3"
GLM52_NUM_LAYERS = 78


class ParallelRole(str, Enum):
    TRAINER = "trainer"
    PRIMITIVE_TEST = "primitive_test"


class ContributionLayout(str, Enum):
    LOCAL_PARTIAL = "local_partial"


class OutputDistribution(str, Enum):
    OWNER_SHARDED = "owner_sharded"
    CONSUMER_SHARDED = "consumer_sharded"
    REPLICATED_CANONICAL = "replicated_canonical"


class CanonicalMoETransport(str, Enum):
    """Byte-transport implementation under the version-1 arithmetic contract."""

    AUTO = "auto"
    DENSE_V1 = "dense_v1"
    PACKED_EP16_V2 = CANONICAL_MOE_PACKED_EP16_TRANSPORT_VERSION
    CP_SHARDED_V3 = CANONICAL_MOE_CP_SHARDED_TRANSPORT_VERSION


def _cp_sharded_v3_incompatibilities(
    *,
    plan: ParallelPlan,
    capacity: int,
    local_rows: int,
    graph_mode: bool,
    consumer_sharded_output: bool,
) -> tuple[str, ...]:
    reasons = []
    if plan.role is not ParallelRole.TRAINER:
        reasons.append("trainer role")
    if plan.cp_size != 16 or plan.ep_size != 16:
        reasons.append("EP16/CP16")
    if graph_mode:
        reasons.append("eager execution")
    if not consumer_sharded_output:
        reasons.append("consumer-sharded output")
    if capacity <= 0 or capacity % 16:
        reasons.append("capacity divisible by 16")
    elif local_rows < 0 or local_rows > capacity // 16:
        reasons.append("local rows within the CP-owned capacity")
    identity = tuple(range(plan.contributor_count))
    if any(ordinals != identity for ordinals in plan.logical_ordinals_by_group):
        reasons.append("identity contributor ordering")
    return tuple(reasons)


def resolve_canonical_moe_transport(
    requested: CanonicalMoETransport | str,
    *,
    plan: ParallelPlan,
    capacity: int,
    local_rows: int,
    graph_mode: bool,
    consumer_sharded_output: bool,
) -> CanonicalMoETransport:
    """Resolve the public transport mode against the actual execution geometry.

    ``auto`` promotes only the fully admitted consumer-sharded EP16/CP16
    trainer shape. Every other admitted canonical shape retains the dense
    executable oracle. Explicit optimized modes never silently fall back.

    The exact GLM-5.2 path resolves internally (there is no user-facing
    transport knob): the admitted eager EP16/CP16 consumer-sharded geometry
    serves ``cp_sharded_v3`` — the transport with the strongest certified
    performance evidence (a direct 64/64 zero-K3 replay and the faster
    measured trainer forward) — and every other admitted canonical shape
    retains the dense executable oracle.
    """

    try:
        mode = requested if isinstance(requested, CanonicalMoETransport) else CanonicalMoETransport(requested)
    except ValueError as error:
        raise ValueError(f"Unsupported canonical MoE transport: {requested!r}") from error

    cp_v3_reasons = _cp_sharded_v3_incompatibilities(
        plan=plan,
        capacity=capacity,
        local_rows=local_rows,
        graph_mode=graph_mode,
        consumer_sharded_output=consumer_sharded_output,
    )
    if mode is CanonicalMoETransport.AUTO:
        return CanonicalMoETransport.CP_SHARDED_V3 if not cp_v3_reasons else CanonicalMoETransport.DENSE_V1
    if mode is CanonicalMoETransport.CP_SHARDED_V3 and cp_v3_reasons:
        raise ValueError("cp_sharded_v3 requires " + ", ".join(cp_v3_reasons))
    if mode is CanonicalMoETransport.PACKED_EP16_V2:
        reasons = []
        if plan.cp_size != 16 or plan.ep_size != 16:
            reasons.append("EP16/CP16")
        if graph_mode:
            reasons.append("eager execution")
        if reasons:
            raise ValueError("packed_ep16_v2 requires " + ", ".join(reasons))
    return mode


class GraphContractStatus(IntEnum):
    OK = 0
    CAPACITY_OVERFLOW = 1
    MISSING_LOGICAL_POSITION = 2
    INVALID_PADDING = 3
    TOPOLOGY_VERSION_MISMATCH = 4


@dataclass(frozen=True)
class ParallelPlan:
    """Exact immutable topology and logical contributor order.

    ``logical_ordinals_by_group`` is parallel to ``combine_groups``. Each
    entry maps the corresponding physical-rank position in that group to the
    logical contributor ordinal used by the arithmetic tree.
    """

    role: ParallelRole
    world_size: int
    pp_size: int
    tp_size: int
    dp_size: int
    cp_size: int
    ep_size: int
    effective_dense_tp: int
    combine_groups: tuple[tuple[int, ...], ...]
    logical_ordinals_by_group: tuple[tuple[int, ...], ...]
    pipeline_layer_ranges: tuple[tuple[int, int], ...]
    cp_ep_aliases: tuple[tuple[int, int], ...]
    contract_version: str = CANONICAL_MOE_REDUCE_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contributor_count(self) -> int:
        return self.cp_size

    def validate(self) -> None:
        if self.contract_version != CANONICAL_MOE_REDUCE_VERSION:
            raise ValueError(f"Unsupported canonical MoE contract version: {self.contract_version}")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.cp_size not in (2, 4, 8, 16):
            raise ValueError("canonical_moe_reduce_v1 admits contributor counts 2, 4, 8, or 16")
        if len(self.combine_groups) != len(self.logical_ordinals_by_group):
            raise ValueError("combine_groups and logical_ordinals_by_group must have equal lengths")

        seen_physical: set[int] = set()
        expected_ordinals = tuple(range(self.cp_size))
        for group, ordinals in zip(self.combine_groups, self.logical_ordinals_by_group, strict=True):
            if len(group) != self.cp_size:
                raise ValueError(f"Every combine group must contain exactly {self.cp_size} ranks")
            if len(set(group)) != len(group):
                raise ValueError(f"Combine group contains duplicate physical ranks: {group}")
            if tuple(sorted(ordinals)) != expected_ordinals:
                raise ValueError(f"Logical ordinals must be a permutation of {expected_ordinals}: {ordinals}")
            if any(rank < 0 or rank >= self.world_size for rank in group):
                raise ValueError(f"Combine group contains a rank outside world_size={self.world_size}: {group}")
            overlap = seen_physical.intersection(group)
            if overlap:
                raise ValueError(f"Physical ranks appear in multiple combine groups: {sorted(overlap)}")
            seen_physical.update(group)

        if seen_physical != set(range(self.world_size)):
            raise ValueError("Combine groups must partition every physical global rank exactly once")
        if tuple(sorted(self.cp_ep_aliases)) != tuple((rank, rank) for rank in range(self.cp_size)):
            raise ValueError("Version 1 requires an explicit identity CP/EP alias map")

        expected_start = 0
        for start, end in self.pipeline_layer_ranges:
            if start != expected_start or end <= start:
                raise ValueError("Pipeline layer ranges must be positive, contiguous, and start at layer 0")
            expected_start = end
        if expected_start != GLM52_NUM_LAYERS and self.role is not ParallelRole.PRIMITIVE_TEST:
            raise ValueError(f"Pipeline ranges must cover exactly {GLM52_NUM_LAYERS} GLM layers")

        if self.role is ParallelRole.TRAINER:
            actual = (
                self.world_size,
                self.pp_size,
                self.tp_size,
                self.dp_size,
                self.cp_size,
                self.ep_size,
                self.effective_dense_tp,
            )
            admitted = {(16, 1, 1, 1, 16, 16, 1): ((0, 78),)}
            expected_ranges = admitted.get(actual)
            if expected_ranges is None:
                raise ValueError(
                    f"Unsupported GLM-5.2 trainer topology {actual}; admitted topologies are {tuple(admitted)}"
                )
            if self.pipeline_layer_ranges != expected_ranges:
                raise ValueError(
                    f"GLM-5.2 trainer topology {actual} requires pipeline ranges {expected_ranges}, "
                    f"got {self.pipeline_layer_ranges}"
                )
            expected_groups = tuple(
                tuple(range(group_start, group_start + self.cp_size))
                for group_start in range(0, self.world_size, self.cp_size)
            )
            if self.combine_groups != expected_groups:
                raise ValueError("GLM-5.2 trainer combine groups must be contiguous groups of cp_size physical ranks")
            if self.logical_ordinals_by_group != (expected_ordinals,) * len(expected_groups):
                raise ValueError("GLM-5.2 trainer requires identity logical contributor ordinals in every group")
        elif self.role is ParallelRole.PRIMITIVE_TEST:
            if self.world_size != self.cp_size or len(self.combine_groups) != 1:
                raise ValueError("Primitive plans use one combine group spanning the test world")
            if self.pp_size != 1 or self.tp_size != 1 or self.dp_size != 1 or self.ep_size != self.cp_size:
                raise ValueError("Primitive plans require PP1/TP1/DP1 and EP equal to contributor count")
        else:
            raise ValueError(f"Unknown parallel role: {self.role}")

    @classmethod
    def glm52_trainer(
        cls,
        *,
        world_size: int = 16,
        pp_size: int = 1,
        dp_size: int = 1,
        contributor_count: int = 16,
    ) -> ParallelPlan:
        identity = tuple(range(contributor_count))
        combine_groups = tuple(
            tuple(range(start, start + contributor_count)) for start in range(0, world_size, contributor_count)
        )
        pipeline_layer_ranges = ((0, 78),)
        return cls(
            role=ParallelRole.TRAINER,
            world_size=world_size,
            pp_size=pp_size,
            tp_size=1,
            dp_size=dp_size,
            cp_size=contributor_count,
            ep_size=contributor_count,
            effective_dense_tp=1,
            combine_groups=combine_groups,
            logical_ordinals_by_group=(identity,) * len(combine_groups),
            pipeline_layer_ranges=pipeline_layer_ranges,
            cp_ep_aliases=tuple((rank, rank) for rank in identity),
        )

    @classmethod
    def primitive(
        cls,
        contributor_count: int,
        *,
        physical_to_logical: tuple[int, ...] | None = None,
    ) -> ParallelPlan:
        physical_ranks = tuple(range(contributor_count))
        ordinals = physical_to_logical if physical_to_logical is not None else physical_ranks
        return cls(
            role=ParallelRole.PRIMITIVE_TEST,
            world_size=contributor_count,
            pp_size=1,
            tp_size=1,
            dp_size=1,
            cp_size=contributor_count,
            ep_size=contributor_count,
            effective_dense_tp=1,
            combine_groups=(physical_ranks,),
            logical_ordinals_by_group=(ordinals,),
            pipeline_layer_ranges=((0, 1),),
            cp_ep_aliases=tuple((rank, rank) for rank in physical_ranks),
        )

    def group_index_for_physical_rank(self, physical_global_rank: int) -> int:
        matches = [index for index, group in enumerate(self.combine_groups) if physical_global_rank in group]
        if len(matches) != 1:
            raise ValueError(f"Physical rank {physical_global_rank} does not belong to exactly one combine group")
        return matches[0]


@dataclass(frozen=True)
class CanonicalMoEGraphMetadata:
    """Fixed-capacity logical row metadata for eager or graph replay."""

    logical_row_ids: torch.Tensor
    absolute_positions: torch.Tensor
    valid_mask: torch.Tensor
    capacity: int
    valid_rows: int
    sentinel: int = -1

    @classmethod
    def build(
        cls,
        logical_row_ids: torch.Tensor,
        absolute_positions: torch.Tensor,
        *,
        capacity: int,
    ) -> CanonicalMoEGraphMetadata:
        if logical_row_ids.ndim != 1 or absolute_positions.ndim != 1:
            raise ValueError("logical_row_ids and absolute_positions must be rank-1")
        if logical_row_ids.shape != absolute_positions.shape:
            raise ValueError("logical_row_ids and absolute_positions must have equal shapes")
        if logical_row_ids.numel() > capacity:
            raise ValueError(f"Row count {logical_row_ids.numel()} exceeds fixed capacity {capacity}")
        if logical_row_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("logical_row_ids must be an integer tensor")
        if absolute_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("absolute_positions must be an integer tensor")
        if bool(torch.any(logical_row_ids < 0)) or bool(torch.any(absolute_positions < 0)):
            raise ValueError("Valid logical row IDs and absolute positions must be nonnegative")
        if torch.unique(logical_row_ids).numel() != logical_row_ids.numel():
            raise ValueError("Valid logical row IDs must be unique within one invocation")

        device = logical_row_ids.device
        padded_rows = torch.full((capacity,), -1, dtype=logical_row_ids.dtype, device=device)
        padded_positions = torch.full((capacity,), -1, dtype=absolute_positions.dtype, device=device)
        valid_mask = torch.zeros((capacity,), dtype=torch.bool, device=device)
        row_count = logical_row_ids.numel()
        padded_rows[:row_count].copy_(logical_row_ids)
        padded_positions[:row_count].copy_(absolute_positions)
        valid_mask[:row_count] = True
        return cls(padded_rows, padded_positions, valid_mask, capacity, row_count)

    def validate_for_replay(self, *, expected_contract_version: str) -> GraphContractStatus:
        if expected_contract_version != CANONICAL_MOE_REDUCE_VERSION:
            return GraphContractStatus.TOPOLOGY_VERSION_MISMATCH
        if self.logical_row_ids.numel() != self.capacity or self.absolute_positions.numel() != self.capacity:
            return GraphContractStatus.CAPACITY_OVERFLOW
        if self.valid_rows < 0 or self.valid_rows > self.capacity:
            return GraphContractStatus.CAPACITY_OVERFLOW
        if bool(torch.any(self.valid_mask & ((self.logical_row_ids < 0) | (self.absolute_positions < 0)))):
            return GraphContractStatus.MISSING_LOGICAL_POSITION
        invalid = ~self.valid_mask
        if bool(torch.any(self.logical_row_ids[invalid] != self.sentinel)) or bool(
            torch.any(self.absolute_positions[invalid] != self.sentinel)
        ):
            return GraphContractStatus.INVALID_PADDING
        return GraphContractStatus.OK

    def owner_ordinals(self, contributor_count: int) -> torch.Tensor:
        owners = torch.full_like(self.absolute_positions, self.sentinel)
        owners[self.valid_mask] = torch.remainder(self.absolute_positions[self.valid_mask], contributor_count)
        return owners


@dataclass(frozen=True)
class LocalMoEContribution:
    tensor: torch.Tensor
    metadata: CanonicalMoEGraphMetadata
    local_partial_policy: str
    layout: ContributionLayout = ContributionLayout.LOCAL_PARTIAL

    def __post_init__(self) -> None:
        if self.layout is not ContributionLayout.LOCAL_PARTIAL:
            raise ValueError("canonical_moe_reduce_v1 accepts only typed local partial contributions")
        if self.tensor.dtype is not torch.bfloat16:
            raise TypeError(f"Local MoE partial must be BF16, got {self.tensor.dtype}")
        if self.tensor.ndim < 2:
            raise ValueError("Local MoE partial must have a row dimension and at least one payload dimension")
        if self.tensor.shape[0] != self.metadata.capacity:
            raise ValueError("Local MoE partial row count must equal graph metadata capacity")
        if not self.local_partial_policy:
            raise ValueError("local_partial_policy must name the routed/shared scaling and add-order contract")


@dataclass(frozen=True)
class CanonicalMoEOutput:
    tensor: torch.Tensor
    owner_mask: torch.Tensor
    distribution: OutputDistribution
    metadata: CanonicalMoEGraphMetadata


@dataclass(frozen=True)
class _RuntimePlan:
    group_physical_ranks: tuple[int, ...]
    physical_to_logical: tuple[int, ...]
    local_group_rank: int
    local_logical_ordinal: int

    @property
    def logical_to_group_rank(self) -> tuple[int, ...]:
        result = [-1] * len(self.physical_to_logical)
        for group_rank, logical in enumerate(self.physical_to_logical):
            result[logical] = group_rank
        return tuple(result)


def _validate_runtime_plan(
    plan: ParallelPlan,
    group: dist.ProcessGroup,
    physical_global_rank: int,
) -> _RuntimePlan:
    if not dist.is_initialized():
        raise RuntimeError("canonical_moe_reduce_v1 requires an initialized torch.distributed process group")
    if dist.get_world_size() != plan.world_size:
        raise ValueError(f"Runtime world size {dist.get_world_size()} does not match plan {plan.world_size}")

    group_index = plan.group_index_for_physical_rank(physical_global_rank)
    expected_physical = plan.combine_groups[group_index]
    expected_logical = plan.logical_ordinals_by_group[group_index]
    if dist.get_world_size(group) != len(expected_physical):
        raise ValueError("Runtime combine-group size does not match the immutable ParallelPlan")

    get_group_ranks = getattr(dist, "get_process_group_ranks", None)
    if get_group_ranks is not None:
        actual_physical = tuple(get_group_ranks(group))
        if actual_physical != expected_physical:
            raise ValueError(
                f"Runtime group physical rank order {actual_physical} does not match plan {expected_physical}"
            )

    local_group_rank = dist.get_rank(group)
    if expected_physical[local_group_rank] != physical_global_rank:
        raise ValueError("Runtime local rank does not match the plan's physical group ordering")
    return _RuntimePlan(
        group_physical_ranks=expected_physical,
        physical_to_logical=expected_logical,
        local_group_rank=local_group_rank,
        local_logical_ordinal=expected_logical[local_group_rank],
    )


def _adjacent_pairwise_bf16(partials: torch.Tensor) -> torch.Tensor:
    """Evaluate the version-1 balanced tree over logical contributor axis 0."""
    if partials.dtype is not torch.bfloat16:
        raise TypeError("Pairwise canonical MoE arithmetic requires BF16 inputs")
    if partials.shape[0] not in (2, 4, 8, 16):
        raise ValueError("Pairwise canonical MoE arithmetic admits 2, 4, 8, or 16 contributors")

    level = [partials[index] for index in range(partials.shape[0])]
    while len(level) > 1:
        level = [(level[index] + level[index + 1]).to(torch.bfloat16) for index in range(0, len(level), 2)]
    return level[0]


def _transport_and_fold(
    local_partial: torch.Tensor,
    absolute_positions: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    runtime: _RuntimePlan,
    distribution: OutputDistribution,
    chunk_rows: int,
    transport: CanonicalMoETransport,
) -> tuple[torch.Tensor, torch.Tensor]:
    contributor_count = len(runtime.physical_to_logical)
    logical_to_group = runtime.logical_to_group_rank
    capacity = local_partial.shape[0]
    output = torch.zeros_like(local_partial)
    owner_mask = torch.zeros((capacity,), dtype=torch.bool, device=local_partial.device)
    payload_shape = local_partial.shape[1:]

    for start in range(0, capacity, chunk_rows):
        end = min(start + chunk_rows, capacity)
        rows = end - start
        chunk = local_partial[start:end]
        chunk_valid = valid_mask[start:end]
        owners = torch.remainder(absolute_positions[start:end].clamp_min(0), contributor_count)

        local_owner_mask = chunk_valid & (owners == runtime.local_logical_ordinal)
        if transport is CanonicalMoETransport.DENSE_V1:
            send = torch.zeros((contributor_count, rows, *payload_shape), dtype=chunk.dtype, device=chunk.device)
            for owner_logical in range(contributor_count):
                destination_group_rank = logical_to_group[owner_logical]
                row_mask = chunk_valid & (owners == owner_logical)
                send[destination_group_rank, row_mask] = chunk[row_mask]

            receive = torch.empty_like(send)
            dist.all_to_all_single(
                receive.view(contributor_count * rows, *payload_shape),
                send.contiguous().view(contributor_count * rows, *payload_shape),
                group=group,
            )
            physical_sources = receive.view(contributor_count, rows, *payload_shape)
            source_group_order = torch.tensor(logical_to_group, dtype=torch.long, device=chunk.device)
            logical_sources = physical_sources.index_select(0, source_group_order)
            folded = _adjacent_pairwise_bf16(logical_sources)
        elif transport is CanonicalMoETransport.PACKED_EP16_V2:
            if contributor_count != 16:
                raise ValueError("packed_ep16_v2 requires exactly 16 logical contributors")
            owner_counts = torch.bincount(owners[chunk_valid], minlength=contributor_count)
            # The packed transport is eager-only, so this host scalar is
            # intentional. Sequence-parallel padding resets padded position
            # IDs to zero, making owner 0 larger than ceil(rows / 16) at real
            # padded server geometries. All contributors share this
            # gathered metadata and therefore derive the same equal split.
            packed_rows = max(1, int(owner_counts.max().item()))
            send = torch.zeros(
                (contributor_count, packed_rows, *payload_shape),
                dtype=chunk.dtype,
                device=chunk.device,
            )
            local_packed_indices = None
            for owner_logical in range(contributor_count):
                destination_group_rank = logical_to_group[owner_logical]
                row_indices = torch.nonzero(chunk_valid & (owners == owner_logical), as_tuple=False).flatten()
                send[destination_group_rank, : row_indices.numel()] = chunk.index_select(0, row_indices)
                if owner_logical == runtime.local_logical_ordinal:
                    local_packed_indices = row_indices

            if local_packed_indices is None:
                raise RuntimeError("packed_ep16_v2 could not resolve the local logical owner's rows")
            receive = torch.empty_like(send)
            dist.all_to_all_single(
                receive.view(contributor_count * packed_rows, *payload_shape),
                send.contiguous().view(contributor_count * packed_rows, *payload_shape),
                group=group,
            )
            physical_sources = receive.view(contributor_count, packed_rows, *payload_shape)
            source_group_order = torch.tensor(logical_to_group, dtype=torch.long, device=chunk.device)
            logical_sources = physical_sources.index_select(0, source_group_order)
            packed_folded = _adjacent_pairwise_bf16(logical_sources)
        else:
            raise ValueError(f"Unknown canonical MoE transport: {transport}")

        owner_values = torch.zeros_like(chunk)
        if transport is CanonicalMoETransport.DENSE_V1:
            owner_values[local_owner_mask] = folded[local_owner_mask]
        else:
            owner_values.index_copy_(0, local_packed_indices, packed_folded[: local_packed_indices.numel()])
        owner_mask[start:end] = local_owner_mask

        if distribution is OutputDistribution.OWNER_SHARDED:
            output[start:end] = owner_values
            continue

        gathered = torch.empty(
            (contributor_count, rows, *payload_shape),
            dtype=chunk.dtype,
            device=chunk.device,
        )
        dist.all_gather_into_tensor(
            gathered.view(contributor_count * rows, *payload_shape),
            owner_values.contiguous(),
            group=group,
        )
        replicated = torch.zeros_like(chunk)
        for owner_logical in range(contributor_count):
            source_group_rank = logical_to_group[owner_logical]
            row_mask = chunk_valid & (owners == owner_logical)
            replicated[row_mask] = gathered[source_group_rank, row_mask]
        output[start:end] = replicated

    return output, owner_mask


class _CanonicalMoEReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_partial: torch.Tensor,
        absolute_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        group: dist.ProcessGroup,
        runtime: _RuntimePlan,
        distribution: OutputDistribution,
        chunk_rows: int,
        transport: CanonicalMoETransport,
    ) -> torch.Tensor:
        output, owner_mask = _transport_and_fold(
            local_partial,
            absolute_positions,
            valid_mask,
            group=group,
            runtime=runtime,
            distribution=distribution,
            chunk_rows=chunk_rows,
            transport=transport,
        )
        ctx.group = group
        ctx.distribution = distribution
        ctx.save_for_backward(owner_mask, valid_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        owner_mask, valid_mask = ctx.saved_tensors
        grad = grad_output.contiguous()
        if ctx.distribution is OutputDistribution.OWNER_SHARDED:
            grad = torch.where(
                owner_mask.view(-1, *([1] * (grad.ndim - 1))),
                grad,
                torch.zeros_like(grad),
            )
        grad = torch.where(
            valid_mask.view(-1, *([1] * (grad.ndim - 1))),
            grad,
            torch.zeros_like(grad),
        )
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None, None, None, None, None, None, None


class _CanonicalMoECPShardedV3(torch.autograd.Function):
    """Route source-CP buckets to their consumers, then fold locally."""

    @staticmethod
    def forward(
        ctx,
        local_partial: torch.Tensor,
        valid_mask: torch.Tensor,
        group: dist.ProcessGroup,
        runtime: _RuntimePlan,
    ) -> torch.Tensor:
        contributor_count = len(runtime.physical_to_logical)
        if local_partial.shape[0] % contributor_count:
            raise ValueError("CP_SHARDED_V3 requires equal padded CP-source capacity")
        local_capacity = local_partial.shape[0] // contributor_count
        payload_shape = local_partial.shape[1:]
        full_valid = valid_mask.reshape(contributor_count, local_capacity)
        masked = torch.where(
            valid_mask.view(-1, *([1] * len(payload_shape))),
            local_partial,
            torch.zeros_like(local_partial),
        )

        # The existing FULL CP gather is physical-group-rank major, which is
        # exactly the destination order required by the inverse all-to-all.
        send = masked.reshape(contributor_count, local_capacity, *payload_shape).contiguous()
        receive = torch.empty_like(send)
        dist.all_to_all_single(
            receive.view(contributor_count * local_capacity, *payload_shape),
            send.view(contributor_count * local_capacity, *payload_shape),
            group=group,
        )

        # Arrivals are in physical source order. Restore the immutable logical
        # contributor order before the first floating-point operation.
        logical_to_group = torch.tensor(runtime.logical_to_group_rank, dtype=torch.long, device=receive.device)
        logical_sources = receive.index_select(0, logical_to_group)
        folded = _adjacent_pairwise_bf16(logical_sources)
        local_valid = full_valid[runtime.local_group_rank]
        output = torch.where(
            local_valid.view(-1, *([1] * len(payload_shape))),
            folded,
            torch.zeros_like(folded),
        )

        ctx.group = group
        ctx.contributor_count = contributor_count
        ctx.local_capacity = local_capacity
        ctx.save_for_backward(valid_mask, local_valid)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        valid_mask, local_valid = ctx.saved_tensors
        payload_shape = grad_output.shape[1:]
        local_grad = torch.where(
            local_valid.view(-1, *([1] * len(payload_shape))),
            grad_output,
            torch.zeros_like(grad_output),
        ).contiguous()
        full_grad = torch.empty(
            (ctx.contributor_count, ctx.local_capacity, *payload_shape),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        # This byte-preserving gather only inverts the row distribution. It
        # does not claim or perform optimizer-domain gradient reduction.
        dist.all_gather_into_tensor(
            full_grad.view(ctx.contributor_count * ctx.local_capacity, *payload_shape),
            local_grad,
            group=ctx.group,
        )
        flat_grad = full_grad.reshape(ctx.contributor_count * ctx.local_capacity, *payload_shape)
        flat_grad = torch.where(
            valid_mask.view(-1, *([1] * len(payload_shape))),
            flat_grad,
            torch.zeros_like(flat_grad),
        )
        return flat_grad, None, None, None


def _canonical_moe_reduce(
    contribution: LocalMoEContribution,
    *,
    plan: ParallelPlan,
    group: dist.ProcessGroup,
    output_distribution: OutputDistribution,
    transport: CanonicalMoETransport,
    physical_global_rank: int | None,
    chunk_rows: int | None,
    graph_mode: bool,
) -> CanonicalMoEOutput:
    if not isinstance(contribution, LocalMoEContribution):
        if isinstance(contribution, CanonicalMoEOutput):
            raise TypeError("CanonicalMoEOutput is already reduced and cannot be passed as a local contribution")
        raise TypeError("canonical MoE reduction requires LocalMoEContribution")
    if contribution.layout is not ContributionLayout.LOCAL_PARTIAL:
        raise TypeError("canonical MoE reduction requires an unreduced local partial")
    status = contribution.metadata.validate_for_replay(expected_contract_version=plan.contract_version)
    if status is not GraphContractStatus.OK:
        raise ValueError(f"Canonical MoE graph metadata failed closed with status {status.name}")

    if transport is CanonicalMoETransport.PACKED_EP16_V2:
        if plan.contributor_count != 16 or plan.ep_size != 16:
            raise ValueError("packed_ep16_v2 admits only an EP16/CP16 ParallelPlan")
        if graph_mode:
            raise ValueError("packed_ep16_v2 does not yet admit CUDA graph capture")

    physical_global_rank = dist.get_rank() if physical_global_rank is None else physical_global_rank
    runtime = _validate_runtime_plan(plan, group, physical_global_rank)
    effective_chunk_rows = contribution.metadata.capacity if chunk_rows is None else chunk_rows
    if effective_chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if transport is CanonicalMoETransport.PACKED_EP16_V2:
        # Dense-v1 chunks the 16x-expanded owner slots to bound its allocation.
        # Packed-v2's full-capacity send is already only one payload tensor, and
        # coalescing here is required because GLM gathers CP shards in
        # source-grouped order: an arbitrary subrange need not contain a
        # balanced number of logical owners even though the complete capacity
        # does. Keep one equal-split A2A over the complete logical row set.
        effective_chunk_rows = contribution.metadata.capacity

    tensor = _CanonicalMoEReduce.apply(
        contribution.tensor,
        contribution.metadata.absolute_positions,
        contribution.metadata.valid_mask,
        group,
        runtime,
        output_distribution,
        effective_chunk_rows,
        transport,
    )
    return CanonicalMoEOutput(
        tensor=tensor,
        owner_mask=contribution.metadata.owner_ordinals(plan.contributor_count) == runtime.local_logical_ordinal,
        distribution=output_distribution,
        metadata=contribution.metadata,
    )


def canonical_moe_reduce_v1(
    contribution: LocalMoEContribution,
    *,
    plan: ParallelPlan,
    group: dist.ProcessGroup,
    output_distribution: OutputDistribution,
    physical_global_rank: int | None = None,
    chunk_rows: int | None = None,
    graph_mode: bool = False,
) -> CanonicalMoEOutput:
    """Canonicalize one typed local partial under the version-1 contract."""
    return _canonical_moe_reduce(
        contribution,
        plan=plan,
        group=group,
        output_distribution=output_distribution,
        transport=CanonicalMoETransport.DENSE_V1,
        physical_global_rank=physical_global_rank,
        chunk_rows=chunk_rows,
        graph_mode=graph_mode,
    )


def canonical_moe_reduce_packed_ep16_v2(
    contribution: LocalMoEContribution,
    *,
    plan: ParallelPlan,
    group: dist.ProcessGroup,
    output_distribution: OutputDistribution,
    physical_global_rank: int | None = None,
    chunk_rows: int | None = None,
    graph_mode: bool = False,
) -> CanonicalMoEOutput:
    """Use packed equal-split owner transport with version-1 fold arithmetic.

    Each destination receives at most ``ceil(chunk_rows / 16)`` rows from
    every logical contributor. The selected rows retain their original order,
    so the existing adjacent-pairwise BF16 fold observes the exact same
    contributor sequence as :func:`canonical_moe_reduce_v1`.
    """
    return _canonical_moe_reduce(
        contribution,
        plan=plan,
        group=group,
        output_distribution=output_distribution,
        transport=CanonicalMoETransport.PACKED_EP16_V2,
        physical_global_rank=physical_global_rank,
        chunk_rows=chunk_rows,
        graph_mode=graph_mode,
    )


def canonical_moe_reduce_cp_sharded_v3(
    contribution: LocalMoEContribution,
    *,
    plan: ParallelPlan,
    group: dist.ProcessGroup,
    physical_global_rank: int | None = None,
    graph_mode: bool = False,
) -> CanonicalMoEOutput:
    """Return only the canonical rows consumed by this CP rank.

    V3 changes placement, not arithmetic. It requires the rank-major FULL CP
    layout and an equal padded source capacity. Dense v1 remains an independent
    executable oracle and shares no packing or row-mapping code with this path.
    """
    if not isinstance(contribution, LocalMoEContribution):
        raise TypeError("canonical MoE reduction requires LocalMoEContribution")
    if contribution.layout is not ContributionLayout.LOCAL_PARTIAL:
        raise TypeError("canonical MoE reduction requires an unreduced local partial")
    status = contribution.metadata.validate_for_replay(expected_contract_version=plan.contract_version)
    if status is not GraphContractStatus.OK:
        raise ValueError(f"Canonical MoE graph metadata failed closed with status {status.name}")
    if plan.contributor_count != 16 or plan.ep_size != 16:
        raise ValueError("cp_sharded_v3 admits only an EP16/CP16 ParallelPlan")
    if graph_mode:
        raise ValueError("cp_sharded_v3 does not yet admit CUDA graph capture")
    if contribution.metadata.capacity % plan.contributor_count:
        raise ValueError("cp_sharded_v3 requires equal padded CP-source capacity")

    physical_global_rank = dist.get_rank() if physical_global_rank is None else physical_global_rank
    runtime = _validate_runtime_plan(plan, group, physical_global_rank)
    if runtime.physical_to_logical != tuple(range(plan.contributor_count)):
        raise ValueError("cp_sharded_v3 currently requires identity contributor ordering")

    local_capacity = contribution.metadata.capacity // plan.contributor_count
    tensor = _CanonicalMoECPShardedV3.apply(
        contribution.tensor,
        contribution.metadata.valid_mask,
        group,
        runtime,
    )
    start = runtime.local_group_rank * local_capacity
    end = start + local_capacity
    local_valid = contribution.metadata.valid_mask[start:end]
    valid_rows = int(local_valid.sum().item())
    local_metadata = CanonicalMoEGraphMetadata(
        logical_row_ids=contribution.metadata.logical_row_ids[start:end],
        absolute_positions=contribution.metadata.absolute_positions[start:end],
        valid_mask=local_valid,
        capacity=local_capacity,
        valid_rows=valid_rows,
    )
    return CanonicalMoEOutput(
        tensor=tensor,
        owner_mask=local_valid,
        distribution=OutputDistribution.CONSUMER_SHARDED,
        metadata=local_metadata,
    )


__all__ = [
    "CANONICAL_MOE_PACKED_EP16_TRANSPORT_VERSION",
    "CANONICAL_MOE_CP_SHARDED_TRANSPORT_VERSION",
    "CANONICAL_MOE_REDUCE_VERSION",
    "CanonicalMoEGraphMetadata",
    "CanonicalMoEOutput",
    "CanonicalMoETransport",
    "ContributionLayout",
    "GraphContractStatus",
    "LocalMoEContribution",
    "OutputDistribution",
    "ParallelPlan",
    "ParallelRole",
    "canonical_moe_reduce_packed_ep16_v2",
    "canonical_moe_reduce_cp_sharded_v3",
    "canonical_moe_reduce_v1",
    "resolve_canonical_moe_transport",
]
