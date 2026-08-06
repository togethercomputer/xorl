"""Topology-aware local storage for multi-adapter LoRA state.

The model is the owner of distributed placement.  Adapter state is deliberately
ordinary local tensors so the per-adapter optimizers can remain compatible with
fused elementwise optimizers.  This module records enough of the model layout to
pack and unpack those tensors without guessing from parameter names or local
shapes.

The layout helpers are intentionally independent of the adapter manager
lifecycle.  They are also used by the distributed layout tests and by the
checkpoint fingerprint code.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from xorl.distributed.gradient_reduction import (
    GradientReductionDomain,
    validate_gradient_reduction_domain,
)


_MASK63 = (1 << 63) - 1


def wait_for_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a local tensor, waiting for async DTensor materialization if needed."""

    wait = getattr(tensor, "wait", None)
    return wait() if wait is not None else tensor


def canonical_parameter_name(name: str) -> str:
    """Remove wrappers that do not change the logical model parameter identity."""

    return name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _mesh_local_rank(mesh: Any) -> int:
    """Get a stable local coordinate from a one-dimensional EP mesh."""

    try:
        return int(mesh.get_local_rank())
    except Exception:
        pass
    try:
        return int(mesh.get_local_rank(mesh_dim=0))
    except Exception:
        pass
    coordinate = mesh.get_coordinate()
    if coordinate is not None:
        return int(coordinate[0])
    return 0


def _ep_plan_matches(model: nn.Module, fqn: str) -> bool:
    """Return whether the model's explicit parallel plan owns this parameter."""

    get_plan = getattr(model, "get_parallel_plan", None)
    if get_plan is None:
        return False
    try:
        plan = get_plan()
        ep_plan = getattr(plan, "ep_plan", {}) or {}
    except Exception:
        return False
    # Import lazily to keep this small layout module usable by CPU-only tests.
    from xorl.distributed.utils import check_fqn_match  # noqa: PLC0415

    return any(check_fqn_match(pattern, fqn) for pattern in ep_plan)


def _spec_info(model: nn.Module, name: str, param: torch.Tensor) -> Any:
    mapping = getattr(model, "_fqn2spec_info", None)
    clean_name = canonical_parameter_name(name)
    if mapping is not None:
        info = mapping.get(clean_name) or mapping.get(name)
        if info is not None:
            return info
    return getattr(param, "spec_info", None)


def _is_explicit_ep_layout(model: nn.Module, name: str, spec_info: Any) -> bool:
    """Distinguish real EP ownership from the generic Replicate annotations.

    ``ParallelPlan.apply`` annotates every parameter with a ``SpecInfo`` when EP
    is enabled.  A replicated non-expert parameter must not therefore be treated
    as an EP replica.  Sharded specs are unambiguously EP-owned; replicated specs
    count only when the model's explicit plan matches the parameter.
    """

    if spec_info is None or getattr(spec_info, "ep_mesh", None) is None:
        return False
    placement = getattr(spec_info, "placement", None)
    if isinstance(placement, Shard):
        return True
    if isinstance(placement, Replicate):
        return _ep_plan_matches(model, canonical_parameter_name(name))
    return False


def _placement_signature(param: torch.Tensor, spec_info: Any, ep_owned: bool) -> tuple[Any, ...]:
    if isinstance(param, DTensor):
        mesh = param.device_mesh
        mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        placements = tuple(
            type(placement).__name__ + ":" + str(getattr(placement, "dim", "")) for placement in param.placements
        )
        return ("dtensor", mesh_names, placements, bool(ep_owned))
    placement = getattr(spec_info, "placement", None)
    if placement is None:
        return ("local", bool(ep_owned))
    return ("local", type(placement).__name__, getattr(placement, "dim", None), bool(ep_owned))


def _active_intersection(
    local_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    logical_shape: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    slices: list[slice] = []
    storage_shape: list[int] = []
    for offset, size, logical_size in zip(local_offset, local_shape, logical_shape, strict=True):
        start = max(0, -offset)
        end = min(size, logical_size - offset)
        if end < start:
            end = start
        slices.append(slice(start, end))
        storage_shape.append(end - start)
    return tuple(slices), tuple(storage_shape)


@dataclass(frozen=True)
class AdapterTensorLayout:
    """Static model placement plus one session's active local rectangle.

    ``substrate_shape`` and ``local_substrate_shape`` describe the maximum-rank
    model tensor.  ``logical_shape`` describes the active session rank.  The
    adapter slot stores exactly ``active_storage_shape`` elements and never
    stores inactive maximum-rank values.
    """

    fqn: str
    dtype: torch.dtype
    rank_dim: int
    substrate_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    local_substrate_shape: tuple[int, ...]
    local_logical_offset: tuple[int, ...]
    local_logical_shape: tuple[int, ...]
    active_local_slices: tuple[slice, ...]
    active_storage_shape: tuple[int, ...]
    replica_count: int = 1
    replica_ranks: tuple[int, ...] = (0,)
    replica_key: tuple[Any, ...] = ()
    placement_signature: tuple[Any, ...] = ()
    gradient_reduction: GradientReductionDomain = GradientReductionDomain.NONE

    @property
    def has_active_storage(self) -> bool:
        return all(size > 0 for size in self.active_storage_shape)

    @property
    def needs_ep_gradient_sync(self) -> bool:
        """Whether this active rectangle needs the canonical EP gradient sum."""

        return self.replica_count > 1 and self.gradient_reduction is GradientReductionDomain.EP_SUM

    @property
    def is_ep_owned(self) -> bool:
        """Whether the model placement explicitly assigns this tensor to EP."""

        # ``_placement_signature`` stores this explicit ownership bit for both
        # DTensor and already-local parameters. Keeping it as a layout
        # property avoids inferring ownership from backend names or shapes.
        return bool(self.placement_signature and self.placement_signature[-1] is True)

    @property
    def active_global_offset(self) -> tuple[int, ...]:
        return tuple(
            offset + (selection.start or 0)
            for offset, selection in zip(self.local_logical_offset, self.active_local_slices, strict=True)
        )

    def canonical_descriptor(self) -> tuple[Any, ...]:
        """Descriptor used for cross-rank layout validation and replica identity."""

        return (
            self.fqn,
            self.logical_shape,
            self.local_logical_offset,
            self.local_logical_shape,
            _dtype_name(self.dtype),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fqn": self.fqn,
            "dtype": _dtype_name(self.dtype),
            "rank_dim": self.rank_dim,
            "substrate_shape": list(self.substrate_shape),
            "logical_shape": list(self.logical_shape),
            "local_substrate_shape": list(self.local_substrate_shape),
            "local_logical_offset": list(self.local_logical_offset),
            "local_logical_shape": list(self.local_logical_shape),
            "active_local_slices": [
                [selection.start, selection.stop, selection.step] for selection in self.active_local_slices
            ],
            "active_storage_shape": list(self.active_storage_shape),
            "replica_count": self.replica_count,
            "replica_ranks": list(self.replica_ranks),
            "replica_key": list(self.replica_key),
            "placement_signature": [
                list(item) if isinstance(item, tuple) else item for item in self.placement_signature
            ],
            "gradient_reduction": self.gradient_reduction.value,
        }

    def pack_from_local(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Extract the active intersection from a model-local substrate tensor."""

        local_tensor = wait_for_local_tensor(local_tensor)
        if tuple(local_tensor.shape) != self.local_substrate_shape:
            raise ValueError(
                f"Local tensor for {self.fqn} has shape {tuple(local_tensor.shape)}, "
                f"expected {self.local_substrate_shape}"
            )
        return local_tensor[self.active_local_slices].detach().clone().contiguous()

    def unpack_to_local(self, storage: torch.Tensor, destination: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Zero inactive substrate values and insert one active local adapter slot."""

        if tuple(storage.shape) != self.active_storage_shape:
            raise ValueError(
                f"Adapter slot for {self.fqn} has shape {tuple(storage.shape)}, expected {self.active_storage_shape}"
            )
        if destination is None:
            destination = torch.zeros(
                self.local_substrate_shape,
                dtype=self.dtype,
                device=storage.device,
            )
        destination = wait_for_local_tensor(destination)
        if tuple(destination.shape) != self.local_substrate_shape:
            raise ValueError(
                f"Destination for {self.fqn} has shape {tuple(destination.shape)}, "
                f"expected {self.local_substrate_shape}"
            )
        destination.zero_()
        if storage.numel():
            destination[self.active_local_slices].copy_(storage)
        return destination


@dataclass
class ShardedAdapterTensor:
    """A local optimizer parameter paired with its layout."""

    layout: AdapterTensorLayout
    param: nn.Parameter


def _base_layout_for_parameter(
    model: nn.Module,
    name: str,
    param: torch.Tensor,
    rank_dim: int,
    active_rank: int,
) -> AdapterTensorLayout:
    raw_param = param.data if isinstance(param, nn.Parameter) else param
    if isinstance(raw_param, DTensor):
        local_tensor = wait_for_local_tensor(raw_param.to_local())
        local_shape, local_offset = compute_local_shape_and_global_offset(
            tuple(raw_param.shape), raw_param.device_mesh, raw_param.placements
        )
        substrate_shape = tuple(int(value) for value in raw_param.shape)
        local_substrate_shape = tuple(int(value) for value in local_tensor.shape)
        local_logical_offset = tuple(int(value) for value in local_offset)
        # The pinned helper is the source of truth; the shape assertion catches
        # a stale DTensor implementation before any adapter bytes are copied.
        if tuple(local_shape) != local_substrate_shape:
            raise RuntimeError(
                f"DTensor local-shape mismatch for {name}: helper={tuple(local_shape)}, actual={local_substrate_shape}"
            )
    else:
        local_tensor = wait_for_local_tensor(raw_param)
        substrate_shape = tuple(int(value) for value in local_tensor.shape)
        local_substrate_shape = substrate_shape
        local_logical_offset = tuple(0 for _ in substrate_shape)

    spec_info = _spec_info(model, name, raw_param)
    gradient_reduction = validate_gradient_reduction_domain(
        getattr(spec_info, "gradient_reduction", GradientReductionDomain.NONE)
    )
    ep_owned = _is_explicit_ep_layout(model, name, spec_info)
    if ep_owned:
        placement = getattr(spec_info, "placement", None)
        ep_mesh = spec_info.ep_mesh
        if isinstance(placement, Shard):
            ep_dim = placement.dim if placement.dim >= 0 else len(substrate_shape) + placement.dim
            ep_size = int(ep_mesh.size())
            ep_rank = _mesh_local_rank(ep_mesh)
            substrate_shape = list(substrate_shape)
            local_logical_offset = list(local_logical_offset)
            substrate_shape[ep_dim] *= ep_size
            local_logical_offset[ep_dim] += ep_rank * int(local_substrate_shape[ep_dim])
            substrate_shape = tuple(substrate_shape)
            local_logical_offset = tuple(local_logical_offset)
        # Replicate() is a genuine shared logical rectangle.  Its logical shape
        # and offset stay unchanged; replica_count is filled after exchange.

    if rank_dim < 0:
        rank_dim += len(substrate_shape)
    if rank_dim < 0 or rank_dim >= len(substrate_shape):
        raise ValueError(f"LoRA rank dimension {rank_dim} is invalid for {name} shape {substrate_shape}")
    if active_rank < 0 or active_rank > substrate_shape[rank_dim]:
        raise ValueError(
            f"Active LoRA rank {active_rank} exceeds substrate dimension {substrate_shape[rank_dim]} for {name}"
        )
    logical_shape = list(substrate_shape)
    logical_shape[rank_dim] = int(active_rank)
    logical_shape = tuple(logical_shape)
    active_local_slices, active_storage_shape = _active_intersection(
        tuple(local_logical_offset),
        local_substrate_shape,
        logical_shape,
    )
    fqn = canonical_parameter_name(name)
    dtype = raw_param.dtype
    placement_signature = _placement_signature(raw_param, spec_info, ep_owned)
    replica_key = (
        fqn,
        logical_shape,
        tuple(local_logical_offset),
        local_substrate_shape,
        _dtype_name(dtype),
    )
    return AdapterTensorLayout(
        fqn=fqn,
        dtype=dtype,
        rank_dim=rank_dim,
        substrate_shape=tuple(substrate_shape),
        logical_shape=logical_shape,
        local_substrate_shape=local_substrate_shape,
        local_logical_offset=tuple(local_logical_offset),
        local_logical_shape=local_substrate_shape,
        active_local_slices=active_local_slices,
        active_storage_shape=active_storage_shape,
        replica_key=replica_key,
        placement_signature=placement_signature,
        gradient_reduction=gradient_reduction,
    )


def _descriptor_from_layout(layout: AdapterTensorLayout) -> dict[str, Any]:
    return {
        "fqn": layout.fqn,
        "logical_shape": list(layout.logical_shape),
        "local_logical_offset": list(layout.local_logical_offset),
        "local_logical_shape": list(layout.local_logical_shape),
        "dtype": _dtype_name(layout.dtype),
        "gradient_reduction": layout.gradient_reduction.value,
    }


def discover_adapter_layouts(
    model: nn.Module,
    parameter_metadata: Mapping[str, Mapping[str, Any]],
    *,
    active_rank: int,
    process_group: Optional[dist.ProcessGroup] = None,
    pipeline_parallel_size: int = 1,
    local_group_memberships: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[
    dict[str, AdapterTensorLayout],
    str,
    dict[str, tuple[tuple[int, ...], ...]],
]:
    """Discover and validate the local layouts for one active adapter rank."""

    if pipeline_parallel_size > 1:
        raise RuntimeError(
            "Multi-adapter topology discovery does not support pipeline parallelism; "
            "use the exact optimizer ownership group or disable PP."
        )
    named_params = dict(model.named_parameters())
    layouts: dict[str, AdapterTensorLayout] = {}
    for name in sorted(parameter_metadata):
        if name not in named_params:
            raise RuntimeError(f"LoRA parameter {name!r} changed identity before layout discovery")
        metadata = parameter_metadata[name]
        layouts[name] = _base_layout_for_parameter(
            model,
            name,
            named_params[name],
            int(metadata["rank_dim"]),
            active_rank,
        )

    group = process_group
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(group=group) if distributed else 1
    local_descriptors = [_descriptor_from_layout(layouts[name]) for name in sorted(layouts)]
    local_payload = {
        "layouts": local_descriptors,
        "group_memberships": {
            str(key): tuple(int(member) for member in members)
            for key, members in (local_group_memberships or {}).items()
        },
    }
    gathered: list[dict[str, Any]] = [local_payload]
    if world > 1:
        gathered = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_payload, group=group)

    expected_names = [descriptor["fqn"] for descriptor in local_descriptors]
    expected_static = {
        descriptor["fqn"]: (
            tuple(descriptor["logical_shape"]),
            descriptor["dtype"],
            descriptor["gradient_reduction"],
        )
        for descriptor in local_descriptors
    }
    for rank, payload in enumerate(gathered):
        descriptors = payload["layouts"]
        if [descriptor["fqn"] for descriptor in descriptors] != expected_names:
            raise RuntimeError(f"Rank {rank} exposed a different ordered LoRA layout")
        for descriptor in descriptors:
            key = descriptor["fqn"]
            static = (
                tuple(descriptor["logical_shape"]),
                descriptor["dtype"],
                descriptor["gradient_reduction"],
            )
            if static != expected_static[key]:
                raise RuntimeError(f"Incompatible logical LoRA layout for {key!r} on rank {rank}")

    replica_members: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for rank, payload in enumerate(gathered):
        descriptors = payload["layouts"]
        for descriptor in descriptors:
            key = (
                descriptor["fqn"],
                tuple(descriptor["logical_shape"]),
                tuple(descriptor["local_logical_offset"]),
                tuple(descriptor["local_logical_shape"]),
                descriptor["dtype"],
            )
            replica_members[key].append(rank)
    for name, layout in list(layouts.items()):
        members = tuple(replica_members[layout.replica_key])
        layouts[name] = replace(layout, replica_count=len(members), replica_ranks=members)

    group_memberships: dict[str, tuple[tuple[int, ...], ...]] = {}
    group_keys = sorted({key for payload in gathered for key in payload["group_memberships"]})
    for key in group_keys:
        group_memberships[key] = tuple(
            sorted(
                {
                    tuple(payload["group_memberships"][key])
                    for payload in gathered
                    if key in payload["group_memberships"]
                }
            )
        )

    fingerprint_payload = {
        "world_size": world,
        "layouts": [layouts[name].to_json_dict() for name in sorted(layouts)],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return layouts, fingerprint, group_memberships


def layout_fingerprint(layouts: Mapping[str, AdapterTensorLayout], *, world_size: int = 1) -> str:
    payload = {
        "world_size": int(world_size),
        "layouts": [layouts[name].to_json_dict() for name in sorted(layouts)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, *components: object) -> int:
    """Derive a deterministic signed-63-bit seed without Python's randomized hash."""

    payload = "\x1f".join([str(int(base_seed)), *(str(component) for component in components)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") & _MASK63


def _stable_uint64(values: torch.Tensor, seed: int) -> torch.Tensor:
    """Vectorized deterministic integer mixing for topology-independent draws."""

    values = (values.to(torch.int64) + int(seed)) & _MASK63
    values = ((values ^ (values >> 30)) * 0xBF58476D1CE4E5B9) & _MASK63
    values = ((values ^ (values >> 27)) * 0x94D049BB133111EB) & _MASK63
    return (values ^ (values >> 31)) & _MASK63


def deterministic_kaiming_uniform(
    layout: AdapterTensorLayout,
    *,
    base_seed: int,
    session_identity: str,
    fan_in: Optional[int] = None,
) -> torch.Tensor:
    """Generate a Kaiming-uniform local slot from global logical coordinates."""

    shape = layout.active_storage_shape
    if any(size == 0 for size in shape):
        return torch.empty(shape, dtype=layout.dtype)
    device = torch.device("cpu")
    # The caller moves this result to the adapter device.  CPU generation keeps
    # registration independent from unrelated CUDA RNG state.
    starts = layout.active_global_offset
    strides: list[int] = []
    stride = 1
    for size in reversed(layout.logical_shape):
        strides.append(stride)
        stride *= size
    strides.reverse()
    linear = torch.zeros(shape, dtype=torch.int64, device=device)
    for dim, (size, start, logical_stride) in enumerate(zip(shape, starts, strides, strict=True)):
        view_shape = [1] * len(shape)
        view_shape[dim] = size
        linear = linear + (torch.arange(size, dtype=torch.int64).reshape(view_shape) + start) * logical_stride
    seed = stable_seed(base_seed, session_identity, layout.fqn)
    mixed = _stable_uint64(linear, seed)
    mantissa = mixed & ((1 << 53) - 1)
    uniform = mantissa.to(torch.float64) / float(1 << 53)
    if fan_in is None:
        fan_in = layout.logical_shape[1] if len(layout.logical_shape) > 1 else max(1, layout.logical_shape[0])
    bound = 1.0 / max(float(fan_in), 1.0) ** 0.5
    return ((uniform * 2.0 - 1.0) * bound).to(dtype=layout.dtype)


def deterministic_local_initialization(
    layout: AdapterTensorLayout,
    *,
    base_seed: int,
    session_identity: str,
    is_lora_b: bool,
) -> torch.Tensor:
    if is_lora_b:
        return torch.zeros(layout.active_storage_shape, dtype=layout.dtype)
    return deterministic_kaiming_uniform(
        layout,
        base_seed=base_seed,
        session_identity=session_identity,
        fan_in=layout.logical_shape[1] if len(layout.logical_shape) > 1 else None,
    )


def pack_logical_tensor(layout: AdapterTensorLayout, logical_tensor: torch.Tensor) -> torch.Tensor:
    """Pack a full active logical tensor into this rank's local slot."""

    if tuple(logical_tensor.shape) != layout.logical_shape:
        raise ValueError(
            f"Logical tensor for {layout.fqn} has shape {tuple(logical_tensor.shape)}, expected {layout.logical_shape}"
        )
    global_slices = tuple(
        slice(offset + (selection.start or 0), offset + (selection.start or 0) + size)
        for offset, selection, size in zip(
            layout.local_logical_offset,
            layout.active_local_slices,
            layout.active_storage_shape,
            strict=True,
        )
    )
    return logical_tensor[global_slices].detach().clone().contiguous()


def unpack_local_tensor(layout: AdapterTensorLayout, local_slot: torch.Tensor) -> torch.Tensor:
    """Create a model-local substrate tensor from a compact adapter slot."""

    return layout.unpack_to_local(local_slot)
