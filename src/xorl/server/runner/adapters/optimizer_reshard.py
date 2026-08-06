"""Topology-aware adapter optimizer reconstruction and commit protocol.

Adapter optimizer tensors are local ``nn.Parameter`` storage rather than
DTensors. This module reconstructs optimizer state by logical tensor
rectangles and coordinates the only mutating boundary. Checkpoint file I/O and
save/load policy remain in :mod:`xorl.server.runner.adapters.manager`.
"""

import hashlib
import logging
import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn

from xorl.server.runner.adapters.sharded_state import (
    AdapterTensorLayout,
    canonical_parameter_name,
)


logger = logging.getLogger(__name__)


class _AdapterOptimizerState(Protocol):
    local_params: Dict[str, nn.Parameter]
    tensor_layouts: Dict[str, AdapterTensorLayout]
    optimizer: torch.optim.Optimizer


def canonical_parameter_order(parameters: Dict[str, Any]) -> List[str]:
    """Return stable optimizer order independent of insertion order."""

    return sorted(parameters, key=lambda name: (canonical_parameter_name(name), name))


def clone_state_to_cpu(value: Any) -> Any:
    """Recursively clone transaction state onto CPU to avoid GPU snapshot spikes."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_to_cpu(item) for item in value)
    return deepcopy(value)


def active_rectangle_from_layout_descriptor(
    descriptor: Dict[str, Any],
    *,
    context: str,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the global offset and shape of one compact adapter slot."""

    try:
        local_offset = tuple(int(value) for value in descriptor["local_logical_offset"])
        local_shape = tuple(int(value) for value in descriptor["local_logical_shape"])
        logical_shape = tuple(int(value) for value in descriptor["logical_shape"])
        active_shape = tuple(int(value) for value in descriptor["active_storage_shape"])
        active_slices = descriptor["active_local_slices"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid adapter layout descriptor in {context}") from exc
    if not (len(local_offset) == len(local_shape) == len(logical_shape) == len(active_shape) == len(active_slices)):
        raise RuntimeError(f"Inconsistent adapter layout dimensionality in {context}")
    if any(value < 0 for values in (local_offset, local_shape, logical_shape, active_shape) for value in values):
        raise RuntimeError(f"Negative adapter layout coordinate or extent in {context}")

    active_offset: List[int] = []
    for dim, (offset, local_size, active_size, selection) in enumerate(
        zip(local_offset, local_shape, active_shape, active_slices, strict=True)
    ):
        if type(selection) is not list or len(selection) != 3:
            raise RuntimeError(f"Invalid active slice for dimension {dim} in {context}")
        start, stop, step = selection
        start = 0 if start is None else int(start)
        stop = local_size if stop is None else int(stop)
        step = 1 if step is None else int(step)
        if step != 1 or start < 0 or stop < start or stop > local_size or stop - start != active_size:
            raise RuntimeError(f"Invalid active slice bounds for dimension {dim} in {context}")
        if active_size > 0 and offset + start + active_size > logical_shape[dim]:
            raise RuntimeError(f"Active adapter rectangle exceeds logical shape in dimension {dim} in {context}")
        active_offset.append(offset + start)
    return tuple(active_offset), active_shape


def _rectangle_intersection(
    left_offset: Tuple[int, ...],
    left_shape: Tuple[int, ...],
    right_offset: Tuple[int, ...],
    right_shape: Tuple[int, ...],
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    if not (len(left_offset) == len(left_shape) == len(right_offset) == len(right_shape)):
        return None
    offset = tuple(max(left, right) for left, right in zip(left_offset, right_offset, strict=True))
    end = tuple(
        min(left + left_size, right + right_size)
        for left, left_size, right, right_size in zip(left_offset, left_shape, right_offset, right_shape, strict=True)
    )
    shape = tuple(stop - start for start, stop in zip(offset, end, strict=True))
    return None if any(size <= 0 for size in shape) else (offset, shape)


def _rectangle_slices(
    intersection_offset: Tuple[int, ...],
    intersection_shape: Tuple[int, ...],
    rectangle_offset: Tuple[int, ...],
) -> Tuple[slice, ...]:
    return tuple(
        slice(start - rectangle_start, start - rectangle_start + size)
        for start, size, rectangle_start in zip(intersection_offset, intersection_shape, rectangle_offset, strict=True)
    )


def _optimizer_parameter_ids(
    optimizer_state: Dict[str, Any],
    parameter_order: List[str],
    *,
    context: str,
) -> Dict[str, Any]:
    param_groups = optimizer_state.get("param_groups")
    if type(param_groups) is not list or len(param_groups) != 1:
        raise RuntimeError(f"Topology-changing adapter optimizer resume requires one Adam parameter group: {context}")
    param_ids = param_groups[0].get("params")
    if type(param_ids) is not list or len(param_ids) != len(parameter_order):
        raise RuntimeError(f"Optimizer parameter order does not match its state dict in {context}")
    if len(set(parameter_order)) != len(parameter_order) or len(set(param_ids)) != len(param_ids):
        raise RuntimeError(f"Optimizer parameter order contains duplicates in {context}")
    return dict(zip(parameter_order, param_ids, strict=True))


def same_optimizer_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_optimizer_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            same_optimizer_value(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class OptimizerRestoreCollectiveFailure(RuntimeError):
    """The restore process group can no longer prove a coordinated outcome."""


def coordinate_restore_error(
    local_error: Optional[BaseException],
    *,
    phase: str,
) -> None:
    """Make a rank-local optimizer restore failure visible on every rank."""

    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        if local_error is not None:
            raise local_error
        return

    local_record = None
    if local_error is not None:
        local_record = f"{type(local_error).__name__}: {local_error}"
    gathered: List[Optional[str]] = [None] * torch.distributed.get_world_size()
    try:
        torch.distributed.all_gather_object(gathered, local_record)
    except Exception as exc:
        raise OptimizerRestoreCollectiveFailure(
            f"Adapter optimizer restore {phase} could not coordinate rank status; the distributed server must terminate"
        ) from exc
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(gathered) if error is not None]
    if failures:
        raise RuntimeError(f"Adapter optimizer restore {phase} failed: " + "; ".join(failures)) from local_error


def validate_live_optimizer_binding(
    adapter_state: _AdapterOptimizerState,
) -> Tuple[List[str], Dict[str, Any]]:
    """Prove that optimizer positions name the live local adapter tensors."""

    ordered_names = [
        name
        for name in canonical_parameter_order(adapter_state.local_params)
        if adapter_state.local_params[name].numel() > 0
    ]
    if not ordered_names:
        raise RuntimeError(
            "Adapter optimizer resume does not yet support coordinator ranks with no active local optimizer parameters"
        )
    live_order = [canonical_parameter_name(name) for name in ordered_names]
    if len(set(live_order)) != len(live_order):
        raise RuntimeError("Duplicate canonical live adapter parameter identity during optimizer restore")

    layouts_by_name: Dict[str, AdapterTensorLayout] = {}
    for name, layout in adapter_state.tensor_layouts.items():
        canonical_name = canonical_parameter_name(name)
        if canonical_name in layouts_by_name:
            raise RuntimeError("Duplicate canonical live adapter layout identity during optimizer restore")
        layouts_by_name[canonical_name] = layout
    active_layout_names = {name for name, layout in layouts_by_name.items() if layout.has_active_storage}
    if set(live_order) != active_layout_names:
        raise RuntimeError("Live optimizer parameters and adapter layout descriptors disagree")

    expected_parameters = []
    for original_name, canonical_name in zip(ordered_names, live_order, strict=True):
        parameter = adapter_state.local_params[original_name]
        layout = layouts_by_name[canonical_name]
        if tuple(parameter.shape) != layout.active_storage_shape or parameter.dtype != layout.dtype:
            raise RuntimeError(
                f"Live optimizer parameter {canonical_name!r} does not match its declared adapter layout"
            )
        expected_parameters.append(parameter)

    param_groups = getattr(adapter_state.optimizer, "param_groups", None)
    if type(param_groups) is not list or len(param_groups) != 1:
        raise RuntimeError("Adapter optimizer restore requires one live optimizer parameter group")
    actual_parameters = param_groups[0].get("params")
    if type(actual_parameters) is not list or len(actual_parameters) != len(expected_parameters):
        raise RuntimeError("Live adapter optimizer parameter count does not match canonical adapter order")
    if any(actual is not expected for actual, expected in zip(actual_parameters, expected_parameters, strict=True)):
        raise RuntimeError("Live adapter optimizer parameter objects are not in canonical adapter order")

    template = adapter_state.optimizer.state_dict()
    _optimizer_parameter_ids(template, live_order, context="live adapter optimizer")
    return live_order, template


def validate_staged_optimizer_compatibility(
    adapter_state: _AdapterOptimizerState,
    optimizer_state: Dict[str, Any],
    live_order: List[str],
    live_template: Dict[str, Any],
) -> None:
    """Reject staged state that ``load_state_dict`` would reinterpret by position."""

    staged_param_ids = _optimizer_parameter_ids(optimizer_state, live_order, context="staged adapter optimizer")
    _optimizer_parameter_ids(live_template, live_order, context="live adapter optimizer")
    staged_state = optimizer_state.get("state")
    if type(staged_state) is not dict or not set(staged_state).issubset(set(staged_param_ids.values())):
        raise RuntimeError("Staged adapter optimizer state contains unknown parameter positions")

    staged_group = optimizer_state["param_groups"][0]
    live_group = live_template["param_groups"][0]
    if set(staged_group) - {"params"} != set(live_group) - {"params"}:
        raise RuntimeError("Staged and live adapter optimizer parameter-group metadata are incompatible")

    parameters_by_name = {
        canonical_parameter_name(name): parameter
        for name, parameter in adapter_state.local_params.items()
        if parameter.numel() > 0
    }
    for name, param_id in staged_param_ids.items():
        parameter_state = staged_state.get(param_id)
        if parameter_state is None:
            continue
        if type(parameter_state) is not dict:
            raise RuntimeError(f"Staged optimizer state for {name!r} is not a mapping")
        for field_name, value in parameter_state.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and tuple(value.shape) != tuple(parameters_by_name[name].shape)
            ):
                raise RuntimeError(
                    f"Staged optimizer field {name}.{field_name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(parameters_by_name[name].shape)}"
                )


def commit_optimizer_state(
    adapter_state: _AdapterOptimizerState,
    optimizer_state: Dict[str, Any],
) -> None:
    """Commit on all ranks, rolling back only while coordination is healthy."""

    resident_state = clone_state_to_cpu(adapter_state.optimizer.state_dict())
    local_error: Optional[BaseException] = None
    try:
        adapter_state.optimizer.load_state_dict(optimizer_state)
    except Exception as exc:  # pragma: no cover - exercised by distributed failure injection
        local_error = exc

    try:
        coordinate_restore_error(local_error, phase="commit")
    except OptimizerRestoreCollectiveFailure as collective_error:
        # The optimizer may already be mutated locally. A failed status
        # collective means the group cannot prove who committed, so issuing a
        # rollback collective on the same group could deadlock.
        raise RuntimeError(
            "Adapter optimizer restore lost distributed coordination after optimizer mutation; "
            "the distributed server must terminate"
        ) from collective_error
    except BaseException as commit_error:
        rollback_error: Optional[BaseException] = None
        try:
            adapter_state.optimizer.load_state_dict(resident_state)
        except Exception as exc:  # pragma: no cover - a broken optimizer is unrecoverable
            rollback_error = exc
        try:
            coordinate_restore_error(rollback_error, phase="rollback")
        except BaseException as coordinated_rollback_error:
            raise RuntimeError(
                "Adapter optimizer restore failed and its resident-state rollback was not globally successful; "
                "the distributed server must terminate"
            ) from coordinated_rollback_error
        raise commit_error


def reconstruct_optimizer_state(
    adapter_state: _AdapterOptimizerState,
    manifest: Dict[str, Any],
    *,
    rank: int,
    world: int,
    load_source_rank: Callable[[int], Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble this rank's Adam state from topology-aware saved rectangles."""

    saved_world = int(manifest["world_size"])
    descriptors_by_rank = manifest.get("per_rank_layout_descriptors")
    orders_by_rank = manifest.get("per_rank_optimizer_parameter_order")
    if (
        type(descriptors_by_rank) is not list
        or len(descriptors_by_rank) != saved_world
        or type(orders_by_rank) is not list
        or len(orders_by_rank) != saved_world
    ):
        raise RuntimeError(
            f"Adapter optimizer checkpoint declares world_size={saved_world} but lacks a complete "
            "per-rank topology manifest; refusing topology-changing resume."
        )

    live_order, live_template = validate_live_optimizer_binding(adapter_state)
    live_layouts: Dict[str, AdapterTensorLayout] = {}
    for name, layout in adapter_state.tensor_layouts.items():
        canonical_name = canonical_parameter_name(name)
        if canonical_name in live_layouts:
            raise RuntimeError(f"Duplicate live adapter parameter identity during optimizer reshard: {canonical_name}")
        live_layouts[canonical_name] = layout
    if set(live_order) != {name for name, layout in live_layouts.items() if layout.has_active_storage}:
        raise RuntimeError("Live optimizer parameters and adapter layout descriptors disagree")

    source_layouts: List[Dict[str, Tuple[Dict[str, Any], Tuple[int, ...], Tuple[int, ...]]]] = []
    for source_rank, raw_descriptors in enumerate(descriptors_by_rank):
        if type(raw_descriptors) is not list or type(orders_by_rank[source_rank]) is not list:
            raise RuntimeError(f"Invalid per-rank topology metadata for saved rank {source_rank}")
        layouts_for_rank: Dict[str, Tuple[Dict[str, Any], Tuple[int, ...], Tuple[int, ...]]] = {}
        for descriptor in raw_descriptors:
            if type(descriptor) is not dict or type(descriptor.get("fqn")) is not str:
                raise RuntimeError(f"Invalid adapter layout descriptor for saved rank {source_rank}")
            fqn = canonical_parameter_name(descriptor["fqn"])
            if fqn in layouts_for_rank:
                raise RuntimeError(f"Duplicate layout for {fqn!r} on saved rank {source_rank}")
            offset, shape = active_rectangle_from_layout_descriptor(
                descriptor,
                context=f"saved rank {source_rank}, parameter {fqn}",
            )
            layouts_for_rank[fqn] = descriptor, offset, shape
        nonempty_layouts = {name for name, (_, _, shape) in layouts_for_rank.items() if math.prod(shape) > 0}
        if list(orders_by_rank[source_rank]) != [
            canonical_parameter_name(name) for name in orders_by_rank[source_rank]
        ]:
            raise RuntimeError(f"Non-canonical optimizer parameter name on saved rank {source_rank}")
        if len(set(orders_by_rank[source_rank])) != len(orders_by_rank[source_rank]):
            raise RuntimeError(f"Duplicate optimizer parameter on saved rank {source_rank}")
        if set(orders_by_rank[source_rank]) != nonempty_layouts:
            raise RuntimeError(f"Optimizer order and layouts disagree on saved rank {source_rank}")
        source_layouts.append(layouts_for_rank)

    fragments_by_fqn: Dict[str, List[Dict[str, Any]]] = {}
    needed_source_ranks: set[int] = set()
    for fqn in live_order:
        live_layout = live_layouts[fqn]
        target_offset = live_layout.active_global_offset
        target_shape = live_layout.active_storage_shape
        rectangles: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]] = {}
        for source_rank, layouts_for_rank in enumerate(source_layouts):
            source = layouts_for_rank.get(fqn)
            if source is None:
                continue
            descriptor, source_offset, source_shape = source
            if tuple(int(value) for value in descriptor.get("logical_shape", ())) != live_layout.logical_shape:
                raise RuntimeError(f"Logical shape changed for adapter optimizer parameter {fqn!r}")
            if descriptor.get("dtype") != str(live_layout.dtype).replace("torch.", ""):
                raise RuntimeError(f"Dtype changed for adapter optimizer parameter {fqn!r}")
            intersection = _rectangle_intersection(target_offset, target_shape, source_offset, source_shape)
            if intersection is None:
                continue
            rectangles.setdefault((source_offset, source_shape), []).append(
                {
                    "rank": source_rank,
                    "source_offset": source_offset,
                    "source_shape": source_shape,
                    "intersection_offset": intersection[0],
                    "intersection_shape": intersection[1],
                }
            )
        fragments = [replicas[0] | {"replicas": replicas} for replicas in rectangles.values()]
        covered = sum(math.prod(fragment["intersection_shape"]) for fragment in fragments)
        for index, left in enumerate(fragments):
            for right in fragments[index + 1 :]:
                if (
                    _rectangle_intersection(
                        left["intersection_offset"],
                        left["intersection_shape"],
                        right["intersection_offset"],
                        right["intersection_shape"],
                    )
                    is not None
                ):
                    raise RuntimeError(f"Overlapping non-replica optimizer rectangles for {fqn!r}")
        if covered != math.prod(target_shape):
            raise RuntimeError(
                f"Saved optimizer rectangles do not exactly cover live parameter {fqn!r}: "
                f"covered={covered}, expected={math.prod(target_shape)}"
            )
        fragments_by_fqn[fqn] = fragments
        needed_source_ranks.update(replica["rank"] for fragment in fragments for replica in fragment["replicas"])

    live_param_ids = _optimizer_parameter_ids(live_template, live_order, context=f"live rank {rank}/{world}")
    restored_by_fqn: Dict[str, Dict[str, Any]] = {fqn: {} for fqn in live_order}
    reference_fields: Dict[str, set[str]] = {}
    nonspatial_values: Dict[Tuple[str, str], Any] = {}
    spatial_digests: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...], str], str] = {}
    copied_spatial_fragments: set[Tuple[str, Tuple[int, ...], Tuple[int, ...], str]] = set()
    source_group_metadata: Optional[Dict[str, Any]] = None

    # The callback yields one complete source shard at a time. This bounds host
    # peak memory to one source shard plus this rank's reconstructed target.
    for source_rank in sorted(needed_source_ranks):
        source_state = load_source_rank(source_rank)
        source_param_ids = _optimizer_parameter_ids(
            source_state,
            orders_by_rank[source_rank],
            context=f"saved rank {source_rank}",
        )
        group = source_state["param_groups"][0]
        group_metadata = {key: value for key, value in group.items() if key != "params"}
        if source_group_metadata is None:
            source_group_metadata = deepcopy(group_metadata)
        elif not same_optimizer_value(group_metadata, source_group_metadata):
            raise RuntimeError(f"Optimizer parameter-group metadata differs on saved rank {source_rank}")

        for fqn in live_order:
            for fragment in fragments_by_fqn[fqn]:
                replica = next(
                    (candidate for candidate in fragment["replicas"] if candidate["rank"] == source_rank),
                    None,
                )
                if replica is None:
                    continue
                try:
                    param_state = source_state["state"][source_param_ids[fqn]]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Saved rank {source_rank} has no optimizer state for parameter {fqn!r}"
                    ) from exc
                fields = set(param_state)
                if fqn not in reference_fields:
                    reference_fields[fqn] = fields
                elif fields != reference_fields[fqn]:
                    raise RuntimeError(f"Optimizer state fields differ across saved rectangles for {fqn!r}")

                rectangle_key = (
                    fqn,
                    fragment["source_offset"],
                    fragment["source_shape"],
                )
                for field_name in sorted(fields):
                    value = param_state[field_name]
                    spatial = isinstance(value, torch.Tensor) and tuple(value.shape) == fragment["source_shape"]
                    if not spatial:
                        if isinstance(value, torch.Tensor) and tuple(value.shape) not in {
                            (),
                            (1,),
                        }:
                            raise RuntimeError(
                                f"Non-spatial optimizer field {fqn}.{field_name} has unsupported shape "
                                f"{tuple(value.shape)}"
                            )
                        key = (fqn, field_name)
                        if key in nonspatial_values and not same_optimizer_value(nonspatial_values[key], value):
                            raise RuntimeError(
                                f"Replicated optimizer field {fqn}.{field_name} differs across saved ranks"
                            )
                        if key not in nonspatial_values:
                            nonspatial_values[key] = deepcopy(value)
                            restored_by_fqn[fqn][field_name] = deepcopy(value)
                        continue

                    digest_key = (*rectangle_key, field_name)
                    digest = _tensor_sha256(value)
                    prior_digest = spatial_digests.setdefault(digest_key, digest)
                    if prior_digest != digest:
                        raise RuntimeError(f"Replicated optimizer field {fqn}.{field_name} differs across saved ranks")
                    if digest_key in copied_spatial_fragments:
                        continue
                    target_tensor = restored_by_fqn[fqn].get(field_name)
                    if target_tensor is None:
                        target_tensor = torch.empty(
                            live_layouts[fqn].active_storage_shape,
                            dtype=value.dtype,
                            device="cpu",
                        )
                        restored_by_fqn[fqn][field_name] = target_tensor
                    elif not isinstance(target_tensor, torch.Tensor) or target_tensor.dtype != value.dtype:
                        raise RuntimeError(
                            f"Optimizer field type changed across saved rectangles for {fqn}.{field_name}"
                        )
                    source_slices = _rectangle_slices(
                        fragment["intersection_offset"],
                        fragment["intersection_shape"],
                        fragment["source_offset"],
                    )
                    target_slices = _rectangle_slices(
                        fragment["intersection_offset"],
                        fragment["intersection_shape"],
                        live_layouts[fqn].active_global_offset,
                    )
                    target_tensor[target_slices].copy_(value[source_slices])
                    copied_spatial_fragments.add(digest_key)

    if source_group_metadata is None:  # pragma: no cover - exact coverage requires a source
        raise RuntimeError("Topology-changing optimizer resume found no source optimizer shard")
    restored_state = {live_param_ids[fqn]: restored_by_fqn[fqn] for fqn in live_order}
    restored_group = deepcopy(source_group_metadata)
    restored_group["params"] = live_template["param_groups"][0]["params"]
    logger.info(
        "Resharded adapter optimizer for live rank %d/%d from saved world_size=%d using source ranks %s",
        rank,
        world,
        saved_world,
        sorted(needed_source_ranks),
    )
    return {"state": restored_state, "param_groups": [restored_group]}
