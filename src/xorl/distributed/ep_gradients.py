"""Gradient synchronization for replicated expert-parallel parameters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.distributed._tensor import DTensor


_DEFAULT_BUCKET_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class GradientSyncStats:
    """Communication facts for one optimizer-boundary gradient reduction."""

    configured_parameter_count: int = 0
    participating_parameter_count: int = 0
    bucket_count: int = 0
    gradient_bytes: int = 0
    reduced_bytes: int = 0

    @property
    def parameter_count(self) -> int:
        """Backward-compatible alias for the configured parameter count."""

        return self.configured_parameter_count


def _wait_for_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    wait = getattr(tensor, "wait", None)
    return wait() if wait is not None else tensor


def _parameter_local_tensor(parameter: torch.Tensor) -> torch.Tensor:
    if isinstance(parameter, DTensor):
        return _wait_for_local_tensor(parameter.to_local())
    return parameter


def _parameter_local_gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    gradient = parameter.grad
    if gradient is None:
        return None
    if isinstance(gradient, DTensor):
        gradient = _wait_for_local_tensor(gradient.to_local())
    return gradient


def _set_parameter_local_gradient(parameter: torch.Tensor, local_gradient: torch.Tensor) -> None:
    gradient = parameter.grad
    if isinstance(gradient, DTensor):
        gradient.to_local().copy_(local_gradient.to(dtype=gradient.dtype))
        return
    if gradient is not None:
        gradient.copy_(local_gradient.to(dtype=gradient.dtype))
        return
    local_gradient = local_gradient.to(dtype=parameter.dtype)
    if isinstance(parameter, DTensor):
        parameter.grad = DTensor.from_local(
            local_gradient,
            parameter.device_mesh,
            parameter.placements,
            run_check=False,
        )
    else:
        parameter.grad = local_gradient


def synchronize_ep_replicated_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Sum one replicated EP gradient across the EP group.

    This hook remains available for small standalone callers. The production
    model path uses the coalesced optimizer-boundary reducer below.
    """

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return gradient

    from .parallel_state import get_parallel_state  # noqa: PLC0415

    parallel_state = get_parallel_state()
    if not parallel_state.ep_enabled:
        return gradient
    ep_group = parallel_state.ep_group
    if ep_group is None or torch.distributed.get_world_size(ep_group) <= 1:
        return gradient

    local_gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
    torch.distributed.all_reduce(local_gradient, group=ep_group)
    return gradient


@torch.no_grad()
def synchronize_ep_replicated_gradients(
    model: torch.nn.Module,
    *,
    bucket_bytes: int = _DEFAULT_BUCKET_BYTES,
) -> GradientSyncStats:
    """Coalesce replicated-factor reductions across EP before optimizer use.

    A participation mask is reduced alongside every gradient bucket. This is
    required because a globally cancelling gradient can sum to zero even when
    only a subset of ranks participated. Every replica then materializes a
    zero gradient whenever any rank contributed, keeping Adam state and weight
    decay transitions identical.
    """

    if not getattr(model, "_ep_replicated_gradient_sync_enabled", False):
        return GradientSyncStats()
    groups = getattr(model, "_ep_param_groups", {})
    if "ep_replicated_gradient_sync" in groups:
        parameters = list(groups["ep_replicated_gradient_sync"])
    else:
        # Compatibility for small callers constructing the older group shape.
        parameters = list(groups.get("ep_replicated", ()))
    if not parameters:
        return GradientSyncStats()
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return GradientSyncStats()

    from .parallel_state import get_parallel_state  # noqa: PLC0415

    parallel_state = get_parallel_state()
    if not parallel_state.ep_enabled:
        return GradientSyncStats()
    ep_group = parallel_state.ep_group
    if ep_group is None or torch.distributed.get_world_size(ep_group) <= 1:
        return GradientSyncStats()

    return synchronize_replicated_gradient_parameters(parameters, ep_group=ep_group, bucket_bytes=bucket_bytes)


@torch.no_grad()
def synchronize_replicated_gradient_parameters(
    parameters: Iterable[torch.Tensor],
    *,
    ep_group: torch.distributed.ProcessGroup,
    bucket_bytes: int = _DEFAULT_BUCKET_BYTES,
) -> GradientSyncStats:
    """Reduce a selected set of replicated gradients across one EP group.

    ``parameters`` may be model parameters or independent adapter-slot
    parameters.  Keeping this primitive independent of model metadata is
    deliberate: adapter slots are the canonical accumulation boundary for
    multi-adapter training, so their gradients must be reduced after all
    streamed ``forward_backward`` calls have been captured.
    """

    parameters = list(parameters)
    if not parameters:
        return GradientSyncStats()
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return GradientSyncStats()
    if torch.distributed.get_world_size(ep_group) <= 1:
        return GradientSyncStats()

    bucket_limit = max(int(bucket_bytes), 1)
    buckets: dict[torch.device, list[torch.Tensor]] = {}
    for parameter in parameters:
        device = _parameter_local_tensor(parameter).device
        buckets.setdefault(device, []).append(parameter)

    configured_parameter_count = 0
    participating_parameter_count = 0
    bucket_count = 0
    gradient_bytes = 0
    reduced_bytes = 0
    for bucket_parameters in buckets.values():
        current: list[torch.Tensor] = []
        current_bytes = 0
        for parameter in bucket_parameters:
            parameter_bytes = _parameter_local_tensor(parameter).numel() * torch.float32.itemsize
            if current and current_bytes + parameter_bytes > bucket_limit:
                stats = _all_reduce_gradient_bucket(current, ep_group)
                configured_parameter_count += stats.configured_parameter_count
                participating_parameter_count += stats.participating_parameter_count
                bucket_count += stats.bucket_count
                gradient_bytes += stats.gradient_bytes
                reduced_bytes += stats.reduced_bytes
                current = []
                current_bytes = 0
            current.append(parameter)
            current_bytes += parameter_bytes
        if current:
            stats = _all_reduce_gradient_bucket(current, ep_group)
            configured_parameter_count += stats.configured_parameter_count
            participating_parameter_count += stats.participating_parameter_count
            bucket_count += stats.bucket_count
            gradient_bytes += stats.gradient_bytes
            reduced_bytes += stats.reduced_bytes

    return GradientSyncStats(
        configured_parameter_count,
        participating_parameter_count,
        bucket_count,
        gradient_bytes,
        reduced_bytes,
    )


def _all_reduce_gradient_bucket(
    parameters: list[torch.Tensor],
    ep_group: torch.distributed.ProcessGroup,
) -> GradientSyncStats:
    if not parameters:
        return GradientSyncStats()

    local_numels = [_parameter_local_tensor(parameter).numel() for parameter in parameters]
    gradient_numel = sum(local_numels)
    # Append one FP32 participation entry per parameter so the data and mask
    # share one collective. This avoids a second all-reduce per bucket.
    flat = torch.zeros(
        gradient_numel + len(parameters), dtype=torch.float32, device=_parameter_local_tensor(parameters[0]).device
    )
    offset = 0
    for index, (parameter, numel) in enumerate(zip(parameters, local_numels, strict=True)):
        gradient = _parameter_local_gradient(parameter)
        if gradient is not None:
            flat[offset : offset + numel].copy_(gradient.reshape(-1))
            flat[gradient_numel + index] = 1.0
        offset += numel

    torch.distributed.all_reduce(flat, group=ep_group)

    # One device-to-host transfer per bucket determines which parameters need
    # a materialized gradient; there is no per-parameter .item() sync.
    participated = set(torch.nonzero(flat[gradient_numel:] > 0, as_tuple=False).flatten().cpu().tolist())
    offset = 0
    for index, (parameter, numel) in enumerate(zip(parameters, local_numels, strict=True)):
        reduced = flat[offset : offset + numel].view_as(_parameter_local_tensor(parameter))
        if index in participated:
            _set_parameter_local_gradient(parameter, reduced)
        offset += numel

    return GradientSyncStats(
        configured_parameter_count=len(parameters),
        participating_parameter_count=len(participated),
        bucket_count=1,
        gradient_bytes=gradient_numel * torch.float32.itemsize,
        reduced_bytes=flat.numel() * torch.float32.itemsize,
    )


def register_ep_replicated_gradient_hooks(parameters: Iterable[torch.Tensor]) -> None:
    """Install one EP-sum hook for standalone tests and diagnostics."""

    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        if getattr(parameter, "_xorl_ep_replicated_gradient_hook", None) is not None:
            continue
        handle = parameter.register_hook(synchronize_ep_replicated_gradient)
        parameter._xorl_ep_replicated_gradient_hook = handle
