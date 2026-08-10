"""Residual transport and logical norm for compiled adapter gradients."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

from xorl.server.runner.adapters.gradient_ownership import (
    AdapterGradientOwnershipError,
    AdapterGradientOwnershipPlan,
    ReductionDomainPlan,
    ReductionOperation,
)
from xorl.server.runner.adapters.sharded_state import canonical_parameter_name


class AdapterGradientCollectiveFailure(RuntimeError):
    """A distributed collective failed; the process group is no longer trusted."""


class AdapterGradientMutationFailure(RuntimeError):
    """Optimizer mutation began and the session must recover from checkpoint."""


@dataclass(frozen=True)
class AdapterGradientTransportStats:
    collective_count: int = 0
    bucket_count: int = 0
    input_bytes: int = 0
    transported_bytes: int = 0
    largest_bucket_bytes: int = 0


@dataclass(frozen=True)
class TransportCompleteLocalGradients(Mapping[str, torch.Tensor]):
    """The common local logical-gradient image consumed by norm and clipping."""

    gradients: Mapping[str, torch.Tensor]

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.gradients[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.gradients)

    def __len__(self) -> int:
        return len(self.gradients)


def _canonical_tensors(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in values.items():
        canonical = canonical_parameter_name(name)
        if canonical in result:
            raise AdapterGradientOwnershipError(f"Duplicate canonical gradient {canonical!r}")
        result[canonical] = tensor
    return result


def _bounded_bucket_slices(
    fqns: list[str],
    gradients: Mapping[str, torch.Tensor],
    bucket_bytes: int,
) -> list[list[tuple[str, int, int]]]:
    buckets: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_bytes = 0
    for fqn in fqns:
        tensor = gradients[fqn]
        element_bytes = tensor.element_size()
        if element_bytes > bucket_bytes:
            raise AdapterGradientOwnershipError(
                f"Transport bucket size {bucket_bytes} cannot hold one {element_bytes}-byte element"
            )
        start = 0
        while start < tensor.numel():
            available_elements = (bucket_bytes - current_bytes) // element_bytes
            if available_elements == 0:
                buckets.append(current)
                current = []
                current_bytes = 0
                continue
            stop = min(tensor.numel(), start + available_elements)
            current.append((fqn, start, stop))
            current_bytes += (stop - start) * element_bytes
            start = stop
            if current_bytes == bucket_bytes:
                buckets.append(current)
                current = []
                current_bytes = 0
    if current:
        buckets.append(current)
    return buckets


def transport_complete_local_gradients(
    *,
    plan: AdapterGradientOwnershipPlan,
    numerators: Mapping[str, torch.Tensor],
    templates: Mapping[str, torch.Tensor],
    multiplier: float,
    resolve_group: Callable[[ReductionDomainPlan], Any],
    bucket_bytes: int,
    consume_numerators: bool = False,
) -> tuple[TransportCompleteLocalGradients, AdapterGradientTransportStats]:
    """Build one transport-complete local logical-gradient image.

    Pending reductions execute in canonical domain, parameter, and
    bounded-bucket order. Production finalization consumes its scratch in place
    after local validation; the copying option exists only for isolated algebra
    tests of this pure transport boundary.
    """

    if not torch.isfinite(torch.tensor(multiplier)) or multiplier <= 0:
        raise AdapterGradientOwnershipError("Finalization multiplier must be finite and positive")
    if bucket_bytes <= 0:
        raise AdapterGradientOwnershipError("Transport bucket size must be positive")
    raw = _canonical_tensors(numerators)
    template_by_fqn = _canonical_tensors(templates)
    expected = {item.fqn for item in plan.parameters}
    if set(raw) - expected or set(template_by_fqn) != expected:
        raise AdapterGradientOwnershipError("Gradient scratch and templates differ from the compiled plan")

    pending_fqns = {item.fqn for item in plan.parameters if item.pending_domains}
    gradients: dict[str, torch.Tensor] = {}
    input_bytes = 0
    for item in plan.parameters:
        source = raw.get(item.fqn)
        if source is None:
            dtype = torch.float32 if item.fqn in pending_fqns else template_by_fqn[item.fqn].dtype
            gradients[item.fqn] = torch.zeros_like(template_by_fqn[item.fqn], dtype=dtype)
            continue
        dtype = torch.float32 if item.fqn in pending_fqns else template_by_fqn[item.fqn].dtype
        if consume_numerators and source.dtype == dtype:
            destination = source
            destination.mul_(multiplier)
        else:
            destination = torch.empty_like(source, dtype=dtype)
            torch.mul(source, multiplier, out=destination)
        gradients[item.fqn] = destination
        input_bytes += source.numel() * source.element_size()

    domains: dict[tuple[str, str, str, str], tuple[ReductionDomainPlan, list[str]]] = {}
    for item in plan.parameters:
        for domain in item.pending_domains:
            key = domain.axis.value, domain.authority.value, domain.operation.value, domain.group_key
            domains.setdefault(key, (domain, []))[1].append(item.fqn)

    collective_count = 0
    transported_bytes = 0
    bucket_count = 0
    largest_bucket_bytes = 0
    for key in sorted(domains):
        domain, fqns = domains[key]
        if domain.operation is not ReductionOperation.SUM:
            raise AdapterGradientOwnershipError("Residual transport currently admits SUM reductions only")
        if not dist.is_available() or not dist.is_initialized():
            raise AdapterGradientOwnershipError("A pending reduction requires initialized distributed state")
        group = resolve_group(domain)
        if group is None:
            raise AdapterGradientOwnershipError(f"Pending reduction group {domain.group_key!r} is unavailable")
        for bucket_slices in _bounded_bucket_slices(sorted(fqns), gradients, bucket_bytes):
            packed = torch.cat([gradients[fqn].reshape(-1)[start:stop] for fqn, start, stop in bucket_slices])
            packed_bytes = packed.numel() * packed.element_size()
            bucket_count += 1
            largest_bucket_bytes = max(largest_bucket_bytes, packed_bytes)
            if dist.get_world_size(group=group) > 1:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
                collective_count += 1
                transported_bytes += packed_bytes
            offset = 0
            for fqn, start, stop in bucket_slices:
                width = stop - start
                gradients[fqn].reshape(-1)[start:stop].copy_(packed[offset : offset + width])
                offset += width

    return TransportCompleteLocalGradients(gradients), AdapterGradientTransportStats(
        collective_count=collective_count,
        bucket_count=bucket_count,
        input_bytes=input_bytes,
        transported_bytes=transported_bytes,
        largest_bucket_bytes=largest_bucket_bytes,
    )


def logical_l2_norm(
    plan: AdapterGradientOwnershipPlan,
    gradients: Mapping[str, torch.Tensor],
    *,
    world_group: Any = None,
) -> torch.Tensor:
    """Count replicated logical coordinates once and owner shards independently."""

    values = _canonical_tensors(gradients)
    if set(values) != {item.fqn for item in plan.parameters}:
        raise AdapterGradientOwnershipError("Transport-complete gradients differ from the compiled plan")
    first = next(iter(values.values()))
    local_sum_square = torch.zeros((), dtype=torch.float32, device=first.device)
    for item in plan.parameters:
        local_sum_square.add_(values[item.fqn].float().square().sum() / item.norm_replica_divisor)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group=world_group) > 1:
        dist.all_reduce(local_sum_square, op=dist.ReduceOp.SUM, group=world_group)
    return local_sum_square.sqrt()


__all__ = [
    "AdapterGradientCollectiveFailure",
    "AdapterGradientMutationFailure",
    "AdapterGradientTransportStats",
    "TransportCompleteLocalGradients",
    "logical_l2_norm",
    "transport_complete_local_gradients",
]
