"""Per-invocation GLM-5.2 IndexShare state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import torch

from xorl.models.transformers.glm5.layer_plan import Glm52LayerPlan, IndexerType


class IndexShareLifecycle(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class CanonicalLogicalIndices:
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.dtype not in (torch.int32, torch.int64):
            raise TypeError("Canonical logical selected indices must use int32 or int64")
        if self.values.ndim < 2:
            raise ValueError("Canonical logical selected indices must include row and selection dimensions")
        if bool(torch.any(self.values < -1)):
            raise ValueError("Canonical logical indices admit only nonnegative positions and -1 sentinels")


class IndexShareContext:
    """Mutable state shared by every decoder layer in one model invocation.

    This intentionally is not a dataclass. FSDP mixed-precision input casting
    recursively rebuilds dataclass kwargs, which would give each wrapped
    decoder layer a distinct copy of this mutable context. Keeping the context
    as an ordinary object preserves its identity across FSDP layer boundaries.
    """

    def __init__(
        self,
        *,
        invocation_id: str,
        plan_identity: str,
        owning_pipeline_stage: tuple[int, int],
    ) -> None:
        self.invocation_id = invocation_id
        self.plan_identity = plan_identity
        self.owning_pipeline_stage = owning_pipeline_stage
        self.lifecycle = IndexShareLifecycle.OPEN
        self._producer_payloads: dict[int, CanonicalLogicalIndices] = {}

    def _require_open(self) -> None:
        if self.lifecycle is not IndexShareLifecycle.OPEN:
            raise RuntimeError(f"IndexShareContext {self.invocation_id} is already closed")

    def publish(
        self,
        *,
        layer_index: int,
        layer_plan: Glm52LayerPlan,
        indices: CanonicalLogicalIndices,
    ) -> None:
        self._require_open()
        self._validate_plan(layer_plan)
        spec = layer_plan.layers[layer_index]
        if spec.indexer_type is not IndexerType.FULL:
            raise RuntimeError(f"Shared-index layer {layer_index} cannot publish IndexShare state")
        if not self._owns_layer(layer_index):
            raise RuntimeError(f"Layer {layer_index} is outside owning pipeline stage {self.owning_pipeline_stage}")
        if layer_index in self._producer_payloads:
            raise RuntimeError(f"Full-indexer layer {layer_index} published twice in one invocation")
        self._producer_payloads[layer_index] = indices

    def consume(self, *, layer_index: int, layer_plan: Glm52LayerPlan) -> CanonicalLogicalIndices:
        self._require_open()
        self._validate_plan(layer_plan)
        spec = layer_plan.layers[layer_index]
        if spec.indexer_type is not IndexerType.SHARED:
            raise RuntimeError(f"Full-indexer layer {layer_index} must compute rather than consume shared indices")
        if not self._owns_layer(layer_index):
            raise RuntimeError(f"Layer {layer_index} is outside owning pipeline stage {self.owning_pipeline_stage}")
        producer = spec.index_producer_layer
        if producer not in self._producer_payloads:
            raise RuntimeError(
                f"Shared-index layer {layer_index} requires producer layer {producer}, but it has not published"
            )
        return self._producer_payloads[producer]

    def close(self) -> None:
        self._producer_payloads.clear()
        self.lifecycle = IndexShareLifecycle.CLOSED

    def _validate_plan(self, layer_plan: Glm52LayerPlan) -> None:
        if self.plan_identity != layer_plan.identity:
            raise RuntimeError("IndexShareContext layer-plan identity does not match the executing model")

    def _owns_layer(self, layer_index: int) -> bool:
        start, end = self.owning_pipeline_stage
        return start <= layer_index < end


class IndexShareContextManager:
    """One-live-context manager for the GLM-5.2 forward invocation."""

    def __init__(self, layer_plan: Glm52LayerPlan, owning_pipeline_stage: tuple[int, int]):
        if owning_pipeline_stage not in layer_plan.pipeline_layer_ranges:
            raise ValueError(f"Unknown pipeline stage {owning_pipeline_stage} for layer plan")
        self.layer_plan = layer_plan
        self.owning_pipeline_stage = owning_pipeline_stage
        self._active: IndexShareContext | None = None
        self._invocation_counter = 0

    @property
    def active(self) -> IndexShareContext | None:
        return self._active

    def begin(self) -> IndexShareContext:
        if self._active is not None:
            raise RuntimeError("GLM-5.2 canonical path supports only one live IndexShareContext per pipeline stage")
        invocation_id = f"{self.layer_plan.identity[:12]}:{self._invocation_counter}"
        self._invocation_counter += 1
        self._active = IndexShareContext(
            invocation_id=invocation_id,
            plan_identity=self.layer_plan.identity,
            owning_pipeline_stage=self.owning_pipeline_stage,
        )
        return self._active

    def end(self, context: IndexShareContext) -> None:
        if context is not self._active:
            raise RuntimeError("Attempted to close a stale or foreign IndexShareContext")
        context.close()
        self._active = None

    @contextmanager
    def invocation(self) -> Iterator[IndexShareContext]:
        context = self.begin()
        try:
            yield context
        finally:
            self.end(context)


__all__ = [
    "CanonicalLogicalIndices",
    "IndexShareContext",
    "IndexShareContextManager",
    "IndexShareLifecycle",
]
