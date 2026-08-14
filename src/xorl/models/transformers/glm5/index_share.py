"""Per-invocation GLM-5.2 IndexShare state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

import torch

from xorl.models.transformers.glm5.layer_plan import Glm52LayerPlan, IndexerType


class IndexShareLifecycle(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class IndexShareMode(str, Enum):
    TRAINING_WITH_BACKWARD = "training_with_backward"
    FORWARD_ONLY = "forward_only"


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
        mode: IndexShareMode,
    ) -> None:
        self.invocation_id = invocation_id
        self.plan_identity = plan_identity
        self.owning_pipeline_stage = owning_pipeline_stage
        self.mode = mode
        self.lifecycle = IndexShareLifecycle.OPEN
        self._producer_payloads: dict[int, CanonicalLogicalIndices] = {}

    def _require_open(self) -> None:
        if self.lifecycle is not IndexShareLifecycle.OPEN:
            raise RuntimeError(f"IndexShareContext {self.invocation_id} is already closed")

    def get_or_publish(
        self,
        *,
        producer_layer_index: int,
        layer_plan: Glm52LayerPlan,
        produce_payload: Callable[[], torch.Tensor | CanonicalLogicalIndices],
    ) -> CanonicalLogicalIndices:
        """Return one invocation-owned producer payload, computing it at most once.

        Full-layer activation checkpointing calls the producer layer again during
        backward.  The retained detached payload is deliberately reused in that
        recomputation instead of rerunning the no-grad indexer or publishing a
        second value for the same logical producer.
        """

        self._require_open()
        self._validate_plan(layer_plan)
        spec = layer_plan.layers[producer_layer_index]
        if spec.indexer_type is not IndexerType.FULL:
            raise RuntimeError(f"Shared-index layer {producer_layer_index} cannot publish IndexShare state")
        if not self._owns_layer(producer_layer_index):
            raise RuntimeError(
                f"Layer {producer_layer_index} is outside owning pipeline stage {self.owning_pipeline_stage}"
            )
        payload = self._producer_payloads.get(producer_layer_index)
        if payload is not None:
            return payload

        produced = produce_payload()
        values = produced.values if isinstance(produced, CanonicalLogicalIndices) else produced
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"Full-indexer layer {producer_layer_index} producer returned {type(values).__name__}, expected Tensor"
            )
        payload = CanonicalLogicalIndices(values.detach())
        self._producer_payloads[producer_layer_index] = payload
        return payload

    def require(
        self,
        *,
        producer_layer_index: int,
        layer_plan: Glm52LayerPlan,
    ) -> CanonicalLogicalIndices:
        """Strictly require one already-published producer payload."""

        self._require_open()
        self._validate_plan(layer_plan)
        spec = layer_plan.layers[producer_layer_index]
        if spec.indexer_type is not IndexerType.FULL:
            raise RuntimeError(f"IndexShare producer layer {producer_layer_index} is not a full-indexer layer")
        if not self._owns_layer(producer_layer_index):
            raise RuntimeError(
                f"Layer {producer_layer_index} is outside owning pipeline stage {self.owning_pipeline_stage}"
            )
        if producer_layer_index not in self._producer_payloads:
            raise RuntimeError(f"IndexShare producer layer {producer_layer_index} has not published")
        return self._producer_payloads[producer_layer_index]

    def close(self) -> None:
        if self.lifecycle is IndexShareLifecycle.CLOSED:
            return
        self._producer_payloads.clear()
        self.lifecycle = IndexShareLifecycle.CLOSED

    def _validate_plan(self, layer_plan: Glm52LayerPlan) -> None:
        if self.plan_identity != layer_plan.identity:
            raise RuntimeError("IndexShareContext layer-plan identity does not match the executing model")

    def _owns_layer(self, layer_index: int) -> bool:
        start, end = self.owning_pipeline_stage
        return start <= layer_index < end


class IndexShareContextManager:
    """Own the in-flight IndexShare contexts for one GLM-5.2 stage.

    Pipeline schedules may issue several forwards before the matching
    backwards.  Each invocation therefore owns a distinct context, retained
    until the schedule has completed every backward (or aborts).  Checkpoint
    recomputation closes over that invocation's context object directly, so
    overlapping microbatches cannot observe one another's producer payloads.
    """

    def __init__(self, layer_plan: Glm52LayerPlan, owning_pipeline_stage: tuple[int, int]):
        if owning_pipeline_stage not in layer_plan.pipeline_layer_ranges:
            raise ValueError(f"Unknown pipeline stage {owning_pipeline_stage} for layer plan")
        self.layer_plan = layer_plan
        self.owning_pipeline_stage = owning_pipeline_stage
        self._active: dict[str, IndexShareContext] = {}
        self._invocation_counter = 0

    @property
    def active(self) -> IndexShareContext | None:
        """Return the sole active context, or the newest one when several overlap.

        This compatibility view is useful for diagnostics.  Lifecycle code
        must use ``active_contexts``/``end_all`` because a PP stage can have
        more than one training microbatch in flight.
        """

        invocation_id = next(reversed(self._active), None)
        return None if invocation_id is None else self._active[invocation_id]

    @property
    def active_contexts(self) -> tuple[IndexShareContext, ...]:
        return tuple(self._active.values())

    def begin(self, *, mode: IndexShareMode | str) -> IndexShareContext:
        try:
            mode = IndexShareMode(mode)
        except ValueError as exc:
            choices = ", ".join(item.value for item in IndexShareMode)
            raise ValueError(f"Unknown IndexShare mode {mode!r}; expected one of: {choices}") from exc
        invocation_id = f"{self.layer_plan.identity[:12]}:{self._invocation_counter}"
        self._invocation_counter += 1
        context = IndexShareContext(
            invocation_id=invocation_id,
            plan_identity=self.layer_plan.identity,
            owning_pipeline_stage=self.owning_pipeline_stage,
            mode=mode,
        )
        self._active[invocation_id] = context
        return context

    def end(self, context: IndexShareContext) -> None:
        active = self._active.get(context.invocation_id)
        if context is active:
            context.close()
            del self._active[context.invocation_id]
            return
        if active is None and context.lifecycle is IndexShareLifecycle.CLOSED:
            return
        raise RuntimeError("Attempted to close a stale or foreign IndexShareContext")

    def end_active(self) -> None:
        """Idempotently release the newest active invocation, if any."""

        context = self.active
        if context is not None:
            self.end(context)

    def end_all(self) -> None:
        """Release every invocation retained for backward by this stage."""

        for context in tuple(self._active.values()):
            self.end(context)

    def finish_forward(self, context: IndexShareContext, *, succeeded: bool) -> None:
        """Apply the model-owned half of the explicit lifecycle contract."""

        if not succeeded or context.mode is IndexShareMode.FORWARD_ONLY:
            self.end(context)

    @contextmanager
    def invocation(self, *, mode: IndexShareMode | str) -> Iterator[IndexShareContext]:
        context = self.begin(mode=mode)
        succeeded = False
        try:
            yield context
            succeeded = True
        finally:
            self.finish_forward(context, succeeded=succeeded)


__all__ = [
    "CanonicalLogicalIndices",
    "IndexShareContext",
    "IndexShareContextManager",
    "IndexShareLifecycle",
    "IndexShareMode",
]
