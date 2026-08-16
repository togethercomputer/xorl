"""Immutable GLM-5.2 layer and IndexShare ownership plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum


class IndexerType(str, Enum):
    FULL = "full"
    SHARED = "shared"


class MLPType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True)
class Glm52LayerSpec:
    layer_index: int
    indexer_type: IndexerType
    index_producer_layer: int
    mlp_type: MLPType


@dataclass(frozen=True)
class Glm52LayerPlan:
    layers: tuple[Glm52LayerSpec, ...]
    index_topk_freq: int
    index_skip_topk_offset: int
    index_topk_pattern: tuple[int, ...] | None
    pipeline_layer_ranges: tuple[tuple[int, int], ...]

    @classmethod
    def from_config(
        cls,
        config,
        *,
        pipeline_layer_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> Glm52LayerPlan:
        num_layers = int(config.num_hidden_layers)
        raw_indexer_types = getattr(config, "indexer_types", None)
        raw_mlp_types = getattr(config, "mlp_layer_types", None)
        if raw_indexer_types is None:
            raise ValueError("GLM-5.2 requires the checkpoint's indexer_types schedule")
        if raw_mlp_types is None:
            raise ValueError("GLM-5.2 requires the checkpoint's mlp_layer_types schedule")
        if len(raw_indexer_types) != num_layers:
            raise ValueError(f"indexer_types has length {len(raw_indexer_types)}, expected {num_layers}")
        if len(raw_mlp_types) != num_layers:
            raise ValueError(f"mlp_layer_types has length {len(raw_mlp_types)}, expected {num_layers}")

        index_topk_freq = int(getattr(config, "index_topk_freq", 0))
        index_skip_topk_offset = int(getattr(config, "index_skip_topk_offset", -1))
        raw_pattern = getattr(config, "index_topk_pattern", None)
        index_topk_pattern = None if raw_pattern is None else tuple(int(value) for value in raw_pattern)
        if index_topk_freq <= 0:
            raise ValueError("index_topk_freq must be positive")
        if index_skip_topk_offset < 0:
            raise ValueError("index_skip_topk_offset must be nonnegative")
        if index_skip_topk_offset > num_layers:
            raise ValueError("index_skip_topk_offset cannot exceed num_hidden_layers")
        if index_topk_pattern is not None:
            if len(index_topk_pattern) != num_layers or any(value not in (0, 1) for value in index_topk_pattern):
                raise ValueError("index_topk_pattern must be a 0/1 entry for every hidden layer")

        ranges = pipeline_layer_ranges or ((0, num_layers),)
        cls._validate_pipeline_ranges(ranges, num_layers)
        stage_by_layer: dict[int, tuple[int, int]] = {}
        for stage_range in ranges:
            for layer_index in range(*stage_range):
                stage_by_layer[layer_index] = stage_range

        layers: list[Glm52LayerSpec] = []
        preceding_full: int | None = None
        for layer_index, (raw_indexer, raw_mlp) in enumerate(zip(raw_indexer_types, raw_mlp_types, strict=True)):
            try:
                indexer_type = IndexerType(raw_indexer)
            except ValueError as error:
                raise ValueError(f"Unknown indexer_types value at layer {layer_index}: {raw_indexer!r}") from error
            try:
                mlp_type = MLPType(raw_mlp)
            except ValueError as error:
                raise ValueError(f"Unknown mlp_layer_types value at layer {layer_index}: {raw_mlp!r}") from error

            if index_topk_pattern is None:
                expected_full = (
                    layer_index < index_skip_topk_offset
                    or (layer_index - index_skip_topk_offset + 1) % index_topk_freq == 0
                )
            else:
                expected_full = bool(index_topk_pattern[layer_index])
            expected_indexer = IndexerType.FULL if expected_full else IndexerType.SHARED
            if indexer_type is not expected_indexer:
                raise ValueError(
                    f"indexer_types[{layer_index}]={indexer_type.value!r} disagrees with "
                    "index_topk_freq/index_skip_topk_offset/index_topk_pattern"
                )

            stage_start, _ = stage_by_layer[layer_index]
            if layer_index == stage_start:
                preceding_full = None
            if indexer_type is IndexerType.FULL:
                preceding_full = layer_index
            elif preceding_full is None:
                raise ValueError(
                    f"Pipeline stage beginning at layer {stage_start} starts with shared-index layer {layer_index}"
                )

            layers.append(
                Glm52LayerSpec(
                    layer_index=layer_index,
                    indexer_type=indexer_type,
                    index_producer_layer=preceding_full,
                    mlp_type=mlp_type,
                )
            )

        return cls(
            layers=tuple(layers),
            index_topk_freq=index_topk_freq,
            index_skip_topk_offset=index_skip_topk_offset,
            index_topk_pattern=index_topk_pattern,
            pipeline_layer_ranges=ranges,
        )

    @staticmethod
    def _validate_pipeline_ranges(ranges: tuple[tuple[int, int], ...], num_layers: int) -> None:
        expected_start = 0
        for start, end in ranges:
            if start != expected_start or end <= start:
                raise ValueError("Pipeline layer ranges must be positive, contiguous, and start at layer 0")
            expected_start = end
        if expected_start != num_layers:
            raise ValueError(f"Pipeline layer ranges must cover exactly {num_layers} layers")

    @property
    def full_indexer_layers(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers if layer.indexer_type is IndexerType.FULL)

    @property
    def shared_indexer_layers(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers if layer.indexer_type is IndexerType.SHARED)

    @property
    def dense_layers(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers if layer.mlp_type is MLPType.DENSE)

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers if layer.mlp_type is MLPType.SPARSE)

    @property
    def identity(self) -> str:
        payload = {
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "indexer_type": layer.indexer_type.value,
                    "index_producer_layer": layer.index_producer_layer,
                    "mlp_type": layer.mlp_type.value,
                }
                for layer in self.layers
            ],
            "index_topk_freq": self.index_topk_freq,
            "index_skip_topk_offset": self.index_skip_topk_offset,
            "index_topk_pattern": self.index_topk_pattern,
            "pipeline_layer_ranges": self.pipeline_layer_ranges,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def derive_glm52_pipeline_layer_ranges(config, pp_size: int) -> tuple[tuple[int, int], ...]:
    """Derive contiguous PP ranges whose stages begin on full indexers.

    GLM shares sparse-index state between neighboring layers, so a pipeline
    stage may begin only where that state is produced locally.  The checkpoint
    schedule is the source of truth; no PP-degree table is maintained here.
    """

    pp_size = int(pp_size)
    if pp_size <= 0:
        raise ValueError("pipeline_parallel_size must be positive")

    base_plan = Glm52LayerPlan.from_config(config)
    num_layers = len(base_plan.layers)
    legal_starts = base_plan.full_indexer_layers
    if pp_size > len(legal_starts):
        raise ValueError(
            f"GLM-5.2 PP{pp_size} requires {pp_size} full-index stage starts, "
            f"but the checkpoint schedule exposes only {len(legal_starts)}"
        )
    if pp_size == 1:
        return ((0, num_layers),)

    starts = [0]
    for cut_index in range(1, pp_size):
        remaining_cuts = pp_size - cut_index - 1
        candidates = [
            start
            for position, start in enumerate(legal_starts)
            if start > starts[-1] and len(legal_starts) - position - 1 >= remaining_cuts
        ]
        if not candidates:
            raise ValueError(f"GLM-5.2 cannot derive {pp_size} non-empty full-index pipeline stages")
        target = cut_index * num_layers / pp_size
        starts.append(min(candidates, key=lambda start: (abs(start - target), start)))

    ranges = tuple(zip(starts, (*starts[1:], num_layers), strict=True))
    # Re-run the model's state-ownership validator over the resolved cuts.
    Glm52LayerPlan.from_config(config, pipeline_layer_ranges=ranges)
    return ranges


def install_glm52_pipeline_module_plan(
    config,
    *,
    num_stages: int,
    input_weight: int = 1,
    output_weight: int = 1,
    num_layers_in_first_stage: int | None = None,
    num_layers_in_last_stage: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Resolve and install the real GLM pipeline module plan before construction.

    Input/output weighting can produce module-only edge stages.  IndexShare
    ownership therefore follows the number of stages that actually own decoder
    layers, not the physical PP degree.  The resolved FQN assignment is stored on
    the config and reused verbatim by ``build_parallelize_model``; it is not
    reconstructed after the GLM layers have already captured their layer plan.
    """

    from xorl.distributed.pipeline_parallel import generate_llm_fqn_per_model_part  # noqa: PLC0415

    layer_prefix = "model.layers"
    provisional = generate_llm_fqn_per_model_part(
        num_stages=int(num_stages),
        num_layers=int(config.num_hidden_layers),
        input_weight=int(input_weight),
        output_weight=int(output_weight),
        input_fqns=["model.embed_tokens"],
        layer_prefix=layer_prefix,
        output_fqns=["model.norm", "lm_head"],
        num_layers_in_first_stage=num_layers_in_first_stage,
        num_layers_in_last_stage=num_layers_in_last_stage,
    )
    decoder_slots = tuple(
        stage_index
        for stage_index, modules in enumerate(provisional)
        if any(name.startswith(f"{layer_prefix}.") for name in modules)
    )
    if not decoder_slots:
        raise ValueError("GLM pipeline module plan must assign at least one stage decoder layers")

    ranges = derive_glm52_pipeline_layer_ranges(config, len(decoder_slots))
    resolved = [[name for name in modules if not name.startswith(f"{layer_prefix}.")] for modules in provisional]
    for stage_index, (start, end) in zip(decoder_slots, ranges, strict=True):
        decoder_names = [f"{layer_prefix}.{layer_index}" for layer_index in range(start, end)]
        output_offset = next(
            (index for index, name in enumerate(resolved[stage_index]) if name in {"model.norm", "lm_head"}),
            len(resolved[stage_index]),
        )
        resolved[stage_index][output_offset:output_offset] = decoder_names

    module_plan = tuple(tuple(modules) for modules in resolved)
    config._glm52_pipeline_layer_ranges = ranges
    config._glm52_pipeline_decoder_stage_indices = decoder_slots
    config._glm52_pipeline_module_names_per_stage = module_plan
    return module_plan


__all__ = [
    "Glm52LayerPlan",
    "Glm52LayerSpec",
    "IndexerType",
    "MLPType",
    "derive_glm52_pipeline_layer_ranges",
    "install_glm52_pipeline_module_plan",
]
