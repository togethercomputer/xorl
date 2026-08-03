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


__all__ = ["Glm52LayerPlan", "Glm52LayerSpec", "IndexerType", "MLPType"]
