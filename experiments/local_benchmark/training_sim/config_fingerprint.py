"""Resolve the subset of XoRL config state needed by the simulator."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


try:
    from .model_metadata import resolve_model_metadata
    from .schemas import RunFingerprint, Topology
except ImportError:  # pragma: no cover - exercised by direct script execution
    from model_metadata import resolve_model_metadata
    from schemas import RunFingerprint, Topology


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def config_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_commit(repo_root: str | Path = REPO_ROOT) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}


def _int_value(section: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = section.get(key, default)
    if value is None:
        return None
    return int(value)


def _find_int(sections: list[dict[str, Any]], keys: tuple[str, ...]) -> int | None:
    for section in sections:
        for key in keys:
            if key in section and section[key] is not None:
                return int(section[key])
    return None


def _infer_world_size(
    train: dict[str, Any],
    *,
    non_dp_size: int,
    world_size: int | None,
) -> int:
    if world_size is not None:
        return int(world_size)
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    replicate_size = _int_value(train, "data_parallel_replicate_size", -1) or -1
    shard_size = _int_value(train, "data_parallel_shard_size", -1) or -1
    if replicate_size > 0 and shard_size > 0:
        return replicate_size * shard_size * non_dp_size
    return 1


def _infer_local_world_size(*, local_world_size: int | None, world_size: int) -> int:
    if local_world_size is not None:
        return int(local_world_size)
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size > 8 and world_size % 8 == 0:
        return 8
    return world_size


def _resolve_dp_split(data_parallel_size: int, replicate_size: int, shard_size: int) -> tuple[int, int]:
    if replicate_size > 0 and shard_size > 0:
        if data_parallel_size != replicate_size * shard_size:
            raise ValueError(
                f"data_parallel_size ({data_parallel_size}) should equal "
                f"data_parallel_replicate_size ({replicate_size}) * data_parallel_shard_size ({shard_size})."
            )
        return replicate_size, shard_size

    if replicate_size > 0:
        if data_parallel_size % replicate_size != 0:
            raise ValueError("data_parallel_size should be a multiple of data_parallel_replicate_size.")
        return replicate_size, data_parallel_size // replicate_size

    if shard_size > 0:
        if data_parallel_size % shard_size != 0:
            raise ValueError("data_parallel_size should be a multiple of data_parallel_shard_size.")
        return data_parallel_size // shard_size, shard_size

    return 1, data_parallel_size


def resolve_topology(
    raw_config: dict[str, Any],
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    num_experts: int | None = None,
    top_k: int | None = None,
) -> Topology:
    train = _section(raw_config, "train")
    data = _section(raw_config, "data")

    ulysses = _int_value(train, "ulysses_parallel_size", 1) or 1
    ringattn = _int_value(train, "ringattn_parallel_size", 1) or 1
    tensor_parallel = _int_value(train, "tensor_parallel_size", 1) or 1
    pipeline_parallel = _int_value(train, "pipeline_parallel_size", 1) or 1
    expert_parallel = _int_value(train, "expert_parallel_size", 1) or 1
    non_dp_size = ulysses * ringattn * tensor_parallel * pipeline_parallel
    resolved_world_size = _infer_world_size(train, non_dp_size=non_dp_size, world_size=world_size)
    resolved_local_world_size = _infer_local_world_size(
        local_world_size=local_world_size, world_size=resolved_world_size
    )
    if resolved_world_size <= 0 or resolved_local_world_size <= 0:
        raise ValueError("world_size and local_world_size must be positive")
    if resolved_world_size % resolved_local_world_size != 0:
        raise ValueError("world_size must be divisible by local_world_size")
    if resolved_world_size % non_dp_size != 0:
        raise ValueError(
            f"world_size ({resolved_world_size}) must be divisible by ulysses ({ulysses}) * ringattn "
            f"({ringattn}) * tensor_parallel ({tensor_parallel}) * pipeline_parallel ({pipeline_parallel})."
        )

    data_parallel_size = resolved_world_size // non_dp_size
    ranks_per_pipeline_stage = resolved_world_size // pipeline_parallel
    ep_fsdp_size = (
        ranks_per_pipeline_stage // expert_parallel if ranks_per_pipeline_stage % expert_parallel == 0 else None
    )
    replicate_size = _int_value(train, "data_parallel_replicate_size", -1) or -1
    shard_size = _int_value(train, "data_parallel_shard_size", -1) or -1
    replicate_size, shard_size = _resolve_dp_split(data_parallel_size, replicate_size, shard_size)

    micro_batch_size = _int_value(train, "micro_batch_size", 1) or 1
    gradient_accumulation_steps = _int_value(train, "gradient_accumulation_steps", 1) or 1
    global_batch_size = micro_batch_size * gradient_accumulation_steps * data_parallel_size
    sample_packing_sequence_len = _int_value(data, "sample_packing_sequence_len", 32000)

    model_metadata = resolve_model_metadata(raw_config, num_experts=num_experts, top_k=top_k)

    return Topology(
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        node_count=resolved_world_size // resolved_local_world_size,
        data_parallel_size=data_parallel_size,
        data_parallel_replicate_size=replicate_size,
        data_parallel_shard_size=shard_size,
        tensor_parallel_size=tensor_parallel,
        pipeline_parallel_size=pipeline_parallel,
        expert_parallel_size=expert_parallel,
        ep_fsdp_size=ep_fsdp_size,
        ulysses_parallel_size=ulysses,
        ringattn_parallel_size=ringattn,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_batch_size=global_batch_size,
        sample_packing_sequence_len=sample_packing_sequence_len,
        num_experts=model_metadata.num_experts,
        top_k=model_metadata.top_k,
    )


def build_fingerprint(
    config_path: str | Path,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    balanced_routing: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> RunFingerprint:
    path = Path(config_path)
    raw_config = load_training_config(path)
    model_metadata = resolve_model_metadata(raw_config, num_experts=num_experts, top_k=top_k)
    topology = resolve_topology(
        raw_config,
        world_size=world_size,
        local_world_size=local_world_size,
        num_experts=model_metadata.num_experts,
        top_k=model_metadata.top_k,
    )
    return RunFingerprint(
        config_path=str(path),
        config_sha256=config_sha256(path),
        config_name=path.name,
        repo_commit=repo_commit(repo_root),
        balanced_routing=balanced_routing,
        topology=topology,
        model_metadata=model_metadata,
    )
