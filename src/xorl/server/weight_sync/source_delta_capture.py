"""Trainer-side source sparse-delta capture helpers.

This module captures optimizer-step deltas in training/source-rank
coordinates. It deliberately stops before receiver translation: downstream
code can load the source-rank packed files, feed them through
``delta-encoding`` translation plans, and then upload receiver-rank packed
files through the sparse-delta sync API.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from xorl.server.weight_sync.sparse_delta_files import (
    SparseTensorUpdate,
    prepare_delta_encoding_runtime,
    write_sparse_delta_file,
)


FORMAT_VERSION = "xorl_sparse_source_delta_capture_v1"
GLOBAL_MANIFEST_VERSION = "xorl_sparse_source_delta_capture_manifest_v1"


try:
    from torch.distributed._tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False


_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def sparse_delta_capture_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether an optim_step sparse-delta capture request is active."""

    return bool(config) and bool(config.get("enabled", True))


def snapshot_sparse_delta_tensors(
    model: Any,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Copy selected local model parameters to CPU before an optimizer step."""

    dtype = _capture_dtype(config)
    snapshots: dict[str, torch.Tensor] = {}
    for name, param in _iter_selected_parameters(model, config):
        snapshots[name] = _local_tensor_cpu_copy(param, dtype=dtype)
    return snapshots


def write_sparse_source_delta_rank(
    *,
    model: Any,
    before: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    rank: int,
    world_size: int,
    model_id: str,
    step: int,
    snapshot_s: float = 0.0,
) -> dict[str, Any]:
    """Diff selected parameters against ``before`` and write one source-rank file."""

    t0 = time.perf_counter()
    dtype = _capture_dtype(config)
    output_dir = resolve_sparse_delta_capture_output_dir(config, model_id=model_id, step=step)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = dict(model.named_parameters())
    updates: list[SparseTensorUpdate] = []
    tensor_stats: list[dict[str, Any]] = []
    diff_s = 0.0
    for name, before_tensor in before.items():
        param = parameters.get(name)
        if param is None:
            raise KeyError(f"Parameter {name!r} disappeared before sparse-delta capture finalization")

        t_diff = time.perf_counter()
        after_tensor = _local_tensor_cpu_copy(param, dtype=dtype)
        update = _make_sparse_update(name, before_tensor, after_tensor)
        diff_s += time.perf_counter() - t_diff

        nnz = int(update.flat_indices.numel()) if update is not None else 0
        layout = _parameter_layout_metadata(param, after_tensor)
        tensor_stats.append(
            {
                "name": name,
                "shape": list(before_tensor.shape),
                "dtype": _dtype_name(before_tensor.dtype),
                "numel": int(before_tensor.numel()),
                "nnz": nnz,
                "layout": layout,
            }
        )
        if update is not None:
            updates.append(update)

        del after_tensor

    packed_path: str | None = None
    packed_bytes = 0
    write_s = 0.0
    if updates:
        t_write = time.perf_counter()
        packed_stats = write_sparse_delta_file(
            updates,
            output_dir / _rank_filename(config, rank=rank),
            delta_encoding_path=_delta_encoding_path(config),
            use_native_extension=bool(config.get("use_native_extension", False)),
        )
        write_s = time.perf_counter() - t_write
        packed_path = packed_stats.path
        packed_bytes = packed_stats.packed_bytes

    manifest = {
        "format": FORMAT_VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "model_id": model_id,
        "step": int(step),
        "capture_dtype": _dtype_name(dtype),
        "output_dir": str(output_dir),
        "packed_path": packed_path,
        "tensors": tensor_stats,
        "totals": {
            "tensors_considered": len(tensor_stats),
            "tensors_changed": sum(1 for stat in tensor_stats if int(stat["nnz"]) > 0),
            "nnz": sum(int(stat["nnz"]) for stat in tensor_stats),
            "packed_bytes": int(packed_bytes),
            "snapshot_s": float(snapshot_s),
            "diff_s": diff_s,
            "write_s": write_s,
            "total_s": time.perf_counter() - t0,
        },
    }

    manifest_path = output_dir / _rank_manifest_filename(config, rank=rank)
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def write_sparse_source_delta_global_manifest(
    rank_manifests: Iterable[Mapping[str, Any] | None],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a rank-0 manifest that points at all source-rank packed files."""

    ranks = [dict(manifest) for manifest in rank_manifests if manifest]
    if not ranks:
        raise ValueError("No rank sparse-delta capture manifests were provided")

    ranks.sort(key=lambda item: int(item["rank"]))
    if output_dir is None:
        output_dir = ranks[0].get("output_dir")
    if output_dir is None:
        raise ValueError("output_dir is required when rank manifests do not include one")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    packed_paths = [str(manifest["packed_path"]) for manifest in ranks if manifest.get("packed_path")]
    totals = {
        "ranks": len(ranks),
        "tensors_considered": sum(int(manifest["totals"]["tensors_considered"]) for manifest in ranks),
        "tensors_changed": sum(int(manifest["totals"]["tensors_changed"]) for manifest in ranks),
        "nnz": sum(int(manifest["totals"]["nnz"]) for manifest in ranks),
        "packed_bytes": sum(int(manifest["totals"]["packed_bytes"]) for manifest in ranks),
        "snapshot_s_max": max(float(manifest["totals"].get("snapshot_s", 0.0)) for manifest in ranks),
        "diff_s_max": max(float(manifest["totals"].get("diff_s", 0.0)) for manifest in ranks),
        "write_s_max": max(float(manifest["totals"].get("write_s", 0.0)) for manifest in ranks),
        "total_s_max": max(float(manifest["totals"].get("total_s", 0.0)) for manifest in ranks),
    }
    global_manifest = {
        "format": GLOBAL_MANIFEST_VERSION,
        "source_format": FORMAT_VERSION,
        "output_dir": str(output_path),
        "model_id": ranks[0].get("model_id"),
        "step": ranks[0].get("step"),
        "world_size": ranks[0].get("world_size"),
        "packed_paths": packed_paths,
        "ranks": ranks,
        "totals": totals,
    }

    manifest_path = output_path / "manifest.json"
    _write_json(manifest_path, global_manifest)
    global_manifest["manifest_path"] = str(manifest_path)
    return global_manifest


def load_sparse_source_delta_inputs(
    manifest_path: str | Path,
    *,
    delta_encoding_path: str | None = None,
    use_native_extension: bool = False,
    tag: str | None = "enc",
    include_empty: bool = False,
) -> list[tuple[Any, Any]]:
    """Load source-rank packed files as ``(StoreKey, EncodedDelta)`` inputs.

    The returned tensors are cloned out of the mmap so callers can close files
    before running the translation engine.
    When ``include_empty`` is true, the returned inputs also include encoded
    empty tensors for manifest entries with ``nnz == 0``. Translation plans for
    sharded source layouts need those empty shards to know the input is
    complete.
    """

    prepare_delta_encoding_runtime(
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )

    from delta_encoding.encoding.compression import encode  # noqa: PLC0415
    from delta_encoding.encoding.packed import MmapPackedFile  # noqa: PLC0415
    from delta_encoding.encoding.types import EncodedDelta  # noqa: PLC0415
    from delta_encoding.ops.types import StoreKey  # noqa: PLC0415

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("format") != GLOBAL_MANIFEST_VERSION:
        raise ValueError(f"Unsupported sparse source-delta manifest format: {manifest.get('format')!r}")

    inputs: list[tuple[Any, Any]] = []
    seen: set[tuple[int, str]] = set()
    for rank_manifest in manifest.get("ranks", []):
        packed_path = rank_manifest.get("packed_path")
        if not packed_path:
            continue
        rank = int(rank_manifest["rank"])
        with MmapPackedFile(packed_path) as packed:
            for entry in packed.entries:
                key = StoreKey(entry.name, rank=rank)
                if tag:
                    key = key.tag(tag)
                encoded = EncodedDelta(
                    packed.flat_deltas_view(entry).clone(),
                    packed.values_view(entry).clone(),
                    tuple(entry.shape),
                )
                inputs.append((key, encoded))
                seen.add((rank, entry.name))

    if include_empty:
        for rank_manifest in manifest.get("ranks", []):
            rank = int(rank_manifest["rank"])
            for tensor in rank_manifest.get("tensors", []):
                name = str(tensor["name"])
                if (rank, name) in seen or int(tensor.get("nnz", 0)) != 0:
                    continue
                key = StoreKey(name, rank=rank)
                if tag:
                    key = key.tag(tag)
                dtype = _dtype_from_name(tensor.get("dtype") or rank_manifest.get("capture_dtype") or "bfloat16")
                shape = tuple(int(dim) for dim in tensor["shape"])
                encoded = encode(
                    torch.empty(0, dtype=torch.int32),
                    torch.empty(0, dtype=dtype),
                    shape,
                )
                inputs.append((key, encoded))
    return inputs


def resolve_sparse_delta_capture_output_dir(
    config: Mapping[str, Any],
    *,
    model_id: str,
    step: int,
) -> Path:
    output_dir = config.get("output_dir")
    if output_dir:
        return Path(str(output_dir)).expanduser()
    safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    return Path(os.getcwd()) / "sparse_delta_capture" / safe_model_id / f"step-{step}"


def _make_sparse_update(name: str, before: torch.Tensor, after: torch.Tensor) -> SparseTensorUpdate | None:
    if before.shape != after.shape:
        raise ValueError(f"Parameter {name!r} shape changed during optimizer step: {before.shape} -> {after.shape}")
    if before.dtype != after.dtype:
        raise ValueError(f"Parameter {name!r} dtype changed during optimizer step: {before.dtype} -> {after.dtype}")
    if before.numel() > torch.iinfo(torch.int32).max:
        raise ValueError(
            f"Sparse-delta source capture requires per-rank tensors with <= int32 numel; "
            f"{name!r} has {before.numel()} elements"
        )

    before_flat = before.reshape(-1)
    after_flat = after.reshape(-1)
    changed = torch.ne(before_flat, after_flat)
    if not bool(changed.any().item()):
        return None

    flat_indices = changed.nonzero(as_tuple=False).flatten().to(torch.int32)
    values = after_flat.index_select(0, flat_indices.to(torch.int64)).contiguous()
    return SparseTensorUpdate(
        name=name,
        flat_indices=flat_indices.contiguous(),
        values=values,
        shape=tuple(int(dim) for dim in after.shape),
    )


def _iter_selected_parameters(model: Any, config: Mapping[str, Any]):
    include = _compile_patterns(config.get("include") or config.get("include_patterns"))
    exclude = _compile_patterns(config.get("exclude") or config.get("exclude_patterns"))
    only_trainable = bool(config.get("only_trainable", True))
    max_tensors = config.get("max_tensors")
    max_numel = config.get("max_numel_per_tensor", config.get("max_numel"))
    selected = 0

    for name, param in model.named_parameters():
        if only_trainable and not bool(getattr(param, "requires_grad", False)):
            continue
        if include and not any(pattern.search(name) for pattern in include):
            continue
        if exclude and any(pattern.search(name) for pattern in exclude):
            continue
        local = _local_tensor_view(param)
        if max_numel is not None and int(local.numel()) > int(max_numel):
            continue
        yield name, param
        selected += 1
        if max_tensors is not None and selected >= int(max_tensors):
            break


def _local_tensor_cpu_copy(param: Any, *, dtype: torch.dtype) -> torch.Tensor:
    local = _local_tensor_view(param).detach()
    return local.to(device="cpu", dtype=dtype, copy=True).contiguous()


def _local_tensor_view(param: Any) -> torch.Tensor:
    dtensor = _as_dtensor(param)
    if dtensor is not None:
        return dtensor.to_local()
    return param


def _as_dtensor(param: Any) -> Any | None:
    if not _HAS_DTENSOR:
        return None
    if isinstance(param, DTensor):
        return param
    data = getattr(param, "data", None)
    if isinstance(data, DTensor):
        return data
    return None


def _parameter_layout_metadata(param: Any, local: torch.Tensor) -> dict[str, Any]:
    dtensor = _as_dtensor(param)
    if dtensor is None:
        return {
            "type": "local",
            "global_shape": [int(dim) for dim in local.shape],
            "local_shape": [int(dim) for dim in local.shape],
            "global_offsets": [0 for _ in local.shape],
            "placements": [],
        }

    global_shape = tuple(int(dim) for dim in dtensor.shape)
    placements = tuple(dtensor.placements)
    mesh = dtensor.device_mesh
    try:
        local_shape, global_offsets = compute_local_shape_and_global_offset(global_shape, mesh, placements)
    except Exception as exc:
        local_shape = tuple(int(dim) for dim in local.shape)
        global_offsets = tuple(0 for _ in local_shape)
        layout_error = repr(exc)
    else:
        layout_error = None

    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is not None:
        mesh_shape = [int(dim) for dim in mesh_tensor.shape]
    else:
        mesh_shape = []
    coordinate = mesh.get_coordinate()
    metadata = {
        "type": "dtensor",
        "global_shape": list(global_shape),
        "local_shape": [int(dim) for dim in local_shape],
        "actual_local_shape": [int(dim) for dim in local.shape],
        "global_offsets": [int(dim) for dim in global_offsets],
        "placements": [repr(placement) for placement in placements],
        "device_mesh_shape": mesh_shape,
        "device_mesh_dim_names": list(mesh.mesh_dim_names or []),
        "device_mesh_coordinate": [int(dim) for dim in coordinate] if coordinate is not None else None,
    }
    if layout_error is not None:
        metadata["layout_error"] = layout_error
    return metadata


def _capture_dtype(config: Mapping[str, Any]) -> torch.dtype:
    value = config.get("dtype", config.get("capture_dtype", "bfloat16"))
    return _dtype_from_name(value)


def _dtype_from_name(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None or str(value).lower() in {"param", "parameter"}:
        raise ValueError("sparse_delta_capture currently requires an explicit CPU capture dtype")
    key = str(value).lower().replace("torch.", "")
    try:
        return _DTYPES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported sparse_delta_capture dtype {value!r}") from exc


def _compile_patterns(value: Any) -> list[re.Pattern[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    return [re.compile(str(pattern)) for pattern in values]


def _delta_encoding_path(config: Mapping[str, Any]) -> str | None:
    path = config.get("delta_encoding_path") or os.environ.get("XORL_DELTA_ENCODING_PATH")
    return str(path) if path else None


def _rank_filename(config: Mapping[str, Any], *, rank: int) -> str:
    template = str(config.get("filename_template", "rank{rank}.packed"))
    return template.format(rank=int(rank))


def _rank_manifest_filename(config: Mapping[str, Any], *, rank: int) -> str:
    template = str(config.get("manifest_filename_template", "rank{rank}.manifest.json"))
    return template.format(rank=int(rank))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
