"""Helpers for writing prepacked sparse-delta files.

This module is the training-side boundary for the fast sparse-delta API path:
callers provide already sparse, inference-coordinate tensor updates and get a
packed file that can be passed to ``sync_inference_weights(sparse_delta_paths=...)``.
It intentionally does not inspect trainer modules or FSDP state.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class SparseTensorUpdate:
    """Sparse absolute-value update for one inference tensor."""

    name: str
    flat_indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SparseDeltaFileStats:
    """Stats for one packed sparse-delta file."""

    path: str
    tensors: int
    nnz: int
    packed_bytes: int


@dataclass(frozen=True)
class SparseTensorShard:
    """Receiver-local shard of a logical sparse tensor."""

    rank: int
    name: str
    shape: tuple[int, ...]
    slices: tuple[tuple[int, int], ...]


def _load_delta_encoding(
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> tuple[Any, Any]:
    prepare_delta_encoding_runtime(
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )

    compression = importlib.import_module("delta_encoding.encoding.compression")
    packed = importlib.import_module("delta_encoding.encoding.packed")
    return compression.encode, packed.write_packed_file


def prepare_delta_encoding_runtime(
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> None:
    """Configure imports for optional ``delta-encoding`` runtime use."""

    if delta_encoding_path:
        resolved = str(Path(delta_encoding_path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    if not use_native_extension:
        sys.modules["delta_encoding.encoding._escape_ext"] = None


def _validate_update(update: SparseTensorUpdate) -> None:
    if not update.name:
        raise ValueError("Sparse tensor update name must be non-empty")
    if not update.shape:
        raise ValueError(f"Sparse tensor update {update.name!r} must have a non-empty shape")
    if any(dim < 0 for dim in update.shape):
        raise ValueError(f"Sparse tensor update {update.name!r} has invalid shape {update.shape}")

    numel = 1
    for dim in update.shape:
        numel *= dim
    if numel > 2**31 - 1:
        raise ValueError(
            f"Sparse-delta packed format requires int32 flat indices, but tensor {update.name!r} has {numel} elements"
        )

    if update.flat_indices.ndim != 1:
        raise ValueError(f"Sparse tensor update {update.name!r} flat_indices must be 1D")
    if (
        update.flat_indices.dtype == torch.bool
        or update.flat_indices.is_floating_point()
        or update.flat_indices.is_complex()
    ):
        raise ValueError(f"Sparse tensor update {update.name!r} flat_indices must use an integer dtype")
    if update.values.ndim != 1:
        raise ValueError(f"Sparse tensor update {update.name!r} values must be 1D")
    if update.flat_indices.numel() != update.values.numel():
        raise ValueError(
            f"Sparse tensor update {update.name!r} has {update.flat_indices.numel()} indices but "
            f"{update.values.numel()} values"
        )

    if update.flat_indices.numel() == 0:
        return
    indices_cpu = update.flat_indices.detach().to("cpu")
    min_index = int(indices_cpu.min().item())
    max_index = int(indices_cpu.max().item())
    if min_index < 0 or max_index >= numel:
        raise ValueError(
            f"Sparse tensor update {update.name!r} flat indices out of range for shape {update.shape}: "
            f"min={min_index}, max={max_index}, numel={numel}"
        )


def _sorted_cpu_update(update: SparseTensorUpdate) -> SparseTensorUpdate:
    indices = update.flat_indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    values = update.values.detach().to(device="cpu").contiguous()
    if indices.numel() <= 1:
        return SparseTensorUpdate(update.name, indices.to(torch.int32), values, update.shape)

    order = torch.argsort(indices, stable=True)
    indices = indices[order]
    values = values[order]
    if bool(torch.any(indices[1:] == indices[:-1]).item()):
        raise ValueError(f"Sparse tensor update {update.name!r} has duplicate flat indices")
    return SparseTensorUpdate(update.name, indices.to(torch.int32), values, update.shape)


def write_sparse_delta_file(
    updates: Iterable[SparseTensorUpdate],
    path: str | Path,
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> SparseDeltaFileStats:
    """Write sparse tensor updates to a packed sparse-delta file.

    Values are absolute receiver-side tensor values, not additive increments.
    Tensor names and flat indices must already be in inference coordinates.
    """

    updates = list(updates)
    if not updates:
        raise ValueError("write_sparse_delta_file requires at least one sparse tensor update")

    encode_fn, write_packed_file = _load_delta_encoding(
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )

    encoded: dict[str, Any] = {}
    total_nnz = 0
    for update in updates:
        _validate_update(update)
        sorted_update = _sorted_cpu_update(update)
        indices = sorted_update.flat_indices
        values = sorted_update.values
        encoded[update.name] = encode_fn(indices, values, tuple(update.shape))
        total_nnz += int(indices.numel())

    written = Path(write_packed_file(encoded, path))
    return SparseDeltaFileStats(
        path=str(written),
        tensors=len(encoded),
        nnz=total_nnz,
        packed_bytes=written.stat().st_size,
    )


def make_contiguous_shards(
    *,
    name: str,
    shape: Sequence[int],
    shard_dim: int,
    num_shards: int,
    shard_sizes: Sequence[int] | None = None,
) -> list[SparseTensorShard]:
    """Describe uniform contiguous receiver shards for one logical tensor."""

    shape = tuple(int(dim) for dim in shape)
    if not shape:
        raise ValueError("shape must be non-empty")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_dim < 0:
        shard_dim += len(shape)
    if shard_dim < 0 or shard_dim >= len(shape):
        raise ValueError(f"shard_dim {shard_dim} is out of range for shape {shape}")

    full = shape[shard_dim]
    if shard_sizes is None:
        if full % num_shards != 0:
            raise ValueError(
                f"shape {shape} is not evenly divisible into {num_shards} shards along dim {shard_dim}; "
                "pass explicit shard_sizes"
            )
        shard_sizes = [full // num_shards] * num_shards
    else:
        shard_sizes = [int(size) for size in shard_sizes]
        if len(shard_sizes) != num_shards:
            raise ValueError(f"expected {num_shards} shard sizes, got {len(shard_sizes)}")
        if any(size < 0 for size in shard_sizes):
            raise ValueError(f"shard sizes must be non-negative, got {shard_sizes}")
        if sum(shard_sizes) != full:
            raise ValueError(f"shard sizes sum to {sum(shard_sizes)}, expected {full}")

    shards: list[SparseTensorShard] = []
    start = 0
    for rank, size in enumerate(shard_sizes):
        stop = start + size
        local_shape = list(shape)
        local_shape[shard_dim] = size
        slices = [(0, dim) for dim in shape]
        slices[shard_dim] = (start, stop)
        shards.append(
            SparseTensorShard(
                rank=rank,
                name=name,
                shape=tuple(local_shape),
                slices=tuple(slices),
            )
        )
        start = stop
    return shards


def split_sparse_update_by_shards(
    update: SparseTensorUpdate,
    shards: Sequence[SparseTensorShard],
) -> dict[int, SparseTensorUpdate]:
    """Convert one logical sparse tensor update into receiver-local updates."""

    _validate_update(update)
    if not shards:
        raise ValueError("split_sparse_update_by_shards requires at least one shard")

    logical_shape = tuple(update.shape)
    for shard in shards:
        if len(shard.shape) != len(logical_shape) or len(shard.slices) != len(logical_shape):
            raise ValueError(
                f"Shard rank {shard.rank} rank mismatch: update shape={logical_shape}, "
                f"shard shape={shard.shape}, slices={shard.slices}"
            )
        for dim, ((start, stop), local_dim, full_dim) in enumerate(zip(shard.slices, shard.shape, logical_shape)):
            if start < 0 or stop < start or stop > full_dim:
                raise ValueError(f"Shard rank {shard.rank} has invalid slice {start, stop} for dim {dim}")
            if stop - start != local_dim:
                raise ValueError(
                    f"Shard rank {shard.rank} local shape {shard.shape} does not match slice {shard.slices}"
                )

    sorted_update = _sorted_cpu_update(update)
    flat = sorted_update.flat_indices.to(torch.int64)
    values = sorted_update.values
    coords = _flat_to_coords(flat, logical_shape)

    by_rank: dict[int, SparseTensorUpdate] = {}
    for shard in shards:
        mask = torch.ones(flat.numel(), dtype=torch.bool)
        local_coords: list[torch.Tensor] = []
        for coord, (start, stop) in zip(coords, shard.slices):
            mask &= (coord >= start) & (coord < stop)
            local_coords.append(coord - start)

        if bool(mask.any().item()):
            selected_coords = [coord[mask] for coord in local_coords]
            local_flat = _coords_to_flat(selected_coords, shard.shape)
            local_values = values[mask]
            if local_flat.numel() > 1:
                order = torch.argsort(local_flat, stable=True)
                local_flat = local_flat[order]
                local_values = local_values[order]
        else:
            local_flat = torch.empty(0, dtype=torch.int32)
            local_values = values[:0]

        by_rank[int(shard.rank)] = SparseTensorUpdate(
            name=shard.name,
            flat_indices=local_flat.to(torch.int32),
            values=local_values.contiguous(),
            shape=tuple(shard.shape),
        )
    return by_rank


def split_sparse_update_by_contiguous_shards(
    update: SparseTensorUpdate,
    *,
    shard_dim: int,
    num_shards: int,
    output_name: str | None = None,
    shard_sizes: Sequence[int] | None = None,
) -> dict[int, SparseTensorUpdate]:
    """Split a logical update into per-rank contiguous local coordinates."""

    shards = make_contiguous_shards(
        name=output_name or update.name,
        shape=update.shape,
        shard_dim=shard_dim,
        num_shards=num_shards,
        shard_sizes=shard_sizes,
    )
    return split_sparse_update_by_shards(update, shards)


def write_sparse_delta_files_by_rank(
    updates_by_rank: Mapping[int, Iterable[SparseTensorUpdate]],
    output_dir: str | Path,
    *,
    filename_template: str = "rank{rank}.packed",
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> dict[int, SparseDeltaFileStats]:
    """Write rank-local sparse updates as one packed file per receiver rank."""

    if not updates_by_rank:
        raise ValueError("write_sparse_delta_files_by_rank requires at least one rank")

    output_dir = Path(output_dir)
    stats: dict[int, SparseDeltaFileStats] = {}
    for rank, updates in sorted(updates_by_rank.items()):
        rank_int = int(rank)
        filename = filename_template.format(rank=rank_int)
        path = output_dir / filename
        stats[rank_int] = write_sparse_delta_file(
            list(updates),
            path,
            delta_encoding_path=delta_encoding_path,
            use_native_extension=use_native_extension,
        )
    return stats


def write_encoded_sparse_delta_files_by_rank(
    encoded_by_rank: Mapping[int, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    filename_template: str = "rank{rank}.packed",
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> dict[int, SparseDeltaFileStats]:
    """Write per-rank ``delta-encoding`` EncodedDelta outputs as packed files."""

    if not encoded_by_rank:
        raise ValueError("write_encoded_sparse_delta_files_by_rank requires at least one rank")

    _, write_packed_file = _load_delta_encoding(
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )

    output_dir = Path(output_dir)
    stats: dict[int, SparseDeltaFileStats] = {}
    for rank, encoded_tensors in sorted(encoded_by_rank.items()):
        if not encoded_tensors:
            raise ValueError(f"Rank {rank} has no encoded sparse-delta tensors")
        rank_int = int(rank)
        path = output_dir / filename_template.format(rank=rank_int)
        written = Path(write_packed_file(dict(encoded_tensors), path))
        nnz = sum(int(getattr(encoded, "values").numel()) for encoded in encoded_tensors.values())
        stats[rank_int] = SparseDeltaFileStats(
            path=str(written),
            tensors=len(encoded_tensors),
            nnz=nnz,
            packed_bytes=written.stat().st_size,
        )
    return stats


def collect_encoded_sparse_deltas_by_rank(
    futures: Iterable[Any],
    *,
    expected_ranks: int | Sequence[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Drain ``delta-encoding`` TranslationFutures into rank-keyed outputs.

    ``TranslationFuture.key`` is a ``delta_encoding.ops.StoreKey`` with a
    receiver rank and tensor name. ``TranslationFuture.wait()`` returns the
    terminal EncodedDelta for that rank/name.
    """

    by_rank: dict[int, dict[str, Any]] = {}
    for future in futures:
        key = getattr(future, "key", None)
        if key is None:
            raise TypeError("Translation future is missing a key attribute")
        strip_tags = getattr(key, "strip_tags", None)
        if callable(strip_tags):
            key = strip_tags()

        rank = getattr(key, "rank", None)
        name = getattr(key, "name", None)
        if rank is None:
            raise ValueError(f"Refusing to serialize unranked sparse-delta terminal output: {key}")
        if not name:
            raise ValueError(f"Refusing to serialize sparse-delta terminal output with empty name: {key}")

        rank_int = int(rank)
        rank_outputs = by_rank.setdefault(rank_int, {})
        name = str(name)
        if name in rank_outputs:
            raise ValueError(f"Duplicate sparse-delta terminal output for rank {rank_int}: {name!r}")
        rank_outputs[name] = future.wait()

    _validate_expected_ranks(by_rank, expected_ranks)
    return by_rank


def write_translation_futures_as_sparse_delta_files(
    futures: Iterable[Any],
    output_dir: str | Path,
    *,
    expected_ranks: int | Sequence[int] | None = None,
    filename_template: str = "rank{rank}.packed",
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> dict[int, SparseDeltaFileStats]:
    """Write ``delta-encoding`` TranslationFuture outputs as packed files."""

    encoded_by_rank = collect_encoded_sparse_deltas_by_rank(futures, expected_ranks=expected_ranks)
    return write_encoded_sparse_delta_files_by_rank(
        encoded_by_rank,
        output_dir,
        filename_template=filename_template,
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )


def _validate_expected_ranks(
    by_rank: Mapping[int, Mapping[str, Any]],
    expected_ranks: int | Sequence[int] | None,
) -> None:
    if expected_ranks is None:
        return
    if isinstance(expected_ranks, int):
        expected = set(range(expected_ranks))
    else:
        expected = {int(rank) for rank in expected_ranks}

    actual = set(by_rank)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing ranks {missing}")
        if extra:
            parts.append(f"unexpected ranks {extra}")
        raise ValueError("Sparse-delta terminal rank mismatch: " + ", ".join(parts))


def _flat_to_coords(flat_indices: torch.Tensor, shape: tuple[int, ...]) -> list[torch.Tensor]:
    remaining = flat_indices.to(torch.int64)
    coords = [torch.empty_like(remaining) for _ in shape]
    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] = remaining % shape[dim]
        remaining = remaining // shape[dim]
    return coords


def _coords_to_flat(coords: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    if not coords:
        return torch.empty(0, dtype=torch.int32)
    flat = torch.zeros_like(coords[0], dtype=torch.int64)
    stride = 1
    for coord, dim in zip(reversed(coords), reversed(shape)):
        flat += coord.to(torch.int64) * stride
        stride *= dim
    if flat.numel() == 0 or stride <= torch.iinfo(torch.int32).max:
        return flat.to(torch.int32)
    return flat
