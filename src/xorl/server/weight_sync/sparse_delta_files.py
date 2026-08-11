"""Helpers for writing prepacked sparse-delta files.

This module is the training-side boundary for the fast sparse-delta API path:
callers provide already sparse, inference-coordinate tensor updates and get a
packed file that can be passed to ``sync_inference_weights(sparse_delta_paths=...)``.
It intentionally does not inspect trainer modules or FSDP state.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

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
        trusted_root = os.environ.get("XORL_TRUSTED_DELTA_ENCODING_ROOT", "").strip()
        if not trusted_root:
            raise ValueError("delta_encoding_path requires XORL_TRUSTED_DELTA_ENCODING_ROOT to be set by the operator")
        trusted_resolved = str(Path(trusted_root).expanduser().resolve(strict=True))
        if resolved != trusted_resolved:
            raise ValueError("delta_encoding_path does not match XORL_TRUSTED_DELTA_ENCODING_ROOT")
        root_path = Path(resolved)
        root_stat = root_path.stat()
        if not root_path.is_dir() or root_path.is_symlink():
            raise ValueError("Trusted delta-encoding root must be a real directory")
        if root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o022:
            raise ValueError("Trusted delta-encoding root must be owned by the current user and not writable by others")
        package_path = root_path / "delta_encoding"
        if not package_path.is_dir() or package_path.is_symlink():
            raise ValueError("Trusted delta-encoding root does not contain a safe delta_encoding package")
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


def _render_rank_filename(filename_template: str, rank: int) -> str:
    """Render one rank filename without allowing directory traversal."""
    filename = filename_template.format(rank=rank)
    candidate = Path(filename)
    if (
        not filename
        or "\x00" in filename
        or candidate.is_absolute()
        or candidate.name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("Sparse-delta filename_template must render a plain filename")
    return filename
