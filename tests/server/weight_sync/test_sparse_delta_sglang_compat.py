from __future__ import annotations

import hashlib
import importlib
import inspect as _inspect  # noqa: E402
import os
import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn

from xorl.server.weight_sync.sparse_delta_files import SparseTensorUpdate, write_sparse_delta_file


if (
    "disk_compression" not in _inspect.signature(write_sparse_delta_file).parameters
):  # pragma: no cover - upstream WIP gap
    pytest.skip(
        "write_sparse_delta_file() has no disk_compression parameter upstream; the zstd-packed "
        "disk-compression sparse-delta path this test exercises is not implemented yet",
        allow_module_level=True,
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _add_import_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if (candidate / "python" / "sglang").is_dir():
        candidate = candidate / "python"
    if not candidate.exists():
        return None
    resolved = str(candidate.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return resolved


def _require_delta_encoding() -> str | None:
    delta_path = _add_import_path(os.environ.get("XORL_DELTA_ENCODING_PATH")) or _add_import_path(
        "/home/apanda/delta-encoding"
    )
    try:
        importlib.import_module("delta_encoding.encoding.packed")
        importlib.import_module("delta_encoding.encoding.compression")
    except Exception as exc:
        pytest.skip(f"delta-encoding is not importable: {exc}")
    return delta_path


def _require_sglang_apply_sparse_delta_file() -> Callable[..., object]:
    _add_import_path(os.environ.get("XORL_SGLANG_PATH")) or _add_import_path("/home/apanda/xorl-sglang-internal")
    try:
        module = importlib.import_module("sglang.srt.weight_sync.sparse_delta")
    except Exception as exc:
        pytest.skip(f"xorl-sglang-internal sparse-delta receiver is not importable: {exc}")
    return module.apply_sparse_delta_file


class _TinyReceiverModel(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)
        self.norm = nn.Parameter(torch.empty(3))
        self.to(dtype=dtype)
        with torch.no_grad():
            self.proj.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype) / 10)
            self.norm.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))


def _sparse_updates_from_dense_diff(
    before: nn.Module,
    after: nn.Module,
) -> list[SparseTensorUpdate]:
    before_params = dict(before.named_parameters())
    updates: list[SparseTensorUpdate] = []
    for name, after_param in after.named_parameters():
        before_param = before_params[name]
        changed = torch.ne(before_param.detach(), after_param.detach()).reshape(-1).nonzero(as_tuple=False).flatten()
        if changed.numel() == 0:
            continue
        flat_after = after_param.detach().reshape(-1)
        updates.append(
            SparseTensorUpdate(
                name=name,
                flat_indices=changed.to(torch.int64),
                values=flat_after[changed].clone(),
                shape=tuple(after_param.shape),
            )
        )
    return updates


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_xorl_packed_sparse_delta_applies_with_sglang_receiver(
    tmp_path: Path,
    dtype: torch.dtype,
) -> None:
    delta_encoding_path = _require_delta_encoding()
    apply_sparse_delta_file = _require_sglang_apply_sparse_delta_file()

    baseline = _TinyReceiverModel(dtype)
    sparse_receiver = _TinyReceiverModel(dtype)
    dense_receiver = _TinyReceiverModel(dtype)
    sparse_receiver.load_state_dict(baseline.state_dict())
    dense_receiver.load_state_dict(baseline.state_dict())

    with torch.no_grad():
        dense_receiver.proj.weight.reshape(-1)[1] = torch.tensor(-2.0, dtype=dtype)
        dense_receiver.proj.weight.reshape(-1)[10] = torch.tensor(7.5, dtype=dtype)
        dense_receiver.norm.reshape(-1)[2] = torch.tensor(9.0, dtype=dtype)

    packed_path = tmp_path / f"delta-{str(dtype).rsplit('.', 1)[-1]}.packed"
    updates = _sparse_updates_from_dense_diff(baseline, dense_receiver)
    write_stats = write_sparse_delta_file(
        updates,
        packed_path,
        delta_encoding_path=delta_encoding_path,
    )

    apply_stats = apply_sparse_delta_file(sparse_receiver, packed_path)

    assert write_stats.nnz == 3
    assert apply_stats.total_nnz == 3
    assert apply_stats.applied_nnz == 3
    assert apply_stats.direct_tensors == 2
    for sparse_param, dense_param in zip(sparse_receiver.parameters(), dense_receiver.parameters()):
        torch.testing.assert_close(sparse_param, dense_param, rtol=0, atol=0)


def test_sglang_receiver_enforces_sparse_delta_sha256(tmp_path: Path) -> None:
    delta_encoding_path = _require_delta_encoding()
    apply_sparse_delta_file = _require_sglang_apply_sparse_delta_file()

    receiver = _TinyReceiverModel(torch.float32)
    before = {name: param.detach().clone() for name, param in receiver.named_parameters()}
    packed_path = tmp_path / "checksummed-delta.packed"

    write_sparse_delta_file(
        [
            SparseTensorUpdate(
                name="proj.weight",
                flat_indices=torch.tensor([0], dtype=torch.int64),
                values=torch.tensor([99.0], dtype=torch.float32),
                shape=tuple(receiver.proj.weight.shape),
            )
        ],
        packed_path,
        delta_encoding_path=delta_encoding_path,
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        apply_sparse_delta_file(receiver, packed_path, expected_sha256="0" * 64)

    for name, param in receiver.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)

    apply_stats = apply_sparse_delta_file(receiver, packed_path, expected_sha256=_sha256_file(packed_path))
    assert apply_stats.applied_nnz == 1
    assert torch.count_nonzero(receiver.proj.weight != before["proj.weight"]) == 1


def test_xorl_zstd_packed_sparse_delta_applies_with_sglang_receiver(tmp_path: Path) -> None:
    pytest.importorskip("zstandard")
    delta_encoding_path = _require_delta_encoding()
    apply_sparse_delta_file = _require_sglang_apply_sparse_delta_file()

    receiver = _TinyReceiverModel(torch.float32)
    before = {name: param.detach().clone() for name, param in receiver.named_parameters()}
    packed_path = tmp_path / "compressed-delta.packed"

    write_stats = write_sparse_delta_file(
        [
            SparseTensorUpdate(
                name="proj.weight",
                flat_indices=torch.tensor([0], dtype=torch.int64),
                values=torch.tensor([99.0], dtype=torch.float32),
                shape=tuple(receiver.proj.weight.shape),
            )
        ],
        packed_path,
        delta_encoding_path=delta_encoding_path,
        disk_compression="zstd",
        zstd_level=1,
    )

    assert write_stats.compression == "zstd"
    assert write_stats.path.endswith(".packed.zst")
    apply_stats = apply_sparse_delta_file(receiver, write_stats.path, expected_sha256=write_stats.sha256)

    assert apply_stats.compression == "zstd"
    assert apply_stats.uncompressed_packed_bytes == write_stats.uncompressed_packed_bytes
    assert apply_stats.applied_nnz == 1
    assert torch.count_nonzero(receiver.proj.weight != before["proj.weight"]) == 1


def test_xorl_noop_packed_sparse_delta_applies_with_sglang_receiver(tmp_path: Path) -> None:
    delta_encoding_path = _require_delta_encoding()
    apply_sparse_delta_file = _require_sglang_apply_sparse_delta_file()

    receiver = _TinyReceiverModel(torch.bfloat16)
    before = {name: param.detach().clone() for name, param in receiver.named_parameters()}
    packed_path = tmp_path / "noop-delta.packed"

    write_stats = write_sparse_delta_file(
        [
            SparseTensorUpdate(
                name="proj.weight",
                flat_indices=torch.empty(0, dtype=torch.int64),
                values=torch.empty(0, dtype=torch.bfloat16),
                shape=tuple(receiver.proj.weight.shape),
            )
        ],
        packed_path,
        delta_encoding_path=delta_encoding_path,
    )

    apply_stats = apply_sparse_delta_file(receiver, packed_path)

    assert write_stats.tensors == 1
    assert write_stats.nnz == 0
    assert apply_stats.total_nnz == 0
    assert apply_stats.applied_nnz == 0
    assert apply_stats.skipped_empty_tensors == 1
    for name, param in receiver.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)


def test_sglang_validate_only_rejects_bad_sparse_delta_without_mutating(tmp_path: Path) -> None:
    delta_encoding_path = _require_delta_encoding()
    apply_sparse_delta_file = _require_sglang_apply_sparse_delta_file()

    receiver = _TinyReceiverModel(torch.float32)
    before = {name: param.detach().clone() for name, param in receiver.named_parameters()}
    packed_path = tmp_path / "mixed-valid-invalid.packed"

    write_sparse_delta_file(
        [
            SparseTensorUpdate(
                name="proj.weight",
                flat_indices=torch.tensor([0], dtype=torch.int64),
                values=torch.tensor([99.0], dtype=torch.float32),
                shape=tuple(receiver.proj.weight.shape),
            ),
            SparseTensorUpdate(
                name="missing.weight",
                flat_indices=torch.tensor([0], dtype=torch.int64),
                values=torch.tensor([1.0], dtype=torch.float32),
                shape=(1,),
            ),
        ],
        packed_path,
        delta_encoding_path=delta_encoding_path,
    )

    with pytest.raises(KeyError, match="missing.weight"):
        apply_sparse_delta_file(receiver, packed_path, validate_only=True)

    for name, param in receiver.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
