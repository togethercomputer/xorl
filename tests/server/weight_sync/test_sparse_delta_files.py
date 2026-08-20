from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from xorl.server.weight_sync.source_delta_capture import (
    snapshot_sparse_delta_tensors,
    write_sparse_source_delta_global_manifest,
    write_sparse_source_delta_rank,
)
from xorl.server.weight_sync.sparse_delta_files import (
    SparseTensorUpdate,
    write_sparse_delta_file,
)


@dataclass
class _FakeEncoded:
    flat_indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]

    @property
    def flat_deltas(self) -> torch.Tensor:
        return self.flat_indices


def _install_fake_delta_encoding(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    root = types.ModuleType("delta_encoding")
    encoding = types.ModuleType("delta_encoding.encoding")
    compression = types.ModuleType("delta_encoding.encoding.compression")
    packed = types.ModuleType("delta_encoding.encoding.packed")
    encoding_types = types.ModuleType("delta_encoding.encoding.types")
    ops = types.ModuleType("delta_encoding.ops")
    ops_types = types.ModuleType("delta_encoding.ops.types")

    class FakeStoreKey:
        def __init__(self, name: str, rank: int | None = None) -> None:
            self.name = name
            self.rank = rank
            self._tags: tuple[str, ...] = ()

        def tag(self, tag: str) -> "FakeStoreKey":
            key = FakeStoreKey(self.name, self.rank)
            key._tags = (*self._tags, tag)
            return key

        def strip_tags(self) -> "FakeStoreKey":
            return FakeStoreKey(self.name, self.rank)

    class FakeMmapPackedFile:
        def __init__(self, path: str | Path) -> None:
            self.path = str(path)
            self.entries = []

        def __enter__(self) -> "FakeMmapPackedFile":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def fake_encode(indices: torch.Tensor, values: torch.Tensor, shape: tuple[int, ...]) -> _FakeEncoded:
        return _FakeEncoded(indices.clone(), values.clone(), tuple(shape))

    def fake_write_packed_file(encoded: dict[str, _FakeEncoded], path: str | Path) -> Path:
        captured["encoded"] = encoded
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"packed")
        return out

    compression.encode = fake_encode
    encoding_types.EncodedDelta = _FakeEncoded
    ops_types.StoreKey = FakeStoreKey
    packed.MmapPackedFile = FakeMmapPackedFile
    packed.write_packed_file = fake_write_packed_file
    monkeypatch.setitem(sys.modules, "delta_encoding", root)
    monkeypatch.setitem(sys.modules, "delta_encoding.encoding", encoding)
    monkeypatch.setitem(sys.modules, "delta_encoding.encoding.compression", compression)
    monkeypatch.setitem(sys.modules, "delta_encoding.encoding.types", encoding_types)
    monkeypatch.setitem(sys.modules, "delta_encoding.ops", ops)
    monkeypatch.setitem(sys.modules, "delta_encoding.ops.types", ops_types)
    monkeypatch.setitem(sys.modules, "delta_encoding.encoding.packed", packed)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.frozen = nn.Parameter(torch.tensor([5.0, 6.0]), requires_grad=False)


def _assert_source_delta_capture_writes_rank_manifest_and_packed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_delta_encoding(monkeypatch, captured)
    model = _TinyModel()

    config = {"output_dir": str(tmp_path), "dtype": "float32"}
    before = snapshot_sparse_delta_tensors(model, config)
    with torch.no_grad():
        model.weight[1] = 20.0
        model.weight[3] = 40.0

    manifest = write_sparse_source_delta_rank(
        model=model,
        before=before,
        config=config,
        rank=0,
        world_size=1,
        model_id="default",
        step=7,
    )

    assert manifest["rank"] == 0
    assert manifest["step"] == 7
    assert manifest["packed_path"] == str(tmp_path / "rank0.packed")
    assert manifest["totals"]["tensors_considered"] == 1
    assert manifest["totals"]["nnz"] == 2
    assert Path(manifest["manifest_path"]).exists()
    encoded = captured["encoded"]["weight"]
    assert encoded.flat_indices.tolist() == [1, 3]
    assert encoded.values.tolist() == [20.0, 40.0]


def _assert_source_delta_capture_rejects_traversing_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_delta_encoding(monkeypatch, {})
    for template_key, template in (
        ("filename_template", "../rank{rank}.packed"),
        ("manifest_filename_template", "../rank{rank}.manifest.json"),
    ):
        model = _TinyModel()
        config = {
            "output_dir": str(tmp_path),
            "dtype": "float32",
            template_key: template,
        }
        before = snapshot_sparse_delta_tensors(model, config)
        with torch.no_grad():
            model.weight[0] = 10.0

        with pytest.raises(ValueError, match="plain filename"):
            write_sparse_source_delta_rank(
                model=model,
                before=before,
                config=config,
                rank=0,
                world_size=1,
                model_id="default",
                step=7,
            )


def _assert_source_delta_global_manifest_collects_rank_outputs(tmp_path: Path) -> None:
    rank0 = {
        "rank": 0,
        "world_size": 2,
        "model_id": "default",
        "step": 3,
        "output_dir": str(tmp_path),
        "packed_path": str(tmp_path / "rank0.packed"),
        "totals": {"tensors_considered": 1, "tensors_changed": 1, "nnz": 2, "packed_bytes": 10},
    }
    rank1 = {
        "rank": 1,
        "world_size": 2,
        "model_id": "default",
        "step": 3,
        "output_dir": str(tmp_path),
        "packed_path": None,
        "totals": {"tensors_considered": 1, "tensors_changed": 0, "nnz": 0, "packed_bytes": 0},
    }

    manifest = write_sparse_source_delta_global_manifest([rank1, rank0])

    assert manifest["packed_paths"] == [str(tmp_path / "rank0.packed")]
    assert manifest["totals"]["ranks"] == 2
    assert manifest["totals"]["nnz"] == 2
    assert [rank["rank"] for rank in manifest["ranks"]] == [0, 1]
    assert Path(manifest["manifest_path"]).exists()


def _assert_write_sparse_delta_file_packs_sparse_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_delta_encoding(monkeypatch, captured)

    stats = write_sparse_delta_file(
        [
            SparseTensorUpdate(
                name="lm_head.weight",
                flat_indices=torch.tensor([0, 17], dtype=torch.int64),
                values=torch.tensor([1.25, -2.5], dtype=torch.bfloat16),
                shape=(4, 8),
            )
        ],
        tmp_path / "delta.packed",
    )

    assert stats.path == str(tmp_path / "delta.packed")
    assert stats.tensors == 1
    assert stats.nnz == 2
    assert stats.packed_bytes == len(b"packed")
    encoded = captured["encoded"]["lm_head.weight"]
    assert encoded.flat_indices.dtype == torch.int32
    assert encoded.flat_indices.device.type == "cpu"
    assert encoded.flat_indices.tolist() == [0, 17]
    assert encoded.values.dtype == torch.bfloat16
    assert encoded.values.tolist() == [1.25, -2.5]
    assert encoded.shape == (4, 8)


def _assert_write_sparse_delta_file_sorts_indices_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_delta_encoding(monkeypatch, captured)

    write_sparse_delta_file(
        [
            SparseTensorUpdate(
                name="lm_head.weight",
                flat_indices=torch.tensor([17, 0], dtype=torch.int64),
                values=torch.tensor([2.5, 1.25], dtype=torch.bfloat16),
                shape=(4, 8),
            )
        ],
        tmp_path / "delta.packed",
    )

    encoded = captured["encoded"]["lm_head.weight"]
    assert encoded.flat_indices.tolist() == [0, 17]
    assert encoded.values.tolist() == [1.25, 2.5]


def _assert_write_sparse_delta_file_rejects_malformed_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_delta_encoding(monkeypatch, {})

    malformed = (
        (torch.tensor([1, 1], dtype=torch.int32), torch.tensor([1.0, 2.0]), "duplicate flat indices"),
        (torch.tensor([0.0], dtype=torch.float32), torch.tensor([1.0]), "integer dtype"),
        (torch.tensor([0, 1], dtype=torch.int32), torch.tensor([1.0]), "2 indices but 1 values"),
        (torch.tensor([4], dtype=torch.int32), torch.tensor([1.0]), "out of range"),
    )
    for case, (indices, values, error) in enumerate(malformed):
        with pytest.raises(ValueError, match=error):
            write_sparse_delta_file(
                [SparseTensorUpdate(name="bad.weight", flat_indices=indices, values=values, shape=(2, 2))],
                tmp_path / f"bad-{case}.packed",
            )


def test_sparse_delta_artifact_and_source_capture_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as capture_patch:
        _assert_source_delta_capture_writes_rank_manifest_and_packed_file(tmp_path, capture_patch)
        _assert_source_delta_capture_rejects_traversing_templates(tmp_path, capture_patch)
    _assert_source_delta_global_manifest_collects_rank_outputs(tmp_path)

    with monkeypatch.context() as artifact_patch:
        _assert_write_sparse_delta_file_packs_sparse_updates(tmp_path, artifact_patch)
        _assert_write_sparse_delta_file_sorts_indices_before_encoding(tmp_path, artifact_patch)
        _assert_write_sparse_delta_file_rejects_malformed_updates(tmp_path, artifact_patch)
