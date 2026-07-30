from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from xorl.server.weight_sync.source_delta_capture import (
    GLOBAL_MANIFEST_VERSION,
    load_sparse_source_delta_inputs,
    snapshot_sparse_delta_tensors,
    write_sparse_source_delta_global_manifest,
    write_sparse_source_delta_rank,
)
from xorl.server.weight_sync.sparse_delta_files import (
    SparseTensorUpdate,
    collect_encoded_sparse_deltas_by_rank,
    split_sparse_update_by_contiguous_shards,
    write_encoded_sparse_delta_files_by_rank,
    write_sparse_delta_file,
    write_sparse_delta_files_by_rank,
    write_translation_futures_as_sparse_delta_files,
)


@dataclass
class _FakeEncoded:
    flat_indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]

    @property
    def flat_deltas(self) -> torch.Tensor:
        return self.flat_indices


@dataclass(frozen=True)
class _FakeKey:
    name: str
    rank: int | None

    def strip_tags(self) -> "_FakeKey":
        return _FakeKey(self.name.split(".__", 1)[0], self.rank)


@dataclass
class _FakeFuture:
    key: _FakeKey
    value: _FakeEncoded

    def wait(self) -> _FakeEncoded:
        return self.value


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


def test_source_delta_capture_writes_rank_manifest_and_packed_file(
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


def test_source_delta_global_manifest_collects_rank_outputs(tmp_path: Path) -> None:
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


def test_load_sparse_source_delta_inputs_can_include_empty_rank_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_delta_encoding(monkeypatch, {})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": GLOBAL_MANIFEST_VERSION,
                "world_size": 2,
                "ranks": [
                    {
                        "rank": 0,
                        "packed_path": None,
                        "capture_dtype": "bfloat16",
                        "tensors": [
                            {
                                "name": "model.norm.weight",
                                "shape": [4],
                                "dtype": "bfloat16",
                                "nnz": 0,
                            }
                        ],
                    },
                    {
                        "rank": 1,
                        "packed_path": None,
                        "capture_dtype": "bfloat16",
                        "tensors": [
                            {
                                "name": "model.norm.weight",
                                "shape": [4],
                                "dtype": "bfloat16",
                                "nnz": 0,
                            }
                        ],
                    },
                ],
            }
        )
    )

    inputs = load_sparse_source_delta_inputs(manifest_path, include_empty=True)

    assert len(inputs) == 2
    assert [key.rank for key, _ in inputs] == [0, 1]
    assert [key.strip_tags().name for key, _ in inputs] == ["model.norm.weight", "model.norm.weight"]
    assert [encoded.shape for _, encoded in inputs] == [(4,), (4,)]
    assert all(encoded.values.dtype == torch.bfloat16 for _, encoded in inputs)
    assert all(encoded.values.numel() == 0 for _, encoded in inputs)


def test_write_sparse_delta_file_packs_sparse_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_write_sparse_delta_file_sorts_indices_before_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_write_sparse_delta_file_rejects_duplicate_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_delta_encoding(monkeypatch, {})

    with pytest.raises(ValueError, match="duplicate flat indices"):
        write_sparse_delta_file(
            [
                SparseTensorUpdate(
                    name="dup.weight",
                    flat_indices=torch.tensor([1, 1], dtype=torch.int32),
                    values=torch.tensor([1.0, 2.0], dtype=torch.float32),
                    shape=(2, 2),
                )
            ],
            tmp_path / "dup.packed",
        )


def test_write_sparse_delta_file_rejects_non_integer_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_delta_encoding(monkeypatch, {})

    with pytest.raises(ValueError, match="integer dtype"):
        write_sparse_delta_file(
            [
                SparseTensorUpdate(
                    name="bad.weight",
                    flat_indices=torch.tensor([0.0], dtype=torch.float32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                    shape=(2, 2),
                )
            ],
            tmp_path / "bad.packed",
        )


def test_write_sparse_delta_file_rejects_bad_update_lengths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_delta_encoding(monkeypatch, {})

    with pytest.raises(ValueError, match="2 indices but 1 values"):
        write_sparse_delta_file(
            [
                SparseTensorUpdate(
                    name="bad.weight",
                    flat_indices=torch.tensor([0, 1], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                    shape=(2, 2),
                )
            ],
            tmp_path / "bad.packed",
        )


def test_split_sparse_update_by_contiguous_shards_localizes_flat_indices() -> None:
    update = SparseTensorUpdate(
        name="lm_head.weight",
        flat_indices=torch.tensor([22, 10, 0, 17, 9], dtype=torch.int64),
        values=torch.tensor([5.0, 3.0, 1.0, 4.0, 2.0], dtype=torch.bfloat16),
        shape=(6, 4),
    )

    by_rank = split_sparse_update_by_contiguous_shards(
        update,
        shard_dim=0,
        num_shards=3,
    )

    assert sorted(by_rank) == [0, 1, 2]
    assert by_rank[0].name == "lm_head.weight"
    assert by_rank[0].shape == (2, 4)
    assert by_rank[0].flat_indices.tolist() == [0]
    assert by_rank[0].values.tolist() == [1.0]
    assert by_rank[1].shape == (2, 4)
    assert by_rank[1].flat_indices.tolist() == [1, 2]
    assert by_rank[1].values.tolist() == [2.0, 3.0]
    assert by_rank[2].shape == (2, 4)
    assert by_rank[2].flat_indices.tolist() == [1, 6]
    assert by_rank[2].values.tolist() == [4.0, 5.0]


def test_split_sparse_update_by_contiguous_shards_emits_empty_local_updates() -> None:
    update = SparseTensorUpdate(
        name="lm_head.weight",
        flat_indices=torch.tensor([0], dtype=torch.int32),
        values=torch.tensor([1.0], dtype=torch.bfloat16),
        shape=(4, 4),
    )

    by_rank = split_sparse_update_by_contiguous_shards(update, shard_dim=0, num_shards=2)

    assert by_rank[0].flat_indices.tolist() == [0]
    assert by_rank[1].shape == (2, 4)
    assert by_rank[1].flat_indices.tolist() == []
    assert by_rank[1].values.numel() == 0


def test_write_sparse_delta_files_by_rank_writes_rank_ordered_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_delta_encoding(monkeypatch, captured)

    stats = write_sparse_delta_files_by_rank(
        {
            1: [
                SparseTensorUpdate(
                    name="lm_head.weight",
                    flat_indices=torch.tensor([0], dtype=torch.int32),
                    values=torch.tensor([2.0], dtype=torch.bfloat16),
                    shape=(2, 4),
                )
            ],
            0: [
                SparseTensorUpdate(
                    name="lm_head.weight",
                    flat_indices=torch.tensor([0], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.bfloat16),
                    shape=(2, 4),
                )
            ],
        },
        tmp_path,
    )

    assert sorted(stats) == [0, 1]
    assert stats[0].path == str(tmp_path / "rank0.packed")
    assert stats[1].path == str(tmp_path / "rank1.packed")
    assert stats[0].nnz == 1
    assert stats[1].nnz == 1


def test_write_encoded_sparse_delta_files_by_rank_uses_packed_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_delta_encoding(monkeypatch, captured)
    encoded_by_rank = {
        0: {"lm_head.weight": _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(1), (2, 4))},
        1: {"lm_head.weight": _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(2), (2, 4))},
    }

    stats = write_encoded_sparse_delta_files_by_rank(encoded_by_rank, tmp_path)

    assert sorted(stats) == [0, 1]
    assert stats[0].path == str(tmp_path / "rank0.packed")
    assert stats[1].path == str(tmp_path / "rank1.packed")
    assert stats[0].nnz == 1
    assert stats[1].nnz == 2


def test_collect_encoded_sparse_deltas_by_rank_drains_translation_futures() -> None:
    encoded0 = _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(1), (2, 4))
    encoded1 = _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(2), (2, 4))

    by_rank = collect_encoded_sparse_deltas_by_rank(
        [
            _FakeFuture(_FakeKey("lm_head.weight.__enc", 1), encoded1),
            _FakeFuture(_FakeKey("lm_head.weight.__enc", 0), encoded0),
        ],
        expected_ranks=2,
    )

    assert sorted(by_rank) == [0, 1]
    assert by_rank[0] == {"lm_head.weight": encoded0}
    assert by_rank[1] == {"lm_head.weight": encoded1}


def test_collect_encoded_sparse_deltas_by_rank_rejects_unranked_output() -> None:
    encoded = _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(1), (2, 4))

    with pytest.raises(ValueError, match="unranked"):
        collect_encoded_sparse_deltas_by_rank([_FakeFuture(_FakeKey("lm_head.weight", None), encoded)])


def test_write_translation_futures_as_sparse_delta_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_delta_encoding(monkeypatch, {})
    futures = [
        _FakeFuture(
            _FakeKey("lm_head.weight", 0),
            _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(1), (2, 4)),
        ),
        _FakeFuture(
            _FakeKey("lm_head.weight", 1),
            _FakeEncoded(torch.tensor([0], dtype=torch.uint8), torch.ones(1), (2, 4)),
        ),
    ]

    stats = write_translation_futures_as_sparse_delta_files(futures, tmp_path, expected_ranks=[0, 1])

    assert sorted(stats) == [0, 1]
    assert stats[0].path == str(tmp_path / "rank0.packed")
    assert stats[1].path == str(tmp_path / "rank1.packed")


def test_write_sparse_delta_file_rejects_out_of_range_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_delta_encoding(monkeypatch, {})

    with pytest.raises(ValueError, match="out of range"):
        write_sparse_delta_file(
            [
                SparseTensorUpdate(
                    name="bad.weight",
                    flat_indices=torch.tensor([4], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                    shape=(2, 2),
                )
            ],
            tmp_path / "bad.packed",
        )
