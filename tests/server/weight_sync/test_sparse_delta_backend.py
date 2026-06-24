from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import requests
import torch

from xorl.server.weight_sync.backends import create_backend
from xorl.server.weight_sync.backends.base import EndpointConfig, TransportConfig
from xorl.server.weight_sync.backends.sparse_delta import SparseDeltaTransportBackend


@dataclass
class _FakeEncoded:
    flat_indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"success": True, "message": "ok"}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(self.text)


def _make_backend(tmp_path: Path, *, world_size: int = 2) -> SparseDeltaTransportBackend:
    SparseDeltaTransportBackend.clear_cached_baselines()
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=world_size)],
        group_name=f"test-sparse-delta-{tmp_path.name}",
        training_rank=0,
        backend_config={"output_dir": str(tmp_path), "keep_files": True},
    )
    backend = SparseDeltaTransportBackend(cfg)
    backend._initialized = True
    backend._encode_fn = lambda indices, values, shape: _FakeEncoded(indices.clone(), values.clone(), tuple(shape))

    def fake_write(encoded_tensors: dict[str, _FakeEncoded], path: str | Path) -> Path:
        del encoded_tensors
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"packed")
        return out

    backend._write_packed_file = fake_write
    return backend


def _make_backend_with_config(
    tmp_path: Path,
    backend_config: dict[str, Any],
    *,
    world_size: int = 1,
) -> SparseDeltaTransportBackend:
    SparseDeltaTransportBackend.clear_cached_baselines()
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=world_size)],
        group_name=f"test-sparse-delta-{tmp_path.name}",
        training_rank=0,
        backend_config={"output_dir": str(tmp_path), "keep_files": True, **backend_config},
    )
    backend = SparseDeltaTransportBackend(cfg)
    backend._initialized = True
    backend._encode_fn = lambda indices, values, shape: _FakeEncoded(indices.clone(), values.clone(), tuple(shape))
    return backend


def test_factory_creates_sparse_delta_backend(tmp_path: Path) -> None:
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=1)],
        backend_config={"output_dir": str(tmp_path)},
    )

    backend = create_backend("sparse_delta", cfg)

    assert isinstance(backend, SparseDeltaTransportBackend)


def test_transfer_bucket_posts_packed_delta_to_each_tp_rank(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, world_size=2)
    captured: list[dict[str, _FakeEncoded]] = []

    def fake_write(encoded_tensors: dict[str, _FakeEncoded], path: str | Path) -> Path:
        captured.append(encoded_tensors)
        out = Path(path)
        out.write_bytes(b"packed")
        return out

    backend._write_packed_file = fake_write

    with patch("requests.post", return_value=_FakeResponse()) as posted:
        backend.transfer_bucket(
            [("model.norm.weight", torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16))],
            flush_cache=True,
            weight_version="sync-1",
        )

    assert posted.call_count == 1
    url = posted.call_args.args[0]
    body = posted.call_args.kwargs["json"]
    assert url == "http://infer-0:30000/update_weights_from_sparse_delta"
    assert body["delta_paths"] == [body["delta_paths"][0], body["delta_paths"][0]]
    assert body["delta_paths"][0].endswith(".packed")
    assert body["flush_cache"] is True
    assert body["weight_version"] == "sync-1"

    encoded = captured[0]["model.norm.weight"]
    assert encoded.flat_indices.tolist() == [0, 1, 2]
    assert encoded.values.dtype == torch.bfloat16
    assert encoded.values.tolist() == [1.0, 2.0, 3.0]
    assert encoded.shape == (3,)


def test_repeated_transfer_encodes_only_exact_byte_changes(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, world_size=1)
    captured: list[dict[str, _FakeEncoded]] = []

    def fake_write(encoded_tensors: dict[str, _FakeEncoded], path: str | Path) -> Path:
        captured.append(encoded_tensors)
        out = Path(path)
        out.write_bytes(b"packed")
        return out

    backend._write_packed_file = fake_write

    with patch("requests.post", return_value=_FakeResponse()):
        backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16))])
        backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0, 4.0, 3.0], dtype=torch.bfloat16))])

    first = captured[0]["model.norm.weight"]
    second = captured[1]["model.norm.weight"]
    assert first.flat_indices.tolist() == [0, 1, 2]
    assert second.flat_indices.tolist() == [1]
    assert second.values.tolist() == [4.0]


def test_unchanged_nonfinal_bucket_skips_post(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, world_size=1)

    with patch("requests.post", return_value=_FakeResponse()) as posted:
        backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0], dtype=torch.bfloat16))])
        backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0], dtype=torch.bfloat16))])

    assert posted.call_count == 1
    assert backend.stats_summary()["skipped_unchanged_buckets"] == 1.0


def test_prime_baseline_posts_only_final_empty_update(tmp_path: Path) -> None:
    backend = _make_backend_with_config(tmp_path, {"prime_baseline": True})
    captured: list[dict[str, _FakeEncoded]] = []

    def fake_write(encoded_tensors: dict[str, _FakeEncoded], path: str | Path) -> Path:
        captured.append(encoded_tensors)
        out = Path(path)
        out.write_bytes(b"packed")
        return out

    backend._write_packed_file = fake_write

    with patch("requests.post", return_value=_FakeResponse()) as posted:
        backend.transfer_bucket([("model.layers.0.weight", torch.tensor([1.0], dtype=torch.bfloat16))])
        backend.transfer_bucket(
            [("model.norm.weight", torch.tensor([2.0, 3.0], dtype=torch.bfloat16))],
            flush_cache=True,
            weight_version="primed",
        )

    assert posted.call_count == 1
    assert posted.call_args.kwargs["json"]["weight_version"] == "primed"
    assert list(captured[0]) == ["model.norm.weight"]
    encoded = captured[0]["model.norm.weight"]
    assert encoded.flat_indices.tolist() == []
    assert encoded.values.tolist() == []
    summary = backend.stats_summary()
    assert summary["primed_tensors"] == 2.0
    assert summary["skipped_unchanged_buckets"] == 1.0
    assert summary["baseline_update_s"] >= 0.0


def test_post_packed_delta_paths_replicates_single_path_to_tp_ranks(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, world_size=2)
    packed = tmp_path / "delta.packed"
    packed.write_bytes(b"packed")

    with patch("requests.post", return_value=_FakeResponse()) as posted:
        backend.post_packed_delta_paths([str(packed)], flush_cache=True, weight_version="sync-1")

    body = posted.call_args.kwargs["json"]
    assert body["delta_paths"] == [str(packed), str(packed)]
    assert body["flush_cache"] is True
    assert body["weight_version"] == "sync-1"
    summary = backend.stats_summary()
    assert summary["posted_files"] == 1.0
    assert summary["total_packed_bytes"] == float(len(b"packed"))


def test_post_packed_delta_paths_threads_fp8_kv_cache_metadata(tmp_path: Path) -> None:
    backend = _make_backend_with_config(
        tmp_path,
        {
            "run_post_process_weights": True,
            "fp8_kv_cache_enabled": True,
            "fp8_kv_cache_postprocess_required": True,
            "fp8_kv_cache_static_scales": True,
        },
    )
    packed = tmp_path / "delta.packed"
    packed.write_bytes(b"packed")

    with patch(
        "requests.post",
        return_value=_FakeResponse(
            payload={
                "success": True,
                "message": "ok",
                "cache_version": "epoch-3",
                "fp8_kv_cache_postprocess_ran": True,
                "fp8_kv_cache_static_scales_updated": True,
            }
        ),
    ) as posted:
        backend.post_packed_delta_paths([str(packed)], flush_cache=True, weight_version="sync-1")

    body = posted.call_args.kwargs["json"]
    assert body["flush_cache"] is True
    assert body["weight_version"] == "sync-1"
    assert body["run_post_process_weights"] is True
    assert body["fp8_kv_cache_enabled"] is True
    assert body["fp8_kv_cache_postprocess_required"] is True
    assert body["fp8_kv_cache_static_scales"] is True
    assert backend.endpoint_results == [
        {
            "host": "infer-0",
            "port": 30000,
            "success": True,
            "message": "ok",
            "cache_epoch": "epoch-3",
            "fp8_kv_cache_postprocess_ran": True,
            "fp8_kv_cache_static_scales_updated": True,
        }
    ]


def test_post_packed_delta_paths_accepts_per_tp_paths(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, world_size=2)
    packed0 = tmp_path / "rank0.packed"
    packed1 = tmp_path / "rank1.packed"
    packed0.write_bytes(b"0")
    packed1.write_bytes(b"11")

    with patch("requests.post", return_value=_FakeResponse()) as posted:
        backend.post_packed_delta_paths([str(packed0), str(packed1)])

    body = posted.call_args.kwargs["json"]
    assert body["delta_paths"] == [str(packed0), str(packed1)]
    assert backend.stats_summary()["total_packed_bytes"] == 3.0


def test_post_only_initialize_does_not_import_delta_encoding(tmp_path: Path) -> None:
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=1)],
        group_name=f"test-sparse-delta-{tmp_path.name}",
        training_rank=0,
        backend_config={
            "output_dir": str(tmp_path),
            "post_only": True,
            "delta_encoding_path": str(tmp_path / "does-not-exist"),
        },
    )
    backend = SparseDeltaTransportBackend(cfg)

    assert backend.initialize() is True

    backend.destroy()


def test_prepacked_only_refuses_streaming_initialize(tmp_path: Path) -> None:
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=1)],
        group_name=f"test-sparse-delta-{tmp_path.name}",
        training_rank=0,
        backend_config={
            "output_dir": str(tmp_path),
            "prepacked_only": True,
        },
    )
    backend = SparseDeltaTransportBackend(cfg)

    assert backend.initialize() is False


def test_prepacked_only_allows_post_only_initialize(tmp_path: Path) -> None:
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=1)],
        group_name=f"test-sparse-delta-{tmp_path.name}",
        training_rank=0,
        backend_config={
            "output_dir": str(tmp_path),
            "post_only": True,
            "prepacked_only": True,
        },
    )
    backend = SparseDeltaTransportBackend(cfg)

    assert backend.initialize() is True

    backend.destroy()


def test_initialize_loads_delta_encoding_with_runtime_helper(tmp_path: Path) -> None:
    delta_path = tmp_path / "delta-encoding"
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host="infer-0", port=30000, world_size=1)],
        group_name=f"test-sparse-delta-{tmp_path.name}",
        training_rank=0,
        backend_config={
            "output_dir": str(tmp_path),
            "delta_encoding_path": str(delta_path),
            "use_native_extension": True,
        },
    )
    backend = SparseDeltaTransportBackend(cfg)
    fake_encode = object()
    fake_write = object()

    def fake_import(name: str) -> SimpleNamespace:
        if name == "delta_encoding.encoding.compression":
            return SimpleNamespace(encode=fake_encode)
        if name == "delta_encoding.encoding.packed":
            return SimpleNamespace(write_packed_file=fake_write)
        raise ModuleNotFoundError(name)

    with (
        patch("xorl.server.weight_sync.backends.sparse_delta.prepare_delta_encoding_runtime") as prepare_runtime,
        patch("xorl.server.weight_sync.backends.sparse_delta.importlib.import_module", side_effect=fake_import),
    ):
        assert backend.initialize() is True

    prepare_runtime.assert_called_once_with(
        delta_encoding_path=str(delta_path),
        use_native_extension=True,
    )
    assert backend._encode_fn is fake_encode
    assert backend._write_packed_file is fake_write

    backend.destroy()


def test_receiver_failure_does_not_update_baseline(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, world_size=1)
    captured: list[dict[str, _FakeEncoded]] = []

    def fake_write(encoded_tensors: dict[str, _FakeEncoded], path: str | Path) -> Path:
        captured.append(encoded_tensors)
        out = Path(path)
        out.write_bytes(b"packed")
        return out

    backend._write_packed_file = fake_write

    with patch("requests.post", return_value=_FakeResponse()):
        backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))])

    with patch("requests.post", return_value=_FakeResponse(400, {"success": False}, "bad sparse delta")):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0, 5.0], dtype=torch.bfloat16))])

    with patch("requests.post", return_value=_FakeResponse()):
        backend.transfer_bucket([("model.norm.weight", torch.tensor([1.0, 5.0], dtype=torch.bfloat16))])

    retry = captured[-1]["model.norm.weight"]
    assert retry.flat_indices.tolist() == [1]
    assert retry.values.tolist() == [5.0]
