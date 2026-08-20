"""Unit tests for the Mooncake-backed OPD teacher hidden-state transport.

These tests use an in-memory fake store so they never need the ``mooncake``
package or a live Mooncake master. They exercise:

* the keyed put/get round trip + shape/dtype metadata contract,
* indexing the fetched cache through ``TeacherActivationCache`` (the surface the
  OPD loss calls), so the loss never sees the backend,
* rank-3 multi-layer caches,
* loud failures on malformed metadata.
"""

from __future__ import annotations

import pytest
import torch

from tests._helpers.opd import FakeMooncakeClient, save_tensor_file
from xorl.distillation import (
    MooncakeHiddenStore,
    MooncakeStoreConfig,
    TeacherActivationCache,
    is_mooncake_entry,
    parse_mooncake_metadata,
)
from xorl.distillation.mooncake_hidden_store import (
    bytes_to_tensor,
    dtype_to_str,
    str_to_dtype,
    tensor_to_bytes,
)


pytestmark = [pytest.mark.cpu]


def _store() -> tuple[MooncakeHiddenStore, FakeMooncakeClient]:
    client = FakeMooncakeClient()
    return MooncakeHiddenStore(client=client, get_retry_max_wait_s=0.0), client


def _assert_mooncake_tensor_codec_policy():
    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int64):
        tensor = (torch.arange(12).reshape(3, 4) % 5).to(dtype)
        restored = bytes_to_tensor(tensor_to_bytes(tensor), (3, 4), dtype)
        assert torch.equal(restored, tensor)
        assert restored.dtype == dtype

    _assert_dtype_string_mapping_is_canonical()


def _assert_dtype_string_mapping_is_canonical():
    assert dtype_to_str(torch.bfloat16) == "bfloat16"
    assert str_to_dtype("bf16") is torch.bfloat16
    assert str_to_dtype("torch.float32") is torch.float32
    with pytest.raises(ValueError):
        str_to_dtype("complex128")


def test_mooncake_hidden_transport_policy():
    _assert_mooncake_tensor_codec_policy()

    store, client = _store()
    tensor = torch.randn(7, 5, dtype=torch.bfloat16)

    meta = store.put_hidden("opd/req-1/teacher/0/hidden", tensor)

    assert meta["backend"] == "mooncake"
    assert meta["key"] == "opd/req-1/teacher/0/hidden"
    assert meta["tensor_key"] == "hidden_states"
    assert meta["tensor_shapes"] == {"hidden_states": [7, 5]}
    assert meta["tensor_dtypes"] == {"hidden_states": "bfloat16"}
    assert meta["num_tokens"] == 7
    assert meta["hidden_size"] == 5
    assert "mooncake" in meta and "master_server_address" in meta["mooncake"]
    # The object is stored under the suffixed key, not the bare base key.
    assert client.put_calls == ["opd/req-1/teacher/0/hidden/hidden_states"]
    assert is_mooncake_entry(meta)

    _assert_put_get_roundtrip_via_metadata()
    _assert_rank3_layer_cache_roundtrip_and_token_count()
    _assert_mooncake_teacher_activation_consumer_policy()


def _assert_put_get_roundtrip_via_metadata():
    store, _ = _store()
    tensor = torch.randn(9, 6, dtype=torch.bfloat16)
    meta = store.put_hidden("k", tensor)

    fetched = store.get_hidden_from_metadata(meta)
    assert fetched.shape == (9, 6)
    assert fetched.dtype == torch.bfloat16
    assert torch.equal(fetched, tensor)


def _assert_rank3_layer_cache_roundtrip_and_token_count():
    store, _ = _store()
    # [layers, tokens, hidden]
    tensor = torch.randn(3, 8, 4, dtype=torch.bfloat16)
    meta = store.put_hidden("layer-key", tensor)

    assert meta["num_tokens"] == 8
    assert meta["hidden_size"] == 4
    fetched = store.get_hidden_from_metadata(meta)
    assert torch.equal(fetched, tensor)


def _assert_mooncake_teacher_activation_consumer_policy():
    store, _ = _store()
    cache_tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    meta = store.put_hidden("opd/req/teacher/0/hidden", cache_tensor)

    # The consumer shares the same store (as it would via the metadata key).
    tac = TeacherActivationCache({"0": meta}, mooncake_store=store, enable_async=False)
    try:
        indices = torch.tensor([[0, 3, 5]])
        out = tac.get("0", indices, device="cpu", dtype=torch.float32)
        assert out.shape == (1, 3, 2)
        assert torch.equal(out[0], cache_tensor[[0, 3, 5]])
    finally:
        tac.close()

    _assert_teacher_activation_cache_rank3_mooncake_entry()
    _assert_multi_teacher_mooncake_caches()


def _assert_teacher_activation_cache_rank3_mooncake_entry():
    store, _ = _store()
    layer_cache = torch.randn(4, 6, 3, dtype=torch.float32)  # [L, tokens, d]
    meta = store.put_hidden("layer", layer_cache)

    tac = TeacherActivationCache({"0": meta}, mooncake_store=store, enable_async=False)
    try:
        indices = torch.tensor([[1, 4]])
        out = tac.get("0", indices, device="cpu", dtype=torch.float32)
        # rank-3 cache returns [*indices.shape, layers, d]
        assert out.shape == (1, 2, 4, 3)
        # token 1, all layers, matches the source layer-major tensor
        assert torch.equal(out[0, 0], layer_cache[:, 1, :])
    finally:
        tac.close()


def _assert_multi_teacher_mooncake_caches():
    store, _ = _store()
    cache_a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    cache_b = torch.arange(100, 110, dtype=torch.float32).reshape(5, 2)
    meta_a = store.put_hidden("a", cache_a)
    meta_b = store.put_hidden("b", cache_b)

    tac = TeacherActivationCache({"0": meta_a, "1": meta_b}, mooncake_store=store, enable_async=False)
    try:
        out_a = tac.get("0", torch.tensor([[1, 3]]), device="cpu", dtype=torch.float32)
        out_b = tac.get("1", torch.tensor([[0, 4]]), device="cpu", dtype=torch.float32)
        assert torch.equal(out_a[0], cache_a[[1, 3]])
        assert torch.equal(out_b[0], cache_b[[0, 4]])
    finally:
        tac.close()


def test_mooncake_metadata_admission_and_store_lifecycle_policy(tmp_path, monkeypatch):
    # The file-backed safetensors cache path was removed; a path/str entry must
    # now fail loudly rather than silently load a file.
    store, _ = _store()
    file_path = save_tensor_file(tmp_path / "legacy.safetensors", "hidden_states", torch.zeros(3, 2))
    tac = TeacherActivationCache({"0": file_path}, mooncake_store=store, enable_async=False)
    try:
        with pytest.raises(ValueError, match="Mooncake metadata dict"):
            tac.get("0", torch.tensor([[0]]), device="cpu", dtype=torch.float32)
    finally:
        tac.close()

    _assert_get_missing_key_raises()
    _assert_size_mismatch_raises()
    _assert_parse_mooncake_metadata_rejects_malformed()
    _assert_mooncake_store_lifecycle_and_configuration_policy(monkeypatch)


def _assert_get_missing_key_raises():
    store, _ = _store()
    meta = {
        "backend": "mooncake",
        "key": "does-not-exist",
        "tensor_key": "hidden_states",
        "tensor_shapes": {"hidden_states": [3, 2]},
        "tensor_dtypes": {"hidden_states": "float32"},
    }
    with pytest.raises(KeyError):
        store.get_hidden_from_metadata(meta)


def _assert_size_mismatch_raises():
    store, client = _store()
    tensor = torch.randn(4, 2, dtype=torch.float32)
    meta = store.put_hidden("k", tensor)
    # Lie about the shape — the byte count no longer matches.
    meta["tensor_shapes"]["hidden_states"] = [5, 2]
    with pytest.raises(ValueError, match="size mismatch"):
        store.get_hidden_from_metadata(meta)


def _assert_parse_mooncake_metadata_rejects_malformed():
    mutations = (
        lambda m: m.pop("key"),
        lambda m: m.pop("tensor_shapes"),
        lambda m: m.pop("tensor_dtypes"),
        lambda m: m["tensor_shapes"].__setitem__("hidden_states", [1, 2, 3, 4]),
    )
    for mutate in mutations:
        meta = {
            "backend": "mooncake",
            "key": "k",
            "tensor_key": "hidden_states",
            "tensor_shapes": {"hidden_states": [3, 2]},
            "tensor_dtypes": {"hidden_states": "float32"},
        }
        mutate(meta)
        with pytest.raises(ValueError):
            parse_mooncake_metadata(meta)


def _assert_mooncake_store_lifecycle_and_configuration_policy(monkeypatch):
    store, client = _store()
    tensor = torch.randn(2, 2)
    meta = store.put_hidden("k", tensor)
    store.remove_hidden(meta["key"])
    assert client.removed == ["k/hidden_states"]

    _assert_store_config_overrides_win_over_env(monkeypatch)


def _assert_store_config_overrides_win_over_env(monkeypatch):
    monkeypatch.setenv("MOONCAKE_MASTER_SERVER", "envhost:50051")
    cfg = MooncakeStoreConfig.from_env(overrides={"master_server_address": "pinned:9999"})
    assert cfg.master_server_address == "pinned:9999"
    cfg_env = MooncakeStoreConfig.from_env()
    assert cfg_env.master_server_address == "envhost:50051"
