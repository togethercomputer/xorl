import json

import pytest
import torch
from safetensors.torch import save_file

from tests._helpers.opd import FakeMooncakeClient, save_tensor_file
from xorl.distillation import (
    MooncakeHiddenStore,
    TeacherActivationCache,
    TeacherHeadManager,
    TeacherHeadStore,
    load_lm_head_weight,
    prepare_lm_head_teacher_store,
)


def _mooncake_cache(hidden, teacher_id="3", *, enable_async=False):
    """Build a TeacherActivationCache backed by an in-memory Mooncake store."""
    store = MooncakeHiddenStore(client=FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    meta = store.put_hidden(f"opd/test/teacher/{teacher_id}/hidden", hidden)
    cache = TeacherActivationCache({teacher_id: meta}, mooncake_store=store, enable_async=enable_async)
    return cache


def test_teacher_head_loading_and_storage_policy(tmp_path):
    weight = torch.randn(11, 7)
    path = tmp_path / "teacher_head.safetensors"
    save_tensor_file(path, "lm_head.weight", weight)

    loaded = load_lm_head_weight(str(path))

    torch.testing.assert_close(loaded, weight)

    _assert_teacher_store_round_trips_lm_head_shards(tmp_path / "store")
    _assert_load_lm_head_weight_uses_tied_embedding(tmp_path / "tied")
    _assert_teacher_head_shard_view_loads_cross_shard_range(tmp_path / "view")
    _assert_teacher_head_manager_residency_policy(tmp_path)


def _assert_teacher_head_manager_residency_policy(tmp_path):
    weight0 = torch.randn(5, 3)
    weight1 = torch.randn(5, 3)
    path0 = tmp_path / "teacher0.safetensors"
    path1 = tmp_path / "teacher1.safetensors"
    save_tensor_file(path0, "lm_head.weight", weight0)
    save_tensor_file(path1, "lm_head.weight", weight1)

    manager = TeacherHeadManager({"0": str(path0), "1": str(path1)})

    loaded0 = manager.get(0, device="cpu")
    loaded1 = manager.get(1, device="cpu")

    torch.testing.assert_close(loaded0, weight0)
    torch.testing.assert_close(loaded1, weight1)
    assert manager._device_teacher_id == "1"

    _assert_teacher_head_manager_reloads_for_dtype_change(tmp_path / "dtype")
    _assert_teacher_head_manager_prefetches_teacher_store(tmp_path / "prefetch")


def _assert_teacher_head_manager_reloads_for_dtype_change(tmp_path):
    tmp_path.mkdir(parents=True)
    weight = torch.randn(5, 3)
    path = tmp_path / "teacher.safetensors"
    save_tensor_file(path, "lm_head.weight", weight)

    manager = TeacherHeadManager({"0": str(path)})

    loaded_fp32 = manager.get(0, device="cpu", dtype=torch.float32)
    loaded_bf16 = manager.get(0, device="cpu", dtype=torch.bfloat16)

    assert loaded_fp32.dtype == torch.float32
    assert loaded_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(loaded_bf16.float(), weight, rtol=1e-2, atol=1e-2)


def _assert_teacher_store_round_trips_lm_head_shards(tmp_path):
    weight = torch.randn(11, 7)
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir(parents=True)
    save_tensor_file(model_dir / "model.safetensors", "lm_head.weight", weight)

    store_dir = tmp_path / "teacher_store"
    manifest = prepare_lm_head_teacher_store(model_dir, store_dir, teacher_id=3, shard_rows=4)
    store = TeacherHeadStore(manifest)

    loaded = store.load_lm_head(3)
    via_loader = load_lm_head_weight(str(store_dir), teacher_id=3)

    torch.testing.assert_close(loaded, weight)
    torch.testing.assert_close(via_loader, weight)
    assert [shard.rows for shard in store.head_spec(3).shards] == [4, 4, 3]


def _assert_load_lm_head_weight_uses_tied_embedding(tmp_path):
    weight = torch.randn(11, 7)
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir(parents=True)
    save_file(
        {
            "model.embed_tokens.weight": weight,
            "model.layers.0.input_layernorm.weight": torch.ones(7),
        },
        str(model_dir / "model-00001-of-00001.safetensors"),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.input_layernorm.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(json.dumps({"tie_word_embeddings": True}), encoding="utf-8")

    loaded = load_lm_head_weight(str(model_dir))

    torch.testing.assert_close(loaded, weight)


def _assert_teacher_head_manager_prefetches_teacher_store(tmp_path):
    weight = torch.randn(5, 3)
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir(parents=True)
    save_tensor_file(model_dir / "model.safetensors", "lm_head.weight", weight)
    store_dir = tmp_path / "teacher_store"
    prepare_lm_head_teacher_store(model_dir, store_dir, teacher_id=0, shard_rows=2)

    manager = TeacherHeadManager({"0": str(store_dir)})
    manager.prefetch(0)
    loaded = manager.get(0, device="cpu")

    torch.testing.assert_close(loaded, weight)


def test_teacher_activation_cache_lifecycle_and_admission_policy():
    hidden = torch.randn(6, 4)
    cache = _mooncake_cache(hidden, enable_async=True)
    cache.prefetch(3)
    gathered = cache.get(3, torch.tensor([0, 5]), device="cpu")

    torch.testing.assert_close(gathered, hidden[[0, 5]])

    _assert_teacher_activation_cache_rejects_out_of_bounds_indices()
    _assert_teacher_activation_cache_reuses_device_cache()
    _assert_teacher_activation_cache_reloads_device_dtype()
    _assert_teacher_activation_cache_gathers_rank3_layer_slices()


def _assert_teacher_activation_cache_rejects_out_of_bounds_indices():
    hidden = torch.randn(6, 4)
    cache = _mooncake_cache(hidden)

    with pytest.raises(IndexError, match="negative"):
        cache.get(3, torch.tensor([[0, 2, -1], [1, 4, 3]]), device="cpu")
    with pytest.raises(IndexError, match="cache only has 6 rows"):
        cache.get(3, torch.tensor([0, 2, 6]), device="cpu")


def _assert_teacher_head_shard_view_loads_cross_shard_range(tmp_path):
    weight = torch.arange(11 * 7, dtype=torch.float32).reshape(11, 7)
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir(parents=True)
    save_tensor_file(model_dir / "model.safetensors", "lm_head.weight", weight)

    manifest = prepare_lm_head_teacher_store(model_dir, tmp_path / "teacher_store", teacher_id=3, shard_rows=4)
    view = TeacherHeadManager({"3": str(manifest)}).sharded_view(3, device="cpu", dtype=torch.float32)

    loaded = view.load_rows_device(3, 9)

    torch.testing.assert_close(loaded, weight[3:9])


def _assert_teacher_activation_cache_reuses_device_cache():
    hidden = torch.randn(6, 4)
    cache = _mooncake_cache(hidden)

    gathered = cache.get(3, torch.tensor([0, 5]), device="cpu", cache_device=True)
    device_cache = cache._device_tensor
    gathered_again = cache.get(3, torch.tensor([2, 4]), device="cpu", cache_device=True)

    torch.testing.assert_close(gathered, hidden[[0, 5]])
    torch.testing.assert_close(gathered_again, hidden[[2, 4]])
    assert cache._device_tensor is device_cache


def _assert_teacher_activation_cache_reloads_device_dtype():
    hidden = torch.randn(6, 4, dtype=torch.float32)
    cache = _mooncake_cache(hidden)

    bf16 = cache.get(3, torch.tensor([0, 1]), device="cpu", dtype=torch.bfloat16, cache_device=True)
    default_dtype = cache.get(3, torch.tensor([0, 1]), device="cpu", dtype=None, cache_device=True)

    assert bf16.dtype == torch.bfloat16
    assert default_dtype.dtype == hidden.dtype
    torch.testing.assert_close(default_dtype, hidden[[0, 1]])


def _assert_teacher_activation_cache_gathers_rank3_layer_slices():
    hidden = torch.arange(5 * 6 * 4, dtype=torch.float32).reshape(5, 6, 4)
    cache = _mooncake_cache(hidden)
    indices = torch.tensor([[0, 2], [5, 1]])

    gathered = cache.get_layer_slice(3, indices, 1, 4, device="cpu")
    expected = hidden[1:4].index_select(1, indices.reshape(-1)).permute(1, 0, 2).reshape(2, 2, 3, 4)
    torch.testing.assert_close(gathered, expected)

    device_indices = torch.tensor([0, 5])
    device_gathered = cache.get_layer_slice(
        3, device_indices, 2, 5, device="cpu", dtype=torch.bfloat16, cache_device=True
    )
    device_expected = hidden[2:5].index_select(1, device_indices).permute(1, 0, 2).to(dtype=torch.bfloat16)
    torch.testing.assert_close(device_gathered, device_expected)
    assert cache.shape(3) == (5, 6, 4)
