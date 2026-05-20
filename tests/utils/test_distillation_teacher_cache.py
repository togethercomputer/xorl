import pytest
import torch

from tests._helpers.opd import save_tensor_file
from xorl.distillation import (
    TeacherActivationCache,
    TeacherHeadManager,
    TeacherHeadStore,
    load_lm_head_weight,
    prepare_lm_head_teacher_store,
)


def test_load_lm_head_weight_from_safetensors_file(tmp_path):
    weight = torch.randn(11, 7)
    path = tmp_path / "teacher_head.safetensors"
    save_tensor_file(path, "lm_head.weight", weight)

    loaded = load_lm_head_weight(str(path))

    torch.testing.assert_close(loaded, weight)


def test_teacher_head_manager_keeps_one_device_head(tmp_path):
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


def test_teacher_head_manager_reloads_for_dtype_change(tmp_path):
    weight = torch.randn(5, 3)
    path = tmp_path / "teacher.safetensors"
    save_tensor_file(path, "lm_head.weight", weight)

    manager = TeacherHeadManager({"0": str(path)})

    loaded_fp32 = manager.get(0, device="cpu", dtype=torch.float32)
    loaded_bf16 = manager.get(0, device="cpu", dtype=torch.bfloat16)

    assert loaded_fp32.dtype == torch.float32
    assert loaded_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(loaded_bf16.float(), weight, rtol=1e-2, atol=1e-2)


def test_teacher_store_round_trips_lm_head_shards(tmp_path):
    weight = torch.randn(11, 7)
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir()
    save_tensor_file(model_dir / "model.safetensors", "lm_head.weight", weight)

    store_dir = tmp_path / "teacher_store"
    manifest = prepare_lm_head_teacher_store(model_dir, store_dir, teacher_id=3, shard_rows=4)
    store = TeacherHeadStore(manifest)

    loaded = store.load_lm_head(3)
    via_loader = load_lm_head_weight(str(store_dir), teacher_id=3)

    torch.testing.assert_close(loaded, weight)
    torch.testing.assert_close(via_loader, weight)
    assert [shard.rows for shard in store.head_spec(3).shards] == [4, 4, 3]


def test_teacher_head_manager_prefetches_teacher_store(tmp_path):
    weight = torch.randn(5, 3)
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir()
    save_tensor_file(model_dir / "model.safetensors", "lm_head.weight", weight)
    store_dir = tmp_path / "teacher_store"
    prepare_lm_head_teacher_store(model_dir, store_dir, teacher_id=0, shard_rows=2)

    manager = TeacherHeadManager({"0": str(store_dir)})
    manager.prefetch(0)
    loaded = manager.get(0, device="cpu")

    torch.testing.assert_close(loaded, weight)


def test_teacher_activation_cache_gathers_indices(tmp_path):
    hidden = torch.randn(6, 4)
    path = tmp_path / "hidden_states.safetensors"
    save_tensor_file(path, "hidden_states", hidden)

    cache = TeacherActivationCache({"3": str(path)})
    indices = torch.tensor([[0, 2, 5], [1, 4, 3]])

    gathered = cache.get(3, indices, device="cpu")

    torch.testing.assert_close(gathered, hidden[indices])


def test_teacher_activation_cache_prefetches(tmp_path):
    hidden = torch.randn(6, 4)
    path = tmp_path / "hidden_states.safetensors"
    save_tensor_file(path, "hidden_states", hidden)

    cache = TeacherActivationCache({"3": str(path)})
    cache.prefetch(3)
    gathered = cache.get(3, torch.tensor([0, 5]), device="cpu")

    torch.testing.assert_close(gathered, hidden[[0, 5]])


def test_teacher_activation_cache_rejects_negative_indices(tmp_path):
    """Negative indices used to be silently clamped to 0, masking producer bugs."""
    hidden = torch.randn(6, 4)
    path = tmp_path / "hidden_states.safetensors"
    save_tensor_file(path, "hidden_states", hidden)

    cache = TeacherActivationCache({"3": str(path)})
    bad_indices = torch.tensor([[0, 2, -1], [1, 4, 3]])

    with pytest.raises(IndexError, match="negative"):
        cache.get(3, bad_indices, device="cpu")


def test_teacher_activation_cache_rejects_out_of_range_indices(tmp_path):
    hidden = torch.randn(6, 4)
    path = tmp_path / "hidden_states.safetensors"
    save_tensor_file(path, "hidden_states", hidden)

    cache = TeacherActivationCache({"3": str(path)})
    bad_indices = torch.tensor([0, 2, 6])

    with pytest.raises(IndexError, match="cache only has 6 rows"):
        cache.get(3, bad_indices, device="cpu")
