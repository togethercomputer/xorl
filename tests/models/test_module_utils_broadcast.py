from types import SimpleNamespace

import pytest

from xorl.models import module_utils


pytestmark = [pytest.mark.cpu]


class _DummyModel:
    def named_buffers(self):
        return []

    def named_parameters(self):
        return []

    def to_empty(self, device):
        self.device = device


def test_rank0_broadcast_path_calls_load_state_dict_on_nonzero_ranks(monkeypatch):
    calls = []

    def fake_broadcast_object_list(obj, src=0):
        if obj[0] is None:
            obj[0] = []

    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        broadcast=lambda tensor, src=0: None,
        broadcast_object_list=fake_broadcast_object_list,
    )

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(
        module_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(global_rank=1, pp_enabled=False),
    )
    monkeypatch.setattr(module_utils, "_build_compiled_key_map", lambda *args, **kwargs: {})
    monkeypatch.setattr(module_utils, "_shrink_expert_params_for_ep", lambda model: None)
    monkeypatch.setattr(module_utils, "post_process_after_weight_loading", lambda *args, **kwargs: None)

    def fake_load_state_dict(weights_path):
        calls.append(weights_path)
        return []

    monkeypatch.setattr(module_utils, "_load_state_dict", fake_load_state_dict)

    module_utils.rank0_load_and_broadcast_weights(_DummyModel(), "dummy-weights", init_device="cpu")

    assert calls == ["dummy-weights"]


def test_try_load_state_dict_uses_rank0_for_local_resolution(monkeypatch):
    local_resolution_calls = []

    def fake_broadcast_object_list(obj, src=0, group=None):
        assert src == 0
        obj[0] = ["shard-0.safetensors", "shard-1.safetensors"]

    fake_dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_rank=lambda: 1,
        get_world_size=lambda: 64,
        broadcast_object_list=fake_broadcast_object_list,
    )

    def fake_try_load_state_dict_local(weights_path, **kwargs):
        local_resolution_calls.append(weights_path)
        return [module_utils.StateDictIterator("local-only.safetensors")]

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "_get_cpu_group", lambda: object())
    monkeypatch.setattr(module_utils, "_try_load_state_dict_local", fake_try_load_state_dict_local)

    iterators = module_utils._try_load_state_dict("dummy-weights")

    assert local_resolution_calls == []
    assert [it.filepath for it in iterators] == ["shard-0.safetensors", "shard-1.safetensors"]
