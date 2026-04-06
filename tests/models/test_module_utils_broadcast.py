from types import SimpleNamespace

import pytest
import torch

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

    def fake_broadcast_object_list(obj, src=0, group=None, device=None):
        if obj[0] is None:
            obj[0] = []

    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        broadcast=lambda tensor, src=0, group=None: None,
        broadcast_object_list=fake_broadcast_object_list,
    )

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "_get_weight_load_group", lambda: None)
    monkeypatch.setattr(module_utils, "_get_weight_load_object_device", lambda: None)
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


def test_rank0_broadcast_path_uses_filtered_prefetch_for_handler_skips(monkeypatch):
    batch_meta_calls = []
    dispatched = []
    handler_calls = {"loaded": [], "skipped": []}

    class _Handler:
        def get_skip_key_fn(self):
            return lambda key: key == "skip.expert.weight"

        def on_skip_weight(self, key):
            handler_calls["skipped"].append(key)
            return [("merged.skip.weight", torch.tensor([1.0]))]

        def on_load_weight(self, key, tensor):
            handler_calls["loaded"].append(key)
            return [(key, tensor)]

        def on_load_complete(self):
            return []

    class _BroadcastModel:
        def named_buffers(self):
            return []

        def named_parameters(self):
            return [
                ("merged.skip.weight", None),
                ("keep.weight", None),
            ]

        def to_empty(self, device):
            self.device = device

        def get_checkpoint_handler(self, **kwargs):
            return _Handler()

    def fake_broadcast_object_list(obj, src=0, group=None, device=None):
        batch_meta_calls.append(obj[0])

    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        broadcast=lambda tensor, src=0, group=None: None,
        broadcast_object_list=fake_broadcast_object_list,
    )

    def fake_prefetch_filtered(state_dict_iterators, skip_key_fn, prefetch_count):
        assert state_dict_iterators == ["shard-0"]
        assert prefetch_count == 2
        assert skip_key_fn("skip.expert.weight")
        yield ({"keep.weight": torch.tensor([2.0])}, ["skip.expert.weight"])

    def fail_prefetch(*args, **kwargs):
        raise AssertionError("_prefetch_shards should not be used when handler skip filtering is available")

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "_get_weight_load_group", lambda: None)
    monkeypatch.setattr(module_utils, "_get_weight_load_object_device", lambda: None)
    monkeypatch.setattr(
        module_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(global_rank=0, pp_enabled=False),
    )
    monkeypatch.setattr(module_utils, "_build_compiled_key_map", lambda *args, **kwargs: {})
    monkeypatch.setattr(module_utils, "_shrink_expert_params_for_ep", lambda model: None)
    monkeypatch.setattr(
        module_utils, "_get_checkpoint_keys", lambda weights_path: {"skip.expert.weight", "keep.weight"}
    )
    monkeypatch.setattr(module_utils, "_load_state_dict", lambda weights_path: ["shard-0"])
    monkeypatch.setattr(module_utils, "_prefetch_shards_filtered", fake_prefetch_filtered)
    monkeypatch.setattr(module_utils, "_prefetch_shards", fail_prefetch)
    monkeypatch.setattr(module_utils, "_dispatch_parameter", lambda *args, **kwargs: dispatched.append(args[1]))
    monkeypatch.setattr(module_utils, "post_process_after_weight_loading", lambda *args, **kwargs: None)
    monkeypatch.setattr(module_utils, "empty_cache", lambda: None)

    module_utils.rank0_load_and_broadcast_weights(_BroadcastModel(), "dummy-weights", init_device="cpu")

    assert handler_calls["skipped"] == ["skip.expert.weight"]
    assert handler_calls["loaded"] == ["keep.weight"]
    assert dispatched == ["merged.skip.weight", "keep.weight"]
    assert batch_meta_calls[0] == [
        ("merged.skip.weight", torch.Size([1]), torch.float32, "broadcast"),
        ("keep.weight", torch.Size([1]), torch.float32, "broadcast"),
    ]


def test_try_load_state_dict_uses_rank0_for_local_resolution(monkeypatch):
    local_resolution_calls = []

    def fake_broadcast_object_list(obj, src=0, group=None, device=None):
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
    monkeypatch.setattr(module_utils, "_get_weight_load_group", lambda: None)
    monkeypatch.setattr(module_utils, "_get_weight_load_object_device", lambda: None)
    monkeypatch.setattr(module_utils, "_get_cpu_group", lambda: object())
    monkeypatch.setattr(module_utils, "_try_load_state_dict_local", fake_try_load_state_dict_local)

    iterators = module_utils._try_load_state_dict("dummy-weights")

    assert local_resolution_calls == []
    assert [it.filepath for it in iterators] == ["shard-0.safetensors", "shard-1.safetensors"]
