import contextlib
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._tensor import Replicate
from torch.distributed._tensor import Shard as DTShard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from xorl.models import module_utils


pytestmark = [pytest.mark.cpu]


class _DummyModel:
    def named_buffers(self):
        return []

    def named_parameters(self):
        return []

    def to_empty(self, device):
        self.device = device


class _FakeDeviceMesh:
    ndim = 1

    def __init__(self, size: int, local_rank: int):
        self._size = size
        self._local_rank = local_rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._local_rank


class _FakeDTensor:
    def __init__(self, local_tensor: torch.Tensor, mesh_size: int, local_rank: int, placements):
        self._local_tensor = local_tensor
        self.device_mesh = _FakeDeviceMesh(mesh_size, local_rank)
        self.placements = placements


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cpu_dtensor_materialize_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        module_utils._cpu_save_device_mesh_cache.clear()
        mesh = DeviceMesh(
            "cpu",
            mesh=torch.arange(world_size).view(2, 2),
            mesh_dim_names=("ep", "fsdp"),
            backend_override=(("gloo", None), ("gloo", None)),
        )
        full_tensor = torch.arange(16, dtype=torch.float32).view(4, 4)
        row = rank // 2
        col = rank % 2
        local_tensor = full_tensor[row * 2 : (row + 1) * 2, col * 2 : (col + 1) * 2].clone()
        dtensor = DTensor.from_local(
            local_tensor,
            device_mesh=mesh,
            placements=[DTShard(0), DTShard(1)],
            shape=full_tensor.shape,
            stride=full_tensor.stride(),
        )
        materialized = module_utils._materialize_tensor_for_save(dtensor)
        assert materialized.device.type == "cpu"
        assert torch.equal(materialized, full_tensor)
    finally:
        dist.destroy_process_group()


def _cpu_dtensor_materialize_to_rank_worker(rank: int, world_size: int, port: int, dst_rank: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        module_utils._cpu_save_device_mesh_cache.clear()
        mesh = DeviceMesh(
            "cpu",
            mesh=torch.arange(world_size),
            mesh_dim_names=("fsdp",),
            backend_override=(("gloo", None),),
        )
        full_tensor = torch.arange(18, dtype=torch.float32).view(9, 2)
        chunk_size = 3
        start = rank * chunk_size
        stop = min(start + chunk_size, full_tensor.shape[0])
        local_tensor = full_tensor[start:stop].clone()
        dtensor = DTensor.from_local(
            local_tensor,
            device_mesh=mesh,
            placements=[DTShard(0)],
            shape=full_tensor.shape,
            stride=full_tensor.stride(),
        )
        materialized = module_utils._materialize_tensor_for_save(dtensor, dst_rank=dst_rank)
        if rank == dst_rank:
            assert materialized is not None
            assert materialized.device.type == "cpu"
            assert torch.equal(materialized, full_tensor)
        else:
            assert materialized is None
    finally:
        dist.destroy_process_group()


def _cpu_dtensor_materialize_2d_to_rank_worker(rank: int, world_size: int, port: int, dst_rank: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        module_utils._cpu_save_device_mesh_cache.clear()
        mesh = DeviceMesh(
            "cpu",
            mesh=torch.arange(world_size).view(2, 2),
            mesh_dim_names=("ep", "fsdp"),
            backend_override=(("gloo", None), ("gloo", None)),
        )
        full_tensor = torch.arange(16, dtype=torch.float32).view(4, 4)
        row = rank // 2
        col = rank % 2
        local_tensor = full_tensor[row * 2 : (row + 1) * 2, col * 2 : (col + 1) * 2].clone()
        dtensor = DTensor.from_local(
            local_tensor,
            device_mesh=mesh,
            placements=[DTShard(0), DTShard(1)],
            shape=full_tensor.shape,
            stride=full_tensor.stride(),
        )
        materialized = module_utils._materialize_tensor_for_save(dtensor, dst_rank=dst_rank)
        if rank == dst_rank:
            assert materialized is not None
            assert materialized.device.type == "cpu"
            assert torch.equal(materialized, full_tensor)
        else:
            assert materialized is None
    finally:
        dist.destroy_process_group()


def test_copy_into_existing_dtensor_shard_for_replicated_tensor():
    dtensor = _FakeDTensor(torch.zeros(4, dtype=torch.float32), mesh_size=4, local_rank=2, placements=(Replicate(),))
    full_tensor = torch.arange(4, dtype=torch.float32)

    copied = module_utils._copy_into_existing_dtensor_shard(dtensor, full_tensor)

    assert copied is True
    assert torch.equal(dtensor._local_tensor, full_tensor)


def test_materialize_tensor_for_save_uses_cpu_mesh_for_dtensors():
    port = _find_free_port()
    mp.start_processes(
        _cpu_dtensor_materialize_worker,
        args=(4, port),
        nprocs=4,
        join=True,
        start_method="fork",
    )


def test_materialize_tensor_for_save_gathers_1d_dtensor_to_writer_rank():
    port = _find_free_port()
    mp.start_processes(
        _cpu_dtensor_materialize_to_rank_worker,
        args=(4, port, 2),
        nprocs=4,
        join=True,
        start_method="fork",
    )


def test_materialize_tensor_for_save_gathers_2d_dtensor_to_writer_rank():
    port = _find_free_port()
    mp.start_processes(
        _cpu_dtensor_materialize_2d_to_rank_worker,
        args=(4, port, 3),
        nprocs=4,
        join=True,
        start_method="fork",
    )


def test_copy_into_existing_dtensor_shard_for_sharded_tensor():
    dtensor = _FakeDTensor(
        torch.zeros(2, 3, dtype=torch.float32),
        mesh_size=4,
        local_rank=1,
        placements=(DTShard(0),),
    )
    full_tensor = torch.arange(24, dtype=torch.float32).view(8, 3)

    copied = module_utils._copy_into_existing_dtensor_shard(dtensor, full_tensor)

    assert copied is True
    assert torch.equal(dtensor._local_tensor, full_tensor[2:4])


def test_copy_into_existing_dtensor_shard_trims_padded_tail_shards():
    dtensor = _FakeDTensor(
        torch.zeros(0, 3, dtype=torch.float32),
        mesh_size=8,
        local_rank=6,
        placements=(DTShard(0),),
    )
    full_tensor = torch.arange(15, dtype=torch.float32).view(5, 3)

    copied = module_utils._copy_into_existing_dtensor_shard(dtensor, full_tensor)

    assert copied is True
    assert tuple(dtensor._local_tensor.shape) == (0, 3)


def test_copy_into_existing_dtensor_shard_rejects_shape_mismatched_replicates():
    dtensor = _FakeDTensor(torch.zeros(1, 3, dtype=torch.float32), mesh_size=4, local_rank=0, placements=(Replicate(),))
    full_tensor = torch.arange(15, dtype=torch.float32).view(5, 3)

    copied = module_utils._copy_into_existing_dtensor_shard(dtensor, full_tensor)

    assert copied is False
    assert torch.equal(dtensor._local_tensor, torch.zeros(1, 3, dtype=torch.float32))


def test_broadcast_object_list_serializes_over_tensor_broadcast_for_nccl_groups(monkeypatch):
    fake_group = object()
    state = {"rank": 3}
    stored = []

    def fake_broadcast(tensor, src=0, group=None):
        assert group is fake_group
        if state["rank"] == src:
            stored.append(tensor.detach().cpu().clone())
        else:
            tensor.copy_(stored.pop(0).to(tensor.device))

    fake_dist = SimpleNamespace(
        get_rank=lambda: state["rank"],
        broadcast=fake_broadcast,
        broadcast_object_list=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected object broadcast")
        ),
    )

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "_get_object_broadcast_device", lambda group: torch.device("cpu"))

    source_payload = [[("payload", torch.Size([2, 3]), torch.float32)]]
    module_utils._broadcast_object_list(source_payload, src=3, group=fake_group)

    state["rank"] = 1
    received_payload = [None]
    module_utils._broadcast_object_list(received_payload, src=3, group=fake_group)

    assert received_payload == source_payload


def test_get_object_broadcast_device_uses_default_nccl_group(monkeypatch):
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_backend=lambda group=None: "nccl",
    )

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(module_utils, "get_device_id", lambda: 3)

    assert module_utils._get_object_broadcast_device(None) == torch.device("cuda:3")


def test_broadcast_object_list_weight_load_uses_weight_load_group(monkeypatch):
    fake_group = object()
    calls = []

    monkeypatch.setattr(module_utils, "_get_weight_load_group", lambda: fake_group)
    monkeypatch.setattr(
        module_utils,
        "_broadcast_object_list",
        lambda obj_list, src=0, group=None: calls.append((obj_list, src, group)),
    )

    payload = [["checkpoint-paths"]]
    module_utils._broadcast_object_list_weight_load(payload, src=7)

    assert calls == [(payload, 7, fake_group)]


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
    monkeypatch.setattr(module_utils, "_get_object_broadcast_device", lambda group: None)
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
    monkeypatch.setattr(module_utils, "_get_object_broadcast_device", lambda group: None)
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
    monkeypatch.setattr(module_utils, "_get_object_broadcast_device", lambda group: None)
    monkeypatch.setattr(module_utils, "_try_load_state_dict_local", fake_try_load_state_dict_local)

    iterators = module_utils._try_load_state_dict("dummy-weights")

    assert local_resolution_calls == []
    assert [it.filepath for it in iterators] == ["shard-0.safetensors", "shard-1.safetensors"]


def test_try_load_state_dict_local_directory_skips_broadcast(monkeypatch):
    local_resolution_calls = []

    fake_dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_rank=lambda: 7,
        get_world_size=lambda: 64,
        broadcast_object_list=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not broadcast")),
    )

    def fake_try_load_state_dict_local(weights_path, **kwargs):
        local_resolution_calls.append(weights_path)
        return [module_utils.StateDictIterator("local-shard.safetensors")]

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils.os.path, "isdir", lambda path: path == "dummy-local-dir")
    monkeypatch.setattr(module_utils, "_try_load_state_dict_local", fake_try_load_state_dict_local)

    iterators = module_utils._try_load_state_dict("dummy-local-dir")

    assert local_resolution_calls == ["dummy-local-dir"]
    assert [it.filepath for it in iterators] == ["local-shard.safetensors"]


def test_grouped_load_weights_uses_filtered_prefetch_on_group_leader(monkeypatch):
    batch_meta_calls = []
    dispatched = []
    transfer_calls = []
    handler_kwargs = []
    handler_calls = {"dense_loaded": [], "expert_loaded": []}
    fake_group = object()
    fake_dense_group = object()

    class _DenseHandler:
        def get_skip_key_fn(self):
            return None

        def on_load_weight(self, key, tensor):
            handler_calls["dense_loaded"].append(key)
            return [(key, tensor)]

        def on_load_complete(self):
            return []

    class _ExpertHandler:
        def get_skip_key_fn(self):
            return None

        def on_load_weight(self, key, tensor):
            handler_calls["expert_loaded"].append(key)
            return [("model.layers.0.mlp.experts.gate_proj", torch.arange(8, dtype=torch.float32).view(8, 1, 1))]

        def on_load_complete(self):
            return []

    class _GroupedModel:
        def named_buffers(self):
            return []

        def named_parameters(self):
            return [
                ("keep.weight", None),
                ("model.layers.0.mlp.experts.gate_proj", None),
            ]

        def named_modules(self):
            return []

        def to_empty(self, device):
            self.device = device

        def get_checkpoint_handler(self, **kwargs):
            handler_kwargs.append(kwargs)
            return _DenseHandler() if kwargs["ep_size"] == 1 else _ExpertHandler()

    def fake_broadcast_object_list(obj, src=0, group=None, device=None):
        batch_meta_calls.append((src, group, obj[0]))
        if obj[0] is None:
            obj[0] = []

    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 128,
        get_process_group_ranks=lambda group: (
            [0, 1, 2, 3, 4, 5, 6, 7] if group is fake_dense_group else [0, 16, 32, 48, 64, 80, 96, 112]
        ),
        broadcast=lambda tensor, src=0, group=None: transfer_calls.append(
            ("broadcast", src, group, tuple(tensor.shape))
        ),
        scatter=lambda tensor, scatter_list=None, src=0, group=None: transfer_calls.append(
            ("scatter", src, group, tuple(tensor.shape), None if scatter_list is None else tuple(scatter_list[0].shape))
        ),
        broadcast_object_list=fake_broadcast_object_list,
    )

    prefetch_calls = []

    def fake_prefetch_filtered(state_dict_iterators, skip_key_fn, prefetch_count):
        assert state_dict_iterators == ["shard-0"]
        if skip_key_fn("model.layers.0.mlp.experts.0.gate_proj.weight"):
            prefetch_calls.append(("dense", prefetch_count))
            assert prefetch_count == 1
            assert not skip_key_fn("keep.weight")
            yield ({"keep.weight": torch.tensor([2.0])}, [])
            return

        prefetch_calls.append(("expert", prefetch_count))
        assert prefetch_count == 1
        assert skip_key_fn("keep.weight")
        yield ({"model.layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([[3.0]])}, [])

    def fail_prefetch(*args, **kwargs):
        raise AssertionError("_prefetch_shards should not be used when handler skip filtering is available")

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "_get_object_broadcast_device", lambda group: None)
    monkeypatch.setattr(module_utils, "_get_grouped_weight_load_group", lambda _ps: fake_group)
    monkeypatch.setattr(module_utils, "_get_grouped_dense_weight_load_group", lambda: fake_dense_group)
    monkeypatch.setattr(
        module_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(global_rank=0, pp_enabled=False, ep_enabled=True, ep_rank=0, ep_size=16),
    )
    monkeypatch.setattr(module_utils, "_build_compiled_key_map", lambda *args, **kwargs: {})
    monkeypatch.setattr(module_utils, "_shrink_expert_params_for_ep", lambda model: None)
    monkeypatch.setattr(
        module_utils,
        "_get_checkpoint_keys",
        lambda weights_path: {"keep.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"},
    )
    monkeypatch.setattr(
        module_utils,
        "_get_expert_scatter_target_shape",
        lambda model, parameter_name, tensor, parallel_plan, parallel_state: (
            (1, 1, 1) if parameter_name == "model.layers.0.mlp.experts.gate_proj" else None
        ),
    )
    monkeypatch.setattr(module_utils, "_load_state_dict", lambda weights_path: ["shard-0"])
    monkeypatch.setattr(module_utils, "_prefetch_shards_filtered", fake_prefetch_filtered)
    monkeypatch.setattr(module_utils, "_prefetch_shards", fail_prefetch)
    monkeypatch.setattr(module_utils, "_dispatch_parameter", lambda *args, **kwargs: dispatched.append(args[1]))
    monkeypatch.setattr(module_utils, "post_process_after_weight_loading", lambda *args, **kwargs: None)
    monkeypatch.setattr(module_utils, "empty_cache", lambda: None)

    module_utils.grouped_load_weights(_GroupedModel(), "dummy-weights", init_device="cpu")

    assert handler_kwargs == [
        {
            "checkpoint_keys": {"keep.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"},
            "ep_rank": 0,
            "ep_size": 1,
            "is_broadcast": False,
            "weights_path": "dummy-weights",
            "device": None,
            "dtype": None,
        },
        {
            "checkpoint_keys": {"keep.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"},
            "ep_rank": 0,
            "ep_size": 16,
            "is_broadcast": False,
            "weights_path": "dummy-weights",
            "device": None,
            "dtype": None,
        },
    ]
    assert prefetch_calls == [("dense", 1), ("expert", 1)]
    assert handler_calls["dense_loaded"] == ["keep.weight"]
    assert handler_calls["expert_loaded"] == ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert dispatched == ["keep.weight", "model.layers.0.mlp.experts.gate_proj"]
    assert transfer_calls == [
        ("broadcast", 0, fake_dense_group, (1,)),
        ("scatter", 0, fake_group, (1, 1, 1), (1, 1, 1)),
    ]
    assert batch_meta_calls[0] == (
        0,
        fake_dense_group,
        [
            ("keep.weight", torch.Size([1]), torch.float32, "broadcast"),
        ],
    )
    assert batch_meta_calls[1] == (
        0,
        fake_group,
        [
            ("model.layers.0.mlp.experts.gate_proj", torch.Size([1, 1, 1]), torch.float32, "expert_scatter"),
        ],
    )


def test_grouped_load_weights_falls_back_without_ep_group(monkeypatch):
    called = []

    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
    )

    monkeypatch.setattr(module_utils, "dist", fake_dist)
    monkeypatch.setattr(module_utils, "_get_grouped_weight_load_group", lambda _ps: None)
    monkeypatch.setattr(
        module_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(global_rank=0, pp_enabled=False, ep_enabled=False, ep_fsdp_device_mesh=None),
    )
    monkeypatch.setattr(
        module_utils,
        "rank0_load_and_broadcast_weights",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    module_utils.grouped_load_weights(_DummyModel(), "dummy-weights", init_device="cpu")

    assert len(called) == 1
