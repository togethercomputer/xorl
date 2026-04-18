"""Lifecycle regression tests for NCCL weight sync rendezvous."""

from types import SimpleNamespace

import torch
import torch.distributed as dist

from xorl.server.weight_sync.backends import nccl_broadcast as nccl_broadcast_module
from xorl.server.weight_sync.backends.nccl_broadcast import EndpointInfo, NCCLWeightSynchronizer


class _JoinDrivenThread:
    """Test double that defers target execution until ``join()``."""

    instances = []

    def __init__(self, target):
        self._target = target
        self.join_called = False
        _JoinDrivenThread.instances.append(self)

    def start(self):
        return None

    def join(self):
        self.join_called = True
        self._target()


class _StickyTCPStore:
    """Fake rendezvous store that keeps ports reserved after destroy."""

    open_ports = set()
    next_ephemeral_port = 31000

    def __init__(self, host_name, port, world_size, is_master, timeout):
        del host_name, world_size, is_master, timeout
        if port == 0:
            port = self.next_ephemeral_port
            _StickyTCPStore.next_ephemeral_port += 1
        if port in self.open_ports:
            raise RuntimeError(f"EADDRINUSE: {port}")
        self.port = port
        self.open_ports.add(port)

    def close(self):
        return None


class _FakePrefixStore:
    def __init__(self, prefix, store):
        self.prefix = prefix
        self.store = store


class _FakeProcessGroup:
    def __init__(self, store):
        self.store = store
        self.destroyed = False


def _make_sync(master_port: int = 0) -> NCCLWeightSynchronizer:
    return NCCLWeightSynchronizer(
        endpoints=[EndpointInfo(host="127.0.0.1", port=12345, world_size=1)],
        master_address="127.0.0.1",
        master_port=master_port,
        group_name="weight_sync_group",
        device="cuda:0",
    )


def test_init_nccl_group_fails_before_starting_inference_when_store_bind_fails(monkeypatch):
    sync = _make_sync()
    inference_started = False

    _JoinDrivenThread.instances.clear()
    monkeypatch.setattr(nccl_broadcast_module, "Thread", _JoinDrivenThread)
    monkeypatch.setattr(sync, "_create_training_store", lambda: (_ for _ in ()).throw(RuntimeError("EADDRINUSE")))

    def fake_inference_init():
        nonlocal inference_started
        inference_started = True
        return []

    monkeypatch.setattr(sync, "_init_inference_endpoints", fake_inference_init)

    assert sync.init_nccl_group() is False
    assert inference_started is False
    assert _JoinDrivenThread.instances == []


def test_destroy_nccl_group_reinit_uses_fresh_ephemeral_rendezvous_port_when_old_one_is_sticky(monkeypatch):
    sync = _make_sync()
    destroyed_groups = []

    _StickyTCPStore.open_ports.clear()
    _StickyTCPStore.next_ephemeral_port = 31000

    def fake_new_process_group_helper(world_size, rank, group_ranks, backend, store, **kwargs):
        del world_size, rank, group_ranks, backend, kwargs
        return _FakeProcessGroup(store.store), None

    def fake_destroy_process_group(process_group):
        process_group.destroyed = True
        destroyed_groups.append(process_group)

    monkeypatch.setattr(nccl_broadcast_module, "TCPStore", _StickyTCPStore)
    monkeypatch.setattr(nccl_broadcast_module, "PrefixStore", _FakePrefixStore)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "destroy_process_group", fake_destroy_process_group)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(nccl_broadcast_module, "Backend", lambda name: name)
    monkeypatch.setattr(nccl_broadcast_module, "_new_process_group_helper", fake_new_process_group_helper)
    monkeypatch.setattr(nccl_broadcast_module, "_world", SimpleNamespace(pg_group_ranks={}))
    monkeypatch.setattr(nccl_broadcast_module, "default_pg_timeout", object())
    monkeypatch.setattr(sync, "_destroy_inference_endpoints", lambda: None)

    first_pg = sync._init_training_process_group()
    sync.process_group = first_pg
    first_port = sync._active_master_port
    sync.destroy_nccl_group()

    assert destroyed_groups == [first_pg]

    sync.process_group = sync._init_training_process_group()
    assert first_port == 31000
    assert sync._active_master_port == 31001


def test_explicit_master_port_is_honored(monkeypatch):
    sync = _make_sync(master_port=29600)

    _StickyTCPStore.open_ports.clear()

    def fake_new_process_group_helper(world_size, rank, group_ranks, backend, store, **kwargs):
        del world_size, rank, group_ranks, backend, kwargs
        return _FakeProcessGroup(store.store), None

    monkeypatch.setattr(nccl_broadcast_module, "TCPStore", _StickyTCPStore)
    monkeypatch.setattr(nccl_broadcast_module, "PrefixStore", _FakePrefixStore)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(nccl_broadcast_module, "Backend", lambda name: name)
    monkeypatch.setattr(nccl_broadcast_module, "_new_process_group_helper", fake_new_process_group_helper)
    monkeypatch.setattr(nccl_broadcast_module, "_world", SimpleNamespace(pg_group_ranks={}))
    monkeypatch.setattr(nccl_broadcast_module, "default_pg_timeout", object())

    process_group = sync._init_training_process_group()

    assert process_group.store.port == 29600
    assert sync._active_master_port == 29600
