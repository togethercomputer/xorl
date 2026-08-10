import sys
import types

import pytest

from xorl.server.runner.grad_sync import hsdp_all_reduce_microbatch_context, should_defer_hsdp_all_reduce


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeFSDP:
    def __init__(self):
        self.requires_all_reduce_calls = []

    def set_requires_all_reduce(self, value):
        self.requires_all_reduce_calls.append(bool(value))


@pytest.fixture(autouse=True)
def fake_fsdp_module(monkeypatch):
    module = types.ModuleType("torch.distributed._composable.fsdp.fully_shard")
    module.FSDPModule = _FakeFSDP
    monkeypatch.setitem(sys.modules, "torch.distributed._composable.fsdp.fully_shard", module)


def test_hsdp_all_reduce_context_defers_until_last_microbatch():
    model = _FakeFSDP()
    train_config = {
        "defer_grad_sync_in_accumulation": True,
        "data_parallel_replicate_size": 2,
    }

    defer_all_reduce = should_defer_hsdp_all_reduce(model, train_config, n_micro_batches=2)
    assert defer_all_reduce is True

    for microbatch_idx in range(2):
        with hsdp_all_reduce_microbatch_context(
            model,
            defer_all_reduce,
            is_last_micro_batch=microbatch_idx == 1,
        ):
            pass

    assert model.requires_all_reduce_calls == [False, True, True, True]


def test_hsdp_all_reduce_deferral_requires_replicate_dimension():
    model = _FakeFSDP()
    train_config = {
        "defer_grad_sync_in_accumulation": True,
        "data_parallel_replicate_size": 1,
    }

    assert should_defer_hsdp_all_reduce(model, train_config, n_micro_batches=2) is False
