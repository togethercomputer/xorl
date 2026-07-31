from types import SimpleNamespace

import pytest

from xorl.checkpoint import checkpointer
from xorl.checkpoint.checkpointer import DistributedCheckpointer


pytestmark = pytest.mark.cpu


def test_sync_dcp_process_group_uses_gloo_for_nccl_default(monkeypatch):
    created = []
    fake_group = object()

    monkeypatch.setattr(
        checkpointer,
        "dist",
        SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_backend=lambda: "nccl",
            new_group=lambda backend: created.append(backend) or fake_group,
        ),
    )
    DistributedCheckpointer._sync_process_group = None

    assert DistributedCheckpointer._get_sync_process_group() is fake_group
    assert DistributedCheckpointer._get_sync_process_group() is fake_group
    assert created == ["gloo"]


def test_sync_dcp_process_group_uses_default_for_gloo(monkeypatch):
    monkeypatch.setattr(
        checkpointer,
        "dist",
        SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_backend=lambda: "gloo",
        ),
    )
    DistributedCheckpointer._sync_process_group = None

    assert DistributedCheckpointer._get_sync_process_group() is None
