import pytest
import torch
from torch import nn

from xorl.server.runner.checkpoint.manager import CheckpointManager
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, device="meta"))
        self.to_empty_device = None

    def to_empty(self, *, device, recurse=True):  # noqa: ARG002
        self.to_empty_device = device
        self.weight = nn.Parameter(torch.empty(2, device="cpu"))
        return self


class _DummyCheckpointer:
    def __init__(self):
        self.path = None
        self.state = None

    def load(self, path, state):
        self.path = path
        self.state = state
        state["extra_state"].update(
            {
                "global_step": 7,
                "global_forward_backward_step": 11,
                "torch_rng_state": torch.get_rng_state(),
            }
        )


def test_checkpoint_manager_materializes_skip_mode_and_omits_missing_optimizer(monkeypatch):
    model = _MetaModel()
    checkpointer = _DummyCheckpointer()
    manager = CheckpointManager(
        model=model,
        optimizer=object(),
        checkpointer=checkpointer,
        lora_config={},
        model_config={},
        train_config={"load_weights_mode": "skip"},
        rank=0,
        local_rank=0,
    )
    monkeypatch.setattr("xorl.server.runner.checkpoint.manager.get_device_type", lambda: "cpu")
    monkeypatch.setattr(manager, "_checkpoint_has_optimizer", lambda _path: False)

    result = manager.load_state("/tmp/model-only-dcp", load_optimizer=True)

    assert model.to_empty_device == "cpu"
    assert checkpointer.path == "/tmp/model-only-dcp"
    assert "optimizer" not in checkpointer.state
    assert result["load_optimizer"] is False
    assert manager.global_step == 7
    assert manager.global_forward_backward_step == 11


def test_model_runner_loads_initial_checkpoint_and_syncs_state():
    class FakeCheckpointManager:
        def __init__(self):
            self.calls = []
            self.global_step = 13
            self.global_forward_backward_step = 17

        def load_state(self, checkpoint_path, load_optimizer=True):
            self.calls.append((checkpoint_path, load_optimizer))

    runner = object.__new__(ModelRunner)
    runner.train_config = {"load_checkpoint_path": "/tmp/initial-dcp"}
    runner.global_step = 0
    runner.global_forward_backward_step = 0
    runner._checkpoint_mgr = FakeCheckpointManager()

    runner._load_initial_checkpoint()

    assert runner._checkpoint_mgr.calls == [("/tmp/initial-dcp", True)]
    assert runner.global_step == 13
    assert runner.global_forward_backward_step == 17
