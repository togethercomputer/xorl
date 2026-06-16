from types import SimpleNamespace

import pytest
import torch

from xorl.checkpoint import checkpointer


pytestmark = [pytest.mark.cpu]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)
        self.register_buffer("persistent_buf", torch.ones(3))
        self.register_buffer("scratch_buf", torch.zeros(2), persistent=False)


def test_reference_state_dict_bypasses_dcp_state_dict_and_skips_nonpersistent_buffers(monkeypatch):
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(dp_mode="none"))
    monkeypatch.setattr(
        checkpointer,
        "get_model_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected DCP state_dict call")),
    )

    model_state = checkpointer.ModelState(_TinyModel())
    state_dict = model_state.reference_state_dict()

    assert "linear.weight" in state_dict
    assert "persistent_buf" in state_dict
    assert "scratch_buf" not in state_dict


def test_distributed_checkpointer_load_skips_missing_optimizer_state(tmp_path, monkeypatch):
    captured = {}

    class _FakeReader:
        def __init__(self, path):
            self.path = path

        def read_metadata(self):
            return SimpleNamespace(state_dict_metadata={"model.linear.weight": object()})

    def fake_dcp_load(state_dict, storage_reader, process_group=None):
        captured["state_keys"] = set(state_dict)
        captured["storage_reader"] = storage_reader
        captured["process_group"] = process_group

    monkeypatch.setattr(checkpointer, "FileSystemReader", _FakeReader)
    monkeypatch.setattr(checkpointer.dcp, "load", fake_dcp_load)

    state = {"model": _TinyModel(), "optimizer": object()}
    result = checkpointer.DistributedCheckpointer.load(str(tmp_path), state)

    assert result is state
    assert captured["state_keys"] == {"model"}
    assert isinstance(captured["storage_reader"], _FakeReader)
