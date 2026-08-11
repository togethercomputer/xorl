from types import SimpleNamespace

import torch

import xorl.server.runner.model_runner as model_runner_module
from xorl.server.runner.model_runner import ModelRunner


def test_per_sample_k3_reassembles_ulysses_tokens_before_packed_means(monkeypatch):
    group = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, sp_group=group),
    )

    full_k3 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 8.0]])
    full_valid = torch.tensor([[True, True, True, True, True, False]])
    calls = []

    def fake_gather(value, *, gather_dim, padding_dim, unpad_dim_size, group):
        calls.append((value.clone(), gather_dim, padding_dim, unpad_dim_size, group))
        return full_valid.to(torch.uint8) if value.dtype == torch.uint8 else full_k3

    monkeypatch.setattr(model_runner_module, "gather_outputs", fake_gather)

    k3, valid, positions = ModelRunner._gather_per_sample_k3_inputs(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([True, True, True]),
        # Sample 0 crosses the CP boundary; sample 1 begins on rank 1.
        torch.tensor([[0, 1, 2, 3, 0, 1]]),
    )

    assert len(calls) == 2
    assert all(call[1:] == (-1, -1, 6, group) for call in calls)
    assert torch.equal(k3, full_k3.view(-1))
    assert torch.equal(valid, full_valid.view(-1))
    assert torch.equal(positions, torch.tensor([0, 1, 2, 3, 0, 1]))
    assert ModelRunner._compute_per_sample_k3(k3, valid, positions) == [2.5, 6.0]


def test_per_sample_k3_non_cp_preserves_local_inputs(monkeypatch):
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=False),
    )
    monkeypatch.setattr(
        model_runner_module,
        "gather_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected gather")),
    )

    k3 = torch.tensor([[0.0, 1.0]])
    valid = torch.tensor([[True, False]])
    positions = torch.tensor([[0, 1]])
    got = ModelRunner._gather_per_sample_k3_inputs(k3, valid, positions)

    assert torch.equal(got[0], k3.view(-1))
    assert torch.equal(got[1], valid.view(-1))
    assert torch.equal(got[2], positions.view(-1))
