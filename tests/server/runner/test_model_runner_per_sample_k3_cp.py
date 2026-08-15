from types import SimpleNamespace

import torch

import xorl.server.runner.model_runner as model_runner_module
from xorl.server.runner.model_runner import ModelRunner


def test_per_sample_k3_reassembles_ulysses_tokens_before_packed_means(monkeypatch):
    group = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, sp_group=group, ringattn_size=1),
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


def test_per_sample_k3_restores_ring_zigzag_before_sample_association(monkeypatch):
    group = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, sp_group=group, cp_size=2, ringattn_size=2),
    )

    # Original token order contains two four-token samples. Ring-2 zigzag order
    # is [doc0:0,3, doc1:0,3, doc0:1,2, doc1:1,2]. Keeping the original
    # position_ids alongside these gathered values would mix the two samples.
    zigzag_k3 = torch.tensor([[1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0]])
    zigzag_valid = torch.ones_like(zigzag_k3, dtype=torch.uint8)

    def fake_gather(value, *, gather_dim, padding_dim, unpad_dim_size, group):
        assert (gather_dim, padding_dim, unpad_dim_size) == (-1, -1, 8)
        return zigzag_valid if value.dtype == torch.uint8 else zigzag_k3

    monkeypatch.setattr(model_runner_module, "gather_outputs", fake_gather)

    k3, valid, positions = ModelRunner._gather_per_sample_k3_inputs(
        torch.tensor([1.0, 1.0, 10.0, 10.0]),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
    )

    assert torch.equal(k3, torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]))
    assert valid.all()
    assert ModelRunner._compute_per_sample_k3(k3, valid, positions) == [1.0, 10.0]


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


def _alignment_physical_rows():
    logical = torch.tensor([[0, 1, 2, -1, -1, 3, 4, -1]])
    live = logical >= 0
    return logical, live


def test_per_token_outputs_restore_post_alignment_rows_before_unpacking(monkeypatch):
    group = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, cp_size=2, sp_group=group),
    )
    logical, live = _alignment_physical_rows()
    physical_logprobs = torch.tensor([[10.0, 11.0, 12.0, 99.0, 99.0, 13.0, 14.0, 99.0]])
    gathered = iter((logical, live.to(torch.uint8), physical_logprobs))

    def fake_gather(_value, *, gather_dim, group, **kwargs):
        assert gather_dim == -1
        assert group is not None
        assert not kwargs
        return next(gathered)

    monkeypatch.setattr(model_runner_module, "gather_outputs", fake_gather)
    runner = object.__new__(ModelRunner)
    accumulators = {"position_ids": [], "logprobs": [], "losses": [], "token_diagnostics": []}
    runner._collect_per_token_outputs(
        {"logprobs": torch.zeros(1, 4)},
        {
            "_original_position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
            "_cp_logical_row_indices": torch.tensor([[0, 1, 2, -1]]),
            "_cp_live_mask": torch.tensor([[True, True, True, False]]),
        },
        accumulators,
    )

    assert torch.equal(accumulators["logprobs"][0], torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]]))
    assert torch.equal(accumulators["position_ids"][0], torch.tensor([[0, 1, 2, 0, 1]]))


def test_per_sample_k3_restores_alignment_rows_before_sample_means(monkeypatch):
    group = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, cp_size=2, sp_group=group),
    )
    logical, live = _alignment_physical_rows()
    physical_k3 = torch.tensor([[1.0, 2.0, 3.0, 99.0, 99.0, 4.0, 6.0, 99.0]])
    gathered = iter((logical, live.to(torch.uint8), physical_k3, live.to(torch.uint8)))

    def fake_gather(_value, *, gather_dim, group, **kwargs):
        assert gather_dim == -1
        assert group is not None
        assert not kwargs
        return next(gathered)

    monkeypatch.setattr(model_runner_module, "gather_outputs", fake_gather)
    k3, valid, positions = ModelRunner._gather_per_sample_k3_inputs(
        torch.zeros(4),
        torch.zeros(4, dtype=torch.bool),
        torch.tensor([[0, 1, 2, 0, 1]]),
        torch.tensor([[0, 1, 2, -1]]),
        torch.tensor([[True, True, True, False]]),
    )

    assert torch.equal(k3, torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0]))
    assert valid.all()
    assert torch.equal(positions, torch.tensor([0, 1, 2, 0, 1]))
    assert ModelRunner._compute_per_sample_k3(k3, valid, positions) == [2.0, 5.0]


def test_teacher_cache_restores_alignment_rows_before_sample_split(monkeypatch):
    group = object()
    logical, live = _alignment_physical_rows()
    physical_hidden = torch.tensor([[[1.0], [2.0], [3.0], [99.0], [99.0], [4.0], [5.0], [99.0]]])
    physical_labels = torch.tensor([[11, 12, 13, -100, -100, 14, 15, -100]])
    gathered = iter((logical, live.to(torch.uint8), physical_hidden, physical_labels))

    def fake_gather(_value, *, gather_dim, group, **kwargs):
        assert group is not None
        assert not kwargs
        return next(gathered)

    monkeypatch.setattr(model_runner_module, "gather_outputs", fake_gather)
    runner = object.__new__(ModelRunner)
    restored_hidden, restored_batch = runner._gather_teacher_cache_sequences(
        torch.zeros(1, 4, 1),
        {
            "labels": torch.zeros(1, 4, dtype=torch.long),
            "_original_position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
            "_cp_logical_row_indices": torch.tensor([[0, 1, 2, -1]]),
            "_cp_live_mask": torch.tensor([[True, True, True, False]]),
        },
        SimpleNamespace(cp_enabled=True, cp_size=2, sp_group=group),
    )

    assert torch.equal(restored_hidden, torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]]))
    assert torch.equal(restored_batch["labels"], torch.tensor([[11, 12, 13, 14, 15]]))


def test_teacher_cache_restores_labels_when_local_and_source_lengths_match(monkeypatch):
    group = object()
    logical = torch.cat(
        (
            torch.arange(32),
            torch.full((32,), -1),
            torch.arange(32, 64),
            torch.full((32,), -1),
        )
    ).view(1, -1)
    live = logical >= 0
    physical_hidden = torch.full((1, 128, 1), 99.0)
    physical_hidden[:, :32, 0] = torch.arange(1, 33, dtype=torch.float32)
    physical_hidden[:, 64:96, 0] = torch.arange(33, 65, dtype=torch.float32)
    physical_labels = torch.full((1, 128), -100, dtype=torch.long)
    physical_labels[:, :32] = torch.arange(101, 133)
    physical_labels[:, 64:96] = torch.arange(133, 165)
    gathered = iter((logical, live.to(torch.uint8), physical_hidden, physical_labels))
    calls = []

    def fake_gather(_value, *, gather_dim, group, **kwargs):
        assert group is not None
        assert not kwargs
        calls.append(gather_dim)
        return next(gathered)

    monkeypatch.setattr(model_runner_module, "gather_outputs", fake_gather)
    runner = object.__new__(ModelRunner)
    restored_hidden, restored_batch = runner._gather_teacher_cache_sequences(
        torch.zeros(1, 64, 1),
        {
            # This local length equals the restored source length, but these
            # rank-local labels must still be gathered through the source map.
            "labels": torch.full((1, 64), -7, dtype=torch.long),
            "_original_position_ids": torch.arange(64).view(1, -1),
            "_cp_logical_row_indices": logical[:, :64],
            "_cp_live_mask": live[:, :64],
        },
        SimpleNamespace(cp_enabled=True, cp_size=2, sp_group=group),
    )

    assert calls == [-1, -1, 1, -1]
    assert torch.equal(restored_hidden[..., 0], torch.arange(1, 65, dtype=torch.float32).view(1, -1))
    assert torch.equal(restored_batch["labels"], torch.arange(101, 165).view(1, -1))
