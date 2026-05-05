import math

import pytest
import torch
import torch.nn as nn

import xorl.trainers.training_utils as training_utils_module
from xorl.data.constants import IGNORE_INDEX
from xorl.trainers.training_utils import (
    clip_gradients,
    count_active_microbatches,
    get_distsign_grad_scale_factor,
    get_effective_grad_clip_value,
    sync_sp_gradients,
)


pytestmark = [pytest.mark.cpu]


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32))


def test_get_effective_grad_clip_value_preserves_regular_clipping():
    model = TinyModule()
    model.weight.grad = torch.ones_like(model.weight)

    grad_norm = clip_gradients(
        model,
        get_effective_grad_clip_value(1.0, use_distsignsgd=False),
    )

    expected_scale = 1.0 / math.sqrt(2.0)
    assert grad_norm == pytest.approx(math.sqrt(2.0))
    assert torch.allclose(model.weight.grad, torch.full_like(model.weight.grad, expected_scale))


def test_get_effective_grad_clip_value_skips_clipping_for_distsignsgd():
    model = TinyModule()
    model.weight.grad = torch.ones_like(model.weight)

    grad_norm = clip_gradients(
        model,
        get_effective_grad_clip_value(1.0, use_distsignsgd=True),
    )

    assert grad_norm == pytest.approx(math.sqrt(2.0))
    assert torch.equal(model.weight.grad, torch.ones_like(model.weight.grad))


def test_get_distsign_grad_scale_factor_returns_mean_vote_scale():
    assert get_distsign_grad_scale_factor(8) == pytest.approx(0.125)


def test_get_distsign_grad_scale_factor_is_noop_without_active_voters():
    assert get_distsign_grad_scale_factor(0) == pytest.approx(1.0)
    assert get_distsign_grad_scale_factor(-1) == pytest.approx(1.0)


def test_count_active_microbatches_batches_reduce_and_returns_voter_total(monkeypatch):
    reduce_calls = []

    def fake_all_reduce(tensor, op, group=None):
        # Simulate a 4-rank DP group where rank counts (per-mb voter totals) are:
        #   mb0: 4 voters, mb1: 0 voters, mb2: 2 voters
        # Local tensor here is the rank-0 contribution; SUM across the group
        # would yield the per-mb voter counts above.
        reduce_calls.append((tensor.clone(), op, group))
        tensor.copy_(torch.tensor([4, 0, 2], dtype=torch.int64))

    monkeypatch.setattr(training_utils_module.dist, "all_reduce", fake_all_reduce)

    micro_batches = [
        {"labels": torch.tensor([1, 2, 3])},
        {"labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX])},
        {"labels": torch.tensor([4, IGNORE_INDEX, 5])},
    ]

    active_microbatches, active_voter_total = count_active_microbatches(micro_batches, group="dp")

    # Exactly one reduce, regardless of microbatch count.
    assert len(reduce_calls) == 1
    _, op, group = reduce_calls[0]
    assert op == torch.distributed.ReduceOp.SUM
    assert group == "dp"
    assert active_microbatches == 2  # mbs with at least one voter
    assert active_voter_total == 6  # 4 + 0 + 2


def test_count_active_microbatches_is_empty_input_safe():
    assert count_active_microbatches([]) == (0, 0)


def test_sync_sp_gradients_reduces_every_grad_by_default(monkeypatch):
    reduced = []

    class FakeParam:
        def __init__(self, grad):
            self.grad = grad

    class FakeModel:
        def parameters(self):
            return [
                FakeParam(torch.tensor([1.0, -2.0])),
                FakeParam(torch.tensor([3.0, 4.0])),
            ]

    def fake_all_reduce(tensor, op, group):
        reduced.append((tensor.clone(), op, group))

    monkeypatch.setattr(training_utils_module.dist, "all_reduce", fake_all_reduce)

    sync_sp_gradients(FakeModel(), sp_grad_sync_group="sp-group")

    assert [t.tolist() for t, _, _ in reduced] == [[1.0, -2.0], [3.0, 4.0]]
    assert all(op == torch.distributed.ReduceOp.SUM for _, op, _ in reduced)
    assert all(group == "sp-group" for _, _, group in reduced)


def test_sync_sp_gradients_skips_dtensor_grads_when_requested(monkeypatch):
    reduced = []

    class FakeDTensor:
        def __init__(self, local_tensor):
            self._local_tensor = local_tensor

    class FakeParam:
        def __init__(self, grad):
            self.grad = grad

    class FakeModel:
        def parameters(self):
            return [
                FakeParam(FakeDTensor(torch.tensor([1.0, -2.0]))),
                FakeParam(torch.tensor([3.0, 4.0])),
            ]

    def fake_all_reduce(tensor, op, group):
        reduced.append((tensor.clone() if isinstance(tensor, torch.Tensor) else tensor, op, group))

    monkeypatch.setattr(training_utils_module, "DTensor", FakeDTensor)
    monkeypatch.setattr(training_utils_module.dist, "all_reduce", fake_all_reduce)

    sync_sp_gradients(FakeModel(), sp_grad_sync_group="sp-group", skip_dtensor_grads=True)

    assert len(reduced) == 1
    tensor, op, group = reduced[0]
    assert torch.equal(tensor, torch.tensor([3.0, 4.0]))
    assert op == torch.distributed.ReduceOp.SUM
    assert group == "sp-group"
