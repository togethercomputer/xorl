import math

import pytest
import torch
import torch.nn as nn

import xorl.trainers.training_utils as training_utils_module
from xorl.data.constants import IGNORE_INDEX
from xorl.trainers.training_utils import (
    clip_gradients,
    count_active_microbatches,
    count_valid_tokens,
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


@pytest.mark.parametrize("max_grad_norm", [0.0, -1.0])
def test_clip_gradients_disabled_when_nonpositive(max_grad_norm):
    model = TinyModule()
    model.weight.grad = torch.ones_like(model.weight)

    grad_norm = clip_gradients(model, max_grad_norm)

    assert grad_norm == 0.0
    assert torch.equal(model.weight.grad, torch.ones_like(model.weight.grad))


def test_get_distsign_grad_scale_factor_returns_mean_vote_scale():
    assert get_distsign_grad_scale_factor(8) == pytest.approx(0.125)


def test_get_distsign_grad_scale_factor_is_noop_without_active_voters():
    assert get_distsign_grad_scale_factor(0) == pytest.approx(1.0)
    assert get_distsign_grad_scale_factor(-1) == pytest.approx(1.0)


def test_count_valid_tokens_uses_metadata_reduce(monkeypatch):
    reduce_calls = []

    def fake_all_reduce_metadata_tensor(tensor, op, group=None, device=None):
        reduce_calls.append((tensor.clone(), op, group, device))
        return torch.tensor(9, dtype=torch.int64, device=device)

    monkeypatch.setattr(training_utils_module, "all_reduce_metadata_tensor", fake_all_reduce_metadata_tensor)

    micro_batches = [
        {"labels": torch.tensor([1, 2, IGNORE_INDEX])},
        {"target_tokens": torch.tensor([3, IGNORE_INDEX, 4])},
    ]

    reduced = count_valid_tokens(micro_batches, group="dp")

    assert reduced.item() == 9
    assert len(reduce_calls) == 1
    tensor, op, group, device = reduce_calls[0]
    assert tensor.device.type == "cpu"
    assert tensor.item() == 4
    assert op == torch.distributed.ReduceOp.SUM
    assert group == "dp"
    assert device == training_utils_module.get_device_type()


def test_count_active_microbatches_batches_reduce_and_returns_voter_total(monkeypatch):
    reduce_calls = []

    def fake_all_reduce_metadata_tensor(tensor, op, group=None, device=None):
        # Simulate a 4-rank DP group where rank counts (per-mb voter totals) are:
        #   mb0: 4 voters, mb1: 0 voters, mb2: 2 voters
        # Local tensor here is the rank-0 contribution; SUM across the group
        # would yield the per-mb voter counts above.
        reduce_calls.append((tensor.clone(), op, group, device))
        return torch.tensor([4, 0, 2], dtype=torch.int64, device=device)

    monkeypatch.setattr(training_utils_module, "all_reduce_metadata_tensor", fake_all_reduce_metadata_tensor)

    micro_batches = [
        {"labels": torch.tensor([1, 2, 3])},
        {"labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX])},
        {"labels": torch.tensor([4, IGNORE_INDEX, 5])},
    ]

    active_microbatches, active_voter_total = count_active_microbatches(micro_batches, group="dp")

    # Exactly one reduce, regardless of microbatch count.
    assert len(reduce_calls) == 1
    tensor, op, group, device = reduce_calls[0]
    assert tensor.device.type == "cpu"
    assert op == torch.distributed.ReduceOp.SUM
    assert group == "dp"
    assert device == "cpu"
    assert active_microbatches == 2  # mbs with at least one voter
    assert active_voter_total == 6  # 4 + 0 + 2


def test_count_active_microbatches_is_empty_input_safe():
    assert count_active_microbatches([]) == (0, 0)


def test_pp_chunked_ce_matches_eager_loss_and_grad(monkeypatch):
    monkeypatch.setenv("XORL_PP_CE_CHUNK_TOKENS", "2")
    labels = torch.tensor([[1, 2, IGNORE_INDEX], [3, 4, 0]])
    pred = torch.randn(2, 3, 5, dtype=torch.bfloat16).requires_grad_()
    ref_pred = pred.detach().clone().requires_grad_()

    chunked_loss = training_utils_module.make_pp_loss_fn("eager")(pred, labels)
    ref_loss = torch.nn.functional.cross_entropy(
        ref_pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )

    chunked_loss.backward()
    ref_loss.backward()

    torch.testing.assert_close(chunked_loss, ref_loss)
    torch.testing.assert_close(pred.grad, ref_pred.grad)


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
