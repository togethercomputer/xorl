import pytest
import torch

import xorl.distributed.gradient_accumulate_loss as loss_module
from xorl.distributed.gradient_accumulate_loss import gradient_accumulate_loss


pytestmark = [pytest.mark.cpu]


def test_gradient_accumulate_loss_uses_requested_group(monkeypatch):
    reduce_calls = []

    def fake_all_reduce(tensor, op, group=None):
        reduce_calls.append((tensor.clone(), op, group))

    monkeypatch.setattr(loss_module.dist, "all_reduce", fake_all_reduce)

    loss = torch.tensor(2.0, requires_grad=True)
    local_valid_tokens = torch.tensor(3.0)
    global_valid_tokens = torch.tensor(6.0)

    ga_loss, loss_sum = gradient_accumulate_loss(
        loss,
        local_valid_tokens,
        global_valid_tokens,
        group="loss-group",
    )
    ga_loss.backward()

    assert ga_loss.item() == pytest.approx(1.0)
    assert loss_sum.item() == pytest.approx(6.0)
    assert loss.grad.item() == pytest.approx(0.5)
    assert len(reduce_calls) == 1
    _, op, group = reduce_calls[0]
    assert op == torch.distributed.ReduceOp.SUM
    assert group == "loss-group"
