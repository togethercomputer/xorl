import pytest
import torch
import torch.nn as nn

from xorl.optim import SignSGD, build_optimizer


pytestmark = [pytest.mark.cpu]


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)


def test_signsgd_applies_sign_updates():
    param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = SignSGD([param], lr=0.1)

    param.grad = torch.tensor([2.0, -0.5, 0.0])
    optimizer.step()

    expected = torch.tensor([0.9, -1.9, 3.0])
    assert torch.allclose(param, expected)


def test_signsgd_applies_decoupled_weight_decay_before_sign_update():
    param = nn.Parameter(torch.tensor([2.0, -2.0]))
    optimizer = SignSGD([param], lr=0.1, weight_decay=0.5)

    param.grad = torch.tensor([3.0, -4.0])
    optimizer.step()

    expected = torch.tensor([1.8, -1.8])
    assert torch.allclose(param, expected)


def test_signsgd_skips_parameters_without_gradients():
    param = nn.Parameter(torch.tensor([1.5, -1.5]))
    optimizer = SignSGD([param], lr=0.1, weight_decay=0.5)

    optimizer.step()

    assert torch.allclose(param, torch.tensor([1.5, -1.5]))


def test_signsgd_rejects_sparse_gradients():
    param = nn.Parameter(torch.ones(4))
    optimizer = SignSGD([param], lr=0.1)
    param.grad = torch.sparse_coo_tensor(indices=[[0, 2]], values=torch.tensor([1.0, -1.0]), size=(4,))

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        optimizer.step()


def test_signsgd_keeps_optimizer_state_empty_across_steps():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = SignSGD([param], lr=0.1)

    for grad in (torch.tensor([1.0]), torch.tensor([-1.0])):
        param.grad = grad
        optimizer.step()
        assert len(optimizer.state) == 0


def test_signsgd_state_dict_round_trips_without_state_tensors():
    source_param = nn.Parameter(torch.tensor([1.0, -1.0]))
    source_optimizer = SignSGD([source_param], lr=0.1, weight_decay=0.25)
    source_param.grad = torch.tensor([1.0, -1.0])
    source_optimizer.step()

    state_dict = source_optimizer.state_dict()

    target_param = nn.Parameter(torch.tensor([0.0, 0.0]))
    target_optimizer = SignSGD([target_param], lr=1.0, weight_decay=0.0)
    target_optimizer.load_state_dict(state_dict)

    assert state_dict["state"] == {}
    assert target_optimizer.state_dict()["state"] == {}
    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert target_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.25)


def test_build_optimizer_supports_signsgd_and_preserves_weight_decay_split():
    model = TinyModule()

    optimizer = build_optimizer(
        model,
        lr=0.1,
        weight_decay=0.01,
        optimizer_type="signsgd",
        no_decay_params=["bias"],
    )

    assert isinstance(optimizer, SignSGD)
    assert len(optimizer.param_groups) == 2

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == pytest.approx(0.01)
    assert no_decay_group["weight_decay"] == pytest.approx(0.0)
    assert decay_group["params"] == [model.linear.weight]
    assert no_decay_group["params"] == [model.linear.bias]
