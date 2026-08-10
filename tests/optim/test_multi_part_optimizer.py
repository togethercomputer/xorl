"""Unit tests for build_optimizer over multiple PP virtual-stage model parts."""

import pytest
import torch
import torch.nn as nn

from xorl.optim import build_optimizer
from xorl.optim.lr_scheduler import build_lr_scheduler
from xorl.optim.multi_optimizer import MultiOptimizer


pytestmark = [pytest.mark.cpu]


class _Part(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.linear = nn.Linear(dim, dim)


def _build_parts(n: int = 2):
    torch.manual_seed(0)
    return [_Part() for _ in range(n)]


def test_multi_part_returns_multi_optimizer():
    parts = _build_parts(2)
    opt = build_optimizer(parts, lr=1e-3, optimizer_type="adamw", fused=False)
    assert isinstance(opt, MultiOptimizer)
    assert opt._is_multi_optimizer
    assert len(opt) == 2
    # per-part model mapping present for DCP FQN resolution
    assert isinstance(opt.model, dict)
    assert set(opt.model.keys()) == set(opt.key_names)


def test_single_part_list_returns_plain_optimizer():
    parts = _build_parts(1)
    opt = build_optimizer(parts, lr=1e-3, optimizer_type="adamw", fused=False)
    assert not getattr(opt, "_is_multi_optimizer", False)


def test_step_updates_every_part():
    parts = _build_parts(2)
    opt = build_optimizer(parts, lr=1e-1, optimizer_type="adamw", fused=False)
    before = [p.linear.weight.detach().clone() for p in parts]
    x = torch.randn(4, 8)
    loss = sum(p.linear(x).sum() for p in parts)
    loss.backward()
    opt.step()
    opt.zero_grad()
    for part, prev in zip(parts, before):
        assert not torch.equal(part.linear.weight.detach(), prev), "part params did not update"
        assert part.linear.weight.grad is None or torch.all(part.linear.weight.grad == 0)


def test_param_groups_cover_all_parts():
    parts = _build_parts(2)
    opt = build_optimizer(parts, lr=1e-3, optimizer_type="adamw", fused=False)
    group_params = {id(p) for g in opt.param_groups for p in g["params"]}
    model_params = {id(p) for part in parts for p in part.parameters()}
    assert model_params <= group_params


def test_lr_scheduler_drives_all_groups():
    parts = _build_parts(2)
    opt = build_optimizer(parts, lr=1e-2, optimizer_type="adamw", fused=False)
    scheduler = build_lr_scheduler(opt, train_steps=10, lr=1e-2, lr_min=0.0, lr_decay_style="linear")
    lrs_before = [g["lr"] for g in opt.param_groups]
    for _ in range(5):
        scheduler.step()
    lrs_after = [g["lr"] for g in opt.param_groups]
    assert len(lrs_before) == len(lrs_after)
    assert all(after < before for before, after in zip(lrs_before, lrs_after) if before > 0)


def test_custom_param_groups_rejected_for_multi_part():
    parts = _build_parts(2)
    with pytest.raises(ValueError):
        build_optimizer(
            parts,
            lr=1e-3,
            optimizer_type="adamw",
            fused=False,
            param_groups=[{"params": list(parts[0].parameters())}],
        )
