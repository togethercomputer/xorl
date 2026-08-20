import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from xorl.optim import build_optimizer
from xorl.optim.lr_scheduler import build_lr_scheduler
from xorl.optim.multi_optimizer import MultiOptimizer


pytestmark = [pytest.mark.cpu]


def _trace(scheduler, steps: int) -> list[float]:
    lrs: list[float] = []
    for _ in range(steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.optimizer.step()
        scheduler.step()
    return lrs


def _make_optimizer(lr: float = 1.0) -> SGD:
    return SGD(nn.Linear(2, 2).parameters(), lr=lr)


class _Part(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.linear = nn.Linear(dim, dim)


def _build_parts(count: int = 2):
    torch.manual_seed(0)
    return [_Part() for _ in range(count)]


def _assert_multi_part_optimizer_and_scheduler_policy():
    parts = _build_parts(2)
    optimizer = build_optimizer(parts, lr=1e-1, optimizer_type="adamw", fused=False)
    assert isinstance(optimizer, MultiOptimizer)
    assert optimizer._is_multi_optimizer
    assert len(optimizer) == 2
    assert isinstance(optimizer.model, dict)
    assert set(optimizer.model) == set(optimizer.key_names)
    assert {id(parameter) for part in parts for parameter in part.parameters()} <= {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }

    before = [part.linear.weight.detach().clone() for part in parts]
    values = torch.randn(4, 8)
    sum(part.linear(values).sum() for part in parts).backward()
    optimizer.step()
    optimizer.zero_grad()
    for part, previous in zip(parts, before):
        assert not torch.equal(part.linear.weight.detach(), previous)
        assert part.linear.weight.grad is None or torch.all(part.linear.weight.grad == 0)

    scheduler = build_lr_scheduler(
        optimizer,
        train_steps=10,
        lr=1e-1,
        lr_min=0.0,
        lr_decay_style="linear",
    )
    learning_rates_before = [group["lr"] for group in optimizer.param_groups]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    learning_rates_after = [group["lr"] for group in optimizer.param_groups]
    assert all(after < before for before, after in zip(learning_rates_before, learning_rates_after) if before > 0)

    assert not getattr(
        build_optimizer(_build_parts(1), lr=1e-3, optimizer_type="adamw", fused=False), "_is_multi_optimizer", False
    )
    with pytest.raises(ValueError):
        build_optimizer(
            parts,
            lr=1e-3,
            optimizer_type="adamw",
            fused=False,
            param_groups=[{"params": list(parts[0].parameters())}],
        )


class TestSchedule:
    def test_schedule_mode_policy(self):
        sched = build_lr_scheduler(
            _make_optimizer(),
            train_steps=10,
            lr=1.0,
            lr_decay_style="constant",
            lr_warmup_ratio=0.4,
            lr_start=0.0,
        )
        lrs = _trace(sched, 8)
        # 4 warmup steps from lr_start=0 to init_lr=1, then constant at 1
        assert lrs[:4] == pytest.approx([0.0, 0.25, 0.5, 0.75])
        assert lrs[4:] == pytest.approx([1.0] * 4)

        self._assert_linear_schedule_policy()
        self._assert_cosine_schedule_policy()
        self._assert_rejects_invalid_configuration()
        _assert_multi_part_optimizer_and_scheduler_policy()

    def _assert_linear_schedule_policy(self):
        sched = build_lr_scheduler(_make_optimizer(), train_steps=10, lr=1.0, lr_decay_style="linear")
        lrs = _trace(sched, 11)
        # No warmup, default lr_min=1e-7, decay_ratio=1: linear 1.0 → ~0 over 10 steps.
        # Last value at step 10 is min_lr_ratio = 1e-7.
        assert lrs[0] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(1e-7)
        diffs = [lrs[i] - lrs[i + 1] for i in range(len(lrs) - 2)]
        assert all(d == pytest.approx(diffs[0], rel=1e-6) for d in diffs)

        self._assert_warmup_decay_ratio_and_floor()

    def _assert_warmup_decay_ratio_and_floor(self):
        sched = build_lr_scheduler(
            _make_optimizer(),
            train_steps=10,
            lr=1.0,
            lr_decay_style="linear",
            lr_warmup_ratio=0.2,
            lr_min=0.1,
            lr_decay_ratio=0.8,
        )
        lrs = _trace(sched, 12)
        # warmup steps 0-1 (lr_start=0 to 1), decay steps 2-7 (1.0 → 0.1), floor 8+.
        assert lrs[0] == pytest.approx(0.0)
        assert lrs[1] == pytest.approx(0.5)
        assert lrs[2] == pytest.approx(1.0)
        assert lrs[7] == pytest.approx(0.25)
        assert lrs[8] == pytest.approx(0.1)
        assert lrs[11] == pytest.approx(0.1)

    def _assert_cosine_schedule_policy(self):
        sched = build_lr_scheduler(
            _make_optimizer(),
            train_steps=10,
            lr=1.0,
            lr_decay_style="cosine",
            lr_min=0.1,
            lr_decay_ratio=0.8,
        )
        lrs = _trace(sched, 12)
        assert lrs[0] == pytest.approx(1.0)
        assert lrs[8] == pytest.approx(0.1)
        assert lrs[11] == pytest.approx(0.1)
        # Monotonically non-increasing within decay window
        for a, b in zip(lrs[:9], lrs[1:9]):
            assert b <= a + 1e-9

        self._assert_warmup_midpoint_and_endpoint()

    def _assert_warmup_midpoint_and_endpoint(self):
        sched = build_lr_scheduler(
            _make_optimizer(),
            train_steps=10,
            lr=1.0,
            lr_decay_style="cosine",
            lr_warmup_ratio=0.2,
            lr_min=0.0,
        )
        lrs = _trace(sched, 11)
        # 2 warmup steps from 0 → 1, then half-cosine from 1 → 0 over 8 steps.
        assert lrs[0] == pytest.approx(0.0)
        assert lrs[1] == pytest.approx(0.5)
        assert lrs[2] == pytest.approx(1.0)
        # midpoint of decay (step 6): cos(π/2) → factor 0.5
        assert lrs[6] == pytest.approx(0.5, abs=1e-6)
        assert lrs[10] == pytest.approx(0.0, abs=1e-6)

    def _assert_rejects_invalid_configuration(self):
        cases = [
            ({"lr": 0.0}, "lr must be > 0"),
            ({"lr": -1e-3}, "lr must be > 0"),
            ({"lr": 1.0, "lr_warmup_ratio": 1.5}, "lr_warmup_ratio"),
            ({"lr": 1.0, "lr_warmup_ratio": -0.1}, "lr_warmup_ratio"),
            ({"lr": 1.0, "lr_decay_style": "bogus"}, "Unknown learning rate decay style"),
        ]
        for kwargs, error in cases:
            with pytest.raises(ValueError, match=error):
                build_lr_scheduler(_make_optimizer(), train_steps=10, **kwargs)
