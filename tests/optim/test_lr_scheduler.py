import pytest
import torch.nn as nn
from torch.optim import SGD

from xorl.optim.lr_scheduler import build_lr_scheduler


pytestmark = [pytest.mark.cpu]


def _trace(scheduler, steps: int) -> list[float]:
    lrs: list[float] = []
    for _ in range(steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


def _make_optimizer(lr: float = 1.0) -> SGD:
    return SGD(nn.Linear(2, 2).parameters(), lr=lr)


class TestConstantSchedule:
    def test_no_warmup_holds_lr(self):
        sched = build_lr_scheduler(_make_optimizer(), train_steps=8, lr=1.0, lr_decay_style="constant")
        assert _trace(sched, 8) == pytest.approx([1.0] * 8)

    def test_linear_warmup_then_constant(self):
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


class TestLinearSchedule:
    def test_default_decays_to_near_zero_over_full_range(self):
        sched = build_lr_scheduler(_make_optimizer(), train_steps=10, lr=1.0, lr_decay_style="linear")
        lrs = _trace(sched, 11)
        # No warmup, default lr_min=1e-7, decay_ratio=1: linear 1.0 → ~0 over 10 steps.
        # Last value at step 10 is min_lr_ratio = 1e-7.
        assert lrs[0] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(1e-7)
        diffs = [lrs[i] - lrs[i + 1] for i in range(len(lrs) - 2)]
        assert all(d == pytest.approx(diffs[0], rel=1e-6) for d in diffs)

    def test_lr_min_floor(self):
        sched = build_lr_scheduler(
            _make_optimizer(),
            train_steps=10,
            lr=1.0,
            lr_decay_style="linear",
            lr_min=0.25,
        )
        lrs = _trace(sched, 12)
        assert lrs[0] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(0.25)
        assert min(lrs) == pytest.approx(0.25)

    def test_warmup_then_decay_to_lr_min_within_decay_ratio(self):
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


class TestCosineSchedule:
    def test_endpoints_and_midpoint(self):
        sched = build_lr_scheduler(
            _make_optimizer(),
            train_steps=10,
            lr=1.0,
            lr_decay_style="cosine",
            lr_min=0.0,
        )
        lrs = _trace(sched, 11)
        assert lrs[0] == pytest.approx(1.0)
        # half-cosine: at progress=0.5, factor = 0.5
        assert lrs[5] == pytest.approx(0.5, abs=1e-6)
        assert lrs[10] == pytest.approx(0.0, abs=1e-6)

    def test_lr_min_floor_after_decay_ratio(self):
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

    def test_warmup_then_cosine(self):
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


class TestValidation:
    def test_rejects_non_positive_lr(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            build_lr_scheduler(_make_optimizer(), train_steps=10, lr=0.0)
        with pytest.raises(ValueError, match="lr must be > 0"):
            build_lr_scheduler(_make_optimizer(), train_steps=10, lr=-1e-3)

    def test_rejects_warmup_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="lr_warmup_ratio"):
            build_lr_scheduler(_make_optimizer(), train_steps=10, lr=1.0, lr_warmup_ratio=1.5)
        with pytest.raises(ValueError, match="lr_warmup_ratio"):
            build_lr_scheduler(_make_optimizer(), train_steps=10, lr=1.0, lr_warmup_ratio=-0.1)

    def test_rejects_unknown_decay_style(self):
        with pytest.raises(ValueError, match="Unknown learning rate decay style"):
            build_lr_scheduler(_make_optimizer(), train_steps=10, lr=1.0, lr_decay_style="bogus")
