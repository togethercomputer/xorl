from types import SimpleNamespace

import pytest
import torch

from xorl.distributed.moe import deepep


pytestmark = pytest.mark.cpu


def test_deepep_async_combine_is_synchronous_by_default(monkeypatch):
    captured = {}

    def fake_apply(expert_output, buffer, ctx, async_combine):
        del buffer, ctx
        captured["async_combine"] = async_combine
        return expert_output

    monkeypatch.setattr(deepep, "_ALLOW_UNSAFE_ASYNC_COMBINE", False)
    monkeypatch.setattr(deepep._FusedUnpermuteAndCombine, "apply", staticmethod(fake_apply))

    expert_output = torch.ones(1, 2)
    result = deepep.tokens_post_combine(
        buffer=None,
        expert_output=expert_output,
        ctx=SimpleNamespace(),
        async_combine=True,
    )

    assert result is expert_output
    assert captured["async_combine"] is False


def test_deepep_async_combine_can_be_unsafely_opted_in(monkeypatch):
    captured = {}

    def fake_apply(expert_output, buffer, ctx, async_combine):
        del buffer, ctx
        captured["async_combine"] = async_combine
        return expert_output

    monkeypatch.setattr(deepep, "_ALLOW_UNSAFE_ASYNC_COMBINE", True)
    monkeypatch.setattr(deepep._FusedUnpermuteAndCombine, "apply", staticmethod(fake_apply))

    expert_output = torch.ones(1, 2)
    result = deepep.tokens_post_combine(
        buffer=None,
        expert_output=expert_output,
        ctx=SimpleNamespace(),
        async_combine=True,
    )

    assert result is expert_output
    assert captured["async_combine"] is True
