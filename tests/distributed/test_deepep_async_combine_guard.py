from types import SimpleNamespace

import torch

from xorl.distributed.moe import deepep


def test_deepep_async_combine_is_synchronous_by_default(monkeypatch):
    captured = {}

    def fake_apply(expert_output, buffer, ctx, async_combine):
        del buffer, ctx
        captured["async_combine"] = async_combine
        return expert_output

    monkeypatch.delenv("XORL_DEEPEP_UNSAFE_ASYNC_COMBINE", raising=False)
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

    monkeypatch.setenv("XORL_DEEPEP_UNSAFE_ASYNC_COMBINE", "1")
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
