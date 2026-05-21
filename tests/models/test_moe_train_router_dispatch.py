from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.arguments import ModelArguments
from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.moe.router import TopKRouter
from xorl.server.server_arguments import ServerArguments


pytestmark = [pytest.mark.cpu]


def test_train_router_true_allowed_with_alltoall():
    moe = MoEBlock(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        intermediate_size=32,
        moe_implementation="eager",
        train_router=True,
    )
    moe.experts.ep_dispatch = "alltoall"
    moe.train()
    for p in moe.parameters():
        nn.init.normal_(p, std=0.01)

    x = torch.randn(1, 4, 16, requires_grad=True)
    out, _ = moe(x)
    out.sum().backward()

    assert moe.gate.weight.grad is not None
    assert torch.isfinite(moe.gate.weight.grad).all()
    assert moe.gate.weight.grad.abs().sum() > 0


def test_train_router_true_rejected_with_deepep():
    moe = MoEBlock(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        intermediate_size=32,
        moe_implementation="eager",
        train_router=True,
    )
    moe.experts.ep_dispatch = "deepep"

    x = torch.randn(1, 4, 16)
    with pytest.raises(AssertionError, match="ep_dispatch='deepep'"):
        moe(x)


def test_model_arguments_default_train_router_false():
    args = ModelArguments(config_path="Qwen/Qwen3-8B")

    assert args.train_router is False


def test_server_arguments_default_train_router_false():
    args = ServerArguments(model_path="Qwen/Qwen3-8B")

    assert args.train_router is False
    assert args.to_config_dict()["model"]["train_router"] is False


def test_from_config_defaults_train_router_false():
    config = SimpleNamespace(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        hidden_act="silu",
        norm_topk_prob=True,
    )

    moe = MoEBlock.from_config(config, moe_implementation="eager")

    assert moe.train_router is False


def test_balanced_synthetic_routing_env(monkeypatch):
    monkeypatch.setenv("XORL_MOE_SYNTHETIC_ROUTING", "balanced")
    router = TopKRouter(num_experts=4, top_k=2)

    logits = torch.randn(8, 4)
    routing_weights, selected_experts = router(logits, torch.bfloat16)

    expected_experts = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
        ]
    )
    assert torch.equal(selected_experts, expected_experts)
    assert routing_weights.dtype == torch.bfloat16
    torch.testing.assert_close(routing_weights.float(), torch.full((8, 2), 0.5))

    counts = torch.bincount(selected_experts.flatten(), minlength=4)
    assert torch.equal(counts, torch.full((4,), 4, dtype=counts.dtype))


def test_balanced_synthetic_routing_replay_regather_uses_uniform_weights(monkeypatch):
    monkeypatch.setenv("XORL_MOE_SYNTHETIC_ROUTING", "balanced")
    moe = MoEBlock(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        intermediate_size=32,
        moe_implementation="eager",
    )

    router_logits = torch.randn(3, 4)
    cached_experts = torch.tensor([[3, 2], [1, 0], [2, 1]])
    selected_experts, routing_weights = moe._regather_routing(router_logits, cached_experts, torch.float32)

    assert torch.equal(selected_experts, cached_experts)
    torch.testing.assert_close(routing_weights, torch.full((3, 2), 0.5))
