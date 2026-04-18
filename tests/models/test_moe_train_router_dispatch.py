from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.arguments import ModelArguments
from xorl.models.layers.moe.moe_block import MoEBlock
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
