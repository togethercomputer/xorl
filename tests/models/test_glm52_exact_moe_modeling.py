from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.modeling_glm5 import Glm5MoEBlock


def _empty_block() -> Glm5MoEBlock:
    block = Glm5MoEBlock.__new__(Glm5MoEBlock)
    nn.Module.__init__(block)
    block.routed_scaling_factor = 2.5
    return block


def test_canonical_routed_boundary_passes_both_global_and_owner_local_ids() -> None:
    block = _empty_block()
    experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(128, 128, ep_rank=7, device="cpu")
    captured = {}

    def forward(self, hidden, routing, selected_experts=None, **kwargs):
        captured.update(
            hidden=hidden,
            routing=routing,
            selected_experts=selected_experts,
            local_ids=kwargs["sglang_ep_native_local_ids"],
            routed_scaling_factor=kwargs["routed_scaling_factor"],
        )
        return torch.ones_like(hidden)

    experts.forward = MethodType(forward, experts)
    block.experts = experts
    hidden = torch.zeros((3, 128), dtype=torch.bfloat16)
    routing = torch.arange(24, dtype=torch.float32).reshape(3, 8).div_(32)
    global_ids = torch.arange(24, dtype=torch.int64).reshape(3, 8).add_(112)
    local_ids = torch.arange(24, dtype=torch.int32).reshape(3, 8).remainder_(16)

    output = block._canonical_routed_local_partial(hidden, routing, global_ids, local_ids)

    assert torch.equal(output, torch.ones_like(hidden))
    assert captured["hidden"] is hidden
    assert captured["routing"] is routing
    assert captured["selected_experts"] is global_ids
    assert captured["local_ids"] is local_ids
    assert captured["routed_scaling_factor"] == 2.5


def test_canonical_shared_boundary_calls_the_exact_root_with_contributor_ordinal() -> None:
    block = _empty_block()
    shared = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta")
    captured = {}

    def forward(self, hidden, *, contributor_ordinal):
        captured.update(hidden=hidden, contributor_ordinal=contributor_ordinal)
        return torch.full_like(hidden, 0.5)

    shared.forward = MethodType(forward, shared)
    block.shared_experts = shared
    hidden = torch.zeros((3, 6144), dtype=torch.bfloat16)

    output = block._canonical_shared_local_partial(
        hidden,
        contributor_ordinal=7,
        contributor_count=16,
    )

    assert torch.equal(output, torch.full_like(hidden, 0.5))
    assert captured["hidden"] is hidden
    assert captured["contributor_ordinal"] == 7
