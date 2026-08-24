from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from xorl.distributed.canonical_moe import canonical_moe_fold_fp64_v3
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


def test_glm_moe_route_exposes_final_independently_computed_routing_diagnostics() -> None:
    block = _empty_block()
    block.canonical_contract_version = None
    block.config = SimpleNamespace()
    block.train_router = False
    block._routing_replay = None
    block.gate = nn.Linear(4, 3, bias=False)
    block.experts = nn.Identity()
    block.experts.ep_dispatch = "alltoall"
    weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
    ids = torch.tensor([[2, 0]], dtype=torch.int32)
    block._route_tokens_to_experts = MethodType(
        lambda self, router_logits, input_dtype, **kwargs: (weights, ids),
        block,
    )
    captured = {}
    block._diagnostic_capture_component = lambda name, value: captured.setdefault(name, value)

    routed_weights, routed_ids, router_logits = block.route(torch.ones((1, 4), dtype=torch.float32))

    assert captured["moe_router_logits"] is router_logits
    assert captured["moe_topk_ids"] is routed_ids
    assert captured["moe_topk_weights"] is routed_weights


def test_glm_moe_canonical_boundary_captures_input_and_combined_output() -> None:
    block = _empty_block()
    block.canonical_contract_version = "test"
    expected = torch.full((1, 2, 4), 0.5, dtype=torch.bfloat16)
    block._canonical_ep_forward = MethodType(lambda self, *args: expected, block)
    captured = {}
    block._diagnostic_capture_component = lambda name, value: captured.setdefault(name, value)
    hidden = torch.zeros_like(expected)

    output = block.forward_experts_with_shared(
        hidden,
        torch.ones((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.int32),
        torch.arange(2),
    )

    assert output is expected
    assert captured["moe_input"] is hidden
    assert captured["moe_experts_output"] is expected


def test_canonical_routed_boundary_accepts_deepep_local_ids_without_global_ids() -> None:
    block = _empty_block()
    experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(128, 128, ep_rank=7, device="cpu")
    captured = {}

    def forward(self, hidden, routing, selected_experts=None, **kwargs):
        captured.update(
            selected_experts=selected_experts,
            local_ids=kwargs["sglang_ep_native_local_ids"],
        )
        return torch.ones_like(hidden)

    experts.forward = MethodType(forward, experts)
    block.experts = experts
    hidden = torch.zeros((3, 128), dtype=torch.bfloat16)
    routing = torch.arange(24, dtype=torch.float32).reshape(3, 8).div_(32)
    local_ids = torch.tensor(
        [[0, -1, -1, 3, -1, -1, -1, -1]] * 3,
        dtype=torch.int32,
    )

    output = block._canonical_routed_local_partial(hidden, routing, None, local_ids)

    assert torch.equal(output, torch.ones_like(hidden))
    assert captured["selected_experts"] is None
    assert captured["local_ids"] is local_ids


def test_native_deepep_routed_boundary_preserves_empty_receive_batch_and_structural_factor_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xorl.distributed.moe import deepep_native_exact

    block = _empty_block()
    experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(128, 128, ep_rank=7, device="cpu")
    experts.ep_dispatch = "deepep"
    experts.deepep_buffer_size_gb = 1.0
    experts.deepep_num_sms = 24

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the fused expert runner must not receive an empty DeepEP batch")

    experts.forward = fail_if_called
    block.experts = experts
    block.train_router = False
    block.num_experts = 256
    captured = {}
    program_kwargs = {}
    block._diagnostic_capture_component = lambda name, value: captured.setdefault(name, value)

    def fake_program(hidden, routing, selected, **kwargs):
        program_kwargs.update(kwargs)
        empty_hidden = hidden.new_empty((0, hidden.shape[1]))
        empty_weights = routing.new_empty((0, routing.shape[1]))
        empty_ids = selected.new_empty((0, selected.shape[1]), dtype=torch.int32)
        empty_leaf = kwargs["runner"](empty_hidden, empty_weights, empty_ids)
        assert empty_leaf.dtype is torch.bfloat16
        assert empty_leaf.shape == empty_hidden.shape
        assert empty_leaf.is_contiguous()
        return torch.zeros_like(hidden) + empty_leaf.sum()

    monkeypatch.setattr(deepep_native_exact, "native_dispatch_runner_combine", fake_program)
    hidden = torch.zeros((2, 128), dtype=torch.bfloat16)
    routing = torch.full((2, 8), 0.125, dtype=torch.float32)
    selected = torch.arange(16, dtype=torch.int32).reshape(2, 8)
    layer_dependency = torch.ones_like(hidden, requires_grad=True)
    shared_dependency = torch.full_like(hidden, 2.0, requires_grad=True)

    output = block._native_deepep_routed_local(
        hidden,
        routing,
        selected,
        torch.ones(2, dtype=torch.bool),
        ep_rank=7,
        ep_size=16,
        ep_group=object(),
        backward_layer_dependency=layer_dependency,
        backward_shared_dependency=shared_dependency,
    )

    assert torch.equal(output, torch.zeros_like(hidden))
    assert captured["moe_native_routed"] is output
    assert program_kwargs["complete_backward_device_boundary"] is True
    assert program_kwargs["backward_trace_label"] == "glm52_layer_unknown"
    assert program_kwargs["backward_layer_dependency"] is layer_dependency
    assert program_kwargs["backward_shared_dependency"] is shared_dependency
    output.float().sum().backward()
    for name in experts.logical_factor_names:
        gradient = getattr(experts, name).grad
        assert gradient is not None
        assert gradient.dtype is torch.float32
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_native_deepep_empty_source_rank_preserves_combine_autograd_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xorl.distributed.moe import deepep_native_exact

    block = _empty_block()
    block.train_router = False
    block.num_experts = 256
    block.experts = SimpleNamespace(
        num_experts=256,
        deepep_buffer_size_gb=1.0,
        deepep_num_sms=24,
    )
    block._diagnostic_capture_component = lambda *_args: None
    combine_edge = torch.tensor(1.0, dtype=torch.bfloat16, requires_grad=True)

    def fake_program(hidden, _routing, _selected, **_kwargs):
        assert hidden.shape == (0, 128)
        return hidden + combine_edge

    monkeypatch.setattr(deepep_native_exact, "native_dispatch_runner_combine", fake_program)
    hidden = torch.zeros((2, 128), dtype=torch.bfloat16, requires_grad=True)
    output = block._native_deepep_routed_local(
        hidden,
        torch.full((2, 8), 0.125, dtype=torch.float32),
        torch.arange(16, dtype=torch.int32).reshape(2, 8),
        torch.zeros(2, dtype=torch.bool),
        ep_rank=0,
        ep_size=16,
        ep_group=object(),
    )

    output.float().sum().backward()

    assert combine_edge.grad is not None
    assert combine_edge.grad.item() == 0.0


def test_native_shared_boundary_executes_all_tp16_leaves_locally_and_masks_padding() -> None:
    block = _empty_block()
    shared = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta")
    captured = {}

    def forward(self, hidden, *, contributor_ordinal=None, all_contributors=False):
        assert all_contributors is True
        assert contributor_ordinal is None
        captured["hidden"] = hidden
        ordinals = torch.arange(16, dtype=torch.float32) - 7.5
        return ordinals[:, None, None].expand(16, *hidden.shape).to(torch.bfloat16).contiguous()

    shared.forward = MethodType(forward, shared)
    block.shared_experts = shared
    block._diagnostic_capture_component = lambda name, value: captured.setdefault(name, value)
    hidden = torch.zeros((3, 6144), dtype=torch.bfloat16)
    valid = torch.tensor([True, False, True])

    actual = block._native_shared_local_fold(hidden, valid)
    expected = canonical_moe_fold_fp64_v3(captured["moe_native_shared_down"])
    expected[1].zero_()

    assert captured["hidden"] is hidden
    assert captured["moe_native_shared_down"].shape == (16, 3, 6144)
    assert torch.equal(actual, expected)
    assert captured["moe_native_shared_folded"] is actual
