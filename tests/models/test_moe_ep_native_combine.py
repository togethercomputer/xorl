"""Guard tests for the structural Qwen3.5-MoE native-EP combine."""

import pytest
import torch

from xorl.distributed.canonical_moe import canonical_moe_fold_fp64_v3
from xorl.models.layers.moe.ep_native_combine import (
    exchange_and_canonical_fold,
    gather_ids_for_ep_combine,
    gather_tokens_for_ep_combine,
    max_rows_for_ep_combine,
    sglang_fused_gate_sigmoid_mul_add,
    validate_native_ep_combine_size,
)


pytestmark = [pytest.mark.cpu]


def test_native_combine_accepts_every_positive_complete_contributor_group():
    for size in (1, 2, 3, 4, 5, 6, 8, 16, 17, 32, 64):
        validate_native_ep_combine_size(size)
    for size in (0, -1):
        with pytest.raises(ValueError, match="positive contributor count"):
            validate_native_ep_combine_size(size)


@pytest.mark.parametrize("ep_size", [1, 3, 5])
def test_native_combine_odd_tail_and_identity_geometries_preserve_bytes_and_gradients(monkeypatch, ep_size):
    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    monkeypatch.setattr(
        _AllToAll,
        "apply",
        staticmethod(lambda _group, partial, _out_splits, _in_splits: partial),
    )
    contributors = (
        torch.arange(ep_size * 6, dtype=torch.float32).reshape(ep_size, 2, 3).sub_(7).div_(8).bfloat16()
    ).requires_grad_(True)

    result = exchange_and_canonical_fold(contributors.reshape(ep_size * 2, 3), group=None, ep_size=ep_size)
    expected = canonical_moe_fold_fp64_v3(contributors)

    assert torch.equal(result.view(torch.uint16), expected.view(torch.uint16))
    result.float().sum().backward()
    assert torch.equal(contributors.grad, torch.ones_like(contributors))


def test_native_combine_32_way_fallback_matches_explicit_adjacent_tree(monkeypatch):
    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    monkeypatch.setattr(
        _AllToAll,
        "apply",
        staticmethod(lambda _group, partial, _out_splits, _in_splits: partial),
    )
    contributors = torch.randn(32, 3, 5, generator=torch.Generator().manual_seed(41)).to(torch.bfloat16)

    result = exchange_and_canonical_fold(contributors.reshape(96, 5), group=None, ep_size=32)

    level = contributors.float()
    while level.shape[0] > 1:
        level = level[0::2] + level[1::2]
    assert torch.equal(result, level[0].bfloat16())


def test_qwen35_exchange_uses_canonical_tree_and_preserves_backward(monkeypatch):
    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    monkeypatch.setattr(
        _AllToAll,
        "apply",
        staticmethod(lambda _group, partial, _out_splits, _in_splits: partial),
    )
    contributors = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0],
        dtype=torch.bfloat16,
    ).view(8, 1, 1)
    leaf = contributors.expand(8, 3, 2).clone()
    leaf[:, -1].zero_()  # deterministic padded row
    leaf.requires_grad_(True)

    result = exchange_and_canonical_fold(leaf.reshape(24, 2), group=None, ep_size=8)

    level = leaf.float()
    while level.shape[0] > 1:
        level = level[0::2] + level[1::2]
    expected = level[0].bfloat16()
    legacy = leaf[-1]
    for ordinal in range(6, -1, -1):
        legacy = legacy + leaf[ordinal]
    assert torch.equal(result, expected)
    assert not torch.equal(result[0], legacy[0])
    assert torch.equal(result[-1], torch.zeros_like(result[-1]))

    result.float().sum().backward()
    assert leaf.grad is not None
    assert torch.equal(leaf.grad, torch.ones_like(leaf.grad))


@pytest.mark.parametrize(
    ("partial", "ep_size", "message"),
    [
        (torch.zeros((16, 2), dtype=torch.float32), 8, "BF16"),
        (torch.zeros((15, 2), dtype=torch.bfloat16), 8, "divisible"),
        (torch.zeros((18, 2), dtype=torch.bfloat16), 0, "positive contributor count"),
    ],
)
def test_qwen35_exchange_fails_closed_before_transport(monkeypatch, partial, ep_size, message):
    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    called = False

    def unexpected_transport(*_args):
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    monkeypatch.setattr(_AllToAll, "apply", staticmethod(unexpected_transport))
    with pytest.raises((TypeError, ValueError), match=message):
        exchange_and_canonical_fold(partial, group=None, ep_size=ep_size)
    assert not called


def _qwen_block(*, exact: bool = False, native_deepep: bool = False):
    from transformers import PretrainedConfig  # noqa: PLC0415

    from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock  # noqa: PLC0415

    cfg = PretrainedConfig(
        hidden_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        hidden_act="silu",
        norm_topk_prob=True,
        shared_expert_intermediate_size=24,
        train_router=False,
        _qwen35_exact_contract=exact,
        _deepep_native_exact=native_deepep,
        _ep_dispatch="deepep" if native_deepep else "alltoall",
    )
    return Qwen3_5MoeSparseMoeBlock(cfg, moe_implementation="eager", layer_idx=0).to(torch.bfloat16)


def test_exact_native_requires_trainer_ep():
    blk = _qwen_block(exact=True)
    x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    routing = torch.zeros(4, 2, dtype=torch.float32)
    selected = torch.zeros(4, 2, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="trainer EP"):
        blk._ep_combine_native(x, routing, selected)


def test_exact_native_combine_is_structural():
    blk = _qwen_block(exact=True)
    assert blk._native_ep_combine
    assert blk._exact_batch_invariant_router
    assert blk.router._exact_batch_invariant


def test_qwen35_native_deepep_selects_shared_transport_and_thin_shared_join():
    blk = _qwen_block(exact=True, native_deepep=True)

    assert blk.deepep_native_exact
    assert blk.experts.deepep_native_exact
    assert blk.experts.ep_dispatch == "deepep"
    assert not blk._native_ep_combine
    assert not blk.supports_routing_replay()


def test_native_routed_partial_enters_through_module_call(monkeypatch):
    """The EP serving-kernel lane must run inside FSDP's pre-forward hooks."""
    blk = _qwen_block()
    hidden = torch.randn(4, 32, dtype=torch.bfloat16)
    routing = torch.randn(4, 2, dtype=torch.float32)
    local_ids = torch.zeros(4, 2, dtype=torch.int32)
    calls = []

    def pre_forward(_module, _args, _kwargs):
        calls.append("pre_forward")

    def routed_partial(got_hidden, got_routing, got_ids):
        calls.append("routed_partial")
        assert got_hidden is hidden
        assert got_routing is routing
        assert got_ids is local_ids
        return torch.zeros_like(hidden)

    blk.experts.register_forward_pre_hook(pre_forward, with_kwargs=True)
    monkeypatch.setattr(blk.experts, "sglang_ep_native_routed_partial", routed_partial)

    result = blk.experts(hidden, routing, sglang_ep_native_local_ids=local_ids)

    assert calls == ["pre_forward", "routed_partial"]
    assert torch.equal(result, torch.zeros_like(hidden))


def test_variable_row_token_gather_unpads_backward(monkeypatch):
    """The live River packer gives EP ranks unequal T; collectives must not."""
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    monkeypatch.setattr(combine.dist, "get_world_size", lambda _group: 2)

    def fake_gather(out, local, group=None):
        del group
        assert local.shape == (3, 2)
        out[:3].copy_(local)
        out[3:].copy_(local + 10)

    def fake_reduce_scatter(out, grad, op=None, group=None):
        del op, group
        out.copy_(grad[:3] + grad[3:])

    monkeypatch.setattr(combine.dist, "all_gather_into_tensor", fake_gather)
    monkeypatch.setattr(combine.dist, "reduce_scatter_tensor", fake_reduce_scatter)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    gathered = gather_tokens_for_ep_combine(x, group=None, padded_rows=3)
    assert gathered.shape == (6, 2)
    assert torch.equal(gathered[2], torch.zeros(2))
    gathered.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_token_gather_orders_backward_before_dependency_producer(monkeypatch):
    """The shared c10d branch must queue before its routed DeepEP sibling."""
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    events = []

    class RoutedBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("routed")
            return grad_output

    monkeypatch.setattr(combine.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(
        combine.dist,
        "all_gather_into_tensor",
        lambda output, local, group=None: output.copy_(local),
    )

    def fake_reduce_scatter(output, grad, op=None, group=None):
        del op, group
        events.append("shared")
        output.copy_(grad)

    monkeypatch.setattr(combine.dist, "reduce_scatter_tensor", fake_reduce_scatter)

    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    routed = RoutedBoundary.apply(x)
    gathered = gather_tokens_for_ep_combine(
        x,
        group=None,
        padded_rows=1,
        backward_dependency=routed,
    )
    (gathered.sum() + routed.sum()).backward()

    assert events == ["shared", "routed"]
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_shared_then_routed_dependency_holds_transformer_residual(monkeypatch):
    """Do not release an earlier layer through the residual bypass."""
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    events = []

    class ResidualBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("residual")
            return grad_output

    class RoutedBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, backward_layer_dependency):
            del backward_layer_dependency
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("routed")
            return grad_output, None

    monkeypatch.setattr(combine.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(
        combine.dist,
        "all_gather_into_tensor",
        lambda output, local, group=None: output.copy_(local),
    )

    def fake_reduce_scatter(output, grad, op=None, group=None):
        del op, group
        events.append("shared")
        output.copy_(grad)

    monkeypatch.setattr(combine.dist, "reduce_scatter_tensor", fake_reduce_scatter)

    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    residual = ResidualBoundary.apply(x)
    routed = RoutedBoundary.apply(x, residual)
    gathered = gather_tokens_for_ep_combine(
        x,
        group=None,
        padded_rows=1,
        backward_dependency=routed,
    )
    (gathered.sum() + routed.sum() + residual.sum()).backward()

    assert events == ["shared", "routed", "residual"]
    assert torch.equal(x.grad, torch.full_like(x, 3.0))


def test_variable_row_id_gather_uses_invalid_padding(monkeypatch):
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    monkeypatch.setattr(combine.dist, "get_world_size", lambda _group: 2)

    def fake_gather(out, local, group=None):
        del group
        assert torch.equal(local, torch.tensor([[4, 5], [-1, -1], [-1, -1]]))
        out[:3].copy_(local)
        out[3:].copy_(local)

    monkeypatch.setattr(combine.dist, "all_gather_into_tensor", fake_gather)
    gathered = gather_ids_for_ep_combine(torch.tensor([[4, 5]]), group=None, padded_rows=3)
    assert gathered.shape == (6, 2)
    assert torch.equal(gathered[1:3], torch.full((2, 2), -1))


def test_max_rows_for_ep_combine(monkeypatch):
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    def fake_max(rows, op=None, group=None):
        del op, group
        rows.fill_(8192)

    monkeypatch.setattr(combine.dist, "all_reduce", fake_max)
    assert max_rows_for_ep_combine(6016, torch.device("cpu"), group=None) == 8192


def test_serving_fused_gate_forward_preserves_trainer_gradients(monkeypatch):
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    def fake_serving_kernel(hidden, weight, shared, final):
        final.copy_(final + torch.sigmoid((hidden * weight).sum(dim=-1, keepdim=True)) * shared)

    monkeypatch.setattr(combine, "_run_sglang_fused_gate_sigmoid_mul_add", fake_serving_kernel)
    hidden = torch.randn(3, 5, requires_grad=True)
    weight = torch.randn(5, requires_grad=True)
    shared = torch.randn(3, 5, requires_grad=True)
    routed = torch.randn(3, 5, requires_grad=True)
    inputs = (hidden, weight, shared, routed)

    actual = sglang_fused_gate_sigmoid_mul_add(*inputs)
    actual.sum().backward()
    actual_grads = tuple(value.grad.detach().clone() for value in inputs)

    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    ref_hidden, ref_weight, ref_shared, ref_routed = reference_inputs
    reference = ref_routed + torch.sigmoid((ref_hidden * ref_weight).sum(dim=-1, keepdim=True)) * ref_shared
    reference.sum().backward()

    torch.testing.assert_close(actual, reference)
    for actual_grad, reference_input in zip(actual_grads, reference_inputs, strict=True):
        torch.testing.assert_close(actual_grad, reference_input.grad)


def test_native_combine_captures_actual_operands(monkeypatch):
    """The layer-selected diagnostic hook exposes every exact-combine boundary."""
    import xorl.distributed.parallel_state as parallel_state  # noqa: PLC0415
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415
    import xorl.ops.batch_invariant_ops as batch_invariant_ops  # noqa: PLC0415

    class DummyParallelState:
        ep_enabled = True
        ep_size = 8
        ep_rank = 2
        ep_group = object()

    class DummyExperts(torch.nn.Module):
        num_experts = 8

        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(1, 2, 32, dtype=torch.bfloat16))

        def forward(self, hidden, routing, *, sglang_ep_native_local_ids):
            del routing, sglang_ep_native_local_ids
            return torch.full_like(hidden, 0.25)

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: DummyParallelState())
    monkeypatch.setattr(combine, "max_rows_for_ep_combine", lambda rows, _device, _group: rows)
    monkeypatch.setattr(combine, "gather_tokens_for_ep_combine", lambda value, _group, _rows: value)
    monkeypatch.setattr(combine, "gather_ids_for_ep_combine", lambda value, _group, _rows: value)
    monkeypatch.setattr(combine, "exchange_and_canonical_fold", lambda value, _group, _size: value)
    monkeypatch.setattr(
        combine,
        "sglang_fused_gate_sigmoid_mul_add",
        lambda hidden, weight, shared, routed: routed
        + torch.sigmoid((hidden * weight).sum(dim=-1, keepdim=True)) * shared,
    )
    monkeypatch.setattr(
        batch_invariant_ops._BatchInvariantTrunkLinearFn,
        "apply",
        lambda value, weight, bias: torch.nn.functional.linear(value, weight, bias),
    )

    blk = _qwen_block()
    blk.experts = DummyExperts()
    captured = {}
    blk._diagnostic_capture_component = lambda name, value: captured.setdefault(name, value.detach().clone())

    hidden = torch.randn(1, 2, 32, dtype=torch.bfloat16)
    routing = torch.randn(2, 2, dtype=torch.float32)
    selected = torch.full((2, 2), 2, dtype=torch.int64)
    output = blk._ep_combine_native(hidden, routing, selected)

    assert set(captured) == {
        "moe_native_gathered_input",
        "moe_native_gathered_routing",
        "moe_native_gathered_ids",
        "moe_native_local_ids",
        "moe_native_routed",
        "moe_native_shared_gate_value",
        "moe_native_shared_gate_up",
        "moe_native_shared_act",
        "moe_native_shared_down",
        "moe_native_local_partial",
        "moe_native_combined",
    }
    torch.testing.assert_close(captured["moe_native_gathered_input"], hidden.reshape(2, 32))
    torch.testing.assert_close(captured["moe_native_gathered_routing"], routing)
    torch.testing.assert_close(captured["moe_native_gathered_ids"], selected)
    torch.testing.assert_close(captured["moe_native_local_ids"], torch.zeros_like(selected, dtype=torch.int32))
    torch.testing.assert_close(captured["moe_native_routed"], torch.full((2, 32), 0.25, dtype=torch.bfloat16))
    torch.testing.assert_close(captured["moe_native_combined"], captured["moe_native_local_partial"])
    torch.testing.assert_close(output.reshape(2, 32), captured["moe_native_combined"])
