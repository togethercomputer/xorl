"""Guard tests for the structural Qwen3.5-MoE native-EP combine."""

import pytest
import torch

from xorl.models.layers.moe.ep_native_combine import (
    NATIVE_EP_COMBINE_QUALIFIED_SIZES,
    QWEN35_NATIVE_EP_COMBINE_SIZES,
    gather_ids_for_ep_combine,
    gather_tokens_for_ep_combine,
    max_rows_for_ep_combine,
    sglang_fused_gate_sigmoid_mul_add,
    validate_native_ep_combine_size,
    validate_qwen35_native_ep_combine_size,
)


pytestmark = [pytest.mark.cpu]


def test_native_combine_qualified_size_registry():
    assert NATIVE_EP_COMBINE_QUALIFIED_SIZES["qwen3_5_moe"] == frozenset({8})
    assert QWEN35_NATIVE_EP_COMBINE_SIZES == frozenset({8})
    validate_native_ep_combine_size("qwen3_5_moe", 8)
    for size in (1, 2, 4, 16):
        with pytest.raises(ValueError, match=r"qualified sizes for family 'qwen3_5_moe': \[8\]"):
            validate_native_ep_combine_size("qwen3_5_moe", size)
    with pytest.raises(ValueError, match="no qualified EP sizes for family 'dsv4'"):
        validate_native_ep_combine_size("dsv4", 8)


def test_qwen35_native_combine_policy(monkeypatch):
    validate_qwen35_native_ep_combine_size(8)
    for size in (1, 2, 4, 16):
        with pytest.raises(ValueError, match=r"qualified sizes for family 'qwen3_5_moe': \[8\]"):
            validate_qwen35_native_ep_combine_size(size)

    blk = _qwen_block(exact=True)
    assert blk._native_ep_combine
    assert blk._exact_batch_invariant_router
    assert blk.router._exact_batch_invariant

    x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    routing = torch.zeros(4, 2, dtype=torch.float32)
    selected = torch.zeros(4, 2, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="trainer EP"):
        blk._ep_combine_native(x, routing, selected)

    with monkeypatch.context() as rows_patch:
        _assert_variable_row_collectives_pad_tokens_and_ids_and_share_max_rows(rows_patch)
    with monkeypatch.context() as gate_patch:
        _assert_serving_fused_gate_forward_preserves_trainer_gradients(gate_patch)
    with monkeypatch.context() as dispatch_patch:
        _assert_native_combine_dispatch_and_actual_operand_policy(dispatch_patch)


def _qwen_block(*, exact: bool = False):
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
    )
    return Qwen3_5MoeSparseMoeBlock(cfg, moe_implementation="eager", layer_idx=0).to(torch.bfloat16)


def _assert_native_routed_partial_enters_through_module_call(monkeypatch):
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


def _assert_variable_row_collectives_pad_tokens_and_ids_and_share_max_rows(monkeypatch):
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

    def fake_id_gather(out, local, group=None):
        del group
        assert torch.equal(local, torch.tensor([[4, 5], [-1, -1], [-1, -1]]))
        out[:3].copy_(local)
        out[3:].copy_(local)

    monkeypatch.setattr(combine.dist, "all_gather_into_tensor", fake_id_gather)
    gathered_ids = gather_ids_for_ep_combine(torch.tensor([[4, 5]]), group=None, padded_rows=3)
    assert gathered_ids.shape == (6, 2)
    assert torch.equal(gathered_ids[1:3], torch.full((2, 2), -1))

    def fake_max(rows, op=None, group=None):
        del op, group
        rows.fill_(8192)

    monkeypatch.setattr(combine.dist, "all_reduce", fake_max)
    assert max_rows_for_ep_combine(6016, torch.device("cpu"), group=None) == 8192


def _assert_serving_fused_gate_forward_preserves_trainer_gradients(monkeypatch):
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


def _assert_native_combine_dispatch_and_actual_operand_policy(monkeypatch):
    """The layer-selected diagnostic hook exposes every exact-combine boundary."""
    with monkeypatch.context() as routed_partial_monkeypatch:
        _assert_native_routed_partial_enters_through_module_call(routed_partial_monkeypatch)

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
        lambda hidden, weight, shared, routed: (
            routed + torch.sigmoid((hidden * weight).sum(dim=-1, keepdim=True)) * shared
        ),
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
