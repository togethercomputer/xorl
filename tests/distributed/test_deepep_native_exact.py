from types import SimpleNamespace

import pytest
import torch

from xorl.distributed.moe import deepep_native_exact as native_exact_module
from xorl.distributed.moe.deepep_native_exact import (
    DeepEPNativeExactError,
    NativeDeepEPGeometry,
    _flatten_native_route_metadata,
    adapt_native_runner_metadata,
    canonicalize_native_routing_metadata,
    reduce_expert_rows_to_bf16_leaf,
    validate_native_receive_metadata,
)


def _dispatch_ctx(*, rows=3, hidden=2, ids=None, weights=None, indices=None):
    if ids is None:
        ids = torch.tensor([[0, -1], [1, -1], [0, 1]], dtype=torch.int64)[:rows]
    if weights is None:
        weights = torch.ones_like(ids, dtype=torch.float32)
    if indices is None:
        indices = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    return SimpleNamespace(
        num_recv_tokens=rows,
        hidden_dim=hidden,
        recv_topk_idx=ids,
        recv_topk_weights=weights,
        permuted_indices=indices,
    )


def test_expert_rows_reduce_in_fp32_then_store_one_bf16_leaf():
    # 256.0 + 1.0 - 256.0 distinguishes FP32 local accumulation from a
    # left-associated BF16 fold, which loses the unit contribution.
    expert_output = torch.tensor(
        [[256.0, 1.0], [1.0, 2.0], [-256.0, 4.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    ctx = _dispatch_ctx(
        rows=2,
        hidden=2,
        ids=torch.tensor([[0, 1], [1, -1]], dtype=torch.int64),
        indices=torch.tensor([0, 0, 0], dtype=torch.long),
    )

    leaf = reduce_expert_rows_to_bf16_leaf(expert_output, ctx)

    assert leaf.dtype is torch.bfloat16
    assert torch.equal(leaf, torch.tensor([[1.0, 7.0], [0.0, 0.0]], dtype=torch.bfloat16))
    leaf.float().sum().backward()
    assert torch.equal(expert_output.grad, torch.ones_like(expert_output))


def test_native_receive_metadata_accepts_empty_receive_batch():
    output = torch.empty((0, 4), dtype=torch.bfloat16).contiguous()
    ctx = _dispatch_ctx(
        rows=0,
        hidden=4,
        ids=torch.empty((0, 2), dtype=torch.int64),
        weights=torch.empty((0, 2), dtype=torch.float32),
        indices=torch.empty(0, dtype=torch.long),
    )

    validate_native_receive_metadata(output, ctx, num_local_experts=2)


def test_native_route_metadata_preserves_topk_on_empty_rank():
    routing = torch.empty((0, 8), dtype=torch.float32)
    experts = torch.empty((0, 8), dtype=torch.int64)

    routing_flat, experts_flat = _flatten_native_route_metadata(
        routing,
        experts,
        row_count=0,
    )

    assert routing_flat.shape == (0, 8)
    assert experts_flat.shape == (0, 8)
    assert routing_flat.is_contiguous()
    assert experts_flat.is_contiguous()


def test_native_runner_metadata_is_int32_ids_and_fp32_weights():
    ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.75, 0.0], [0.5, 0.25]], dtype=torch.float32)

    runner_ids, runner_weights = adapt_native_runner_metadata(ids, weights)

    assert runner_ids.dtype is torch.int32
    assert torch.equal(runner_ids.to(torch.int64), ids)
    assert runner_weights.dtype is torch.float32
    assert torch.equal(runner_weights, weights)


def test_native_routing_metadata_preserves_sampler_fp32_coefficients():
    weights = torch.tensor(
        [[0.31519484519958496, 0.09776496887207031]],
        dtype=torch.float32,
    )

    canonical = canonicalize_native_routing_metadata(weights)

    assert canonical.dtype is torch.float32
    assert canonical.is_contiguous()
    assert torch.equal(canonical, weights)
    assert not torch.equal(canonical, weights.to(torch.bfloat16).to(torch.float32))


@pytest.mark.parametrize(
    ("ids", "weights", "match"),
    [
        (torch.tensor([[-1, -1]]), torch.ones((1, 2)), "no local route"),
        (torch.tensor([[2, -1]]), torch.ones((1, 2)), "outside this rank"),
        (torch.tensor([[-2, -1]]), torch.ones((1, 2)), "below -1"),
        (torch.tensor([[0, -1]]), torch.tensor([[float("nan"), 0.0]]), "not finite"),
    ],
)
def test_native_receive_metadata_fails_closed(ids, weights, match):
    output = torch.zeros((1, 4), dtype=torch.bfloat16).contiguous()
    ctx = _dispatch_ctx(rows=1, hidden=4, ids=ids, weights=weights)

    with pytest.raises(DeepEPNativeExactError, match=match):
        validate_native_receive_metadata(output, ctx, num_local_experts=2)


def test_native_receive_metadata_rejects_wider_wire_value():
    output = torch.zeros((1, 4), dtype=torch.float32).contiguous()
    ctx = _dispatch_ctx(
        rows=1,
        hidden=4,
        ids=torch.tensor([[0, -1]], dtype=torch.int64),
        weights=torch.ones((1, 2), dtype=torch.float32),
    )

    with pytest.raises(DeepEPNativeExactError, match="must be BF16"):
        validate_native_receive_metadata(output, ctx, num_local_experts=2)


def test_normal_default_is_deterministic_and_admits_ep16_one_call(monkeypatch):
    monkeypatch.setattr(
        native_exact_module,
        "resolve_native_deepep_geometry",
        lambda _group, hidden: NativeDeepEPGeometry(ep_size=16, ep_rank=0, hidden_size=hidden),
    )
    monkeypatch.setattr(
        native_exact_module,
        "validate_native_receive_metadata",
        lambda *_args, **_kwargs: None,
    )

    calls = []

    def fake_apply(
        recv_output,
        _buffer,
        _dispatch_ctx,
        geometry,
        _backward_layer_dependency,
        _backward_trace_label,
    ):
        calls.append(geometry)
        return recv_output.clone()

    monkeypatch.setattr(
        native_exact_module._DeepEPDeterministicCombineBF16,
        "apply",
        staticmethod(fake_apply),
    )
    recv_output = torch.zeros((1, 4), dtype=torch.bfloat16).contiguous()

    combined = native_exact_module.native_receive_combine_and_fold(
        recv_output,
        buffer=object(),
        dispatch_ctx=object(),
        ep_group=object(),
        num_local_experts=1,
    )

    assert torch.equal(combined, recv_output)
    assert len(calls) == 1
    assert calls[0].ep_size == 16


def test_expert_order_adapter_uses_only_deterministic_receive(monkeypatch):
    leaf = torch.zeros((1, 4), dtype=torch.bfloat16)
    calls = []
    monkeypatch.setattr(
        native_exact_module,
        "reduce_expert_rows_to_bf16_leaf",
        lambda _output, _dispatch_ctx: leaf,
    )

    def fake_receive(recv_output, **kwargs):
        calls.append(kwargs)
        return recv_output

    monkeypatch.setattr(
        native_exact_module,
        "native_receive_combine_and_fold",
        fake_receive,
    )
    common = dict(
        buffer=object(),
        dispatch_ctx=object(),
        ep_group=object(),
        num_local_experts=1,
    )

    assert native_exact_module.native_expert_combine_and_fold(leaf, **common) is leaf
    assert len(calls) == 1


def test_normal_deterministic_rejects_unsupported_ep_before_kernel(monkeypatch):
    monkeypatch.setattr(
        native_exact_module,
        "resolve_native_deepep_geometry",
        lambda _group, hidden: NativeDeepEPGeometry(ep_size=24, ep_rank=0, hidden_size=hidden),
    )
    monkeypatch.setattr(
        native_exact_module,
        "validate_native_receive_metadata",
        lambda *_args, **_kwargs: None,
    )
    called = False

    def unexpected_apply(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("deterministic kernel must not run")

    monkeypatch.setattr(
        native_exact_module._DeepEPDeterministicCombineBF16,
        "apply",
        staticmethod(unexpected_apply),
    )

    with pytest.raises(DeepEPNativeExactError, match=r"EP sizes.*EP24"):
        native_exact_module.native_receive_combine_and_fold(
            torch.zeros((1, 4), dtype=torch.bfloat16).contiguous(),
            buffer=object(),
            dispatch_ctx=object(),
            ep_group=object(),
            num_local_experts=1,
        )

    assert not called


def test_normal_deterministic_rejects_unsupported_ep_before_dispatch(monkeypatch):
    from xorl.distributed.moe import deepep as deepep_module

    monkeypatch.setattr(
        native_exact_module,
        "resolve_native_deepep_geometry",
        lambda _group, hidden: NativeDeepEPGeometry(ep_size=24, ep_rank=0, hidden_size=hidden),
    )
    called = False

    def unexpected_buffer(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("DeepEP buffer must not be acquired")

    monkeypatch.setattr(deepep_module, "get_default_buffer", unexpected_buffer)

    with pytest.raises(DeepEPNativeExactError, match=r"EP sizes.*EP24"):
        native_exact_module.native_dispatch_runner_combine(
            torch.zeros((1, 4), dtype=torch.bfloat16),
            torch.ones((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.int64),
            ep_group=object(),
            num_experts=24,
            num_local_experts=1,
            buffer_size_gb=1.0,
            num_sms=1,
            runner=lambda hidden, _weights, _ids: hidden,
        )

    assert not called
