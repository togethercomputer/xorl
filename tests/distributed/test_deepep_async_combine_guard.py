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


def test_native_dispatch_exposes_transported_receive_metadata(monkeypatch):
    recv_x = torch.ones(2, 4)
    recv_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
    recv_weights = torch.tensor([[0.5, 0.0], [0.75, 0.25]], dtype=torch.float32)
    ctx = SimpleNamespace(recv_topk_idx=recv_ids, recv_topk_weights=recv_weights)
    monkeypatch.setattr(
        deepep,
        "token_pre_dispatch_no_permute",
        lambda **_kwargs: (recv_x, torch.tensor([1, 3]), ctx),
    )

    got_x, got_ids, got_weights, got_ctx = deepep.token_pre_dispatch_native(
        buffer=None,
        hidden_states=torch.zeros(1, 4),
        routing_weights=torch.ones(1, 2),
        selected_experts=torch.zeros(1, 2, dtype=torch.int32),
        num_experts=4,
    )

    assert got_x is recv_x
    assert got_ids is recv_ids
    assert got_weights is recv_weights
    assert got_ctx is ctx


def test_native_combine_keeps_receive_layout_and_forces_safe_sync(monkeypatch):
    captured = {}

    def fake_apply(recv_output, buffer, ctx, async_combine):
        captured.update(buffer=buffer, ctx=ctx, async_combine=async_combine)
        return recv_output

    monkeypatch.delenv("XORL_DEEPEP_UNSAFE_ASYNC_COMBINE", raising=False)
    monkeypatch.setattr(deepep._FusedNativeReceiveCombine, "apply", staticmethod(fake_apply))
    recv_output = torch.ones(2, 4)
    ctx = SimpleNamespace()
    result = deepep.tokens_post_combine_native(
        buffer="buffer",
        recv_output=recv_output,
        ctx=ctx,
        async_combine=True,
    )

    assert result is recv_output
    assert captured == {"buffer": "buffer", "ctx": ctx, "async_combine": False}


def test_native_dispatch_backward_can_require_device_completion(monkeypatch):
    events = []

    class FakeEvent:
        def current_stream_wait(self):
            events.append("event_wait")

    class FakeGrad:
        dtype = torch.bfloat16
        device = torch.device("cuda:0")

        def record_stream(self, _stream):
            events.append("record_stream")

        def to(self, _dtype):
            raise AssertionError("the matching BF16 gradient must not be cast")

    class FakeDeepEP:
        def combine(self, **_kwargs):
            events.append("combine")
            return FakeGrad(), None, FakeEvent()

    class FakeStream:
        def synchronize(self):
            events.append("device_complete")

    monkeypatch.setattr(deepep, "EventHandle", lambda: object())
    monkeypatch.setattr(deepep, "EventOverlap", lambda _handle: object())
    monkeypatch.setattr(deepep.torch.cuda, "current_stream", lambda _device=None: FakeStream())
    ctx = SimpleNamespace(
        buffer=SimpleNamespace(buffer=FakeDeepEP(), combine_config=object()),
        handle=object(),
        input_dtype=torch.bfloat16,
        call_id=17,
        complete_backward_device_boundary=True,
        backward_trace_label="glm52_layer_7",
        backward_layer_dependency_meta=((1,), torch.float32, torch.device("cpu")),
        backward_shared_dependency_meta=((2, 2), torch.bfloat16, torch.device("cpu")),
    )

    result = deepep._FusedDispatchNoPermute.backward(
        ctx,
        torch.ones(2, 4, dtype=torch.bfloat16),
        None,
        None,
    )

    assert events == ["combine", "event_wait", "record_stream", "device_complete"]
    assert isinstance(result[0], FakeGrad)
    assert result[1:7] == (None,) * 6
    assert torch.equal(result[7], torch.zeros(1, dtype=torch.float32))
    assert torch.equal(result[8], torch.zeros((2, 2), dtype=torch.bfloat16))


def test_terminal_dispatch_dependency_holds_shared_and_residual_backward():
    events = []

    class SharedBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("shared")
            return grad_output

    class ResidualBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("residual")
            return grad_output

    class TerminalDispatchBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, layer_dependency, shared_dependency):
            ctx.layer_shape = layer_dependency.shape
            ctx.shared_shape = shared_dependency.shape
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("routed_terminal")
            return (
                grad_output,
                torch.zeros(ctx.layer_shape),
                torch.zeros(ctx.shared_shape),
            )

    value = torch.ones((2, 2), requires_grad=True)
    shared = SharedBoundary.apply(value)
    residual = ResidualBoundary.apply(value)
    routed = TerminalDispatchBoundary.apply(value, residual, shared)
    (routed.sum() + shared.sum() + residual.sum()).backward()

    assert events[0] == "routed_terminal"
    assert set(events[1:]) == {"shared", "residual"}
    assert torch.equal(value.grad, torch.full_like(value, 3.0))
