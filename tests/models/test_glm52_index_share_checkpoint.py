from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from xorl.models.transformers.glm5.index_share import (
    CanonicalLogicalIndices,
    IndexShareContextManager,
    IndexShareLifecycle,
    IndexShareMode,
)
from xorl.models.transformers.glm5.layer_plan import Glm52LayerPlan


def _producer_shared_plan() -> Glm52LayerPlan:
    return Glm52LayerPlan.from_config(
        SimpleNamespace(
            num_hidden_layers=2,
            indexer_types=["full", "shared"],
            index_topk_freq=2,
            index_skip_topk_offset=0,
            index_topk_pattern=[1, 0],
            mlp_layer_types=["dense", "sparse"],
        )
    )


@pytest.mark.cpu
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpointed_producer_shared_backward_reuses_detached_payload(use_reentrant):
    plan = _producer_shared_plan()
    manager = IndexShareContextManager(plan, (0, 2))
    context = manager.begin(mode=IndexShareMode.TRAINING_WITH_BACKWARD)
    input_tensor = torch.randn(4, requires_grad=True)
    scale = torch.nn.Parameter(torch.tensor(0.75))
    calls = {"producer_layer": 0, "produce_payload": 0, "shared_layer": 0}

    def produce_payload() -> CanonicalLogicalIndices:
        calls["produce_payload"] += 1
        return CanonicalLogicalIndices(torch.tensor([[0, 1]], dtype=torch.int64))

    def producer_layer(hidden_states: torch.Tensor) -> torch.Tensor:
        calls["producer_layer"] += 1
        payload = context.get_or_publish(
            producer_layer_index=0,
            layer_plan=plan,
            produce_payload=produce_payload,
        )
        assert not payload.values.requires_grad
        return hidden_states.sin() * scale + payload.values.sum().to(hidden_states) * 0.0

    def shared_layer(hidden_states: torch.Tensor) -> torch.Tensor:
        calls["shared_layer"] += 1
        payload = context.require(producer_layer_index=0, layer_plan=plan)
        return hidden_states.cos() + payload.values.sum().to(hidden_states) * 0.0

    try:
        output = checkpoint(producer_layer, input_tensor, use_reentrant=use_reentrant)
        output = checkpoint(shared_layer, output, use_reentrant=use_reentrant)
        output.sum().backward()
    finally:
        manager.end(context)

    assert calls["producer_layer"] == 2
    assert calls["shared_layer"] == 2
    assert calls["produce_payload"] == 1
    assert input_tensor.grad is not None
    assert scale.grad is not None
    assert manager.active is None
    assert context.lifecycle is IndexShareLifecycle.CLOSED


@pytest.mark.cpu
def test_mode_owned_success_and_failure_cleanup_is_idempotent():
    plan = _producer_shared_plan()
    manager = IndexShareContextManager(plan, (0, 2))

    forward_only = manager.begin(mode=IndexShareMode.FORWARD_ONLY)
    forward_only.get_or_publish(
        producer_layer_index=0,
        layer_plan=plan,
        produce_payload=lambda: torch.tensor([[0, 1]], dtype=torch.int32),
    )
    manager.finish_forward(forward_only, succeeded=True)
    manager.end(forward_only)
    assert manager.active is None

    forward_failure = manager.begin(mode=IndexShareMode.TRAINING_WITH_BACKWARD)
    manager.finish_forward(forward_failure, succeeded=False)
    manager.end(forward_failure)
    assert manager.active is None

    backward_failure = manager.begin(mode=IndexShareMode.TRAINING_WITH_BACKWARD)
    manager.finish_forward(backward_failure, succeeded=True)
    assert manager.active is backward_failure
    with pytest.raises(RuntimeError, match="backward failed"):
        try:
            raise RuntimeError("backward failed")
        finally:
            manager.end(backward_failure)
    manager.end(backward_failure)
    assert manager.active is None


@pytest.mark.cpu
def test_overlapping_checkpointed_microbatches_keep_invocation_local_payloads_until_schedule_end():
    """Model a 1F1B stage issuing F0/F1 before the matching B1/B0 recomputes."""

    plan = _producer_shared_plan()
    manager = IndexShareContextManager(plan, (0, 2))
    scale = torch.nn.Parameter(torch.tensor(0.5))
    payload_calls = [0, 0]
    contexts = []
    outputs = []

    for microbatch in range(2):
        context = manager.begin(mode=IndexShareMode.TRAINING_WITH_BACKWARD)
        contexts.append(context)
        input_tensor = torch.tensor([float(microbatch + 1)], requires_grad=True)

        def layer(hidden_states, *, _context=context, _microbatch=microbatch):
            payload = _context.get_or_publish(
                producer_layer_index=0,
                layer_plan=plan,
                produce_payload=lambda: (
                    payload_calls.__setitem__(_microbatch, payload_calls[_microbatch] + 1)
                    or torch.tensor([[_microbatch]], dtype=torch.int64)
                ),
            )
            return hidden_states * scale + payload.values.sum().to(hidden_states) * 0.0

        outputs.append(checkpoint(layer, input_tensor, use_reentrant=False))
        manager.finish_forward(context, succeeded=True)

    assert manager.active_contexts == tuple(contexts)
    outputs[1].sum().backward(retain_graph=True)
    outputs[0].sum().backward()
    assert payload_calls == [1, 1]
    assert manager.active_contexts == tuple(contexts)

    manager.end_all()
    assert manager.active_contexts == ()
    assert all(context.lifecycle is IndexShareLifecycle.CLOSED for context in contexts)
