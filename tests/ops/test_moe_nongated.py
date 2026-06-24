"""Non-gated MoE expert tests (Nemotron-3-Ultra style: ``down(relu2(up(x)))``).

CPU: eager non-gated experts vs a plain per-expert torch loop reference
(forward + gradients), plus constructor/activation-registry contracts.
GPU: triton/native non-gated backends vs eager (forward + gradients).
"""

import pytest
import torch
import torch.nn.functional as F


def _import_experts():
    """Import MoEExperts or skip (mirrors test_eager_vs_native_moe.py)."""
    try:
        from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415

        return MoEExperts
    except Exception as e:
        pytest.skip(f"Cannot import MoE layers: {e}")


def _make_routing(num_tokens, num_experts, top_k, device, dtype):
    """Random softmax-top-k routing weights and expert assignments."""
    logits = torch.randn(num_tokens, num_experts, device=device)
    weights, selected = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(dtype), selected


def _make_nongated_experts(MoEExperts, ne, hd, inter, backend, device, dtype, seed=42):
    torch.manual_seed(seed)
    experts = MoEExperts(ne, hd, inter, hidden_act="relu2", moe_implementation=backend, gated=False)
    torch.nn.init.xavier_normal_(experts.gate_up_proj.data)
    torch.nn.init.xavier_normal_(experts.down_proj.data)
    return experts.to(device, dtype)


def _eager_moe_forward(experts, hidden_states, routing_weights, selected_experts):
    """Per-expert dispatch loop mirroring ``MoEBlock._eager_forward()``."""
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=experts.num_experts).permute(2, 1, 0)
    for expert_idx in range(experts.num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
        current_hidden_states = experts(current_state, expert_idx=expert_idx) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states


def _reference_forward(hidden_states, routing_weights, selected_experts, up_proj, down_proj):
    """Plain per-expert loop: ``down(relu(x @ up) ** 2)`` weighted over top-k."""
    num_tokens, top_k = selected_experts.shape
    rows = []
    for t in range(num_tokens):
        acc = torch.zeros(down_proj.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        for k in range(top_k):
            e = int(selected_experts[t, k])
            h = torch.square(F.relu(hidden_states[t] @ up_proj[e]))
            acc = acc + routing_weights[t, k] * (h @ down_proj[e])
        rows.append(acc)
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_eager_nongated_matches_reference():
    """Eager non-gated forward + gradients match a plain per-expert torch loop."""
    MoEExperts = _import_experts()
    ne, hd, inter, top_k, num_tokens = 4, 16, 24, 2, 10  # hd is a latent dim, not a model hidden size

    experts = _make_nongated_experts(MoEExperts, ne, hd, inter, "eager", "cpu", torch.float32)
    assert experts.gate_up_proj.shape == (ne, hd, inter)
    assert experts.up_proj.shape == (ne, hd, inter)

    up_ref = experts.gate_up_proj.detach().clone().requires_grad_(True)
    down_ref = experts.down_proj.detach().clone().requires_grad_(True)

    torch.manual_seed(7)
    x = torch.randn(num_tokens, hd, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    routing_weights, selected_experts = _make_routing(num_tokens, ne, top_k, "cpu", torch.float32)

    out = _eager_moe_forward(experts, x, routing_weights, selected_experts)
    ref = _reference_forward(x_ref, routing_weights, selected_experts, up_ref, down_ref)
    torch.testing.assert_close(out, ref)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(experts.gate_up_proj.grad, up_ref.grad)
    torch.testing.assert_close(experts.down_proj.grad, down_ref.grad)


@pytest.mark.cpu
def test_relu2_activation_registry():
    """relu2 is registered, normalized, and equals relu(x)**2 when gate ≡ up."""
    from xorl.ops.moe.activations import (  # noqa: PLC0415
        MOE_ACTIVATIONS,
        UNGATED_HIDDEN_ACTS,
        apply_moe_activation,
        normalize_hidden_act,
    )

    assert normalize_hidden_act("relu2") == "relu2"
    assert "relu2" in MOE_ACTIVATIONS
    assert "relu2" in UNGATED_HIDDEN_ACTS

    x = torch.randn(32)
    torch.testing.assert_close(apply_moe_activation("relu2", x, x), torch.square(F.relu(x)))


@pytest.mark.cpu
def test_nongated_quack_raises():
    """quack backend rejects non-gated experts with a clear error."""
    MoEExperts = _import_experts()
    with pytest.raises(NotImplementedError, match="non-gated"):
        MoEExperts(4, 16, 24, hidden_act="relu2", moe_implementation="quack", gated=False)


@pytest.mark.cpu
def test_nongated_rejects_gated_activation():
    """Non-gated experts require an ungated activation (relu2)."""
    MoEExperts = _import_experts()
    with pytest.raises(ValueError, match="non-gated"):
        MoEExperts(4, 16, 24, hidden_act="silu", moe_implementation="eager", gated=False)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "native"])
def test_nongated_backend_matches_eager(backend):
    """Triton/native non-gated forward + gradients match eager."""
    MoEExperts = _import_experts()
    from xorl.models.layers.moe.backend import MOE_EXPERT_BACKENDS  # noqa: PLC0415

    if backend not in MOE_EXPERT_BACKENDS:
        pytest.skip(f"{backend} backend not available")

    ne, hd, inter, top_k, num_tokens = 8, 64, 128, 2, 32
    device, dtype = "cuda", torch.bfloat16

    eager = _make_nongated_experts(MoEExperts, ne, hd, inter, "eager", device, dtype)
    other = MoEExperts(ne, hd, inter, hidden_act="relu2", moe_implementation=backend, gated=False).to(device, dtype)
    with torch.no_grad():
        other.gate_up_proj.copy_(eager.gate_up_proj)
        other.down_proj.copy_(eager.down_proj)

    torch.manual_seed(7)
    x_eager = torch.randn(num_tokens, hd, device=device, dtype=dtype, requires_grad=True)
    x_other = x_eager.detach().clone().requires_grad_(True)
    routing_weights, selected_experts = _make_routing(num_tokens, ne, top_k, device, dtype)

    out_eager = _eager_moe_forward(eager, x_eager, routing_weights, selected_experts)
    out_other = other(x_other, routing_weights, selected_experts)
    torch.testing.assert_close(out_other, out_eager, atol=0.05, rtol=0.05, msg=f"{backend} forward mismatch")

    grad_out = torch.randn_like(out_eager)
    out_eager.backward(grad_out)
    out_other.backward(grad_out)

    atol, rtol = 0.05, 0.05
    torch.testing.assert_close(x_other.grad, x_eager.grad, atol=atol, rtol=rtol, msg=f"{backend} input grad mismatch")
    torch.testing.assert_close(
        other.gate_up_proj.grad,
        eager.gate_up_proj.grad,
        atol=atol,
        rtol=rtol,
        msg=f"{backend} gate_up_proj grad mismatch",
    )
    torch.testing.assert_close(
        other.down_proj.grad, eager.down_proj.grad, atol=atol, rtol=rtol, msg=f"{backend} down_proj grad mismatch"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
