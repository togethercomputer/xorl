"""Correctness comparison between eager and native MoE backends.

Tests forward-pass output agreement, backward-pass gradient agreement,
determinism, and scaling behavior across varying expert counts, hidden dims,
top-k values, and batch/sequence sizes.
"""

import pytest
import torch
import torch.nn as nn

DEVICE = "cuda"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Helpers (lazy-import to avoid torchvision env crash at module level)
# ---------------------------------------------------------------------------

def _import_moe():
    """Import MoE layers; returns (MoEBlock, MoEExperts) or skips."""
    try:
        from xorl.models.layers.moe.moe_block import MoEBlock
        from xorl.models.layers.moe.experts import MoEExperts
        return MoEBlock, MoEExperts
    except Exception as e:
        pytest.skip(f"Cannot import MoE layers: {e}")


def _make_pair(num_experts, hidden_dim, intermediate, top_k, seed=42):
    """Create eager + native MoEBlocks with identical weights."""
    MoEBlock, _ = _import_moe()

    torch.manual_seed(seed)
    eager = MoEBlock(hidden_dim, num_experts, top_k, intermediate, moe_implementation="eager")
    nn.init.xavier_normal_(eager.experts.gate_proj.data)
    nn.init.xavier_normal_(eager.experts.up_proj.data)
    nn.init.xavier_normal_(eager.experts.down_proj.data)
    nn.init.xavier_normal_(eager.gate.weight.data)
    eager = eager.to(DEVICE, DTYPE)

    native = MoEBlock(hidden_dim, num_experts, top_k, intermediate, moe_implementation="native")
    native = native.to(DEVICE, DTYPE)
    with torch.no_grad():
        native.gate.weight.copy_(eager.gate.weight)
        native.experts.gate_proj.copy_(eager.experts.gate_proj)
        native.experts.up_proj.copy_(eager.experts.up_proj)
        native.experts.down_proj.copy_(eager.experts.down_proj)
    return eager, native


# ---------------------------------------------------------------------------
# Test 1: Forward output agreement
# ---------------------------------------------------------------------------

FORWARD_CONFIGS = [
    # (num_experts, hidden_dim, intermediate, top_k, batch, seq)
    (4, 64, 128, 2, 2, 8),
    (8, 128, 256, 2, 4, 16),
    (4, 64, 128, 1, 2, 8),    # top_k=1
    (8, 128, 256, 4, 2, 16),   # top_k=4
    (16, 64, 128, 2, 1, 4),   # 16 experts
    (4, 64, 128, 2, 1, 1),    # minimal seq
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ne,hd,inter,topk,bs,seq", FORWARD_CONFIGS)
def test_forward_agreement(ne, hd, inter, topk, bs, seq):
    """Eager and native forward outputs should match (same weights, same routing)."""
    eager_block, native_block = _make_pair(ne, hd, inter, topk)
    torch.manual_seed(999)
    x = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        eager_out, eager_logits = eager_block(x)
        native_out, native_logits = native_block(x)

    # Router logits must be identical (same gate weights)
    torch.testing.assert_close(eager_logits, native_logits, atol=0, rtol=0)

    # Expert outputs
    max_diff = (eager_out - native_out).abs().max().item()
    mean_diff = (eager_out - native_out).abs().mean().item()
    rel_diff = ((eager_out - native_out).abs() / (eager_out.abs() + 1e-8)).mean().item()

    print(f"\n  E={ne} H={hd} K={topk} B={bs} S={seq}: "
          f"max={max_diff:.6f} mean={mean_diff:.6f} rel={rel_diff:.6f}")

    torch.testing.assert_close(
        native_out, eager_out, atol=0.05, rtol=0.02,
        msg=f"Forward mismatch: max_diff={max_diff:.6f}",
    )


# ---------------------------------------------------------------------------
# Test 2: Backward gradient agreement
# ---------------------------------------------------------------------------

BACKWARD_CONFIGS = [
    (4, 64, 128, 2, 2, 8),
    (8, 128, 256, 2, 4, 16),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ne,hd,inter,topk,bs,seq", BACKWARD_CONFIGS)
def test_backward_agreement(ne, hd, inter, topk, bs, seq):
    """Eager and native backward gradients should match."""
    eager_block, native_block = _make_pair(ne, hd, inter, topk)

    torch.manual_seed(999)
    x_eager = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_native = x_eager.detach().clone().requires_grad_(True)

    eager_out, _ = eager_block(x_eager)
    eager_out.sum().backward()

    native_out, _ = native_block(x_native)
    native_out.sum().backward()

    atol, rtol = 0.05, 0.05

    # Input gradient
    torch.testing.assert_close(
        x_native.grad, x_eager.grad, atol=atol, rtol=rtol,
        msg="Input gradient mismatch",
    )

    # Weight gradients
    for name in ["gate_proj", "up_proj", "down_proj"]:
        eager_grad = getattr(eager_block.experts, name).grad
        native_grad = getattr(native_block.experts, name).grad
        assert eager_grad is not None, f"eager {name} grad is None"
        assert native_grad is not None, f"native {name} grad is None"
        diff = (eager_grad - native_grad).abs().max().item()
        print(f"\n  {name} grad max_diff: {diff:.6f}")
        torch.testing.assert_close(
            native_grad, eager_grad, atol=atol, rtol=rtol,
            msg=f"{name} gradient mismatch (max_diff={diff:.6f})",
        )

    # Gate gradient
    gate_diff = (eager_block.gate.weight.grad - native_block.gate.weight.grad).abs().max().item()
    print(f"\n  gate.weight grad max_diff: {gate_diff:.6f}")
    torch.testing.assert_close(
        native_block.gate.weight.grad, eager_block.gate.weight.grad,
        atol=atol, rtol=rtol,
        msg=f"Gate weight gradient mismatch (max_diff={gate_diff:.6f})",
    )


# ---------------------------------------------------------------------------
# Test 3: Determinism
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["eager", "native"])
def test_determinism(backend):
    """Same input should produce identical output across two calls."""
    MoEBlock, _ = _import_moe()

    torch.manual_seed(42)
    block = MoEBlock(64, 4, 2, 128, moe_implementation=backend)
    nn.init.xavier_normal_(block.experts.gate_proj.data)
    nn.init.xavier_normal_(block.experts.up_proj.data)
    nn.init.xavier_normal_(block.experts.down_proj.data)
    nn.init.xavier_normal_(block.gate.weight.data)
    block = block.to(DEVICE, DTYPE)

    torch.manual_seed(999)
    x = torch.randn(2, 8, 64, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out1, _ = block(x)
        out2, _ = block(x)

    assert torch.equal(out1, out2), f"{backend} is not deterministic"


# ---------------------------------------------------------------------------
# Test 4: Larger scale (closer to real model dims)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_scale_forward():
    """Forward agreement at larger dimensions (E=64, H=512, I=1024, K=8)."""
    ne, hd, inter, topk = 64, 512, 1024, 8
    eager_block, native_block = _make_pair(ne, hd, inter, topk)

    torch.manual_seed(42)
    x = torch.randn(1, 128, hd, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        eager_out, _ = eager_block(x)
        native_out, _ = native_block(x)

    max_diff = (eager_out - native_out).abs().max().item()
    mean_diff = (eager_out - native_out).abs().mean().item()
    rel_diff = ((eager_out - native_out).abs() / (eager_out.abs() + 1e-8)).mean().item()
    print(f"\n  Large scale: max={max_diff:.6f} mean={mean_diff:.6f} rel={rel_diff:.6f}")

    torch.testing.assert_close(
        native_out, eager_out, atol=0.1, rtol=0.05,
        msg=f"Large scale forward mismatch: max_diff={max_diff:.6f}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_scale_backward():
    """Backward agreement at larger dimensions."""
    ne, hd, inter, topk = 64, 512, 1024, 8
    eager_block, native_block = _make_pair(ne, hd, inter, topk)

    torch.manual_seed(42)
    x_eager = torch.randn(1, 128, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_native = x_eager.detach().clone().requires_grad_(True)

    eager_out, _ = eager_block(x_eager)
    eager_out.sum().backward()
    native_out, _ = native_block(x_native)
    native_out.sum().backward()

    atol, rtol = 0.1, 0.1

    input_diff = (x_eager.grad - x_native.grad).abs().max().item()
    print(f"\n  input_grad max_diff: {input_diff:.6f}")
    torch.testing.assert_close(
        x_native.grad, x_eager.grad, atol=atol, rtol=rtol,
        msg=f"Large scale input gradient mismatch: {input_diff:.6f}",
    )

    for name in ["gate_proj", "up_proj", "down_proj"]:
        eg = getattr(eager_block.experts, name).grad
        ng = getattr(native_block.experts, name).grad
        diff = (eg - ng).abs().max().item()
        print(f"  {name} grad max_diff: {diff:.6f}")
        torch.testing.assert_close(
            ng, eg, atol=atol, rtol=rtol,
            msg=f"Large scale {name} gradient mismatch: {diff:.6f}",
        )


# ---------------------------------------------------------------------------
# Test 5: Edge case — all tokens routed to same expert
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_tokens_same_expert():
    """When all tokens route to the same expert, outputs should still match."""
    _, MoEExperts = _import_moe()
    from xorl.models.layers.moe.backend.native import native_expert_forward

    ne, hd, inter = 4, 64, 128
    torch.manual_seed(42)

    experts = MoEExperts(ne, hd, inter, moe_implementation="eager").to(DEVICE, DTYPE)
    nn.init.xavier_normal_(experts.gate_proj.data)
    nn.init.xavier_normal_(experts.up_proj.data)
    nn.init.xavier_normal_(experts.down_proj.data)

    num_tokens, top_k = 16, 2
    x = torch.randn(num_tokens, hd, device=DEVICE, dtype=DTYPE)

    # Force all tokens to expert 0
    routing_weights = torch.ones(num_tokens, top_k, device=DEVICE, dtype=DTYPE) / top_k
    selected_experts = torch.zeros(num_tokens, top_k, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        native_out = native_expert_forward(
            x, routing_weights, selected_experts,
            experts.gate_proj, experts.up_proj, experts.down_proj,
            num_experts=ne,
        )

        # Eager: all tokens → expert 0, weight per slot = 1/topk, top_k slots → sums to 1.0
        eager_out = experts(x, expert_idx=0)

    max_diff = (eager_out - native_out).abs().max().item()
    print(f"\n  Same-expert max_diff: {max_diff:.6f}")
    torch.testing.assert_close(
        native_out, eager_out, atol=0.01, rtol=0.01,
        msg=f"Same-expert output mismatch: {max_diff:.6f}",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
