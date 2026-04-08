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
        from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415
        from xorl.models.layers.moe.moe_block import MoEBlock  # noqa: PLC0415

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
# Test 1: Forward + backward agreement across all configs
# ---------------------------------------------------------------------------

ALL_CONFIGS = [
    # (num_experts, hidden_dim, intermediate, top_k, batch, seq)
    (4, 64, 128, 2, 2, 8),
    (8, 128, 256, 2, 4, 16),
    (4, 64, 128, 1, 2, 8),  # top_k=1
    (8, 128, 256, 4, 2, 16),  # top_k=4
    (16, 64, 128, 2, 1, 4),  # 16 experts
    (4, 64, 128, 2, 1, 1),  # minimal seq
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ne,hd,inter,topk,bs,seq", ALL_CONFIGS)
def test_forward_and_backward_agreement(ne, hd, inter, topk, bs, seq):
    """Eager and native forward outputs and backward gradients should match."""
    eager_block, native_block = _make_pair(ne, hd, inter, topk)

    # --- Forward agreement ---
    torch.manual_seed(999)
    x = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        eager_out, eager_logits = eager_block(x)
        native_out, native_logits = native_block(x)

    torch.testing.assert_close(eager_logits, native_logits, atol=0, rtol=0)
    max_diff = (eager_out - native_out).abs().max().item()
    torch.testing.assert_close(
        native_out,
        eager_out,
        atol=0.05,
        rtol=0.02,
        msg=f"Forward mismatch: max_diff={max_diff:.6f}",
    )

    # --- Backward agreement (for larger configs) ---
    if ne <= 8 and hd >= 64:
        torch.manual_seed(999)
        x_eager = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)
        x_native = x_eager.detach().clone().requires_grad_(True)

        eager_out2, _ = eager_block(x_eager)
        eager_out2.sum().backward()
        native_out2, _ = native_block(x_native)
        native_out2.sum().backward()

        atol, rtol = 0.05, 0.05
        torch.testing.assert_close(x_native.grad, x_eager.grad, atol=atol, rtol=rtol, msg="Input gradient mismatch")
        for name in ["gate_proj", "up_proj", "down_proj"]:
            eager_grad = getattr(eager_block.experts, name).grad
            native_grad = getattr(native_block.experts, name).grad
            assert eager_grad is not None, f"eager {name} grad is None"
            assert native_grad is not None, f"native {name} grad is None"
            torch.testing.assert_close(native_grad, eager_grad, atol=atol, rtol=rtol, msg=f"{name} gradient mismatch")
        torch.testing.assert_close(
            native_block.gate.weight.grad,
            eager_block.gate.weight.grad,
            atol=atol,
            rtol=rtol,
            msg="Gate weight gradient mismatch",
        )


# ---------------------------------------------------------------------------
# Test 2: Determinism + edge case (all tokens same expert)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_determinism_and_edge_cases():
    """Determinism: same input produces identical output. Edge case: all tokens to same expert."""
    MoEBlock, MoEExperts = _import_moe()

    for backend in ["eager", "native"]:
        # --- Determinism ---
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

    # --- All tokens to same expert ---
    from xorl.models.layers.moe.backend.native import native_expert_forward  # noqa: PLC0415

    ne, hd, inter = 4, 64, 128
    torch.manual_seed(42)
    experts = MoEExperts(ne, hd, inter, moe_implementation="eager").to(DEVICE, DTYPE)
    nn.init.xavier_normal_(experts.gate_proj.data)
    nn.init.xavier_normal_(experts.up_proj.data)
    nn.init.xavier_normal_(experts.down_proj.data)

    num_tokens, top_k = 16, 2
    x_edge = torch.randn(num_tokens, hd, device=DEVICE, dtype=DTYPE)
    routing_weights = torch.ones(num_tokens, top_k, device=DEVICE, dtype=DTYPE) / top_k
    selected_experts = torch.zeros(num_tokens, top_k, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        native_out = native_expert_forward(
            x_edge,
            routing_weights,
            selected_experts,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
            num_experts=ne,
        )
        eager_out = experts(x_edge, expert_idx=0)

    torch.testing.assert_close(
        native_out,
        eager_out,
        atol=0.01,
        rtol=0.01,
        msg="Same-expert output mismatch",
    )


# ---------------------------------------------------------------------------
# Test 3: Large scale forward + backward
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_scale():
    """Forward + backward agreement at larger dimensions (E=64, H=512, I=1024, K=8)."""
    ne, hd, inter, topk = 64, 512, 1024, 8
    eager_block, native_block = _make_pair(ne, hd, inter, topk)

    # --- Forward ---
    torch.manual_seed(42)
    x = torch.randn(1, 128, hd, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        eager_out, _ = eager_block(x)
        native_out, _ = native_block(x)
    torch.testing.assert_close(
        native_out,
        eager_out,
        atol=0.1,
        rtol=0.05,
        msg="Large scale forward mismatch",
    )

    # --- Backward ---
    torch.manual_seed(42)
    x_eager = torch.randn(1, 128, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_native = x_eager.detach().clone().requires_grad_(True)

    eager_out2, _ = eager_block(x_eager)
    eager_out2.sum().backward()
    native_out2, _ = native_block(x_native)
    native_out2.sum().backward()

    atol, rtol = 0.1, 0.1
    torch.testing.assert_close(x_native.grad, x_eager.grad, atol=atol, rtol=rtol, msg="Large scale input grad mismatch")

    for name in ["gate_proj", "up_proj", "down_proj"]:
        eg = getattr(eager_block.experts, name).grad
        ng = getattr(native_block.experts, name).grad
        torch.testing.assert_close(ng, eg, atol=atol, rtol=rtol, msg=f"Large scale {name} gradient mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
