"""GPU integration test: QARLMoEExperts composes with the real triton MoE kernel.

Validates that the __dict__-shadow fake-quant feeds the actual triton grouped-GEMM
expert forward (not just CPU eager), producing finite output, a lossy delta vs the
unquantized weights, and STE gradients on the real weight Parameters. Single GPU,
EP disabled (weight fake-quant is rank-local, so EP composition is the same op per
rank); the multi-rank EP path is exercised by the at-scale run.
"""

import pytest
import torch

from xorl.models.layers.moe.experts import MoEExperts
from xorl.qarl.moe_experts import QARLMoEExperts, convert_moe_experts_to_qarl


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _routing(num_tokens, num_experts, top_k, device):
    logits = torch.randn(num_tokens, num_experts, device=device)
    weights, idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    return weights.to(torch.bfloat16), idx.to(torch.int64)


def _make(num_experts=8, hidden=256, inter=256, device="cuda"):
    torch.manual_seed(0)
    e = MoEExperts(
        num_experts=num_experts,
        hidden_dim=hidden,
        intermediate_size=inter,
        hidden_act="silu",
        moe_implementation="triton",
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        e.gate_up_proj.normal_(std=0.02)
        e.down_proj.normal_(std=0.02)
    return e


def test_triton_forward_finite_lossy_and_grad():
    device = "cuda"
    num_tokens, num_experts, top_k = 64, 8, 2
    x = torch.randn(num_tokens, 256, device=device, dtype=torch.bfloat16)
    rw, se = _routing(num_tokens, num_experts, top_k, device)

    ref = _make()
    out_ref = ref(x, rw, se)
    assert torch.isfinite(out_ref).all()

    q = convert_moe_experts_to_qarl(_make(), group_size=16)  # same seed -> same init
    out_q = q(x, rw, se)
    assert isinstance(q, QARLMoEExperts)
    assert out_q.shape == out_ref.shape
    assert torch.isfinite(out_q).all()
    # Fake-quant must perturb the output (lossy) but stay close-ish (good approx).
    rel = (out_q.float() - out_ref.float()).norm() / out_ref.float().norm().clamp_min(1e-6)
    assert rel > 0, "fake-quant produced an identical output (no-op)"
    assert rel < 0.5, f"fake-quant output diverged too far: rel={rel:.3f}"

    out_q.sum().backward()
    assert q.gate_up_proj.grad is not None and torch.isfinite(q.gate_up_proj.grad).all()
    assert q.down_proj.grad is not None and torch.isfinite(q.down_proj.grad).all()


def test_weight_quant_disable_matches_baseline():
    device = "cuda"
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    rw, se = _routing(64, 8, 2, device)
    ref = _make()
    out_ref = ref(x, rw, se)
    q = convert_moe_experts_to_qarl(_make(), quantize_weight=False)
    out_q = q(x, rw, se)
    torch.testing.assert_close(out_q, out_ref, rtol=0, atol=0)
