"""Module-level grad-engagement gate for the exact-contract one-round SwiGLU.

K3 lane doctrine: a lever that participates in the training graph ships with
a grad-engagement gate — the module forward/backward must produce non-None,
finite gradients for every parameter and for the input, and the gradients
must match a reference trajectory. The op-level backward test
(tests/ops/test_exact_fp32_silu_and_mul.py) covers the kernel; this gate
covers engagement through Qwen3_5MLP, where the fused one-round path is
selected by the exact-contract policy rather than by a flag.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP


def _exact_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu",
        _activation_native=True,  # the exact contract must override this
        _qwen35_exact_contract=True,
    )


def _one_round_reference_mlp(mlp: Qwen3_5MLP, x: torch.Tensor) -> torch.Tensor:
    gate, up = mlp.gate_up_proj(x).chunk(2, dim=-1)
    activated = (F.silu(gate.float()) * up.float()).to(gate.dtype)
    return mlp.down_proj(activated)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_mlp_grad_engagement_matches_reference_trajectory():
    torch.manual_seed(7)
    mlp = Qwen3_5MLP(_exact_config()).to(device="cuda", dtype=torch.bfloat16)
    assert mlp._use_fused_silu, "exact contract did not select the fused one-round path"
    ref = Qwen3_5MLP(_exact_config()).to(device="cuda", dtype=torch.bfloat16)
    ref.load_state_dict(mlp.state_dict())

    x = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)

    out = mlp(x)
    out_ref = _one_round_reference_mlp(ref, x_ref)

    # Forward bytes: the module's fused path is the one-round program.
    assert torch.equal(out, out_ref), "exact MLP forward != one-round reference bytes"

    out.backward(grad_out)
    out_ref.backward(grad_out)

    # Engagement: every parameter and the input received a finite, nonzero grad.
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"{name} received no grad"
        assert torch.isfinite(param.grad.float()).all(), f"{name} grad not finite"
        assert param.grad.abs().sum() > 0, f"{name} grad identically zero"
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()

    # Reference trajectory: backward is stock numerics; match to tolerance.
    ref_params = dict(ref.named_parameters())
    for name, param in mlp.named_parameters():
        torch.testing.assert_close(param.grad, ref_params[name].grad, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0.02, atol=0.0078125)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_mlp_grads_flow_in_train_mode():
    torch.manual_seed(11)
    mlp = Qwen3_5MLP(_exact_config()).to(device="cuda", dtype=torch.bfloat16).train()
    x = torch.randn(24, 64, device="cuda", dtype=torch.bfloat16)
    loss = mlp(x).float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad.float()).all() for p in mlp.parameters())
