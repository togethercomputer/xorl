import torch

from xorl.models.layers import normalization


EPS = 1e-6


def _cpu_v2_forward(x, weight, eps, *, residual=None, zero_centered=False):
    norm_input = x if residual is None else x + residual
    fp32 = norm_input.float()
    inv_rms = torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + eps)
    scale = weight.float() + 1.0 if zero_centered else weight.float()
    out = (fp32 * inv_rms * scale).to(x.dtype)
    return out if residual is None else (out, norm_input)


def _cpu_rms_backward(normed_input, weight, eps, grad_output, grad_residual_out=None):
    with torch.enable_grad():
        x = normed_input.detach().float().requires_grad_(True)
        w = weight.detach().float().requires_grad_(True)
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        out = x * inv_rms * w
        objective = (out * grad_output.float()).sum()
        if grad_residual_out is not None:
            objective = objective + (x * grad_residual_out.float()).sum()
        return torch.autograd.grad(objective, (x, w))


def test_qwen_v2_zero_centered_backward_uses_effective_weight(monkeypatch):
    monkeypatch.setattr(normalization, "rms_norm_v2", _cpu_v2_forward)
    monkeypatch.setattr(normalization, "fused_rms_norm_backward", _cpu_rms_backward)
    torch.manual_seed(11)
    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    grad = torch.randn_like(x)

    out = normalization._FamiliesV2ZeroCenteredRMSNorm.apply(x, weight, EPS)
    out.backward(grad)
    candidate_dx = x.grad.detach().clone()
    candidate_dw = weight.grad.detach().clone()

    x_ref = x.detach().requires_grad_(True)
    weight_ref = weight.detach().requires_grad_(True)
    ref = _cpu_v2_forward(x_ref, weight_ref, EPS, zero_centered=True)
    ref.backward(grad)

    assert torch.allclose(candidate_dx.float(), x_ref.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(candidate_dw, weight_ref.grad, atol=2e-5, rtol=2e-5)


def test_qwen_v2_residual_backward_preserves_both_gradient_paths(monkeypatch):
    monkeypatch.setattr(normalization, "rms_norm_v2", _cpu_v2_forward)
    monkeypatch.setattr(normalization, "fused_rms_norm_backward", _cpu_rms_backward)
    torch.manual_seed(13)
    x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn_like(x)
    grad_residual = torch.randn_like(residual)

    out, residual_out = normalization._FamiliesV2ZeroCenteredResidualRMSNorm.apply(x, residual, weight, EPS)
    torch.autograd.backward((out, residual_out), (grad_out, grad_residual))
    candidate = (x.grad.detach().clone(), residual.grad.detach().clone(), weight.grad.detach().clone())

    x_ref = x.detach().requires_grad_(True)
    residual_ref = residual.detach().requires_grad_(True)
    weight_ref = weight.detach().requires_grad_(True)
    ref_out, ref_residual = _cpu_v2_forward(
        x_ref,
        weight_ref,
        EPS,
        residual=residual_ref,
        zero_centered=True,
    )
    torch.autograd.backward((ref_out, ref_residual), (grad_out, grad_residual))

    assert torch.allclose(candidate[0].float(), x_ref.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.equal(candidate[0], candidate[1])
    assert torch.allclose(candidate[1].float(), residual_ref.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(candidate[2], weight_ref.grad, atol=2e-5, rtol=2e-5)
