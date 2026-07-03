import torch

from xorl.models.layers.normalization import RMSNorm, native_rms_norm, set_rmsnorm_mode


def test_sglang_jit_rmsnorm_mode_cpu_falls_back_to_native_residual():
    set_rmsnorm_mode("sglang_jit")
    try:
        norm = RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))

        hidden_states = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
        residual = torch.tensor([[1.0, 0.5, -0.25, 0.125]], dtype=torch.float32)

        out, residual_out = norm(hidden_states, residual=residual, prenorm=True)

        expected_residual = hidden_states + residual
        expected = native_rms_norm(expected_residual, norm.weight, norm.variance_epsilon)
        assert torch.equal(residual_out, expected_residual)
        assert torch.equal(out, expected)
    finally:
        set_rmsnorm_mode("native")


def test_sglang_jit_rmsnorm_mode_cpu_accepts_packed_shape():
    set_rmsnorm_mode("sglang_jit")
    try:
        norm = RMSNorm(4, eps=1e-6)
        hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
        residual = torch.randn(2, 3, 4, dtype=torch.float32)

        out, residual_out = norm(hidden_states, residual=residual, prenorm=True)

        expected_residual = hidden_states + residual
        expected = native_rms_norm(expected_residual, norm.weight, norm.variance_epsilon)
        assert out.shape == hidden_states.shape
        assert residual_out.shape == hidden_states.shape
        assert torch.equal(residual_out, expected_residual)
        assert torch.allclose(out, expected)
    finally:
        set_rmsnorm_mode("native")


def test_sglang_kernel_rmsnorm_mode_cpu_falls_back_to_native_residual():
    set_rmsnorm_mode("sglang_kernel")
    try:
        norm = RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))

        hidden_states = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
        residual = torch.tensor([[1.0, 0.5, -0.25, 0.125]], dtype=torch.float32)

        out, residual_out = norm(hidden_states, residual=residual, prenorm=True)

        expected_residual = hidden_states + residual
        expected = native_rms_norm(expected_residual, norm.weight, norm.variance_epsilon)
        assert torch.equal(residual_out, expected_residual)
        assert torch.equal(out, expected)
    finally:
        set_rmsnorm_mode("native")
