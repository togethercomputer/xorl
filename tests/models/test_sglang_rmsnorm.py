import torch

from xorl.models.layers.normalization import RMSNorm, get_rmsnorm_mode, set_rmsnorm_mode


def test_sglang_rmsnorm_force_residual_uses_fp32_weight_multiply():
    hidden = torch.tensor([[1.5, -2.0, 0.25, 4.0]], dtype=torch.bfloat16)
    norm = RMSNorm(hidden_size=4, eps=1e-6, mode="sglang")
    norm.weight.data = torch.tensor([1.0, 1.25, -0.5, 2.0], dtype=torch.float32)

    out = norm(hidden, force_sglang_residual=True)

    hidden_f = hidden.float()
    expected = hidden_f * torch.rsqrt(hidden_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = (expected * norm.weight.float()).to(hidden.dtype)
    torch.testing.assert_close(out, expected)


def test_global_sglang_rmsnorm_mode_is_accepted():
    previous = get_rmsnorm_mode()
    try:
        set_rmsnorm_mode("sglang")
        assert get_rmsnorm_mode() == "sglang"
        assert RMSNorm(hidden_size=2).mode == "sglang"
    finally:
        set_rmsnorm_mode(previous)
