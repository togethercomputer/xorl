import torch

from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5_moe import modeling_qwen3_5_moe


def test_qwen3_5_moe_sglang_rmsnorm_keeps_no_residual_native(monkeypatch):
    calls = []

    def fake_native(hidden_states, weight, variance_epsilon):
        calls.append(("native", hidden_states.clone()))
        return hidden_states + 1

    def fake_eager(hidden_states, weight, variance_epsilon):
        calls.append(("eager", hidden_states.clone()))
        return hidden_states + 2

    def fake_native_no_batch_invariant(hidden_states, weight, variance_epsilon):
        calls.append(("native_no_batch_invariant", hidden_states.clone()))
        return hidden_states + 3

    monkeypatch.setattr(modeling_qwen3_5_moe, "native_zero_centered_rms_norm", fake_native)
    monkeypatch.setattr(modeling_qwen3_5_moe, "eager_zero_centered_rms_norm", fake_eager)
    monkeypatch.setattr(
        modeling_qwen3_5_moe,
        "native_zero_centered_rms_norm_without_batch_invariant",
        fake_native_no_batch_invariant,
    )

    set_rmsnorm_mode("sglang")
    try:
        norm = modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm(4)

        x = torch.ones(2, 4)
        out = norm(x)
        assert torch.equal(out, x + 1)
        assert calls[-1][0] == "native"

        residual = torch.full((2, 4), 3.0)
        out, residual_out = norm(x, residual=residual, prenorm=True)
        assert torch.equal(residual_out, x + residual)
        assert torch.equal(out, x + residual + 3)
        assert calls[-1][0] == "native_no_batch_invariant"

        out = norm(x, force_sglang_residual=True)
        assert torch.equal(out, x + 3)
        assert calls[-1][0] == "native_no_batch_invariant"
    finally:
        set_rmsnorm_mode("native")
