"""Dense and MoE Qwen3.5 RMSNorm contract owners.

The two model families carry parallel zero-centered RMSNorm implementations.
Their shared dispatch and site-assignment policy is exercised once here, with
both implementations passed through the same cases. GPU arithmetic is covered
by one representative model lifecycle because both classes call the same
normalization kernels.
"""

import pytest
import torch

from xorl.models.layers import normalization
from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5 import modeling_qwen3_5
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
)
from xorl.models.transformers.qwen3_5_moe import modeling_qwen3_5_moe
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeModel,
    Qwen3_5MoeRMSNorm,
)
from xorl.ops.batch_invariant_ops import rms_norm_batch_invariant, set_batch_invariant_mode


HIDDEN = 2048
N_TOKENS = 512
EPS = 1e-6

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _tiny_dense_config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        layer_types=["full_attention", "full_attention"],
        _attn_implementation="eager",
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return Qwen3_5Config(**kwargs)


def _tiny_moe_config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=0,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        layer_types=["full_attention", "full_attention"],
        _attn_implementation="eager",
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return Qwen3_5MoeConfig(**kwargs)


VARIANTS = (
    (
        "dense",
        modeling_qwen3_5,
        Qwen3_5RMSNorm,
        _tiny_dense_config,
        Qwen3_5TextModel,
        Qwen3_5DecoderLayer,
    ),
    (
        "moe",
        modeling_qwen3_5_moe,
        Qwen3_5MoeRMSNorm,
        _tiny_moe_config,
        Qwen3_5MoeModel,
        Qwen3_5MoeDecoderLayer,
    ),
)


def _assert_dispatch_policy(module, norm_cls, monkeypatch):
    calls = []

    def fake_native(hidden, _weight, _eps):
        calls.append("native")
        return hidden + 1

    def fake_residual(hidden, _weight, _eps):
        calls.append("residual")
        return hidden + 3

    def fake_family1(hidden, _weight, _eps):
        calls.append("family1")
        return hidden + 5

    monkeypatch.setattr(module, "native_zero_centered_rms_norm", fake_native)
    monkeypatch.setattr(module, "native_zero_centered_rms_norm_without_batch_invariant", fake_residual)
    monkeypatch.setattr(module, "fast_zero_centered_batch_invariant_rms_norm", fake_family1)
    monkeypatch.setattr(module, "fast_zero_centered_batch_invariant_residual_rms_norm", fake_residual)

    x = torch.ones(2, 4)
    residual = torch.full_like(x, 3)
    for mode in ("sglang", "sglang_fused"):
        set_rmsnorm_mode(mode)
        try:
            ordinary = norm_cls(4, exact_contract=False)
            exact = norm_cls(4, exact_contract=True)

            assert torch.equal(ordinary(x), x + 1)
            assert calls[-1] == "native"
            out, residual_out = ordinary(x, residual=residual, prenorm=True)
            assert torch.equal(out, x + residual + 3)
            assert torch.equal(residual_out, x + residual)
            assert calls[-1] == "residual"
            assert torch.equal(ordinary(x, force_sglang_residual=True), x + 3)
            assert calls[-1] == "residual"

            expected = "family1" if mode == "sglang_fused" else "native"
            expected_delta = 5 if mode == "sglang_fused" else 1
            assert torch.equal(exact(x), x + expected_delta)
            assert calls[-1] == expected
            assert torch.equal(exact(x, force_sglang_residual=True), x + 3)
            assert calls[-1] == "residual"

            # Exact selection is instance-owned and cannot leak into an
            # ordinary module living in the same process.
            assert torch.equal(ordinary(x), x + 1)
            assert calls[-1] == "native"
        finally:
            set_rmsnorm_mode("native")


def _assert_v2_admission_and_dispatch(module, norm_cls, monkeypatch):
    calls = []

    def fake_v2(hidden, _weight, _eps, *, residual=None):
        calls.append(residual is not None)
        if residual is None:
            return hidden + 7
        residual_out = hidden + residual
        return residual_out + 7, residual_out

    monkeypatch.setattr(module, "fast_zero_centered_families_v2_rms_norm", fake_v2)
    x = torch.ones(2, 4)
    residual = torch.full_like(x, 3)

    set_rmsnorm_mode("sglang_fused")
    try:
        default = norm_cls(4, exact_contract=True)
        candidate = norm_cls(4, exact_contract=True, rmsnorm_family="v2")
        assert default.rmsnorm_family == "v1"
        assert torch.equal(candidate(x), x + 7)
        out, residual_out = candidate(x, residual=residual, prenorm=True)
        assert torch.equal(out, x + residual + 7)
        assert torch.equal(residual_out, x + residual)
        assert calls == [False, True]
    finally:
        set_rmsnorm_mode("native")

    with pytest.raises(RuntimeError, match="only in the exact training lane"):
        norm_cls(4, exact_contract=False, rmsnorm_family="v2")

    rejected = norm_cls(4, exact_contract=True, rmsnorm_family="v2")
    with pytest.raises(RuntimeError, match="requires rmsnorm_mode='sglang_fused'"):
        rejected(x)


class CaptureNorm(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.force_values = []

    def forward(self, hidden_states, residual=None, prenorm=False, *, force_sglang_residual=False, **_kwargs):
        self.force_values.append(force_sglang_residual)
        if prenorm:
            return hidden_states, residual
        return hidden_states


class IdentityAttention(torch.nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return hidden_states, None


def _assert_v2_reaches_every_norm_site(variant):
    name, _module, norm_cls, config_factory, model_cls, _layer_cls = variant
    config = config_factory()
    config._qwen35_exact_contract = True
    config._qwen35_rmsnorm_family = "v2"
    set_rmsnorm_mode("sglang_fused")
    try:
        model = model_cls(config)
    finally:
        set_rmsnorm_mode("native")

    resolved = {
        module_name: module.rmsnorm_family
        for module_name, module in model.named_modules()
        if isinstance(module, norm_cls)
    }
    assert resolved, name
    assert set(resolved.values()) == {"v2"}, name
    assert "norm" in resolved
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        for suffix in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            assert resolved[f"{prefix}.{suffix}"] == "v2", (name, suffix)


def _run_to_post_attention_norm(name, layer, hidden):
    layer.input_layernorm = CaptureNorm(layer.input_layernorm.mode)
    captured = layer.input_layernorm
    layer.self_attn = IdentityAttention()
    layer.post_attention_layernorm = CaptureNorm("native")
    if name == "dense":
        layer.mlp = torch.nn.Identity()
        layer(hidden, position_embeddings=(hidden, hidden))
    else:
        layer._pre_mlp_forward(hidden, position_embeddings=(hidden, hidden))
    return captured


def _assert_layer_and_final_norm_site_policy(variant):
    name, _module, _norm_cls, config_factory, model_cls, layer_cls = variant
    hidden = torch.ones(1, 2, 8)
    for layer_idx, mode, expected in (
        (0, "sglang", False),
        (1, "native", False),
        (1, "sglang", True),
        (0, "sglang_fused", False),
        (1, "sglang_fused", True),
    ):
        layer = layer_cls(config_factory(), layer_idx=layer_idx)
        layer.input_layernorm.mode = mode
        captured = _run_to_post_attention_norm(name, layer, hidden)
        assert captured.force_values == [expected], (name, layer_idx, mode)

    class StubLayer(torch.nn.Module):
        layer_type = "full_attention"

        def forward(self, hidden_states, *_args, **_kwargs):
            return (hidden_states,)

    for mode, expected in (("native", False), ("sglang", True), ("sglang_fused", True)):
        model = model_cls(config_factory())
        model.layers = torch.nn.ModuleList([StubLayer()])
        model.norm = CaptureNorm(mode)
        model(input_ids=torch.tensor([[0, 1]]))
        assert model.norm.force_values == [expected], (name, mode)


def _assert_dense_gdn_norm_remains_separate():
    config = _tiny_dense_config(layer_types=["linear_attention", "full_attention"])
    config._qwen35_exact_contract = True
    config._qwen35_rmsnorm_family = "v2"
    set_rmsnorm_mode("sglang_fused")
    try:
        layer = Qwen3_5DecoderLayer(config, layer_idx=0)
    finally:
        set_rmsnorm_mode("native")

    assert layer.linear_attn is not None
    assert type(layer.linear_attn.o_norm).__name__ == "FusedRMSNormGated"
    assert not hasattr(layer.linear_attn.o_norm, "rmsnorm_family")


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


def _assert_v2_zero_centered_backward(monkeypatch):
    monkeypatch.setattr(normalization, "rms_norm_v2", _cpu_v2_forward)
    monkeypatch.setattr(normalization, "fused_rms_norm_backward", _cpu_rms_backward)
    torch.manual_seed(11)

    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    grad = torch.randn_like(x)
    normalization._FamiliesV2ZeroCenteredRMSNorm.apply(x, weight, EPS).backward(grad)

    x_ref = x.detach().requires_grad_(True)
    weight_ref = weight.detach().requires_grad_(True)
    _cpu_v2_forward(x_ref, weight_ref, EPS, zero_centered=True).backward(grad)
    assert torch.allclose(x.grad.float(), x_ref.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(weight.grad, weight_ref.grad, atol=2e-5, rtol=2e-5)

    x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn_like(x)
    grad_residual = torch.randn_like(residual)
    out, residual_out = normalization._FamiliesV2ZeroCenteredResidualRMSNorm.apply(x, residual, weight, EPS)
    torch.autograd.backward((out, residual_out), (grad_out, grad_residual))

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
    assert torch.allclose(x.grad.float(), x_ref.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.equal(x.grad, residual.grad)
    assert torch.allclose(residual.grad.float(), residual_ref.grad.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(weight.grad, weight_ref.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.cpu
def test_qwen3_5_norm_dispatch_site_and_backward_contract(monkeypatch):
    for variant in VARIANTS:
        _name, module, norm_cls, *_rest = variant
        with monkeypatch.context() as dispatch_patch:
            _assert_dispatch_policy(module, norm_cls, dispatch_patch)
        with monkeypatch.context() as v2_patch:
            _assert_v2_admission_and_dispatch(module, norm_cls, v2_patch)
        _assert_v2_reaches_every_norm_site(variant)
        _assert_layer_and_final_norm_site_policy(variant)

    _assert_dense_gdn_norm_remains_separate()
    with monkeypatch.context() as backward_patch:
        _assert_v2_zero_centered_backward(backward_patch)


def _assert_module_realization_matches_for_both_variants():
    torch.manual_seed(3)
    hidden = torch.randn(N_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(N_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    for norm_cls in (Qwen3_5RMSNorm, Qwen3_5MoeRMSNorm):
        set_rmsnorm_mode("sglang")
        try:
            sglang = norm_cls(HIDDEN, eps=EPS).cuda()
            set_rmsnorm_mode("sglang_fused")
            fused = norm_cls(HIDDEN, eps=EPS).cuda()
        finally:
            set_rmsnorm_mode("native")
        with torch.no_grad():
            sglang.weight.copy_(torch.randn(HIDDEN, device="cuda"))
            fused.weight.copy_(sglang.weight)

        with set_batch_invariant_mode(True), torch.no_grad():
            sglang_out, sglang_residual = sglang(hidden, residual=residual, prenorm=True)
            fused_out, fused_residual = fused(hidden, residual=residual, prenorm=True)
            assert torch.equal(sglang_out, fused_out)
            assert torch.equal(sglang_residual, fused_residual)
            assert torch.equal(
                sglang(hidden, force_sglang_residual=True),
                fused(hidden, force_sglang_residual=True),
            )
            assert torch.equal(sglang(hidden), fused(hidden))


def _assert_family1_matches_interpose():
    torch.manual_seed(21)
    head_dim = 128
    x = torch.randn(256, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    set_rmsnorm_mode("sglang_fused")
    try:
        norm = Qwen3_5MoeRMSNorm(head_dim, eps=EPS, exact_contract=True).cuda()
    finally:
        set_rmsnorm_mode("native")
    with torch.no_grad():
        norm.weight.copy_(torch.randn(head_dim, device="cuda"))
        out = norm(x)
        reference = rms_norm_batch_invariant(x.float(), 1.0 + norm.weight.float(), eps=EPS).to(x.dtype)
        with set_batch_invariant_mode(True):
            interpose = torch.nn.functional.rms_norm(
                x.float(),
                (head_dim,),
                1.0 + norm.weight.float(),
                eps=EPS,
            ).to(x.dtype)
    assert torch.equal(out, reference)
    assert torch.equal(out, interpose)


def _assert_moe_layer_forward_matches_realizations():
    torch.manual_seed(7)
    config = _tiny_moe_config(
        hidden_size=HIDDEN,
        intermediate_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    set_rmsnorm_mode("sglang")
    try:
        layer = Qwen3_5MoeDecoderLayer(config, layer_idx=1).to(device="cuda", dtype=torch.bfloat16)
    finally:
        set_rmsnorm_mode("native")
    layer.self_attn = IdentityAttention()
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16))
        layer.post_attention_layernorm.weight.copy_(torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16))

    hidden = torch.randn(1, 128, HIDDEN, device="cuda", dtype=torch.bfloat16)
    position = torch.zeros_like(hidden)
    with set_batch_invariant_mode(True), torch.no_grad():
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.mode = "sglang"
        (sglang_out,) = layer(hidden, position_embeddings=(position, position))
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.mode = "sglang_fused"
        (fused_out,) = layer(hidden, position_embeddings=(position, position))
    assert torch.equal(sglang_out, fused_out)


def _assert_family2_residual_matches_serving_tree():
    from xorl.models.layers.normalization import (  # noqa: PLC0415
        fast_zero_centered_batch_invariant_residual_rms_norm,
        native_zero_centered_rms_norm,
    )
    from xorl.ops.batch_invariant_ops import mean_dim  # noqa: PLC0415

    torch.manual_seed(11)
    x = torch.randn(513, HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(513, HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)

    set_rmsnorm_mode("sglang_fused")
    try:
        exact = Qwen3_5MoeRMSNorm(HIDDEN, eps=EPS, exact_contract=True).cuda()
        ordinary = Qwen3_5MoeRMSNorm(HIDDEN, eps=EPS, exact_contract=False).cuda()
    finally:
        set_rmsnorm_mode("native")
    with torch.no_grad():
        exact.weight.copy_(weight)
        ordinary.weight.copy_(weight)

        summed = (x + residual).float()
        variance = mean_dim(summed * summed, dim=-1, keepdim=True)
        reference = (summed * torch.rsqrt(variance + EPS) * (1.0 + weight.float())).to(torch.bfloat16)
        assert torch.equal(exact(x, residual=residual), reference)
        assert torch.equal(
            ordinary(x, residual=residual),
            native_zero_centered_rms_norm(x + residual, weight, EPS),
        )
        assert torch.equal(
            fast_zero_centered_batch_invariant_residual_rms_norm(x + residual, weight, EPS),
            reference,
        )


@requires_cuda
@pytest.mark.gpu
def test_qwen3_5_norm_bit_exact_model_contract():
    _assert_module_realization_matches_for_both_variants()
    _assert_family1_matches_interpose()
    _assert_moe_layer_forward_matches_realizations()
    _assert_family2_residual_matches_serving_tree()
