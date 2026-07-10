import pytest
import torch

from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5_moe import modeling_qwen3_5_moe
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeModel,
    Qwen3_5MoeRMSNorm,
)
from xorl.ops.batch_invariant_ops import (
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
    set_trunk_linear_contract,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HIDDEN = 2048
N_TOKENS = 512
EPS = 1e-6


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


def test_qwen3_5_moe_sglang_fused_rmsnorm_routes_same_families_as_sglang(monkeypatch):
    """Norm-seed contract (§14) port gate: ``sglang_fused`` must dispatch the same
    family functions as ``sglang`` (family-2 = no-batch-invariant residual tree for
    residual/forced norms, family-1 = native for plain no-residual norms)."""
    calls = []

    def fake_native(hidden_states, weight, variance_epsilon):
        calls.append("native")
        return hidden_states + 1

    def fake_native_no_batch_invariant(hidden_states, weight, variance_epsilon):
        calls.append("native_no_batch_invariant")
        return hidden_states + 3

    def fake_contract(hidden_states, weight, variance_epsilon):
        calls.append("contract_family1")
        return hidden_states + 5

    monkeypatch.setattr(modeling_qwen3_5_moe, "native_zero_centered_rms_norm", fake_native)
    monkeypatch.setattr(
        modeling_qwen3_5_moe,
        "native_zero_centered_rms_norm_without_batch_invariant",
        fake_native_no_batch_invariant,
    )
    monkeypatch.setattr(
        modeling_qwen3_5_moe,
        "fast_zero_centered_batch_invariant_rms_norm",
        fake_contract,
    )

    set_rmsnorm_mode("sglang_fused")
    try:
        norm = modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm(4)
        x = torch.ones(2, 4)

        out = norm(x)
        assert torch.equal(out, x + 1)
        assert calls[-1] == "native"

        residual = torch.full((2, 4), 3.0)
        out, residual_out = norm(x, residual=residual, prenorm=True)
        assert torch.equal(residual_out, x + residual)
        assert torch.equal(out, x + residual + 3)
        assert calls[-1] == "native_no_batch_invariant"

        out = norm(x, force_sglang_residual=True)
        assert torch.equal(out, x + 3)
        assert calls[-1] == "native_no_batch_invariant"

        # Trunk-contract lane: the no-residual (family-1) dispatch swaps to the
        # batch-invariant contract kernel; residual/forced stay family-2.
        set_trunk_linear_contract(True)
        try:
            out = norm(x)
            assert torch.equal(out, x + 5)
            assert calls[-1] == "contract_family1"

            out = norm(x, force_sglang_residual=True)
            assert torch.equal(out, x + 3)
            assert calls[-1] == "native_no_batch_invariant"
        finally:
            set_trunk_linear_contract(False)
    finally:
        set_rmsnorm_mode("native")


# --------------------------------------------------------------------------- #
# Call sites: layer>0 input norm and the final norm must force family-2 in both
# sglang and sglang_fused modes (§14 family assignment, ported from qwen3_moe).
# --------------------------------------------------------------------------- #
class CaptureInputNorm(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.force_sglang_residual_values = []

    def forward(self, hidden_states, *, force_sglang_residual=False):
        self.force_sglang_residual_values.append(force_sglang_residual)
        return hidden_states


class IdentityAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class IdentityPostAttentionNorm(torch.nn.Module):
    def forward(self, hidden_states, residual=None, prenorm=False, **kwargs):
        return hidden_states, residual


def _tiny_config(**overrides) -> Qwen3_5MoeConfig:
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


@pytest.mark.parametrize(
    ("layer_idx", "mode", "expected_force"),
    [
        (0, "sglang", False),
        (1, "native", False),
        (1, "sglang", True),
        (0, "sglang_fused", False),
        (1, "sglang_fused", True),
    ],
)
def test_qwen3_5_moe_layer_input_norm_forces_sglang_residual_after_layer0(layer_idx, mode, expected_force):
    layer = Qwen3_5MoeDecoderLayer(_tiny_config(), layer_idx=layer_idx)
    assert layer.layer_idx == layer_idx

    input_norm = CaptureInputNorm(mode)
    layer.input_layernorm = input_norm
    layer.self_attn = IdentityAttention()
    layer.post_attention_layernorm = IdentityPostAttentionNorm()

    hidden_states = torch.ones(1, 2, 8)
    layer._pre_mlp_forward(hidden_states, position_embeddings=(hidden_states, hidden_states))

    assert input_norm.force_sglang_residual_values == [expected_force]


@pytest.mark.parametrize(
    ("mode", "expected_force"),
    [
        ("native", False),
        ("sglang", True),
        ("sglang_fused", True),
    ],
)
def test_qwen3_5_moe_model_final_norm_forces_sglang_residual(mode, expected_force):
    class StubLayer(torch.nn.Module):
        layer_type = "full_attention"

        def forward(self, hidden_states, *args, **kwargs):
            return (hidden_states,)

    model = Qwen3_5MoeModel(_tiny_config())
    model.layers = torch.nn.ModuleList([StubLayer()])
    final_norm = CaptureInputNorm(mode)
    model.norm = final_norm

    model(input_ids=torch.tensor([[0, 1]]))

    assert final_norm.force_sglang_residual_values == [expected_force]


# --------------------------------------------------------------------------- #
# GPU: module-level bitwise verify — sglang_fused == sglang on every dispatch
# shape, and the trunk-contract family-1 kernel == the aten interpose lane.
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
def test_qwen3_5_module_sglang_fused_equals_sglang_bitwise():
    torch.manual_seed(3)
    device = "cuda"
    hidden = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)

    set_rmsnorm_mode("sglang")
    try:
        sg = Qwen3_5MoeRMSNorm(HIDDEN, eps=EPS).to(device)
        set_rmsnorm_mode("sglang_fused")
        sf = Qwen3_5MoeRMSNorm(HIDDEN, eps=EPS).to(device)
    finally:
        set_rmsnorm_mode("native")
    with torch.no_grad():
        sg.weight.copy_(torch.randn(HIDDEN, device=device))
        sf.weight.copy_(sg.weight)
    assert sg.mode == "sglang" and sf.mode == "sglang_fused"

    with set_batch_invariant_mode(True), torch.no_grad():
        # Residual (post-attention layernorm) path — family-2.
        out_sg, rout_sg = sg(hidden, residual=residual, prenorm=True)
        out_sf, rout_sf = sf(hidden, residual=residual, prenorm=True)
        assert torch.equal(out_sg, out_sf)
        assert torch.equal(rout_sg, rout_sf)

        # force_sglang_residual (layer>0 input norm / final norm) — family-2.
        out_sg2 = sg(hidden, force_sglang_residual=True)
        out_sf2 = sf(hidden, force_sglang_residual=True)
        assert torch.equal(out_sg2, out_sf2)

        # No-residual, no-force (qk-norm / layer-0 input) — family-1 (interpose).
        out_sg3 = sg(hidden)
        out_sf3 = sf(hidden)
        assert torch.equal(out_sg3, out_sf3)


@requires_cuda
@pytest.mark.gpu
def test_qwen3_5_trunk_contract_family1_bit_matches_interpose_kernel():
    """Family-1 term of the Q3.5 norm-seed contract: under the trunk-contract
    lane, the no-residual dispatch (qk-norm shape) must equal the aten::rms_norm
    interpose lane bit-for-bit — the same guarantee qwen3_moe has post-§14."""
    torch.manual_seed(21)
    head_dim = 128
    x = torch.randn(256, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    set_rmsnorm_mode("sglang_fused")
    try:
        norm = Qwen3_5MoeRMSNorm(head_dim, eps=EPS).to("cuda")
    finally:
        set_rmsnorm_mode("native")
    with torch.no_grad():
        norm.weight.copy_(torch.randn(head_dim, device="cuda"))

    set_trunk_linear_contract(True)
    try:
        with torch.no_grad():
            out = norm(x)
            ref = rms_norm_batch_invariant(x.float(), 1.0 + norm.weight.float(), eps=EPS).to(x.dtype)
            with set_batch_invariant_mode(True):
                ref_interpose = torch.nn.functional.rms_norm(
                    x.float(), (head_dim,), 1.0 + norm.weight.float(), eps=EPS
                ).to(x.dtype)
    finally:
        set_trunk_linear_contract(False)

    assert torch.equal(out, ref)
    assert torch.equal(out, ref_interpose), "contract-lane family-1 must equal the aten interpose lane bit-for-bit"


@requires_cuda
@pytest.mark.gpu
def test_qwen3_5_layer_forward_bit_exact_sglang_vs_fused():
    """Full Q3.5 full-attention decoder-layer forward must be bit-identical
    between sglang and sglang_fused (model-level §14 gate for the hybrid).
    Exercises the layer>0 input norm (forced family-2) and the post-attention
    residual norm (family-2) with the real layer modules; family-1 (qk-norm)
    is covered by the module-level tests above."""
    torch.manual_seed(7)
    device = "cuda"
    cfg = _tiny_config(
        hidden_size=HIDDEN,
        intermediate_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    set_rmsnorm_mode("sglang")
    try:
        layer = Qwen3_5MoeDecoderLayer(cfg, layer_idx=1).to(device=device, dtype=torch.bfloat16)
    finally:
        set_rmsnorm_mode("native")
    layer.self_attn = IdentityAttention()
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(torch.randn(HIDDEN, device=device).to(torch.bfloat16))
        layer.post_attention_layernorm.weight.copy_(torch.randn(HIDDEN, device=device).to(torch.bfloat16))

    hidden = torch.randn(1, 128, HIDDEN, device=device, dtype=torch.bfloat16)
    pos = torch.zeros(1, 128, HIDDEN, device=device, dtype=torch.bfloat16)

    with set_batch_invariant_mode(True), torch.no_grad():
        for m in (layer.input_layernorm, layer.post_attention_layernorm):
            m.mode = "sglang"
        (out_sg,) = layer(hidden, position_embeddings=(pos, pos))
        for m in (layer.input_layernorm, layer.post_attention_layernorm):
            m.mode = "sglang_fused"
        (out_sf,) = layer(hidden, position_embeddings=(pos, pos))

    assert torch.equal(out_sg, out_sf), "Q3.5 layer forward diverged between sglang and sglang_fused"


@requires_cuda
@pytest.mark.gpu
def test_qwen3_5_family2_residual_norm_contract(monkeypatch):
    """XORL_BI_RESIDUAL_NORM=1: the residual-tree (family-2) norm must run the
    eager-with-BI-mean composition (pairs with the BI-ops sampler's
    forward_native under the interposed aten::mean.dim), and the flag off must
    preserve the F.rms_norm path bit-for-bit."""
    from xorl.models.layers.normalization import (  # noqa: PLC0415
        fast_zero_centered_batch_invariant_residual_rms_norm,
        native_zero_centered_rms_norm,
    )
    from xorl.ops.batch_invariant_ops import mean_dim  # noqa: PLC0415

    torch.manual_seed(11)
    device = "cuda"
    x = torch.randn(513, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(513, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(HIDDEN, device=device) * 0.02).to(torch.bfloat16)

    set_rmsnorm_mode("sglang_fused")
    try:
        norm = modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm(HIDDEN, eps=1e-6).to(device)
    finally:
        set_rmsnorm_mode("native")
    with torch.no_grad():
        norm.weight.copy_(weight)

    # reference: eager fp32 composition with the BI mean kernel (the sampler's
    # forward_native under SGLANG_BATCH_INVARIANT_OPS=all)
    y = (x + residual).float()
    var = mean_dim(y * y, dim=-1, keepdim=True)
    ref = (y * torch.rsqrt(var + 1e-6) * (1.0 + weight.float())).to(torch.bfloat16)

    monkeypatch.setenv("XORL_BI_RESIDUAL_NORM", "1")
    with torch.no_grad():
        out_on = norm(x, residual=residual)
    assert torch.equal(out_on, ref), "family-2 contract output != eager-with-BI-mean composition"

    monkeypatch.delenv("XORL_BI_RESIDUAL_NORM")
    with torch.no_grad():
        out_off = norm(x, residual=residual)
    expected_off = native_zero_centered_rms_norm(x + residual, weight, 1e-6)
    assert torch.equal(out_off, expected_off), "flag off must preserve the native family-2 path"

    # the standalone helper equals the contract dispatch
    helper = fast_zero_centered_batch_invariant_residual_rms_norm(x + residual, weight, 1e-6)
    assert torch.equal(helper, ref)
