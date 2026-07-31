"""Parity + gradient tests for the fused batch-invariant ``sglang_fused`` RMSNorm.

``sglang_fused`` must be *bit-for-bit* identical to ``sglang`` mode on the dense
Qwen3 forward (that is what preserves the static K3 logprob parity), while
replacing the eager residual-style norms with fused Triton kernels. The GPU
tests assert ``torch.equal`` against the eager reference under batch-invariant
mode (the K3 regime), and check that the closed-form backward matches autograd
of the eager reference. The CPU tests exercise the eager fallback.
"""

import os

import pytest
import torch

from xorl.models.layers.normalization import (
    RMSNorm,
    fast_batch_invariant_rms_norm,
    fast_sglang_residual_rms_norm,
    fast_sglang_rms_norm,
    set_rmsnorm_mode,
    sglang_residual_rms_norm,
)
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from xorl.ops.batch_invariant_ops import (
    fused_add_rms_norm_batch_invariant,
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
    set_trunk_linear_contract,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Dense Qwen3-1.7B hidden size; a packed multi-prompt token count.
HIDDEN = 2048
N_TOKENS = 1024
EPS = 1e-6


# --------------------------------------------------------------------------- #
# CPU fallback (no Triton): sglang_fused must equal the eager sglang path.
# --------------------------------------------------------------------------- #
def test_sglang_fused_cpu_residual_matches_eager():
    set_rmsnorm_mode("sglang_fused")
    try:
        norm = RMSNorm(4, eps=EPS)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
        hidden_states = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
        residual = torch.tensor([[1.0, 0.5, -0.25, 0.125]], dtype=torch.float32)

        out, residual_out = norm(hidden_states, residual=residual, prenorm=True)

        expected_residual = hidden_states + residual
        expected = sglang_residual_rms_norm(expected_residual, norm.weight, EPS)
        assert torch.equal(residual_out, expected_residual)
        assert torch.equal(out, expected)
    finally:
        set_rmsnorm_mode("native")


# This suite pins the v1 family kernels through the fast_* dispatchers; with
# families-v2 default-on those dispatchers route to the v2 tree, so pin the
# kill switch (env is read per call). v2 has its own suite (test_bi_families_v2.py).
os.environ["XORL_FAMILIES_V2"] = "0"


def test_sglang_fused_cpu_force_no_residual_matches_eager():
    set_rmsnorm_mode("sglang_fused")
    try:
        norm = RMSNorm(4, eps=EPS)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
        hidden_states = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)

        out = norm(hidden_states, force_sglang_residual=True)
        expected = sglang_residual_rms_norm(hidden_states, norm.weight, EPS)
        assert torch.equal(out, expected)
    finally:
        set_rmsnorm_mode("native")


# --------------------------------------------------------------------------- #
# GPU bit-exactness under batch-invariant mode (the K3 regime).
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_residual_bit_exact_vs_eager(dtype):
    torch.manual_seed(0)
    device = "cuda"
    hidden = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    residual = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    weight = torch.randn(HIDDEN, device=device, dtype=dtype)

    with set_batch_invariant_mode(True):
        expected_residual = hidden + residual
        expected = sglang_residual_rms_norm(expected_residual, weight, EPS)
        out, residual_out = fast_sglang_residual_rms_norm(hidden, residual, weight, EPS)

    # Residual carry must be bit-identical (it feeds the next layer's stream).
    assert torch.equal(residual_out, expected_residual), "fused residual add diverged from torch add"
    assert torch.equal(out, expected), "fused residual RMSNorm diverged from eager"


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_no_residual_bit_exact_vs_eager(dtype):
    torch.manual_seed(1)
    device = "cuda"
    hidden = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    weight = torch.randn(HIDDEN, device=device, dtype=dtype)

    with set_batch_invariant_mode(True):
        expected = sglang_residual_rms_norm(hidden, weight, EPS)
        out = fast_sglang_rms_norm(hidden, weight, EPS)

    assert torch.equal(out, expected), "fused no-residual RMSNorm diverged from eager"


@requires_cuda
@pytest.mark.gpu
def test_fused_residual_matches_3d_packed_shape():
    torch.manual_seed(2)
    device = "cuda"
    hidden = torch.randn(2, 96, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(2, 96, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)

    with set_batch_invariant_mode(True):
        expected_residual = hidden + residual
        expected = sglang_residual_rms_norm(expected_residual, weight, EPS)
        out, residual_out = fast_sglang_residual_rms_norm(hidden, residual, weight, EPS)

    assert out.shape == hidden.shape
    assert residual_out.shape == hidden.shape
    assert torch.equal(residual_out, expected_residual)
    assert torch.equal(out, expected)


# --------------------------------------------------------------------------- #
# GPU: sglang_fused RMSNorm module == sglang module, bit-for-bit.
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
def test_module_sglang_fused_equals_sglang():
    torch.manual_seed(3)
    device = "cuda"
    hidden = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)

    def run(mode, **kwargs):
        norm = RMSNorm(HIDDEN, eps=EPS, mode=mode).to(device)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(HIDDEN, device=device))
        return norm, norm(hidden, **kwargs)

    with set_batch_invariant_mode(True):
        # Residual (post-attention layernorm) path.
        sg = RMSNorm(HIDDEN, eps=EPS, mode="sglang").to(device)
        with torch.no_grad():
            sg.weight.copy_(torch.randn(HIDDEN, device=device))
        sf = RMSNorm(HIDDEN, eps=EPS, mode="sglang_fused").to(device)
        with torch.no_grad():
            sf.weight.copy_(sg.weight)

        out_sg, rout_sg = sg(hidden, residual=residual, prenorm=True)
        out_sf, rout_sf = sf(hidden, residual=residual, prenorm=True)
        assert torch.equal(out_sg, out_sf)
        assert torch.equal(rout_sg, rout_sf)

        # force_sglang_residual (input layernorm layer>0 / final norm) path.
        out_sg2 = sg(hidden, force_sglang_residual=True)
        out_sf2 = sf(hidden, force_sglang_residual=True)
        assert torch.equal(out_sg2, out_sf2)

        # No-residual, no-force (q/k norm, layer-0 input) path -> both native.
        # Under the trunk contract the interposed aten::rms_norm refuses grad-requiring
        # inputs (it records no graph), so this forward-only check runs no_grad.
        with torch.no_grad():
            out_sg3 = sg(hidden)
            out_sf3 = sf(hidden)
        assert torch.equal(out_sg3, out_sf3)


# --------------------------------------------------------------------------- #
# GPU: closed-form backward matches autograd of the eager reference.
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
def test_fused_residual_backward_matches_autograd():
    torch.manual_seed(4)
    device = "cuda"
    dtype = torch.bfloat16

    # Build inputs once; both paths differentiate the *same* tensors.
    h0 = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    r0 = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    w0 = torch.randn(HIDDEN, device=device, dtype=torch.float32)
    g_out = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    g_rout = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)

    def leaf(t):
        return t.clone().detach().requires_grad_(True)

    # Reference: eager add + eager norm, differentiated by autograd.
    h_ref, r_ref, w_ref = leaf(h0), leaf(r0), leaf(w0)
    residual_out_ref = h_ref + r_ref
    out_ref = sglang_residual_rms_norm(residual_out_ref, w_ref, EPS)
    torch.autograd.backward([out_ref, residual_out_ref], [g_out, g_rout])

    # Fused: same inputs, closed-form backward.
    h_f, r_f, w_f = leaf(h0), leaf(r0), leaf(w0)
    out_f, residual_out_f = fast_sglang_residual_rms_norm(h_f, r_f, w_f, EPS)
    torch.autograd.backward([out_f, residual_out_f], [g_out, g_rout])

    assert torch.allclose(h_f.grad.float(), h_ref.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(r_f.grad.float(), r_ref.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(w_f.grad.float(), w_ref.grad.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.gpu
def test_dense_qwen3_layer_forward_bit_exact_sglang_vs_fused():
    """Full dense Qwen3 decoder-layer forward must be bit-identical between
    sglang and sglang_fused (the model-level K3-preservation gate). Exercises
    input_layernorm (force_sglang_residual path at layer>0) and
    post_attention_layernorm (residual path) with the real RMSNorm modules.
    """

    class IdentityAttention(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    torch.manual_seed(7)
    device = "cuda"
    cfg = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=4096,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=2,
        _attn_implementation="eager",
    )
    layer = Qwen3DecoderLayer(cfg, layer_idx=1).to(device=device, dtype=torch.bfloat16)
    layer.self_attn = IdentityAttention()
    # Randomize the norm weights away from the all-ones init.
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

    assert torch.equal(out_sg, out_sf), "dense layer forward diverged between sglang and sglang_fused"


@requires_cuda
@pytest.mark.gpu
def test_single_tensor_force_call_bit_matches_serving_fused_residual_tree():
    """The layer>0 input-norm / final-norm call shape (pre-summed single tensor,
    force_sglang_residual=True) must be bit-identical to serving's fused residual
    tree (``fused_add_rms_norm_batch_invariant``). Guards the norm-seed trap where
    this call fell through native -> batch-invariant aten interception -> the
    vllm-style rms_norm kernel, which disagrees at 1 ulp on rare boundary values
    (the k3(1 fp32 ulp) floor)."""
    torch.manual_seed(11)
    x = torch.randn(512, HIDDEN, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(HIDDEN, eps=EPS, mode="sglang_fused").to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HIDDEN, device="cuda").to(torch.bfloat16))

    with set_batch_invariant_mode(True), torch.no_grad():
        out_module = norm(x, force_sglang_residual=True)
        ref, residual_out = fused_add_rms_norm_batch_invariant(x, torch.zeros_like(x), norm.weight, EPS)

    assert torch.equal(residual_out, x)
    assert torch.equal(out_module, ref)


@requires_cuda
@pytest.mark.gpu
def test_fused_no_residual_backward_matches_autograd():
    torch.manual_seed(5)
    device = "cuda"
    dtype = torch.bfloat16

    h0 = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)
    w0 = torch.randn(HIDDEN, device=device, dtype=torch.float32)
    g_out = torch.randn(N_TOKENS, HIDDEN, device=device, dtype=dtype)

    def leaf(t):
        return t.clone().detach().requires_grad_(True)

    h_ref, w_ref = leaf(h0), leaf(w0)
    out_ref = sglang_residual_rms_norm(h_ref, w_ref, EPS)
    out_ref.backward(g_out)

    h_f, w_f = leaf(h0), leaf(w0)
    out_f = fast_sglang_rms_norm(h_f, w_f, EPS)
    out_f.backward(g_out)

    assert torch.allclose(h_f.grad.float(), h_ref.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(w_f.grad.float(), w_ref.grad.float(), rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# Trunk contract lane (XORL_BI_TRUNK_LINEAR): the no-residual dispatch (qk-norm)
# must bit-match serving's family-1 batch-invariant kernel — which is the
# aten::rms_norm interpose kernel, NOT the fused sglang residual tree (the two
# disagree at 1 ulp on rare bf16 boundary values).
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
def test_trunk_contract_no_residual_bit_matches_interpose_kernel():
    torch.manual_seed(21)
    # qk-norm call shape: [tokens, heads, head_dim] with a head_dim-sized weight.
    head_dim = 128
    x = torch.randn(256, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(head_dim, eps=EPS, mode="sglang_fused").to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(head_dim, device="cuda").to(torch.bfloat16))

    set_trunk_linear_contract(True)
    try:
        with torch.no_grad():
            out = norm(x)
            ref = rms_norm_batch_invariant(x, norm.weight, eps=EPS)
            with set_batch_invariant_mode(True):
                ref_interpose = torch.nn.functional.rms_norm(x, (head_dim,), norm.weight, eps=EPS)
    finally:
        set_trunk_linear_contract(False)

    assert torch.equal(out, ref)
    assert torch.equal(out, ref_interpose), "contract-lane qk-norm must equal the aten interpose lane bit-for-bit"


@requires_cuda
@pytest.mark.gpu
def test_no_residual_dispatch_unchanged_without_contract():
    torch.manual_seed(22)
    x = torch.randn(N_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(HIDDEN, eps=EPS, mode="sglang_fused").to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = norm(x)
        ref = torch.nn.functional.rms_norm(x, (HIDDEN,), norm.weight, eps=EPS)
    assert torch.equal(out, ref)


@requires_cuda
@pytest.mark.gpu
def test_trunk_contract_no_residual_backward_matches_eager():
    # Same convention as test_fused_no_residual_backward_matches_autograd: fp32
    # weight leaf and the fp32-multiply eager reference (the kernel's semantics),
    # so the comparison is not dominated by bf16 grad-accumulation rounding.
    torch.manual_seed(23)
    head_dim = 128
    h0 = torch.randn(512, head_dim, device="cuda", dtype=torch.bfloat16)
    w0 = torch.randn(head_dim, device="cuda", dtype=torch.float32)
    g_out = torch.randn(512, head_dim, device="cuda", dtype=torch.bfloat16)

    def leaf(t):
        return t.clone().detach().requires_grad_(True)

    h_ref, w_ref = leaf(h0), leaf(w0)
    out_ref = sglang_residual_rms_norm(h_ref, w_ref, EPS)
    out_ref.backward(g_out)

    h_c, w_c = leaf(h0), leaf(w0)
    out_c = fast_batch_invariant_rms_norm(h_c, w_c, EPS)
    out_c.backward(g_out)

    assert torch.isfinite(h_c.grad.float()).all() and torch.isfinite(w_c.grad.float()).all()
    assert torch.allclose(h_c.grad.float(), h_ref.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(w_c.grad.float(), w_ref.grad.float(), rtol=2e-2, atol=2e-2)
