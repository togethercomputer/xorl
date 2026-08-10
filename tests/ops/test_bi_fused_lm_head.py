import pytest
import torch

from xorl.ops.loss.causallm_loss import causallm_loss_function


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

N, H, V = 256, 128, 12_800
IGNORE = -100


def _inputs(seed=0):
    torch.manual_seed(seed)
    hidden = (torch.randn(1, N, H, device="cuda") * 0.5).to(torch.bfloat16)
    weight = (torch.randn(V, H, device="cuda") * 0.05).to(torch.bfloat16)
    labels = torch.randint(0, V, (1, N), device="cuda")
    labels[0, :7] = IGNORE
    return hidden, weight, labels


@requires_cuda
@pytest.mark.gpu
def test_bi_fused_matches_eager_fp32_reference():
    hidden, weight, labels = _inputs()
    out = causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=True, return_per_token=True)
    ref = causallm_loss_function(hidden, weight, labels, ce_mode="eager", lm_head_fp32=True, return_per_token=True)
    assert torch.allclose(out.per_token_logprobs, ref.per_token_logprobs, rtol=1e-4, atol=1e-5)
    assert torch.allclose(out.loss.float(), ref.loss.float(), rtol=1e-4, atol=1e-5)
    # ignored positions contribute exactly zero
    assert (out.per_token_loss.view(-1)[:7] == 0).all()


@requires_cuda
@pytest.mark.gpu
def test_bi_fused_backward_matches_eager_autograd():
    hidden, weight, labels = _inputs(1)

    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    causallm_loss_function(h1, w1, labels, ce_mode="bi_fused", lm_head_fp32=True).loss.backward()

    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    causallm_loss_function(h2, w2, labels, ce_mode="eager", lm_head_fp32=True).loss.backward()

    assert torch.allclose(h1.grad.float(), h2.grad.float(), rtol=2e-2, atol=2e-4)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), rtol=2e-2, atol=2e-4)


@requires_cuda
@pytest.mark.gpu
def test_bi_fused_deterministic_and_batch_invariant():
    hidden, weight, labels = _inputs(2)
    kw = dict(ce_mode="bi_fused", lm_head_fp32=True, return_per_token=True)
    a = causallm_loss_function(hidden, weight, labels, **kw).per_token_logprobs
    b = causallm_loss_function(hidden, weight, labels, **kw).per_token_logprobs
    assert torch.equal(a, b)
    sub = causallm_loss_function(hidden[:, 64:128], weight, labels[:, 64:128], **kw).per_token_logprobs
    assert torch.equal(sub, a[:, 64:128])


@requires_cuda
@pytest.mark.gpu
def test_bi_fused_guards():
    hidden, weight, labels = _inputs(3)
    with pytest.raises(NotImplementedError, match="lm_head_fp32"):
        causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=False)
    with pytest.raises(NotImplementedError, match="softmax_auxiliary_loss"):
        causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=True, z_loss_coef=0.1)


@requires_cuda
@pytest.mark.gpu
def test_bi_fused_temperature_matches_eager_reference():
    hidden, weight, labels = _inputs(4)
    kw = dict(lm_head_fp32=True, logprob_temperature=0.7, return_per_token=True)
    out = causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", **kw)
    ref = causallm_loss_function(hidden, weight, labels, ce_mode="eager", **kw)
    assert torch.allclose(out.per_token_logprobs, ref.per_token_logprobs, rtol=1e-4, atol=1e-5)
    assert (out.per_token_loss.view(-1)[:7] == 0).all()


@requires_cuda
@pytest.mark.gpu
def test_bi_fused_temperature_backward_matches_eager_autograd():
    hidden, weight, labels = _inputs(5)

    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    causallm_loss_function(
        h1, w1, labels, ce_mode="bi_fused", lm_head_fp32=True, logprob_temperature=0.7
    ).loss.backward()

    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    causallm_loss_function(h2, w2, labels, ce_mode="eager", lm_head_fp32=True, logprob_temperature=0.7).loss.backward()

    assert torch.allclose(h1.grad.float(), h2.grad.float(), rtol=2e-2, atol=2e-4)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), rtol=2e-2, atol=2e-4)


@requires_cuda
@pytest.mark.gpu
def test_bi_kernel_unit_temperature_is_exact_identity():
    from xorl.ops.batch_invariant_ops import bi_lm_head_selected_logprob

    hidden, weight, _ = _inputs(6)
    hidden = hidden.view(-1, hidden.shape[-1])
    ids = torch.randint(0, weight.shape[0], (hidden.shape[0],), device="cuda")
    ones = torch.ones(hidden.shape[0], dtype=torch.float32, device="cuda")
    lp_none, lse_none, sel_none = bi_lm_head_selected_logprob(hidden, weight, ids)
    lp_ones, lse_ones, sel_ones = bi_lm_head_selected_logprob(hidden, weight, ids, temperature=ones)
    # x * (1/1.0) is an IEEE identity: the temperature path at T=1 is bitwise
    # the proven contract path.
    assert torch.equal(lp_none, lp_ones)
    assert torch.equal(lse_none, lse_ones)
    assert torch.equal(sel_none, sel_ones)


@requires_cuda
@pytest.mark.gpu
def test_bi_kernel_p1_tokens_clamp_to_exact_zero():
    from xorl.ops.batch_invariant_ops import bi_lm_head_selected_logprob

    torch.manual_seed(7)
    N, H, V = 8192, 1024, 12800
    weight = (torch.randn(V, H, device="cuda") * 0.05).to(torch.bfloat16)
    ids = torch.randint(0, V, (N,), device="cuda")
    scale = 1.0 + torch.rand(N, 1, device="cuda") * 30
    hidden = (weight[ids].float() * scale).to(torch.bfloat16)
    temp = torch.full((N,), 0.7, dtype=torch.float32, device="cuda")
    for t in (None, temp):
        lp, _, _ = bi_lm_head_selected_logprob(hidden, weight, ids, temperature=t)
        # In exact math logprob <= 0; the one-ulp LSE boundary case (p~1 tokens,
        # observed live as +2**-18) must clamp to exactly 0.0, never positive.
        assert (lp <= 0).all()
        assert (lp == 0).any()
