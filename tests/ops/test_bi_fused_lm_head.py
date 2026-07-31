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
    with pytest.raises(NotImplementedError, match="temperature"):
        causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=True, logprob_temperature=0.7)
    with pytest.raises(NotImplementedError, match="softmax_auxiliary_loss"):
        causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=True, z_loss_coef=0.1)
