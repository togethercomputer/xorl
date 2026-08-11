import pytest
import torch

from xorl.ops import batch_invariant_ops as v1
from xorl.ops import bi_families_v2 as v2
from xorl.ops.loss.bi_fused_lm_head import bi_fused_per_token_ce
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
def test_bi_fused_forward_backward_and_kernel_edge_policy():
    for seed, temperature in ((0, None), (4, 0.7)):
        hidden, weight, labels = _inputs(seed)
        fused_hidden = hidden.clone().requires_grad_(True)
        fused_weight = weight.clone().requires_grad_(True)
        eager_hidden = hidden.clone().requires_grad_(True)
        eager_weight = weight.clone().requires_grad_(True)
        kwargs = {"lm_head_fp32": True, "return_per_token": True}
        if temperature is not None:
            kwargs["logprob_temperature"] = temperature

        out = causallm_loss_function(fused_hidden, fused_weight, labels, ce_mode="bi_fused", **kwargs)
        ref = causallm_loss_function(eager_hidden, eager_weight, labels, ce_mode="eager", **kwargs)
        assert torch.allclose(out.per_token_logprobs, ref.per_token_logprobs, rtol=1e-4, atol=1e-5)
        assert torch.allclose(out.loss.float(), ref.loss.float(), rtol=1e-4, atol=1e-5)
        assert (out.per_token_loss.view(-1)[:7] == 0).all()
        out.loss.backward()
        ref.loss.backward()
        assert torch.allclose(fused_hidden.grad.float(), eager_hidden.grad.float(), rtol=2e-2, atol=2e-4)
        assert torch.allclose(fused_weight.grad.float(), eager_weight.grad.float(), rtol=2e-2, atol=2e-4)

    _assert_bi_fused_deterministic_and_batch_invariant()
    _assert_bi_fused_guards()
    _assert_bi_kernel_unit_temperature_is_exact_identity()
    _assert_head_v2_projection_stats_invariance_and_training_policy()


def _assert_bi_fused_deterministic_and_batch_invariant():
    hidden, weight, labels = _inputs(2)
    kw = dict(ce_mode="bi_fused", lm_head_fp32=True, return_per_token=True)
    a = causallm_loss_function(hidden, weight, labels, **kw).per_token_logprobs
    b = causallm_loss_function(hidden, weight, labels, **kw).per_token_logprobs
    assert torch.equal(a, b)
    sub = causallm_loss_function(hidden[:, 64:128], weight, labels[:, 64:128], **kw).per_token_logprobs
    assert torch.equal(sub, a[:, 64:128])


def _assert_bi_fused_guards():
    hidden, weight, labels = _inputs(3)
    with pytest.raises(NotImplementedError, match="lm_head_fp32"):
        causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=False)
    with pytest.raises(NotImplementedError, match="softmax_auxiliary_loss"):
        causallm_loss_function(hidden, weight, labels, ce_mode="bi_fused", lm_head_fp32=True, z_loss_coef=0.1)


def _assert_bi_kernel_unit_temperature_is_exact_identity():
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

    _assert_bi_kernel_p1_tokens_clamp_to_exact_zero()


def _assert_bi_kernel_p1_tokens_clamp_to_exact_zero():
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


def _assert_head_v2_projection_stats_invariance_and_training_policy():
    def make(shape, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16).cuda()

    hidden, weight = make((16, 128), 1), make((512, 128), 2)
    tokens = torch.arange(16, device="cuda") * 29 % 512

    logits, decode_lse = v2.head_v2_full_logits_with_lse(hidden, weight)
    reference_logits = torch.empty_like(logits)
    v1._bi_lm_head_chunk_gemm_fp32(hidden, weight.t(), reference_logits)
    assert torch.equal(logits, reference_logits)

    logprob, score_lse, selected = v2.head_v2_selected_logprob(hidden, weight, tokens)
    assert torch.equal(decode_lse, score_lse)
    assert torch.equal(selected, logits.gather(1, tokens[:, None]).squeeze(1))
    assert torch.equal(logprob, torch.clamp_max(selected - decode_lse, 0.0))

    hidden, weight = make((32, 128), 3), make((512, 128), 4)
    tokens = torch.arange(32, device="cuda") * 31 % 512
    full, _, _ = v2.head_v2_selected_logprob(hidden, weight, tokens)
    sub, _, _ = v2.head_v2_selected_logprob(hidden[8:16].contiguous(), weight, tokens[8:16].contiguous())
    assert torch.equal(sub, full[8:16])

    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    ce = bi_fused_per_token_ce(hidden, weight, tokens)
    assert torch.equal(ce, -full)
    ce.mean().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(weight.grad).all()

    v2._select_qwen35_families_v1()
    try:
        qualified_v1 = bi_fused_per_token_ce(hidden.detach(), weight.detach(), tokens, vocab_chunk=256)
        assert torch.isfinite(qualified_v1).all()
    finally:
        v2._select_nonexact_families()
