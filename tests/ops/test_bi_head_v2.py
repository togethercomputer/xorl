"""Focused gates for the one-launch final token-probability contract."""

import pytest
import torch

from xorl.ops import batch_invariant_ops as v1
from xorl.ops import bi_families_v2 as v2
from xorl.ops.loss.bi_fused_lm_head import bi_fused_per_token_ce


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16).cuda()


@requires_cuda
@pytest.mark.gpu
def test_head_v2_keeps_projection_bits_and_one_stats_tree():
    hidden, weight = _make((16, 128), 1), _make((512, 128), 2)
    tokens = torch.arange(16, device="cuda") * 29 % 512

    logits, decode_lse = v2.head_v2_full_logits_with_lse(hidden, weight)
    reference_logits = torch.empty_like(logits)
    v1._bi_lm_head_chunk_gemm_fp32(hidden, weight.t(), reference_logits)
    assert torch.equal(logits, reference_logits)

    logprob, score_lse, selected = v2.head_v2_selected_logprob(hidden, weight, tokens)
    assert torch.equal(decode_lse, score_lse)
    assert torch.equal(selected, logits.gather(1, tokens[:, None]).squeeze(1))
    assert torch.equal(logprob, torch.clamp_max(selected - decode_lse, 0.0))


@requires_cuda
@pytest.mark.gpu
def test_head_v2_is_batch_invariant_and_trainable(monkeypatch):
    hidden, weight = _make((32, 128), 3), _make((512, 128), 4)
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

    monkeypatch.setenv("XORL_BI_HEAD_V2", "0")
    rollback = bi_fused_per_token_ce(hidden.detach(), weight.detach(), tokens, vocab_chunk=256)
    assert torch.isfinite(rollback).all()
