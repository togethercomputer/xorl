"""Functional tests for the trainer's families-v2 reduction trees."""

import os
from pathlib import Path

import pytest
import torch

from xorl.ops import bi_families_v2 as v2
from xorl.ops.batch_invariant_ops import bi_lm_head_selected_logprob


H, V = 1024, 20480

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _payload(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16).cuda()


def test_vendored_module_is_self_contained():
    """No engine imports: the same bytes have to run under either engine."""
    source = Path(v2.__file__).read_text()
    for banned in ("import xorl", "from xorl", "import sglang", "from sglang"):
        assert banned not in source, f"engine import {banned!r} in the vendored module"


def test_nonexact_family_selection_preserves_legacy_rollback():
    saved = {name: os.environ.get(name) for name in v2.FAMILIES_V2_ENV_VARS}
    selected = v2._EXACT_FAMILIES_VERSION
    try:
        v2._EXACT_FAMILIES_VERSION = None
        for name in v2.FAMILIES_V2_ENV_VARS:
            os.environ.pop(name, None)
        assert v2.families_v2_enabled() is True
        for name in v2.FAMILIES_V2_ENV_VARS:
            os.environ[name] = "0"
            assert v2.families_v2_enabled() is False
            del os.environ[name]
    finally:
        v2._EXACT_FAMILIES_VERSION = selected
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_exact_family_selection_ignores_legacy_rollback(monkeypatch):
    selected = v2._EXACT_FAMILIES_VERSION
    try:
        monkeypatch.setenv("XORL_FAMILIES_V2", "0")
        v2._select_glm52_families_v2()
        assert v2.families_v2_enabled() is True

        monkeypatch.setenv("XORL_FAMILIES_V2", "1")
        v2._select_qwen35_families_v1()
        assert v2.families_v2_enabled() is False
    finally:
        v2._EXACT_FAMILIES_VERSION = selected


@requires_cuda
@pytest.mark.gpu
def test_head_v2_selected_logit_is_bitwise_equal_to_v1():
    """The head redefinition keeps v1's pinned GEMM K chain, so the selected
    logit is bitwise equal to v1; the vocabulary statistics layout is what
    changed. The log-sum-exp is NOT claimed equal across the two generations —
    a v2 trainer must be paired with a v2 sampler."""
    hidden, weight = _payload((16, H), 9), _payload((V, H), 10)
    tokens = torch.arange(16, device="cuda") * 733 % V
    logits, lse_decode = v2.head_v2_full_logits_with_lse(hidden, weight)
    logprob, lse_scoring, selected = v2.head_v2_selected_logprob(hidden, weight, tokens)

    assert torch.equal(lse_decode, lse_scoring), "decode and scoring must share one tree"
    assert torch.equal(selected, logits.gather(1, tokens[:, None]).squeeze(1))
    _, _, selected_v1 = bi_lm_head_selected_logprob(hidden, weight, tokens)
    assert torch.equal(selected, selected_v1)
    expected = torch.clamp_max(logits.gather(1, tokens[:, None]).squeeze(1) - lse_decode, 0.0)
    assert torch.equal(expected, logprob)


@requires_cuda
@pytest.mark.gpu
def test_head_v2_is_batch_composition_invariant():
    hidden, weight = _payload((32, H), 11), _payload((V, H), 12)
    tokens = torch.arange(32, device="cuda") * 601 % V
    logprob, lse, _ = v2.head_v2_selected_logprob(hidden, weight, tokens)
    for rows in (1, 4):
        lp, ls, _ = v2.head_v2_selected_logprob(hidden[:rows].contiguous(), weight, tokens[:rows].contiguous())
        assert torch.equal(lp, logprob[:rows]) and torch.equal(ls, lse[:rows])
