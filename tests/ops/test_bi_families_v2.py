"""Vendoring gates for the families-v2 contract module, trainer side.

Serving twin: sglang test/registered/rl/test_bi_families_v2.py, which pins the
same digest. ``bi_families_v2.py`` is vendored byte-identical into both engines
because the trainer and the sampler must evaluate the same reduction trees; if
the two copies can drift, that guarantee is only a convention.

Single-repo CI has no sibling checkout, so each suite pins the file's sha256.
Editing either copy reddens that repo's gate until the digest is re-pinned in
both suites, which is exactly the byte-equality contract.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from xorl.ops import bi_families_v2 as v2
from xorl.ops.batch_invariant_ops import bi_lm_head_selected_logprob


BI_FAMILIES_V2_SHA256 = "fd4c5bac2a52d2148b8e4d0e9afa4e46e8c62689a68c2bc0e309f671597799e6"

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


def test_vendored_module_matches_pinned_digest():
    ours = hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest()
    assert ours == BI_FAMILIES_V2_SHA256, (
        f"vendored bi_families_v2.py changed (sha256 {ours}); if that is intended, re-pin "
        f"BI_FAMILIES_V2_SHA256 in xorl tests/ops/test_bi_families_v2.py AND in the serving "
        f"engine's test/registered/rl/test_bi_families_v2.py, and land both together"
    )


def test_vendored_copies_are_byte_identical_if_sibling_present():
    """Strictly stronger than the digest gate, but needs both trees present."""
    sibling = os.environ.get("BI_FAMILIES_V2_SIBLING")
    if not sibling or not Path(sibling).exists():
        pytest.skip("sibling engine checkout not available (set BI_FAMILIES_V2_SIBLING)")
    ours = hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest()
    theirs = hashlib.sha256(Path(sibling).read_bytes()).hexdigest()
    assert ours == theirs, "vendored bi_families_v2.py copies drifted"


def test_kill_switch_is_a_paired_rollback():
    """Either engine's variable rolls this engine back, so one setting applied
    to both moves both. A setting that moved only one would put the trainer and
    the sampler on different trees."""
    saved = {name: os.environ.get(name) for name in v2.FAMILIES_V2_ENV_VARS}
    try:
        for name in v2.FAMILIES_V2_ENV_VARS:
            os.environ.pop(name, None)
        assert v2.families_v2_enabled() is True, "families v2 is default on"
        for name in v2.FAMILIES_V2_ENV_VARS:
            os.environ[name] = "0"
            assert v2.families_v2_enabled() is False, f"{name}=0 must roll back"
            del os.environ[name]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
