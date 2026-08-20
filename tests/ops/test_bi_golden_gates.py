"""Frozen-bits golden gates for the K3 BI contract trees (Rule B, in CI).

A/B-equality tests only catch a moved reduction tree if both engines recompile
together — venvs don't guarantee that across compiler bumps. These gates pin
the trees to frozen BYTES: deterministic integer-assembled inputs (no torch
RNG, no libm — identical on every venv) run through this repo's kernels and
must hash to the goldens recorded at the 2026-07-09 N6 re-anchor
(data/bi_golden_trees.json; serving twin checks the same file). v1 trees are
the frozen reference; v2 trees are the [PROD] contract. Goldens are
H100/sm90-specific. Regenerate with a paired trainer/serving golden capture
and update the JSON fixture in the same change.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xorl.ops import batch_invariant_ops as v1
from xorl.ops import bi_families_v2 as v2


pytestmark = pytest.mark.gpu  # explicit: the CI gate is the H100 GPU job, never `-m cpu`

GOLDENS = json.loads((Path(__file__).parent / "data/bi_golden_trees.json").read_text())["cases"]
REAL_VOCAB, REAL_H, DQK, EPS = 128256, 3840, 128, 1e-6

requires_h100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="goldens are H100/sm90-specific",
)


@pytest.fixture(autouse=True)
def _pin_triton_reference_tree(monkeypatch):
    """Keep this module's golden oracle local to each test invocation."""
    monkeypatch.setattr(v1, "_ENABLE_MM_DEEPGEMM", False)


def _splitmix64(n, seed):
    stream = (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (np.arange(1, n + 1, dtype=np.uint64) + np.uint64(stream)) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def bf16(shape, seed, exp_lo=119, exp_hi=134, edge_rows=True):
    n = int(np.prod(shape))
    r = _splitmix64(n, seed)
    sign = (r & np.uint64(1)).astype(np.uint16) << np.uint16(15)
    span = np.uint64(exp_hi - exp_lo + 1)
    exp = ((r >> np.uint64(1)) % span + np.uint64(exp_lo)).astype(np.uint16) << np.uint16(7)
    mant = ((r >> np.uint64(17)) & np.uint64(0x7F)).astype(np.uint16)
    bits = (sign | exp | mant).reshape(shape)
    if edge_rows and len(shape) == 2 and shape[0] >= 2:
        bits[0, :] = 0x3F80 | (seed % 3)
        bits[1, :] = bits[1, :] & np.uint16(0x80FF) | np.uint16(exp_hi << 7)
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).cuda()


def fp32(shape, seed, exp_lo=119, exp_hi=130):
    n = int(np.prod(shape))
    r = _splitmix64(n, seed)
    sign = (r & np.uint64(1)).astype(np.uint32) << np.uint32(31)
    span = np.uint64(exp_hi - exp_lo + 1)
    exp = ((r >> np.uint64(1)) % span + np.uint64(exp_lo)).astype(np.uint32) << np.uint32(23)
    mant = ((r >> np.uint64(9)) & np.uint64(0x7FFFFF)).astype(np.uint32)
    return torch.from_numpy((sign | exp | mant).reshape(shape).view(np.int32)).view(torch.float32).cuda()


def w(n, seed):
    return bf16((n,), seed, exp_lo=126, exp_hi=128, edge_rows=False)


def ids(n, seed, mod):
    return torch.from_numpy((_splitmix64(n, seed) % np.uint64(mod)).astype(np.int64)).cuda()


def _assert_golden(case, **outs):
    for name, t in outs.items():
        got = hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        want = GOLDENS[case][name]["sha256"]
        assert got == want, f"{case}:{name} bits moved (golden {want[:16]} != got {got[:16]})"


@requires_h100
def test_v1_contract_tree_goldens():
    _assert_golden(
        "family1_qk_4096x128",
        out=v1.bi_rms_norm(bf16((4096, DQK), 101), w(DQK, 102), EPS, family=v1.RMS_NORM_FAMILY_NO_RESIDUAL),
    )
    _assert_golden(
        "family1_h3840_m64",
        out=v1.bi_rms_norm(bf16((64, REAL_H), 111), w(REAL_H, 112), EPS, family=v1.RMS_NORM_FAMILY_NO_RESIDUAL),
    )
    _assert_golden(
        "family1_zero_centered_512x128",
        out=v1.bi_rms_norm(
            bf16((512, DQK), 121), w(DQK, 122), EPS, family=v1.RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True
        ),
    )

    _assert_v1_family2_and_mean_goldens()
    _assert_v1_log_softmax_and_mm_goldens()
    _assert_v1_lmhead_scoring_goldens()


def _assert_v1_family2_and_mean_goldens():
    out, res = v1.bi_fused_add_rms_norm(
        bf16((64, REAL_H), 131), bf16((64, REAL_H), 132), w(REAL_H, 133), EPS, family=v1.RMS_NORM_FAMILY_RESIDUAL_TREE
    )
    _assert_golden("family2_residual_m64_h3840", out=out, residual_out=res)
    out, res = v1.bi_fused_add_rms_norm(
        bf16((1, REAL_H), 141, edge_rows=False),
        bf16((1, REAL_H), 142, edge_rows=False),
        w(REAL_H, 143),
        EPS,
        family=v1.RMS_NORM_FAMILY_RESIDUAL_TREE,
    )
    _assert_golden("family2_residual_m1_h3840", out=out, residual_out=res)
    _assert_golden(
        "family2_noresidual_m64_h3840",
        out=v1.bi_rms_norm(bf16((64, REAL_H), 151), w(REAL_H, 152), EPS, family=v1.RMS_NORM_FAMILY_RESIDUAL_TREE),
    )
    _assert_golden(
        "mean_dim_m64_h3840",
        out=v1.mean_dim(fp32((64, REAL_H), 161, exp_lo=110, exp_hi=132), -1, True),
    )


def _assert_v1_log_softmax_and_mm_goldens():
    _assert_golden(
        "log_softmax_4x128256",
        out=v1.log_softmax(fp32((4, REAL_VOCAB), 171, exp_lo=124, exp_hi=131), dim=-1),
    )
    _assert_golden(
        "mm_trunk_64x3840x5120",
        out=v1.matmul_persistent(bf16((64, REAL_H), 181), bf16((REAL_H, 5120), 182, edge_rows=False)),
    )
    _assert_golden(
        "mm_offtable_17x1024x1000",
        out=v1.matmul_persistent(bf16((17, 1024), 191), bf16((1024, 1000), 192, edge_rows=False)),
    )


def _assert_v1_lmhead_scoring_goldens():
    h, wt, tk = bf16((8, REAL_H), 201), bf16((REAL_VOCAB, REAL_H), 202, edge_rows=False), ids(8, 203, REAL_VOCAB)
    lp, lse, sel = v1.bi_lm_head_selected_logprob(h, wt, tk)
    _assert_golden("lmhead_scoring_n8_real", logprob=lp, lse=lse, selected=sel)
    h, wt, tk = bf16((8, REAL_H), 211), bf16((REAL_VOCAB, REAL_H), 212, edge_rows=False), ids(8, 213, REAL_VOCAB)
    lp, lse, sel = v1.bi_lm_head_selected_logprob(h, wt, tk, torch.full((8,), 0.7, dtype=torch.float32, device="cuda"))
    _assert_golden("lmhead_scoring_n8_temp0.7", logprob=lp, lse=lse, selected=sel)
    h, wt, tk = bf16((64, 512), 221), bf16((20480, 512), 222, edge_rows=False), ids(64, 223, 20480)
    lp, lse, sel = v1.bi_lm_head_selected_logprob(h, wt, tk)
    _assert_golden("lmhead_scoring_n64_small", logprob=lp, lse=lse, selected=sel)


@requires_h100
def test_v2_contract_tree_goldens():
    out, res = v2.rms_norm_v2(bf16((64, REAL_H), 131), w(REAL_H, 133), EPS, residual=bf16((64, REAL_H), 132))
    _assert_golden("norm_v2_residual_m64_h3840", out=out, residual_out=res)
    _assert_golden(
        "norm_v2_noresidual_m64_h3840",
        out=v2.rms_norm_v2(bf16((64, REAL_H), 131), w(REAL_H, 133), EPS),
    )
    _assert_golden(
        "norm_v2_zero_centered",
        out=v2.rms_norm_v2(bf16((64, REAL_H), 131)[:, :DQK].contiguous(), w(DQK, 102), EPS, zero_centered=True),
    )
    out, res = v2.rms_norm_v2(
        bf16((64, REAL_H), 131),
        w(REAL_H, 133),
        EPS,
        residual=bf16((64, REAL_H), 132),
        zero_centered=True,
    )
    _assert_golden("norm_v2_zero_centered_residual_m64_h3840", out=out, residual_out=res)

    _assert_v2_head_goldens()


def _assert_v2_head_goldens():
    h, wt, tk = bf16((64, REAL_H), 501), bf16((REAL_VOCAB, REAL_H), 502, edge_rows=False), ids(64, 503, REAL_VOCAB)
    lp, lse, sel = v2.head_v2_selected_logprob(h, wt, tk)
    _assert_golden("head_v2_scoring_n64_real", logprob=lp, lse=lse, selected=sel)
    lp, lse, _ = v2.head_v2_selected_logprob(h, wt, tk, torch.full((64,), 0.7, dtype=torch.float32, device="cuda"))
    _assert_golden("head_v2_scoring_temp0.7", logprob=lp, lse=lse)
    logits, lse_d = v2.head_v2_full_logits_with_lse(h, wt)
    _assert_golden("head_v2_decode_n64", logits=logits, lse=lse_d)
    hs, ws2, ts = bf16((8, 512), 512), bf16((20580, 512), 511, edge_rows=False), ids(8, 513, 20580)
    lp, lse, _ = v2.head_v2_selected_logprob(hs, ws2, ts)
    _assert_golden("head_v2_small_tail", logprob=lp, lse=lse)
