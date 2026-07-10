"""Regression gate: chunk_fwd_kernel_o config pin is cross-triton bit-stable.

BK/BV/num_warps are bit-relevant axes of the fwd-o kernel and the stock
autotune choice flips bits across triton 3.5->3.7 (measured: 11/8.4M elements,
<=7.6e-6, k3-relevant). The pinned BK128/BV128/w4 config reproduces the
triton-3.5.1 anchor bits on both. Goldens frozen 2026-07-09 on H100 under
torch 2.9.1/triton 3.5.1 AND torch 2.12.1/triton 3.7.1 (bitwise equal).
"""

import hashlib
import os

import pytest
import torch

from xorl.ops.linear_attention.ops.common.chunk_o import _FWD_O_CONFIGS
from xorl.ops.linear_attention.ops.gated_delta_rule.chunk import chunk_gated_delta_rule


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

GOLDEN_O = "d9770c98021d38d6477af092244eec9bd8a75ec75bc70d9453cec8d83cfbab16"
GOLDEN_FS = "5c7a5f925c0c1429457dfa54ecad94a1d91a12961496d5832081fa64e40092a0"


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_fwd_o_default_config_is_pinned():
    if os.environ.get("XORL_FLA_FWD_O_AUTOTUNE", "0") == "1":
        pytest.skip("autotune escape hatch enabled")
    assert len(_FWD_O_CONFIGS) == 1
    cfg = _FWD_O_CONFIGS[0]
    assert cfg.kwargs == {"BK": 128, "BV": 128}
    assert cfg.num_warps == 4


@requires_cuda
@pytest.mark.gpu
def test_chunk_scan_output_matches_frozen_golden():
    gen = torch.Generator(device="cpu").manual_seed(20260709)
    B, T, H, K, V = 1, 512, 4, 128, 128
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
    v = (torch.randn(B, T, H, V, generator=gen) * 0.5).to(torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, generator=gen) * 2.0)
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen)).to(torch.bfloat16)
    h0 = torch.randn(2, H, K, V, generator=gen) * 0.1
    cu = torch.tensor([0, 320, 512], dtype=torch.long)

    o, fs = chunk_gated_delta_rule(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        g.cuda(),
        beta.cuda(),
        initial_state=h0.cuda(),
        output_final_state=True,
        cu_seqlens=cu.cuda(),
    )
    o2, fs2 = chunk_gated_delta_rule(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        g.cuda(),
        beta.cuda(),
        initial_state=h0.cuda(),
        output_final_state=True,
        cu_seqlens=cu.cuda(),
    )
    assert torch.equal(o, o2) and torch.equal(fs, fs2)
    assert _sha(o) == GOLDEN_O
    assert _sha(fs) == GOLDEN_FS
