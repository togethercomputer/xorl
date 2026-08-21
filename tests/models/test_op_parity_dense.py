"""Op-parity regression guard for the dense Qwen3-1.7B lockstep ops.

These are the non-aten operators that must stay bit-for-bit aligned with SGLang
for the static K3 (train/serve logprob) parity to hold: the eager RoPE apply and
the exact one-round SwiGLU. (RMSNorm is guarded by ``test_rmsnorm_sglang_fused.py``;
attention is the shared FA kernel / irreducible paged-vs-contiguous floor and is
not asserted here.) The SGLang reference is inlined verbatim from its
``forward_native`` so the test needs no SGLang install.

The K3 recipe uses the *eager* RoPE (flash rope unavailable -> naive path,
``rope_native``) and the one-round FP32 SwiGLU shared by the exact model paths.
"""

import pytest
import torch
import torch.nn.functional as F

from xorl.models.layers import rope as xrope
from xorl.ops.exact.fused_silu_and_mul import exact_fp32_silu_and_mul


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda"
DT = torch.bfloat16
HEAD_DIM = 128
N_Q_HEADS = 16
N_KV_HEADS = 8
ROPE_THETA = 1_000_000
SEQ = 96
MAX_POS = 40960


def _sg_cos_sin_cache():
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float, device=DEV) / HEAD_DIM))
    t = torch.arange(MAX_POS, dtype=torch.float, device=DEV)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # fp32 cache (SGLang keeps fp32 on CUDA)


def _sg_forward_native(positions, query, key, cache):
    """Inlined SGLang RotaryEmbedding.forward_native (is_neox_style=True)."""

    def apply(x, cos, sin):
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    cos_sin = cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    qs = query.shape
    q_rot = apply(query.view(SEQ, -1, HEAD_DIM), cos, sin).reshape(qs)
    ks = key.shape
    k_rot = apply(key.view(SEQ, -1, HEAD_DIM), cos, sin).reshape(ks)
    return q_rot, k_rot


@requires_cuda
@pytest.mark.gpu
def test_rope_xorl_eager_bit_exact_vs_sglang_native():
    torch.manual_seed(0)

    q = torch.randn(1, SEQ, N_Q_HEADS, HEAD_DIM, device=DEV, dtype=DT)
    k = torch.randn(1, SEQ, N_KV_HEADS, HEAD_DIM, device=DEV, dtype=DT)
    positions = torch.arange(SEQ, device=DEV)

    class Cfg:
        rope_scaling = None
        head_dim = HEAD_DIM
        hidden_size = N_Q_HEADS * HEAD_DIM
        num_attention_heads = N_Q_HEADS
        max_position_embeddings = MAX_POS
        rope_theta = ROPE_THETA
        rope_parameters = {}

    xrot = xrope.RotaryEmbedding(Cfg(), device=DEV)
    # K3 recipe runs the naive (eager) rope path (flash apply unavailable).
    assert xrope._flash_apply_rotary_emb is None
    cos, sin = xrot.forward(q.view(1, SEQ, -1), positions.unsqueeze(0))
    xq, xk = xrope.apply_rotary_pos_emb(q, k, cos, sin)

    cache = _sg_cos_sin_cache()
    sq = q.view(SEQ, N_Q_HEADS, HEAD_DIM).reshape(SEQ, -1).to(DT)
    sk = k.view(SEQ, N_KV_HEADS, HEAD_DIM).reshape(SEQ, -1).to(DT)
    sq_out, sk_out = _sg_forward_native(positions, sq.clone(), sk.clone(), cache)
    sq_out = sq_out.view(SEQ, N_Q_HEADS, HEAD_DIM).unsqueeze(0)
    sk_out = sk_out.view(SEQ, N_KV_HEADS, HEAD_DIM).unsqueeze(0)

    assert torch.equal(xq, sq_out), "xorl eager RoPE(Q) diverged from SGLang forward_native"
    assert torch.equal(xk, sk_out), "xorl eager RoPE(K) diverged from SGLang forward_native"


@requires_cuda
@pytest.mark.gpu
def test_swiglu_xorl_one_round_bit_exact_vs_sglang_fp32():
    torch.manual_seed(1)

    inter = 6144
    gate_up = torch.randn(SEQ, 2 * inter, device=DEV, dtype=DT)

    x_xorl_exact = exact_fp32_silu_and_mul(gate_up)
    d = gate_up.shape[-1] // 2
    x_sg = (F.silu(gate_up[..., :d].float()) * gate_up[..., d:].float()).to(DT)

    assert torch.equal(x_xorl_exact, x_sg), "xorl exact SwiGLU diverged from SGLang fp32_silu_and_mul"
