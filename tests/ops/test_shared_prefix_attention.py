"""GPU equivalence tests for shared-prefix attention vs dense causal attention.

The shared-prefix backend (duplicate-boundary layout: shared prefix block
``[p_0..p_{P-2}]`` + per-member decoded block ``[p_{P-1}, r_0, ...]``) must
reproduce, up to fp tolerance, standard causal attention over the
un-deduplicated ``N*(P+R)`` layout — forward and gradients (including the prefix
K/V gradient, which accumulates over all N members).
"""

import itertools

import pytest
import torch


pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("shared-prefix attention requires CUDA", allow_module_level=True)

from flash_attn_interface import flash_attn_varlen_func  # noqa: E402

from xorl.models.layers.attention.backend.shared_prefix_attention import (  # noqa: E402
    shared_prefix_attention_forward,
)
from xorl.ops.shared_prefix.repack import SharedPrefixContext  # noqa: E402


def _single_group_ctx(P, Rs, device):
    """Context for one group: shared prefix [P-1] + per-member [R_i + 1] blocks."""
    nshared = P - 1
    blocks = [r + 1 for r in Rs]
    num_dec = sum(blocks)
    rep_len = nshared + num_dec
    full = torch.arange(num_dec, device=device)
    return SharedPrefixContext(
        shared_idx=torch.arange(nshared, device=device),
        dec_idx=torch.arange(nshared, rep_len, device=device),
        cu_shared=torch.tensor([0, nshared], dtype=torch.int32, device=device),
        max_shared=nshared,
        cross_local_idx=full if nshared > 0 else torch.empty(0, dtype=torch.long, device=device),
        cu_cross_q=torch.tensor([0, num_dec], dtype=torch.int32, device=device),
        max_cross_q=num_dec,
        cu_dec=torch.tensor([0, *itertools.accumulate(blocks)], dtype=torch.int32, device=device),
        max_dec=max(blocks),
        dec_orig_idx=full,  # unused by attention
        repacked_len=rep_len,
        orig_len=rep_len,
        num_groups=1,
        has_shared_prefix=True,
        orig_position_ids=torch.arange(rep_len, device=device).unsqueeze(0),
    )


def _repacked(t_prefix, t_resp, P, Rs):
    """Build a repacked [shared, per-member [boundary, resp]] tensor."""
    offs = [0, *itertools.accumulate(Rs)]
    parts = [t_prefix[: P - 1]]
    for i in range(len(Rs)):
        parts.append(t_prefix[P - 1 : P])
        parts.append(t_resp[offs[i] : offs[i + 1]])
    return torch.cat(parts)


def _reference_dense(qp, kp, vp, qd, kd, vd, Rs, scale):
    """Standard causal attention over [prefix, resp_i] per member; prefix reused."""
    P = qp.size(0)
    offs = [0, *itertools.accumulate(Rs)]
    q_parts, k_parts, v_parts, cu = [], [], [], [0]
    for i, r in enumerate(Rs):
        sl = slice(offs[i], offs[i + 1])
        q_parts += [qp, qd[sl]]
        k_parts += [kp, kd[sl]]
        v_parts += [vp, vd[sl]]
        cu.append(cu[-1] + P + r)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=qp.device)
    m = max(P + r for r in Rs)
    out = flash_attn_varlen_func(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu_t,
        cu_seqlens_k=cu_t,
        max_seqlen_q=m,
        max_seqlen_k=m,
        softmax_scale=scale,
        causal=True,
    )
    # decoded outputs per member = reference positions [P-1 : P+R_i] (boundary + responses)
    dec_idx = []
    for i, r in enumerate(Rs):
        base = cu[i]
        dec_idx += list(range(base + P - 1, base + P + r))
    return out.index_select(0, torch.tensor(dec_idx, device=qp.device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2)])
def test_shared_prefix_matches_dense_fwd_bwd(dtype, head_dim, hq, hkv):
    torch.manual_seed(0)
    device = "cuda"
    P, Rs = 48, [16, 8, 24]
    scale = head_dim**-0.5
    tol = 1e-2 if dtype == torch.float16 else 2e-2

    def leaf(n, h):
        return torch.randn(n, h, head_dim, device=device, dtype=dtype, requires_grad=True)

    qp, kp, vp = leaf(P, hq), leaf(P, hkv), leaf(P, hkv)
    qd, kd, vd = leaf(sum(Rs), hq), leaf(sum(Rs), hkv), leaf(sum(Rs), hkv)
    leaves = [qp, kp, vp, qd, kd, vd]

    ref_dec = _reference_dense(qp, kp, vp, qd, kd, vd, Rs, scale)

    ctx = _single_group_ctx(P, Rs, device)
    q_rep = _repacked(qp, qd, P, Rs).unsqueeze(0)
    k_rep = _repacked(kp, kd, P, Rs).unsqueeze(0)
    v_rep = _repacked(vp, vd, P, Rs).unsqueeze(0)
    out_rep, _ = shared_prefix_attention_forward(
        None, q_rep, k_rep, v_rep, None, scaling=scale, shared_prefix_context=ctx
    )
    sp_dec = out_rep.squeeze(0)[ctx.dec_idx]

    torch.testing.assert_close(sp_dec, ref_dec, atol=tol, rtol=tol)

    seed = torch.randn_like(ref_dec)
    g_ref = torch.autograd.grad((ref_dec * seed).sum(), leaves, allow_unused=True)
    g_sp = torch.autograd.grad((sp_dec * seed).sum(), leaves, allow_unused=True)
    for name, a, b in zip(["qp", "kp", "vp", "qd", "kd", "vd"], g_ref, g_sp):
        torch.testing.assert_close(b, a, atol=tol, rtol=tol, msg=f"grad mismatch: {name}")


def test_shared_prefix_singleton_equals_plain_causal():
    """A 1-member group reduces to ordinary causal attention over [prefix, resp]."""
    torch.manual_seed(1)
    device, dtype, head_dim, h = "cuda", torch.bfloat16, 64, 4
    P, R = 32, 20
    scale = head_dim**-0.5

    qp = torch.randn(P, h, head_dim, device=device, dtype=dtype)
    kp = torch.randn(P, h, head_dim, device=device, dtype=dtype)
    vp = torch.randn(P, h, head_dim, device=device, dtype=dtype)
    qd = torch.randn(R, h, head_dim, device=device, dtype=dtype)
    kd = torch.randn(R, h, head_dim, device=device, dtype=dtype)
    vd = torch.randn(R, h, head_dim, device=device, dtype=dtype)

    ref = _reference_dense(qp, kp, vp, qd, kd, vd, [R], scale)

    ctx = _single_group_ctx(P, [R], device)
    out, _ = shared_prefix_attention_forward(
        None,
        _repacked(qp, qd, P, [R]).unsqueeze(0),
        _repacked(kp, kd, P, [R]).unsqueeze(0),
        _repacked(vp, vd, P, [R]).unsqueeze(0),
        None,
        scaling=scale,
        shared_prefix_context=ctx,
    )
    torch.testing.assert_close(out.squeeze(0)[ctx.dec_idx], ref, atol=2e-2, rtol=2e-2)


def test_shared_prefix_p_equals_one_no_cross():
    """P==1 group (empty shared block) uses decoded self-attn only."""
    torch.manual_seed(2)
    device, dtype, head_dim, h = "cuda", torch.bfloat16, 64, 4
    P, Rs = 1, [12, 9]
    scale = head_dim**-0.5

    qp = torch.randn(P, h, head_dim, device=device, dtype=dtype)
    kp = torch.randn(P, h, head_dim, device=device, dtype=dtype)
    vp = torch.randn(P, h, head_dim, device=device, dtype=dtype)
    qd = torch.randn(sum(Rs), h, head_dim, device=device, dtype=dtype)
    kd = torch.randn(sum(Rs), h, head_dim, device=device, dtype=dtype)
    vd = torch.randn(sum(Rs), h, head_dim, device=device, dtype=dtype)

    ref = _reference_dense(qp, kp, vp, qd, kd, vd, Rs, scale)
    ctx = _single_group_ctx(P, Rs, device)
    assert ctx.shared_idx.numel() == 0 and ctx.cross_local_idx.numel() == 0
    out, _ = shared_prefix_attention_forward(
        None,
        _repacked(qp, qd, P, Rs).unsqueeze(0),
        _repacked(kp, kd, P, Rs).unsqueeze(0),
        _repacked(vp, vd, P, Rs).unsqueeze(0),
        None,
        scaling=scale,
        shared_prefix_context=ctx,
    )
    torch.testing.assert_close(out.squeeze(0)[ctx.dec_idx], ref, atol=2e-2, rtol=2e-2)
