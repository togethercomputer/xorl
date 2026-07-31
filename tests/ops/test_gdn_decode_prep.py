"""Bitwise gates for the decode-shaped GDN prep kernels (P5).

The decode-scheduled triangular solve must be bit-identical to the pinned
(num_warps=2) ``solve_tril`` — the frozen prefill-contract kernel — and the
opt-in recompute-decode composition must reproduce the pinned FlashQLA prefill
bitwise through per-step partial-chunk recompute.
"""

import pytest
import torch
import torch.nn.functional as F


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FlashQLA is SM90-only",
)

HK, HV, DK, DV = 16, 32, 128, 128
CHUNK = 64
SCALE = DK**-0.5


def _decode_shape_A(batch: int, seed: int, regime: str = "normal"):
    """A = chunk_scaled_dot_kkt output on l2normed k at the padded decode shape."""
    from xorl.ops.linear_attention.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = batch * CHUNK
    k = torch.randn(1, total, HV, DK, generator=gen, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, total, HV, generator=gen, device="cuda", dtype=torch.float32)
    if regime == "k_large":
        k = (k.float() * 100).to(torch.bfloat16)
    elif regime == "padded":
        k = k.view(1, batch, CHUNK, HV, DK).clone()
        k[:, :, 37:] = 0
        k = k.view(1, total, HV, DK)
        beta = beta.view(1, batch, CHUNK, HV).clone()
        beta[:, :, 37:] = 0
        beta = beta.view(1, total, HV)
    k = F.normalize(k.float(), dim=-1).to(torch.bfloat16)
    cu = torch.arange(0, total + 1, CHUNK, device="cuda", dtype=torch.long)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=None, beta=beta, cu_seqlens=cu, output_dtype=torch.float32)
    return A, cu


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("batch", [1, 16, 64])
@pytest.mark.parametrize("regime", ["normal", "k_large", "padded"])
def test_solve_tril_decode_bitwise_vs_pinned(batch, regime):
    from xorl.ops.linear_attention.ops.utils import solve_tril
    from xorl.ops.linear_attention.ops.utils.solve_tril_decode import solve_tril_decode

    A, cu = _decode_shape_A(batch, seed=batch + 17, regime=regime)
    ref = solve_tril(A=A, cu_seqlens=cu, output_dtype=torch.bfloat16)
    got = solve_tril_decode(A=A, cu_seqlens=cu, output_dtype=torch.bfloat16)
    assert torch.equal(ref, got)


@requires_cuda
@pytest.mark.gpu
def test_solve_tril_decode_bitwise_non_varlen_and_partial_chunks():
    from xorl.ops.linear_attention.ops.utils import solve_tril
    from xorl.ops.linear_attention.ops.utils.solve_tril_decode import solve_tril_decode

    A, _ = _decode_shape_A(64, seed=3)
    Ab = A.view(64, CHUNK, HV, CHUNK).contiguous()
    assert torch.equal(
        solve_tril(A=Ab, cu_seqlens=None, output_dtype=torch.bfloat16),
        solve_tril_decode(A=Ab, cu_seqlens=None, output_dtype=torch.bfloat16),
    )
    cu = torch.tensor([0, 64, 257, 450, 707, 1000], device="cuda", dtype=torch.long)
    Ap = A[:, :1000].contiguous()
    assert torch.equal(
        solve_tril(A=Ap, cu_seqlens=cu, output_dtype=torch.bfloat16),
        solve_tril_decode(A=Ap, cu_seqlens=cu, output_dtype=torch.bfloat16),
    )


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("diag_group", [1, 4, 16])
@pytest.mark.parametrize("diag_warps", [2, 8])
@pytest.mark.parametrize("merge_warps", [2, 8])
def test_solve_tril_decode_launch_config_invariance(diag_group, diag_warps, merge_warps):
    # The reduction tree is spelled out structurally (fma + explicit adds), so
    # bits must not move with the launch config; this pins that property.
    from xorl.ops.linear_attention.ops.utils import solve_tril
    from xorl.ops.linear_attention.ops.utils import solve_tril_decode as mod
    from xorl.ops.linear_attention.ops.utils.index import prepare_chunk_indices

    A, cu = _decode_shape_A(64, seed=29)
    ref = solve_tril(A=A, cu_seqlens=cu, output_dtype=torch.bfloat16)
    ci = prepare_chunk_indices(cu, CHUNK)
    B, T, H, BT = A.shape
    Ai = torch.empty_like(A, dtype=torch.bfloat16)
    Di = torch.empty(B, T, H, 16, dtype=torch.float32, device="cuda")
    mod.solve_tril_64x64_diag_inv_grouped_kernel[len(ci) * 4, B * (H // diag_group)](
        A=A,
        Di=Di,
        cu_seqlens=cu,
        chunk_indices=ci,
        T=T,
        H=H,
        BT=BT,
        G=diag_group,
        IS_VARLEN=True,
        num_warps=diag_warps,
        num_stages=1,
    )
    mod.solve_tril_64x64_merge_inv_kernel[len(ci), B * H](
        A=A,
        Di=Di,
        Ai=Ai,
        cu_seqlens=cu,
        chunk_indices=ci,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
        IS_VARLEN=True,
        num_warps=merge_warps,
        num_stages=1,
    )
    assert torch.equal(ref, Ai)


def _make_inputs(T: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16)

    q = rnd(1, T, HK, DK).repeat_interleave(HV // HK, dim=2).contiguous()
    k = rnd(1, T, HK, DK).repeat_interleave(HV // HK, dim=2).contiguous()
    v = rnd(1, T, HV, DV)
    a_in = rnd(1, T, HV)
    b_in = rnd(1, T, HV)
    A_log = torch.empty(HV, device="cuda", dtype=torch.float32).uniform_(0, 2, generator=gen).log()
    dt_bias = torch.rand(HV, device="cuda", dtype=torch.float32, generator=gen)
    g = -A_log.exp().view(1, 1, -1) * F.softplus(a_in.float() + dt_bias.view(1, 1, -1))
    beta = b_in.float().sigmoid().to(torch.bfloat16).float()
    return q, k, v, g, beta


def _pinned_prefill(q, k, v, g, beta, initial_state=None, cu_seqlens=None):
    from xorl.ops.linear_attention import tilelang_gemm_v1

    tilelang_gemm_v1.patch()
    from xorl.ops.linear_attention.flashqla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from xorl.ops.linear_attention.flashqla.utils import l2norm

    q = l2norm(q)
    k = l2norm(k)
    _, _, o, _, final_state = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=SCALE,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=False,
        auto_cp=False,
    )
    return o.to(q.dtype), final_state


@requires_hopper
@pytest.mark.gpu
def test_recompute_decode_bitwise_vs_pinned_prefill():
    # gate-5 protocol at T=193: padded fixed-shape per-step recompute from fp32
    # chunk checkpoints through the decode prep path reproduces the pinned
    # prefill bitwise.
    from xorl.ops.linear_attention.gdn_decode_prep import chunk_gated_delta_rule_fwd_decode

    T = 193
    q, k, v, g, beta = _make_inputs(T, seed=51)
    o_ref, _ = _pinned_prefill(q, k, v, g, beta)

    checkpoint = None
    outs = []
    for t in range(T):
        t0 = (t // CHUNK) * CHUNK
        L = t + 1 - t0
        pad = CHUNK - L
        qp = F.pad(q[:, t0 : t + 1], (0, 0, 0, 0, 0, pad))
        kp = F.pad(k[:, t0 : t + 1], (0, 0, 0, 0, 0, pad))
        vp = F.pad(v[:, t0 : t + 1], (0, 0, 0, 0, 0, pad))
        gp = F.pad(g[:, t0 : t + 1], (0, 0, 0, pad))
        bp = F.pad(beta[:, t0 : t + 1], (0, 0, 0, pad))
        init = checkpoint.transpose(-1, -2).contiguous() if checkpoint is not None else None
        o, s = chunk_gated_delta_rule_fwd_decode(qp, kp, vp, gp, bp, scale=SCALE, initial_state=init)
        outs.append(o[:, L - 1 : L])
        if (t + 1) % CHUNK == 0:
            checkpoint = s.transpose(-1, -2).contiguous()
    o_dec = torch.cat(outs, dim=1)
    assert torch.equal(o_ref, o_dec)


@requires_hopper
@pytest.mark.gpu
def test_padded_decode_call_bitwise_and_graph_capturable():
    from xorl.ops.linear_attention.gdn_decode_prep import chunk_gated_delta_rule_fwd_decode
    from xorl.ops.linear_attention.ops.utils.index import prepare_chunk_indices

    batch = 16
    gen = torch.Generator(device="cuda").manual_seed(7)
    total = batch * CHUNK
    q = torch.randn(1, total, HV, DK, generator=gen, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, HV, DK, generator=gen, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, total, HV, DV, generator=gen, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, total, HV, generator=gen, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, total, HV, generator=gen, device="cuda", dtype=torch.float32)
    init = torch.randn(batch, HV, DK, DV, generator=gen, device="cuda", dtype=torch.float32)
    cu = torch.arange(0, total + 1, CHUNK, device="cuda", dtype=torch.long)
    ci = prepare_chunk_indices(cu, CHUNK)

    with torch.no_grad():
        o_pin, s_pin = _pinned_prefill(q, k, v, g, beta, initial_state=init, cu_seqlens=cu)

        def fn():
            return chunk_gated_delta_rule_fwd_decode(
                q, k, v, g, beta, scale=SCALE, initial_state=init, cu_seqlens=cu, chunk_indices=ci
            )

        o_dec, s_dec = fn()
        assert torch.equal(o_pin, o_dec)
        assert torch.equal(s_pin, s_dec)

        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o_g, s_g = fn()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(o_pin, o_g)
        assert torch.equal(s_pin, s_g)
