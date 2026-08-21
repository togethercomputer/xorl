"""P5 GDN backend selection and contract-pin regression tests for FlashQLA.

Ports the FlashQLA certification gates 2 (M/batch-invariance) and 4
(chunk-chaining exactness through the fp32 pool-layout handoff) as regression
tests: with ``auto_cp`` pinned OFF by the exact Qwen3.5 model program, the
FlashQLA forward must be bitwise batch/M-invariant and bitwise
chain-exact. The default (heuristic) mode is intentionally not asserted
bitwise: its intra-card CP warmup h0-drop approximation is bitwise-breaking by
design, which is exactly why the contract lane pins it off.

All GPU tests run at the Qwen3.5/3.6-35B-A3B GDN contract shape: HK16/GVA32
(q/k repeated 16->32 before the kernel), dk=dv=128, chunk 64, bf16 q/k/v,
fp32 g/beta, ``use_qk_l2norm_in_kernel=True``.
"""

import warnings

import pytest
import torch

import xorl.models.layers.gated_deltanet as gated_deltanet
from xorl.models.layers.gated_deltanet import GatedDeltaNet
from xorl.ops.linear_attention.backend import FLASHQLA_AUTOCP_ENV, resolve_flashqla_auto_cp
from xorl.ops.linear_attention.modules.bi_contract import gdn_contract


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HK, HV, DK, DV = 16, 32, 128, 128
CHUNK = 64
SCALE = DK**-0.5


def _flashqla_chunk_or_skip():
    import inspect  # noqa: PLC0415

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashQLA requires a Hopper (SM90) GPU")
    import tilelang.language as _tl  # noqa: PLC0415

    if "prefer_instruction" not in inspect.signature(_tl.copy).parameters:
        pytest.skip("tilelang lacks prefer_instruction (PR #2303); FlashQLA TMA path unavailable")
    from xorl.ops._vendored.flashqla import chunk_gated_delta_rule as flashqla_chunk  # noqa: PLC0415

    return flashqla_chunk


@pytest.fixture
def armed_contract_lane(monkeypatch):
    monkeypatch.delenv(FLASHQLA_AUTOCP_ENV, raising=False)
    with gdn_contract(True):
        yield


def _make_inputs(T, *, seed, batch=1, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.bfloat16)

    q = rnd(batch, T, HK, DK).repeat_interleave(HV // HK, dim=2).contiguous()
    k = rnd(batch, T, HK, DK).repeat_interleave(HV // HK, dim=2).contiguous()
    v = rnd(batch, T, HV, DV)
    g = -torch.rand(batch, T, HV, generator=gen, device=device, dtype=torch.float32)
    beta = torch.rand(batch, T, HV, generator=gen, device=device, dtype=torch.float32)
    return q, k, v, g, beta


def _run(fn, q, k, v, g, beta, *, initial_state=None, cu_seqlens=None):
    return fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=SCALE,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )


def _make_states(fn, n, *, seed, device="cuda"):
    """Realistic fp32 checkpoints: scan one random chunk per request."""
    states = []
    for i in range(n):
        q, k, v, g, beta = _make_inputs(CHUNK, seed=seed + i, device=device)
        _, s = _run(fn, q, k, v, g, beta)
        states.append(s)
    return torch.cat(states, dim=0)


def _assert_bitwise(got, exp, label):
    assert torch.equal(got, exp), f"{label}: frac_neq={(got != exp).float().mean().item():.2e}"


@pytest.mark.cpu
def test_resolve_auto_cp_precedence(monkeypatch):
    monkeypatch.delenv(FLASHQLA_AUTOCP_ENV, raising=False)
    with gdn_contract(False):
        assert resolve_flashqla_auto_cp(None) is True  # non-contract default: heuristic stays on
    with gdn_contract(True):
        assert resolve_flashqla_auto_cp(None) is False
        assert resolve_flashqla_auto_cp(True) is False
        monkeypatch.setenv(FLASHQLA_AUTOCP_ENV, "1")
        assert resolve_flashqla_auto_cp(None) is False
    monkeypatch.setenv(FLASHQLA_AUTOCP_ENV, "0")
    with gdn_contract(False):
        assert resolve_flashqla_auto_cp(None) is False
        assert resolve_flashqla_auto_cp(False) is False

    _assert_gdn_backend_env_dispatches_to_flashqla(monkeypatch)


def _assert_gdn_backend_env_dispatches_to_flashqla(monkeypatch):
    calls = []

    def fake_flashqla_chunk(**kwargs):
        calls.append(kwargs)
        assert "cp_context" not in kwargs
        return kwargs["v"], None

    monkeypatch.setenv("XORL_GDN_BACKEND", "flashqla")
    monkeypatch.setattr(gated_deltanet, "flashqla_chunk_gated_delta_rule", fake_flashqla_chunk)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = GatedDeltaNet(
            hidden_size=128,
            expand_v=1.0,
            head_dim=128,
            num_heads=1,
            num_v_heads=1,
            mode="chunk",
            use_gate=False,
            use_short_conv=False,
        )
    layer.train()
    out, _, _ = layer(torch.randn(1, 8, 128))

    assert len(calls) == 1
    assert calls[0]["q"].shape == (1, 8, 1, 128)
    assert out.shape == (1, 8, 128)


@requires_cuda
@pytest.mark.gpu
def test_contract_lane_pins_autocp_off(monkeypatch):
    """The armed lane must skip intra_card_cp_preprocess; arg/env must restore it."""
    fn = _flashqla_chunk_or_skip()
    import xorl.ops._vendored.flashqla.ops.gated_delta_rule.chunk as chunk_mod  # noqa: PLC0415

    calls = []
    real = chunk_mod.intra_card_cp_preprocess

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(chunk_mod, "intra_card_cp_preprocess", spy)
    # bs1 T=4096: CP-eligible (Be*H=32 <= 40) so the default mode engages the preprocess
    q, k, v, g, beta = _make_inputs(4096, seed=0)

    monkeypatch.delenv(FLASHQLA_AUTOCP_ENV, raising=False)
    with gdn_contract(True):
        _run(fn, q, k, v, g, beta)
    assert not calls, "armed contract lane must pin auto_cp off"

    with gdn_contract(True):
        fn(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE, use_qk_l2norm_in_kernel=True, auto_cp=True)
    assert not calls, "exact model program must ignore an explicit auto_cp override"

    with gdn_contract(False):
        _run(fn, q, k, v, g, beta)
    assert len(calls) == 1, "default (non-contract) mode must keep the heuristic on"

    monkeypatch.setenv(FLASHQLA_AUTOCP_ENV, "0")
    _run(fn, q, k, v, g, beta)
    assert len(calls) == 1, "XORL_GDN_FLASHQLA_AUTOCP=0 must pin the heuristic off"


@requires_cuda
@pytest.mark.gpu
def test_gate2_shape_invariance_policy(armed_contract_lane):
    """Gate 2a: packed varlen rows with fp32 initial states == per-request calls, bitwise."""
    fn = _flashqla_chunk_or_skip()
    partial_lens = [1, 17, 64, 33]
    T = sum(partial_lens)
    q, k, v, g, beta = _make_inputs(T, seed=21)
    init = _make_states(fn, len(partial_lens), seed=100)

    cu = torch.tensor([0, *torch.tensor(partial_lens).cumsum(0).tolist()], device="cuda", dtype=torch.long)
    o_pack, s_pack = _run(fn, q, k, v, g, beta, initial_state=init, cu_seqlens=cu)

    for i, L in enumerate(partial_lens):
        lo, hi = int(cu[i]), int(cu[i + 1])
        o_i, s_i = _run(
            fn, q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], initial_state=init[i : i + 1]
        )
        _assert_bitwise(o_pack[:, lo:hi], o_i, f"row {i} (len {L}) out")
        _assert_bitwise(s_pack[i : i + 1], s_i, f"row {i} (len {L}) state")

    _assert_gate2_same_row_bits_invariant_to_total_m()
    _assert_gate2_block_dv_tile_heuristic_bit_invariant()


def _assert_gate2_same_row_bits_invariant_to_total_m():
    """Gate 2b: the same row alone (CP-eligible bs1) vs packed 4xT — bitwise under the pin."""
    fn = _flashqla_chunk_or_skip()
    for T in (2048, 4096):
        q, k, v, g, beta = _make_inputs(4 * T, seed=22)

        o_alone, s_alone = _run(fn, q[:, :T], k[:, :T], v[:, :T], g[:, :T], beta[:, :T])

        cu = torch.arange(0, 4 * T + 1, T, device="cuda", dtype=torch.long)
        o_pack, s_pack = _run(fn, q, k, v, g, beta, cu_seqlens=cu)

        _assert_bitwise(o_pack[:, :T], o_alone, f"T={T} out")
        _assert_bitwise(s_pack[0:1], s_alone, f"T={T} state")


def _assert_gate2_block_dv_tile_heuristic_bit_invariant():
    """Gate 2c: identical row 0 at B=1/2/3 (block_DV tile heuristic 32/64/128) — bitwise."""
    fn = _flashqla_chunk_or_skip()
    q, k, v, g, beta = _make_inputs(1024, seed=24, batch=3)
    outs = {}
    for B in (1, 2, 3):
        o, s = _run(fn, q[:B], k[:B], v[:B], g[:B], beta[:B])
        outs[B] = (o[0:1], s[0:1])
    for a, b in ((1, 2), (1, 3), (2, 3)):
        _assert_bitwise(outs[a][0], outs[b][0], f"B{a} vs B{b} out")
        _assert_bitwise(outs[a][1], outs[b][1], f"B{a} vs B{b} state")


@requires_cuda
@pytest.mark.gpu
def test_gate4_chunk_chaining_bitwise_through_pool_layout(armed_contract_lane):
    """Gate 4: one call == chained calls with fp32 state handoff through the sglang
    pool layout ([N, HV, V, K] transpose round-trip) — the recompute-decode prerequisite."""
    fn = _flashqla_chunk_or_skip()
    for T, step in ((256, CHUNK), (4096, CHUNK), (4096, 256)):
        q, k, v, g, beta = _make_inputs(T, seed=42)

        o_ref, s_ref = _run(fn, q, k, v, g, beta)

        pool = None
        outs = []
        for t0 in range(0, T, step):
            sl = slice(t0, min(t0 + step, T))
            init = pool.transpose(-1, -2).contiguous() if pool is not None else None
            o, s = _run(fn, q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=init)
            pool = s.transpose(-1, -2).contiguous()
            outs.append(o)
        o_chain = torch.cat(outs, dim=1)

        _assert_bitwise(o_ref, o_chain, f"T={T} step={step} out")
        _assert_bitwise(s_ref.transpose(-1, -2).contiguous(), pool, f"T={T} step={step} state")
