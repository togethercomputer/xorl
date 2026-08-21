"""Contracts for DSV4's literal serving-shaped exact attention forward."""

from __future__ import annotations

import sys
import types

import pytest
import torch


def _exact_attention_module():
    from xorl.ops.families.dsv4 import exact_attention  # noqa: PLC0415

    return exact_attention


def _cache_state(num_tokens: int, ratio: int):
    exact_attention = _exact_attention_module()
    state = exact_attention.Dsv4DecodeCarryState()
    state.kvcache = exact_attention._ensure_paged_kvcache(None, num_tokens, 128, torch.device("cpu"))
    state.num_tokens = num_tokens
    if ratio:
        state.num_compressed = num_tokens // ratio
        state.compressed_kvcache = exact_attention._ensure_paged_kvcache(
            None,
            state.num_compressed,
            256 // ratio,
            torch.device("cpu"),
        )
    return state


def _install_fake_flash_mla(monkeypatch, implementation):
    package = types.ModuleType("sgl_kernel")
    module = types.ModuleType("sgl_kernel.flash_mla")
    module.flash_mla_with_kvcache = implementation
    module.get_mla_metadata = lambda: (object(), None)
    package.flash_mla = module
    monkeypatch.setitem(sys.modules, "sgl_kernel", package)
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_mla", module)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("ratio", "num_tokens", "positions", "expected_blocks"),
    [
        (0, 130, [0, 4, 128], None),
        (4, 12, [2, 3, 7], [0, 1, 2]),
        (128, 130, [126, 127, 128], [0, 1, 1]),
    ],
)
def test_serving_rows_dispatch_literal_m1_with_causal_metadata(
    monkeypatch,
    ratio,
    num_tokens,
    positions,
    expected_blocks,
):
    exact_attention = _exact_attention_module()
    calls = []

    def fake_flash_mla_with_kvcache(**kwargs):
        calls.append(
            {
                name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
        )
        return kwargs["q"].clone(), torch.empty(0)

    _install_fake_flash_mla(monkeypatch, fake_flash_mla_with_kvcache)
    state = _cache_state(num_tokens, ratio)
    q = torch.arange(len(positions), dtype=torch.bfloat16).view(1, -1, 1, 1).expand(-1, -1, 64, 512)
    positions_tensor = torch.tensor(positions, dtype=torch.int64)

    output = exact_attention._serving_attention_rows(
        q,
        state,
        positions_tensor,
        ratio,
        torch.zeros(64),
        512**-0.5,
    )

    assert torch.equal(output, q)
    assert len(calls) == len(positions)
    for row, (position, call) in enumerate(zip(positions, calls)):
        assert call["q"].shape == (1, 1, 64, 512)
        assert torch.equal(call["q"], q[:, row : row + 1])
        swa_length = min(position + 1, 128)
        expected_swa = list(range(position, position - swa_length, -1))
        actual_swa = call["indices"][0, 0]
        assert actual_swa[:swa_length].tolist() == expected_swa
        assert (actual_swa[swa_length:] == -1).all()
        assert call["topk_length"].tolist() == [swa_length]
        if ratio:
            blocks = expected_blocks[row]
            actual_extra = call["extra_indices_in_kvcache"][0, 0]
            assert call["extra_indices_in_kvcache"].shape == (1, 1, 512 if ratio == 4 else 64)
            assert actual_extra[:blocks].tolist() == list(range(blocks))
            assert (actual_extra[blocks:] == -1).all()
            assert call["extra_topk_length"].tolist() == [max(blocks, 1)]
        else:
            assert "extra_indices_in_kvcache" not in call


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("ratio", "num_tokens", "position"),
    [(0, 8, 4), (4, 12, 3), (128, 256, 127)],
)
def test_serving_row_excludes_future_raw_and_compressed_cache_bytes(monkeypatch, ratio, num_tokens, position):
    exact_attention = _exact_attention_module()

    def fake_flash_mla_with_kvcache(**kwargs):
        raw_indices = kwargs["indices"].reshape(-1)
        raw_indices = raw_indices[raw_indices >= 0].to(torch.int64)
        raw_pages = kwargs["k_cache"]
        value = raw_pages[
            raw_indices[0] // raw_pages.shape[1],
            raw_indices[0] % raw_pages.shape[1],
            0,
            0,
        ].float()
        if kwargs.get("extra_k_cache") is not None:
            extra_indices = kwargs["extra_indices_in_kvcache"].reshape(-1)
            extra_indices = extra_indices[extra_indices >= 0].to(torch.int64)
            if extra_indices.numel():
                extra_pages = kwargs["extra_k_cache"]
                value = (
                    value
                    + extra_pages[
                        extra_indices // extra_pages.shape[1],
                        extra_indices % extra_pages.shape[1],
                        0,
                        0,
                    ]
                    .float()
                    .sum()
                )
        return kwargs["q"] * 0 + value.to(kwargs["q"].dtype), torch.empty(0)

    _install_fake_flash_mla(monkeypatch, fake_flash_mla_with_kvcache)
    state = _cache_state(num_tokens, ratio)
    raw = exact_attention._paged_cache_kernel_view(state.kvcache, 128)
    raw_values = torch.arange(1, num_tokens + 1, dtype=torch.int64).remainder(251).to(torch.uint8)
    raw[
        torch.arange(num_tokens) // 128,
        torch.arange(num_tokens) % 128,
        0,
        0,
    ] = raw_values
    if ratio:
        extra_page_size = 256 // ratio
        extra = exact_attention._paged_cache_kernel_view(state.compressed_kvcache, extra_page_size)
        extra_locs = torch.arange(state.num_compressed)
        extra[
            extra_locs // extra_page_size,
            extra_locs % extra_page_size,
            0,
            0,
        ] = torch.arange(1, state.num_compressed + 1, dtype=torch.uint8)

    q = torch.zeros((1, 1, 64, 512), dtype=torch.bfloat16)
    before = exact_attention._serving_decode_attention(q, state, position, ratio, torch.zeros(64), 512**-0.5)
    future_raw = torch.arange(position + 1, num_tokens)
    raw[future_raw // 128, future_raw % 128, 0, 0] = 251
    if ratio:
        first_future_block = (position + 1) // ratio
        future_extra = torch.arange(first_future_block, state.num_compressed)
        extra[
            future_extra // extra_page_size,
            future_extra % extra_page_size,
            0,
            0,
        ] = 253
    after_future_mutation = exact_attention._serving_decode_attention(
        q,
        state,
        position,
        ratio,
        torch.zeros(64),
        512**-0.5,
    )
    assert torch.equal(before, after_future_mutation)

    page, slot = divmod(position, 128)
    raw[page, slot, 0, 0] = (int(raw[page, slot, 0, 0]) + 17) % 251
    after_visible_mutation = exact_attention._serving_decode_attention(
        q,
        state,
        position,
        ratio,
        torch.zeros(64),
        512**-0.5,
    )
    assert not torch.equal(before, after_visible_mutation)


@pytest.mark.cpu
@pytest.mark.parametrize("ratio", [4, 128])
def test_full_hybrid_over_128_materializes_each_cache_once(monkeypatch, ratio):
    exact_attention = _exact_attention_module()
    calls = {"raw_store": 0, "compressed_store": 0, "attention": 0}

    def fake_raw_store(kv, _weight, _freqs, _eps, state, offset, *, dequantize):
        calls["raw_store"] += 1
        assert offset == 0
        assert dequantize is False
        state.num_tokens = kv.shape[1]
        return kv[0]

    def fake_compressed_store(x, *_args, carry_state, dequantize):
        calls["compressed_store"] += 1
        assert dequantize is False
        observed_ratio = _args[-1]
        assert observed_ratio == ratio
        carry_state.num_compressed = x.shape[1] // ratio
        return x.new_zeros((carry_state.num_compressed, 512))

    def fake_attention(q, state, positions, observed_ratio, _sink, _scale):
        calls["attention"] += 1
        assert observed_ratio == ratio
        assert state.num_tokens == 129
        assert torch.equal(positions, torch.arange(129))
        return q.clone()

    monkeypatch.setattr(exact_attention, "_store_raw_kv_carry", fake_raw_store)
    monkeypatch.setattr(exact_attention, "_serving_compressed_kv", fake_compressed_store)
    monkeypatch.setattr(exact_attention, "_serving_attention_rows", fake_attention)
    q = torch.zeros((1, 129, 64, 512), dtype=torch.bfloat16)
    kv = torch.zeros((1, 129, 512), dtype=torch.bfloat16)
    x = torch.zeros((1, 129, 4096), dtype=torch.bfloat16)

    output = exact_attention.exact_compressed_attention(
        q,
        kv,
        x,
        torch.ones(512, dtype=torch.bfloat16),
        torch.zeros(64),
        torch.ones((129, 32), dtype=torch.complex64),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        1e-6,
        512**-0.5,
        ratio,
    )

    assert torch.equal(output, q)
    assert calls == {"raw_store": 1, "compressed_store": 1, "attention": 1}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_c0_full_batch_materializes_one_independent_cache_per_request(monkeypatch):
    exact_attention = _exact_attention_module()
    state_ids = []
    store_calls = []

    def fake_raw_store(kv, _weight, _freqs, _eps, state, offset, *, dequantize):
        store_calls.append(kv.detach().clone())
        assert dequantize is False
        state.num_tokens = kv.shape[1]
        state_ids.append(id(state))
        assert offset == 0
        return kv[0]

    def fake_attention(q, state, positions, ratio, _sink, _scale):
        assert ratio == 0
        assert state.num_tokens == 3
        assert torch.equal(positions, torch.arange(3, device=q.device))
        return q.clone()

    monkeypatch.setattr(exact_attention, "_store_raw_kv_carry", fake_raw_store)
    monkeypatch.setattr(exact_attention, "_serving_attention_rows", fake_attention)
    device = torch.device("cuda")
    q = torch.randn((2, 3, 64, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((2, 3, 512), dtype=torch.bfloat16, device=device)
    output = exact_attention.exact_c0_attention(
        q,
        kv,
        torch.ones(512, dtype=torch.bfloat16, device=device),
        torch.zeros(64, device=device),
        torch.ones((3, 32), dtype=torch.complex64, device=device),
        1e-6,
        512**-0.5,
    )

    assert torch.equal(output, q)
    assert len(store_calls) == 2
    assert len(set(state_ids)) == 2
    assert torch.equal(store_calls[0], kv[:1])
    assert torch.equal(store_calls[1], kv[1:])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_full_sequence_surrogate_backward_is_finite_and_reaches_prior_kv(monkeypatch, ratio):
    exact_attention = _exact_attention_module()

    def fake_raw_store(kv, _weight, _freqs, _eps, state, _offset, *, dequantize):
        assert dequantize is False
        state.num_tokens = kv.shape[1]
        return kv[0]

    def fake_compressed_store(x, *_args, carry_state, dequantize):
        assert dequantize is False
        ratio = _args[-1]
        carry_state.num_compressed = x.shape[1] // ratio
        return x.new_zeros((carry_state.num_compressed, 512))

    def fake_attention(q, _state, _positions, _ratio, _sink, _scale):
        return q.clone()

    monkeypatch.setattr(exact_attention, "_store_raw_kv_carry", fake_raw_store)
    monkeypatch.setattr(exact_attention, "_serving_attention_rows", fake_attention)
    monkeypatch.setattr(exact_attention, "_serving_compressed_kv", fake_compressed_store)
    sequence_length = 3 if ratio == 0 else ratio
    device = torch.device("cuda")
    q = (0.02 * torch.randn((1, sequence_length, 64, 512), device=device)).to(torch.bfloat16).requires_grad_()
    kv = (0.02 * torch.randn((1, sequence_length, 512), device=device)).to(torch.bfloat16).requires_grad_()
    weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    sink = torch.zeros(64, dtype=torch.float32, device=device)
    freqs = torch.ones((sequence_length, 32), dtype=torch.complex64, device=device)

    if ratio == 0:
        output = exact_attention.exact_c0_attention(q, kv, weight, sink, freqs, 1e-6, 512**-0.5)
        x = None
    else:
        x = (0.02 * torch.randn((1, sequence_length, 4096), device=device)).to(torch.bfloat16).requires_grad_()
        coff = 2 if ratio == 4 else 1
        compressor_wkv = 0.001 * torch.randn((coff * 512, 4096), device=device)
        compressor_wgate = 0.001 * torch.randn((coff * 512, 4096), device=device)
        output = exact_attention.exact_compressed_attention(
            q,
            kv,
            x,
            weight,
            sink,
            freqs,
            compressor_wkv,
            compressor_wgate,
            torch.zeros((ratio, coff * 512), device=device),
            torch.ones(512, device=device),
            1e-6,
            512**-0.5,
            ratio,
        )
    output[:, -1].float().sum().backward()

    for grad in (q.grad, kv.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
    assert kv.grad[0, 0].abs().sum() > 0
    assert kv.grad[0, -1].abs().sum() > 0
    if x is not None:
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
