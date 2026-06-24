"""CPU smoke tests for ``DeepSeekV4Attention``.

Covers the two non-tilelang variants:

- ``compress_ratio == 0``: pure window attention (no compressor, no indexer).
- ``compress_ratio == 128``: window + static-pool compressed KV (compressor only).

The ``compress_ratio == 4`` (DSA indexer) variant requires tilelang and is
exercised by the kernel tests (``tests/ops/dsv4/``) plus a Phase-6 e2e job.

Tests use compact dims so they finish in seconds on CPU. The ``XORL_DSV4_ROPE_MAX_SEQ_LEN``
override keeps the precomputed freqs_cis tensor small.
"""

import pytest
import torch


pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _small_rope_buffer(monkeypatch):
    """Avoid precomputing 65k×D/2 freqs_cis when the test only needs ~16."""
    monkeypatch.setenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", "1024")
    monkeypatch.setenv("XORL_DSV4_SPARSE_ATTN_IMPL", "sparse")  # pure-torch ref
    # Clear the @lru_cache on precompute_freqs_cis so cross-test state with
    # different (dim, seqlen, factor, ...) keys doesn't leak.
    from xorl.ops.dsv4.rope import precompute_freqs_cis  # noqa: PLC0415

    precompute_freqs_cis.cache_clear()
    yield
    precompute_freqs_cis.cache_clear()


def _tiny_config(*, compress_ratios):
    """Compact DSv4 config that satisfies all internal consistency asserts.

    hidden_size=64, n_heads=4, n_groups=2, q_lora_rank=32, o_lora_rank=16,
    head_dim=32, qk_rope_head_dim=8, sliding_window=8.
    """
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config  # noqa: PLC0415

    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=len(compress_ratios),
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        max_position_embeddings=1024,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        sliding_window=8,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        hc_mult=2,
        compress_ratios=list(compress_ratios),
        compress_rope_theta=160000.0,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 256,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        # MTP slot is consumed by compressor_ratios validator only when present
        num_nextn_predict_layers=0,
    )


@pytest.mark.parametrize("compress_ratio", [0, 128])
def test_attention_forward_backward_shapes(compress_ratio):
    """Forward + backward at every variant produce the right shapes and finite grads."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415

    torch.manual_seed(0)
    cfg = _tiny_config(compress_ratios=[compress_ratio, compress_ratio])
    layer = DeepSeekV4Attention(cfg, layer_id=0).to(torch.float32)
    # ``attn_sink`` and the compressor's fp32 params (``ape``, ``wkv``,
    # ``wgate``) are ``torch.empty``-allocated and never touched by any of
    # the standard PyTorch ``nn.*`` constructors. Calling the production
    # ``DeepseekV4PreTrainedModel._init_weights`` zero-inits them, but this
    # test bypasses ``post_init`` so we replicate the contract here. Without
    # this, a stray ``inf`` / ``nan`` in the uninitialized memory propagates
    # through softmax and the assertion below trips intermittently depending
    # on torch's allocator state.
    with torch.no_grad():
        layer.attn_sink.zero_()
        if layer.compressor is not None:
            layer.compressor.ape.zero_()
            for m in (layer.compressor.wkv, layer.compressor.wgate):
                m.weight.normal_(0.0, 0.02)
            layer.compressor.norm.weight.fill_(1.0)
    # Window-only requires seqlen >= window_size; C128 requires seqlen % 128 == 0
    # — but the smallest C128-friendly seqlen is 128, which is fine. For the
    # window-only path we use window_size as the seqlen.
    seqlen = 128 if compress_ratio == 128 else cfg.sliding_window
    x = torch.randn(1, seqlen, cfg.hidden_size, requires_grad=True)

    out = layer(x)

    assert out.shape == (1, seqlen, cfg.hidden_size), out.shape
    assert torch.isfinite(out).all(), "non-finite forward output"

    out.sum().backward()
    assert torch.isfinite(x.grad).all(), "non-finite input grad"

    # Every trainable param should have a finite gradient.
    for name, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_attn_sink_is_fp32_and_keep_fp32_marked():
    """attn_sink is per-head fp32 and tagged for the FSDP2 dtype policy."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415

    cfg = _tiny_config(compress_ratios=[0])
    layer = DeepSeekV4Attention(cfg, layer_id=0)

    assert layer.attn_sink.dtype == torch.float32
    assert layer.attn_sink.shape == (cfg.num_attention_heads,)
    assert getattr(layer.attn_sink, "_keep_fp32", False) is True


def test_kv_qat_enabled_from_quantization_config():
    """Xorl mirrors Miles' config-driven FP8-QAT gate instead of an env toggle."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415
    from xorl.ops.dsv4.utils import dsv4_kv_qat_enabled  # noqa: PLC0415

    cfg = _tiny_config(compress_ratios=[0])
    assert dsv4_kv_qat_enabled(cfg) is False

    cfg.quantization_config = {"quant_method": "fp8"}
    assert dsv4_kv_qat_enabled(cfg) is True
    assert DeepSeekV4Attention(cfg, layer_id=0)._kv_qat_enabled is True

    cfg.quantization_config = {"quant_method": "awq"}
    assert dsv4_kv_qat_enabled(cfg) is False


def test_attn_sink_promoted_for_tilelang_after_bf16_cast(monkeypatch):
    """Tilelang sparse attention requires the per-head sink tensor in fp32."""
    from xorl.models.transformers.deepseek_v4 import modeling_deepseek_v4  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415

    monkeypatch.setenv("XORL_DSV4_SPARSE_ATTN_IMPL", "tilelang")
    seen = {}

    def fake_sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale):
        del kv, topk_idxs, sm_scale
        seen["attn_sink_dtype"] = attn_sink.dtype
        return torch.zeros_like(q)

    monkeypatch.setattr(modeling_deepseek_v4, "sparse_attn_tilelang", fake_sparse_attn_tilelang)

    cfg = _tiny_config(compress_ratios=[0])
    layer = DeepSeekV4Attention(cfg, layer_id=0).to(torch.bfloat16)
    assert torch.is_complex(layer.freqs_cis)
    assert layer.freqs_cis.imag.abs().max() > 0
    with torch.no_grad():
        layer.attn_sink.zero_()

    x = torch.randn(1, cfg.sliding_window, cfg.hidden_size, dtype=torch.bfloat16)
    layer(x)

    assert layer.attn_sink.dtype == torch.bfloat16
    assert seen["attn_sink_dtype"] == torch.float32


def test_compressor_present_only_when_needed():
    """C0 has no compressor/indexer; C128 has compressor; C4 has both."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415

    cfg = _tiny_config(compress_ratios=[0, 128, 4])

    l0 = DeepSeekV4Attention(cfg, layer_id=0)
    l128 = DeepSeekV4Attention(cfg, layer_id=1)
    l4 = DeepSeekV4Attention(cfg, layer_id=2)

    assert l0.compressor is None and l0.indexer is None
    assert l128.compressor is not None and l128.indexer is None
    assert l4.compressor is not None and l4.indexer is not None


def test_tp_size_gt_1_rejected():
    """Stub guard until the xorl-style SP gather is wired."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415

    cfg = _tiny_config(compress_ratios=[0])

    class _FakeGroup:
        def size(self):
            return 2

    with pytest.raises(AssertionError, match="TP > 1 is not implemented"):
        DeepSeekV4Attention(cfg, layer_id=0, tp_group=_FakeGroup())
