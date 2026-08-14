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


@pytest.mark.parametrize("compress_ratio", [0, 128])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_exact_cp_selects_the_serving_kv_boundary(monkeypatch, compress_ratio, cp_size):
    """Every CP layout gathers raw WKV before the fused serving cache store."""

    from xorl.models.transformers.deepseek_v4 import modeling_deepseek_v4  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        DeepSeekV4Attention,
    )
    from xorl.ops.dsv4 import exact_attention  # noqa: PLC0415

    class _FakeCPGroup:
        def size(self):
            return cp_size

        def rank(self):
            return cp_size - 1

    cfg = _tiny_config(compress_ratios=[compress_ratio])
    cfg._dsv4_flash_exact_mode = True
    group = _FakeCPGroup() if cp_size > 1 else None
    layer = DeepSeekV4Attention(cfg, layer_id=0, cp_group=group)
    sequence_length = 4
    hidden = torch.arange(
        sequence_length * cfg.hidden_size,
        dtype=torch.float32,
    ).view(1, sequence_length, cfg.hidden_size)
    raw_kv = layer.wkv(hidden).detach()

    observed = {"gathers": [], "attention": []}

    monkeypatch.setattr(
        modeling_deepseek_v4._ExactBatchInvariantRmsNorm,
        "apply",
        lambda value, _weight, _eps: value,
    )
    monkeypatch.setattr(exact_attention, "exact_q_norm_rope", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(exact_attention, "exact_inverse_rope", lambda value, *_args, **_kwargs: value)

    def fake_gather(value, dim, cp_group):
        assert cp_group is group
        result = torch.cat([value + 1000.0 * rank for rank in range(cp_size)], dim=dim)
        observed["gathers"].append((value.detach().clone(), result.detach().clone()))
        return result

    def fake_c0(q, kv, *_args, **kwargs):
        observed["attention"].append((kv.detach().clone(), None, kwargs))
        return q

    def fake_compressed(q, kv, source, *_args, **kwargs):
        observed["attention"].append((kv.detach().clone(), source.detach().clone(), kwargs))
        return q

    monkeypatch.setattr(modeling_deepseek_v4, "all_gather_cp", fake_gather)
    monkeypatch.setattr(exact_attention, "exact_c0_attention", fake_c0)
    monkeypatch.setattr(exact_attention, "exact_compressed_attention", fake_compressed)

    output = layer(hidden)
    assert output.shape == hidden.shape
    assert len(observed["attention"]) == 1
    attention_kv, attention_source, kwargs = observed["attention"][0]

    if cp_size == 1:
        assert observed["gathers"] == []
        assert torch.equal(attention_kv, raw_kv)
        assert kwargs["kv_preprocessed"] is False
        assert kwargs["query_positions"] is None
        if compress_ratio:
            assert torch.equal(attention_source, hidden)
    else:
        expected_kv = torch.cat(
            [raw_kv + 1000.0 * rank for rank in range(cp_size)],
            dim=1,
        )
        assert torch.equal(attention_kv, expected_kv)
        assert kwargs["kv_preprocessed"] is False
        assert torch.equal(
            kwargs["query_positions"],
            torch.arange(sequence_length, 2 * sequence_length),
        )
        assert len(observed["gathers"]) == (2 if compress_ratio else 1)
        if compress_ratio:
            expected_source = torch.cat(
                [hidden + 1000.0 * rank for rank in range(cp_size)],
                dim=1,
            )
            assert torch.equal(attention_source, expected_source)


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_exact_ring_cp_restores_gathered_rows_and_uses_local_rope_positions(monkeypatch, compress_ratio):
    """The exact seam consumes zigzag Q locally and logical KV/compressor rows globally."""

    from xorl.models.transformers.deepseek_v4 import modeling_deepseek_v4  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        DeepSeekV4Attention,
    )
    from xorl.ops.dsv4 import exact_attention  # noqa: PLC0415
    from xorl.ops.dsv4.cp_utils import Dsv4ExactCPLayout  # noqa: PLC0415

    class _FakeCPGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 1

    cfg = _tiny_config(compress_ratios=[compress_ratio])
    cfg._dsv4_flash_exact_mode = True
    group = _FakeCPGroup()
    layer = DeepSeekV4Attention(cfg, layer_id=0, cp_group=group)
    local_length = 4
    hidden = torch.arange(local_length * cfg.hidden_size, dtype=torch.float32).view(
        1,
        local_length,
        cfg.hidden_size,
    )
    # ring2 storage for S=8 is rank0=[0,1,6,7], rank1=[2,3,4,5].
    local_positions = torch.tensor([2, 3, 4, 5])
    restore_order = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
    layout = Dsv4ExactCPLayout(
        local_storage_indices=torch.arange(4),
        local_logical_rows=local_positions,
        local_request_ids=torch.zeros(4, dtype=torch.int64),
        local_request_positions=local_positions,
        local_live_count=4,
        compute_rows=4,
        gather_order=restore_order,
        global_logical_rows=torch.arange(8),
        global_request_ids=torch.zeros(8, dtype=torch.int64),
        global_request_positions=torch.arange(8),
        request_ids=(0,),
        local_request_row_indices=(torch.arange(4),),
        global_request_row_indices=(torch.arange(8),),
    )

    observed = {"q_freqs": None, "inverse_freqs": None, "attention": None, "gathers": []}
    monkeypatch.setattr(
        modeling_deepseek_v4._ExactBatchInvariantRmsNorm,
        "apply",
        lambda value, _weight, _eps: value,
    )

    def fake_q_norm_rope(value, freqs, *_args, **_kwargs):
        observed["q_freqs"] = freqs.detach().clone()
        return value

    def fake_inverse_rope(value, freqs, *_args, **_kwargs):
        observed["inverse_freqs"] = freqs.detach().clone()
        return value

    def fake_gather(value, *, dim, layout, cp_group):
        assert cp_group is group
        gathered = torch.cat((value + 1000.0, value), dim=dim)
        observed["gathers"].append((gathered.detach().clone(), dim))
        return gathered.index_select(dim, layout.gather_order)

    def fake_c0(q, kv, *_args, **kwargs):
        observed["attention"] = (kv.detach().clone(), None, kwargs)
        return q

    def fake_compressed(q, kv, source, *_args, **kwargs):
        observed["attention"] = (kv.detach().clone(), source.detach().clone(), kwargs)
        return q

    monkeypatch.setattr(exact_attention, "exact_q_norm_rope", fake_q_norm_rope)
    monkeypatch.setattr(exact_attention, "exact_inverse_rope", fake_inverse_rope)
    monkeypatch.setattr(modeling_deepseek_v4, "gather_dsv4_exact_cp_rows", fake_gather)
    monkeypatch.setattr(exact_attention, "exact_c0_attention", fake_c0)
    monkeypatch.setattr(exact_attention, "exact_compressed_attention", fake_compressed)

    output = layer(hidden, exact_cp_layout=layout)

    assert output.shape == hidden.shape
    expected_local_freqs = layer.freqs_cis.index_select(
        0,
        local_positions.to(layer.freqs_cis.device),
    ).to(hidden.device)
    assert torch.equal(observed["q_freqs"], expected_local_freqs)
    assert torch.equal(observed["inverse_freqs"], expected_local_freqs)
    attention_kv, attention_source, kwargs = observed["attention"]
    gathered_kv = observed["gathers"][0][0]
    assert torch.equal(attention_kv, gathered_kv.index_select(1, restore_order))
    assert torch.equal(kwargs["query_positions"], local_positions)
    assert kwargs["kv_preprocessed"] is False
    if compress_ratio:
        gathered_source = observed["gathers"][1][0]
        assert torch.equal(attention_source, gathered_source.index_select(1, restore_order))


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_exact_packed_requests_reset_c0_c4_c128_state(monkeypatch, compress_ratio):
    """Every packed request gets its own attention/cache/compressor program."""

    from xorl.models.transformers.deepseek_v4 import modeling_deepseek_v4  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepSeekV4Attention  # noqa: PLC0415
    from xorl.ops.dsv4 import exact_attention  # noqa: PLC0415
    from xorl.ops.dsv4.cp_utils import build_dsv4_exact_cp_layout  # noqa: PLC0415

    cfg = _tiny_config(compress_ratios=[compress_ratio])
    cfg._dsv4_flash_exact_mode = True
    layer = DeepSeekV4Attention(cfg, layer_id=0)
    hidden = torch.arange(9 * cfg.hidden_size, dtype=torch.float32).view(1, 9, cfg.hidden_size)
    layout = build_dsv4_exact_cp_layout(
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, -1]]),
        torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, -1]]),
        torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4, 0]]),
        torch.tensor([[True, True, True, True, True, True, True, True, False]]),
        compute_rows=9,
        cp_group=None,
    )
    calls = []
    monkeypatch.setattr(
        modeling_deepseek_v4._ExactBatchInvariantRmsNorm,
        "apply",
        lambda value, _weight, _eps: value,
    )
    monkeypatch.setattr(exact_attention, "exact_q_norm_rope", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(exact_attention, "exact_inverse_rope", lambda value, *_args, **_kwargs: value)

    def fake_c0(q, kv, _weight, _sink, freqs, *_args, **kwargs):
        calls.append((q, kv, None, freqs, kwargs))
        return q

    def fake_compressed(q, kv, source, _weight, _sink, freqs, *_args, **kwargs):
        calls.append((q, kv, source, freqs, kwargs))
        return q

    monkeypatch.setattr(exact_attention, "exact_c0_attention", fake_c0)
    monkeypatch.setattr(exact_attention, "exact_compressed_attention", fake_compressed)

    output = layer(hidden, exact_cp_layout=layout)
    assert output.shape == hidden.shape
    assert len(calls) == 2
    for call, expected_length in zip(calls, (3, 5)):
        q, kv, source, freqs, kwargs = call
        assert q.shape[1] == kv.shape[1] == expected_length
        assert torch.equal(kwargs["query_positions"], torch.arange(expected_length))
        assert kwargs["kv_preprocessed"] is False
        assert torch.equal(freqs, layer.freqs_cis[:expected_length])
        if compress_ratio:
            assert source.shape[1] == expected_length
        else:
            assert source is None
    assert not any(name.startswith("_dsv4_zigzag") for name in layer.__dict__)


def test_dsv4_pipeline_carries_hyperconnection_state_first_middle_last(monkeypatch):
    """A physical PP cut carries 4-D state and matches the uncut model."""

    import copy
    import types
    from types import SimpleNamespace

    from xorl.distributed import pipeline_parallel
    from xorl.distributed.pipeline_parallel import _pp_forward, _recursive_prune
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
    from xorl.trainers.training_utils import _set_pp_batch_metadata

    monkeypatch.setattr(pipeline_parallel, "get_parallel_state", lambda: SimpleNamespace(cp_size=1))
    cfg = _tiny_config(compress_ratios=[0, 0, 0])
    cfg.tie_word_embeddings = False
    cfg._moe_implementation = "eager"
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")
    model.eval()
    input_ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]])
    position_ids = torch.arange(8).view(1, -1)
    with torch.no_grad():
        expected = model.lm_head(model(input_ids=input_ids, position_ids=position_ids).last_hidden_state)

    plan = [
        ["model.embed_tokens", "model.layers.0"],
        ["model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ]
    parts = []
    for stage_idx, module_names in enumerate(plan):
        part = copy.deepcopy(model)
        _recursive_prune(part, "", set(module_names))
        part._configure_pp_stage(stage_idx=stage_idx, num_stages=3)
        part._pp_is_first = stage_idx == 0
        part._pp_is_last = stage_idx == 2
        part._pp_stage_idx = stage_idx
        part._pp_original_forward = part.forward
        part.forward = types.MethodType(_pp_forward, part)
        parts.append(part)

    assert parts[0].model.hc_head_fn is None
    assert parts[1].model.hc_head_fn is None
    assert parts[2].model.hc_head_fn is not None

    _set_pp_batch_metadata(
        parts,
        [{"input_ids": input_ids, "position_ids": position_ids}],
    )
    first_wire = parts[0](input_ids)
    assert first_wire.shape == (1, 8, cfg.hc_mult, cfg.hidden_size)
    middle_wire = parts[1](first_wire)
    assert middle_wire.shape == first_wire.shape
    logits = parts[2](middle_wire)
    assert logits.shape == (1, 8, cfg.vocab_size)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    logits.square().mean().backward()
    assert parts[0].model.embed_tokens.weight.grad is not None
    assert next(layer for layer in parts[1].model.layers if layer is not None).self_attn.wq_a.weight.grad is not None
    assert parts[2].lm_head.weight.grad is not None


def test_exact_ragged_cp_pipeline_uses_storage_row_wire_first_middle_last_and_backward(monkeypatch):
    """Ragged exact rows stay compact for compute and storage-sized on the PP wire."""

    import copy
    import types
    from types import SimpleNamespace

    import torch.nn as nn

    from xorl.distributed import pipeline_parallel
    from xorl.distributed.pipeline_parallel import _pp_forward, _recursive_prune
    from xorl.models.transformers.deepseek_v4 import modeling_deepseek_v4
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
    from xorl.trainers.training_utils import _set_pp_batch_metadata

    parallel_state = SimpleNamespace(
        cp_size=2,
        fsdp_enabled=False,
        ep_enabled=False,
        fsdp_group=None,
        ep_group=None,
        sp_group=None,
    )
    monkeypatch.setattr(pipeline_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(modeling_deepseek_v4, "get_parallel_state", lambda: parallel_state)

    class _ScaleLayer(nn.Module):
        def __init__(self, layer_id: int):
            super().__init__()
            self.layer_id = layer_id
            self.scale = nn.Parameter(torch.tensor(1.0 + 0.125 * layer_id))
            self.seen_rows = []

        def forward(self, hidden, **_kwargs):
            self.seen_rows.append(int(hidden.shape[1]))
            return hidden * self.scale

    class _MeanHyperConnection:
        @staticmethod
        def block_expand(hidden):
            return hidden.unsqueeze(2).expand(-1, -1, 2, -1).clone()

        @staticmethod
        def block_head(hidden, *_args):
            return hidden.mean(dim=2)

    torch.manual_seed(37)
    cfg = _tiny_config(compress_ratios=[0, 0, 0])
    cfg.tie_word_embeddings = False
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")
    # The boundary row plan is selected dynamically by the resolved exact
    # config.  Lightweight layers isolate that plan from the CUDA attention
    # implementation while retaining real differentiable stage parameters.
    cfg._dsv4_flash_exact_mode = True
    model.model.layers = nn.ModuleList([_ScaleLayer(index) for index in range(3)])
    model.model.hc_util = _MeanHyperConnection()
    model.model.norm = nn.Identity()
    model.to(torch.bfloat16)
    model.train()

    input_ids = torch.tensor([[3, 0, 5, 7, 0, 9, 11, 13]])
    row_metadata = {
        "_cp_logical_row_indices": torch.tensor([[0, -1, 1, 2, -1, 3, 4, 5]]),
        "_cp_request_ids": torch.tensor([[0, -1, 0, 1, -1, 1, 1, 1]]),
        "_cp_request_positions": torch.tensor([[0, 0, 1, 0, 0, 1, 2, 3]]),
        "_cp_live_mask": torch.tensor([[True, False, True, True, False, True, True, True]]),
        "_r3_sample_lengths": [2, 4],
        "num_samples": 2,
    }
    # CP keeps local storage rows on each PP lane while position/FA metadata
    # describe the full global stream.
    position_ids = torch.arange(16).view(1, -1)
    batch = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "cu_seq_lens_q": torch.tensor([0, 16], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 16], dtype=torch.int32),
        "max_length_q": 16,
        "max_length_k": 16,
        **row_metadata,
    }

    baseline = copy.deepcopy(model)
    baseline_hidden = baseline(
        input_ids=input_ids,
        position_ids=position_ids,
        **row_metadata,
    ).last_hidden_state
    baseline_logits = baseline.lm_head(baseline_hidden)
    baseline_logits.square().sum().backward()

    plan = [
        ["model.embed_tokens", "model.layers.0"],
        ["model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ]
    parts = []
    boundary_state = {
        "rank": 4,
        "dtype": torch.bfloat16,
        "shape_suffix": (cfg.hc_mult, cfg.hidden_size),
        "state": "completed_hyperconnection_residual",
    }
    for stage_idx, module_names in enumerate(plan):
        part = copy.deepcopy(model)
        _recursive_prune(part, "", set(module_names))
        part._configure_pp_stage(stage_idx=stage_idx, num_stages=3)
        part._pp_is_first = stage_idx == 0
        part._pp_is_last = stage_idx == 2
        part._pp_stage_idx = stage_idx
        part._pp_exact_boundary_contract = True
        part._pp_pipeline_boundary_state = boundary_state
        part._pp_original_forward = part.forward
        part.forward = types.MethodType(_pp_forward, part)
        parts.append(part)

    _set_pp_batch_metadata(parts, [batch])
    first_wire = parts[0](input_ids)
    first_wire.retain_grad()
    assert first_wire.shape == (1, 8, cfg.hc_mult, cfg.hidden_size)
    assert torch.count_nonzero(first_wire[:, 6:]) == 0

    middle_wire = parts[1](first_wire)
    middle_wire.retain_grad()
    assert middle_wire.shape == first_wire.shape
    assert torch.count_nonzero(middle_wire[:, 6:]) == 0

    staged_logits = parts[2](middle_wire)
    assert staged_logits.shape == (1, 8, cfg.vocab_size)
    assert [parts[index].model.layers[index].seen_rows for index in range(3)] == [[6], [6], [6]]
    staged_logits.square().sum().backward()

    torch.testing.assert_close(staged_logits, baseline_logits, rtol=0, atol=0)
    assert torch.count_nonzero(first_wire.grad[:, 6:]) == 0
    assert torch.count_nonzero(middle_wire.grad[:, 6:]) == 0
    baseline_params = dict(baseline.named_parameters())
    staged_params = {}
    for part in parts:
        staged_params.update(dict(part.named_parameters()))
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.scale",
        "model.layers.1.scale",
        "model.layers.2.scale",
        "lm_head.weight",
    ):
        torch.testing.assert_close(staged_params[name].grad, baseline_params[name].grad, rtol=0, atol=0)


def test_exact_layout_is_threaded_through_interleaved_checkpoint_microbatches(monkeypatch):
    """Checkpoint recompute receives each call's immutable layout, not layer state."""

    from functools import partial
    from types import SimpleNamespace

    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint

    from xorl.models.transformers.deepseek_v4 import modeling_deepseek_v4
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

    monkeypatch.setattr(
        modeling_deepseek_v4,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_size=1, fsdp_group=None, ep_group=None, sp_group=None),
    )

    calls = []

    class _RecordingLayer(nn.Module):
        layer_id = 0

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.25))

        def forward(self, hidden, *, exact_cp_layout, **_kwargs):
            calls.append(tuple(int(rows.numel()) for rows in exact_cp_layout.global_request_row_indices))
            return hidden * self.scale

    class _IdentityHC:
        @staticmethod
        def block_expand(hidden):
            return hidden.unsqueeze(2).expand(-1, -1, 2, -1).clone()

        @staticmethod
        def block_head(hidden, *_args):
            return hidden.mean(dim=2)

    cfg = _tiny_config(compress_ratios=[0])
    cfg._dsv4_flash_exact_mode = True
    model = DeepseekV4Model(cfg, moe_implementation="eager")
    recorder = _RecordingLayer()
    model.layers = nn.ModuleList([recorder])
    model.hc_util = _IdentityHC()
    model.norm = nn.Identity()
    model.gradient_checkpointing = True
    model._gradient_checkpointing_method = "recompute_full_layer"
    model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    model.train()

    def run(request_lengths, token_offset):
        request_ids = []
        request_positions = []
        for request_id, length in enumerate(request_lengths):
            request_ids.extend([request_id] * length)
            request_positions.extend(range(length))
        live_rows = sum(request_lengths)
        storage_rows = 8
        return model(
            input_ids=(torch.arange(storage_rows) + token_offset).remainder(cfg.vocab_size).view(1, -1),
            position_ids=torch.arange(storage_rows).view(1, -1),
            _cp_logical_row_indices=torch.tensor([list(range(live_rows)) + [-1] * (storage_rows - live_rows)]),
            _cp_request_ids=torch.tensor([request_ids + [-1] * (storage_rows - live_rows)]),
            _cp_request_positions=torch.tensor([request_positions + [0] * (storage_rows - live_rows)]),
            _cp_live_mask=torch.tensor([[True] * live_rows + [False] * (storage_rows - live_rows)]),
            _r3_sample_lengths=request_lengths,
            num_samples=len(request_lengths),
        ).last_hidden_state

    output_a = run([2, 4], 1)
    output_b = run([3, 3], 9)
    assert calls[:2] == [(2, 4), (3, 3)]
    assert torch.count_nonzero(output_a[:, 6:]) == 0
    assert torch.count_nonzero(output_b[:, 6:]) == 0
    (output_a.square().sum() + output_b.square().sum()).backward()
    assert calls.count((2, 4)) >= 2
    assert calls.count((3, 3)) >= 2
    assert recorder.scale.grad is not None
    assert model.embed_tokens.weight.grad is not None
    assert not hasattr(recorder, "exact_cp_layout")
