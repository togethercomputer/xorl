"""End-to-end CPU smoke test for DeepseekV4Model + DeepseekV4ForCausalLM.

Builds a tiny model with mixed compress_ratios (one C0, one C128, one
window-only repeat) and one hash-routed MoE layer. Runs forward+backward
through the full stack: embed → block_expand → N decoder layers (each
HC-wrapped) → block_head → norm → lm_head.

The tilelang/CUDA-only paths (compress_ratio=4 sparse-MLA, full Flash
dims) are exercised in Phase-6 e2e jobs.
"""

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _cpu_env(monkeypatch):
    monkeypatch.setenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", "256")
    monkeypatch.setenv("XORL_DSV4_SPARSE_ATTN_IMPL", "sparse")
    from xorl.ops.dsv4.rope import precompute_freqs_cis  # noqa: PLC0415

    precompute_freqs_cis.cache_clear()
    yield
    precompute_freqs_cis.cache_clear()


def _tiny_config(*, num_hidden_layers=3, compress_ratios=None, num_hash_layers=0):
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config  # noqa: PLC0415

    if compress_ratios is None:
        compress_ratios = [0] * num_hidden_layers

    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=4,
        max_position_embeddings=256,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=num_hash_layers,
        hc_mult=2,
        compress_ratios=list(compress_ratios),
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        num_nextn_predict_layers=0,
        _moe_implementation="eager",
    )


def _make_model(cfg, model_cls):
    model = model_cls(cfg, moe_implementation="eager").to(torch.float32)

    # Spot-check that there are no NaNs left after init (catches any param
    # type _init_weights forgot to handle).
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite params after init in {name}"
    for name, b in model.named_buffers():
        if b.is_floating_point():
            assert torch.isfinite(b).all(), f"non-finite buffer after init in {name}"
    return model


def _lm_logits(model, input_ids):
    outputs = model(input_ids)
    return model.lm_head(outputs.last_hidden_state)


def test_for_causal_lm_rejects_pipeline_parallelism():
    """DSv4 requires a dedicated PP forward because hyperconnection state is 4-D."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    cfg = _tiny_config(num_hidden_layers=1, compress_ratios=[0])
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")

    assert model.config.base_model_pp_plan is None
    with pytest.raises(ValueError, match="Pipeline parallelism is not supported"):
        model.get_pp_module_config()


def test_model_forward_shape_window_only():
    """3 layers, all window-only (no compressor, no indexer)."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Model  # noqa: PLC0415

    torch.manual_seed(0)
    cfg = _tiny_config(num_hidden_layers=3, compress_ratios=[0, 0, 0])
    model = _make_model(cfg, DeepseekV4Model)

    bsz, seqlen = 1, cfg.sliding_window
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)

    out = model(input_ids).last_hidden_state
    assert out.shape == (bsz, seqlen, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_model_forward_shape_with_c128():
    """One C128 layer + two C0 layers."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Model  # noqa: PLC0415

    torch.manual_seed(1)
    cfg = _tiny_config(num_hidden_layers=3, compress_ratios=[0, 128, 0])
    model = _make_model(cfg, DeepseekV4Model)

    bsz, seqlen = 1, 128  # multiple of 128 for the C128 layer
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)

    out = model(input_ids).last_hidden_state
    assert out.shape == (bsz, seqlen, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_for_causal_lm_forward_backward():
    """Full forward + backward through the LM head."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    torch.manual_seed(2)
    cfg = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 0])
    model = _make_model(cfg, DeepseekV4ForCausalLM)

    bsz, seqlen = 2, cfg.sliding_window
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)

    outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=False)
    assert outputs.last_hidden_state.shape == (bsz, seqlen, cfg.hidden_size)
    logits = model.lm_head(outputs.last_hidden_state)
    assert logits.shape == (bsz, seqlen, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    assert torch.isfinite(loss)

    loss.backward()

    # Trainable params should have finite grads. HC params (hc_attn_*,
    # hc_ffn_*, hc_head_*) are intentionally NOT in this list — the V4
    # HyperConnection mixer runs under torch.no_grad(), so its params are
    # effectively frozen at training time (initialized from the checkpoint
    # and never updated).
    must_grad = [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.wq_a.weight",
        "model.layers.0.mlp.gate.weight",
    ]
    grads = dict(model.named_parameters())
    for name in must_grad:
        p = grads[name]
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"

    # HC params should have grad=None (no autograd path through the mixer).
    for name in (
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.0.hc_attn_fn",
        "model.layers.0.hc_ffn_fn",
    ):
        assert grads[name].grad is None, f"{name} unexpectedly received a gradient"


def test_for_causal_lm_with_hash_layer():
    """First layer is hash-routed; verify input_ids threads through correctly."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    torch.manual_seed(3)
    cfg = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 0], num_hash_layers=1)
    model = _make_model(cfg, DeepseekV4ForCausalLM)

    # Populate the hash table deterministically.
    table = (
        torch.arange(cfg.vocab_size).unsqueeze(1) + torch.arange(cfg.num_experts_per_tok).unsqueeze(0)
    ) % cfg.n_routed_experts
    model.model.layers[0].mlp.tid2eid.copy_(table.to(torch.int32))
    # Layer 1 (non-hash) keeps its random e_score_correction_bias init.

    bsz, seqlen = 1, cfg.sliding_window
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)

    logits = _lm_logits(model, input_ids)
    assert logits.shape == (bsz, seqlen, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()
    # Layer 0 is hash, so its gate gets grads via the gather of routing weights.
    assert torch.isfinite(model.model.layers[0].mlp.gate.weight.grad).all()


def test_keep_fp32_marks_propagated():
    """All HC + attn_sink + compressor fp32 params carry _keep_fp32 = True."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    cfg = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 128])
    model = _make_model(cfg, DeepseekV4ForCausalLM)

    expected_fp32 = []
    for li in range(cfg.num_hidden_layers):
        for sub in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            expected_fp32.append(f"model.layers.{li}.{sub}")
        expected_fp32.append(f"model.layers.{li}.self_attn.attn_sink")
    expected_fp32.extend(["model.hc_head_fn", "model.hc_head_base", "model.hc_head_scale"])

    params = dict(model.named_parameters())
    for name in expected_fp32:
        p = params[name]
        assert p.dtype == torch.float32, f"{name} is not fp32"
        assert getattr(p, "_keep_fp32", False) is True, f"{name} missing _keep_fp32"


def test_from_config_wires_parallel_groups(monkeypatch):
    """DSv4 factory passes xorl's sequence-parallel group into attention."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    class FakeGroup:
        def size(self):
            return 1

    class FakeParallelState:
        tp_enabled = False
        cp_enabled = True
        sp_group = FakeGroup()

    monkeypatch.setattr("xorl.distributed.parallel_state.get_parallel_state", lambda: FakeParallelState())

    cfg = _tiny_config(num_hidden_layers=1, compress_ratios=[0])
    model = DeepseekV4ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="native")

    attn = model.model.layers[0].self_attn
    assert attn.tp_group is None
    assert attn.cp_group is FakeParallelState.sp_group
    assert attn.cp_size == 1
    assert model.config._attn_implementation == "native"


def test_from_config_dtype_cast_preserves_rope_and_fp32_params(monkeypatch):
    """Registry construction keeps complex RoPE caches and fp32-only params intact."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    class FakeParallelState:
        tp_enabled = False
        cp_enabled = False

    monkeypatch.setattr("xorl.distributed.parallel_state.get_parallel_state", lambda: FakeParallelState())

    cfg = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 128])
    model = DeepseekV4ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16)

    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.layers[0].self_attn.wq_a.weight.dtype == torch.bfloat16

    fp32_params = [
        model.model.hc_head_fn,
        model.model.layers[0].self_attn.attn_sink,
        model.model.layers[1].self_attn.compressor.ape,
        model.model.layers[1].self_attn.compressor.wkv.weight,
        model.model.layers[1].self_attn.compressor.wgate.weight,
    ]
    for param in fp32_params:
        assert param.dtype == torch.float32
        assert getattr(param, "_keep_fp32", False) is True

    freqs_cis = {name: buf for name, buf in model.named_buffers() if name.endswith("freqs_cis")}
    assert freqs_cis
    for name, buf in freqs_cis.items():
        assert torch.is_complex(buf), f"{name} was cast away from complex dtype"
        assert buf.imag.abs().max() > 0, f"{name} lost its imaginary RoPE component"


def test_direct_to_dtype_preserves_rope_and_fp32_params():
    """Direct DSv4 dtype casts share the same RoPE/fp32 carve-outs as registry construction."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    cfg = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 128])
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")
    model.to(torch.bfloat16)

    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.layers[1].self_attn.compressor.ape.dtype == torch.float32
    assert model.model.layers[1].self_attn.compressor.wkv.weight.dtype == torch.float32

    freqs_cis = {name: buf for name, buf in model.named_buffers() if name.endswith("freqs_cis")}
    assert freqs_cis
    for name, buf in freqs_cis.items():
        assert torch.is_complex(buf), f"{name} was cast away from complex dtype"
        assert buf.imag.abs().max() > 0, f"{name} lost its imaginary RoPE component"


def test_outer_gradient_checkpointing_wraps_decoder_layers():
    """DSv4Model has the checkpoint flag consumed by the decoder loop."""
    from xorl.models.module_utils import DEFAULT_GRADIENT_CHECKPOINTING_METHOD  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Model  # noqa: PLC0415

    cfg = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 0])
    model = _make_model(cfg, DeepseekV4Model)
    model.train()

    calls = []

    def fake_checkpoint(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    model.gradient_checkpointing = True
    model._gradient_checkpointing_method = DEFAULT_GRADIENT_CHECKPOINTING_METHOD
    model._gradient_checkpointing_func = fake_checkpoint

    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.sliding_window), dtype=torch.long)
    out = model(input_ids).last_hidden_state

    assert out.shape == (1, cfg.sliding_window, cfg.hidden_size)
    assert len(calls) == cfg.num_hidden_layers
    assert all(call_kwargs["input_ids"] is input_ids for _, _, call_kwargs in calls)
