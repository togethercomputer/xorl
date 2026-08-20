"""Test torch.compile compatibility for MoE models.

Tests per-layer compilation (like torchtitan's apply_compile) on:
- MoEBlock alone (native/eager/triton/quack backends)
- Qwen3MoeDecoderLayer
- Full Qwen3MoeForCausalLM forward + backward
- TFLOPS benchmark: compiled vs uncompiled

Known issues:
- fullgraph=True: graph break from logger.warning_once in get_parallel_state()
- triton/quack backends: custom autograd.Function causes graph breaks but
  works with fullgraph=False (torch.compile splits around the opaque kernels).
"""

import pytest
import torch
import torch.nn as nn

from xorl.models.layers.rope import RotaryEmbedding
from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
)


DEVICE = "cuda"
DTYPE = torch.bfloat16


def _tiny_moe_config(**overrides):
    """Create a minimal Qwen3MoeConfig for fast testing."""

    defaults = dict(
        vocab_size=1000,
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        output_router_logits=False,
        _moe_implementation="native",
        max_position_embeddings=128,
        pad_token_id=0,
        _attn_implementation="sdpa",
    )
    defaults.update(overrides)
    return Qwen3MoeConfig(**defaults)


def _make_position_embeddings(config, seq_len, device, dtype):
    """Create position_embeddings (cos, sin) for decoder layer tests."""

    rotary = RotaryEmbedding(config=config).to(device)
    dummy_hidden = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rotary(dummy_hidden, position_ids)
    return cos, sin


def _make_moe_block(moe_backend, hidden_size=128, num_experts=4, top_k=2, intermediate=128):
    """Create an MoEBlock with xavier init for numerical stability."""
    from xorl.models.layers.moe.moe_block import MoEBlock  # noqa: PLC0415

    block = MoEBlock(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate,
        moe_implementation=moe_backend,
    )
    nn.init.xavier_normal_(block.experts.gate_proj.data)
    nn.init.xavier_normal_(block.experts.up_proj.data)
    nn.init.xavier_normal_(block.experts.down_proj.data)
    nn.init.xavier_normal_(block.gate.weight.data)
    return block.to(DEVICE, DTYPE)


def _available_backends():
    """Return list of available MoE backends on this system."""
    from xorl.utils.import_utils import is_fused_moe_available  # noqa: PLC0415

    backends = ["native", "eager"]
    if is_fused_moe_available():
        backends.append("triton")
    backends.append("quack")
    return backends


AVAILABLE_BACKENDS = _available_backends() if torch.cuda.is_available() else []


# ---------------------------------------------------------------------------
# Test 1: MoEBlock compile -- aot_eager + inductor + fullgraph + correctness
# ---------------------------------------------------------------------------


def _assert_moe_block_compile(moe_backend):
    """MoEBlock compile: aot_eager tracing, inductor compile, and correctness."""
    # --- aot_eager tracing (forward + backward) ---
    block = _make_moe_block(moe_backend)
    compiled_aot = torch.compile(block, fullgraph=False, backend="aot_eager")

    x = torch.randn(2, 8, 128, device=DEVICE, dtype=DTYPE, requires_grad=True)
    out, router_logits = compiled_aot(x)
    assert out.shape == x.shape
    assert router_logits.shape == (16, 4)
    out.sum().backward()
    assert x.grad is not None

    # --- inductor compile (forward + backward) ---
    block2 = _make_moe_block(moe_backend)
    compiled_ind = torch.compile(block2, fullgraph=False, backend="inductor")
    x2 = torch.randn(2, 8, 128, device=DEVICE, dtype=DTYPE, requires_grad=True)
    out2, _ = compiled_ind(x2)
    assert out2.shape == x2.shape
    out2.sum().backward()
    assert x2.grad is not None

    # --- correctness: compiled vs uncompiled match ---
    torch.manual_seed(42)
    block4 = _make_moe_block(moe_backend)
    x4 = torch.randn(2, 8, 128, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        ref_out, ref_logits = block4(x4)
    compiled_block4 = torch.compile(block4, fullgraph=False, backend="aot_eager")
    with torch.no_grad():
        comp_out, comp_logits = compiled_block4(x4)
    torch.testing.assert_close(ref_out, comp_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(ref_logits, comp_logits, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_block_decoder_and_full_model_compile_policy():
    for moe_backend in AVAILABLE_BACKENDS:
        _assert_moe_block_compile(moe_backend)
        _assert_decoder_layer_compile(moe_backend)

    # Lower-level contracts already compile every available MoE backend with
    # both compiler backends. Full-model composition only needs each compiler
    # path once.
    for moe_backend, compile_backend in (("native", "aot_eager"), ("eager", "inductor")):
        _assert_full_model_per_layer_compile(moe_backend, compile_backend)


# ---------------------------------------------------------------------------
# Test 2: Qwen3MoeDecoderLayer compile (aot_eager + inductor)
# ---------------------------------------------------------------------------


def _assert_decoder_layer_compile(moe_backend):
    """Decoder layer compile: aot_eager and inductor, forward + backward."""

    seq_len = 8
    for compile_backend in ["aot_eager", "inductor"]:
        config = _tiny_moe_config(_moe_implementation=moe_backend)
        layer = Qwen3MoeDecoderLayer(config, layer_idx=0).to(DEVICE, DTYPE)
        compiled_layer = torch.compile(layer, fullgraph=False, backend=compile_backend)

        x = torch.randn(2, seq_len, 128, device=DEVICE, dtype=DTYPE, requires_grad=True)
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(2, -1)
        position_embeddings = _make_position_embeddings(config, seq_len, DEVICE, DTYPE)

        outputs = compiled_layer(
            hidden_states=x,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        hidden_out = outputs[0]
        assert hidden_out.shape == x.shape
        hidden_out.sum().backward()
        assert x.grad is not None
        x.grad = None


# ---------------------------------------------------------------------------
# Test 3: Full model per-layer compile (torchtitan style)
# ---------------------------------------------------------------------------


def _assert_full_model_per_layer_compile(moe_backend, compile_backend):
    """Apply torch.compile to each decoder layer, run forward + backward."""

    config = _tiny_moe_config(_moe_implementation=moe_backend)
    model = Qwen3MoeForCausalLM(config).to(DEVICE, DTYPE)

    compiled_count = 0
    for layer_id, mod in model.model.layers.named_children():
        if isinstance(mod, Qwen3MoeDecoderLayer):
            compiled_mod = torch.compile(mod, fullgraph=False, backend=compile_backend)
            model.model.layers.register_module(layer_id, compiled_mod)
            compiled_count += 1

    input_ids = torch.randint(0, 1000, (2, 16), device=DEVICE)

    output = model(input_ids=input_ids)
    assert output.last_hidden_state is not None

    output.last_hidden_state.sum().backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients found"
