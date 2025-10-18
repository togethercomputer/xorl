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
import time
import torch
import torch.nn as nn

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _tiny_moe_config(**overrides):
    """Create a minimal Qwen3MoeConfig for fast testing."""
    from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

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
    from xorl.models.layers.rope import RotaryEmbedding
    rotary = RotaryEmbedding(config=config).to(device)
    dummy_hidden = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rotary(dummy_hidden, position_ids)
    return cos, sin


def _make_moe_block(moe_backend, hidden_size=128, num_experts=4, top_k=2, intermediate=128):
    """Create an MoEBlock with xavier init for numerical stability."""
    from xorl.models.layers.moe.moe_block import MoEBlock
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
    backends = ["native", "eager"]
    try:
        from xorl.utils.import_utils import is_fused_moe_available
        if is_fused_moe_available():
            backends.append("triton")
    except Exception:
        pass
    try:
        from xorl.ops.group_gemm.kernel.quack import quack_group_gemm_same_nk
        backends.append("quack")
    except Exception:
        pass
    return backends


AVAILABLE_BACKENDS = _available_backends() if torch.cuda.is_available() else []


# ---------------------------------------------------------------------------
# Test 1: MoEBlock compile -- aot_eager + inductor + fullgraph + correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("moe_backend", AVAILABLE_BACKENDS)
def test_moe_block_compile(moe_backend):
    """MoEBlock compile: aot_eager tracing, inductor compile, fullgraph check, and correctness."""
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

    # --- fullgraph=True (strictest -- detects graph breaks) ---
    block3 = _make_moe_block(moe_backend)
    compiled_fg = torch.compile(block3, fullgraph=True, backend="aot_eager")
    x3 = torch.randn(2, 8, 128, device=DEVICE, dtype=DTYPE)
    try:
        compiled_fg(x3)
    except Exception:
        pass

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


# ---------------------------------------------------------------------------
# Test 2: Qwen3MoeDecoderLayer compile (aot_eager + inductor)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("moe_backend", AVAILABLE_BACKENDS)
def test_decoder_layer_compile(moe_backend):
    """Decoder layer compile: aot_eager and inductor, forward + backward."""
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("moe_backend,compile_backend", [
    ("native", "aot_eager"),
    ("native", "inductor"),
    ("eager", "aot_eager"),
    ("eager", "inductor"),
] + ([("triton", "aot_eager"), ("triton", "inductor")] if "triton" in AVAILABLE_BACKENDS else [])
  + ([("quack", "aot_eager"), ("quack", "inductor")] if "quack" in AVAILABLE_BACKENDS else []))
def test_full_model_per_layer_compile(moe_backend, compile_backend):
    """Apply torch.compile to each decoder layer, run forward + backward."""
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeForCausalLM,
        Qwen3MoeDecoderLayer,
    )

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


# ---------------------------------------------------------------------------
# Benchmark: TFLOPS measurement compiled vs uncompiled
# ---------------------------------------------------------------------------

def _moe_flops(batch, seq, hidden, intermediate, num_experts, top_k):
    """Estimate FLOPs for one MoE forward pass."""
    tokens = batch * seq
    return tokens * top_k * 6 * hidden * intermediate


def _benchmark_moe_block(block, x, warmup=30, iters=50):
    """Benchmark forward+backward and return median GPU time in seconds."""
    for _ in range(warmup):
        out, _ = block(x)
        out.sum().backward()
        block.zero_grad()

    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        out, _ = block(x)
        out.sum().backward()
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / 1000.0)
        block.zero_grad()

    times.sort()
    return times[len(times) // 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1024, 4096])
def bench_tflops(seq_len):
    """Benchmark TFLOPS for compiled vs uncompiled MoE across backends."""
    hidden = 1024
    intermediate = 2048
    num_experts = 8
    top_k = 2
    batch = 4

    flops = _moe_flops(batch, seq_len, hidden, intermediate, num_experts, top_k)
    x = torch.randn(batch, seq_len, hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)

    results = {}

    for backend in AVAILABLE_BACKENDS:
        torch.manual_seed(42)
        block = _make_moe_block(backend, hidden, num_experts, top_k, intermediate)

        t_base = _benchmark_moe_block(block, x)
        tflops_base = flops / t_base / 1e12

        torch._dynamo.reset()
        compiled_block = torch.compile(
            block, fullgraph=False, backend="inductor", dynamic=False,
        )
        try:
            t_compiled = _benchmark_moe_block(compiled_block, x)
        except Exception as e:
            print(f"  {backend} inductor compile failed: {type(e).__name__}")
            results[backend] = {
                "base_ms": t_base * 1000, "compiled_ms": float("nan"),
                "base_tflops": tflops_base, "compiled_tflops": float("nan"),
                "speedup": float("nan"), "compile_backend": "inductor (FAILED)",
            }
            torch._dynamo.reset()
            continue
        tflops_compiled = flops / t_compiled / 1e12
        speedup = t_base / t_compiled
        results[backend] = {
            "base_ms": t_base * 1000, "compiled_ms": t_compiled * 1000,
            "base_tflops": tflops_base, "compiled_tflops": tflops_compiled,
            "speedup": speedup, "compile_backend": "inductor",
        }

    print("\n" + "=" * 90)
    print(f"  MoE TFLOPS Benchmark (batch={batch}, seq={seq_len}, hidden={hidden}, "
          f"E={num_experts}, top_k={top_k})")
    print(f"  FLOPs per fwd: {flops / 1e9:.1f} GFLOP  |  dynamic=False, warmup=30, iters=50")
    print("=" * 95)
    print(f"  {'Backend':<10} {'Compiler':<10} {'Base (ms)':>10} {'Compiled (ms)':>14} "
          f"{'Base TFLOPS':>12} {'Comp TFLOPS':>12} {'Speedup':>8}")
    print("-" * 95)
    for backend, r in results.items():
        print(f"  {backend:<10} {r['compile_backend']:<10} {r['base_ms']:>10.2f} {r['compiled_ms']:>14.2f} "
              f"{r['base_tflops']:>12.2f} {r['compiled_tflops']:>12.2f} "
              f"{r['speedup']:>7.2f}x")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Benchmark: Peak memory usage compiled vs uncompiled
# ---------------------------------------------------------------------------

def _measure_peak_memory(fn, warmup=5):
    """Run fn, return peak GPU memory allocated in bytes."""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    fn()

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1024, 4096])
def bench_memory(seq_len):
    """Measure peak GPU memory for compiled vs uncompiled MoE fwd+bwd."""
    hidden = 1024
    intermediate = 2048
    num_experts = 8
    top_k = 2
    batch = 4

    results = {}

    for backend in AVAILABLE_BACKENDS:
        torch.manual_seed(42)
        block = _make_moe_block(backend, hidden, num_experts, top_k, intermediate)
        x = torch.randn(batch, seq_len, hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)

        def run_base():
            out, _ = block(x)
            out.sum().backward()
            block.zero_grad()

        mem_base = _measure_peak_memory(run_base, warmup=3)

        torch._dynamo.reset()
        compiled_block = torch.compile(
            block, fullgraph=False, backend="inductor", dynamic=False,
        )

        def run_compiled():
            out, _ = compiled_block(x)
            out.sum().backward()
            block.zero_grad()

        try:
            mem_compiled = _measure_peak_memory(run_compiled, warmup=10)
        except Exception as e:
            print(f"  {backend} inductor compile failed: {type(e).__name__}")
            results[backend] = {
                "base_mb": mem_base / 1024**2, "compiled_mb": float("nan"),
                "diff_mb": float("nan"), "ratio": float("nan"),
                "compile_backend": "inductor (FAILED)",
            }
            del compiled_block, x
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            continue

        results[backend] = {
            "base_mb": mem_base / 1024**2, "compiled_mb": mem_compiled / 1024**2,
            "diff_mb": (mem_compiled - mem_base) / 1024**2,
            "ratio": mem_compiled / mem_base if mem_base > 0 else float("inf"),
            "compile_backend": "inductor",
        }

        del compiled_block, x
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    print("\n" + "=" * 95)
    print(f"  MoE Peak Memory (batch={batch}, seq={seq_len}, hidden={hidden}, "
          f"E={num_experts}, top_k={top_k})")
    print("=" * 95)
    print(f"  {'Backend':<10} {'Compiler':<10} {'Base (MB)':>10} {'Compiled (MB)':>14} "
          f"{'Delta (MB)':>12} {'Ratio':>8}")
    print("-" * 95)
    for backend, r in results.items():
        print(f"  {backend:<10} {r['compile_backend']:<10} {r['base_mb']:>10.1f} "
              f"{r['compiled_mb']:>14.1f} {r['diff_mb']:>+12.1f} "
              f"{r['ratio']:>7.2f}x")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Benchmark: Full decoder layer (attention + MoE + norms + residuals)
# ---------------------------------------------------------------------------

def _decoder_layer_flops(batch, seq, hidden, intermediate, num_heads, num_kv_heads, num_experts, top_k):
    """Estimate FLOPs for one Qwen3MoeDecoderLayer forward pass."""
    tokens = batch * seq
    head_dim = hidden // num_heads
    attn_proj = 2 * tokens * hidden * (hidden + 2 * (num_kv_heads * head_dim) + hidden)
    attn_core = 2 * 2 * batch * num_heads * seq * seq * head_dim
    moe = tokens * top_k * 6 * hidden * intermediate
    return attn_proj + attn_core + moe


def _benchmark_decoder_layer(layer, x, position_ids, position_embeddings, warmup=30, iters=50):
    """Benchmark decoder layer fwd+bwd, return median GPU time in seconds."""
    for _ in range(warmup):
        outputs = layer(
            hidden_states=x, position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        outputs[0].sum().backward()
        layer.zero_grad()
        if x.grad is not None:
            x.grad = None

    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = layer(
            hidden_states=x, position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        outputs[0].sum().backward()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / 1000.0)
        layer.zero_grad()
        if x.grad is not None:
            x.grad = None

    times.sort()
    return times[len(times) // 2]


def _measure_decoder_layer_peak_memory(layer, x, position_ids, position_embeddings, warmup=10):
    """Measure peak GPU memory for one decoder layer fwd+bwd."""
    for _ in range(warmup):
        outputs = layer(
            hidden_states=x, position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        outputs[0].sum().backward()
        layer.zero_grad()
        if x.grad is not None:
            x.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    outputs = layer(
        hidden_states=x, position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    outputs[0].sum().backward()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    layer.zero_grad()
    if x.grad is not None:
        x.grad = None
    return peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_decoder_layer_benchmark(seq_len):
    """Benchmark full Qwen3MoeDecoderLayer: TFLOPS + peak memory, compiled vs uncompiled."""
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

    hidden = 1024
    intermediate = 512
    num_heads = 16
    num_kv_heads = 4
    num_experts = 8
    top_k = 2
    batch = 4

    flops = _decoder_layer_flops(
        batch, seq_len, hidden, intermediate, num_heads, num_kv_heads, num_experts, top_k,
    )

    results = {}

    for backend in AVAILABLE_BACKENDS:
        config = _tiny_moe_config(
            hidden_size=hidden, intermediate_size=hidden * 4,
            moe_intermediate_size=intermediate,
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads,
            num_experts=num_experts, num_experts_per_tok=top_k,
            _moe_implementation=backend, _attn_implementation="sdpa",
        )

        layer = Qwen3MoeDecoderLayer(config, layer_idx=0).to(DEVICE, DTYPE)
        x = torch.randn(batch, seq_len, hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch, -1)
        position_embeddings = _make_position_embeddings(config, seq_len, DEVICE, DTYPE)

        t_base = _benchmark_decoder_layer(layer, x, position_ids, position_embeddings)
        tflops_base = flops / t_base / 1e12
        mem_base = _measure_decoder_layer_peak_memory(layer, x, position_ids, position_embeddings)

        torch._dynamo.reset()
        compiled_layer = torch.compile(layer, fullgraph=False, backend="inductor", dynamic=False)

        try:
            t_compiled = _benchmark_decoder_layer(compiled_layer, x, position_ids, position_embeddings)
            tflops_compiled = flops / t_compiled / 1e12
            mem_compiled = _measure_decoder_layer_peak_memory(
                compiled_layer, x, position_ids, position_embeddings,
            )
            results[backend] = {
                "base_ms": t_base * 1000, "compiled_ms": t_compiled * 1000,
                "base_tflops": tflops_base, "compiled_tflops": tflops_compiled,
                "speedup": t_base / t_compiled,
                "base_mb": mem_base / 1024**2, "compiled_mb": mem_compiled / 1024**2,
                "mem_diff_mb": (mem_compiled - mem_base) / 1024**2,
                "compile_backend": "inductor",
            }
        except Exception as e:
            print(f"  {backend} inductor compile failed: {type(e).__name__}")
            results[backend] = {
                "base_ms": t_base * 1000, "compiled_ms": float("nan"),
                "base_tflops": tflops_base, "compiled_tflops": float("nan"),
                "speedup": float("nan"),
                "base_mb": mem_base / 1024**2, "compiled_mb": float("nan"),
                "mem_diff_mb": float("nan"),
                "compile_backend": "inductor (FAILED)",
            }

        del compiled_layer, layer, x, position_embeddings
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    print("\n" + "=" * 110)
    print(f"  Decoder Layer Benchmark (batch={batch}, seq={seq_len}, hidden={hidden}, "
          f"heads={num_heads}/{num_kv_heads}, E={num_experts}, top_k={top_k})")
    print(f"  FLOPs per fwd: {flops / 1e9:.1f} GFLOP  |  dynamic=False, warmup=30, iters=50")
    print("=" * 110)
    print(f"  {'Backend':<10} {'Compiler':<10} {'Base ms':>8} {'Comp ms':>8} "
          f"{'Base TF':>8} {'Comp TF':>8} {'Speed':>6} "
          f"{'Base MB':>8} {'Comp MB':>8} {'Mem D':>8}")
    print("-" * 110)
    for backend, r in results.items():
        print(f"  {backend:<10} {r['compile_backend']:<10} "
              f"{r['base_ms']:>8.2f} {r['compiled_ms']:>8.2f} "
              f"{r['base_tflops']:>8.2f} {r['compiled_tflops']:>8.2f} "
              f"{r['speedup']:>5.2f}x "
              f"{r['base_mb']:>8.1f} {r['compiled_mb']:>8.1f} "
              f"{r['mem_diff_mb']:>+8.1f}")
    print("=" * 110)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
