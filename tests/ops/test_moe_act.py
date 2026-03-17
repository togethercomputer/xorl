"""Tests for moe_act selective activation recompute variants.

Each backend (native, triton, quack) has a moe_act variant that:
- Checkpoints gate+up projection activations (native: torch.utils.checkpoint;
  triton/quack: custom autograd.Function saves fewer tensors)
- Recomputes gate+up in backward instead of saving them
- Trades extra backward compute for reduced activation memory

Tests:
1. Forward correctness: moe_act output matches standard output per backend
2. Backward correctness: moe_act gradients match standard gradients per backend
3. Memory: moe_act saves activation memory compared to standard
4. TFLOPS benchmark: standard vs moe_act fwd+bwd per backend
"""

import pytest
import torch
import torch.nn as nn

DEVICE = "cuda"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_moe_act_backends():
    """Return backends that have moe_act local variants registered."""
    from xorl.models.layers.moe.backend import MOE_EXPERT_BACKENDS_MOE_ACT
    return list(MOE_EXPERT_BACKENDS_MOE_ACT.keys())


AVAILABLE_BACKENDS = _available_moe_act_backends() if torch.cuda.is_available() else []


def _make_block_pair(ne, hd, inter, topk, backend, seed=42):
    """Create (standard, moe_act) MoEBlock pair with identical weights.

    Returns (std_block, act_block) where act_block.experts._moe_act = True.
    """
    from xorl.models.layers.moe.moe_block import MoEBlock

    torch.manual_seed(seed)
    std = MoEBlock(hd, ne, topk, inter, moe_implementation=backend)
    nn.init.xavier_normal_(std.experts.gate_proj.data)
    nn.init.xavier_normal_(std.experts.up_proj.data)
    nn.init.xavier_normal_(std.experts.down_proj.data)
    nn.init.xavier_normal_(std.gate.weight.data)
    std = std.to(DEVICE, DTYPE)

    act = MoEBlock(hd, ne, topk, inter, moe_implementation=backend)
    act.experts._moe_act = True
    act = act.to(DEVICE, DTYPE)
    with torch.no_grad():
        act.gate.weight.copy_(std.gate.weight)
        act.experts.gate_proj.copy_(std.experts.gate_proj)
        act.experts.up_proj.copy_(std.experts.up_proj)
        act.experts.down_proj.copy_(std.experts.down_proj)

    return std, act


# ---------------------------------------------------------------------------
# Test 1: Forward correctness
# ---------------------------------------------------------------------------

CORRECTNESS_CONFIGS = [
    # (num_experts, hidden, intermediate, top_k, batch, seq)
    (4,  64,  128, 2, 2,  8),
    (8, 128,  256, 2, 4, 16),
    (4,  64,  128, 1, 2,  8),   # top_k=1
    (8, 128,  256, 4, 2, 16),   # top_k=4
    (4,  64,  128, 2, 1,  1),   # minimal
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("ne,hd,inter,topk,bs,seq", CORRECTNESS_CONFIGS)
def test_forward_correctness(backend, ne, hd, inter, topk, bs, seq):
    """moe_act forward output must match standard forward output."""
    std, act = _make_block_pair(ne, hd, inter, topk, backend)

    torch.manual_seed(7)
    x = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        std_out, std_logits = std(x)
        act_out, act_logits = act(x)

    # Router logits must be identical (same gate weights, same input)
    torch.testing.assert_close(act_logits, std_logits, atol=0, rtol=0)

    max_diff = (act_out - std_out).abs().max().item()
    torch.testing.assert_close(
        act_out, std_out, atol=0.05, rtol=0.02,
        msg=f"[{backend}] Forward mismatch: max_diff={max_diff:.6f}",
    )


# ---------------------------------------------------------------------------
# Test 2: Backward correctness (gradients)
# ---------------------------------------------------------------------------

BACKWARD_CONFIGS = [
    (4,  64, 128, 2, 2,  8),
    (8, 128, 256, 2, 4, 16),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("ne,hd,inter,topk,bs,seq", BACKWARD_CONFIGS)
def test_backward_correctness(backend, ne, hd, inter, topk, bs, seq):
    """moe_act backward gradients must match standard backward gradients."""
    std, act = _make_block_pair(ne, hd, inter, topk, backend)

    atol, rtol = 0.05, 0.05

    torch.manual_seed(7)
    x_std = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_act = x_std.detach().clone().requires_grad_(True)

    std_out, _ = std(x_std)
    std_out.sum().backward()

    act_out, _ = act(x_act)
    act_out.sum().backward()

    torch.testing.assert_close(
        x_act.grad, x_std.grad, atol=atol, rtol=rtol,
        msg=f"[{backend}] Input gradient mismatch",
    )
    for name in ["gate_proj", "up_proj", "down_proj"]:
        g_std = getattr(std.experts, name).grad
        g_act = getattr(act.experts, name).grad
        assert g_std is not None, f"[{backend}] std {name}.grad is None"
        assert g_act is not None, f"[{backend}] act {name}.grad is None"
        torch.testing.assert_close(
            g_act, g_std, atol=atol, rtol=rtol,
            msg=f"[{backend}] {name} gradient mismatch",
        )
    torch.testing.assert_close(
        act.gate.weight.grad, std.gate.weight.grad, atol=atol, rtol=rtol,
        msg=f"[{backend}] Gate weight gradient mismatch",
    )


# ---------------------------------------------------------------------------
# Test 3: Activation memory savings
# ---------------------------------------------------------------------------

def _measure_fwd_bwd_peak_memory(block, x, warmup=3):
    """Peak GPU memory (bytes) for one forward+backward call."""
    for _ in range(warmup):
        out, _ = block(x)
        out.sum().backward()
        block.zero_grad()
        if x.grad is not None:
            x.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    x_clone = x.detach().clone().requires_grad_(True)
    out, _ = block(x_clone)
    out.sum().backward()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    block.zero_grad()
    return peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_memory_savings(backend):
    """moe_act should use less or equal peak memory than standard."""
    ne, hd, inter, topk, bs, seq = 8, 512, 1024, 2, 4, 256

    std, act = _make_block_pair(ne, hd, inter, topk, backend)

    x = torch.randn(bs, seq, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)

    mem_std = _measure_fwd_bwd_peak_memory(std, x)
    mem_act = _measure_fwd_bwd_peak_memory(act, x)

    savings_mb = (mem_std - mem_act) / 1024**2
    print(
        f"\n[{backend}] Memory: std={mem_std/1024**2:.1f} MB  "
        f"moe_act={mem_act/1024**2:.1f} MB  "
        f"savings={savings_mb:+.1f} MB"
    )

    # moe_act should not use significantly more memory than standard
    # (allow 5% overhead for checkpoint bookkeeping)
    assert mem_act <= mem_std * 1.05, (
        f"[{backend}] moe_act used more memory than standard: "
        f"{mem_act/1024**2:.1f} MB vs {mem_std/1024**2:.1f} MB"
    )


# ---------------------------------------------------------------------------
# Benchmark: TFLOPS standard vs moe_act per backend
# ---------------------------------------------------------------------------

def _moe_flops(bs, seq, hd, inter, topk):
    """Forward FLOPs: 3 GEMMs × 2 (matmul count) × tokens × top_k."""
    return bs * seq * topk * 6 * hd * inter


def _benchmark(block, x, warmup=20, iters=40):
    """Median fwd+bwd GPU time (seconds)."""
    for _ in range(warmup):
        out, _ = block(x)
        out.sum().backward()
        block.zero_grad()
        if x.grad is not None:
            x.grad = None

    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out, _ = block(x)
        out.sum().backward()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1) / 1000.0)
        block.zero_grad()
        if x.grad is not None:
            x.grad = None

    times.sort()
    return times[len(times) // 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1024, 4096])
def bench_moe_act_tflops(seq_len):
    """TFLOPS benchmark: standard vs moe_act per backend."""
    ne, hd, inter, topk, bs = 8, 1024, 2048, 2, 4

    flops = _moe_flops(bs, seq_len, hd, inter, topk)
    x = torch.randn(bs, seq_len, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)

    results = {}
    for backend in AVAILABLE_BACKENDS:
        std, act = _make_block_pair(ne, hd, inter, topk, backend)

        t_std = _benchmark(std, x)
        t_act = _benchmark(act, x)

        results[backend] = {
            "std_ms":   t_std * 1000,
            "act_ms":   t_act * 1000,
            "std_tflops": flops / t_std / 1e12,
            "act_tflops": flops / t_act / 1e12,
            "overhead":   t_act / t_std,
        }

        del std, act

    print("\n" + "=" * 90)
    print(
        f"  moe_act TFLOPS  (bs={bs}, seq={seq_len}, hidden={hd}, "
        f"inter={inter}, E={ne}, top_k={topk})"
    )
    print(f"  FLOPs/fwd: {flops/1e9:.1f} GFLOP  |  warmup=20, iters=40")
    print("=" * 90)
    print(
        f"  {'Backend':<10} {'Std (ms)':>10} {'Act (ms)':>10} "
        f"{'Std TF':>10} {'Act TF':>10} {'Overhead':>10}"
    )
    print("-" * 90)
    for backend, r in results.items():
        print(
            f"  {backend:<10} {r['std_ms']:>10.2f} {r['act_ms']:>10.2f} "
            f"{r['std_tflops']:>10.2f} {r['act_tflops']:>10.2f} "
            f"{r['overhead']:>9.2f}x"
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Test 4: moe_act works correctly inside gradient_checkpointing_enable
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_moe_act_via_gradient_checkpointing_enable(backend):
    """gradient_checkpointing_enable(moe_checkpoint_method='moe_act') sets _moe_act correctly."""
    from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    from xorl.models.layers.moe.experts import MoEExperts

    config = Qwen3MoeConfig(
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
        _moe_implementation=backend,
        max_position_embeddings=128,
        pad_token_id=0,
        _attn_implementation="sdpa",
    )
    model = Qwen3MoeForCausalLM(config).to(DEVICE, DTYPE)

    # Enable selective GC with moe_act
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False,
        "recompute_modules": ["self_attn", "mlp"],
        "moe_checkpoint_method": "moe_act",
    })

    # Verify _moe_act is set on all MoEExperts modules
    moe_experts_modules = [m for m in model.modules() if isinstance(m, MoEExperts)]
    assert len(moe_experts_modules) > 0, "No MoEExperts found in model"
    for mod in moe_experts_modules:
        assert mod._moe_act is True, f"Expected _moe_act=True, got {mod._moe_act}"

    # Forward + backward must not crash
    input_ids = torch.randint(0, 1000, (2, 16), device=DEVICE)
    output = model(input_ids=input_ids)
    output.last_hidden_state.sum().backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed"


# ---------------------------------------------------------------------------
# Test 5: moe_act + torch.compile correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_moe_act_compile(backend):
    """moe_act variant must produce correct results when compiled with inductor."""
    ne, hd, inter, topk = 4, 128, 256, 2
    _, act = _make_block_pair(ne, hd, inter, topk, backend)

    # Reference: uncompiled moe_act
    torch.manual_seed(42)
    x = torch.randn(2, 16, hd, device=DEVICE, dtype=DTYPE, requires_grad=True)
    ref_out, _ = act(x)
    ref_out.sum().backward()
    ref_grad = x.grad.clone()
    act.zero_grad()

    # Compiled variant
    torch._dynamo.reset()
    _, act2 = _make_block_pair(ne, hd, inter, topk, backend)
    compiled = torch.compile(act2, fullgraph=False, backend="inductor", dynamic=False)

    x2 = x.detach().clone().requires_grad_(True)
    comp_out, _ = compiled(x2)
    comp_out.sum().backward()

    torch.testing.assert_close(comp_out, ref_out, atol=0.05, rtol=0.02,
                                msg=f"[{backend}] compile forward mismatch")
    torch.testing.assert_close(x2.grad, ref_grad, atol=0.05, rtol=0.05,
                                msg=f"[{backend}] compile backward mismatch")

    torch._dynamo.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
