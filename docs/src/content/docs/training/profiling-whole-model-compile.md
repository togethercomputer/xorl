---
title: Profiling — Qwen3-1.7B whole-model-compile (single H100)
---

`torch.profiler` analysis of the Qwen3-1.7B single-GPU config
(`examples/local/dummy/configs/full/qwen3_1.7b.yaml`) on this branch's regime:
per-layer `torch.compile` + `reduce-overhead` CUDA graphs (`XORL_COMPILE_REDUCE_OVERHEAD=1`),
full-weight AMP (fp32 master + `torch.autocast(bf16)`), FlashAttention-4 (cute).

Captured on a single H100_SXM via `vast/launch.sh`, steady-state window (profiler
steps 15–20, past the step-0 compile spike). Trace: `results/<stem>-<ts>/trace/*.pt.trace.json.gz`.

## Per-step training time

Measured on ProfilerStep#15–17 (steady state):

| Metric | Value |
|---|---|
| **GPU compute / step** (Σ kernel time — profiler-independent) | **192.5 ms** |
| Wall / step, **unprofiled** (40.6k tok/s ÷ 8192 tok/step) | **~202 ms** |
| Wall / step, **under profiler** (ProfilerStep span) | ~225 ms |
| dev, non-compiled (reference) | 212 ms |

**vs dev:** compiled ≈ **202 ms** unprofiled vs **212 ms** non-compiled → **~10 ms (~5%)
faster**. The win is small because the step is compute-bound: `torch.compile` + CUDA graphs
mainly collapse per-kernel launch overhead, but do not shrink the matmuls/attention that
fill ~192 ms. The ~225 ms profiled wall is inflated — the profiler records ~416k events on
the host thread, which balloons the host-side bubbles; **GPU-busy 192.5 ms is the reliable
figure**, and real non-compute overhead is ~9.5 ms/step (202 − 192.5, ≈4.7%).

## Where the 192.5 ms of GPU compute goes

| Time | Share | Group |
|---|---|---|
| 106.9 ms | 56% | cutlass GEMMs (`nvjet`) — QKV/O projections + MLP |
| 28.0 ms | 15% | triton (compiled: rmsnorm / silu / casts) |
| 26.4 ms | 14% | fp32-master AdamW (`multi_tensor_apply`) — notably heavy |
| 18.8 ms | 10% | FA4 attention (cute) |
| ~12 ms | ~6% | elementwise / reductions / other |

## Non-compute latency that remains (~9.5 ms real / 32.8 ms under profiler)

GPU idle is **14.6% of the profiled wall**. Relative attribution of the bubbles
(profiled, GPU-idle gaps >20 µs unless noted):

1. **FA4 custom-op host bubbles — 11.6 ms across 28 gaps (one per decoder layer).**
   The single biggest non-compute cost. `xorl_fa4::fwd` sits **outside** the compiled /
   CUDA-graphed region (the custom op breaks graph capture), so every layer pays exposed
   host launch latency (~0.4 ms each) before the attention kernel fires.
2. **`cudaStreamSynchronize` — 3.0 ms** (mostly one 2.8 ms stall): per-step
   `loss`/`grad-norm`/`.item()` logging forces a D2H sync.
3. **Eager `aten::mm` launch gaps — 2.1 ms** (56 small gaps; matmuls outside graphs, e.g. lm_head).
4. **Optimizer boundary — 2.7 ms:** `AdamW.step` host bubble (1.74 ms) + `zero_grad` (0.94 ms).
5. **Step boundary / dataloader HtoD — 1.2 ms; per-layer dtype-cast — 0.86 ms; grad-norm
   `_foreach_norm` — 0.64 ms.**
6. **~1837 small inter-kernel launch gaps (<20 µs) — 4.5 ms:** death-by-a-thousand-cuts
   launch latency on the portions not inside a captured graph.

## Biggest lever

Pulling FA4 inside the CUDA graph — e.g. `XORL_COMPILE_WHOLE_BACKBONE=1`, which this branch
adds to fold the whole layer loop into one graph — would reclaim most of the ~11.6 ms FA4
bubble plus the small launch gaps. The run above used **per-layer** compile only; the
whole-backbone / whole-step paths are the obvious next experiment.

## Reproduce

```bash
cd vast && ./launch.sh examples/local/dummy/configs/full/qwen3_1.7b.yaml
# traces -> results/<stem>-<ts>/trace/*.pt.trace.json.gz  (view at https://ui.perfetto.dev)
```

## Notes on the FA4 dtype path

FA4's cute kernel rejects fp32 q/k. On the full-weight AMP path the naive rotary fallback
promotes q/k to fp32, so `model.attention_cast_bf16: true` casts them back to bf16 right
after RoPE. This was confirmed empirically here: removing it makes FA4 assert
`inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2`. The field + its trainer
plumbing are ported from dev (this branch's upstream base had dropped them).
