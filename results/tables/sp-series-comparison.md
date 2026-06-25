# Qwen3-8B sp-series: Ulysses (sequence) parallelism vs throughput & MFU

8×H100_SXM · Qwen3-8B · FSDP2 · `flash_attention_3` · 64k `sample_packing_sequence_len`
(packing-matched across all three) · grad-checkpointing on · 3-whole-step steady-state
window per run (3 fwd + 3 bwd, no clipped steps), rank 0 · profiler steps 15–20.

The 8 GPUs are split between Ulysses (sp) and FSDP shard (dp), product = 8, so the
global batch = `data_parallel_shard_size` (sp1 → 8 seqs, sp2 → 4, sp8 → 1).

| Metric | **sp1** (ulysses=1, dp=8) | **sp2** (ulysses=2, dp=4) | **sp8** (ulysses=8, dp=1) |
|---|---|---|---|
| **Throughput** (global) | **~64.1k tok/s** | ~60.5k tok/s | ~51.5k tok/s |
| **MFU** (model FLOPs) | **51.3%** | 48.4% | 41.2% |
| MFU (causal-attn variant) | 45.5% | 43.0% | 36.6% |
| **Compute busy** | **98.8%** | 95.2% | 81.3% |
| **Comm busy** | 9.4% | 15.5% | 22.6% |
| GPU idle | **0.9%** | 1.7% | 6.2% |
| **Exposed (non-overlapped) comm** | **0.2%** (57 ms, 3% of comm) | 3.1% (355 ms, 20%) | 12.5% (428 ms, 55%) |
| Dominant exposed | FSDP `AllGather` 16 ms (1%) | Ulysses `SendRecv` 280 ms (48%) | `SendRecv` 181 ms (74%) + FSDP AllGather 125 ms + ReduceScatter 119 ms |
| `SendRecv` (Ulysses A2A) busy | none | 586 ms | 244 ms (74% exposed) |
| Fits 64k seqlen? | yes, no OOM | yes | yes |

## Takeaway

- **Pure-FSDP sp1 dominates every axis**: highest throughput (64.1k tok/s), 98.8%
  compute-busy, 51.3% MFU, and comm essentially free (0.2% exposed) — FSDP
  `AllGather`/`ReduceScatter` hide almost completely behind the large per-GPU compute.
- Adding Ulysses is **monotonically worse** (sp1 -> sp2 -> sp8): splitting the sequence
  shrinks per-GPU compute while growing the all-to-all (`SendRecv`), which overlaps
  progressively worse (48% -> 74% exposed). At sp8 there's so little compute left that
  even FSDP collectives become exposed, driving idle to 6.2% and MFU down to 41%.
- **64k seqlen fits with pure FSDP + gradient checkpointing — no OOM** — so the
  sequence-parallel overhead buys nothing at this 8k per-sample / 64k-packed scale.
  Ulysses only earns its keep when activations genuinely don't fit a single GPU.

## MFU method

`model_flops_per_token = 6N + 12*L*h*s`, with N = 8.19e9 (Qwen3-8B total params),
L = 36, h = 4096, s = 8000 (per-sample attention span; attention is block-diagonal
over the packed 64k sequence). Attention ~= 22% of per-token FLOPs.

`MFU = (model_flops_per_token * global_tok_per_s) / (8 * 989.5 TFLOP/s)`, where
989.5 TFLOP/s is the H100 SXM dense BF16 tensor-core peak (no sparsity; training
runs `enable_mixed_precision: true`). Peak across 8 GPUs = 7916 TFLOP/s.

The "causal-attn variant" row halves the attention term (`6*L*h*s`) to credit
flash-attention's triangular-mask savings; the ranking is unchanged.

Sources: Qwen3-8B config (huggingface.co/Qwen/Qwen3-8B) · NVIDIA H100 datasheet.

## Per-run trace dirs (results/)

- sp1: `qwen3_8b_sp1-20260624-211559` (config `qwen3_8b_sp1.yaml`)
- sp2: `qwen3_8b_sp2-20260624-204659` (config `qwen3_8b_sp2.yaml`)
- sp8: `qwen3_8b_sp8-20260624-210358` (config `qwen3_8b_sp8.yaml`)
