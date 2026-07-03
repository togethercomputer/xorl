# Fused training megakernel (single GPU): a true one-kernel fwd+bwd

A TRUE fused megakernel for training: ONE persistent CUDA kernel executes the entire
forward+backward pass of a Qwen3-architecture model — embedding, L decoder layers
(RMSNorm, fused-QKV, per-head qk-RMSNorm + RoPE, causal GQA flash attention, o-proj,
SwiGLU MLP), final norm, lm_head, cross-entropy, and the complete backward down to every
weight gradient — with zero kernel boundaries and zero CPU involvement mid-step. This
goes beyond the whole-step CUDAGraph capture (`train.enable_cudagraph_step`), which
replays many kernels with one launch but keeps every kernel boundary.

## Architecture

- **Persistent cooperative kernel** (`megakernel.cu`): blocks sized to fill the GPU
  (H100: 264 blocks x 256 threads at 100KB dynamic smem), launched with
  `cudaLaunchCooperativeKernel` so `grid.sync()` is available.
- **In-kernel interpreter**: the host (`mk.py Program`) builds an instruction stream —
  each instruction = (op, ntiles, buffer-table indices + shape ints) — grouped into
  dependency-free *waves*. Blocks self-schedule (instr, tile) work items within a wave;
  `grid.sync()` (~1.7us) separates waves. A full nano fwd+bwd is 84 waves.
- **Device op library** (`ops.cuh`, `attention.cuh`): templated bf16 WMMA GEMM (all
  layout variants for fwd / dX / dW, residual-add + fp32-out + accumulate epilogues,
  register-prefetch software pipelining, split-K with fp32 atomics for small dW
  matrices), RMSNorm fwd/bwd, warp-parallel per-head qk-RMSNorm+RoPE fwd/bwd (smem-staged
  weight-grad atomics), SwiGLU fwd/bwd, embedding gather/scatter-add, materialized-logits
  CE fwd/bwd, and flash attention fwd + FA2-style two-pass bwd (dKV pass + dQ pass, P
  recomputed from LSE, GQA, D in {64,128}, causal).
- **Dtypes**: params/activations bf16, all accumulation fp32, weight grads fp32.
  Gradient zeroing happens IN the kernel (wave 0, overlapped with the embedding gather).
- **Fixed shapes**, optimizer outside (goal scope = the fwd+bwd pass).

## Correctness (test_ops.py, test_model.py)

- Every op unit-tested vs PyTorch: GEMM layouts exact at fp32, attention fwd/bwd within
  bf16 tolerance (incl. GQA, D=128, ragged S).
- Full-model gradient parity vs a pure-PyTorch fp32 reference: loss matches to <2e-3
  rel; EVERY weight gradient within 2.2% max-rel (bf16-appropriate), across two configs.
- Rerun-stable up to fp32-atomic summation order (loss/norm-grads use atomics, same
  caveat as FA backward); 40 steps of raw SGD on megakernel gradients drives loss
  9.05 -> 5.6 (it learns).

## Performance (H100, median of 50 steps, bench.py)

| config | eager | torch.compile | compile+CUDAGraph | megakernel |
|---|---|---|---|---|
| nano: H256 L4 S512 V8192 | 10.5 ms | 2.25 ms | 0.83 ms | **2.53 ms** |
| small: H512 L8 S1024 V16k | 19.7 ms | 4.08 ms | 2.99 ms | **12.0 ms** |

v0 -> current step time: nano 6.56 -> 2.53 ms, small 30.9 -> 12.0 ms, via (profiled with
the in-kernel per-wave clock64 attribution, `profile_waves.py`):
1. fp32 wmma smem strides must be multiples of 4 (silent corruption otherwise);
2. vectorized coalesced tile loads for all four GEMM layouts (+2x);
3. instruction lookup scans offsets only (the 104B struct copy per scan step was
   dominating many-instruction waves);
4. warp-parallel qk-norm with smem-staged grad atomics (global-atomic contention on the
   tiny [D] grad buffers was 8% of the step);
5. split-K dW GEMMs (16-tile matrices on 264 blocks were 6% occupancy);
6. register-prefetch pipelining in the GEMM K-loop (single-buffered loads were
   latency-bound at ~33us/wave).

## Honest assessment + v1 roadmap

The megakernel beats eager (4.1x) and matches plain torch.compile at nano scale, but
compile+CUDAGraph is still ~3x faster. Remaining structural gaps, in order:
- **Wave barriers**: 84 grid.syncs serialize ops that could pipeline; narrow waves
  (a 32-tile layer GEMM on 264 blocks) idle 85% of the SMs. The fix is the inference-
  megakernel design: per-tile dependency counters instead of global waves, so tiles of
  consecutive ops overlap. This is THE v1 item — it attacks both barriers and occupancy.
- **GEMM pipeline depth**: cuBLAS uses multi-stage cp.async/TMA pipelines and tuned
  tile shapes; ours is a 1-deep register prefetch (roughly 60-100 TF vs 200+ TF at these
  sizes).
- **Attention bwd tiling**: 32x32 tiles with a serial q-loop per kv tile (24% of step).
- CE currently materializes logits (fine at small V; chunk for real vocab).

Files: `megakernel.cu` (interpreter + launcher), `ops.cuh` / `attention.cuh` (device op
library), `mk.py` (program builder), `model.py` (Qwen3 program + `MKQwen3.step()`),
`test_ops.py` / `test_model.py` (correctness), `bench.py`, `profile_waves.py`.

Env: torch 2.14 cu130 (`.venv-fa4`) + system CUDA 13.1 nvcc, sm_90. Run everything with
`CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python <script>.py` from this directory.
