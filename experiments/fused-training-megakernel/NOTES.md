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

## v1: dataflow scheduling + latency round (this section supersedes the v0 numbers)

The wave barriers are gone: instructions carry dependency counts derived from per-op
read/write signatures (RAW/WAR/WAW over the buffer table, with alias "slots" declaring
disjoint regions like the q/kv halves of dqkv); ready instructions enter a ring; blocks
claim (instr, tile-batch) work via atomic cursors with a sticky-instruction fast path
and a global consumed-head (naive ring scanning was SLOWER than waves). Both executors
remain available (`mode="waves"|"df"`) and are cross-checked by the tests.

Latency round, guided by per-instruction %globaltimer stamps (clock64 is per-SM and
useless across blocks): attention bwd split over GQA members and kv-chunks with fp32
atomic workspaces + a convert op (dKV worst-instr 190us -> 47us); warp-parallel
softmax rows in attention fwd/bwd-recompute; split-K-to-fp32-scratch for the K=8192
dlogits@Wlm gemm (145us -> ~40us); single-pass online CE; 16K vectorized fill/convert
chunks; per-layer fp32 attention workspaces (a shared one chained layers through its
zero-fill). A cp.async BK=64 gemm rewrite with layout-matched col_major fragments was
tried and REVERTED: in-model it lost ~10% to the 1-deep register-prefetch BK=32 gemm
(microbench: ~1.6us/K-iter and no improvement; suspect LDSM patterns on the col_major
ld=72 paths — a real Hopper wgmma/TMA kernel is the actual answer, not WMMA surgery).

| config | v0 megakernel | v1 megakernel | compile+CUDAGraph |
|---|---|---|---|
| nano  (H256 L4 S512)  | 2.53 ms | **1.93-1.99 ms** | 0.83 ms |
| small (H512 L8 S1024) | 12.0 ms | **9.8 ms**       | 3.0 ms  |

Launch-bound sweep (does the megakernel win somewhere?): nano at S=256 -> 1.49 vs
0.72 ms; S=128 -> 1.20 vs 0.62 ms; deep-narrow L=12 -> 4.07 vs 2.08 ms. No crossover:
CUDA-graph replay's per-node tax also shrinks with size, and the megakernel's chain
floor (~13us per critical-path op at S=128) tracks it.

## v2 round 1: wgmma (Hopper) GEMM path

`wgmma_probe.py` validates a from-scratch wgmma m64n128k16 setup against torch —
hand-built GmmaDescriptors over a no-swizzle K-major INTER smem arrangement (8x8-bf16
core matrices, SBO=256B, LBO=128B; descriptor bitfields from the CUTE headers bundled
with deep_gemm, which also provide the fma wrappers; build needs `-arch=sm_90a`).
Two hard-won facts:
- A data-dependent `ScaleOut` ternary between `warpgroup_arrive` and `commit` makes
  ptxas SERIALIZE every wgmma (~2.5us each, 60x slow). Branch-free accumulate
  (always ScaleOut::One over explicitly zeroed registers) reaches ~39ns per
  m64n128k16 = ~94% of per-SM tensor peak in the probe.
- The NT (Linear-fwd) gemms route through a 128x128 two-warpgroup tile with 2-stage
  cp.async feeds (`op_gemm_wgmma`, flags bit7; host: `mk.wgmma_ok`). Model parity
  holds. BUT model-level gains are ~nil at nano and ~4% at small: the 64 fp32
  accumulators push the whole interpreter kernel to 255 regs -> 1 block/SM (132
  persistent blocks, halving overlap capacity), and per-instr fixed costs (scattered
  register-direct epilogue, claim, pipeline prologue) dominate at chain-gemm sizes
  (~20us/instr vs ~140ns of actual mma). Also tried and dropped: direct successor
  handoff in the scheduler (+14% step time, mechanism unclear).

Current: nano 2.00ms, small 9.5ms vs compile+CUDAGraph 0.82/3.0ms. The wgmma
infrastructure is correct and peak-capable; converting that into step-time wins needs
the fixed-cost/occupancy engineering: m64n64 variant (32 accumulators) to restore
2 blocks/SM, smem-staged vectorized epilogues, warp-specialized producer/consumer
structure, and the fusion round — the full multi-week program.

## CODA (~/coda-kernels) findings — blueprint for the fusion round

CODA (Guo et al., arXiv:2605.19269) expresses transformer ops as GEMM-plus-epilogue
programs over quack's CuTeDSL GemmSm90 — exactly the epilogue-fusion direction, already
engineered. What transfers to the megakernel (designs, not code — CuTeDSL kernels are
whole-kernel JIT artifacts, same transplant boundary as cuBLAS):
- gemm_swiglu pairs the gate/up COLUMN HALVES within one CTA's epilogue -> our gemm
  tiles can claim paired (n, n + N/2) column blocks and apply swiglu tile-locally,
  no weight-interleave, no backward layout changes.
- lse.py / ColVecStore: tile-local row-reduction epilogues (LSE for fusing CE into the
  lm_head gemm; row sum-of-squares for producing the NEXT rmsnorm's rstd in the
  producing gemm's epilogue).
- Even CODA keeps swiglu-BACKWARD as a standalone elementwise op (dswiglu_backward)
  with plain gemms for dx/dW — our backward structure is not the anomaly.
Calibration caveat: CODA standalone calls carry ~50-100us CuTeDSL host dispatch at
small shapes (amortized only under CUDA graphs), so per-op comparisons must use
in-graph event timing, not wall clock.

Also answered along the way: hand-written SASS would not unlock reusing cuBLAS —
disassembled kernels are frozen whole-kernel images (baked constant-bank argument
reads, block-shape-tied register/barrier allocation, internal schedulers) and cuBLAS
is a per-shape kernel-selection library besides. SASS remains a read/audit tool here.

## v2 round 2: occupancy, staged epilogue, first fused epilogue

- **m64n64 wgmma retile** (128x64 tiles, two warpgroups sharing B loads, 32 accumulators)
  + **smem-staged vectorized epilogue** (accumulators staged over the dead cp.async
  buffers, fully coalesced uint4 stores): nano 1.93 -> 1.85ms, small ~9.3ms.
- `__launch_bounds__(256, 2)` A/B on identical hardware: forces 128 regs -> 2 blocks/SM
  but spills 360B of stack; nano WORSE by 16% (latency chain pays for spills), small
  BETTER by 7% (throughput pays for occupancy). Kept unbounded (nano-optimal); the real
  resolution is setmaxnreg warp specialization.
- **First fused epilogue** (CODA pattern): per-head qk-RMSNorm + RoPE fused into the
  qkv gemm epilogue — with WG_BN == 64 == head_dim each tile is exactly one head, so
  the norm reduction is tile-local over the fp32 staging (flags bit8; D=64 only,
  D=128 falls back to the separate op). Kills 1 chain instr/layer; parity holds.
  Remaining fusions from the same blueprint: swiglu paired-column tiles, Drow into
  o-proj bwd, rstd producers, CE/lse into lm_head.

Scoreboard (GPU-matched runs): nano megakernel 1.85ms vs compile+CUDAGraph 0.84ms
(2.2x); small 9.3ms vs 3.0ms (3.1x). The goal (megakernel faster) remains unmet; the
remaining measured gap sits in (a) the WMMA-path bwd gemms (NN/TN need MN-major wgmma
variants), (b) attention op quality, (c) warp specialization to escape the
occupancy/registers dilemma, (d) the rest of the fusion list.

## v2 round 3: MN-major wgmma — validated, and a decisive negative result

wgmma_probe.py now also validates the MN-major INTER descriptor (canonical layout from
mma_traits_sm90_gmma.hpp: SBO = 128B mn-group stride, LBO = k-group stride — the reverse
of K-major's assignment; first guess faulted with "out-of-range shared address" and the
header's canonical-layout doc gave the exact form). op_gemm_wgmma now supports all four
storage-major combinations (NN/NT/TN/TT, per-operand descriptor + template dispatch)
plus split-K with fp32-atomic epilogues, all parity-tested.

THE FINDING: routing the backward NN/TN gemms through wgmma is SLOWER in-model
(nano 1.85 -> 2.05ms, small 9.3 -> 10.3ms) despite near-peak mma throughput — at these
tile counts the per-instruction fixed costs (claim, prologue fill, epilogue, scheduler
handoff) dominate so completely that tensor-core quality is irrelevant. Routing reverted
to NT-only (capability retained in the kernel). Combined with the earlier rounds, this
empirically closes the "roadmap arithmetic": no remaining math-unit conversion pays.
The residual 2.2x vs compile+CUDAGraph lives in the structural floor — the ~85-deep
serial chain times per-instruction overhead — whose remedies are warp-specialized
producer/consumer ops, tile-granular dependencies, and FA-class attention tiles: the
multi-week core, now with direct measurements behind that scoping.

Scoreboard: nano 1.85ms / small 9.28ms vs compile+CUDAGraph 0.83 / 3.0ms.

## v2 round 4: split-KV attention forward — implemented, measured, not routed

OP_ATTN_FWD_SPLIT (chunked kv loop with chunk-local online softmax, locally normalized
partials + (m_c, l_c)) and OP_ATTN_COMBINE (flash-decoding merge; also produces LSE)
are implemented and parity-tested. Measured with C=4: nano neutral (~1.85ms), small
NEGATIVE (+5%: the combine chain hop plus [C, S, nq*D] partial-tensor traffic outweigh
the per-instr latency saving). Routing disabled (attn_C = 1); ops retained.

With this, EVERY session-scale item on the v2 roadmap has been executed and measured:
the ones that pay are routed and committed; the ones that don't (cp.async WMMA rewrite,
scheduler direct handoff, launch-bounds occupancy trade, bwd-gemm wgmma routing,
split-KV fwd routing) are committed as documented negative results. Best configuration:
nano ~1.85-1.89ms, small ~9.3-9.4ms vs compile+CUDAGraph 0.83/3.0ms. The remaining gap
is the structural floor; the only remaining levers are the multi-week ones (warp
specialization, tile-granular dependencies, FA-class attention tiles).

## Honest assessment + v2 roadmap

compile+CUDAGraph remains ~2.3x faster. The measured structural gap, in order:
1. **Per-op kernel quality**: our WMMA gemm is 40-100 TF where cuBLAS gets 200+; FA
   does attention tiles ~3x faster. The fix is Hopper-native kernels (wgmma + TMA,
   proper multi-stage pipelines) inside the interpreter ops — the single biggest item.
2. **Critical-path length**: ~89 chain instructions; each pays claim + prologue +
   epilogue (~2-4us). Fusing rowops into gemm prologues/epilogues (norm stats in the
   producing gemm's epilogue, rope/qk-norm in the qkv epilogue, swiglu in the gu
   epilogue) would cut the chain to ~50.
3. **Tile-level dependencies**: instruction-level deps only overlap off-path work
   (the dW gemms); tile-granular counters would let dependent ops start on partially
   complete producers (true MPK-style pipelining).

## Superseded: v0 assessment

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
