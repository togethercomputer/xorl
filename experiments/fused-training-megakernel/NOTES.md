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

## v3 Phase 0: measurement round — the gap model, corrected

Plan: gated phases P0 measure → P1 fusions → P2 warp-spec prototype (go/no-go) → P3
region-watermark tile deps → P4 full warp-spec port → P5 FA-class attention → P6
negative re-runs. New meters: `profile_df.py` (consumes the per-instr %globaltimer
stamps; realized-critical-path walk splits every hop into wait vs span), `hop_bench.py`
(serial chains over distinct buffers), `trace_baseline.py` (nsys per-kernel gap
analysis — needs `--cuda-graph-trace=node`, the default traces graphs opaquely).

THE CORRECTION: the residual gap is NOT "per-instr fixed cost x chain depth". Measured:
- nano 1.79 ms = 243 us on-path wait (13.6%) + 1542 us on-path SPAN (86.4%); small
  9.4 ms = 16.1% / 83.9%. The scheduler hop is cheap (axpy chain 3.1 us/hop = 1.7 gap
  + 1.4 span; wgmma chain 5.6; wmma chain 9.0). An 85-hop chain at clean-chain cost
  would be ~0.4 ms — the rest is intrinsic op latency (serial kv-loops / k-loops /
  row loops), inflated by co-scheduling contention.
- Attention is the #1 lever at BOTH configs: megakernel ~650 us at nano vs baseline
  flash 169 us; small ~5.9 ms vs ~935 us (inductor even uses a partly math-path bwd).
- Baseline decomposition (median replay): nano 263 kernels, 737 us active / 133 us gap
  (15.3%, 0.51 us/kernel); small 507 kernels, 2833/264 (8.5%). Advantage pool at nano
  (gap + elementwise round-trip overhead + gemm shape inefficiency) ≈ 480-530 us ≥ the
  0.35x stop-rule threshold (291) -> the win is arithmetically OPEN, slack ~150-250 us.
- HARDENED baseline (foreach grad zeroing + max-autotune-no-cudagraphs, manual capture;
  all warmup must run on a side stream or AccumulateGrad stream refs break capture):
  nano 828 -> 711 us, small 2987 -> 2730 us. These are the goalposts now.
- Phase-1 addendum found by the profiler: the 5 on-path CVT hops (112 us nano / 346 us
  small) are deletable — QKNORM_ROPE_BWD and head RMSNORM_BWD read the fp32 atomic
  workspaces directly (dy_f32 flag); no dtype constraint on elementwise consumers.

## v3 Phase 1: fusion round — two keepers, two negatives, one discovery

Measured on GPU-matched runs (hardened goalposts 711/2730 us):

| change | nano | small | verdict |
|---|---|---|---|
| v2 tip | 1888 | 9391 | — |
| CVT deletion (dy_f32: qknorm-bwd + head rmsnorm-bwd read fp32 workspaces) | 1833 | 9275 | KEEP |
| Drow fused into dOatt gemm epilogue (bit10, per-layer drow buffers) | 1841 | 9230 | KEEP |
| swiglu paired-column tiles (bit9) | +88 | +297 | REMOVED |
| CE/lse partials in lm_head epilogue (bit11) | ±noise (A/B: on 1865/9275, off 1861/9317) | | KEEP ON (cheapens the CE hop ~5x for free) |

Phase-1 end state: **nano ~1860, small ~9280; chain 85→76 / 161→144.**
(Run-to-run noise on these timings is ±20-40us — single-fusion deltas below that are
calls made on direction consistency across both configs, not on one number.)

Why the fusion estimates missed (this is the durable lesson): with waits at only ~2-7us
per hop, deleting a chain hop pays only its SPAN, and only if that span isn't re-added
to an on-path producer. bit9 halved the gu tiles → doubled per-tile serial span
(parallelism traded for fusion — needs tile-granular deps first); bit11 added SFU
reductions to 512 on-path lm-gemm tiles that cost more than the CE scan they saved.
rstd-producer fusion (bit12) NOT attempted — same class as bit11, predicted negative
by two data points. Fusion at instruction granularity is now exhausted; the remaining
wins are op latency (attention, gemm pipelines) and span overlap (tile-granular deps).

TOOLING RULE learned the hard way: check `cuobjdump -res-usage` on every device-code
change. The bit9 pass-loop restructure spilled 320B of stack in the wgmma hot path and
silently slowed EVERYTHING ~8% — invisible in tests, only visible in the spill count.

DISCOVERY: the interpreter kernel is REG:255 → **1 block/SM → 132 blocks, including at
the committed v2 tip** (probably since round 3's four-major wgmma dispatch). The
"264 blocks / 2 per SM" claims in earlier sections and the mk.py claim heuristic are
stale; all round-3/4 numbers were measured at 132 blocks. Consequences: Phase 4's
1-block/SM warp-spec design gives up nothing; `claim_sz`'s hardcoded 264 is a cheap
tuning knob; and there may be a free win in getting register pressure back under 128
(worth one bounded attempt before the prototype).

## v3 Phase 2: warp-specialization prototype — GATE PASSED (GO)

`ws_probe.py` (standalone, torch load_inline, ext xorl_ws_probe): a persistent
132-block kernel where each block = 3 warpgroups — WG0 scheduler/producer, WG1+2 the
256-thread consumer op — with 2 smem pages and flag handoff, executing the same serial
256-link wgmma chain hop_bench.py measures at 5.69us/hop on the flat interpreter.

**Result: 2.77-2.80us/hop NT, 3.69 alternating-major (vs flat 5.69 / WMMA 10.38
same-GPU). The ≤4us gate PASSED; the "structural floor" does not survive warp
specialization — the warp-spec gemm hop is cheaper than the flat TRIVIAL-op hop (3.1).**

Config that wins (adopt for the Phase 4 port): producer prefetches B before polling
the predecessor-done flag; PRODUCER-side completion (consumers arrive empty
immediately; the __threadfence + release sits on the otherwise-idle producer — worth
13% vs consumer-side); plain __pipeline_wait_prior handoff (cp.async.mbarrier.arrive
neutral at these page sizes); 2 pages x 24KB; NO setmaxnreg.

Hard-won specifics (full log: results/mkv3-p2-wsprobe.md):
- setmaxnreg WORKS but is perf-negative here: consumers already fit in 126 regs and
  dec-to-40/56 spills the producer (REG 168 / STACK 80-144). Compile recipe if ever
  needed: explicit -gencode=arch=compute_90a,code=sm_90a (CUDA 13's -arch=sm_90a also
  runs a compute_90 PTX pass that rejects unguarded setmaxnreg) AND __maxnreg__ on the
  kernel (else ptxas ignores it with C7508).
- The remaining 2.8us hop = 0.64us cross-SM signal (st.release.gpu -> ld.acquire.gpu)
  + ~2us span dominated by the A-tile gmem round trip (predecessor C -> L2 -> cp.async)
  and epilogue. Tile-granular deps (consumer starts on partial predecessor) and
  same-block chaining (C stays in smem) attack exactly these — Phase 3/4 material.
- Named barriers (bar.sync 1,256) for consumer-only sync; __syncthreads would hang WG0.
- globaltimer footgun: subtract stamps in int64 BEFORE any float conversion (fp32
  granularity at ~1e14ns is 16ms — diffs read as exactly 0).
- Parity method: per-link one-step checks vs torch with an orthogonal B (max 7.8e-3);
  chain-end drift after 256 links is rounding-order walk, not error.

Implication for the plan: Phase 4's projected chain overhead at ~2.8us/hop over a
~76-hop chain is ~215us — the port is GO. An independent replication (second harness)
is running to confirm the headline number.

## v3 Phase 3: region-watermark tile deps (df2) — built, correct, and PARKED (negative)

`megakernel_df2` + host gate emission (mk.py `_build_gates`): producers publish
completed 128-row REGIONS (per-region tile counters → frontier); each gated consumer's
claim cursor is bounded by watermark = frontier × k_edge (host-precomputed tiles-per-
region). ≤1 gated in-edge per consumer (latest row-linear producer); everything else
stays instruction-granular. Parity: full test_model agreement (66/143 nano, 126/271
small instrs gated), rerun-stable, losses bit-identical to df.

Protocol lessons (hard-won, in order):
1. NEVER route unbounded claims through a CAS loop — with ~all blocks claiming from one
   big instr it degrades quadratically (df2 first cut: 2.5x slower; fetch-add fast path
   for bound==ntiles recovered half).
2. Parked-in-ring gated instrs pin the consumed head and every idle block rescans them
   continuously (memory-system saturation). Fix: event-driven wakeup — exhausted gated
   instrs are KILLED out of the ring (slot -> -3) and re-pushed by the producer's
   watermark raise (queued[] dedupe flag + Dekker-style re-check with __threadfence for
   the lost-wakeup race; ring sized n + Σ(regions+1) per gate).
3. Volatile reads are mandatory on the region-counter prefix scan (stale L1 would
   strand the frontier); a final unbounded watermark publish on producer completion
   removes all residual deadlock risk.

THE VERDICT: after all fixes, df2 = df + 300-400us at BOTH configs (nano 1853 vs 2158,
small 9028 vs 9440). Region overlap cannot pay here because op spans are INTRINSIC-
LATENCY-bound (long serial kv/k loops per tile; 64-256 tiles on 132 blocks = no queue
backlog) — tile-granular deps eat the QUEUEING component of span, and there is none.
The machinery (wakeups, bounded claims, region accounting, longer ring scans) is pure
overhead at this scale. Parked per the plan's kill rule; executor + emission retained
(mode="df2"), df remains default. Would bind for throughput-bound programs (tiles >>
blocks) — not this regime.

KEEPERS from the round:
- **qt-outer attention tile order** (tile = qt*nq + qh, attention.cuh): small df
  9268 -> 9028 (-240us) — short causal tiles first lets blocks pick up off-path work
  while the long tiles run. Kept unconditionally (also what any prefix gating needs).
- claim quantum: 264 beats the "true" 132 at small (bigger batches worsen tail balance
  on multi-round instrs; nano mildly prefers 132). Left at 264; per-config knob for P6.

Scoreboard after Phase 3: **nano 1853 / small 9028 (df)** vs hardened 711/2730.

## v3 Phase 4a: warp-spec scheduler offload (megakernel_ws) — protocol works, register
## tax eats it; shipped as capability, df stays default

`megakernel_ws` (mode="ws", parity-green in test_model): 384 threads (see below why not
288), scheduler warpgroup owns claim + completion accounting off the consumer path,
double work slots with eager pre-claim; ops now use `consumer_sync()` (bar.sync 1,256)
and `MK_CONSUMERS` instead of __syncthreads/blockDim.x (equivalent for all executors;
conversion validated green everywhere).

Best result: nano 1938 / small 9198 vs df 1907/8994 same-run — **gate not met**. The
decisive diagnostic: df recompiled at 224 regs runs 2002/9388, i.e. the REGISTER TAX
alone is +97/+385; at equal budget the ws protocol WINS by −64/−190 (on-path wait at
small: 1495 -> 587us). The protocol is right; the hardware charges for it.

CRITICAL HARDWARE FACT (cost us the phase): **H100 allocates registers at 4-WARP
granularity — any block >256 threads is charged 12 warps**, so 288 threads has the
same 168-reg entry ceiling as 384 (65536/(12x32x4)); `__maxnreg__(224)` at 288 threads
fails occupancy. The first 288-thread build spilled the op path (STACK 544) and lost
14% uniformly. Consequences: (a) warp-spec on this op library REQUIRES setmaxnreg
(224 consumers / 56 scheduler measured best; 232/40 and 240/24 worse — scheduler spill
slows cross-block completion publication), plus the explicit
-gencode=arch=compute_90a,code=sm_90a (now in mk.py; CUDA 13.1's -arch=sm_90a emits
compute_90 PTX that silently rejects setmaxnreg); (b) probe findings that did NOT
transfer to the full executor: setmaxnreg-is-negative, 128B state padding (+10..30
here: scans touch 32x more L2 lines).

Path to harvest the standing −64/−190 protocol win (deferred, not abandoned):
either port claim-before-account + overlapped accounting into 256-thread df (no tax,
part of the win), or shrink the WMMA/attention op register footprints under 224 so the
consumers stop paying the ceiling — the natural companion of the Phase-5 attention
rewrite. Full log: results/mkv3-p4a.md.

## v3 Phase 5: FA-class wgmma attention — the big lever lands

`wgmma_attention.cuh` (OP_ATTN_{FWD,DKV,DQ}_WG, routed when D==64 and S%128==0; WMMA
ops remain the fallback and the D=128 ragged-S test config exercises it): block tile =
(head, 128 q-rows), two consumer warpgroups on 64-row halves, 2-stage K/V streaming,
REGISTER online softmax on the wgmma accumulator layout (2 shfl_xor per row), P via
bf16 smem, second accumulator bank for O += P@V; bwd keeps the validated two-pass
structure (dqkv_f32 atomics, P from LSE, Drow input) with chunked streaming (DKV C=2;
DQ C=4 at S=512, C=2 at S=1024 — C=1 is latency-bound at nano, C=4 tail-bound at
small, same lesson as the claim quantum).

Standalone: fwd 3.34x/5.63x, dkv 2.1x/3.3x, dq 2.2x/4.9x vs the WMMA ops (nano/small
shapes); parity vs SDPA/autograd everywhere; REG 109-168, zero spill, dkv 96KB smem
(the new smem-max op; 4KB headroom).

**Scoreboard: nano 1853 -> 1775 (df); small 9028 -> 7118 (df) / 7024 (ws) — mode=ws
beats df for the first time**, confirming Phase 4a's prediction that ops ≤224 regs let
the scheduler-offload protocol win surface. df remains default (nano still prefers it).
Attention on-path at small: ~5.9ms -> ~1.6ms; attention is NO LONGER the #1 lever —
small DQ is now co-scheduling-bound (76us in-model vs 31 standalone), and the top
remaining items are the NN/TN bwd gemm latency, the dispatch-spill tax (below), and
per-hop costs.

KEY LAYOUT DISCOVERY: a single no-swizzle 64x64 smem arrangement
off64(r,c) = ((r>>3)<<10)+((c>>3)<<7)+((r&7)<<4)+((c&7)<<1) is readable under BOTH
wgmma majors — K-view LBO=128B/SBO=1024B (k-tile step +256B) and MN-view
LBO=1024B/SBO=128B (k-tile +2048B, = wg_desc_mn) — so every operand loads once
row-major and every bwd transpose is a descriptor change, zero data movement. Also:
generic smem stores feeding wgmma need `fence.proxy.async.shared::cta`.

COST SURPRISE (open, P6 candidate): integrating the new ops grew the shared interpreter
switch's max pressure — df STACK 272 -> 528 with a ~6% uniform tax on the OLD path
(spills at the dispatch call sites, none in op bodies; __noinline__ made it worse; a
register diet on the bwd ops — P parked in smem bf16 across the dP gemm — recovered
part). The residual is structural to the one-255-reg-switch design; fixes are
executor-level (ABI-isolated dispatch or setmaxnreg partitioning). Net integration
effect decisively positive despite it. Full log: results/mkv3-p5-attnprobe.md.

## v3 Phase 6 round 1: rowop batching + criticality rings — the post-attention levers

Plan + historical-megakernel survey (Laine 2013, Diamos persistent RNNs, Whippletree,
stream-K, Hazy no-bubbles, MPK): results/mkv3-p6-plan.md. P6.0 re-profile OVERTURNED
the P5 "top remaining items" list: with attention fixed, ROW OPS were the #1 on-path
span at both configs (RMSNORM_BWD 132/842us nano/small — ~50us per instr for a ~3MB-
traffic op), NN bwd-dX gemms #2 (~426/1515us, 10 TF, 16-tile latency-bound).

**Shipped (parity-green everywhere, both test suites):**
1. **Batched row ops** (`MK_ROW_R=8` rows/tile, one warp per row): warp-shuffle
   reductions (block_sum + its 3 block barriers deleted), uint4 8xbf16 vectorized IO,
   rmsnorm_bwd/qknorm_bwd stage dw in smem — ONE global atomic per element per 8 rows
   (the per-row atomics serialized on the tiny [H]/[D] grad buffers were the span).
   mk.py `_ROW_TILE_R`/`rowop_tiles` keep df2 gates R-aware; swiglu_bwd gained dy_f32.
   Scalar fallback for H%8!=0. nano -117us, small -722us — biggest single round since
   P5. Rowop spans: RMSNORM_FWD 3.3x, SWIGLU_BWD 2.1x, RMSNORM_BWD 1.35x (residual is
   bandwidth contention with co-running cold work, NOT op quality — see 3).
2. **Hot/cold criticality ready rings in df** (the Whippletree/MPK lesson): COLD =
   sinks (dW gemms, embed_bwd) + fills, HOT = everything else; idle blocks drain hot
   first; cold-sticky blocks yield when the hot tail moves (invisible-slot guard for
   the push race). Host computes crit[] in finalize (adj_off + FILL check). WAITS
   COLLAPSED: EMBED_FWD 91->0, RMSNORM_BWD worst-hop 140->4.4us wait; step only
   -7/-38us direct (waits became contended spans) BUT it flipped experiment 3:
3. **Split-K fp32 dX routing, tile-gated** (`model.dx_split_k`, gate: < 32 MN tiles):
   the 16-tile dXN gemms route via split-K atomics into per-layer fp32 workspaces,
   consumers read them dy_f32 (no CVT). Under the OLD single ring: +127/+467
   (claim contention ate it). On hot/cold rings: nano -27us, small ~0 (its gemms
   have >= 64 tiles — gated off). Order-of-experiments mattered.

**Measured negatives (documented, keep off):**
- cold_cap (bound blocks on cold work, MK_COLD_CAP): +/-20us wash both configs —
  bandwidth contention trades ~1:1 against tail serialization at these sizes.
- wgmma NN routing re-run (MK_WGMMA_NN=1, bit10 excluded): STILL +58/+69 — the
  round-3 verdict survives the new fixed-cost regime.
- DQ/DKV "co-scheduling overhead" (76 vs 31us standalone) diagnosed BENIGN: the pair
  runs concurrently in the same window (concurrent max ~= sequential sum) — stop
  chasing it.
- ws mode now LOSES to df at both configs (1686/6616 vs 1627/6356) — the P6 wins are
  df-only; a ws port of the hot/cold rings is the obvious (deferred) follow-up.

**Round-1 scoreboard: nano 1622 / small 6356 (df)** vs hardened 712/2735.

## v3 Phase 6 round 2: the spill-tax kill + claim retune — 2.05x

1. **Dispatch spill tax SOLVED** (the P5 open item, and Laine-2013's canonical
   megakernel pathology): the spilled caller state was dominated by the 104-byte
   `const Instr I = instrs[ins]` register copy live across every dispatch call.
   Fix: stage the claimed Instr into STATIC smem once per claim (26-int parallel
   copy + the already-present barrier) and dispatch a reference to it. df STACK
   624 -> 336; **nano -125us, small -484us (~8% uniform, both configs)** — the
   highest value-per-line change of the phase. Applied to df/df2/waves.
   **ws consumers MUST keep the register copy**: the same trick there (reference
   into the control slot) hangs intermittently at small — see 3.
2. **Claim quantum default 264 -> 132**: the old "264 beats 132" optimum was an
   artifact of expensive claims; after the rings + spill fix, finer batches' tail
   balance wins (-27/-237us). MK_CLAIM env knob for sweeps.
3. **ws lookahead=2 pre-claim race, mitigated**: with the new (8x shorter) rowop
   batches, the eager slot-B pre-claim hangs ~1 in 2-6 rounds of 20 small steps
   (pre-P6 commit clean at the old cadence; df clean everywhere; every new op clean
   200x in ws isolation; la=1 clean 160 steps). The race predates P6 — the cadence
   change widened its window. Default is now lookahead=1 (MK_WS_LOOKAHEAD to
   override); root-cause fix belongs to the deferred ws round. ws (la=1, stable):
   1633/6291 — still behind df.

**Final flag-planting scoreboard** (df megakernel vs hardened compile+cudagraph+,
median-of-50, FRESH PROCESS PER CONFIG — benching multiple sequence lengths in one
process poisons torch.compile with dynamic-shape recompiles: the in-process S=256
"baseline" was 2121us vs the honest 608):
| config                  | megakernel | hardened | gap  |
|-------------------------|-----------|----------|------|
| nano  (H256 L4 S512)    | 1468      | 711      | 2.06x |
| small (H512 L8 S1024)   | 5626      | 2730     | 2.06x |
| deep-narrow (L12 S512)  | 3638      | 1985     | 1.83x |
| S=128 (nano width)      | 1046      | 459      | 2.28x |
| S=256                   | 1233      | 608      | 2.03x |
| S=1024                  | 1937      | 961      | 2.02x |
(v3 start: 2.65x/3.44x; post-P5: 2.5x/2.6x.) Phase-6 total: -307us nano (-17%),
-1480us small (-21%), all parity-green + racecheck/synccheck clean on the new df
protocol. Full logs: results/mkv3-p6-*.

Remaining structural items, in measured order: (1) per-op math throughput (wgmma NT
gemms ~40TF in-model, NN dX gemms ~10TF WMMA — needs producer-fed multi-stage
pipelines, i.e. P4b, now unblocked since ops fit 224 regs and the spill tax is gone);
(2) bandwidth contention between the chain and cold dW work (cold_cap was a wash, but
op-level BW efficiency isn't); (3) deferred retunes (attention C, m64n128 tiles);
(4) the ws pre-claim race + hot/cold port, if ws is to matter again.

## v3 Phase 6 round 3: row-gradient reduction + post-P6 retunes

**RMSNorm backward row-gradient reduction (KEEP):** `op_rmsnorm_bwd` no longer has all
8 row warps atomically contending on one shared `[H]` weight-gradient buffer. Each row
now writes a private `[row,H]` partial in smem, then the block reduces the 8 rows once
before the global `dw` atomic. This is the exact contention pattern P6.1 identified in
the remaining row-op spans. Correctness is green (`test_ops.py`, `test_model.py`; DF,
waves, df2, and ws all agree). Profile effect is local and real:

| metric | before | after |
|---|---:|---:|
| nano RMSNORM_BWD on-path span | ~199us | ~159us |
| small RMSNORM_BWD on-path span | ~674-682us | ~224us |

The step-time win is smaller than the local span win because the realized critical path
moved into head/attention/GEMM work once RMSNorm stopped dominating. Final same-run
headline benchmark (`results/mkv3-p6-rmsrow-ckv3-bench.log`):

| config | megakernel | hardened compile+CUDAGraph+ | gap |
|---|---:|---:|---:|
| nano  (H256 L4 S512) | 1421us | 709us | 2.00x |
| small (H512 L8 S1024) | 5612us | 2740us | 2.05x |

**Attention backward chunk retune (KEEP, shape-gated):** added `MK_ATTN_DKV_C` and
`MK_ATTN_DQ_C` overrides and retuned defaults after the row-op change. dQ stays at
the old picks (C=4 for S=512, C=2 for S=1024). dKV changes to C=3 only for S=512:
it improves the S=512 nano/deep family, but hurts S=256 and S=1024. The baked default
is therefore `Ckv=3 if S == 512 else 2`, with env overrides for future sweeps.

**Head dX split-K target retune (MEASURED, left override-only):** added
`MK_HEAD_DX_TARGET_TILES` around the `dlogits @ Wlm` split-K target because the
post-row-reduction profile exposed it as a worst hop at small. Sweeping 256..2048
showed no robust default improvement: 384 tiles was slightly best in one small run, but
512 stayed better at nano/deep/S1024. This older verdict was superseded by the P4b
current-base retune below, which promotes 256 after later scheduler/op changes.

Remaining measured top items after this round (`results/mkv3-p6-rmsrow-ckv3-prof.log`):
small is led by ATTN_DQ_WG, head dX/lm_head GEMMs, and GEMMNN MLP dX spans; nano is
now split across RMSNorm_BWD, ATTN_DKV_WG, and small GEMMNN spans. This reinforces the
existing P4b direction: producer-fed multi-stage GEMM/attention pipelines, not more
instruction-level fusion.

## v3 P4b gate recheck: deeper wgmma mainloops do not clear the gate

The parallel P4b probe (`pipe_probe.py`, uncommitted as of this note) tested the next
registered hypothesis: replacing `op_gemm_wgmma`'s current 2-stage/drained mainloop
with 3/4-stage and one-in-flight GMMA variants. Gate was >=1.5x over current on small
NT shapes. Measured result (`results/mkv3-p4b-pipe-quick.log` and
`results/mkv3-p4b-pipe-full.log`): no gate. The small NT heavy hitters were flat:

| shape | current | best deeper/in-flight |
|---|---:|---:|
| small NT gu 1024x3072x512 | ~50us / 64TF | ~50us / 64TF |
| small NT lm_head 1024x16384x512 | ~225us / 76TF | ~223-224us / 77TF |
| small NT down 1024x512x1536 | ~46us / 35TF | slower or noise |

The full probe currently fails correctness on the first NN shape (`S2W0(cur)` rel
~0.095), so use it as an NT gate result only until the NN path is fixed. A current
`MK_WGMMA_NN=1` recheck was also non-promotable: nano lost slightly and small was
neutral, despite the small profile showing the head split-K hop itself can shrink.
The cost reappears as wait/other-op span. Do not spend the next round on generic
pipeline-depth or broad NN routing unless a new probe changes this evidence.

## v3 Phase 6 round 4: attention-bwd chunk retune after the P4b miss

**Attention chunk defaults retuned (KEEP):** after generic P4b pipeline-depth failed,
the cheapest remaining attention lever was the routed chunk count. A fresh sweep
(`results/mkv3-p6-r4-retune-sweep.log`,
`results/mkv3-p6-r4-retune-combos.log`) found two stable changes:

- `ATTN_DQ_WG` now defaults to `Cq=2` for the wgmma path. The old S=512 default
  (`Cq=4`) over-split: nano/deep improved by ~60-100us in megakernel-only medians.
- `ATTN_DKV_WG` now uses `Ckv=1` only when `nq * (S/128) >= 64` (small has enough
  natural chunks), otherwise `Ckv=2` preserves tail parallelism for nano/S1024-H256.

The head split-K target sweep again showed only small/noisy wins at this checkpoint;
this older verdict was superseded by the P4b current-base retune below, which promotes
`MK_HEAD_DX_TARGET_TILES=256` after later scheduler/op changes.

Correctness: `test_ops.py` and `test_model.py` are green. Hardened headline benchmark
(`results/mkv3-p6-r4-attnretune-bench.log`):

| config | megakernel | hardened compile+CUDAGraph+ | gap |
|---|---:|---:|---:|
| nano  (H256 L4 S512) | 1406us | 708us | 1.99x |
| small (H512 L8 S1024) | 5510us | 2731us | 2.02x |

Profile (`results/mkv3-p6-r4-attnretune-prof.log`): nano improves locally by trimming
`ATTN_DKV_WG` chunks from 48 to 32 tiles per layer (on-path `ATTN_DKV_WG` span ~137us
-> ~121us). Small trades the old high-wait `ATTN_DQ_WG` path for fewer longer
`ATTN_DKV_WG` chunks; on-path wait drops (~618us -> ~425us) and the benchmark wins
~100us. The next real work is still op quality for GEMMNN/head and attention kernels,
not another instruction-level fusion pass.

## v3 P4b SW128 + gated NN routing: the GEMM limiter was smem bank conflict

The parallel P4b branch found the missing GEMM lever: the old no-swizzle INTER smem
layout made the cp.async stores 8-way bank-conflicted. Generic 3/4-stage mainloops did
not help because the limiter was the stage fill itself. `op_gemm_wgmma` now uses the
Hopper SW128/B128 layout for both K-major and MN-major operand slabs, with 1024B-aligned
stage bases so the absolute-address swizzle phase matches the descriptor. Attention
keeps its INTER layout because its descriptor-swap trick relies on the no-swizzle
symmetry.

Routing update: SW128 also flips the old NN verdict for sufficiently tiled dX GEMMs.
NN now routes when `gemm_tiles_wgmma(M, N) >= MK_WGMMA_NN_MIN` (default 64). This
routes small's 64+ tile dX GEMMs but keeps nano's 16-48 tile dX GEMMs on WMMA/split-K.
TN/dW routing remains OFF by default (`MK_WGMMA_TN=1` is only a re-run knob): direct
A/B showed dW sinks get faster locally but steal bandwidth from the on-path chain
(`results/mkv3-p4b-sw128-route-ab.log`).

Post-SW128 attention retune: `ATTN_DQ_WG` now defaults to `Cq=1` (env override
`MK_ATTN_DQ_C`). The earlier Cq=2 default over-splits once NN dX is no longer the main
small bottleneck. `DKV_C` keeps the prior shape gate (`Ckv=1` only when
`nq * (S/128) >= 64`).

Validation:
- `test_ops.py`, `test_model.py` green (`results/mkv3-p4b-sw128-testops.log`,
  `results/mkv3-p4b-sw128-route-testmodel.log`).
- Direct large NT GEMM checks for the routed model shapes pass, including
  512x8192x256, 1024x3072x512, 1024x16384x512, and residual 1024x512x1536
  (`results/mkv3-p4b-sw128-large-gemm-check.log`).
- Direct routed NN and TN/split-K checks pass (`results/mkv3-p4b-sw128-route-gemm-check.log`).
- `cuobjdump -res-usage`: df stays `REG:255 STACK:144`; no new spill cliff
  (`results/mkv3-p4b-sw128-cuobjdump.log`).

Clean hardened benchmark (`results/mkv3-p4b-dq1-default-bench.log`):

| config | megakernel | hardened compile+CUDAGraph+ | gap |
|---|---:|---:|---:|
| nano  (H256 L4 S512) | 1260us | 710us | 1.77x |
| small (H512 L8 S1024) | 4465us | 2731us | 1.64x |

Clean profile (`results/mkv3-p4b-dq1-prof.log`): SW128 makes the big NT spans
real progress instead of probe-only throughput, and gated NN removes most of the small
dX GEMM tax. Small `GEMMNN 1024x512x3072` drops ~516us -> ~254us, `GEMMNN
1024x1536x512` ~427us -> ~216us, and head dX ~190us -> ~146us. Remaining small top
items are now `RMSNORM_BWD`, `ATTN_DKV_WG`, `SWIGLU_BWD`, and the large NT lm_head
fwd. Nano remains split across `RMSNORM_BWD`, `ATTN_DKV_WG`, and small WMMA `GEMMNN`.

Follow-up checked and NOT promoted: widening `dx_split_k` to include 64-tile dX GEMMs
looked positive in one noisy run, but clean same-GPU A/B showed old gate, default, and
forced `MK_DX_SPLIT_MAX_TILES=64` within noise for small while forced 64 hurt nano
(`results/mkv3-p6-r4-dxsplit-ab-gpu5.log`). Keep the existing `<32` split-K gate.

## v3 P4b round 2 (session 0b544181): the latency-bound verdict + protocol/rowop harvest

Context: ran concurrently with a peer session (coordination:
results/AGENT-COORDINATION.md); the SW128 + NN-gating work is documented in the
section above (6fe8fcb). This section covers everything after it.

**THE measurement of the round** — nsys gpu-metrics sampled inside the megakernel
window (small, 5 steps, clean GPU): **SM issue 19%, compute warps in flight 12%
(= the single 256-thread block), unallocated warp slots 87%, DRAM read+write
UNDER 10% combined.** The interpreter is latency-bound at 1/8 occupancy;
the P6-era "bandwidth contention" framing is dead. Every subsequent experiment
was interpreted against this.

Shipped (each measured on idle GPUs with before/after util guards):
- **Drow (bit10) on the wgmma epilogue** (-43us small): warp-per-row reduction over
  Cs; qh = n0/D so D=128 half-head tiles accumulate correctly; falls under the NN
  >=64-tile gate (small routes, nano's 16-tile dOatt stays WMMA).
- **Rowop MLP split** (nano -12, small -30): rmsnorm fwd/bwd take TWO rows per warp
  with interleaved load streams (MK_ROW_R2=16; RMSNORM_BWD span 565->467 small);
  swiglu keeps single-row (its 6-iteration rows are already MLP-saturated) but
  gains __expf (SFU) over libm expf; qknorm stays R=8 (R=16 doubled its serial
  per-warp task chain, +142us). dw partials fold both rows into one smem slot.
- **op_ce_bwd uint4 IO** (CE_BWD 79 -> 46us small): the V=16384 dlogits pass was
  2-byte scalar accesses; libm expf kept deliberately (peer measured __expf here
  and reverted).
- **df completion-hint stickiness** (~-20us both): the block whose accounting
  enables a HOT dependent adopts it as its own sticky claim — the chain's next hop
  starts on a warm block without ring rediscovery (the ws scheduler's hint path,
  ported).
- **ws hot/cold rings + consumer-owned Instr snapshot**: ws STACK 544->304, nano ws
  1633->1399; ws still trails df, stays non-default.

Measured negatives (documented, reverted or default-off):
- **MK_OCC2** (__launch_bounds__(256,2) -> 2 blocks/SM at 128 regs): REG:128
  STACK:944, nano +32% / small +40%. Occupancy-via-spill loses; more blocks is not
  the path.
- **REGISTER-LIFETIME LAW (measured twice, now a design rule)**: register-resident
  value reuse LOSES to re-reading in the 8-warp regime — rmsnorm_bwd single-pass
  (hold x/dy/w across the dot; +240us small) and qknorm register-dw accumulation
  (+58us small) both reverted. Long register lifetimes block the scoreboard's load
  overlap; the winning shape is short-lived registers + more independent streams
  (the interleave). Corollary for future op work: prefer re-reads and extra load
  streams over caching values in registers.
- Rowop claim floor (MK_ROWOP_CLAIM 2/4): +275/+838 small — tail balance beats
  claim amortization (Stream-K physics, again).
- Attention loader lane remap (conflict-free stores under INTER): NEUTRAL — kept
  (strictly fewer smem port cycles; attention stage fills hide under mma+softmax,
  unlike the bare gemm loop).
- TN dW wgmma routing re-run under SW128: STILL +226/+570 — dW gemms are sinks;
  2x more BW-hungry sinks steal from the chain.
- Attention chunk re-sweeps: DKV C=1 / DQ C=2 confirmed optimal in-model (probe
  standalone prefers C=2/C=4 — the in-model optimum is a scheduling optimum, not
  an op optimum).

Attention verdict: the WG ops are FA-CLASS STANDALONE (attention_probe: fwd 35us,
dKV 45, dQ 30 at small shapes ~= baseline flash parity); the +30-50% in-model
instr tax is the global latency environment. Attention-internal work is not the
next lever.

ws race postmortem: the "la=1 stall" reported mid-session was GPU CO-TENANCY (an
sglang server landed on the measurement GPU; context time-slicing makes one batch
span ~2.5ms — a perfect fake stall signature). Retracted with clean-GPU evidence
(ws small 4890, no dribble). One REAL protocol bug found by inspection (la=2
only): the flip fast path stages without the k < acct+lookahead bound, so at
ds-acct=2 it overwrites slot (ds&1)'s q_ins bookkeeping while batch acct (same
parity) is un-accounted -> done[] corruption -> lost dependents. Likely the
historical la=2 hang; FIXED this round (ds - acct < 2 guard on the flip fast
path, a no-op at la=1): la=2 stress 200 small steps clean (previously hung ~1 in
40-120). la=2 remains SLOWER in-model (small 5494 vs 4826, nano 1377 vs 1269 —
the P4a tail-imbalance mechanism), so la=1 stays default; the guard is a
correctness fix.

Operational lessons (multi-agent, now standing practice):
- Per-session TORCH_EXTENSIONS_DIR: concurrent rebuilds of the name-keyed torch
  extension cache race and one process can load a mid-edit .so (flaky asserts).
- Guard every timed run with before/after nvidia-smi util checks; local GPUs churn
  (an inference server arrived mid-session and invalidated 30 minutes of numbers).
- Claim board + small frequent commits + never committing the peer's in-flight
  files kept two sessions productive in one tree.

SCOREBOARD (median-of-50, fresh process per config, hardened baseline):
| config | megakernel | hardened | gap |
|---|---|---|---|
| nano  (H256 L4 S512)  | 1250 | 711 | 1.76x |
| small (H512 L8 S1024) | 4426 | 2732 | 1.62x |
| deep-narrow (L12)     | 3138 | 1984 | 1.58x |
| S=128                 | 922  | 538  | 1.72x |
| S=256                 | 1070 | 610  | 1.75x |
| S=1024 (nano width)   | 1643 | 959  | 1.71x |
(v3 start 2.65x/3.44x; post-P6 2.06x/2.06x; post-SW128 1.79x/1.65x.)

## v3 P4b: RETRACTION + baseline correction — the crossover was a baseline bug

The section below reported a long-S crossover. Chasing its mechanism exposed a
BASELINE BUG present since bench.py was written: TorchQwen3 called
F.scaled_dot_product_attention with 3-D [H,S,D] tensors, which the flash backend
rejects — SDPA silently math-decomposed (materialized S x S softmax, tf32 gemms)
at EVERY S, in every v3 measurement. torch.profiler on the baseline shows
safe_softmax + xmma_f32f32_tf32f32 kernels, no flash, at S=1024 and S=4096 alike.

With the fixed baseline (4-D + enable_gqa; parity vs the math twin verified,
grads ~3%):

| config | megakernel | flash-baseline | honest gap | (old soft gap) |
|---|---|---|---|---|
| nano  (H256 L4 S512)  | ~1235 | **633**  | **1.95x** | 1.76x |
| small (H512 L8 S1024) | ~4370 | **1905** | **2.29x** | 1.62x |
| S=4096 (nano width)   | ~4365 | **1560** | **2.80x** | "1.19x faster" |
| S=8192 (nano width)   | ~10440| **3120** | **3.35x** | "1.68x faster" |
(measured twice each; the megakernel column is unchanged and stands.)

Full honest gauntlet vs the fixed flash baseline (median-of-50, fresh process
per config, util-guarded):

| config | megakernel | flash-baseline | gap |
|---|---|---|---|
| nano  (H256 L4 S512)  | 1244 | 631  | 1.97x |
| small (H512 L8 S1024) | 4413 | 1750 | 2.52x |
| deep-narrow (L12)     | 3146 | 1569 | 2.01x |
| S=128                 | 919  | 493  | 1.86x |
| S=256                 | 1056 | 550  | 1.92x |
| S=1024 (nano width)   | 1636 | 777  | 2.11x |

Consequences, honestly:
- The long-S crossover is RETRACTED — an artifact of the baseline's quadratic
  math-attention. Against real flash the megakernel falls FURTHER behind as S
  grows (our attention ops are FA-class per-instr at S~1024 shapes but scale
  worse than FA at long S).
- Every "hardened baseline" number in the v3 program (711/2733 goalposts, all
  gap ratios) was soft. The honest current gaps: nano 1.95x, small 2.29x.
- The chunked-CE test in the section below remains valid methodology; its
  conclusion ("CE was not the baseline's problem") is unchanged — the problem
  was attention all along.
- bench.py is fixed at the source (4-D + enable_gqa) so every future number is
  against the real baseline. The v3 win gates restated: nano <= ~0.63ms,
  small <= ~1.9ms — i.e., the true remaining program is ~2x, at its hardest
  precisely where the megakernel thesis (boundary overheads) was weakest.

## v3 P4b: CROSSOVER — the megakernel BEATS compile+CUDAGraph at S >= ~3k

Extending the S-sweep past the flag-planting configs (nano width H256 L4, then
small width, hardened baseline, median-of-50, fresh process per config, clean
GPU with util guards):

| S (nano width) | megakernel | hardened | mk/baseline |
|---|---|---|---|
| 128   | 922   | 538   | 1.72x slower |
| 512   | 1250  | 711   | 1.76x slower |
| 1024  | 1643  | 959   | 1.71x slower |
| 2048  | 2452  | 1928  | 1.27x slower |
| 3072  | 3386  | 3445  | **1.02x FASTER** |
| 4096  | 4387  | 5232  | **1.19x FASTER** (reproduced twice) |
| 8192  | 10398 | 17418 | **1.68x FASTER** |
| 4096 @ small width (H512 L8) | 16358 | 19433 | **1.19x FASTER** |

Parity verified at S=4096 AND S=8192 (loss matches the bf16 eager twin to 4dp;
worst grad max-rel 2.0%/2.4%). The win WIDENS with S (1.19x at 4096, 1.68x at 8192). The megakernel's step time scales near-linearly in S over this
range (1250 -> 4387 for 8x S) while the baseline turns superlinear past S=2048
(711 -> 5232). The obvious objection — "the baseline's fp32 [S,V] CE
materialization is the handicap" — was TESTED AND REFUTED: a chunked-CE
hardened baseline (per-1024-row lm_head + CE, no logits materialization,
compiled + CUDAGraphed identically) measures 5238us at S=4096 and 17620 at
S=8192 — within noise of the plain baseline. The megakernel's long-S advantage
comes from its attention/gemm/tile-co-scheduling scaling, not from CE; the
crossover claim survives the fused-CE honesty check (ratios 1.19x/1.69x
unchanged). Per-kernel decomposition of WHERE the baseline goes superlinear is
an open item (nsys --cuda-graph-trace=node, per the P0 method).

The v3 win gates at nano/small (0.66/2.4ms) remain unmet — short-S is still
1.6-1.8x behind — but the original goal ("BEAT compile+cudagraph") is now MET
for S >= ~3072 at both tested widths.

Late-round addenda from 0b544181 (all committed with data):
- ws snapshot A/B (MK_WS_REGCOPY): the consumer-owned smem Instr snapshot is worth
  -717us at small vs the old register copy; the residual ws-vs-df gap (4862 vs
  ~4400) is a UNIFORM ~8-20% per-op span tax = the 224-reg consumer ceiling
  (P4a's register-tax verdict re-confirmed with per-op evidence).
- MPK-topology probe (mpk_probe.py): dedicated scheduler block + 131 full-register
  consumer blocks over gmem mailboxes = 5.09us/hop (vs df ~3.0) — two cross-SM
  signals lose to df's self-claiming; scheduler-issued prefetch.global.L2 made it
  WORSE (9.86 — the issue loop starves done-polling). The register-tax escape is
  NO-GO at chain-hop granularity; both protocol escapes from the latency tax are
  now measured and closed. What remains: per-op MLP where streams are starved,
  honest acceptance of the ~1.6-1.8x floor at this architecture, or the Diamos
  weight-stationary v4 option (P6 survey).
- compute-sanitizer: df racecheck + synccheck CLEAN (covers the completion-hint
  and claim changes); ws racecheck reports are the by-design lock-free
  release/acquire handoff (racecheck cannot model PTX acquire/release).
- MK_CLAIM re-sweep post-SW128: 132 still optimal (66: +366 small, 264: +162).

Remaining structural items, in measured order: (1) the uniform in-model latency
tax (8 warps/SM) — the only true fixes are protocol-level (producer-fed loads /
cross-instr prefetch, i.e. the original P4b endgame, on a stabilized ws) or
per-op MLP where load streams are still starved; (2) head-dX target retune (below;
lm_head n128 was rechecked separately and not promoted); (3) waits ~410us small
(144 hops x ~2.9us).

P4b current-base head dX retune: after the committed ws/checkpoint verdicts, the
`dlogits @ Wlm` split-K target was rechecked without n128 on clean GPU 3. The old
512 target lost consistently to `MK_HEAD_DX_TARGET_TILES=256`: on `2a5bc25`, nano
512=1246.6/1235.4us vs 256=1225.3/1222.8us and small 512=4392.9/4390.8us vs
256=4362.5/4363.5us (`results/mkv3-p4b-wsbase-headtarget-256-ab.log`). After
CE_BWD vectorization (`91b7f74`), 256 still won: nano 512=1250.7/1227.9us vs
256=1221.6/1217.8us and small 512=4374.3/4367.8us vs 256=4335.6/4336.2us
(`results/mkv3-p4b-headtarget256-after-cebwd-ab.log`). After the df scheduler
completion-hint commit (`eba44d5`), the win held again: nano 512=1249.6/1229.7us vs
256=1220.2/1217.1us and small 512=4370.0/4368.5us vs 256=4338.8/4338.1us
(`results/mkv3-p4b-headtarget256-after-hint-ab.log`). The default moved to 256; the env
knob stays for reruns.
Post-DQ retune moved the default to `MK_HEAD_DX_TARGET_TILES=192`. A broad GPU2 sweep
suggested 192 over 256 (`mkv3-p4b-postdq-headtarget-sweep-20260704T192304Z.log`), but
its alternating repeat had drift
(`mkv3-p4b-postdq-headtarget-192-ab-20260704T192444Z.log`). The decisive GPU3 repeat
kept 192 ahead for the configs where head dX is material: small 4315.7/4313.9us vs
4332.8/4333.2us for 256, and S4096 4297.5/4293.4us vs 4315.2/4311.6us
(`mkv3-p4b-postdq-headtarget-192-gpu3-repeat-20260704T192636Z.log`). Nano was
effectively neutral after warmup (second-pass 192 1215.2us vs 256 1214.0us;
`mkv3-p4b-postdq-headtarget-192-nano-gpu3-20260704T192814Z.log`). The env knob stays.

Post-headtarget attention chunk recheck: with the current defaults at `431ed91`, a
fresh env-only sweep kept the routed WG attention chunk defaults unchanged. Small is
best at the existing `Ckv=1,Cq=1` (4350us vs >=4378us for the nearest alternates;
`results/mkv3-p4b-after-headtarget-attn-small-sweep.log`). Nano's broad sweep showed a
small apparent `Ckv=3,Cq=1` edge, but the alternating A/B was not stable: default
`Ckv=2,Cq=1` ran 1227.1/1214.9us, while `Ckv=3,Cq=1` ran 1216.3/1216.3us and
`Ckv=2,Cq=2` ran 1221.6/1221.8us
(`results/mkv3-p4b-after-headtarget-attn-nano-sweep.log`,
`results/mkv3-p4b-after-headtarget-attn-nano-ckv3-ab.log`). No attention env default
change was promoted.

Long-S attention chunk recheck after the fixed flash-baseline correction: an env-only
S=4096 sweep initially suggested `Ckv=2,Cq=1` might beat the default (4368.5us vs
4409.9us; `mkv3-p4b-longS-attn-chunk-sweep-s4096.log`), but alternating A/B refuted
it. S=4096 default won all three repeats (4373.2/4325.7/4345.2us vs
4417.8/4413.0/4416.3us for `Ckv=2,Cq=1`), and S=8192 also kept `Ckv=1,Cq=1`
(10408.1us vs 10509.7us for Ckv=2 and 10726.3us for Ckv=4;
`mkv3-p4b-longS-attn-chunk-ab.log`). Keep routed attention chunk defaults unchanged.

Post-fixed-baseline DQ C=1 epilogue cleanup: the current S=4096 profile showed
`ATTN_DQ_WG` as the largest on-path span (~1000us across four layers;
`mkv3-p4b-s4096-default-profile.log`). Since the default `Cq=1` has exactly one writer
per q slice, the dQ epilogue can store directly into the fp32 workspace instead of
using `atomicAdd`; `Cq>1` still uses atomics. Correctness is green post-merge
(`mkv3-p4b-dq-c1-postmerge-testattention.log`,
`mkv3-p4b-dq-c1-postmerge-testmodel.log`). Two clean same-GPU A/B runs promoted the
change: GPU5 control/variant medians were nano 1242.6/1231.9us, small 4351.5/4328.0us,
S4096 4325.7/4315.4us
(`mkv3-p4b-dq-c1-{control,variant}-20260704T190138Z-mkonly.log`); reverse-order GPU3
medians were nano 1248.3/1243.7us, small 4340.3/4326.9us, S4096 4318.1/4308.5us
(`mkv3-p4b-dq-c1-reverse-*-mkonly.log`). The earlier
`mkv3-p4b-dq-c1-outer-ab.log` was contaminated by an SGLang server and is ignored.
Follow-up DQ C=1 register-direct epilogue: once C=1 used plain stores, the shared-memory
stage/drain was no longer required on that path. The final route now stores each
thread's accumulator fragments directly to `dQKV_f32` for `C==1`; chunked `C>1` still
uses the old staged atomic path. Correctness is green (`mkv3-p4b-dq-reg-epilogue-testattention-20260704T194449Z.log`,
`mkv3-p4b-dq-reg-epilogue-testmodel-20260704T195504Z.log`). The first same-GPU A/B was
mixed (control/variant medians: nano 1234.3/1232.7us, small 4311.8/4312.0us, S4096
4288.3/4292.6us; `mkv3-p4b-dq-reg-epilogue-ab-20260704T194629Z.log`), so a reverse and
cache-warm repeat were required. Reverse-order GPU3 favored the variant strongly:
small 4304.9us vs 4326.8us, S4096 4248.1us vs 4314.5us
(`mkv3-p4b-dq-reg-epilogue-reverse-20260704T194944Z.log`). Cache-warm alternating GPU3
confirmed the win: control A/B small 4317.9/4324.2us and S4096 4293.0/4292.6us;
variant A/B small 4308.0/4314.2us and S4096 4251.9/4261.1us
(`mkv3-p4b-dq-reg-epilogue-warm-alt-20260704T195326Z.log`). This is a real long-S
attention win with a smaller short-S gain.

Post-headtarget DKV first-pair batching recheck: a macro-gated branch
`MK_ATTN_DKV_X2_SD=1` batched the first two independent DKV wgmma groups
(`S=QK^T` and `dP=dO V^T`) into one `wga_mma64_x2` call. This was correctness-clean
(`mkv3-p4b-attn-dkv-x2-testattention.log`) but a profile A/B on GPU 3 was negative:
control nano/small totals 1145.0us / 4325.4us, variant 1169.1us / 4510.8us
(`mkv3-p4b-attn-dkv-x2-control-prof.log`,
`mkv3-p4b-attn-dkv-x2-variant-prof.log`). DKV span itself was flat/noisy (nano
115.9us -> 114.5us, small 551.9us -> 556.9us), while other spans worsened. Do not
merge the branch; this is another instance of the register-lifetime law.
Post-S2048-headtarget DKV G2 fusion recheck: for `G==2,Ckv==1`, an env-gated branch
`MK_ATTN_DKV_G2_FUSE=1` fused the two GQA group-member DKV tiles for each KV tile,
accumulated both into one register pair, and drained dK/dV with plain stores instead
of two atomic epilogues. Correctness was green
(`mkv3-p4b-dkv-g2-testattention-20260704T2336DKVG2.log`,
`mkv3-p4b-dkv-g2-modelparity-20260704T2336DKVG2.log`), but the route was decisively
negative despite halving DKV tiles: variant-control deltas were +221us small, +323us
S2048, and +455us S4096, with control winning 80/80 paired samples in every direction
(`mkv3-p4b-dkv-g2-ab-20260704T2336DKVG2.log`). Lost G-parallelism dominates removed
K/V reloads and atomic drains; keep separate G-member DKV tiles.

Post-headtarget RMSNorm-bwd four-row fold recheck: `MK_RMS_BWD_R4=1` made each warp
fold four rows into one smem `dw` slot (32 rows/tile) instead of the default two-row
fold. Correctness was green (`mkv3-p4b-rmsbwd-r4-testrmsnorm.log`,
`mkv3-p4b-rmsbwd-r4-testmodel-fixedtiles.log`) and the RMSNorm-bwd span improved
(nano 146.3us -> 126.3us, small 466.3us -> 443.9us), but the whole-step medians did
not promote: control nano/small 1244.2us / 4344.0us, variant 1241.2us / 4370.6us
(`mkv3-p4b-rmsbwd-r4-control-mkonly.log`,
`mkv3-p4b-rmsbwd-r4-variant-fixedtiles-mkonly.log`). Keep the two-row default.

Post-DQ-reg RMSNorm-bwd dx/dw split: `RMSNORM_BWD` now emits default-on split ops
(`MK_RMS_BWD_SPLIT_DW=0` restores the old combined instruction). The dx-only op remains
on the residual-gradient chain; the dw-only op is a cold sink, so dX consumers no longer
wait for the weight-gradient atomic drain. Correctness is green
(`mkv3-p4b-rmsbwd-splitdw-testmodel-20260704T212233Z.log`). Same-source A/B with the
split env toggled promoted the change twice. GPU2 control/split medians were nano
1266.8/1260.7us, small 4507.1/4498.3us, S4096 4403.9/4403.1us
(`mkv3-p4b-rmsbwd-splitdw-mkonly-20260704T212708Z.log`). Variant-first GPU3
confirmation remained positive: nano 1274.7/1269.9us, small 4510.9/4503.7us, S4096
4406.5/4397.7us (`mkv3-p4b-rmsbwd-splitdw-confirm-20260704T213139Z.log`). This is a
small structural win, not a local RMS kernel-quality win: it adds one cold instruction per
norm but removes the dw drain from the critical dependency.

Post-headtarget Drow-WGMMA route recheck: the fused `dOatt = dX @ Wo` + Drow
epilogue path still builds as WMMA on current `model.py`, despite the WGMMA epilogue
support. A narrow route branch was correctness-clean (`mkv3-p4b-drow-wg-route-testmodel.log`)
and changed small's profile label from `GEMMNN 1024x512x512` to
`GEMMNN 1024x512x512.wg`, cutting that span to 213.8us. The profile total still lost
because `ATTN_DKV_WG` grew in the same run (small total 4373.5us;
`mkv3-p4b-drow-wg-route-prof.log`), and 80-step medians were only noise-level positive
versus recent control (route 1242.6us / 4341.0us vs control 1244.2us / 4344.0us;
`mkv3-p4b-drow-wg-route-mkonly.log`). Do not merge without a repeatable step-level win.
Post-DQ C=1 re-gate on a fresh current-base worktree kept the no-go. Correctness stayed
green (`mkv3-p4b-drow-wg-current-testmodel-20260704T191524Z.log`) and the target span
again shrank (`GEMMNN 1024x512x512.wg` at 214.1us), but `ATTN_DKV_WG` grew to 613.2us
and small regressed. Same-GPU direct medians were control/variant nano
1232.5/1235.2us, small 4328.3/4367.3us, S4096 4310.8/4302.6us
(`mkv3-p4b-drow-wg-current-ab-20260704T191524Z.log`). The tiny S4096 win is not worth
the short-S loss; leave this route isolated.

Post-RMS-split long-S Drow-WGMMA route: the prior route's short-shape loss is avoided by
gating the fused `dOatt = dX @ Wo` + Drow epilogue WGMMA path to `S >= 2048` only
(`MK_DROW_WG_LONGONLY=0` restores the WMMA route, `=1` forces the WGMMA route where
`wgmma_ok` allows it). A forced-route model test with `MK_WGMMA_NN_MIN=1` exercised the
bit10 WGMMA epilogue and passed (`mkv3-p4b-drow-wg-longonly-forced-testmodel-20260704T221753Z.log`).
S4096 timing promoted the long-only gate: GPU2 control/variant medians were
4428.1/4398.3us with 19/20 paired wins
(`mkv3-p4b-drow-wg-longonly-s4096-20260704T220831Z.log`); variant-first GPU3 confirmed
4443.7/4438.5us with 18/20 paired wins
(`mkv3-p4b-drow-wg-longonly-s4096-confirm-20260704T221402Z.log`). This remains off for
nano/small by default because the earlier small regression was real.

Post-headtarget SWIGLU_BWD two-row fold recheck: `MK_SWIGLU_BWD_R2=1` made each warp
handle rows `r` and `r+8` under a 16-row tile, matching the RMSNorm two-row tile
plumbing but keeping the per-row register lifetime short. Correctness was green
(`mkv3-p4b-swiglu-r2-testops.log`, `mkv3-p4b-swiglu-r2-testmodel.log`). Profile A/B was
not promotable: control totals were 1145.7us / 4375.4us and variant totals were
1152.9us / 4358.8us, but the SWIGLU_BWD span itself worsened (nano 38.2us -> 51.4us,
small 323.5us -> 347.4us; `mkv3-p4b-swiglu-r2-control-prof.log`,
`mkv3-p4b-swiglu-r2-variant-prof.log`). The direct 80-step medians were clearly
negative: control nano/small 1240.1us / 4352.0us, variant 1250.0us / 4391.4us
(`mkv3-p4b-swiglu-r2-mkonly.log`). Keep SWIGLU_BWD at one row/warp.

## v3 P4b round 3 (session 0b544181): the register-file accounting + n128 tiles

Direction set by the user mid-session: keep pushing, rethink bottlenecks.

**The dual-stream executor — built, then refuted by first principles.** The nsys
verdict (81% of issue slots empty) begged for more warps. Design: 512 threads =
two independent 256-thread df claim loops per block, asymmetric registers (fat
half 192: gemms/attention off the hot+cold rings; lean half 64: barrier-free
streaming ops off a third LEAN ring). Enabler shipped and kept: the op library
now indexes mk_tid() (= threadIdx.x & 255) with a group-derived barrier id, so
ops run unchanged in any multiple-of-256 block shape — side effect, df STACK
144 -> 32. The executor itself died at the compiler: ptxas gives setmaxnreg
regions NO extra budget — the whole kernel compiles at the __maxnreg__(128)
entry cap (REG:128 STACK:848 = the OCC2 spill signature). The general
accounting, now stated once for all future rounds: 256 threads x 255 regs =
the 64K register file EXACTLY; every add-warps design (ws 224, occ2 128, dual
192/64, two-block splits) pays fat-path registers, and the measured spill/
register tax (5-40%) always exceeds what the added warps can hide (~10-15% of
step). Eight 255-reg warps is the architecture's Pareto point.

**What the accounting DOES allow: bigger tiles at full registers.** m64n128 NT
wgmma tiles (64 fp32 accs/thread ~= 200 regs, fits 255): double mma work per
sync, half the B-traffic per FLOP — the dependent chain per FLOP shortens,
which is the one lever the register-lifetime law permits. Generalized from the
peer session's lm_head-only route (branch commit 8008126; their 'not promoted'
verdict was correct for the narrow scope — the win needed every NT gemm to
clear absorption), plus residual (bit16) support and an NN variant: B[K,N]
loads into two 64-mn MN-major SW128 slabs with the canonical B128 MN
descriptor using LBO=8192B as the 64-mn group stride (validated by grad
parity). Routing: flags bit12, MK_WGMMA_N128 / _NN / _NN_MIN knobs; NN
tile-gated at >=32 n128 tiles (nano's 24-tile dX stays m64n64/WMMA).
Measured (df, clean GPU): small 4252 -> 4095 (NT -146, NN -28), nano ~flat
(1093-1115 band). lm_head span 280 -> 194; gu 185 -> 138.

Knob re-sweeps post-n128: MK_CLAIM=132 and DKV C=1 still optimal. Cold-cap needs
shape gating: the peer default cap16 is right for short shapes, but S4096 regresses
hard unless cold work is uncapped. Broad c742 sweep
(`mkv3-p4b-c742-coldcap-shape-sweep-20260704T215225Z.log`) and focused repeat
(`mkv3-p4b-c742-coldcap-shape-alt-20260704T215407Z.log`) confirmed the robust part:
S4096 cap16/cap0/cap0/cap16/cap33/cap0 was
4009.8/3917.8/3916.9/4009.5/3934.1/3916.1us. Short-shape cap33 was not strong enough
to retune, so the default is cap16 for S < 2048 and uncapped for S >= 2048; `MK_COLD_CAP`
still overrides.

SCOREBOARD (fixed flash baseline, median-of-50, fresh process per config):
| config | megakernel | flash-baseline | gap |
|---|---|---|---|
| nano  (H256 L4 S512)  | 1205 | 631 | 1.91x |
| small (H512 L8 S1024) | 4105 | 1766 | 2.32x |
| deep-narrow (L12)     | 3088 | 1770 | 1.74x |
| S=128                 | 919 | 484 | 1.90x |
| S=256                 | 1046 | 482 | 2.17x |
| S=1024 (nano width)   | 1544 | 711 | 2.17x |
Long-S (separate processes): S=4096 4010/1568 = 2.56x, S=8192 9503/3156 =
3.01x — n128 pays more at long S too (mk S=8192 was 10434 this morning).
(Morning honest reset: nano 1.97x / small 2.52x; the day's compounded work —
both sessions — moved small ~14% and nano ~3%. Baseline medians wobble ±5%
run-to-run from inductor autotune variance; gaps quoted per-run.)

Residual decomposition at small (~4100 vs 1900): the uniform in-model tax
(25-50% on every op class vs standalone — co-resident instrs sharing the SM's
issue/memory pipes; this IS the overlap working), per-op gaps vs vendor
kernels (gemms ~140TF vs cuBLAS 200+, attention ~1.3x off FA3), waits ~445us
(144 hops x ~3.1us), fills/CE/embeds. No remaining single item measures
>~150us. The honest framing stands: a one-kernel design must pick one
register/occupancy point for all ops; the baseline runs every kernel at its
own optimum. The remaining ~2x is the price of that constraint at these model
sizes — future rounds should either attack per-op quality within the 255-reg
point (TMA, deeper attention pipelining) or accept the flag-planting scope.

## v3 P4b r3 coda: cold_cap flip + the attention-pipelining strike-three

- cold_cap default 0 -> 16: the pre-SW128 wash flipped positive once the ops got
  faster (cold dW work turned net-contentious): small ~4110 -> ~4042, nano ~1102
  -> ~1092. Flat across 8-33.
- FA2-style software-pipelined attention fwd (MK_ATTN_PIPE build): parity-green,
  in-model NEGATIVE (fwd span 245 -> 297, step +245 at small). With the pipe_probe
  stage sweep and the MPK prefetch, that's three independent measurements saying
  the same thing: in this 8-warp latency-bound regime, overlap/depth machinery
  costs more than it hides. The megakernel's ops are as fast as this architecture
  lets them be; the residual vs the baseline is the one-register-point constraint,
  not op micro-structure.
- Dynamic-smem default corrected after the pipe artifact: the default build only
  needs the old 100KB carveout; 120KB is now selected only when `MK_ATTN_PIPE=1`.
  Same-root A/B (`mkv3-p4b-smem-default-*-20260704T220504Z-mkonly.log`) was a small
  win or neutral: control/variant nano 1192.7/1185.6us, small 4084.6/4079.7us,
  S4096 3921.8/3919.3us. Reverse-order GPU3
  (`mkv3-p4b-smem-default-rev-*-20260704T220819Z-mkonly.log`) confirmed the
  short-shape direction: variant/control nano 1192.1/1196.6us, small
  4060.4/4073.0us, S4096 neutral 3922.3/3920.4us.
- Mid-S cold_cap retune: keep cap16 for S<1024, move 1024<=S<2048 to cap33, and keep
  S>=2048 uncapped. Current-head broad sweep
  (`mkv3-p4b-coldcap-current-sweep-20260704T2227COLDCAP.log`) kept nano at cap16 and
  S4096 at cap0, while small favored cap24/33. Alternating repeat
  (`mkv3-p4b-coldcap-current-alt-20260704T2230COLDCAPALT.log`) confirmed small cap33
  over cap16 by -9.3us median with 15/20 wins; nano cap24 was negative by +7.0us.
  The final-bench S1024 guard was weakly positive for cap33
  (`mkv3-p4b-coldcap-s1024-alt-20260704T2231COLDCAPS1024.log`: -2.0us median,
  16/24 wins). Default-path side-worktree validation passed full model parity
  (`mkv3-p4b-coldcap-midS-testmodel-20260704T2232COLDCAPMIDS.log`) and selected
  caps 16/33/33/0 for nano/small/S1024/S4096.
- n128 short-row auto-gate: keep all-eligible m64n128 only when the GEMM row count is
  at least 1024; below that, default to off for M<256 and lm-head-only for
  256<=M<1024. The current-head mode sweep showed all-eligible n128 still necessary
  for small and S4096, but expensive on short rows
  (`mkv3-p4b-n128-mode-current-20260704T2238N128MODE.log`,
  `mkv3-p4b-n128-short-current-20260704T2241N128SHORT.log`). Side-worktree paired
  A/B against old mode=1 confirmed default wins on the affected configs:
  S128 -49.8us (23/24 wins), S256 -19.1us (22/24), nano -9.1us (20/24), deep -38.1us
  (19/20), with S1024 unchanged in route count
  (`mkv3-p4b-n128-auto-paired-20260704T2247N128AUTO.log`). Full model parity passed
  (`mkv3-p4b-n128-auto-testmodel-20260704T2247N128AUTO.log`).
- S2048 head-dX target gate: after the S2048 RMS dx R4 route, the global
  `MK_HEAD_DX_TARGET_TILES=192` default still protects small, but S2048 wins by
  reducing the `dlogits @ Wlm` split-K from 3 to 1. Broad current-head sweep
  (`mkv3-p4b-headdx-post-rmsdx-sweep-20260704T2325HEADDX.log`) kept small best at
  192 and showed S2048 target 96 at 2215.6us vs 2232.0us for 192. Paired repeats
  confirmed the S2048-only default: env 96 beat env 192 by -39.0us and -14.4us
  (`mkv3-p4b-headdx-s2048-96-ab-20260704T2325HEADDX.log`), and patched auto beat
  forced old 192 by -35.8us and -10.4us
  (`mkv3-p4b-headdx-s2048-default-ab-20260704T2325HEADDX.log`). Route guard:
  auto leaves nano/small/S4096 unchanged and emits S2048 head dX as 64 tiles, sk=1
  (`mkv3-p4b-headdx-s2048-route-20260704T2325HEADDX.log`). Full model parity passed
  (`mkv3-p4b-headdx-s2048-testmodel-20260704T2325HEADDX.log`).
- Cold dW split-K retune: the old implicit target 512 over-parallelized off-path TN
  weight-gradient GEMMs and stole issue/memory bandwidth from the hot chain. The new
  implicit target is 192 for `K < 2048` and 128 for `K >= 2048`; explicit split targets
  such as head dX and nano dX are unchanged, and `MK_DW_TARGET_TILES=512` restores the
  old behavior. Broad and paired sweeps
  (`mkv3-p4b-dw-split-target-sweep-20260704T2342DWSK.log`,
  `mkv3-p4b-dw-split-target-paired-20260704T2343DWSK.log`) plus direct env
  default-vs-old timing (`mkv3-p4b-dw-split-target-default-ab-20260704T2344DWSK.log`)
  confirmed old-vs-new deltas: nano -57.3/-62.1us, small -232.2/-230.4us, S2048
  -109.6/-101.6us, and S4096 -170.6/-167.9us. Long-S also prefers 128 over 192
  (S2048 -20us, S4096 -43..45us).
- S2048 attention-bwd C=2 gate: post-dW profile put S2048 back on `ATTN_DKV_WG`
  span, but broad env sweep rejected larger attention C for nano/small/S4096
  (`mkv3-p4b-attn-c-sweep-580db4d-20260704T2351ATTNC.log`). The one surviving shape
  is H256/S2048 with both `MK_ATTN_DKV_C=2` and `MK_ATTN_DQ_C=2`: paired/reverse env
  A/B (`mkv3-p4b-attn-c-s2048-paired-580db4d-20260704T2352ATTNC.log`) measured
  -42.4us, +29.7us, -30.8us, and +28.8us old-vs-new across four construction orders,
  with the C=2 combo winning 135-138/140 samples each block. Patched default vs forced
  old `C=1/1` (`mkv3-p4b-attn-s2048-c2-default-ab-20260704T2354ATTNC.log`) confirmed
  -39.8/+26.1/-25.6/+35.4us with 135-140/140 wins. Keep the gate S2048/H256 only;
  individual `DKV_C=2` or `DQ_C=2` and S4096 C=2 regressed.
- Long-S TN WGMMA dW gate: the old global `MK_WGMMA_TN=1` no-go became stale after the
  cold dW split-target retune. Current-head resweep
  (`mkv3-p4b-current-knob-resweep-65d9f1e-20260705T0000KNOBS.log`) still regressed
  nano/small/S2048, but S4096 improved. Paired long-S A/B
  (`mkv3-p4b-tnwg-long-paired-65d9f1e-20260705T0001TNWG.log`) set the boundary:
  S2048 loses (+22.1/+19.6us for TN), while S3072 wins -61.4/+55.5us, S4096 wins
  -93.8/+105.8us, and S8192 wins -335.2/+340.1us, with all long-S samples favoring
  TN. Patched default vs forced old
  (`mkv3-p4b-tnwg-long-default-ab-20260705T0003TNWG.log`) confirmed S3072
  -54.2/+52.1us, S4096 -106.8/+105.8us, and S8192 -332.8/+337.7us, while S2048 kept
  zero TN-WG routes. Default TN WGMMA is therefore gated to implicit `K >= 3072`;
  `MK_WGMMA_TN=0/1` still force-disables/enables it.
- Drow WGMMA default broadening: the old `S >= 2048` gate was stale after the later
  N128, dW, and long-S TN retunes. Current-head paired recheck
  (`mkv3-p4b-current-knob-paired-f687ef5-20260705T0007KNOBS.log`) showed small
  `MK_DROW_WG_LONGONLY=1` winning -24.8/+31.4us across order reversals with
  113/120 and 117/120 wins, while nano does not route Drow WGMMA and long-S shapes
  already default to it. Patched default vs forced old
  (`mkv3-p4b-drow-wg-default-ab-20260705T0009DROW.log`) confirmed small
  -21.4/+19.9us with 131/150 and 134/150 wins for the new default. Default now attempts
  Drow WGMMA whenever `mk.wgmma_ok` accepts the Drow epilogue GEMM;
  `MK_DROW_WG_LONGONLY=0` restores the old WMMA route.
- QKNORM_ROPE_BWD D=64 cache fast path: the D=64 model path has one rope pair per lane,
  so `op_qknorm_rope_bwd` now keeps `da/db`, weights, and normalized `x` live across the
  dot-product reduction instead of reloading/recomputing them for dx/dw. The generic
  D!=64 loop is unchanged, and `MK_QKBWD_D64_CACHE=0` keeps the old loop for A/B/bisects.
  Correctness passed full op and model suites
  (`mkv3-p4b-qknorm-d64-cache-flag-testops-20260705T0034QKBWD.log`,
  `mkv3-p4b-qknorm-d64-cache-flag-testmodel-20260705T0036QKBWD.log`,
  `mkv3-p4b-qknorm-d64-cache-default-testmodel-20260705T0040QKBWD.log`). Fresh-cache
  same-process A/B (`mkv3-p4b-qknorm-d64-cache-ab-20260705T0036QKBWD.log`) measured
  paired-diff medians: nano -6.7us (175/240 wins), small -26.6us (175/180), S2048
  -17.7us (167/180), S4096 -30.5us (126/128). `cuobjdump -res-usage` was unchanged
  for `megakernel_df`/`df2`/`ws`, so there is no spill/regression tax
  (`mkv3-p4b-qknorm-d64-cache-resusage-20260705T0040QKBWD.log`).
- S3072 head-dX target gate: after the qknorm-bwd cache, the S3072 profile showed
  `dlogits @ Wlm` still using target 192 (split-K=2, 192 tiles), while S2048 already
  used target 96. Current-head A/B
  (`mkv3-p4b-s3072-headdx-target-post-qknorm-86a174b-20260705T0042HEADDX.log`) showed
  target 96/128 both emit split-K=1 (96 tiles) and beat the default. Target 96 measured
  -19.8us and -15.5us paired medians across order reversals, with 112/120 and 106/120
  wins. Default head dX target is now 96 for H256/S2048 and H256/S3072; the env
  `MK_HEAD_DX_TARGET_TILES=192` restores the old route.
- Small head-dX target gate: after the qknorm-bwd cache and S3072 retune, the H512/S1024
  small profile still had `dlogits @ Wlm` on path at target 192 (split-K=3, 192 tiles).
  Current-head A/B (`mkv3-p4b-small-headdx-target-post-f0888dc-20260705T0045HEADDX.log`)
  showed target 96 emits split-K=1 (64 tiles) and wins -15.7us/-11.8us paired medians
  across order reversals, with 98/100 and 92/100 wins. Targets 128 and 256 regressed.
  Default head dX target is now 96 for H512/S1024 too; `MK_HEAD_DX_TARGET_TILES=192`
  restores the old route.
- Small RMS dx R4 gate: the earlier H512 R4 no-go flipped only after the qknorm-bwd cache
  and head-dX target retunes moved the small critical path. Post-`d23c38a` recheck kept
  S3072/S4096 as no-go (+2..6us), but H512/S1024 small was repeatably positive
  (`mkv3-p4b-rmsdx-r4-post-headdx-d23c38a-20260705T0049RMSDX.log`). The longer small-only
  confirmation (`mkv3-p4b-small-rmsdx-r4-confirm-d23c38a-20260705T0050RMSDX.log`)
  measured combined paired median -9.23us with 326/440 wins. Default `rms_dx_r4` now
  covers H512/S1024 small in addition to H256/S2048; `MK_RMS_DX_R4=0` restores the old
  R2 route. Route guard and parity passed
  (`mkv3-p4b-small-rmsdx-r4-route-20260705T005116Z.log`,
  `mkv3-p4b-small-rmsdx-r4-testmodel-20260705T005248Z.log`,
  `mkv3-p4b-small-rmsdx-r4-small-parity-20260705T005307Z.log`). Patched default vs
  forced-old timing (`mkv3-p4b-small-rmsdx-r4-default-ab-clean-20260705T005339Z.log`)
  measured combined paired median -5.23us with 310/440 wins.

- Nano NN WGMMA threshold gate: the older "nano dX WGMMA loses" result became stale after
  the qknorm/head-dX/RMS route retunes. Current route check with `MK_WGMMA_NN_MIN=16`
  sends nano's M=512 NN shapes `(512,256,256)`, `(512,768,256)`, and split-K head-dX
  `(512,256,8192)` through WGMMA while leaving the split-K `(512,256,512)` path WMMA.
  Paired confirmation
  (`mkv3-p4b-nano-nnmin16-confirm-post-33646d6-20260705T005643Z.log`) measured combined
  paired median -18.59us with 435/480 wins. Broad paired check
  (`mkv3-p4b-nnmin16-broad-paired-post-33646d6-20260705T005710Z.log`) kept small and
  long shapes neutral/noise-level, so the default threshold is now 16 only for M=512 NN
  gemms and remains 64 elsewhere; `MK_WGMMA_NN_MIN=64` restores the old nano route.
  Patched route guard and full model parity passed
  (`mkv3-p4b-nano-nnmin16-route-20260705T005841Z.log`,
  `mkv3-p4b-nano-nnmin16-testmodel-20260705T005850Z.log`). Patched default vs forced
  old timing (`mkv3-p4b-nano-nnmin16-default-ab-clean-20260705T005906Z.log`) measured
  combined paired median -19.76us with 479/480 wins.
- Nano head-dX target gate: after nano's M=512 NN path switched to WGMMA, the old
  head-dX split-K target 192 over-parallelized `(512,256,8192)` (192 tiles, sk=12).
  Current sweep (`mkv3-p4b-nano-headdx-target-post-59d68b0-20260705T010139Z.log`) showed
  targets 48/64/96 all beating 192, with target 96 best in the quick pass (96 tiles,
  sk=6). Paired confirmation
  (`mkv3-p4b-nano-headdx96-confirm-post-59d68b0-20260705T010154Z.log`) measured combined
  paired median -14.64us with 492/520 wins. Nano RMS dx R4 and attention chunk retunes
  were rejected in the same post-`59d68b0` pass, so the only new nano default is head-dX
  target 96; `MK_HEAD_DX_TARGET_TILES=192` restores the old route. Patched route guard
  and full model parity passed
  (`mkv3-p4b-nano-headdx96-route-20260705T010223Z.log`,
  `mkv3-p4b-nano-headdx96-testmodel-20260705T010232Z.log`). Patched default vs forced
  old timing (`mkv3-p4b-nano-headdx96-default-ab-clean-20260705T010245Z.log`) measured
  combined paired median -17.06us with 445/480 wins.

- ATTN_DKV_WG direct-atomic epilogue: after the route toggles were exhausted, the
  staged-smem dK/dV epilogue remained inside the top attention bucket. A compile-flagged
  direct-atomic variant writes the existing accumulator lane layout straight to
  `dQKV_f32`, skipping the smem stage and drain loop. Focused attention and full-model
  correctness passed
  (`mkv3-p4b-attndkv-directatomic-flag-testattention-20260705T011304Z.log`,
  `mkv3-p4b-attndkv-directatomic-flag-testmodel-20260705T011640Z.log`). Paired timing
  (`mkv3-p4b-attndkv-directatomic-ab-85ec29f-20260705T011444Z.log`) won on every tracked
  shape: nano -17.81us, small -49.39us, S2048 -40.51us, S3072 -7.30us, S4096 -19.23us
  combined paired. Default now compiles this direct-atomic DKV epilogue; set
  `MK_ATTN_DKV_DIRECT_ATOMIC=0` to restore the old staged-smem atomic route. Default-on
  full op/model suites passed
  (`mkv3-p4b-attndkv-directatomic-default-testops-20260705T011720Z.log`,
  `mkv3-p4b-attndkv-directatomic-default-testmodel-20260705T011720Z.log`), and patched
  default-vs-old timing (`mkv3-p4b-attndkv-directatomic-default-ab-clean-20260705T011741Z.log`)
  confirmed wins on all tracked shapes: nano -19.26us, small -47.94us, S2048 -41.76us,
  S3072 -8.83us, S4096 -20.90us combined paired. `cuobjdump -res-usage` showed no
  `megakernel_df`/`df2` register or stack change versus the old route
  (`mkv3-p4b-attndkv-directatomic-resusage-20260705T011825Z.log`,
  `mkv3-p4b-attndkv-old-resusage-20260705T011830Z.log`). Rechecked no-go side paths in
  the same pass: `MK_ATTN_PIPE=1` remained strongly negative
  (`mkv3-p4b-attn-pipe-recheck-85ec29f-20260705T010542Z.log`), and direct fwd WG stores
  were rejected after long-S regressions
  (`mkv3-p4b-attnfwd-direct-ab-85ec29f-20260705T011027Z.log`).

- ATTN_FWD_WG LSE fast-log: WGMMA forward stores now use `__logf` for the LSE epilogue
  by default; `MK_ATTN_FAST_LOG=0` restores precise `logf`. The blast radius is only the
  WGMMA attention fwd LSE sites; generic attention and CE stay on `logf`. Focused
  attention and full-model parity passed
  (`mkv3-p4b-aflog-testattention-20260705T0208AFLOG.log`,
  `mkv3-p4b-aflog-testmodel-20260705T0209AFLOG.log`,
  `mkv3-p4b-aflog-default-testmodel-20260705T0212AFLOG.log`). Opt-in paired timing
  (`mkv3-p4b-aflog-ab-20260705T0212AFLOG.log`) was neutral on nano (+0.40us median
  variant-control, 175/360 wins) and positive on small (-13.92us, 219/240 wins), S2048
  (-12.21us, 150/180 wins), and S4096 (-11.28us, 106/120 wins). Patched default-vs-old
  timing (`mkv3-p4b-aflog-default-vs-old-20260705T0212AFLOG.log`) confirmed smaller but
  still favorable medians: nano -5.84us, small -7.58us, S2048 -8.34us, S4096 -4.75us.
  End-to-end final-score refresh is noise-level at nano and modestly favorable at small
  (`mkv3-p4b-score-nano-aflog-20260705T0213SCORE.log`: 1060.7us vs graph+ 561.2us;
  `mkv3-p4b-score-small-aflog-20260705T0213SCORE.log`: 3738.7us vs graph+ 1908.5us).

- CE_FWD LSE fast-log no-go: a default-off `MK_CE_FAST_LOG` probe changed only
  `op_ce_fwd`'s final `logf(se)` to `__logf(se)` and kept CE backward `expf` untouched.
  Correctness was fine (`mkv3-p4b-ceflog-testce-20260705T0215CEFLOG.log`,
  `mkv3-p4b-ceflog-testmodel-20260705T0217CEFLOG.log`), but paired timing was mixed
  (`mkv3-p4b-ceflog-ab-20260705T0219CEFLOG.log`): nano regressed (+1.55us median,
  158/360 wins), small was order/noise-level (+1.26us paired median, 116/240 wins),
  S2048 was tiny positive (-2.91us, 105/180 wins), and S4096 was positive (-8.37us,
  86/120 wins). Keep CE on precise `logf`; the current CE op does not have enough
  shape information to split nano from S4096 safely.

- Post-fast-log profile + SWIGLU reciprocal no-go: current profile
  (`mkv3-p4b-post-aflog-profile-34d6635-20260705T0221PROFILE.log`) has nano 958.5us
  and small 3644.0us; small's top spans are still `ATTN_DKV_WG` 363.6us,
  `GEMMNN 1024x512x3072.wg` 311.8us, `SWIGLU_BWD` 261.9us, and `ATTN_FWD_WG`
  242.8us. A default-off `MK_SWIGLU_RCP_RN` probe replaced the sigmoid division with
  `__frcp_rn`. Global reciprocal correctness passed
  (`mkv3-p4b-swrcp-testswiglu-20260705T0223SWRCP.log`,
  `mkv3-p4b-swrcp-testmodel-20260705T0225SWRCP.log`) and helped small
  (`mkv3-p4b-swrcp-ab-20260705T0225SWRCP.log`: -16.22us median, 210/240 wins), but
  regressed nano (+6.74us, 90/360 wins), S2048 (+0.30us, 87/180 wins), and S4096
  badly (+20.16us, 2/120 wins). A shape-gated S=1024/I=1536 version passed the actual
  small-shape op test and model fallback checks
  (`mkv3-p4b-swrcp-gated-testswiglu-20260705T0227SWRCP.log`,
  `mkv3-p4b-swrcp-gated-testmodel-20260705T0229SWRCP.log`), but still regressed fallback
  shapes and left only a weak small win
  (`mkv3-p4b-swrcp-gated-ab-20260705T0229SWRCP.log`: nano +2.98us, small -4.11us,
  S2048 +3.18us, S4096 +18.02us). Keep SWIGLU's division form.

- ATTN_FWD_WG output reciprocal no-go: a default-off `MK_ATTN_FAST_INV` probe changed
  only the two WGMMA fwd output-normalization reciprocals (`1.0f / l[...]`) to
  `__frcp_rn`. Focused attention correctness passed
  (`mkv3-p4b-afinv-testattention-20260705T0233AFINV.log`), but the full model gate
  failed before timing: `test_model.py` kept per-step gradient parity within tolerance,
  then failed the 40-step learning sanity (`9.0496 -> 7.1534`, less than the required
  2.0 loss drop; `mkv3-p4b-afinv-testmodel-20260705T0235AFINV.log`). Keep WGMMA
  attention output normalization on division.

- Post-fast-log `MK_CLAIM` sweep no-change: current-tip env sweep
  (`mkv3-p4b-claim-sweep-f9cc513-20260705T0237CLAIM.log`) tested 64/96/132/160/192/264
  with randomized ordering. Default 132 remains the only safe scheduler claim quantum:
  nano saw 160 only -1.44us vs 132 while small, S2048, and S4096 regressed for every
  non-132 value (small nearest 264 was +37.28us; S2048 nearest 264 was +25.01us;
  S4096 nearest 160 was +3.92us and 264 was +115.86us). Keep `MK_CLAIM=132`.

- Post-fast-log `MK_COLD_CAP` retune: keep nano/short at cap16, move
  `1024 <= S < 2048` from cap33 to cap48, and keep `S >= 2048` uncapped. Broad
  current-tip sweep (`mkv3-p4b-coldcap-sweep-3babb35-20260705T0242COLD.log`) kept
  nano best at cap16; small had cap48 only -0.4us vs cap33, while S2048/S4096
  alternates were all noise-level. Focused paired repeats made the defensible part
  narrow: small cap48 beat cap33 by -2.7us median with 60/100 wins, and nano-width
  S1024 cap48 beat cap33 by -1.9us median with 67/120 wins
  (`mkv3-p4b-coldcap-confirm-3babb35-20260705T0243COLD.log`,
  `mkv3-p4b-coldcap-s1024-confirm-3babb35-20260705T0244COLD.log`). Long-S stays
  uncapped: S2048 cap64 was a 48/90 coin flip, S4096 cap8 lost by paired median, and
  S4096 cap132's -3.9us median was too small/noisy to undo the prior uncapped choice.
  Default-path model validation passed
  (`mkv3-p4b-coldcap-cap48-testmodel-20260705T0245COLD.log`), and affected score
  refreshes landed at small 3757.8us vs graph+ 1887.1us and S1024 1456.3us vs graph+
  778.1us (`mkv3-p4b-score-small-cap48-20260705T0245SCORE.log`,
  `mkv3-p4b-score-s1024-cap48-20260705T0246SCORE.log`). `MK_COLD_CAP` still overrides
  the model-selected default.

- Post-cap48 attention chunk retune: keep small at `DKV_C=1/DQ_C=1`, but move the
  H256/S512 WG attention shape to `DKV_C=3/DQ_C=2`. Current-profile refresh
  (`mkv3-p4b-profile-cap48-d6bbf0d-20260705T0245PROFILE.log`) still had nano led by
  `ATTN_DQ_WG` and small led by `ATTN_DKV_WG`, so an env-only resweep tested nearby
  chunk counts. The broad pass (`mkv3-p4b-attn-c-resweep-d6bbf0d-20260705T0246ATTNC.log`)
  rejected small changes hard (small `2/1` +66.6us, `1/2` +27.1us) but found nano
  `3/2` at -7.7us vs default. Focused confirmations kept the narrow H256/S512 gate:
  nano `3/2` beat `2/1` by -4.8us median with 128/180 wins, and deep-L12 beat by
  -6.3us median with 84/120 wins
  (`mkv3-p4b-attn-c-nano-confirm-d6bbf0d-20260705T0247ATTNC.log`). Route check and
  model validation passed
  (`mkv3-p4b-attn-c32-route-20260705T0248ATTNC.log`,
  `mkv3-p4b-attn-c32-testmodel-20260705T0248ATTNC.log`), and affected score refreshes
  landed at nano 1053.0us vs graph+ 630.9us and deep 2638.4us vs graph+ 1765.5us
  (`mkv3-p4b-score-nano-attnc32-20260705T0249SCORE.log`,
  `mkv3-p4b-score-deep-attnc32-20260705T0249SCORE.log`). Env overrides
  `MK_ATTN_DKV_C` and `MK_ATTN_DQ_C` still force the chunk counts for sweeps.

- Post-attn-chunk cold dW split-K retune: lower off-path dW split targets again after
  cap48 and nano attention chunking. Current profile
  (`mkv3-p4b-profile-attnc32-00c5aff-20260705T0249PROFILE.log`) still had large
  overlapped dW volume, and the env resweep
  (`mkv3-p4b-dw-target-resweep-00c5aff-20260705T0253DWSK.log`) showed lower targets
  winning broadly. Paired confirmations vs current defaults were decisive: target64
  beat default by -27.4us nano, -14.0us S1024, -15.3us S2048, -75.2us S4096, and also
  won the short scoreboard shapes by -15.6us S128 and -27.5us S256
  (`mkv3-p4b-dw-target-confirm-00c5aff-20260705T0254DWSK.log`,
  `mkv3-p4b-dw-target-short-confirm-00c5aff-20260705T0256DWSK.log`). Small prefers the
  slightly wider target96 over target64 by ~4us, and target96 beat the old default by
  -16.5us; S1024 target96 vs target64 was noise-level
  (`mkv3-p4b-dw-target-s1024-96v64-00c5aff-20260705T0255DWSK.log`). Default split-K
  target is therefore 96 for `K == 1024` and 64 otherwise; explicit head-dX targets are
  unchanged, and `MK_DW_TARGET_TILES` still force-overrides for sweeps. Route inspection
  and default-path model validation passed
  (`mkv3-p4b-dw-target-route2-20260705T0257DWSK.log`,
  `mkv3-p4b-dw-target-testmodel-20260705T0257DWSK.log`). Score refreshes after the
  retune are in the gauntlet below
  (`mkv3-p4b-score-nano-dwtarget-20260705T0258SCORE.log`,
  `mkv3-p4b-score-small-dwtarget-20260705T0259SCORE.log`,
  `mkv3-p4b-score-deep-dwtarget-20260705T0259SCORE.log`,
  `mkv3-p4b-score-s128-dwtarget-20260705T0300SCORE.log`,
  `mkv3-p4b-score-s256-dwtarget-20260705T0300SCORE.log`,
  `mkv3-p4b-score-s1024-dwtarget-20260705T0301SCORE.log`).

- Post-attn-chunk rejected rechecks: H256/S512 `MK_RMS_DX_R4=1` remains negative
  (`mkv3-p4b-rmsdx-r4-nano-recheck-00c5aff-20260705T0250RMS.log`: nano +9.2us
  median, deep +9.1us), and the apparent deep-L12 head-dX target48 signal from the
  broad sweep was noise (`mkv3-p4b-headdx-target-recheck-00c5aff-20260705T0251HEADDX.log`,
  `mkv3-p4b-headdx48-deep-confirm-00c5aff-20260705T0252HEADDX.log`: deep target48
  +13.2us paired median on confirmation). Keep nano/deep RMS dx on R2 and head-dX
  target96.

- Post-dW cold-cap retune: keep nano at cap16, small at cap48, and long S uncapped, but
  move H256/S1024 from cap48 to cap64. The broad current-head sweep
  (`mkv3-p4b-coldcap-resweep-postdw-0ee579e-20260705T0300COLD.log`) kept nano best at
  cap16 and S2048 uncapped; S4096 cap96 was only -2.8us noise. S1024 cap64 was the only
  repeatable change, confirmed at -5.4us paired median with 93/140 wins
  (`mkv3-p4b-coldcap-s1024-confirm-postdw-0ee579e-20260705T0301COLD.log`). Small cap33
  over cap48 was only -1.4us with 66/120 wins, so keep H512/S1024 small at cap48.
  Route check and default-path validation passed
  (`mkv3-p4b-coldcap-s1024-route-20260705T0302COLD.log`,
  `mkv3-p4b-coldcap-s1024-testmodel-20260705T0302COLD.log`). The affected score refresh
  landed at S1024 1461.2us vs graph+ 713.2us
  (`mkv3-p4b-score-s1024-coldcap64-20260705T0303SCORE.log`); the pairwise cap A/B is
  the stronger signal here because the graph+ baseline moved substantially in this
  fresh process. `MK_COLD_CAP` still force-overrides for sweeps.

- S1024 WGMMA NN threshold gate: the S-sweep profile showed H256/S1024 still running
  several `GEMMNN 1024x256x...` hops on WMMA because the generic NN threshold was 64
  tiles (`mkv3-p4b-profile-ssweep-4a1cf3e-20260705T0300PROFILE.log`). Lowering only the
  `M=1024,N=256` NN threshold to 32 routes those hops through m64n64 WGMMA while leaving
  small's already-routed `1024x512` shapes unchanged. Env sweep
  (`mkv3-p4b-s1024-wg-nn-threshold-4a1cf3e-20260705T0301N128.log`) measured S1024
  `MK_WGMMA_NN_MIN=32` at -102.3us versus default; the n128 min16 route was weaker
  (-74.3us). Direct confirmation
  (`mkv3-p4b-s1024-wg32-confirm-4a1cf3e-20260705T0302N128.log`) measured -108.1us
  paired median with 180/180 wins. Route check and model validation passed
  (`mkv3-p4b-s1024-wg32-route-20260705T0303N128.log`,
  `mkv3-p4b-s1024-wg32-testmodel-20260705T0303N128.log`), and the affected score
  refresh landed at S1024 1338.0us vs graph+ 774.8us
  (`mkv3-p4b-score-s1024-wg32-20260705T0304SCORE.log`). `MK_WGMMA_NN_MIN` still
  force-overrides the threshold for sweeps.

- Short-row WGMMA NN threshold gate: after the S1024 route, S128/S256 were still led by
  tiny WMMA `GEMMNN {128,256}x256x256` hops. Env sweep
  (`mkv3-p4b-short-wg-nn-threshold-692ae50-20260705T0304N128.log`) found threshold8
  modestly positive on S128/S256 while nano's existing threshold16 stayed best. Direct
  confirmation (`mkv3-p4b-short-wg8-confirm-692ae50-20260705T0305N128.log`) measured
  S128 -6.4us median with 113/180 wins and S256 -7.3us median with 145/180 wins. Default
  NN threshold is therefore 8 for `M in {128,256}, N=256`; M512 stays 16, M1024/N256
  stays 32, and the generic threshold stays 64. Route check and model validation passed
  (`mkv3-p4b-short-wg8-route-20260705T0306N128.log`,
  `mkv3-p4b-short-wg8-testmodel-20260705T0306N128.log`). Score refreshes landed at
  S128 853.1us vs graph+ 485.8us and S256 914.7us vs graph+ 549.9us
  (`mkv3-p4b-score-s128-wg8-20260705T0307SCORE.log`,
  `mkv3-p4b-score-s256-wg8-20260705T0307SCORE.log`); as usual, use the paired A/B for
  the small S128/S256 route delta because graph+ moves between fresh processes.

- Post-route `MK_CLAIM` resweep no-change: after the dW and WGMMA route retunes,
  current-head claim sweep (`mkv3-p4b-claim-resweep-48e9ff3-20260705T0306CLAIM.log`)
  kept 132 as the only safe scheduler quantum. S128, nano, S1024, and small all favored
  132; S256 had claim192 only -2.9us, too small and isolated to justify regressing the
  rest. Keep `MK_CLAIM=132`.

- Current-head executor-mode rechecks no-change: fresh profile at `0536d5f`
  (`mkv3-p4b-profile-current-0536d5f-20260705T0308PROFILE.log`) measured nano 918.2us
  and small 3644.4us, with small still led by `ATTN_DKV_WG`, WGMMA NN dX GEMMs,
  `SWIGLU_BWD`, `ATTN_FWD_WG`, and RMS dx. Paired current-mode checks kept df as the
  default: `ws` lost by +78.4us on nano and +227.4us on small with zero paired wins
  (`mkv3-p4b-mode-ws-current-0536d5f-20260705T0311MODE.log`), and `df2` lost by
  +229.5us on nano and +458.0us on small with zero paired wins
  (`mkv3-p4b-mode-df2-current-0536d5f-20260705T0312MODE.log`).

- Current-head H512/S1024 RMS dx retune: after the dW/WGMMA route changes, small moved
  back from `RMSNORM_BWD_DX_R4` to the normal two-row `RMSNORM_BWD_DX`. The broad
  current-head check (`mkv3-p4b-rmsdx-r4-current-0536d5f-20260705T0313RMS.log`)
  rejected R4 for nano (+13.2us) and H256/S1024 (+15.1us), while small R2 beat R4 by
  -6.2us. Longer small confirmation
  (`mkv3-p4b-rmsdx-r4-small-confirm-0536d5f-20260705T0314RMS.log`) measured R2-R4 at
  -9.6us paired median with 142/180 wins. Route and validation passed
  (`mkv3-p4b-rmsdx-r2-small-route-20260705T0315RMS.log`,
  `mkv3-p4b-rmsdx-r2-small-testmodel-20260705T0315RMS.log`); H256/S2048 still uses the
  R4 fold, and `MK_RMS_DX_R4` still force-overrides for sweeps. Affected score refresh:
  small 3714.4us vs graph+ 1898.8us
  (`mkv3-p4b-score-small-rmsr2-20260705T0316SCORE.log`).

- Post-RMS-retune small no-go rechecks: refreshed small profile at `1ea7e4f`
  (`mkv3-p4b-profile-small-rmsr2-1ea7e4f-20260705T0317PROFILE.log`) still leads with
  `ATTN_DKV_WG`, WGMMA NN dX, `SWIGLU_BWD`, `ATTN_FWD_WG`, and RMS dx. Focused
  current-head rechecks did not expose a second safe default change: disabling all NN
  n128 regressed +27.4us and lm-head-only n128 regressed +40.0us
  (`mkv3-p4b-n128-nn-current-1ea7e4f-20260705T0318N128.log`); selective NN n128
  thresholds 48/64/96/128 all regressed
  (`mkv3-p4b-n128-nn-thresh-small-1ea7e4f-20260705T0325N128.log`); head-dX targets
  128/192/256 regressed and target64 was the same effective split as default/noise
  (`mkv3-p4b-headdx-small-current-1ea7e4f-20260705T0319HEADDX.log`); cap16 was only a
  weak -3.9us with 112/180 wins after confirmation, so keep small cap48
  (`mkv3-p4b-coldcap-small-rmsr2-1ea7e4f-20260705T0320COLD.log`,
  `mkv3-p4b-coldcap16-small-confirm-1ea7e4f-20260705T0321COLD.log`); dW target64/80
  stayed noise and target128 regressed
  (`mkv3-p4b-dwtarget-small-rmsr2-1ea7e4f-20260705T0322DWSK.log`); attention bwd chunk
  variants all regressed by +35us to +96us
  (`mkv3-p4b-attn-c-small-rmsr2-1ea7e4f-20260705T0323ATTNC.log`); and forcing the old
  Drow WMMA route lost +31.4us with 1/140 wins
  (`mkv3-p4b-drow-small-rmsr2-1ea7e4f-20260705T0324DROW.log`).

- Current-head S256 head-dX split retune: the all-in-one gauntlet
  (`mkv3-p4b-score-all-current-cf137d6-20260705T0320SCORE.log`) reproduced the old
  Torch compile recompile-limit pitfall for later S-sweep baseline rows, so use
  fresh-process rows (`mkv3-p4b-score-s256-current-cf137d6-20260705T0327SCORE.log`,
  `mkv3-p4b-score-s1024-current-cf137d6-20260705T0327SCORE.log`) and paired A/B for
  small route deltas. S256 profile
  (`mkv3-p4b-profile-s256-current-cf137d6-20260705T0329PROFILE.log`) put the
  split-K `dlogits @ Wlm` hop at the top of the worst-hop list. Smaller
  `MK_HEAD_DX_TARGET_TILES` values beat the old target192 decisively
  (`mkv3-p4b-headdx-s256-current-cf137d6-20260705T0332HEADDX.log`: target32/48/64 all
  about -17us; target256 regressed). Confirmation selected target64 for H256/S256 only:
  -16.5us paired median with 167/180 wins; S128 target64 was too weak to promote
  (-4.6us, 68/120), and target64 was slightly better than target32 in direct A/B
  (`mkv3-p4b-headdx-s256-confirm-cf137d6-20260705T0333HEADDX.log`). Route and
  correctness passed (`mkv3-p4b-headdx-s256-route-20260705T0334HEADDX.log`,
  `mkv3-p4b-headdx-s256-parity-20260705T0334HEADDX.log`,
  `mkv3-p4b-headdx-s256-testmodel-20260705T0334HEADDX.log`). A fresh score row after
  the change landed at S256 934.5us vs graph+ 559.5us
  (`mkv3-p4b-score-s256-headdx64-20260705T0335SCORE.log`), but the paired A/B is the
  stronger evidence because both megakernel and graph+ medians wobble between fresh
  processes.

- Current-head S128 WGMMA NN threshold gate: S128 profile
  (`mkv3-p4b-profile-s128-current-8d8f95b-20260705T0336PROFILE.log`) showed the
  `GEMMNN 128x256x256` Drow hops as the top on-path bucket and still on WMMA because
  the short-row gate required 8 m64n64 tiles while this shape has 4. A current-head
  threshold sweep (`mkv3-p4b-wg-nn-s128-current-8d8f95b-20260705T0337N128.log`) measured
  `MK_WGMMA_NN_MIN=4` at -12.4us median with 100/160 wins; route inspection confirmed
  the intended `128x256` head-dX and Drow WGMMA routes while leaving S256/nano routes
  unchanged (`mkv3-p4b-wg-nn-s128-route-20260705T0338N128.log`,
  `mkv3-p4b-wg4-s128-route-20260705T0339N128.log`). Focused S128 parity and default
  model validation passed (`mkv3-p4b-wg4-s128-parity-20260705T0339N128.log`,
  `mkv3-p4b-wg4-s128-testmodel-20260705T0340N128.log`). Fresh score after the patch:
  S128 874.7us vs graph+ 495.4us (`mkv3-p4b-score-s128-wg4-20260705T0340SCORE.log`);
  use the paired A/B for the small route delta.

- Current-head S128 head-dX split retune: after the S128 WGMMA NN gate, `dlogits @ Wlm`
  now routes through WGMMA, and the old target192 over-parallelized this short shape.
  Broad current-head sweep (`mkv3-p4b-headdx-s128-post-wg4-a21b7a8-20260705T0341HEADDX.log`)
  measured target32 at -29.8us paired median, target64 at -23.8us, target96 at -15.9us,
  and target128 at -12.9us versus the current default. The target32 signal held when
  isolated from the rejected dX split side probe: target32 beat target192 by -20.0us
  with 157/180 wins while both models forced the old dX split target128
  (`mkv3-p4b-headdx-s128-target32-isolated-20260705T0342HEADDX.log`). Default route now
  emits S128 head dX as 32 WGMMA split-K tiles (`sk=8`) while S256/nano/S1024 routes are
  unchanged (`mkv3-p4b-headdx-s128-target32-route-20260705T0343HEADDX.log`). Full model
  validation passed (`mkv3-p4b-headdx-s128-target32-testmodel-20260705T0344HEADDX.log`),
  and patched default-vs-forced-old timing confirmed -14.2us paired median with 137/180
  wins (`mkv3-p4b-headdx-s128-target32-default-vs-old-20260705T0345HEADDX.log`). Fresh
  score after the patch: S128 867.2us vs graph+ 484.7us
  (`mkv3-p4b-score-s128-headdx32-20260705T0346SCORE.log`).

- Rejected S128 dX split side probes: opt-in WGMMA split-K inside `gemm_dx` was a clear
  no-go despite halving tile counts, regressing S128 by +80.1us, S256 by +119.4us, and
  nano by +122.1us with zero variant wins
  (`mkv3-p4b-dxwg-splitk-current-a21b7a8-20260705T0333DXWG.log`). Lowering the existing
  WMMA split-K target to 64 was initially positive on S128 but construction-position
  checks exposed allocator/order bias; patched default-vs-old was only -2.4us with
  90/160 wins and profile attribution showed only a ~2.8us total difference, while S256
  was noise and nano slightly lost (`mkv3-p4b-dxsplit-target-current-a21b7a8-20260705T0334DXSK.log`,
  `mkv3-p4b-dxsplit-s128-target64-confirm-a21b7a8-20260705T0335DXSK.log`,
  `mkv3-p4b-dxsplit-s128-default-vs-old-20260705T0338DXSK.log`,
  `mkv3-p4b-dxsplit-s128-default-vs-old-reverse-20260705T0339DXSK.log`,
  `mkv3-p4b-dxsplit-s128-profile-default-vs-old-20260705T0340DXSK.log`). Keep
  `gemm_dx` split target128.

- Current-head S1024 head-dX split retune: fresh post-S128 profile at `b919f9f`
  (`mkv3-p4b-profile-current-b919f9f-20260705T0341PROFILE.log`) showed S1024 now has the
  WGMMA NN route active, with remaining leaders `ATTN_DQ_WG`, `ATTN_FWD_WG`, the lm-head
  NT gemm, and head dX still at target192 (`sk=6`, 192 tiles). Current-head sweep
  (`mkv3-p4b-headdx-s1024-current-b919f9f-20260705T0342HEADDX.log`) selected target64:
  target64 beat target192 by -20.2us paired median with 96/100 wins; target96 was also
  positive (-14.9us), while target32 and target256 regressed. Longer confirmation kept
  target64 at -11.7us with 164/180 wins
  (`mkv3-p4b-headdx-s1024-target64-confirm-b919f9f-20260705T0343HEADDX.log`). Default
  route now emits H256/S1024 head dX as 64 WGMMA split-K tiles (`sk=2`) while S128,
  S256, nano, S2048, S3072, and H512/S1024 small routes are unchanged
  (`mkv3-p4b-headdx-s1024-target64-route-20260705T0344HEADDX.log`). Full model
  validation passed (`mkv3-p4b-headdx-s1024-target64-testmodel-20260705T0345HEADDX.log`),
  patched default-vs-forced-old timing confirmed -12.3us with 168/180 wins
  (`mkv3-p4b-headdx-s1024-target64-default-vs-old-20260705T0345HEADDX.log`), and fresh
  score after the patch was S1024 1283.3us vs graph+ 776.7us
  (`mkv3-p4b-score-s1024-headdx64-20260705T0346SCORE.log`).

- Current-head S1024 attention dQ chunk retune: after the S1024 head-dX route shortened,
  a focused S1024 attention chunk sweep kept `DKV_C=2` but moved `DQ_C` from 1 to 2.
  The broad pass (`mkv3-p4b-attn-c-s1024-current-8248cbe-20260705T0348ATTNC.log`)
  rejected `1/1`, `1/2`, `3/2`, and `4/1` hard, while `2/2` beat current `2/1` by
  -8.1us with 92/100 wins. Longer confirmation
  (`mkv3-p4b-attn-c-s1024-c22-confirm-8248cbe-20260705T0349ATTNC.log`) measured -13.0us
  with 139/180 wins. Route guard (`mkv3-p4b-attn-c-s1024-c22-route-20260705T0350ATTNC.log`)
  shows only S1024 changes to `DKV_C=2/DQ_C=2`; nano, S2048, S4096, and small are
  unchanged. Full model validation passed
  (`mkv3-p4b-attn-c-s1024-c22-testmodel-20260705T0351ATTNC.log`), patched default-vs-old
  timing confirmed -10.9us with 141/180 wins
  (`mkv3-p4b-attn-c-s1024-c22-default-vs-old-20260705T0352ATTNC.log`), and the
  construction-order check was clean (`mkv3-p4b-attn-c-s1024-c22-ordercheck-20260705T0355ATTNC.log`:
  -13.3us and -14.1us with 158/160 and 159/160 wins). Fresh score rows after this
  second S1024 change were noisy/worse than the paired delta would imply: 1332.2us and
  1352.3us vs graph+ ~778us
  (`mkv3-p4b-score-s1024-attnc22-20260705T0353SCORE.log`,
  `mkv3-p4b-score-s1024-attnc22-repeat-20260705T0354SCORE.log`); use the paired A/B for
  the route decision.

- Post-S1024-retune profile and compile-flag guard checks: current-head profile at
  `7beef96` shows nano 924.6us, small 3620.4us, and H256/S1024 1223.5us in the
  `%globaltimer` critical-path meter (`mkv3-p4b-profile-current-7beef96-20260705T0349PROFILE.log`,
  `mkv3-p4b-profile-s1024-current-7beef96-20260705T0352PROFILE.log`). S1024 is now led by
  `ATTN_DKV_WG` 149.1us, `ATTN_FWD_WG` 134.2us, lm-head NT 92.0us, MLP dX 86.0us, Drow
  74.7us, and RMS dx 70.4us. The DKV direct-atomic epilogue was not part of the original
  S1024 coverage, so a forced-old staged-smem check confirmed the current default
  decisively: old-minus-default +50.5us combined paired median with 200/200 default wins
  (`mkv3-p4b-attndkv-directatomic-s1024-current-7beef96-20260705T0353DKVA.log`). WGMMA
  attention fast-log remains safe but not an S1024 win: precise-minus-fastlog was -0.2us
  combined paired median with order-sensitive 97/200 fast-log wins
  (`mkv3-p4b-attnfwd-fastlog-s1024-current-7beef96-20260705T0402AFLOG.log`). QKNORM
  D=64 cache, also previously unmeasured at H256/S1024, is weakly positive and stays
  default-on: old-minus-qkbc +4.7us combined paired median with 134/200 qkbc wins
  (`mkv3-p4b-qknorm-d64-cache-s1024-current-7beef96-20260705T0404QKBWD.log`). No source
  route change from this guard pass.

- Current-head dW split-target resweep after the S1024/S128 retunes: no change. The
  low-memory paired sweep at `0277b86`
  (`mkv3-p4b-dwtarget-current-0277b86-20260705T0416DWSK.log`) found nano target48 only
  weakly positive (-5.7us paired median, 52/80 wins), while target32 and target96 lost.
  Small and S1024 did not support a new default either: small target64 was +3.6us with
  38/80 wins, and S1024 target64 was +3.8us with 37/80 wins; the other small/S1024
  candidates were noise or regressions. Keep the dW split target at K==1024 -> 96 and
  otherwise 64; no source route change.

- Early `dXN_f32` zero-fill no-go: moving the final head-dX fp32 workspace fill from the
  CE-bwd boundary into wave 0 passed full model parity under the temporary
  `MK_EARLY_DXN_ZERO` toggle
  (`mkv3-p4b-early-dxnzero-testmodel-20260705T0419DXNZ.log`), but paired timing rejected
  the schedule (`mkv3-p4b-early-dxnzero-ab-8cae8a5-20260705T0419DXNZ.log`). Nano
  regressed by +3.1us paired median with 32/80 early wins, H256/S1024 regressed by
  +2.8us with 29/80 wins, and small's -2.2us median with 50/80 wins was too weak and
  isolated to gate. Keep the late zero plus its wave boundary; no source route change.

- Current-head cold-cap resweep after the S1024 route retunes: move only the shallow
  nano-width S512 default from cap16 to uncapped cold work. The broad env resweep at
  `61f455f` (`mkv3-p4b-coldcap-current-61f455f-20260705T0420COLD.log`) found cap0
  positive on nano (-7.1us paired median, 53/70 wins) while small and H256/S1024 were
  noise. Longer guards (`mkv3-p4b-coldcap-cap0-confirm-61f455f-20260705T0421COLD.log`)
  confirmed nano cap0 at -6.6us with 129/180 wins, but rejected a broad S512 rule
  because deep-L12 regressed +10.5us with only 25/140 cap0 wins. S128/S256 cap0 were
  weak positives, so leave them at cap16. Route guard
  (`mkv3-p4b-coldcap-nano-route-20260705T0422COLD.log`) shows only nano changes to
  default cap0; deep/S128/S256 stay cap16, H256/S1024 stays cap64, small stays cap48,
  and long S stays uncapped. Full model validation passed
  (`mkv3-p4b-coldcap-nano-testmodel-20260705T0422COLD.log`), and patched default-vs-old
  timing confirmed nano cap0 over forced old cap16 by -7.2us paired median with 146/200
  default wins (`mkv3-p4b-coldcap-nano-default-vs-old-20260705T0422COLD.log`).

- Post-nano-cap profile and S512 attention chunk no-go: current-head profile at
  `0a4560a` (`mkv3-p4b-profile-current-0a4560a-20260705T0423PROFILE.log`) has nano
  at 920.2us with cap0, small at 3640.7us, and H256/S1024 at 1225.3us. Nano's cap0
  profile made `ATTN_DQ_WG` more visible on the hot path, but a focused S512 chunk
  resweep (`mkv3-p4b-attn-c-nano-cap0-0a4560a-20260705T0423ATTNC.log`) found no
  repeatable improvement over the current `DKV_C=3/DQ_C=2` default. Nearby `2/1`,
  `3/1`, `4/2`, and `2/2` were all noise-level (-1.0us to +0.4us paired medians with
  39-43/80 wins), while `DQ_C=3` variants regressed and `DKV_C=1/DQ_C=2` lost hard
  (+32.3us, 2/80 wins). Keep the S512 attention chunk gate unchanged.

- Post-nano-cap RMS dx R4 recheck: the cap0 scheduler change did not flip the prior
  H256/S512 R4 no-go. Forced `MK_RMS_DX_R4=1`
  (`mkv3-p4b-rmsdx-r4-current-7bb4905-20260705T0424RMS.log`) regressed nano by +9.9us
  paired median with only 1/120 R4 wins, deep-L12 by +33.4us with 2/100 wins, and
  H256/S1024 by +8.9us with 12/100 wins. Keep the default R2 dx path for these shapes;
  only the existing H256/S2048 R4 gate remains.

- Current-head `MK_ALLHOT` scheduler-policy recheck: after nano moved to cap0, forcing
  every instruction into the hot ready ring was still decisively negative
  (`mkv3-p4b-allhot-current-8e66b13-20260705T0426HOT.log`). All-hot regressed nano by
  +34.5us paired median with 1/100 wins, small by +64.7us with 1/80 wins, and
  H256/S1024 by +28.9us with 0/100 wins. Keep the hot/cold criticality split; uncapped
  nano cold work is not equivalent to single-ring scheduling.

- Current-head `MK_CLAIM` resweep after nano cap0: keep the scheduler claim quantum at
  132 (`mkv3-p4b-claim-current-f7d70c6-20260705T0427CLAIM.log`). Nano claim160 was only
  a weak isolated -2.4us paired median with 53/80 wins, while claim64/96/112 regressed
  by +56us to +111us, claim192 regressed +9.6us, and claim264 regressed +4.9us. The
  likely alternate claim160 is not shape-safe: small regressed +189.2us with 0/60 wins
  and H256/S1024 regressed +17.2us with 7/70 wins. No source route change.

- Current-head nano/deep head-dX target resweep after cap0: no change
  (`mkv3-p4b-headdx-nano-cap0-9cd0a88-20260705T0428HEADDX.log`). Nano target48/64 had
  superficially negative medians (-10.1us and -8.8us), but the paired distributions
  were heavily contaminated/noisy (p25/p75 roughly -55us/+40us) with only 51/80 and
  47/80 wins. Target32/128/192 regressed, and deep-L12 did not provide a clean guard
  for lower targets. Keep the current target96 for H256/S512 and deep-L12.

- Current-head short cold-cap retune: extend the shallow cap0 gate to H256/L4/S128
  only. A clean-GPU longer confirmation
  (`mkv3-p4b-coldcap-short-confirm-0482227-20260705T0430COLD.log`) measured S128 cap0
  over cap16 at -6.2us paired median with 165/240 wins, while S256 stayed too weak
  (-2.4us, 138/240 wins) and remains cap16. Route guard
  (`mkv3-p4b-coldcap-s128-route-20260705T0430COLD.log`) shows S128 and nano S512 at
  cap0, S256/deep at cap16, H256/S1024 at cap64, and small at cap48. Focused S128
  parity passed (`mkv3-p4b-coldcap-s128-parity-20260705T0430COLD.log`), full model
  validation passed (`mkv3-p4b-coldcap-s128-testmodel-20260705T0431COLD.log`), and
  patched default-vs-old timing confirmed S128 cap0 over forced old cap16 by -8.2us
  paired median with 189/220 default wins
  (`mkv3-p4b-coldcap-s128-default-vs-old-20260705T0431COLD.log`).

- Post-S128-cap profile and S128 attention chunk no-go: refreshed short profiles
  (`mkv3-p4b-profile-short-966add6-20260705T0432PROFILE.log`) show S128 at 725.5us
  with cap0 and S256 at 809.1us with cap16. S128 is now led by tiny WGMMA NN dX,
  `ATTN_DKV_WG`, RMS dx, and qk/MLP NT hops, so a focused attention chunk sweep tested
  nearby `DKV_C/DQ_C` pairs
  (`mkv3-p4b-attn-c-s128-current-966add6-20260705T0433ATTNC.log`). The broad pass made
  `4/1` look best (-10.6us, 75/90 wins), but longer confirmation weakened it to -4.2us
  with p75 above zero and 134/220 wins
  (`mkv3-p4b-attn-c-s128-c41-confirm-966add6-20260705T0434ATTNC.log`). The conservative
  `2/2` alternative regressed on confirmation (+7.0us, 66/220 wins:
  `mkv3-p4b-attn-c-s128-c22-confirm-966add6-20260705T0435ATTNC.log`). S256 guards also
  rejected `1/1` and `1/2` hard, with `2/2` weakly negative. Keep the current S128/S256
  `DKV_C=2/DQ_C=1` default.

- Post-S128-cap short n128 no-go: current-head route recheck
  (`mkv3-p4b-n128-s128-current-08e773c-20260705T0435N128.log`) kept the earlier short
  n128 verdict. S128 default has zero n128 GEMMs; forcing `MK_WGMMA_N128=1` routed 13
  NT GEMMs through n128 and regressed by +19.3us paired median with 1/80 wins. The
  lm-head-only mode was noise (+0.4us, 37/80 wins), and forcing lower n128 NN thresholds
  did not add NN n128 routes for the current flagged/split paths. S256 mode1 also
  regressed by +21.9us with 12/60 wins. Keep short-row n128 disabled except the existing
  lm-head route where the default gate already selects it.

- In-kernel valid-label reciprocal: remove the host-side PyTorch `labels >= 0` sum from
  the timed `MKQwen3.step()` path by adding a one-tile `OP_INV_VALID` root instruction
  that writes the same `inv_valid = 1 / max(valid_count, 1)` scalar consumed by CE
  forward/backward. The old behavior remains available with `MK_INV_VALID_IN_KERNEL=0`.
  Full model validation passed under the default path
  (`mkv3-p4b-invvalid-testmodel-20260705T0440INV.log`). Direct old-host-count vs
  new-in-kernel-count paired `step()` timing
  (`mkv3-p4b-invvalid-step-ab-20260705T0442INV.log`) was decisive on every flag shape:
  nano -75.3us, small -93.0us, S128 -71.5us, S256 -69.8us, and H256/S1024 -79.8us,
  each with 160/160 new-path wins. This does not shorten `prog.run()` profiles; it
  removes prelaunch work from the user-visible step path and the final scoreboard.

Post-input-bind focused scoreboard (df defaults, median-of-50, fresh process per config;
GPU 6 had no visible pmon process but retained an opaque allocation; baseline medians
wobble +-5-8% across runs from inductor autotune variance). Logs:
`mkv3-p4b-score-{nano,small,s128,s256,s1024,deep}-bindinputs-81fe5d0-20260705T0500SCORE.log`.
| config | megakernel | flash-baseline | gap |
|---|---|---|---|
| nano | 945 | 635 | 1.49x |
| small | 3649 | 1895 | 1.93x |
| deep-L12 | 2456 | 1774 | 1.38x |
| S=128 | 759 | 496 | 1.53x |
| S=256 | 833 | 549 | 1.52x |
| S=1024 | 1242 | 776 | 1.60x |
(Morning honest reset: nano 1.97x / small 2.52x.)

- Input-binding probe no-go: after moving `inv_valid` into the kernel, the remaining
  `step()` input-copy overhead looked tempting, but directly patching the device buftab
  entries to external token/label pointers was slower than the normal device copies
  (`mkv3-p4b-input-bind-probe-4aaf11a-20260705T0450BIND.log`). The pointer-bound path
  matched losses, but event timing regressed S128 by +15.0us, nano by +19.1us, and small
  by +8.8us, with wall-time medians also positive. Keep the simple token/label copies
  unless a C++ launcher-side pointer override is implemented and remeasured.

- Launcher-side input binding: the C++/kernel-side version of the previous idea is
  promoted. `megakernel_df` now accepts two optional buftab pointer overrides and writes
  them before the existing initialization sync; default `MKQwen3.step()` binds canonical
  CUDA int32 token/label inputs in df mode, records their stream lifetime, and falls
  back to the internal-copy path for alternate executors or non-canonical inputs
  (`MK_BIND_INPUTS=0` restores the old copy path). Full model validation passed
  (`mkv3-p4b-bindinputs-testmodel-final-20260705T0458BIND.log`). Fixed old-copy vs
  new-bind paired timing (`mkv3-p4b-bindinputs-step-ab-fixed-20260705T0457BIND.log`)
  measured S128 -12.6us (199/200 wins), nano -11.7us (197/200), H256/S1024 -12.3us
  (185/200), and small -22.5us (198/200). This is a user-visible `step()` win only;
  `prog.run()` profiles are unchanged.

- CE ignore-row skip no-go: skipping CE forward/backward work for `ignore_index` rows is
  not worth a route. A compile-flag branch version passed validation
  (`mkv3-p4b-ceskip-testmodel-20260705T0505CESKIP.log`) and looked strongly positive
  on small (`mkv3-p4b-ceskip-step-ab-20260705T0507CESKIP.log`: -31.9us, 200/200 wins),
  but the same branch was noise/slightly negative on nano. A later separate-opcode route
  kept V8192 on the old CE kernels and passed validation
  (`mkv3-p4b-ceopskip-testmodel-20260705T0512CESKIP.log`), but timing
  (`mkv3-p4b-ceopskip-step-ab-20260705T0514CESKIP.log`) erased the small win (-1.8us,
  121/200 wins) and regressed H256/S1024 by +9.5us with only 22/200 wins. Keep the
  current CE opcodes unchanged.

- Current post-step-path profile refresh: `%globaltimer` profiles at `e771105`
  (`mkv3-p4b-profile-current-e771105-20260705T0520PROFILE.log`,
  `mkv3-p4b-profile-ssweep-e771105-20260705T0521PROFILE.log`) show the kernel-side
  bottleneck map is unchanged by the `inv_valid` and input-bind wins. Small is still led
  by `ATTN_DKV_WG` 394us, `GEMMNN 1024x512x3072.wg` 331us, `SWIGLU_BWD` 279us, and
  `ATTN_FWD_WG` 262us on path. H256/S1024 is led by `ATTN_DKV_WG` 146us,
  `ATTN_FWD_WG` 130us, lm-head forward 93us, and MLP dX 86us. Short S128/S256 remain
  dominated by small WGMMA NN dX/head/attention spans plus fixed wait. The remaining
  high-value work is true op quality (attention/GEMM/SwiGLU/RMS) or a new scheduler
  mechanism; the known route knobs around attention chunks, n128, cold cap, claim, CE
  skip, and rowop folds have been rechecked and are no-go or already promoted.

- SWIGLU_BWD FMA derivative promotion: `MK_SWIGLU_FMA_DERIV` is now default-on and
  rewrites `dsilu` from `sig + sg * (1-sig)` to `fmaf(-sg, sig, sig + sg)`, with
  `MK_SWIGLU_FMA_DERIV=0` restoring the old expression for A/B. Focused SwiGLU and
  full-model correctness passed
  (`mkv3-p4b-swfma-testswiglu-20260705T0527SWFMA.log`,
  `mkv3-p4b-swfma-testmodel-20260705T0528SWFMA.log`). Paired same-process timing
  (`mkv3-p4b-swfma-step-ab-20260705T0529SWFMA.log`) was order-stable: nano -8.19us
  (189/200 wins), small -16.80us (194/200 wins), and H256/S1024 -10.67us (184/200
  wins). This is a narrow op-quality win; it does not change the remaining bottleneck
  ordering.

- Post-SWIGLU-FMA profile + attention masked-exp no-go: current-head profiles at
  `38c762f` (`mkv3-p4b-profile-post-swfma-38c762f-20260705T0535PROFILE.log`,
  `mkv3-p4b-profile-s1024-post-swfma-38c762f-20260705T0538PROFILE.log`) show the same
  op-quality bottleneck order after the SWIGLU arithmetic win. Small is led by
  `ATTN_DKV_WG` 397us, `GEMMNN 1024x512x3072.wg` 336us, `SWIGLU_BWD` 277us,
  `GEMMNN 1024x512x512.wg` 257us, and `ATTN_FWD_WG` 245us. H256/S1024 is led by
  `ATTN_DKV_WG` 140us, `ATTN_FWD_WG` 122us, lm-head forward 93us, and MLP dX 88us.
  A default-off `MK_ATTN_MASKED_EXP_SKIP` probe skipped `__expf(-inf)` for causal-masked
  diagonal entries inside the WGMMA attention fwd/dKV/dQ loops. Correctness passed
  (`mkv3-p4b-attnmask-testattention-20260705T0541MASK.log`,
  `mkv3-p4b-attnmask-testmodel-20260705T0543MASK.log`), but paired timing was a
  decisive regression (`mkv3-p4b-attnmask-step-ab-20260705T0544MASK.log`): S128 was
  only a weak/noisy -2.58us, while S256 regressed +32.02us, nano +31.30us, small
  +223.23us, and H256/S1024 +65.73us. Keep masked attention probabilities on the
  existing `exp(-inf) -> 0` path; the branch/predicate cost overwhelms saved SFU work.

- Attention dS FMA no-go: a default-off `MK_ATTN_DS_FMA` probe rewrote WGMMA attention
  backward `dS = P * (dP-Drow) * scale` as `P * fmaf(dP, scale, -Drow*scale)` in both
  dKV and dQ. Focused attention and full-model correctness passed
  (`mkv3-p4b-attndsfma-testattention-20260705T0550DSFMA.log`,
  `mkv3-p4b-attndsfma-testmodel-20260705T0552DSFMA.log`). All-shape timing
  (`mkv3-p4b-attndsfma-step-ab-20260705T0553DSFMA.log`) was mixed: S128 -14.24us,
  small -12.24us, and H256/S1024 -4.75us, but S256 regressed +4.62us and nano
  regressed +2.11us. A shape-gated version for `S==128 || S==1024` also passed
  correctness (`mkv3-p4b-attndsfma-gated-testattention-20260705T0557DSFMA.log`,
  `mkv3-p4b-attndsfma-gated-testmodel-20260705T0600DSFMA.log`) but still regressed
  protected shapes and lost H256/S1024 (`mkv3-p4b-attndsfma-gated-step-ab-20260705T0601DSFMA.log`:
  S128 -14.37us, small -8.10us, S256 +8.24us, nano +8.34us, H256/S1024 +5.63us).
  Keep the original `(dP-Drow)*scale` expression; the FMA form is not a safe default.

- RMSNorm split-dx FMA route promotion: the broad `MK_RMS_DX_FMA` arithmetic rewrite
  passed focused RMS/full-model validation but was mixed enough to reject as an
  all-shape default (`mkv3-p4b-rmsxfma-step-ab-20260705T0613RMSX.log`: S128 -14.45us,
  S256 +5.49us, nano +3.39us, H256/S2048 +9.73us). The promoted route uses a separate
  `OP_RMSNORM_BWD_DX_FMA` opcode only for H256/S128 split RMSNorm dx; other shapes keep
  the old opcode. Routed H256/S128 full-model validation passed with nine emitted route
  opcodes (`mkv3-p4b-rmsxfma-route-s128-testmodel-20260705T0626RMSX.log`), default
  validation passed (`mkv3-p4b-rmsxfma-route-default-testmodel-20260705T0627RMSX.log`;
  no-env H256/S128 default validation:
  `mkv3-p4b-rmsxfma-default-s128-testmodel-20260705T0636RMSX.log`), and route
  inspection confirmed S256/nano/small/H256S1024 emit zero route opcodes with identical
  instruction and claim tensors. Paired route-off/route-on timing
  (`mkv3-p4b-rmsxfma-route-step-ab-20260705T0629RMSX.log`) measured H256/S128 at
  -5.60us median with 167/200 route wins. Post-promotion default-vs-forced-old timing
  (`mkv3-p4b-rmsxfma-default-vs-old-s128-20260705T0637RMSX.log`) confirmed -5.98us
  median with 161/200 default wins. `MK_RMS_DX_FMA_ROUTE=0` forces the old path for A/B.

- ATTN_FWD_WG fast-log restoration: current-head profiling after `d2f4db7`
  (`mkv3-p4b-profile-current-d2f4db7-20260705T0602PROFILE.log`,
  `mkv3-p4b-profile-s1024-d2f4db7-20260705T0608PROFILE.log`) showed attention fwd/dkv
  still dominating small and H256/S1024, and source inspection found the existing
  `MK_ATTN_FAST_LOG` compile flag no longer affected `attention.cuh`. The WGMMA forward
  LSE write again uses `__logf` by default, with `MK_ATTN_FAST_LOG=0` restoring precise
  `logf` for A/B. Focused attention and full-model validation passed
  (`mkv3-p4b-aflogrestore-testattention-20260705T0611AFLOG.log`,
  `mkv3-p4b-aflogrestore-testmodel-20260705T0614AFLOG.log`). Paired default-vs-precise
  timing (`mkv3-p4b-aflogrestore-step-ab-20260705T0615AFLOG.log`) confirmed S128
  -9.66us (153/160 fast wins), S256 neutral -0.38us (85/160), nano -9.41us (193/200),
  small -20.08us (160/160), H256/S1024 -6.08us (126/160), and H256/S2048 -15.15us
  (120/120).

- Drow n128 WGMMA no-go: after the post-fast-log profile still showed the Drow-fused
  `dOatt = dX @ Wo` hop as a visible small/S1024 cost
  (`mkv3-p4b-profile-current-966990a-20260705T0610PROFILE.log`,
  `mkv3-p4b-profile-s1024-966990a-20260705T0614PROFILE.log`), a default-off
  `MK_DROW_WG_N128=1` probe added n128 Drow epilogue support and halved Drow tile
  counts on every tested shape. Focused epilogue correctness passed for small and
  H256/S1024 dimensions (`mkv3-p4b-drown128-focused-gemm-20260705T0620DROWN128.log`),
  and full-model validation passed
  (`mkv3-p4b-drown128-testmodel-20260705T0621DROWN128.log`), but paired timing
  (`mkv3-p4b-drown128-step-ab-20260705T0622DROWN128.log`) was a decisive regression:
  S128 +57.79us, S256 +62.86us, nano +62.78us, small +54.42us, H256/S1024 +64.37us,
  and H256/S2048 +34.30us. The temporary source route was removed; keep Drow on the
  existing m64n64 WGMMA epilogue.

- Post-fast-log attention chunk resweep: current-head env sweep after `05de747`
  (`mkv3-p4b-attn-c-resweep-05de747-20260705T0617ATTNC.log`) kept the existing small,
  H256/S1024, and H256/S2048 defaults, but H256/S512 nano flipped from `DKV_C=3,DQ_C=2`
  to `DKV_C=2,DQ_C=2`. Direct confirmation
  (`mkv3-p4b-attn-c-nano-c22-confirm-05de747-20260705T0621ATTNC.log`) measured -4.72us
  with 193/240 C22 wins. Route inspection confirmed only nano changes while small and
  long-S retain their defaults, full default model validation passed
  (`mkv3-p4b-attn-c-nano-c22-testmodel-20260705T0624ATTNC.log`), and post-promotion
  default-vs-forced-old timing
  (`mkv3-p4b-attn-c-nano-c22-default-vs-old-20260705T0625ATTNC.log`) confirmed -4.99us
  with 195/240 default wins. `MK_ATTN_DKV_C`/`MK_ATTN_DQ_C` still force the chunk counts.

- Current scoreboard after the fast-log/RMS/nano-attention wins:
  `mkv3-p4b-score-both-cfc2fb3-20260705T0627SCORE.log` measured nano at 948.6us
  megakernel vs 634.6us compile+CUDAGraph+ (1.49x gap) and small at 3638.2us
  megakernel vs 1895.1us compile+CUDAGraph+ (1.92x gap). The remaining small gap is
  still dominated by op quality rather than obvious route knobs: post-fast-log profiles
  show `ATTN_DKV_WG`, large WGMMA/n128 GEMMs, `SWIGLU_BWD`, Drow, and RMS dx as the
  largest repeated spans.

- SWIGLU exp2 sigmoid no-go: replacing the SwiGLU sigmoid's `__expf(-g)` with inline
  `ex2.approx.ftz.f32(-log2(e)*g)` was correctness-clean
  (`mkv3-p4b-swex2-ptx2-testswiglu-20260705T062457Z.log`,
  `mkv3-p4b-swex2-ptx2-testmodel-20260705T062457Z.log`), but all-shape timing was
  mixed and not promotable (`mkv3-p4b-swex2-ptx2-step-ab-20260705T062457Z.log`):
  S128 improved by -16.08us, while small regressed by +5.62us and H256/S1024 by
  +9.42us. Runtime S128 gating avoided the math on other shapes but added enough
  branch/code cost to regress fallbacks (`mkv3-p4b-swex2-s128-gpu5-step-ab-20260705T063639Z.log`:
  S128 -10.11us, S256 +4.53us, nano +3.41us, small +20.43us, H256/S1024 +6.43us,
  H256/S2048 +17.42us). A templated S128-only body was also correctness-clean
  (`mkv3-p4b-swex2-s128tmpl-testswiglu-20260705T063639Z.log`,
  `mkv3-p4b-swex2-s128tmpl-testmodel-20260705T063639Z.log`,
  `mkv3-p4b-swex2-promoted-s128-testmodel-20260705T063639Z.log`) and looked positive
  in co-resident A/B (`mkv3-p4b-swex2-s128tmpl-gpu5-step-ab-20260705T063639Z.log`:
  S128 -12.43us), but reverse construction order inverted the apparent win
  (`mkv3-p4b-swex2-promoted-default-ab-20260705T063639Z.log` vs
  `mkv3-p4b-swex2-promoted-s128-reverse-ab-20260705T063639Z.log`). Fresh single-model
  process medians removed that bias and were neutral
  (`mkv3-p4b-swex2-s128-freshproc-ab-20260705T063639Z.log`: median default-old
  +0.29us, mean -0.33us, 3/6 default wins). The temporary source route was removed;
  keep SwiGLU on `__expf` plus the promoted FMA derivative.

- ATTN_DKV old smem-drain epilogue recheck no-go: with `ATTN_DKV_WG` still a top
  repeated span, the pre-direct-atomic epilogue was revalidated via
  `MK_ATTN_DKV_DIRECT_ATOMIC=0`. Full-model correctness passed
  (`mkv3-p4b-adkvoff-testmodel-20260705T064840Z.log`), but paired timing
  (`mkv3-p4b-adkvoff-step-ab-20260705T064840Z.log`) strongly kept the current
  direct-atomic default: S128 -6.43us, S256 -21.52us, nano -29.34us, small -49.23us,
  H256/S1024 -47.76us, and H256/S2048 -49.46us for default minus old-smem. Keep
  `MK_ATTN_DKV_DIRECT_ATOMIC=1`.

- Small SwiGLU-BWD two-warp route: the rejected `MK_SWIGLU_BWD_R2=1` probe folded two
  rows into one warp and lost. The opposite split, two warps per row, is now a narrow
  H512/S1024 small default. `OP_SWIGLU_BWD_2W` maps each 8-warp block to four rows and
  splits the feature dimension across a row-local warp pair, reducing the six serial
  vector chunks per I=1536 row to three. Focused op correctness passed
  (`mkv3-p4b-swb2w-testops-20260705T065511Z.log`), as did full-model checks with the
  route forced on and with the promoted default
  (`mkv3-p4b-swb2w-testmodel-20260705T065511Z.log`,
  `mkv3-p4b-swb2w-small-testmodel-20260705T065511Z.log`,
  `mkv3-p4b-swb2w-default-testmodel-20260705T065511Z.log`). Broad timing was mixed
  (`mkv3-p4b-swb2w-step-ab-20260705T065511Z.log`): S128 looked positive but reversed
  under construction-order control, S256 was neutral, nano/H256-S1024/H256-S2048
  regressed, and small was weakly positive. Reverse-order timing confirmed only small
  (`mkv3-p4b-swb2w-short-reverse-ab-20260705T065511Z.log`), and fresh single-model
  process medians kept the small win in 6/6 pairs
  (`mkv3-p4b-swb2w-small-freshproc-ab-20260705T065511Z.log`: median -5.76us,
  mean -6.80us). Route inspection confirmed only small emits the new op by default
  (`mkv3-p4b-swb2w-route-20260705T065511Z.log`), and final default-vs-forced-old timing
  measured -12.22us median with 193/240 default wins
  (`mkv3-p4b-swb2w-default-vs-old-20260705T065511Z.log`). `MK_SWIGLU_BWD_2W=0`
  restores the old route for A/B. Post-promotion small profile
  (`mkv3-p4b-post-swb2w-small-profile-20260705T070715Z.log`) measured 3648.0us total;
  the next small leaders are `ATTN_DKV_WG`, `GEMMNN 1024x512x3072.wg`,
  `SWIGLU_BWD_2W`, Drow `GEMMNN 1024x512x512.wg`, `ATTN_FWD_WG`, and RMS dx.

- Post-SwiGLU-2W small RMS dx FMA recheck: because RMS dx remains on path after the
  small SwiGLU win, `MK_RMS_DX_FMA_ROUTE=1` was temporarily loosened to force
  `OP_RMSNORM_BWD_DX_FMA` on H512/S1024 small. The route emitted 17 FMA dx ops and
  passed small gradient parity (`mkv3-p4b-rmsxfma-small-testmodel-20260705T071005Z.log`),
  but paired timing was negative/noise-level
  (`mkv3-p4b-rmsxfma-small-step-ab-20260705T071005Z.log`: +2.40us median, +1.86us
  mean, 94/240 route wins). The temporary source relaxation was reverted; keep the FMA
  route limited to H256/S128.

- Small SwiGLU-BWD four-warp split no-go: extending the promoted small 2W route to an
  opt-in 4W route (four warps per I=1536 row, two rows per block) was correctness-clean
  but slower. Focused opcode validation passed for S64/I160, S128/I768, S128/I1536, and
  S1024/I1536, including both bf16 and fp32 `dh` inputs
  (`mkv3-p4b-swb4w-testops-20260705T0716Z.log`). Small model route inspection under
  `MK_SWIGLU_BWD_4W=1` emitted 8 `OP_SWIGLU_BWD_4W` instructions and passed one-step
  gradient parity with worst rel err 0.0229
  (`mkv3-p4b-swb4w-small-testmodel-20260705T0716Z.log`). Paired timing against the
  current 2W default lost decisively
  (`mkv3-p4b-swb4w-small-step-ab-20260705T0716Z.log`: 2W 3653.41us median, 4W
  3663.52us median, delta +10.46us median / +10.26us mean, 36/240 4W wins). The
  temporary 4W source route was reverted; keep `OP_SWIGLU_BWD_2W` as the small default.

## Honest assessment + v2 roadmap

compile+CUDAGraph remains ~2.0x faster on the current flag-planting configs. The
measured structural gap, in order:
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
