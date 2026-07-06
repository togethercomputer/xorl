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

- WGMMA Drow direct-store promotion: the fused Drow epilogue in `dOatt = dX @ Wo`
  used `atomicAdd` even when D=64 makes each WGMMA tile own a complete head and every
  `drow[qh,row]` element has a single writer. `MK_DROW_DIRECT_STORE=1` compiles a
  direct assignment for that safe case. Small one-step gradient parity passed with all
  8 Drow GEMMs on WGMMA (`mkv3-p4b-drowstore-small-testmodel-20260705T0726Z.log`).
  Broad env A/B was positive for S128 (-26.45us), S256 (-15.30us), nano/S512
  (-17.47us), small (-20.67us), and H256/S1024 (-16.66us), but H256/S2048 regressed
  slightly (`mkv3-p4b-drowstore-broad-step-ab-20260705T0726Z.log`). A runtime branch
  inside the `_drowst` extension still hurt S2048
  (`mkv3-p4b-drowstore-guard-step-ab-20260705T0735Z.log`), so the default gate is at
  extension selection: D=64 and S<2048 build `_drowst`; S2048 and D=128 keep the old
  atomic extension. Route checks confirmed the gate and env override
  (`mkv3-p4b-drowstore-route-20260705T0735Z.log`), and default model validation passed
  (`mkv3-p4b-drowstore-default-testmodel-20260705T0735Z.log`). Final default-vs-old
  timing (`mkv3-p4b-drowstore-default-vs-old-20260705T0735Z.log`) measured S128
  -1.52us, S256 -14.62us, nano/S512 -12.18us, small -31.62us, and H256/S1024 -4.51us
  median; H256/S2048 default compiles the old no-`_drowst` extension. Set
  `MK_DROW_DIRECT_STORE=0` to force the old atomic Drow path for A/B.

- Post-Drow-direct-store scoreboard: `mkv3-p4b-score-both-e71d706-20260705T0749Z.log`
  measured nano at 932.3us megakernel vs 632.1us compile+CUDAGraph+ (1.47x gap) and
  small at 3607.3us megakernel vs 1904.6us compile+CUDAGraph+ (1.89x gap). Compared
  with the pre-Drow scoreboard at `cfc2fb3`, this trims the megakernel side by about
  16us nano and 31us small while the hardened graph baseline moved within normal
  compile-run variance.

- WGMMA attention `exp2.approx` promotion: `MK_ATTN_EXP2_APPROX=1` swaps WGMMA
  attention's exp calls to `ex2.approx.ftz.f32` with a log2(e) scale. Focused opcode
  validation passed full `test_ops.py`, including WGMMA attention forward/backward
  (`mkv3-p4b-aex2-testops-20260705T0757Z.log`), forced-on model parity passed
  (`mkv3-p4b-aex2-testmodel-20260705T0757Z.log`), and the final gated default model
  test passed (`mkv3-p4b-aex2-gated-default-testmodel-20260705T0819Z.log`). The first
  default gate covered all D=64 WGMMA attention shapes, but short-S timing was not
  robust enough (S128 regressed, S256 was neutral), so the promoted default is D=64,
  S>=512, and S%128==0. Route checks confirmed S128/S256 and D128 keep the old
  extension by default, while nano/S512, small/S1024, and H256/S2048 use `_aex2`;
  `MK_ATTN_EXP2_APPROX=0/1` remains an explicit A/B override
  (`mkv3-p4b-aex2-gated-route-20260705T0811Z.log`). Final gated default-vs-old timing
  (`mkv3-p4b-aex2-gated-default-vs-old-20260705T0819Z.log`) measured nano/S512
  -4.93us, small/H512/S1024 -37.70us, H256/S1024 -16.83us, and H256/S2048 -20.10us
  median; the S128 and D128 entries are same-extension controls, not changed code
  paths.

- Post-attention-exp2 scoreboard: `mkv3-p4b-score-both-9746ccd-20260705T0822Z.log`
  measured nano at 926.3us megakernel vs 635.2us compile+CUDAGraph+ (1.46x gap).
  The same both-shape run had a contaminated small megakernel row (4008.9us), so small
  was rerun separately: `mkv3-p4b-score-small-rerun-9746ccd-20260705T0826Z.log`
  measured 3586.4us megakernel vs 1901.5us compile+CUDAGraph+ (1.89x gap), with a
  megakernel-only sanity rerun at 3529.8us median
  (`mkv3-p4b-score-small-megakernel-rerun-9746ccd-20260705T0822Z.log`). Relative to
  the post-Drow scoreboard, the committed `_aex2` default trims about 6us nano and
  about 21us small on the standard bench path.

- Post-attention-exp2 profile and lm-head CE partial `exp2.approx` promotion: the
  current post-`91a8fb7` profile (`mkv3-p4b-profile-post-aex2-91a8fb7-20260705T0827Z.log`)
  measured nano at 909.6us and small at 3606.1us. Small is still led by `ATTN_DKV_WG`
  376.3us, MLP dX `GEMMNN 1024x512x3072.wg` 326.1us, `SWIGLU_BWD_2W` 289.1us,
  `ATTN_FWD_WG` 254.9us, RMS dx 244.6us, Drow 226.2us, and lm-head forward 200.2us.
  `MK_LMHEAD_EXP2_APPROX=1` changes only the fused lm-head GEMM CE/LSE partial
  epilogue's online sumexp from `__expf` to `ex2.approx.ftz.f32`; CE backward and the
  standalone CE reduction are unchanged. Forced-on full model validation passed
  (`mkv3-p4b-lex2-testmodel-20260705T0828Z.log`), and default gated validation passed
  (`mkv3-p4b-lex2-default-testmodel-20260705T0846Z.log`). The final default gate is
  V>=8192, V%64==0, and S>=256: route checks confirmed S128 and D128/V4096 keep the
  old extension by default, while S256, nano/S512, small/S1024, and H256/S2048 use
  `_lex2`; `MK_LMHEAD_EXP2_APPROX=0/1` remains an explicit A/B override
  (`mkv3-p4b-lex2-gated-route-20260705T0847Z.log`). Broad forced timing was positive
  on the main shapes (`mkv3-p4b-lex2-step-ab-20260705T0832Z.log`), and final gated
  default-vs-old timing (`mkv3-p4b-lex2-gated-default-vs-old-20260705T0847Z.log`)
  measured S256 -8.32us, nano/S512 -19.58us, small/H512/S1024 -56.99us, H256/S1024
  -36.00us, and H256/S2048 -65.95us median. S128 and D128 in that final matrix are
  same-extension controls, not changed code paths.

- Post-lm-head-exp2 scoreboard: `mkv3-p4b-score-both-400dc74-20260705T0849Z.log`
  measured nano at 905.2us megakernel vs 632.2us compile+CUDAGraph+ (1.43x gap) and
  small at 3542.8us megakernel vs 1912.9us compile+CUDAGraph+ (1.85x gap). Relative to
  the clean post-attention-exp2 rows, `_lex2` trims about 21us nano and about 44us
  small on the standard bench path; the remaining small gap is still mostly
  attention/GEMM/SwiGLU/RMS op quality rather than CE materialization.

- Post-lm-head-exp2 profile: `mkv3-p4b-profile-post-lex2-bd76802-20260705T0854Z.log`
  measured nano at 882.2us and small at 3541.7us. The lm-head forward hop dropped to
  130.2us small / 38.7us nano, so it is no longer the near-top small item. Current
  small leaders are `ATTN_DKV_WG` 372.4us, MLP dX `GEMMNN 1024x512x3072.wg` 333.6us,
  `SWIGLU_BWD_2W` 287.6us, `ATTN_FWD_WG` 257.2us, RMS dx 245.2us, and Drow
  `GEMMNN 1024x512x512.wg` 216.9us. The remaining high-value work is true op quality
  in those kernels or a scheduler/dependency mechanism, not another CE/lm-head epilogue
  cleanup.

- CE forward `exp2.approx` no-go: a default-off `MK_CE_EXP2_APPROX=1` probe swapped
  only `OP_CE_FWD`'s online sumexp/merge exponentials from `expf` to
  `ex2.approx.ftz.f32`; CE backward kept libm `expf` per the earlier measured revert.
  The first shared-checkout timing attempt was invalidated by overlapping SSQ/RMS WIP,
  so the decision moved to clean sibling worktree `/home/apanda/xorl-oss-cex2-probe`.
  Forced-on full model validation passed
  (`mkv3-p4b-cex2-clean-testmodel-20260705T0908Z.log`; training sanity 9.0496 ->
  6.0712). The first GPU 5 paired A/B built old first and looked positive
  (`mkv3-p4b-cex2-gpu5-step-ab-20260705T0917Z.log`: S128 -15.94us, S256 -12.22us,
  nano/S512 -13.55us, small -7.33us, H256/S1024 -4.05us, H256/S2048 -37.10us,
  D128/ragged -10.03us median). But the actual promoted-default check reversed the
  construction order and refuted the change
  (`mkv3-p4b-cex2-gated-default-vs-old-20260705T0937Z.log`): default `_cex2`
  regressed S128 +18.19us, S256 +8.58us, nano/S512 +7.26us, small +7.73us, and
  H256/S2048 +9.74us median; only H256/S1024 (-0.53us) and D128/ragged (-0.78us) were
  neutral/tiny positive. Source experiment reverted, including the inert `mk.py`
  `MK_CE_EXP2_APPROX` plumbing that had slipped into the idle-poll commit; keep
  `OP_CE_FWD` precise.

## Overnight-state certified scoreboard (2026-07-05 ~10:00Z, both sessions' work)

Clean-GPU util guards, median-of-50, fresh process per config, flash baseline:

| config | megakernel | baseline | gap | (24h ago) |
|---|---|---|---|---|
| nano  (H256 L4 S512)  | 917  | 634  | 1.45x | 2.06x-soft |
| small (H512 L8 S1024) | 3570 | 1904 | 1.88x | 2.06x-soft |
| deep-narrow (L12)     | 2400 | 1762 | **1.36x** | 1.83x-soft |
| S=128                 | 750  | 485  | 1.55x | 2.28x-soft |
| S=256                 | 815  | 554  | 1.47x | 2.03x-soft |
| S=1024 (nano width)   | 1209 | 780  | 1.55x | 2.02x-soft |
| S=4096                | 3349 | 1569 | 2.13x | — |
| S=8192                | 8395 | 3125 | 2.69x | — |

The short-S / deep-chain regime — the megakernel's original thesis territory —
is where the gap is closing fastest (deep-L12 1.36x). The long-S regime remains
attention-scaling-bound vs FA3. Note the "24h ago" column is vs the SOFT (math-
attention) baseline; the honest morning reset was nano 1.97x / small 2.52x, so
the true 24h movement is 1.97->1.45 and 2.52->1.88.

- QKNORM/ROPE-bwd V split no-go: current-head profile after the SSQ/scoreboard/CE
  cleanup (`mkv3-p4b-profile-current-1ef905d-20260705T1003Z.log`) still showed
  QKNORM/ROPE bwd at 54.2us nano and 188.0us small, so an isolated worktree
  `/home/apanda/xorl-oss-qkbwd-splitv` tested `MK_QKBWD_SPLIT_V=1`: V-head
  fp32->bf16 pass-through moved out of `OP_QKNORM_ROPE_BWD` into a separate disjoint-slot
  row op while Q/K norm+rope wrote only the q/k slot of `dQKVraw`. Focused QKNORM op
  correctness and full model validation passed
  (`mkv3-p4b-qkbwd-splitv-testqk-20260705T1014Z.log`,
  `mkv3-p4b-qkbwd-splitv-testmodel-20260705T1014Z.log`; training sanity 9.0496 ->
  5.9191), but timing did not promote. Paired step timing
  (`mkv3-p4b-qkbwd-splitv-step-ab-20260705T1014Z.log`) was order-sensitive: nano
  measured -1.15us/-4.30us split-minus-control across construction orders, while small
  flipped +3.58us/-3.76us. Profile A/B showed why this is not a safe default: QKNORM
  span dropped locally (nano 43.7->35.6us, small 163.0->157.5us), but on-path wait
  rose (nano 9.6->13.7us, small 20.6->32.3us) and total profiles regressed/flattened
  (nano 889.3->891.4us, small 3556.8->3575.5us;
  `mkv3-p4b-qkbwd-splitv-control-prof-20260705T1014Z.log`,
  `mkv3-p4b-qkbwd-splitv-variant-prof-20260705T1014Z.log`). Leave the implementation
  unmerged; the dependency/gating cost of the extra row op eats the local V-copy saving.

- Current-head small cold-cap recheck: after the late op changes, an env-only sweep
  retested H512/S1024 `MK_COLD_CAP` around the current cap48 default. The broad pass
  (`mkv3-p4b-coldcap-current-eae531e-20260705T1030Z.log`) made cap16 look best
  (3507.49us vs cap48 3516.03us), but paired confirmation refuted it as
  construction-order noise (`mkv3-p4b-coldcap16-confirm-eae531e-20260705T1030Z.log`):
  cap16-cap48 measured -7.47us with cap48 built first, then +4.19us with cap16 built
  first. Keep the H512/S1024 small default at cap48.

- CE backward `ex2.approx` promotion: `MK_CE_BWD_EXP2_APPROX=1` changes only
  `OP_CE_BWD`'s softmax exponentials from libm `expf` to the inline
  `ex2.approx.ftz.f32` helper; CE_FWD stays on the existing `MK_CE_EXP2_APPROX` path and
  is not touched by the default. The earlier broad `MK_CE_EXP2_APPROX` validation was
  treated as exploratory because it also affected CE_FWD, then the corrected
  CE-BWD-only route passed focused and full validation
  (`mkv3-p4b-ceb2-isolated-testce-20260705T1049Z.log`,
  `mkv3-p4b-ceb2-isolated-wide-ce-20260705T1049Z.log`,
  `mkv3-p4b-ceb2-isolated-testmodel-20260705T1049Z.log`; training sanity 9.0496 ->
  5.6440). Broad forced timing was cleanly positive for S>=1024 but mixed/order-sensitive
  for short/nano/D128 (`mkv3-p4b-ceb2-isolated-broad-ab-20260705T1049Z.log`), so the
  default gate is `S >= 1024`, `V >= 8192`, and `V % 8 == 0`; env
  `MK_CE_BWD_EXP2_APPROX=0/1` force-overrides for A/B. Route check
  (`mkv3-p4b-ceb2-route-20260705T1049Z.log`) confirmed S128/S256/nano/D128 stay old by
  default while H256/S1024, H512/S1024 small, and H256/S2048 use `_ceb2`. Final
  default-vs-forced-old timing (`mkv3-p4b-ceb2-default-vs-old-20260705T1049Z.log`)
  measured H256/S1024 -6.27us/-5.58us, small -35.65us/-34.35us, and H256/S2048
  -16.53us/-21.95us across construction orders. Small default-vs-old gradient diff also
  passed (`mkv3-p4b-ceb2-small-old-vs-default-grads-20260705T1049Z.log`; worst rel err
  0.009625).

- Post-CE-backward-exp2 scoreboard: `mkv3-p4b-score-both-4129b84-20260705T1130Z.log`
  measured nano at 917.6us megakernel vs 631.2us compile+CUDAGraph+ (1.45x gap), as
  expected unchanged by the default gate because S512 keeps the old CE-BWD route. Small
  measured 3511.9us megakernel vs 1906.1us compile+CUDAGraph+ (1.84x gap), improving
  from the overnight 3570/1904 row and the post-lm-head 3542.8/1912.9 row. The paired
  default-vs-old CE-BWD timing attributed roughly 30-35us of small-step savings to the
  `_ceb2` route; the scoreboard sees the same improvement class.

- Current-head TN-WGMMA dW gate broadening: the earlier long-S-only TN gate became
  stale after the late n128/lm-head/CE-BWD route changes. A small route resweep
  (`mkv3-p4b-small-route-resweep-3994aa1-20260705T1150Z.log`) rejected nearby attention
  chunk, head-dX, dW-target, and n128 changes, but forced `MK_WGMMA_TN=1` routed 33
  H512/S1024 dW instructions and beat the default by -95.7us/-70.7us across
  construction orders. The follow-up guard
  (`mkv3-p4b-tnwg-current-guard-3994aa1-20260705T1155Z.log`) passed default-vs-forced-TN
  parity for nano, H256/S1024, and small, then set the new boundary: forced TN regressed
  S128 (+4.6/+34.3us), S256 (+44.0/+38.5us), nano (+40.4/+43.2us), and H256/S1024
  (+40.4/+55.7us), but won H512/S1024 small (-128.2/-56.6us) and H256/S2048
  (-61.5/-32.4us). The default gate is now `K >= 2048` plus the H512-style K=1024 dW
  shapes with `min(M, N) >= 512`; `MK_WGMMA_TN=0/1` still force-disables/enables it for
  A/B. Route check (`mkv3-p4b-tnwg-gated-route-20260705T1202Z.log`) confirmed S128,
  S256, nano, and H256/S1024 keep zero TN routes by default while small, H256/S2048,
  and H256/S3072 route TN. Final default-vs-forced-old validation
  (`mkv3-p4b-tnwg-gated-default-vs-old-20260705T1205Z.log`) passed small and
  H256/S2048 gradient parity, then measured default-minus-old at -77.4us/-87.4us for
  small and -28.6us/-63.7us for H256/S2048 across construction orders.

- Post-TN-WGMMA-dW scoreboard: `mkv3-p4b-score-both-40887f3-20260705T1210Z.log`
  measured nano at 916.8us megakernel vs 628.4us compile+CUDAGraph+ (1.46x gap),
  effectively unchanged because nano keeps zero TN dW routes by default. Small measured
  3428.4us megakernel vs 1890.8us compile+CUDAGraph+ (1.81x gap), an 83.5us
  megakernel-side reduction from the post-CE-BWD scoreboard's 3511.9us row and matching
  the final default-vs-old TN timing class.

- Current small `SWIGLU_BWD_2W` reciprocal no-go: re-testing the old sigmoid
  reciprocal idea only inside the current two-warps-per-row body was correctness-clean
  but slower. The isolated `MK_SWIGLU_BWD_2W_RCP_RN` probe in
  `/home/apanda/xorl-oss-swrcp2w-probe` passed small default-vs-variant parity
  (`mkv3-p4b-swrcp2w-small-parity-20260705T1238Z.log`: both routes emit 8
  `OP_SWIGLU_BWD_2W` instructions, loss delta 9.54e-07, worst grad rel effectively 0),
  but paired timing rejected it (`mkv3-p4b-swrcp2w-small-step-ab-20260705T1240Z.log`):
  default-then-variant median +13.4us, reverse construction order median +16.7us, and
  overall +14.6us median with reciprocal winning only 1/24 pairs. Keep the exact
  division form in `SWIGLU_BWD_2W`.

- Current-head small-only attention dS-FMA retest no-go: the older
  `MK_ATTN_DS_FMA` arithmetic rewrite helped small weakly before but was rejected as an
  S-only gate because protected H256/S1024 and nano shapes regressed. A new isolated
  compile-flag probe in `/home/apanda/xorl-oss-attndsfma-small-probe` retested it as a
  possible H512/S1024-only extension default. Small parity passed
  (`mkv3-p4b-attndsfma-small-parity-20260705T1302Z.log`: 8 `ATTN_DKV_WG` and 8
  `ATTN_DQ_WG` ops in both routes, loss delta 9.54e-07, worst grad rel 0.000001), but
  paired timing was noise/negative (`mkv3-p4b-attndsfma-small-step-ab-20260705T1308Z.log`):
  default-then-FMA median -0.74us, reverse construction order +4.53us, overall +2.05us
  median / +0.49us mean with FMA winning 10/24 pairs. Do not revive the dS-FMA route
  on the current head.

- H256/S2048 head-dX `sk=1` no-atomic promotion: route inspection showed the small and
  H256/S2048 `dlogits @ Wlm` head-dX GEMMs both used the split-K/fp32 atomic path with
  `sk_head=1`, which pays a zero-fill plus atomics without actually splitting K. An
  isolated program-construction probe removed the split-K flag only when
  `MK_HEAD_DX_NO_ATOMIC_SK1=1` and `sk_head==1`. Small parity passed
  (`mkv3-p4b-headdx-sk1-small-parity-20260705T1318Z.log`) but timing was neutral
  (`mkv3-p4b-headdx-sk1-small-step-ab-20260705T1322Z.log`: overall -0.64us median,
  +0.45us mean, 12/24 wins), so small remains on the old default. H256/S2048 parity
  passed (`mkv3-p4b-headdx-sk1-s2048-parity-20260705T1329Z.log`: old head row
  `flags=168, sk=1`, new `flags=136`, one fewer fill/instruction, worst grad rel
  0.005997), and paired timing won decisively
  (`mkv3-p4b-headdx-sk1-s2048-step-ab-20260705T1330Z.log`: -7.70us median, -7.46us
  mean, 19/20 wins). The default is therefore enabled only for H256/S2048; route/parity
  on the promoted main branch passed
  (`mkv3-p4b-headdx-sk1-promote-route-parity-20260705T1334Z.log`) and final
  default-vs-forced-old timing confirmed -8.06us median / -7.30us mean with 19/20 wins
  (`mkv3-p4b-headdx-sk1-default-vs-old-s2048-20260705T1340Z.log`). Set
  `MK_HEAD_DX_NO_ATOMIC_SK1=0` to force the old split atomic route, or `=1` to force the
  no-atomic `sk=1` route for A/B.

- QKNORM/ROPE-bwd D=64 FMA dx no-go: a default-off `MK_QKBWD_D64_FMA_DX` probe rewrote
  the cached D=64 q/k norm backward dx expression from `da*w - xh*dot` to
  `fmaf(-xh, dot, da*w)`. Nano and small default-vs-variant parity passed
  (`mkv3-p4b-qkbwdfma-parity-20260705T1402Z.log`: nano worst rel 0.004982, small worst
  rel 0.000001), but paired timing rejected promotion
  (`mkv3-p4b-qkbwdfma-step-ab-20260705T1408Z.log`). Nano was only noise-level
  (-0.99us median, -0.49us mean, 15/24 FMA wins), while small regressed (+2.78us
  median, +2.18us mean, 6/24 FMA wins). Keep the existing multiply/subtract expression.

- H512/S1024 small head-dX n128/fp32 promotion: the m64n128 WGMMA body now supports
  fp32 stores for the explicit head-dX no-atomic route. Generic `wgmma_n128_ok` still
  excludes fp32/split-K/accumulating routes; only `dlogits @ Wlm` with `sk_head==1`
  can opt into `flags=4232`. An isolated probe
  `/home/apanda/xorl-oss-headdx-n128f32-probe` passed default-vs-variant parity for
  small and H256/S2048 (`mkv3-p4b-headdx-n128f32-parity-20260705T1425Z.log`), then
  paired timing promoted small only
  (`mkv3-p4b-headdx-n128f32-step-ab-20260705T1426Z.log`: small -17.57us median /
  -18.84us mean, 20/24 wins; S2048 +7.17us median / +6.41us mean, 5/24 wins).
  Promoted route/parity on main passed
  (`mkv3-p4b-headdx-n128f32-promote-route-parity-20260705T1430Z.log`): small default
  is `[(155, 32, 4232, None)]`, `MK_HEAD_DX_N128_F32=0` restores old
  `[(156, 64, 168, 1)]`, while S2048 default remains the prior m64 no-atomic
  `[(83, 64, 136, None)]`. Final promoted default-vs-old small timing confirmed
  -10.61us median / -12.68us mean with 24/32 wins
  (`mkv3-p4b-headdx-n128f32-default-vs-old-small-20260705T1438Z.log`). Standard
  `test_model.py` passed (`mkv3-p4b-headdx-n128f32-testmodel-20260705T1440Z.log`).
  Use `MK_HEAD_DX_N128_F32=0` to force the old small split-atomic route, or
  `MK_HEAD_DX_N128_F32=1` with `MK_HEAD_DX_NO_ATOMIC_SK1=1` for forced A/B.

- Post-head-dX-n128 scoreboard/profile at `e5e641d`: `profile_df.py both df`
  (`mkv3-p4b-profile-current-e5e641d-20260705T1450Z.log`) measured nano 890.6us and
  small 3439.1us. Small now has `n_instr=288`; the promoted head-dX row is
  `GEMMNN 1024x512x16384.wg` with 32 tiles. `bench.py both`
  (`mkv3-p4b-score-both-e5e641d-20260705T1452Z.log`) measured nano 917.8us
  megakernel vs 628.3us compile+CUDAGraph+ (1.46x gap), and small 3435.0us
  megakernel vs 1914.1us compile+CUDAGraph+ (1.79x gap). The small scoreboard row is
  noise-close to the post-TN number but directionally consistent with the paired
  default-vs-old microbench; no further route knob is implied by this profile.

- Fused Drow/dOatt n128 route no-go: a default-off `MK_DROW_N128=1` probe in
  `/home/apanda/xorl-oss-drow-n128-probe` added Drow epilogue support to the m64n128
  WGMMA body and routed only the fused dOatt/Drow GEMM through it. Route smoke
  (`mkv3-p4b-drow-n128-route-20260705T1502Z.log`) showed the intended tile-count
  halving: nano Drow rows 16 -> 8 tiles and small rows 64 -> 32 tiles with
  `flags=5248`. Default-vs-n128 parity passed
  (`mkv3-p4b-drow-n128-parity-20260705T1510Z.log`: nano worst grad rel 0.005917,
  small worst rel 0.000001), but paired step timing rejected it decisively
  (`mkv3-p4b-drow-n128-step-ab-20260705T1512Z.log`): nano regressed +44.51us median /
  +45.98us mean with 0/24 wins, and small regressed +61.31us median / +66.87us mean
  with 0/24 wins. Do not route Drow/Drow-direct-store through n128.

- Fused qkv qk-norm/RoPE n128 route no-go: a default-off `MK_QKROPE_N128=1` probe in
  `/home/apanda/xorl-oss-qkrope-n128-probe` added q/k RMSNorm+RoPE epilogue support to
  the m64n128 WGMMA body and routed only the fused qkv projection through it. Route
  smoke (`mkv3-p4b-qkrope-n128-route-20260705T1523Z.log`) confirmed the intended
  tile-count halving: nano qkv+qkrope rows 32 -> 16 tiles and small rows 128 -> 64
  tiles with `flags=4482`. Default-vs-n128 parity passed
  (`mkv3-p4b-qkrope-n128-parity-20260705T1530Z.log`: nano worst grad rel 0.005032,
  small worst rel 0.000001), but timing rejected the route
  (`mkv3-p4b-qkrope-n128-step-ab-20260705T1532Z.log`): nano regressed +34.34us median /
  +34.90us mean with 0/24 wins, and small regressed +25.01us median / +25.10us mean
  with 1/24 wins. Do not route the fused qkv qkrope epilogue through n128.

- Nano head-dX n128 split-K promotion: the m64n128 WGMMA body now supports fp32
  split-K atomics for the explicit head-dX path. Generic `wgmma_n128_ok` still rejects
  split-K/fp32 routes; only `dlogits @ Wlm` opts in. An isolated probe
  `/home/apanda/xorl-oss-headdx-n128split-probe` showed target 48 is the only useful
  n128 split point for nano: route smoke
  (`mkv3-p4b-headdx-n128split-route-20260705T1542Z.log`) changed the nano head-dX row
  from `[(92, 96, 168, 6)]` to target-48 `[(92, 48, 4264, 6)]`, while H512/S1024 small
  stayed on the promoted no-atomic row `[(155, 32, 4232, None)]`. Default-vs-variant
  parity passed for targets 48/64/96
  (`mkv3-p4b-headdx-n128split-parity-20260705T1548Z.log`; target-48 loss delta
  -3.81e-06, worst grad rel 0.006172). Timing selected target 48:
  `mkv3-p4b-headdx-n128split-step-ab-20260705T1550Z.log` measured target 48 at
  -4.04us median / -3.58us mean with 14/16 wins, target 64 neutral, and target 96
  regressed; the stronger target-48 repeat
  (`mkv3-p4b-headdx-n128split-target48-repeat-20260705T1555Z.log`) measured
  -4.19us median / -3.73us mean with 30/32 wins. Promoted main route/parity passed
  (`mkv3-p4b-headdx-n128split-promote-route-parity-20260705T1600Z.log`): nano default
  is now `[(92, 48, 4264, 6)]`, `MK_HEAD_DX_N128_SPLIT=0` restores old
  `[(92, 96, 168, 6)]`, and small remains unchanged. Standard `test_model.py` passed
  (`mkv3-p4b-headdx-n128split-testmodel-20260705T1605Z.log`). Final promoted
  default-vs-old timing confirmed -4.87us median / -4.36us mean with 27/32 wins
  (`mkv3-p4b-headdx-n128split-default-vs-old-20260705T1610Z.log`). Use
  `MK_HEAD_DX_N128_SPLIT=0` to force the old nano split-K route or
  `MK_HEAD_DX_N128_SPLIT_TARGET=<tiles>` for target sweeps.

- Post-head-dX-n128-split scoreboard/profile at `d3276f1`: `profile_df.py both df`
  (`mkv3-p4b-profile-current-d3276f1-20260705T1620Z.log`) measured nano 894.0us and
  small 3473.2us. Nano now shows the promoted head-dX row as
  `GEMMNN 512x256x8192.wg.splitK` with 48 tiles and 37.6us span; small remains the
  unchanged 32-tile no-atomic `GEMMNN 1024x512x16384.wg`. `bench.py both`
  (`mkv3-p4b-score-both-d3276f1-20260705T1622Z.log`) measured nano at 921.0us
  megakernel vs 637.1us compile+CUDAGraph+ (1.45x gap), and small at 3454.7us
  megakernel vs 1915.5us compile+CUDAGraph+ (1.80x gap). The scoreboard run is
  noisier than the paired default-vs-old timing; use the paired A/B above for route
  attribution and this row as the current end-to-end snapshot.

- Nano MLP/qkv dX n128 split-K no-go: after adding n128 split-K support for head-dX,
  an isolated `MK_DX_N128_SPLIT=1` probe in
  `/home/apanda/xorl-oss-dx-n128split-probe` tried the same tile shape for the remaining
  nano `gemm_dx` fp32 workspaces. Corrected route inspection
  (`mkv3-p4b-dx-n128split-route-corrected-20260705T1638Z.log`) showed the intended
  changes: `512x256x1536` rows moved from 128 WMMA split-K tiles to 48/64/96/128 n128
  tiles by target, while `512x256x512` rows bottomed out at 64 n128 tiles for targets
  64+. Default-vs-variant parity passed for targets 48/64/96/128
  (`mkv3-p4b-dx-n128split-parity-20260705T1640Z.log`; worst grad rel <=0.007736). The
  first paired timing run rejected every target
  (`mkv3-p4b-dx-n128split-step-ab-20260705T1642Z.log`): target 48 regressed +110.69us
  median / +110.84us mean with 0/16 wins, target 64 regressed +160.30us, target 96
  regressed +187.81us, and target 128 regressed +183.49us. A later pmon check found
  another Claude session resident on GPU 3, so treat those timings as no-promote
  evidence rather than final clean attribution. Keep nano MLP/qkv dX on the existing
  WMMA split-K route unless a future clean repeat overturns this large negative result;
  n128 split-K is only promoted for head-dX.

- Current-head attention chunk resweep no-change: after the nano head-dX n128 split
  promotion, an env-only GPU 5 sweep rechecked nearby `MK_ATTN_DKV_C` /
  `MK_ATTN_DQ_C` values on the current head
  (`mkv3-p4b-attn-c-current-53f8233-20260705T1710Z.log`). Route inspection confirmed
  nano default remains `DKV_C=2/DQ_C=2` and small default remains `DKV_C=1/DQ_C=1`.
  Nano `C=1/1` and `1/2` lost hard (+29.32us and +33.26us median), while `2/1` and
  `3/2` were construction-order-biased/neutral (+2.00us and +0.30us overall median).
  Small over-split variants all regressed: `2/1` +63.72us median, `1/2` +20.32us,
  `2/2` +63.90us, and `3/1` +63.66us. Keep the current attention chunk gates. GPU 5
  had no pmon/compute-app process before and after, but did have resident memory; since
  no candidate was positive, no repeat is needed.

- Current-head cold-cap resweep no-change: an env-only GPU 5 sweep after the head-dX
  n128 promotions rechecked `MK_COLD_CAP` around nano's cap0 default and small's cap48
  default (`mkv3-p4b-coldcap-current-e5b5a9f-20260705T1720Z.log`). Nano cap8/16/32/48
  were pure construction-order bias: built second lost and built first won, with
  overall medians between +0.52us and +1.10us except cap8 at +1.01us. Small cap0/64/80
  were also order-biased/neutral (overall -0.88us, -1.63us, +0.26us medians), cap32
  regressed (+5.40us), and cap16's weak overall -6.70us median was refuted by reverse
  construction order (+8.02us median). Keep nano uncapped, small cap48, H256/S1024
  cap64, and long-S uncapped.

- Current-head dW split-target resweep no-change: an env-only GPU 5 sweep rechecked
  `MK_DW_TARGET_TILES` after the head-dX n128 promotions
  (`mkv3-p4b-dwtarget-current-992a6a7-20260705T1730Z.log`). Nano target32 regressed
  (+4.06us median), target96 regressed hard (+24.26us), and target48 was only
  order-biased/neutral (-0.66us median, 7/12 wins). Small target64 and target80 showed
  apparent wins only when built second, then reversed or went neutral when built first
  (overall -7.20us and +0.81us medians); target128 and target160 regressed hard
  (+60.94us and +62.88us). Keep the current split target policy:
  `K == 1024 -> 96`, otherwise 64.

- Current-head claim-quantum resweep no-change: an env-only GPU 5 sweep rechecked
  `MK_CLAIM` after the latest route changes
  (`mkv3-p4b-claim-current-906f272-20260705T1740Z.log`). Nano claim96/112 regressed
  by +56.22us/+55.38us median, claim192 regressed +12.30us, and claim160 was only
  order-biased/neutral (+0.39us median, 6/12 wins). Small rejected every non-default
  claim decisively: claim96 +267.50us, claim112 +273.32us, claim160 +131.69us, and
  claim192 +102.34us median. Keep the global claim quantum at 132.

- Current-head CE ignore-row skip recheck no-go: the old compile-flag branch that
  skipped CE math for ignored rows was re-probed from sibling worktree
  `/home/apanda/xorl-oss-ceskip-current-probe` as `MK_CE_IGNORE_SKIP=1`. Forced-on
  full-model validation passed (`mkv3-p4b-ceskip-current-testmodel-20260705T1825Z.log`).
  Current-head paired timing first rejected protected shapes but showed an apparent
  H512/S1024 small win (`mkv3-p4b-ceskip-current-step-ab-20260705T1830Z.log`: nano
  +3.02us, H256/S1024 +8.91us, small -18.72us). A guarded main-tree promotion attempt
  also passed standard plus small parity
  (`mkv3-p4b-ceskip-promote-validation-20260705T1845Z.log`) and a first small
  default-vs-old A/B was weakly positive
  (`mkv3-p4b-ceskip-default-vs-old-small-20260705T1855Z.log`: -2.56us median,
  141/240 wins). The construction-order repeat refuted the route decisively
  (`mkv3-p4b-ceskip-default-vs-old-small-repeat-20260705T1900Z.log`: +19.39us median,
  only 34/320 wins). The temporary main-tree promotion was reverted; keep the current
  CE fwd/bwd opcodes unchanged.

- Current-head SSQ epilogue recheck no-change: an env-only `MK_SSQ_FUSE=0` recheck on
  GPU 5 tested whether disabling WGMMA sum-of-squares epilogues still helps after the
  late route retunes (`mkv3-p4b-ssq-current-bf2934c-20260705T1915Z.log`). Nano was
  noise-level (-0.85us off-default median, 83/160 off wins), and small was
  construction-order-sensitive/no-promote (default-first -13.36us, off-first +5.31us,
  combined -2.64us with 64/120 off wins). H256/S1024 initially looked consistent
  (-4.96us, 110/140 off wins), but the focused confirmation refuted it
  (`mkv3-p4b-ssq-s1024-confirm-bf2934c-20260705T1930Z.log`): default-first favored
  SSQ-off by -11.81us while off-first reversed to +12.48us, combined +1.97us with only
  191/400 off wins. Keep `MK_SSQ_FUSE=1` default-on.

- Small SwiGLU sigmoid-cache promotion: H512/S1024 small now writes a bf16 sigmoid cache
  from `OP_SWIGLU_FWD` and feeds it to the existing `OP_SWIGLU_BWD_2W` route, removing
  the backward `__expf` recompute on the hot small SwiGLU backward path. The route is
  small-only by default (`H=512/S=1024/I=1536`) and `MK_SWIGLU_CACHE_SIG=0` restores the
  old recompute path for A/B. The isolated probe in
  `/home/apanda/xorl-oss-swiglu-cache-probe` passed explicit small validation
  (`mkv3-p4b-swcache-small-parity-20260705T1945Z.log`: 8 cached fwd ops, 8 cached 2W
  bwd ops, loss 9.79777 vs 9.79781, worst grad rel 0.025832) and paired timing
  (`mkv3-p4b-swcache-small-step-ab-20260705T1955Z.log`) measured cache-control
  -17.57us when control built first and -10.58us when cache built first, combined
  -13.44us with 264/320 cache wins. Main-tree validation passed explicit small parity
  (`mkv3-p4b-swcache-main-small-parity-20260705T2005Z.log`) and full default
  `test_model.py` coverage (`mkv3-p4b-swcache-main-testmodel-20260705T2010Z.log`).
  Final main default-vs-forced-old timing
  (`mkv3-p4b-swcache-default-vs-old-small-20260705T2020Z.log`) confirmed the promoted
  default: -16.10us when default built first, -2.72us when old built first, combined
  -8.37us with 279/400 default wins.
  Post-promotion profile/score refresh at `674d0ad`
  (`mkv3-p4b-profile-small-post-swcache-674d0ad-20260705T2030Z.log`,
  `mkv3-p4b-score-small-post-swcache-674d0ad-20260705T2035Z.log`) measured small at
  3509.7us vs compile+CUDAGraph+ 1897.8us. `SWIGLU_BWD_2W` is now below the top three
  small spans; current small leaders are `ATTN_DKV_WG`, MLP dX
  `GEMMNN 1024x512x3072.wg`, and `ATTN_FWD_WG`.

- SwiGLU DSSG cache no-go: the follow-up default-off branch
  `/home/apanda/xorl-oss-swiglu-dssg-cache-probe` tested
  `MK_SWIGLU_CACHE_DSSG=1`, writing bf16 `silu(g)` plus bf16 `dsilu(g)` from
  `OP_SWIGLU_FWD` so `OP_SWIGLU_BWD_2W` can skip the gate-half load and sigmoid
  derivative arithmetic. Correctness passed for H512/S1024 small
  (`mkv3-p4b-swdssg-small-parity-20260705T1355Z.log`: 8 cached fwd ops, 8 cached
  2W bwd ops, loss 9.79777 vs 9.79781, worst grad rel 0.025717), but paired timing
  against the current sigmoid-cache default was clearly negative
  (`mkv3-p4b-swdssg-vs-sig-small-step-ab-20260705T1355Z.log`): `sig_first`
  dssg-sig +65.15us median with only 1/200 DSSG wins, `dssg_first` +38.22us with
  2/200 wins, combined +53.15us with 3/400 wins. Do not promote the doubled cache;
  keep the smaller bf16 sigmoid cache as the default small route.

- N128 direct bf16-store no-go: the default-off branch
  `/home/apanda/xorl-oss-n128-direct-probe` tested `MK_N128_DIRECT_BF16=1`, bypassing
  the n128 WGMMA shared-memory accumulator drain for plain bf16-output tiles only.
  Correctness passed for H512/S1024 small
  (`mkv3-p4b-n128ds-small-parity-20260705T1403Z.log`: extension `_n128ds`, 50 n128
  GEMM rows, 32 plain direct-store candidates, loss 9.79777 vs 9.79781, worst grad rel
  0.025832). Paired timing was not promotable
  (`mkv3-p4b-n128ds-vs-default-small-step-ab-20260705T1403Z.log`): default-first
  regressed by +31.94us median with only 5/200 variant wins; variant-first was neutral
  at +0.98us median with 98/200 wins; combined regressed by +16.77us with 103/400
  wins. Keep the current coalesced shared-memory epilogue for n128 GEMMs.

- N128 three-stage feed no-go: the default-off branch
  `/home/apanda/xorl-oss-n128-stage3-probe` tested `MK_N128_STAGE3=1`, changing only
  the m64n128 WGMMA GEMM body from the current two-stage cp.async feed to a three-stage
  two-tile lead. The first schedule was caught by parity as a stage-reuse bug
  (`mkv3-p4b-n128s3-small-parity-20260705T1411Z.log`: loss 9.83752 vs 9.79781);
  the fixed schedule passed H512/S1024 small parity
  (`mkv3-p4b-n128s3-small-parity-20260705T1415Z.log`: extension `_n128s3`, 50 n128
  rows, loss 9.79776 vs 9.79781, worst grad rel 0.025832). Paired timing was a
  decisive no-go (`mkv3-p4b-n128s3-vs-default-small-step-ab-20260705T1415Z.log`):
  default-first regressed +82.40us with 0/200 stage3 wins, stage3-first regressed
  +51.38us with 0/200 wins, combined +66.62us with 0/400 wins. Keep the current
  two-stage n128 feed.

- Current-head executor-mode recheck no-change: after the small SwiGLU cache and n128
  no-go probes, a same-model H512/S1024 timing rechecked `df`, `df2`, and `ws`
  (`mkv3-p4b-mode-current-359a231-small-20260705T1420Z.log`). Current `df` remains
  decisively best. In order `df->df2->ws`, `df2-df` regressed +497.90us median and
  `ws-df` regressed +295.89us, both with 0/160 wins. In reverse order, `df2-df`
  regressed +460.51us and `ws-df` +335.46us, again both 0/160 wins. Combined:
  `df2-df` +482.78us and `ws-df` +314.10us with 0/320 wins each. Keep `mode="df"`.

- `megakernel_df2` hot/cold ready-ring port correctness no-go: the isolated branch
  `/home/apanda/xorl-oss-df2-hotcold-probe` tested whether the full region-watermark
  executor only looked bad because it lacked the current `df` hot/cold ready rings and
  cold-cap policy. The CUDA/host patch built and the loss stayed close, but executor
  agreement was intermittently outside the repo tolerance on H512/S1024 small, so no
  timing was trusted. Initial hot/cold+completion-hint correctness matched loss
  (9.797764 vs 9.797768) but drifted `w2.5` by 0.0825 rel vs `df`
  (`mkv3-p4b-df2hc-small-correctness-20260705T1425Z.log`). Removing the completion
  hint still drifted up to 0.1121 rel across alternating `df`/`df2` runs
  (`mkv3-p4b-df2hc-nohint-small-repeat-20260705T1429Z.log`), and uncapping cold work
  with `MK_COLD_CAP=0` still drifted up to 0.0711 rel
  (`mkv3-p4b-df2hc-nohint-cap0-small-repeat-20260705T1431Z.log`). Treat the hot/cold
  `df2` port as a scheduler-race no-go; leave the sibling dirty and keep `df2` parked.

- H256/S1024 1W SwiGLU sigmoid-cache promotion: the earlier small-only cache wrote a
  bf16 sigmoid cache only when the 2-warps-per-row small backward route was active,
  but the normal 1W `OP_SWIGLU_BWD` already had optional cache support. The isolated
  branch `/home/apanda/xorl-oss-swiglu-cache-1w-probe` wired the cache through the 1W
  route and forced it for non-small shapes. Route/parity passed for nano and H256/S1024
  (`mkv3-p4b-swcache1w-route-parity-20260705T1433Z.log`): both cached 4 SwiGLU fwd and
  4 normal 1W bwd ops with zero `OP_SWIGLU_BWD_2W`, with worst grad rel 0.006326 nano
  and 0.008897 H256/S1024. Timing kept nano as no-default-change/order-biased
  (`mkv3-p4b-swcache1w-step-ab-20260705T1440Z.log`: +3.44us when control constructed
  first, -4.38us in reverse, combined -1.26us with 176/320 wins), but H256/S1024 was a
  clean win (same log: -18.90us and -21.57us across construction orders, combined
  -20.24us with 319/320 wins). H256/S2048 was rejected
  (`mkv3-p4b-swcache1w-s2048-20260705T1441Z.log`: combined +16.56us with 6/192 wins).
  Main promotion gates the sigmoid cache for `H=256,S=1024,I=768` in addition to the
  existing `H=512,S=1024,I=1536` small route, and passes the optional cache arg to both
  2W and 1W SwiGLU backward bodies. Promoted validation
  (`mkv3-p4b-swcache1w-promote-h256s1024-20260705T1448Z.log`) confirmed default route
  `4 fwd/4 1W bwd cached`, forced-old route uncached, parity worst 0.009197, and
  default-vs-old timing at -20.13us / -5.76us by construction order, combined -12.32us
  with 286/320 wins. Use `MK_SWIGLU_CACHE_SIG=0` to force the old uncached route.

- Post-H256/S1024 sigmoid-cache profile/score refresh at `77603c1`: targeted
  `Cfg(S=1024)` attribution
  (`mkv3-p4b-profile-s1024-post-swcache1w-77603c1-20260705T1450Z.log`) measured
  1218.9us total, 76 chain hops, 193.3us on-path wait, and 1025.6us on-path span.
  The top H256/S1024 path totals remain op-quality bound: `ATTN_DKV_WG` 141.5us,
  `ATTN_FWD_WG` 120.6us, MLP dX `GEMMNN 1024x256x1536.wg` 90.2us, RMS dx 76.7us,
  qkv+qkrope `GEMMNT 1024x512x256.wg.+qkrope` 71.3us, GU/down GEMMs, lm-head,
  QKNORM/ROPE bwd, and cached 1W `SWIGLU_BWD` 55.7us. The matching hardened benchmark
  row (`mkv3-p4b-score-s1024-post-swcache1w-77603c1-20260705T1455Z.log`) measured
  megakernel 1237.7us vs compile+CUDAGraph+ 778.0us (1.59x gap). No new route knob is
  implied; this points back to attention/GEMM/RMS kernel quality and dependency length.

- H256/S1024 cached SwiGLU-BWD 2W promotion: after the 1W sigmoid-cache route became
  default for `H=256,S=1024,I=768`, an env-only check retested the existing
  two-warps-per-row backward body with the cache enabled. Forced cached 2W vs forced
  cached 1W (`mkv3-p4b-sw2w-cache-s1024-20260705T1456Z.log`) passed route/parity
  (`4 fwd/0 1W/4 2W` vs `4 fwd/4 1W/0 2W`, worst grad rel 0.008834) and won both
  construction orders: -11.39us and -4.78us, combined -7.92us with 278/320 2W wins.
  Main promotion widens `swiglu_bwd_2w_default` only to the same H256/S1024 shape
  already gated for sigmoid-cache. Final default-vs-forced-old validation
  (`mkv3-p4b-sw2w-cache-s1024-promote-20260705T1505Z.log`) confirmed default route
  `4 fwd/0 1W/4 2W`, forced-old cached 1W route, parity worst 0.007067, and
  default-old timing -2.67us / -14.10us by construction order, combined -8.46us with
  262/320 default wins. `MK_SWIGLU_BWD_2W=0` restores the cached 1W route for A/B;
  nano and H256/S2048 remain off per the earlier broad and S2048 no-go evidence.
  Post-promotion H256/S1024 profile
  (`mkv3-p4b-profile-s1024-post-sw2w-2e4a5cb-20260705T1508Z.log`) measured 1208.2us
  total; the top path is still attention/GEMM led (`ATTN_DKV_WG` 136.0us,
  `ATTN_FWD_WG` 128.9us, MLP dX 90.5us), with cached `SWIGLU_BWD_2W` at 75.9us.
  The matching score refresh
  (`mkv3-p4b-score-s1024-post-sw2w-2e4a5cb-20260705T1520Z.log`) measured megakernel
  1228.4us vs compile+CUDAGraph+ 778.0us (1.58x gap).

- Current-head H256/S1024 n128 NN recheck no-change: because the post-2W profile still
  had `GEMMNN 1024x256x1536.wg` on the path, an env-only `MK_WGMMA_N128_NN_MIN=16`
  recheck retested the earlier n128 NN alternative at current head. Route inspection
  (`mkv3-p4b-s1024-n128nn-current-20260705T1510Z.log`) flipped 8 `1024x256` NN rows
  through n128 (tile count 32 -> 16 for the K=1536/512 rows) while keeping lm-head
  split-K and qkrope rows on their existing routes. Parity passed (worst rel 0.006850),
  but paired timing rejected the route decisively: +22.24us and +26.98us
  n128-default by construction order, combined +24.37us with 2/320 n128 wins. Keep the
  current m64n64 WGMMA NN threshold for H256/S1024.

- Current-head H256/S1024 attention chunk recheck no-change: the post-2W profile put
  attention back at the top, so an env-only sweep rechecked nearby `MK_ATTN_DKV_C` /
  `MK_ATTN_DQ_C` values against the current `2/2` default. Route/parity/timing
  (`mkv3-p4b-s1024-attn-chunk-current-20260705T1515Z.log`) confirmed all tested
  variants were correctness-clean but slower: `1/2` regressed +62.19us with 0/240
  wins, `2/1` regressed +14.67us with 16/240 wins, `1/1` regressed +61.12us with 0/240
  wins, and `3/2` regressed +49.47us with 0/240 wins. Keep `DKV_C=2/DQ_C=2` for
  H256/S1024.

- H256/S2048 combined cached SwiGLU-BWD 2W promotion: the earlier S2048 cache-only
  result was a no-go, and the older broad 2W route was also not enough on its own, but
  the combined current-head route is positive. Forced `MK_SWIGLU_CACHE_SIG=1
  MK_SWIGLU_BWD_2W=1` vs current default
  (`mkv3-p4b-sw2w-cache-s2048-current-20260705T1525Z.log`) routed `4 fwd/0 1W/4 2W`
  instead of uncached `4 fwd/4 1W/0 2W`, passed parity (worst rel 0.009004), and won
  both construction orders: -10.21us and -9.73us variant-default, combined -10.11us
  with 165/192 variant wins. Main promotion widens both `swiglu_cache_sig_default` and
  `swiglu_bwd_2w_default` to `H=256,S=2048,I=768`. Final default-vs-forced-old
  validation (`mkv3-p4b-sw2w-cache-s2048-promote-20260705T1535Z.log`) confirmed default
  cached 2W route, forced-old uncached 1W route, parity worst 0.007088, and
  default-old timing -6.10us / -5.34us by construction order, combined -5.62us with
  145/192 default wins. `MK_SWIGLU_CACHE_SIG=0 MK_SWIGLU_BWD_2W=0` restores the old
  route for A/B.
  Post-promotion H256/S2048 profile
  (`mkv3-p4b-profile-s2048-post-sw2w-250838c-20260705T1540Z.log`) measured 1895.5us
  total; the path is attention-dQ led (`ATTN_DQ_WG` 335.3us, `ATTN_FWD_WG` 236.1us),
  with lm-head 114.6us, cached `SWIGLU_BWD_2W` 102.0us, head-dX 101.0us, QKNORM/ROPE
  bwd 99.9us, and MLP dX 98.1us next. The matching score refresh
  (`mkv3-p4b-score-s2048-post-sw2w-250838c-20260705T1542Z.log`) measured megakernel
  1908.9us vs compile+CUDAGraph+ 1043.3us (1.83x gap).

- H256/S3072 combined cached SwiGLU-BWD 2W promotion: after the S2048 win, the same
  combined route was rechecked at the next long-S shape. Forced `MK_SWIGLU_CACHE_SIG=1
  MK_SWIGLU_BWD_2W=1` vs current default
  (`mkv3-p4b-sw2w-cache-s3072-current-20260705T1548Z.log`) routed `4 fwd/0 1W/4 2W`
  instead of uncached `4 fwd/4 1W/0 2W`, passed parity (worst rel 0.010763), and won
  both construction orders: -13.73us and -8.48us variant-default, combined -10.99us
  with 113/128 variant wins. Main promotion widens both `swiglu_cache_sig_default` and
  `swiglu_bwd_2w_default` to `H=256,S=3072,I=768`. Final default-vs-forced-old
  validation (`mkv3-p4b-sw2w-cache-s3072-promote-20260705T1552Z.log`) confirmed default
  cached 2W route, forced-old uncached 1W route, parity worst 0.010763, and
  default-old timing -16.37us / -13.15us by construction order, combined -15.38us with
  124/128 default wins. `MK_SWIGLU_CACHE_SIG=0 MK_SWIGLU_BWD_2W=0` restores the old
  route for A/B.

- H256/S4096 combined cached SwiGLU-BWD 2W promotion: the same route was checked as a
  long-S boundary with fewer paired samples. Forced `MK_SWIGLU_CACHE_SIG=1
  MK_SWIGLU_BWD_2W=1` vs current default
  (`mkv3-p4b-sw2w-cache-s4096-current-20260705T1558Z.log`) routed `4 fwd/0 1W/4 2W`,
  passed parity (worst rel 0.007641), and was a smaller win: -3.71us and -5.07us
  variant-default by construction order, combined -4.21us with 59/80 variant wins.
  Main promotion widens both default gates to `H=256,S=4096,I=768`. Final
  default-vs-forced-old validation
  (`mkv3-p4b-sw2w-cache-s4096-promote-20260705T1602Z.log`) confirmed default cached 2W
  route, forced-old uncached 1W route, parity worst 0.007642, and default-old timing
  -6.61us / -0.54us by construction order, combined -5.23us with 52/80 default wins.
  This is a weaker long-S win than S2048/S3072 but still positive under construction
  order control; `MK_SWIGLU_CACHE_SIG=0 MK_SWIGLU_BWD_2W=0` restores the old route for
  A/B.
  Post-promotion H256/S4096 profile
  (`mkv3-p4b-profile-s4096-post-sw2w-ffb835f-20260705T1620Z.log`) measured 3397.8us
  total and is strongly attention-dQ led: `ATTN_DQ_WG` 922.8us, `ATTN_FWD_WG` 439.6us,
  RMS dx 253.7us, lm-head 224.2us, QKNORM/ROPE bwd 181.6us, and cached
  `SWIGLU_BWD_2W` 174.5us. The matching score refresh
  (`mkv3-p4b-score-s4096-post-sw2w-ffb835f-20260705T1622Z.log`) measured megakernel
  3370.2us vs compile+CUDAGraph+ 1584.1us (2.13x gap).

- H256/S8192 cached SwiGLU-BWD 2W boundary no-change: the same combined route was
  tested at S8192 and should not be defaulted. Forced `MK_SWIGLU_CACHE_SIG=1
  MK_SWIGLU_BWD_2W=1` vs current default
  (`mkv3-p4b-sw2w-cache-s8192-current-20260705T1608Z.log`) routed `4 fwd/0 1W/4 2W`
  and passed parity (worst rel 0.007589), but timing regressed in both construction
  orders: +50.88us and +45.22us variant-default, combined +47.47us with only 10/40
  variant wins. Keep the cached-2W default capped at H256/S4096.

- Current-head H256/S2048 attention DQ over-split no-change: after the cached-2W
  promotion, S2048's profile was attention-dQ led, so an env-only retest tried higher
  `MK_ATTN_DQ_C` around the current `DKV_C=2/DQ_C=2` default. Route/parity/timing
  (`mkv3-p4b-s2048-attn-dqsplit-current-20260705T1615Z.log`) kept the existing gate:
  `2/3` regressed +67.41us with 0/160 wins, `2/4` regressed +62.40us with 0/160 wins,
  and `1/3` regressed +46.82us with 0/160 wins. Keep H256/S2048 attention chunks at
  `DKV_C=2/DQ_C=2`.

- Current-head H256/S4096 attention chunk no-change: S4096's post-SwiGLU profile is
  even more attention-dQ led, but the current `DKV_C=1/DQ_C=1` default is still the
  best route. Env-only route/parity/timing
  (`mkv3-p4b-s4096-attn-dqsplit-current-20260705T1516Z.log`) tried `1/2`, `1/3`,
  `2/2`, and `2/1`; all variants passed default-vs-variant parity (worst rel <=0.007247)
  but regressed in paired timing. `1/2` lost +77.57us with 0/80 wins, `1/3` lost
  +125.12us with 0/80 wins, `2/2` lost +100.21us with 0/80 wins, and `2/1` still lost
  +21.63us with only 1/80 wins. Keep H256/S4096 attention chunks at `DKV_C=1/DQ_C=1`.

- Isolated `OP_ATTN_DQ_WG` S/dP WGMMA batching no-go: because S2048/S4096 remain
  attention-dQ led after the route knobs were exhausted, sibling worktree
  `/home/apanda/xorl-oss-attn-dq-x2-probe` added a default-off `MK_ATTN_DQ_X2_SD=1`
  probe that batches the independent `S=QK^T` and `dP=dO V^T` groups through
  `wga_mma64_x2`. The variant compiled and passed default-vs-variant parity
  (`/home/apanda/xorl-oss-attn-dq-x2-probe/results/mkv3-p4b-attn-dq-x2-current-20260705T1521Z.log`:
  H256/S2048 worst grad rel 0.003925; H256/S4096 worst grad rel 0.000001), but timing
  rejected it. S2048 was construction-order biased and negative overall (+3.87us
  combined with 28/80 wins), while S4096 regressed in both orders (+14.11us combined
  with 4/64 wins). Do not promote the x2 DQ source path; the dirty sibling remains
  no-go evidence only.

- Current-head long-shape RMS dx R4 recheck no-change: with S4096 still showing a large
  RMS dx span after cached-SwiGLU, an env-only `MK_RMS_DX_R4` sweep rechecked the long
  H256 defaults (`mkv3-p4b-rmsdx-r4-long-current-20260705T1526Z.log`). All routes passed
  default-vs-variant parity. H256/S2048 still keeps default R4; forced R2 was
  construction-order biased and negative overall (+6.51us variant-control with 26/80
  wins). H256/S3072 and H256/S4096 still keep default R2; forced R4 lost +3.74us
  (26/80 wins) and +4.37us (15/64 wins), respectively. Keep `rms_dx_r4` gated only for
  H256/S2048; `MK_RMS_DX_R4` remains just a sweep override elsewhere.

- Current-head H256/S3072 profile/score refresh: after the cached-SwiGLU promotion and
  follow-on no-go probes, S3072 now measures
  (`mkv3-p4b-profile-s3072-current-8058040-20260705T1528Z.log`) 2640.6us total with 76
  chain hops, 210.0us wait, and 2430.6us span. The path is attention-dQ led:
  `ATTN_DQ_WG` 569.3us, `ATTN_FWD_WG` 336.9us, lm-head forward
  `GEMMNT 3072x8192x256.wg` 222.7us, RMS dx 144.4us, QKNORM/ROPE bwd 141.7us, cached
  `SWIGLU_BWD_2W` 137.8us, qkv+qkrope 119.3us, and head-dX 105.5us. The matching
  hardened score (`mkv3-p4b-score-s3072-current-8058040-20260705T1528Z.log`) measured
  megakernel 2651.2us vs compile+CUDAGraph+ 1327.3us (2.00x gap). The same attention
  chunk/RMS/source-batching knobs just rechecked for S2048/S4096 do not change this
  route map.

- H256/S>=3072 head-dX `sk=1` no-atomic promotion: the S3072 profile exposed a
  head-dX row with split count one still paying the split-K zero-fill and fp32 atomic
  route. Env-only probe `mkv3-p4b-headdx-sk1-long-current-20260705T1530Z.log` moved
  H256/S3072 and H256/S4096 from `(flags 168, sk=1)` to the existing no-zero/no-atomic
  route `(flags 136, sk=None)`, passed parity, and improved S3072 by -7.39us
  (75/96 wins) and S4096 by -9.47us (71/80 wins). Boundary probe
  `mkv3-p4b-headdx-sk1-s8192-current-20260705T1530Z.log` also passed parity and
  improved S8192 by -21.92us (37/40 wins). After widening the default from H256/S2048
  to H256/S>=2048, clean promoted-default vs forced-old validation
  `mkv3-p4b-headdx-sk1-long-promote-20260705T1531Z.log` passed with worst grad rel
  <= 0.000001 and measured S3072 -6.48us (66/96 wins), S4096 -10.38us (74/80 wins),
  and S8192 -16.22us (33/40 wins). Keep the gate tied to `sk_head==1`; force the old
  split-K/atomic route with `MK_HEAD_DX_NO_ATOMIC_SK1=0`.

- Post-head-dX-sk1 long-shape profile/score refresh at `e1606c5`: profile
  `mkv3-p4b-profile-long-post-headdx-e1606c5-20260705T1533Z.log` confirms both
  H256/S3072 and H256/S4096 now emit the head-dX row as `GEMMNN ...x256x8192.wg`
  without `.splitK`, reducing `n_instr` to 152. S3072 measured 2642.2us total
  (76 hops, 226.9us wait, 2415.3us span); head-dX fell to 94.9us on path from the
  pre-promotion 105.5us row, but attention-dQ still dominates at 568.2us. S4096
  measured 3332.4us total (76 hops, 286.9us wait, 3045.5us span), led by attention-dQ
  906.9us, attention fwd 432.4us, RMS dx 248.9us, lm-head forward 218.4us, and
  head-dX 93.6us. Hardened score
  `mkv3-p4b-score-long-post-headdx-e1606c5-20260705T1533Z.log` measured S3072
  megakernel 2627.4us vs compile+CUDAGraph+ 1329.4us (1.98x gap) and S4096
  megakernel 3341.7us vs compile+CUDAGraph+ 1569.5us (2.13x gap). The next long-shape
  bottleneck remains attention-dQ, not head-dX.

- H256/S3072 and H256/S4096 scheduler idle-poll cadence promotion: the existing
  `MK_IDLE_NS` constant is now exposed as an extension build knob, with the model
  default set to 32ns only for H256/S3072 and H256/S4096; `MK_IDLE_NS=256` forces the
  old cadence. Paired env sweep
  `mkv3-p4b-idle-ns-long-probe-20260705T1537Z.log` tested 128/64/32ns against the old
  256ns. S3072 favored faster polling, with idle64 -12.32us (46/48 wins) and idle32
  -11.06us (45/48 wins). S4096 favored idle32, -11.20us with 40/40 wins. The S8192
  boundary is explicitly not promoted: `mkv3-p4b-idle-ns-s8192-boundary-20260705T1543Z.log`
  was only weakly positive for env idle32 (-10.14us, 20/24 wins), and clean
  promoted-default validation `mkv3-p4b-idle32-long-promote-20260705T1545Z.log`
  rejected it (+13.22us default-old median, 1/16 wins). That same clean validation
  kept the exact promoted shapes positive: S3072 -6.93us (33/40 wins) and S4096
  -5.89us (28/36 wins). Route checks confirmed S2048/S8192 stay on the old build and
  S3072/S4096 pick `_idle32`; `mkv3-p4b-idle32-testmodel-20260705T1546Z.log` passed
  full `test_model.py`.

- Post-idle32 long-shape profile/score refresh at `ed57b25`: profile
  `mkv3-p4b-profile-long-post-idle32-ed57b25-20260705T1552Z.log` confirms the promoted
  shapes use `_idle32`. S3072 measured 2614.2us total (76 hops, 210.1us wait,
  2404.2us span), led by attention-dQ 563.9us, attention fwd 335.0us, lm-head forward
  221.9us, RMS dx 144.6us, QKNORM/ROPE bwd 139.8us, cached `SWIGLU_BWD_2W` 139.3us,
  and head-dX 95.3us. S4096 measured 3343.3us total (76 hops, 284.9us wait,
  3058.4us span), led by attention-dQ 914.5us, attention fwd 435.5us, RMS dx 256.0us,
  lm-head forward 220.6us, QKNORM/ROPE bwd 179.0us, cached `SWIGLU_BWD_2W` 174.0us,
  and head-dX 93.9us. Hardened score
  `mkv3-p4b-score-long-post-idle32-ed57b25-20260705T1553Z.log` measured S3072
  megakernel 2647.5us vs compile+CUDAGraph+ 1341.6us (1.97x gap) and S4096
  megakernel 3331.1us vs compile+CUDAGraph+ 1581.2us (2.11x gap). The next useful
  long-shape work remains attention kernel quality or a new attention-DQ mechanism;
  scheduler poll cadence is now exhausted for the tested exact shapes.

- H256/S3072 and H256/S4096 attention-dQ C=1 `float2` direct-store promotion: the
  `OP_ATTN_DQ_WG` C=1 epilogue now vector-stores the adjacent accumulator pair via
  `float2` for the exact same shapes that use idle32; `MK_ATTN_DQ_FLOAT2_STORE=0`
  forces the old scalar-store epilogue. Route checks confirmed S2048/S8192 stay on
  the old build while S3072/S4096 pick `_adqf2`. Initial default-off probe
  `mkv3-p4b-attndq-float2-store-probe-20260705T1554Z.log` measured S3072 -9.68us
  (45/48 wins) and S4096 -15.95us (39/40 wins). Clean promoted-default vs forced-old
  timing `mkv3-p4b-attndq-float2-store-promote-20260705T1558Z.log` remained positive:
  S3072 -11.46us (36/40 wins) and S4096 -7.17us (30/36 wins). Long-shape gradient
comparison `mkv3-p4b-attndq-float2-store-gradcheck-20260705T1559Z.log` matched the
old path to fp32-noise: worst grad rel was 7.61e-7 for S3072 and 1.20e-6 for S4096.
After S2048 later joined idle32 and retuned bwd banding to T12, rechecking forced
`MK_ATTN_DQ_FLOAT2_STORE=1`
(`mkv3-p4b-s2048-attndq-float2-post-t12-20260705T1953Z.log`) was still only
noise/order-mixed (+1.10us then -1.86us, weak win counts), so S2048 remains
outside the `_adqf2` build bucket.

- H256/S8192 attention-dQ `float2` direct-store promotion, but no broad Cq=1 widening:
  after DKV-float2 made S8192 even more dQ-led, env-only expansion
  `mkv3-p4b-attndq-float2-cq1-expand-probe-20260705T1717Z.log` tested Cq=1 shapes.
  S8192 was strongly positive (-32.53us, 12/12 wins; worst grad rel 9.2e-7), but S256
  rejected (+3.90us, 28/120 wins) and small was too weak (-5.33us, 29/48). S128 looked
  positive in the env probe (-12.06us, 158/160), but the clean promoted-default repeat
  `mkv3-p4b-attndq-float2-cq1-promote-20260705T1725Z.log` refuted it (+1.18us, 66/160
  wins). The same clean repeat kept S8192 positive (-29.92us, 10/12 wins). Final route
  `mkv3-p4b-attndq-float2-cq1-final-route-20260705T1727Z.log` confirms `_adqf2` only
  for H256/S3072, S4096, and S8192; S128/S256/small stay on the scalar dQ store.

- Post-attention-dQ-float2 profile/score refresh at `665d8cc`: profile
  `mkv3-p4b-profile-long-post-dqf2-665d8cc-20260705T1602Z.log` confirms S3072/S4096
  use both `_idle32` and `_adqf2`. S3072 measured 2608.1us total (76 hops, 203.6us
  wait, 2404.4us span); the realized path now splits attention backward across
  `ATTN_DQ_WG` 276.6us and `ATTN_DKV_WG` 275.0us, with attention fwd 339.8us and
  lm-head forward 222.3us. S4096 measured 3362.9us total (76 hops, 282.1us wait,
  3080.8us span), still led by `ATTN_DQ_WG` 916.5us, attention fwd 442.7us, RMS dx
  251.3us, lm-head forward 223.0us, and QKNORM/ROPE bwd 180.3us. Combined score
  `mkv3-p4b-score-long-post-dqf2-665d8cc-20260705T1603Z.log` measured S4096
  megakernel 3300.2us vs compile+CUDAGraph+ 1586.8us (2.08x gap). Its S3072 row was
  an outlier at 3007.4us despite the profile and paired timings, so use the clean
  rerun `mkv3-p4b-score-s3072-post-dqf2-rerun-665d8cc-20260705T1604Z.log`: S3072
  megakernel 2610.5us vs compile+CUDAGraph+ 1324.5us (1.97x gap). S4096 still needs a
  real attention-DQ kernel-quality improvement; S3072's path has become more balanced
  across attention fwd/dQ/dKV and lm-head forward.

- Attention-dQ split-mask loop no-go: a default-off source probe
  `MK_ATTN_DQ_SPLIT_MASK=1` tested splitting the dQ probability-store loop into a
  no-mask path for fully unmasked K stages and the existing masked path for the
  diagonal stage. Unlike the earlier `MK_ATTN_MASKED_EXP_SKIP` no-go, this kept the
  same `wga_exp` math and only tried to remove per-element mask predicates from the
  common unmasked stages. Timing `mkv3-p4b-attndq-splitmask-probe-20260705T1605Z.log`
  passed loss checks but regressed decisively: S3072 +41.98us with 0/40 wins and
  S4096 +45.98us with 0/36 wins. The source probe was reverted; keep the compact
  single-loop predicate form.

- Attention-dKV direct-atomic `float2` promotion: the `OP_ATTN_DKV_WG` direct-atomic
  epilogue now atomically adds each adjacent accumulator pair as a `float2` for all
  D=64 WGMMA attention shapes (`c.D == 64 and c.S % 128 == 0`). The env override
  `MK_ATTN_DKV_FLOAT2_ATOMIC=0` restores the old scalar direct-atomic epilogue for A/B.
  Initial default-off timing was strongly positive on H256/S3072 (-12.70us, 40/40
  wins) and H256/S4096 (-25.47us, 36/36 wins), then on H256/S2048 (-16.02us, 36/36)
  and H256/S8192 (-55.49us, 16/16) in
  `mkv3-p4b-attndkv-float2-atomic-probe-20260705T1611Z.log` and
  `mkv3-p4b-attndkv-float2-atomic-boundary-20260705T1616Z.log`. The short/small
  extension `mkv3-p4b-attndkv-float2-atomic-short-20260705T1622Z.log` also supported a
  broad gate: nano -10.48us (46/48 wins), H256/S1024 -21.33us (40/40), and H512/S1024
  small -38.18us (32/32).

- Clean promoted-default vs forced-old timing
  `mkv3-p4b-attndkv-float2-atomic-promote-20260705T1635Z.log` confirmed the broad
  route and printed the intended `_adkvf2` suffix only on the default path: nano
  -5.70us (42/48 wins), H256/S1024 -7.78us (34/40), H512/S1024 small -39.86us
  (32/32), H256/S2048 -10.26us (29/32), H256/S3072 -7.20us (26/32), H256/S4096
  -23.44us (24/24), and H256/S8192 -47.39us (11/12). Gradient comparison
  `mkv3-p4b-attndkv-float2-atomic-gradcheck-20260705T1645Z.log` stayed inside the
  existing model tolerance, with worst rel 0.00610 on H256/S1024 and long shapes around
  1e-6. Full `test_model.py` passed in
  `mkv3-p4b-attndkv-float2-atomic-testmodel-20260705T1646Z.log`; nano's PyTorch
  reference worst grad rel was 0.0281, below the existing 0.03 gate, and the D=128
  ragged fallback plus executor-mode checks still passed.

- S128/S256 boundary check for the broad dKV `float2` gate: because `c.D == 64 and
  c.S % 128 == 0` also covers the shortest WGMMA attention routes, a post-commit
  boundary run `mkv3-p4b-attndkv-float2-atomic-s128-s256-boundary-20260705T1656Z.log`
  checked default vs forced-old. S128 was neutral-positive (-0.85us median, 94/160
  wins; worst grad rel 0.00562), while S256 was clearly positive (-15.15us, 119/120
  wins; worst grad rel 0.00651). Keep the broad D=64/S%128 default.

- Post-attention-dKV-float2 profile/score refresh at `1568829`: `profile_df.py both df`
  in `mkv3-p4b-profile-both-post-dkvf2-1568829-20260705T1705Z.log` measured nano
  887.9us total (76 hops, 203.0us wait, 685.0us span) and small 3448.2us total
  (144 hops, 376.7us wait, 3071.5us span). Nano's path is now led by attention-dQ
  85.6us, attention-fwd 74.7us, RMS dx 66.1us, and qkrope GEMM 65.4us; dKV is
  off-path at 70.3us total. Small is led by `ATTN_DKV_WG` 412.2us, MLP dX
  `GEMMNN 1024x512x3072.wg` 386.7us, attention-fwd 254.4us, and cached
  `SWIGLU_BWD_2W` 242.2us.

- Long-shape profile `mkv3-p4b-profile-long-post-dkvf2-1568829-20260705T1706Z.log`
  measured S3072 2578.5us total, S4096 3325.0us, and S8192 8415.5us. On S3072/S4096,
  the realized path no longer includes dKV; attention-dQ dominates (S3072 546.0us,
  S4096 902.4us), followed by attention-fwd and lm-head/RMS rowops. S8192 is still an
  attention-quality problem: attention-dQ is 3345.9us and attention-fwd 1520.4us on
  path, while dKV is off-path at 2627.3us span.

- Fresh-process score refresh:
  `mkv3-p4b-score-final-fresh-post-dkvf2-1568829-20260705T1709Z.log` measured
  nano 912.8us vs compile+CUDAGraph+ 631.4us (1.45x gap), small 3457.7us vs 1904.0us
  (1.82x), deep 2382.7us vs 1785.8us (1.33x), S128 736.0us vs 428.1us (1.72x),
  S256 812.8us vs 547.4us (1.49x), and S1024 1217.7us vs 780.4us (1.56x). The
  same-process `final_bench.py all` log hit TorchDynamo's recompile limit after S128,
  so use the fresh-process log for compile/CUDAGraph baselines. Long fresh-process score
  `mkv3-p4b-score-long-fresh-post-dkvf2-1568829-20260705T1711Z.log` measured S3072
  2593.8us vs graph+ 1340.5us (1.93x), S4096 3323.2us vs 1569.4us (2.12x), and S8192
  8221.0us vs 3123.3us (2.63x).

- Small DKV chunk retune after DKV/DQ `float2` no-change: even with the cheaper dKV
  `float2` atomics, H512/S1024 small still rejects over-splitting the DKV backward.
  `mkv3-p4b-small-dkv-c-retune-post-float2-20260705T1732Z.log` compared current
  `DKV_C=1/DQ_C=1` against `DKV_C=2/DQ_C=1` and `DKV_C=3/DQ_C=1`; both passed
  gradient checks but lost decisively. `DKV_C=2` was +59.58us median with 0/80 wins,
  and `DKV_C=3` was +55.68us with 0/80 wins. Keep small at `DKV_C=1/DQ_C=1`.

## v3 P4b long-S scaling round (session 2853e0de): the C=1 straggler diagnosis

Standalone per-op measurement of WHY WG attention scales worse than FA3 with S
(single-instruction df programs on a clean GPU 3, median-of-50, fresh process per S,
H256 long config nq=4/nkv=2/D=64; flash side = CUDA-graph-captured SDPA 4-D+gqa,
double-measured to within 3%). Logs: `mkv3-p4b-attn-scaling-probe-20260705T164743Z.log`,
`mkv3-p4b-flash-graph-probe-*.log`, `mkv3-p4b-flash-graph-rerun-*.log`, plus the v1
eager-autograd flash numbers retracted inside the first log (CPU autograd overhead
polluted them; only the graph-captured flash numbers are valid).

| S | mk fwd | fl fwd | mk dkv | mk dq | mk bwd sum | fl bwd | bwd ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 (C=2/2) | 42.7 | 16.3 | 40.0 | 38.5 | 78.5 | 69.1 | 1.14x |
| 2048 (C=2/2) | 64.5 | 28.4 | 62.7 | 54.6 | 117.3 | 107.8 | 1.09x |
| 3072 (C=1/1) | 88.1 | 41.2 | 142.7 | 104.8 | 247.5 | 152.1 | 1.63x |
| 4096 (C=1/1) | 111.8 | 75.2 | 186.4 | 134.5 | 320.9 | 199.1 | 1.61x |
| 8192 (C=1/1) | 390.9 | 233.2 | 669.2 | 481.2 | 1150.4 | 520.8 | 2.21x |

Findings (all per layer, standalone):
- The bwd ops are COMPETITIVE with graph-captured flash at the chunked shapes
  (S<=2048: 1.09-1.14x). The long-S collapse is specific to the C=1 regime.
- At C=1 the ops are STRAGGLER-BOUND: makespan == the longest causal tile's serial
  stage chain. Marginal cost is 2.7us per 64-row stage for dkv and 1.9us for dq —
  S3072->S4096 dkv grew +43.7us for exactly +16 stages while the tile count grew
  96->128 and absorbed nothing (SMs idle around the straggler). fwd matches the same
  model (111.8us ~= 64 stages x 1.5us + fill at S4096).
- The S2048->S3072 dkv discontinuity (+127% time for +50% S) is the Ckv=2->1 default
  flip tripling the longest chain, not a bandwidth or wave effect.
- Uniform C=2 at S4096 (the earlier measured no-go, +77us) fails because it doubles
  fill/atomic cost on ALL tiles, including the short tail tiles where the fill
  dominates. The measurement-supported fix is BANDED chunking — chunk count
  proportional to per-tile stage count (split only the long tiles) — in progress in
  `/home/apanda/xorl-oss-attn-band` (kernel decode packs C | kv/q-tile-off<<8 |
  band-width<<16 into the existing C arg; per-band ws slots keep the disjoint row
  ranges parallel in the dep analysis; bands emitted longest-first).
- Re-run knobs: `results/attn_scaling_probe_2853e0de.py <S>` (megakernel ops +
  per-stage fits) and `results/flash_graph_probe_2853e0de.py <S>` (flash side),
  driver `results/run_attn_scaling_probe_2853e0de.sh <gpu>`.

## v3 P4b banded attention-bwd chunking (session 2853e0de): the straggler fix lands

**PROMOTED (commits `06ab5b6` + `8508a7d`), the largest single long-S win of the
P4b program**: `OP_ATTN_DKV_WG` / `OP_ATTN_DQ_WG` now support banded chunking —
the host splits each op into contiguous kv/q-tile bands whose chunk count is
proportional to the band's causal stage count (`C = ceil(stages / T)`), so only
the long triangle tiles get split and the short tail tiles keep their cheap
single fill. This is the fix the C=1 straggler diagnosis (section above)
predicted: at S>=3072 the bwd ops were makespan-bound on the longest tile's
serial chain while most SMs idled, and uniform C>1 (the old no-go) paid
fill/atomic overhead on every tile to fix only the one straggler.

Mechanics: the band spec packs into the existing C arg (`C | kv_or_q_tile_off<<8
| band_width<<16`), so stale callers passing a bare C decode unchanged (off=0,
width=full). DQ bands emitted with C==1 keep the direct-store epilogue (bands are
q-disjoint, still one writer per slice); C>1 bands use the existing atomic drain
into the pre-zeroed workspace. Per-band `dQKV_f32` slots (`kv0/kv1/...`,
`q0/q1/...`) declare the bands' truly disjoint row ranges to the dependency
analysis so they schedule in parallel. Bands are emitted longest-chunk-first.
`MK_ATTN_BAND=<T>` overrides; `MK_ATTN_BAND=0` restores the uniform Ckv/Cq path
(where `MK_ATTN_DKV_C`/`MK_ATTN_DQ_C` still apply).

Default gate (H256/D64): `{2048: 16, 3072: 16, 4096: 32, 8192: 32}`; all other
shapes keep the uniform path. Env-on-vs-off in-model paired A/B, both
construction orders, every config parity-clean (losses equal, worst grad rel
<= 0.0063):

| shape | T | delta (order1 / order2) | wins |
|---|---|---|---|
| H256/S2048 | 16 | -20.6 / -16.0us | 40/40, 40/40 |
| H256/S3072 | 16 | -37.0 / -42.4us | 40/40, 40/40 |
| H256/S4096 | 32 | -80.7 / -78.7us | 40/40, 40/40 |
| H256/S8192 | 32 | -531.7 / -524.6us | 16/16, 16/16 |

Promoted-default vs forced-old (`MK_ATTN_BAND=0`) on the merged tree (which
includes the peer's dq-float2-s8192) confirmed: S2048 +12.5/+14.4us, S3072
+35.1/+36.4us, S4096 +70.4/+76.1us, S8192 +505.0us (0/16, clean guard) and
+515.0us (1/16; an sglang server landed memory-parked on the GPU during this
last reverse-order rep — treat it as corroborating-only; the three clean
measurements agree). Route check: S512/S1024 instruction streams unchanged;
S2048/S3072/S4096/S8192 gain exactly the predicted +2/+4/+2/+6 instrs per layer.
`test_ops.py`, `test_model.py` (default), and `test_model.py` with
`MK_ATTN_BAND=16` forced all passed.

Rejected variants: S2048 T=32 degenerates to uniform C=1 and loses (+41/+48us,
0/40 both orders); S8192 T=16 over-splits (-321/-332us, worse than T=32).
Standalone op-level A/B agrees (S8192 dkv 675->420us, dq 485->327us at T=32;
S4096 banded dq ALONE is flat from wave quantization — 192+ tiles on a 132-SM
machine — but the in-model window absorbs it via cross-op work stealing, so the
shape still wins -80us). Logs: `mkv3-p4b-attn-band-probe-20260705T170354Z.log`,
`mkv3-p4b-attn-band-model-ab-20260705T170925Z.log`,
`mkv3-p4b-attn-band-validate-20260705T171650Z.log` in the attn-band worktree
results/ (repro: `results/attn_band_probe.py <S> <T>`,
`results/attn_band_model_ab.py <S> <T|0> <order>`).

Post-merge long-S scoreboard at `1b4e194` (hardened bench_cfg, fresh process per
shape, GPU 6, guards clean; `mkv3-p4b-score-long-post-band-20260705T173446Z.log`):

| shape | megakernel | compile+CUDAGraph+ | gap | pre-band gap |
|---|---:|---:|---:|---:|
| S2048 | 1843.9 | 1043.3 | **1.77x** | 1.83x |
| S3072 | 2526.9 | 1336.1 | **1.89x** | 1.97x |
| S4096 | 3206.0 | 1589.1 | **2.02x** | 2.08-2.13x |
| S8192 | 7782.2 | 3044.8 | **2.56x** | 2.63-2.69x |

Short-S banding is a measured NO-GO (env-only sweep on the merged tree, both
construction orders, all parity-clean;
`mkv3-p4b-attn-band-short-sweep-20260705T174217Z.log` and
`mkv3-p4b-attn-band-s512-t4-20260705T174635Z.log`): S1024 T=8 is neutral
(-0.2us 21/40, then +4.5us 12/40), S512 T=4 is negative (+15.2us 2/40, +4.1us
9/40), and the degenerate T>=8 points at S512 / T=16 at S1024 collapse to
uniform C=1 and lose outright (+22 to +70us, ~0/40). Banding pays only where
the straggler chain is long (>= 32 stages, i.e. S >= 2048); below that the
extra fills/atomics have nothing to amortize against. Keep the short-S uniform
C defaults. Re-run knobs: `results/attn_band_model_ab_main.py <S> <T> <order>`
with `MK_ATTN_BAND` env override.

Band emission order follow-on: the post-band S8192 profile showed the critical DQ
bands were wait-dominated behind same-wave DKV work. `MK_ATTN_BAND_ORDER` now accepts
`lpt` (the original longest-stage-first order) and `dq_first` (DQ bands first, largest
chunk count first). The default is `dq_first` only for H256/D64/S8192; all shorter
gated shapes keep `lpt`, and `MK_ATTN_BAND_ORDER=lpt` restores the old S8192 route.
Validation in `/home/apanda/xorl-oss-attn-band-order-probe`: default route hashes for
S4096 still match the shared tree exactly, while S8192 default changes to
`DQ C4/C3/C2/C1, DKV C4/C3/C2/C1`. S8192 env A/B on the old default won
-88.4us/-105.0us (16/16, 16/16); promoted-default vs forced `lpt` confirmed
-103.2us (16/16) and -74.2us (15/16). All parity checks passed
(`worst_grad_rel` <= 0.006477). Do NOT broaden the default: S2048 regressed
+8.9/+22.1us, S3072 +52.7/+62.9us, and S4096 +47.3/+51.8us. Logs:
`mkv3-p4b-attn-band-order-s8192-20260705T1801Z.log`,
`mkv3-p4b-attn-band-order-long-sweep-20260705T1804Z.log`, and
`mkv3-p4b-attn-band-order-s8192-promoted-default-20260705T1808Z.log`.

ATTN_FWD_WG banding follow-on (session 2853e0de implementation, current-head
validation by the band-order session) is promoted for H256/D64 long shapes. The WG
fwd path now supports `MK_ATTN_FWD_BAND=<T>`: split q bands run as flash-decoding
kv chunks that write locally-normalized `fopart/fmpart/flpart`, then a range-limited
`OP_ATTN_COMBINE` merges only the split rows; C=1 bands keep the direct O/LSE
epilogue. Default gate: `{2048:16, 3072:32, 4096:32, 8192:64}` for H256/D64, and
`MK_ATTN_FWD_BAND=0` restores the old direct fwd route. The S8192 validation is
against current `94c1a2a` after the DQ-first bwd band-order merge, not the older
`cfeaaf4` baseline. Current-head paired A/B log:
`/home/apanda/xorl-oss-attn-fwdband-current-eval/results/mkv3-p4b-attn-fwdband-current-eval-20260705T1822Z.log`.
Results: S2048/T16 -19.7/-15.7us (39/40, 39/40), S3072/T32 -45.7/-43.9us
(40/40, 40/40), S4096/T32 -4.4/-11.1us (30/40, 40/40), S8192/T64
-452.2/-481.5us (16/16, 16/16). All parity checks passed (`worst_grad_rel`
<= 0.013994). Keep rejected fwd-band variants out of the default: S4096 T16 and
T48 lose, T64 is neutral; S3072 T16 is neutral/negative. Promoted-default vs forced
old confirmed the env gate itself: S2048 -23.9us, S3072 -39.8us, S4096 -6.6us,
S8192 -481.9us (`mkv3-p4b-attn-fwdband-promoted-default-20260705T1828Z.log`).
Post-promotion S8192 profile (`mkv3-p4b-profile-s8192-post-fwdband-d86b5ce-20260705T1830Z.log`)
is `7345.8us` total. Fwd is no longer the lead span (`ATTN_FWD_WG` fell to
`849.1us`, plus `ATTN_COMBINE` `306.7us`); the path is back to bwd DKV wait:
`ATTN_DKV_WG` `2662.3us` total with `2304.7us` wait. Retesting
`MK_ATTN_BAND_ORDER=lpt` after fwd-band was still a no-go (+114/+105us), so keep
S8192 DQ-first. Long scoreboard at `d86b5ce`
(`mkv3-p4b-score-long-post-band-20260705T183210Z.log`) gives S8192 `7327.2us`
vs compile+CUDAGraph+ `3137.1us` (2.34x); shorter-shape scoreboard medians were
noisy and should be interpreted through the paired A/B above.

S4096 fwd-band retune after the S4096 bwd-band T29 default and combine row
batching: isolated worktree `/home/apanda/xorl-oss-attn-fwdband-retune` retested
nearby `MK_ATTN_FWD_BAND` budgets against committed current head `98591ae`.
Broad log `mkv3-p4b-fwdband-retune-broad-20260705T1905Z.log` showed T24 as the
only positive broad candidate (-17.0/-11.6us, 23/24 and 24/24 wins); T28/T29/T30/
T36/T40 lost, and S8192 stayed on T64 because T48/T56/T72/T80 all lost. Narrow
S4096 log `mkv3-p4b-fwdband-retune-s4096-narrow-20260705T1912Z.log` picked T22:
current T32 vs forced T22 was -23.26us/-22.64us with 40/40 wins in both
construction orders, parity clean (`worst_grad_rel` <= 0.010605). T24 also won
but less (-16.56/-15.87us), and T26 was neutral/bad. Promote H256/D64/S4096 fwd
banding from T32 to T22; promoted-default confirmation
`mkv3-p4b-fwdband-retune-s4096-promoted-default-20260705T1917Z.log` beat forced
old T32 by +20.77us/+21.71us old-minus-new (old wins only 2/40 and 1/40), with
parity clean (`worst_grad_rel` <= 0.010257). Keep S2048 T16, S3072 T32, and
S8192 T64 unchanged.

## v3 P4b fused one-pass attention bwd (session 2853e0de): NO-GO, and why

FA2-style fusion of dQ into `OP_ATTN_DKV_WG` (per stage: x3 wgmma commit batch
adding dQp = dS @ K_wg via dS's K-view against owned K's MN-view, dqp reusing
the dead s-bank lifetime, then 16 float2 atomicAdds into the workspace q slot;
host drops `OP_ATTN_DQ_WG` entirely) is **correctness-clean but decisively
slower**: S4096 in-model +118.98us (0/40 wins), loss identical, worst grad rel
0.005769 (`mkv3-p4b-attn-fusedbwd-smoke-20260705T175625Z.log` in the
attn-fusedbwd worktree). Mechanism: fusion multiplies dQ WRITE traffic by
n_kvt x 2WG — every q row receives (S/128)x2 fp32 atomic contributions
(~135MB/layer of atomics at S4096) instead of one direct store, and atomic
writes serialize at L2/DRAM, while the two-pass structure's extra K/V re-reads
(the cost fusion tries to save) are L2-RESIDENT at these shapes (whole qkv
buffer ~4MB << 50MB L2). The pass split converts store-amplification into
cache-absorbed load-amplification; keep the two-pass bwd. The amplification
scales with S, so longer shapes lose harder — do not revisit without a
mechanism that accumulates dQ before the atomic (none exists: stage q-rows are
disjoint by construction). Code stays default-off on branch
`megakernel-attn-fusedbwd` (`/home/apanda/xorl-oss-attn-fusedbwd`,
`MK_ATTN_FUSED_BWD=1` + `_afb` build); repro
`results/attn_fused_model_ab.py <S> 1 <order>` in that worktree.

## v3 P4b long-S day close: composed scoreboard (2026-07-05 ~18:35Z)

Certified composed state at `d86b5ce` (bwd banding + dq_first order + fwd
banding + the day's dkv/dq float2 promotions), hardened bench_cfg, fresh
process per shape, GPU 6, guards clean
(`mkv3-p4b-score-long-post-band-20260705T183210Z.log`):

| shape | megakernel | compile+CUDAGraph+ | gap | day start |
|---|---:|---:|---:|---:|
| S2048 | 1865.3 | 1050.2 | 1.78x | 1.83x |
| S3072 | 2532.2 | 1341.6 | 1.89x | 1.97x |
| S4096 | 3245.2 | 1575.6 | 2.06x | 2.08-2.13x |
| S8192 | **7327.2** | 3137.1 | **2.34x** | 2.63-2.69x |

Instrument note: the S2048/S3072/S4096 megakernel columns read ~20-40us above
the paired-A/B instrument on the same head (paired absolutes: ~1848 / ~2504 /
~3221 — single bench_cfg runs bounce at that scale; see the S3072 3007-outlier
precedent). The S8192 row agrees with the paired instrument to 0.2us. Total
S8192 movement today: 8395 -> 7327 megakernel-side (-12.7%), gap 2.69x -> 2.34x.

Long-S lane state after this round: the causal-straggler family of fixes is
mined (bwd bands, fwd bands, band order); the fused one-pass bwd is refuted
with mechanism. At S4096 the residual attention deficit is mostly the band
quantization awkwardness (any split exceeds 132 tiles) plus the DQ
SM-contention wait; the larger residual at ALL long shapes is now the flat
non-attention deficit (RMS dx 253.9us, lm-head 224.4us, QKNORM 181.3us,
SWIGLU_BWD 176.0us on-path at S4096 per
`mkv3-p4b-profile-long-post-band-20260705T175128Z.log`). Next lanes in
measured-value order after the S4096 retune below: (1) a post-composition
profile refresh to re-rank; (2) the non-attention long-S scaling items (RMS dx
grows superlinearly 144.6 -> 251.3us from S3072 -> S4096 - unexplained and
worth a probe); (3) knob consolidation (the gate maps are now ~11 deep).

S4096 attention-bwd band-budget retune: after fwd-band composition, an env-only
GPU 5 sweep around the current `MK_ATTN_BAND=32` default found that the
three-band split near the boundary is better than the two-band default. Broad
sweep log `mkv3-p4b-s4096-band-budget-sweep-20260705T1841Z.log`: T24 was
noise/negative (+5.6us then -5.6us, low wins), T28 won -53.9/-59.4us (40/40
both), T32 same-route control was -3.3/-0.5us noise, T36 was neutral/negative,
T40 lost +29.5/+20.1us, and T48 was a smaller win (-8.2/-15.4us). Narrow log
`mkv3-p4b-s4096-band-budget-narrow-20260705T1845Z.log`: T26 lost +7.7/+7.3us,
T27 won -31.6/-35.2us, T28 repeated -52.1/-53.4us, T29 won -60.3/-60.4us, and
T30 won -57.4us in the clean order (the other order had contaminated absolute
medians but still paired -66.8us). Confirmation log
`mkv3-p4b-s4096-band-budget-confirm-20260705T1847Z.log`: T29 repeated
-62.9/-60.4us with 40/40 wins in both construction orders, while T31 shrank to
-16.9/-15.7us. Current H256/D64 band defaults after the S2048/S8192 retunes below
are `{2048:12, 3072:16, 4096:29, 8192:40}`; `MK_ATTN_BAND=32` restores the old
S4096 route for A/B and `MK_ATTN_BAND=0` still restores the uniform C route.

S2048 attention-bwd band-budget retune after the idle32/row-batching composition:
GPU5 env-only log `mkv3-p4b-attnband-retune-s2048-post-resweep-20260705T1929Z.log`
found that the old T16 default is no longer best. T12 beat T16 by -29.94us and
-36.69us with 40/40 wins in both construction orders, parity clean
(`worst_grad_rel` <= 0.007364). T10/T14/T24 also won but less; T18/T20 were
neutral/mixed. Promoted-default validation
`mkv3-p4b-attnband-s2048-promoted-default-20260705T1936Z.log` beat forced old
T16 by +34.85us/+32.27us old-minus-new, with old wins 0/40 in both orders.
Promote H256/D64/S2048 `MK_ATTN_BAND` from T16 to T12; keep S3072 T16 and S4096
T29 unchanged, with the S8192 retune below applied later. Rechecking
`MK_ATTN_BAND_ORDER=dq_first` under the
new S2048 T12 geometry (`mkv3-p4b-s2048-t12-band-order-recheck-20260705T1939Z.log`)
still lost +18.50us/+21.31us with only 1/40 wins in both construction orders, so
keep S2048 on default `lpt` order.

S8192 attention-bwd band-budget retune after the idle32 and cached-SwiGLU
composition: GPU5 broad log `mkv3-p4b-attnband-retune-s8192-current-20260705T1956Z.log`
found T24/T28/T36 negative, T34 positive (-35.8/-28.5us), and T40 best in the
broad pass (-85.8/-77.3us, 8/8 wins both orders). Focused confirmation
`mkv3-p4b-attnband-retune-s8192-focused-20260705T2002Z.log` kept T40 best:
-88.48us/-109.62us with 12/12 wins in both construction orders, parity clean
(`worst_grad_rel` <= 0.004956). Promoted-default validation
`mkv3-p4b-attnband-s8192-promoted-default-20260705T2006Z.log` beat forced old
T32 by +75.81us/+80.37us old-minus-new, with old wins 0/16 in both orders and
parity clean (`worst_grad_rel` <= 0.005753). Promote H256/D64/S8192
`MK_ATTN_BAND` from T32 to T40; current H256/D64 band defaults are now
`{2048:12, 3072:16, 4096:29, 8192:40}`.

H256/D64/S8192 QKNORM/ROPE-bwd split-V follow-up after the T40 retune: the
old `MK_QKBWD_SPLIT_V=1` probe was a nano/small no-go because its extra V
pass-through row op added dependency wait, but the post-T40 S8192 profile
(`mkv3-p4b-profile-score-s8192-post-t40-a8f137c-20260705T2012Z.log`) puts
`QKNORM_ROPE_BWD` at `349.2us` on path. Current-base port
`/home/apanda/xorl-oss-qkbwd-splitv-s8192` first exposed a correctness hazard:
the V pass-through reader must use the unslotted `dQKV_f32` root so it depends on
all current banded-attention `kv0/kv1/...` writers; reading a stale `kv` slot
matched loss but broke downstream grads (`w1.3` rel err ~1.03). With that fixed,
env-only S8192 timing `mkv3-p4b-qkbwd-splitv-s8192-current-20260705T2018Z.log`
was parity-clean and positive (-34.59us/-63.47us, 15/16 and 16/16 wins).
Promoted-default validation
`mkv3-p4b-qkbwd-splitv-s8192-promoted-default-20260705T2020Z.log` beat forced
old `MK_QKBWD_SPLIT_V=0` by +42.88us/+31.78us old-minus-new, with old wins
2/16 and 1/16 and parity clean (`worst_grad_rel` <= 0.004937). Default only the
exact H256/D64/S8192 shape first; earlier nano/small no-go evidence still stands.
Boundary check at H256/D64/S4096 after that promotion found the same split-V route
positive: `mkv3-p4b-qkbwd-splitv-s4096-current-20260705T2031Z.log` measured
-16.59us and -3.52us (40/40 and 36/40 wins), then fresh-process confirmation
`mkv3-p4b-qkbwd-splitv-s4096-confirm-20260705T2033Z.log` measured -9.60us and
-12.38us (36/40 and 40/40 wins), all parity-clean (`worst_grad_rel` <= 0.005769).
Promoted-default validation
`mkv3-p4b-qkbwd-splitv-s4096-promoted-default-20260705T2035Z.log` beat forced
old `MK_QKBWD_SPLIT_V=0` by +28.61us/+17.62us old-minus-new, with old wins 8/40
and 3/40 and parity clean (`worst_grad_rel` <= 0.005769). Default
`MK_QKBWD_SPLIT_V` now covers exact H256/D64/S4096 and S8192 only.
Post-promotion S4096 profile/score
`mkv3-p4b-profile-score-s4096-post-qkbv-63f4f11-20260705T2038Z.log` measured
megakernel 3126.1us vs compile+CUDAGraph+ 1585.0us (1.97x gap). Profile total
was 3133.1us (`n_instr=188`, critical path 80, gated 63); on-path leaders are
attention-dQ 762.6us, attention-fwd 303.9us, RMS dx 252.4us, lm-head NT 225.2us,
SwiGLU-BWD 2W 172.6us, and `QKNORM_ROPE_BWD` 162.4us. The split-V target is now
below the attention and RMS/lm-head bottlenecks at S4096.
Boundary check at H256/D64/S3072 also held:
`mkv3-p4b-qkbwd-splitv-s3072-current-20260705T2041Z.log` measured -14.67us and
-8.99us (40/40 and 37/40 wins), and cached fresh-process confirmation
`mkv3-p4b-qkbwd-splitv-s3072-confirm-20260705T2043Z.log` measured -11.28us and
-12.85us (38/40 and 40/40 wins), all parity-clean (`worst_grad_rel` <= 0.005486).
Promoted-default validation
`mkv3-p4b-qkbwd-splitv-s3072-promoted-default-20260705T2044Z.log` beat forced
old `MK_QKBWD_SPLIT_V=0` by +9.17us/+15.81us old-minus-new, with old wins 5/40
and 2/40 and parity clean (`worst_grad_rel` <= 0.003898). Default
`MK_QKBWD_SPLIT_V` now covers exact H256/D64/S3072/S4096/S8192.
H256/D64/S2048 remains a no-change boundary:
`mkv3-p4b-qkbwd-splitv-s2048-current-20260705T2046Z.log` was parity-clean but
construction-order mixed, with default-first neutral/slower (+0.22us, 20/40
wins) and split-first positive (-12.85us, 38/40 wins). Cached confirmation
`mkv3-p4b-qkbwd-splitv-s2048-confirm-20260705T2048Z.log` repeated the split-first
win (-11.98us, 40/40) but default-first stayed too weak/noisy (-1.26us, 17/40).
Keep the default gate at S3072+.
Post-promotion S3072 profile/score
`mkv3-p4b-profile-score-s3072-post-qkbv-ed18a16-20260705T2050Z.log` measured
megakernel 2485.3us vs compile+CUDAGraph+ 1273.2us (1.95x gap). Profile total
was 2485.9us (`n_instr=180`, critical path 80, gated 63); on-path leaders are
attention-dQ 516.8us, lm-head NT 223.9us, attention-fwd 183.4us, SwiGLU-BWD 2W
143.6us, RMS dx 139.9us, and `QKNORM_ROPE_BWD` 131.4us.
Long-shape SSQ-fusion recheck after the split-V promotions:
`mkv3-p4b-ssq-long-current-20260705T2054Z.log` rejected S4096 SSQ-off as
construction-order mixed (-7.60us then +4.18us), but S3072 was weakly positive
in both orders (-5.39us and -2.06us). Cached S3072 confirmation
`mkv3-p4b-ssq-s3072-confirm-20260705T2056Z.log` held (-4.85us and -3.38us).
Promoted-default validation
`mkv3-p4b-ssq-s3072-promoted-default-20260705T2058Z.log` beat forced old
`MK_SSQ_FUSE=1` by +3.38us/+7.73us old-minus-new, with old wins 11/40 and 9/40
and parity clean (`worst_grad_rel` <= 0.008040). Default `MK_SSQ_FUSE` is now
off only for exact H256/D64/S3072; force `MK_SSQ_FUSE=1` for the old fused-SSQ
route.
Post-SSQ S3072 profile/score
`mkv3-p4b-profile-score-s3072-post-ssq-637102c-20260705T2100Z.log` measured
megakernel 2468.5us; compile+CUDAGraph+ was 1339.9us in that fresh process
(higher than the previous 1273.2us graph+ row, so keep paired A/B as the SSQ
delta authority). Profile total was 2488.5us (`n_instr=180`, critical path 80,
gated 71), led by attention-dQ 516.5us, lm-head NT 223.6us, attention-fwd
185.1us, SwiGLU-BWD 2W 139.9us, RMS dx 139.6us, and `QKNORM_ROPE_BWD` 131.4us.
S8192 endpoint check `mkv3-p4b-ssq-s8192-current-20260705T2103Z.log` was a
parity-clean order-mixed wash (-2.27us then +2.34us), so keep SSQ fused at
S8192. S2048 boundary check `mkv3-p4b-ssq-s2048-current-20260705T2106Z.log`
was positive (-3.02us and -7.63us), and cached confirmation
`mkv3-p4b-ssq-s2048-confirm-20260705T2108Z.log` held (-10.66us and -3.66us),
all parity-clean (`worst_grad_rel` <= 0.005908). Promoted-default validation
`mkv3-p4b-ssq-s2048-promoted-default-20260705T2110Z.log` beat forced old
`MK_SSQ_FUSE=1` by +8.37us/+3.02us old-minus-new, with old wins 8/40 and 11/40
and parity clean (`worst_grad_rel` <= 0.007364). Together with the S4096 mixed
result, the SSQ-off default now covers exact H256/D64/S2048 and S3072 only.
Post-SSQ S2048 profile/score
`mkv3-p4b-profile-score-s2048-post-ssq-def4cbb-20260705T2112Z.log` measured
megakernel 1780.9us vs compile+CUDAGraph+ 1044.8us (1.70x gap). Profile total
was 1763.3us (`n_instr=176`, critical path 80, gated 71), led by attention-dQ
274.9us, attention-fwd 131.0us, lm-head NT 114.1us, SwiGLU-BWD 2W 103.5us,
head-dX/lm-head-bwd 101.3us, and `QKNORM_ROPE_BWD` 99.1us.
Post-SSQ S2048 split-V recheck
`mkv3-p4b-qkbwd-splitv-s2048-post-ssq-20260705T2115Z.log` stayed no-change:
default-first was a wash (-0.70us, 19/40 wins) even though split-first was
positive (-12.75us, 40/40). Keep default `MK_QKBWD_SPLIT_V` at S3072+.
Current-head S2048 Drow direct-store recheck
`mkv3-p4b-drowstore-s2048-current-20260705T2120Z.log` kept the old decision:
forced `MK_DROW_DIRECT_STORE=1` was parity-clean but slower in both construction
orders (+13.58us and +3.22us). Keep S2048 on the atomic Drow epilogue.

Post-T29 follow-up no-gos: current default `lpt` order remains best. A
pre-combine env-only retest at `77346e2`
(`mkv3-p4b-s4096-t29-band-order-retest-20260705T1852Z.log`) forced
`MK_ATTN_BAND_ORDER=dq_first` under the new T29 geometry; it was parity-clean
but slower by +51.7/+48.1us with 0/40 wins in both construction orders. A
post-combine worktree probe at `fe39656`
(`/home/apanda/xorl-oss-attn-t29-tiebreak-probe/results/mkv3-p4b-s4096-t29-lpt-dq-tie-20260705T1855Z.log`)
tried only the narrower equal-stage DQ-before-DKV tie-break
(`MK_ATTN_BAND_ORDER=lpt_dq_tie`); it was also parity-clean but slower by
+11.9/+10.3us with only 2/40 and 6/40 wins. Keep S4096 T29 on `lpt`; do not
promote either DQ-first ordering. The pre-combine `MK_RMS_DX_R4=1` recheck
(`mkv3-p4b-s4096-rmsdx-r4-post-t29-20260705T1854Z.log`) still lost
+5.4/+5.2us, so S4096 remains on RMS dx R2.

S3072 post-composition retune no-gos after the S2048/S4096/S8192 promotions:
current profile `mkv3-p4b-profile-s3072-current-1b9cb20-20260705T1941Z.log`
shows S3072 at `2493.9us`, led by `ATTN_DQ_WG` `518.4us`, lm-head NT `224.6us`,
and `ATTN_FWD_WG` `183.7us`. Retesting attention-bwd bands around the T16
default (`mkv3-p4b-attnband-retune-s3072-current-20260705T1943Z.log`) found no
promotion: T10/T12/T14/T18/T28 lost both construction orders, T20 was order-mixed
(-6.64us then +0.86us), and T24 was noise-level (-2.72/-0.32us). Retesting fwd
bands around the T32 default (`mkv3-p4b-fwdband-retune-s3072-current-20260705T1948Z.log`)
also found no promotion: T20/T24/T36/T40/T48 lost both construction orders, and
T28 was order-mixed/noise (-1.44us then +9.01us). Keep S3072 on bwd T16 and fwd
T32.

## v3 P4b attention-combine row batching (session 2853e0de): PROMOTED

The post-compose profile showed `OP_ATTN_COMBINE` (the fwd-band merge op) fat on
the realized path — S3072 100.1us, S4096 181.0us, S8192 301.3us — because its
tile was ONE ROW, so a long-S combine instruction was 2-4k tiny claim-ring
transactions. `op_attn_combine` now takes R rows per tile (new trailing arg,
zero-pads to R=1 for stale callers) with work unit = (row, head) so all 8 warps
stay busy at nq < 8; the fwd-band emission defaults `MK_ATTN_COMBINE_R=8`.
Paired default-vs-forced-R1, both construction orders, parity clean (worst grad
rel <= 0.0053): S3072 -33.1/-27.9us (old wins 0/40 both), S4096 -28.5/-23.6us
(1/40, 2/40), S8192 -69.5/-46.2us (0/16 both). test_ops + test_model green.
Merged as `fe39656`. Log
`mkv3-p4b-attn-combine-r-ab-*.log` in the attn-combine-r worktree.

Precision note discovered during validation (pre-existing, NOT from this
change; identical on main to 4 decimals): forcing fwd bands at nano with tiny T
(`MK_ATTN_FWD_BAND=4` at S512) overshoots the test_model PyTorch-reference grad
bar marginally (kn.3 0.0365 vs 0.03) — the bf16 locally-normalized partial
round-trip costs ~2x error on the tiny qk-norm grads. The default gates
(S >= 2048) avoid the config; if fwd bands are ever gated at short S, budget
tolerance or store partials fp32.

## v3 P4b post-band knob rechecks (session 2853e0de): one flip, one hold

Structural changes invalidate old knob verdicts — two env-only rechecks after
the banding round (`mkv3-p4b-postband-knob-recheck-20260705T185629Z.log`):

- **SwiGLU cached 2W at H256/S8192 FLIPPED and is PROMOTED** (commit `da7e525`):
  pre-band it was rejected (+47.5us, 20260705T1608Z); post-band it wins
  -111.9/-128.3us with 16/16 both construction orders (parity worst rel 0.0087).
  Promoted-default vs forced-old validation
  (`mkv3-p4b-sw2w-s8192-promote-20260705T190241Z.log`): +128.8/+116.0us
  for the default, 0/16 for old, test_model green. The banding round rebalanced
  SM occupancy in exactly the window where the S8192 SwiGLU bwd sits (300.9us
  on-path in the post-compose profile). S8192 megakernel is now ~7246us
  (day start 8395: -13.7%). Gates: `swiglu_cache_sig_default` and
  `swiglu_bwd_2w_default` each gained the exact H256/S8192/I768 shape.
- **RMS dx R4 at H256/S4096 HOLDS negative** post-band: +5.6/+8.1us with the R4
  variant winning only 11/40 in each order (parity clean). The 1.94-wave R2
  quantization hypothesis did not pay; keep R2. (S4096 absolutes in this
  recheck: ~3151us after the peer's T=29 band-budget retune.)

Lesson worth keeping: re-run the cheap env knob sweeps after every structural
scheduling change — this flip was worth 11x the average recent promotion and
cost two A/B runs to find.

Resweep batch 1 (S8192, `mkv3-p4b-postband-resweep1-*.log` +
`mkv3-p4b-idle32-s8192-decide-*.log`):
- **idle32 at S8192 FLIPPED and is PROMOTED** (commit after `21e9f86`): rejected
  pre-band (+13.2us, 1/16), post-band 8/8 paired runs negative (deltas -1.4 to
  -15.4, pooled median -11.2us, wins 100/128). Promoted-default vs forced-256
  validation (`mkv3-p4b-idle32-s8192-promote-20260705T191108Z.log`):
  +14.2 (3/16 for old) / +2.3 (6/16), parity clean. `idle_ns_default` map is now
  `{3072, 4096, 8192} -> 32`.
- cold_cap {16, 33} at S8192: order-mixed wash (+7.4/-4.0 and +8.9/-4.0) — keep
  uncapped.
- head-dx target {96, 384} at S8192: neutral-to-mixed (+0.3/-2.0 and
  +4.0/-11.9) — keep 192.
Remaining resweep batch 2 for whoever picks it up: cold_cap + dW targets + TN
gate + n128 modes at S3072/S4096, and the same matrix at S2048.

Resweep batch 2 (S2048/S3072/S4096, `mkv3-p4b-postband-resweep2-*.log`): NO
flips — the batch-1 flips were both S8192 scheduler-timing knobs; the shorter
gated shapes were already re-tuned post-band by the concurrent sessions.
- cold_cap {16, 33}: order-mixed washes at all three shapes (|delta| <= 3.4us,
  13-30/40 wins either way) — keep current caps.
- `MK_WGMMA_TN=0`: +161.5/+165.3 (S3072) and +169.2/+173.9 (S4096), 0/40
  everywhere — the TN dW gate is load-bearing post-band, strongly confirmed.
- `MK_DW_TARGET_TILES=192`: +166.9/+167.7 (S3072) and +169.2/+172.0 (S4096),
  0/40 — the K>=2048 -> 128 dW split target likewise.
Follow-on S2048 idle-poll sweep (separate GPU5 lane,
`mkv3-p4b-idle-ns-s2048-postband-20260705T1915Z.log`) found one more cheap
scheduler flip outside the batch-2 matrix: `MK_IDLE_NS=32` beat the old S2048
256ns default by -12.13us/-14.74us with 40/40 and 38/40 wins, parity clean
(`worst_grad_rel` <= 0.005891). `MK_IDLE_NS=64` also won (-8.66/-13.14us) and
128 was weaker (-4.35/-8.16us), so promote S2048 into the idle32 bucket.
Promoted-default validation `mkv3-p4b-idle32-s2048-promote-20260705T1920Z.log`
beat forced old 256ns by +15.04us/+11.34us old-minus-new (old wins only 1/40
and 2/40), parity clean. The post-band resweep vein is otherwise mined at the
long shapes; the yielded flips are cached SwiGLU 2W at S8192 plus idle32 at
S2048/S8192.

S8192 fwd-band retune on the new T40-bwd base (session 2853e0de,
`mkv3-p4b-s8192-fwdband-retune-20260705T195210Z.log`): **fwd T=64 is a
sharp optimum, confirmed** — T48 +105.3/+90.2us (0/16 both orders), T56
+40.9/+27.7 (1/16), and the coarse side falls off the unsplit-straggler cliff:
T72 +398.8/+387.5, T80 +510.3/+476.1 (T>64 leaves a >64-stage q-tile at C=1;
T=64 splits exactly everything above one 64-stage chunk). idle_ns fine sweep on
the same base: idle16 +8.7/-11.0 and idle64 +9.9/-3.8 — order-mixed washes,
keep idle32. S8192 defaults after the combined day: bwd band T40, fwd band
T64, idle32, cached SwiGLU 2W; absolute ~7120-7150us in the resweep harness.

Post-band n128 recheck (session 2853e0de, GPU 6,
`mkv3-p4b-n128-postband-recheck-gpu6-*.log`): all n128 routing defaults are
CONFIRMED load-bearing post-band, no flips. S4096: `MK_WGMMA_N128=0`
+132.4/+123.7us (0/40 both), `=2` (lm-head-only) +65.7/+54.4 (0/40), `_NN=0`
+33.2/+27.1 (<=2/40). S8192: `=0` +423.0/+400.0 (0/16), `=2` +268.0 clean
order (its default_first window had a transient co-tenant spike to 14.7ms
absolutes — delta +286.7 corroborates but is flagged), `_NN=0` +96.4/+71.6
(<=1/16). GPU-2 lane note: that GPU reads 1.6-1.8x-high absolutes under
clean-looking guards (invisible out-of-container tenant) — do not time there.

Combine-R sweep (session 2853e0de, `mkv3-p4b-combine-r-sweep-*.log`): the R=8
default (promoted vs R=1 only) is CONFIRMED against its neighbors. R=4 loses
(S4096 +5.4/+8.0 with <=9/40; S8192 +15.7/+0.6), R=16 is order-mixed neutral
(S4096 -2.5/-0.2 at 31/40 and 24/40; S8192 +4.3/-10.8) — under the promotion
bar; keep `MK_ATTN_COMBINE_R=8`.

S8192 attention-dKV row-scalar broadcast: the prior default-off row-broadcast
probe had only tested nano/small pre-long-S retunes, so the current S8192
DKV-heavy profile reopened it. `MK_ATTN_DKV_ROW_BCAST=1` makes each 4-lane
column group load one `LSE`/`Drow` scalar and distribute it with a warp shuffle.
Flagged WGMMA attention parity passed
(`mkv3-p4b-rowbcast-s8192-current-20260705T2128Z.log`, WGMMA bwd dqkv
max_abs_err `6.958e-03`). S8192 env A/B won both orders by -10.90us and
-9.02us; same-cache confirmation strengthened to -16.85us and -13.97us
(`mkv3-p4b-rowbcast-s8192-confirm-20260705T2135Z.log`). Promoted-default vs
forced old `MK_ATTN_DKV_ROW_BCAST=0` favored the new default by +10.61us/+7.60us
old-minus-new with parity clean
(`mkv3-p4b-rowbcast-s8192-promoted-default-20260705T2140Z.log`). The default is
exact H256/D64/S8192 only in the consolidated tuning table; the env override
still forces old/new for A/B.

Post-GEMM-mbarrier-ring correction (20260706): row-broadcast no longer composes with
the long-D64 default. After `31dad00` (`MK_GEMM_MBAR_RING` default for long D64),
forced old `MK_ATTN_DKV_ROW_BCAST=0` beat the promoted S8192 default in both
construction orders: -32.43us and -30.77us, 16/16 wins each, with parity clean
(`mkv3-p4b-post-gmbar-s8192-rowbcast-default-first-20260706T0807Z.log`,
`mkv3-p4b-post-gmbar-s8192-rowbcast-variant-first-20260706T0807Z.log`). The exact
default gate is therefore empty again; `MK_ATTN_DKV_ROW_BCAST=1` remains available for
A/B and future composition checks.

Post-row-broadcast S8192 profile/score refresh:
`mkv3-p4b-profile-score-s8192-post-rowbcast-d3581ed-20260705T2145Z.log` (first
attempt failed before CUDA due to a relative venv path; retry completed with GPU5
pre/post guards clean). Score is megakernel 7016.6us, eager 8413.0us, compile
3593.3us, compile+CUDAGraph 3293.9us, and compile+CUDAGraph+ 3123.3us, so the
long-S gap is 2.25x vs the hardened graph baseline. The profile total is
7092.3us with `n_instr=188`, `critical_path=80`, `gated=63`; on-path leaders are
still `ATTN_DKV_WG` 2584.4us (2073.0us wait + 511.4us span across four hops),
`ATTN_FWD_WG` 1092.6us, RMSNorm bwd-dx 462.7us, lm-head NT 435.8us, qknorm/rope
bwd 279.8us, and swiglu bwd 278.8us. Conclusion: the row-broadcast win is real
but small; the next long-S work should attack DKV dependency/wait structure or
attention-rowop fusion, not scalar-load cleanup.

S8192 H256 RMSNorm bwd-dx fixed-width route: commit `f54b4dd` adds
`OP_RMSNORM_BWD_DX_H256`, a two-row-per-warp dx-only body specialized for H=256
(no runtime H loop/divide), and gates it only for exact H256/S8192 via
`_H256_RMS_DX_H256_S`. `MK_RMS_DX_H256=1/0` still force-enables/restores the old
route for A/B. Route/parity
(`mkv3-p4b-rmsdx-h256-route-parity-20260705T2152Z.log`) passed: forced route
emits nine H256 dx instructions and zero old dx-route opcodes at S512/S4096/S8192;
`test_model` passed under the forced route (S512 worst grad rel 0.0281,
D128-ragged 0.0201, rerun/waves/df2/ws and SGD sanity clean). Timing
(`mkv3-p4b-rmsdx-h256-ab-20260705T2158Z.log`) rejected S4096 as order-mixed
(-16.45us default-first, +1.23us variant-first) but promoted S8192 (-25.10us and
-43.33us, 16/16 wins both orders). Promoted-default confirmation
(`mkv3-p4b-rmsdx-h256-promoted-default-20260705T2201Z.log`) proved S4096 remains
old R2 (`h256=0 r2=9`) while S8192 uses the new opcode (`h256=9 r2=0`), and
forced-old `MK_RMS_DX_H256=0` was slower by +23.33us/+26.99us old-minus-new with
parity clean. Do not widen to S4096 without a fresh post-composition recheck.

H256/S512+S1024 RMSNorm bwd-dx fixed-width route promotion: after the short-shape
profile refresh (`mkv3-p4b-profile-short-post-drowzero-06f3101-20260705T2203Z.log`)
put `RMSNORM_BWD_DX` at 64-65us on-path for S256/nano, an env-only
`MK_RMS_DX_H256=1` A/B rechecked short H256 shapes. S256 is **not** promoted:
default-first was neutral/slightly negative (+0.58us variant-minus-default in the
focused repeat), despite a variant-first win
(`mkv3-p4b-rmsdx-h256-short-confirm-20260705T2207Z.log`). Nano/S512 and H256/S1024
were promoted after the balanced-order confirmation removed second-call bias:
nano old-minus-new +2.42us/+6.40us with 196/300 and 260/300 default wins, and
S1024 +15.28us/+5.89us with 236/240 and 197/240 default wins
(`mkv3-p4b-rmsdx-h256-short-balanced-20260705T2207Z.log`). Promoted-default route
guard (`mkv3-p4b-rmsdx-h256-short-promoted-default-20260705T2207Z.log`) proved
S128 stays on the FMA route, S256 and S4096 stay on R2, and nano/S1024 use nine
H256 dx opcodes. Final route/ref validation passed
(`mkv3-p4b-rmsdx-h256-short-route-validate-20260705T2207Z.log`: S1024 rel loss
4e-6, worst grad rel 0.020981), and full `test_model.py` passed
(`mkv3-p4b-rmsdx-h256-short-testmodel-20260705T2207Z.log`: nano worst 0.0281,
D128-ragged 0.0195, rerun/waves/df2/ws and SGD sanity clean). The exact default
gate is now H256/S512, H256/S1024, and H256/S8192.

Post-RMS S8192 profile/score refresh:
`mkv3-p4b-profile-score-s8192-post-rmsdx-9eb4c2e-20260705T2205Z.log` measured
profile total 7233.5us with `n_instr=188`, `critical_path=80`, `gated=63`.
`RMSNORM_BWD_DX_H256` is now on path at 428.4us, but the structural leaders are
still `ATTN_DKV_WG` 2649.7us (2112.1us wait + 537.6us span), `ATTN_FWD_WG`
1132.5us, lm-head NT 440.0us, qknorm/rope bwd 312.1us, and MLP dx 308.4us.
The refreshed score was megakernel 7156.9us vs compile+CUDAGraph+ 3119.0us
(2.29x gap). Absolute S8192 medians remain noisy; use the paired default-vs-old
logs above as promotion evidence. For next work, DKV wait/dependency structure is
still a larger target than local row-op scalar cleanup.

S4096 H256/I768 SwiGLU-BWD 3W source probe no-go: a temporary
`MK_SWIGLU_BWD_3W=1` route mapped each row across three warps and emitted four
3W instructions for H256 long shapes. Forced-route validation passed
(`mkv3-p4b-swiglu-3w-route-parity-20260705T2212Z.log`: S512 worst grad rel
0.0281, D128-ragged 0.0211, rerun/waves/df2/ws and SGD sanity clean), and the
initial S4096 A/B looked positive after a noisy first window
(`mkv3-p4b-swiglu-3w-ab-20260705T2218Z.log`, then focused confirmation
`mkv3-p4b-swiglu-3w-s4096-confirm-20260705T2224Z.log`: about -50us both
orders). Promoted-default confirmation invalidated the change: with 3W selected
as the unset default at S4096, forced old `MK_SWIGLU_BWD_3W=0` was -6.42us
faster in default-first and neutral (+0.38us old-minus-new) in variant-first
(`mkv3-p4b-swiglu-3w-promoted-default-20260705T2228Z.log`). S8192 was also
neutral-to-slightly-worse in the initial A/B (+6.08us/+1.36us). The source probe
was reverted; keep the existing cached 2W defaults and do not re-add a 3W route
without a new post-composition reason.

H256/S256+S512 Drow zero-fill skip: the D=64/S<2048 direct-store Drow epilogue
overwrites each `drow[qh,row]` once, so the upfront `drow` zero-fill is
unnecessary only when that compile-time direct-store route is actually selected.
Route checks (`mkv3-p4b-drow-zeroskip-route-20260705T2141Z.log`,
`mkv3-p4b-drow-zeroskip-route2-20260705T2141Z.log`) showed the intended
instruction deltas: nano and H256/S1024 remove four Drow fill instructions,
small removes eight, while S2048/D128 keep the fill. Validation
(`mkv3-p4b-drow-zeroskip-validation-control-20260705T2141Z.log`) showed the
small worst-grad envelope was identical with and without the skip, so the route
is correctness-clean. Paired timing rejected a broad gate: small preferred the
old zero-fill by 13.74us and 7.47us, and H256/S1024 was order-mixed (+2.06us,
-6.54us zero-minus-skip). S128 was also order-mixed (-1.23us, +4.96us). The
promoted exact default is therefore H256/L4/D64 at S256 and S512 only: S256 won
by +7.89us/+1.82us zero-minus-skip, and nano/S512 won by +0.93us/+2.16us
(`mkv3-p4b-drow-zeroskip-short-ab-20260705T2141Z.log`,
`mkv3-p4b-drow-zeroskip-ab-20260705T2141Z.log`). `MK_DROW_ZERO_FILL=1` restores
the old zero-fill on direct-store overwrite shapes for A/B; atomic Drow shapes
keep the fill even if the env is set to 0. Post-promotion route/ref validation
passed (`mkv3-p4b-drow-zeroskip-promoted-route-validate-20260705T2158Z.log`:
S256 rel loss 9e-6, worst grad rel 0.023252), and full `test_model.py` passed
(`mkv3-p4b-drow-zeroskip-promoted-testmodel-20260705T2158Z.log`: nano worst
0.0281, D128-ragged 0.0208, rerun/waves/df2/ws and SGD sanity clean).

## v3 P4b knob consolidation (session 2853e0de): one tuning table, routes verified

The secondary lane from the /goal: model.py's ~12 scattered per-shape gate
expressions are consolidated (commit `56ab960`) into one module-level "measured
per-shape tuning" section: `_SWIGLU_CACHED_2W` (drives both cache-sig and 2W
defaults), `_H256_IDLE32_S`, `_H256_DQ_FLOAT2_S`, `_ATTN_BWD_BAND_T`,
`_ATTN_FWD_BAND_T`, `_ATTN_BAND_DQ_FIRST_S`, `_H256_D64_QKBWD_SPLIT_V_S`,
`_H256_ATTN_CHUNKS`, `_HEAD_DX_TARGET`, and `_cold_cap()`. Each knob keeps the
exact gate dimensionality it was measured under (keys deliberately differ —
collapsing them onto one (H, S) key would change semantics off-gauntlet), so
this is a pure relocation. Formula gates (drow direct store, the exp2 family,
dkv float2, dx_split_k tile gates) stay inline: they are shape math, not tuned
constants. Env overrides unchanged.

VERIFIED route-preserving: `results/route_snapshot.py` (knob-consol worktree)
hashes the instruction stream, claim/crit vectors, dep counts, adjacency, and
cold cap per shape; old-vs-new model.py compared IDENTICAL on all 11 gauntlet
shapes (nano, deep-L12, S128, S256, S1024, S2048, S3072, S4096, S8192, small,
D128-ragged) — `results/mkv3-p4b-knob-consol-verify-20260705T204708Z.log`.
Retunes now edit one table line instead of a scattered expression, which also
cuts the cross-session model.py merge conflicts that recurred all day.
Operational note for the snapshot harness: a killed extension build leaves a
stale `<extdir>/<name>/lock` that silently wedges the NEXT build — the
symptom is a snapshot process alive with zero output; delete the lock file.

S8192 post-specialization resweep (session 2853e0de, after rowbcast-dkv
`b54d322` + rms-dx-H256 `f54b4dd`;
`mkv3-p4b-s8192-postspec-resweep-*.log`): ALL current S8192 defaults hold under
the newly specialized ops — bwd band T40 confirmed (T36 +102.0/+104.1 with
0/16 both orders, T44 +10.7/+22.6 with <=2/16), fwd band T64 confirmed (T56
+26.1 clean order 0/16 — its reverse window hit the transient co-tenant, 14.9ms
absolutes, delta +9.0 corroborating-only; T72 +413.3/+417.7, the unsplit-
straggler cliff again), idle32 confirmed (idle64 +1.6/+5.8, 7-8/16). The two
op-specializations changed spans but not the band/idle optima; the resweep law
is satisfied for this regime shift with no changes.

## v3 P4b operator-gap GEMM lane: deep SW128 stages are a no-go under global smem

The operator-gap report's first GEMM recommendation was tested by extending
`pipe_probe.py` with SW128 S6/S8/S9 long-K variants and a `longdx` mode. The
standalone result is real but too small for the current executor's global smem
page. Repeat log `mkv3-p4b-pipe-longdx-deepstage-repeat-20260705T2218Z.log`
showed S8 as the best deep variant versus current S2 SW128:

- `8192x256x8192` NN: 187.4us -> 175.8us.
- `8192x256x1536` NN: 44.3us -> 42.9us.
- `1024x512x3072` NN: 43.1us -> 41.5us, but production small uses the n128
  route for this shape, so this is only a m64n64 reference point.

Focused head-dX slice/full-shape probes
(`mkv3-p4b-pipe-headdx-deepstage-20260705T2224Z.log`) found similar small wins:
S1024 split slice `1024x256x4096` 53.6us -> 51.7us, S2048
`2048x256x8192` 95.8us -> 91.2us, S4096 `4096x256x8192` 96.2us -> 91.8us, and
nano slice `512x256x4096` 53.6us -> 51.3us. S9 was consistently slower than S8.

The integration blocker is that an S8 128x64 ring needs a 208KB dynamic-smem
request for the whole cooperative megakernel launch. Current-code smem controls
with no deep route measured the cost directly:
`mkv3-p4b-smem208-control-20260705T2220Z.log` gave nano +6.1us, H256/S1024
+11.9us, and small +42.2us; `mkv3-p4b-smem208-long-control-20260705T2225Z.log`
gave S2048 +2.4us, S4096 +67.2us, and S8192 +94.9us. Those costs dominate the
per-GEMM S8 win at every candidate shape, especially the long shapes where the
global page perturbation is largest. Do not promote a default route by simply
setting a new WGMMA flag and bumping `Program.run()` smem. Revisit only if the
executor grows a per-op big-smem page, a separate deep-GEMM kernel, or a
warp-specialized producer/consumer variant that can amortize the page without
changing the whole step.

## v3 P4b qwen4b-l1 scheduler retune: uncap cold sinks for giant L1 shapes

The operator-gap qwen4b-l1 addendum exposed an L=1 scheduling regime that the
small-shape cold-cap rule actively hurts: with one layer, there is little later hot
work to overlap with giant dW sinks, so cap48 leaves the enormous lm-head dW work
unfinished at the end of the step. Env-only sweep
`mkv3-p4b-qwen4b-coldcap-sweep-20260705T2229Z.log` on
`Cfg(H=2560, L=1, nq=32, nkv=8, D=128, I=9728, V=151936, S=1024)` found cap0
decisively best: cap48 29432.9us, cap0 22094.0us, cap64 23021.5us; smaller caps
starved the sinks (cap4 215.8ms, cap8 62.9ms, cap16 41.0ms).

Focused repeat `mkv3-p4b-qwen4b-coldcap-confirm-20260705T2234Z.log` confirmed both
orders: cap48 29390.4/29438.5us, cap0 22062.3/22096.4us, cap64
23087.2/23047.4us. Default `_cold_cap()` now returns 0 only for conservative
single-layer giant-vocab shapes (`L==1`, `H>=1024`, `V>=32768`). Existing gauntlet
defaults are unchanged; `MK_COLD_CAP=48` restores the old qwen4b-l1 behavior for A/B.

Post-cap qwen4b-l1 profile
`mkv3-p4b-qwen4b-profile-post-coldcap-0578594-20260705T2238Z.log` confirms the
sink starvation was removed, but the shape is still far from the baseline: 22118.5us,
26 chain hops, 6883.5us wait, 15235.0us span. On-path leaders are giant lm-head
fwd/dX (`GEMMNN 1024x2560x151936.wg.splitK` 4440.0us and
`GEMMNT 1024x151936x2560.wg` 3553.3us), then the generic D=128 attention fallback
(`ATTN_FWD` 1813.3us, `ATTN_DQ` 1503.3us). The D=128 WGMMA/FA4-C route remains
the real attention fix; the following is only a narrow fallback cleanup.

Generic D=128 attention dQ Cq retune: the non-WGMMA path had a hardcoded `Cq=4`
whenever `n_qt>=8`. A no-source instruction-stream sweep on the qwen4b-l1 shape
(`mkv3-p4b-qwen4b-generic-dq-cq-sweep-20260705T2240Z.log`) found lower chunking
better: Cq1 medians 21948.0/21935.2us, Cq2 21970.8/21975.4us, Cq3
21996.6/22029.3us, old Cq4 22089.3/22100.9us; Cq6/8 regressed. Promoted an
exact attention-shape default gate `(H,S,nq,nkv,D)=(2560,1024,32,8,128)` to Cq1;
`MK_ATTN_DQ_C=4` restores the old generic fallback. Promoted A/B log
`mkv3-p4b-qwen4b-generic-dq-c1-promoted-20260705T2243Z.log` verified route emission
(`OP_ATTN_DQ` 1024 tiles/Cq1 default vs 4096 tiles/Cq4 forced old), identical loss,
focused max-abs grad diffs at numerical-noise scale, and paired timing wins of
85.1us and 94.8us old-minus-new. Full `test_model.py` passed in
`mkv3-p4b-qwen4b-generic-dq-c1-testmodel-20260705T2244Z.log`, including the
existing D=128 ragged fallback guard and SGD sanity.

## v3 P4b lm-head fat-tile standalone probe: staged wins, cooperative page blocks

`fat_gemm_probe.py` adds a standalone NT-only WGMMA probe for the lm-head forward
tile-count problem: current-style 128x128 n128 control, a staged 128x256 tile using
m64n256 GMMA, and a 100KB-compatible 128x256 direct-store epilogue. The first attempt
(`mkv3-p4b-fat-gemm-probe-20260705T2247Z.log`) faulted because the standalone smem
arrays were sized in bytes-as-elements; fixed in the probe only.

Validated retry `mkv3-p4b-fat-gemm-probe-retry-20260705T2250Z.log` showed the staged
128x256 tile is a real standalone win: small lm_head 73.9us -> 60.7us, S8192 lm_head
182.8us -> 154.6us. Qwen prefix log
`mkv3-p4b-fat-gemm-probe-qwen-20260705T2251Z.log` showed
`1024x151808x2560` 2791.5us -> 2459.0us. But the staged epilogue needs a 160KB
dynamic-smem page, and unchanged-model launch control
`mkv3-p4b-smem160-tax-20260705T2252Z.log` failed immediately at 160KB with
`cudaErrorCooperativeLaunchTooLarge` for the 132-block cooperative launch. Do not
integrate the staged 128x256 route into the current single cooperative kernel.

The 100KB direct-store 128x256 variant avoids the cooperative page limit but is not a
broad route: `mkv3-p4b-fat-gemm-probe-direct-20260705T2254Z.log` measured small
lm_head 73.1us control vs 72.2us direct (noise) and S8192 182.8us control vs 193.2us
direct (regression). It does win the qwen-like high-K/high-V prefix:
`mkv3-p4b-fat-gemm-probe-direct-qwen-20260705T2255Z.log` measured
`1024x151808x2560` 2784.7us control vs 2490.9us direct. A production attempt should
therefore be qwen-specific, direct-store only, and must add the missing pieces before
route promotion: CE/lse partials from registers, a guarded 128-column tail for
`V=151936`, and focused old-vs-new gradient/timing validation. Do not widen the
general lm-head route from these standalone numbers.

## v3 P4b qwen lm-head direct n256 route: exact-shape promotion

Promoted the production follow-up as an exact qwen4b-l1 lm-head gate, not a general
route. `ops.cuh` flag bit14 selects a 100KB m64n256 NT WGMMA direct-store path with
register CE/LSE partials and a guarded final 128-column tail; `mk.py` defaults it only
for `(M,N,K)=(1024,151936,2560)` with lm-head CE flags. `MK_WGMMA_N256_DIRECT=0`
restores the old n128 route; `=1` force-enables all structurally eligible lm-head CE
shapes for future probes.

Build/route log `mkv3-p4b-qwen-lmhead-n256d-prod-20260705T2257Z.log` verified the
qwen head emits one route-changed instruction: old n128 `ntiles=9496, flags=6274`;
new n256d `ntiles=4752, flags=18562`, with `nparts=2374` unchanged. Focused A/B log
`mkv3-p4b-qwen-lmhead-n256d-ab-20260705T2301Z.log` used identical params/tokens and
found loss rel-diff `3.0e-7`; selected gradient rel diffs stayed below `5.6e-3`.
Paired timings were decisive: old median `22051.4us`, new median `21142.3us`,
old-minus-new `+909.1us`.

Attribution log `mkv3-p4b-qwen-lmhead-n256d-profile-20260705T2301Z.log` confirms the
win lands on the intended hop: qwen `GEMMNT 1024x151936x2560.wg` span dropped
`3528.9us -> 2675.1us`, and step total dropped `21991.8us -> 21062.8us`. Full
`test_model.py` passed in `mkv3-p4b-qwen-lmhead-n256d-testmodel-20260705T2301Z.log`
(nano, D=128 ragged fallback, df/waves/df2/ws agreement, and SGD sanity). Keep the
standalone verdict intact: staged n256 remains blocked by cooperative smem, and broad
direct n256 remains rejected.

## v3 P4b qwen lm-head dX n128 no-atomic route: exact-shape promotion

After the qwen lm-head forward fix, the remaining largest on-path GEMM was
`dlogits @ Wlm`: default qwen emitted `GEMMNN 1024x2560x151936.wg.splitK` as
`ntiles=320, flags=168, sk=1`, still paying the split-K zero-fill/atomic route even
though the split count was one. Existing head-dX n128 support already had the needed
implementation, so this was an env-only qwen check before source promotion.

Initial three-arm log `mkv3-p4b-qwen-headdx-n128split-20260705T2306Z.log` proved route
emission but OOMed while comparing giant gradients across three resident qwen models.
The memory-safe rerun `mkv3-p4b-qwen-headdx-n128split-rerun-20260705T2309Z.log`
compared one candidate at a time with chunked gradient diffs. Forced n128 split-atomic
(`MK_HEAD_DX_N128_SPLIT=1`) changed the row to `ntiles=160, flags=4264, sk=1` and won
`21235.3us -> 19674.4us` (`+1560.9us`). Forced n128 fp32/no-atomic
(`MK_HEAD_DX_N128_F32=1`, `MK_HEAD_DX_NO_ATOMIC_SK1=1`) changed it to
`ntiles=160, flags=4232` and was better: `21204.1us -> 19576.0us`
(`+1628.1us`). Loss rel-diff was `-1.5e-7`; selected gradient rel diffs were below
`3.7e-7`.

Promoted only the exact qwen4b-l1 shape `(H,S,V,nq,nkv,D,L) =
(2560,1024,151936,32,8,128,1)` into the existing `MK_HEAD_DX_N128_F32` default.
`MK_HEAD_DX_N128_F32=0` restores the old split-atomic route. This does not broaden the
earlier n128 split verdicts for nano/small/general MLP dX.

## v3 P4b qwen lm-head dX n256 no-atomic route: exact-shape promotion

The post-head-route profile `mkv3-p4b-qwen-post-headroutes-profile-20260705T2312Z.log`
showed the qwen head-dX row was still a top non-attention leader even after the n128
promotion: `GEMMNN 1024x2560x151936.wg` emitted `ntiles=160, flags=4232` and took
`3609.3us` on path. This is orthogonal to the peer D=128 attention lane.

`fat_gemm_nn_probe.py` is the standalone guardrail for this route. It compares the
current m64n128 NN fp32 path against a 100KB-compatible m64n256 NN direct-store path
using the same SW128 descriptor patterns as the production op. Probe log
`mkv3-p4b-qwen-headdx-fat-nn-probe-20260705T2313Z.log` passed correctness
(`rel=1.24e-5`) and measured qwen head-dX `1024x2560x151936` at n128 `2740.9us`
(160 tiles, 290.6 TF) versus n256 `2126.1us` (80 tiles, 374.7 TF).

Promoted only exact qwen4b-l1 head-dX `(M,N,K)=(1024,2560,151936)` into a new
`MK_HEAD_DX_N256_F32` default gate. It reuses bit14 for a direct m64n256 NN fp32
path; `MK_HEAD_DX_N256_F32=0` restores the previous n128/no-atomic default, and `=1`
force-enables structurally eligible head-dX shapes for future probes. Route log
`mkv3-p4b-qwen-headdx-n256-prod-route-20260705T2315Z.log` verified default emission:
head-dX changed from n128 `160` tiles to n256 `80` tiles with flags `16520`.

Focused A/B `mkv3-p4b-qwen-headdx-n256-prod-ab-20260705T2318Z.log` used identical
params/tokens and old arm `MK_HEAD_DX_N256_F32=0`: old route
`[(30, 160, 4232, None, False, True)]`, new route
`[(30, 80, 16520, None, True, False)]`, loss rel-diff `-2.3e-7`, selected gradient
rel diffs below `4.9e-7`, and paired medians `19587.0us -> 18990.3us`
(`+596.7us` old-minus-new). Profile
`mkv3-p4b-qwen-headdx-n256-prod-profile-20260705T2319Z.log` showed current default
total `18839.3us`, with head-dX span `2465.5us`, lm-head fwd `2684.4us`, and generic
D=128 attention still large (`ATTN_FWD` `1822.1us`, `ATTN_DQ` `1542.2us`). Full
`test_model.py` passed in `mkv3-p4b-qwen-headdx-n256-prod-testmodel-20260705T2319Z.log`.

## v3 P4b qwen dW sk1 no-atomic route: exact-shape promotion

After the lm-head/head-dX promotions, the qwen4b-l1 profile
`mkv3-p4b-qwen-headdx-n256-prod-profile-20260705T2319Z.log` exposed a remaining dW
scheduler artifact: `GEMMTN 6144x2560x1024.wg.splitK` was on path with `5854.2us`
wait even though every qwen dW GEMM computed `sk=1`. Route inspection
`mkv3-p4b-qwen-dw-noatomic-route-20260705T2324Z.log` confirmed all five dW rows used
`flags=169, sk=1` and could structurally use the plain fp32 WGMMA TN store path.

Promoted only exact qwen4b-l1 dW shapes `(M,N,K)`:
`(151936,2560,1024)`, `(2560,9728,1024)`, `(19456,2560,1024)`,
`(2560,4096,1024)`, and `(6144,2560,1024)`. The new default emits `flags=137`
without split-K or atomics. `MK_DW_NO_ATOMIC_SK1=0` restores the prior split-K route;
`=1` force-enables other structurally eligible `sk=1` dW shapes for future probes.

Focused qwen A/B `mkv3-p4b-qwen-dw-noatomic-ab-20260705T2328Z.log` used identical
params/tokens and old arm `MK_DW_NO_ATOMIC_SK1=0`: old dW routes were five
`flags=169, sk=1` rows, new routes were five `flags=137` rows, loss rel-diff was
`-1.5e-7`, selected gradient rel diffs were below `5.0e-7`, and paired medians were
`19017.4us -> 15897.3us` (`+3120.0us` old-minus-new). Profile
`mkv3-p4b-qwen-dw-noatomic-profile-20260705T2331Z.log` showed current default total
`15843.5us`; the former on-path `GEMMTN 6144x2560x1024` is now off-path with a
`236.6us` span, and the visible wait moved to `EMBED_BWD` (`3561.7us`) behind the
remaining off-path giant vocab dW. Full `test_model.py` passed in
`mkv3-p4b-qwen-dw-noatomic-testmodel-20260705T2332Z.log`.

Post-promotion cold-cap resweep
`mkv3-p4b-qwen-coldcap-post-dwnoatomic-20260705T2334Z.log` was a no-go. The current
uncapped default still wins: cap64 lost `2661.7us`, cap96 lost `1082.2us`, and even
cap128 lost `50.2us` with only `1/12` wins. Keep qwen4b-l1 `default_cold_cap=0`; the
new `EMBED_BWD` wait is the remaining giant vocab dW drain, not a profitable cap target.

Follow-up TN fat-tile probe `mkv3-p4b-qwen-dwtn-fat-probe-20260705T2335Z.log`
validated a direct m64n256 fp32 TN tile for qwen dW: vocab dW `151936x2560x1024`
measured n64 `3480.7us`, n128 `2505.5us`, n256 `2112.3us`; wqkv dW
`6144x2560x1024` measured n64 `161.2us`, n256 `98.3us`. The production route reuses
the bit14 direct fp32 path with an A-transposed MN-major loader and is exact-gated by
`MK_DW_N256_TN_F32`: unset promotes only the five qwen dW shapes, `=0` restores the
current n64 no-atomic route, and `=1` force-enables structurally eligible TN probes.

Focused qwen A/B `mkv3-p4b-qwen-dwtn-n256-prod-ab-20260705T2338Z.log` used old arm
`MK_DW_N256_TN_F32=0`: old dW routes were five `flags=137` rows, new routes were five
`flags=16521` rows with tile counts quartered, loss rel-diff was `-1.5e-7`, selected
gradient rel diffs were below `5.9e-7`, and paired medians were
`16151.7us -> 12973.8us` (`+3178.0us` old-minus-new). Profile
`mkv3-p4b-qwen-dwtn-n256-prod-profile-20260705T2341Z.log` showed current total
`13047.0us` and on-path wait only `331.9us`; the remaining leaders are head-dX
`2924.8us`, lm-head forward `2684.8us`, and generic D=128 attention. Full
`test_model.py` and `test_ops.py` passed in
`mkv3-p4b-qwen-dwtn-n256-prod-testmodel-20260705T2342Z.log` and
`mkv3-p4b-qwen-dwtn-n256-prod-testops-20260705T2344Z.log`.

Post-dW route interaction resweep kept the existing qwen n256 head routes. Head-dX
n256 still beat forced old `MK_HEAD_DX_N256_F32=0` by `504.5us` median with `16/16`
wins (`mkv3-p4b-qwen-post-dwtn-route-interactions-20260706T0000Z.log`). The first
process OOMed before the lm-head half due to resident qwen model count, so the fresh
lm-head rerun `mkv3-p4b-qwen-post-dwtn-lmhead-route-20260706T0001Z.log` checked
`MK_WGMMA_N256_DIRECT=0` separately and kept default n256 by `862.0us` median with
`16/16` wins.

## v3 P4b D=128 WGMMA attention (session 2853e0de): the trio lands in-model

**PROMOTED, default-on for `D==128 && S%64==0`** (commits `1566e51..dac7321`,
merged): `OP_ATTN_{FWD,DKV,DQ}_WG128` replace the generic WMMA attention ops at
D=128. This lands the opgap session's FA4-C fallback-replacement spec (their
standalone trio in `attention_probe.cu` on `megakernel-opgap`,
`results/operator-gap/fa4c-d128-trio-round.md`) — independently converged on
the same two design rules: **split-D** (each WG owns a 64-wide D-half of every
output, so all accumulators stay standard m64n64 [32]-fragments and no new
descriptor layouts exist anywhere) and **P-parking** (only s[32] transient).
This implementation's third rule: **redundant-S** — both WGs compute S = Q K^T
(k=128 dual-subtile, one 8-fma commit batch) and the softmax/dS algebra
redundantly; WG0 publishes P/dS once; redundant tensor/SFU work is free in the
latency-bound regime and removes all cross-WG serialization except one
publication barrier per stage. 64-row tiles kill the beyond-diagonal skip
stages (n_stages = q0/64 + 1) and double instruction-level parallelism.

qwen4b-l1 (H2560/L1/nq32/nkv8/D128/V151936/S1024): env A/B **-1586.5/-1846.4us
(12/12 both orders)**; promoted-default vs forced-old WMMA **+1885.4/+1627.5us
(old wins 0/12 both)**; worst grad rel 0.008648, losses equal. Full test_ops +
test_model green with the route default (including df2/ws executor agreement at
the D128/S192 ragged config). ~9% of the qwen4b step from one route.

Two infrastructure fixes the route needed (both would have bitten any D=128
landing): (1) all four executor runners cached
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` in process-lifetime statics,
so mixed-carveout processes launched with a stale attribute
(cooperative-launch-too-large / illegal smem access); they now re-configure
whenever the requested carveout grows (`c590974`). (2) The DKV128 smem struct
is 112KB and ws mode offsets ops by 256B of control smem — a 112KB carveout
fits df byte-exactly but overruns in ws by 256B (illegal address only under
real timing; memcheck missed it; found by bisect + arithmetic). The route takes
the 120KB carveout (`f33f604`; MK_ATTN_PIPE precedent, measured neutral).

Follow-ons queued (integration deltas vs the opgap memos): (a) cross-time the
DQ register topology — the opgap probe holds both D-half accumulators
(dq0/dq1, REG:172) instead of redundant-S; adopt if faster (may close the gap
to their ~2.4ms standalone projection); (b) port the FA4-B fwd KV-widening
spec (+11-19% standalone, REG:165) to OP_ATTN_FWD_WG — the mechanism (halved
per-stage boundary costs) is the measured 1.5us/stage fwd overhead from the
straggler diagnosis; banding composes per their spec. Repro:
`results/d128_qwen4b_ab.py <order>` and
`mkv3-p4b-d128-{family-smoke-v3,promote}-*.log` in the attn-d128 worktree.

Current-head addendum after the qwen dW n256 route: full `test_model.py` with the
D=128 route default and full `test_ops.py` both passed
(`mkv3-p4b-d128-main-default-testmodel-20260706T0031Z.log`,
`mkv3-p4b-d128-main-default-testops-20260706T0036Z.log`). Fresh qwen4b-l1
default-vs-forced-old (`MK_ATTN_D128_WG=0`) A/B measured default `11043.2us` vs old
`13124.7us` in default-first order and default `11156.9us` vs old `13031.7us` in
old-first order, with old-minus-default `+2081.5us` / `+1874.8us`, `12/12` wins in
both orders, and worst selected grad rel `0.008648` on `emb`
(`mkv3-p4b-d128-main-default-vs-old-qwen4b-defaultfirst-20260706T0038Z.log`,
`mkv3-p4b-d128-main-default-vs-old-qwen4b-oldfirst-20260706T0042Z.log`).

Profiler footgun fixed: `profile_df.py` and `profile_waves.py` used to call
`Program.run` directly without `MKQwen3._smem_bytes`, so D=128 profiles silently
fell back to the 100KB default smem allocation. `compute-sanitizer` pinned the crash
to an invalid shared write in `op_attn_dq_wg128` at `wgmma_attention.cuh:1411`,
address just above 100KB
(`mkv3-p4b-d128-main-ragged-iclk-compute-sanitizer-20260706T0104Z.log`). The
profilers now pass `getattr(m, "_smem_bytes", None)` to `Program.run`, matching
`MKQwen3.step()`.

Post-fix qwen4b-l1 profile
`mkv3-p4b-d128-main-default-qwen4b-profile-20260706T0108Z.log` measured total
`11070.1us`, `n_instr=51`, `critical_path=26`, `gated=23`, and `360.7us` on-path
wait. Remaining on-path leaders are head-dX `GEMMNN 1024x2560x151936.wg`
(`2912.8us`), lm-head fwd `GEMMNT 1024x151936x2560.wg` (`2698.5us`), MLP
fwd/bwd (`956.3us`, `673.2us`), then `ATTN_DQ_WG128` (`598.8us`). `ATTN_FWD_WG128`
is `219.0us`, and `ATTN_DKV_WG128` is off-path at `466.1us`; the next qwen-class
lane is back to the giant vocab/head GEMMs, not D=128 attention.

## v3 P4b qwen no-residual NT bf16 n256 direct route

The existing m64n256 direct NT kernel was previously CE-only in dispatch even though
its body already performs ordinary bf16 output stores before the optional CE/LSE
partials. Extending it to no-CE NT bf16 needed two pieces: dispatch now sends any
`flags & 2` n256-direct GEMM to `op_gemm_wgmma_n256_direct` (the NN/TN fp32 route
still handles non-NT bit14 cases), and the CE partial epilogue returns early when
bit11 is absent. A permanent `test_ops.py` case now covers direct `2|128|16384`
NT bf16 output. The initial standalone probe caught the old dispatch bug because
non-CE n256 NT was incorrectly sent to the fp32 NN/TN kernel and produced garbage;
the fixed standalone check passed `128x256x64`, qwen `1024x6144x2560`, and qwen
`1024x19456x2560` (`mkv3-p4b-n256-ntbf16-standalone-fix-20260706T0130Z.log`).

The promoted default is exact-gated by `MK_WGMMA_N256_NT_BF16`: unset routes only
qwen4b-l1 no-residual NT bf16 forwards `(M,N,K)=(1024,19456,2560)` (`wgu`) and
`(1024,6144,2560)` (`wqkv`); `=0` restores the old n128 route, and `=1` force-probes
all structurally eligible NT bf16 shapes. Focused qwen A/B changed those rows from
`ntiles=1216/384, flags=4226` to `ntiles=608/192, flags=16514`; lm-head stayed on
the existing `flags=18562` n256+CE route. Default-first timing measured old
`11045.7us`, new `10816.7us`, old-minus-new `+228.9us`, `16/16` wins, with loss
rel diff `+2.26e-7` and worst selected grad rel `3.34e-7`. Reverse construction
order measured old `11015.3us`, new `10840.8us`, old-minus-new `+174.5us`, `16/16`
wins, worst grad rel `5.84e-7`
(`mkv3-p4b-qwen-n256-ntbf16-ab-fix-20260706T0134Z.log`,
`mkv3-p4b-qwen-n256-ntbf16-ab-fix-rev-20260706T0138Z.log`).

Post-route profile `mkv3-p4b-qwen-n256-ntbf16-profile-20260706T0139Z.log` measured
total `10830.1us`. The intended `GEMMNT 1024x19456x2560.wg` hop is now
`325.1us` after the tile count halves; remaining top on-path work is lm-head fwd
`3016.1us`, head-dX `2926.1us`, MLP dX `615.5us`, `ATTN_DQ_WG128` `501.7us`, and
residual NT MLP/WO forwards (`375.2us`, `196.6us`). Full `test_ops.py` and
`test_model.py` passed in
`mkv3-p4b-qwen-n256-ntbf16-testops2-20260706T0146Z.log` and
`mkv3-p4b-qwen-n256-ntbf16-testmodel-20260706T0141Z.log`.

## v3 P4b qwen residual NT bf16 n256 direct route

Extended the exact qwen `MK_WGMMA_N256_NT_BF16` route to the residual NT bf16
producers `(M,N,K)=(1024,2560,4096)` (`wo`) and `(1024,2560,9728)` (`wd`). The
n256 direct epilogue now adds bit4 residual before bf16 conversion, and when bit13
is set it emits sum-of-squares partials from the post-residual bf16-rounded output,
matching the existing RMSNorm SSQ contract. The model emission passes the SSQ buffer
and returns `do_ssq` for this path, so qwen `wo`/`wd` preserve the fused RMSNorm
producer shortcut instead of falling back to a separate variance pass.

Validation:
- Permanent unit coverage now includes direct `NT n256 residual` and `NT n256 ssq`;
  `test_ops.py` passed in
  `mkv3-p4b-qwen-n256-ntbf16-res-testops-20260706T0152Z.log`.
- Focused qwen A/B changed residual rows from n128 `ntiles=160, flags=12434` to
  n256 `ntiles=80, flags=24722` while keeping the no-residual rows on the earlier
  `flags=16514` route and lm-head on `flags=18562`.
- Default-first timing:
  `mkv3-p4b-qwen-n256-ntbf16-res-ab-20260706T0156Z.log` measured old `11181.2us`,
  new `10905.1us`, old-minus-new `+276.1us`, `16/16` wins, loss rel diff
  `+7.534e-08`, worst grad rel `0.006289`.
- Reverse construction/order timing:
  `mkv3-p4b-qwen-n256-ntbf16-res-ab-rev-20260706T0200Z.log` measured old
  `11069.7us`, new `10821.2us`, old-minus-new `+248.5us`, `16/16` wins, loss rel
  diff `+5.295e-07`, worst grad rel `0.006168`.
- Post-route profile
  `mkv3-p4b-qwen-n256-ntbf16-res-profile-20260706T0204Z.log` measured total
  `10850.8us`. On-path residual NT spans are now `GEMMNT 1024x2560x9728.wg`
  `287.1us` and `GEMMNT 1024x2560x4096.wg` `155.9us`; no-residual `wgu`/`wqkv`
  remain `313.1us` and `123.3us`. The remaining top work is still lm-head fwd
  `3045.1us` and head-dX `2947.6us`, so the next qwen GEMM lane is those giant
  vocab/head paths or the shared mbarrier-ring port, not another narrow NT-forward
  expansion.
- Full `test_model.py` passed in
  `mkv3-p4b-qwen-n256-ntbf16-res-testmodel-20260706T0208Z.log`.

## v3 P4b mbarrier feed-ring in-model probe: NO-GO for current WGMMA ops

Tried the operator-gap barrier-free feed-ring prescription as an in-model integration
slice, but did not promote it. The prototype was kept opt-in during measurement and
then removed from tracked code after timing.

Coverage and correctness:
- Generic m64n64 depth-4 ring direct NN/NT cases passed `test_ops.py`
  (`mkv3-p4b-mbar-generic-testops2-20260706T0224Z.log`), and full
  `test_model.py` with only the generic ring opt-in passed
  (`mkv3-p4b-mbar-generic-testmodel-20260706T0228Z.log`).
- n128 depth-3 ring direct NN/NT cases also passed `test_ops.py`
  (`mkv3-p4b-mbar-n128-testops-20260706T0240Z.log`), and full `test_model.py`
  with only `MK_WGMMA_N128_MBAR=1` passed
  (`mkv3-p4b-mbar-n128-testmodel-20260706T0244Z.log`).

Timing rejected both slices:
- Generic m64n64 ring (`mkv3-p4b-mbar-generic-ab-20260706T0232Z.log`): nano routed
  16 rows and regressed old `933.9us` -> mbar `949.0us` (old-minus-mbar `-15.2us`,
  `0/20` wins). Small routed zero rows through the generic path, so its `+6.8us`
  old-minus-mbar result was noise, not a promotion signal.
- n128 ring (`mkv3-p4b-mbar-n128-ab-20260706T0248Z.log`): nano routed zero rows
  (noise: old-minus-n128_mbar `-2.6us`, `9/24` wins). Small routed 48 rows and
  regressed decisively: old `3549.5us`, n128_mbar `3600.4us`,
  old-minus-n128_mbar `-51.0us`, `1/24` wins.

Interpretation: the standalone barrier-free ring win does not transfer when bolted
onto the current interpreter op bodies. The current helpers still drain each GMMA
batch with `warpgroup_wait<0>`, and the deeper smem page/phase protocol adds overhead
without reducing the realized model path. Keep the operator-gap result as a future
rewrite spec for a deeper mainloop/producer design, but do not carry this direct
port in main.

## v3 P4b D=128 dQ row-split route: exact-shape qwen promotion

The operator-gap D=128 dQ topology handoff (`attn_dq_d128`: 128-row q tile,
each WG owns 64 q rows and accumulates both D halves) does transfer if we keep
the carveout tight and avoid the standalone probe's unnecessary C==1 atomics.

Implementation:
- `OP_ATTN_DQ_WG128` reuses a high bit in `Craw` (`1 << 24`) to select the
  row-split body; the old 64-row redundant-S body remains the fallback.
- Row-split smem is 144KB; qwen default uses a 148KB dynamic-smem launch
  (`MK_ATTN_D128_DQ_RS=0` restores old 120KB redundant-S, `=1` forces the
  row-split route for eligible `D==128 && S%128==0` configs).
- The route uses the existing conflict-reduced 64x128 loader mapping, fuses
  S and dP as one k=128 x2 WGMMA batch, then direct-stores both D halves when
  `C==1` (the qwen path). The atomic/staged epilogue remains only for `C>1`.
- Default gate is exact qwen4b-l1: `(H,S,I,V,nq,nkv,D,L) =
  (2560,1024,9728,151936,32,8,128,1)`.

Evidence:
- Promoted-default A/B vs forced old (`mkv3-p4b-d128-dqrs-promoted-ab-20260706T0119Z.log`):
  unset default routes `[(256, 16777217)]` at 148KB; forced old routes
  `[(512, 1)]` at 120KB. Parity is clean in both construction orders
  (loss diff <= `2.9e-06`, worst grad rel `0.006289`). Paired timing over
  64 pairs: old-minus-default `+32.54us` / `+35.79us` medians, `50/64` wins
  in both orders.
- Instruction profile (`mkv3-p4b-d128-dqrs-direct-profile-20260706T0118Z.log`):
  dQ span drops `317.6us -> 186.0us`; best profiled step total drops
  `10823.9us -> 10711.2us`, median `10853.2us -> 10792.8us`.
- Earlier direct-store paired delta before promotion
  (`mkv3-p4b-d128-dqrs-direct-paired-20260706T0118Z.log`) was positive in both
  pair orders: `+19.71us` / `+19.22us` median old-minus-new.

Validation:
- Route/default checks: unset env rowsplit enabled, `MK_ATTN_D128_DQ_RS=0`
  old route, `=1` forced row-split.
- `test_model.py` PASS (`mkv3-p4b-d128-dqrs-testmodel-20260706T0119Z.log`).
- `test_ops.py` PASS (`mkv3-p4b-d128-dqrs-testops-20260706T0120Z.log`).
- `py_compile`, `ruff check` (system `/home/apanda/.local/bin/ruff`), and
  `git diff --check` passed.

## v3 P4b D=128 fwd mbarrier ring: exact-shape qwen promotion

The operator-gap round-3 D=128 forward ring is now in-model for the exact
qwen4b-l1 shape. Unlike the rejected generic GEMM ring, this path removes the
forward attention stage boundary that remained visible after composition, and
it fits inside the existing high-smem qwen launch envelope.

Implementation:
- `OP_ATTN_FWD_WG128` reuses a high bit in its D argument (`1 << 24`) to select
  the mbarrier body; `MK_ATTN_D128_FWD_MB=0` restores the old forward route,
  `=1` forces the ring for eligible `D==128 && S%128==0` configs.
- The ring uses 128-row q tiles, each WG owns 64 rows and accumulates both D
  halves, with a two-stage K/V mbarrier ring (`cp.async.mbarrier.arrive.noinc`)
  and per-WG P visibility barriers. It uses `consumer_sync()`, not
  `__syncthreads()`, so ws mode remains valid.
- Default gate is exact qwen4b-l1: `(H,S,I,V,nq,nkv,D,L) =
  (2560,1024,9728,151936,32,8,128,1)`. It composes with the qwen dQ row-split
  route and keeps the 148KB launch carveout already needed by dQ row-split.

Evidence:
- Promoted-default A/B vs forced old (`mkv3-p4b-d128-fwdmb-promoted-ab-20260706T0130Z.log`):
  unset default routes forward `[(256, 16777344)]`; forced old routes
  `[(512, 128)]`, both at 148KB because dQ row-split remains active. Parity is
  clean in both construction orders (loss diff <= `9.6e-07`, worst grad rel
  printed as `0.000000`). Paired timing over 64 pairs: old-minus-default
  `+38.13us` / `+31.73us` medians, `58/64` and `52/64` wins.
- Instruction profile (`mkv3-p4b-d128-fwdmb-profile-20260706T0129Z.log`):
  forward span drops `225.4us -> 131.5us`; best profiled step total drops
  `10813.8us -> 10742.8us`, median `10842.3us -> 10770.4us`.

Validation:
- Route/default checks: unset env fwd ring enabled, `MK_ATTN_D128_FWD_MB=0`
  old forward route, `=1` forced ring.
- qwen one-step smoke with the private extension build passed.
- `test_model.py` PASS (`mkv3-p4b-d128-fwdmb-testmodel-20260706T0131Z.log`).
- `test_ops.py` PASS (`mkv3-p4b-d128-fwdmb-testops-20260706T0131Z.log`).
- `py_compile`, `ruff check`, and `git diff --check` passed.

## v3 P4b D=128 dQ mbarrier/split-wait: NO-GO in-model

After the qwen dQ row-split and fwd mbarrier promotions, the remaining
operator-gap D=128 bwd lead was a dQ mbarrier K/V ring with split S/dP WGMMA
waits (run exp/mask while dP is still flying). It was tested as an opt-in
`OP_ATTN_DQ_WG128` high-bit route on top of the current qwen default:
forward mbarrier still active, dQ row-split still active, 148KB carveout.

Evidence:
- Route/smoke: `MK_ATTN_D128_DQ_MB=1` emitted dQ `[(256, 50331649)]`
  (`row-split | mbar`) and completed the qwen one-step smoke.
- Paired A/B vs current default (`mkv3-p4b-d128-dqmb-ab-20260706T0138Z.log`):
  parity was clean in both construction orders (loss diff `2.9e-06`, worst
  grad rel `0.000001`), but timing did not clear the bar:
  default-minus-mbar medians `-0.82us` and `-11.39us`, with weak `31/64` and
  `22/64` wins.
- Instruction profile (`mkv3-p4b-d128-dqmb-profile-20260706T0138Z.log`) showed
  why: dQ span barely moved (`201.9us -> 198.6us`) and total median worsened
  (`10848.0us -> 10879.5us`). The local split-wait overlap is too small after
  the row-split/direct-store route, and the extra mbarrier protocol is absorbed.

Outcome: candidate source was reverted; do not carry `MK_ATTN_D128_DQ_MB`.
The next D=128 bwd direction should be the operator-gap S^T/register-A feed
design, not another K/V-ring bolt-on.

## v3 P4b qwen direct-store dW fill elision: exact-route promotion

After the qwen dW sk1/no-atomic and n256 TN promotions, the current qwen4b-l1
program still zero-filled every fp32 gradient up front. The two giant vocab fills
were visible in the post-D128 profile: `grad:emb` and `grad:wlm` both zeroed
`388956160` fp32 elements. `grad:emb` is still required because `EMBED_BWD`
atomic-adds into vocab rows, but `grad:wlm` and the four layer dW gradients are
now overwritten by direct-store dW GEMMs (`flags=16521`) whenever
`MK_DW_NO_ATOMIC_SK1` selects the no-atomic sk1 route.

Promoted a builder-side skip for exactly those direct-store dW gradients. The
route decision reuses the existing `MK_DW_NO_ATOMIC_SK1` logic and split-K check;
it does not skip fills for atomic/split-K dW, norm gradients, embedding gradients,
or workspaces. `MK_DW_DIRECT_SKIP_FILL=0` restores the old fill instructions
while keeping the direct-store GEMM routes, and `MK_DW_NO_ATOMIC_SK1=0` still
restores the older split-K/atomic dW rows.

Evidence:
- Route check (`mkv3-p4b-qwen-dwskipfill-ab-20260706T0154Z.log`): qwen default
  drops five dW fills and moves from `n_instr=51` to `46`; old-control
  `MK_DW_DIRECT_SKIP_FILL=0` keeps the same five `flags=16521` direct-store dW
  rows but retains the fills.
- Correctness in the same log: identical loss (`12.54405022` both arms) and
  full gradient comparison worst relative diff `3.129190e-07` (`w1.0`; the
  direct-store dW gradients including `wlm` were exactly equal).
- Paired timing in both construction orders: old-minus-new medians `+523.90us`
  and `+514.19us`, with `48/48` wins in both orders.
- Fresh qwen profile (`mkv3-p4b-qwen-dwskipfill-profile-20260706T0154Z.log`):
  `n_instr=46`, best total `10270.6us`. The remaining off-path fill volume is
  `1146.9us`, dominated by required `grad:emb`; off-path vocab dW remains the
  largest sink span (`GEMMTN 151936x2560x1024.wg`, `5349.4us`), while on-path
  leaders are head-dX (`2932.6us`) and lm-head forward (`2702.2us`).
- Validation: `test_model.py` PASS
  (`mkv3-p4b-dwskipfill-testmodel-20260706T0154Z.log`), `test_ops.py` PASS
  (`mkv3-p4b-dwskipfill-testops-20260706T0154Z.log`), `py_compile`,
  `ruff check`, and `git diff --check` passed.

## v3 P4b qwen n256 3-stage operand ring: exact-route promotion

The qwen4b-l1 default already needs a 148KB dynamic-smem page for the D=128 dQ
row-split route, while the promoted n256 direct GEMMs still used a two-stage
96KB operand page. A full staged 128x256 route remains blocked at 160KB, but a
3-stage operand ring for the existing direct-store n256 bodies fits: each stage
is 16KB of A plus 32KB of B, so three stages are 144KB before the existing 1KB
alignment pad.

Implementation:
- `WgmmaSmemN256T<STAGES>` now backs both n256 direct kernels. The old path
  instantiates `STAGES=2`; the new exact qwen path instantiates `STAGES=3` and
  preissues `STAGES-1` cp.async groups, then waits with the correct shrinking
  tail depth.
- GEMM flag bit25 (`GEMM_N256_STAGE3_FLAG`) selects the 3-stage body under the
  existing bit14 n256 route. It does not change tile counts or epilogues.
- The model builder defaults `self.n256_stage3_enabled` on only for the exact
  qwen4b-l1 shape and requests the 148KB launch page whenever that route is
  enabled. `MK_WGMMA_N256_STAGE3=0` restores the previous two-stage n256 bodies.
  The default affects all 11 qwen n256 rows: no-residual/residual NT forwards,
  lm-head forward with its 128-column tail, head-dX, and the five direct-store
  dW GEMMs.

Evidence:
- Route/default check (`mkv3-p4b-qwen-n256-stage3-route-20260706T0208Z.log`):
  unset default emits all 11 n256 rows with bit25 and `smem=151552`; forced
  `MK_WGMMA_N256_STAGE3=0` restores the old flags (`16514/24722/18562/16520/16521`)
  while keeping the rest of the qwen program unchanged.
- Qwen smoke (`mkv3-p4b-qwen-n256-stage3-smoke2-20260706T0206Z.log`): all 11
  rows carry bit25 under the candidate, loss diff is `-9.53674316e-07`, and
  chunked full-gradient comparison has worst relative diff `3.527165e-07`.
- Paired timing (`mkv3-p4b-qwen-n256-stage3-ab-20260706T0207Z.log`): old
  two-stage vs new 3-stage medians are old-minus-new `+424.24us` and
  `+387.22us` in the two run orders, `32/32` wins each (`64/64` overall).
- Matched profile comparison: forced old
  `mkv3-p4b-qwen-n256-stage3-old-profile-20260706T0217Z.log` measured total
  `10815.7us`; promoted stage3
  `mkv3-p4b-qwen-n256-stage3-profile-20260706T0207Z.log` measured `10434.9us`.
  The on-path lm-head forward span dropped `2756.3us -> 2574.6us`, head-dX
  `3256.6us -> 3157.5us`, and the off-path vocab dW sink
  `5779.2us -> 5107.1us`.

Validation:
- `test_model.py` PASS (`mkv3-p4b-n256-stage3-testmodel-20260706T0208Z.log`).
- `test_ops.py` now has permanent bit25 coverage for NT stage3 with a 128-column
  tail plus fp32 NN/TN stage3 dispatch; PASS
  (`mkv3-p4b-n256-stage3-testops2-20260706T0219Z.log`).
- `py_compile`, `ruff check`, and `git diff --check` passed.

Follow-up scheduler check: post-stage3 `MK_COLD_CAP` resweep
(`mkv3-p4b-qwen-coldcap-post-stage3-20260706T0218Z.log`) still rejects low
cold caps; cap96/cap132 were within noise of cap0. The focused cap0-vs-cap96
A/B (`mkv3-p4b-qwen-coldcap96-post-stage3-ab-20260706T0218Z.log`) measured
cap0-minus-cap96 median `+7.26us` in one order but `-9.07us` in the reverse,
`-3.14us` overall with `47/96` wins. Keep qwen `default_cold_cap=0`.

## v3 P4b cooperative cluster launch enablement

The operator-gap branch proved that `cudaLaunchKernelEx` can combine cluster
dimensions with cooperative grid launch on this driver, and that 132 blocks /
66 two-block clusters remain co-resident with no launch-latency penalty. The
main branch now carries the matching host-side opt-in:

- `MK_CLUSTER_X=2` switches all megakernel executors (`waves`, `df`, `df2`,
  `ws`) from `cudaLaunchCooperativeKernel` to `cudaLaunchKernelEx` with
  `cudaLaunchAttributeClusterDimension` and `cudaLaunchAttributeCooperative`.
- The default remains `MK_CLUSTER_X=1`, preserving the old launch path. The
  current support is intentionally limited to cluster size 2 because that is the
  target for adjacent-M-tile B-multicast and is what the residency probe cleared.

Validation:
- Clustered `test_ops.py` PASS
  (`mkv3-cluster-launch-testops-20260706T0225Z.log`), including the permanent
  n256 stage3 NT-tail, NN, and TN cases.
- Clustered `test_model.py` PASS
  (`mkv3-cluster-launch-testmodel-20260706T0228Z.log`), covering `df`, `waves`,
  `df2`, and `ws` executor modes.
- Fresh qwen profile at HEAD
  (`mkv3-p4b-qwen-current-profile-20260706T0221Z.log`) measured `10284.3us`;
  remaining on-path leaders are still head-dX
  `GEMMNN 1024x2560x151936.wg` at `3122.5us` and lm-head fwd
  `GEMMNT 1024x151936x2560.wg` at `2561.0us`.
- Launch-only qwen A/B
  (`mkv3-p4b-qwen-cluster-launch-ab-20260706T0233Z.log`) was neutral:
  cluster1-minus-cluster2 medians were `-7.73us` and `+1.97us` by order,
  `-2.69us` overall with `31/64` wins. This confirms the launch path itself is
  not a speedup and does not impose a measurable tax.

Next work is the real cluster mechanism: paired adjacent-M tile claiming plus a
GEMM body that uses cluster DSMEM/TMA multicast for the shared B operand. The
existing scheduler still claims one tile at a time from a global cursor, so
cluster launch alone cannot reduce the qwen giant-vocab spans.

## v3 P4b qwen n256 N-major tile order: exact-route promotion

The cluster multicast lane wants adjacent M bands to work on the same 256-column
B tile at the same time. The existing n256 tile order was M-major
(`m_tile * n_tiles + n_tile`), which schedules all columns for one M band before
the next M band. Bit26 now selects an N-major decode for the exact qwen n256
direct routes: `n_tile * m_tiles + m_tile`. This keeps the same tile count,
math, smem page, and epilogues, but groups all M bands for a B tile. It is also
the natural precondition for a later cluster-2 B-multicast body.

Route/control:
- `MK_WGMMA_N256_NMAJOR=0` restores the old M-major tile order.
- Unset default and `MK_WGMMA_N256_NMAJOR=1` set bit26 on all 11 exact qwen
  n256 rows; route log `mkv3-p4b-qwen-n256-nmajor-route-20260706T0242Z.log`.
- df2 region-watermark gating treats bit26 GEMMs as non-row-linear, because the
  old watermark proof depends on M-major producer tile order.

Evidence:
- Old-vs-new qwen smoke
  (`mkv3-p4b-qwen-n256-nmajor-smoke-20260706T0239Z.log`) was clean: loss diff
  `-1.90734863e-06`, worst full-gradient relative diff `4.488093e-07`.
- Paired qwen timing (`mkv3-p4b-qwen-n256-nmajor-ab-20260706T0242Z.log`) won in
  both construction orders: old-minus-new medians `+39.87us` and `+29.97us`,
  overall `+35.36us` with `64/96` wins.
- Profile (`mkv3-p4b-qwen-n256-nmajor-profile-20260706T0242Z.log`) showed the
  intended local movement despite noisy total attribution: lm-head fwd
  `2558.7us -> 2514.7us` against the pre-nmajor HEAD profile, and off-path vocab
  dW `5517.6us -> 5049.1us`.

Validation:
- Expanded `test_ops.py` includes bit26 NT-tail, NN, and TN n256 stage3 cases
  with `M=256` so the remap is actually exercised; PASS
  (`mkv3-p4b-n256-nmajor-testops-20260706T0243Z.log`).
- `test_model.py` PASS (`mkv3-p4b-n256-nmajor-testmodel-20260706T0245Z.log`).

## v3 P4b cluster DSMEM-fed GMMA probe: NO-GO, use TMA multicast

Before wiring paired-M cluster tiles into the interpreter, `dsmem_gmma_probe.py`
tested the cheap alternative to TMA multicast: rank0 stages the shared B tile in
its DSMEM, rank1 maps rank0's shared address with `mapa.shared::cluster`, then a
normal GMMA B descriptor is built from that mapped address. The same probe first
checks a scalar `ld.shared::cluster` so a negative GMMA result is not confused
with a broken cluster launch or DSMEM address.

Evidence (`mkv3-dsmem-gmma-probe-20260706T025558Z.log`, GPU5, private torch
extension cache):
- Scalar DSMEM works: rank0 read rank1 probe value `1001`, rank1 read rank0
  probe value `1000`.
- Rank0 local-B GMMA is correct: max error `1.907349e-06`.
- Rank1 remote-B GMMA is not viable: max error `3.250951e+01`, output abs max
  `0.000000e+00` because rank1's local B slab was intentionally zeroed.
- Address diagnostics explain the failure mode: rank1 local B address was
  `0x1002400`, the mapped rank0 B address was `0x2400`, and the GMMA descriptor
  start field for both local and mapped addresses collapsed to `0x240`. The
  14-bit GMMA shared descriptor cannot carry the cluster-rank part of a DSMEM
  address, so GMMA consumes local shared memory only.

Outcome: do not implement a paired-tile body where one CTA stages B and the peer
CTA points GMMA at that DSMEM. The cluster GEMM route needs TMA
`.multicast::cluster` (or equivalent bulk multicast) to materialize the B tile
into each CTA's local smem before GMMA. DSMEM remains available for scalar
exchange and possible epilogue/reduction protocols, but not as a direct GMMA
operand source.

## v3 P4b cluster TMA multicast + local GMMA probe: PASS

`tma_multicast_gmma_probe.py` validates the viable paired-M cluster primitive.
Rank0 issues one `cp.async.bulk.tensor.2d.shared::cluster.global...multicast::cluster`
load of a 128x64 B tile from a CUtensorMap into the same CTA-relative SW128 B
slab in both CTAs (`ctaMask=0x3`). Each CTA pre-arms its local mbarrier with
`mbarrier.arrive.expect_tx`, waits locally, then runs an ordinary local GMMA B
descriptor. Rank1's local B slab is zeroed before the multicast, so rank1
correctness proves the multicast populated local smem; no remote-GMMA descriptor
is involved.

Evidence (`mkv3-tma-multicast-gmma-probe-20260706T030112Z.log`, GPU5, private
torch extension cache):
- Both destination barriers completed: rank0 `wait_ok=1`, rank1 `wait_ok=1`.
- Rank0 and rank1 both matched torch with max error `3.814697e-06`.
- The first B bf16 bits matched on both CTAs (`0xbf22`), while rank1's local B
  address carried the cluster-rank tag (`0x1002800`). Both GMMA descriptor start
  fields were local (`0x280`) because TMA materialized B into each CTA's own
  smem.
- Launch-plus-probe timing was `6.37us` for this tiny two-CTA harness; treat it
  as a sanity datapoint, not a model-speed estimate.

Outcome: the next production path should pair adjacent M tiles inside a cluster
and have the elected CTA issue the B TMA multicast for the pair, with each CTA
keeping its own A load and local GMMA. This is the concrete mechanism that the
bit26 N-major tile order prepared for.

## v3 P4b production-shaped n256 pair TMA probe: NO-GO for current body

`n256_pair_tma_probe.py` moved beyond the minimal primitive and ran a full grid
of 2-CTA clusters, each cluster computing adjacent 128-row M bands for one
256-column B tile. It compares:
- `cpasync`: current-style duplicated per-CTA B `cp.async` loads.
- `tma-sync`: rank0 TMA multicast with a conservative cluster sync before each
  TMA issue.
- `tma-nosync`: rank0 TMA multicast after local mbarrier arming, relying on
  paired-CTA lockstep and skipping the per-stage cluster sync.

Evidence:
- Smoke shape (`mkv3-n256-pair-tma-nosync-smoke-20260706T030736Z.log`):
  all three variants parity-clean (`max_abs=1.907349e-05`); K512 timings were
  cpasync `10.981us`, tma-sync `18.746us`, tma-nosync `12.492us`.
- Qwen-shaped body (`mkv3-n256-pair-tma-nosync-qwen-20260706T030736Z.log`):
  full 66 clusters, M=16896, N=256, K=2560, 20 reps. cpasync `44.070us`
  (502.5 TF), tma-sync `80.608us`, tma-nosync `46.722us`.
- Reversed-order qwen check
  (`mkv3-n256-pair-tma-nosync-qwen-rev-20260706T030736Z.log`) confirmed the
  no-go: tma-nosync medians `47.451us` and `47.868us`; cpasync medians
  `44.164us` and `44.382us`.

Outcome: do not spend the scheduler/ABI complexity to integrate cluster-paired
TMA multicast into the current n256 direct body. The primitive is correct, but
without a decoupled producer warp or a different ring protocol, the TMA path does
not beat the existing per-CTA SW128 cp.async body at the qwen K=2560 body shape.
The bit26 N-major order remains a small standalone win and keeps the option open,
but the next megakernel work should pivot away from n256 B-multicast plumbing.

## v3 P4b qwen n256 scheduler follow-ups: NO-GO

After rejecting paired TMA, two source-free follow-up probes checked whether the
current n256 routes wanted a smaller scheduler adjustment instead of another body.

1. Selective head-dX M-major: a monkeypatch cleared bit26 only for qwen head-dX
   `(M,N,K)=(1024,2560,151936)` while leaving all other exact n256 rows N-major.
   Parity stayed clean, but timing was order-mixed
   (`mkv3-p4b-qwen-nmajor-headdx-mask-20260706T031350Z.log`): default-minus-variant
   `+11.52us` in default-first order with weak `14/32` variant wins, then
   `-68.35us` in variant-first order with only `7/32` variant wins. Keep head-dX
   on the same default N-major policy as the rest of the qwen n256 rows.
2. Global claim-batch retune: `MK_CLAIM=64/32/16` tried to raise the 80-tile
   n256 rows from one-tile claims to 2/3/5 tile claims. This is decisively
   wrong for qwen (`mkv3-p4b-qwen-claim-sweep-20260706T031422Z.log`):
   claim64 regressed by `3359.17us` and `3596.67us`, claim32 by `9012.93us`
   and `8998.40us`, and claim16 by `17797.44us` and `17584.43us` in the two
   construction orders, all with `0` variant wins. Keep the default `MK_CLAIM`
   behavior.

Outcome: no n256 scheduler-only tweak was promoted. The remaining profitable
qwen work came from support memory traffic, not wider tile claims.

## v3 P4b qwen sparse embedding-gradient clear: exact-regime promotion

The post-n256 qwen profile still showed a large support fill bucket. Five
direct-store dW fills were already elided, but `grad:emb` was still fully
zero-filled every step (`151936 x 2560` fp32 elements) even though embedding
backward only atomically adds into rows touched by the current token batch.

Implementation:
- Added `OP_EMBED_ZERO_ROWS`: row-tiled sparse clear of both `prev_tokens[t]`
  and `tokens[t]` embedding-gradient rows before `OP_EMBED_BWD`.
- Added `OP_COPY_I32`: copies current tokens into a persistent `prev_tokens`
  buffer at the end of the step.
- The invariant is: after a step, only current-token rows may be nonzero. The
  next step clears previous-token rows and current-token rows before atomic
  accumulation. Duplicate tokens and previous/current overlap are benign zero
  races.
- Default gate is conservative and matches the qwen-class giant single-layer
  regime: `L==1`, `H>=1024`, `V>=32768`. `MK_EMB_SPARSE_ZERO=0` restores the
  old full embedding-gradient fill; `=1` force-enables the sparse clear for
  validation/probing.

Evidence:
- Two-step changing-token parity with sparse forced passed all executors
  (`mkv3-emb-sparse-two-step-parity-20260706T031614Z.log`): `df`, `waves`,
  `df2`, and `ws` all kept loss diffs within `1.91e-06`; worst selected grad
  rel stayed below `5.88e-03`; nonzero embedding-gradient row counts matched.
- Initial qwen env A/B (`mkv3-p4b-qwen-emb-sparse-ab-20260706T031837Z.log`)
  changed qwen from `n_instr=46`, `FILL_F32=10` to `n_instr=47`, `FILL_F32=9`,
  `EMBED_ZERO_ROWS=1`, `COPY_I32=1`; parity was clean across two token batches
  and timing won both orders: default-minus-sparse `+234.46us` and `+254.32us`,
  `32/32` sparse wins in both.
- Promoted-default vs forced old (`mkv3-p4b-qwen-emb-sparse-default-ab-20260706T032116Z.log`):
  unset default emits sparse clear, `MK_EMB_SPARSE_ZERO=0` restores old full fill.
  Old-minus-default was `+259.01us` and `+234.91us`, with `32/32` default wins in
  both orders; selected grad rel stayed below `3.37e-07` and embedding nonzero row
  counts matched (`1021/1021`).
- Fresh qwen profile (`mkv3-p4b-qwen-emb-sparse-profile-20260706T032129Z.log`)
  measured `10159.6us` best total. The old off-path `FILL_F32` bucket no longer
  appears in the top eight; `EMBED_ZERO_ROWS` appears as a small `42.2us` off-path
  support op.

Validation:
- `test_ops.py` PASS (`mkv3-emb-sparse-testops-20260706T032141Z.log`).
- `MK_EMB_SPARSE_ZERO=1 test_model.py` PASS
  (`mkv3-emb-sparse-testmodel-20260706T032400Z.log`), covering rerun stability,
  `waves`, `df2`, `ws`, and SGD sanity with the sparse path forced on non-qwen
  shapes.
- `py_compile`, `ruff check`, and `git diff --check` passed.

Qwen H2560 RMS dX R4 recheck: NO-GO. After sparse embedding clearing removed
the support-fill distraction, the qwen profile still showed three generic
`RMSNORM_BWD_DX` hops at ~484us total on path, so an env-only
`MK_RMS_DX_R4=1` sweep tested the existing four-row fold outside its H256
default gate. Log `mkv3-p4b-qwen-rmsdx-r4-20260706T032840Z.log` changed the
three RMS dX instructions from `OP_RMSNORM_BWD_DX` with 192 total tiles to
`OP_RMSNORM_BWD_DX_R4` with 96 total tiles and kept parity clean over two
changing-token steps: loss diffs were `-9.54e-07` and `+1.91e-06`, worst
selected-gradient rel was `3.73e-07`, and embedding nonzero row counts matched
(`1018/1018`). Timing rejected the fold in both construction orders:
default-minus-R4 median deltas were `-22.70us` and `-25.10us`, with only `5/32`
R4 wins in each order. Keep qwen H2560 on the existing two-row RMS dX body;
`MK_RMS_DX_R4` remains a sweep override, not a qwen default.

Qwen combined RMS backward recheck: NO-GO. Source-free
`MK_RMS_BWD_SPLIT_DW=0` collapsed the three split `RMSNORM_BWD_DX` plus three
`RMSNORM_BWD_DW` instructions back into three combined `RMSNORM_BWD` ops
(`n_instr=47 -> 44`, region-gated `14 -> 11`) without changing tile count.
Log `mkv3-p4b-qwen-rms-combined-ab-20260706T033617Z.log` kept two-step parity
clean: loss diffs `0` and `-2.86e-06`, worst selected-gradient rel
`3.21e-07`, and embedding nonzero row counts matched (`1021/1021`). Timing was
construction-order biased, not a win: default-minus-combined was `+11.04us`
median / `+14.72us` paired in the default-first order, but `-50.11us` median /
`-31.36us` paired in the reverse order. Keep the current split RMS backward;
the cold dW drain still belongs off the dX path.

Qwen CE forward/backward fusion probe: NO-GO, source reverted. A temporary
`OP_CE_FWD_BWD` combined the lm-head partial-lse reduction/loss update and the
in-place dlogits pass, replacing `CE_FWD` + `CE_BWD` with one row op and one
fewer wave. Unit coverage passed, including the partial-lse form used by qwen
(`mkv3-cefwdbwd-testops-20260706T034126Z.log`). Qwen two-step parity was clean:
route changed from `CE_FWD=1`, `CE_BWD=1`, `n_instr=47`, `critical_path=26` to
`CE_FWD_BWD=1`, `n_instr=46`, `critical_path=25`; loss diffs were
`+1.91e-06` and `+2.86e-06`, worst selected-gradient rel was `4.88e-07`, and
embedding nonzero row counts matched (`1018/1018`). Timing rejected promotion:
default-minus-fused was `-109.87us` median / `-104.10us` paired in the
default-first order, then only `+29.12us` median / `+29.62us` paired in reverse
(`mkv3-p4b-qwen-cefwdbwd-ab-20260706T034413Z.log`). Profile
`mkv3-p4b-qwen-cefwdbwd-profile-20260706T034640Z.log` explained the weak result:
the fused op's span was `327.1us`, barely below split `CE_FWD+CE_BWD`
(`34.7us + 296.4us`) while whole-step attribution stayed dominated by the
giant n256 head/lm-head GEMMs. Do not carry a fused CE opcode unless it also
changes the downstream head/dW dataflow, not just the local CE hop.

Qwen D=128 WGMMA attention claim1: PROMOTED. The old absorption-ledger note
said `MK_ATTN_D128_CLAIM1=1` was an order-mixed wash before the sparse-embed
and current qwen defaults. A source-free recheck on the current qwen4b-l1 shape
(`Cfg(H=2560,L=1,nq=32,nkv=8,D=128,I=9728,V=151936,S=1024)`) changed only the
claim tensor for `ATTN_FWD_WG128`, `ATTN_DKV_WG128`, and `ATTN_DQ_WG128`:
default claims were `2/4/2`, claim1 claims were `1/1/1`, with identical
`n_instr=47`, `critical_path=26`, `gated=14`, and `waves=26`. Two-step parity
was clean for a scheduler-only atomic-order change: loss diffs were
`-2.86e-06` and `-1.91e-06` across the two construction orders, worst selected
grad rel was `1.33e-03` on sparse `grad:emb` rows, and embedding nonzero row
counts matched (`1019/1019`). Timing cleared both order checks:
default-minus-claim1 was `+37.30us` median / `+50.56us` paired with `31/32`
claim1 wins, then `+35.07us` median / `+30.74us` paired with `30/32` wins in
reverse (`mkv3-p4b-qwen-d128claim1-ab-20260706T035004Z.log`). Default now uses
claim1 for D=128 WGMMA attention rows; set `MK_ATTN_D128_CLAIM1=0` to restore
the previous ntiles/132 batching for A/B. Post-promotion validation:
`py_compile`, `ruff`, and `git diff --check` passed; `test_model.py` passed
nano plus D128-ragged correctness/executor/SGD coverage
(`mkv3-d128claim1-testmodel-20260706T035329Z.log`). Patched-default qwen A/B
against forced old kept the win: old-minus-default was `+36.46us` median /
`+34.05us` paired with `26/32` default wins, then `+33.90us` median /
`+38.42us` paired with `26/32` wins in reverse
(`mkv3-p4b-qwen-d128claim1-default-ab-20260706T035810Z.log`).

Qwen D=128 backward DQ-before-DKV emission order: NO-GO, source reverted. The
post-claim1 profile (`mkv3-p4b-qwen-d128claim1-profile-20260706T035905Z.log`)
showed `ATTN_DQ_WG128` on path mostly as wait (`327.0us` wait + `110.1us`
span) while `ATTN_DKV_WG128` was off path at `372.4us`, so a temporary
`MK_ATTN_D128_BWD_DQ_FIRST=1` probe emitted the dQ instruction before dKV
inside the same backward wave. Route/parity were clean: `n_instr=47`,
`critical_path=26`, `gated=14`, `waves=26`, dQ/dKV instruction indices swapped,
loss diffs stayed within `2.86e-06`, and worst selected-gradient rel was
`2.89e-03` on sparse `grad:emb`. Timing did not justify promotion:
default-minus-dqfirst was `-0.86us` median / `+1.70us` paired with `17/32`
dq-first wins, then only `+9.55us` median / `+7.04us` paired with `20/32` wins
in reverse (`mkv3-p4b-qwen-d128-dqfirst-ab-20260706T035952Z.log`). Keep the
current dKV-then-dQ emission; the apparent dQ wait is mostly balanced against
the dKV completion requirement before qk-norm backward.

Qwen post-claim1 all-hot scheduler recheck: NO-GO. The fresh qwen profile still
has a giant off-path vocab dW sink (`GEMMTN 151936x2560x1024.wg`, `5135.6us`
span), so a source-free `MK_ALLHOT=1` A/B checked whether the hot/cold split is
still load-bearing after D128 claim1. Route/parity were clean: default had
`hot=31/cold=16`, all-hot had `hot=47/cold=0`, D128 attention claims stayed
`1/1/1`, loss diffs were `0` and `-1.91e-06`, and worst selected-gradient rel
was `1.37e-03` on sparse `grad:emb`. Timing still rejected all-hot:
default-minus-allhot was `-18.05us` median / `-10.03us` paired with only
`9/32` all-hot wins, then `-16.10us` median / `-14.19us` paired with `8/32`
wins in reverse (`mkv3-p4b-qwen-allhot-postclaim1-ab-20260706T040318Z.log`).
Keep hot/cold criticality; the remaining qwen gap is structural n256/rowop work,
not ready-ring policy.

Qwen SwiGLU two-warp backward: PROMOTED; sigmoid cache rejected for qwen. The
post-claim1 profile showed qwen's one-warp `SWIGLU_BWD` still on path at
`5.8us` wait plus `227.9us` span
(`mkv3-p4b-qwen-d128claim1-profile-20260706T035905Z.log`), so a source-free
`MK_SWIGLU_CACHE_SIG=1 MK_SWIGLU_BWD_2W=1` A/B first checked whether the
existing H256/H512 combined route transfers to the H2560 qwen MLP. Route/parity
were clean: the forced route kept `n_instr=47`, `critical_path=26`, `gated=14`,
`waves=26`, changed `swsig=False`/`SWIGLU_BWD=1/128` to
`swsig=True`/`SWIGLU_BWD_2W=1/256`, kept D128 attention claims at `1/1/1`, and
kept loss diffs within `2.86e-06` with worst selected-gradient rel
`6.29e-03` on sparse `grad:emb`. Timing won both construction orders:
default-minus-forced-new was `+302.45us` median / `+305.85us` paired with
`32/32` wins, then `+314.83us` median / `+300.90us` paired with `32/32` wins
in reverse (`mkv3-p4b-qwen-swiglu-cache2w-ab-20260706T040611Z.log`).

Follow-up decomposition found the qwen win comes from two-warp backward, not
from storing the sigmoid cache. Against the cached+2W default,
`MK_SWIGLU_CACHE_SIG=0 MK_SWIGLU_BWD_2W=1` kept the same `SWIGLU_BWD_2W=1/256`
route but removed `swsig`; parity stayed clean (loss diffs `+9.54e-07` and
`0`, worst selected-gradient rel `6.25e-03` on `grad:emb`) and 2W-only won by
`+95.54us` and `+94.21us` medians with `31/32` and `29/32` wins
(`mkv3-p4b-qwen-swiglu-2wonly-ab-20260706T041818Z.log`). Cache-only
(`MK_SWIGLU_CACHE_SIG=1 MK_SWIGLU_BWD_2W=0`) lost by `251.36us` and `280.80us`
medians with `0/32` wins in both construction orders
(`mkv3-p4b-qwen-swiglu-cacheonly-ab-20260706T041818Z.log`).

The final production gate is exact qwen4b-l1
`(H,S,I,V,nq,nkv,D,L)=(2560,1024,9728,151936,32,8,128,1)` via
`_QWEN_L1_SWIGLU_BWD_2W`: qwen defaults to `MK_SWIGLU_BWD_2W=1` while leaving
`MK_SWIGLU_CACHE_SIG=0`. `MK_SWIGLU_BWD_2W=0` restores the old one-warp route;
`MK_SWIGLU_CACHE_SIG=1` remains an A/B-only override for qwen. Final
patched-default vs forced-old timing showed the intended default route
(`swsig=False`, `SWIGLU_BWD_2W=1/256`) and kept parity clean (loss diffs
`-9.54e-07` and `+1.91e-06`, worst selected-gradient rel `3.14e-03` on
`grad:emb`). Default beat old by `399.79us` and `386.42us` medians with
forced-old winning `0/32` in both orders
(`mkv3-p4b-qwen-swiglu-2wonly-default-ab-20260706T042404Z.log`). Validation for
the final source path: `py_compile`, `ruff`, `git diff --check`, and
`test_model.py` passed (`mkv3-p4b-qwen-swiglu-2wonly-testmodel-20260706T042404Z.log`).
Fresh qwen profile after the final 2W-only gate measured total `9735.6us`, with
`SWIGLU_BWD_2W` at `5.2us` wait plus `120.1us` span. The remaining qwen leaders
are still giant n256 head-dX/lm-head rows, MLP GEMMs, D128 dQ wait, and
qk-norm/CE rowops (`mkv3-p4b-qwen-swiglu-2wonly-profile-20260706T042404Z.log`).

Qwen head-dX n256 split-K probe: NO-GO, source reverted. The top current qwen
hop is still the 80-tile n256 direct `dlogits @ Wlm` row
(`GEMMNN 1024x2560x151936.wg`), so a temporary default-off
`MK_HEAD_DX_N256_SPLIT=1` source probe taught the n256 fp32 body to split K and
atomically accumulate into the existing fp32 `dXN` workspace. Smoke log
`mkv3-p4b-qwen-headdx-n256split-smoke-20260706T033134Z.log` changed head-dX
from `ntiles=80, flags=100679816` to `ntiles=160, flags=100679848, sk=2` and
added one `FILL_F32` instruction. Two-step selected-gradient parity was clean
within the usual atomic-order tolerance: loss diffs were `+1.91e-06` on both
steps, worst selected grad rel was `5.78e-03`, and embedding nonzero row counts
matched (`1021/1021`). Timing did not clear the bar: the first paired check was
order-mixed (`+1.55us` median but `-10.88us` paired for sk2, then `+35.09us`
median and `+17.73us` paired in reverse). The target sweep
`mkv3-p4b-qwen-headdx-n256split-targets-20260706T033447Z.log` rejected sk2/sk3/sk4:
default-minus-split was negative in both sweep halves for every target
(`-34.90us/-16.67us` for sk2, `-60.80us/-31.66us` for sk3, and
`-78.19us/-29.44us` for sk4). The extra fill, wave, and fp32 atomics eat the
shorter-K tile benefit. Do not add n256 split-K support to the production body
without a different accumulation strategy.

## v3 P4b D=128 dQ register-A feed: NO-GO in-model

The operator-gap standalone `attn_dq_d128_rf` result was ported narrowly onto
the promoted qwen row-split dQ route as an opt-in test: dS was packed directly
from the C-fragment layout into 16 bf16x2 A-registers for an RS WGMMA, bypassing
the dS smem store, `fence.proxy.async`, WG sync, and SS A-smem read. The test
kept the current non-mbarrier row-split scheduling and qwen's 148KB launch
carveout.

Evidence:
- Smoke/control log `mkv3-p4b-d128-dqrf-smoke-20260706T0156Z.log`: forced
  control route emitted dQ `[(256,16777217)]`; forced RS-feed emitted
  `[(256,83886081)]`. Qwen one-step parity was clean: loss diff `-1.9e-06`,
  worst full-gradient relative diff `4.632988e-07`.
- Paired timing `mkv3-p4b-d128-dqrf-ab-20260706T0156Z.log` rejected the route:
  old-minus-new medians were `-6.40us` and `-14.08us`, with weak `30/64` and
  `23/64` wins. The standalone per-stage work removal is too small after the
  row-split/direct-store dQ route and is absorbed by the composed qwen step.

Outcome: candidate source was reverted; do not carry `MK_ATTN_D128_DQ_RF` in
production. If this mechanism is revisited, it should be as part of the larger
S^T/full-TMA D=128 bwd rewrite, not another narrow bolt-on to the current dQ
row-split body.

## v3 P4b fwd KV-widening in-model (session 2853e0de): NO-GO — absorption

The FA4-B w128 fwd port (OP_ATTN_FWD_WGW128, `megakernel-attn-w128` worktree,
default-off) is parity-consistent but its +11-19% STANDALONE win does not
survive in-model composition anywhere
(`mkv3-p4b-w128-{smoke,unbanded-sweep}-*.log`):
- Banded long shapes (S4096): +18.7/+17.4us (2/40 both orders). Mechanism: at
  banded shapes only the C==1 bands (the SHORT, off-path tail tiles) can take
  the plain w128 op — no realized path gain, while the 120KB carveout + fatter
  code shape cost a little everywhere.
- Unbanded shapes: S512 -6.5/-1.9us (33/40 then 25/40 — order-decaying, under
  the bar), S1024 +3.3/+7.7. The fwd bucket at short S is co-scheduled and
  partially off-path: op-local boundary savings get absorbed (the absorption
  ledger's 5th confirmation).
- Numerics note: the single 128-wide online-softmax pass gives a DIFFERENT fp
  path than two 64-wide rescales — in-model worst grad rel vs w64 is ~0.021
  (kn), and a nano-forced run overshoots the test_model fp32-reference bar
  marginally (kn.3 0.0306 vs 0.03) — same kn-sensitivity class as the fwd-band
  nano-T4 note. Any future gate must respect this.
The one theoretically-aligned continuation (unclaimed, with caveats): port the
w128 body into the banded C>1 PARTIALS path so the on-path straggler chunks
get the halved boundary count (chunk stage counts halve; band T values would
need re-tuning). Given five absorption strikes, treat as low-prior. The op
stays default-off as a building block; the standalone win in the opgap ledger
stands — this entry records only its in-model fate.

## v3 P4b rowop long-S gap decomposition (session 2853e0de): bodies are FINE

The last undecomposed on-path bucket is settled. In-model, the rowop family
reads 540/740/1330us on-path at S3072/4096/8192 vs the baseline's triton-fused
170-337us (opgap baseline CSVs) — a deficit that GROWS to ~1ms at S8192. The
standalone probe (`results/rowop_scaling_probe_2853e0de.py <S>`, single-instr
programs, GPU 6, `mkv3-p4b-rowop-scaling-v2-20260706T033903Z.log`) shows
the OP BODIES ARE AT BASELINE PARITY — the deficit is entirely critical-path
span stretch:

| S | rms_dx | rms_dw | qknorm_bwd | swiglu_bwd_2w | (launch floor) |
|---|---:|---:|---:|---:|---:|
| 2048 | 19.3 | 19.4 | 32.5 | 20.9 | 16.8 |
| 4096 | 22.6 | 21.9 | 48.4 | 24.3 | 16.6 |
| 8192 | 26.2 | 27.1 | 107.3 | 40.0 | 16.6 |

Launch-floor-subtracted per-model-pass op cost at S4096: rms_dx ~54 (9 instrs)
+ qknorm ~127 (4) + swiglu ~31 (4) + rms_dw ~48 (9) ≈ 260us vs the baseline's
254us — PARITY. The in-model per-instance span (e.g. rms_dx 29.1us vs ~6us
op-only) is tile spread across a window where co-scheduled gemm/attention work
holds the SMs: the rowops are serially chained between gemms, so their
realized span absorbs the surrounding contention. Exception: QKNORM_ROPE_BWD
at S8192 is genuinely op-body-bound (~91us/instr standalone == its in-model
span; it moves ~24MB/instr of L2-resident fp32 workspace — bandwidth-class,
little headroom).

CONSEQUENCE for the roadmap: there is NO rowop op-quality lane. The only lever
that moves this bucket is the v2-roadmap STRUCTURAL item — fusing rowops into
the producing gemm epilogues (the ssq/bit13 fusion already killed the variance
pass; the dx-side and swiglu-side fusions are unexplored). That is a major
multi-session arc: each fused epilogue deletes a chain hop AND removes the
rowop's exposure to co-scheduling stretch. Flagged unclaimed on the board.

## v3 P4b dispatch-spill regression caught by certification (session 2853e0de)

The full certified scoreboard at the 0345Z-class head read UNIFORMLY high vs
the freshest per-shape measurements (`mkv3-p4b-score-full-1cb68c8-20260706T034649Z.log`):

| shape | mk | graph+ | gap | vs fresh expectation |
|---|---:|---:|---:|---|
| nano | 939.0 | 632.4 | 1.48x | +22us |
| small | 3660.2 | 1904.1 | 1.92x | +200us |
| deep-L12 | 2448.1 | 1774.4 | 1.38x | +48us |
| S2048 | 1899.7 | 983.1 | 1.93x | +74us |
| S3072 | 2651.2 | 1330.2 | 1.99x | +150us |
| S4096 | 3362.7 | 1581.1 | 2.13x | +230us |
| S8192 | 7744.0 | 3153.0 | 2.46x | +624us (+8.8%) |

Root cause found by compile-only STACK bisect
(`mkv3-p4b-stack-bisect-20260706T040413Z.log`): `megakernel_df` STACK
went 48 (my d128 merge) -> 32 (the rowbcast/rms-h256/drow commits IMPROVED it)
-> **208 at `544640f` (qwen n256 stage3 ring)** and 176-208 since. Classic P1
dispatch-spill law: ~176B of spill at the dispatch call sites taxes every op
on every shape, biggest absolute cost on the longest steps. Fix recipe per P6
round 2: __noinline__ the fat body or re-stage its hoisted mainloop state;
verify STACK <= 48 and re-run the stage3 A/B (its measured margin partly paid
this tax). Handed to the owning session on the board with a 45-min window.
META: this is why the periodic full certification exists — per-lane paired
A/Bs cannot see a uniform tax that lands between their two arms.

### CORRECTION (20260706T0545Z, same session): STACK is not the mechanism

Runtime bisect + perturbation probes (all median-of-50 small on clean GPU 6,
graph+ control stable 1899-1910us in every run) revise the paragraph above:

| binary | megakernel_df STACK | small us |
|---|---:|---:|
| 06f3101 (pre-window) | 32 | 3472.0 |
| 06f3101 + fat NEVER-EXECUTED n128 clone | 32 | 3458.9 |
| c590974 (mid-window, pre-stage3) | 64 | 3548.1 |
| 544640f (stage3 ring) | 208 | 3657.8 |
| 1cb68c8 (cert head) | 176 | 3662.4 |
| 1cb68c8 + ENTIRE n256 family compiled out | 80 | 3735.6 |

- The regression is CUMULATIVE across the ungated n256 device-code commits
  (+76 by c590974 before stage3 exists, +110 more by 544640f), not a stage3
  cliff. Route at small is bit-identical across the window (invariant-field
  instr diff: only a +1 buffer-table index shift) — python emission exonerated.
- Dead code volume is FREE (row 2: +173 dead lines, STACK 32, runtime equal).
- Removing all n256 code did NOT recover and made it worse (row 6): the probe's
  `__trap()` (noreturn) sat in the hot gemm dispatch CFG, and other window code
  keeps STACK at 80 — so STACK/res-usage CANNOT certify a fix; the n256off
  binary has near-clean stack and the worst runtime.
- Surviving mechanism: the shape of the HOT GEMM DISPATCH caller codegen
  (flag-decode + four template call sites hoisted into it by the n256 commits).
  Fix probe in flight: one `__noinline__` n256 trampoline so the caller regains
  a single call site; runtime-certified only. Logs:
  `mkv3-p4b-small-at-{544640f,c590974}-*.log`,
  `mkv3-p4b-small-n256off-1cb68c8-*.log`,
  `mkv3-p4b-small-lottery-06f3101-*.log`,
  `mkv3-p4b-small-regression-control-20260706T044823Z.log`.

RESOLUTION (20260706T0620Z): ncu SourceCounters on a `-lineinfo` head build
attributed ~15M of 16M executed local sectors to `megakernel.cu:351-424` — the
scheduler claim loop spilling around the inlined dispatch switch. The n256
fanout was exonerated twice (a `__noinline__` trampoline was runtime-neutral
both above and below the pressure cliff). The dominant term was the five D=128
WGMMA attention ops (`op_attn_fwd_wg128{,_mbar}`, `op_attn_dkv_wg128`,
`op_attn_dq_wg128{,_rowsplit}`) inlining fat accumulator frames into the
dispatch switch against the n128 precedent. `__noinline__` on all five:
small **3660.4 -> 3557.7us** (STACK 176 -> 128), qwen4b-l1 neutral
(9890.0 vs 9955.1 unpatched, three-arm within p10-p90 band;
`mkv3-p4b-small-d128noinline-9930a29-*.log`,
`mkv3-p4b-qwen-d128noinline-ab-*.log`). A residual ~85us of window tax at
small (3557.7 vs 3472.0 pre-window) remains unattributed — candidates: the
remaining inline dispatch additions (nt-bf16 n256 routing block, sparse
embed-clear) — re-run knob: the same ncu one-pass
(`l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`, target 0) after any
candidate fix. LAW (supplements P1/P6): fat op bodies MUST be `__noinline__`
before entering the dispatch switch; certify with executed local-load sectors
+ runtime, never STACK/res-usage alone.

### PROMOTED (20260706T0635Z): D=64 wg attention trio noinline — spill zeroed

`__noinline__` on `op_attn_fwd_wg` / `op_attn_dkv_wg` / `op_attn_dq_wg`
(stacked on 6b78314) zeroes executed local-LD at small (4.09M -> 0 sectors)
and recovers the remaining tax: small 3557.7 -> **3511.1us** (~39us from the
3472.0 pre-window anchor, inside the cross-binary noise band). These ops
pre-date the regression window and were fine under the old pressure baseline;
the window's additions raised global allocation and ptxas began spilling
scheduler state around them — noinline fixes the interaction, not the ops.
Full certified scoreboard vs the 1cb68c8 certification (all guards clean,
S8192 pre-guard read a transient 80% so its gain is understated;
`mkv3-p4b-score-full-d64noinline-*.log`): nano 927.5 (-11.5), small 3510.2
(-150), deep-L12 2422.8 (-25), S2048 1830.8 (-69), S3072 2539.7 (-112),
S4096 3196.3 (-166), S8192 7241.8 (-502). Exonerated en route: 53f8c99
sparse embed-clear (removal is WORSE: 3618.0/LD 4.57M), a354e8f cluster
launch (identical), n256 fanout trampoline (neutral twice). test_ops +
test_model green (`mkv3-d64noinline-tests-*.log`). Re-run knob: the ncu
one-pass LD metric; any future kernel-touching promotion should keep
`l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum == 0` at small.


## Epilogue-fusion phase 1 (MLP rmsnorm dissolution) NO-GO (session 2853e0de, 20260706T0740Z)

Implemented per `results/epilogue-fusion-spec-2853e0de.md` (bit15 rstd
row-scale n128 epilogue, rstd-only RMSNORM_FWD mode via args[9], raw-x2 +
host-folded diag(w2)@wgu consumer, env `MK_EPIFUSE_MLP`, worktree wt-epifuse,
default off). Route: n_instr 288->296, **critical_path 144->144** — the
rstd-only op replaces the rmsnorm at the SAME hop count (both feed off the
wo-gemm bit13 partials), so only span could win, and the rowop span was
absorbed (7th absorption-ledger strike). Paired A/B medians-of-50 both orders
(`mkv3-p4b-epifuse-small-parity-ab-20260706T064529Z.log`): small
3565.6/3567.0 vs 3567.1/3565.0 (wash); S4096 3236.9/3248.6 vs 3255.6/3250.6
(+10-19 WORSE); S8192 7359.4/7361.1 vs 7377.9/7380.3 (+17-19 WORSE). Parity:
loss diff 5e-5, grad absmax rel ~1.2e-4 (bf16 weight-fold rounding — would
need a fp32-fold variant at promotion). LESSON: hop REPLACEMENT is not hop
DELETION — the fusion only pays if the consumer's rstd dep lands with the
PRODUCER (rstd finalized inside the wo-gemm epilogue, needs cross-tile
completion signaling) or if the bwd-side xn2 buffer deletion (phase 2,
different mechanism: memory traffic) carries it. Re-run knob:
`MK_EPIFUSE_MLP=1` in wt-epifuse @c9d77d5.

## Long-S head block routed through n256 direct (session 2853e0de, 20260706T0945Z)

Fresh S8192 profile at 3c08fc3 (`mkv3-p4b-profile-s8192-3c08fc3-*.log`) showed
the head block on-path: lm-head fwd GEMMNT 8192x8192x256 at 442.5us span
(78 TF/s effective) + head-dX 156.6us — the qwen n256 direct-store routes were
exact-gated to qwen shapes and never fired on the gauntlet. Extended
`wgmma_n256_direct_ok` / `wgmma_n256_head_dx_ok` exact sets with the
(3072|4096|8192, 8192, 256) head triples (and dX companions). Paired A/B both
orders, patched-default vs forced-old
(`mkv3-p4b-n256head-promo-20260706T093701Z.log`): S3072 2439.4/2452.4 vs
2497.2/2498.7; S4096 3084.4/3102.6 vs 3138.0/3144.3; **S8192 6853.3/6858.7 vs
7100.6/7101.5 (-245us)**. Losses bit-identical per shape. Gate excludes
small/S2048 (triage regressions +99/+66) and leaves nano/deep neutral —
verified unchanged (929.2/3440.3;
`mkv3-p4b-n256head-gauntlet-20260706T092847Z.log`). test_ops + test_model
green. Arms compose additively (lm -165, dx -82 at S8192). Next structural
S8192 target per the same profile: ATTN_DKV_WG on-path 2565us at 80% wait.

## bwd-attention S/dP one-batch + fused ALU pass (session 2853e0de, 20260706T1010Z)

Follow-through on the scheduling-exoneration diagnosis (packing 77-83%; the
gap is per-stage op cost). In both `op_attn_dkv_wg` and `op_attn_dq_wg`, the
S=QK^T and dP=dO V^T gemms are independent but were issued as two fully
drained `wga_mma64` batches, serialized only by sharing one fp32 bank (the old
register diet). Change: `wga_mma64_x2` batches both in ONE warpgroup commit
(s+s2 banks: dkv 96->128, dq 64->96 live), and with both banks live the
softmax + dS ALU fuse into one pass — dkv drops the P smem re-read, dq drops
the P park/re-read entirely (dS written once, P never materialized;
`bf2f(f2bf(p))` preserves the exact old rounding). Deletes one full drain +
one smem round-trip per 64-row stage from the serial chain. Validation:
test_ops + test_model green, dqkv wgmma error unchanged (7.535e-03), losses
bit-identical every shape, REG:255 STACK:32 LOCAL:0, executed local-LD = 0
(the spill gate). Gauntlet patched-vs-control
(`mkv3-p4b-dqdkv-sdp-batch-gauntlet-20260706T095454Z.log`): nano -5.6,
small -9.9, S2048 -2.9, S3072 -13.2, S4096 -20.0, **S8192 -113.1**; reverse
order S8192 -93.8 (`...-seal-*.log`); dkv-only intermediate was -40/-49
(`mkv3-p4b-dkv-sdp-batch-*.log`), dq added the rest. Next slope-changer for
the remaining bwd-attention gap (~180 TF/s effective vs flash ~500):
cross-stage pipelining (issue stage t+1's S/dP batch during stage t's
accumulation drain) — needs P/dS smem doubling and commit-group bookkeeping;
scoped, unclaimed.

## D=128 dQ rowsplit register-A dS feed (this session, 20260706T1155Z)

Ported the FA4 RS-feed handoff into `op_attn_dq_wg128_rowsplit`: dS is now
packed from the computed C-fragment values directly into `MMA_64x64x16...
_RS<K,MN>` A registers, deleting the dS smem tile, proxy fence, WG sync, and
smem-A reads before `dQ += dS @ K`. The rowsplit smem struct drops 144KB ->
128KB, while the qwen launch carveout remains 148KB because neighboring qwen
n256 routes still need it.

Exact qwen dQ op (`S=1024,nq=32,nkv=8,D=128`) is parity-clean vs PyTorch
(`max_abs=1.26e-03`, `max_rel=2.52e-03`) and improves the same harness from
103.65us base to 99.62us. Resource usage is unchanged (`megakernel_df STACK:32`,
`df2 STACK:48`, `ws STACK:64`), avoiding the hidden dispatch tax that killed
the dKV S^T port. Validation: `test_ops.py`, `test_model.py`, and a forced
`MK_ATTN_D128_DQ_RS=1` S256 model parity check including df2/ws agreement all
passed (`mkv3-p4b-d128-dq-rsfeed-main-*.log`).

## Qwen D=128 QKNORM_ROPE_BWD cached-pair path (this session, 20260706T1355Z)

Promoted a D=128 companion to the D=64 qknorm-bwd cache path: each lane owns
two rope pairs and keeps the inverse-rope/RMS intermediates live across the dot
reduction, deleting the generic D!=64 second pass reload/recompute. The default
compile flag is `MK_QKBWD_D128_CACHE=1`, with `=0` retaining the old generic
loop for A/B. The device body is runtime-scoped to the measured qwen attention
shape (`S=1024,nq=32,nkv=8,D=128`) so other D=128 layouts remain on the old
path.

Fresh-cache qwen A/B against forced-old passed parity
(`loss_diff=9.54e-07`, worst selected grad rel `3.14e-03`) and won both timed
orders: default-new then forced-old median +25.5us old-minus-new, new wins
35/40 (`mkv3-p4b-qkbwd-d128-cache-final-defaultfirst-20260706T1342Z.log`);
forced-old then default-new median +26.8us, new wins 32/40
(`mkv3-p4b-qkbwd-d128-cache-final-reverse2-20260706T1353Z.log`). A profile with
the path enabled moved qwen `QKNORM_ROPE_BWD` from the current-main profile's
304.5us total to 281.2us and whole-step profile time 9345.0 -> 9324.0us
(`mkv3-p4b-profile-qwen-qkbwd-d128-cache-20260706T1337Z.log`). Validation:
`diff --check`, `py_compile`, `ruff`, `test_ops.py`, and `test_model.py` passed
with the guarded default-on build (`mkv3-p4b-qkbwd-d128-cache-test*.log`).
After applying the same patch in the main checkout, the qwen activated-path A/B
remained positive: median +28.7us old-minus-new, mean +38.3us, new wins 20/24
(`mkv3-p4b-qkbwd-d128-cache-main-qwen-ab-20260706T1412Z.log`).

## Qwen MLP dX m64n256 BF16 NN route (this session, 20260706T1515Z)

Promoted a qwen-only BF16-output use of the existing m64n256 NN/TN direct body
for the two remaining on-path MLP dX rows:
`GEMMNN 1024x9728x2560` (wd dX) and `GEMMNN 1024x2560x19456` (wgu dX). Current
main routed both through n128 (`flags=4224`, tiles 608/160). The new route is
exact-gated by `wgmma_n256_nn_bf16_ok`, uses the qwen stage3+n-major n256 bits,
and can be disabled with `MK_WGMMA_N256_NN_BF16=0`. The earlier small-shape
`GEMMNN 1024x512x3072` n256 BF16 probe remains rejected; this gate intentionally
does not cover it.

Scratch route check changed only those rows: tiles `608 -> 304` and `160 -> 80`,
with `n_instr=47`, `critical_path=26`, and the existing qwen 148KB smem page.
Qwen A/B against forced-old was parity-clean under the standard model-style
gradient scale (`loss_diff=-1.91e-06`, worst grad `emb` rel `3.05e-03`) and won
both construction/timing orders: forced-old first median +120.5us old-minus-new,
new wins 24/24 (`mkv3-p4b-qwen-n256-nn-bf16-ab-20260706T1502Z.log`); candidate
first median +102.8us, new wins 24/24
(`mkv3-p4b-qwen-n256-nn-bf16-reverse-20260706T1507Z.log`). Candidate profile
measured step total 9220.1us vs the refreshed current-main 9314.2us, with the
targeted MLP dX spans dropping to 502.5us and 314.0us
(`mkv3-p4b-profile-qwen-n256-nn-bf16-20260706T1510Z.log`). Validation:
`diff --check`, `py_compile`, `ruff`, focused n256 BF16 NN `test_ops.py`, and
full `test_model.py` passed in the scratch worktree.
After applying the same patch in main, the qwen A/B remained positive:
`loss_diff=+2.86e-06`, worst grad `emb` rel `3.05e-03`, median +98.7us
old-minus-new, new wins 16/16
(`mkv3-p4b-qwen-n256-nn-bf16-main-ab-20260706T1519Z.log`). Main `test_ops.py`
also passed with the new BF16 n256 NN cases
(`mkv3-p4b-qwen-n256-nn-bf16-main-testops-20260706T1524Z.log`).

## Qwen qkv dX m64n256 BF16 NN route (this session, 20260706T1235Z)

Extended the qwen-only BF16-output m64n256 NN route to the remaining on-path
qkv dX row: `GEMMNN 1024x2560x6144`. The route is still exact-shape gated by
`wgmma_n256_nn_bf16_ok`; `MK_WGMMA_N256_NN_BF16=0` disables all BF16 NN n256
probes, while the narrower `MK_WGMMA_N256_QKVDX_BF16=0` disables only this qkv
dX addition for A/B against the prior MLP dX promotion.

Scratch route check against `6ae088f` left the MLP dX rows unchanged and moved
only qkv dX from n128 to qwen stage3+n-major n256:
`tiles 160 -> 80`, flags `[7,12] -> [7,14,25,26]`, `n_instr=47`,
`critical_path=26`, qwen smem still 148KB. The isolated qwen A/B was parity
clean (`loss_diff=0`, worst grad `emb` rel `3.00e-03`) and won both
construction orders: old-then-new median +21.7us old-minus-new, new wins 15/16;
new-then-old median +27.1us, new wins 14/16
(`mkv3-p4b-qwen-n256-qkvdx-ab-20260706T1230Z.log`). A profile with qkv enabled
showed the targeted row span moving from 288.6us to 264.2us, with whole-step
best profile time 9368.5us -> 9148.0us
(`mkv3-p4b-profile-qwen-n256-qkvdx-20260706T1232Z.log`).
After applying the same patch in main, the qwen A/B remained positive:
`loss_diff=0`, worst grad `emb` rel `3.00e-03`, old-then-new median +25.6us
old-minus-new with new wins 13/16, and new-then-old median +31.5us with new
wins 15/16 (`mkv3-p4b-qwen-n256-qkvdx-main-ab-20260706T1233Z.log`). Main
`test_ops.py` passed, including the n256 BF16 NN coverage
(`mkv3-p4b-qwen-n256-qkvdx-main-testops-20260706T1235Z.log`).

## Qwen fused dOatt/Drow m64n256 route (this session, 20260706T1245Z)

Added exact qwen D=128 support for the fused dOatt/Drow GEMM through the n256
NN direct body: `GEMMNN 1024x4096x2560` with flags bit10. The n256 Drow
specialization is compile-time separated from the non-Drow n256 NN routes, and
the route is gated by `wgmma_n256_nn_bf16_drow_ok`; set
`MK_WGMMA_N256_DROW_BF16=0` to restore the previous WGMMA Drow route.

Scratch route check moved only the Drow row from WGMMA (`flags=1152`, tiles 512)
to qwen stage3+n-major n256 (`flags=100680832`, tiles 128), leaving `n_instr=47`,
`critical_path=26`, and qwen smem at 148KB. Qwen smoke parity was clean
(`loss_diff=+1.91e-06`, worst grad `emb` rel `3.00e-03`;
`mkv3-p4b-drow-n256-qwen-smoke-20260706T1241Z.log`). Paired qwen timing won
both construction orders: old-then-new median +152.8us old-minus-new, new wins
16/16; new-then-old median +127.5us, new wins 16/16
(`mkv3-p4b-drow-n256-qwen-ab-20260706T1243Z.log`). Profile attribution showed
the targeted Drow row span moving from 350.7us to 244.7us, with best traced step
9287.3us -> 9022.7us
(`mkv3-p4b-profile-drow-n256-qwen-20260706T1243Z.log`). Scratch `test_ops.py`
passed with new direct coverage for both `NN n256 drow bf16` and `NN n256 drow`
(`mkv3-p4b-drow-n256-qwen-testops-20260706T1244Z.log`).
After applying the same patch in main, the qwen A/B remained positive:
`loss_diff=+3.81e-06`, worst grad `emb` rel `3.00e-03`, old-then-new median
+120.2us old-minus-new with new wins 16/16, and new-then-old median +140.3us
with new wins 15/16 (`mkv3-p4b-drow-n256-qwen-main-ab-20260706T1247Z.log`).
Main `test_ops.py` passed with the new n256 Drow cases
(`mkv3-p4b-drow-n256-qwen-main-testops-20260706T1249Z.log`).

## ATTN_FWD_WG cross-stage PV pipeline (session 2853e0de + subagent, 20260706T1245Z)

Triple-buffered K/V (AttnWgFwdSmem 64->80KB, fits the 100KB carveout), S
issued via the existing no-wait helper with one `warpgroup_wait<0>` before the
softmax ALU (in-order retirement covers PV(t-1)+S(t)), PV commits with no
drain and flies across the bottom consumer_sync and the (t+2)%3 refill
(disjoint by parity); post-loop drain + a NEW consumer_sync before the fp32
epilogue overlay (cross-WG P/Q overlay hazard found in implementation).
Op-level O/LSE BITWISE identical; test_ops/test_model green; REG:255 STACK:32
LD-gate 0 at small. Isolated A/B vs clean a7a75d3 both orders: S8192
**-53.6/-41.0**, small -18.3/-15.7, S4096/nano noise. Far below the ~-270
fwd-bucket hope — the PV drain was mostly absorbed by co-scheduling
(absorption ledger, 8th entry) — kept as a safe monotone win. NOTE
(follow-up, unclaimed): S8192 shows 786432 local-LD sectors in BOTH arms —
pre-existing since the 43803eb n256 head route (proven by control equality);
small remains 0. Worth an ncu source pass on the n256 impls at long-S.
Logs: mkv3-p4b-fwdpipe-*-20260706T122413Z.log.

## S256 lm-head n128 default retune (session codex, 20260706T1402Z)

Current-head source-free P5 retune from `smallshape-gemm-study.md`: disabling the
generic m64n128 route with `MK_WGMMA_N128=0` only changed the S256 lm-head fwd row
from n128 (`GEMMNT 256x8192x256.wg`, 128 tiles, flags 6274) to the older n64 route
(`256` tiles, flags 2178), leaving `n_instr=161`, `critical_path=76`, and
`gated=67`. Parity stayed in the normal route-tolerance class (`loss_diff
-1.91e-06`, worst grad `emb` rel `6.82e-03`), and paired timing won both orders:
default-then-n128off `+6.56us` median / `+6.79us` mean with `145/160` wins, and
n128off-then-default `+5.90us` median / `+5.98us` mean with `139/160` wins.
Nano/S512 was a no-change boundary (`+0.45us` then `+0.16us` medians, second-order
mean negative), so keep nano on n128. The default `wgmma_n128_ok` gate now uses
`M < 512 -> off`, `512 <= M < 1024 -> lm_head-only`, and `M >= 1024 -> all
eligible`. Set `MK_WGMMA_N128=2` to force the old S256 lm-head n128 route for A/B.
Logs: `mkv3-p4b-s256-nano-n128off-current-20260706T1402Z.log` and
`mkv3-p4b-s256-n128off-profile-20260706T1408Z.log`.

## Long-S local-LD trace: LAW AMENDED, no action (2853e0de + subagent, 20260706T1440Z)

The 786,432 local-LD sectors at S8192 are NOT spill and NOT the n256 route
(same-binary env A/B: identical LD with route on/off). 100% comes from ONE
dynamically-indexed `float w[8]` in `op_attn_combine` (attention.cuh:296) —
local-by-construction, executes only where fwd split-bands run (never small),
latency-hidden: stall-priced at ~1-2us (all local traffic ~20-25us). The 13.1M
local ST sectors are matching-LDL-free semi-dead stores (ld8bf staging f[8]) —
ABI-frame class, harmless. n256 mainloop d[128] fully register-resident.
**LAW AMENDMENTS**: (1) the promotion LD==0 gate is SHAPE-SCOPED — certify
small==0 plus control-equality at band-enabled long S; (2) when attributing
via nvdisasm line info, use the INNERMOST inline frame — the outermost frame
collapses everything to the dispatch line (megakernel.cu:423), which polluted
the first reading of the morning's spill hunt (runtime fixes stood anyway).
Optional hygiene (not time): #pragma unroll the three C-loops in combine with
c<C guards. Log: mkv3-p4b-n256-longs-ld-20260706T143417Z.log.

## Qwen SwiGLU backward 4W route (this session, 20260706T1515Z)

Added an exact qwen4b-L1 route for `SWIGLU_BWD_4W`: four warps cooperate on one
very-wide I=9728 row, replacing the promoted qwen 2W route at the same program
point. The 2W code still compiles for the shape, but default runtime routing now
prefers 4W; set `MK_SWIGLU_BWD_4W=0` to force the old 2W route for A/B. This
shape is separate from the earlier small H512/I1536 4W no-go.

Scratch validation from `/home/apanda/xorl-oss-qwen-swiglu-4w` first proved the
source route default-off, then the promoted default. The promoted scratch route
was `sw_bwd4w=1/512` by default and `sw_bwd2w=1/256` with
`MK_SWIGLU_BWD_4W=0`, with selected-gradient parity clean (`loss_diff`
`-9.54e-07` / `-9.54e-07`, worst grad `emb` rel `3.14e-03`). Timing won both
construction orders: old2w-minus-default medians `+355.71us` and `+329.78us`,
default4w wins `47/48` and `48/48`. Scratch `test_model.py` and `test_ops.py`
passed. Logs:
`mkv3-p4b-qwen-swiglu-4w-promoted-default-first-20260706T1456Z.log`,
`mkv3-p4b-qwen-swiglu-4w-promoted-old2w-first-20260706T1456Z.log`.

After applying the identical patch in main, the qwen A/B stayed positive with the
same route split and parity clean (`loss_diff=+2.86e-06` / `-1.91e-06`, worst
grad `emb` rel `1.57e-03` / `3.14e-03`). Timing won both construction orders:
old2w-minus-default medians `+355.97us` and `+342.80us`, default4w wins `48/48`
and `47/48`. Main validation passed `py_compile`, `git diff --check`,
`test_model.py`, and `test_ops.py`. Logs:
`mkv3-p4b-qwen-swiglu-4w-main-default-first-20260706T1505Z.log`,
`mkv3-p4b-qwen-swiglu-4w-main-old2w-first-20260706T1505Z.log`.

## Small H512/S1024 SwiGLU backward 4W route (this session, 20260706T1530Z)

Extended the existing `SWIGLU_BWD_4W` opcode to the exact small benchmark shape
`H512/S1024/I1536/V16384/nq8/nkv4/D64/L8`. The first promotion routed small's
cached-sigmoid SwiGLU backward rows through four warps per row, while
`MK_SWIGLU_BWD_4W=0` restored the old 2W worker-count route for A/B. This
current-head recheck overturned the earlier small 4W no-go after the surrounding
scheduler and 4W implementation changed. Follow-up below moves the final small
default to cache-off + 4W; set `MK_SWIGLU_CACHE_SIG=1 MK_SWIGLU_BWD_4W=0` to
restore the full pre-1530Z small behavior.

Current-head refresh before the change measured small at `3601.0us` in
`profile_df.py` and `3576.2us` vs graph+ `1903.9us` in `final_bench.py`; the
on-path `SWIGLU_BWD_2W` bucket was `31.5us` wait + `210.7us` span. Source-free
env A/B with `MK_SWIGLU_BWD_4W=1` routed default `sw_bwd2w=8/2048` versus
candidate `sw_bwd4w=8/4096`, kept selected-gradient parity clean
(`loss_diff=+0.00e+00` / `+2.86e-06`, worst grad `qn.0` rel `<5.3e-07`), and
won both construction orders: default-minus-4W medians `+132.27us` and
`+121.23us`, 4W wins `80/80` and `80/80`. Logs:
`mkv3-p4b-small-swiglu-4w-current-default-first-20260706T1517Z.log`,
`mkv3-p4b-small-swiglu-4w-current-sw4w-first-20260706T1517Z.log`.

After promoting the shape default, forced-old A/B stayed positive with the same
route split and clean parity (`loss_diff=-4.77e-06` / `+0.00e+00`, worst grad
`qn.0` rel `4.41e-07`). Default4W won both construction orders:
old2w-minus-default medians `+126.06us` and `+133.02us`, wins `80/80` and
`80/80`. The refreshed default profile measured `3458.7us`; refreshed score
measured small `3446.7us` vs graph+ `1890.7us` (gap `1.82x`). Validation passed
`py_compile`, `git diff --check`, `test_model.py`, and `test_ops.py`. Logs:
`mkv3-p4b-small-swiglu-4w-promoted-default-first-20260706T1528Z.log`,
`mkv3-p4b-small-swiglu-4w-promoted-old2w-first-20260706T1528Z.log`,
`mkv3-p4b-profile-small-swiglu-4w-default-20260706T1528Z.log`,
`mkv3-p4b-score-small-swiglu-4w-default-20260706T1528Z.log`.

## Small SwiGLU cache-off with 4W follow-up (this session, 20260706T1540Z)

With small on the 4W backward body, rechecked whether cached sigmoid still pays.
Source-free env A/B kept `SWIGLU_BWD_4W` active in both arms and toggled only
`MK_SWIGLU_CACHE_SIG=0`. Routes stayed `sw_bwd4w=8/4096`; default had
`swsig=True`, cache-off had `swsig=False`. Parity stayed inside tolerance
(`loss_diff=+3.81e-06` / `-1.91e-06`, worst grad `kn.0` rel `1.31e-02`), and
cache-off won both construction orders: default-minus-cacheoff medians
`+25.68us` and `+22.70us`, cacheoff wins `78/80` and `78/80`. Logs:
`mkv3-p4b-small-swiglu-4w-cacheoff-default-first-20260706T1530Z.log`,
`mkv3-p4b-small-swiglu-4w-cacheoff-cacheoff-first-20260706T1530Z.log`.

The final small default removes `(512, 1024, 1536)` from `_SWIGLU_CACHED_2W`,
keeps a separate `_SMALL_SWIGLU_BWD_2W` fallback for worker-count A/B, and keeps
`_SMALL_SWIGLU_BWD_4W` enabled. Against the full previous default
(`MK_SWIGLU_CACHE_SIG=1 MK_SWIGLU_BWD_4W=0`), the final default routed
`swsig=False, sw_bwd4w=8/4096` versus old `swsig=True, sw_bwd2w=8/2048`, with
parity clean (`loss_diff=+2.86e-06` / `-1.91e-06`, worst grad `kn.0` rel
`1.31e-02`) and wins both orders: old-minus-default medians `+156.70us` and
`+148.88us`, default wins `80/80` and `80/80`. The refreshed default profile
measured `3421.6us`; refreshed score measured small `3425.9us` vs graph+
`1916.7us` (gap `1.79x`). Validation passed `py_compile`, `git diff --check`,
`test_model.py`, and `test_ops.py`. Logs:
`mkv3-p4b-small-swiglu-4w-cacheoff-promoted-default-first-20260706T1530Z.log`,
`mkv3-p4b-small-swiglu-4w-cacheoff-promoted-old-first-20260706T1530Z.log`,
`mkv3-p4b-profile-small-swiglu-4w-cacheoff-default-20260706T1530Z.log`,
`mkv3-p4b-score-small-swiglu-4w-cacheoff-default-20260706T1530Z.log`.

Follow-up local-load certification on the final `ab4ffde` cache-off default
closed the 80d0053 advisory. SourceCounters on H512/S1024 small found **zero**
local-load source rows and only local stores; the one-pass metric gate confirmed
`l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum = 0` and
`local_op_st.sum = 8,437,760` with `gpu__time_duration.sum = 3.95 ms` under NCU.
So the earlier 2,906 local-load sectors belonged to the intermediate cached
4W arm, not the final small default. No source change is needed; keep the
shape-scoped LD law as small==0, long-S budgeted separately. Logs:
`mkv3-p4b-small-source-ab4ffde-20260706T154058Z.{log,csv,ncu-rep}`,
`mkv3-p4b-small-ldgate-ab4ffde-20260706T154439Z.{log,details.csv,ncu-rep}`.
Detailed note: `results/operator-gap/small-swiglu-4w-cacheoff-ldgate.md`.

## Post-4W source-free follow-ups: nano 4W, small cold cap, and small gmbar (this session, 20260706T1605Z)

Current-head follow-ups after the small 4W/cache-off route:

- H256/S128 nano `MK_SWIGLU_BWD_4W=1` is an order-sensitive wash. The forced
  route changed nano from `sw_bwd=4/256` to `sw_bwd4w=4/1024` and parity stayed
  within the usual noisy envelope (`loss_diff=+3.81e-06`, worst selected grad
  `emb` rel `4.26e-03`), but timing failed the both-order gate: default-first
  showed forced 4W faster by `+10.51us` median with wins `72/80`, while
  force4w-first flipped negative at `-5.34us` with wins `18/80`. Keep nano on
  the generic SwiGLU backward route. Log:
  `mkv3-p4b-nano-swiglu-4w-env-ab-20260706T154841Z.log`; detail note:
  `results/operator-gap/nano-swiglu-4w-env-nogo.md`.
- H512/S1024 small `MK_COLD_CAP` also remains at the existing cap48 default.
  Broad source-free sweep over `0/16/33/48/64/96` suggested cap64/cap96 about
  `12us` faster than cap48, but the focused cap48-vs-cap64 A/B failed
  construction-order reversal: cap48-first measured cap48-minus-cap64
  `+15.09us` median with cap64 wins `68/80`, while cap64-first flipped to
  `-1.02us` with cap64 wins `38/80`. Keep `_cold_cap()` unchanged. Log:
  `mkv3-p4b-small-coldcap-current-f66c432-20260706T1555Z.log`; detail note:
  `results/operator-gap/small-coldcap-post4w-nogo.md`.
- H512/S1024 small `MK_GEMM_MBAR_RING=0` confirmed the current D64 mbar-ring
  default is still load-bearing. Forced old two-stage GEMM feed path preserved
  parity (`loss_diff=-9.54e-07`, worst selected grad `wf` rel `2.88e-07`) but
  lost both construction orders: old-minus-default `+38.03us` median with
  default wins `62/80`, and `+44.38us` with default wins `79/80`. Keep
  `gemm_mbar_ring_default = c.D == 64 and c.S >= 1024 and c.S % 128 == 0`.
  Log: `mkv3-p4b-small-gmbar-current-f66c432-20260706T1600Z.log`; detail
  note: `results/operator-gap/small-gmbar-post4w-keep.md`.

## Small H512/S1024 NN n128 route retunes (this session, 20260706T1725Z)

After the small 4W/cache-off route, the remaining MLP dX stack made the two
repeated H512/S1024 NN bf16 n128 rows worth rechecking. A broad source-free
`MK_WGMMA_N128_NN=0` probe first disabled all NN n128 rows and won both
construction orders (`+63.90us` and `+56.69us`, `80/80` wins each), but the
source promotion is narrower: with `MK_WGMMA_N128_NN` unset,
`wgmma_n128_ok()` now routes only `(1024,512,3072)` and `(1024,512,1024)` NN
rows through the normal m64n64 WGMMA path. `MK_WGMMA_N128_NN=1` still restores
the old NN n128 behavior for A/B, and `=0` still disables NN n128 broadly.

The explicit promoted-default route check on H512/S1024 small kept
`n_instr=288`, `critical_path=144`, `gated=127`, and `splitK=33`. Default had
`n128=34`; forced old had `n128=50`. The 16 changed rows are exactly 8x
`GEMMNN 1024x512x3072` and 8x `GEMMNN 1024x512x1024`, moving from n128 flags
`4224` / `32` tiles to m64n64 flags `128` / `64` tiles. Parity stayed clean
(`loss_diff=-1.91e-06` / `-2.86e-06`, worst selected grad rel `<5.4e-07`), and
timing won both construction orders: old-minus-default medians `+28.67us` and
`+28.40us`, default wins `78/80` and `77/80`. Logs:
`mkv3-p4b-small-n128nn-post4w-20260706T1610Z.log`,
`mkv3-p4b-small-n128nn-promoted-explicit-20260706T1640Z.log`,
`mkv3-p4b-small-n128nn-promoted-route-explicit-20260706T1644Z.log`.
Discard `mkv3-p4b-small-n128nn-promoted-20260706T1620Z.log` as decision
evidence: its route print showed the default side at `n128=0`, not the intended
promoted route.

The refreshed narrow-default profile measured `3374.9us` (previous
post-cache-off profile `3421.6us`). The refreshed small score measured
megakernel `3392.1us` vs compile+CUDAGraph+ `1896.8us` (gap still `1.79x`).
Validation passed `py_compile`, `git diff --check`, `test_model.py`, and
`test_ops.py`. Logs:
`mkv3-p4b-profile-small-n128nn-default-20260706T1655Z.log`,
`mkv3-p4b-score-small-n128nn-default-20260706T1656Z.log`. Detail note:
`results/operator-gap/small-n128nn-post4w-promote.md`.

Follow-up after the narrow promotion compared current default against broad
`MK_WGMMA_N128_NN=0` and found only one remaining changed row family:
8x `GEMMNN 1024x1536x512`, moving from n128 flags `4224` / `96` tiles to
m64n64 flags `128` / `192` tiles. The 160-sample repeat survived both
construction orders with clean parity (`loss_diff=-1.91e-06` / `+0.00e+00`,
worst selected grad rel `<7.1e-07`): default-minus-broad-off medians
`+11.76us` and `+7.01us`, broad-off wins `127/160` and `114/160`. The default
gate now also excludes exact shape `(1024,1536,512)` when
`MK_WGMMA_N128_NN` is unset; force-on/force-off env semantics are unchanged.
Logs: `mkv3-p4b-small-n128nn-broad-after-promote-20260706T1710Z.log`,
`mkv3-p4b-small-n128nn-broad-after-promote-repeat-20260706T1715Z.log`.

Final promoted-default validation against `MK_WGMMA_N128_NN=1` restored all 24
old NN n128 rows in the forced-old arm (`n128=26 -> 50`) and won both
construction orders: old-minus-default medians `+39.22us` and `+33.17us`,
default wins `80/80` and `80/80`, with clean parity (`loss_diff=+2.86e-06` /
`-1.91e-06`, worst selected grad rel `<7.2e-07`). Post-edit validation again
passed `py_compile`, `git diff --check`, `test_model.py`, and `test_ops.py`.
The final refreshed profile measured `3379.2us`; the final small score measured
megakernel `3373.2us` vs compile+CUDAGraph+ `1904.9us` (gap `1.77x`). Logs:
`mkv3-p4b-small-n128nn-final-promoted-20260706T1725Z.log`,
`mkv3-p4b-profile-small-n128nn-final-default-20260706T1732Z.log`,
`mkv3-p4b-score-small-n128nn-final-default-20260706T1735Z.log`.

## Small H512/S1024 NT n128 route retune (this session, 20260706T1800Z)

After the NN n128 retunes, `MK_WGMMA_N128=2` was used as a source-free probe to
keep lm-head/head-dX n128 behavior but disable the general NT n128 route. On
H512/S1024 small this changed exactly 24 rows: 8x
`GEMMNT 1024x512x512` (`12434` / 32 tiles -> `8338` / 64 tiles), 8x
`GEMMNT 1024x3072x512` (`4226` / 192 tiles -> `130` / 384 tiles), and 8x
`GEMMNT 1024x512x1536` (`12434` / 32 tiles -> `8338` / 64 tiles). Route shape
stayed `n_instr=288`, `critical_path=144`, `gated=127`, while `n128` dropped
from `26` to `2`. Parity stayed clean (`loss_diff=-2.86e-06` / `+3.81e-06`,
worst selected grad rel `<4.3e-07`), and timing won both construction orders:
default-minus-mode2 medians `+56.46us` and `+55.62us`, mode2 wins `80/80` and
`80/80`. Log: `mkv3-p4b-small-n128-mode2-after-final-20260706T1740Z.log`.

The promoted default now excludes only those exact NT shapes when
`MK_WGMMA_N128` is unset; `MK_WGMMA_N128=1` still restores old general n128 for
A/B, and `=2` still means lm-head-only mode. Promoted validation against
`MK_WGMMA_N128=1` kept `n128=2` in default vs `n128=26` in forced-old, with clean
parity (`loss_diff=+9.54e-07` / `-1.91e-06`, worst selected grad rel
`<6.2e-07`) and wins both orders: old-minus-default medians `+53.36us` and
`+44.85us`, default wins `79/80` and `80/80`. Validation passed `py_compile`,
`git diff --check`, `test_model.py`, and `test_ops.py`. The refreshed final
profile measured `3354.8us`; the refreshed final score measured megakernel
`3356.6us` vs compile+CUDAGraph+ `1888.5us` (gap `1.78x`). Logs:
`mkv3-p4b-small-n128nt-promoted-20260706T1750Z.log`,
`mkv3-p4b-profile-small-n128nt-default-20260706T1755Z.log`,
`mkv3-p4b-score-small-n128nt-default-20260706T1757Z.log`. Detail note:
`results/operator-gap/small-n128nt-post4w-promote.md`.

## Post-n128 small cold-cap recheck (this session, 20260706T1815Z)

After the NN/NT n128 route changes, re-swept H512/S1024 small `MK_COLD_CAP` over
`0/16/33/48/64/96`. Route shape stayed `n_instr=288`, `critical_path=144`,
`gated=127`, `hot=197`, `cold=91`, and parity against cap48 stayed clean
(worst selected grad rel `<5.4e-07`). The broad 60-sample pass was flat:
cap48 `3283.47us`, cap64 `3280.27us`, cap16/cap33/cap96 all about `3283us`,
with only cap0 clearly worse at `3322.08us`. The focused cap48-vs-cap64
160-sample gate failed the decision bar: cap48-first measured
cap48-minus-cap64 `-0.13us` median, cap64 wins `79/160`; cap64-first measured
`+1.36us`, cap64 wins `86/160`. Keep cap48. Logs:
`mkv3-p4b-small-coldcap-after-n128-20260706T1805Z.log`,
`mkv3-p4b-small-coldcap-after-n128-focused-20260706T1810Z.log`. Detail note:
`results/operator-gap/small-coldcap-post-n128-nogo.md`.

## Post-n128 small GEMM mbar-ring recheck (this session, 20260706T1825Z)

After the n128 row gates, rechecked H512/S1024 small `MK_GEMM_MBAR_RING=0`.
Program shape stayed `n_instr=288`, `critical_path=144`, `gated=127`, and
parity stayed clean (`loss_diff=+3.81e-06` / `+0.00e+00`, worst selected grad
rel `<6.8e-07`). Forced old two-stage GEMM feed path lost both construction
orders by a large margin: old-minus-default medians `+154.51us` and `+150.94us`,
default wins `80/80` and `80/80`. Keep the D64 mbar-ring default. Log:
`mkv3-p4b-small-gmbar-after-n128-20260706T1820Z.log`. Detail note:
`results/operator-gap/small-gmbar-post-n128-keep.md`.

## Post-NT small lm-head n128 check (this session, 20260706T1835Z)

After the NN/NT n128 row gates, rechecked the remaining generic n128 row on
H512/S1024 small by comparing current default against `MK_WGMMA_N128=0`. This
only moved lm-head forward row 152 (`GEMMNN 1024x16384x512`) off n128, changing
`n128=2 -> 1`; head-dX row 155 remained on its explicit n128 path. Program shape
stayed `n_instr=288`, `critical_path=144`, `gated=127`, and parity stayed clean
(`loss_diff=-2.86e-06` / `-3.81e-06`, worst selected grad rel `<7.2e-07`).
Disabling the lm-head n128 row lost both construction orders:
default-minus-n128off medians `-48.88us` and `-48.67us`, with n128off wins
`1/80` and `0/80`. Keep the current lm-head n128 route. Log:
`mkv3-p4b-small-n128off-after-nt-20260706T1830Z.log`. Detail note:
`results/operator-gap/small-lmhead-n128-post-nt-keep.md`.

## Post-n128 small attention chunk recheck (this session, 20260706T1900Z)

After the 4W/cache and NN/NT n128 route changes, rechecked H512/S1024 small
attention chunk envs against the current `DKV_C=1/DQ_C=1` default. Route shape
stayed `n_instr=288`, `critical_path=144`, `gated=127`; default remained
`ATTN_FWD_WG=8/512`, `ATTN_DKV_WG=8/512`, `ATTN_DQ_WG=8/512`. The variants
`DKV_C=2/DQ_C=1`, `DKV_C=3/DQ_C=1`, `DKV_C=1/DQ_C=2`, and `DKV_C=2/DQ_C=2`
changed only the expected DKV/DQ tile counts and stayed parity-clean
(`loss_diff` within `+/-4.77e-06`, worst selected grad rel `<1.18e-02`), but all
lost both build orders with zero wins: default-minus-variant medians were
`-77.71us`/`-72.86us`, `-81.81us`/`-77.81us`, `-34.61us`/`-32.99us`, and
`-69.42us`/`-65.44us`. Keep small attention chunks at `DKV_C=1/DQ_C=1`. Log:
`mkv3-p4b-small-attnc-post-n128-20260706T1840Z.log`. Detail note:
`results/operator-gap/small-attnc-post-n128-nogo.md`.

## Small direct-BF16 GEMM epilogue post-n128 promotion (this session, 20260706T1925Z)

The earlier direct-BF16 GEMM epilogue result was S128-only because H512/S1024
small lost or was neutral. After the NN/NT n128 retunes, forced
`MK_GEMM_DIRECT_BF16_EPILOGUE=1` now covers 32 ordinary WGMMA BF16-output rows
(`5632` tiles): 8x `GEMMNN 1024x512x1024`, 8x `GEMMNN 1024x512x3072`, 8x
`GEMMNN 1024x1536x512`, and 8x `GEMMNT 1024x3072x512`. Route shape stayed
`n_instr=288`, `critical_path=144`, `gated=127`, and parity stayed clean
(`loss_diff=+/-9.54e-07`, worst selected grad rel `<5.3e-07`). Forced direct
won both source-free orders: default-minus-direct medians `+27.82us` and
`+27.01us`, direct wins `80/80` and `80/80`.

The default gate now also enables direct-BF16 GEMM epilogues for exact
H512/L8/S1024 small, while preserving the earlier S128 default. Forced old
`MK_GEMM_DIRECT_BF16_EPILOGUE=0` lost both promoted-default orders: default beat
old by `28.82us` and `23.71us`, with old wins `1/80` and `2/80`. Validation
passed `py_compile`, `git diff --check`, `test_model.py`, and `test_ops.py`.
Refreshed profile measured `3320.1us`; refreshed score measured megakernel
`3312.5us` vs compile+CUDAGraph+ `1903.3us` (gap `1.74x`). Logs:
`mkv3-p4b-small-directbf16-post-n128-20260706T1905Z.log`,
`mkv3-p4b-small-directbf16-promoted-20260706T1915Z.log`,
`mkv3-p4b-profile-small-directbf16-default-20260706T1925Z.log`, and
`mkv3-p4b-score-small-directbf16-default-20260706T1925Z.log`. Detail note:
`results/operator-gap/small-directbf16-post-n128-promote.md`.

Post-direct focused attention recheck: because the direct-BF16 promotion moved
attention around in the realized profile, rechecked the closest prior attention
chunk no-go, `MK_ATTN_DKV_C=1 MK_ATTN_DQ_C=2`, on current head. It still only
changed `ATTN_DQ_WG=8/512 -> 8/1024`, stayed parity-clean
(`loss_diff=+9.54e-07` / `0`, worst selected grad rel `<1.18e-02`), and lost
both orders: default-minus-variant `-29.44us` and `-28.46us`, variant wins
`2/80` and `1/80`. Keep small attention chunks at `DKV_C=1/DQ_C=1`. Log:
`mkv3-p4b-small-attnc-post-directbf16-20260706T1930Z.log`. Detail note:
`results/operator-gap/small-attnc-post-n128-nogo.md`.

Post-direct focused cold-cap recheck: because cap64 was close before direct-BF16,
rechecked current cap48/default against `MK_COLD_CAP=64`. Route shape stayed
`n_instr=288`, `critical_path=144`, `gated=127`, `hot=197`, `cold=91`, and
parity stayed clean (worst selected grad rel `<6.7e-07`). Cap64 still failed:
default-minus-cap64 was `-3.04us` with cap64 wins `65/160` in default-first,
and `-0.85us` with cap64 wins `75/160` in cap64-first. Keep cap48. Log:
`mkv3-p4b-small-coldcap-post-directbf16-20260706T1940Z.log`. Detail note:
`results/operator-gap/small-coldcap-post-n128-nogo.md`.

Post-direct small RMS dX R4 recheck: because RMS dX is back in the small
top-five after the n128 and direct-BF16 route moves, rechecked forced
`MK_RMS_DX_R4=1` against the current default. Route shape stayed
`n_instr=288`, `critical_path=144`, `gated=127`; default used 17 two-row
`RMSNORM_BWD_DX` ops / 1088 tiles, while R4 used 17 `RMSNORM_BWD_DX_R4` ops /
544 tiles. Parity stayed clean (`loss_diff=-3.81e-06` / `+2.86e-06`, worst
selected grad rel `<7.1e-07`), but R4 lost both orders: default-minus-R4
`-34.08us` and `-35.23us`, R4 wins `0/160` and `1/160`. Keep H512/S1024 small
on the normal two-row RMS dX route. Log:
`mkv3-p4b-small-rmsdx-r4-post-directbf16-20260706T1657Z.log`. Detail note:
`results/operator-gap/small-rmsdx-r4-post-directbf16-nogo.md`.

Post-direct small dKV float2 recheck: because `ATTN_DKV_WG` remains a large
on-path bucket, rechecked forced scalar direct atomics via
`MK_ATTN_DKV_FLOAT2_ATOMIC=0`. The scalar extension correctly dropped `_adkvf2`
while route shape stayed `n_instr=288`, `critical_path=144`, `gated=127`,
`ATTN_DKV_WG=8/512`. Parity stayed clean (`loss_diff=+9.54e-07` /
`-9.54e-07`, worst selected grad rel `<6.3e-07`), and scalar lost both orders:
default-minus-scalar `-30.77us` and `-28.56us`, scalar wins `1/80` and `0/80`.
Keep the dKV float2 direct-atomic default after direct-BF16. Log:
`mkv3-p4b-small-dkvf2-post-directbf16-20260706T1659Z.log`. Detail note:
`results/operator-gap/small-dkvf2-post-directbf16-keep.md`.

Post-direct small attention fast-log recheck: because `ATTN_FWD_WG` remains
top-three but H256/S1024 had already shown fast-log neutral, rechecked exact
H512/L8/S1024 small against `MK_ATTN_FAST_LOG=0`. Precise `logf` dropped the
`_aflog` suffix, kept route shape unchanged (`n_instr=288`,
`critical_path=144`, `gated=127`, `ATTN_FWD_WG=8/512`), and stayed within the
normal q/k norm tolerance (worst selected grad `qn.0` rel around `9.54e-03`).
The signal is small but repeatable: source-free precise beat fast by
`+1.34/+1.46us` over 80 reps, `+2.35/+1.07us` over 240 reps, and
`+0.83/+1.22us` over 480 reps. Exact small now requests precise attention LSE
log by default; `MK_ATTN_FAST_LOG=1` restores the old `_aflog` route. Promoted
default beat forced fast by `-1.04us` and `-1.65us` over 240 reps, validation
passed `py_compile`, `test_model.py`, and `test_ops.py`, refreshed profile was
`3331.4us`, and refreshed score was megakernel `3312.6us` vs graph+ `1898.0us`.
Logs: `mkv3-p4b-small-aflog-post-directbf16-20260706T1702Z.log`,
`mkv3-p4b-small-aflog-post-directbf16-repeat-20260706T1702Z.log`,
`mkv3-p4b-small-aflog-post-directbf16-repeat2-20260706T1702Z.log`,
`mkv3-p4b-small-aflog-promoted-20260706T1702Z.log`,
`mkv3-p4b-profile-small-aflog-precise-default-20260706T1702Z.log`, and
`mkv3-p4b-score-small-aflog-precise-default-20260706T1702Z.log`. Detail note:
`results/operator-gap/small-aflog-post-directbf16-promote.md`.

Post-fast-log small attention dQ float2 recheck: after the precise-log default
made `ATTN_DQ_WG` more exposed in the profile, forced
`MK_ATTN_DQ_FLOAT2_STORE=1` for exact H512/S1024 small. The variant compiled the
expected `_adqf2` extension, kept route shape unchanged (`n_instr=288`,
`critical_path=144`, `gated=127`, `ATTN_DQ_WG=8/512`), and stayed parity-clean
(`loss_diff=-1.91e-06` / `+9.54e-07`, worst selected grad rel `<4.7e-07`), but
lost both orders: default-minus-float2 `-3.18us` and `-5.76us`, float2 wins
`33/80` and `27/80`. Keep `_adqf2` limited to the existing H256 long-shape
gates. Log: `mkv3-p4b-small-dqf2-post-aflog-20260706T1709Z.log`. Detail note:
`results/operator-gap/small-dqf2-post-aflog-nogo.md`.

Post-fast-log small attention exp2 recheck: forced the old precise exp path with
`MK_ATTN_EXP2_APPROX=0` after the small fast-log retune. The forced-old
extension correctly dropped `_aex2`, kept route shape unchanged (`n_instr=288`,
`critical_path=144`, `gated=127`, `ATTN_FWD/DKV/DQ=8/512`), and parity stayed
clean (`loss_diff=-1.91e-06` / `-9.54e-07`, worst selected grad rel
`<5.4e-07`). Precise exp lost both orders: default-minus-precise `-28.80us`
and `-25.66us`, precise wins `3/80` and `3/80`. Keep `_aex2` enabled for
H512/S1024 small. Log: `mkv3-p4b-small-aex2-post-aflog-20260706T1712Z.log`.
Detail note: `results/operator-gap/small-aex2-post-aflog-keep.md`.

Post-fast-log small qknorm D64 cache recheck: forced the old generic loop with
`MK_QKBWD_D64_CACHE=0`. The forced-old extension correctly dropped `_qkbc`,
kept route shape unchanged (`n_instr=288`, `critical_path=144`, `gated=127`,
`QKNORM_ROPE_BWD=8/1024`), and stayed parity-clean (`loss_diff=-3.81e-06` /
`0`, worst selected grad `kn.0` rel around `1.03e-02`). The old loop lost both
orders: default-minus-old `-29.25us` and `-26.21us`, old wins `2/80` and
`1/80`. Keep `_qkbc` enabled for H512/S1024 small. Log:
`mkv3-p4b-small-qkbc-post-aflog-20260706T1715Z.log`. Detail note:
`results/operator-gap/small-qkbc-post-aflog-keep.md`.

Post-qkbc small SwiGLU derivative FMA recheck: because the original FMA win
predated the later `SWIGLU_BWD_4W` and cache-off route, forced the old
derivative expression with `MK_SWIGLU_FMA_DERIV=0`. The forced-old extension
correctly dropped `_swfma`, kept route shape unchanged (`n_instr=288`,
`critical_path=144`, `gated=127`, `SWIGLU_FWD=8/1024`,
`SWIGLU_BWD_4W=8/4096`, `swsig=False`), and parity stayed clean (worst
selected grad `kn.0` rel around `1.01e-02`). The first 80-rep pass only weakly
favored old (`+0.30us` and `+2.24us`, old wins `43/80` and `46/80`), and the
240-rep repeat split by construction order (`+1.28us`, then `-1.30us`; old wins
`127/240` and `109/240`). Keep `_swfma` enabled: the forced-old derivative did
not survive the both-order repeat gate with a meaningful margin. Logs:
`mkv3-p4b-small-swfma-post-qkbc-20260706T1719Z.log` and
`mkv3-p4b-small-swfma-post-qkbc-repeat-20260706T1724Z.log`. Detail note:
`results/operator-gap/small-swfma-post-qkbc-keep.md`.

Post-swfma small lm-head exp2 recheck: forced the old precise lm-head
CE-partial exponential path with `MK_LMHEAD_EXP2_APPROX=0`. The forced-precise
extension correctly dropped `_lex2`, kept route shape unchanged
(`n_instr=288`, `critical_path=144`, `gated=127`, `lmhead_gemm=1/1024`,
`gemm=99/13856`, `ce_fwd=1/1024`), and parity stayed clean (`loss_diff`
`-2.86e-06` / `+5.72e-06`, worst selected grad rel `<4.5e-07`). Precise exp
lost both construction orders: default-minus-precise `-68.45us` and `-67.42us`,
precise wins `0/80` and `0/80`. Keep `_lex2` enabled for H512/S1024 small. Log:
`mkv3-p4b-small-lex2-post-swfma-20260706T1728Z.log`. Detail note:
`results/operator-gap/small-lex2-post-swfma-keep.md`.

Post-lex2 small Drow direct-store recheck: forced the old atomic fused-Drow
epilogue with `MK_DROW_DIRECT_STORE=0`. The forced-old extension correctly
dropped `_drowst`, kept route shape unchanged (`n_instr=288`,
`critical_path=144`, `gated=127`, `drow_gemm=8/512`, `gemm=99/13856`), and
parity stayed clean (`loss_diff=-4.77e-06` / `+2.86e-06`, worst selected grad
rel `<4.7e-07`). The old atomic path lost both construction orders:
default-minus-old `-46.56us` and `-49.86us`, old wins `0/80` and `0/80`. Keep
`_drowst` enabled for H512/S1024 small. Log:
`mkv3-p4b-small-drowst-post-lex2-20260706T1727Z.log`. Detail note:
`results/operator-gap/small-drowst-post-lex2-keep.md`.

Post-drowst small attention dKV direct-atomic recheck: forced the old smem-drain
dKV epilogue with `MK_ATTN_DKV_DIRECT_ATOMIC=0`. The forced-old extension
correctly dropped both `_adkva` and `_adkvf2`, kept route shape unchanged
(`n_instr=288`, `critical_path=144`, `gated=127`, `ATTN_FWD/DKV/DQ=8/512`
each), and parity stayed clean (`loss_diff=-1.91e-06` / `+2.86e-06`, worst
selected grad rel `<9.7e-07`). The old smem-drain path lost both construction
orders: default-minus-old `-91.33us` and `-94.29us`, old wins `0/80` and
`0/80`. Keep `_adkva` and `_adkvf2` enabled for H512/S1024 small. Log:
`mkv3-p4b-small-adkva-post-drowst-20260706T1731Z.log`. Detail note:
`results/operator-gap/small-adkva-post-drowst-keep.md`.

Post-adkva small CE backward exp2 recheck: forced the old precise CE backward
exponential path with `MK_CE_BWD_EXP2_APPROX=0`. The forced-precise extension
correctly dropped `_ceb2`, kept route shape unchanged (`n_instr=288`,
`critical_path=144`, `gated=127`, `CE_FWD=1/1024`, `CE_BWD=1/1024`), and parity
stayed clean (worst selected grad `kn.0` rel around `1.13e-02`). Precise lost
both construction orders: default-minus-precise `-13.98us` and `-12.46us`,
precise wins `11/80` and `8/80`. Keep `_ceb2` enabled for H512/S1024 small; the
current-stack margin is smaller than the original promotion check, but remains
order-stable. Log: `mkv3-p4b-small-ceb2-post-adkva-20260706T1734Z.log`. Detail
note: `results/operator-gap/small-ceb2-post-adkva-keep.md`.

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
