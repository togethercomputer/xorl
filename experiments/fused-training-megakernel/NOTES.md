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

**Scoreboard: nano 1622 / small 6356 (df)** vs hardened 712/2735 — gap 2.28x/2.32x
(from 2.5x/2.6x after P5; v3 started at 2.65x/3.44x). STACK note: df 528 -> 608 after
the rowop integration (same dispatch-call-site spill class as P5; the executor-level
fix is still open, now ~the last uniform tax).

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
