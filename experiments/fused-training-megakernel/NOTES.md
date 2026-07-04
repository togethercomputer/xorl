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

End-of-session certified gauntlet (df defaults, clean-GPU util guards,
median-of-50, fresh process per config; baseline medians wobble +-5-8% across
runs from inductor autotune variance):
| config | megakernel | flash-baseline | gap |
|---|---|---|---|
| nano | 1200 | 631 | 1.90x |
| small | 4114 | 1904 | 2.16x |
| deep-L12 | 3067 | 1562 | 1.96x |
| S=128 | 928 | 491 | 1.89x |
| S=256 | 1029 | 548 | 1.88x |
| S=1024 | 1545 | 775 | 1.99x |
(Morning honest reset: nano 1.97x / small 2.52x.)

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
