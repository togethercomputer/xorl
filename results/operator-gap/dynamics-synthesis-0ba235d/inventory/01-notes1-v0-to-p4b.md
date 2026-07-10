# Era slice: NOTES.md part 1 (v0 -> P4b operator-gap start; pre Jul-5 mostly)

## ARCHITECTURE (foundational decisions)
- Persistent cooperative kernel: one CUDA kernel, full Qwen3 fwd+bwd; 132 blocks x 256 threads (later reality), 100KB dyn smem, cooperative launch for grid.sync.
- In-kernel interpreter: host builds instruction stream; blocks self-schedule (instr, tile).
- Waves executor first (grid.sync ~1.7us between dep-free waves; nano = 84 waves); superseded by dataflow.
- Device op library: templated bf16 WMMA GEMM all layouts, rowops, embed, CE, FA2-style attention.
- Dtypes: bf16 params/acts, fp32 accum everywhere, fp32 weight grads; grad zeroing in-kernel wave 0.
- fp32 workspaces + atomics for grad accumulation; rerun-stable.
- Correctness contract: per-op unit tests vs PyTorch + full-model grad parity + 40-step learning sanity.

## v0 -> EARLY (nano 6.56->2.53ms, small 30.9->12.0ms)
- fp32 WMMA smem strides multiple of 4 (silent corruption else); vectorized coalesced tile loads (+2x); instruction lookup scans offsets only; warp-parallel qk-norm w/ smem-staged atomics (global atomic contention was 8%/step); split-K dW (occupancy); register-prefetch GEMM K-loop pipelining. All KEEP.

## v1 DATAFLOW (nano 2.53->1.93ms)
- df executor: dependency counts from per-op read/write signatures, ready ring, atomic claim cursors, sticky-instruction fast path. Naive ring scanning SLOWER than waves initially.
- %globaltimer per-instr stamps = profiling foundation (clock64 per-SM useless).
- Attn bwd split over GQA+kv-chunks w/ fp32 atomic workspaces: dKV 190->47us. KEEP.
- Split-K lm-head dX K=8192: 145->40us. KEEP. Online CE single-pass KEEP.
- cp.async BK=64 rewrite REVERTED (-10% in-model vs 1-deep register prefetch).
- No launch-bound crossover found vs CUDAGraph (both scale).

## v2 WGMMA
- From-scratch wgmma m64n128k16 validated, ~94% per-SM peak in probe.
- LAW: data-dependent ternary between warpgroup_arrive and commit => ptxas serializes every wgmma (60x); branch-free accumulate mandatory.
- NT gemms via 128x128 2-WG tile: parity but ~nil gains — 255 regs -> 1 block/SM; per-instr fixed cost (~20us) dominates.
- Direct successor handoff DROPPED +14%.
- MN-major wgmma all-major routing bwd: SLOWER (nano 1.85->2.05) despite near-peak mma -> REVERTED. "No math-unit conversion pays at these tile counts."
- m64n64 retile + uint4 epilogue: nano 1.93->1.85. KEEP.
- __launch_bounds__(256,2): nano WORSE 16% (latency pays spills), small BETTER 7% -> kept unbounded.
- First fused epilogue: qk-RMSNorm+RoPE into qkv gemm epilogue (bit8, WG_BN==64==head_dim). KEEP.
- Split-KV attn fwd + combine at C=4: nano neutral, small +5% -> disabled (combine hop + partial traffic).

## v3 P0 MEASUREMENT
- Gap model CORRECTED: nano = 13.6% wait + 86.4% SPAN; hop cheap (3-9us); rest intrinsic op latency + co-scheduling contention.
- Attention #1 lever both configs. Hardened baseline (foreach zero + max-autotune-no-cudagraphs): the goalposts.

## v3 P1 FUSION
- CVT deletion (read fp32 workspaces directly): nano -55us, small KEEP.
- Drow fused into dOatt gemm epilogue (bit10) KEEP marginal.
- SwiGLU paired-column tiles: +88/+297us REMOVED (halved tiles -> doubled serial span; SPAN LAW).
- CE/lse partials in lm_head epilogue: ±noise KEEP ON (cheapens CE hop 5x).
- TOOLING RULE: cuobjdump -res-usage on every device change (bit9 silently spilled 320B, ~8% uniform).
- Interpreter is REG:255 -> 1 block/SM -> 132 blocks (discovered late).
- LESSON: deleting a chain hop pays only its SPAN (waits 2-7us); instruction-granularity fusion exhausted.

## v3 P2 WS PROTOTYPE: 2.77us/hop vs flat 5.69 -> gate PASSED. Producer prefetch before flag poll; producer-side completion; no setmaxnreg (spills).
## v3 P3 df2 REGION-WATERMARK TILE DEPS: built, correct, PARKED: +300-400us both configs — op spans intrinsic-latency-bound, no queueing to overlap. CAS-loop claims quadratic; event-driven wakeup needed; volatile reads on prefix scans.
- qt-outer attention tile order: small -240us KEEP.
## v3 P4a WS SCHEDULER OFFLOAD: gate NOT met (nano 1938 vs df 1907); register tax +97/+385us; at equal budget ws WINS -64/-190.
- HW FACT: registers allocated at 4-warp granularity; >256 threads charged 12 warps. Warp-spec REQUIRES setmaxnreg.
## v3 P5 FA-CLASS WGMMA ATTENTION (big lever): D64 S%128 trio; single no-swizzle 64x64 layout readable under BOTH wgmma majors -> every bwd transpose is a descriptor change. nano 1853->1775, small 9028->7118. ws beats df first time at <=224-reg ops.
- COST SURPRISE: new ops grew shared dispatch switch pressure: df STACK 272->528, ~6% uniform tax.
## v3 P6 ROWOP BATCHING + RINGS (2.06x): MK_ROW_R=8 batched rowops (nano -117, small -722us); hot/cold criticality rings (waits collapsed); dispatch spill tax SOLVED (104B Instr staged to static smem: -125/-484us, ~8% uniform); claim 264->132; RMSNorm bwd row-grad smem partials.
## v3 P4b:
- Deeper mainloops NO GATE; SW128 DISCOVERY (no-swizzle cp.async was 8-way bank conflicted): small NN 516->254us etc. KEEP; NN routing flips positive post-SW128 (tile-gated >=64).
- THE measurement: SM issue 19%, warps in flight 12%, unallocated warp slots 87%, DRAM <10% -> LATENCY-BOUND at 1/8 occupancy.
- REGISTER-LIFETIME LAW (x2): register-resident reuse LOSES to re-reading in 8-warp regime (rmsnorm_bwd single-pass +240us; qknorm register-dw +58us).
- Dual-stream 512-thread executor DIED at compiler: setmaxnreg regions get NO extra ptxas budget; 256 x 255 = 64K register file exactly; "8x255-reg warps is the Pareto point."
- m64n128 NT tiles: small 4252->4095 KEEP (the one lever register budget permits).
- BASELINE RETRACTION: SDPA 3-D tensors silently math-decomposed; honest gaps nano 1.95x, small 2.29x, S4096 2.80x, S8192 3.35x. Long-S crossover retracted.
- Chunked-CE baseline within noise -> attention was the baseline's problem, not CE.
- MPK topology probe (scheduler block + mailboxes): 5.09us/hop vs df 3.0 -> closed. OCC2 (128 regs, 2 blocks): +32-40% loses.
- Micro-op wins (mostly narrow/gated): fast-log/exp2.approx family, float2 direct stores, direct-atomic epilogues, FMA rewrites, sigmoid caches, RMS fixed-width H256, per-shape SKR/dW targets, cold-cap flip 0->16 (shape-gated), idle32 long-S, in-kernel INV_VALID (-70..-93us), launcher input binding (-12..-22us).
- Micro-op NO-GOs: rowop claim floor (+275/+838); DKV x2 batching (+185, register-lifetime); DKV G2 fusion (+221/+455 lost G-parallelism); ONE-PASS BWD FUSED (+119us S4096: store amplification 135MB atomics vs L2-resident re-reads — pass split converts store-amp into cache-absorbed load-amp); MK_ATTN_PIPE (+245); masked-exp skip (+223); n128 direct bf16 store; deep SW128 stages + fat 128x256 tiles REAL standalone wins but blocked by cooperative smem page (208KB/160KB -> launch fail); ws/df2 rechecks (df best); ALLHOT +34/+65.
- LONG-S ROUND: straggler-bound diagnosis (makespan = longest causal tile chain) -> BANDED bwd chunking {2048:12,3072:16,4096:29,8192:40} S8192 -531us; banded fwd {..8192:64} -481us; combine row batching -70us; dq_first band order S8192 only; split-V long-S only. Short-S banding NO-GO.
- LESSON: re-run cheap knob sweeps after every structural change (SwiGLU 2W S8192 flipped +47 -> -128us post-band).
- QWEN4B-L1 program: cold_cap0 L=1 (29433->22094us); D128 dQ C=1 (-85/-95); lm-head n256 direct route (+909us win); head-dX n256 no-atomic (+597); dW n256 TN direct (+3178); D128 WGMMA attention trio (split-D, P-parking, redundant-S; +1875-2081us ~9%/step); no-residual NT bf16 n256 (+229).
- D128 landing traps: cudaFuncSetAttribute cached in statics (stale carveout); ws 256B control overrun (memcheck missed); profiler smem fallback crash.

## ERAS: v0 correctness -> v1 dataflow -> v2 wgmma (fixed costs dominate) -> P0/P1 measure+fuse -> P2-P4a warp-spec (register tax) -> P3 tile deps (parked) -> P5 wgmma attention -> P6 rowops+rings -> P4b SW128 + latency-bound acceptance + micro-op grind -> long-S banded round -> qwen giant-shape program.

## PATTERNS (this slice)
WINS: remove serialization pinning shared resources (smem-staged atomics, direct single-writer stores); split only the straggler (bands, gated split-K); cheap arithmetic in latency-bound regime (SFU/tensor work ~free); short-lived registers + independent streams; cold work off critical dependency; fix bank conflicts (SW128) to unlock previously-useless depth.
FAILURES: adding warps/occupancy (register/spill tax 5-40% > 10-15% hidden — 5+ independent strikes); fusion trading parallelism for hops (swiglu pair, one-pass, G2/x2); overlap/pipeline machinery in 8-warp regime (3+ strikes); tensor-core quality at chain sizes; cross-SM signaling; cooperative smem page ceiling kills real standalone wins.
MEASUREMENT TRAPS: construction-order bias; co-tenancy; soft baselines (SDPA math fallback); silent spills; stale carveouts; globaltimer fp32 granularity; stale ext locks.
