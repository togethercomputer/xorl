# Hypothesis evolution by era (reconstructed from the recorded belief states)

## Era A — v0-v2 (pre-P0): "make it exist" + wrong gap model
Beliefs held: (a) residual gap = per-instruction fixed cost x chain depth; (b) tensor-core
quality (wgmma everywhere) will close GEMMs; (c) more occupancy = faster.
Fate: (a) CORRECTED in P0 (wait 13.6% vs span 86.4%); (b) REFUTED at chain-gemm sizes
(MN-major routing slower despite near-peak mma); (c) REFUTED repeatedly (launch_bounds,
OCC2, ws, dual-stream).
Survivors from this era: branch-free wgmma accumulate; alignment laws; correctness contract.

## Era B — P0-P4a: measurement first; the machine's true regime
New laws: wait-vs-span gap model; latency-bound at 1/8 occupancy (DRAM <10% — killed the
"bandwidth contention" frame); 4-warp register granularity; 8x255-reg = Pareto point;
REGISTER-LIFETIME LAW (formed on 2 reverts); protocol escapes (MPK, df2 tile deps) closed
empirically — tile deps pay only when tiles >> blocks.
Revision: warp-spec "protocol right, hardware charges" — ws wins at equal registers,
loses with the tax. This later becomes the per-shape register-point ROUTING idea.

## Era C — P5-P6 + P4b: op quality + scheduling; first structural laws
New: dual-major smem layout (transposes = descriptor swaps); dispatch-spill law (Instr in
smem, -8% uniform); hot/cold rings; SW128 ("limiter was fill, not depth" — a hidden-cause
correction); banding/straggler law (largest long-S win); one-pass bwd REFUTED #1
(store-amplification law); register-lifetime law re-confirmed (x2 batching, G2 fusion).
Meta-revision: BASELINE RETRACTION (SDPA 3-D bug) — birth of baseline-honesty law; both-
order + fresh-process discipline institutionalized after ~dozens of order-flips.

## Era D — Jul 5-6 (opgap matrix + probe rounds): at-shape science
New: dual-bound meter (onpath..onpath+offp); gemm_dx = worst bucket -> CLOSED via
per-family register-point routing ladder (SKR + n256 + thresholds) — "routing over
uniformity" proven (uniform 384/168 rewrite REFUTED, producer confined to fitting ops);
wave-gate law (fat tiles iff >=~132 CTAs); boundary-removal-beats-overlap (STSM refuted
by register-lifetime; w128 confirmed standalone); absorption law accumulates strikes 5-8;
NT fwd floor DECLARED (writeback-rate, "not ours to close"); qwen 4.05x -> ~1.2x via n256
cascade + D128 trio (biggest absolute wins of the project; all exact-gated).
Wrong-at-the-time: "flash ~500 TF/s" ceiling (folklore); "one-pass impossible" (drain-
artifact confound); "cluster launch incompatible" (refuted — composes, graph-capturable).

## Era E — Jul 7: executors as routing dimension; ceilings corrected; floors broken
New: pdf producer executor (WG2 TMA producer + setmaxnreg region idiom) — register-point
routing EXTENDED to executors; per-shape gates (small stays df: residency law); TMA feeds
gated by K-length (mechanism-boundary, not shape superstition); FA4 at-shape ground truth
(389/285/212/137 — ceiling correction kills folklore targets); ONE-PASS ARC completes:
standalone GO (old refutation was the drain) -> in-model NO-GO xN (monolithic AND banded)
=> OVERLAP-PRESERVATION LAW: deleting band/LPT exposure loses even at 5/7 GEMM work;
EVICT_FIRST TMA store BREAKS the declared NT floor => floors are mechanism-scoped;
WAIT-COLUMN principle (SKR_REDUCE absorbed in DQ wait slack); itax floor measured (~3.4-4
us/hop; floor > whole gap at short-S) => mbi (microbatch interleave) as the ANSWER to
short-S — add parallelism instead of shaving hops; resweep law weaponized (flips worth 11x
avg promotion); STACK->runtime metric WITHDRAWN (executed local-LD is the metric).

## Era F — Jul 8-9: diminishing returns; compiler-level truths; DAG algebra
New: wait-RELOCATION law (5 scheduler lanes at s8192: wait moves to siblings under the
single-sticky-head executor — scheduler space CLOSED without executor redesign);
call-reached-noinline WGMMA poison fully root-caused (~30 lanes; image-level partitioning
required; ALL source respellings dead); precision COMPOSITIONS win where isolated knobs
lose (small DQ+DKV fp32-P); TRUE DAG deletion via algebraic commutation (RMS fold: norm
commutes through GEMM) with realized-value correction: DAG-cut worth ~1us/hop in
starvation regime, NOT the 3.5us itax price -> only pays where L amortizes sink mass;
dead-fallback compile-pruning = the reliable qwen pattern; hit-rate collapse on
picked-over shapes (~3-8 promotions vs ~70 closes); landing bottleneck (dirty shared
tree) becomes the binding process constraint; qwen-l1 reaches 1.028x (parity threshold).

## Law-revision episodes (dated, cross-checked vs known-evolution-episodes.md)
E1 one-pass: refuted(P4b, store-amp) -> reopened(FA4 map) -> standalone GO -> in-model
   NO-GO x4 -> surviving principle = overlap exposure dominates arithmetic savings.
E2 NT floor: declared(Jul-6) -> broken(Jul-7, store-policy axis unexplored) -> in-model
   win EXCEEDS standalone (L2 relief) — floors and standalone deltas are both bounded views.
E3 split-K: "rejected, atomics win"(Jul-5) -> SKR promoted w/ gate map(Jul-7) -> SKR law
   (one giant sub-wave long-K gemm, one reduce; K/CTAs>=~90).
E4 coop/cluster launch: "incompatible" -> composes -> graph-capturable (+8-17us).
E5 qwen: 4.05x -> 1.028x entirely via routing + feeds + pruning; zero exotic math.
E6 dW TN TMA: standalone-neutral -> -300us in-model (off-path-is-not-free law).
E7 wgi frame war: spill tax killed 60%, still 0 wins — working set is real; binding
   constraint identification failure mode.
E8 qwen NT / attention singleton drain: ~30-lane root cause; compiler-level, not source.
E9 fusion algebra: bit8 qkrope fwd fold WIN vs qkrope BWD fold +811us vs RMS fwd fold
   WIN(deep/small only) — commutation + no-new-serialization + L-amortization decide.
E10 producer residency: pdf wins long-S/qwen, taxes short-S (+34..+145) — per-shape image.

## Trajectory observations (for projection)
1. Laws have moved from SCHEDULING-level (eras B-C) -> ROUTING-level (D) -> EXECUTOR-level
   (E) -> COMPILER/IMAGE-level and DAG-ALGEBRA-level (F). Each era's residual migrated
   down one level of the stack.
2. Refutation half-life: micro-knob verdicts rot fastest (resweep law); mechanism laws
   (register-lifetime, absorption) have never been overturned, only refined at boundaries
   (e.g., cache-to-delete-a-second-pass exception; off-path-is-not-free exception).
3. "Impossibility" claims rot fastest of all (one-pass, NT floor, cluster launch, split-K)
   — every one was mechanism-scoped, not physics-scoped, and fell to an unexplored axis.
4. The stable core (never overturned): latency-bound regime, register-file conservation,
   absorption/wait-column accounting, exact-shape gating, both-order measurement.
