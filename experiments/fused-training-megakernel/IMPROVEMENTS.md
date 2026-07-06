# Improvements ledger

Every proposed improvement, its measured verdict, the mechanism behind the
outcome, and the transferable principle. One entry per idea — including the
failures, especially the failures. Sessions MUST append an entry when landing
a promotion or recording a no-go (a one-liner here + the full record in
NOTES.md). Format:

> **Name** — VERDICT (shapes, numbers, commit/log)
> Why: the mechanism that made it win or lose.
> Principle: what generalizes.

Verdicts: WIN (landed), NO-GO (measured worse/neutral, recorded), NEUTRAL
(kept or dropped for other reasons), SUPERSEDED.

## Scheduling / dispatch

**Noinline fat attention ops (D=128 five + D=64 trio)** — WIN (every shape;
small 3660->3511, S8192 -502 vs cert; 6b78314+c9d77d5)
Why: all ops inline into ONE kernel function; ptxas solves one global register
allocation, so fat op frames push SPILLS into the scheduler claim loop — ops
that never execute on a shape still tax it.
Principle: fat op bodies must be `__noinline__` before entering the dispatch
switch; certify with executed local-LD sectors (shape-scoped: small==0,
long-S by control-equality), never STACK/res-usage alone.

**n256 fanout trampoline** — NO-GO (neutral twice, above and below the
pressure cliff)
Why: call-site COUNT was not the pressure source; fat inline frames were.
Principle: isolating flag-decode is worthless; isolate REGISTER-FAT bodies.

**Dead-code compile gating (n256 family)** — NO-GO (removing 500+ never-executed
lines made small WORSE, 3662->3736)
Why: code volume is free when hot codegen is untouched; the probe's noreturn
trap perturbed the hot dispatch CFG instead.
Principle: never certify by binary size or STACK; only runtime + executed-LD.

**Band/claim/order knob sweeps at S8192 bwd attention** — NO-GO (many, all
sessions)
Why: packing math showed bands already 77-83% packed vs the theoretical
minimum makespan; knobs can recover <=10%.
Principle: before sweeping scheduling knobs on a wait-heavy bucket, compute
packing efficiency from the iclk timeline (total stage-units x per-stage cost
/ 132 blocks vs actual window) — it splits scheduling from op cost in minutes.

**Small idle_ns 256->64** — WIN (small ~-20us; 32/64/128 ALL beat 256 in 10
paired runs post-SKR; fe0fe19)
Why: SKR+n256 shortened small's hop structure; the ready-ring poll cadence
tuned for the old hop lengths left idle blocks sleeping through short waits.
Principle: poll cadence is a resweep-law knob — recheck it at a shape after
any promotion that changes that shape's hop-length distribution.

**MK_CLAIM 64/264 + cold_cap 8/32 at small (post-SKR recheck)** — NO-GO
(claim: +468/+131 0/40; cold_cap: died at 120 reps after 40-rep positives)
Why: claim quantum verdicts are robust to today's structural changes; the
cold_cap "wins" were window noise.
Principle: sub-6us 40-rep verdicts are NOISE — require 120+ reps or a
double-order margin >2x the window drift before believing knob deltas.

## Attention op bodies

**S/dP one-commit batch + fused ALU (dkv+dq)** — WIN (all D64 shapes; S8192
-113/-94 both orders; 5b8b3db)
Why: the two independent gemms were serialized only by sharing one fp32
register bank (an old register diet); batching them deletes a full warpgroup
drain and a P smem round-trip from every 64-row stage.
Principle: check INDEPENDENT gemms sharing accumulator banks — the register
diet that avoids spill can cost a serial drain per stage; +32 banks is often
affordable post-noinline.

**Fwd PV cross-stage pipeline (triple-buffered K/V)** — WIN, small (S8192
-47 isolated, small -17; bitwise parity; 1e8420a)
Why: the PV drain was already mostly absorbed by co-scheduled work; only the
long-S fwd-bound residue paid.
Principle: drain-hiding pays only where the op monopolizes the machine;
absorption eats it elsewhere (absorption-ledger entry #8).

**QKNORM_ROPE_BWD per-warp dw partials** — WIN (small -21, S4096 -17, S8192
-43/-66; 61e37bd)
Why: every lane of every warp atomicAdd'ed into ONE shared [D] array —
block-wide serialization on 64 addresses, ~30x above bandwidth floor.
Principle: block-shared smem accumulators with per-lane atomics serialize;
use per-warp slices + one cross-warp reduce. Grep `atomicAdd(&*_s[` when
writing rowops.

## Routing / tiles

**n256 direct routes for long-S head block** — WIN (S3072 -52/-58, S4096
-45/-54, S8192 -245/-247; 43803eb)
Why: the lm-head fwd ran at 78 TF/s on the generic m64n64 tile; the
giant-N direct-store tile existed but was exact-gated to qwen shapes.
Principle: exact-shape gates rot — when a new shape class appears (or a
profile shows a gemm far off roofline), re-audit which existing routes could
serve it. Gate by measurement: the same route LOSES at small/S2048 (+99/+66).

**Epilogue fusion phase 1 (MLP rmsnorm dissolution)** — NO-GO (long-S +10-19
worse, small wash; critical_path unchanged; e117807)
Why: the rstd-only op replaced the rmsnorm at the SAME dep-chain hop count —
hop REPLACEMENT is not hop DELETION; the rowop span was absorbed anyway.
Principle: fusion pays only if the consumer's dependency lands with the
PRODUCER (true hop deletion) or deletes real traffic; count chain hops before
building.

**head-dX SKR in-model port (splitK + separate reduce)** — WIN small skr=2
(-115/-112 40/40 x4, 193a108), nano/deep skr=4 (-12, -31/-35; 0abc08f);
NO-GO s128/s256 (+36/+12: n128 tile collapses at M<=256), s1024 (noise),
s2048 (+18: direct route stands), per-layer gate_up dX (+83/+65 0/80)
Why: K-splitting a HALF-WAVE long-K head gemm doubles tile parallelism to one
wave and halves the serial K chain, for ONE ~11us reduce hop; per-layer
application multiplies the reduce tax past the split gain; replaces zero-fill
+ fp32-atomic epilogues (the round-12 16-30us atomic tax) at nano/deep.
Principle: SKR pays iff ONE giant sub-wave long-K gemm amortizes ONE reduce;
never apply per-layer. The portable pieces of a standalone ladder (K-split +
plain slabs + reduce) can beat the ladder's in-model expectation because the
win is wave-filling, which standalone probes cannot see.

**lm_head fwd n256 resweep cascade (small/s1024/s2048)** — WIN (-33/-23,
-11/-12, -25/-21 both orders; 7f77378, 5c6e234)
Why: the 0928Z gauntlet CORRECTLY rejected these cells; the same-day SKR and
mbar-ring/commit-batching promotions restructured the surrounding schedule
and flipped them.
Principle: exact-shape gates rot in BOTH directions — after any structural
promotion, re-run the cheap env probes for neighboring REJECTED routes at the
affected shapes; 4 of this session's 6 promotions were resweep flips found by
2-run env A/Bs.

## Meta / measurement

**STACK-is-not-runtime** — STACK/res-usage is a smell, never certification.
**Absorption ledger** — op-local savings off the critical path don't move the
step; 8 strikes and counting. Ask "does this op monopolize the machine?"
first.
**Innermost-frame attribution** — ncu lineinfo mapping must use the innermost
inline frame; the outermost collapses everything onto the dispatch line.
**First-run-after-build artifact** — the first timed process on a fresh
extension variant can read wildly high (nano +286 fluke); rerun before
believing any single-process outlier.
**Cross-binary noise band** — small has +-40-130us sensitivity to unrelated
codegen shifts; same-binary env A/Bs or same-day controls only.

## GEMM tiles / feeds

**TMA feed for qwen n256 mbar ring (NN dX + TN dW)** — WIN (qwen4b-l1
-622us on merged head, 9695->9073; ported commits 9146c9c+ae6dca2;
`mkv3-p4b-n256tma-*.log`)
Why: one elected thread issuing cp.async.bulk.tensor.2d per ring stage
replaces 12 per-thread cp.async slices + 256 arrivals; tensormaps built
host-side per program. TN dW rows were standalone-NEUTRAL but win -300
in-model — off-path sinks stop burning issue slots the on-path chain needs.
Principle: (1) feed machinery (TMA vs per-thread cp.async) is worth porting
even when tile geometry is already right; (2) standalone-neutral changes to
OFF-PATH ops can still win in-model by freeing issue slots — measure
in-model before rejecting; (3) validated ports carry across divergent heads
when gated per-shape — confirm with a same-instrument control, not the
original session's absolutes.

## Register architecture / warp specialization

(Consolidates the megakernel-paper-style "reallocate registers from task
managers/loaders to consumers" question — asked 2026-07-06; every cell below
was already measured across P2/P4a/P4b-r3 and GEMM rounds 5-12.)

**setmaxnreg reallocation, scheduler WG -> consumers (ws mode)** — WIN inside
`megakernel_ws`, where it is MANDATORY (entry `__maxnreg__(168)` at 384thr;
consumers inc->224, scheduler warpgroup dec->56; without it ptxas spills the
op hot paths at the 168 entry ceiling: REG:168 STACK:544, +14% both configs;
megakernel.cu:702-715, results/mkv3-p4a.md)
Why: H100 charges registers at 4-WARP granularity — any block >256 threads
pays the 65536/384 = 168-reg entry ceiling; reallocation is how consumers
climb back to 224. Needs explicit `-gencode=arch=compute_90a,code=sm_90a`
(plain -arch=sm_90a also embeds compute_90 PTX where setmaxnreg is rejected)
AND an entry maxnreg, or ptxas ignores it.
Principle: reallocation is a prerequisite for warp-spec on this op library,
not a speedup by itself — ws still trails df by a uniform ~8-20%/op, which IS
the 224-vs-255 consumer ceiling (structural).

**Reallocation split sweep (224/56 vs 240/24 vs 512-thr 192/64)** — 224/56
best; 240/24 NO-GO both configs (the dec-24 scheduler spills its
claim/accounting path; the slower handoff costs more than the 16 extra
consumer regs recover); 512-thr dual-stream 192/64 NO-GO (ptxas compiled the
whole image at the 128-reg entry cap, STACK:848).
Why: freed registers pay only if the thin warpgroup's own code fits its
budget — a pure TMA producer fits 24 regs (round-5-proven), a scheduler
needs ~56.
Principle: size dec targets to the thin path's real register need; every
added warpgroup takes its registers straight off the consumer ceiling
(256thr x 255regs = the exact-64K Pareto point of this op library).

**Full MK-paper/nvjet producer topology for GEMM (384/168 + WG2 dec-24 pure
TMA producer)** — WIN standalone / NO-GO as a uniform library point.
Standalone: s8192 dX-head 69.1us/497TF = 1.03x nvjet, +16-36% across the dX
family (pipe_probe_prod.py, results/operator-gap/gemmb-probe-round5.md).
Uniform point: REFUTED by the accumulated register-point map (rounds 5-12 +
attention register-feed rounds: dkv S^T REG:224, supertile 180-224, generic
df ops at 255 are what a uniform 168 ceiling sacrifices).
Why: ptxas gives one kernel image one register point; the producer dividend
is confined to 168-fitting ops = the dX GEMM family, which then reached
1.03-1.29x standalone anyway via splitK+separate-reduce (SKR, round 12) and
elected-thread TMA feeds — both of which fit the 255-pt df image with ZERO
extra warps and are the mechanisms actually promoted in-model.
Principle: harvest loader-decoupling where it fits the image (elected-thread
TMA inside consumer warps) instead of paying a warpgroup for it. Surviving
UNMEASURED cells for the design conversation: (a) per-op-class register
modes (multi-image / launch-select executor variants); (b) single-image
240/24 producer-df — df self-scheduling consumers at 240 (the exact-balance
split: 128*(168-24) == 256*(240-168)) + WG2 as a parked pure-TMA producer
fed by a smem mailbox on GEMM rows only. Cost side bounded by the measured
224-tax (fat ops +4-12% at 224, so <that at 240); win side = the round-5
producer dividend, IF the mailbox handoff beats elected-thread issue
in-model. Nobody has measured (b).
