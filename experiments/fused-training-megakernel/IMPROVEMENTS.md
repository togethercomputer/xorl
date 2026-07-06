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
