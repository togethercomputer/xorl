# Candidate laws (to VALIDATE against full trial corpus — predict PROMOTED vs NO-GO)

Drafted before reading extraction reports, from residual maps + FA4/DeepGEMM concept maps
+ scoreboards. Each law must be scored: confirming trials / counterexamples / precision.

## H1 — Wait-column law (bottleneck-unit accounting)
A change wins iff it reduces realized critical-path WAIT at the current bottleneck;
static instruction/ALU savings on overlapped or off-path rows are neutral-to-noise.
Predicts: micro-ALU/store-format tweaks on off-path ops → NO-GO; changes that move a
top wait row → PROMOTED.

## H2 — Overlap-preservation law (the PDR lesson)
Restructures that delete existing overlap (bands/LPT tail exposure, off-path absorption)
lose even when they cut logical work substantially (one-pass = 5/7 GEMM work, still
+1.9ms). Two-pass-with-overlap > one-pass-without. Any topology change must carry an
explicit replacement for the overlap it deletes.

## H3 — Register/working-set cliff law
Bodies live near hard register points {168, 224, 240/255}. Additions that push past the
point (or grow stack/frame/LOCAL) fail regardless of algorithmic merit; register-file
accounting is conserved (producer donation only pays when consumers actually use it —
setmaxnreg region idiom). Spill-tax removal alone is insufficient if the working set is
real (wgi frame war).

## H4 — Regime-match law (latency vs throughput)
Sub-wave/short-S shapes respond to BOUNDARY-COUNT cuts (fusion, wider stages, fewer
syncs/hops); machine-filling shapes respond to FEED/DRAIN-RATE work (TMA producer,
EVICT_FIRST stores, deep rings). Applying one regime's mechanism to the other fails.
Includes wave-gate law: fat tiles iff (M/128)(N/128) >= ~132.

## H5 — Itax/DAG-depth law (floor shapes)
Interpreter machinery ~3.4-4us per critical-path hop. At floor shapes (deep/nano/s128)
machinery floor exceeds the whole remaining gap → only true hop DELETION, added
parallelism (mbi), or contract changes can help. Hop REPLACEMENT is neutral. Fusions
that add on-path arithmetic or new serialization (atomics, last-arrival counters) to
delete hops lose (qkrope fold +811us).

## H6 — Atomic/drain-placement law
Hot-path global atomics are poison; fp32 workspace + separate reduce wins. Bulk-reduce
(cp.reduce.async.bulk) drains win only where the completion wait lands OFF the tail
chain (promoted exact-s4096, refuted s8192).

## H7 — At-shape ceiling law
Improvement targets must be at-our-shape measurements, not machine-filling folklore
(FA4 "500 TF/s" is really 389/285/212/137 at our S). Lanes justified by folklore
ceilings overrun their real EV and die.

## H8 — In-model-only measurement law
Standalone/bare-kernel results mispredict in-model outcomes in BOTH directions
(interpreter/route tax ~22%; in-model latency environment +30-50% on attention; TN dW
rows standalone-neutral yet -300us in-model). Only whole-step paired A/B, both
construction orders, fresh processes, is promotable evidence.

## H9 — SASS-visibility gate (compiler-behavior law)
If the intended structural change is not visible in SASS (DEPBAR cadence, HGMMA
grouping, UBLKRED counts), timing will not show it. ptxas lowering is body-context
dependent (qwen NT singleton drain: reproduced only by whole-translation-unit shape;
forceinline wrapper call was the collapse trigger; every plain C++ respelling failed).

## H10 — Routing-over-uniformity law
Per-shape/per-family routing across register points {168,224,255}, executor images
(df/pdf/ws), and per-shape gates beats any uniform design. Uniform rewrites REFUTED;
parked producer WG costs +34..+145us at short-S (residency tax) → gate it by shape.

## H11 — Contract/meter law
A chunk of remaining "gap" is meter policy, not kernel deficiency: graphed-step
(+8-17us/step), native microbatch interleave (beats baseline per-sequence at
s128/s256/nano/deep), baseline bounce ±8%. Distinguish kernel headroom from contract
decisions.

## H12 — Cheap-probe staging law (process)
The pipeline route-audit → SASS/res-usage gate → smoke → paired x8 → x40/x80 both
orders kills bad lanes cheaply; lanes that skipped SASS gates wasted GPU time.
(Process-level; validate by counting where the pipeline caught failures early.)

# Validation TODO
For each law: list trials it correctly classifies (both promotions AND no-gos),
counterexamples, and refine. Target: complete explanation of the underlying dynamics —
what is and is not efficient in this system.
