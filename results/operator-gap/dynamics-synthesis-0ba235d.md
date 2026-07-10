# Underlying dynamics of the training megakernel: validated laws, gap map, and forward portfolio

Date: 2026-07-09 (revised same day after review). Head: `0ba235d` (megakernel branch).

Method: full-corpus READ of AGENT-COORDINATION.md + both archives + IMPROVEMENTS.md
+ NOTES.md (~3.3MB, ~500+ verdict-bearing entries, v0 through Jul-9), time-sliced into
six eras and distilled into era inventories; per-era hypothesis reconstruction from
recorded belief states; adversarial validation of the law set against that distilled
record; projection of the law trajectory onto a future-trial portfolio. EVIDENCE
STATUS: the follow-up pass completed the row-level extraction — **`trials_full.csv`
holds all 2,313 verdict-bearing corpus entries, each with source file + line number**
(spot-check-verified). The synthesis layer on top remains curated: 126 deduplicated
ledger rows + 32 evidence events are connected to the laws by 177 links; precision
figures are the validation reports' claims, mechanically auditable through the
law-links join and, at full depth, against trials_full.csv source lines (see package
README for schemas and duplication semantics).

**Evidence package (audit trail): `results/operator-gap/dynamics-synthesis-0ba235d/`**
- `README.md` — scope, honest claim, schemas, SUPERSESSION MAP (read this first).
- `laws.csv` — 39 laws (C1/C2 split out of P5) with status, precision, counterexamples,
  refinement, probe.
- `trial_ledger.csv` — terminal verdict rows (normalized, split-by-shape, landed status).
- `evidence_events.csv` — non-terminal evidence (measurements, SASS proofs, scaffolds,
  standalone probes, law declarations, retractions, parked candidates).
- `law_trial_links.csv` — law_id x ref_id x role(confirm|counterexample|boundary|probe)
  x expected/observed verdict: the mechanical audit path for the precision claims.
- `inventory/` — the 8 era-slice extraction reports (source-file line ranges named;
  not retro-edited — see README for known superseded statements).
- `validation/` — the 4 per-domain adversarial validation reports.
- `laws-evolution.md`, `known-evolution-episodes.md` — the era-by-era belief trajectory.
- `candidate-laws.md` — the pre-registered H1-H12 hypothesis set (written before corpus
  contact) whose scoring is in `validation/validation-V4-epistemics.md`.

## 0. CHANGED SINCE FIRST WRITE (2026-07-09, later the same day)

1. **SwiGLU-BWD fold: PROMOTED at nano+deep+small** (deep -26.5/-27.6us cp 189->177,
   small -82.6/-86.2 cp 130->122, nano -20.0/-16.6 cp 77->73; 40/40 both orders;
   bit-exact parity at small). Portfolio item 3 is DONE, and its info-gain fired: nano
   won this fold after losing the RMS fold at comparable dcp — **the DAG-cut value law
   is hereby revised: zero-added-mass hop deletion pays at EVERY chain depth;
   sink-buying deletion pays only where L amortizes the sink mass** (board
   20260709T021106Z; results/operator-gap/dagdepth-swbf-0ba235d.md).
2. **CE fwd warp-per-row (warprow): evidence COMPLETE** — s8192 -94.0/-105.1 (16/16 x2),
   s4096 -39.4/-33.8 (40/40 x2), s3072 -38.8/-40.2, s2048 -28.5; nano/small wash never
   negative. Portfolio item 4 is validated; landing parked behind the shared-tree
   supertile WIP (board 20260709T0420Z/0445Z; wt-cewr commit c8569a1 cherry-pickable).
3. **qwen NT pruned-image SASS proof**: removing unrelated fallback dispatch/op bodies
   from the focused launched image RECOVERS grouped WGMMA issue (HGMMA max-run 8,
   hist {8:1}, CALL=0, STACK=0 vs default {1:216}) — not a production promotion (it
   no-ops non-matching rows), but decisive for routing: the singleton drain is a
   property of the FULL-image composition, and image partitioning is a live mechanism
   (results/operator-gap/qwen-nt-pruned-0ba235d.md).
4. **qwen NT sidecar: launch ABI exported + host split/rejoin contract proven** —
   cutpoint at instr 37, `valid_topological_split=true`, `runnable_now=false`; next
   step is a typed boundary/no-op or partial skip/resume executor, then one-tile
   sidecar correctness (results/operator-gap/qwen-nt-sidecar-{launchabi,splitplan}-0ba235d.md).
   Portfolio item 2 has moved from "prototype" to "contract in hand."
5. **LATER SAME DAY — portfolio item 2's ATTENTION branch CLOSED (projection's LOSES
   arm fired)**: the split-entry SASS probe showed a clone entry reaching ONLY
   OP_ATTN_DQ_WG (CALL=0, fully inlined) STILL emits {1:12} singleton cadence — the
   attention drain is BODY-STRUCTURAL, not image-level call-poison; subsequently
   root-caused to ptxas 13.1 descriptor-slot management (heuristic, not pressure;
   non-constructible at 13.1; standing gate = rerun the bisect on every CUDA toolkit
   bump). Band-preserving one-pass at s4096 also killed by cell arithmetic. Long-S
   attention-bwd wait reclassifies toward a CONSERVATION floor; remaining levers are
   contract-level (native-mbi long-S) + FA4-class in-body restructures. The qwen
   sidecar/image lane (item 2's other branch) is unaffected in mechanism but shares
   the reduced upside. See attn-splitentry-sass-0ba235d-refuted.md,
   attn-hgmma-drain-mechanism-0ba235d.md, onepass-band-pdf-audit-0ba235d.md.
6. **CE warprow + qknorm vec8 LANDED** (3b5701e; -105us + -130us at s8192, s8192
   ~1.90x); precision payload v2.1 (~-235us over 5 shapes) integration-ready;
   composed dagdepth cert @e91f729: deep 1.234 (new floor), s8192 1.948, qwen-l1
   1.031 — every H-family absolute improved vs 05e9e97.
7. **The live ranked state now lives in
   `results/operator-gap/frontier-20260709-postsession.md`** (supersedes this note's
   section 4 ordering); this note remains the law model + evidence package + the
   record of how the projection resolved.

## 1. VALIDATED CURRENT STATE: gap map

Provenance: ratios are the fresh series adopted into `0ba235d`
(`talk/facts.json` fresh block). **Non-qwen rows certified @`05e9e97`**
(remote isolated dedicated H100, median-of-50, fresh process/shape;
`results/operator-gap/postburst-scoreboard-05e9e97.md`,
`results/operator-gap/talk-fresh-sweep-jul8-postburst-20260708T1020Z.log`).
**Qwen rows @`6d4e86d` refresh** (results/operator-gap/qwen-postntpdf-score-profile-6d4e86d.md).
Exposure columns from the certified reanchor residual map
(`results/operator-gap/certified-reanchor-residual-map-05e9e97.md`) and the
s8192/s4096 profile refreshes at `0ba235d`/`ed49255`.

| shape | gap | abs us | binding constraint (source) |
|---|---|---:|---|
| qwen-l1 | 1.028x | 205 | parity threshold post NT-supertile-PDF (qwen-postntpdf @6d4e86d) |
| qwen-l2 | 1.109x | 972 | NT/head-dX/RMS spans, 5.1% wait (post-pair2 triage @6d4e86d) |
| deep | 1.267x | 472 | itax floor: 213 cp hops x ~3.5us (itax note; dagdepth census) |
| nano/s128/s256 | 1.38-1.42x | ~220 | itax floor = 110-133% of gap (itax decomposition) |
| small | 1.533x | 1020 | D64 attn bodies + RMS dX 103 + lm-head 201 (reanchor map) |
| s1024 | 1.593x | 463 | attn-fwd 123 + lm-head 113 (reanchor map) |
| s2048 | 1.641x | 672 | DQ wait 144 + attn-fwd 125 + lm-head 78 + CE 76 (reanchor map) |
| s3072 | 1.704x | 930 | DQ wait 278 + attn-fwd 184 + CE 115 (reanchor map) |
| s4096 | 1.849x | 1350 | DKV wait 465 + attn-fwd 288 + CE 150 (reanchor + s4096-post-rsfeed) |
| s8192 | 1.942x | 2996 | DQ wait 1751 + attn-fwd 1024 + CE 285 + lm-head 503 (s8192 refresh @0ba235d) |

Program arc: Jul-5 matrix 1.47-2.30x + qwen 4.05x -> Jul-8 certified 1.03-1.94x; every
cell improved at every certification.

Root causes (each with closed-lane evidence in trials.csv):
- Long-S attention-bwd wait: two-pass recompute structurally required in the CURRENT
  banded/LPT schedule family (one-pass arc: 6 in-model variants incl banded PDR lost);
  scheduler priority space closed under the CURRENT single-sticky-head executor
  (wait-relocation, 5 lanes); WGMMA singleton drain — TWO DISTINCT MECHANISMS (laws
  C1/C2): the qwen NT drain is launched-image COMPOSITION (source respelling dead;
  pruned image recovers grouped issue — qwen-nt-pruned proof; runnable sidecar split
  pending), while the attention D64 drain is BODY-STRUCTURAL ptxas 13.1 descriptor-slot
  policy (CALL=0 image isolation still drains; toolchain-bound; toolkit-bump bisect gate
  standing — split-entry + drain-mechanism notes).
- Attention-fwd long-S span: body already carries 3-ring PV pipeline + banding; at-shape
  ceiling is FA4 389 TF/s @s8192.
- Short-S/deep: machinery floor; bodies at baseline parity; levers = DAG deletion (now
  two promoted folds), mbi, meter adoption.
- qwen-l2: span-dominated body residual; dead-fallback pruning still paying; big jumps
  couple to the sidecar/image-partition lane.
- CE long-S: addressed by warprow (validated, parked).

## 2. INTERPRETIVE LAW MODEL (validated against the record; see laws.csv for per-law status)

This section is a MODEL — the best compression of ~500 verdicts — not itself a
measurement. Per-law precision, counterexamples, and falsification probes are in
`dynamics-synthesis-0ba235d/laws.csv` and `validation/`.

One 8-warp, 255-register CTA per SM, latency-bound at ~1/8 occupancy (SM issue 19%,
DRAM <10%), under a greedy single-sticky-head dataflow executor. Step time = makespan of
the realized dependent chain + congestion on three shared currencies (per-SM issue slots;
write-drain serialization; warp/register residency). Resources are statically partitioned
at compiled-image granularity, so every optimization is a zero-sum repartition that nets
to zero-minus-tax unless it (1) deletes work without minting serialization, (2) shortens
the argmax chain, (3) releases a shared currency, or (4) routes a (shape,family) to a
less-taxed partition {register point x executor x route}. Reads ~free (L2 dedup); writes
priced by serialization x tail position; off-path work free in span, real in issue slots.
Mechanism-scoped floors fell 5/5 to unexplored axes; conservation-scoped floors held 3/3.

Law tiers (full detail in laws.csv):
- Near-perfect within their recorded scope: overlap-preservation (8/8),
  fusion algebra (12/13, now including the SWBF sink-mass refinement), itax floor
  (quantitatively predictive via the mbi yield curve), register-lifetime (9/9; loss at
  identical resource counters), straggler+exclusivity (11/12), wait-column (9/10),
  STACK-not-runtime (6/6), both-order gate (0 recorded FP escapes; FN side unpriced).
- High with named boundary clauses: absorption (issue-slot/L2 congestion clause),
  wait-relocation (scoped to the current executor at saturation), residency, smem page
  tax, sink/cold-cap (leaf reclassification), exact-gating (wave/feed knobs only —
  codegen laws promote broadly).
- Moving frontier (highest info-gain): dispatch-spill/noinline boundary (deepest form =
  ptxas per-image allocation -> image partitioning; pruned-image proof supports),
  feed gating (needs displacement + wait-on-operand terms), sibling-closure composition,
  working-set primacy (algorithmic axis untried), SASS gate as a GO signal (0-for-3;
  the sidecar lane is the direct test of whether grouped cadence buys time).

SCOPE DISCIPLINE for strong words in this note: "closed" means closed in the current
full-image/source-shape family, executor, and schedule epoch — per the resweep and
impossibility-rot laws, mechanism-scoped closures carry a half-life and reopen when a
genuinely new axis lands (store policy, drain type, image composition). The do-not-spend
list below inherits this scoping.

## 3. Evolution and projection

Constraint migration per era: scheduling (P0-P6) -> per-shape routing (Jul 5-6) ->
executor images (Jul 7) -> compiler-image composition + DAG algebra (Jul 8-9).
Refutation half-life: knob verdicts rot fastest (resweep flips ~11x an average
promotion); mechanism laws only gain boundary clauses; mechanism-scoped impossibility
claims fell 4/4, conservation-scoped never.

Projection under the live trials:
- Sidecar/image-partition at qwen-l2 row 37 (and later s8192 attention): WINS => the
  call-poison corollary converts to time, wait-relocation re-scopes to executor
  artifact, ~1.7ms long-S residual opens, and the next law prices cross-image handoff.
  LOSES => grouped cadence is cosmetic; long-S attention-bwd reclassifies toward a
  conservation floor; endgame there becomes contract-level (mbi at long-S).
- SWBF already resolved the DAG-cut question in the optimistic direction: zero-mass
  folds pay everywhere => enumerate remaining commuting folds (fold-vs-cache cross at
  s1024 posted).
- Band-preserving one-pass under pdf at exact-s4096 remains the write-amp deconfounder.
- s512 held-out routing prediction: if the gate laws transfer, stop paying per-gate
  measurements.

## 4. Stack-ranked forward portfolio (updated)

1. LAND the validated stack + one resweep pass: RMS-fwd-fold + SwiGLU-BWD fold
   (cumulative dagdepth deltas deep -51 / small -131 / nano -18us), CE warprow
   (~-105us s8192, -39 s4096, -39 s3072, -28 s2048), elected-TMA dq (s2048/s3072),
   qwen kind6 + head-dX PDF-only (l2 -27..-37), s1024 SKR=4, mbi/graphstep adoption
   decision. Integration throughput remains below verdict throughput — the dirty-tree
   landing bottleneck is still the cheapest us in the program.
2. Sidecar/typed-boundary executor for the qwen-l2 lm-head cutpoint (contract proven,
   runnable_now=false), then the same mechanism class at s8192/s4096 attention bands.
   Highest EV x info-gain; the pruned-image SASS proof de-risks the cadence half.
3. Remaining zero-mass DAG folds (fold-vs-cache cross at s1024 first; QKRoPE DQ-only
   isolate if it can be made counter-free).
4. Zero-GPU law->predictor analyses: per-sibling wait attribution in the standard meter;
   feed 2-feature separability backtest; s512 routing prediction; runtime-vs-local-LD
   regression.
5. qwen-l2 dead-fallback pruning continuation.
6. Sibling-closure 2^3 precision lattice at s2048.

Do-not-spend (scoped per section 2's discipline — closed in the current image family /
executor / schedule epoch): s8192 scheduler knobs and ready-order priorities; monolithic
or banded one-pass outside the band-preserving pdf frame; WGMMA cadence source
respelling and attribute-inlining; broad TMA at short-K; occupancy adders; in-place
ping-pong; DKV RS/split-wait family; qwen CE reducer variants (superseded by warprow).

## 5. Process epistemics

False positives were measurement-construction artifacts (order bias ~18/era, sub-6us
40-rep phantoms, co-tenancy, soft baselines); false negatives were staleness artifacts
(resweep flips, impossibility rot, standalone-neutral in-model wins). Keep the
deterministic kill ladder (route -> SASS/executed counters -> contract audit ->
both-order paired timing escalating only inside the noise band -> forced-old confirm);
spend reclaimed GPU on the FN side (scheduled marginal-verdict resweeps, expiry-dated
NO-GOs with unexplored-axis lists, baseline fingerprinting per cert).
Four leaks: (1) integration < verdict throughput (now gating ~6 validated results);
(2) noise-gate FN rate at floor shapes unmeasured; (3) no per-sibling wait attribution
in the standard meter (would have killed the 5-lane scheduler family after lane one);
(4) era-F hit-rate collapse says lane SELECTION is where GPU-hours die — the corpus
supports a scored triage (wait-slack + smem-page delta + SASS visibility predicts ~80%
of NO-GOs) currently applied as folklore.
