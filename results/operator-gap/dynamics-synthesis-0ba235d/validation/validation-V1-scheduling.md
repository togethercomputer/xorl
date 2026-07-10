# V1 validation: scheduling/critical-path laws (condensed)

L1 Absorption: ~13 predicts (w128 5th strike, claim=1 6th, PV 8th, mbar-ring port, per-layer scratch, anatomy sub-translation). 2 TRUE counterexamples = off-path-is-not-free (TN dW TMA -294/-333; NT C-store in-model > standalone) — shared-resource (issue slots, L2) effects. PRECISION high (~13/15 with clause). REFINEMENT: only savings that release a shared resource or shorten the argmax chain move the step. PROBE: synthetic off-path work of controlled type (issue vs latency vs L2-heavy) -> quantitative resource boundary for absorption.

L2 Wait-column: ~9 predicts (P0 13.6/86.4; SKR_REDUCE ±3us in 536us slack; sink-steal refuted; dq bulk-red s8192 +55->+98 strengthened; 83%-packed iclk). No true counterexamples. PRECISION high. REFINEMENT: a hop's wait is other ops' overlap; deletable quantity is span; on-path mandatory waits are negative-value to widen. PROBE: pre-registered predicted step delta from reconstructed realized chain before flipping one knob.

L3 Wait-relocation: ~10 no-gos predicted (tail wait 3268->30 total +127; hot-fair wait -1427 span +1629). ~6 boundary wins (qt-outer, dq_first, hot-leaf trio, slot deps) all fed a SPECIFIC starved consumer or locality. PRECISION high in-regime (10/10), medium unscoped (10/16). REFINEMENT: under work-conserving single-sticky-head executor at saturation, priority edits conserve total sibling wait. PROBE: group-aware admission executor at s8192 — THE cluster-level falsifier.

L4 Overlap-preservation: 8/8 (one-pass arc all variants incl banded PDR +428 — banding cannot rescue overlap deletion; L4 dominates L5). REFINEMENT: step prices EXPOSED overlap, not arithmetic; (old span - new span) must exceed overlap destroyed. PROBE: one-pass under pdf producer budget WITH band overlap preserved.

L5 Straggler: ~11/12 (banding -532; SKR gate map incl qwen SKR +1160 fail = no exclusive straggler). REFINEMENT: split only the makespan-owning op, proportionally, and only if freed SMs would idle (sink pool recycling defeats it). PROBE: synthetic straggler injection at fixed tile count.

L6 Hop-replacement/fusion algebra: ~11/11 CLEANEST LAW (epifuse cp 144->144; qkrope fold +811 via added atomics/counter; wins = bit8, CVT, CE partials, RMS fold). commutation + no-new-serialization + L-amortization jointly decide. PROBE: SwiGLU-BWD fold (pure hop deletion, zero sink).

L7 Itax floor: ~8/8, QUANTITATIVELY predictive (mbi yield curve tracks starvation fraction exactly: 1.47/1.75/1.89x n2/3/4 short-S declining to 1.08x s8192). REFINEMENT: short-S gap = cp x hop-price; beat it only by cp reduction, price reduction (executor), or amortization (mbi). PROBE: delete known hop count at s128, check gap closes by cp x 3.4-4us (coefficient never intervened on).

L8 DAG-cut realized value: 6/6 but n=1 family (RMS fold). Tension with itax law resolved by regime-dependent hop value (~3.5us when hops bind, ~1us at starvation). PROBE: SwiGLU fold below L=8 (zero sink) — decides if L8 is independent or collapses into L7.

L9 Sink/cold-cap: ~10/13 naive, 13/13 with leaf classification (hot-leaf promotions were mislabeled sinks on the EMBED_BWD join). REFINEMENT: sink-ness is path-relative and resweep-fragile; re-classify leaves after every structural promotion. PROBE: bracket one op-speed promotion with dense cap sweep -> sensitivity coefficient.

L10 Tile deps: ~6/6 narrow (df2 parked; qkv-v slot deps positive branch). REFINEMENT: granularity should match measured queue backlog. PROBE: df2-style deps scoped ONLY to s8192 attention bands.

## CROSS-LAW MECHANISM (the deep statement)
Machine runs latency-bound at ~1/8 occupancy under a work-conserving single-sticky-head
dataflow executor => step time = span-sum of longest realized dependent chain; everything
else is fungible slack already filled with overlapped work. L1/L2/L3/L4/L8 are five views
of this. L5/L6/L10 are corollaries (only argmax-chain-shortening pays). L7/L9 are boundary
conditions (machinery floor per hop; sinks off-chain until step end).
DEEPEST FORM: step = makespan of realized chain under greedy scheduling + CONGESTION TERMS
on issue slots / L2 / register-file residency. The laws hold where congestion terms are
small; every recorded counterexample IS a congestion term (TN dW TMA, NT C-store, pdf
residency tax). Cluster-level falsifier: group-aware-admission executor at s8192 — attacks
the single-sticky-head clause; if wait genuinely shrinks, L3 re-scopes from machine truth
to executor artifact.
