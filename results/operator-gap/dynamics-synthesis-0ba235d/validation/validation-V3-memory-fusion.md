# V3 validation: memory/feed/fusion laws (condensed)

M1 Write-amp: ~9/12 medium-high. TRUE counterexample: the founding +119us one-pass refutation was later shown DRAIN+CONTEXT artifact (standalone GO with bulk drain) — the in-model kill was overlap deletion, not bytes. REFINEMENT: price writes by (serialization mechanism x tail position), not amplification; L2 absorbs load side nearly free. PROBE: one-pass in-model at exact-s4096 in a band-preserving schedule — separates write-path cost from overlap deletion.

M2 L2-free-multicast: 4/4 high, small sample. Boundary: L2 is a free READ-multicast, not a free write-buffer (store-EVICT_FIRST wins, load-EVICT_FIRST harmful). PROBE: force claim-order dispersion, measure DRAM re-fetch -> co-residency horizon.

M3 Boundary-removal > overlap-machinery: ~15/18 high in-scope. TRUE counterexamples: tail-barrier elision +186 (barrier was data dependency, not seam); pdf producer = overlap machinery that WON (but in feed-bound regime, taxing latency-bound shapes exactly as predicted). REFINEMENT: boundary deletion pays iff the boundary is a sync convention (not data dep) and freed span isn't absorbed; overlap machinery pays only with a decoupled issue slot + feed-bound stages. PROBE: sweep mbar ring across shape ladder recording (K/stage, occupancy) crossover.

M4 Wave/filling: ~12/14 high with EXCLUSIVITY AMENDMENT (qwen SKR +1160 at giant K: sub-wave SMs already recycled into sink pool — filling accounting must count ALL schedulable work; hole filled by sinks is not a hole). Gates K/CTAs>=~90 and >=~132 CTAs have no post-amendment violations. PROBE: sweep off-path sink mass at fixed shape, measure SKR win decay -> exclusivity coefficient.

M5 TMA/feed K-gating: ~7/10 medium as K-length alone; near-perfect with displacement + wait-on-operand terms. TRUE: s8192 not feed-hostile (WG2 cp.async dq-feed promoted there; TMA loss was FEED DISPLACEMENT of the incumbent). REFINEMENT: feed pays iff bytes/stage amortize issue overhead AND consumer stalls on that operand AND no incumbent displacement. PROBE: 2-feature separability test (bytes/stage, consumer wait) over all ledger feed trials.

M6 Off-path-not-free: ~9/9 high. Resolution of tension with absorption: off-path work is free in SPAN, expensive in ISSUE SLOTS/residency/feed contention. Delta-span absorbed & worthless; delta-issue-pressure real and signed both ways. PROBE: per-SM issue-slot attribution before/after TN dW TMA; synthetic re-injection.

M7 Floors-mechanism-scoped: 5/5 mechanism floors FELL (NT store policy, one-pass drain, split-K SKR, coop launch, 500TF folklore); 3/3 conservation floors HELD (itax hop quantum, 240-reg working set, latency-bound regime). REFINEMENT: classify floors mechanism vs conservation; only mechanism floors are attack targets; enumerate untested axes before declaring. PROBE: singleton-drain image partitioning at s8192 = the direct M7 test of the current "attention-bwd throughput" floor.

M8 Fusion algebra: ~12/13 high; every loss has a clause autopsy (b: qkrope fold +811 serialization; c: epifuse cp 144->144 replacement; cache-state: partials-fed RMS). Add clause (d): price deletion at the MEASURED wait-column rate (~1us starvation vs 3.5 nominal), re-evaluate after route changes. PROBE: SwiGLU-BWD fold = clean discriminator (all clauses satisfied, zero sink) — if fails below L=8, L-amortization dominates even at zero cost.

M9 Precision-composition: ~5-6 confirm + 2 INVERSE (qkrope register-epilogue sibling-positive -> shared wash; pdf+gates). REFINEMENT: knob sign is a property of the SIBLING-CLOSURE — knobs shifting work between parallel critical-path siblings (DQ/DKV) must be evaluated on the group; single knob swaps which sibling is critical. PROBE: full 2^3 knob lattice at s2048 predicted from single-knob path-swap telemetry.

M10 Sparsity: ~4/4 above mass threshold, 5 correct refusals below (s1024 x4 order-mixed). REFINEMENT: special case of delete-step-invariant-work, priced by (invariant bytes/step time) x path position. PROBE: tabulate fill bytes vs win across shapes, test the boundary shape.

## CROSS-LAW: the machine does not sell FLOPs or bytes — it sells ISSUE SLOTS and
CRITICAL-PATH NANOSECONDS, and L2 gives the read side away free. Reads ~free (L2 dedup +
re-reads beat caching); writes priced by drain serialization x tail position; off-path
costs issue slots not span; filling counts sink work; feeds pay only where bytes/stage
amortize the issue slot they consume; fusion pays only on real DAG-edge deletion without
new serialization, valued at the local wait-column rate; knob signs live at sibling-
closure level. Durable model = conservation-based pricing; mechanism claims provisional.
