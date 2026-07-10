# V2 validation: resource-accounting laws (condensed)

R1 Register-lifetime: ~9/9, 0 overturns. Strongest: DKV RS family lost +85..+105 with IDENTICAL REG/STACK/LOCAL => the lifetime itself is the cost (pure ILP). Boundary: caches that delete a pass (QKBWD cache) are the stated exception. REFINEMENT: register file buys time-shifting, not bandwidth; caching pays iff it deletes a DAG node; loss scales with lifetime span. PROBE: re-run rmsnorm single-pass cache inside pdf 240-reg image — decides machine-law vs 255-reg-image-law.

R2 4-warp/Pareto: ~9/9. Flat caps ALWAYS lose (+30..+307); pdf shows a second Pareto point (168-entry + 240-region) exists. REFINEMENT: register file purchasable only in 4-warp quanta at IMAGE granularity; points {168,224,240-region,255} are different compiled images; decision is discrete routing, never tuning. PROBE: image-partitioned 512-thread executor (the 512-thr death was compiler-budget scoped; partitioning is the unexplored axis).

R3 Residency: 5/5, narrow. Parked WG2 +34 nano even nanosleep'd; pdf value = feed work minus fixed dilution, positive only wait-bound shapes. mbi is the constructive dual (fill idle issue with independent WORK, not protocol warps). PROBE: pdf shell with truly-empty WG2 vs df — separates issue dilution from 168-image cost (currently confounded).

R4 Smem page tax: 5/5 + 1 in-clause boundary (D128 trio funded 120KB page on-path for its whole family). REFINEMENT: two regimes — smooth per-KB tax on EVERY shape + hard coop-launch cliff ~148-160KB; promotion unit is the PAGE, not the op. PROBE: carveout sweep 100->208KB in 16KB steps, no code change (only 3 points exist today).

R5 Dispatch-spill/noinline: ~7/9 MEDIUM — mechanism unrefuted, rule has 2 refuted classes (pdf shell ABI boundary: noinline made LD WORSE 4.83->6.32M; WGMMA call-poison where forceinline pipelines at zero reg delta yet inline rounds all lose in-model). REFINEMENT: the invariant is "no live caller state across an ABI/dispatch boundary"; noinline is one enforcement, sometimes the artifact; the true variable is ptxas per-image global allocation => fixes ultimately need image partitioning. PROBE: 2x2x3 bisect {caller state} x {WGMMA} x {inline/noinline/separate-image}.

R6 STACK-not-runtime: 6/6 (score-mask split WINS with static SASS GROWTH; boundary bodies grow stack with cadence unchanged). Runtime is senior partner, executed local-LD the explainer (DKV family: identical counters, +85 loss). PROBE: regress runtime delta on local-LD sector delta across ~20 recorded trials -> us/Msector by shape.

R7 Register-point routing: ~10/10 in-domain. Same mechanism, opposite sign by shape (SKR -115 small vs +1160 qwen). Boundary: uniform HYGIENE wins (claim132, Instr-smem, SW128) are outside domain (protocol correctness, not operating point). REFINEMENT: no single optimal image exists; architecture IS the routing table {register point x executor x tile route} keyed by (family, shape). PROBE: predict a held-out shape's (s512) routing table from the gate laws, then measure — transfers = science, else curve-fit.

R8 Working-set primacy: 4/4 arcs (~20 lanes) but YOUNG + single-op (attention-bwd). wgi killed 60% of spill tax, still +442; one-pass all-arrangements dead; ~70% residual = S^2-scaled body-internal local traffic at 240-reg wall. CAUTION: demand claim; the algorithmic axis (smaller live-set body) untried — and impossibility claims rot fastest. REFINEMENT: artifact spill (removable) vs demand spill (conserved under rearrangement; exits = smaller-live-set algorithm or bigger register point where its taxes clear). PROBE: FA4-style smaller-live-set restructure at s8192 under 240-reg pdf image — win kills the floor reading; instrumented loss hardens it to physics.

## CROSS-LAW GENERATING REALITY
Measured directly: SM issue 19%, warps-in-flight 12%, 87% warp slots unallocated, DRAM <10%.
=> Each SM runs ONE resident 8-warp CTA owning the entire statically partitioned budget
(64K regs in 4-warp quanta at compiled-image granularity via ptxas global allocation; one
dynamic-smem page with a coop-launch cliff; 8 issue streams), latency-bound, no elastic
resource to hide anything. Every optimization is a zero-sum REPARTITION: repartitions net
to zero minus tax. Only four things ever paid: work deletion, pass deletion, dependent-
chain shortening, per-shape selection of the least-taxed partition. Each era's "law" =
discovery of one more direction in which the single static budget refuses to stretch.
