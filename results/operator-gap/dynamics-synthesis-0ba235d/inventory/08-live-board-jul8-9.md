# Era slice: LIVE BOARD (both halves; Jul 8 late -> Jul 9: singleton-drain root-cause, boundary nodes, folds, precision compositions)

## PROMOTED (only ~8 in this era — hit-rate collapsing)
- s1024 FWD score-mask split (-2.9/-5.6 x80; dynamic branch-skip; static SASS GROWS).
- qwen head-dX PDF-only exact body (-27..-37; deletes dead non-PDF TMA path; UTMALDG -75).
- qwen NT supertile PDF-only compile-prune (L1 -108/-69, L2 -72/-85; same dead-fallback-prune mechanism).
- qwen NT postinit-nosync L2-only (-43/-25; deletes provably-redundant barrier; L1 order-mixed left off).
- short-GEMM N64 fast-dispatch shape-gated (s128 +9.7/+11.6 ... deep +57/+43 forced-old).
- small DQ+DKV precision composition (DQ_RS_FEED+DQ_FP32_P+DKV_FP32_P): -33/-31 40/40 — COMPOSITION wins where each isolated knob is net-negative (isolated DKV fp32-P moved path to DQ +6.8).
- RMS-fwd-fold (MK_RMS_FWD_FOLD, true DAG deletion — norm commutes into consumer GEMM; pre-scaled weight sink + epilogue row-scale): deep -22/-25, small -46/-46 40/40; NO-GO nano/s128/s256/s1024 — realized DAG-cut value ~1us/hop in starvation regime, not the 3.5us itax price; landing PARKED on peer WIP.
- elected-TMA DQ staging s2048 (-14/-3) s3072 (-7/-9) promoted; s8192 kind-5 producer NO-GO (off-path). LANDING BLOCKED on dirty tree.
- PARKED-READY: native microbatches port (1.18-1.42x); graphed microbatches (s128 n=4 352us/seq).

## NO-GO / structural findings
- S8192 scheduler family (5 lanes): DQ-tail, tail-pair, DQ-family, hot-fair-quantum, DKV-band — wait RELOCATES to siblings, never shrinks (e.g. tail wait 3268->30 but total +127; hot-fair wait -1427 but span +1629). Root: executor single-sticky-head/first-hot-dependent contract. Multi-sibling fanout = executor-structural, needs group-aware admission.
- small selective/joint banding NO-GO (+57/+46 — must cut DQ+DKV together, then still no); DKV fp32-P isolated NO-GO (path swaps to DQ).
- qkrope register-epilogue: sibling-positive, shared-composition WASH (reverted; deep parity fail 0.031).
- RMS-dX dot-partials (+30..+51 — producer GEMM span grows); s256 RMS FMA order-mixed.
- QKRoPE-BWD fold body (candidate 2 of dagdepth): correctness-PASS but +811us (!) — direct per-row atomics + DKV last-arrival counter erase DAG-depth benefit; smem-partials variant still +88; DQ-only isolate in flight.
- Boundary-node program (itax): full scaffolding built parity-clean (H256/n256 typed boundary ops, replay bridges, DAG rewrite proofs) but ALL replay bodies NO-GO — stack cliffs (176->272/320/416/528), cadence unchanged max=1, one NaN (missing materialization). Correctness baselines only; next = true split protocols, not full-body replay.
- WGMMA singleton-drain root cause (~30 diagnostic lanes, the era's big science): ANY WGMMA body reachable via function call in the launched image => per-HGMMA DEPBAR singleton (ptxas C7510). Triggers isolated: DKV = runtime skip predicate q0s<kv0wg; DQ = RS-feed dQ WGMMA group; FWD = two real D64 WGMMA groups in one body; qwen NT = forceinline wrapper CALL; n256 = ANY non-n256 WGMMA-family body in callable image. Separate-global/standalone code always groups (max=4..16). ALL source-level fixes failed: inline attrs, direct in-body, raw asm, descriptor forms, split dispatch, call-free pruned image (still singleton with CALL=0 for attention bodies), top-level embedding. _pwgi1 WGMMA-isolated image restores grouping 40/16 max=4 but can't execute non-n256 rows. => grouping requires image-level partitioning (replay/boundary executor or separate kernels), not source spelling.
- ACTIVE at capture: qwen NT pruned image SASS proof; H256 CE pair op-body lane (s4096/s8192 CE_FWD 137 + CE_BWD 145 vs baseline ~90+92); QKRoPE DQ-only fold isolate; SwiGLU-BWD epilogue fold (dagdepth round 2, zero sink cost, may extend below L=8).

## PATTERNS
- Wait-relocation (5); call-reached-noinline WGMMA poison (~15 lanes, fully root-caused); sibling-positive->shared-wash; full-body replay adds stack without cadence (6); atomic-count overhead kills epilogue folds (+811, +88); pybind export-name-length trap (4 lanes).
- WINS: compile-prune dead fallback paths (the reliable qwen pattern); delete provably-redundant barrier; dynamic branch-skip; bypass interpreter for tiny rows; combined precision compositions; true DAG deletion via algebraic commutation (RMS fold) gated by L amortization.
- Meta: every promoted win REMOVES work or a barrier; every attempt to REARRANGE WGMMA issue/wait cadence at source level failed.
- Verdict counts this era: ~3-8 PROMOTED vs ~18 NO-GO + ~10 ROUTE-AWAY + ~40 CLOSED-BY-ANALYSIS — the survivor-yield per lane is collapsing on picked-over shapes; landing bottleneck (dirty shared tree) now gates 3 ready promotions.
