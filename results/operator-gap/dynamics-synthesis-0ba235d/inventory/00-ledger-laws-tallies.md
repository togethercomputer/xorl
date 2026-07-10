# IMPROVEMENTS.md ledger: stated laws (verbatim-condensed) + tallies
(Authoritative distilled record, ~223 entries. Use as the primary law list for validation.)

## STATED LAWS
1. Register-lifetime law: prefer re-reads + extra independent load streams over register caching at 8 warps/SM; long register lifetimes block load overlap; only dependent-chain shortening pays. Corollaries: don't trade parallel axes for traffic; cache to delete a second pass, not to avoid a re-read.
2. Dispatch-spill law (Laine-2013): no caller state live across the fat dispatch switch; stage claim state in smem.
3. Absorption law (8+ strikes): op-local savings off the realized critical path don't move the step; ask "does this op monopolize the machine?"
4. Resweep law: every structural promotion invalidates neighboring knob AND gate verdicts; re-run cheap env sweeps; flips both directions are the cheapest wins.
5. Stream-K/claim-batch law: claim size trades overhead vs tail imbalance; don't batch small-tile claims; 132 optimal.
6. Tile-granular-dependency law: tile deps pay only when tiles >> blocks; measure queue backlog before building overlap machinery.
7. Global smem page tax law: dynamic-smem request taxes EVERY shape (+6..+95us); 160KB fails coop launch; new big-smem routes must share an already-funded page.
8. Split-K law: converts occupancy holes into atomics; only while tiles << blocks; kill sk==1 atomics on sight.
9. ptxas setmaxnreg region idiom: inc-region honored only as if(consumer){inc;body;return;}dec; check max SASS register index not res-usage.
10. 4-warp register granularity / Pareto point: >256thr pays 168-entry ceiling; 256thr x 255regs = exact-64K Pareto; register/executor point is a PER-SHAPE routing decision.
11. Residency law: extra resident warps dilute per-SM issue even parked (+34..+145us short-S).
12. itax hop/floor law: floor = cp x ~3.4-4.0us/hop (df; 5.4-6.2 pdf); deleted hop worth span + ~3.5us; floor EXCEEDS whole gap at s128/s256/nano/deep; 74-79% claim+idle.
13. Wait-vs-span (gap model): split every hop into wait vs span; deleting a hop pays only its span; WAIT-column principle — check the wait column before optimizing a hop.
14. Hop-replacement law: fusion pays only on true hop DELETION or real traffic deletion; replacement at same depth does nothing; don't fuse ops whose spans concatenate.
15. Write-amplification law: count write amplification before fusing passes; atomic stores don't cache; L2-resident re-reads beat DRAM-serialized atomic stores.
16. L2-is-a-free-multicast law: co-scheduled tiles get L2 dedup free; measure DRAM re-fetch before building multicast.
17. Both-order + confirmation gate law: paired A/B both construction orders + promoted-vs-forced-old confirm; one-order win = bias; sub-6us/40-rep = noise (need 120+ reps).
18. STACK-is-not-runtime law: certify with runtime + executed local-LD sectors only.
19. Baseline-honesty law: profile the baseline too (SDPA 3-D bug poisoned everything).
20. Measurement-hygiene law: env guards, private build dirs, fresh process; periodic full certification catches uniform drift invisible to A/Bs.
21. Sink-under-parallelization/cold-cap law: sinks only need to finish before step end; cap trade is two-sided, moves with op speed/shape.
22. Noinline + reflow laws: fat op bodies __noinline__ before dispatch; never-taken alternate paths tax hot path (separate loops); noinline does NOT generalize to ABI call-boundary local traffic (pdf shell — fix state placement).
23. Accounting-fence law: the accounting __threadfence is the data-visibility point; no localized weakening (parity fails).
24. Numerics-trajectory law: numerics need 40-step SGD sanity, not just per-step parity.
25. Executor-regime law: executor-regime wins do not compose across executors; measure mechanism ON target executor; port the mechanism, not the vehicle.
26. Straggler law: fix stragglers with work-proportional decomposition (banding), not uniform splitting; needs an actual straggler (gate by stage count).
27. Enqueue/ready-order law: enqueue order is not a dependency edit; ready-order pays only for a specific starved consumer; apparent waits are absorbed overlap; (live-board addition) wait RELOCATES to siblings under single-sticky-head executor.
28. Filling-vs-fatness law: sub-wave fat tiles worse than more smaller tiles when CTAs << SMs; wave-gate (M/128)(N/128)>=~132; SKR pays iff ONE giant sub-wave long-K gemm amortizes ONE reduce (K/CTAs >= ~90).
29. Ablate-before-architect law: SASS-verified ablation of the specific op before porting a pattern; "obvious" targets are often the smallest slice; fwd/dkv/dq drain anatomies differ.
30. Off-path-is-not-free law: standalone-neutral off-path ops can WIN in-model by freeing issue slots (TN dW TMA -300us); removing off-path pressure amplifies on-path wait laws.
31. Sparsity-invariant law: clear-what-you-touched (sparse embed zero).
32. Meter-scope law: profile bucket is a bound (onpath..onpath+offp); score meter includes host launch path; standalone never substitutes for in-model.
33. Contract-audit laws: fused epilogues must register buffer args in _access_sets; cp.reduce source shape is part of the contract; per-clone SASS audit (ptxas 13.1 U64 bug).
34. Register-point routing (from arc): architecture is per-family routing across {168,224,255} register points — uniform rewrite refuted.
35. Boundary-removal beats overlap-machinery (opgap probes): fusion/boundary-removal class wins where overlap/batching class loses.
36. fp32 WMMA stride x4 alignment law (silent corruption).
37. Call-reached-noinline WGMMA poison (live board): any WGMMA body reachable via function call in the launched image forces per-HGMMA DEPBAR singleton drain (ptxas C7510); separate-global/standalone code groups fine; plain source respelling cannot fix it. DKV trigger = runtime skip predicate; DQ trigger = RS-feed group; qwen NT trigger = forceinline wrapper call.
38. DAG-cut realized value (live board, rmsfold): realized DAG-cut value ~1us/hop in starvation regime, NOT the full 3.5us itax price — fold pays only where L amortizes the off-path sink mass (deep/small yes; nano/s128/s256/s1024 no).

## TALLIES (ledger)
- Heavily NO-GO-dominated by design. Highest-yield promoted clusters: (1) D64/D128 WGMMA attention trio + banding; (2) qwen n256 direct-store + TMA + pdf executor stack (22.1ms -> ~8.5ms compounded, l2 -26%); (3) rowop batching/register-lifetime family; (4) scheduling hot/cold rings + claim-132 + resweep flips.
- Per category (PROMOTED/NO-GO approx): gemm-dx 8/6; gemm-fwd 7/4+1 floor; lm-head 6/2; gemm-dw 4/2; attn-fwd 7/8; attn-dkv 6/6; attn-dq 5/4; onepass 0/1(ledger)+4 more (archives); executor 7/8; scheduling ~18/~16; rowop ~13/~8; ce 3/4; itax-dag 4/7; qwen 4/~12; precision 6/4; tma-smem 6/6.
