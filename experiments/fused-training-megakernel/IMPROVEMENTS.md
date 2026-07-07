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

**Instruction lookup scans offsets only (v0)** — WIN (many-instruction waves
stopped being scan-dominated)
Why: the ready-scan copied the whole 104B Instr struct per scan step.
Principle: interpreter inner loops must touch indices, not payloads.

**Dataflow executor replacing wave barriers (v1)** — WIN (nano 2.53->1.93-1.99ms,
small 12.0->9.8ms with the latency round)
Why: 84 grid.syncs serialized ops that could overlap; dependency counts from
per-op read/write signatures + a ready ring with a sticky-instruction fast
path and a global consumed-head. Naive ring scanning was SLOWER than waves.
Principle: self-scheduling only pays with a fast-path claim (sticky instr +
atomic cursor); ring scans are a memory-system tax.

**Scheduler direct successor handoff (v2 r1)** — NO-GO (+14% step, mechanism
unclear)
Principle: pushing work at completers loses to blocks self-claiming; don't
build handoff without a measured queueing problem.

**Region-watermark tile deps (df2, P3)** — NO-GO (df + 300-400us both configs:
nano 1853 vs 2158, small 9028 vs 9440; executor retained as mode="df2",
parked; later hot/cold port failed on scheduler-race correctness)
Why: op spans are intrinsic-latency-bound (long serial kv/k loops; 64-256
tiles on 132 blocks = no queue backlog); tile-granular deps eat the QUEUEING
component of span, and there is none — the wakeup/bounded-claim machinery is
pure overhead. Protocol facts paid for: unbounded claims through a CAS loop
degrade quadratically; parked gated instrs must be killed out of the ring and
event-woken (Dekker re-check for the lost wakeup); volatile reads on region
counters.
Principle: tile-granular dependencies pay only when tiles >> blocks; measure
queue backlog before building overlap machinery.

**qt-outer attention tile order (P3)** — WIN (small df 9268->9028, -240us)
Why: short causal tiles first lets blocks pick up off-path work while the
long tiles run.
Principle: within an op, emit long-pole tiles where the scheduler can overlap
the tail; tile order is a free scheduling knob.

**Warp-spec scheduler offload (mode=ws, P4a)** — NO-GO as default executor
(nano 1938/small 9198 vs df 1907/8994; briefly won post-P5 at small, lost
again after the P6 df-only wins; final rechecks +78/+227us and +296/+335 with
0 paired wins; results/mkv3-p4a.md)
Why: the protocol WINS at equal registers (-64/-190; on-path wait small
1495->587us) but the 4-warp register-allocation granularity puts consumers at
a 224-reg ceiling, and the register tax (+97/+385) exceeds the protocol win
(see Register architecture section). la=2 pre-claim had a real flip-fast-path
race (fixed) and is still slower (tail imbalance); ws consumers must keep a
register Instr copy (the smem-reference trick hangs); the consumer-owned smem
snapshot was worth -717us small but ws still trails df.
Principle: a correct protocol can lose to the hardware's pricing; measure the
tax and the win separately (equal-register A/B) before blaming the design.

**Hot/cold criticality ready rings (P6)** — WIN (waits collapsed: EMBED_FWD
91->0us, RMSNORM_BWD worst-hop 140->4.4us; step -7/-38us direct, and it
flipped the split-K dX experiment)
Why: dW sinks and fills competed with the chain at claim time; idle blocks
drain hot first, cold-sticky blocks yield when the hot tail moves.
Principle: separate sink work from chain work at CLAIM time. MK_ALLHOT
rechecks stay decisively negative everywhere (+34/+65/+29us; qwen too) — the
split is load-bearing.

**Tile-gated split-K fp32 dX routing (P6)** — WIN only on hot/cold rings
(nano -27us; +127/+467 under the old single ring; gate < 32 MN tiles; the
64-tile widening was refuted, mkv3-p4b-p6-r4-dxsplit-ab-gpu5.log)
Why: split-K atomics multiply claim traffic; only viable once cold work stops
contending.
Principle: order of experiments matters — a no-go under one scheduler can be
a win under the next; re-run interacting experiments after scheduler changes.

**Dispatch-spill kill: Instr staged in smem (P6 r2)** — WIN (nano -125, small
-484, ~8% uniform; df STACK 624->336)
Why: the 104-byte `const Instr I = instrs[ins]` register copy was live across
every dispatch call site — ptxas spilled it around the switch, taxing every
op (the dispatch-spill law; Laine-2013's canonical megakernel pathology).
Principle: keep NO caller state live across a fat dispatch switch; stage
per-claim state in static smem and pass references.

**Claim quantum 264 -> 132 (P6 r2, then invariant)** — WIN (-27/-237us);
every later resweep keeps 132 (66: +366 small; 264: +162; post-SW128,
post-fast-log, post-route, post-cap0 sweeps all reject alternates; qwen
claim64/32/16 lose +3.4/+9.0/+17.8ms; S8192 claim64 +1.2ms)
Why: the old 264 optimum was an artifact of expensive claims; once claims are
cheap, finer batches win on tail balance (Stream-K physics).
Principle: claim-batch size trades claim overhead vs tail imbalance; retune
once after making claims cheaper, then leave it alone.

**df completion-hint stickiness** — WIN (~-20us both configs; eba44d5)
Why: the block whose accounting enables a HOT dependent adopts it as its own
sticky claim — the chain's next hop starts on a warm block without ring
rediscovery.
Principle: completion is the cheapest possible dispatch signal; use it.

**cold_cap (bounded blocks on cold work)** — WIN as a shape-gated family
after many retunes (pre-SW128: wash; post-SW128 flip to cap16, small
~4110->4042; then S>=2048 uncapped (S4096 4010 vs 3917), mid-S cap33->48->64,
nano/S128 cap0 (-7.2/-8.2us), qwen-L1 cap0 decisive: 22094 vs 29433us cap48;
late resweeps all no-change)
Why: the optimum tracks op speed — when ops got faster, cold dW turned
net-contentious at short S; at L=1 giant-vocab there is no later hot work to
overlap, so capping starves the sinks.
Principle: cold-capping trades bandwidth contention against tail
serialization; the balance moves every time op speed or shape changes —
resweep after structural rounds, and expect L1/giant-sink regimes to want
uncapped.

**Rowop claim floor (MK_ROWOP_CLAIM 2/4)** — NO-GO (+275/+838us small)
Why: tail balance beats claim amortization at these tile counts.
Principle: same Stream-K law as the claim quantum — don't batch small-tile
claims.

**MPK-topology probe (dedicated scheduler block + gmem mailboxes)** — NO-GO
(5.09us/hop vs df ~3.0; scheduler-issued L2 prefetch made it WORSE, 9.86)
Why: two cross-SM signals per hop lose to self-claiming; the issue loop
starves done-polling.
Principle: chain-hop-granularity work distribution cannot afford cross-SM
round trips; both protocol escapes from the register tax are measured closed.

**Cold-ring demotion of attention ops (DQ short-S / DKV long-S)** — NO-GO
(DQ demote: small 4063->4500us; DKV demote: +41.3us, 1/24 wins)
Why: attention readiness matters even when the op is off-path in the profile;
demoting it starves the realized path.
Principle: profile off-path-ness is not schedulable slack — the realized
critical path moves when you deprioritize.

**Per-layer non-state scratch (false WAR/WAW dep removal)** — NO-GO (host DAG
deps 2030->1190, max fan-in 68->54; timing flat/worse: nano 1226.8/1237.2)
Why: under the hot/cold scheduler the false deps were never binding.
Principle: dependency-graph beauty is not speed; only deps on the realized
path matter.

**Attention-bwd DQ-first instruction order** — NO-GO (S4096 control 4266.7
vs DQ-first 4304.7us, 19/20 control; qwen D128 dq-first also no-go
-0.9/+9.6)
Why: the DQ wait is balanced against DKV completion feeding qk-norm bwd;
swapping enqueue order does not create readiness.
Principle: enqueue order is not a dependency edit; fix the structure (banding
did) or leave it.

**Long-S scheduler idle-poll cadence (MK_IDLE_NS 256 -> 32)** — WIN,
shape-gated (S3072 -11.1, S4096 -11.2us; post-band flips added S2048 -12/-15
and S8192 pooled -11.2 — the pre-band S8192 rejection reversed; idle16/64
washes)
Why: idle blocks re-poll the ring; at long S the ready set turns over fast
enough that 256ns sleeps strand work.
Principle: poll cadence has a shape-dependent optimum; include it in
post-structural resweeps.

**D=128 attention claim-1** — NEUTRAL then WIN (standalone: claim batching
cost -48/-75/-85us per op; in-model first an order-mixed wash at qwen
(absorption strike 6, knob committed default-off), later PROMOTED
post-sparse-embed: +37.3/+35.1us old-minus-new, ~31/32 both orders)
Why: the claim-batching straggler law — batched claims (quantum 4 over 512
tiles) serialize the causally-longest tiles on single blocks; whether the
model absorbs that depends on what else is co-scheduled.
Principle: per-op claim quantum matters where tiles have skewed serial
length; recheck absorbed no-gos after the surrounding schedule changes.

**Early dXN zero-fill (move to wave 0)** — NO-GO (nano +3.1, S1024 +2.8us)
Why: the fill overlaps fine where it is; earlier placement steals wave-0
bandwidth from the embedding gather.
Principle: moving support work earlier only helps if its current position
actually blocks something.

**Drow zero-fill skip on direct-store shapes** — WIN narrow (S256 +7.9/+1.8,
S512 +0.9/+2.2 zero-minus-skip; small PREFERS the fill by 13.7/7.5us; late
small recheck order-mixed — gate stays S256/S512 only)
Why: the direct-store epilogue overwrites every element, so the fill is dead
work — but at small the fill's schedule slot was load-bearing overlap.
Principle: dead work is only free to delete if its slot wasn't absorbing
contention; gate deletions per shape.

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

**Attention bwd split over GQA members + kv chunks (v1)** — WIN (dKV worst
instr 190->47us; fp32 atomic workspaces + convert op)
Why: one monolithic bwd instruction serialized the whole head dimension.
Principle: split attention bwd along its natural parallel axes before tuning
anything inside it.

**Split-KV forward + combine op (v2 r4, C=4)** — NO-GO then SUPERSEDED (nano
neutral, small +5%: the combine chain hop plus partial-tensor traffic
outweigh the per-instr latency saving; ops retained — the same machinery
became the WIN inside long-S fwd banding a phase later)
Principle: keep correct-but-unrouted mechanisms around; a losing op at one
shape class can be the fix at another (flash-decoding partials paid only
where the straggler chain was long).

**FA-class WGMMA attention trio, D=64 (P5)** — WIN (small 9028->7118; fwd
3.34/5.63x, dkv 2.1/3.3x, dq 2.2/4.9x vs the WMMA ops standalone; attention
stopped being the #1 lever; results/mkv3-p5-attnprobe.md)
Why: block tile = (head, 128 q-rows), two consumer warpgroups, 2-stage K/V
streaming, register online softmax. KEY LAYOUT: one no-swizzle 64x64 smem
arrangement readable under BOTH wgmma majors — every bwd transpose is a
descriptor change, zero data movement (generic smem stores feeding wgmma need
`fence.proxy.async.shared::cta`).
Principle: design smem layouts for descriptor-level transposes; the
dual-major trick deletes whole staging passes.

**Attention chunk counts (MK_ATTN_DKV_C / MK_ATTN_DQ_C)** — WIN as a
per-shape gate map, retuned after every structural change (P5 defaults ->
r3/r4 retunes -> post-SW128 Cq=1 -> S1024 2/2, S2048 2/2, nano 3/2 then 2/2,
small 1/1 repeatedly confirmed against every variant; qwen D128 C=1
everywhere; generic D128 fallback Cq 4->1, -85/-95us)
Why: the in-model optimum is a SCHEDULING optimum, not an op optimum
(standalone prefers more chunks; in-model, fill/atomic overhead and tail
balance dominate). The DQ/DKV "co-scheduling overhead" was diagnosed benign —
the pair runs concurrently; concurrent max ~= sequential sum.
Principle: never tune chunk counts standalone; only in-model, per shape, and
re-sweep after structural rounds.

**FA2-style software-pipelined attention fwd (MK_ATTN_PIPE)** — NO-GO (fwd
span 245->297, step +245us at small; later recheck still strongly negative)
Why: with the pipe-probe stage sweep and the MPK prefetch, three independent
measurements agree — in the 8-warp latency-bound regime, generic
overlap/depth machinery costs more than it hides.
Principle: pipeline depth is not free below ~1/8 occupancy; only targeted
drain-hiding on a monopolizing op pays (see the PV pipeline entry).

**Attention loader lane remap (conflict-free INTER stores)** — NEUTRAL, kept
Why: strictly fewer smem port cycles, but attention stage fills hide under
mma+softmax (unlike the bare gemm loop, where SW128 was decisive).
Principle: bank conflicts only matter where the fill is exposed.

**DKV S/dP wgmma x2 batching, first attempts** — NO-GO then SUPERSEDED
(MK_ATTN_DKV_X2_SD profile A/B: totals 1145/4325 -> 1169/4511; the isolated
DQ x2 probe also lost S2048 +3.9/S4096 +14.1) — superseded by the 5b8b3db
one-commit batch WIN once the fused-ALU restructure and dispatch unspill
landed.
Why: batching alone extended register lifetimes without deleting the serial
drain; the win needed both accumulator banks live AND the softmax/dS ALU
fused into one pass.
Principle: a mechanism can lose half-done and win whole; record the losing
half with its missing ingredient.

**DKV G2 fusion (fuse GQA-pair tiles per KV tile)** — NO-GO (+221 small,
+323 S2048, +455 S4096; control won 80/80 despite halving DKV tiles)
Why: lost G-parallelism dominates the removed K/V reloads and atomic drains.
Principle: don't trade parallel axes for traffic in the latency-bound regime
(register-lifetime-law corollary).

**Forward register-direct O epilogue** — NO-GO (reverse order: nano
1247/1230, S4096 4298/4262 variant/control); later direct fwd WG stores also
rejected on long-S regressions.
Why: the smem-coalesced O drain is already hidden; register-direct stores
scatter.
Principle: coalesced staged epilogues beat register-direct stores unless a
single-writer direct path removes atomics too.

**DQ epilogue family (C=1 direct store -> register-direct -> float2)** — WIN
in stages (C=1 direct store: small -23, S4096 -10; register-direct: S4096
-66/-40 on reverse+warm repeats; float2 store gated S3072/S4096 -11.5/-7.2
and S8192 -30; small/S128/S256 rejected); chunked C>1 direct atomics NO-GO
(4341.7 vs default 4299.7).
Why: Cq=1 has exactly one writer per q slice, so every layer of staging and
atomics is deletable; wider stores pay only where the op is long enough.
Principle: hunt single-writer cases — they convert atomic epilogues into
plain stores; gate vector-width wins by shape.

**DKV epilogue family (direct atomic -> float2)** — first NO-GO (20260704:
small/S4096 worse, keep smem drain), then SUPERSEDED/PROMOTED after the route
landscape changed (direct-atomic: -17.8 nano ... -49.4 small on every
tracked shape; float2 atomics broad D64 gate: -5.7 nano ... -47.4/-55.5
long-S; forced-old rechecks lose -29..-94us).
Why: skipping the smem stage/drain pays once the surrounding schedule stopped
absorbing the drain; float2 halves atomic transactions.
Principle: epilogue verdicts are not permanent — resweep after op/route
changes; pair adjacent accumulator lanes for vector atomics.

**Attention tail-barrier elision** — NO-GO (nano 1240->1276, small
4328->4515, S4096 4267->4430; 0 paired wins)
Why: the final syncs are either throughput discipline or cheaper than the
runtime branch that guards them.
Principle: barriers inside warm loops are rarely the cost; measure before
"optimizing" synchronization away.

**Masked-work skip family (masked-exp skip; dQ/dKV split-mask loops)** —
NO-GO (masked-exp: small +223, nano +31; dQ split-mask S3072 +42/S4096 +46;
dKV split-mask dead neutral +0.19)
Why: per-element branch/predicate cost overwhelms the saved SFU work; the
mask is already nearly free.
Principle: arithmetic on the masked fraction (`exp(-inf)->0`) beats
branching; don't specialize loops around a predicate the compiler hoists.

**C=1 straggler diagnosis + banded bwd chunking** — WIN, the largest single
long-S win (S2048 -20.6/-16.0, S3072 -37/-42, S4096 -80.7/-78.7, S8192
-531.7/-524.6us; 40/40 everywhere; 06ab5b6+8508a7d)
Why: at C=1 the bwd ops are makespan-bound on the longest causal tile's
serial stage chain (dkv 2.7us, dq 1.9us per 64-row stage) while SMs idle;
uniform C>1 (the old no-go) pays fill/atomic overhead on ALL tiles to fix one
straggler. Banding splits only the long tiles (C = ceil(stages/T)), packs the
band spec into the existing C arg, and gives bands disjoint workspace slots.
Principle: fix stragglers with work-proportional decomposition, not uniform
splitting; measure marginal cost per serial stage first (single-instruction
programs), and let degenerate specs decode as the old path.

**Short-S banding** — NO-GO (S512 T4 +15.2us; S1024 T8 neutral; degenerate T
collapses to uniform C=1 and loses +22..+70)
Why: below ~32 stages the extra fills/atomics have nothing to amortize.
Principle: straggler fixes need a straggler; gate by stage count (S >= 2048).

**Band emission order (lpt vs dq_first)** — WIN narrow (dq_first S8192 only:
-88/-105us 16/16; broadening loses S2048 +9/+22, S3072 +53/+63, S4096
+47/+52; dkv_first +52, dkv_long_first +158, lpt_dq_tie +12 all NO-GO;
post-combine and post-n256 rechecks keep dq_first at S8192 only)
Why: at S8192 the critical DQ bands were wait-dominated behind same-wave DKV
work; everywhere else the apparent DKV wait is useful overlap.
Principle: ready-order edits only pay where the timeline shows a specific
starved consumer; apparent waits are usually absorbed overlap.

**Forward banding (flash-decoding partials + range-limited combine)** — WIN
(S2048 T16 -19.7, S3072 T32 -45.7, S4096 T32 -4.4 then retuned T22 -23,
S8192 T64 -452/-481; MK_ATTN_FWD_BAND=0 recheck: +525us, 0/16)
Why: same straggler mechanism as bwd; C=1 bands keep the direct O/LSE
epilogue. Numerics: bf16 locally-normalized partials cost ~2x error on tiny
qk-norm grads — short-S gates must budget tolerance or store fp32.
Principle: the fwd fix reuses the parked split-KV ops; sharp budget optima
exist (T>64 leaves an unsplit straggler: +400/+510 cliff).

**Band budget retunes (T per shape)** — WIN via repeated resweeps (S4096
T32->T29 -60us x3; S2048 T16->T12 -30/-37; S8192 T32->T40 -88/-110; fwd
S4096 T32->T22 -23; later rechecks hold: T36/T44/T48 reject)
Why: every structural change (idle32, row batching, combine-R, n256 routes)
moves the packing balance.
Principle: band budgets are cheap env resweeps with occasional 10x payoffs —
the resweep law's home turf.

**Combine row batching (MK_ATTN_COMBINE_R=8)** — WIN (S3072 -33/-28, S4096
-28/-24, S8192 -69/-46us; R=4 loses, R=16 neutral; fe39656)
Why: the combine tile was ONE ROW — a long-S combine instruction was 2-4k
tiny claim-ring transactions.
Principle: never let per-tile work shrink below claim cost; batch rows so
all 8 warps stay busy.

**Combine weight-array unroll (kill the `float w[8]` local)** — WIN gated
(S8192 -29/-44; S4096 clean +18.4/+7.3 for unroll; S3072 regresses — gate
exact)
Why: the dynamically-indexed local array was the only real local-memory
traffic at banded long-S (latency-hidden but nonzero).
Principle: scalarize small dynamic arrays in hot ops; verify per shape — the
win rides on how exposed the op is.

**Fused one-pass attention bwd (dQ inside DKV)** — NO-GO (S4096 +119us,
0/40; parity clean)
Why: fusion multiplies dQ WRITE traffic by n_kvt x 2WG — ~135MB/layer of
fp32 atomics at S4096 that serialize at L2/DRAM — while the two-pass K/V
re-reads it saves are L2-RESIDENT (~4MB << 50MB L2). Store-amplification
traded for cache-absorbed load-amplification; loses harder as S grows.
Principle: count write amplification before fusing passes; atomic stores
don't cache.

**Fwd KV-widening w128 (FA4-B port)** — NO-GO in-model by absorption (banded
shapes: only short off-path tiles can take it, S4096 +18.7; unbanded: S512
-6.5/-1.9 order-decaying, S1024 +3.3; the C>1 partials-path port also
failed: S4096 +23/+25, S8192 washes) despite +11-19% standalone.
Why: absorption strike 5 — the halved boundary cost lands off the realized
path; also a real numerics delta (single 128-wide softmax pass ~0.021 grad
rel vs w64).
Principle: standalone op wins must survive composition with banding and
co-scheduling; port to where the on-path chunks actually run.

**DKV row-scalar broadcast (LSE/Drow via shuffle)** — WIN then SUPERSEDED
the other way (S8192 -16.9/-14.0 promoted; after the long-D64 gemm mbar ring
landed, forced-old beat it -32.4/-30.8 16/16 — default gate emptied again;
small and DQ variants NO-GO)
Why: scalar-load dedup composes with the surrounding schedule; the mbar ring
changed that schedule and the win inverted.
Principle: micro-wins near the noise floor are composition-fragile; re-run
them after structural neighbors change, and be willing to un-promote.

**Cheap DKV cross-stage reschedule (no extra smem)** — NO-GO (test_ops hung
at RUN DKV_WG twice; results/operator-gap/attn-bwd-pipe-conservative-nogo.md)
Why: issuing S/dP(t) while draining dV/dK(t-1) without a real ring deadlocks
buffer reuse.
Principle: cross-stage overlap needs the full ping-pong/ring ownership
protocol; there is no cheap version.

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

**D=128 WGMMA attention trio (fwd/dkv/dq WG128)** — WIN (qwen4b-l1
-1586/-1846us env, +1885/+1628 vs forced-old — ~9% of the step; default
D==128 && S%64==0; 1566e51..dac7321)
Why: split-D (each WG owns a 64-wide D-half, all accumulators stay m64n64
fragments — no new descriptor layouts), P-parking, redundant-S (both WGs
compute S/softmax: free in the latency-bound regime, kills cross-WG
serialization). Independently converged with the opgap FA4-C spec.
Principle: at D=128, split the head dimension across warpgroups instead of
inventing layouts; redundant compute beats synchronization when issue slots
are idle.

**D=128 dQ row-split + register-A dS feed** — WIN (row-split: +32.5/+35.8
old-minus-new, dQ span 317.6->186.0us; RS-feed: op 103.65->99.62us, fa58afb);
C-chunk sweeps keep C=1 (C=2 +97us); the mbar split-wait variant NO-GO
(-0.8/-11.4 weak); the dKV S^T register feed NO-GO in-model (op +10.3us AND
STACK 32->160 = a +6.9us tax on the old path just from compiling it).
Why: row-split keeps the direct-store single-writer epilogue; the RS feed
deletes dS smem + fence + WG sync. The dKV S^T port died on the
dispatch-spill interaction, not its own math.
Principle: measure the binary-wide resource tax of a port, not just the op;
a fatter frame taxes every other op through the shared allocation.

**D=128 fwd mbarrier ring** — WIN (qwen +38.1/+31.7 old-minus-new; fwd span
225->131.5us)
Why: 128-row q tiles with a two-stage K/V mbarrier ring remove the forward
stage boundary; uses consumer_sync so ws mode stays valid.
Principle: mbarrier rings pay on ops with long streamed operands and an
exposed stage boundary — not as a generic bolt-on (see the GEMM ring entry).

## GEMM tiles / feeds

**Vectorized coalesced tile loads, all four layouts (v0)** — WIN (+2x)
Principle: get global-load vectorization right before anything else.

**Register-prefetch K-loop pipelining (v0)** — WIN (single-buffered loads
were latency-bound at ~33us/wave)
Principle: even 1-deep prefetch beats none; more depth needs occupancy to
hide it (see the deep-mainloop no-gos).

**Split-K for tiny-tile matrices (v0 dW; v1 dlogits@Wlm K=8192: 145->40us)**
— WIN
Why: 16-tile matrices on 264 blocks were 6% occupancy.
Principle: split-K converts occupancy holes into atomics; worth it only while
tiles << blocks (the same law later gates every split-K knob).

**cp.async BK=64 WMMA rewrite** — NO-GO (~-10% in-model; microbench flat)
Why: LDSM patterns on the col_major ld=72 paths; WMMA surgery is not a
Hopper pipeline.
Principle: don't polish WMMA; the answer was wgmma+TMA-class kernels.

**wgmma NT path (v2 r1)** — WIN small (~4%), nano nil; plus the decisive
probe fact: a data-dependent ScaleOut ternary between arrive and commit makes
ptxas SERIALIZE every wgmma (~2.5us each, 60x); branch-free accumulate over
zeroed registers reaches ~94% of per-SM peak.
Why: 64 fp32 accumulators pushed the kernel to 255 regs -> 1 block/SM, and
per-instr fixed costs dominate at chain-gemm sizes.
Principle: keep wgmma issue branch-free; tensor-core quality is irrelevant
until fixed costs and occupancy are handled.

**m64n64 retile + smem-staged vectorized epilogue (v2 r2)** — WIN (nano
1.93->1.85ms)
Why: 32 accumulators restore block density; staging accumulators over dead
cp.async buffers makes stores fully coalesced.
Principle: epilogues belong in smem staging unless a single-writer direct
path exists.

**Deeper wgmma mainloops (3/4-stage, one-in-flight)** — NO-GO (pipe probe:
small NT heavy hitters flat at ~50/~224/~46us)
Why: the limiter was the stage FILL (bank conflicts), not depth; and at 8
warps/SM there is nothing to hide latency with.
Principle: identify the limiter before adding stages; depth multiplies smem
cost for nothing when fill-bound.

**SW128 swizzled operand slabs** — WIN (small GEMMNN 516->254us, 427->216,
head dX 190->146; scoreboard 1468->1260 nano / 5626->4465 small; 6fe8fcb)
Why: the no-swizzle INTER layout made cp.async stores 8-way bank-conflicted —
THE GEMM limiter. Attention keeps INTER (its descriptor-swap trick needs the
symmetry).
Principle: swizzle the operand path feeding wgmma; a fill-bound mainloop
mimics "pipeline too shallow" and sends you tuning the wrong knob.

**m64n128 (n128) tiles** — WIN (small NT -146, NN -28; lm_head span 280->194;
generalized from the peer's lm-head-only branch 8008126 whose narrow-scope
'not promoted' verdict was correct)
Why: 64 fp32 accumulators/thread fit 255 regs; double mma per sync, half
B-traffic per FLOP — the dependent chain per FLOP shortens, the one lever the
register-lifetime law permits.
Principle: bigger tiles at full registers is the sanctioned way to spend the
register file; gate by row count (short rows can't fill the tile).

**n128 feed/store variants (3-stage feed; direct bf16 store)** — NO-GO
(stage3 +66.6us 0/400; direct-store +16.8 combined)
Why: same fill-bound + absorption physics as the generic deep-mainloop no-go.
Principle: n128's win was tile geometry, not feed depth.

**Direct-BF16 GEMM epilogue (generic WGMMA, skip the smem drain)** — WIN
narrow (S128 exact gate ~-1..-5us; post-n128 small promoted -27.8/-27.0
80/80; do-not-broaden: S256 order-dominated, H256/S1024 lost)
Principle: the smem drain is only worth skipping where the store stays
coalesced enough and the op is exposed — exact-shape gate it.

**Fat lm-head tiles (staged 128x256)** — NO-GO for the cooperative kernel
(needs a 160KB smem page: cudaErrorCooperativeLaunchTooLarge at 132 blocks;
the 100KB direct variant loses generally and wins only qwen-class
high-K/high-V — which became the n256 route)
Why: the one-kernel design shares a single smem carveout; smem208 controls
show the global page itself costs +6..+95us across shapes.
Principle: in a one-image megakernel, per-op smem appetite is a GLOBAL tax;
fat-tile routes must fit the shared page or stay probes.

**qwen n256 direct-store route family** — WIN, exact-gated ladder (lm-head
fwd +909us; head-dX n128 then n256 +1628/+597; dW sk1 no-atomic +3120; TN
n256 dW +3178; NT bf16 no-residual/residual +229/+276; MLP dX +120/+98; qkv
dX +25/+31; Drow n256 +152/+120; direct-dW fill skip +524/+514; qwen step
22.1ms -> ~8.5ms compounded over the day) — with broad-route rejections
everywhere else (small MLP dX n256 +210/+251; S8192 NT broad +150/+162;
S8192 NN exact-triple WIN -29/-18).
Why: giant-K/giant-N single rows are the one regime where 256-wide
direct-store tiles amortize; sk=1 split-K rows were paying zero-fill+atomics
for no actual split.
Principle: audit `sk==1` split-K routes (pure waste); exact-gate giant-tile
routes and let broad-force probes define the boundary empirically.

**n256 stage3 operand ring + N-major tile order** — WIN (stage3: +424/+387
old-minus-new 32/32, forced-stage2 rechecks lose up to +2.6ms; N-major
bit26: +35 then +45..+87 in later rechecks)
Why: the 148KB qwen page was already funded by the D128 dQ row-split; three
48KB stages fit. N-major groups all M bands per B tile (precondition for
multicast work that ultimately didn't pay — kept anyway, it wins alone).
Principle: if a bigger smem page is already funded by another route, respend
it on feed depth; keep tile-order wins even when their motivating follow-up
dies.

**mbarrier feed-ring for GEMM** — NO-GO as a generic port (nano +15.2 0/20;
small n128 ring +51 1/24), then SUPERSEDED by the gated long-D64 landing
(31dad00: small -76.0 80/80, S2048 -27.7, S3072 -50.6, S4096 -63.2, S8192
-150.1; short shapes explicitly off) and the qwen n256 port (-169/-191) with
the NT lm-head body split BACK to the old refill loop (_gmbar_n256ntold: NT
mbar costs +112/+145); stage-depth S5/S6 NO-GO (24-32KB/stage blows the
100KB page — illegal memory access).
Why: barrier-free full/empty rings pay only where the CTA-sync refill was an
exposed boundary (long streamed K at D64, giant NN/TN rows).
Principle: port protocols per BODY and per REGIME, not per library; the same
ring is a win, a loss, and a crash depending on operand length and smem
budget.

**Cluster B-sharing family (DSMEM-fed GMMA; paired-M TMA multicast;
B-multicast probe)** — NO-GO (DSMEM: the 14-bit GMMA descriptor cannot
encode a cluster rank — remote-B reads local zeros; the TMA-multicast
primitive itself PASSES, but the production pair probe loses to plain
cp.async 44.1 vs 46.7us; round-6: B was already L2-absorbed in our claim
order)
Why: B-operand reuse was never the bottleneck — L2 already dedupes it; the
cluster protocol adds pair-coupling per stage.
Principle: before building multicast, measure whether the shared operand is
actually re-fetched from DRAM; L2 is a free multicast for co-scheduled tiles.

**Deep SW128 stages for long-K dX (S8, standalone win)** — NO-GO to
integrate (187.4->175.8us standalone, but needs a 208KB global page costing
+6..+95us across shapes)
Principle: same global-smem-page law as fat lm-head tiles; standalone
per-GEMM wins below the page tax are unroutable in the one-image design.

**256-row supertile + NT-floor diagnosis (opgap)** — standalone WIN at the
224-reg point (NT lm_head 177->144us, +18%; exclusive with the 168-pt
producer), but the cheap direct-store 256x128 variant loses everywhere
(184->195us), and the NT family is declared FLOORED at 1.66x vs nvjet —
C-write rate 0.95 vs 1.54 TB/s, raster order neutral, all SM-side mechanisms
exhausted; residual is DRAM writeback policy.
Principle: know when a gap is not yours to close — after the mechanism
ledger is exhausted, the next step is a diagnostic (write-path counters),
not more mechanisms.

**NN transpose-staging (K-major via smem transpose)** — NO-GO (standalone:
NT 12.93 vs NN MN-major 12.96us — descriptor path exonerated; explicit
transpose loses +2.4us)
Principle: the small-shape NN penalty lives at the in-model shell level
(flags/epilogue/claim), not the MN-major descriptor; don't port a transpose.

**Post-spill side probes (splitK-direct epilogue; generic wgmma noinline;
iclk compile-out; MK_CLAIM=264 recheck)** — NO-GO (3596/3608/3592 vs clean
3569us; claim264 fails the gauntlet)
Why: after the D64/D128 noinline fixes, remaining local traffic is ABI-frame
class; broad noinline of the generic wrapper adds a call on hot n128/n256
routes.
Principle: once executed local-LD reads zero, stop hunting spill; lower
STACK/local-store counts are not a promotion criterion (runtime only).

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

**TMA feed at H256/D64 n256 shapes (boundary sweep)** — S3072 WIN, S4096 +
S8192 NO-GO (s3072 -7.1/-10.2 35-39/40 both orders + promoted-old
+11.2/+8.1, commit 0b7ed2a; s4096 +6.1/+11.9 <=7/40; s8192 +27.3/+33.6 2/16;
`mkv3-p4b-s{3072,4096,8192}-n256tma-*.log`)
Why: the win class is LONG-K rows only (s3072's sole eligible row = head-dX
K=8192, 128 ring iters). 12 of s8192's 13 eligible rows run 4-24 iters where
the elected-thread fence+expect_tx serialization exceeds the per-thread issue
work it deletes; s4096's 32-tile head-dX packing loses where s3072's 24-tile
wins. The s8192 probe also parity-proved the STAGES=2 TMA arm.
Principle: TMA-feed eligibility is a K-length question, not a route question —
short-K rings keep the 256-thread cp.async feed; boundary-sweep per shape
because tile-count packing flips neighbors (3072 vs 4096).

## Rowops / elementwise

**Warp-parallel qk-norm with smem-staged grad atomics (v0)** — WIN (~8% of
the step)
Why: global-atomic contention on the tiny [D] grad buffers.
Principle: stage tiny-buffer atomics through smem; never point a block of
global atomics at a [D]-sized target.

**Vectorized fills/converts + per-layer fp32 workspaces (v1)** — WIN
Why: 16K-chunk fill/convert; a SHARED attention workspace chained layers
through its own zero-fill.
Principle: shared scratch creates false serialization through its zeroing;
per-layer workspaces are cheap dependency surgery.

**CVT hop deletion via dy_f32 consumers (P1)** — WIN (5 on-path CVT hops,
112/346us; nano 1888->1833)
Why: elementwise consumers have no dtype constraint — read the fp32 atomic
workspaces directly.
Principle: convert-at-consumer beats convert-as-instruction; grep for
convert ops on the critical path.

**Batched row ops (MK_ROW_R=8, warp per row)** — WIN (nano -117, small -722
— biggest single round since P5; rowop spans 1.35-3.3x)
Why: warp-shuffle reductions delete 3 block barriers; uint4 8xbf16 IO; dw
staged in smem = ONE global atomic per element per 8 rows.
Principle: per-row atomics into [H]/[D] grad buffers serialize; batch rows
and stage partials.

**RMSNorm-bwd row-gradient reduction (private [row,H] partials)** — WIN
(small on-path span 674->224us)
Why: 8 row warps were atomically contending on ONE shared [H] buffer.
Principle: same contention pattern one level up — per-warp/per-row slices,
one block reduce, then one global atomic.

**Rowop MLP split (R2=16 interleaved streams) + the register-lifetime law**
— WIN (nano -12, small -30; RMSNORM_BWD 565->467; swiglu gains __expf) with
two REVERTED register-caching attempts (rmsnorm single-pass +240us small;
qknorm register-dw +58; qknorm R16 +142 — doubled serial per-warp chain)
Why: register-resident value reuse LOSES to re-reading at 8 warps/SM — long
register lifetimes block load overlap; the winning shape is short-lived
registers + more independent streams.
Principle: THE REGISTER-LIFETIME LAW — prefer re-reads and extra load
streams over caching values in registers; only dependent-chain shortening
pays.

**op_ce_bwd uint4 IO** — WIN (CE_BWD 79->46us small); CE_BWD two-row tile
NO-GO (reverse order refuted; local span barely moved).
Principle: vectorize the V-wide pass; don't halve tile count of an op whose
per-row warps are already saturated.

**RMS four-row folds (BWD_R4 / DX_R4)** — NO-GO as defaults (whole-step
+26.6 small despite local span 466->444; DX_R4 kept ONLY at H256/S2048
(-6.6/-13.1); the small DX_R4 gate flip-flopped with surrounding routes and
ended at R2; every long-S/nano/qwen recheck negative)
Why: folding rows halves tiles (parallelism) to save reduction work the
machine wasn't short of.
Principle: local span improvements that cut tile count are usually absorbed
or inverted at step level; promote only on step time, per exact shape.

**RMSNorm bwd dx/dw split** — WIN (c3fa4c2: dX consumers stop waiting for
the dw atomic drain; small -8/-7) — but QKNORM dx/dw split NO-GO twice
(small +100/S4096 +89; qwen -69 0/32: the extra cold op adds more scheduling
pressure than the drain costs), and the qwen combined-RMS recheck confirms
the split.
Principle: splitting an atomic sink off the chain pays when the drain is the
binding dependency — not when the op is small enough that a second
instruction costs more than the drain.

**QKNORM/ROPE-bwd V-passthrough split (split-V)** — NO-GO short then WIN
long (nano/small: local span down but on-path wait up, totals flat; S8192
-34.6/-63.5, S4096 -9.6/-12.4, S3072 -11/-13 post-band; S2048 boundary
stays off; correctness hazard: the reader must depend on the UNSLOTTED
dQKV_f32 root or banded slot-writers race it)
Principle: an extra row op's dependency cost outweighs its local saving
until the op is long enough; slot-aliased buffers demand the conservative
dep root.

**SwiGLU backward worker-count arc (R2/2W/3W/4W/4W_V4)** — WIN via repeated
supersession (R2 fold NO-GO, span 38->51; 2W two-warps-per-row WIN small -12
then widened to H256 S1024-4096, S8192 FLIPPED post-band -112/-128; 3W
NO-GO; 4W: small first NO-GO +10.5 then WIN -132/-121 after the scheduler
and body changed, qwen I=9728 4W WIN +356/+330; 4W_V4 NO-GO; nano 4W
order-wash)
Why: warps-per-row must match row length (serial chunks per row) and the
current scheduler's tail behavior — the optimum moved every structural
round.
Principle: worker-count-per-row is a shape x scheduler knob, not an op
property; the resweep law applies to op-internal geometry too.

**SwiGLU sigmoid cache arc** — WIN then SUPERSEDED (cache WIN small -13.4
and H256 S1024-4096 + S8192 post-band flip; qwen: cache REJECTED — 2W-only
beats cache+2W by +95; post-4W small: cache-off wins -25.7/-22.7 — final
small default is cache-off+4W; the DSSG double cache NO-GO +53)
Why: caching trades an `__expf` recompute for bf16 traffic; the balance
flips with worker count and shape — 4W rows re-read the cache too often.
Principle: memoization is a bandwidth-vs-SFU trade that must be re-measured
whenever the consumer's access pattern changes.

**qk-norm bwd caches (D64 cached-pair; D128 cached-pair)** — WIN (D64:
-6.7/-26.6/-17.7/-30.5 across shapes, res-usage unchanged; D128 qwen:
+25.5/+26.8 old-minus-new)
Why: these caches shorten the dependent chain (one pass instead of
reload+recompute) WITHOUT extending accumulator lifetimes — the allowed side
of the register-lifetime law.
Principle: cache to delete a second pass, not to avoid a re-read.

**Fixed-width RMS dx bodies (H256/H512/H2560 exact)** — WIN narrow (H256:
S8192 -25/-43, S512/S1024 +2.4..+15.3 old-minus-new; S256/S4096 rejected;
H512 exact body NO-GO — local 150->140 but step flat; qwen H2560 R4 NO-GO)
Why: killing the runtime H loop/divide pays only where the op is exposed
enough on-path.
Principle: specialization is a routing decision; local op improvement
without step-time proof is not a promotion (absorption again).

**QKNORM_ROPE_BWD per-warp dw partials** — WIN (small -21, S4096 -17, S8192
-43/-66; 61e37bd)
Why: every lane of every warp atomicAdd'ed into ONE shared [D] array —
block-wide serialization on 64 addresses, ~30x above bandwidth floor.
Principle: block-shared smem accumulators with per-lane atomics serialize;
use per-warp slices + one cross-warp reduce. Grep `atomicAdd(&*_s[` when
writing rowops.

**CE fwd/bwd fusion (OP_CE_FWD_BWD)** — NO-GO (qwen: -110 default-minus-fused
one order, +29 reverse; fused span 327 vs 34.7+296.4 split — no real
deletion)
Why: the fused op replaced two hops with one op of the same total span; the
step is dominated by the surrounding head GEMMs.
Principle: don't fuse ops whose spans simply concatenate; fusion must change
dataflow (traffic or hop deletion), not instruction count.

**CE_BWD label fixup (branch-free vocab loop + one overwrite)** — WIN gated
(qwen -26/-25; S8192 -44.8/-31.8 16/16; small -2.7/-3.3 confirmed; the H256
S1024-S4096 sweep fails the two-order gate — keep exact)
Why: removes a per-element compare from a V-wide hot loop; only giant-V
on-path CE rows feel it.
Principle: per-element predicates in V-wide loops are worth one
write-after fixup — where V is big enough to matter.

**CE ignore-row skip** — NO-GO twice (first: the small win was erased by the
separate-opcode route, S1024 +9.5; recheck: -2.6 weak then +19.4 on the
construction-order repeat — reverted)
Why: the skip's branch cost and route plumbing eat the saved work at these
ignore densities.
Principle: data-dependent skips need a measured density argument; a weak win
that fails order reversal is a loss.

**Sparse embedding-gradient clear (qwen)** — WIN (+234/+254us old-minus-new,
32/32 both; OP_EMBED_ZERO_ROWS clears prev+current token rows instead of
zero-filling [V,H])
Why: EMBED_BWD only touches batch-token rows; the full fill was 389M fp32
zeros per step. Duplicate/overlap zero races are benign.
Principle: exploit sparsity invariants across steps
(clear-what-you-touched).

**SSQ fused epilogue defaults (bit13)** — mostly keep-on, flipped OFF at
S2048/S3072 post-split-V (+8.4/+3.0 and +3.4/+7.7 old-minus-new; S4096
order-mixed, S8192 wash, small/nano confirmed on)
Why: the fused sum-of-squares rides the wo/wd epilogue; at mid-long-S that
epilogue slot became contended.
Principle: even "free" fused reductions have a shape-dependent price; keep
them behind per-shape gates.

**Rowop long-S gap decomposition** — NO LANE (launch-floor-subtracted op
cost at S4096 ~260us vs the baseline's 254us — bodies at PARITY; the
in-model 540-1330us bucket is critical-path span stretch from co-scheduling;
exception: QKNORM bwd at S8192 is genuinely body/bandwidth-bound
~91us/instr)
Principle: decompose before optimizing — the only lever for a
parity-at-body/stretched-in-model bucket is structural (fuse into producer
epilogues), not kernel quality.

## Routing / tiles

**qk-RMSNorm+RoPE fused into the qkv gemm epilogue (v2 r2, bit8)** — WIN
(kills 1 chain instr/layer at D=64; WG_BN==64==head_dim makes the norm
reduction tile-local); unfuse rechecks NO-GO twice (S128 -16.7us for fused;
S128 probe +7.4/+8.0 for the split).
Principle: epilogue fusion pays when the tile boundary IS the op boundary
(one head per tile) — the exception that proves the hop-deletion rule.

**SwiGLU paired-column tiles (bit9)** — NO-GO (+88 nano/+297 small; its
pass-loop restructure also silently spilled 320B in the wgmma hot path, -8%
everywhere, visible only in the spill count)
Why: halving gu tiles doubled per-tile serial span — parallelism traded for
fusion.
Principle: fusion that reduces tile count needs tile-granular deps first;
and check `cuobjdump -res-usage` on EVERY device-code change.

**CE/LSE partials in the lm_head epilogue (bit11)** — NEUTRAL, kept on (A/B
within noise; cheapens the CE hop ~5x for free; rstd-producer bit12 was not
attempted — predicted negative by bit9+bit11)
Principle: keep free reductions that ride an existing epilogue; don't add
SFU work to on-path tiles for an off-path saving.

**wgmma NN/TN bwd routing (rounds 3/P6)** — NO-GO three times (nano
1.85->2.05ms, small 9.3->10.3; re-runs +58/+69) then SUPERSEDED by SW128 +
tile-gating (NN >= 64 tiles wins; nano's 16-48-tile dX stays WMMA/split-K)
Why: pre-SW128 the wgmma path was fill-bound, so per-instruction fixed costs
dominated; the verdict was about the feed, not the math unit.
Principle: a routing no-go can be a body bug in disguise; re-run routing
verdicts after fixing the body.

**NN threshold ladder (MK_WGMMA_NN_MIN per shape)** — WIN arc (generic 64;
M=512 nano 16: -18.6/-19.8 479/480; M=1024/N=256 32: S1024 -108 180/180;
short rows 8: S128/S256 -6.4/-7.3; S128 Drow shape 4: -12.4) with a late
NARROWING at small (exact 1024x512x{3072,1024}/1536x512 NN rows and three NT
n128 rows moved back to m64n64: +28.7/+39.2 and +56.5/+55.6 for the narrower
default)
Why: the threshold proxies "does the fatter tile fill the machine and beat
its fixed cost HERE"; every body/scheduler change moves it.
Principle: thresholds are measurements, not constants — keep them per exact
shape and resweep after structural rounds.

**TN dW wgmma routing** — NO-GO twice (+226/+570: dW sinks are 2x more
BW-hungry and steal from the chain) then SUPERSEDED by gates as dW split
targets and bodies changed: long-S K>=3072 WIN (S3072 -61, S4096 -94/-107,
S8192 -333), broadened to K=1024 with min(M,N)>=512 (small -77/-87, S2048
-29/-64); post-band resweeps show the gate is now load-bearing (forcing it
off: +161..+174, 0/40).
Principle: sink-op routing verdicts depend on how much bandwidth the chain
needs at that moment; revisit them when the chain gets faster.

**Drow/dOatt fused-epilogue route arc** — WIN by stages (bit10 fused Drow in
P1; wgmma-epilogue Drow -43us small; small WGMMA route NO-GO twice ->
long-only S>=2048 gate WIN (S4096 -30/-25, 49e9481) -> broadened default WIN
(small -21/-25; forcing old WMMA later loses +31.4); Drow n128 NO-GO
(+54..+64 everywhere); qwen D128 Drow n256 WIN (+152/+120); n256 Drow
direct-store bit27 NO-GO (order-mixed weak).
Principle: one fusion, four tile widths, four verdicts — route width and
fusion are separate decisions; measure each pairing.

**head-dX split-K target arc (dlogits @ Wlm)** — WIN by successive
supersession (512 -> 256 (-21/-28) -> 192 -> per-shape 96/64/32 as routes
changed; sk=1 no-atomic promotion at S2048+ (-7.7..-21.9) and qwen; small
n128-fp32 no-atomic -17.6/-10.6; nano n128 split target 48 -4.2/-4.9)
Why: every upstream change moves the optimal split; over-parallelized long-K
splits lose to fewer fatter tiles; sk==1 rows pay pure atomic/fill waste.
Principle: split-K targets are the most-retuned knob in the program — always
re-derive them from the current route, and kill sk==1 atomics on sight.

**Cold dW split-target arc** — WIN then invariant (512 -> 192/128 (-57 nano,
-232 small, -110 S2048, -170 S4096) -> 96/64 (-27 nano ... -75 S4096);
subsequent sweeps all no-change)
Why: off-path sinks over-parallelized at target 512 stole issue/memory
bandwidth from the hot chain.
Principle: deliberately under-parallelize sinks; they only need to finish
before the step ends.

**dX split side-knobs** — NO-GO family (dxwg split-K +80..+122 0 wins; nano
MLP/qkv dX n128 split-K +110..+188; WMMA split target 64 order-biased, keep
128; MK_DX_SPLIT_MAX_TILES=64 refuted)
Principle: the <32-tile split-K gate survived every attack; don't re-derive
it without new structure.

**lm-head n128 per-shape gates** — WIN arc (auto-gate by M: S128 -49.8, S256
-19.1, nano -9.1, deep -38.1 vs old mode 1; S256 lm-head n128 OFF as a late
retune +6.5/+5.9, nano boundary stays on; small lm-head n128 confirmed
load-bearing: forcing it off loses -48.9)
Why: short rows can't fill 128-wide tiles; the boundary is a measured M
ladder, not intuition.
Principle: route gates by row count with per-shape overrides — and re-verify
the survivors when neighbors change (see the lm_head n256 cascade below).

**Fused-qkrope tile widening** — mixed by width: n256 qkrope NO-GO
(+34/+25 nano/small); n128 two-head qkrope epilogue WIN at long-S (S8192
+56/+52 old-minus-new; S3072/S4096 -41/-43 40/40; S2048 order-mixed, stays
wg64).
Principle: fusion epilogues constrain tile width to the head structure;
widen only to the width the epilogue math actually tiles (two D=64 heads =
n128).

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

**Epilogue fusion phase 2 (MLP `xn2` deletion via scaled dW B-load)** — NO-GO
(H512/L1/S1024/V4096 `MK_EPIFUSE_MLP=1`: +16.32/+15.58us, 1/40+0/40; attr:
fwd +1.01/+4.26, bwd +9.63/+17.20; note
`results/operator-gap/rowop-epifuse-phase2-nogo.md`)
Why: correctness holds, but deleting `xn2` requires the generic WGMMA TN
B-load to materialize `x2 * rstd2 * w2` elementwise; that scaled-load path is
the measured loser. The forward row-scale/rstd-only half is only neutral and
cannot delete traffic alone.
Principle: buffer-deletion fusion pays only if the replacement consumer can
absorb the transformation cheaply; moving rowop math into an exposed operand
feed is still on-path arithmetic, not free traffic deletion.

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

**head-dX SKR at s3072/s4096 (SKR round 4)** — WIN skr=2 both (-118.6/-120.6
40/40 and -90.0/-75.6 40/40+39/40 at 0abc08f; re-anchored forced-old at
322f344 +95.2/+89.4 and +77.4/+80.9, 0/40 x4; skr=4 inferior; s1024
order-flipped at head after winning both orders at 5c6e234 — left atomic)
Why: their n256 direct head-dX ran 24/32 CTAs — parallelism-starved with the
full K=8192 chain per tile; SKR-2 = 96/128 n128-CTAs, half the serial K, one
reduce. K/CTAs 170/128 >= nvjet's ~90 split gate; s2048 (exposure-hidden) and
s8192 (one-wave, K/CTAs 64) bracket the window. SUPERSEDES the 0b7ed2a s3072
n256-TMA default (its only TMA row leaves the route; long-K TMA principle
untouched).
Principle: "the direct route stands" verdicts are S-local — a NO-GO at one
shape does not close its neighbors; walk the boundary. Sub-wave n256 tiles
can be WORSE than more, smaller n128 tiles when CTAs << SMs: tile fatness
buys nothing idle SMs could have eaten.

**lm_head fwd n256 resweep cascade (small/s1024/s2048)** — WIN (-33/-23,
-11/-12, -25/-21 both orders; 7f77378, 5c6e234)
Why: the 0928Z gauntlet CORRECTLY rejected these cells; the same-day SKR and
mbar-ring/commit-batching promotions restructured the surrounding schedule
and flipped them.
Principle: exact-shape gates rot in BOTH directions — after any structural
promotion, re-run the cheap env probes for neighboring REJECTED routes at the
affected shapes; 4 of this session's 6 promotions were resweep flips found by
2-run env A/Bs.

**lm_head fwd n256 short-S boundary at 9199394** — NO-GO for forced n256 at
s128/s256; nano weak/noise, keep current gates (s128 +37.20/+21.50us 0/80
both-ish; s256 +14.96/+11.50us 1/80+4/80; nano -4.22/-0.40us 66/80+47/80;
job `lmhead-n256-boundary-9199394-2325`, note
`results/operator-gap/lmhead-n256-short-boundary-9199394-nogo.md`)
Why: the n256 direct route still needs enough rows to fill its fatter tile;
s128/s256 expose pure underfill, and nano's tiny one-order movement is below
the promotion bar after the current idle/SKR route state.
Principle: route-boundary resweeps should close both sides of the boundary.
Decisive neighbor losses and sub-noise middle movement mean keep the gate, not
promote an exact-shape exception.

## Numerics / approximation

**fp32 WMMA smem strides must be multiples of 4 (v0)** — correctness law
(silent corruption otherwise).
Principle: alignment constraints in accumulator staging corrupt silently;
encode them structurally.

**Attention LSE fast-log (`__logf`)** — WIN gated then partially SUPERSEDED
(broad: small -20.1, S2048 -15.2, S128 -9.7; S256 neutral; the flag later
ROTTED out of attention.cuh and was restored; post-directbf16 SMALL flipped
to precise: +1.3/+1.5 repeatable — small now defaults precise)
Principle: approx-transcendental wins are small, real, and
composition-fragile; guard the flag with route checks so it can't silently
rot, and let per-shape rechecks reverse it.

**WGMMA attention exp2.approx (`ex2.approx.ftz`)** — WIN gated (small -37.7,
S1024 -16.8, S2048 -20.1, nano -4.9; S128 regressed — gate D64, S>=512;
small recheck confirms keep: precise loses -28.8/-25.7)
Principle: SFU-approx swaps pay on softmax-heavy long ops; always gate out
the shortest shapes.

**lm-head CE-partial exp2** — WIN gated (small -57.0, S2048 -66.0, nano
-19.6; gate V>=8192, S>=256; small recheck: precise loses -68.5/-67.4)
Principle: the online sumexp in a giant-N epilogue is the one place exp is
hot enough to matter.

**CE forward fast-log / exp2** — NO-GO (fast-log mixed; exp2 looked positive
built-old-first (-7..-37) and REVERSED on the promoted-default order
(+7..+18) — reverted, including inert plumbing)
Why: CE_FWD is one small hop; the deltas were construction-order artifacts.
Principle: the both-order gate exists exactly for this class; a win that
appears only in one build order is a bias, not a speedup.

**CE backward exp2** — WIN gated (small -35.7/-34.4, S2048 -16.5/-22.0,
S1024 -6.3; gate S>=1024 V>=8192; short/nano order-mixed; small recheck
keeps it: precise loses -14.0/-12.5)
Principle: same op family, opposite verdict from CE_FWD — gate by where the
op is actually hot (the V-wide backward pass), not by op name.

**SwiGLU arithmetic forms** — split verdicts: FMA derivative WIN (nano -8.2,
small -16.8, S1024 -10.7; forced-old recheck fails the both-order gate —
keep); `__frcp_rn` reciprocal NO-GO (small -16 but S4096 +20; gated variant
regresses fallbacks; 2W retest +14.6); exp2 sigmoid NO-GO (S128-only win,
everything else regresses; fresh-process neutral).
Principle: contraction-safe FMA rewrites are the only broadly safe
arithmetic change; approx reciprocals/exponentials in a fused elementwise
body leak cost into other shapes via code shape.

**Attention output fast-inv (`__frcp_rn` normalization)** — NO-GO on
CORRECTNESS (per-step grads within tolerance, but the 40-step learning
sanity failed: loss 9.05->7.15 vs the required 2.0 drop)
Why: systematic normalization error compounds across steps.
Principle: numerics gates need a training-trajectory check, not just
per-step parity — this is what the SGD sanity test is for.

**Attention dS FMA rewrite** — NO-GO twice (mixed: S128/small win,
S256/nano lose; gated retest +2.05 overall; small-only retest fails the
order gate)
Principle: an FMA re-association that helps some shapes but regresses
protected ones on code shape alone is not gateable — leave it.

**RMS dx FMA opcode** — WIN narrow (separate OP_RMSNORM_BWD_DX_FMA routed
ONLY at H256/S128: -5.6/-6.0; the broad compile-flag rewrite was mixed:
S128 -14.5 but S256/nano/S2048 regress; small retest negative)
Principle: when an arithmetic form wins only one shape, make it a routed
opcode rather than a compile flag — isolation beats gating.

## Launch / executor

**Occupancy via register cap (__launch_bounds__(256,2) / MK_OCC2)** — NO-GO
twice (v2 A/B: nano -16%/small +7%, kept unbounded; P4b re-run: REG:128
STACK:944, nano +32%/small +40%)
Why: forcing 2 blocks/SM spills the fat-op paths; the latency chain pays
more for spills than the added warps hide (see Register architecture for the
full accounting).
Principle: occupancy-via-spill decisively loses; more blocks is not the
path.

**Dynamic-smem default correction (100KB unless MK_ATTN_PIPE)** — WIN (nano
-7/-5, small -5/-13; a 120KB carveout had leaked in as a global default from
a default-off artifact)
Principle: the smem carveout is a global knob — audit it after any
experiment that touched it.

**Global smem page tax** — LAW (208KB controls: +6.1 nano, +42.2 small,
+94.9 S8192 with NO route using it; 160KB cooperative launch FAILS outright
at 132 blocks; qwen's 148KB page is paid once and reused by dQ-rowsplit +
fwd-ring + stage3)
Principle: in a cooperative one-kernel design the dynamic-smem request taxes
every shape and has a hard launch ceiling; new big-smem routes must share an
already-funded page.

**Launcher smem-attribute caching fix + ws 256B offset** — WIN
(infrastructure; c590974/f33f604 — would have bitten ANY D=128 landing)
Why: executors cached cudaFuncSetAttribute(MaxDynamicSharedMemorySize) in
process-lifetime statics — mixed-carveout processes launched with a stale
attribute (cooperative-launch-too-large / illegal smem access); ws mode
offsets ops by 256B of control smem, so a byte-exact 112KB struct overruns
only in ws, only under real timing.
Principle: re-apply any per-launch attribute that can grow; leave carveout
headroom for executor-mode control smem.

**mk_tid()/group-derived barrier helper (vs legacy constant semantics)** —
WIN/keep (legacy revert NO-GO: nano 1189/1262, small 4212/4508, S4096
4042/4374 on the clean control; side effect of the helper: df STACK 144->32)
Why: the dual-stream executor that motivated it died, but the helper
generalizes the op library over block shapes and compiles better.
Principle: judge helpers by measurement, not by whether their motivating
feature survived.

**Cooperative cluster launch (MK_CLUSTER_X=2)** — NEUTRAL (launch-only A/B
-2.7us overall; kept as opt-in capability)
Why: cluster dims + cooperative launch coexist via cudaLaunchKernelEx; the
launch path itself is neither a win nor a tax.
Principle: capability landings are fine at neutral cost; the win
(B-multicast) must be earned separately — and wasn't (see the cluster
family entry).

**In-kernel valid-label reciprocal (OP_INV_VALID)** — WIN on the
user-visible step (nano -75.3, small -93.0, S128 -71.5us; 160/160)
Why: the host-side `labels >= 0` sum forced a sync + prelaunch work every
step.
Principle: anything the host computes per step from device data belongs in
wave 0 of the kernel.

**Input binding (device buftab patch -> launcher-side binding)** — NO-GO
then WIN (Python-side pointer patch: +15/+19/+8.8 — slower than the copies;
C++ launcher-side override: S128 -12.6, nano -11.7, small -22.5us, ~198/200)
Why: patching device memory from Python costs more than the copy it saves;
the launcher writes the override before the existing init sync for free.
Principle: put fast-path plumbing below the Python boundary; measure the
plumbing, not the concept.

**Qwen4b-L2 gate-extension lane (cluster + support + head-dX)** — WIN
(l2 16.21ms -> ~12.06ms, -26%; commits 6c4d63c+e1e23b2+c3cc32c;
`mkv3-p4b-qwenl2-*.log`)
Why: every l1-exact qwen gate was dark at L=2 (tuple sets and
exact_qwen4b_l1 all L-keyed) — the l2 n256 rows ran 2-stage no-ring
cp.async and head-dX ran splitK. Extending the proven l1 cluster
(stage3+nmajor+ring+TMA, -1.53ms), the support knobs (sparse-embed -219us,
cefix, swb4w), and the head-dX no-split n256 route (-2.15ms, loss-exact)
recovered it. Peel attribution: stage3 alone +3.3ms, TN-TMA +1.6ms on top
of NN, TMA +0.7ms. dq RS-feed was the ONE l1 component that HURT l2
(RS-off faster in 4 windows, -53..-166us) and was decoupled to L1-only.
Principle: (1) exact-tuple gates silently orphan sibling shapes — sweep
every L/S neighbor of a promoted config with the cluster, not per-knob;
(2) peel-resweep after cluster promotion catches components that do not
transfer (dq-RS); (3) NN-only TMA inverted ~1.5ms WORSE than no-TMA at l2
(elected-thread issue contends with doubled cp.async TN sink pressure) —
partial feed conversions can be worse than none.

**S3072 head-dX TMA gate (promoted then reverted)** — SUPERSEDED
(promote 0b7ed2a -7/-10us 35-39/40; revert after fe15e24 head-dX SKR:
TMA-off faster -16.3/-20.2 40/40; revert-confirm +16.2/+19.0 1-2/40)
Why: the SKR promotion moved the gated row off the n256 route; the leftover
_gtma compile taxes the binary with no active rows (the noinline lesson).
Principle: a shape gate whose target row is re-routed by a LATER promotion
is not merely vacuous — retest and revert it; resweep-law applies to gates,
not just knobs.

**Qwen4b-l2 cold-cap-0 (flipped on, then killed by pdf)** — NO-GO
(pre-pdf resweep: cap0 -108.9/-49.7 11/16 both orders at 5f566dc; post-pdf
recheck at 589ee3d: +17.1/+4.9 wash-to-loss; stashed edit discarded;
`mkv3-p4b-qwenl2-coldcap-postpdf-k8s-20260707T0510Z.log`)
Why: the head-dX restructure opened a cold-ring window that uncapping
exploited; the producer-df executor restructured scheduling again and
closed it — two head-moves flipped the same knob twice in one day.
Principle: between measuring a knob win and committing its flip, re-check
what landed; a verdict is only valid at the structure it was measured on
(the resweep law applies at commit time, not just probe time).

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
in-model. Cell (b) is now BUILT (megakernel_pdf, worktree xorl-oss-pdf240,
session e5225c66) — measurement in flight.

**Producer-df executor (megakernel_pdf: 384thr, consumers inc-240, WG2 =
pure TMA producer via smem mailbox)** — WIN at TMA-row shapes / NO-GO
short-S (head-rebased e8837a5, k8s both orders + GPU-5 profile, parity
clean: qwen4b-l1 -1056/-1127us 12/12 with lm-head dX span 2453->1680 (-32%);
qwen4b-l2 -1511/-1370us 12/12 on top of the l2 gemm-cluster promotion;
s3072 -21us 40/40+39/40; s8192 -165us 16/16 at head; small +145us 0/40.
Session e5225c66, logs mkv3-p4b-pdf240-*-head-20260706T2345Z.log)
Why: the WG2 producer (dec-24 region, R18 used) replays each posted tile's
full stage schedule gated only by ring empties — issue decouples from the
consumers' mma cadence, the exact serialization the long-D64 elected-feed
no-go isolated. Separately, the region-compiled 240 image wins ~-405us at
qwen while MK_DF_MAXNREG=240 (flat entry cap, 256thr df) is +30/+44us
WORSE — the shell, not the lower ceiling, carries the phase-1 win.
Principle: (1) the round-5 producer topology IS reachable inside the
one-image megakernel — as a per-SHAPE executor mode, not a uniform library
point; register-point routing extends to executors (gate mode=pdf like a
knob); (2) requests strictly serial per block make the mailbox ordering
free (tile T+1 posts program-order after T's last full-wait); racecheck
shows only the by-design acquire/release class, synccheck 0.

**ptxas setmaxnreg REGION IDIOM (CUDA 13.1)** — TOOLCHAIN LAW (measured on
megakernel_pdf, 2026-07-06; board entry has the SASS evidence)
ptxas honors a setmaxnreg.inc region ONLY in the ws/CUTLASS idiom: the inc'd
body self-contained INSIDE the taken branch, ending in return, with the dec
path on the fallthrough. Inc on the fallthrough (dec-branch-first) is
SILENTLY compiled at the entry cap — every value 224/232/240 refused (max
SASS reg R165, STACK:208 spill) while the runtime USETMAXREG.TRY_ALLOC is
still emitted, so warps own registers ptxas never allocates. Restructured to
the idiom: 240 honored (R237, STACK 80).
Why: ptxas's region formation appears to require the raised budget to be
scoped to a single-entry single-exit branch body; a fallthrough region that
merges with the function exit is kept at entry budget.
Principle: always structure warp-spec as `if (consumer) { inc; body; return; }
dec; thin-path` and CHECK max SASS register index (not res-usage REG, which
reports the entry value) after every change near these branches. Retro-note:
the P4b-r3 512-thr dual-stream refutation ("no extra compile budget") likely
hit this same structure trap; re-run in the idiom before citing it as a
hardware limit.

**dkv stage ablation ladder (no-exp/no-alu/no-gemm compile variants)** —
MEASUREMENT (S8192; `mkv3-p4b-dkv-ablate-*.log`)
Why: decomposed the 2.5us stage: ALU 41%, accum drain 21%, score gemm 6-10%,
syncs 28%; span cuts compound into wait cuts (192us span -> 485us step).
Principle: ablate before architecting — the "obvious" gemm target was the
SMALLEST slice; ALU+sync overlap (warp-spec/ping-pong) owns ~70%. Beware
invisible co-tenant preemption faking waits (repeat until two runs agree).

**TMA feed for long-S D64 gemm rings (all majors + split-K TN)** — WIN
(S3072 -24, S4096 -30, S8192 -15; both orders 15-16/16; 17b1596+27c8084)
Why: elected-thread bulk-tensor loads + count-1 expect_tx replace 12-16
per-thread cp.async slices + 256 arrivals per ring stage; biggest on long-K
TN dW sinks (issue-slot relief for the co-scheduled chain).
Principle: (1) never-taken alternate paths tax the hot path via codegen
reflow — keep old/new as separate loops, extract shared fat bodies
__noinline__; (2) df-mode loss is not bit-stable within an arm (~5e-6 fp32
atomic spread) — parity means cross-arm delta <= within-arm replay spread.

**dq cross-stage ping-pong (no-wait accum + triple K/V + double dS)** —
NO-GO (S3072 +3 / S4096 +13 / S8192 +32, scales with S; correctness-clean;
wt-dqpipe, `mkv3-p4b-dqpipe-*.log`)
Why: dq's stage path is score->ALU — the accum drain was already hidden; two
in-flight batches per WG contend on the shared tensor pipe with the sibling
WG's score pair.
Principle: ablate the SPECIFIC op before porting a pipeline pattern —
fwd/dkv/dq have different drain anatomies; and in a 2-WG cooperative body,
extra in-flight batches steal the sibling's tensor-pipe slots. In-place
ping-pong is dead here; the ALU share needs warp-spec or direct ALU work.

**dkv LSE/Drow stage prefetch (cp.async with Q/dO fills)** — WIN (S8192
-55..-64, 3 clean pairs both orders; S4096/small neutral; +1KB smem)
Why: two per-row gmem scalar loads sat on the per-stage ALU critical chain;
staging them with the existing cp.async group moves their latency off-chain.
Principle: scalar gmem loads inside a stage's ALU pass are chain links —
prefetch them with the stage's bulk loads; the ablation's "loads" share is
often the cheapest slice to delete.

**Single-image 240/24 producer-df SHELL at short-S (cell (b), cost side)** —
NO-GO at short-S/non-TMA shapes, measured (shell tax small +156.8/+137, nano
+42.4/+45.8 0/40; decomposition: named bar.sync 1,256 = 0, flat-240 ceiling =
+11.0/+8.2, remainder = WG2 residency ~+34us at nano even PARKED in
nanosleep-8192; commits 16be37e inert knob, logs mkv3-p4b-dfprod-*, design
results/operator-gap/producer-df-design.md)
Why: 4 extra resident warps dilute per-SM issue in the latency-bound 8-warp
regime even when they never execute work — the cost is residency, not
registers, spills, or barriers (all isolated).
Principle (SCOPE reconciled with the producer-df executor WIN entry above,
session e5225c66 — the two data sets agree everywhere both measured): the
residency tax is real and unconditional (~+34us nano / part of small's +145
band), so 256thr/255regs stays the Pareto point at shapes with NO producer
work — which is why the _PDF_MODE gate keeps short-S on df with bit-identical
binaries. It does NOT close the design at TMA-row shapes, where the measured
producer dividend + region effect exceed the tax by an order of magnitude
(qwen4b-l1 -1056..-1133us 12/12 across k8s + clean GPU 3, l2 -1370..-1511,
s3072 -21 40/40, s8192 -165 16/16). Corrected law: the register/executor
point is a PER-SHAPE routing decision — elected-thread feeds where the tax
dominates, the WG2 producer where giant TMA rows dominate.

**D64 TMA resweep at excluded shapes** — WIN at S2048 (-24, gate widened);
S1024/small/nano exclusions UPHELD (+15/+64/inert)
Why: the TMA fence/expect_tx cost amortizes only when K=S>=2048 per ring
stage; K=1024 rows pay more in fence than they save in issue slots.
Principle: resweep-law passes are cheap (env-only) and should test the
MECHANISM boundary (here: K), not just shapes — the boundary told us which
future shapes will win without measuring them.

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

**Baseline honesty (the SDPA 3-D bug)** — RETRACTION (bench.py passed 3-D
[H,S,D] tensors, the flash backend rejected them, and SDPA silently
math-decomposed in EVERY v3 measurement; the celebrated long-S crossover was
an artifact — against real flash the megakernel falls FURTHER behind with S;
honest gaps restated nano 1.95x / small 2.52x; dcb24e9)
Principle: profile the BASELINE too (torch.profiler shows which kernels
actually ran); a soft baseline poisons every derived conclusion — and the
chunked-CE control shows how to falsify the convenient alternative
explanation before believing a crossover.

**Fresh process per config** — benching multiple sequence lengths in one
process poisons torch.compile with dynamic-shape recompiles (in-process
S=256 "baseline" 2121us vs the honest 608; Dynamo recompile-limit warnings);
single scoreboard rows are snapshots (the S3072 3007us outlier) — paired A/B
is the decision instrument.

**Timing instrumentation footguns** — clock64 is per-SM and useless across
blocks (%globaltimer only); subtract globaltimer stamps in int64 BEFORE any
float conversion (fp32 granularity at ~1e14ns is 16ms — diffs read as
exactly 0); profilers must pass the model's smem carveout to Program.run
(D=128 profiles silently crashed at the 100KB default until fixed).

**The gap model (P0)** — the residual is NOT per-instr fixed cost x chain
depth: nano = 243us on-path wait (13.6%) + 1542us on-path SPAN; hops cost
2-7us; the rest is intrinsic op latency inflated by co-scheduling.
Principle: split every hop into wait vs span before choosing a lever;
deleting a hop pays only its span, and only if not re-added to an on-path
producer.

**Latency-bound verdict (nsys gpu-metrics inside the window)** — SM issue
19%, compute warps in flight 12%, DRAM read+write <10%: the interpreter is
latency-bound at 1/8 occupancy.
Principle: sample GPU metrics inside the kernel window before reasoning
about bandwidth; this one measurement killed the "bandwidth contention"
framing and re-priced every subsequent idea.

**Both-order + confirmation gate** — many "wins" were construction-order
bias (CE exp2 reversed on the promoted-default order; ceskip -2.6 became
+19.4; swb2w's S128 arm reversed; broad sweeps routinely contradict focused
repeats).
Principle: promote only on paired A/B in BOTH construction orders plus a
promoted-default-vs-forced-old confirmation; a one-order win is noise until
proven otherwise.

**GPU hygiene (co-tenancy + private build dirs)** — standing practice: guard
every timed run with before/after nvidia-smi (sglang co-tenants manufactured
a fake ws-stall signature and poisoned medians twice; GPU2 reads 1.6-1.8x
high under clean-looking guards; GPUs 1/4/7 went hidden-util mid-session);
per-session TORCH_EXTENSIONS_DIR (the name-keyed cache races between
concurrent builds; a killed build leaves a stale lock that silently wedges
the next one).
Principle: a timing without environment guards is not a measurement.

**Resweep law** — structural changes invalidate old knob verdicts: idle32
and cached-SwiGLU-2W at S8192 flipped from NO-GO to WIN post-banding;
lm-head n256 flipped post-SKR; TN dW flipped post-SW128 and
post-dW-retune; claim1 flipped post-sparse-embed; the S8192 SwiGLU flip
alone was worth 11x the average recent promotion for two A/B runs.
Principle: after every structural promotion, re-run the cheap env knob
sweeps on the affected shapes — the flips are the cheapest wins on the
board.

**Periodic full certification** — the uniform dispatch-spill tax
(+22..+624us per shape) was invisible to per-lane paired A/Bs (it sat in
both arms) and was caught only by a full certified scoreboard against fresh
expectations.
Principle: paired A/Bs measure deltas, not drift; run the full gauntlet
periodically and bisect uniform regressions by RUNTIME (the compile-only
STACK bisect misattributed it — see the n256 exoneration).

**Knob consolidation + route snapshots** — one module-level tuning table
replaced ~12 scattered per-shape gates (56ab960), verified route-preserving
by hashing instruction streams/claims/deps across all 11 gauntlet shapes
(results/route_snapshot.py).
Principle: measured gates accrete into unmaintainable expressions — relocate
them behind a hash-verified refactor so retunes edit one line, and keep each
gate at the exact dimensionality it was measured under.

**s3072/s4096 post-SKR knob resweep (R4 round, session b603d819)** — ALL
DEFAULTS STAND (bwd band: s3072 12/+43, 20/-2.5 81/120 sub-noise, 24/wash;
s4096 24/+67, 32/+65; fwd band: s3072 24/wash, 40/+34; s4096 20/+18, 24/+11;
dq_first +47/+50 0/40 both; idle 64/32 wash-to-sub-noise both shapes;
mkv3-p4b-resweep-*.log)
Why: the SKR promotion moved head-dX off the path but the s3072/s4096 chains
are ATTN_DQ-wait-bound (305/536us wait) — knob nudges reshuffle slack, not
throughput.
Principle: a 40-rep -6..-10us "win" that fails at 120 reps is the noise rule
working — always escalate marginal band/idle candidates to 120+ reps before
promoting.

**OP_SKR_REDUCE fatter chunks (claim-tax hypothesis)** — NO-GO/ABSORBED
(chunk 8192/16384/32768 at s4096 all ±3us 17-26/40; s3072 16384 -5.9 29/40
sub-noise; patch parked at results/skr_reduce_chunk_probe_b603d819.patch)
Why: the reduce shows 122.7us on-path span at s4096 (256 one-shot 4096-elem
tiles), but shrinking it buys nothing — the chain sits in a 536us ATTN_DQ
wait-slack regime that absorbs on-path hop savings (exposure logic beats the
span model).
Principle: profile "on-path span" is not automatically exposed time — check
the WAIT column of the surrounding chain before optimizing a hop; in
wait-slack regimes only the wait producer itself is worth attacking.
