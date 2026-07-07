# D64 DQ C>1 bulk-reduce drain prototype (session b1d36305)

Date: 2026-07-07. Worktree: /home/apanda/xorl-oss/.claude/worktrees/agent-a99c122c4935ff7d4
(branch `dq-bulkred-b1d36305`, base 2a41f6a). Flag: `MK_ATTN_DQ_BULK_RED=1`
(extension suffix `_adqbr`; flag-off builds are name/flag-identical to base).
Commits: 22e0824 (drain), e4704f4 (SASS audit + salt), 8895682 (forceinline
workaround), b9003a3 (kwarg plumbing).

## Verdict

**SPLIT: WIN at exact-S4096, REFUTED at S8192 (the original target).**

Paired alternating A/B (dq_bulkred_ab.py, env_ab_main pattern), both
construction orders, GPU 3 idle (0%/0MiB before/after every window), pdf route
(the shapes' default executor):

| shape | order | default | variant | delta | wins |
|---|---|---|---|---|---|
| s8192 | default_first | 6380.54us | 6435.54us | +54.99us | 0/24 |
| s8192 | variant_first | 6386.43us | 6428.77us | +42.34us | 2/24 |
| s8192 | default_first (rerun) | 6382.22us | 6434.30us | +52.08us | 2/24 |
| s8192 | variant_first (rerun) | 6396.48us | 6435.38us | +38.90us | 1/24 |
| s4096 | default_first | 3019.62us | 2978.94us | **-40.67us** | 40/40 |
| s4096 | variant_first | 2990.16us | 2957.44us | **-32.72us** | 40/40 |
| s4096 | default_first (rerun) | 2994.38us | 2952.98us | **-41.41us** | 40/40 |
| s4096 | variant_first (rerun) | 2999.44us | 2958.77us | **-40.67us** | 40/40 |

Parity in the timed configuration: loss bit-identical (9.06573 s8192 / 9.06420
s4096), worst_grad_rel 0.0039-0.0064 (atomic-order tolerance class). ROUTE
identical (188/189 instrs).

Mechanism reading of the split: at S8192 the tail `ATTN_DQ_WG C=4 off=60
width=4` bands ARE the critical path (441-447us waits), and the drain's
mandatory full-completion `cp.async.bulk.wait_group 0` (required so the region
watermark cannot observe the tile complete before the reduction is globally
visible) adds TMA round-trip latency directly on that chain — the fp32 atomic
drain it replaces is fire-and-forget (visibility covered by the executor's
post-batch `__threadfence`). At S4096 the chunked-DQ drains sit off the critical
chain and removing 8192 REDG issues per tile (32/thread) nets a steady ~1.3%
step win. A future one-pass-bwd drain (concept map item 1) with a producer-warp
drain that overlaps the wait would not pay the S8192 latency the same way.

## What was built

In `op_attn_dq_wg`'s C>1 chunked epilogue ONLY (C==1 direct-store and DKV
float2 atomics untouched; no one-pass coupling): keep the existing Cs[128][68]
smem staging, replace the 256-thread coalesced fp32 atomicAdd drain with the
FA4 zero-atomic idiom:

- every writing thread: `fence.proxy.async.shared::cta` BEFORE consumer_sync
  (stores -> proxy fence -> barrier -> elected issue; a lane-0-only
  post-barrier fence leaves the async proxy reading the stale bf16 K/V view of
  the overlay — measured err ~1e35);
- one elected lane per warp (ln==0, 8 warps) issues its warp's 16 rows as
  per-row 256B `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`
  (the [S, stride] workspace interleaves dq/dkv slots per row, so the drain is
  128 x 256B row segments; Cs rows at 272B stride are 16B-aligned, dst rows
  256B-aligned) — via the L2::cache_hint spelling with an evict_normal policy
  (semantically identical to plain add.f32);
- `cp.async.bulk.commit_group` + `wait_group.read 0` + `wait_group 0` (full
  completion) BEFORE the op returns.

## FINDING (toolchain, high value): ptxas 13.1 assembles add.f32 as ADD.U64

The single source asm line mis-assembles on CUDA 13.1 (ptxas V13.1.115,
sm_90a) whenever ONE PTX `.func` containing it is materialized by ptxas into
MULTIPLE kernel entries: exactly one entry receives the correct
`UBLKRED.G.S.ADD.F32.RN`; every other clone silently receives
`UBLKRED.G.S.ADD.U64` — a 64-bit INTEGER reduce over the f32 data. Corruption
signature (verified bit-exact): dst_bits + src_bits mod 2^32 per f32 lane with
u64 carry between adjacent pairs, e.g. 1000.0f (0x447A0000) + 0.0225f
(0x3CB85B2E) -> 0x81325B2E (-3.3e-38).

Forensics: parity fail ~1e35 at C=4 -> single-tile run on ZEROED ws is clean
(int-add onto 0x0 is invisible — beware this disguise in any future cp.reduce
parity test) -> nonzero-init (1000.0) single-tile shows EVERY drained element =
exact integer bit-sum -> per-function SASS audit: 16 UBLKREDs per executor
clone, F32.RN/U64 split across clones.

Ruled out empirically: issue pattern (single-thread vs 8-warp elect), staging
stride, in-flight LDGSTS, cooperative launch, 100KB smem, UR pressure
(standalone probes always correct); `__noinline__` helper (ptxas re-inlines at
-O3), volatile-fn-pointer ABI call (ptxas devirtualizes), `.L2::cache_hint`
spelling, NOP-salt perturbations (winner reshuffles but stays exactly-one
across 6 builds); UR register numbers (the fixed build is F32.RN with dst pairs
at UR16/UR26). Concurrent racing reduces to the same 256B rows are EXACT (new
concurrent-overlap unit probe, 64 CTAs vs atomicAdd baseline: max_err 0).

Root cause model: `op_attn_dq_wg` is `__noinline__` -> cicc emits one `.func`
holding the 16 asm statements (verified in PTX: exactly 16 `cp.reduce`
occurrences, all inside `.func _Z22wga_dq_bulkred_drain16...`) -> ptxas inlines
that single PTX object into every executor entry and only the first
materialization encodes the type field correctly. Standalone probes never
trigger it because each `__global__` owns PTX-distinct statements.

WORKAROUND (verified): flag-gated `__forceinline__` on `op_attn_dq_wg` + the
drain helper (dispatch() is already forceinline), so every executor entry owns
its own PTX copy of the asm. SASS audit after: ALL images at ADD.F32.RN —
megakernel/df/df2/ws in the default build, + megakernel_pdf in the pdf builds
(80/80 UBLKREDs). Every flag-on build is hard-gated by
`mk.py::_audit_bulkred_sass` (raises on any non-F32 UBLKRED in a launchable
image; `MK_ATTN_DQ_BULK_RED_AUDIT` scopes, `MK_ATTN_DQ_BULK_RED_SALT` rerolls).

Upstream action: minimal repro = one `.noinline` `.func` with the asm called
from >=2 entries; report to NVIDIA. ANY future cp.reduce.async.bulk use in the
megakernel (one-pass-bwd dQ drain, DKV P2) MUST keep the per-entry-inline +
per-clone SASS audit pattern, and MUST parity-test against a NONZERO-initialized
destination.

## Parity

- flag OFF: test_ops ALL PASSED, test_model ALL PASSED (incl waves/df2/ws
  agreement) — build is name/flag-identical to base.
- flag ON: test_ops ALL PASSED (bwd dqkv (wgmma) err 5.8e-03, same as base);
  test_model ALL PASSED (waves-vs-df, df2, ws agreement all OK; training sanity
  9.0496 -> 5.6530/40 steps).
- timed configuration (pdf): loss bit-identical, worst_grad_rel <= 0.0064.

## res-usage (flag-on vs flag-off, default df-family build)

- megakernel_df: REG 255/255, STACK 48 vs 32 (+16B frame, LOCAL 0 = no spill)
- megakernel: REG 255/255, STACK 48/48; megakernel_df2: REG 255/255, 48/48;
  megakernel_ws: REG 168/168, STACK 80/80.
- Entry ceilings respected (255 for 256-thread, 168 for 384-thread images).
- No smem change: drain reuses the existing Cs[128][68] overlay (34816B < 80KB
  op slab); no new barriers beyond the existing consumer_sync.

## Landing checklist (human/main session; this worktree never touched the shared tree)

1. Cherry-pick branch `dq-bulkred-b1d36305` (4 commits: wgmma_attention.cuh,
   mk.py, dq_bulkred_ab.py).
2. Exact model.py gate-table edit to promote the S4096 win (mirrors the
   pdf_d64_feed S4096-wins/S8192-regresses pattern), next to the other
   attn_dq_* defaults (~line 492):

       # D64 dQ C>1 zero-atomic bulk-reduce drain: exact S4096 won both orders
       # (-33..-41us, 40/40 x4 windows); exact S8192 regressed (+39..+55us) --
       # its tail C=4 DQ bands are critical-path and pay the drain's
       # full-completion wait. MK_ATTN_DQ_BULK_RED=0/1 remains the A/B override.
       self.attn_dq_bulk_red_default = exact_s4096

   and in the load_ext(...) call add:

       attn_dq_bulk_red=self.attn_dq_bulk_red_default,

3. Keep the SASS audit mandatory for any flag-on build (silent integer
   corruption otherwise); do not relax `_audit_bulkred_sass`.
4. Note the +16B STACK frame on megakernel_df (LOCAL 0, no spill) and that the
   flag-on build force-inlines op_attn_dq_wg (all-images code-size grows; the
   measured s4096 win already includes that cost).
5. Upstream ptxas bug report.

## Artifacts

- AB runner: worktree experiments/fused-training-megakernel/dq_bulkred_ab.py
  (copy also expected to be run via MKAB_TREE).
- Unit probes (scratchpad, session b1d36305): cp_reduce_overlap_probe.py
  (concurrent overlapping reduces), cp_reduce_modelmimic_probe.py (issue
  pattern/launch mimics), cp_reduce_urpressure_probe.py (UR pressure).
- SASS audits: adqbr_sass{,2,3,4,5}.txt in the session scratchpad.
