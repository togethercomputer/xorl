# Attention producer-feed Phase 0+1 (attn-pdf-feed lane)

Session b1d36305, 2026-07-07. Spec:
`results/operator-gap/s8192-attn-producer-offload-design-2a41f6a.md` (+ the
revised Phase 1 from `attention-tma-layout-feasibility-2a41f6a.md`).
Worktree: `/home/apanda/xorl-oss/.claude/worktrees/agent-afe5cb98663ff1e67`,
branch `attn-pdf-feed-b1d36305` off base `2a41f6a`.
Commits: `216afd9` (Phase 0 scaffold), `e3e0c36` (Phase 1 WG2 cp.async stream).
Probe: `results/attn_pdf_feed_probe_b1d36305.py` (in the worktree branch).

## What was built

Branch A of the feasibility note — the layout-PRESERVING WG2 `cp.async`
producer. No TMA, no new smem layout, no repack stage; the GEMM SW128 TMA
reuse stays a NO-GO as audited.

- **Flag**: `MK_ATTN_PDF_FEED=0/1/2` (env + `load_ext(attn_pdf_feed=)` kwarg).
  `1` = dq K/V stage feed, `2` = + dkv Q/dO stage feed. Extension suffixes
  `_apdff` / `_apdff2`. Requires `pdf_producer` (forced 0 without it; device
  `#error` if defined without `MK_PDF_PRODUCER`). Default OFF everywhere:
  `model.py::attn_pdf_feed_default = 0`; flag-off extension name unchanged.
- **Request format (spec blocker #2 resolved)**: ONE generic mailbox kind
  (`kind=4`) covers BOTH streaming directions, because the dkv LSE/Drow
  scalar prefetch stays on the consumer cp.async path (spec's own
  recommendation). Both directions then reduce to "stream two 64x64 bf16
  tiles per stage into `wga_off64` slabs": per operand a gmem row-0 base
  (`tmA/tmB`), a gmem row stride in bytes (`m0/n0` — dq: `stride*2` both;
  dkv: `stride*2` and `nq*D*2`), and a per-stage gmem byte step
  (`k_base/bk` = `C*64*rowstride`), plus dst slabs `a0/b0` with per-stage
  smem strides `a_stride/b_stride` (8KB), `iters=n_stages`, `stages=2`.
  Field reuse documented on the `MkPdfFeed` struct; no struct change.
- **Producer topology**: kind-4 requests are served WARPGROUP-WIDE — all 128
  WG2 threads issue 4 x 16B `__pipeline_memcpy_async` slices per operand tile
  with the exact consumer-loader geometry (`v = ptid + i*128`; per `i`:
  `r += 16`, `c8` fixed, `wga_off64 += 2048`, src `+= 16*rowstride`), so the
  destination bytes are identical to the consumer path by construction. GEMM
  TMA replay (kinds 0-3) stays on elected thread 256 unchanged; threads
  257-383 track `seq` and skip GEMM requests. The consumer inc-region idiom
  (taken branch ending in `return`) is untouched.
- **Handshake** (replaces the consumers' commit-group pairing for streamed
  stages): per-tile 2-deep mbarrier ring past the smem struct
  (dq: offset 81920, dkv: 99328; +32B, fits the 100KB carveout).
  `bfull[st]` count-128, fired via `cp.async.mbarrier.arrive.noinc` per
  producer thread (the proven n256 mbar-ring pattern); consumers
  `wg_mbar_wait(bfull[st], (t>>1)&1)`. `bempty[st]` count-1, armed by
  consumer tid0 BEHIND the existing end-of-stage `consumer_sync` (all WGMMA
  helpers end in `warpgroup_wait<0>`, so slot release is safe); producer
  waits `(t/stages - 1)&1` before slot reuse. Barriers are (re)initialized by
  tid0 before each request post, with a REQUIRED `consumer_sync` between
  init and any consumer wait; the `st.release` on `seq` orders init for the
  producer's acquire. Serial-request invariant holds for kind 4: the
  consumer's last bfull wait needs all 128 producer arrivals, so the
  producer finished issuing tile T before tile T+1 can post.
- **Consumers keep**: owned-tile loads (own commit group; `wait_prior(0)` at
  t=0), all softmax/dS math, dq RS-feed, fp32-P, exp2-prebias, direct-store
  epilogues. In dkv the `issue_qdo_stage` lambda keeps LSE/Drow + the same
  one-commit-per-stage group structure, so the `wait_prior` expressions are
  unchanged under feed.
- **Executor hardening**: `g_pdf_feed.active` is now also cleared by the
  `waves` and `df2` executor images under
  `#if defined(MK_PDF_PRODUCER) && defined(MK_ATTN_PDF_FEED)` (they could
  previously dispatch attention ops with garbage `active` in forced-env
  builds; df/ws/pdf already handled it). Flag-off images unchanged.

## Gate outcomes

### Compile / res-usage (spec blocker #4)

Builds in `TORCH_EXTENSIONS_DIR=/tmp/torch-ext-b1d36305-apdff*`:

| image | REG | STACK | LOCAL | WG2 region max R | consumer region max R | consumer LDL/STL |
|---|---|---|---|---|---|---|
| pdf240p flag-off (base 2a41f6a) | 168 | 80 | 0 | — | 237 | 34/87 |
| pdf240p flag-off (worktree) | 168 | 80 | 0 | — | 237 | 34/87 |
| pdf240p_apdff (=1) | 168 | 96 | 0 | **R21** | 237 | 34/87 |
| pdf240p_apdff2 (=2) | 168 | 96 | 0 | **R21** | 237 | 34/87 |

- **240/24 fits — no pool split needed.** SASS proof: the WG2 dec region
  (between `USETMAXREG.DEALLOC.CTAPOOL 0x18` and `TRY_ALLOC 0xf0`) tops out
  at R21 (<= R23 budget; URs are the separate uniform bank, unaffected by
  setmaxnreg). The 232/48 / 224/56 splits stay unused.
- The +16 STACK is 3 STL/3 LDL of per-thread invariants stashed ACROSS the
  poll loop, once per REQUEST (not per stage) — that is what lets the region
  fit 24 registers; negligible for a dedicated producer.
- Consumer inc region is IDENTICAL flag-on vs flag-off (max R237, same
  LDL/STL counts): the added consumer branches cost no registers/spills.
- No-collapse check: consumer region allocates at 240 (max R237), not the
  168 entry cap; region idiom preserved.

### Flag-off bit-identity

`cuobjdump -sass` of the flag-off pdf240p extension built from the worktree
vs from pristine base `2a41f6a`: **identical except the embedded source-path
identifier line** (4 diff lines total = 1 path hunk). Flag-off default-name
extensions are byte-for-byte the same build inputs (kwarg/env absent).

### Layout/handshake proof (standalone, pdf executor, GPU 3, production
attention body flags: exp2+prebias, dq rs-feed, dq fp32-P, dq float2 store,
dkv float2 atomic)

`results/attn_pdf_feed_probe_b1d36305.py` — feed-on (`_apdff2`) vs feed-off
(`pdf240p`), `mode="pdf"`:

- dq S8192 C=1 (direct-store epilogue): **BITWISE identical**.
- dq S8192 C=1 under `mode="df"` with the feed extension (active=0 dormant
  fallback): **BITWISE identical** to the feed-off pdf run.
- dq S8192 C=4 (chunked, multi-request ring reuse): rel_err 0.0 vs feed-off;
  1.5e-07 vs the C=1 golden (fp32 atomic order, expected).
- dkv S8192 G=2 C=1: rel_err 0.0.
- dkv S2048 G=1 C=1 (single atomic per address): **BITWISE identical**.
- combined dkv+dq instructions in ONE pdf launch (mailbox request
  interleaving across ops): rel_err 0.0.
- No hangs anywhere (mbarrier ring + halt path clean under timeout guard).

### Standard parity gates (GPU 3)

- `test_ops.py`: green flag-off, `=1`, `=2` (ALL OP TESTS PASSED).
- `test_model.py`: green flag-off, `=1`, `=2` (worst grad rel err
  0.0172-0.0177, normal band; rerun stable; waves/df2/ws agreement OK —
  exercises the dormant fallback and the new `active` clears).

## Phase 2 (feed-only in-model prototype, s8192, default-off)

Paired both-order A/B via the `env_ab_main.py` harness (model A = defaults,
model B = defaults + env), GPU 3 (0% util guards before/after each window,
defaults stable 6411-6434us across all four windows), 16 reps, step medians:

| variant | order | default | variant | delta | wins | worst_grad |
|---|---|---|---|---|---|---|
| `MK_ATTN_PDF_FEED=1` (dq feed) | default_first | 6414.03us | 6367.31us | **-46.72us** | 16/16 | 0.0041 |
| `MK_ATTN_PDF_FEED=1` (dq feed) | variant_first | 6419.36us | 6339.95us | **-79.41us** | 16/16 | 0.0041 |
| `MK_ATTN_PDF_FEED=2` (+dkv feed) | default_first | 6434.40us | 6375.41us | **-58.99us** | 16/16 | 0.0053 |
| `MK_ATTN_PDF_FEED=2` (+dkv feed) | variant_first | 6410.91us | 6353.89us | **-57.02us** | 15/16 | 0.0043 |

- **dq feed-only is a both-order win at s8192** (-46.7/-79.4us, 32/32) —
  the spec's Phase 2 go criterion is met, and the effect is larger than the
  "modest" expectation.
- =2 (dq+dkv) also wins both orders (-59.0/-57.0us) but sits INSIDE the =1
  band: dkv feed adds no clear win on top of dq in this prototype window
  (consistent with dkv's wait-column being narrower than dq's at s8192). A
  direct =1-vs-=2 paired A/B plus the certified re-cert battery should
  decide which value (if any) promotes.
- Route parity: n_instr 188 both sides (feed is executor-internal, no
  program change). Loss identical to 5 decimals in all four runs.

## Go / no-go

- Phase 0: PASS. Phase 1: PASS (both directions proven, dq and dkv).
- Phase 2 prototype: **GO signal** — dq feed-only improves both construction
  orders at s8192. Still default-off; promotion (default-on at exact S8192,
  =1 vs =2 choice, cross-shape no-widen check per the resweep law) requires
  the standard certified battery and is the main session's call.
- Phase 3 (softmax/dS offload): out of scope for this claim; the mailbox +
  warpgroup-wide producer + 240/24-fit groundwork it needs is now in place.
  Note for Phase 3 sizing: the pure cp.async issue producer peaked at R21 of
  the 24-reg dec budget — any ALU offload WILL need the 232/48 split as the
  spec predicted.
