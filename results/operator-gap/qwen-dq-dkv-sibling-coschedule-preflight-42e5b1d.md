# qwen DQ/DKV Sibling Co-Schedule Preflight @ 42e5b1d

Date: 2026-07-10
Status: SOURCE-FREE DESIGN PREFLIGHT / NO SOURCE SLOT YET

## Scope

This is the follow-on to `qwen-dq-sibling-wait-preflight-42e5b1d.md`.
It does not edit shared source, create a source patch, build CUDA, launch a GPU
kernel, or claim timing. The goal is to pin the minimum credible source bar for
the qwen D128 DQ wait lane after DQ-only ready ordering was classified as a
duplicate of the S8192 no-promote class.

## Evidence Reused

- qwen route/dependency helper:
  `results/qwen_dq_sibling_wait_preflight_42e5b1d.py`
- qwen route/dependency summary:
  `results/qwen-dq-sibling-wait-preflight-42e5b1d-20260710T0030Z.json`
- prior source-free note:
  `results/operator-gap/qwen-dq-sibling-wait-preflight-42e5b1d.md`
- S8192 executor no-promote class:
  `results/operator-gap/s8192-multi-sibling-fanout-executor-preflight-0ba235d.md`
- live qwen gap register:
  `results/qwen-live-gap-register-42e5b1d-20260710T0016Z.json`
- current executor source read only:
  `experiments/fused-training-megakernel/megakernel.cu`

## Pair Map

The route preflight already proves the qwen DQ tails are paired DKV/DQ sibling
sets, not isolated DQ leaves.

qwen4b-l1:

- route: `47/26/9`, `pdf`, `smem=151552`
- pair: `(38,39)`
- DKV row `38`: `ATTN_DKV_WG128`, `ntiles=512`, `claim=1`
- DQ row `39`: `ATTN_DQ_WG128`, `ntiles=256`, `claim=1`, `arg10=16777217`
- shared producers: `[9,10,14,15,36]`
- every shared producer orders DKV before DQ

qwen4b-l2:

- route: `78/44/14`, `pdf`, `smem=151552`
- pair: `(53,54)`
  - DKV row `53`: `ATTN_DKV_WG128`, `ntiles=512`, `claim=1`
  - DQ row `54`: `ATTN_DQ_WG128`, `ntiles=256`, `claim=1`, `arg10=16777217`
  - shared producers: `[15,16,29,30,51]`
- pair: `(69,70)`
  - DKV row `69`: `ATTN_DKV_WG128`, `ntiles=512`, `claim=1`
  - DQ row `70`: `ATTN_DQ_WG128`, `ntiles=256`, `claim=1`, `arg10=16777217`
  - shared producers: `[13,14,20,21,51,67]`
- every shared producer orders DKV before DQ
- producer `51` fans out to all four L2 sibling rows in dependency order
  `[53,54,69,70]`

Current qwen4b-l2 profile context still shows DQ wait as the exposed issue:
rows `54` and `70` together contribute about `304.6us` wait in the preflight
note, while the live gap register records comparable row-level DQ waits
(`156.992us` and `159.712us`). The register does not expose enough row-level
DKV wait to certify a combined improvement without a fresh profile.

## Executor Contract Read

Both df and pdf carry a single sticky instruction and a single completion hint:

- df sticky claim: `megakernel.cu:436-459`
- df hot-ring scan/head advance: `megakernel.cu:460-487`
- df completion hint: `megakernel.cu:559-579`
- pdf sticky claim: `megakernel.cu:674-694`
- pdf hot-ring scan/head advance: `megakernel.cu:695-721`
- pdf completion hint: `megakernel.cu:813-831`
- ws also keeps a single `hint` on completion: `megakernel.cu:1484-1505`

This is exactly why adjacency lanes can erase one local wait while worsening
the selected profile: the finisher pushes all ready dependents, but only one hot
dependent becomes the local sticky successor. The hot-ring consumed head then
lets that one instruction drain claims before sibling rows become equally cheap.

The S8192 multi-sibling preflight is therefore applicable to qwen: a DQ-only or
DKV-first/DQ-first rank swap can choose which sibling waits, but it is not a
credible promotion path unless the combined DKV plus DQ sibling wait falls.

## Credible Source Shape

A future source lane should be opened only if it implements a group-aware
successor mechanism, not another local body or adjacency retune.

Minimum acceptable mechanism:

- Keep normal dependency semantics and ready-ring publication unchanged.
- Detect a bounded hot sibling group among dependents that became ready from the
  same producer completion.
- Represent only the qwen D128 DKV/DQ sibling sets as groups:
  - L1 `(38,39)`
  - L2 `(53,54)`
  - L2 `(69,70)`
  - producer `51` group `[53,54,69,70]`
- Use scalar per-block hint state, not a variable local-memory array, to avoid
  stack/local growth in the 240-register pdf consumer region.
- Rotate claims across the hinted group with a bounded quantum so one row does
  not monopolize the sticky path until its cursor drains.
- Do not preclaim lookahead work and do not add busy-loop cursor rescans; both
  are in the measured-negative ws/preclaim class.

One plausible prototype is a default-off qwen-only executor flag that adds up to
four scalar group-hint registers (`g0..g3`, `g_n`, `g_next`) in the df/pdf
completion path. When a producer completion enables a recognized sibling set,
the finisher records all ready siblings as group hints while still pushing the
same ready-ring entries. The next claim attempt tries the group hint in
round-robin order before falling back to `last_ins` and the hot ring. The group
hint must expire when every member cursor is drained or when none can be claimed
without additional probing.

This is deliberately narrower than a general executor rewrite: it tests whether
qwen's paired D128 rows can share immediate successor privilege without changing
the graph or broad scheduler regime.

## Reject Duplicate Source Attempts

Do not open source work for:

- DQ-only ready-first or DQ-rank swaps
- DKV-first versus DQ-first pair ordering without group claims
- local DQ body cleanup, descriptor rewrites, split-K, sidecar, helper, or final
  sync changes
- claim-size-only retunes
- lookahead/preclaim or busy-rescan schemes already rejected by ws/S8192 notes

## Required Gates For A Source Prototype

Before timing:

- build from a clean sibling worktree, not the shared checkout
- default-off env/cflag only
- route unchanged:
  - qwen4b-l1 `47/26/9`, `smem=151552`
  - qwen4b-l2 `78/44/14`, `smem=151552`
- DQ rows remain C=1 rowsplit:
  - `ATTN_DQ_WG128`, `ntiles=256`, `claim=1`, `arg10=16777217`
- DKV rows remain:
  - `ATTN_DKV_WG128`, `ntiles=512`, `claim=1`
- dependency sets for `(38,39)`, `(53,54)`, `(69,70)` remain identical to the
  source-free preflight
- SASS/resource gate for `megakernel_pdf` shows no new LOCAL memory, no stack
  growth, no unexpected device calls, and no register-limit regression

First GPU gate:

- run a profile, not only step timing
- report selected total and row-level wait/span for rows `53`, `54`, `69`, `70`
- require combined wait for those four rows to fall, not just row `54` or `70`
- reject if DQ wait falls but DKV wait rises enough to keep combined sibling
  wait flat or worse
- reject if selected profile total regresses, even when local DQ wait improves

Only after that should a paired full-step timing gate run.

## Verdict

This lane is ready for a source-design implementation slot, but only for a
group-aware sibling successor protocol. The artifact explicitly rules out a
quick adjacency or DQ-only source patch, because qwen has the same single-hint
multi-sibling shape as the closed S8192 no-promote evidence.
