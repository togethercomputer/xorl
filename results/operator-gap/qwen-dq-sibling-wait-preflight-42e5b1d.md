# qwen attention-DQ sibling wait preflight @42e5b1d

Date: 2026-07-10
Claim: `qwen attention-DQ sibling wait preflight @42e5b1d`
Status: SOURCE-FREE PREFLIGHT / NO SOURCE SLOT YET

## Scope

This lane checks the next-ranked non-N256 wait-path target after the resident
N256 branch SASS no-go. It is deliberately source-free: no shared source edit,
no isolated source patch, no CUDA extension build, no GPU or remote isolated launch, no
timing claim, and no default-policy change.

The question was whether qwen D128 `ATTN_DQ_WG128` has a non-duplicate
producer-readiness source hook, or whether the local DQ-ready-first idea is
already closed by the S8192 ready-order no-promote evidence.

## Artifacts

- helper: `results/qwen_dq_sibling_wait_preflight_42e5b1d.py`
  - sha256: `9d589a96cda24529cf891796e9aa39bcab6c5248b0c31621e95624599d7b8aed`
  - size: `283` lines / `9249` bytes
- summary JSON:
  `results/qwen-dq-sibling-wait-preflight-42e5b1d-20260710T0030Z.json`
  - sha256: `22ddc5904481f4af7322e46d1e12d840459779ca3278d5fc4ed153311db512f8`
  - size: `2961` lines / `63933` bytes
- source tree inspected:
  `/tmp/xorl-oss-qwen-hdxrmsdot-7abb355-codex`
  - head: `42e5b1d3ff53`

The helper stubs `mk.load_ext`, uses CPU tensors, suppresses TMA descriptor
injection, and reuses the real host `Program._build_deps(...)` path. It builds
the host route/dependency graph without compiling or loading a CUDA extension.

## Current Profile Context

The current `42e5b1d` qwen4b-l2 profile still shows DQ wait, not DQ local span,
as the attention-DQ issue:

- `ATTN_DQ_WG128`: `509.4us` on path
- wait: `304.6us`
- span: `204.7us`
- worst-hop rows:
  - row `54`: wait `152.1us`, span `102.7us`
  - row `70`: wait `152.5us`, span `102.0us`

The older qwen DQ inventory says the same thing structurally: D128 rowsplit is
already promoted, and remaining qwen attention-DQ is mostly wait. Local body
variants such as descriptor cleanup, C1 body clone, exp2-prebias, final-sync
deletion, and tensor-map reduce are closed.

## Route Findings

qwen4b-l1 route:

- `n_instr=47`
- `critical_path=26`
- `gated=9`
- mode `pdf`
- `smem_bytes=151552`

qwen4b-l1 DKV/DQ pair:

- DKV row `38`, DQ row `39`
- shared producers: `[9, 10, 14, 15, 36]`
- all shared producers order DKV before DQ

qwen4b-l2 route:

- `n_instr=78`
- `critical_path=44`
- `gated=14`
- mode `pdf`
- `smem_bytes=151552`

qwen4b-l2 DKV/DQ pairs:

- DKV row `53`, DQ row `54`
  - shared producers: `[15, 16, 29, 30, 51]`
  - all shared producers order DKV before DQ
- DKV row `69`, DQ row `70`
  - shared producers: `[13, 14, 20, 21, 51, 67]`
  - all shared producers order DKV before DQ

Diagnosis counts:

- qwen4b-l1 has `1` DQ-tail pair and `5` producer groups with DQ-tail fanout
- qwen4b-l2 has `2` DQ-tail pairs and `10` producer groups with DQ-tail fanout
- all qwen DQ-tail pairs are DKV-first across their shared producers

## Verdict

Do not open a qwen DQ-only tail-ready-first source patch.

The narrow mechanism is structurally available, but it duplicates the S8192
ready-order no-promote class:

- S8192 tail-ready-first collapsed DQ tail wait from roughly `431-434us` to
  roughly `3us`
- selected profile still got slower by `+67.520us` and `+104.864us`
- the wait moved to sibling DKV rows instead of shrinking the combined tail

qwen has the same sibling shape: each current DQ tail row is paired with a DKV
row over the same producer set, and DKV is currently first. A DQ-only rank swap
would most likely trade the qwen DQ wait cells for qwen DKV wait cells, which
the S8192 profile already proved is not a promotion path.

## Next Useful DQ Gate

The next useful DQ lane must be a combined sibling-set objective, not a local
DQ body patch and not a DQ-only adjacency reorder.

Candidate class:

`qwen D128 DQ/DKV sibling co-schedule protocol preflight`

Minimum gate before source patch:

- preserve qwen4b-l1 route `47/26/9`
- preserve qwen4b-l2 route `78/44/14`
- preserve C=1 rowsplit DQ rows
- reason over each DKV/DQ pair as a unit:
  - L1 pair `(38,39)`
  - L2 pairs `(53,54)` and `(69,70)`
- source hook must reduce combined `ATTN_DQ_WG128 + ATTN_DKV_WG128` wait, not
  just swap which sibling appears on the realized path
- first GPU profile must show selected total down and no transfer of wait from
  DQ to DKV before paired timing

Possible mechanisms to preflight next:

- executor-level paired hint or two-hint completion path for D128 DKV/DQ sibling
  groups
- a host route annotation that lets both siblings be claimed as a pair without
  changing CUDA body codegen
- a deliberate DKV/DQ pair topology experiment that keeps dependency sets
  equivalent but changes the ready/claim protocol for the pair

Reject duplicate mechanisms:

- DQ-only ready-first adjacency
- local DQ body cleanup
- DQ claim-size-only retune
- descriptor-only or exp2-only DQ changes

## Verification

Passed:

```bash
python3 -m py_compile results/qwen_dq_sibling_wait_preflight_42e5b1d.py
python3 results/qwen_dq_sibling_wait_preflight_42e5b1d.py \
  --summary results/qwen-dq-sibling-wait-preflight-42e5b1d-20260710T0030Z.json
git diff --check -- \
  results/qwen_dq_sibling_wait_preflight_42e5b1d.py \
  results/qwen-dq-sibling-wait-preflight-42e5b1d-20260710T0030Z.json
```

No source edit or GPU/remote isolated work was performed.
