# Qwen integration chain @42e5b1d

Date: 2026-07-09 UTC
Agent: codex

## Verdict

Integration branch assembled, static/route gates passed, same-commit GPU
evidence was verified, fresh qwen profile refresh passed, the full 12-shape
scoreboard completed, and the chain is now folded into the canonical
`<repo-root>` `megakernel` checkout.

Branch `megakernel` now has the intended qwen source chain plus one doc-only
phase-closeout commit:

```text
3b5701e megakernel: land CE fwd warprow + qknorm bwd vec8
be55098 snapshot dirty qwen frontier
7abb355 port qwen nt sidecar to dirty frontier
965a0c1 add qwen head-dx rmsdot partials gate
42e5b1d default qwen head-dx rmsdot partials
9eb4d03 megakernel: record operator-gap phase notes
```

Canonical worktree:

```text
<repo-root>
branch: megakernel
head: 9eb4d03
qwen source/evidence anchor: 42e5b1d
```

The temporary replay worktree
`/tmp/xorl-oss-qwen-integration-codex-20260709T235457Z` and branch
`qwen-integration-codex-20260709T235457Z` were removed after the canonical
checkout was fast-forwarded.

## Source Packaging

Exported replay artifacts:

```text
results/qwen-integration-chain-42e5b1d-20260709T2359Z.mbox
sha256 622d30ec9e17b6a73a7f26e51e17266bdc575eb99080aede6354b54f93d45350
size 363004 bytes

results/qwen-integration-chain-42e5b1d-20260709T2359Z.patch
sha256 4fbbc9853bc88c9ddfecab64972ba2a1a8c0ae2b75055bb4edbf63a1164f81b0
size 347731 bytes
```

The old text patch
`results/qwen-dirty-frontier-source-3b5701e-20260709T2126Z.patch` did not apply
cleanly to pristine `3b5701e`. The equivalent durable commit `be55098` exists,
is an ancestor of `7abb355`, and was used instead of replaying stale patch
context.

## Gates Run

Static:

- `py_compile` passed for `mk.py` and `model.py`.
- `py_compile` passed for the sidecar route/default-policy helpers and
  head-DX RMS-dot helper modules.
- `git diff --check 3b5701e..HEAD` passed.

Qwen NT sidecar host route:

- Helper: `results/qwen_nt_sidecar_dirty_manual_route_be55098.py`
- Result: `pass=true`, `errors=[]`
- L1 default: one `_ntscbnd` boundary row at flat `22`, route `47/26/9`,
  `smem=151552`, split plan valid, direct NT target removed.
- L2 default: one `_ntscbnd` boundary row at flat `37`, route `78/44/14`,
  `smem=151552`, split plan valid, direct NT target removed.
- Forced-old rollback restores the direct GEMM row for L1/L2.
- `MK_QWEN_NT_SIDECAR_STEP=0` keeps the boundary API but disables default step
  request.

Qwen head-DX RMS-dot no-GPU route:

- Route built with extension loading stubbed.
- Result: `pass=true`.
- Rollback route: no `_hdxrmsdot`, one head-DX row at `idx=40`, no partials.
- Policy route: `_hdxrmsdot` present with `_ntscbnd` and `_hdxexpdf`, route
  `78/44/14`, `smem=151552`.
- Policy head-DX row writes partials arg `80`, `nparts=10`, `x=2`, `wf=72`.
- First RMS dX row `idx=42` consumes partials arg `80`, `nparts=10`; later RMS
  dX rows do not consume partials.

## Verified Existing GPU Evidence

These GPU gates were not relaunched from
`/tmp/xorl-oss-qwen-integration-codex-20260709T235457Z`, but they were verified
from durable notes on the same source commit content:

- `results/operator-gap/qwen-nt-sidecar-dirty-hardening-7abb355-pass.md`:
  `test_ops.py` and `test_model.py` passed for the sidecar source package, and
  qwen4b L1/L2 sidecar SASS/resource stayed `LOCAL0`; L2 route stayed
  `78/44/14`, `smem=151552`, with the sidecar symbol at `STACK0 LOCAL0` and
  `max_hgmma_between_depbar=8`.
- `results/operator-gap/qwen-nt-sidecar-integrated-profile-7abb355.md`:
  qwen4b-l2 sidecar graph replay won both orders (`-438.336us`,
  `-337.440us`) and fresh no-graph step-only replay won both orders
  (`-323.296us`, `-464.224us`).
- `results/operator-gap/qwen-hdxrmsdot-default-42e5b1d-positive.md`:
  default-policy `_hdxrmsdot` selected with no env, rollback selected without
  `_hdxrmsdot`, route stayed `78/44/14`, `smem=151552`, graph capture/parity
  passed both orders, and timing was positive (`-10.784us` median in
  rollback-first, `-36.976us` median in policy-first).
- `results/operator-gap/qwen-hdxrmsdot-hardening-42e5b1d-pass.md`:
  generic `test_ops.py` and `test_model.py` passed on clean worktree
  `/tmp/xorl-oss-qwen-hdxrmsdot-7abb355-codex` at `42e5b1d`; `test_model.py`
  worst grad rel errors stayed below the `0.03` threshold.
- `results/operator-gap/qwen-sidecar-hdxrmsdot-profile-42e5b1d-pass.md`:
  fresh sidecar-aware qwen profile/timing passed for both qwen4b-l1 and
  qwen4b-l2 on `42e5b1d`. qwen4b-l2 step deltas were `-291.712us` and
  `-882.144us`; graph deltas were `-402.112us` and `-768.960us`. qwen4b-l1
  step deltas were `-31.840us` and `-415.424us`; graph deltas were
  `-404.064us` and `-381.440us`.

## Full Scoreboard

`results/operator-gap/qwen-integration-scoreboard-42e5b1d-pass.md` records the
full 12-shape scoreboard from the integration worktree with `rc=0`, `pyrc=0`,
and 12 JSON rows.

Artifacts:

```text
results/qwen-integration-scoreboard-localgpu-42e5b1d-20260710T0008Z.log
sha256 8f147a3294820b7d66e80a197522c63bd91f50bc57a95055572d910d0e0ae3c5

results/qwen-integration-scoreboard-summary-42e5b1d-20260710T0008Z.jsonl
sha256 cfa4f85780a0d9b895feca71abdf00d072866d50f77cfb264e6950ef001753eb
rows 12
```

Scoreboard interpretation:

- qwen4b-l1 was a full-score win in this sweep: `6961.952us` vs
  `7314.464us`, ratio `0.9518`.
- qwen4b-l2 was close but still behind the `compile+cudagraph+` baseline:
  `9134.784us` vs `8815.360us`, ratio `1.0362`.
- Standard broad shapes remain slower than `compile+cudagraph+`; this sweep is
  an execution/score artifact and does not replace rollback-specific promotion
  evidence.

## Gates Not Relaunched Here

The earlier GPU hardening/profile gates were run or verified on clean source worktree
`/tmp/xorl-oss-qwen-hdxrmsdot-7abb355-codex` at the same `42e5b1d` commit,
not from `/tmp/xorl-oss-qwen-integration-codex-20260709T235457Z` itself.
The integration scoreboard was originally run from
`/tmp/xorl-oss-qwen-integration-codex-20260709T235457Z` through `MK_AB_TREE`.
That worktree has since been removed because `megakernel` now points at the
same validated source content. The branch HEAD is `9eb4d03` because a doc-only
operator-gap phase note commit was added after the qwen source anchor.

Do not treat this note as replacing the prior GPU evidence in:

- `qwen-nt-sidecar-manual-dirty-7abb355-promote.md`
- `qwen-nt-sidecar-dirty-hardening-7abb355-pass.md`
- `qwen-nt-sidecar-integrated-profile-7abb355.md`
- `qwen-hdxrmsdot-default-42e5b1d-positive.md`
- `qwen-hdxrmsdot-hardening-42e5b1d-pass.md`
- `qwen-sidecar-hdxrmsdot-profile-42e5b1d-pass.md`

## Next

Remaining integration work after landing to `megakernel`:

```text
optional racecheck/synccheck on the qwen sidecar split path
rerun qwen profile/scoreboard if future source changes touch the qwen route
```

Keep rollback envs documented:

```text
MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY=0
MK_QWEN_NT_SIDECAR_STEP=0
MK_QWEN_HEADDX_RMS_DOT_PARTIALS=0
```
