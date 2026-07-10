# qwen head-DX first-RMS update split @42e5b1d

Date: 2026-07-10
Claim: `qwen head-DX first-RMS update-split SASS gate @42e5b1d`
Status: SASS/ROUTE/DEPENDENCY PASS / NO PROMOTION

## Scope

This lane tested a default-off qwen4b-l2 candidate that extends the existing
head-DX RMS-dot partial path. The candidate makes the final head-DX epilogue
also apply the first final-RMS dX update term `r * dy * w` into `W["dX"]`, then
marks the first final RMS dX row as correction-only so it applies only
`-r * xhat * m`.

This is not a parity, timing, graph, or default-policy result. It did not
launch a kernel and did not use GPU/remote isolated. local GPU was not touched.

## Source Package

- source worktree: `/tmp/xorl-oss-qwen-hdxrmsupd-42e5b1d-codex`
- source checkpoint: `875fa39132e2` (`add qwen head-dx first rms update split gate`)
- base frontier: `_hdxrmsdot` qwen chain at `42e5b1d3ff53`
- shared checkout source: untouched
- source diff scope: `experiments/fused-training-megakernel/mk.py`,
  `model.py`, and `ops.cuh`
- replay patch:
  `results/qwen-hdxrmsupd-42e5b1d-20260710T0025Z.patch`
  - sha256: `b21f4cec9008f173f67edb41971dae13a31c3184ff1f35bd080045e495e50234`
  - size: `261` lines / `12492` bytes

Static source gates passed before SASS:

```bash
python3 -m py_compile experiments/fused-training-megakernel/mk.py \
  experiments/fused-training-megakernel/model.py
git diff --check
```

## SASS And Resource Gate

- helper: `results/qwen_hdxrmsupd_sass_42e5b1d.py`
  - sha256: `4a50b8ff7fe7fc22b4ae33ed30f4183917c0d720f16d73abf1c53ede1457a304`
  - size: `359` lines / `12943` bytes
- log: `results/qwen-hdxrmsupd-sass-gate-42e5b1d-20260710T0034Z.log`
  - sha256: `f91d14179a306c1f4391f3928ee84a1664169134ebdba39d9e60c570d633c933`
  - size: `8` lines / `7384` bytes
  - ended `JOB_DONE rc=0 2026-07-10T00:40:44Z`
- summary:
  `results/qwen-hdxrmsupd-sass-summary-42e5b1d-20260710T0034Z.json`
  - sha256: `0b6971f10ac39dd876fea210462a815047812b2e43b18381a8cd1cb2407e30ec`
  - size: `305` lines / `10008` bytes
  - `pass=true`

The helper compiled and disassembled:

- default `_hdxrmsdot` image
  - `.so` sha256:
    `d8d9a7d65a4137e9c6ae915671b326ffd01d3d400aaf5d14c0aadc81bfccb09c`
  - full SASS sha256:
    `9c05025ec21408c31ceef3d5d2dbdab1eb02eac849b983e909ca13a0e458d48c`
  - `megakernel_pdf` SASS sha256:
    `22de372948126c2d85d62639fa930019e663fba56534964230380619a56ffb77`
- candidate `_hdxrmsdot_hdxrmsupd` image
  - `.so` sha256:
    `27a5b0ebdf2617f839396007e91d9296b99a2f05df76898844ab5f0d899d90f2`
  - full SASS sha256:
    `7e5fc0548f3cfd73b256ad749f6c842bebcc883f5bda0da5a0bf100bfba13dc7`
  - `megakernel_pdf` SASS sha256:
    `6882fe1f31d9f7bb89e122e96deaa9ebd345045b2afa2aff866e4702dcf67eec`

Passing checks:

- qwen4b-l2 route stayed `78/44/14`
- smem stayed `151552`
- NT sidecar boundary remained available
- default image suffix had `_hdxrmsdot` and not `_hdxrmsupd`
- candidate image suffix had `_hdxrmsdot_hdxrmsupd`
- head-DX row stayed flat row `40`, `ntiles=80`
- first partial RMS dX row stayed flat row `42`, `ntiles=64`
- candidate head-DX row has `GEMM_HEADDX_RMSUPD_FLAG`
- candidate head-DX row passes dX buffer `23` and rstd buffer `74`
- candidate first partial RMS dX row has correction-only arg `11 == 1`
- `megakernel_pdf` resources did not regress against this default image:
  - default: `REG168 STACK96 SHARED1264 LOCAL0`
  - candidate: `REG168 STACK96 SHARED1264 LOCAL0`

The SASS body changed, but the total `megakernel_pdf` counts increased in the
candidate:

- `FFMA`: `3786 -> 4074`
- `FMUL`: `5319 -> 5767`
- `F2F`: `2302 -> 2430`
- `LDG`: `897 -> 899`
- `HGMMA`: unchanged at `216`
- `WARPGROUP.DEPBAR`: unchanged at `216`
- `CALL`: unchanged at `127`

So this gate proves the candidate is SASS/resource legal, not that it is faster.

## Dependency Audit

- helper: `results/qwen_hdxrmsupd_dependency_audit_42e5b1d.py`
  - sha256: `1aee11d15c078db6ab44b29e7e702853a3e28b9e3836c4661188214a61683d79`
  - size: `274` lines / `9225` bytes
- summary:
  `results/qwen-hdxrmsupd-dependency-audit-42e5b1d-20260710T0041Z.json`
  - sha256: `9af9ea16e7d55539cfcd141ea03a765e4ed2d7e3583d567cc7d5de049f92ab1e`
  - size: `177` lines / `3985` bytes
  - `pass=true`

Dependency findings:

- qwen4b-l1 remains unchanged under the env toggle:
  - default and enabled routes both `47/26/9`
  - `rmsdot_enabled=false`
  - `rmsupd_enabled=false`
- qwen4b-l2 stays route-stable:
  - default and enabled routes both `78/44/14`
  - both keep `151552` smem
- candidate qwen4b-l2 dependency contract:
  - head-DX row `40`
  - first partial RMS dX row `42`
  - row `42` deps include row `40`: deps `[17, 35, 36, 40]`
  - shared partial buffer: `80`
  - shared dX buffer: `23`
  - correction-only arg: `1`

This proves the first RMS correction row is ordered after the head-DX update
through the shared partial buffer, while both rows touch the same `dX` buffer.

## Verdict

PASS for source packaging, SASS/resource legality, route stability, qwen4b-l1
guarding, and host dependency contract.

Do not promote this candidate yet. The candidate writes the first RMS update
through bf16 `dX` before the correction row, and the candidate SASS adds visible
arithmetic. The next gate must be qwen4b-l2 parity before any timing claim.
Only if parity is clean should a GPU timing gate compare default `_hdxrmsdot`
against `_hdxrmsdot_hdxrmsupd` in both construction orders.

## Verification

Passed:

```bash
python3 -m py_compile results/qwen_hdxrmsupd_sass_42e5b1d.py
python3 -m py_compile results/qwen_hdxrmsupd_dependency_audit_42e5b1d.py
python3 -m json.tool \
  results/qwen-hdxrmsupd-sass-summary-42e5b1d-20260710T0034Z.json >/dev/null
python3 results/qwen_hdxrmsupd_dependency_audit_42e5b1d.py \
  --summary results/qwen-hdxrmsupd-dependency-audit-42e5b1d-20260710T0041Z.json
```

No shared source edit, kernel launch, parity run, timing run, graph capture,
default-policy claim, remote isolated job, or local GPU use was performed.
