# qwen NT sidecar dirty-frontier hardening pass @7abb355

Date: 2026-07-09
Agent: codex
Claim: qwen NT sidecar promotion hardening @7abb355
Source: `/tmp/xorl-oss-qwen-nt-sidecar-manual-be55098-codex`
Base: dirty-frontier snapshot `be55098`
Candidate commit: `7abb355`
Shared checkout source edits: none
local GPU: not used

## Verdict

PASS. The manual qwen NT sidecar dirty-frontier candidate passed standard GPU tests and the focused route/resource/SASS hardening gate on local GPU.

This hardening does not itself promote the patch into the shared checkout or default production branch. It raises `7abb355` from "timing promote candidate" to "standard-test and SASS-hardened promote candidate". Remaining recommended gates are optional racecheck and an integrated dirty-frontier qwen production profile after applying/promoting the patch in an isolated integration worktree.

## Standard GPU Tests

Ran sequentially on local GPU from the clean source worktree at `7abb355`, with private torch extension caches.

- `experiments/fused-training-megakernel/test_ops.py`: PASS, `rc=0`
- `experiments/fused-training-megakernel/test_model.py`: PASS, `rc=0`

Artifacts:

- `results/qwen-nt-sidecar-hardening-summary-7abb355-20260709T2201Z.json`
  - sha256 `e6a743158d121fb7b8317b16625ca8141d41e167ec3dd3fc466dfff2f6e30a2b`
  - `1` line / `512` bytes
- `results/qwen-nt-sidecar-hardening-testops-localgpu-7abb355-20260709T2201Z.log`
  - sha256 `b3c600e77099fec5eed1d1f7da9ca71abcf96977fc147645766d9a293dc3223d`
  - `50` lines / `3434` bytes
  - ends `ALL OP TESTS PASSED`
- `results/qwen-nt-sidecar-hardening-testmodel-localgpu-7abb355-20260709T2201Z.log`
  - sha256 `986d14e29e97d9dea900051bd023ace2cf96db0f76fa745abe4117d0a68e0c5e`
  - `77` lines / `3722` bytes
  - ends `ALL MODEL TESTS PASSED`

Model-test coverage included nano D64 and D128 configs, rerun stable, waves-vs-df, df2, warp-specialized scheduler, graph replay, graph input rewrite, training sanity, and graphed training sanity.

## SASS And Resource Gate

Helper:

- `results/qwen_nt_sidecar_dirty_default_sass_7abb355.py`
  - sha256 `0d5c55cd1cde9ab67cacc3b632472d9691142ec9e3b9e280aeb749b9cc169a13`
  - `288` lines / `10360` bytes
  - `py_compile` PASS

Run artifacts:

- `results/qwen-nt-sidecar-dirty-default-sass-localgpu-7abb355-20260709T2209Z.log`
  - sha256 `5e26ccb35aecef7a8f4ddd34fe8d53deb24e243207c56baa585f50cb70997be8`
  - `5` lines / `19163` bytes
  - wrapper `JOB_DONE rc=0 2026-07-09T22:21:11Z`
- `results/qwen-nt-sidecar-dirty-default-sass-summary-7abb355-20260709T2209Z.json`
  - sha256 `5d993b8455273469340b8ad2b690c08868a6cefab21e424873c3251115c6e554`
  - `348` lines / `13060` bytes
  - `pass=true`

### Route And SASS Summary

`qwen4b-l1 forced_old`:

- Route `n_instr=47 critical_path=26 gated=9 smem=151552`
- Direct NT head row: flat `22`, `op=3`, `ntiles=4748`
- No `_ntscbnd`, `api_available=false`, `step_requested=false`
- `megakernel_pdf`: `REG168 STACK96 SHARED1264 LOCAL0`
- Counts: `HGMMA=212`, `HGMMA.64x128=16`, `HGMMA.64x256=48`, `CALL=148`, `LDL_STL=117`, `TMA=149`, `max_hgmma_between_depbar=1`

`qwen4b-l1 promoted_default`:

- Route `n_instr=47 critical_path=26 gated=9 smem=151552`
- `_ntscbnd=true`, one boundary row at flat `22`, `op=39`, `ntiles=4748`
- `step_requested=true`, `split_plan_valid=true`, `main_row_replaced_by_boundary=true`
- `megakernel_pdf`: `REG168 STACK96 SHARED1264 LOCAL0`, same top-level HGMMA/CALL/LDL_STL/TMA counts as forced-old
- `qwen_nt_lmhead_sidecar`: `REG168 STACK0 SHARED1248 LOCAL0`
- Sidecar counts: `HGMMA=8`, `HGMMA.64x128=8`, `CALL=0`, `LDL_STL=0`, `TMA=12`, `WARPGROUP.ARRIVE=1`, `WARPGROUP.DEPBAR=2`, `max_hgmma_between_depbar=8`

`qwen4b-l2 forced_old`:

- Route `n_instr=78 critical_path=44 gated=14 smem=151552`
- Direct NT head row: flat `37`, `op=3`, `ntiles=4748`
- No `_ntscbnd`, `api_available=false`, `step_requested=false`
- `megakernel_pdf`: `REG168 STACK112 SHARED1264 LOCAL0`
- Counts: `HGMMA=216`, `HGMMA.64x128=16`, `HGMMA.64x256=52`, `CALL=127`, `LDL_STL=177`, `TMA=153`, `max_hgmma_between_depbar=1`

`qwen4b-l2 promoted_default`:

- Route `n_instr=78 critical_path=44 gated=14 smem=151552`
- `_ntscbnd=true`, one boundary row at flat `37`, `op=39`, `ntiles=4748`
- `step_requested=true`, `split_plan_valid=true`, `main_row_replaced_by_boundary=true`
- `megakernel_pdf`: `REG168 STACK112 SHARED1264 LOCAL0`, same top-level HGMMA/CALL/LDL_STL/TMA counts as forced-old
- `qwen_nt_lmhead_sidecar`: `REG168 STACK0 SHARED1248 LOCAL0`
- Sidecar counts: `HGMMA=8`, `HGMMA.64x128=8`, `CALL=0`, `LDL_STL=0`, `TMA=12`, `WARPGROUP.ARRIVE=1`, `WARPGROUP.DEPBAR=2`, `max_hgmma_between_depbar=8`

## SASS Artifacts

Hashes and sizes:

- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l1-forced_old-resusage-7abb355.txt`
  - sha256 `232224d4b8a57e848615aaa289650769b2579dd426b4b0175f1cc926cab33f01`
  - `23` lines / `969` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l1-forced_old-megakernel_pdf-7abb355.sass`
  - sha256 `9b3946ca5c7e57ef5008728ea247a39ce9c12da53cffcc92cd1e1b0453bfad78`
  - `154721` lines / `21194910` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l1-forced_old-qwen_nt_lmhead_sidecar-7abb355.sass`
  - sha256 `3f67c5e849af8d54b4bfc25119dd4dbeeffe3c7d8d61c738f31ea6f36416af03`
  - `11` lines / `249` bytes
  - expected missing-symbol warning
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l1-promoted_default-resusage-7abb355.txt`
  - sha256 `506e0e574246208b55fc1c36ac5eb4ba8a3c782d2e07212aa891ed7a2fbeceb3`
  - `25` lines / `1087` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l1-promoted_default-megakernel_pdf-7abb355.sass`
  - sha256 `5297148db437aaac836cafd4257cb79c2d1c4bf2278f24501989b4cc575442d6`
  - `154705` lines / `21192718` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l1-promoted_default-qwen_nt_lmhead_sidecar-7abb355.sass`
  - sha256 `d6b61462b83635020e088fc1ad8b57787dd90cc3e72776b3105985053238e8e3`
  - `5121` lines / `592534` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l2-forced_old-resusage-7abb355.txt`
  - sha256 `7964aa9a40982fc2832e6c285c4b1b71b6ce2e9bc68786d4412b7e5751032826`
  - `23` lines / `970` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l2-forced_old-megakernel_pdf-7abb355.sass`
  - sha256 `4bcf707f234733b7f6cd52a07f48f10760716c22ab621652b0c411d4ae78a322`
  - `159841` lines / `21896350` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l2-forced_old-qwen_nt_lmhead_sidecar-7abb355.sass`
  - sha256 `3f67c5e849af8d54b4bfc25119dd4dbeeffe3c7d8d61c738f31ea6f36416af03`
  - `11` lines / `249` bytes
  - expected missing-symbol warning
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l2-promoted_default-resusage-7abb355.txt`
  - sha256 `4a8f83c7dd0026c9ada7603560a9a14e239e0a413d7d7ac809fb79aa5dcbda0a`
  - `25` lines / `1088` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l2-promoted_default-megakernel_pdf-7abb355.sass`
  - sha256 `f9b852b30ae542e2fd277a9b918eb4129007e3d7321a45f6f9c09c3a89c99364`
  - `159841` lines / `21896350` bytes
- `results/operator-gap/qwen-nt-sidecar-dirty-default-qwen4b-l2-promoted_default-qwen_nt_lmhead_sidecar-7abb355.sass`
  - sha256 `d21b6b4c5fb4cd44f3dfcb935c5970e90752bf67a1e5e5abd6af7e302d30aac8`
  - `5121` lines / `592534` bytes

## Interpretation

The top-level PDF kernel resource usage is unchanged by the boundary sidecar route for both qwen4b L1 and L2. L1 remains `STACK96`, L2 remains `STACK112`, both remain `LOCAL0`, and the main-kernel HGMMA/CALL/LDL_STL/TMA counts are flat versus forced-old.

The promoted-default route replaces the direct lm-head NT row with a boundary op (`op=39`) and emits a separate `qwen_nt_lmhead_sidecar` symbol with real GMMA work. The sidecar is spill-free (`LOCAL0`, `STACK0`) and has the expected `8` `HGMMA.64x128` instructions with no calls or local load/store churn.

Together with the earlier local and remote isolated timing/parity evidence, this removes the main promotion risk that the timing win was hiding a resource/SASS regression.

## Remaining Work

Recommended next gates before merging/promoting:

1. Optional `compute-sanitizer --tool racecheck` or equivalent on the promoted-default qwen4b L1/L2 path.
2. Apply/promote `7abb355` into an isolated integration worktree and rerun the current dirty-frontier integrated qwen production profile, so the board has an end-to-end profile after the sidecar route is active in the intended branch shape.
3. If profile remains positive, prepare the narrow PR/patch with rollback envs documented:
   - `MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY=0`
   - `MK_QWEN_NT_SIDECAR_STEP=0`

## Resource Cleanup

Local GPU lock `results/.gpulock-qwen-nt-sidecar-sass-7abb355-localgpu` was released by the wrapper. local GPU was observed free (`0 MiB`, `0%`) after the run. No remote isolated resources were launched for this hardening gate.
