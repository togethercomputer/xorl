# qwen NT sidecar integrated L2 profile @7abb355

Date: 2026-07-09 UTC
Agent: codex
Source worktree: `/tmp/xorl-oss-qwen-nt-sidecar-manual-be55098-codex`
Commit: `7abb355`
GPU: local GPU
Private cache: `/tmp/torchext-qwen-nt-sidecar-integrated-profile-7abb355-20260709T2222Z`
Shared checkout source edits: none
local GPU: not used

## Verdict

PASS as the post-hardening integrated qwen4b-l2 timing/profile anchor for the sidecar candidate.

The old `profile_df` helper output is retained only as a main-kernel attribution view. It is not a full sidecar step meter, because the sidecar candidate runs the real step as prefix + sidecar + post launches while `profile_df.profile()` times a plain `prog.run()` trial after warmup.

The sidecar-aware and step-only follow-ups provide the actual end-to-end result:

- Fresh no-graph step-only rerun: promoted wins both orders.
  - old-first: `9477.376us -> 9154.080us`, delta `-323.296us`
  - promoted-first: `9673.728us -> 9209.504us`, delta `-464.224us`
- Graph replay in the sidecar-aware helper: promoted wins both orders.
  - old-first: `9565.728us -> 9127.392us`, delta `-438.336us`
  - promoted-first: `9659.424us -> 9321.984us`, delta `-337.440us`

The transient mixed ungraphed step result from the sidecar-aware helper (`old_first -246.368us`, `promoted_first +427.392us`) is classified as a helper-state/order artifact, because the fresh no-graph step-only rerun immediately recovered both-order wins with fresh model builds and no CUDA graph capture/replay.

## Route

Forced-old route:

- extension suffix: `_ntstnosync_hdxexpdf_pdf240p`
- `n_instr=78`, `critical_path=44`, `gated=14`, `smem_bytes=151552`
- lm-head NT row: flat `37`, op `3`, `ntiles=4748`
- sidecar API unavailable, `policy_requested=false`

Promoted-default route:

- extension suffix: `_ntstnosync_ntscbnd_hdxexpdf_pdf240p`
- `n_instr=78`, `critical_path=44`, `gated=14`, `smem_bytes=151552`
- lm-head NT row replaced by boundary row: flat `37`, op `39`, `ntiles=4748`
- one boundary row, one cutpoint, valid split plan
- prefix/post subprograms: `37` / `40` instructions
- prefix/post critical paths: `20` / `23`
- sidecar tile range: `[0, 4748]`
- `policy_requested=true`

## End-to-End Timing

Sidecar-aware helper:

- helper: `results/qwen_nt_sidecar_integrated_profile_7abb355.py`
- helper sha256: `b088cd2571ca4f3373e9b0d9d6963a49c29a06d623e6742f42207db52255efad`
- helper size: `329` lines / `11172` bytes
- `py_compile`: PASS
- log: `results/qwen-nt-sidecar-integrated-profile-aware-localgpu-7abb355-20260709T2227Z.log`
- log sha256: `8830f035499f899a6d4c097f75c8d0e14f9c298754b74c44cf4b4f154f32d497`
- log size: `1` line / `6898` bytes
- summary: `results/qwen-nt-sidecar-integrated-profile-aware-summary-7abb355-20260709T2227Z.json`
- summary sha256: `42ff329b997c2bc9c628c51af44f41d4ab39880e3fb0c50a17a23d1513c320fd`
- summary size: `371` lines / `10949` bytes
- wrapper: `JOB_DONE rc=0 2026-07-09T22:29:29Z`

Sidecar-aware graph timings:

| Order | Forced-old graph median | Promoted graph median | Delta |
| --- | ---: | ---: | ---: |
| old-first | `9565.728us` | `9127.392us` | `-438.336us` |
| promoted-first | `9659.424us` | `9321.984us` | `-337.440us` |

Promoted component timings:

| Component | Median |
| --- | ---: |
| prefix | `1969.088us` |
| sidecar | `1379.712us` |
| post | `5815.168us` |
| prefix+sidecar+post kernel total | `9174.528us` |

Step-only no-graph rerun:

- helper: `results/qwen_nt_sidecar_step_only_rerun_7abb355.py`
- helper sha256: `a67fb2763dc96adc762cb1aa82bac84c721e68f47338d920636882371568369e`
- helper size: `183` lines / `6196` bytes
- `py_compile`: PASS
- log: `results/qwen-nt-sidecar-step-only-rerun-localgpu-7abb355-20260709T2230Z.log`
- log sha256: `6528cc24572a0c054a50580cc5fc457bebc150e699ddfce34eb11a49d2cce37b`
- log size: `1` line / `5116` bytes
- summary: `results/qwen-nt-sidecar-step-only-rerun-summary-7abb355-20260709T2230Z.json`
- summary sha256: `9eec8daa8f0705d5d0f2349feeae0d3b88368dfad92217883766d2d00330bbff`
- summary size: `261` lines / `7906` bytes
- wrapper: `JOB_DONE rc=0 2026-07-09T22:30:34Z`

Step-only timings:

| Order | Forced-old step median | Promoted step median | Delta |
| --- | ---: | ---: | ---: |
| old-first | `9477.376us` | `9154.080us` | `-323.296us` |
| promoted-first | `9673.728us` | `9209.504us` | `-464.224us` |

## Main-Kernel Attribution View

The older helper is still useful for residual-bucket triage, with the caveat above.

- helper: `results/qwen_l2_integrated_profile_3b5701e.py`
- log: `results/qwen-nt-sidecar-integrated-profile-localgpu-7abb355-20260709T2222Z.log`
- log sha256: `53f7772275ade8b30a55441dda3d2b39a4e78b6bd2b36d45936ecd5d22bed6e8`
- log size: `54` lines / `6150` bytes

Route/profile facts from that log:

- extension suffix includes `_ntscbnd`
- `target_rows=[]` because the direct `OP_GEMM` lm-head NT row is gone
- `QWEN_NT_SIDECAR_BOUNDARY` appears on-path at `20.2us`
- plain `prog.run()` profile median is `7670.0us`, but this excludes the real sidecar split step and must not be quoted as end-to-end step time

Largest remaining on-path buckets in the caveated main-kernel attribution view:

| Bucket | On-path us | Upper-bound us |
| --- | ---: | ---: |
| `GEMMNN 1024x2560x151936.wg` | `1629.7` | `1629.7` |
| `RMSNORM_BWD_DX` | `1560.2` | `1560.2` |
| `ATTN_DQ_WG128` | `547.0` | `547.0` |
| `GEMMNT 1024x19456x2560.wg` | `538.7` | `538.7` |
| `GEMMNN 1024x2560x19456.wg` | `457.6` | `457.6` |
| `ATTN_DKV_WG128` | `0.0` | `419.5` |
| `GEMMTN 151936x2560x1024.wg` | `0.0` | `2935.8` |

## Interpretation

This closes the qwen lm-head NT sidecar integration gap for the current dirty-frontier candidate. The promoted route is active in the end-to-end `step()` path, graph capture works, and both fresh no-graph step timings and graph timings show a several-hundred-microsecond win.

Do not use the old helper's `7670.0us` profile total as a scoreboard step number. Use the step-only and graph timing tables above for end-to-end sidecar timing. Use the old helper only for residual main-kernel attribution after the direct lm-head NT GEMM row has been replaced by the boundary row.

The residual qwen L2 performance frontier is now:

1. head-dX: `GEMMNN 1024x2560x151936.wg`, about `1629.7us` on-path in the caveated attribution view;
2. RMS dX: `RMSNORM_BWD_DX`, about `1560.2us` on-path;
3. attention DQ row-split: `ATTN_DQ_WG128`, about `547.0us` on-path;
4. off-path lm-head dW only if a scheduler/overlap change can make its upper bound actionable.

## Remaining Promotion Work

Before merging/promoting into the intended branch:

1. Optional racecheck/synccheck on the promoted sidecar split path.
2. Prepare the narrow integration patch/PR from `7abb355`.
3. Keep rollback envs documented:
   - `MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY=0`
   - `MK_QWEN_NT_SIDECAR_STEP=0`

## Cleanup

local GPU lock `results/.gpulock-qwen-nt-sidecar-integrated-profile-7abb355-localgpu` was released. local GPU was observed free (`0 MiB`, `0%`) after the runs. No remote isolated resources were launched for this profile gate.
