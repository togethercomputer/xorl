# Qwen head-DX first-RMS update-split parity pass

Date: 2026-07-10

Agent: codex

Source tree: `/tmp/xorl-oss-qwen-hdxrmsupd-42e5b1d-codex`

Source checkpoint: `875fa39132e207d4d47cf272da2b138fc4e19483` on frontier `42e5b1d3ff537ed0a988d73fcaa0b88650fbf510`

Scope: parity only for the default-off `_hdxrmsupd` candidate after the SASS/resource gate passed. No timing, graph, default-policy, remote isolated, or shared checkout source edit was performed. local GPU was not used.

## Artifacts

- Helper: `results/qwen_hdxrmsupd_parity_42e5b1d.py`
  - sha256 `2d7af6c9a512cd58997f48d7c3f3d628387ffe531758b712539f356f09cba65d`
  - `304` lines / `11355` bytes
- Log: `results/qwen-hdxrmsupd-parity-localgpu-875fa39-20260710T0045Z.log`
  - sha256 `731d353655b0aa762182295371ee7ec5eee31864e3f1dfc1fede8d873d82b879`
  - `7` lines / `10424` bytes
  - `JOB_START 2026-07-10T00:43:46Z`
  - `JOB_DONE rc=0 2026-07-10T00:49:36Z`
- Summary: `results/qwen-hdxrmsupd-parity-summary-875fa39-20260710T0045Z.json`
  - sha256 `4bd962e8981762ffd7e09ce38dd573245c45a04e97824ad06e94ea3483bf7f6d`
  - `569` lines / `17394` bytes
  - `pass=true`, `sha=875fa39`
- Runtime cache: `/tmp/torchext-qwen-hdxrmsupd-parity-875fa39-20260710T0045Z`

## Gate

The helper compared default `_hdxrmsdot` against candidate `_hdxrmsdot_hdxrmsupd` on qwen4b-l2 in both construction orders:

- `default_first`
- `candidate_first`

Tolerances:

- loss absolute tolerance: `0.005`
- gradient relative tolerance: `0.05`
- gradient absolute tolerance: `0.0005`

Both orders passed the route gate and full-step parity:

| order | loss diff | worst grad | worst abs | worst rel | pass |
| --- | ---: | --- | ---: | ---: | --- |
| `default_first` | `-4.76837158203125e-06` | `emb` | `0.000244140625` | `0.009615384615384616` | yes |
| `candidate_first` | `-1.9073486328125e-06` | `emb` | `0.000244140625` | `0.009615384615384616` | yes |

Candidate route checks:

- route/page: `78/44/14`
- shared memory: `151552`
- sidecar boundary rows: `1`
- head-DX row: `idx=40`, `ntiles=80`, `has_rmsdot_flag=true`, `has_rmsupd_flag=true`
- head-DX update args: `dx_arg=23`, `rstd_arg=74`, `partials_arg=80`, `nparts_arg=10`
- first RMS dX row: `idx=42`, `ntiles=64`, `has_partials=true`, `partials_arg=80`, `nparts_arg=10`, `correction_only=1`

The parity result addresses the SASS gate caveat: the candidate writes the first RMS update through bf16 `dX` before applying the correction, but full-step loss and all gradients are within the declared tolerances in both construction orders.

## Validation

- `python3 -m json.tool results/qwen-hdxrmsupd-parity-summary-875fa39-20260710T0045Z.json` PASS
- parity log tail ends `JOB_DONE rc=0`
- lock `results/.gpulock-qwen-hdxrmsupd-parity-875fa39-localgpu` is absent after completion
- local GPU observed `0 MiB, 0%` after the run
- pending final scoped checks: `git diff --check` over this note, helper, summary, log, and `results/AGENT-COORDINATION.md`

## Verdict

`_hdxrmsupd` is now SASS/resource positive and parity positive. The next required gate is timing/graph A/B on qwen4b-l2, comparing default `_hdxrmsdot` against candidate `_hdxrmsdot_hdxrmsupd` in both construction orders before any default-policy promotion.
