# Qwen head-DX first-RMS update-split timing no-go

Date: 2026-07-10

Agent: codex

Source tree: `/tmp/xorl-oss-qwen-hdxrmsupd-42e5b1d-codex`

Source checkpoint: `875fa39132e207d4d47cf272da2b138fc4e19483`

Scope: qwen4b-l2 timing and graph replay only, comparing default `_hdxrmsdot` against candidate `_hdxrmsdot_hdxrmsupd` after the SASS/resource and parity gates passed. No shared checkout source edit, default-policy promotion, remote isolated launch, or local GPU use was performed.

## Artifacts

- Helper: `results/qwen_hdxrmsupd_timing_42e5b1d.py`
  - sha256 `cbca0e810cdfc3825a972b75e68ca63d04ead9c3b1954323ebde7dee16def2d2`
  - `488` lines / `18272` bytes
- Log: `results/qwen-hdxrmsupd-timing-localgpu-875fa39-20260710T0052Z.log`
  - sha256 `e9a706abc2ad56ae02e25337041587f0d4e4202ba82ed27daa7b003027f07488`
  - `11` lines / `27685` bytes
  - `JOB_START 2026-07-10T00:52:28Z`
  - `JOB_DONE rc=0 2026-07-10T00:58:39Z`
- Summary: `results/qwen-hdxrmsupd-timing-summary-875fa39-20260710T0052Z.json`
  - sha256 `30ed42bad242d056179b79e261f3facf81d46ab0ad69e58343875eb74eb52eb5`
  - `1382` lines / `41497` bytes
  - `diagnostic_complete=true`
  - `timing_positive=false`
  - `pass=false`
- Runtime cache: `/tmp/torchext-qwen-hdxrmsupd-timing-875fa39-20260710T0052Z`

## Route And Parity Guard

Both construction orders passed the route guard and parity guard before timing:

- route/page: `78/44/14`
- shared memory: `151552`
- sidecar boundary rows: `1`
- candidate head-DX row: `idx=40`, `ntiles=80`, `has_rmsdot_flag=true`, `has_rmsupd_flag=true`, `dx_arg=23`, `rstd_arg=74`, `partials_arg=80`, `nparts_arg=10`
- first RMS dX row: `idx=42`, `ntiles=64`, `has_partials=true`, `partials_arg=80`, `nparts_arg=10`, `correction_only=1`
- worst grad in both orders: `emb`, abs `0.000244140625`, rel `0.009615384615384616`
- loss diffs: `-3.814697265625e-06` and `-5.7220458984375e-06`

## Timing Result

64 repetitions were run per order with 8 warmup iterations and 3 graph-capture warmup iterations.

| order | mode | default median us | candidate median us | candidate minus default us | paired mean us | wins | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `default_first` | eager step | `9220.000267028809` | `9193.74418258667` | `-26.256084442138672` | `-23.948952555656433` | `46/64` | positive |
| `default_first` | graph replay | `9260.208129882812` | `9231.823921203613` | `-28.38420867919922` | `-20.955443382263184` | `44/64` | positive |
| `candidate_first` | eager step | `9204.544067382812` | `9210.335731506348` | `+5.791664123535156` | `+12.003511190414429` | `27/64` | negative |
| `candidate_first` | graph replay | `9215.216159820557` | `9220.720291137695` | `+5.504131317138672` | `-1.3214647769927979` | `30/64` | negative |

The candidate is order-mixed: it wins when timed second in `default_first`, but it does not survive the reverse construction/timing order. This is not promotion-grade.

## Verdict

`_hdxrmsupd` remains a useful SASS/parity-clean reference, but timing/graph evidence is negative for default-policy promotion. Do not promote this candidate as-is. A follow-up source lane would need to reduce the extra head-DX epilogue pressure or remove enough work from the first RMS dX body to win in both construction orders.

## Validation

- helper `py_compile` PASS before launch
- helper `--help` PASS before launch
- summary `json.tool` PASS after run
- wrapper exited `JOB_DONE rc=0`
- timing lock was released after completion
- local GPU observed `0 MiB, 0%` after completion
- final scoped `git diff --check` PASS after the timing note and coordination close
