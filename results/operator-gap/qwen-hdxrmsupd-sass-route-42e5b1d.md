# qwen head-DX first-RMS update-split SASS gate @42e5b1d

## Verdict

PASS for route/SASS/resource only. Do not promote yet.

The candidate source package makes the first final-RMS dX update a split protocol:
head-DX still writes the RMS-dot partials, and now also applies the first RMS
term `r * dy * w` into `W["dX"]`; the first final RMS dX row then consumes the
same partials and applies only the correction term.

This is not a parity/timing/default-policy result. Numerics are not a pure
refactor because the candidate writes the first term through bf16 `dX` before
the correction, while the old body rounds after combining first term and
correction. Parity must be measured before any runtime or policy promotion.

## Source Package

- Base frontier: `42e5b1d3ff53` (`default qwen head-dx rmsdot partials`).
- Isolated worktree: `/tmp/xorl-oss-qwen-hdxrmsupd-42e5b1d-codex`.
- Local source checkpoint: `875fa39132e207d4d47cf272da2b138fc4e19483` (`add qwen head-dx first rms update split gate`).
- Shared checkout source: untouched.
- Patch artifact: `results/qwen-hdxrmsupd-42e5b1d-20260710T0025Z.patch`, sha256 `b21f4cec9008f173f67edb41971dae13a31c3184ff1f35bd080045e495e50234`, `261` lines / `12492` bytes.

## Gate Artifacts

- Helper: `results/qwen_hdxrmsupd_sass_42e5b1d.py`, sha256 `4a50b8ff7fe7fc22b4ae33ed30f4183917c0d720f16d73abf1c53ede1457a304`, `359` lines / `12943` bytes.
- Canonical log: `results/qwen-hdxrmsupd-sass-gate-42e5b1d-20260710T0034Z.log`, sha256 `f91d14179a306c1f4391f3928ee84a1664169134ebdba39d9e60c570d633c933`, `8` lines / `7384` bytes, ended `JOB_DONE rc=0`.
- Summary JSON: `results/qwen-hdxrmsupd-sass-summary-42e5b1d-20260710T0034Z.json`, sha256 `0b6971f10ac39dd876fea210462a815047812b2e43b18381a8cd1cb2407e30ec`, `305` lines / `10008` bytes.
- SASS/resusage directory: `results/qwen-hdxrmsupd-sass-42e5b1d-20260710T0034Z/`.

Superseded failed-helper launches:

- `20260710T0027Z`: default image compiled, route audit failed at the CUDA-tensor assertion in `Program.buf`.
- `20260710T0031Z`: default image compiled, route audit failed when CPU pointers reached CUDA tensor-map encoding.

Both were helper CPU-route issues only; neither emitted a SASS verdict.

## Route Contract

Default `_hdxrmsdot` and candidate `_hdxrmsdot_hdxrmsupd` both preserved the qwen4b-l2 PDF route:

- `n_instr=78`, `critical_path=44`, `gated=14`.
- `smem_bytes=151552`.
- NT sidecar boundary active: `sidecar_boundary_rows=1`.
- Head-DX row: `idx=40`, `ntiles=80`, shape `(1024, 2560, 151936)`.
- First final RMS dX row: `idx=42`, `ntiles=64`.

Candidate-specific route proof:

- Head-DX kept RMS-dot partials and added RMSUPD: default `has_rmsupd_flag=false`, candidate `has_rmsupd_flag=true`.
- Candidate head-DX args added live `dX`/`rstd` buffers: `arg13_dx=23`, `arg14_rstd=74`.
- First final RMS dX consumed the same partial buffer: head `arg9_partials=80`, RMS `arg9_partials=80`, `arg10_nparts=10`.
- Candidate first RMS row has correction-only sentinel `arg11_correction_only=1`; default has `0`.

## Resource and SASS

`megakernel_pdf` resource usage did not regress:

| image | REG | STACK | SHARED | LOCAL |
| --- | ---: | ---: | ---: | ---: |
| default `_hdxrmsdot` | 168 | 96 | 1264 | 0 |
| candidate `_hdxrmsdot_hdxrmsupd` | 168 | 96 | 1264 | 0 |

`megakernel_pdf` SASS changed while control structure stayed stable:

| count | default | candidate | delta |
| --- | ---: | ---: | ---: |
| HGMMA | 216 | 216 | 0 |
| WARPGROUP.DEPBAR | 216 | 216 | 0 |
| CALL | 127 | 127 | 0 |
| F2F | 2302 | 2430 | +128 |
| FFMA | 3786 | 4074 | +288 |
| FMUL | 5319 | 5767 | +448 |
| LDG | 897 | 899 | +2 |
| STL | 128 | 128 | 0 |

This is enough for the first SASS/resource gate: the candidate branch is present,
the route contract changed exactly where intended, and no LOCAL/stack/call growth
appeared. It does not prove runtime speed or numerical parity.

## Next Gate

The next useful gate is a qwen4b-l2 parity check for `_hdxrmsupd` against current
`_hdxrmsdot`, focused on `dX`, `wf` grad, and downstream gradients. If parity is
clean or within an explicitly acceptable tolerance, then run a timing/graph A/B.
If parity fails, the source likely needs a different contract that avoids the
extra bf16 round, such as keeping the first term in an fp32 side buffer until the
correction is applied.

No GPU kernel launch, timing, graph capture, default-policy promotion, remote isolated job,
or local GPU use occurred in this gate.
