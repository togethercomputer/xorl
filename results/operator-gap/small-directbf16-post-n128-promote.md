# Small direct-BF16 GEMM epilogue post-n128 promotion

Date: 2026-07-06
Base: `2276fd7` (`megakernel: record post-n128 attention chunk check`)

## Question

The earlier direct-BF16 GEMM epilogue probe promoted only exact S128. H512/S1024
small previously lost or was neutral. After the 4W/cache and NN/NT n128 route
retunes, however, many H512/S1024 BF16-output GEMM rows moved back onto the
ordinary WGMMA body where `MK_GEMM_DIRECT_BF16_EPILOGUE` can bypass the shared
fp32 accumulator drain.

## Source-free recheck

Command log:

`results/mkv3-p4b-small-directbf16-post-n128-20260706T1905Z.log`

Forced `MK_GEMM_DIRECT_BF16_EPILOGUE=1` changed no route shape:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `GEMM=99/13856`
- direct-eligible WGMMA rows: `32/5632` tiles

Eligible row families:

- 8x `GEMMNN 1024x512x1024`, flags `128`, 512 tiles
- 8x `GEMMNN 1024x512x3072`, flags `128`, 512 tiles
- 8x `GEMMNN 1024x1536x512`, flags `128`, 1536 tiles
- 8x `GEMMNT 1024x3072x512`, flags `130`, 3072 tiles

Parity stayed clean:

- default-first: `loss_diff=-9.53674316e-07`, worst selected grad `kn.0`,
  relative error `3.599130e-07`.
- direct-first: `loss_diff=+9.53674316e-07`, worst selected grad `qn.0`,
  relative error `5.291102e-07`.

Forced direct won both construction orders:

- default-first: default `3292.50us`, direct `3266.40us`,
  default-minus-direct `+27.82us`, direct wins `80/80`.
- direct-first: default `3287.68us`, direct `3261.63us`,
  default-minus-direct `+27.01us`, direct wins `80/80`.

## Promoted default validation

Source now enables the direct-BF16 GEMM epilogue by default for exact
H512/L8/S1024 small, while preserving the earlier S128 default. Force old with
`MK_GEMM_DIRECT_BF16_EPILOGUE=0`.

Promoted-default vs forced-old log:

`results/mkv3-p4b-small-directbf16-promoted-20260706T1915Z.log`

Route shape and eligible rows matched the source-free recheck. Parity stayed
clean:

- default-first: `loss_diff=-9.53674316e-07`, worst selected grad `kn.0`,
  relative error `4.049021e-07`.
- old-first: `loss_diff=+9.53674316e-07`, worst selected grad `kn.0`,
  relative error `5.398695e-07`.

Promoted default beat forced old both orders:

- default-first: default `3276.99us`, old `3304.03us`,
  default-minus-old `-28.82us`, old wins `1/80`.
- old-first: default `3262.70us`, old `3289.86us`,
  default-minus-old `-23.71us`, old wins `2/80`.

Validation:

- `py_compile` for `model.py`
- `git diff --check`
- `test_model.py`: `ALL MODEL TESTS PASSED`
- `test_ops.py`: `ALL OP TESTS PASSED`

Refreshed promoted-default profile:

- log: `results/mkv3-p4b-profile-small-directbf16-default-20260706T1925Z.log`
- total: `3320.1us`

Refreshed promoted-default score:

- log: `results/mkv3-p4b-score-small-directbf16-default-20260706T1925Z.log`
- megakernel: `3312.5us`
- eager: `16960.5us`
- compile: `4195.1us`
- compile+cudagraph: `2093.7us`
- compile+cudagraph+: `1903.3us`

## Decision

Promote the exact H512/L8/S1024 small default. The broad conclusion remains
unchanged: do not enable direct-BF16 GEMM epilogues for all D64 shapes without
new evidence.
