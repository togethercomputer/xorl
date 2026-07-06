# Small N128 NN Post-4W Promotion

Date: 2026-07-06

Base: `b552463` (`megakernel: record post-4w gmbar check`)

Verdict: **promote** a narrower H512/S1024 small default: keep the non-NN
m64n128 routes, but route the repeated MLP dX NN bf16 rows
`GEMMNN 1024x512x3072`, `GEMMNN 1024x512x1024`, and
`GEMMNN 1024x1536x512` through the normal m64n64 WGMMA path.

## Candidate

Source change in `wgmma_n128_ok()`:

- `MK_WGMMA_N128_NN=0` still disables all NN n128 rows.
- `MK_WGMMA_N128_NN=1` still forces the old NN n128 behavior for A/B.
- with the env unset, only `(M,N,K) in {(1024,512,3072), (1024,512,1024),
  (1024,1536,512)}` falls back to m64n64.

Shape:

- `Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)`

## Evidence

Logs:

- `results/mkv3-p4b-small-n128nn-post4w-20260706T1610Z.log`
- `results/mkv3-p4b-small-n128nn-promoted-explicit-20260706T1640Z.log`
- `results/mkv3-p4b-small-n128nn-promoted-route-explicit-20260706T1644Z.log`
- `results/mkv3-p4b-profile-small-n128nn-default-20260706T1655Z.log`
- `results/mkv3-p4b-score-small-n128nn-default-20260706T1656Z.log`
- `results/mkv3-p4b-small-n128nn-broad-after-promote-20260706T1710Z.log`
- `results/mkv3-p4b-small-n128nn-broad-after-promote-repeat-20260706T1715Z.log`
- `results/mkv3-p4b-small-n128nn-final-promoted-20260706T1725Z.log`
- `results/mkv3-p4b-profile-small-n128nn-final-default-20260706T1732Z.log`
- `results/mkv3-p4b-score-small-n128nn-final-default-20260706T1735Z.log`

The initial source-free knob `MK_WGMMA_N128_NN=0` showed the opportunity, but it
was broader than the final source change:

| Order | Default | NN n128 off | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3412.38us` / `3415.91us` | `3348.38us` / `3349.98us` | default-minus-non128 `+63.90us` median, `+65.93us` mean, non128 wins `80/80` |
| non128 first | `3391.73us` / `3397.83us` | `3334.72us` / `3336.18us` | default-minus-non128 `+56.69us` median, `+61.66us` mean, non128 wins `80/80` |

The promoted source route kept parity clean and changed only the intended NN
rows in the small MLP dX stack:

- default route: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=34`,
  `splitK=33`
- forced old `MK_WGMMA_N128_NN=1`: `n_instr=288`, `critical_path=144`,
  `gated=127`, `n128=50`, `splitK=33`
- 16 rows move from n128 flags `4224`, `32` tiles each, to m64n64 flags `128`,
  `64` tiles each: 8x `GEMMNN 1024x512x3072` and 8x
  `GEMMNN 1024x512x1024`

Promoted default vs forced old timing, 80 CUDA-event samples per arm:

| Order | Default | Forced old NN n128 | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3351.14us` | `3380.14us` | old-minus-default `+28.67us` median, `+28.01us` mean, default wins `78/80` |
| old first | `3350.34us` | `3379.15us` | old-minus-default `+28.40us` median, `+29.47us` mean, default wins `77/80` |

Parity:

- default-first loss diff: `-1.90734863e-06`, worst selected grad `qn.0`
  rel `3.968325e-07`
- old-first loss diff: `-2.86102295e-06`, worst selected grad `kn.0`
  rel `5.398695e-07`

The earlier `results/mkv3-p4b-small-n128nn-promoted-20260706T1620Z.log` is
discarded as decision evidence: its route print showed the default side with
`n128=0`, which is not the intended promoted route. The explicit rerun above
cleared the env knobs and printed the correct route.

Refreshed default profile:

- `profile_df.py small df`: `3374.9us`, down from the previous post-4W/cache-off
  profile `3421.6us`
- on-path `GEMMNN 1024x512x3072.wg`: `19.5us` wait + `254.1us` span
- on-path `GEMMNN 1024x512x1024.wg`: `20.4us` wait + `127.3us` span

Refreshed small score:

| Variant | Time |
| --- | ---: |
| megakernel | `3392.1us` |
| eager | `17213.5us` |
| compile | `5179.1us` |
| compile+cudagraph | `2094.9us` |
| compile+cudagraph+ | `1896.8us` |

Remaining small gap vs compile+cudagraph+ is `1.79x`.

Validation:

- `py_compile mk.py model.py profile_df.py`
- `git diff --check`
- `test_model.py`
- `test_ops.py`

## Decision

Keep this as a shape-specific H512/S1024 post-4W retune. It is smaller than the
source-free all-NN-n128-off knob, survives construction-order reversal, preserves
the existing force-on/force-off env gates, and reduced the small score from the
previous post-cache-off `3425.9us` to the final `3373.2us`.

## Follow-Up Row Family

After the first narrow promotion, current default was compared against broad
`MK_WGMMA_N128_NN=0`. The only remaining route delta was 8x
`GEMMNN 1024x1536x512`, from n128 flags `4224` with `96` tiles to m64n64 flags
`128` with `192` tiles.

First pass, 80 samples per arm:

| Order | Default | Broad NN n128 off | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3359.06us` | `3346.56us` | default-minus-broadoff `+12.29us` median, `+13.62us` mean, broadoff wins `66/80` |
| broadoff first | `3347.84us` | `3340.14us` | default-minus-broadoff `+7.57us` median, `+6.98us` mean, broadoff wins `58/80` |

Repeat pass, 160 samples per arm:

| Order | Default | Broad NN n128 off | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3351.65us` | `3339.33us` | default-minus-broadoff `+11.76us` median, `+11.27us` mean, broadoff wins `127/160` |
| broadoff first | `3345.02us` | `3337.44us` | default-minus-broadoff `+7.01us` median, `+7.21us` mean, broadoff wins `114/160` |

Parity stayed clean in both repeat orders:

- default-first loss diff: `-1.90734863e-06`, worst selected grad `qn.0`
  rel `7.054799e-07`
- broadoff-first loss diff: `+0.00000000e+00`, worst selected grad `w1.0`
  rel `4.677046e-07`

Decision: include `(1024,1536,512)` in the exact default-off set too, while
keeping `MK_WGMMA_N128_NN=1` as the old-route force-on override.

## Final Promoted Default

After adding `(1024,1536,512)`, forced-old `MK_WGMMA_N128_NN=1` restored all
24 old NN n128 rows:

- final default: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=26`,
  `splitK=33`
- forced old: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=50`,
  `splitK=33`

Final promoted default vs forced old, 80 samples per arm:

| Order | Final default | Forced old NN n128 | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3358.29us` | `3399.02us` | old-minus-default `+39.22us` median, `+40.16us` mean, default wins `80/80` |
| old first | `3342.03us` | `3376.48us` | old-minus-default `+33.17us` median, `+35.35us` mean, default wins `80/80` |

Parity stayed clean:

- default-first loss diff: `+2.86102295e-06`, worst selected grad `kn.0`
  rel `5.398694e-07`
- old-first loss diff: `-1.90734863e-06`, worst selected grad `kn.0`
  rel `7.198259e-07`

Final profile/score:

- `profile_df.py small df`: `3379.2us`
- `final_bench.py small`: megakernel `3373.2us`, compile+cudagraph+ `1904.9us`,
  remaining gap `1.77x`

Validation after the final source edit:

- `py_compile mk.py model.py profile_df.py`
- `git diff --check`
- `test_model.py`
- `test_ops.py`
