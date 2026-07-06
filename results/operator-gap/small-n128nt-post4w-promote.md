# Small NT N128 Post-4W Promotion

Date: 2026-07-06

Base: `b6d0b7f` (`megakernel: keep small mlp dx off n128`)

Verdict: **promote** exact H512/S1024 small NT n128 gates for
`GEMMNT 1024x512x512`, `GEMMNT 1024x3072x512`, and
`GEMMNT 1024x512x1536`. These rows run faster on the normal m64n64 WGMMA path
after the small 4W/cache-off and NN n128 retunes.

## Candidate

Source-free probe:

- default: current `b6d0b7f`
- candidate: `MK_WGMMA_N128=2`
- interpretation: keep lm-head n128 behavior, but disable the general NT n128
  route

Promoted source behavior:

- with `MK_WGMMA_N128` unset, only the exact NT shapes above fall back to
  m64n64
- `MK_WGMMA_N128=1` still forces old general n128 behavior for A/B
- `MK_WGMMA_N128=2` still means lm-head-only mode

Shape:

- `Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)`

## Evidence

Logs:

- `results/mkv3-p4b-small-n128-mode2-after-final-20260706T1740Z.log`
- `results/mkv3-p4b-small-n128nt-promoted-20260706T1750Z.log`
- `results/mkv3-p4b-profile-small-n128nt-default-20260706T1755Z.log`
- `results/mkv3-p4b-score-small-n128nt-default-20260706T1757Z.log`

Source-free `MK_WGMMA_N128=2` route:

- default: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=26`
- mode2: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=2`
- changed rows: 8x each
  - `GEMMNT 1024x512x512`: flags `12434`, 32 tiles -> flags `8338`, 64 tiles
  - `GEMMNT 1024x3072x512`: flags `4226`, 192 tiles -> flags `130`, 384 tiles
  - `GEMMNT 1024x512x1536`: flags `12434`, 32 tiles -> flags `8338`, 64 tiles

Source-free timing, 80 samples per arm:

| Order | Default | `MK_WGMMA_N128=2` | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3358.14us` | `3301.62us` | default-minus-mode2 `+56.46us` median, `+55.19us` mean, mode2 wins `80/80` |
| mode2 first | `3347.07us` | `3292.08us` | default-minus-mode2 `+55.62us` median, `+54.24us` mean, mode2 wins `80/80` |

Source-free parity:

- default-first loss diff: `-2.86102295e-06`, worst selected grad `w2.0`
  rel `4.208155e-07`
- mode2-first loss diff: `+3.81469727e-06`, worst selected grad `kn.0`
  rel `4.049021e-07`

Promoted default vs forced old `MK_WGMMA_N128=1`:

- default: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=2`
- old NT n128: `n_instr=288`, `critical_path=144`, `gated=127`, `n128=26`

Promoted timing, 80 samples per arm:

| Order | Default | Forced old NT n128 | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3296.42us` | `3348.45us` | old-minus-default `+53.36us` median, `+51.82us` mean, default wins `79/80` |
| old first | `3292.96us` | `3339.46us` | old-minus-default `+44.85us` median, `+46.83us` mean, default wins `80/80` |

Promoted parity:

- default-first loss diff: `+9.53674316e-07`, worst selected grad `qn.0`
  rel `6.172947e-07`
- old-first loss diff: `-1.90734863e-06`, worst selected grad `qn.0`
  rel `4.850175e-07`

Validation after the source edit:

- `py_compile mk.py model.py profile_df.py`
- `git diff --check`
- `test_model.py`
- `test_ops.py`

Refreshed final profile/score:

- `profile_df.py small df`: `3354.8us`
- `final_bench.py small`: megakernel `3356.6us`, compile+cudagraph+ `1888.5us`,
  remaining gap `1.78x`

## Decision

Keep the exact NT shape gates. The broad source-free knob was not merely a
mode artifact: the promoted source route reproduces the win against
`MK_WGMMA_N128=1` and keeps the explicit env override available for regression
A/B.
