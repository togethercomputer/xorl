# Small Cold-Cap Post-N128 Re-Sweep No-Go

Date: 2026-07-06

Base: `0de64db` (`megakernel: keep small nt rows off n128`)

Verdict: **do not retune** the H512/S1024 small `MK_COLD_CAP` default after the
NN/NT n128 route changes. Keep cap48.

## Candidate

Source-free current-head scheduler cap re-sweep after small moved its remaining
general NN/NT n128 rows to m64n64.

Shape:

- `Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)`

Caps tested:

- `0`, `16`, `33`, `48`, `64`, `96`

## Evidence

Logs:

- `results/mkv3-p4b-small-coldcap-after-n128-20260706T1805Z.log`
- `results/mkv3-p4b-small-coldcap-after-n128-focused-20260706T1810Z.log`

Route summary was unchanged across caps:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `hot=197`
- `cold=91`

Parity against cap48 was clean for all tested caps; worst selected-gradient rel
error stayed below `5.4e-07`.

Broad single-order medians over 60 samples:

| Cap | Median |
| --- | ---: |
| `0` | `3322.08us` |
| `16` | `3283.41us` |
| `33` | `3283.58us` |
| `48` | `3283.47us` |
| `64` | `3280.27us` |
| `96` | `3283.60us` |

Cap64 was only about `3us` better than cap48 in the broad pass, so it received
the focused order-reversal gate.

Focused cap48 vs cap64, 160 samples per arm:

| Order | Cap48 | Cap64 | Paired delta |
| --- | ---: | ---: | --- |
| cap48 first | `3286.86us` | `3287.81us` | cap48-minus-cap64 `-0.13us` median, `+0.72us` mean, cap64 wins `79/160` |
| cap64 first | `3287.68us` | `3286.40us` | cap48-minus-cap64 `+1.36us` median, `+1.78us` mean, cap64 wins `86/160` |

## Decision

Reject. The remaining cap64 signal is within timing noise after the n128 route
changes and does not justify changing `_cold_cap()`. Keep the H512/S1024 small
default cap48.

## Post-direct-BF16 addendum

After `fc32323` promoted direct-BF16 GEMM epilogues for exact H512/L8/S1024
small, cap64 was rechecked against the current cap48 default because the earlier
cap64 result was close.

Log: `results/mkv3-p4b-small-coldcap-post-directbf16-20260706T1940Z.log`

Route shape stayed unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `hot=197`
- `cold=91`

Parity stayed clean:

- default-first: `loss_diff=+1.90734863e-06`, worst selected grad `kn.0`,
  relative error `3.599131e-07`.
- cap64-first: `loss_diff=+1.90734863e-06`, worst selected grad `qn.0`,
  relative error `6.613873e-07`.

Cap64 still failed the focused gate:

| Order | Cap48/default | Cap64 | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3267.70us` | `3269.31us` | default-minus-cap64 `-3.04us` median, `-2.51us` mean, cap64 wins `65/160` |
| cap64 first | `3261.01us` | `3262.67us` | default-minus-cap64 `-0.85us` median, `-1.91us` mean, cap64 wins `75/160` |

Keep cap48 after the direct-BF16 promotion.
