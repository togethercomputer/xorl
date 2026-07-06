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
