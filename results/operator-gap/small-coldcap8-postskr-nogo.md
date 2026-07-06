# Small Cold-Cap8 Post-SKR Re-Sweep No-Go

Date: 2026-07-06

Base: current main after `61e37bd` (`megakernel: per-warp qknorm rope bwd dw partials`);
the candidate originated immediately after `5c6e234` (`s1024/s2048 lm_head n256`).

Verdict: **do not retune** the exact H512/S1024 small `MK_COLD_CAP` default to
8 after the SKR and n256 head-route promotions. Keep the existing cap48 default.

## Candidate

The post-SKR small resweep found a weak cap8 signal after the head-dX SKR and
lm-head n256 route changes. Because scheduler knobs often flip after structural
promotions, cap8 received high-rep paired order reversal.

Shape:

- `Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)`

Route shape stayed unchanged:

- `n_instr=289`
- default and cap8 both use the same instruction graph; only the cold-work
  scheduler cap changes.

## Evidence

Logs:

- `results/mkv3-p4b-small-coldcap8-repeat-20260706T2322Z.log`
- `results/mkv3-p4b-small-coldcap8-postskr-default-first-20260706T2155Z.log`
- `results/mkv3-p4b-small-coldcap8-postskr-variant-first-20260706T2155Z.log`

First high-rep repeat, 120 reps/order:

| Order | Default | Cap8 | Delta |
| --- | ---: | ---: | --- |
| default first | `3239.25us` | `3238.98us` | `-0.27us`, cap8 wins `75/120` |
| cap8 first | `3230.37us` | `3239.07us` | `+8.70us`, cap8 wins `22/120` |

Independent confirmation, 160 reps/order:

| Order | Default | Cap8 | Delta |
| --- | ---: | ---: | --- |
| default first | `3238.21us` | `3235.70us` | `-2.51us`, cap8 wins `88/160` |
| cap8 first | `3208.94us` | `3216.66us` | `+7.71us`, cap8 wins `40/160` |

Parity stayed clean in every run; worst reported selected-gradient relative
error was at or below `1e-6`.

## Decision

Reject. Cap8 does not survive construction-order reversal. The default-first
signal is below the local timing noise band, while the reverse order is a clear
regression in both high-rep runs. Keep `_cold_cap()` unchanged for H512/S1024
small.
