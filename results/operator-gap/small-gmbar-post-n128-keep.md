# Small D64 GEMM Mbar-Ring Post-N128 Keep

Date: 2026-07-06

Base: `520d2fa` (`megakernel: record post-n128 coldcap check`)

Verdict: **keep** the current H512/S1024 small `MK_GEMM_MBAR_RING` default.
After the NN/NT n128 route changes, forcing the old two-stage GEMM feed path is
even more negative than the earlier post-4W check.

## Candidate

Source-free current-head A/B:

- default: unset `MK_GEMM_MBAR_RING`, model default `True` for `D=64,S>=1024`
- old: `MK_GEMM_MBAR_RING=0`
- shape: `Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024)`

## Evidence

Log:

- `results/mkv3-p4b-small-gmbar-after-n128-20260706T1820Z.log`

Program shape was unchanged in both arms:

- `n_instr=288`
- `critical_path=144`
- `gated=127`

Parity:

- default-first loss diff: `+3.81469727e-06`, worst selected grad `kn.0`
  rel `5.398697e-07`
- old-first loss diff: `+0.00000000e+00`, worst selected grad `w2.0`
  rel `6.733047e-07`

Timing, 80 CUDA-event samples per arm:

| Order | Default | Old two-stage | Paired delta |
| --- | ---: | ---: | --- |
| default first | `3289.98us` | `3443.94us` | old-minus-default `+154.51us` median, `+153.93us` mean, default wins `80/80` |
| old first | `3294.26us` | `3445.34us` | old-minus-default `+150.94us` median, `+150.53us` mean, default wins `80/80` |

## Decision

Keep `gemm_mbar_ring_default = c.D == 64 and c.S >= 1024 and c.S % 128 == 0`.
The D64 mbar-ring path is strongly load-bearing after the n128 row gates.
