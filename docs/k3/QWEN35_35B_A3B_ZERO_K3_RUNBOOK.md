# Qwen3.5 35B-A3B MoE/GDN contract

This is the portable trainer-side contract for Qwen3.5 35B-A3B. It does not
claim that this repository alone certifies a sampler build.

Set `XORL_BI_GDN=1`. The model then selects the paired dense-GDN defaults:

- GDN convolution and gating contracts enabled;
- batch-invariant trunk linear and residual-norm paths enabled;
- deterministic MoE router/top-k behavior enabled;
- family-v1 GDN numerics and fixed L2Norm launch selection;
- native ordered EP combine by default only for the certified EP8 topology.

Explicit environment values always override these defaults. Native EP combine
must enter experts through `nn.Module.__call__`; this lets FSDP materialize the
BF16 compute parameters before the serving-layout expert kernel runs.

The recurrent GDN decode opt-out is a speed comparison, not a zero-K3 mode.
The retained local attribution result reduced exact-decode overhead from 41.40%
to 25.62% versus recurrent decode while preserving literal zero token K3. This
is local evidence, not a fleet-throughput or production rollout claim.

Promotion requires all of the following on the paired trainer and sampler
revisions:

1. MoE expert-forward equality and routing-vitality/replay checks.
2. Native EP-combine forward and backward tests on the target EP topology.
3. Static trainer/sampler replay with exact token logprobs.
4. A real rollout-to-update gate with `ratio_mean == 1.0`, policy and behavior
   K3 equal to `0.0`, finite nonzero gradients, and a completed optimizer step.
5. A production-length step-zero rerun after real weight synchronization.
