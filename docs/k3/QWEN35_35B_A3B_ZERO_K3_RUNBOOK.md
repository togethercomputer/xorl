# Qwen3.5 35B-A3B MoE/GDN contract

This is the portable trainer-side contract for Qwen3.6-35B-A3B (whose model
family is Qwen3.5 in the implementation). It does not claim that this
repository alone certifies a sampler build.

There is no component flag recipe. In server-training mode, official model
identity and geometry select the complete exact program:

- GDN convolution and gating contracts enabled;
- batch-invariant trunk linear and residual-norm paths enabled;
- deterministic MoE router/top-k behavior enabled;
- family-v1 GDN numerics and fixed L2Norm launch selection;
- native ordered EP combine by default only for the certified EP8 topology.

Incompatible numerical overrides and any topology other than the admitted
WORLD16/DP16 (replicate 2, shard 8)/EP8 program raise before execution. Native
EP combine enters experts through `nn.Module.__call__`; this lets FSDP
materialize the BF16 compute parameters before the serving-layout expert kernel
runs.

On serving, `--rl-on-policy-target xorl` is the only activation switch. The
official geometry selects the qualified FA4 graph/radix program, exact GDN
state handling, router/head kernels, and ordered expert combine as one unit.
There is no public recurrent-decode opt-out.

Promotion requires all of the following on the paired trainer and sampler
revisions:

1. MoE expert-forward equality and routing-vitality/replay checks.
2. Native EP-combine forward and backward tests on the target EP topology.
3. Static trainer/sampler replay with exact token logprobs.
4. A real rollout-to-update gate with `ratio_mean == 1.0`, policy and behavior
   K3 equal to `0.0`, finite nonzero gradients, and a completed optimizer step.
5. A production-length step-zero rerun after real weight synchronization.
