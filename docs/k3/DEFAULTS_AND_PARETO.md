# Numerical-contract selection

Generic XoRL defaults support many models and topologies; they do not promise
exact trainer/sampler parity everywhere. For the model families below, server
training selects one architecture-aware numerical program before module
construction and rejects incompatible overrides.

## Architecture-selected programs

| Model lane | Admitted trainer topology | Contract ownership |
|---|---|---|
| Qwen3.5 dense | single rank | model resolver selects GDN, RoPE, RMSNorm, trunk, and head arithmetic |
| Qwen3.6-35B-A3B | WORLD8/DP8/EP8 or WORLD16/DP16/EP8 | model resolver additionally selects router, expert forward, and the canonical logical adjacent-pair EP fold |
| GLM-5.2 native FP8 | WORLD16/PP1/TP1/DP1/EP16/CP16 | model resolver selects sparse MLA, native-FP8 projections, routed/shared experts, head, and distributed combine |

On the paired serving engine, `--rl-on-policy-target xorl` selects the matching
architecture program. Component environment variables are diagnostic surfaces,
not launch instructions for these lanes.

The selected program owns all bit-relevant choices together:

- RoPE table construction and Class-B rotary application;
- RMSNorm-v2 row formation, reduction, reciprocal, and affine rounding;
- dense GEMM K-axis accumulation;
- attention/GDN backend and state handling;
- router arithmetic and selected-expert ordering;
- expert projection, activation, routing-weight, and local-combine arithmetic;
- distributed expert transport and the versioned logical adjacent-pair FP32
  fold with one final low-precision cast; and
- LM-head projection and vocabulary normalization.

## LoRA programs

Qwen's exact single-adapter program uses a canonical folded weight for its
trainer forward and synchronized sampler weight. GLM-5.2 instead keeps the
native-FP8 base frozen and executes active rank-1 LoRA through the same SGLang
forward kernels in trainer and sampler, with trainer-owned autograd.

These are separate admitted programs. Dynamic multi-adapter serving and
arbitrary ranks, scales, or target sets do not inherit either contract.

## Generic mechanisms

The lower-level attention, RMSNorm, GEMM, head, expert, and fold mechanisms in
this directory remain useful for other models, but a new combination must earn
its own topology and end-to-end qualification. Do not assemble a model launch
by copying individual environment flags from historical campaigns.

## Qualification

For each trainer/sampler revision pair:

1. assert the resolved model program and every fail-closed topology check;
2. generate tokens and retain the sampler's decision-time FP32 log-probability
   bytes;
3. replay the same token IDs through the full trainer; and
4. require byte equality for every retained token and K3 exactly zero.

Performance results and cluster-specific launch records belong with the run
that produced them, not in this source contract.
