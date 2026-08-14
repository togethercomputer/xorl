# Qwen3.5-family exact server-training program

Qwen3.5-0.8B and Qwen3.6-35B-A3B use one architecture-selected numerical
program in server training. Users do not assemble it from component flags.

## Admitted geometries

- Qwen3.5-0.8B dense: single rank.
- Qwen3.6-35B-A3B MoE: WORLD8/DP8/EP8 or WORLD16/DP16/EP8.

The resolver selects Class-B RoPE, RMSNorm-v2, exact GDN state handling,
batch-invariant trunk and head programs, and the qualified attention program.
The MoE model additionally selects deterministic routing, the serving-value
expert forward, and `canonical_moe_fold_fp32_v2`: logical contributor order
followed by an adjacent-pair tree whose leaves and nodes are FP32, with one
final cast at the consumer boundary. Raw EP8 transport does not perform the
addition.

On serving, `--rl-on-policy-target xorl` activates the paired program. The
loader rejects incompatible topology, precision, attention, routing, cache,
graph, and sampling choices rather than falling back to a different path.

## LoRA

The supported single-adapter Qwen path uses canonical merged weights in the
trainer forward and publishes the same bytes to serving. Dynamic multi-adapter
execution is a different program and is not covered by this contract.

## Qualification

Qualification requires a repeatable sampler capture and full-depth trainer
replay of the retained token IDs and decision-time FP32 log-probability bytes.
Every retained byte must match and K3 must be exactly zero. A model or runtime
revision is not qualified solely because it descends from an earlier passing
revision.

Changing the contributor fold is a new numerical program. Checkpoints remain
compatible, but rollout denominators from the former descending-chain program
must not be replayed against this program. Sampler and trainer revisions must
switch together between rollout generations, after draining old trajectories
and flushing captured graphs and caches.
