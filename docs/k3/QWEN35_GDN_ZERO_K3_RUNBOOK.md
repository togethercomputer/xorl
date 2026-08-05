# Qwen3.5-family exact server-training program

The official Qwen3.5-0.8B and Qwen3.6-35B-A3B geometries select their complete
exact numerical programs automatically in server-training mode. On the paired
SGLang server, `--rl-on-policy-target xorl` is the only activation switch.
There is no per-component environment recipe and no faster non-exact fallback
inside the admitted program.

## System design

The trainer and sampler execute the same numerical choices at every
bit-relevant boundary:

- Q/K L2 normalization uses the fixed serving launch geometry rather than an
  autotuned reduction.
- GDN convolution, gating, recurrent composition, and gated RMSNorm use the
  paired exact kernels. Decode reconstructs the current partial chunk from the
  last 64-token fp32 boundary state, matching teacher-forced prefill.
- RoPE inverse frequencies, positions, and cosine/sine tables are constructed
  on CPU in fp32 before bit-exact device transfer. Qwen uses the Class-A rotary
  application program.
- Trunk projections, norm families, router selection, and the float32 LM-head
  scoring path use their batch-invariant implementations.
- Qwen3.6 MoE uses the native EP8 ordered combine. The dense model does not
  inherit MoE-only graph or transport optimizations.

The Qwen3.6 serving program uses FA4, DP attention, one local CUDA-graph bucket
of 10, global admission 80 at DP8, radix reuse, and 64-aligned continuation
chunks. The corresponding trainer program is WORLD16 with DP16 (two replicas,
shard size 8) and EP8. Qwen3.5-0.8B is admitted only at the single-rank dense
geometry.

## Fail-closed envelope

The model loader rejects unqualified model geometry, topology, attention or
MoE backends, precision overrides, and incompatible cache/graph settings.
Serving rejects sampling transforms the trainer cannot replay, speculative
decoding, session-state restoration, quantized model weights, and LoRA-wrapped
LM heads for this program. It does not silently fall back to another numerical
path.

## Qualification

A trainer/serving revision pair is qualified only by behavior produced by that
pair. Capture the same prompt and seed in at least three sampler lifecycles and
require identical retained token IDs and raw float32 logprob bytes. Replay the
immutable capture through the full trainer model and require:

- every retained token ID unchanged;
- every raw float32 behavior-logprob byte unchanged;
- token K3 and aggregate K3 exactly `0.0`;
- an all-zero per-request error inventory.

Run the exact and non-exact throughput arms in the same warm session when
reporting overhead. Historical development-lineage results explain the design,
but do not qualify a rewritten or released pair by ancestry.

For generic dense and historical component recipes, see
[DEFAULTS_AND_PARETO.md](DEFAULTS_AND_PARETO.md). Those recipes are not launch
instructions for the architecture-selected Qwen3.5-family program.
