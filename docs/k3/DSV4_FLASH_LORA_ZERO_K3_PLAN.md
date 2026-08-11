# DSV4-Flash active-LoRA zero-K3 program

## Goal and support boundary

Add a fail-closed DSV4-Flash training and serving program whose retained
decision-time FP32 log-probability bytes match exactly. The first admitted lane
is deliberately narrow:

- the official DSV4-Flash geometry;
- one active adapter, rank 1, alpha 1;
- native serving-value FP8 dense weights and MXFP4 routed experts;
- one-request eager decoding without radix reuse, speculative decoding, or
  CUDA graphs;
- temperature 1 and top-p 1; and
- MTP disabled.

This is a new architecture contract. Passing the GLM-5.2 or Qwen contracts does
not implicitly qualify DSV4-Flash. Existing DSV4 CPU tests cover attention
LoRA injection and gradient flow but do not establish the routed-expert path.

## What can be reused

The qualification method and much of the adapter lifecycle already exist:

| Reuse directly | Generalize with DSV4 mappings | New DSV4 work |
| --- | --- | --- |
| Frozen sampler trace, raw-logprob byte comparison, K3 scorer, four-decision and 64-decision gates | GLM active rank-1 factor masters, BF16 views, explicit physical target manifest, update and synchronization protocol | Four-stream mHC pre/post/head arithmetic |
| First-divergence localization and fail-closed contract resolution | GLM native-FP8 serving-value linear forwards and trainer-only VJPs | C0/C4/C128 hybrid attention, compressor, and cache layout |
| Gradient-ownership and optimizer-step checks | Distributed LM-head projection and vocabulary normalization | DSV4 indexer operands, rotations, quantization, top-k ordering, and page mapping |
| Negative controls and distinguishable nonzero factors | Ordered expert contribution transport and combine | Native MXFP4 routed-expert forward with active LoRA, clamped SwiGLU, and a trainer-owned backward |
| Existing exact RoPE, GEMM, norm, and head primitives, after DSV4 shape-specific byte gates | SGLang adapter loading, memory-pool ownership, and post-step refresh | Hash routing in the first three blocks, later correction-bias routing, and DSV4-specific expert folding |

Qwen's folded-weight LoRA path is not the default donor: folding into native
FP8/MXFP4 bases would add a new requantization program. The initial candidate is
therefore GLM-style active LoRA over immutable quantized base payloads.

## Current implementation gaps

The repository already models DSV4's 43-layer hybrid-attention, mHC, routing,
and MoE structure and can inject ordinary attention LoRA. Its generic routed
expert wrapper correctly rejects DSV4's clamped SwiGLU semantics, so that path
already requires a model-specific implementation. The repository does not yet
provide the paired exact program:

1. The training checkpoint path dequantizes block-FP8 and MXFP4 payloads to
   BF16. Exact replay must retain the serving payloads, scales, layouts, and
   rounding boundaries.
2. Trainer attention/indexer kernels are mathematically aligned but are not
   literal serving-value forwards. C0, C4, and C128 each need an independently
   qualified forward path.
3. The pinned SGLang DSV4 model has no active adapter inventory or LoRA wiring;
   its low-rank projection names describe the base architecture, not PEFT.
4. Model admission does not select a DSV4 exact contract, validate its complete
   geometry, or reject incompatible topology/backend/cache choices.
5. Existing DSV4 attention-LoRA tests do not cover quantized serving forwards,
   physical fused mappings, cross-engine bytes, factor synchronization, or
   optimizer progress; the routed-expert and end-to-end LoRA baseline tests are
   currently stopped by the clamped-SwiGLU guard.

## Proposed adapter contract

Freeze routing and selection in the first lane: hash tables, router and
correction-bias parameters, the DSV4 indexer, compressors, mHC parameters, and
attention sinks are not adapter targets. They must still execute the exact
serving forward.

The provisional logical target universe is:

- every layer's `wq_a`, `wq_b`, `wkv`, `wo_a`, and `wo_b` projections;
- every layer's shared-expert gate, up, and down projections;
- every layer's routed-expert gate, up, and down projection banks; and
- the LM head.

For 43 layers this is 345 non-routed logical projections, 43 routed banks of
three projections, and 948 factor tensors at rank 1. This count is an admission
assertion, not a best-effort target list: checkpoint discovery must reproduce
it exactly or stop. A DSV4 physical manifest must map those factors into fused
SGLang storage such as QKV-A and grouped expert payloads without changing value
order.

## Implementation workstreams

### 1. Architecture admission

- Add a complete official-geometry validator and DSV4 exact-contract resolver.
- Select the contract before model wrapping or sharding.
- Add trainer and sampler fail-closed checks for topology, quantization,
  attention backend, page size, graph/radix/speculation state, sampling, and
  adapter shape.
- Make unsupported configurations raise; never fall back to ordinary DSV4
  kernels under an exact-mode request.

Likely touch points are `src/xorl/models/auto.py`,
`src/xorl/trainers/model_builder.py`, DSV4 configuration code, server
arguments, and their unit tests.

### 2. Native quantized base ownership

- Preserve block-FP8 payloads/scales and MXFP4 expert payloads in the trainer
  load path instead of materializing BF16 base weights.
- Define immutable base-buffer ownership, DCP serialization, and restore
  validation.
- Prove base-only trainer self-consistency before comparing with serving.

The main starting point is
`src/xorl/models/transformers/deepseek_v4/checkpoint_handler.py`.

### 3. Paired active-LoRA forwards

- Add a DSV4 factor inventory and physical-layout mapper analogous to the GLM
  adapter module in the pinned SGLang tree.
- Wire the sampler model, LoRA backend, memory pool, and update endpoint for the
  complete target universe.
- In XoRL, call the same serving-value FP8/MXFP4/fused projection programs with
  the same BF16 factor views and addition/cast order.
- Supply checked trainer-only VJPs for activations and factor masters.
- Treat grouped `wo_a`, QKV-A fusion, shared experts, routed expert banks, and
  the sharded LM head as explicit physical families.

### 4. Exact DSV4 trunk

Qualify and repair the earliest differing component in this order:

1. embeddings and four-stream mHC pre-mix;
2. RoPE and query/KV projection families;
3. C0/C4/C128 attention, including compressor overlap, attention sinks, legal
   key ranges, and logical-to-paged cache order;
4. indexer rotation/Hadamard operands, FP8 or FP4 encoding and scales, causal
   top-k eligibility, tie order, and page mapping;
5. mHC post-mix;
6. first-three-layer hash routing, later routing/correction bias, shared and
   routed experts, and ordered EP combine;
7. final mHC head mix, norm, LM head, and selected-token probability.

Matching selected indices alone is insufficient; the decision-time probability
bytes remain the acceptance criterion.

### 5. Backward, optimizer, and synchronization

- Prove every admitted factor receives its intended gradient and frozen base,
  selector, router, and mHC tensors do not.
- Add distributed ownership and completion-rendezvous tests for routed banks.
- Run a finite-loss backward and one real optimizer step.
- Publish every updated factor view atomically, verify sampler refresh, then
  capture a fresh trace. An exact pre-step replay does not qualify post-step
  behavior.
- Cover checkpoint save/restore of factor masters and optimizer state without
  rewriting the immutable quantized base.

## Qualification sequence

1. **Source and topology freeze.** Record the exact trainer/sampler revisions,
   geometry, native dtypes, physical adapter inventory, and resolved runtime
   topology. Establish repeated sampler-capture denominator bytes.
2. **Base ruler.** With no adapter installed, replay terminal prefill plus the
   first three KV-cached decode decisions. Localize the first differing tensor
   and repair component-by-component.
3. **Adapter A join.** Install all-zero factors, then deterministic
   distinguishable nonzero factors across all 948 tensors. Require the same
   four decisions to match and require a negative control to detect a perturbed
   factor or ordering error.
4. **Training gate.** Establish complete gradient ownership, finite backward,
   and a real optimizer update.
5. **Adapter B join.** Synchronize the updated factors, take a fresh sampler
   capture, and match the same four decisions.
6. **Promotion replay.** Match a fresh 64-decision prefix byte-for-byte with
   exactly zero K3.
7. **Runtime expansion.** Admit batching, radix/cache reuse, CUDA graphs,
   longer contexts, and additional topologies only through separate exactness
   and performance gates.

Topology is resolved, not assumed. The first useful candidate should keep
attention tensor-parallel width at one and preserve native expert ownership,
but trainer DP/FSDP/EP and sampler DP-attention/EP must be accepted only after a
runtime byte proxy and the full replay agree.

## Definition of done

Completion requires all of the following independently:

- exact base and active-LoRA serving-value forwards for every admitted physical
  family;
- a complete, fail-closed 948-factor inventory and update path;
- four-decision A1/A2 and post-update B1 byte equality;
- complete adapter gradient ownership, finite optimizer progress, and atomic
  factor refresh;
- a fresh 64-decision B2 replay with byte equality at every retained decision
  and K3 exactly zero; and
- measured endpoint throughput after correctness, with unsupported runtime
  combinations still rejected.

Model construction, graph capture, index agreement, finite loss, a lifecycle
smoke, or a zero-valued adapter by itself is not completion.

## First implementation slice

The shortest evidence-producing slice is:

1. implement the DSV4 geometry/topology resolver and exact 948-factor inventory
   tests in both trees;
2. retain native base quantization in the trainer load path;
3. add sampler loading for a zero-valued rank-1 adapter;
4. freeze one repeated four-decision sampler trace; and
5. obtain the first-divergence report for base-only full-depth replay.

That result decides which new DSV4 surface needs repair first without spending
time on optimizer or scale-out work before the forward contract is real.
