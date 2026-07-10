# K3 train/serve parity

This directory is the current operator entry point for minimizing trainer-to-sampler logprob
divergence without giving up measured throughput. It describes the current contract as of
2026-07-10; older notes under `docs/notes/` are evidence records, not launch recipes.

Start with [DEFAULTS_AND_PARETO.md](DEFAULTS_AND_PARETO.md). It contains:

- the actual xorl defaults and the overrides required by each validated lane;
- exact dense, MoE, hybrid, TP>1, LoRA, and routing-replay recipes;
- the evidence class and measured speed status for each recipe;
- production, diagnostic-only, and rejected flag classifications.

## Evidence classes

- **[GATE]**: a frozen-input component, layer, or end-to-end equivalence gate.
- **[LOCAL]**: a real-stack local run using real weights and generated traces.
- **[PROD]**: an observed production training run.

Never promote a claim to a stronger class without new evidence. Always report the K3 aggregation
(token or length-normalized sequence); those values differ by roughly sequence length.

## Current frontier

- Dense, softmax-attention, TP1-serving RL has an exact-zero recipe. The current families-v2
  step-0 production gate was exactly `0.0` over 4,096 rollouts / 65.5M valid tokens. Older
  pre-v2 full-run anchors were 83/85 zero steps in v12 and 140/140 in v15; do not conflate them
  with the current trees.
- Softmax-attention MoE scoring has a bitwise contract **[GATE]**. Live MoE must also preserve
  routing capture/replay and validate the serving topology.
- Hybrid GDN+MoE is bitwise for teacher-forced prefill/scoring **[GATE]**; live recurrent decode
  still has a declared nonzero floor. Do not advertise it as a zero-K3 live lane.
- Conventional tensor-sharded serving with effective `attn_tp_size` or `head_tp_size` greater
  than 1 has no certified BI lm-head/trunk contract. Top-level TP with EP8+DP-attention can still
  have effective attention/head TP1. The validated TP2 Wordle fallback is minimum-K3, not zero.
- Single-adapter LoRA is contracted by folding the adapter into the weights before the forward;
  multi-adapter serving remains uncontracted.

## Non-negotiable rules

1. Both engines must execute the same numerical contract. A flag that cannot cover a requested
   shape or topology must raise; silent fallback invalidates the result.
2. Forward bits are the K3 contract. Backward may use stock numerics only after a gradient gate.
3. Keep routing capture/replay enabled by default on supported MoE lanes. The EP8+DP-attention
   all-zero capture bug and EP8+packing split bug are topology-specific blockers, not reasons to
   disable routing globally.
4. Never enable the global batch-invariant aten interpose in a training graph. Use the scoped
   trunk-linear contract described in the matrix.
5. Run a full K3 gate at production sequence length before fleet launch. Step 0 must equal the
   lane's declared null hypothesis: exact zero or its written floor.

For measurement, use paired sampler/trainer traces at production sequence length and retain the
exact environment and aggregation method with the result. This directory intentionally carries
only the current operational contract.
