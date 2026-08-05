# K3 train/serve parity

This directory is the operator entry point for minimizing trainer-to-sampler
logprob divergence without giving up measured throughput. For the official
GLM-5.2 and Qwen3.5-family server-training models, the exact program is
architecture-selected; the historical per-component environment recipes are
not current launch instructions.

Start with [DEFAULTS_AND_PARETO.md](DEFAULTS_AND_PARETO.md). It contains:

- the actual xorl defaults and the overrides required by each validated lane;
- exact dense, MoE, hybrid, TP>1, LoRA, and routing-replay recipes;
- the evidence class and measured speed status for each recipe;
- production, diagnostic-only, and rejected flag classifications.

Model-family bring-up guide:

- [QWEN35_GDN_ZERO_K3_RUNBOOK.md](QWEN35_GDN_ZERO_K3_RUNBOOK.md) — current Qwen3.5-family
  automatic exact program, admitted envelope, and direct qualification gate.

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
- Qwen3.6-35B-A3B has an architecture-selected exact GDN+MoE program at its
  admitted EP8 topology **[GATE]**. A released trainer/serving revision pair is
  not qualified by ancestry: it still requires its own repeatable capture and
  raw-logprob replay.
- Dense Qwen3.5-0.8B GDN at TP1 has an exact live rollout-to-optimizer mechanics gate
  **[LOCAL]** under the correctness-oriented rescan contract. This closes the blanket claim that
  GDN live decode has an unavoidable numerical floor; it does **not** certify Qwen3.5 MoE,
  distributed serving, production length, or production throughput. See
  [QWEN35_GDN_ZERO_K3_RUNBOOK.md](QWEN35_GDN_ZERO_K3_RUNBOOK.md).
- GLM-5.2 has a full 78-block exact native-FP8 sparse-MLA/MoE program at
  WORLD16/EP16/CP16 **[GATE]**. Its public revision pair likewise requires a
  direct final-head replay before release.
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
