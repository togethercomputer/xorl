# Train/serve numerical contracts

This directory documents stable production mechanisms and conventional tests.
It intentionally excludes campaign receipts, frozen benchmark outputs,
cluster-specific launch records, and promotion manifests.

## Shared arithmetic contracts

- [Attention](ATTENTION_CONTRACT.md): backend identity and KV-reduction shape.
- [RMSNorm](RMSNORM_CONTRACT.md): canonical BF16 rows and explicit reduction trees.
- [GEMM](GEMM_CONTRACT.md): fixed K-axis accumulation with tunable output geometry.
- [LM head](LM_HEAD_CONTRACT.md): projection and vocabulary-normalization trees.
- [LoRA](LORA_CONTRACT.md): Qwen folded and GLM active-LoRA forward contracts.
- [MoE experts](moe-serving-expert-kernel.md): serving-value forwards with trainer-owned backward.
- [Contract selection](DEFAULTS_AND_PARETO.md): architecture-selected support envelopes.
- [Qwen3.5 family](QWEN35_GDN_ZERO_K3_RUNBOOK.md): supported dense and MoE geometries.
- [DSV4-Flash](DSV4_FLASH_LORA_ZERO_K3_PLAN.md): scoped active-LoRA zero-K3 program and qualification gates.

## Rules

1. Match arithmetic and rounding boundaries, not merely mathematical formulas.
2. Unsupported shapes, topologies, and backends fail instead of silently
   selecting another numerical program.
3. Forward bytes define the train/serve contract. A trainer may use a separate
   checked backward implementation.
4. Qualification belongs to an exact trainer/sampler revision pair and uses
   real decision-time FP32 log-probability bytes.
