# Train/serve numerical contracts

This directory documents the stable arithmetic contracts used to make trainer
and sampler forward values agree. It contains production mechanisms and their
conventional verification commands, not campaign receipts, benchmark outputs,
or environment-specific launch records.

## Contracts

- [Attention](ATTENTION_CONTRACT.md): backend identity and KV-reduction shape.
- [RMSNorm](RMSNORM_CONTRACT.md): canonical BF16 rows and explicit reduction trees.
- [GEMM](GEMM_CONTRACT.md): fixed K-axis accumulation with tunable output geometry.
- [LM head](LM_HEAD_CONTRACT.md): projection and vocabulary-normalization trees.
- [LoRA](LORA_CONTRACT.md): canonical folds and exact-forward/trainable-backward boundaries.
- [Contract selection](DEFAULTS_AND_PARETO.md): supported generic lanes and fail-closed rules.

## Rules

1. Match arithmetic and rounding boundaries, not merely mathematical formulas.
2. A shape, topology, or backend outside a documented envelope must fail rather
   than silently use another numerical program.
3. Forward bytes define the train/serve contract. A trainer may use a separate
   checked backward implementation.
4. Qualify a trainer/sampler revision pair with real decision-time FP32
   log-probability bytes. Results from another source pair are evidence about
   the design, not qualification of the new pair.
