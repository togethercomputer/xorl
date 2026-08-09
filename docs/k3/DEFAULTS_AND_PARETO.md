# Numerical-contract selection

The generic XoRL defaults support many model and topology combinations; they do
not promise exact trainer/sampler parity everywhere. Exact lanes opt into a
complete, paired numerical program and fail on unsupported combinations.

## Generic contract surfaces

| Surface | Trainer selection | Required paired behavior |
|---|---|---|
| Attention | `attn_implementation` | Same backend family and KV split schedule |
| RMSNorm | `rmsnorm_mode: sglang_fused` | Same canonical BF16 row and RMSNorm-v2 tree |
| RoPE | `rope_native: true` | Same table construction and rotary application arithmetic |
| Dense projections | `XORL_BI_TRUNK_LINEAR=1` | Paired batch-invariant GEMM contract |
| LM head | `lm_head_fp32: true`, `ce_mode: bi_fused` | Paired projection and vocabulary-normalization trees |
| Router | `router_fp32: true`, `XORL_MOE_BI_ROUTER=1` | Paired gate GEMM, selection, and renormalization |
| Experts | `XORL_MOE_SGLANG_FUSED_EXPERTS=1` where supported | Same expert forward; trainer-owned backward |
| Single-adapter LoRA | `XORL_LORA_MERGED_FORWARD=1` | Canonical folded weight used by trainer and sampler |

These are lower-level contract mechanisms, not a mix-and-match recipe for an
arbitrary model. A model-specific resolver may select them as one unit.

## Supported generic lane

The generic dense lane requires BF16 TP1 projection and head execution, a plain
LM head, a paired SGLang build, and matching attention, RoPE, RMSNorm, GEMM, and
LM-head implementations. Tensor-sharded heads, speculative decoding, sampling
transforms the trainer does not replay, and silent kernel fallback are outside
this envelope.

The generic MoE overlay additionally requires paired router arithmetic and a
serving-layout expert forward. Expert dispatch and distributed combine are
separate contracts; enabling the expert kernel alone does not qualify an EP
topology.

The generic folded-LoRA lane is single-adapter only. Multi-adapter dynamic
serving uses a different forward program and requires its own contract.

## Qualification

Before treating a configuration as exact:

1. assert that every selected contract engaged;
2. generate tokens and retain the sampler's decision-time FP32 log-probability
   bytes;
3. replay the same token IDs through the trainer; and
4. require byte equality for every retained token and K3 exactly zero.

Performance measurements are meaningful only after this correctness gate and
belong with the benchmark run that produced them, not in this source document.
