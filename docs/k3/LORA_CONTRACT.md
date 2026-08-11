# LoRA train/serve forward contracts

## Why one generic LoRA path is insufficient

The same factors can produce different bytes when trainer and sampler use
different base GEMMs, low-rank kernels, scaling placement, additions, or BF16
rounding boundaries. Exactness therefore belongs to a complete model forward,
not to an adapter checkpoint format alone.

This stack contains two model-specific solutions.

## Qwen: canonical merged forward

The Qwen exact resolver marks injected adapter modules for a merged forward.
Each adapted weight is formed as:

```text
delta_fp32 = factor_product_fp32 * float(scaling)
forward_weight = (base_weight_fp32 + delta_fp32).to(base_weight.dtype)
```

Dense factors use `B @ A`; expert factors use the GKN-oriented batched product.
The trainer runs the already-contracted base forward on that folded weight, and
weight synchronization publishes the same folded bytes. Straight-through
autograd differentiates the fold into the trainable factors.

The resolver owns this choice through module state; there is no public
`XORL_LORA_MERGED_FORWARD` launch flag on the architecture-selected path.

Qwen3.5/3.6 training defaults keep the GDN input as four
independent rank-r adapters: `q_proj`, `k_proj`, `v_proj`, and `g_proj` (the
trainer name for serving's `in_proj_z`). The optional River/SGLang-shaped
`in_proj_qkvz` adapter remains available only when explicitly targeted.

On Qwen3.6 MoE layers, `gate_proj`, `up_proj`, and `down_proj` also cover every
shared expert. Gate and up own independent factors even though their frozen
base remains one fused `gate_up_proj` tensor. The trainer folds those logical
factors into the fused GEMM, the ordered EP combine consumes the same folded
weights, and weight synchronization publishes only the corresponding fused
base-weight bytes. Gate and up must therefore be selected together.

## GLM-5.2: native-FP8 base plus active rank-1 LoRA

Folding adapters into GLM's native-FP8 expert base would require FP8
requantization and would define a different serving program. The exact GLM
lane instead keeps the FP8 base and scales frozen and executes the sampler's
active-LoRA value path literally in the trainer.

The admitted configuration uses FP32 factor masters with rank 1, alpha 1, and
one explicit BF16 factor view consumed by both engines. It covers the LM head,
attention/MLA projections, dense MLPs, shared experts, and routed-expert
gate/up/down projections. Router and DSA indexer parameters remain frozen.

The exact forward calls the matching SGLang block-FP8, fused projection, MoE
hook, and LM-head programs with the same layouts and rounding boundaries.
Trainer-only autograd supplies activation and factor gradients from the
effective BF16 factor values. It is a checked straight-through treatment of the
frozen quantized base, not a derivative of FP8 quantization.

## Unsupported inheritance

Neither lane implicitly covers multiple simultaneously active adapters,
arbitrary LoRA rank or scaling, a different target universe, a different FP8
format, or a new TP/EP topology. Those combinations require an explicit
forward contract and replay gate.

## Verification

```bash
pytest tests/models/test_lora_merged_forward.py -q
pytest tests/models/test_qwen35_lora_projection_topology.py -q
pytest tests/e2e/qwen3_5/test_lora_projection_topology.py -q  # one GPU
pytest tests/models/test_glm52_exact_qlora.py -q
pytest tests/models/test_glm52_exact_gate_up_qlora.py -q
pytest tests/models/test_glm52_exact_shared_expert_qlora.py -q
pytest tests/models/test_glm52_exact_routed_experts_qlora.py -q
pytest tests/models/test_glm52_exact_lm_head_qlora.py -q
```

The public tests cover fold order, model admission, factor layouts, forward
rounding, gradients, and fail-closed combinations. Cross-engine and full-model
replay qualify the paired revisions; campaign receipts are intentionally kept
outside the public source tree.
