# LoRA forward contract

## Problem

Dynamic LoRA can make trainer and sampler evaluate different programs. The
trainer may compute `base(x) + lora(x)`, while the sampler may use SGMV or fused
MoE hooks with different K reductions, scaling placement, additions, and BF16
rounding boundaries. Matching factor values does not make those programs
bitwise equivalent.

## Canonical folded lane

The generic single-adapter exact lane folds the adapter before the forward:

```text
delta_fp32 = factor_product_fp32 * float(scaling)
forward_weight = (base_weight_fp32 + delta_fp32).to(base_weight.dtype)
```

Dense factors use `B @ A`; expert factors use the GKN-oriented batched product.
The result is accumulated in FP32 and cast once. Training forwards and weight
synchronization consume the same folded bytes, while autograd differentiates
the fold into the adapter factors.

Enable this generic mechanism with `XORL_LORA_MERGED_FORWARD=1`. Cache entries
are keyed by parameter version; `XORL_LORA_MERGED_FORWARD_CACHE=0` disables the
cache for diagnosis. The lane is restricted to one active adapter and to base
kernels already covered by a train/serve contract.

This is not a contract for dynamic multi-adapter serving. A model-specific
implementation may instead contract active LoRA by running the sampler's LoRA
forward literally and supplying a trainer-only backward.

## Verification

```bash
pytest tests/models/test_lora_merged_forward.py -q
pytest tests/ops/test_lora_utils.py -q
```

The public tests cover fold order, cache invalidation, adapter gradients,
expert-layout commutation, and fail-closed composition. Cross-engine and
end-to-end replay remain release gates for the paired trainer and sampler
revisions; campaign artifacts are not stored here.
