# MoE expert forward contract

## The expert block is one numerical program

An MoE expert forward includes token alignment, gate/up projections, BF16
rounding, activation, the down projection, routing-weight placement, local
top-k reduction, and distributed combine. Matching only a GEMM formula or
weight layout is insufficient.

The exact lanes make training evaluate the sampler's expert value program and
attach a trainer-owned backward. The equality boundary is the forward output;
the sampler does not need to implement autograd.

## Generic BF16 mechanism

For supported BF16 experts, XoRL exposes GKN parameters through serving-layout
views and invokes SGLang's fused expert forward. The custom autograd wrapper
returns activation, routing-weight, and original-layout parameter gradients.
Unsupported activation, bias, layout, or distribution combinations fail rather
than falling back to the ordinary grouped-GEMM forward.

Architecture-selected Qwen MoE uses this principle together with its qualified
EP8 dispatch and `canonical_moe_fold_v1`. Transport restores logical
contributor order; the fold then evaluates a BF16-rounded adjacent-pair tree.
The resolver, rather than a user-provided component flag, owns the production
choice.

## GLM-5.2 native-FP8 experts

GLM's exact expert lane preserves the native block-FP8 weights and FP32 scales.
Trainer and sampler use the same deterministic SGLang expert configuration,
token alignment, activation roundpoints, and local reduction. Changing the
number of routed rows changes the number of aligned M blocks, not the fixed
K-axis program.

Active rank-1 LoRA uses the same SGLang MoE hooks in both engines. Its shrink
and expand GEMMs have fixed rank-aware blocks and no split-K partial merge.
Trainer-only autograd supplies factor and activation gradients.

EP dispatch and the same 16-leaf logical adjacent-pair fold remain explicit
parts of the GLM contract; local expert equality alone does not qualify the
distributed model.

## Verification

```bash
pytest tests/models/test_moe_sglang_fused_experts.py -q
pytest tests/models/test_moe_sglang_fused_experts_ep.py -q
pytest tests/models/test_glm52_exact_routed_experts_qlora.py -q
```

Package or kernel upgrades require a fresh cross-engine replay because the
executed serving program—not its API name—is the contract.
