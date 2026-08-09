# MoE serving-kernel forward contract

## Problem

An MoE expert block is a program, not one matrix multiplication. Kernel launch
shape, SiLU multiplication, routing-weight placement, and the top-k reduction
tree all affect the final bf16 bits. Running XoRL's grouped-GEMM forward in the
trainer while SGLang runs `fused_experts_impl` in the sampler therefore leaves
a trainer-sampler log-probability mismatch even when inputs, routes, and weights
are identical.

## Contract

Set `XORL_MOE_SGLANG_FUSED_EXPERTS=1` to make an EP1/TP1 trainer call the same
SGLang expert forward used by serving. XoRL's GKN parameters are exposed as
zero-copy transpose views (`[E,H,2I] -> [E,2I,H]` and `[E,I,H] -> [E,H,I]`),
and the local orchestration preserves SGLang's Triton launches and combine.

The equality boundary ends at the forward output. The sampler does not run
backward, so the custom autograd wrapper reuses XoRL's grouped-GEMM backward to
produce `dX`, routing-weight gradients, and gradients in the original GKN
parameter layout. This keeps the scored function shared without requiring a
serving kernel to become a training kernel.

The opt-in fails loudly for EP/TP expert distribution, expert biases, and
unsupported activations. The current OSS expert container is gated-only; a
future non-gated container needs its own explicit guard and contract. EP
dispatch and combine order are separate contracts; this switch must not
silently claim to cover them.

SGLang and `sgl_kernel` must be importable in the training environment. The
default path is unchanged when the variable is unset or `0`.

## Gates

Development layer replays established the mechanism, but campaign receipts and
benchmark artifacts are intentionally not part of the public source tree. The
public repository keeps the implementation and its conventional equality tests.

Run the dependency-light contract suite with:

```bash
pytest tests/models/test_moe_sglang_fused_experts.py -q
```

Before enabling the path in a new environment, also replay a captured layer
through both the live sampler kernel and this XoRL call and require
`torch.equal`. Kernel-package upgrades require a fresh replay because the
serving program itself is the contract.
