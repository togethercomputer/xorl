# RMSNorm kernel-family contract

## Problem

Two batch-invariant RMSNorm kernels are in use. One accumulates the sum of
squares in a looped `tl.sum` and divides by `tl.sqrt`. The other squares
elementwise, reduces with the shared batch-invariant `mean_dim`, and scales by
`tl.rsqrt`. Both are correct implementations of the same formula, and on rare
bf16 boundary values they round apart by one ulp (roughly 2 elements in 524288
at shape `[4096, 128]`).

Which of the two a sampler runs is decided by the call: SGLang dispatches the
looped kernel when there is no residual argument and the `mean_dim` kernels on
the residual path. The trainer used to reach a kernel the same way, as a
consequence of how each call site happened to be written. Rewriting a call site
could therefore change kernel without changing any declaration, leaving the
trainer and the sampler on different kernels for the same layer. Five one-ulp
seed elements accounted for a measured 2.99e-5 trainer/sampler log-probability
gap that grows through the layers.

## Contract

Every RMSNorm site names the kernel family that its serving counterpart runs:

- `serving_no_residual`: the looped `tl.sum` kernel SGLang dispatches when
  `residual is None`, and the kernel behind the `aten::rms_norm` interpose.
  Sites are the qk-norms and the layer-0 input layernorm.
- `serving_residual_tree`: the `mean_dim` and `tl.rsqrt` kernels SGLang
  dispatches on residual calls. Sites are the input layernorm at layer>0, the
  post-attention layernorm, and the final norm.

`bi_rms_norm` and `bi_fused_add_rms_norm` in `xorl.ops.batch_invariant_ops` are
the entry points, and both take the family as a required keyword argument.
Nothing calls the family kernels directly.

A site declares its family at module construction, `RMSNorm(..., family=...)`,
or on the call, `norm(x, family=...)`. The declaration takes the place of the
`force_sglang_residual=<expression>` arguments the call sites used to carry, and
dispatch is bit-identical to those expressions in every combination of rmsnorm
mode, batch-invariant mode, and trunk contract.

Combinations that serving cannot produce raise instead of picking a kernel: a
`serving_no_residual` site called with a residual stream, a
`serving_no_residual` site forced onto the residual tree, and a fused residual
add requested through `serving_no_residual`.

An undeclared call in a parity configuration (an sglang mode with either the
global interpose or the scoped trunk contract active) still dispatches as before
and warns once per mode and call shape, naming the family it fell through to.
`XORL_RMSNORM_REQUIRE_FAMILY=1` turns that warning into a `RuntimeError`, which
is the setting for a lane that must be certified.

The Qwen3.5 zero-centered norm is `serving_no_residual` with
`zero_centered=True`: an fp32 upcast with `1 + weight` folded in fp32 around the
same reduction tree. It is a fold on the no-residual family, not a third family,
and it has no residual form.

This contract governs which kernel the forward runs. Backward stays on the
closed-form RMSNorm gradient, which does not enter the trainer/sampler
comparison.

## Verification

```bash
pytest tests/models/test_rmsnorm_family_contract.py -q
pytest tests/models/test_rmsnorm_family_cross_engine.py -q
```

The first file pins each family funnel to its kernel bitwise, checks that
family-declared module calls reproduce the legacy call shapes bitwise across
rmsnorm modes and batch-invariant settings, checks that the Qwen3 layers and the
shared attention module declare their site families, and exercises the violation
errors and the undeclared-call tripwire. It also asserts that the two families
still disagree on the `[4096, 128]` seed shape. If they ever agree there, every
bitwise gate in both files passes for free and this contract needs
re-examination.

The second file compares xorl's dispatch against SGLang's dispatch per site
class on four shapes, covering the scoped trunk-contract lane and the
zero-centered form. It needs `sglang.srt.batch_invariant_ops` importable (pure
Triton, no `sgl_kernel`); point `SGLANG_REPO` at an SGLang checkout's `python`
directory if SGLang is not installed. It skips when SGLang is absent. Both files
need a Hopper GPU for the bitwise gates.

A kernel-package upgrade on either side calls for a fresh run of the
cross-engine file, because the dispatched program is what the contract pins.
