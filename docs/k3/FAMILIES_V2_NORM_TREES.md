# Families v2: the redefined norm reduction trees

## Problem

The RMSNorm kernel-family contract makes every norm site name the kernel its
serving counterpart runs, which keeps the trainer and the sampler on the same
kernel. It does that by pinning two existing kernels in place. Both of them
leave the reduction order partly to the compiler: they accumulate a sum of
squares through `tl.sum`, whose lowering the kernel author does not control. A
compiler or Triton bump can therefore move the bits under a contract that says
nothing has changed, and it takes three kernel launches per norm call to
evaluate the residual path.

## Contract

Families v2 replaces both trees with one tree that is written out explicitly:

- within a block, an adjacent-pairwise balanced binary tree — `tl.split` plus
  one `float32` add per level. A two-element reduction has exactly one possible
  association, so no tree choice is left to the compiler.
- across blocks, a sequential scalar chain in index order.
- `tl.sum` over more than two elements is not used in any bit-relevant
  position.

The tree is a function of the hidden size alone. A row therefore normalizes to
the same bits whatever batch it arrives in, which is the batch-invariance the
contract exists to provide, and the same bytes run in the trainer and in the
sampler because the module is vendored byte-identical into both.

One tree covers every norm site class: the residual path, the no-residual path,
the zero-centered path, and per-head qk-norm.

## Two realizations, one tree

The tree has two realizations:

- **fused**: one launch per call. `grid=(rows,)`, walking the tile chain
  serially within each row.
- **split**: three launches — per-tile partial trees, the pinned combine, then
  an elementwise normalize — exposing `rows * n_tiles` independent programs.

They compute the identical tree and are **bitwise identical**; adjacent pairing
preserves contiguity at every level, and the `float32` partials and the
`bfloat16` residual round trip are stored and reloaded exactly. That equality is
what allows the choice between them to be made on performance grounds alone.

`_v2_norm_use_split` makes that choice:

```python
V2_NORM_SPLIT_MIN_TILES = 10
# n_tiles is ceil(hidden_size / V2_NORM_TILE), where V2_NORM_TILE = 512
split = n_tiles >= V2_NORM_SPLIT_MIN_TILES and rows <= n_tiles
```

Splitting only pays when the split kernel's 512-wide tile chain is deep and
there are few rows. The tile basis is load-bearing: using the fused kernel's
4096-wide chunk count understates the split path's parallelism by exactly 8x
and selects the wrong realization. The rule above is fitted to the frozen
72-cell H100 record at
`experiments/k3_tests/families_v2/results/norm_structure_switch_h100.json`;
it makes two raw slower choices, only one beyond the 1.01x tie margin, with
1.092x worst regret. Re-fit it with
`experiments/k3_tests/families_v2/bench_norm_structure_switch.py`.

The gates force each realization explicitly rather than reaching one through
this rule. A gate that let the rule choose would compare a realization against
itself as soon as the rule changed, and pass while testing nothing.

## Rollback, and why it is paired

`XORL_FAMILIES_V2=0`, or equivalently `SGLANG_FAMILIES_V2=0`, selects the v1
kernels. Either variable rolls back the engine that reads it, so one setting
applied to both rolls back both.

It has to be applied to both. v1 and v2 are different trees, and the trainer and
the sampler must evaluate the same one; a setting that moved only one engine
would put them on different trees, which is exactly the failure the contract
exists to prevent.

## Migration

v1 and v2 both hold the trainer and the sampler bitwise equal — that is the
contract, and v2 satisfies it. But the two trees do not agree with each other,
so the reported values move when you switch:

- most decode log-probabilities in a sequence change value;
- greedy token selection can change on a small fraction of sequences;
- goldens, frozen anchors, captured bundles, and A/B baselines recorded under v1
  are not valid against v2 and must be re-taken.

This is a re-certification requirement, not a contract violation. Both engines
flip together, and anchors are re-frozen on the new trees before any cross-run
bitwise comparison.
