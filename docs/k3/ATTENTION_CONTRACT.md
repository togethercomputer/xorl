# Attention contract

Attention reduces over the KV axis. Whether the trainer and the sampler agree bit for bit
depends on two things: **which** flash-attention build runs, and **in how many pieces** it
splits that reduction.

This document covers the trainer half. The sampler half lives in the serving engine's
deterministic-inference documentation.

## The two knobs

### 1. Same backend on both sides

`flash_attn_interface` FA3, `sgl_kernel.flash_attn` FA3 and the FA4 CUTE kernel are
separate compilations of the same algorithm, and reduction order is a property of the
compilation rather than of the version number. Pair the trainer with the sampler it will be
scored against:

- FA4 trainer with an FA4 sampler is gated bitwise: offline attention-block replay against
  a sampler capture returns `frac_neq = 0.0`, and a live dense bench reads 1889/1889
  generated tokens bitwise-equal at `k3 = 0.0` with kernel engagement asserted.
- FA3 trainer with an FA3 sampler holds in offline block replay, provided both sides run
  the same FA3 build. It has no bitwise gate in this repo yet.
- Cross-pairing — an FA3 trainer against an FA4 sampler, or two different FA3 builds — is
  untested. Match the sampler's backend rather than assume the pair holds.

### 2. Pin the split count

Flash attention can split one query block's KV range into `num_splits` chunks, reduce each
chunk independently, and combine the partials afterwards. `num_splits=0` asks for a
heuristic keyed on batch size and sequence shape, so **the same request gets a different
reduction order — and different bits — depending on who else is in the batch.**

`num_splits=1` removes the cross-block combine entirely: one query block accumulates its
whole KV range in a single fixed serial order, which is batch-invariant by construction.

Which entry point you call decides whether the heuristic runs at all. FA4's
`flash_attn_func` and `flash_attn_varlen_func`, and FA3's non-cache entry points, all
default `num_splits=1`; only the `*_with_kvcache` family defaults to `0`. The trainer's
packed varlen path therefore never reached the heuristic — the serving path, which is
`*_with_kvcache`, is where the pin does work.

Where it is load-bearing, measured on `sgl_kernel` FA3 over a page-size-1 KV cache at decode
shapes: `num_splits` 0 versus 1 changes bits **at batch size 1 only**, by exactly one bf16
ulp at the magnitude of the affected elements, and is bit-identical at batch 8, 64 and 128.

On Hopper there is no split-KV path in FA4 to pin at all — the CUTE kernel asserts
`not is_split_kv` on SM90. At batch 1 the heuristic asks for 4 splits and the kernel raises;
at batch 8 and above it asks for none, and `num_splits` 0 and 1 are bitwise identical.

Pinning is not free everywhere. In a controlled A/B with `ignore_eos` and the sampling seed
pinned (one 9.7B dense model, 1×H100, CUDA graphs on), deterministic and stock FA4 agree on
attention step time to within −0.9 %…+3.2 % across seven batch/context shapes, since FA4 has
nothing to split on SM90. On FA3 — the one backend that does split there — the pin costs
8.3 % of step time at batch 1 and 4k context, rising to 26.2 % at 32k.

## What this repo does

The default packed trainer path calls `flash_attn_varlen_func`, which is not the entry
point the serving engine uses. Two environment-gated routes in
[`flash_attention.py`](../../src/xorl/models/layers/attention/backend/flash_attention.py)
run packed self-attention through the serving entry point instead — a page-size-1 KV cache
(one page per token) driven by `page_table`/`cache_seqlens` metadata, with `num_splits=1`:

| Flag | Kernel | Use |
|---|---|---|
| `XORL_FLASH_ATTN_SGL_KERNEL=1` | `sgl_kernel.flash_attn.flash_attn_with_kvcache` | The exact serving build. Takes the build out of the variables when localizing a mismatch. |
| `XORL_FLASH_ATTN_PAGED_KVCACHE=1` | `flash_attn_interface.flash_attn_with_kvcache` | Same entry point, xorl's own FA3 build. Separates entry-point effects from build effects. |

Both are opt-in diagnostics rather than parity requirements. The two FA3 builds are
bitwise equal on the shapes measured here — 6 of 6, maximum difference exactly 0.0 — so
the build is not a source of divergence; a mismatch once attributed to it was root-caused
to RoPE, and with matching post-RoPE q/k the attention core reproduces the serving
engine's output.

`XORL_FLASH_ATTN_SGL_KERNEL` covers packed varlen batches and the single-sequence `B=1`
batched shape (a causal forward over `[1, S]` is one varlen sequence of length `S`, so
`cu_seqlens=[0, S]` is synthesized). Shapes the route cannot represent — cross-attention
`cu_seqlens`, mismatched q/k/v spans — **raise**. A silent fallback would return plausible
numbers from a different reduction order, which is exactly the failure this contract
exists to prevent.

The FA4 CUTE path pins `num_splits=1` on both its varlen and batched calls, and forwards
`softmax_scale`, the module's `causal` flag, and `deterministic` — the FA4 and FA3 branches
now take identical arguments, so switching backends changes the kernel and nothing else.

## FA4-only environments

Some CUDA 13 images ship `flash_attn.cute` but not `flash_attn_interface`. The FA3 import
is therefore optional, and `flash_attention_forward` falls through to FA4 when FA3 is
absent.

That fallback needs the registry to agree. Dispatch used to read
`ATTENTION_FUNCTIONS.get(impl, eager_attention_forward)`, so a `flash_attention_3` key that
is registered only when FA3 imports made `attn_implementation: flash_attention_3`
**silently run eager attention** in an FA4-only environment — a correct-looking run with no
contract at all. Two changes close that:

- The `flash_attention_2` and `flash_attention_3` keys are registered whenever any flash
  build is importable, and `is_flash_attention` reports membership in the live registry
  rather than in a static name set.
- Dispatch goes through `get_attention_fn`, which **raises** for a flash implementation that
  no installed build provides, naming what is missing. `xorl.models.auto` calls it before
  loading weights, so an unavailable backend fails at startup rather than at the first
  forward. Only the flash family raises; non-flash names still resolve to eager.

## Checking it

```bash
pytest tests/ops/test_attention.py -v
```

The tests assert the routed kernel receives `num_splits=1`, a page-size-1 `k_cache`
(`[total_tokens, 1, num_heads, head_dim]`), the derived `page_table`/`cache_seqlens`, and
the caller's `softmax_scale`; that both flags off leaves the default varlen path untouched;
that flash keys stay registered when FA3 is unavailable; and that a requested flash backend
with no installed build raises instead of degrading to eager.
