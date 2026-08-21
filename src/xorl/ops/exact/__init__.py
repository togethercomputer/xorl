"""The serving-parity (exact) contract programs (issue #78 phase 3).

One home and one name for what the historical ``bi_`` / ``exact_`` /
``canonical_`` / ``class_b`` prefixes all meant: byte-pinned programs shared
with the serving engine.

Physically here: ``rope_class_b``, ``canonical_moe_leaf``,
``canonical_moe_cast``, ``kernel_config_pin``, ``bi_gemm_configs``,
``block_fp8_native``, ``fused_silu_and_mul``.

Aliased here but deliberately NOT moved:

- ``families_v2`` -> :mod:`xorl.ops.bi_families_v2` — vendored byte-identical
  into the serving engine and sha256-gated; the file cannot move.
- ``batch_invariant`` -> :mod:`xorl.ops.batch_invariant_ops` — vendored-
  adapted from SGLang's ``srt/batch_invariant_ops``; it stays a single file
  at its path so it remains diffable against the serving twin.
- ``sampling_transforms`` -> :mod:`xorl.ops.exact_sampling_transforms` — the
  replay contract; kept in place while in-flight work (#74) rewrites it, to
  be flipped to canonical here afterwards.
"""
