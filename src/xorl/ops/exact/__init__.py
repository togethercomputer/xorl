"""The serving-parity (exact) contract programs (issue #78 phase 3).

One home and one name for what the historical ``bi_`` / ``exact_`` /
``canonical_`` / ``class_b`` prefixes all meant: byte-pinned programs shared
with the serving engine.

Here: ``sampling_transforms`` (the replay contract), ``rope_class_b``,
``canonical_moe_leaf``, ``canonical_moe_cast``, ``kernel_config_pin``,
``bi_gemm_configs``, ``block_fp8_native``, ``fused_silu_and_mul``.

The modules that exist as literal twins inside the serving engine
(``bi_families_v2``, ``batch_invariant_ops``) live in
:mod:`xorl.ops.sglang`, which carries the paired-edit policy.
"""
