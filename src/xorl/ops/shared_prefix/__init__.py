"""Shared-prefix attention for RL policy updates.

When an RL rollout samples N responses for one shared prompt P, naive attention
runs over ``N*(P+R)`` tokens. SharedPrefix repacks the packed micro-batch so the
shared prefix's KV is computed once: per group, a shared prefix block plus one
decoded block per response (with the last prompt token duplicated into each, so
the repacked<->original token mapping is 1:1). For long shared prompts (P >> R)
this saves attention and MLP compute across every layer, and composes with
Ulysses sequence parallelism.

This package provides the framework-side data path:
- ``repack``: detect shared-prefix groups, repack tokens + loss fields, and remap
  per-token outputs back to the original layout.

The attention kernel lives in
``xorl.models.layers.attention.backend.shared_prefix_attention`` and is driven by
the :class:`SharedPrefixContext` produced here.
"""

from .repack import (
    SharedPrefixContext,
    detect_prompt_groups,
    shared_prefix_remap_to_original,
    shared_prefix_repack_batch,
)


__all__ = [
    "SharedPrefixContext",
    "detect_prompt_groups",
    "shared_prefix_remap_to_original",
    "shared_prefix_repack_batch",
]
