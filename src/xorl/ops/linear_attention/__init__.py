"""Linear attention kernels used by Qwen3.5.

The ``GatedDeltaNet`` layer class moved to
:mod:`xorl.models.layers.gated_deltanet` (issue #78 phase 4); it is
re-exported here lazily for one deprecation cycle.
"""

from .ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def __getattr__(name: str):
    if name == "GatedDeltaNet":
        from xorl.models.layers.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GatedDeltaNet",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
