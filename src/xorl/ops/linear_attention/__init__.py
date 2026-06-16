"""Linear attention ops and layers used by Qwen3.5."""

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from .layers.gated_deltanet import GatedDeltaNet


__all__ = [
    "GatedDeltaNet",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
