"""Linear attention ops and layers used by Qwen3.5."""

from .layers.gated_deltanet import GatedDeltaNet
from .ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


__all__ = [
    "GatedDeltaNet",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
