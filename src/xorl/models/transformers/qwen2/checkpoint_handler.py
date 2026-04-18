"""Checkpoint handler for Qwen2 dense models.

Identical to Qwen3CheckpointHandler — same merge/split logic for
qkv_proj and gate_up_proj fused projections.
"""

from ..qwen3.checkpoint_handler import Qwen3CheckpointHandler


Qwen2CheckpointHandler = Qwen3CheckpointHandler
