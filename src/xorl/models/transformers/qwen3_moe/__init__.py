"""Qwen3 MoE model."""

from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3_moe import (
    Qwen3MoeSparseExperts,
    Qwen3MoeTritonExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseTritonMoeBlock,
    Qwen3MoeSparseNativeMoeBlock,
)

# Canonical MoE layer abstractions
from ...layers.moe import (
    MoEBlock,
    MoEExperts,
    MoEExpertsLoRA,
    MoELoRAConfig,
    TopKRouter,
    MOE_EXPERT_BACKENDS,
)

__all__ = [
    # Base model
    "Qwen3MoeConfig",
    "Qwen3MoeSparseExperts",
    "Qwen3MoeTritonExperts",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoeSparseTritonMoeBlock",
    "Qwen3MoeSparseNativeMoeBlock",
    # Canonical MoE layers
    "MoEBlock",
    "MoEExperts",
    "MoEExpertsLoRA",
    "MoELoRAConfig",
    "TopKRouter",
    "MOE_EXPERT_BACKENDS",
]
