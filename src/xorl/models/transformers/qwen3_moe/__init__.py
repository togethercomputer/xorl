"""Qwen3 MoE model."""

# Canonical MoE layer abstractions
from ...layers.moe import (
    MOE_EXPERT_BACKENDS,
    MoEBlock,
    MoEExperts,
    MoEExpertsLoRA,
    MoELoRAConfig,
    TopKRouter,
)
from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseExperts,
    Qwen3MoeSparseNativeMoeBlock,
    Qwen3MoeSparseTritonMoeBlock,
    Qwen3MoeTritonExperts,
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
