"""
Loss functions for training.

This module provides various loss functions for language model training:
- causallm_loss_function: Standard causal language modeling loss
- importance_sampling_loss_function: Importance sampling loss for GRPO/RL
- miles_policy_loss_function: Miles-style policy loss with PPO clipping
- distillation_loss_function: Knowledge distillation loss using JSD
- fast_fused_linear_cross_entropy: Memory-efficient fused linear cross-entropy
"""

from .causallm_loss import causallm_loss_function
from .distillation_loss import distillation_loss_function
from .fused_linear import fast_fused_linear_cross_entropy
from .importance_sampling_loss import importance_sampling_loss_function
from .miles_policy_loss import (
    apply_tis_correction,
    compute_ppo_loss,
    miles_policy_loss_function,
)

__all__ = [
    # Causal LM loss
    "causallm_loss_function",
    # Importance sampling loss
    "importance_sampling_loss_function",
    # Miles policy loss
    "compute_ppo_loss",
    "apply_tis_correction",
    "miles_policy_loss_function",
    # Distillation loss
    "distillation_loss_function",
    # Fused linear cross-entropy
    "fast_fused_linear_cross_entropy",
]
