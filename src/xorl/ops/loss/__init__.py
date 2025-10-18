"""
Loss functions for training.

This module provides various loss functions for language model training:
- causallm_loss_function: Standard causal language modeling loss
- importance_sampling_loss_function: Importance sampling loss for GRPO/RL
- policy_loss_function: PPO-style policy loss with TIS correction
"""

from typing import Callable, Dict

from xorl.ops.loss.causallm_loss import causallm_loss_function
from xorl.ops.loss.importance_sampling_loss import importance_sampling_loss_function
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.policy_loss import policy_loss_function
from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


# ---------------------------------------------------------------------------
# Loss function registry
# ---------------------------------------------------------------------------
LOSS_REGISTRY: Dict[str, Callable] = {
    "causallm_loss": causallm_loss_function,
    "cross_entropy": causallm_loss_function,  # alias
    "importance_sampling": importance_sampling_loss_function,
    "policy_loss": policy_loss_function,
}


def get_loss_function(name: str) -> Callable:
    """Look up a loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]


def register_loss_function(name: str, fn: Callable) -> None:
    """Register a custom loss function."""
    LOSS_REGISTRY[name] = fn


__all__ = [
    "LossOutput",
    "LOSS_REGISTRY",
    "get_loss_function",
    "register_loss_function",
    "causallm_loss_function",
    "importance_sampling_loss_function",
    "policy_loss_function",
    "vocab_parallel_cross_entropy",
]
