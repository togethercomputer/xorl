"""Training objectives (issue #78 phase 2).

The RL and supervised objective functions, their reducers, and the loss
registry.  These consume the cross-entropy/selected-logprob KERNELS in
:mod:`xorl.ops.loss`; the kernels stay there — this package is the
trainer-facing API.
"""

from typing import Callable, Dict

from xorl.objectives.causallm_loss import causallm_loss_function, fsdp_sharded_causallm_loss_function
from xorl.objectives.cispo_loss import cispo_loss_function
from xorl.objectives.grpo_loss import drgrpo_loss_function
from xorl.objectives.importance_sampling_loss import importance_sampling_loss_function
from xorl.objectives.loss_output import LossOutput
from xorl.objectives.opd_loss import OPDLossMetrics, opd_loss_function, opd_vocab_parallel_loss_function
from xorl.objectives.policy_loss import policy_loss_function
from xorl.objectives.reducers import Reducer, SequencePartial, TokenPartial


# ---------------------------------------------------------------------------
# Loss function registry
# ---------------------------------------------------------------------------
LOSS_REGISTRY: Dict[str, Callable] = {
    "causallm_loss": causallm_loss_function,
    "cross_entropy": causallm_loss_function,  # alias
    "importance_sampling": importance_sampling_loss_function,
    "cispo": cispo_loss_function,
    "policy_loss": policy_loss_function,
    "drgrpo": drgrpo_loss_function,
    "opd_loss": opd_loss_function,
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
    "LOSS_REGISTRY",
    "LossOutput",
    "OPDLossMetrics",
    "Reducer",
    "SequencePartial",
    "TokenPartial",
    "causallm_loss_function",
    "cispo_loss_function",
    "drgrpo_loss_function",
    "fsdp_sharded_causallm_loss_function",
    "get_loss_function",
    "importance_sampling_loss_function",
    "opd_loss_function",
    "opd_vocab_parallel_loss_function",
    "policy_loss_function",
    "register_loss_function",
]
