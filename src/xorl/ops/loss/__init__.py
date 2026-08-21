"""Cross-entropy / selected-logprob KERNELS.

The RL and supervised objective functions moved to :mod:`xorl.objectives`
(issue #78 phase 2).  Their public API is re-exported here lazily (PEP 562)
for one deprecation cycle so ``from xorl.ops.loss import ...`` keeps working
without creating an import cycle (the objectives import the kernels in this
package).
"""

from typing import Literal

from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


# Cross-entropy computation mode shared by the local-trainer (TrainingArguments)
# and server-runner (ServerArguments) entry points so the Literal stays in sync.
# ``bi_fused`` runs the shared batch-invariant projection and fixed-order LSE.
CrossEntropyMode = Literal["eager", "compiled", "bi_fused", "quack_linear", "fused_quack"]

_OBJECTIVE_EXPORTS = frozenset(
    {
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
    }
)


def __getattr__(name: str):
    if name in _OBJECTIVE_EXPORTS:
        import xorl.objectives as _objectives

        return getattr(_objectives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # noqa: F822  (objective names resolve via __getattr__)
    "CrossEntropyMode",
    "vocab_parallel_cross_entropy",
    *sorted(_OBJECTIVE_EXPORTS),
]
