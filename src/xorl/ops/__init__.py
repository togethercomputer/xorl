from .linear_attention import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from .moe.quack import quack_moe_forward
from .moe.triton import TritonMoeExpertsFunction, triton_moe_forward
from .moe.triton_lora import (
    TritonMoeExpertsLoRAFunction,
    triton_moe_lora_forward,
)
from .ssm import ssd_chunked


def __getattr__(name: str):
    # Objective functions moved to xorl.objectives (#78 phase 2); resolve the
    # historical xorl.ops re-exports lazily to avoid an import cycle (the
    # objectives import the loss kernels under this package).
    if name in ("causallm_loss_function", "importance_sampling_loss_function"):
        import xorl.objectives as _objectives

        return getattr(_objectives, name)
    # Layer classes moved to xorl.models.layers (#78 phase 4).
    if name == "GatedDeltaNet":
        from xorl.models.layers.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet
    if name == "Mamba2Mixer":
        from xorl.models.layers.mamba2_mixer import Mamba2Mixer

        return Mamba2Mixer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TritonMoeExpertsFunction",
    "triton_moe_forward",
    "quack_moe_forward",
    "fused_silu_and_mul",
    "TritonMoeExpertsLoRAFunction",
    "triton_moe_lora_forward",
    "causallm_loss_function",
    "importance_sampling_loss_function",
    "GatedDeltaNet",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "Mamba2Mixer",
    "ssd_chunked",
]
