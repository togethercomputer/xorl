"""MoE activation registry.

A single source of truth for the activation functions used by all MoE
backends (native / eager / triton / quack).  Each activation is a binary op
``(gate_out, up_out) -> h`` whose output is fed to the down projection.

New activations are added by:

1. Writing the implementation as a function of ``(gate_out, up_out)``.
2. Registering it in ``MOE_ACTIVATIONS`` under the canonical name.
3. Extending ``normalize_hidden_act`` to recognize any HF aliases.
4. Adding the name to the ``SUPPORTED_HIDDEN_ACTS`` set of each backend
   whose kernel actually implements it (backends validate at entry).
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation constants
# ---------------------------------------------------------------------------

# GPT-OSS clamped SwiGLU. If a future model needs different values, register
# a new ``hidden_act`` kind rather than making these runtime-configurable.
CLAMPED_SWIGLU_ALPHA: float = 1.702
CLAMPED_SWIGLU_LIMIT: float = 7.0


# ---------------------------------------------------------------------------
# Activation implementations
# ---------------------------------------------------------------------------


def silu_swiglu(gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
    """Standard SwiGLU: ``silu(gate) * up``."""
    return F.silu(gate_out) * up_out


def gelu_tanh_glu(gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
    """GeGLU (tanh approx): ``gelu_tanh(gate) * up``."""
    return F.gelu(gate_out, approximate="tanh") * up_out


def clamped_swiglu(gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
    """GPT-OSS clamped SwiGLU.

    Clamp both branches, then ``silu(alpha * gate) * (up + 1)``.
    """
    gate_out = gate_out.clamp(max=CLAMPED_SWIGLU_LIMIT)
    up_out = up_out.clamp(min=-CLAMPED_SWIGLU_LIMIT, max=CLAMPED_SWIGLU_LIMIT)
    return (gate_out * torch.sigmoid(CLAMPED_SWIGLU_ALPHA * gate_out)) * (up_out + 1)


# ---------------------------------------------------------------------------
# Registry & dispatch
# ---------------------------------------------------------------------------

MOE_ACTIVATIONS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "silu": silu_swiglu,
    "gelu_tanh": gelu_tanh_glu,
    "clamped_swiglu": clamped_swiglu,
}

SUPPORTED_HIDDEN_ACTS: frozenset[str] = frozenset(MOE_ACTIVATIONS.keys())


def normalize_hidden_act(hidden_act: str | None) -> str:
    """Normalize a HF-style ``hidden_act`` string to a canonical MoE act kind."""
    if hidden_act is None or hidden_act == "silu":
        return "silu"
    if hidden_act in ("gelu_tanh", "gelu_pytorch_tanh"):
        return "gelu_tanh"
    if hidden_act == "clamped_swiglu":
        return "clamped_swiglu"
    raise ValueError(f"Unsupported hidden_act={hidden_act!r}. Supported: {sorted(SUPPORTED_HIDDEN_ACTS)}")


def check_hidden_act_supported(hidden_act: str, backend: str, supported: frozenset[str]) -> None:
    """Raise if ``hidden_act`` is not in the backend's supported set."""
    if hidden_act not in supported:
        raise ValueError(
            f"MoE backend {backend!r} does not support hidden_act={hidden_act!r}. Supported: {sorted(supported)}"
        )


def apply_moe_activation(
    hidden_act: str,
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
) -> torch.Tensor:
    """Apply the activation named by ``hidden_act`` to split gate/up tensors.

    Uses explicit ``if`` chain rather than a dict lookup so ``torch.compile``
    specializes on the string value.
    """
    if hidden_act == "silu":
        return silu_swiglu(gate_out, up_out)
    if hidden_act == "gelu_tanh":
        return gelu_tanh_glu(gate_out, up_out)
    if hidden_act == "clamped_swiglu":
        return clamped_swiglu(gate_out, up_out)
    raise ValueError(f"Unknown hidden_act={hidden_act!r}. Supported: {sorted(SUPPORTED_HIDDEN_ACTS)}")
