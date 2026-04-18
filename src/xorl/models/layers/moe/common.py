"""Shared helpers for MoE expert weight layouts."""

import torch


def split_gate_up_proj(
    gate_up_proj: torch.Tensor,
    intermediate_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused ``gate_up_proj`` tensor into gate and up views."""
    if intermediate_size is None:
        intermediate_size = gate_up_proj.shape[-1] // 2
    return (
        gate_up_proj[..., :intermediate_size],
        gate_up_proj[..., intermediate_size:],
    )


def fuse_gate_up_proj(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
) -> torch.Tensor:
    """Concatenate separate gate/up tensors into fused ``gate_up_proj`` format."""
    return torch.cat([gate_proj, up_proj], dim=-1)
