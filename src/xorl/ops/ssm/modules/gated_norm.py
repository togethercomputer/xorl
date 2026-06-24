from __future__ import annotations

# Matches transformers' Zamba2RMSNormGated (used by NemotronHMamba2Mixer as `norm`):
# the gate (silu) is applied BEFORE normalization, and the RMS statistic is computed
# per group of `group_size` channels, all in fp32.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupRMSNormGated(nn.Module):
    """Gated grouped RMSNorm: ``weight * group_rmsnorm(x * silu(gate))``."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by group_size={group_size}.")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        *prefix_dims, last_dim = hidden_states.shape
        group_count = last_dim // self.group_size
        hidden_states_group = hidden_states.view(*prefix_dims, group_count, self.group_size)
        variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
        hidden_states_group = hidden_states_group * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_group.view(*prefix_dims, last_dim)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, group_size={self.group_size}, eps={self.variance_epsilon}"
