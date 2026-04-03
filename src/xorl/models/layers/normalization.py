from typing import Callable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn


RMSNormMode = Literal["eager", "native", "compile"]
_RMSNORM_MODE: RMSNormMode = "native"
_COMPILED_NATIVE_RMS_NORM: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None
_COMPILED_ZERO_CENTERED_RMS_NORM: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None


def set_rmsnorm_mode(mode: RMSNormMode) -> None:
    global _RMSNORM_MODE
    if mode not in {"eager", "native", "compile"}:
        raise ValueError(f"Unsupported rmsnorm_mode: {mode}")
    _RMSNORM_MODE = mode


def get_rmsnorm_mode() -> RMSNormMode:
    return _RMSNORM_MODE


def eager_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Eager implementation of RMSNorm."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def native_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    return F.rms_norm(hidden_states, (weight.shape[0],), weight, eps=variance_epsilon)


def compiled_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    global _COMPILED_NATIVE_RMS_NORM
    if _COMPILED_NATIVE_RMS_NORM is None:
        _COMPILED_NATIVE_RMS_NORM = torch.compile(native_rms_norm)
    return _COMPILED_NATIVE_RMS_NORM(hidden_states, weight, variance_epsilon)


def eager_zero_centered_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    output = hidden_states.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + variance_epsilon)
    output = output * (1.0 + weight.float())
    return output.type_as(hidden_states)


def native_zero_centered_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    output = F.rms_norm(
        hidden_states.float(),
        (weight.shape[0],),
        1.0 + weight.float(),
        eps=variance_epsilon,
    )
    return output.type_as(hidden_states)


def compiled_zero_centered_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    global _COMPILED_ZERO_CENTERED_RMS_NORM
    if _COMPILED_ZERO_CENTERED_RMS_NORM is None:
        _COMPILED_ZERO_CENTERED_RMS_NORM = torch.compile(native_zero_centered_rms_norm)
    return _COMPILED_ZERO_CENTERED_RMS_NORM(hidden_states, weight, variance_epsilon)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, mode: Optional[RMSNormMode] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.mode: RMSNormMode = mode or get_rmsnorm_mode()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual_out: Optional[torch.Tensor] = None
        norm_input = hidden_states
        if residual is not None:
            residual_out = hidden_states + residual
            norm_input = residual_out

        if self.mode == "eager":
            out = eager_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "native":
            out = native_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "compile":
            out = compiled_rms_norm(norm_input, self.weight, self.variance_epsilon)
        else:
            raise ValueError(f"Unsupported rmsnorm_mode: {self.mode}")

        if residual_out is not None and prenorm:
            return out, residual_out
        return out

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, mode={self.mode}"
