from __future__ import annotations

# Minimal local RMSNorm adapted from flash-linear-attention/fla/modules/layernorm.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del residual_in_fp32
        residual_out = x if residual is None else x + residual
        norm_input = residual_out.float()
        inv_rms = torch.rsqrt(norm_input.square().mean(dim=-1, keepdim=True) + self.eps)
        y = norm_input * inv_rms
        if self.weight is not None:
            y = y * self.weight.float()
        if self.bias is not None:
            y = y + self.bias.float()
        y = y.to(x.dtype)
        return y if not prenorm else (y, residual_out)
