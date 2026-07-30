from __future__ import annotations

# Minimal local RMSNorm-gate path adapted from flash-linear-attention/fla/modules/fused_norm_gate.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.
import torch
import torch.nn as nn

from xorl.ops.linear_attention.modules.bi_contract import bi_rms_norm_gated, is_gdn_contract_enabled


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del residual_in_fp32
    if activation not in {"swish", "silu", "sigmoid"}:
        raise ValueError(f"Unsupported activation: {activation}")

    residual_out = x if residual is None else x + residual
    norm_input = residual_out.float()
    inv_rms = torch.rsqrt(norm_input.square().mean(dim=-1, keepdim=True) + eps)
    y = norm_input * inv_rms
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()

    gate = g.float()
    if activation in {"swish", "silu"}:
        y = y * gate * torch.sigmoid(gate)
    else:
        y = y * torch.sigmoid(gate)

    y = y.to(x.dtype)
    return y if not prenorm else (y, residual_out)


class FusedRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if is_gdn_contract_enabled() and x.is_cuda:
            # GDN contract: serving's layernorm_gated kernel kills the
            # rare bf16-ULP gated-RMSNorm tail. Only the GDN o_norm config is
            # contracted; anything else voids the bitwise claim, so fail loud.
            if residual is not None or prenorm or self.bias is not None or self.weight is None:
                raise NotImplementedError(
                    "XORL_BI_GDN gated-RMSNorm contract covers the GDN o_norm config only (no residual/prenorm/bias)."
                )
            return bi_rms_norm_gated(x, self.weight, g, self.eps, activation=self.activation)
        return rms_norm_gated(
            x=x,
            g=g,
            weight=self.weight,
            bias=self.bias,
            activation=self.activation,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=self.eps,
        )
