# Adapted from flash-linear-attention/fla/modules/short_conv.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.ops.linear_attention.ops.cp import FLACPContext, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd


def _activate_tensor(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in {None, "identity"}:
        return x
    if activation in {"silu", "swish"}:
        return F.silu(x)
    raise ValueError(f"Unsupported activation: {activation}")


def _depthwise_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    x_t = x.transpose(1, 2)
    y = F.conv1d(x_t, weight, bias, padding=weight.shape[-1] - 1, groups=weight.shape[0])
    y = y[:, :, : x.shape[1]]
    return _activate_tensor(y.transpose(1, 2), activation)


def _make_tail_buffer(x_2d: torch.Tensor, prefix_width: int) -> torch.Tensor:
    tails = x_2d.new_zeros(prefix_width, x_2d.shape[-1])
    if prefix_width == 0:
        return tails
    copy_len = min(x_2d.shape[0], prefix_width)
    if copy_len > 0:
        tails[-copy_len:] = x_2d[-copy_len:]
    return tails


def _prepare_cp_prefix(
    x: torch.Tensor,
    cp_context: FLACPContext,
    kernel_size: int,
) -> torch.Tensor:
    prefix_width = kernel_size - 1
    x_2d = x.squeeze(0)
    prefix = x_2d.new_zeros(prefix_width, x_2d.shape[-1])
    if prefix_width <= 0:
        return prefix

    tails = _make_tail_buffer(x_2d, prefix_width)
    heads = conv_cp_send_recv_fwd(tails.contiguous(), cp_context.group)
    if not cp_context.is_first_rank:
        valid_len = min(prefix_width, cp_context.pre_num_conv_tokens or 0)
        if valid_len > 0:
            prefix[-valid_len:] = heads[-valid_len:]
    return prefix


def _cp_short_conv_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor,
    prefix: torch.Tensor,
) -> torch.Tensor:
    if x.shape[0] != 1:
        raise ValueError(f"CP short conv requires packed input with batch size 1, got {x.shape}.")

    prefix_width = prefix.shape[0]
    zero_prefix = prefix.new_zeros(prefix.shape) if prefix_width > 0 else prefix
    outputs: list[torch.Tensor] = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)):
        seq = x[:, start:end]
        if prefix_width > 0:
            seq_prefix = prefix if seq_idx == 0 else zero_prefix
            ext = torch.cat((seq_prefix.unsqueeze(0), seq), dim=1)
            y = _depthwise_causal_conv(ext, weight, bias, activation)
            y = y[:, prefix_width:, :]
        else:
            y = _depthwise_causal_conv(seq, weight, bias, activation)
        outputs.append(y)
    return torch.cat(outputs, dim=1) if outputs else x.new_zeros(x.shape)


class _ShortConvolutionCPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        cu_seqlens: torch.Tensor,
        cp_context: FLACPContext,
    ) -> torch.Tensor:
        prefix = _prepare_cp_prefix(x=x, cp_context=cp_context, kernel_size=weight.shape[-1])
        y = _cp_short_conv_forward(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            cu_seqlens=cu_seqlens,
            prefix=prefix,
        )

        bias_tensor = bias if bias is not None else x.new_empty(0, dtype=x.dtype)
        ctx.save_for_backward(x, weight, bias_tensor, prefix, cu_seqlens)
        ctx.has_bias = bias is not None
        ctx.activation = activation
        ctx.cp_context = cp_context.copy_for_backward()
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias_tensor, prefix, cu_seqlens = ctx.saved_tensors

        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            weight_req = weight.detach().requires_grad_(True)
            inputs: list[torch.Tensor] = [x_req, weight_req]

            if ctx.has_bias:
                bias_req = bias_tensor.detach().requires_grad_(True)
                inputs.append(bias_req)
            else:
                bias_req = None

            if prefix.numel() > 0:
                prefix_req = prefix.detach().requires_grad_(True)
                inputs.append(prefix_req)
            else:
                prefix_req = prefix

            y = _cp_short_conv_forward(
                x=x_req,
                weight=weight_req,
                bias=bias_req,
                activation=ctx.activation,
                cu_seqlens=cu_seqlens,
                prefix=prefix_req,
            )
            grads = torch.autograd.grad(y, inputs, grad_output, allow_unused=False)

        dx = grads[0]
        dw = grads[1]
        grad_idx = 2
        db = grads[grad_idx] if ctx.has_bias else None
        if ctx.has_bias:
            grad_idx += 1
        if prefix.numel() > 0:
            dprefix = grads[grad_idx]
            recv_dprefix = conv_cp_send_recv_bwd(dprefix.contiguous(), ctx.cp_context.group)
            tail_len = min(dx.shape[1], prefix.shape[0])
            if tail_len > 0:
                dx[:, -tail_len:, :].add_(recv_dprefix[-tail_len:].unsqueeze(0))

        return dx, dw, db, None, None, None


class ShortConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=0,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return _activate_tensor(x, self.activation)

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        return _depthwise_causal_conv(x, self.weight, self.bias, self.activation)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **_: object,
    ):
        del chunk_indices
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        if cp_context is not None:
            if cache is not None:
                raise ValueError("CP short conv does not support cache/input state yet.")
            if output_final_state:
                raise ValueError("CP short conv does not support output_final_state yet.")
            if cp_context.group is None or cp_context.cu_seqlens is None:
                raise ValueError("CP short conv requires a fully initialized cp_context.")
            y = _ShortConvolutionCPFunction.apply(
                x,
                self.weight,
                self.bias,
                self.activation,
                cp_context.cu_seqlens,
                cp_context,
            )
            if residual is not None:
                y = y + residual
            return y, cache

        if cu_seqlens is None:
            y = self._conv(x)
            if residual is not None:
                y = y + residual
            final_state = None
            if output_final_state:
                width = self.kernel_size[0]
                final_state = x.new_zeros(x.shape[0], x.shape[-1], width)
                tail = x[:, -width:, :].transpose(1, 2)
                final_state[:, :, -tail.shape[-1] :] = tail
            return y, final_state if output_final_state else cache

        if x.shape[0] != 1:
            raise ValueError("Packed varlen path expects batch size 1.")

        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
            seq = x[:, start:end]
            outputs.append(self._conv(seq))
        y = torch.cat(outputs, dim=1) if outputs else x.new_zeros(x.shape)
        if residual is not None:
            y = y + residual
        return y, cache
