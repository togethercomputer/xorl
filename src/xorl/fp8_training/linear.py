"""Trainable block-FP8 linear layer.

This module implements FP8 compute for full-weight training while retaining
ordinary BF16/FP32 master parameters. The optimizer and checkpoints still see
``weight`` and ``bias`` parameters; the forward/backward compute path quantizes
operands to block FP8 and dispatches the existing Triton FP8 GEMM.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


FP8BackwardMode = Literal["bf16", "fp8"]
FP8CorrectionMode = Literal["none", "activation", "activation2", "weight", "first_order", "full"]
FP8OutputDType = Literal["input", "float32"]
_VALID_BACKWARD_MODES: set[str] = {"bf16", "fp8"}
_VALID_CORRECTION_MODES: set[str] = {"none", "activation", "activation2", "weight", "first_order", "full"}
_VALID_OUTPUT_DTYPES: set[str] = {"input", "float32"}


def _validate_backward_mode(mode: str) -> FP8BackwardMode:
    if mode not in _VALID_BACKWARD_MODES:
        raise ValueError(f"FP8 backward mode must be one of {_VALID_BACKWARD_MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def _validate_output_dtype(output_dtype: str) -> FP8OutputDType:
    if output_dtype not in _VALID_OUTPUT_DTYPES:
        raise ValueError(f"FP8 output dtype must be one of {_VALID_OUTPUT_DTYPES}, got {output_dtype!r}")
    return output_dtype  # type: ignore[return-value]


def _validate_correction_mode(mode: str) -> FP8CorrectionMode:
    if mode not in _VALID_CORRECTION_MODES:
        raise ValueError(f"FP8 correction mode must be one of {_VALID_CORRECTION_MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def _pad_last_dim(x: Tensor, multiple: int) -> Tensor:
    pad = (-x.shape[-1]) % multiple
    if pad == 0:
        return x.contiguous()
    return F.pad(x, (0, pad)).contiguous()


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = []
    next_stride = 1
    for size in reversed(shape):
        stride.append(next_stride)
        next_stride *= int(size)
    return tuple(reversed(stride))


def _validate_smoothquant_alpha(alpha: float | None) -> float | None:
    if alpha is None:
        return None
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"FP8 SmoothQuant alpha must be in [0, 1], got {alpha}")
    return alpha


def _validate_amax_scale(scale: float) -> float:
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError(f"FP8 amax scale must be positive, got {scale}")
    return scale


def _apply_smoothquant(a: Tensor, b: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    a_absmax = a.abs().amax(dim=0).clamp_min(1e-12)
    b_absmax = b.abs().amax(dim=0).clamp_min(1e-12)
    smooth = (a_absmax.pow(alpha) / b_absmax.pow(1.0 - alpha)).clamp(1e-6, 1e6)
    return a / smooth, b * smooth


def _fp8_matmul(
    a: Tensor,
    b: Tensor,
    block_size: int,
    smoothquant_alpha: float | None = None,
    activation_amax_scale: float = 1.0,
    weight_amax_scale: float = 1.0,
    correction_mode: FP8CorrectionMode = "none",
) -> Tensor:
    """Compute ``a @ b.T`` with block-FP8 quantized operands.

    ``block_fp8_quantize`` requires the contracting dimension to be divisible by
    ``block_size``. Padding the last dimension is mathematically exact because
    both operands are padded with zeros along the same contraction axis.
    """
    from xorl.ops.quantize import (  # noqa: PLC0415
        block_fp8_dequantize,
        block_fp8_dequantize_gkn_rowwise,
        block_fp8_gemm,
        block_fp8_quantize,
        block_fp8_quantize_gkn_rowwise,
    )

    if a.dim() < 1 or b.dim() != 2:
        raise RuntimeError(f"FP8 matmul expects a rank>=1 lhs and rank-2 rhs, got {a.shape=} {b.shape=}")
    if a.shape[-1] != b.shape[-1]:
        raise RuntimeError(f"FP8 matmul contraction mismatch: {a.shape[-1]} != {b.shape[-1]}")
    correction_mode = _validate_correction_mode(correction_mode)

    out_shape = (*a.shape[:-1], b.shape[0])
    a_2d = a.reshape(-1, a.shape[-1])
    a_float = a_2d.float()
    b_float = b.float()
    if smoothquant_alpha is not None:
        a_float, b_float = _apply_smoothquant(a_float, b_float, smoothquant_alpha)
    a_padded = _pad_last_dim(a_float, block_size)
    b_padded = _pad_last_dim(b_float, block_size)

    a_fp8, a_scales = block_fp8_quantize(
        a_padded,
        block_size=block_size,
        amax_scale=activation_amax_scale,
    )
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(
        b_padded,
        block_size=block_size,
        amax_scale=weight_amax_scale,
    )
    out = block_fp8_gemm(
        a_fp8,
        a_scales,
        b_fp8,
        b_scales,
        block_size=block_size,
        weight_scale_layout="row",
        backend="auto",
    )
    if correction_mode != "none":
        a_dequant = None
        b_dequant = None
        a_res_fp8 = None
        a_res_scales = None
        b_res_fp8 = None
        b_res_scales = None

        if correction_mode in {"activation", "activation2", "first_order", "full"}:
            a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=block_size)
            a_residual = (a_padded - a_dequant).contiguous()
            a_res_fp8, a_res_scales = block_fp8_quantize(
                a_residual,
                block_size=block_size,
                amax_scale=activation_amax_scale,
            )
            out = out + block_fp8_gemm(
                a_res_fp8,
                a_res_scales,
                b_fp8,
                b_scales,
                block_size=block_size,
                weight_scale_layout="row",
                backend="auto",
            )
            if correction_mode == "activation2":
                a_res_dequant = block_fp8_dequantize(a_res_fp8, a_res_scales, block_size=block_size)
                a_second_residual = (a_residual - a_res_dequant).contiguous()
                a_second_res_fp8, a_second_res_scales = block_fp8_quantize(
                    a_second_residual,
                    block_size=block_size,
                    amax_scale=activation_amax_scale,
                )
                out = out + block_fp8_gemm(
                    a_second_res_fp8,
                    a_second_res_scales,
                    b_fp8,
                    b_scales,
                    block_size=block_size,
                    weight_scale_layout="row",
                    backend="auto",
                )

        if correction_mode in {"weight", "first_order", "full"}:
            b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=block_size)
            b_residual = (b_padded - b_dequant).contiguous()
            b_res_fp8, b_res_scales = block_fp8_quantize_gkn_rowwise(
                b_residual,
                block_size=block_size,
                amax_scale=weight_amax_scale,
            )
            out = out + block_fp8_gemm(
                a_fp8,
                a_scales,
                b_res_fp8,
                b_res_scales,
                block_size=block_size,
                weight_scale_layout="row",
                backend="auto",
            )

        if correction_mode == "full":
            if a_res_fp8 is None or a_res_scales is None:
                a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=block_size)
                a_residual = (a_padded - a_dequant).contiguous()
                a_res_fp8, a_res_scales = block_fp8_quantize(
                    a_residual,
                    block_size=block_size,
                    amax_scale=activation_amax_scale,
                )
            if b_res_fp8 is None or b_res_scales is None:
                b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=block_size)
                b_residual = (b_padded - b_dequant).contiguous()
                b_res_fp8, b_res_scales = block_fp8_quantize_gkn_rowwise(
                    b_residual,
                    block_size=block_size,
                    amax_scale=weight_amax_scale,
                )
            out = out + block_fp8_gemm(
                a_res_fp8,
                a_res_scales,
                b_res_fp8,
                b_res_scales,
                block_size=block_size,
                weight_scale_layout="row",
                backend="auto",
            )
    return out.reshape(out_shape)


class _FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        block_size: int,
        backward_mode: str,
        smoothquant_alpha: float | None,
        activation_amax_scale: float,
        weight_amax_scale: float,
        correction_mode: str,
        output_dtype: str,
    ) -> Tensor:
        backward_mode = _validate_backward_mode(backward_mode)
        smoothquant_alpha = _validate_smoothquant_alpha(smoothquant_alpha)
        activation_amax_scale = _validate_amax_scale(activation_amax_scale)
        weight_amax_scale = _validate_amax_scale(weight_amax_scale)
        correction_mode = _validate_correction_mode(correction_mode)
        output_dtype = _validate_output_dtype(output_dtype)
        ctx.block_size = block_size
        ctx.backward_mode = backward_mode
        ctx.smoothquant_alpha = smoothquant_alpha
        ctx.activation_amax_scale = activation_amax_scale
        ctx.weight_amax_scale = weight_amax_scale
        ctx.correction_mode = correction_mode
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.save_for_backward(x, weight)

        out = _fp8_matmul(
            x,
            weight,
            block_size,
            smoothquant_alpha=smoothquant_alpha,
            activation_amax_scale=activation_amax_scale,
            weight_amax_scale=weight_amax_scale,
            correction_mode=correction_mode,
        )
        if output_dtype == "input":
            out = out.to(x.dtype)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weight = ctx.saved_tensors
        block_size = ctx.block_size
        backward_mode = ctx.backward_mode
        smoothquant_alpha = ctx.smoothquant_alpha
        activation_amax_scale = ctx.activation_amax_scale
        weight_amax_scale = ctx.weight_amax_scale
        correction_mode = ctx.correction_mode

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        grad_input = None
        grad_weight = None
        grad_bias = None

        use_fp8_backward = backward_mode == "fp8" and grad_output.is_cuda and x.is_cuda and weight.is_cuda

        if ctx.needs_input_grad[0]:
            if use_fp8_backward:
                grad_input_2d = _fp8_matmul(
                    grad_output_2d,
                    weight.t().contiguous(),
                    block_size,
                    smoothquant_alpha=smoothquant_alpha,
                    activation_amax_scale=activation_amax_scale,
                    weight_amax_scale=weight_amax_scale,
                    correction_mode=correction_mode,
                )
            else:
                grad_input_2d = grad_output_2d.float().matmul(weight.float())
            grad_input = grad_input_2d.reshape_as(x).to(x.dtype)

        if ctx.needs_input_grad[1]:
            if use_fp8_backward:
                grad_weight = _fp8_matmul(
                    grad_output_2d.t().contiguous(),
                    x_2d.t().contiguous(),
                    block_size,
                    smoothquant_alpha=smoothquant_alpha,
                    activation_amax_scale=activation_amax_scale,
                    weight_amax_scale=weight_amax_scale,
                    correction_mode=correction_mode,
                )
            else:
                grad_weight = grad_output_2d.t().float().matmul(x_2d.float())
            grad_weight = grad_weight.to(weight.dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0).to(ctx.bias_dtype)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class FP8Linear(nn.Linear):
    """Drop-in ``nn.Linear`` replacement with block-FP8 compute.

    The layer stores normal trainable master parameters and quantizes operands
    just-in-time for GEMM. This is a compute-mode replacement, not a quantized
    checkpoint format.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        block_size: int = 128,
        backward_mode: FP8BackwardMode = "fp8",
        smoothquant_alpha: float | None = None,
        activation_amax_scale: float = 1.0,
        weight_amax_scale: float = 1.0,
        correction_mode: FP8CorrectionMode = "none",
        output_dtype: FP8OutputDType = "input",
        allow_bf16_fallback: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if block_size <= 0:
            raise ValueError(f"FP8 block_size must be positive, got {block_size}")
        self.fp8_block_size = int(block_size)
        self.fp8_backward_mode = _validate_backward_mode(backward_mode)
        self.fp8_smoothquant_alpha = _validate_smoothquant_alpha(smoothquant_alpha)
        self.fp8_activation_amax_scale = _validate_amax_scale(activation_amax_scale)
        self.fp8_weight_amax_scale = _validate_amax_scale(weight_amax_scale)
        self.fp8_correction_mode = _validate_correction_mode(correction_mode)
        self.fp8_output_dtype = _validate_output_dtype(output_dtype)
        self.fp8_allow_bf16_fallback = allow_bf16_fallback
        self.fp8_module_name: str | None = None
        self.last_forward_used_fp8 = False

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        block_size: int = 128,
        backward_mode: FP8BackwardMode = "fp8",
        smoothquant_alpha: float | None = None,
        activation_amax_scale: float = 1.0,
        weight_amax_scale: float = 1.0,
        correction_mode: FP8CorrectionMode = "none",
        output_dtype: FP8OutputDType = "input",
        allow_bf16_fallback: bool = True,
    ) -> "FP8Linear":
        fp8 = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            block_size=block_size,
            backward_mode=backward_mode,
            smoothquant_alpha=smoothquant_alpha,
            activation_amax_scale=activation_amax_scale,
            weight_amax_scale=weight_amax_scale,
            correction_mode=correction_mode,
            output_dtype=output_dtype,
            allow_bf16_fallback=allow_bf16_fallback,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        fp8.weight = module.weight
        fp8.bias = module.bias
        return fp8

    def _can_use_fp8_with_weight(self, x: Tensor, weight: Tensor) -> bool:
        return (
            x.is_cuda
            and weight.is_cuda
            and x.is_floating_point()
            and weight.is_floating_point()
            and x.shape[-1] == self.in_features
        )

    def _can_use_local_fp8(self, x: Tensor, weight: Tensor) -> bool:
        return (
            x.is_cuda
            and weight.is_cuda
            and x.is_floating_point()
            and weight.is_floating_point()
            and x.shape[-1] == weight.shape[-1]
        )

    def _forward_local_impl(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self._can_use_local_fp8(x, weight):
            self.last_forward_used_fp8 = True
            out = _FP8LinearFunction.apply(
                x,
                weight,
                bias,
                self.fp8_block_size,
                self.fp8_backward_mode,
                self.fp8_smoothquant_alpha,
                self.fp8_activation_amax_scale,
                self.fp8_weight_amax_scale,
                self.fp8_correction_mode,
                self.fp8_output_dtype,
            )
            from xorl.fp8_training.profiler import (  # noqa: PLC0415
                linear_error_profiling_enabled,
                record_linear_error,
            )

            if linear_error_profiling_enabled():
                record_linear_error(self, x, out)
            return out

        self.last_forward_used_fp8 = False
        if not self.fp8_allow_bf16_fallback:
            raise RuntimeError(
                "FP8Linear cannot use FP8 compute for this input. "
                f"input_device={x.device}, weight_device={weight.device}, "
                f"input_shape={tuple(x.shape)}, weight_shape={tuple(weight.shape)}, in_features={self.in_features}"
            )
        out = F.linear(x, weight, bias)
        if self.fp8_output_dtype == "float32":
            out = out.float()
        return out

    def _forward_dtensor_impl(self, x: Tensor, weight: DTensor, bias: Tensor | None) -> Tensor:
        input_is_dtensor = isinstance(x, DTensor)
        x_local = x.to_local() if input_is_dtensor else x
        weight_local = weight.to_local()
        bias_is_dtensor = isinstance(bias, DTensor)

        placements = tuple(weight.placements)
        shard_dim = next((placement.dim for placement in placements if isinstance(placement, Shard)), None)
        mesh = weight.device_mesh
        output_shape = torch.Size((*x.shape[:-1], weight.shape[0]))
        output_stride = _contiguous_stride(output_shape)

        if shard_dim == 1:
            local_out = self._forward_local_impl(x_local, weight_local, None)
            out = DTensor.from_local(
                local_out,
                mesh,
                [Partial()],
                run_check=False,
                shape=output_shape,
                stride=output_stride,
            )
            if bias is not None:
                out = out.redistribute(placements=[Replicate()], async_op=True)
                bias_dtensor = bias if bias_is_dtensor else DTensor.from_local(
                    bias,
                    mesh,
                    [Replicate()],
                    run_check=False,
                    shape=torch.Size([weight.shape[0]]),
                    stride=(1,),
                )
                out = out + bias_dtensor
            return out

        local_bias = bias.to_local() if bias_is_dtensor else bias
        local_out = self._forward_local_impl(x_local, weight_local, local_bias)
        if shard_dim == 0:
            placement = Shard(local_out.ndim - 1)
        else:
            placement = Replicate()
        return DTensor.from_local(
            local_out,
            mesh,
            [placement],
            run_check=False,
            shape=output_shape,
            stride=output_stride,
        )

    def _forward_impl(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if isinstance(weight, DTensor):
            return self._forward_dtensor_impl(x, weight, bias)

        if self._can_use_fp8_with_weight(x, weight):
            return self._forward_local_impl(x, weight, bias)

        return self._forward_local_impl(x, weight, bias)

    def forward_with_weight(self, x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        """Run this FP8 recipe with an explicit local weight shard.

        Tensor-parallel loss code can call this with ``lm_head.weight.to_local()``
        so the FP8 module recipe is used without handing DTensor operands to the
        block-FP8 kernels.
        """

        return self._forward_impl(x, weight, bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, fp8_block_size={self.fp8_block_size}, "
            f"fp8_backward_mode={self.fp8_backward_mode}, "
            f"fp8_smoothquant_alpha={self.fp8_smoothquant_alpha}, "
            f"fp8_activation_amax_scale={self.fp8_activation_amax_scale}, "
            f"fp8_weight_amax_scale={self.fp8_weight_amax_scale}, "
            f"fp8_correction_mode={self.fp8_correction_mode}, "
            f"fp8_output_dtype={self.fp8_output_dtype}, "
            f"fp8_allow_bf16_fallback={self.fp8_allow_bf16_fallback}"
        )
