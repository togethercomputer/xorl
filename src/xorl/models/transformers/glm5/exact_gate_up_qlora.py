"""Exact active-LoRA value path for the fused GLM-5.2 dense gate/up pair.

SGLang serves the dense MLP's gate and up projections as one block-FP8
projection followed by its stacked-A and fused-B LoRA kernels.  This module
owns that same value program as one mixed-precision-protected leaf.  Its
backward is deliberately separate: the frozen base follows the existing BF16
QLoRA surrogate while the two logical LoRA branches accumulate in FP32.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.ops.block_fp8_native import NativeBlockFP8Linear, _sglang_native_block_fp8_linear_value


GLM52_EXACT_TP1_GATE_UP_QLORA_CONTRACT_VERSION = "glm52_exact_tp1_fused_gate_up_rank1_qlora_v1"


@lru_cache(maxsize=64)
def _single_adapter_gate_up_batch_info(device_index: int, rows: int):
    """Cache the one-live-adapter metadata consumed by the literal kernels."""

    from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415

    device = torch.device("cuda", device_index)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )


class _LogicalRankOneProjection(nn.Module):
    """Hold one logical projection's FP32 masters without a second value path."""

    # Gradient-ownership discovery evaluates the producer on the module that
    # directly registers each factor; it does not inherit the fused parent's
    # declaration while walking child modules.
    adapter_gradient_producer_family = "module_managed"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty((1, in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty((out_features, 1), dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _apply(self, fn, recurse: bool = True):
        # Model-wide BF16 conversion must move these masters but never cast
        # them. The fused parent is an MP-ignored NativeBlockFP8Linear unit.
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.lora_A.device))
        protected = {name: self._parameters.pop(name) for name in ("lora_A", "lora_B")}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                if parameter.is_meta:
                    value = torch.empty_like(parameter, dtype=torch.float32, device=probe.device)
                    replacements[name] = nn.Parameter(value, requires_grad=parameter.requires_grad)
                else:
                    # Preserve the Parameter object so a harmless dtype/device
                    # conversion cannot invalidate optimizer or ownership
                    # references established by the caller.
                    with torch.no_grad():
                        parameter.data = parameter.data.to(device=probe.device, dtype=torch.float32)
                        if parameter.grad is not None:
                            parameter.grad.data = parameter.grad.data.to(device=probe.device, dtype=torch.float32)
                    replacements[name] = parameter
        except Exception:
            self._parameters.update(protected)
            raise
        self._parameters.update(replacements)
        return result


class _Glm52ExactTP1GateUpQLoRAFunction(torch.autograd.Function):
    """Keep literal fused-forward values separate from the hybrid VJP."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        module,
    ) -> Tensor:
        effective_gate_A = gate_A.to(torch.bfloat16).contiguous()
        effective_gate_B = gate_B.to(torch.bfloat16).contiguous()
        effective_up_A = up_A.to(torch.bfloat16).contiguous()
        effective_up_B = up_B.to(torch.bfloat16).contiguous()
        output = module._exact_forward_value(
            input,
            effective_gate_A,
            effective_gate_B,
            effective_up_A,
            effective_up_B,
        )
        expected_shape = (*input.shape[:-1], 2 * module.intermediate_size)
        if output.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact fused gate/up produced {output.dtype}, expected torch.bfloat16")
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"GLM-5.2 exact fused gate/up output shape {tuple(output.shape)} does not match {expected_shape}"
            )
        ctx.module = module
        # Saving both effective values and masters preserves ordinary autograd
        # version-counter checks if an optimizer mutates a factor too early.
        ctx.save_for_backward(
            input.detach(),
            effective_gate_A,
            effective_gate_B,
            effective_up_A,
            effective_up_B,
            gate_A,
            gate_B,
            up_A,
            up_B,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            input,
            effective_gate_A,
            effective_gate_B,
            effective_up_A,
            effective_up_B,
            _gate_A_master,
            _gate_B_master,
            _up_A_master,
            _up_B_master,
        ) = ctx.saved_tensors
        gradients = ctx.module._surrogate_vjp(
            input,
            effective_gate_A,
            effective_gate_B,
            effective_up_A,
            effective_up_B,
            grad_output,
            needs_input_grad=ctx.needs_input_grad[:5],
        )
        return (*gradients, None)


class Glm52ExactTP1FusedGateUpBlockFP8QLoRA(NativeBlockFP8Linear):
    """One literal TP1 dense gate/up block with four logical rank-1 masters.

    The direct base state has fused ``[gate; up]`` row order.  The child names
    intentionally preserve the logical adapter inventory as
    ``gate_proj.lora_A/B`` and ``up_proj.lora_A/B``; neither child implements a
    competing projection forward.
    """

    adapter_gradient_producer_family = "module_managed"
    contract_version = GLM52_EXACT_TP1_GATE_UP_QLORA_CONTRACT_VERSION
    _glm52_exact_active_lora_component = True
    logical_factor_names = (
        "gate_proj.lora_A",
        "gate_proj.lora_B",
        "up_proj.lora_A",
        "up_proj.lora_B",
    )

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        *,
        r: int = 1,
        lora_alpha: int = 1,
        bias: bool = False,
        device: torch.device | str | None = None,
        enable_aqn: bool = False,
        tp_size: int = 1,
    ) -> None:
        if r != 1 or lora_alpha != 1:
            raise ValueError(
                f"GLM-5.2 exact fused gate/up requires rank=1 and alpha=1; received rank={r}, alpha={lora_alpha}"
            )
        if bias:
            raise ValueError("GLM-5.2 exact fused gate/up is bias-free")
        if enable_aqn:
            raise ValueError("GLM-5.2 exact fused gate/up rejects adaptive quantization noise")
        if tp_size != 1:
            raise ValueError(f"GLM-5.2 exact fused gate/up admits only effective TP1, got TP{tp_size}")
        if intermediate_size <= 0 or intermediate_size % 128:
            raise ValueError("GLM-5.2 fused gate/up intermediate_size must be a positive multiple of 128")

        super().__init__(in_features, 2 * intermediate_size, device=device)
        self.intermediate_size = int(intermediate_size)
        self.r = self.active_r = int(r)
        self.lora_alpha = self.active_lora_alpha = int(lora_alpha)
        self.scaling = 1.0
        self.enable_aqn = False
        self.tp_size = 1
        self._exact_gate_up_base_loaded = False
        self.gate_proj = _LogicalRankOneProjection(in_features, intermediate_size, device=device)
        self.up_proj = _LogicalRankOneProjection(in_features, intermediate_size, device=device)

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        if (lora_rank, lora_alpha) != (1, 1):
            raise ValueError(
                "GLM-5.2 exact fused gate/up runtime admits only lora_rank=1 and lora_alpha=1; "
                f"got rank={lora_rank}, alpha={lora_alpha}"
            )
        self.active_r = 1
        self.active_lora_alpha = 1
        self.scaling = 1.0

    @classmethod
    def from_linear(cls, module: nn.Linear) -> Glm52ExactTP1FusedGateUpBlockFP8QLoRA:
        raise RuntimeError(
            "GLM-5.2 exact fused gate/up cannot be built from one linear; "
            "construction must provide the explicit gate/up pair"
        )

    def load_prequantized(self, weight: Tensor, weight_scale_inv: Tensor) -> None:
        raise RuntimeError(
            "GLM-5.2 exact fused gate/up requires load_gate_up_prequantized so gate/up row order is explicit"
        )

    def load_gate_up_prequantized(
        self,
        gate_weight: Tensor,
        gate_weight_scale_inv: Tensor,
        up_weight: Tensor,
        up_weight_scale_inv: Tensor,
    ) -> None:
        """Load separate official tensors into literal SGLang ``[gate; up]`` order."""

        projection_shape = (self.intermediate_size, self.in_features)
        scale_shape = (self.intermediate_size // 128, (self.in_features + 127) // 128)
        for role, weight, scales in (
            ("gate", gate_weight, gate_weight_scale_inv),
            ("up", up_weight, up_weight_scale_inv),
        ):
            if weight.dtype is not torch.float8_e4m3fn:
                raise TypeError(f"{role}_weight must remain float8_e4m3fn, got {weight.dtype}")
            if tuple(weight.shape) != projection_shape:
                raise ValueError(f"{role}_weight shape {tuple(weight.shape)} does not match {projection_shape}")
            if scales.dtype is not torch.float32:
                raise TypeError(f"{role}_weight_scale_inv must remain FP32, got {scales.dtype}")
            if tuple(scales.shape) != scale_shape:
                raise ValueError(f"{role}_weight_scale_inv shape {tuple(scales.shape)} does not match {scale_shape}")
            if weight.device != scales.device:
                raise RuntimeError(f"{role} weight and scales must share a device")
        if gate_weight.device != up_weight.device:
            raise RuntimeError("gate and up prequantized tensors must share a device")

        fused_weight = torch.cat((gate_weight, up_weight), dim=0).contiguous()
        fused_scales = torch.cat((gate_weight_scale_inv, up_weight_scale_inv), dim=0).contiguous()
        NativeBlockFP8Linear.load_prequantized(self, fused_weight, fused_scales)
        self._exact_gate_up_base_loaded = True

    def _validate_engaged_contract(self, input: Tensor) -> None:
        if self.r != 1 or self.active_r != 1 or self.lora_alpha != 1 or self.active_lora_alpha != 1:
            raise RuntimeError("GLM-5.2 exact fused gate/up runtime requires rank=1 and alpha=1")
        if self.scaling != 1.0 or self.tp_size != 1 or self.enable_aqn:
            raise RuntimeError("GLM-5.2 exact fused gate/up runtime contract was mutated")
        expected_factors = {
            "gate_proj.lora_A": (self.gate_proj.lora_A, (1, self.in_features)),
            "gate_proj.lora_B": (self.gate_proj.lora_B, (self.intermediate_size, 1)),
            "up_proj.lora_A": (self.up_proj.lora_A, (1, self.in_features)),
            "up_proj.lora_B": (self.up_proj.lora_B, (self.intermediate_size, 1)),
        }
        for name, (factor, shape) in expected_factors.items():
            if factor.dtype is not torch.float32 or tuple(factor.shape) != shape:
                raise TypeError(f"{name} must remain FP32 with shape {shape}, got {factor.dtype} {tuple(factor.shape)}")
            if factor.device != input.device:
                raise RuntimeError(f"{name} and activations must share one device")
        if self.packed_weight_f32.requires_grad or self.weight_scale_inv.requires_grad:
            raise RuntimeError("GLM-5.2 exact fused gate/up base weights and scales must remain frozen")
        if input.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact fused gate/up requires BF16 activations, got {input.dtype}")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"GLM-5.2 exact fused gate/up input width {input.shape[-1]} does not match {self.in_features}"
            )
        if not input.is_contiguous():
            raise ValueError("GLM-5.2 exact fused gate/up requires contiguous sampler-layout activations")
        if input.numel() == 0:
            raise ValueError("GLM-5.2 exact fused gate/up does not admit an empty TP1 component batch")

    def _exact_forward_value(
        self,
        input: Tensor,
        effective_gate_A: Tensor,
        effective_gate_B: Tensor,
        effective_up_A: Tensor,
        effective_up_B: Tensor,
    ) -> Tensor:
        """Run one fused W8A8 base, stacked-A, then fused gate/up B-add."""

        if input.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact fused gate/up forward requires CUDA and pinned SGLang kernels")
        if self.packed_weight_f32.device != input.device or self.weight_scale_inv.device != input.device:
            raise RuntimeError("GLM-5.2 exact fused gate/up base state and activations must share one CUDA device")
        effective_factors = (effective_gate_A, effective_gate_B, effective_up_A, effective_up_B)
        if any(factor.device != input.device for factor in effective_factors):
            raise RuntimeError("GLM-5.2 exact fused gate/up factors and activations must share one CUDA device")
        if any(factor.dtype is not torch.bfloat16 for factor in effective_factors):
            raise TypeError("GLM-5.2 exact fused gate/up effective factors must be BF16")

        try:
            from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang fused gate/up LoRA kernels are required") from exc

        rows = input.numel() // self.in_features
        input_2d = input.view(rows, self.in_features)
        base_output = _sglang_native_block_fp8_linear_value(
            input_2d,
            self.fp8_weight().contiguous(),
            self.weight_scale_inv.contiguous(),
            block_size=self.block_size,
        )
        # SGLang normalizes separate logical tensors into these exact physical
        # buffer orders: [gate_A; up_A] and [gate_B; up_B].
        stacked_A = torch.cat((effective_gate_A, effective_up_A), dim=0).unsqueeze(0).contiguous()
        stacked_B = torch.cat((effective_gate_B, effective_up_B), dim=0).unsqueeze(0).contiguous()
        batch_info = _single_adapter_gate_up_batch_info(input.device.index, rows)
        lora_a_output = sgemm_lora_a_fwd(input_2d, stacked_A, batch_info, stack_num=2)
        output = gate_up_lora_b_fwd(
            lora_a_output,
            stacked_B,
            batch_info,
            self.intermediate_size,
            base_output=base_output,
        )
        return output.view(*input.shape[:-1], 2 * self.intermediate_size)

    def _dequantize_base_weight(self) -> Tensor:
        return NativeBlockFP8Linear.forward(self, return_dequantized_weight=True)

    def _surrogate_vjp(
        self,
        input: Tensor,
        effective_gate_A: Tensor,
        effective_gate_B: Tensor,
        effective_up_A: Tensor,
        effective_up_B: Tensor,
        grad_output: Tensor,
        *,
        needs_input_grad: tuple[bool, bool, bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Evaluate the BF16-base/FP32-logical-factor surrogate VJP."""

        need_input, need_gate_A, need_gate_B, need_up_A, need_up_B = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None, None, None

        with torch.enable_grad(), torch.autocast(device_type=input.device.type, enabled=False):
            gate_base_grad_input = None
            up_base_grad_input = None
            if need_input:
                base_weight = self._dequantize_base_weight().to(input.dtype)
                gate_weight, up_weight = base_weight.split(self.intermediate_size, dim=0)
                gate_grad_output, up_grad_output = grad_output.split(self.intermediate_size, dim=-1)
                gate_base_input = input.detach().requires_grad_(True)
                up_base_input = input.detach().requires_grad_(True)
                gate_base_output = F.linear(gate_base_input, gate_weight)
                up_base_output = F.linear(up_base_input, up_weight)
                gate_base_grad_input, up_base_grad_input = torch.autograd.grad(
                    (gate_base_output, up_base_output),
                    (gate_base_input, up_base_input),
                    grad_outputs=(
                        gate_grad_output.to(gate_base_output.dtype),
                        up_grad_output.to(up_base_output.dtype),
                    ),
                )
            reference_gate_input = input.float().detach().requires_grad_(need_input)
            reference_up_input = input.float().detach().requires_grad_(need_input)
            reference_gate_A = effective_gate_A.float().detach().requires_grad_(need_gate_A)
            reference_gate_B = effective_gate_B.float().detach().requires_grad_(need_gate_B)
            reference_up_A = effective_up_A.float().detach().requires_grad_(need_up_A)
            reference_up_B = effective_up_B.float().detach().requires_grad_(need_up_B)
            gate_output = F.linear(F.linear(reference_gate_input, reference_gate_A), reference_gate_B)
            up_output = F.linear(F.linear(reference_up_input, reference_up_A), reference_up_B)
            lora_output = torch.cat((gate_output, up_output), dim=-1)

            requested = []
            labels = []
            for label, required, value in (
                ("gate_input", need_input, reference_gate_input),
                ("up_input", need_input, reference_up_input),
                ("gate_A", need_gate_A, reference_gate_A),
                ("gate_B", need_gate_B, reference_gate_B),
                ("up_A", need_up_A, reference_up_A),
                ("up_B", need_up_B, reference_up_B),
            ):
                if required:
                    labels.append(label)
                    requested.append(value)
            gradients = torch.autograd.grad(
                lora_output,
                requested,
                grad_outputs=grad_output.float(),
                allow_unused=False,
            )

        by_label = dict(zip(labels, gradients, strict=True))
        grad_input = None
        if need_input:
            gate_grad_input = gate_base_grad_input.float() + by_label["gate_input"].float()
            up_grad_input = up_base_grad_input.float() + by_label["up_input"].float()
            # The trainer has two logical projections feeding one BF16
            # activation. Preserve both storage-boundary casts and their
            # gate-then-up accumulation even though the sampler value path is
            # one fused projection.
            grad_input = gate_grad_input.to(input.dtype) + up_grad_input.to(input.dtype)
        return (
            grad_input,
            by_label.get("gate_A"),
            by_label.get("gate_B"),
            by_label.get("up_A"),
            by_label.get("up_B"),
        )

    def forward_partition(self, *args, **kwargs) -> Tensor:
        raise RuntimeError("GLM-5.2 exact fused gate/up cannot bypass active LoRA through a base-only partition")

    def forward(self, input: Tensor) -> Tensor:
        self._validate_engaged_contract(input)
        return _Glm52ExactTP1GateUpQLoRAFunction.apply(
            input,
            self.gate_proj.lora_A,
            self.gate_proj.lora_B,
            self.up_proj.lora_A,
            self.up_proj.lora_B,
            self,
        )


__all__ = [
    "GLM52_EXACT_TP1_GATE_UP_QLORA_CONTRACT_VERSION",
    "Glm52ExactTP1FusedGateUpBlockFP8QLoRA",
]
