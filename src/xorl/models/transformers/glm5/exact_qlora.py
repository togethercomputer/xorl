"""Exact-value rank-1 QLoRA primitives for the canonical GLM-5.2 trainer.

The forward below deliberately uses the same public SGLang block-FP8 and
dynamic-LoRA kernels as serving.  Autograd is supplied separately by the
existing differentiable QLoRA program, evaluated on the effective BF16 factor
bytes consumed by the exact forward.  The resulting backward is a validated
straight-through surrogate; it is not the derivative of FP8 quantization.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.ops.block_fp8_native import _sglang_native_block_fp8_linear_value
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear


GLM52_EXACT_TP1_QLORA_CONTRACT_VERSION = "glm52_exact_tp1_rank1_qlora_v1"


@lru_cache(maxsize=64)
def _single_adapter_batch_info(device_index: int, rows: int):
    """Cache immutable production-graph-shaped metadata across all TP1 layers."""

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


class _Glm52ExactTP1QLoRAFunction(torch.autograd.Function):
    """Own exact forward values while delegating only the VJP to QLoRA."""

    @staticmethod
    def forward(ctx, input: Tensor, lora_A: Tensor, lora_B: Tensor, module) -> Tensor:
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = lora_B.to(torch.bfloat16).contiguous()
        output = module._exact_forward_value(input, effective_A, effective_B)
        if output.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact TP1 QLoRA produced {output.dtype}, expected torch.bfloat16")
        if tuple(output.shape) != (*input.shape[:-1], module.out_features):
            raise RuntimeError(
                "GLM-5.2 exact TP1 QLoRA output shape mismatch: "
                f"got {tuple(output.shape)}, expected {(*input.shape[:-1], module.out_features)}"
            )
        ctx.module = module
        # Saving the masters as well as their effective BF16 values gives
        # autograd its normal version-counter protection against mutation
        # between the exact forward and surrogate backward.
        ctx.save_for_backward(input.detach(), effective_A, effective_B, lora_A, lora_B)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, effective_A, effective_B, _lora_A_master, _lora_B_master = ctx.saved_tensors
        grad_input, grad_A, grad_B = ctx.module._surrogate_vjp(
            input,
            effective_A,
            effective_B,
            grad_output,
            needs_input_grad=ctx.needs_input_grad[:3],
        )
        return grad_input, grad_A, grad_B, None


class Glm52ExactTP1BlockFP8QLoRALinear(BlockFP8QLoRALinear):
    """Canonical GLM-5.2 TP1 block-FP8 linear with active rank-1 LoRA.

    This wrapper is intentionally narrower than :class:`BlockFP8QLoRALinear`.
    It admits only the rank-1/alpha-1, bias-free, non-AQN tuple certified by the
    exact active-LoRA contract.  TP-sharded shared experts and the LM head have
    different physical programs and must use their own wrappers.
    """

    contract_version = GLM52_EXACT_TP1_QLORA_CONTRACT_VERSION
    _glm52_exact_active_lora_component = True
    # The exact contract keeps the trainable masters in FP32 and performs its
    # single BF16 factor rounding explicitly inside the value path. Decoder
    # FSDP mixed precision must therefore own this module as a separate unit
    # without a parameter-cast policy.
    fsdp_requires_full_precision = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 1,
        lora_alpha: int = 1,
        bias: bool = False,
        device: Optional[torch.device] = None,
        enable_aqn: bool = False,
        aqn_alpha: float = 1.0,
    ) -> None:
        if r != 1 or lora_alpha != 1:
            raise ValueError(
                f"GLM-5.2 exact TP1 QLoRA requires rank=1 and alpha=1; received rank={r}, alpha={lora_alpha}"
            )
        if bias:
            raise ValueError("GLM-5.2 exact TP1 block-FP8 projections are bias-free")
        if enable_aqn:
            raise ValueError("GLM-5.2 exact TP1 QLoRA rejects adaptive quantization noise")
        super().__init__(
            in_features,
            out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=False,
            device=device,
            enable_aqn=False,
            aqn_alpha=aqn_alpha,
        )

    def _apply(self, fn, recurse: bool = True):
        # Preserve exact packed bytes and FP32 LoRA masters across a model-wide
        # dtype conversion. Non-meta Parameters retain object identity so
        # optimizer and ownership references cannot be invalidated.
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.lora_A.device))
        protected_names = ("lora_A", "lora_B", "packed_weight_f32")
        protected = {name: self._parameters.pop(name) for name in protected_names}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                if parameter.is_meta:
                    value = torch.empty_like(parameter, dtype=torch.float32, device=probe.device)
                    replacements[name] = nn.Parameter(value, requires_grad=parameter.requires_grad)
                else:
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

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        if (lora_rank, lora_alpha) != (1, 1):
            raise ValueError(
                "GLM-5.2 exact TP1 QLoRA runtime admits only lora_rank=1 and lora_alpha=1; "
                f"got rank={lora_rank}, alpha={lora_alpha}"
            )
        self.active_r = 1
        self.active_lora_alpha = 1
        self.scaling = 1.0

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        r: int = 1,
        lora_alpha: int = 1,
        **kwargs,
    ) -> Glm52ExactTP1BlockFP8QLoRALinear:
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
        replacement = cls(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=module.bias is not None,
            device=module.weight.device,
            **kwargs,
        )
        replacement._quantize_and_store(module.weight.detach())
        return replacement

    def _validate_engaged_contract(self, input: Tensor) -> None:
        if self.active_r != 1 or self.active_lora_alpha != 1 or self._active_scaling() != 1.0:
            raise RuntimeError("GLM-5.2 exact TP1 QLoRA runtime requires active rank=1, alpha=1, scaling=1")
        if self.lora_A.dtype is not torch.float32 or self.lora_B.dtype is not torch.float32:
            raise TypeError("GLM-5.2 exact TP1 QLoRA master factors must remain FP32")
        if self.enable_aqn:
            raise RuntimeError("GLM-5.2 exact TP1 QLoRA cannot engage AQN")
        if input.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact TP1 QLoRA requires BF16 activations, got {input.dtype}")
        if input.shape[-1] != self.in_features:
            raise ValueError(f"GLM-5.2 exact TP1 QLoRA input width {input.shape[-1]} does not match {self.in_features}")
        if not input.is_contiguous():
            raise ValueError("GLM-5.2 exact TP1 QLoRA requires contiguous sampler-layout activations")
        if input.numel() == 0:
            raise ValueError("GLM-5.2 exact TP1 QLoRA does not admit an empty TP1 component batch")

    def _exact_forward_value(self, input: Tensor, effective_A: Tensor, effective_B: Tensor) -> Tensor:
        """Run the literal SGLang base, A-store, B-store, and fused-add order."""

        if input.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact TP1 QLoRA forward requires CUDA and pinned SGLang kernels")
        if self.packed_weight_f32.requires_grad:
            raise RuntimeError("GLM-5.2 exact TP1 QLoRA base weights must remain frozen")
        if self.packed_weight_f32.device != input.device or self.weight_block_scales.device != input.device:
            raise RuntimeError("GLM-5.2 exact TP1 QLoRA state and activations must share one CUDA device")
        if effective_A.device != input.device or effective_B.device != input.device:
            raise RuntimeError("GLM-5.2 exact TP1 QLoRA factors and activations must share one CUDA device")
        if effective_A.dtype is not torch.bfloat16 or effective_B.dtype is not torch.bfloat16:
            raise TypeError("GLM-5.2 exact TP1 QLoRA effective factors must be BF16")

        try:
            from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang FP8 and LoRA kernels are required") from exc

        rows = input.numel() // self.in_features
        input_2d = input.view(rows, self.in_features)
        weight = (
            self._read_packed_weight_uint8()
            .view(torch.float8_e4m3fn)
            .reshape(self.out_features, self.in_features)
            .contiguous()
        )
        scales = self._recover_tensor(
            self.weight_block_scales,
            self._scale_dtypes["weight_block_scales"],
        ).contiguous()
        base_output = _sglang_native_block_fp8_linear_value(
            input_2d,
            weight,
            scales,
        )

        batch_info = _single_adapter_batch_info(input.device.index, rows)
        lora_a_output = sgemm_lora_a_fwd(
            input_2d,
            effective_A.unsqueeze(0),
            batch_info,
        )
        output = sgemm_lora_b_fwd(
            lora_a_output,
            effective_B.unsqueeze(0),
            batch_info,
            base_output=base_output,
        )
        return output.view(*input.shape[:-1], self.out_features)

    def _surrogate_vjp(
        self,
        input: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        grad_output: Tensor,
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Evaluate the hybrid QLoRA VJP on effective BF16 factor bytes.

        The frozen-base branch retains its declared BF16 compute program.  The
        LoRA branch widens the exact forward's saved BF16 x/A/B bytes and runs
        in FP32.  Their activation gradients are summed in FP32 and cast only
        by autograd at the caller's BF16 activation-storage boundary.
        """

        need_input, need_A, need_B = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None

        with torch.enable_grad(), torch.autocast(device_type=input.device.type, enabled=False):
            base_grad_input = None
            if need_input:
                base_input = input.detach().requires_grad_(True)
                base_weight = self._dequantize_weight().to(base_input.dtype)
                base_output = F.linear(base_input, base_weight, self.bias)
                (base_grad_input,) = torch.autograd.grad(
                    base_output,
                    base_input,
                    grad_outputs=grad_output.to(base_output.dtype),
                )

            reference_input = input.float().detach().requires_grad_(need_input)
            reference_A = effective_A.float().detach().requires_grad_(need_A)
            reference_B = effective_B.float().detach().requires_grad_(need_B)
            lora_output = F.linear(
                F.linear(reference_input, reference_A),
                reference_B,
            )

            requested = []
            labels = []
            for label, required, value in (
                ("input", need_input, reference_input),
                ("A", need_A, reference_A),
                ("B", need_B, reference_B),
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
            grad_input = base_grad_input.float() + by_label["input"].float()
        return grad_input, by_label.get("A"), by_label.get("B")

    def forward(self, input: Tensor) -> Tensor:
        self._validate_engaged_contract(input)
        return _Glm52ExactTP1QLoRAFunction.apply(input, self.lora_A, self.lora_B, self)


__all__ = [
    "GLM52_EXACT_TP1_QLORA_CONTRACT_VERSION",
    "Glm52ExactTP1BlockFP8QLoRALinear",
]
