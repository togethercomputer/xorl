"""Exact active-LoRA local partials for the GLM-5.2 shared expert.

The admitted sampler executes the shared expert at genuine TP16.  Gate/up are
merged-column projections with replicated low-rank A factors and output-row
sharded B factors.  Down is row parallel: its A input columns and frozen base
columns are sharded, while B is replicated.  Each rank returns one unreduced
BF16 partial; the already-versioned canonical MoE owner folds those partials
after adding the routed contribution.

This module owns one logical shared expert, not sixteen model replicas.  A
``contributor_ordinal`` selects the physical sampler shard for one invocation.
The forward uses the pinned public SGLang FP8 and dynamic-LoRA kernels.  The
backward is the validated QLoRA surrogate evaluated on the effective BF16
factor bytes, with FP32 logical factor gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.models.transformers.glm5.exact_lora_contract import glm52_exact_lora_scaling
from xorl.ops.exact.block_fp8_native import NativeBlockFP8Linear, _sglang_native_block_fp8_linear_value
from xorl.ops.exact.fused_silu_and_mul import exact_fp32_silu_and_mul


GLM52_EXACT_TP16_SHARED_EXPERT_QLORA_CONTRACT_VERSION = "glm52_exact_tp16_shared_expert_qlora_v2"
GLM52_SHARED_EXPERT_HIDDEN_SIZE = 6144
GLM52_SHARED_EXPERT_INTERMEDIATE_SIZE = 2048
GLM52_SHARED_EXPERT_TP_SIZE = 16
GLM52_SHARED_EXPERT_SHARD_SIZE = 128


@lru_cache(maxsize=64)
def _single_adapter_batch_info(device_index: int, rows: int, rank: int, scaling: float):
    """Return the one-live-adapter metadata consumed by literal SGLang kernels."""

    from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415

    device = torch.device("cuda", device_index)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.full((1,), rank, dtype=torch.int32, device=device),
        scalings=torch.full((1,), scaling, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )


class _SharedExpertProjection(NativeBlockFP8Linear):
    """One frozen logical projection plus its FP32 low-rank masters."""

    adapter_gradient_producer_family = "module_managed"
    fsdp_requires_full_precision = True
    _glm52_exact_active_lora_component = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        role: str,
        rank: int,
        device: torch.device | str | None,
    ) -> None:
        super().__init__(in_features, out_features, device=device)
        if role not in {"gate", "up", "down"}:
            raise ValueError(f"Unsupported GLM-5.2 shared-expert projection role {role!r}")
        self.role = role
        self.lora_A = nn.Parameter(torch.empty((rank, in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank), dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _apply(self, fn, recurse: bool = True):
        # NativeBlockFP8Linear already protects its packed base. Protect the
        # FP32 masters as well and retain their Parameter identities.
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

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "GLM-5.2 exact shared-expert projections cannot run independently; "
            "invoke the TP16 shared-expert root with a contributor ordinal"
        )

    def forward_partition(self, *args, **kwargs):
        raise RuntimeError(
            "GLM-5.2 exact shared-expert projections cannot bypass active LoRA through a base-only partition"
        )


@dataclass(frozen=True)
class Glm52SharedExpertPhysicalFactors:
    """One sampler rank's live slot-zero factor views."""

    gate_up_A: Tensor
    gate_up_B: Tensor
    down_A: Tensor
    down_B: Tensor


@dataclass(frozen=True)
class _PhysicalBaseViews:
    gate_up_weight: Tensor
    gate_up_scales: Tensor
    down_weight: Tensor
    down_scales: Tensor


@dataclass(frozen=True)
class _ExactLocalValue:
    output: Tensor
    gate_up_base: Tensor
    gate_up_A_output: Tensor
    gate_up: Tensor
    activated: Tensor
    down_base: Tensor
    down_A_output: Tensor


class _Glm52ExactTP16SharedExpertFunction(torch.autograd.Function):
    """Keep one physical exact value path separate from its surrogate VJP."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        down_A: Tensor,
        down_B: Tensor,
        module,
        contributor_ordinal: int,
    ) -> Tensor:
        effective = tuple(
            factor.to(torch.bfloat16).contiguous() for factor in (gate_A, gate_B, up_A, up_B, down_A, down_B)
        )
        value = module._exact_forward_value(input, *effective, contributor_ordinal=contributor_ordinal)
        expected_shape = (*input.shape[:-1], module.hidden_size)
        output = value.output.view(expected_shape)
        if output.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact TP16 shared expert produced {output.dtype}, expected torch.bfloat16")
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"GLM-5.2 exact TP16 shared-expert output shape {tuple(output.shape)} does not match {expected_shape}"
            )

        ctx.module = module
        ctx.contributor_ordinal = int(contributor_ordinal)
        # Saving the masters as well as effective values preserves normal
        # autograd version checks against optimizer mutation before backward.
        ctx.save_for_backward(
            input.detach(),
            *effective,
            gate_A,
            gate_B,
            up_A,
            up_B,
            down_A,
            down_B,
            value.gate_up,
            value.activated,
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
            effective_down_A,
            effective_down_B,
            _gate_A_master,
            _gate_B_master,
            _up_A_master,
            _up_B_master,
            _down_A_master,
            _down_B_master,
            exact_gate_up,
            exact_activated,
        ) = ctx.saved_tensors
        gradients = ctx.module._surrogate_vjp(
            input,
            effective_gate_A,
            effective_gate_B,
            effective_up_A,
            effective_up_B,
            effective_down_A,
            effective_down_B,
            exact_gate_up,
            exact_activated,
            grad_output,
            contributor_ordinal=ctx.contributor_ordinal,
            needs_input_grad=ctx.needs_input_grad[:7],
        )
        return (*gradients, None, None)


class Glm52ExactTP16SharedExpertBlockFP8QLoRA(nn.Module):
    """Produce one unreduced sampler-equivalent GLM-5.2 shared-expert partial."""

    adapter_gradient_producer_family = "module_managed"
    contract_version = GLM52_EXACT_TP16_SHARED_EXPERT_QLORA_CONTRACT_VERSION
    fsdp_requires_full_precision = True
    _glm52_exact_active_lora_component = True
    logical_factor_names = (
        "gate_proj.lora_A",
        "gate_proj.lora_B",
        "up_proj.lora_A",
        "up_proj.lora_B",
        "down_proj.lora_A",
        "down_proj.lora_B",
    )

    def __init__(
        self,
        hidden_size: int = GLM52_SHARED_EXPERT_HIDDEN_SIZE,
        intermediate_size: int = GLM52_SHARED_EXPERT_INTERMEDIATE_SIZE,
        *,
        r: int = 1,
        lora_alpha: int = 1,
        tp_size: int = GLM52_SHARED_EXPERT_TP_SIZE,
        bias: bool = False,
        enable_aqn: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size != GLM52_SHARED_EXPERT_HIDDEN_SIZE:
            raise ValueError(
                f"GLM-5.2 exact shared expert requires hidden_size={GLM52_SHARED_EXPERT_HIDDEN_SIZE}, got {hidden_size}"
            )
        if intermediate_size != GLM52_SHARED_EXPERT_INTERMEDIATE_SIZE:
            raise ValueError(
                "GLM-5.2 exact shared expert requires "
                f"intermediate_size={GLM52_SHARED_EXPERT_INTERMEDIATE_SIZE}, got {intermediate_size}"
            )
        if tp_size != GLM52_SHARED_EXPERT_TP_SIZE:
            raise ValueError(f"GLM-5.2 exact shared expert requires TP{GLM52_SHARED_EXPERT_TP_SIZE}, got TP{tp_size}")
        scaling = glm52_exact_lora_scaling(r, lora_alpha)
        if bias:
            raise ValueError("GLM-5.2 exact shared expert is bias-free")
        if enable_aqn:
            raise ValueError("GLM-5.2 exact shared expert rejects adaptive quantization noise")
        if intermediate_size // tp_size != GLM52_SHARED_EXPERT_SHARD_SIZE:
            raise RuntimeError("GLM-5.2 exact shared-expert shard geometry changed")

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.tp_size = int(tp_size)
        self.shard_size = GLM52_SHARED_EXPERT_SHARD_SIZE
        self.r = self.active_r = int(r)
        self.lora_alpha = self.active_lora_alpha = int(lora_alpha)
        self.scaling = scaling
        self.enable_aqn = False
        self.gate_proj = _SharedExpertProjection(hidden_size, intermediate_size, role="gate", rank=r, device=device)
        self.up_proj = _SharedExpertProjection(hidden_size, intermediate_size, role="up", rank=r, device=device)
        self.down_proj = _SharedExpertProjection(intermediate_size, hidden_size, role="down", rank=r, device=device)
        self._checkpoint_source_prefix: str | None = None

    def bind_checkpoint_sources(self, shared_expert_fqn: str) -> None:
        """Bind the three official projection pairs to this logical root."""

        if not shared_expert_fqn or shared_expert_fqn.endswith("."):
            raise ValueError(f"Invalid GLM-5.2 shared-expert checkpoint prefix: {shared_expert_fqn!r}")
        if self._checkpoint_source_prefix not in {None, shared_expert_fqn}:
            raise RuntimeError(
                "GLM-5.2 exact shared-expert checkpoint sources are immutable once bound: "
                f"existing={self._checkpoint_source_prefix!r}, requested={shared_expert_fqn!r}"
            )
        self._checkpoint_source_prefix = shared_expert_fqn
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(self, projection_name)
            projection._source_fqn = f"{shared_expert_fqn}.{projection_name}"
            projection._source_quant_format = "block_fp8"
            projection._is_prequantized = True
            projection._merge_sources = None
            # The official FP8 weight is a source-only checkpoint key. The
            # scale is transformed inline into this live native parameter and
            # must remain eligible for ordinary checkpoint dispatch.
            projection._qlora_expected_skip_keys = {"weight"}

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        self.scaling = glm52_exact_lora_scaling(lora_rank, lora_alpha)
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha

    def load_prequantized(
        self,
        gate_weight: Tensor,
        gate_weight_scale_inv: Tensor,
        up_weight: Tensor,
        up_weight_scale_inv: Tensor,
        down_weight: Tensor,
        down_weight_scale_inv: Tensor,
    ) -> None:
        """Load the three official logical FP8 tensors without materialization."""

        NativeBlockFP8Linear.load_prequantized(self.gate_proj, gate_weight, gate_weight_scale_inv)
        NativeBlockFP8Linear.load_prequantized(self.up_proj, up_weight, up_weight_scale_inv)
        NativeBlockFP8Linear.load_prequantized(self.down_proj, down_weight, down_weight_scale_inv)

    def _validate_ordinal(self, contributor_ordinal: int) -> int:
        if isinstance(contributor_ordinal, bool) or not isinstance(contributor_ordinal, int):
            raise TypeError("GLM-5.2 shared-expert contributor_ordinal must be an integer")
        if not 0 <= contributor_ordinal < self.tp_size:
            raise ValueError(
                f"GLM-5.2 shared-expert contributor_ordinal must be in [0, {self.tp_size}), got {contributor_ordinal}"
            )
        return contributor_ordinal

    def _validate_factor_state(self) -> None:
        expected = {
            "gate_proj.lora_A": (self.gate_proj.lora_A, (self.r, self.hidden_size)),
            "gate_proj.lora_B": (self.gate_proj.lora_B, (self.intermediate_size, self.r)),
            "up_proj.lora_A": (self.up_proj.lora_A, (self.r, self.hidden_size)),
            "up_proj.lora_B": (self.up_proj.lora_B, (self.intermediate_size, self.r)),
            "down_proj.lora_A": (self.down_proj.lora_A, (self.r, self.intermediate_size)),
            "down_proj.lora_B": (self.down_proj.lora_B, (self.hidden_size, self.r)),
        }
        for name, (factor, shape) in expected.items():
            if factor.dtype is not torch.float32 or tuple(factor.shape) != shape:
                raise TypeError(f"{name} must remain FP32 with shape {shape}, got {factor.dtype} {tuple(factor.shape)}")

    def _validate_engaged_contract(self, input: Tensor, contributor_ordinal: int) -> int:
        ordinal = self._validate_ordinal(contributor_ordinal)
        if (
            self.tp_size != GLM52_SHARED_EXPERT_TP_SIZE
            or self.shard_size != GLM52_SHARED_EXPERT_SHARD_SIZE
            or self.active_r != self.r
            or self.active_lora_alpha != self.lora_alpha
            or self.scaling != glm52_exact_lora_scaling(self.r, self.lora_alpha)
            or self.enable_aqn
        ):
            raise RuntimeError("GLM-5.2 exact TP16 shared-expert runtime contract was mutated")
        self._validate_factor_state()
        if input.ndim < 2:
            raise ValueError("GLM-5.2 exact shared expert requires at least a row and hidden dimension")
        if input.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact shared expert requires BF16 activations, got {input.dtype}")
        if input.shape[-1] != self.hidden_size:
            raise ValueError(
                f"GLM-5.2 exact shared-expert input width {input.shape[-1]} does not match {self.hidden_size}"
            )
        if not input.is_contiguous():
            raise ValueError("GLM-5.2 exact shared expert requires contiguous sampler-layout activations")
        if input.numel() == 0:
            raise ValueError("GLM-5.2 exact shared expert does not admit an empty component batch")
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            if projection.packed_weight_f32.requires_grad or projection.weight_scale_inv.requires_grad:
                raise RuntimeError("GLM-5.2 exact shared-expert base weights and scales must remain frozen")
            if (
                projection.packed_weight_f32.device != input.device
                or projection.weight_scale_inv.device != input.device
            ):
                raise RuntimeError("GLM-5.2 exact shared-expert base state and activations must share one device")
            if projection.lora_A.device != input.device or projection.lora_B.device != input.device:
                raise RuntimeError("GLM-5.2 exact shared-expert factors and activations must share one device")
        return ordinal

    def _physical_factor_views_from_effective(
        self,
        effective_gate_A: Tensor,
        effective_gate_B: Tensor,
        effective_up_A: Tensor,
        effective_up_B: Tensor,
        effective_down_A: Tensor,
        effective_down_B: Tensor,
        contributor_ordinal: int,
    ) -> Glm52SharedExpertPhysicalFactors:
        ordinal = self._validate_ordinal(contributor_ordinal)
        start = ordinal * self.shard_size
        end = start + self.shard_size
        return Glm52SharedExpertPhysicalFactors(
            gate_up_A=torch.cat((effective_gate_A, effective_up_A), dim=0).unsqueeze(0).contiguous(),
            gate_up_B=torch.cat((effective_gate_B[start:end], effective_up_B[start:end]), dim=0)
            .unsqueeze(0)
            .contiguous(),
            down_A=effective_down_A[:, start:end].unsqueeze(0).contiguous(),
            down_B=effective_down_B.unsqueeze(0).contiguous(),
        )

    def physical_factor_views(self, contributor_ordinal: int) -> Glm52SharedExpertPhysicalFactors:
        """Derive one live SGLang slot view from the FP32 logical masters."""

        self._validate_factor_state()
        effective = tuple(
            factor.to(torch.bfloat16).contiguous()
            for factor in (
                self.gate_proj.lora_A,
                self.gate_proj.lora_B,
                self.up_proj.lora_A,
                self.up_proj.lora_B,
                self.down_proj.lora_A,
                self.down_proj.lora_B,
            )
        )
        return self._physical_factor_views_from_effective(*effective, contributor_ordinal)

    @staticmethod
    def _partition_base_state(
        projection: NativeBlockFP8Linear,
        *,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        out_start, out_end = projection._validate_partition(output_range, projection.out_features, "output")
        in_start, in_end = projection._validate_partition(input_range, projection.in_features, "input")
        weight = projection.fp8_weight()[out_start:out_end, in_start:in_end].contiguous()
        scales = projection.weight_scale_inv[
            out_start // 128 : (out_end + 127) // 128,
            in_start // 128 : (in_end + 127) // 128,
        ].contiguous()
        return weight, scales

    def _physical_base_views(self, contributor_ordinal: int) -> _PhysicalBaseViews:
        ordinal = self._validate_ordinal(contributor_ordinal)
        start = ordinal * self.shard_size
        end = start + self.shard_size
        gate_weight, gate_scales = self._partition_base_state(self.gate_proj, output_range=(start, end))
        up_weight, up_scales = self._partition_base_state(self.up_proj, output_range=(start, end))
        down_weight, down_scales = self._partition_base_state(self.down_proj, input_range=(start, end))
        return _PhysicalBaseViews(
            gate_up_weight=torch.cat((gate_weight, up_weight), dim=0).contiguous(),
            gate_up_scales=torch.cat((gate_scales, up_scales), dim=0).contiguous(),
            down_weight=down_weight,
            down_scales=down_scales,
        )

    def _exact_forward_value(
        self,
        input: Tensor,
        effective_gate_A: Tensor,
        effective_gate_B: Tensor,
        effective_up_A: Tensor,
        effective_up_B: Tensor,
        effective_down_A: Tensor,
        effective_down_B: Tensor,
        *,
        contributor_ordinal: int,
    ) -> _ExactLocalValue:
        """Execute the literal merged-column, SwiGLU, and row-partial program."""

        if input.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact shared-expert forward requires CUDA and pinned SGLang kernels")
        effective_factors = (
            effective_gate_A,
            effective_gate_B,
            effective_up_A,
            effective_up_B,
            effective_down_A,
            effective_down_B,
        )
        if any(factor.dtype is not torch.bfloat16 for factor in effective_factors):
            raise TypeError("GLM-5.2 exact shared-expert effective factors must be BF16")
        if any(factor.device != input.device for factor in effective_factors):
            raise RuntimeError("GLM-5.2 exact shared-expert factors and activations must share one CUDA device")

        try:
            from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang shared-expert LoRA kernels are required") from exc

        rows = input.numel() // self.hidden_size
        input_2d = input.view(rows, self.hidden_size)
        batch_info = _single_adapter_batch_info(input.device.index, rows, self.r, self.scaling)
        factors = self._physical_factor_views_from_effective(*effective_factors, contributor_ordinal)
        base = self._physical_base_views(contributor_ordinal)

        gate_up_base = _sglang_native_block_fp8_linear_value(
            input_2d,
            base.gate_up_weight,
            base.gate_up_scales,
        )
        gate_up_base_witness = gate_up_base.clone()
        gate_up_A_output = sgemm_lora_a_fwd(input_2d, factors.gate_up_A, batch_info, stack_num=2)
        gate_up = gate_up_lora_b_fwd(
            gate_up_A_output,
            factors.gate_up_B,
            batch_info,
            self.shard_size,
            base_output=gate_up_base,
        )
        # Exact SGLang target mode resolves SiluAndMul.forward_exact to the
        # one-round FP32 SwiGLU (xorl-sglang f10b907d8); this site must
        # produce those bytes.
        activated = exact_fp32_silu_and_mul(gate_up)

        down_base = _sglang_native_block_fp8_linear_value(
            activated,
            base.down_weight,
            base.down_scales,
        )
        down_base_witness = down_base.clone()
        down_A_output = sgemm_lora_a_fwd(activated, factors.down_A, batch_info)
        output = sgemm_lora_b_fwd(
            down_A_output,
            factors.down_B,
            batch_info,
            base_output=down_base,
        )
        return _ExactLocalValue(
            output=output,
            gate_up_base=gate_up_base_witness,
            gate_up_A_output=gate_up_A_output,
            gate_up=gate_up,
            activated=activated,
            down_base=down_base_witness,
            down_A_output=down_A_output,
        )

    def _dequantized_partition_weight(
        self,
        projection: NativeBlockFP8Linear,
        *,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> Tensor:
        weight = NativeBlockFP8Linear.forward(projection, return_dequantized_weight=True)
        out_start, out_end = projection._validate_partition(output_range, projection.out_features, "output")
        in_start, in_end = projection._validate_partition(input_range, projection.in_features, "input")
        return weight[out_start:out_end, in_start:in_end].contiguous()

    def _projection_surrogate_vjp(
        self,
        input: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        grad_output: Tensor,
        projection: NativeBlockFP8Linear,
        *,
        output_range: tuple[int, int] | None,
        input_range: tuple[int, int] | None,
        A_input_range: tuple[int, int] | None,
        B_output_range: tuple[int, int] | None,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        need_input, need_A, need_B = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None

        with torch.enable_grad(), torch.autocast(device_type=input.device.type, enabled=False):
            base_grad_input = None
            if need_input:
                base_input = input.detach().requires_grad_(True)
                base_weight = self._dequantized_partition_weight(
                    projection,
                    output_range=output_range,
                    input_range=input_range,
                ).to(base_input.dtype)
                base_output = F.linear(base_input, base_weight)
                (base_grad_input,) = torch.autograd.grad(
                    base_output,
                    base_input,
                    grad_outputs=grad_output.to(base_output.dtype),
                )

            reference_input = input.float().detach().requires_grad_(need_input)
            reference_A = effective_A.float().detach().requires_grad_(need_A)
            reference_B = effective_B.float().detach().requires_grad_(need_B)
            A = reference_A
            if A_input_range is not None:
                A = A[:, A_input_range[0] : A_input_range[1]]
            B = reference_B
            if B_output_range is not None:
                B = B[B_output_range[0] : B_output_range[1]]
            lora_output = self.scaling * F.linear(F.linear(reference_input, A), B)

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

    def _surrogate_vjp(
        self,
        input: Tensor,
        effective_gate_A: Tensor,
        effective_gate_B: Tensor,
        effective_up_A: Tensor,
        effective_up_B: Tensor,
        effective_down_A: Tensor,
        effective_down_B: Tensor,
        exact_gate_up: Tensor,
        exact_activated: Tensor,
        grad_output: Tensor,
        *,
        contributor_ordinal: int,
        needs_input_grad: tuple[bool, bool, bool, bool, bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Evaluate the staged BF16-base/FP32-factor QLoRA surrogate."""

        need_input, need_gate_A, need_gate_B, need_up_A, need_up_B, need_down_A, need_down_B = needs_input_grad
        if not any(needs_input_grad):
            return (None,) * 7
        ordinal = self._validate_ordinal(contributor_ordinal)
        start = ordinal * self.shard_size
        end = start + self.shard_size
        rows = input.numel() // self.hidden_size
        input_2d = input.view(rows, self.hidden_size)
        grad_output_2d = grad_output.view(rows, self.hidden_size)

        need_activation = need_input or need_gate_A or need_gate_B or need_up_A or need_up_B
        down_grad_input, grad_down_A, grad_down_B = self._projection_surrogate_vjp(
            exact_activated,
            effective_down_A,
            effective_down_B,
            grad_output_2d,
            self.down_proj,
            output_range=None,
            input_range=(start, end),
            A_input_range=(start, end),
            B_output_range=None,
            needs_input_grad=(need_activation, need_down_A, need_down_B),
        )

        grad_input = grad_gate_A = grad_gate_B = grad_up_A = grad_up_B = None
        if need_activation:
            with torch.enable_grad(), torch.autocast(device_type=input.device.type, enabled=False):
                gate_up_input = exact_gate_up.detach().requires_grad_(True)
                # VJP reference differentiates the same one-round program the
                # forward emits (backward stays trainer-owned numerics).
                activation = exact_fp32_silu_and_mul(gate_up_input)
                (gate_up_grad,) = torch.autograd.grad(
                    activation,
                    gate_up_input,
                    grad_outputs=down_grad_input.to(activation.dtype),
                )
            gate_grad, up_grad = gate_up_grad.split(self.shard_size, dim=-1)
            gate_grad_input, grad_gate_A, grad_gate_B = self._projection_surrogate_vjp(
                input_2d,
                effective_gate_A,
                effective_gate_B,
                gate_grad,
                self.gate_proj,
                output_range=(start, end),
                input_range=None,
                A_input_range=None,
                B_output_range=(start, end),
                needs_input_grad=(need_input, need_gate_A, need_gate_B),
            )
            up_grad_input, grad_up_A, grad_up_B = self._projection_surrogate_vjp(
                input_2d,
                effective_up_A,
                effective_up_B,
                up_grad,
                self.up_proj,
                output_range=(start, end),
                input_range=None,
                A_input_range=None,
                B_output_range=(start, end),
                needs_input_grad=(need_input, need_up_A, need_up_B),
            )
            if need_input:
                # Match two unfused logical QLoRA projections feeding the
                # same BF16 activation: each branch crosses its storage
                # boundary before gate-then-up accumulation.
                grad_input = (gate_grad_input.to(input.dtype) + up_grad_input.to(input.dtype)).view_as(input)

        return grad_input, grad_gate_A, grad_gate_B, grad_up_A, grad_up_B, grad_down_A, grad_down_B

    def forward(self, input: Tensor, *, contributor_ordinal: int) -> Tensor:
        ordinal = self._validate_engaged_contract(input, contributor_ordinal)
        return _Glm52ExactTP16SharedExpertFunction.apply(
            input,
            self.gate_proj.lora_A,
            self.gate_proj.lora_B,
            self.up_proj.lora_A,
            self.up_proj.lora_B,
            self.down_proj.lora_A,
            self.down_proj.lora_B,
            self,
            ordinal,
        )


__all__ = [
    "GLM52_EXACT_TP16_SHARED_EXPERT_QLORA_CONTRACT_VERSION",
    "GLM52_SHARED_EXPERT_HIDDEN_SIZE",
    "GLM52_SHARED_EXPERT_INTERMEDIATE_SIZE",
    "GLM52_SHARED_EXPERT_SHARD_SIZE",
    "GLM52_SHARED_EXPERT_TP_SIZE",
    "Glm52ExactTP16SharedExpertBlockFP8QLoRA",
    "Glm52SharedExpertPhysicalFactors",
]
