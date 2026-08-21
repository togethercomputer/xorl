"""Exact active-LoRA routed-expert bank for the GLM-5.2 EP16 row.

The admitted sampler has outer TP16 and EP16, but its routed experts execute
with ``moe_tp_size == 1``.  Each EP owner therefore holds sixteen complete
experts.  This component preserves that physical program: SGLang's block-FP8
MoE sequence supplies the value path, its ordinary MoE-LoRA hooks inject the
gate/up deltas before SwiGLU and the routed down delta before the owner fold,
and a separate FP32-factor surrogate supplies autograd.

The frozen base is deliberately rank-local.  Before EP parallelization the
six adapter parameters retain the complete logical inventory; the EP plan then
replicates the three shared outer factors and owner-shards the three
expert-specific factors.  Component tests may still exercise the pre-EP global
form, while production consumes the post-EP sixteen-expert form directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.distributed.gradient_reduction import GradientReductionDomain
from xorl.lora.expert_adapter_contract import (
    ExpertAdapterGradientContract,
    gated_expert_factor_ownership,
    gated_expert_factor_shapes,
)
from xorl.models.layers.moe.backend import expert_adapter_backend_contract
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.transformers.glm5.exact_lora_contract import glm52_exact_lora_scaling
from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
from xorl.ops.exact.block_fp8_native import pack_fp8_as_float32


GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION = "glm52_exact_ep16_routed_qlora_v2"
GLM52_ROUTED_GLOBAL_EXPERTS = 256
GLM52_ROUTED_EP_SIZE = 16
GLM52_ROUTED_LOCAL_EXPERTS = 16
GLM52_ROUTED_MAX_LORAS_PER_BATCH = 8


@dataclass(frozen=True)
class Glm52ExactRoutedValueTrace:
    """Sampler-owned intermediate bytes captured by a component test."""

    gate_up_base: Tensor
    gate_up_post_lora: Tensor
    activated: Tensor
    down_base_routed: Tensor
    down_post_lora_routed: Tensor
    owner_output: Tensor


def localize_glm52_ep16_expert_ids(global_ids: Tensor, ep_rank: int) -> Tensor:
    """Map global expert IDs to one EP16 owner's contiguous local slots."""

    if not isinstance(ep_rank, int) or isinstance(ep_rank, bool) or not 0 <= ep_rank < GLM52_ROUTED_EP_SIZE:
        raise ValueError(f"GLM-5.2 routed expert owner must be in [0, 15], got {ep_rank!r}")
    if global_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"GLM-5.2 global expert IDs must be int32 or int64, got {global_ids.dtype}")
    if not global_ids.is_contiguous():
        raise ValueError("GLM-5.2 global expert IDs must use a contiguous route-major layout")
    if global_ids.numel() and (
        bool(torch.any(global_ids < 0)) or bool(torch.any(global_ids >= GLM52_ROUTED_GLOBAL_EXPERTS))
    ):
        raise ValueError("GLM-5.2 global expert IDs must be in [0, 255]")

    start = ep_rank * GLM52_ROUTED_LOCAL_EXPERTS
    owned = (global_ids >= start) & (global_ids < start + GLM52_ROUTED_LOCAL_EXPERTS)
    return torch.where(owned, global_ids - start, global_ids.new_full((), -1)).to(torch.int32).contiguous()


class _Glm52ExactEP16RoutedQLoRAFunction(torch.autograd.Function):
    """Keep sampler values isolated from the differentiable surrogate."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        down_A: Tensor,
        down_B: Tensor,
        routed_scaling_factor: float,
        module,
    ) -> Tensor:
        effective = tuple(
            factor.to(torch.bfloat16).contiguous() for factor in (gate_A, gate_B, up_A, up_B, down_A, down_B)
        )
        output, _ = module._sampler_value(
            hidden,
            routing,
            local_ids,
            *effective,
            routed_scaling_factor=float(routed_scaling_factor),
            capture_trace=False,
        )
        expected_shape = (hidden.shape[0], module.hidden_size)
        if output.dtype is not torch.bfloat16 or tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "GLM-5.2 exact routed sampler output contract mismatch: "
                f"got {output.dtype} {tuple(output.shape)}, expected torch.bfloat16 {expected_shape}"
            )
        ctx.module = module
        ctx.routed_scaling_factor = float(routed_scaling_factor)
        ctx.save_for_backward(
            hidden.detach(),
            routing.detach(),
            local_ids,
            *effective,
            gate_A,
            gate_B,
            up_A,
            up_B,
            down_A,
            down_B,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        hidden, routing, local_ids = saved[:3]
        effective = saved[3:9]
        gradients = ctx.module._surrogate_vjp(
            hidden,
            routing,
            local_ids,
            *effective,
            grad_output=grad_output,
            routed_scaling_factor=ctx.routed_scaling_factor,
            needs_input_grad=ctx.needs_input_grad[:9],
        )
        return (*gradients[:2], None, *gradients[2:], None, None)


class Glm52ExactEP16BlockFP8QLoRARoutedExperts(Glm52NativeBlockFP8Experts):
    """One local base slice plus one complete six-factor logical routed bank."""

    adapter_gradient_producer_family = "module_managed"
    contract_version = GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION
    _glm52_exact_active_lora_component = True
    fsdp_requires_full_precision = True
    quant_format = "block_fp8"
    quant_group_size = 128
    moe_implementation = "triton"
    hybrid_shared = True
    _ep_already_local_parameter_names = frozenset(
        {
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        }
    )
    _ep_force_shard_parameter_names = frozenset({"gate_proj_lora_B", "up_proj_lora_B", "down_proj_lora_A"})
    _ep_gradient_reduction_by_parameter = {
        "gate_proj_lora_A": GradientReductionDomain.EP_SUM,
        "up_proj_lora_A": GradientReductionDomain.EP_SUM,
        "down_proj_lora_B": GradientReductionDomain.EP_SUM,
    }
    logical_factor_names = (
        "gate_proj_lora_A",
        "gate_proj_lora_B",
        "up_proj_lora_A",
        "up_proj_lora_B",
        "down_proj_lora_A",
        "down_proj_lora_B",
    )

    @property
    def expert_adapter_gradient_contract(self) -> ExpertAdapterGradientContract:
        """Declare generic ownership plus the exact lane's immutable geometry."""

        if (
            self.contract_version != GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION
            or self.ep_size != GLM52_ROUTED_EP_SIZE
            or self.num_experts != GLM52_ROUTED_GLOBAL_EXPERTS
            or self.num_local_experts != GLM52_ROUTED_LOCAL_EXPERTS
            or self.moe_tp_size != 1
            or self.active_r != self.r
            or self.active_lora_alpha != self.lora_alpha
            or self.ep_dispatch != "alltoall"
            or self.hybrid_shared is not True
        ):
            raise ValueError("exact EP16 alltoall routed lane geometry no longer matches its declared contract")
        roles = ("gate_proj", "up_proj", "down_proj")
        return ExpertAdapterGradientContract(
            backend=replace(
                expert_adapter_backend_contract(self.moe_implementation),
                producer_family="module_managed",
            ),
            factor_layout="gkn_gate_up_down",
            projection_roles=roles,
            factor_ownership=gated_expert_factor_ownership(roles, hybrid_shared=True),
            factor_shapes=gated_expert_factor_shapes(
                roles,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                rank=self.r,
                hybrid_shared=True,
            ),
            supported_quantized_base_formats=("block_fp8",),
            quantized_base_format=self.quant_format,
            supports_efsdp_replication=True,
            requires_managed_fsdp=True,
            guard_fields=(
                ("expert_exact_contract", self.contract_version),
                ("expert_ep_size", self.ep_size),
                ("expert_moe_tp_size", self.moe_tp_size),
            ),
        )

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        ep_rank: int,
        r: int = 1,
        lora_alpha: int = 1,
        num_experts: int = GLM52_ROUTED_GLOBAL_EXPERTS,
        ep_size: int = GLM52_ROUTED_EP_SIZE,
        num_local_experts: int = GLM52_ROUTED_LOCAL_EXPERTS,
        moe_tp_size: int = 1,
        ep_dispatch: str = "alltoall",
        hidden_act: str = "silu",
        device: torch.device | str | None = None,
    ) -> None:
        scaling = glm52_exact_lora_scaling(r, lora_alpha)
        if (num_experts, ep_size, num_local_experts) != (
            GLM52_ROUTED_GLOBAL_EXPERTS,
            GLM52_ROUTED_EP_SIZE,
            GLM52_ROUTED_LOCAL_EXPERTS,
        ):
            raise ValueError(
                "GLM-5.2 exact routed experts require 256 global experts, EP16, and 16 local experts per owner"
            )
        if moe_tp_size != 1:
            raise ValueError(f"GLM-5.2 routed experts admit only effective MoE-TP1, got TP{moe_tp_size}")
        if ep_dispatch != "alltoall":
            raise ValueError(f"GLM-5.2 exact routed experts reject ep_dispatch={ep_dispatch!r}; DeepEP is not admitted")
        if not isinstance(ep_rank, int) or isinstance(ep_rank, bool) or not 0 <= ep_rank < ep_size:
            raise ValueError(f"GLM-5.2 routed expert owner must be in [0, 15], got {ep_rank!r}")

        # The native parent allocates this owner's physical sixteen-expert
        # slice.  The logical global count is restored below for canonical EP
        # mapping and the sampler's LoRA alignment contract.
        super().__init__(
            num_local_experts,
            hidden_size,
            intermediate_size,
            hidden_act=hidden_act,
            device=device,
        )
        self.num_local_experts = GLM52_ROUTED_LOCAL_EXPERTS
        self.num_experts = GLM52_ROUTED_GLOBAL_EXPERTS
        self.ep_size = GLM52_ROUTED_EP_SIZE
        self.ep_rank = ep_rank
        self.expert_offset = ep_rank * self.num_local_experts
        self.moe_tp_size = 1
        self.ep_dispatch = "alltoall"
        self.r = self.active_r = r
        self.lora_alpha = self.active_lora_alpha = lora_alpha
        self.scaling = scaling
        self.max_loras_per_batch = GLM52_ROUTED_MAX_LORAS_PER_BATCH

        self.gate_proj_lora_A = nn.Parameter(torch.empty((1, hidden_size, r), dtype=torch.float32, device=device))
        self.gate_proj_lora_B = nn.Parameter(
            torch.empty((num_experts, r, intermediate_size), dtype=torch.float32, device=device)
        )
        self.up_proj_lora_A = nn.Parameter(torch.empty((1, hidden_size, r), dtype=torch.float32, device=device))
        self.up_proj_lora_B = nn.Parameter(
            torch.empty((num_experts, r, intermediate_size), dtype=torch.float32, device=device)
        )
        self.down_proj_lora_A = nn.Parameter(
            torch.empty((num_experts, intermediate_size, r), dtype=torch.float32, device=device)
        )
        self.down_proj_lora_B = nn.Parameter(torch.empty((1, r, hidden_size), dtype=torch.float32, device=device))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        for factor in (self.gate_proj_lora_A, self.up_proj_lora_A, self.down_proj_lora_A):
            for expert in range(factor.shape[0]):
                nn.init.kaiming_uniform_(factor.data[expert], a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj_lora_B)
        nn.init.zeros_(self.up_proj_lora_B)
        nn.init.zeros_(self.down_proj_lora_B)

    def _apply(self, fn, recurse: bool = True):
        factor_names = self.logical_factor_names
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.gate_proj_lora_A.device))
        protected = {name: self._parameters.pop(name) for name in factor_names}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                if parameter.is_meta:
                    value = torch.empty_like(parameter, dtype=torch.float32, device=probe.device)
                    replacement = nn.Parameter(value, requires_grad=parameter.requires_grad)
                    for attribute in (
                        "spec_info",
                        "_ep_gradient_reduction_domain",
                        "_xorl_ep_force_shard",
                    ):
                        if hasattr(parameter, attribute):
                            setattr(replacement, attribute, getattr(parameter, attribute))
                    replacements[name] = replacement
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
        self.scaling = glm52_exact_lora_scaling(lora_rank, lora_alpha)
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha

    def load_prequantized(self, gate_up, gate_up_scale_inv, down, down_scale_inv) -> None:
        """Load the already-localized sixteen-expert official source slice."""

        expected = {
            "gate_up": (
                gate_up,
                (self.num_local_experts, self.hidden_size, 2 * self.intermediate_size),
                torch.float8_e4m3fn,
            ),
            "gate_up_scale_inv": (
                gate_up_scale_inv,
                (self.num_local_experts, self.hidden_size // 128, 2 * self.intermediate_size // 128),
                torch.float32,
            ),
            "down": (
                down,
                (self.num_local_experts, self.intermediate_size, self.hidden_size),
                torch.float8_e4m3fn,
            ),
            "down_scale_inv": (
                down_scale_inv,
                (self.num_local_experts, self.intermediate_size // 128, self.hidden_size // 128),
                torch.float32,
            ),
        }
        for name, (tensor, shape, dtype) in expected.items():
            if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
            if tensor.device != gate_up.device:
                raise RuntimeError("GLM-5.2 routed expert base tensors must share one device")
            if dtype is torch.float32 and not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"{name} contains non-finite values")
        with torch.no_grad():
            self.gate_up_packed_weight_f32.copy_(pack_fp8_as_float32(gate_up).to(self.gate_up_packed_weight_f32.device))
            self.gate_up_weight_scale_inv.copy_(gate_up_scale_inv.to(self.gate_up_weight_scale_inv.device))
            self.down_packed_weight_f32.copy_(pack_fp8_as_float32(down).to(self.down_packed_weight_f32.device))
            self.down_weight_scale_inv.copy_(down_scale_inv.to(self.down_weight_scale_inv.device))

    def localize_global_expert_ids(self, global_ids: Tensor) -> Tensor:
        return localize_glm52_ep16_expert_ids(global_ids, self.ep_rank)

    def _owner_local_factor(self, factor: Tensor, *, name: str) -> Tensor:
        """Return this owner's physical bank from either pre- or post-EP storage."""

        if factor.shape[0] == self.num_local_experts:
            return factor
        if factor.shape[0] == self.num_experts:
            expert_end = self.expert_offset + self.num_local_experts
            return factor[self.expert_offset : expert_end]
        raise RuntimeError(
            f"{name} must have either {self.num_experts} pre-EP or {self.num_local_experts} owner-local rows; "
            f"got {factor.shape[0]}"
        )

    def _validate_runtime_contract(self, hidden: Tensor, routing: Tensor, local_ids: Tensor) -> None:
        if self.ep_dispatch == "deepep" or self.ep_dispatch != "alltoall":
            raise RuntimeError("GLM-5.2 exact routed experts require canonical alltoall and reject DeepEP")
        if (self.num_experts, self.ep_size, self.num_local_experts, self.moe_tp_size) != (256, 16, 16, 1):
            raise RuntimeError("GLM-5.2 exact routed EP16/MoE-TP1 topology was mutated")
        if (
            self.active_r != self.r
            or self.active_lora_alpha != self.lora_alpha
            or self.scaling != glm52_exact_lora_scaling(self.r, self.lora_alpha)
        ):
            raise RuntimeError("GLM-5.2 exact routed expert adapter contract was mutated")
        if hidden.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact routed expert values require CUDA and pinned SGLang kernels")
        if hidden.ndim != 2 or hidden.dtype is not torch.bfloat16 or hidden.shape[1] != self.hidden_size:
            raise TypeError(
                "GLM-5.2 routed hidden states must be BF16 [rows, hidden_size]; "
                f"got {hidden.dtype} {tuple(hidden.shape)}"
            )
        if not hidden.is_contiguous() or hidden.shape[0] == 0:
            raise ValueError("GLM-5.2 routed hidden states must be non-empty and contiguous")
        if routing.ndim != 2 or routing.dtype is not torch.float32 or routing.shape[0] != hidden.shape[0]:
            raise TypeError(
                f"GLM-5.2 routing weights must be FP32 [rows, topk], got {routing.dtype} {tuple(routing.shape)}"
            )
        if tuple(local_ids.shape) != tuple(routing.shape) or local_ids.dtype is not torch.int32:
            raise TypeError(
                "GLM-5.2 local expert IDs must be int32 with the routing shape; "
                f"got {local_ids.dtype} {tuple(local_ids.shape)}"
            )
        if not routing.is_contiguous() or not local_ids.is_contiguous():
            raise ValueError("GLM-5.2 routing weights and local IDs must use contiguous route-major layouts")
        if not bool(torch.all(torch.isfinite(routing))):
            raise ValueError("GLM-5.2 routing weights contain non-finite values")
        if bool(torch.any(local_ids < -1)) or bool(torch.any(local_ids >= self.num_local_experts)):
            raise ValueError("GLM-5.2 local IDs admit only the -1 sentinel or slots [0, 15]")

        shared_factor_shapes = {
            "gate_proj_lora_A": (1, self.hidden_size, self.r),
            "up_proj_lora_A": (1, self.hidden_size, self.r),
            "down_proj_lora_B": (1, self.r, self.hidden_size),
        }
        expert_factor_shapes = {
            "gate_proj_lora_B": (self.r, self.intermediate_size),
            "up_proj_lora_B": (self.r, self.intermediate_size),
            "down_proj_lora_A": (self.intermediate_size, self.r),
        }
        factor_names = set(shared_factor_shapes) | set(expert_factor_shapes)
        for name, shape in shared_factor_shapes.items():
            factor = getattr(self, name)
            if factor.dtype is not torch.float32 or tuple(factor.shape) != shape or factor.device != hidden.device:
                raise TypeError(
                    f"{name} must remain FP32 {shape} on {hidden.device}, got "
                    f"{factor.dtype} {tuple(factor.shape)} on {factor.device}"
                )
        for name, trailing_shape in expert_factor_shapes.items():
            factor = getattr(self, name)
            if (
                factor.dtype is not torch.float32
                or factor.shape[0] not in (self.num_local_experts, self.num_experts)
                or tuple(factor.shape[1:]) != trailing_shape
                or factor.device != hidden.device
            ):
                raise TypeError(
                    f"{name} must remain FP32 [{self.num_local_experts}|{self.num_experts}, "
                    f"{', '.join(str(value) for value in trailing_shape)}] on {hidden.device}, got "
                    f"{factor.dtype} {tuple(factor.shape)} on {factor.device}"
                )
        if any(parameter.requires_grad for name, parameter in self.named_parameters() if name not in factor_names):
            raise RuntimeError("GLM-5.2 exact routed base weights and scales must remain frozen")

    def _physical_factor_buffers(
        self,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        down_A: Tensor,
        down_B: Tensor,
    ) -> dict[str, Tensor]:
        """Build the literal rank-local SGLang memory-pool views."""

        slots = self.max_loras_per_batch
        device = gate_A.device
        gate_up_A = torch.zeros((slots, 1, 2 * self.r, self.hidden_size), dtype=torch.bfloat16, device=device)
        gate_up_B = torch.zeros(
            (slots, self.num_local_experts, 2 * self.intermediate_size, self.r),
            dtype=torch.bfloat16,
            device=device,
        )
        down_A_buffer = torch.zeros(
            (slots, self.num_local_experts, self.r, self.intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        )
        down_B_buffer = torch.zeros((slots, 1, self.hidden_size, self.r), dtype=torch.bfloat16, device=device)
        local_gate_B = self._owner_local_factor(gate_B, name="gate_proj_lora_B")
        local_up_B = self._owner_local_factor(up_B, name="up_proj_lora_B")
        local_down_A = self._owner_local_factor(down_A, name="down_proj_lora_A")
        gate_up_A[0, 0].copy_(torch.cat((gate_A.transpose(1, 2), up_A.transpose(1, 2)), dim=1)[0])
        gate_up_B[0].copy_(self.scaling * torch.cat((local_gate_B.transpose(1, 2), local_up_B.transpose(1, 2)), dim=1))
        down_A_buffer[0].copy_(local_down_A.transpose(1, 2))
        down_B_buffer[0, 0].copy_(self.scaling * down_B.transpose(1, 2)[0])
        return {
            "gate_up_lora_a_weights": gate_up_A,
            "gate_up_lora_b_weights": gate_up_B,
            "down_lora_a_weights": down_A_buffer,
            "down_lora_b_weights": down_B_buffer,
        }

    def physical_factor_buffers(self) -> dict[str, Tensor]:
        effective = tuple(getattr(self, name).to(torch.bfloat16).contiguous() for name in self.logical_factor_names)
        return self._physical_factor_buffers(*effective)

    def _lora_info(self, rows: int, physical: dict[str, Tensor]):
        try:
            from sglang.srt.lora.lora_moe_runners import LoRAInfo  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang MoE-LoRA hooks are required") from exc

        device = physical["gate_up_lora_a_weights"].device
        ranks = torch.zeros(self.max_loras_per_batch, dtype=torch.int32, device=device)
        ranks[0] = self.r
        enabled = torch.zeros_like(ranks)
        enabled[0] = 1
        return LoRAInfo(
            **physical,
            seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
            req_to_lora=torch.zeros(1, dtype=torch.int32, device=device),
            lora_ranks=ranks,
            adapter_enabled=enabled,
            token_lora_mapping=torch.zeros(rows, dtype=torch.int32, device=device),
            max_lora_rank=self.r,
            num_experts=self.num_experts,
            has_active_lora=True,
            experts_shared_outer_loras=True,
            cg_buffers=None,
            fully_sharded=False,
            tp_size=1,
            tp_rank=0,
            hidden_size=self.hidden_size,
            lora_use_virtual_experts=False,
        )

    def _sampler_value(
        self,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        down_A: Tensor,
        down_B: Tensor,
        *,
        routed_scaling_factor: float,
        capture_trace: bool,
    ) -> tuple[Tensor, Glm52ExactRoutedValueTrace | None]:
        """Invoke S4's literal base/hook/activation/hook/fold sequence."""

        physical = self._physical_factor_buffers(gate_A, gate_B, up_A, up_B, down_A, down_B)
        try:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # noqa: PLC0415
                _fused_moe_kernel_sequence,
                _prepare_fused_moe_run,
            )
            from sglang.srt.lora.lora_moe_runners import build_lora_hooks  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang block-FP8 MoE-LoRA runner is required") from exc

        MoEExperts._ensure_sglang_server_args()
        w1 = self.gate_up_proj.transpose(1, 2)
        w2 = self.down_proj.transpose(1, 2)
        w1_scale = self.gate_up_weight_scale_inv.transpose(1, 2)
        w2_scale = self.down_weight_scale_inv.transpose(1, 2)
        if w1.transpose(1, 2).is_contiguous() is False or w2.transpose(1, 2).is_contiguous() is False:
            raise ValueError("GLM-5.2 routed base must retain the zero-copy GKN transpose stride contract")

        config, down_config, down_tma, sorted_ids, expert_ids, padded = _prepare_fused_moe_run(
            hidden,
            w1,
            w2,
            local_ids,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[128, 128],
        )
        hooks = build_lora_hooks(hidden, self._lora_info(hidden.shape[0], physical), local_ids)
        trace_values: dict[str, Tensor] = {}
        if capture_trace:
            original_gate_up = hooks.after_gate_up
            original_down = hooks.after_down

            def traced_gate_up(x, cache, weights, ids):
                trace_values["gate_up_base"] = cache.clone()
                assert original_gate_up is not None
                original_gate_up(x, cache, weights, ids)
                trace_values["gate_up_post_lora"] = cache.clone()

            def traced_down(activated, cache, weights, ids):
                trace_values["activated"] = activated.clone()
                trace_values["down_base_routed"] = cache.clone()
                assert original_down is not None
                original_down(activated, cache, weights, ids)
                trace_values["down_post_lora_routed"] = cache.clone()

            hooks.after_gate_up = traced_gate_up
            hooks.after_down = traced_down

        output = _fused_moe_kernel_sequence(
            hidden,
            w1,
            w2,
            routing,
            local_ids,
            sorted_ids,
            expert_ids,
            padded,
            config,
            down_config,
            down_tma,
            b1=None,
            b2=None,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=[128, 128],
            activation="silu",
            is_gated=True,
            no_combine=False,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=None,
            gemm1_limit=None,
            filter_expert=True,
            hooks=hooks,
            swiglu_limit=None,
            gate_up_interleaved=False,
            a1_q=None,
        )
        trace = None
        if capture_trace:
            trace = Glm52ExactRoutedValueTrace(owner_output=output.clone(), **trace_values)
        return output, trace

    def _dequantized_base(self) -> tuple[Tensor, Tensor]:
        try:
            from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang block-FP8 dequantization is required for the surrogate") from exc
        gate_up = block_quant_dequant(
            self.gate_up_proj.transpose(1, 2),
            self.gate_up_weight_scale_inv.transpose(1, 2),
            [128, 128],
            torch.bfloat16,
        )
        down = block_quant_dequant(
            self.down_proj.transpose(1, 2),
            self.down_weight_scale_inv.transpose(1, 2),
            [128, 128],
            torch.bfloat16,
        )
        return gate_up, down

    def _surrogate_program(
        self,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        down_A: Tensor,
        down_B: Tensor,
        *,
        routed_scaling_factor: float,
    ) -> Tensor:
        """Existing BF16 base branch plus FP32 logical LoRA dot products."""

        gate_up_weight, down_weight = self._dequantized_base()
        rows, topk = local_ids.shape
        local_gate_B = self._owner_local_factor(gate_B, name="gate_proj_lora_B")
        local_up_B = self._owner_local_factor(up_B, name="up_proj_lora_B")
        local_down_A = self._owner_local_factor(down_A, name="down_proj_lora_A")
        result = hidden.float() * 0.0 + routing.sum() * 0.0
        # Exact-zero anchors make unused expert slots participate with a
        # mathematically zero gradient instead of returning ``None``.
        result = result + (gate_A.sum() + gate_B.sum() + up_A.sum() + up_B.sum() + down_A.sum() + down_B.sum()) * 0.0
        for local_expert in range(self.num_local_experts):
            pair_rows, pair_topk = (local_ids == local_expert).nonzero(as_tuple=True)
            if pair_rows.numel() == 0:
                continue
            expert_input_fp32 = hidden.index_select(0, pair_rows).float()
            expert_input_bf16 = expert_input_fp32.to(torch.bfloat16)
            base_gate_up = F.linear(expert_input_bf16, gate_up_weight[local_expert])
            base_gate, base_up = base_gate_up.split(self.intermediate_size, dim=-1)
            gate_lora = self.scaling * F.linear(
                F.linear(expert_input_fp32, gate_A[0].transpose(0, 1)),
                local_gate_B[local_expert].transpose(0, 1),
            )
            up_lora = self.scaling * F.linear(
                F.linear(expert_input_fp32, up_A[0].transpose(0, 1)),
                local_up_B[local_expert].transpose(0, 1),
            )
            gate = (base_gate.float() + gate_lora).to(torch.bfloat16)
            up = (base_up.float() + up_lora).to(torch.bfloat16)
            activated = F.silu(gate.float()).to(torch.bfloat16) * up
            base_down = F.linear(activated, down_weight[local_expert])
            down_lora = self.scaling * F.linear(
                F.linear(activated.float(), local_down_A[local_expert].transpose(0, 1)),
                down_B[0].transpose(0, 1),
            )
            down = (base_down.float() + down_lora).to(torch.bfloat16).float()
            scores = routing[pair_rows, pair_topk].unsqueeze(1)
            result = result.index_add(0, pair_rows, down * scores * routed_scaling_factor)
        return result

    def _surrogate_vjp(
        self,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        gate_A: Tensor,
        gate_B: Tensor,
        up_A: Tensor,
        up_B: Tensor,
        down_A: Tensor,
        down_B: Tensor,
        *,
        grad_output: Tensor,
        routed_scaling_factor: float,
        needs_input_grad: tuple[bool, ...],
    ) -> tuple[Tensor | None, ...]:
        needs = (needs_input_grad[0], needs_input_grad[1], *needs_input_grad[3:9])
        if not any(needs):
            return (None,) * 8
        with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
            values = [hidden, routing, gate_A, gate_B, up_A, up_B, down_A, down_B]
            # The surrogate's logical derivatives accumulate against FP32
            # leaves.  BF16 casts inside ``_surrogate_program`` retain the
            # frozen base branch's declared compute/storage boundaries.
            references = [
                value.float().detach().requires_grad_(needed) for value, needed in zip(values, needs, strict=True)
            ]
            output = self._surrogate_program(
                references[0],
                references[1],
                local_ids,
                *references[2:],
                routed_scaling_factor=routed_scaling_factor,
            )
            requested = [value for value, needed in zip(references, needs, strict=True) if needed]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_outputs=grad_output.float(),
                allow_unused=False,
            )
        iterator = iter(computed)
        return tuple(next(iterator) if needed else None for needed in needs)

    def sampler_value_trace(
        self,
        hidden: Tensor,
        routing: Tensor,
        global_ids: Tensor,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> Glm52ExactRoutedValueTrace:
        local_ids = self.localize_global_expert_ids(global_ids)
        self._validate_runtime_contract(hidden, routing, local_ids)
        effective = tuple(getattr(self, name).to(torch.bfloat16).contiguous() for name in self.logical_factor_names)
        _, trace = self._sampler_value(
            hidden,
            routing,
            local_ids,
            *effective,
            routed_scaling_factor=float(routed_scaling_factor),
            capture_trace=True,
        )
        assert trace is not None
        return trace

    def forward(
        self,
        hidden_states: Tensor,
        routing_weights: Tensor,
        selected_experts: Tensor | None = None,
        *,
        sglang_ep_native_local_ids: Tensor | None = None,
        routed_scaling_factor: float = 1.0,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise TypeError(f"Unexpected GLM-5.2 exact routed expert arguments: {sorted(kwargs)}")
        if selected_experts is not None:
            expected_local = self.localize_global_expert_ids(selected_experts)
            if sglang_ep_native_local_ids is None:
                sglang_ep_native_local_ids = expected_local
            elif not torch.equal(expected_local, sglang_ep_native_local_ids):
                raise RuntimeError("Global expert IDs disagree with the canonical EP16 owner/local remap")
        if sglang_ep_native_local_ids is None:
            raise RuntimeError("GLM-5.2 exact routed experts require canonical rank-local IDs")
        routed_scaling_factor = float(routed_scaling_factor)
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ValueError("GLM-5.2 routed scaling factor must be finite and positive")
        self._validate_runtime_contract(hidden_states, routing_weights, sglang_ep_native_local_ids)
        return _Glm52ExactEP16RoutedQLoRAFunction.apply(
            hidden_states,
            routing_weights,
            sglang_ep_native_local_ids,
            self.gate_proj_lora_A,
            self.gate_proj_lora_B,
            self.up_proj_lora_A,
            self.up_proj_lora_B,
            self.down_proj_lora_A,
            self.down_proj_lora_B,
            routed_scaling_factor,
            self,
        )


__all__ = [
    "GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION",
    "GLM52_ROUTED_EP_SIZE",
    "GLM52_ROUTED_GLOBAL_EXPERTS",
    "GLM52_ROUTED_LOCAL_EXPERTS",
    "Glm52ExactEP16BlockFP8QLoRARoutedExperts",
    "Glm52ExactRoutedValueTrace",
    "localize_glm52_ep16_expert_ids",
]
