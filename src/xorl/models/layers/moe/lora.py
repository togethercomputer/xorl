"""LoRA adapter injection for MoE experts.

Provides:
- :class:`MoELoRAConfig` — LoRA configuration dataclass.
- :class:`MoEExpertsLoRA` — unified weight container for MoE experts with LoRA
  (handles eager, triton, native, and quack backends).
- :func:`inject_lora_into_experts` — replaces ``MoEBlock.experts`` in-place.

All weights are stored in (G, K, N) format — ``[num_experts, in_features, out_features]``.

LoRA weight parameter names are preserved exactly for checkpoint compatibility::

    gate_proj_lora_A: [E, hidden, r]     gate_proj_lora_B: [E, r, inter]
    up_proj_lora_A:   [E, hidden, r]     up_proj_lora_B:   [E, r, inter]
    down_proj_lora_A: [E, inter, r]      down_proj_lora_B: [E, r, hidden]
"""

import math
from dataclasses import dataclass, replace
from typing import List, Optional

import torch
import torch.nn as nn

from xorl.distributed.gradient_reduction import GradientReductionDomain

from ....lora.expert_adapter_contract import (
    ExpertAdapterFactorOwnership,
    ExpertAdapterGradientContract,
    gated_expert_factor_ownership,
    gated_expert_factor_shapes,
    validate_gated_silu_expert_adapter_semantics,
)
from ....lora.fold import (
    FoldedLoraWeightGateUpGKN,
    FoldedLoraWeightGKN,
    canonical_lora_fold_gkn,
    lora_merged_cache_enabled,
    lora_merged_forward_enabled,
)
from ....lora.modules.base import LoraModule
from ....ops.group_gemm.kernel import compute_lora_scaling
from ....utils import logging
from ..activations import ACT2FN
from .backend import (
    EP_COMBINE,
    EP_DISPATCH,
    EP_EXPERT_COMPUTE_LORA,
    MOE_EXPERT_BACKENDS_LORA,
    ep_lora_gradient_reduction_domain,
    expert_adapter_backend_contract,
    zero_token_lora_output,
)
from .common import split_gate_up_proj


logger = logging.get_logger(__name__)


class _SglangNativeLoRAHooksTrainFunction(torch.autograd.Function):
    """Keep the literal serving-hook value path separate from its VJP.

    The forward is SGLang's ordinary base/hook/activation/hook/combine
    sequence.  Backward reuses XoRL's existing trainable fused-MoE surrogate,
    so changing the value path does not discard hidden, router, or adapter
    gradients.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        routing: torch.Tensor,
        local_ids: torch.Tensor,
        gate_A: torch.Tensor,
        gate_B: torch.Tensor,
        up_A: torch.Tensor,
        up_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        module,
    ) -> torch.Tensor:
        effective = tuple(
            factor.to(torch.bfloat16).contiguous() for factor in (gate_A, gate_B, up_A, up_B, down_A, down_B)
        )
        output = module._sglang_native_lora_hook_value(
            hidden,
            routing,
            local_ids,
            *effective,
        )
        expected = (hidden.shape[0], module.hidden_dim)
        if output.dtype is not torch.bfloat16 or tuple(output.shape) != expected:
            raise RuntimeError(
                "Native Qwen MoE-LoRA hook output contract mismatch: "
                f"got {output.dtype} {tuple(output.shape)}, expected torch.bfloat16 {expected}"
            )
        ctx.module = module
        ctx.save_for_backward(
            hidden.detach(),
            routing.detach(),
            local_ids,
            gate_A,
            gate_B,
            up_A,
            up_B,
            down_A,
            down_B,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, routing, local_ids, *factors = ctx.saved_tensors
        gradients = ctx.module._sglang_native_lora_hook_surrogate_vjp(
            hidden,
            routing,
            local_ids,
            *factors,
            grad_output=grad_output,
            needs_input_grad=ctx.needs_input_grad,
        )
        return gradients[0], gradients[1], None, *gradients[2:], None


@dataclass
class MoELoRAConfig:
    """Configuration for MoE LoRA adapters."""

    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = None
    use_rslora: bool = False
    hybrid_shared: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["gate_proj", "up_proj", "down_proj"]
        selected = tuple(self.target_modules)
        supported = {"gate_proj", "up_proj", "down_proj"}
        unsupported = set(selected) - supported
        if not selected or len(set(selected)) != len(selected) or unsupported:
            raise ValueError(
                f"MoE LoRA target_modules must be a non-empty subset of {sorted(supported)}; got {list(selected)!r}"
            )
        self.target_modules = list(selected)


class MoEExpertsLoRA(LoraModule, nn.Module):
    """MoE experts with LoRA adapters.

    Handles all backends: eager (per-expert loop), triton/quack (group GEMM),
    and native (torch._grouped_mm). Base weights are frozen; only LoRA weights
    are trainable.

    Base weights use fused ``gate_up_proj`` storage in (G, K, N) format —
    ``[num_experts, in_features, out_features]``. ``gate_proj`` and
    ``up_proj`` remain available as views for compatibility.

    When Expert Parallelism is enabled, pass ``num_local_experts`` to create
    weights at the local (sharded) shape.
    """

    def adapter_gradient_producer_family(self) -> str:
        """Return the fixed producer selected by this execution plan."""

        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        if lora_merged_forward_enabled(self) or get_parallel_state().ep_enabled or self.moe_implementation != "eager":
            return "fused_managed"
        return "module_managed"

    @property
    def expert_adapter_gradient_contract(self) -> ExpertAdapterGradientContract:
        """Declare the configured backend and factor ownership to the compiler."""

        try:
            validate_gated_silu_expert_adapter_semantics(self)
        except NotImplementedError as error:
            raise ValueError(str(error)) from error
        roles = tuple(self.lora_config.target_modules)
        backend = replace(
            expert_adapter_backend_contract(self.moe_implementation),
            producer_family=self.adapter_gradient_producer_family(),
        )
        return ExpertAdapterGradientContract(
            backend=backend,
            factor_layout="gkn_gate_up_down",
            projection_roles=roles,
            factor_ownership=gated_expert_factor_ownership(
                roles,
                hybrid_shared=self.lora_config.hybrid_shared,
            ),
            factor_shapes=gated_expert_factor_shapes(
                roles,
                num_experts=self.num_experts,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size,
                rank=self.r,
                hybrid_shared=self.lora_config.hybrid_shared,
            ),
            supports_efsdp_replication=True,
            guard_fields=(
                ("expert_hybrid_shared", self.lora_config.hybrid_shared),
                ("expert_lora_semantics", getattr(self, "expert_lora_semantics", "generic_unclamped")),
                # Ownership guards admit only canonical scalar wire types.
                # Hex preserves the configured binary64 value exactly.
                ("expert_swiglu_limit_hex", self.swiglu_limit.hex()),
            ),
        )

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        moe_implementation: str = "triton",
        lora_config: Optional[MoELoRAConfig] = None,
        num_local_experts: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        self.num_global_experts = int(num_experts)
        self.num_experts = num_local_experts if num_local_experts is not None else num_experts
        self.num_local_experts = int(self.num_experts)
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.gated = True
        self.moe_implementation = moe_implementation
        self.lora_config = lora_config or MoELoRAConfig()
        self.r = self.lora_config.r
        self.lora_alpha = self.lora_config.lora_alpha
        self.swiglu_limit = float(swiglu_limit)
        self.gate_up_bias = None
        self.down_bias = None
        self.active_r = self.r
        self.active_lora_alpha = self.lora_alpha
        self.use_rslora = self.lora_config.use_rslora
        self._ep_gradient_reduction_domain = ep_lora_gradient_reduction_domain(moe_implementation)
        factor_ownership = gated_expert_factor_ownership(
            self.lora_config.target_modules,
            hybrid_shared=self.lora_config.hybrid_shared,
        )
        self._ep_gradient_reduction_by_parameter = {
            name: (
                self._ep_gradient_reduction_domain
                if ownership is ExpertAdapterFactorOwnership.EP_REPLICATED
                else GradientReductionDomain.NONE
            )
            for name, ownership in factor_ownership
        }

        # Base weights (frozen) in (G, K, N) format
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, hidden_dim, 2 * intermediate_size),
            requires_grad=False,
        )
        self.gate_up_proj._fused_gate_up = True
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, intermediate_size, hidden_dim),
            requires_grad=False,
        )

        # Activation function
        self.act_fn = ACT2FN[hidden_act]

        # LoRA weights in (G, K, N) format:
        #   A: [E, in_features, r]
        #   B: [E, r, out_features]
        r = self.lora_config.r
        num_exp = self.num_experts
        hybrid = self.lora_config.hybrid_shared

        # For hybrid_shared mode:
        # - gate/up: lora_A shared [1, hidden, r], lora_B per-expert [E, r, inter]
        # - down: lora_A per-expert [E, inter, r], lora_B shared [1, r, hidden]
        shared_exp = 1 if hybrid else num_exp

        # ParallelPlan shards parameters, not buffers. Unselected projection
        # factors are structural zeros consumed by the same fused backend API,
        # so their owner-specific expert dimension must already be EP-local.
        # A post-EP construction explicitly supplies a smaller num_local_experts;
        # a pre-EP construction derives the future local size from parallel state.
        self._zero_factor_experts = num_exp
        if num_local_experts is None or num_local_experts == num_experts:
            try:
                from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

                parallel_state = get_parallel_state()
                ep_size = int(parallel_state.ep_size) if parallel_state.ep_enabled else 1
                if ep_size > 1:
                    if num_exp % ep_size:
                        raise ValueError(f"Expert count {num_exp} is not divisible by expert parallel size {ep_size}")
                    self._zero_factor_experts = num_exp // ep_size
            except RuntimeError:
                # Construction outside an initialized distributed runtime is
                # local; a later EP plan is responsible for parameters.
                pass

        self._create_lora_params("gate_proj", shared_exp, num_exp, r, hidden_dim, intermediate_size)
        self._create_lora_params("up_proj", shared_exp, num_exp, r, hidden_dim, intermediate_size)
        self._create_lora_params("down_proj", num_exp, (1 if hybrid else num_exp), r, intermediate_size, hidden_dim)

        # Scaling factor
        self.scaling = compute_lora_scaling(self.lora_alpha, self.r, self.use_rslora)

        self.reset_lora_parameters()

        # EP dispatch strategy (inherited from source MoEExperts via inject_lora)
        self.ep_dispatch: str = "alltoall"
        self.deepep_buffer_size_gb: float = 2.0
        self.deepep_num_sms: int = 20
        self.deepep_async_combine: bool = False
        self.deepep_native_exact: bool = False
        self.lora_serving_mode: str | None = None
        self.alltoall_combine_hidden_chunk_size: int = 0

    @property
    def gate_proj(self) -> torch.Tensor:
        gate_proj, _ = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        gate_proj.grad = (
            None if self.gate_up_proj.grad is None else self.gate_up_proj.grad[..., : self.intermediate_size]
        )
        return gate_proj

    @property
    def up_proj(self) -> torch.Tensor:
        _, up_proj = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        up_proj.grad = None if self.gate_up_proj.grad is None else self.gate_up_proj.grad[..., self.intermediate_size :]
        return up_proj

    def _create_lora_params(
        self, name: str, A_experts: int, B_experts: int, r: int, in_features: int, out_features: int
    ):
        """Create LoRA A and B parameters in (G, K, N) format.

        A: [experts, in_features, r]
        B: [experts, r, out_features]
        """
        if name in self.lora_config.target_modules:
            setattr(self, f"{name}_lora_A", nn.Parameter(torch.empty(A_experts, in_features, r)))
            setattr(self, f"{name}_lora_B", nn.Parameter(torch.empty(B_experts, r, out_features)))
        else:
            local_A_experts = 1 if A_experts == 1 else self._zero_factor_experts
            local_B_experts = 1 if B_experts == 1 else self._zero_factor_experts
            self.register_buffer(
                f"{name}_lora_A",
                torch.zeros(local_A_experts, in_features, r),
                persistent=False,
            )
            self.register_buffer(
                f"{name}_lora_B",
                torch.zeros(local_B_experts, r, out_features),
                persistent=False,
            )

    def reset_lora_parameters(self):
        """Initialize LoRA weights: kaiming_uniform for A, zeros for B."""
        for name in self.lora_config.target_modules:
            lora_A = getattr(self, f"{name}_lora_A")
            lora_B = getattr(self, f"{name}_lora_B")
            if isinstance(lora_A, nn.Parameter):
                for i in range(lora_A.shape[0]):
                    nn.init.kaiming_uniform_(lora_A.data[i], a=math.sqrt(5))
                nn.init.zeros_(lora_B.data)

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        """Update the active LoRA slice used during forward/merge/export."""
        if lora_rank <= 0 or lora_rank > self.r:
            raise ValueError(f"Active LoRA rank must be in [1, {self.r}], got {lora_rank}")
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha

    def _active_scaling(self) -> float:
        return compute_lora_scaling(self.active_lora_alpha, self.active_r, self.use_rslora)

    def _active_lora_views(self, proj_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        lora_A = getattr(self, f"{proj_name}_lora_A")[..., : self.active_r].contiguous()
        lora_B = getattr(self, f"{proj_name}_lora_B")[:, : self.active_r, ...].contiguous()
        return lora_A, lora_B

    def _compute_proj_delta(self, proj_name: str) -> torch.Tensor:
        """Compute LoRA delta for one projection. Returns [E, K, N] in GKN format."""
        lora_A, lora_B = self._active_lora_views(proj_name)
        E = max(lora_A.shape[0], lora_B.shape[0])
        A = lora_A.expand(E, -1, -1)  # [E, in, r]
        B = lora_B.expand(E, -1, -1)  # [E, r, out]
        return torch.bmm(A, B) * self._active_scaling()  # [E, in, out] = [E, K, N]

    def merge_weights(self) -> None:
        """Merge LoRA weights into base weights for inference.

        After merging: weight = weight + delta_weight for each active projection.
        Resets LoRA parameters after merge.
        """
        with torch.no_grad():
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                if proj_name not in self.lora_config.target_modules:
                    continue
                base = getattr(self, proj_name)
                delta = self._compute_proj_delta(proj_name)
                merged = base.to(torch.float32) + delta
                base.data.copy_(merged.to(base.dtype))
        self.reset_lora_parameters()

    # ------------------------------------------------------------------
    # Merged-forward exact-model contract lane
    # ------------------------------------------------------------------

    def sglang_moe_tp_sim_enabled(self, parallel_state) -> bool:
        """TP-sim is outside the LoRA merged-forward envelope."""
        return False

    def sglang_fused_experts_auto_supported(self) -> bool:
        """Auto-default eligibility mirror of :meth:`MoEExperts.sglang_fused_experts_auto_supported`:
        under the exact model program the adapted experts fold their delta
        (canonical fold) and run the contracted serving kernel on the merged
        weights, so the ep=1 auto-enable applies to them too."""
        return (
            lora_merged_forward_enabled(self)
            and self.hidden_act in {"silu", "gelu", "gelu_tanh"}
            and self.swiglu_limit == 0.0
        )

    def invalidate_merged_weight_cache(self) -> None:
        self._merged_weight_cache = {}

    def _merged_weight_key(self) -> tuple:
        params = (
            self.gate_proj_lora_A,
            self.gate_proj_lora_B,
            self.up_proj_lora_A,
            self.up_proj_lora_B,
            self.down_proj_lora_A,
            self.down_proj_lora_B,
            self.gate_up_proj,
            self.down_proj,
        )
        return (
            tuple(t._version for t in params),
            tuple(t.data_ptr() for t in params),
            self.active_r,
            self.active_lora_alpha,
        )

    def _merged_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonically folded ``(gate_up' [E, H, 2I], down' [E, I, H])``.

        Fold = :func:`xorl.lora.fold.canonical_lora_fold_gkn` per projection
        (fp32 accumulate, cast once) — the exact arithmetic the weight-sync
        merged extraction ships under the same exact program, so the trainer forward and
        the serving engine see identical merged bytes. Cached per module, keyed
        on adapter/base param versions and the active rank/alpha; the optimizer
        step's in-place update invalidates the key."""
        cache = getattr(self, "_merged_weight_cache", None)
        if cache is None:
            cache = {}
            self._merged_weight_cache = cache
        key = self._merged_weight_key()
        if lora_merged_cache_enabled() and cache.get("key") == key:
            return cache["gate_up"], cache["down"]
        scaling = self._active_scaling()
        inter = self.intermediate_size
        with torch.no_grad():
            gate_A, gate_B = self._active_lora_views("gate_proj")
            up_A, up_B = self._active_lora_views("up_proj")
            down_A, down_B = self._active_lora_views("down_proj")
            gate_f = canonical_lora_fold_gkn(self.gate_up_proj[..., :inter], gate_A, gate_B, scaling)
            up_f = canonical_lora_fold_gkn(self.gate_up_proj[..., inter:], up_A, up_B, scaling)
            gate_up_f = torch.cat([gate_f, up_f], dim=-1)
            down_f = canonical_lora_fold_gkn(self.down_proj, down_A, down_B, scaling)
        if lora_merged_cache_enabled():
            cache["key"] = key
            cache["gate_up"] = gate_up_f
            cache["down"] = down_f
        else:
            cache.clear()
        return gate_up_f, down_f

    def canonical_merged_proj_weight(self, proj_name: str) -> torch.Tensor:
        """Per-projection view of the canonically folded weights (weight-sync
        extraction consumes this so the shipped bytes are exactly the bytes the
        merged forward trains with)."""
        gate_up_f, down_f = self._merged_weights()
        if proj_name == "down_proj":
            return down_f
        inter = self.intermediate_size
        if proj_name == "gate_proj":
            return gate_up_f[..., :inter]
        if proj_name == "up_proj":
            return gate_up_f[..., inter:]
        raise KeyError(f"unknown projection {proj_name!r}")

    def _merged_trainable_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Straight-through merged weights: forward bits = the cached fold,
        backward = exact chain rule through W' = W + scaling * (A @ B) into the
        (active slices of the) LoRA factors."""
        gate_up_f, down_f = self._merged_weights()
        r = self.active_r
        scaling = self._active_scaling()
        gate_up_w = FoldedLoraWeightGateUpGKN.apply(
            gate_up_f,
            self.gate_proj_lora_A[..., :r],
            self.gate_proj_lora_B[:, :r, :],
            self.up_proj_lora_A[..., :r],
            self.up_proj_lora_B[:, :r, :],
            scaling,
            self.intermediate_size,
        )
        down_w = FoldedLoraWeightGKN.apply(
            down_f,
            self.down_proj_lora_A[..., :r],
            self.down_proj_lora_B[:, :r, :],
            scaling,
        )
        return gate_up_w, down_w

    def _merged_lora_needs_grad(self, *activations: torch.Tensor) -> bool:
        if not torch.is_grad_enabled():
            return False
        if any(t is not None and t.requires_grad for t in activations):
            return True
        return any(
            getattr(self, f"{proj}_lora_{f}").requires_grad
            for proj in ("gate_proj", "up_proj", "down_proj")
            for f in ("A", "B")
            if isinstance(getattr(self, f"{proj}_lora_{f}"), nn.Parameter)
        )

    def sglang_fused_experts_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """LoRA K3 contract lane: canonical fold + the contracted serving-kernel
        forward on the merged weights (bit-identical to a serving engine that
        received the folded weights), backward through the low-rank factors.

        Mirrors :meth:`MoEExperts.sglang_fused_experts_forward`; the exact
        model program must select canonical merged-LoRA execution."""
        from .experts import (  # noqa: PLC0415
            _MOE_SGLANG_FUSED_EXPERTS_ENV,
            MoEExperts,
            _sglang_fused_experts_kernel_call,
            _SglangFusedExpertsTrainFunction,
            moe_sglang_fused_experts_weight_mode,
        )

        if not lora_merged_forward_enabled(self):
            raise NotImplementedError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 on LoRA-adapted experts requires canonical "
                "merged-LoRA execution, selected automatically by the exact model program."
            )
        if self.hidden_act not in {"silu", "gelu_tanh"} or self.swiglu_limit != 0.0:
            raise NotImplementedError(
                "Canonical merged-LoRA experts support gated silu/gelu_tanh without swiglu_limit only"
            )
        if moe_sglang_fused_experts_weight_mode() == "cached":
            raise NotImplementedError(
                "Canonical merged-LoRA execution does not compose with "
                "XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE=cached (the transpose cache cannot track "
                "per-step merged weights); use the strided (default) or transient mode."
            )
        fused_experts_impl = MoEExperts._load_sglang_fused_experts_impl()

        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act

        if self._merged_lora_needs_grad(hidden_flat, routing_flat):
            gate_up_w, down_w = self._merged_trainable_weights()
            output = _SglangFusedExpertsTrainFunction.apply(
                hidden_flat,
                routing_flat,
                selected_flat,
                gate_up_w,
                down_w,
                fused_experts_impl,
                activation,
                self.hidden_act,
                self.swiglu_limit,
                self.num_experts,
                None,
            )
        else:
            gate_up_f, down_f = self._merged_weights()
            output = _sglang_fused_experts_kernel_call(
                hidden_flat,
                gate_up_f,
                down_f,
                routing_flat,
                selected_flat,
                fused_experts_impl,
                activation,
                self.swiglu_limit,
                None,
                weight_cache=None,
            )
        return output.reshape(original_shape)

    def _merged_ep_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Merged-forward EP lane: alltoall dispatch, canonical fold of the
        LOCAL expert slice, the contracted per-rank serving-kernel compute
        (routing weight applied in-kernel, mirroring
        :meth:`MoEExperts.sglang_fused_experts_ep_compute`), stock combine."""
        from .experts import (  # noqa: PLC0415
            _MOE_SGLANG_FUSED_EXPERTS_ENV,
            MoEExperts,
            _sglang_fused_experts_ep_kernel_call,
            _SglangFusedExpertsEPTrainFunction,
            moe_sglang_fused_experts_weight_mode,
        )

        if self.ep_dispatch != "alltoall":
            raise NotImplementedError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 supports ep_dispatch='alltoall' only (got {self.ep_dispatch!r})"
            )
        if self.hidden_act not in {"silu", "gelu_tanh"} or self.swiglu_limit != 0.0:
            raise NotImplementedError(
                "Canonical merged-LoRA experts support gated silu/gelu_tanh without swiglu_limit only"
            )
        if moe_sglang_fused_experts_weight_mode() == "cached":
            raise NotImplementedError(
                "Canonical merged-LoRA execution does not compose with WEIGHT_MODE=cached; use strided/transient."
            )
        fused_experts_impl = MoEExperts._load_sglang_fused_experts_impl()
        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act

        dispatch_kwargs = self._build_dispatch_kwargs(hidden_states, routing_weights, selected_experts, parallel_state)
        permute_tokens, cumsum, ctx = EP_DISPATCH[self.ep_dispatch](**dispatch_kwargs)
        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        if expert_scores is None:
            raise ValueError("Canonical merged-LoRA EP compute requires dispatched expert scores")

        if permute_tokens.shape[0] == 0 and self._merged_lora_needs_grad(permute_tokens, expert_scores):
            factors = tuple(
                value
                for projection in ("gate_proj", "up_proj", "down_proj")
                for value in self._active_lora_views(projection)
            )
            expert_output = zero_token_lora_output(permute_tokens, self.hidden_dim, *factors)
        elif self._merged_lora_needs_grad(permute_tokens, expert_scores):
            gate_up_w, down_w = self._merged_trainable_weights()
            expert_output = _SglangFusedExpertsEPTrainFunction.apply(
                permute_tokens,
                expert_scores,
                gate_up_w,
                down_w,
                cumsum,
                fused_experts_impl,
                activation,
                self.hidden_act,
                self.swiglu_limit,
                True,
                None,
            )
        else:
            gate_up_f, down_f = self._merged_weights()
            permute_tokens = permute_tokens.contiguous()
            if permute_tokens.shape[0] == 0:
                expert_output = permute_tokens.new_zeros(permute_tokens.shape)
            else:
                expert_output = _sglang_fused_experts_ep_kernel_call(
                    permute_tokens,
                    gate_up_f,
                    down_f,
                    expert_scores,
                    cumsum,
                    fused_experts_impl,
                    activation,
                    self.swiglu_limit,
                    gate_up_bias=None,
                    gated=True,
                    weight_cache=None,
                )

        # Routing weights were applied in-kernel (serving semantics) — no
        # post-hoc expert_scores multiply on this lane.
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
        return EP_COMBINE[self.ep_dispatch](**combine_kwargs)

    def sglang_ep_native_routed_partial(
        self,
        hidden_flat: torch.Tensor,
        routing_flat: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Literal SGLang active-LoRA local leaf for native exact EP.

        The base weights stay unmerged.  SGLang's ordinary MoE-LoRA hooks add
        gate/up deltas before SwiGLU and down deltas before its fused
        ``no_combine=False`` route fold.  A zero LoRA-B therefore still
        constructs and executes the same hooks as a live adapter; only their
        numerical delta is zero.
        """
        from .experts import moe_sglang_fused_experts_weight_mode  # noqa: PLC0415

        if not lora_merged_forward_enabled(self):
            raise NotImplementedError(
                "Native EP combine on LoRA-adapted experts requires the exact LoRA execution contract"
            )
        mode = getattr(self, "lora_serving_mode", None)
        if mode == "merged":
            return self._sglang_ep_native_merged_partial(
                hidden_flat,
                routing_flat,
                local_ids,
            )
        if mode != "separate":
            raise ValueError(f"Unknown LoRA serving mode {mode!r}")
        if not self.lora_config.hybrid_shared:
            raise NotImplementedError("Native exact MoE-LoRA hooks currently require hybrid shared-outer factors")
        if self.hidden_act not in {"silu", "gelu_tanh"} or self.swiglu_limit != 0.0:
            raise NotImplementedError("LoRA native EP combine supports gated silu/gelu_tanh without swiglu_limit only")
        if moe_sglang_fused_experts_weight_mode() == "cached":
            raise NotImplementedError(
                "Native exact MoE-LoRA hooks do not compose with WEIGHT_MODE=cached; use strided/transient."
            )
        factors = tuple(
            value
            for projection in ("gate_proj", "up_proj", "down_proj")
            for value in self._active_lora_views(projection)
        )
        if self._merged_lora_needs_grad(hidden_flat, routing_flat):
            return _SglangNativeLoRAHooksTrainFunction.apply(
                hidden_flat,
                routing_flat,
                local_ids,
                *factors,
                self,
            )
        effective = tuple(factor.to(torch.bfloat16).contiguous() for factor in factors)
        return self._sglang_native_lora_hook_value(
            hidden_flat,
            routing_flat,
            local_ids,
            *effective,
        )

    def _sglang_ep_native_merged_partial(
        self,
        hidden_flat: torch.Tensor,
        routing_flat: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Existing canonical folded-weight native local leaf."""

        from .experts import (  # noqa: PLC0415
            MoEExperts,
            _sglang_fused_experts_kernel_call,
            _SglangFusedExpertsTrainFunction,
            moe_sglang_fused_experts_weight_mode,
        )

        if moe_sglang_fused_experts_weight_mode() == "cached":
            raise NotImplementedError(
                "Canonical merged-LoRA execution does not compose with WEIGHT_MODE=cached; use strided/transient."
            )
        fused_experts_impl = MoEExperts._load_sglang_fused_experts_impl()
        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act
        e_local = int(self.gate_up_proj.shape[0])
        if self._merged_lora_needs_grad(hidden_flat, routing_flat):
            gate_up_w, down_w = self._merged_trainable_weights()
            return _SglangFusedExpertsTrainFunction.apply(
                hidden_flat,
                routing_flat,
                local_ids,
                gate_up_w,
                down_w,
                fused_experts_impl,
                activation,
                self.hidden_act,
                self.swiglu_limit,
                e_local,
                None,
                True,
            )
        gate_up_f, down_f = self._merged_weights()
        if hidden_flat.shape[0] == 0:
            return hidden_flat.clone()
        return _sglang_fused_experts_kernel_call(
            hidden_flat,
            gate_up_f,
            down_f,
            routing_flat,
            local_ids,
            fused_experts_impl,
            activation,
            self.swiglu_limit,
            None,
            weight_cache=None,
            filter_expert=True,
        )

    def _sglang_native_lora_physical_buffers(
        self,
        gate_A: torch.Tensor,
        gate_B: torch.Tensor,
        up_A: torch.Tensor,
        up_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the one-adapter SGLang shared-outer memory-pool views."""

        scaling = self._active_scaling()
        return {
            "gate_up_lora_a_weights": torch.cat((gate_A.transpose(1, 2), up_A.transpose(1, 2)), dim=1)
            .unsqueeze(0)
            .contiguous(),
            "gate_up_lora_b_weights": (scaling * torch.cat((gate_B.transpose(1, 2), up_B.transpose(1, 2)), dim=1))
            .unsqueeze(0)
            .to(torch.bfloat16)
            .contiguous(),
            "down_lora_a_weights": down_A.transpose(1, 2).unsqueeze(0).contiguous(),
            "down_lora_b_weights": (scaling * down_B.transpose(1, 2)).unsqueeze(0).to(torch.bfloat16).contiguous(),
        }

    def _sglang_native_lora_info(self, rows: int, physical: dict[str, torch.Tensor]):
        try:
            from sglang.srt.lora.lora_moe_runners import LoRAInfo  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned SGLang MoE-LoRA hooks are required") from exc

        device = physical["gate_up_lora_a_weights"].device
        return LoRAInfo(
            **physical,
            seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
            req_to_lora=torch.zeros(1, dtype=torch.int32, device=device),
            lora_ranks=torch.tensor([self.active_r], dtype=torch.int32, device=device),
            adapter_enabled=torch.ones(1, dtype=torch.int32, device=device),
            token_lora_mapping=torch.zeros(rows, dtype=torch.int32, device=device),
            max_lora_rank=self.active_r,
            num_experts=self.num_global_experts,
            has_active_lora=True,
            single_adapter_id=0,
            experts_shared_outer_loras=True,
            cg_buffers=None,
            fully_sharded=False,
            tp_size=1,
            tp_rank=0,
            hidden_size=self.hidden_dim,
            lora_use_virtual_experts=False,
        )

    def _sglang_native_lora_hook_value(
        self,
        hidden: torch.Tensor,
        routing: torch.Tensor,
        local_ids: torch.Tensor,
        gate_A: torch.Tensor,
        gate_B: torch.Tensor,
        up_A: torch.Tensor,
        up_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
    ) -> torch.Tensor:
        """Run SGLang's literal base/hook/activation/hook/fold sequence."""

        from .experts import MoEExperts  # noqa: PLC0415

        try:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # noqa: PLC0415
                _fused_moe_kernel_sequence,
                _prepare_fused_moe_run,
            )
            from sglang.srt.lora.lora_moe_runners import build_lora_hooks  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned SGLang MoE-LoRA hook runner is required") from exc

        if hidden.shape[0] == 0:
            return hidden.clone()
        physical = self._sglang_native_lora_physical_buffers(gate_A, gate_B, up_A, up_B, down_A, down_B)
        MoEExperts._ensure_sglang_server_args()
        w1 = self.gate_up_proj.transpose(1, 2)
        w2 = self.down_proj.transpose(1, 2)
        local_ids = local_ids.contiguous()
        routing = routing.to(torch.float32).contiguous()
        config, down_config, down_tma, sorted_ids, expert_ids, padded = _prepare_fused_moe_run(
            hidden,
            w1,
            w2,
            local_ids,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        hooks = build_lora_hooks(
            hidden,
            self._sglang_native_lora_info(hidden.shape[0], physical),
            local_ids,
            mul_routed_weight=True,
        )
        return _fused_moe_kernel_sequence(
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
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            w1_scale=None,
            w2_scale=None,
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            activation=("gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act),
            is_gated=True,
            no_combine=False,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_limit=None,
            filter_expert=True,
            hooks=hooks,
            swiglu_limit=None,
            gate_up_interleaved=False,
            a1_q=None,
        )

    def _sglang_native_lora_hook_surrogate_vjp(
        self,
        hidden: torch.Tensor,
        routing: torch.Tensor,
        local_ids: torch.Tensor,
        gate_A: torch.Tensor,
        gate_B: torch.Tensor,
        up_A: torch.Tensor,
        up_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        *,
        grad_output: torch.Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        """VJP through the existing folded-weight training surrogate."""

        from .experts import (  # noqa: PLC0415
            MoEExperts,
            _SglangFusedExpertsTrainFunction,
        )

        needs = (needs_input_grad[0], needs_input_grad[1], *needs_input_grad[3:9])
        if not any(needs):
            return (None,) * 8
        values = (hidden, routing, gate_A, gate_B, up_A, up_B, down_A, down_B)
        with torch.enable_grad():
            references = [value.detach().requires_grad_(needed) for value, needed in zip(values, needs, strict=True)]
            hidden_ref, routing_ref, gate_A_ref, gate_B_ref, up_A_ref, up_B_ref, down_A_ref, down_B_ref = references
            scaling = self._active_scaling()
            inter = self.intermediate_size
            gate = canonical_lora_fold_gkn(self.gate_up_proj[..., :inter], gate_A_ref, gate_B_ref, scaling)
            up = canonical_lora_fold_gkn(self.gate_up_proj[..., inter:], up_A_ref, up_B_ref, scaling)
            down = canonical_lora_fold_gkn(self.down_proj, down_A_ref, down_B_ref, scaling)
            output = _SglangFusedExpertsTrainFunction.apply(
                hidden_ref,
                routing_ref,
                local_ids,
                torch.cat((gate, up), dim=-1),
                down,
                MoEExperts._load_sglang_fused_experts_impl(),
                ("gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act),
                self.hidden_act,
                self.swiglu_limit,
                # ParallelPlan/FSDP may shard the expert tensors after this
                # module is constructed.  In that case the construction-time
                # attribute still names the global expert count, while the
                # physical weight leading dimension is the authoritative
                # local grouped-GEMM count.
                int(self.gate_up_proj.shape[0]),
                None,
                True,
            )
            requested = [value for value, needed in zip(references, needs, strict=True) if needed]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_outputs=grad_output,
                allow_unused=False,
            )
        iterator = iter(computed)
        return tuple(next(iterator) if needed else None for needed in needs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        selected_experts: torch.Tensor = None,
        expert_idx: int = None,
        sglang_ep_native_local_ids: torch.Tensor = None,
        dsv4_exact_native: bool = False,
        dsv4_exact_lora_live: bool = True,
        dsv4_exact_return_routes: bool = False,
    ) -> torch.Tensor:
        """Forward pass with LoRA.

        For **eager**: called per-expert with ``expert_idx`` when EP is disabled.
        For all implementations: checks EP first, falls back to local path.
        """
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        if dsv4_exact_native:
            from xorl.models.transformers.deepseek_v4.native_payload import (  # noqa: PLC0415
                dsv4_native_mxfp4_routed_partial,
            )

            return dsv4_native_mxfp4_routed_partial(
                hidden_states,
                routing_weights,
                selected_experts,
                self,
                lora_live=dsv4_exact_lora_live,
                return_routes=dsv4_exact_return_routes,
            )

        if sglang_ep_native_local_ids is not None:
            return self.sglang_ep_native_routed_partial(
                hidden_states,
                routing_weights,
                sglang_ep_native_local_ids,
            )

        parallel_state = get_parallel_state()

        if parallel_state.ep_enabled:
            return self._ep_forward(hidden_states, routing_weights, selected_experts, parallel_state)

        if self.moe_implementation == "eager":
            assert expert_idx is not None
            return self._eager_lora_forward(hidden_states, expert_idx)

        # Local path — registry-based
        fn = MOE_EXPERT_BACKENDS_LORA[self.moe_implementation]
        gate_proj = self.gate_proj.contiguous()
        up_proj = self.up_proj.contiguous()
        gate_proj_lora_A, gate_proj_lora_B = self._active_lora_views("gate_proj")
        up_proj_lora_A, up_proj_lora_B = self._active_lora_views("up_proj")
        down_proj_lora_A, down_proj_lora_B = self._active_lora_views("down_proj")
        return fn(
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=self.down_proj,
            gate_proj_lora_A=gate_proj_lora_A,
            gate_proj_lora_B=gate_proj_lora_B,
            up_proj_lora_A=up_proj_lora_A,
            up_proj_lora_B=up_proj_lora_B,
            down_proj_lora_A=down_proj_lora_A,
            down_proj_lora_B=down_proj_lora_B,
            scaling=self._active_scaling(),
            swiglu_limit=self.swiglu_limit,
        )

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Unified EP forward with LoRA: dispatch → compute → combine.

        Uses the same dispatch/combine as ``MoEExperts._ep_forward()`` but
        routes to the LoRA-aware EP compute registry. Under the explicit
        ``XORL_MOE_SGLANG_FUSED_EXPERTS=1`` contract flag the merged-forward
        lane replaces the LoRA compute; the exact model program selects that
        lane automatically.
        """
        from .experts import _moe_sglang_fused_experts_env_state  # noqa: PLC0415

        if self.deepep_native_exact:
            from xorl.distributed.moe.deepep_native_exact import (  # noqa: PLC0415
                canonicalize_native_routing_metadata,
                native_dispatch_runner_combine,
            )

            if self.ep_dispatch != "deepep" or self.deepep_async_combine:
                raise RuntimeError("LoRA native DeepEP exact requires synchronous ep_dispatch='deepep'")
            # The exact router stores coefficients at BF16 precision.  DeepEP
            # exposes its routing-metadata ABI as FP32, so widen those stored
            # BF16 values without introducing any additional information.
            routing_weights = canonicalize_native_routing_metadata(routing_weights)
            return native_dispatch_runner_combine(
                hidden_states,
                routing_weights,
                selected_experts,
                ep_group=parallel_state.ep_group,
                num_experts=self.num_global_experts,
                num_local_experts=int(self.gate_up_proj.shape[0]),
                buffer_size_gb=self.deepep_buffer_size_gb,
                num_sms=self.deepep_num_sms,
                runner=self.sglang_ep_native_routed_partial,
            )

        explicit_sglang_fused = _moe_sglang_fused_experts_env_state()
        if explicit_sglang_fused is True:
            if not lora_merged_forward_enabled(self):
                from .experts import _MOE_SGLANG_FUSED_EXPERTS_ENV  # noqa: PLC0415

                raise NotImplementedError(
                    f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 on LoRA-adapted experts requires "
                    "canonical merged-LoRA execution (canonical fold + serving kernel on merged weights); "
                    "a partially-contracted LoRA lane would silently void the contract."
                )
            return self._merged_ep_forward(hidden_states, routing_weights, selected_experts, parallel_state)

        if self.moe_implementation not in EP_EXPERT_COMPUTE_LORA:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"EP with LoRA. Available: {list(EP_EXPERT_COMPUTE_LORA.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. Available: {list(EP_DISPATCH.keys())}"
            )

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]
        compute_fn = EP_EXPERT_COMPUTE_LORA[self.moe_implementation]
        gate_proj = self.gate_proj.contiguous()
        up_proj = self.up_proj.contiguous()
        gate_proj_lora_A, gate_proj_lora_B = self._active_lora_views("gate_proj")
        up_proj_lora_A, up_proj_lora_B = self._active_lora_views("up_proj")
        down_proj_lora_A, down_proj_lora_B = self._active_lora_views("down_proj")

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(hidden_states, routing_weights, selected_experts, parallel_state)
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)

        # Step 2: Expert computation with LoRA
        if permute_tokens.shape[0] == 0:
            expert_output = zero_token_lora_output(
                permute_tokens,
                self.hidden_dim,
                gate_proj_lora_A,
                gate_proj_lora_B,
                up_proj_lora_A,
                up_proj_lora_B,
                down_proj_lora_A,
                down_proj_lora_B,
            )
        else:
            expert_output = compute_fn(
                permute_tokens,
                cumsum,
                gate_proj,
                up_proj,
                self.down_proj,
                gate_proj_lora_A,
                gate_proj_lora_B,
                up_proj_lora_A,
                up_proj_lora_B,
                down_proj_lora_A,
                down_proj_lora_B,
                self._active_scaling(),
                self.swiglu_limit,
            )

        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        if expert_scores is not None:
            expert_output = expert_output * expert_scores.unsqueeze(1).to(expert_output.dtype)

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
        return combine_fn(**combine_kwargs)

    def _build_dispatch_kwargs(self, hidden_states, routing_weights, selected_experts, parallel_state):
        """Build dispatch kwargs based on ep_dispatch strategy."""
        kwargs = dict(
            hidden_states=hidden_states,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=self.num_experts,
        )
        if self.ep_dispatch == "alltoall":
            kwargs["ep_group"] = parallel_state.ep_group
        elif self.ep_dispatch == "deepep":
            from xorl.distributed.moe.deepep import get_default_buffer  # noqa: PLC0415

            kwargs["buffer"] = get_default_buffer(
                ep_group=parallel_state.ep_group,
                buffer_size_gb=self.deepep_buffer_size_gb,
                num_sms=self.deepep_num_sms,
            )
            kwargs["num_local_experts"] = self.gate_up_proj.shape[0]
        return kwargs

    def _build_combine_kwargs(self, expert_output, ctx, dispatch_kwargs, parallel_state):
        """Build combine kwargs based on ep_dispatch strategy."""
        if self.ep_dispatch == "alltoall":
            return dict(
                expert_output=expert_output,
                ctx=ctx,
                ep_group=parallel_state.ep_group,
                hidden_chunk_size=self.alltoall_combine_hidden_chunk_size,
            )
        elif self.ep_dispatch == "deepep":
            return dict(
                buffer=dispatch_kwargs["buffer"],
                expert_output=expert_output,
                ctx=ctx,
                async_combine=self.deepep_async_combine,
            )

    def _eager_lora_forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Per-expert LoRA forward (eager mode).

        All weights in (G, K, N) format — direct matmul, no transpose.
        """
        compute_dtype = hidden_states.dtype

        # x @ W — no transpose needed with (G, K, N) format
        gate_proj_out = torch.matmul(hidden_states, self.gate_proj[expert_idx])
        up_proj_out = torch.matmul(hidden_states, self.up_proj[expert_idx])
        active_scaling = self._active_scaling()

        if "gate_proj" in self.lora_config.target_modules:
            gate_A, gate_B = self._active_lora_views("gate_proj")
            A = gate_A[min(expert_idx, gate_A.shape[0] - 1)].to(compute_dtype)
            B = gate_B[expert_idx].to(compute_dtype)
            gate_proj_out = gate_proj_out + torch.matmul(torch.matmul(hidden_states, A), B) * active_scaling

        if self.swiglu_limit > 0:
            gate_proj_out = gate_proj_out.clamp(-self.swiglu_limit, self.swiglu_limit)

        if "up_proj" in self.lora_config.target_modules:
            up_A, up_B = self._active_lora_views("up_proj")
            A = up_A[min(expert_idx, up_A.shape[0] - 1)].to(compute_dtype)
            B = up_B[expert_idx].to(compute_dtype)
            up_proj_out = up_proj_out + torch.matmul(torch.matmul(hidden_states, A), B) * active_scaling

        out = self.act_fn(gate_proj_out) * up_proj_out

        down_out = torch.matmul(out, self.down_proj[expert_idx])
        if "down_proj" in self.lora_config.target_modules:
            down_A, down_B = self._active_lora_views("down_proj")
            A = down_A[expert_idx].to(compute_dtype)
            B = down_B[min(expert_idx, down_B.shape[0] - 1)].to(compute_dtype)
            down_out = down_out + torch.matmul(torch.matmul(out, A), B) * active_scaling

        return down_out

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_size={self.intermediate_size}, r={self.active_r}, max_r={self.r}, "
            f"lora_alpha={self.active_lora_alpha}, "
            f"target_modules={self.lora_config.target_modules}, swiglu_limit={self.swiglu_limit}"
        )

    @classmethod
    def from_module(cls, module: nn.Module, r: int, lora_alpha: int, **kwargs):
        """Create from an existing MoEExperts module, copying base weights."""
        validate_gated_silu_expert_adapter_semantics(module)
        target_modules = kwargs.get("target_modules", ["gate_proj", "up_proj", "down_proj"])
        use_rslora = kwargs.get("use_rslora", False)
        hybrid_shared = kwargs.get("hybrid_shared", False)
        lora_config = MoELoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            use_rslora=use_rslora,
            hybrid_shared=hybrid_shared,
        )

        base_gate_up = getattr(module, "gate_up_proj", None)
        num_exp = base_gate_up.shape[0] if base_gate_up is not None else module.gate_proj.shape[0]
        hidden_dim = module.hidden_dim
        intermediate_size = module.intermediate_size
        moe_implementation = getattr(module, "moe_implementation", "triton")

        lora_experts = cls(
            num_experts=num_exp,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            moe_implementation=moe_implementation,
            lora_config=lora_config,
            num_local_experts=num_exp,
            swiglu_limit=float(getattr(module, "swiglu_limit", 0.0)),
        )
        lora_experts.act_fn = module.act_fn
        lora_experts.ep_dispatch = getattr(module, "ep_dispatch", "alltoall")
        lora_experts.deepep_buffer_size_gb = getattr(module, "deepep_buffer_size_gb", 2.0)
        lora_experts.deepep_num_sms = getattr(module, "deepep_num_sms", 20)
        lora_experts.deepep_async_combine = getattr(module, "deepep_async_combine", False)
        lora_experts.deepep_native_exact = getattr(module, "deepep_native_exact", False)
        lora_experts.lora_serving_mode = getattr(module, "lora_serving_mode", None)
        lora_experts.alltoall_combine_hidden_chunk_size = getattr(module, "alltoall_combine_hidden_chunk_size", 0)
        lora_experts.swiglu_limit = float(getattr(module, "swiglu_limit", 0.0))
        if hasattr(module, "expert_lora_semantics"):
            lora_experts.expert_lora_semantics = module.expert_lora_semantics
        if hasattr(module, "native_mxfp4_payload"):
            lora_experts.add_module(
                "native_mxfp4_payload",
                module.native_mxfp4_payload,
            )
            lora_experts.fsdp_requires_full_precision = True

        base_weight = base_gate_up if base_gate_up is not None else module.gate_proj
        lora_experts = lora_experts.to(
            device=base_weight.device,
            dtype=base_weight.dtype,
        )
        with torch.no_grad():
            if base_gate_up is not None:
                lora_experts.gate_up_proj.copy_(base_gate_up)
            else:
                lora_experts.gate_proj.copy_(module.gate_proj)
                lora_experts.up_proj.copy_(module.up_proj)
            lora_experts.down_proj.copy_(module.down_proj)

        return lora_experts


def inject_lora_into_experts(
    block: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    hybrid_shared: bool = False,
) -> None:
    """Replace ``block.experts`` with a :class:`MoEExpertsLoRA` instance."""
    validate_gated_silu_expert_adapter_semantics(block.experts)
    if target_modules is None:
        target_modules = ["gate_proj", "up_proj", "down_proj"]

    lora_config = MoELoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        hybrid_shared=hybrid_shared,
    )

    gate_up_proj = getattr(block.experts, "gate_up_proj", None)
    num_local_experts = gate_up_proj.shape[0] if gate_up_proj is not None else block.experts.gate_proj.shape[0]
    hidden_dim = block.experts.hidden_dim
    intermediate_size = block.experts.intermediate_size
    moe_implementation = getattr(block.experts, "moe_implementation", "triton")

    lora_experts = MoEExpertsLoRA(
        num_experts=block.experts.num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        moe_implementation=moe_implementation,
        lora_config=lora_config,
        num_local_experts=num_local_experts,
        swiglu_limit=float(getattr(block.experts, "swiglu_limit", 0.0)),
    )
    lora_experts.act_fn = block.experts.act_fn
    lora_experts.ep_dispatch = getattr(block.experts, "ep_dispatch", "alltoall")
    lora_experts.deepep_buffer_size_gb = getattr(block.experts, "deepep_buffer_size_gb", 2.0)
    lora_experts.deepep_num_sms = getattr(block.experts, "deepep_num_sms", 20)
    lora_experts.deepep_async_combine = getattr(block.experts, "deepep_async_combine", False)
    lora_experts.deepep_native_exact = getattr(block.experts, "deepep_native_exact", False)
    lora_experts.lora_serving_mode = getattr(block.experts, "lora_serving_mode", None)
    lora_experts.alltoall_combine_hidden_chunk_size = getattr(block.experts, "alltoall_combine_hidden_chunk_size", 0)
    lora_experts.swiglu_limit = float(getattr(block.experts, "swiglu_limit", 0.0))
    if hasattr(block.experts, "expert_lora_semantics"):
        lora_experts.expert_lora_semantics = block.experts.expert_lora_semantics
    if hasattr(block.experts, "native_mxfp4_payload"):
        lora_experts.add_module(
            "native_mxfp4_payload",
            block.experts.native_mxfp4_payload,
        )
        lora_experts.fsdp_requires_full_precision = True

    base_weight = gate_up_proj if gate_up_proj is not None else block.experts.gate_proj
    lora_experts = lora_experts.to(
        device=base_weight.device,
        dtype=base_weight.dtype,
    )

    with torch.no_grad():
        if gate_up_proj is not None:
            lora_experts.gate_up_proj.copy_(gate_up_proj)
        else:
            lora_experts.gate_proj.copy_(block.experts.gate_proj)
            lora_experts.up_proj.copy_(block.experts.up_proj)
        lora_experts.down_proj.copy_(block.experts.down_proj)

    block.experts = lora_experts

    logger.debug(f"Injected MoE LoRA with r={r}, alpha={lora_alpha}, target_modules={target_modules}")


# ---------------------------------------------------------------------------
# Utility functions (kept for backward compat — used by lora/ module)
# ---------------------------------------------------------------------------


def copy_weights_to_lora_experts(source_experts: nn.Module, target_experts: nn.Module):
    """Copy base weights from source experts to LoRA experts."""
    with torch.no_grad():
        if hasattr(source_experts, "gate_up_proj") and hasattr(target_experts, "gate_up_proj"):
            target_experts.gate_up_proj.copy_(source_experts.gate_up_proj)
        else:
            target_experts.gate_proj.copy_(source_experts.gate_proj)
            target_experts.up_proj.copy_(source_experts.up_proj)
        target_experts.down_proj.copy_(source_experts.down_proj)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none"):
    """Mark only LoRA parameters as trainable."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = True


def lora_state_dict(model: nn.Module, bias: str = "none") -> dict:
    """Get state dict containing only LoRA parameters."""
    state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            state_dict[name] = param
        elif bias == "all" and "bias" in name:
            state_dict[name] = param
    return state_dict
