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

from dataclasses import dataclass, field
import math
from typing import Optional, List

import torch
import torch.nn as nn

from ....ops.group_gemm.kernel import compute_lora_scaling
from ....lora.modules.base import LoraModule
from ....utils import logging
from ..activations import ACT2FN

logger = logging.get_logger(__name__)


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



class MoEExpertsLoRA(LoraModule, nn.Module):
    """MoE experts with LoRA adapters.

    Handles all backends: eager (per-expert loop), triton/quack (group GEMM),
    and native (torch._grouped_mm). Base weights are frozen; only LoRA weights
    are trainable.

    All weights in (G, K, N) format — ``[num_experts, in_features, out_features]``.

    When Expert Parallelism is enabled, pass ``num_local_experts`` to create
    weights at the local (sharded) shape.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        moe_implementation: str = "triton",
        lora_config: Optional[MoELoRAConfig] = None,
        num_local_experts: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_local_experts if num_local_experts is not None else num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.moe_implementation = moe_implementation
        self.lora_config = lora_config or MoELoRAConfig()
        self.r = self.lora_config.r
        self.lora_alpha = self.lora_config.lora_alpha

        # Base weights (frozen) in (G, K, N) format
        self.gate_proj = nn.Parameter(
            torch.empty(self.num_experts, hidden_dim, intermediate_size),
            requires_grad=False,
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.num_experts, hidden_dim, intermediate_size),
            requires_grad=False,
        )
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

        self._create_lora_params("gate_proj", shared_exp, num_exp, r, hidden_dim, intermediate_size)
        self._create_lora_params("up_proj", shared_exp, num_exp, r, hidden_dim, intermediate_size)
        self._create_lora_params("down_proj", num_exp, (1 if hybrid else num_exp), r, intermediate_size, hidden_dim)

        # Scaling factor
        self.scaling = compute_lora_scaling(
            self.lora_config.lora_alpha,
            self.lora_config.r,
            self.lora_config.use_rslora,
        )

        self.reset_lora_parameters()

        # EP dispatch strategy (inherited from source MoEExperts via inject_lora)
        self.ep_dispatch: str = "alltoall"
        self.deepep_buffer_size_gb: float = 2.0
        self.deepep_num_sms: int = 20
        self.deepep_async_combine: bool = False

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
            self.register_buffer(f"{name}_lora_A", torch.zeros(A_experts, in_features, r))
            self.register_buffer(f"{name}_lora_B", torch.zeros(B_experts, r, out_features))

    def reset_lora_parameters(self):
        """Initialize LoRA weights: kaiming_uniform for A, zeros for B."""
        for name in self.lora_config.target_modules:
            lora_A = getattr(self, f"{name}_lora_A")
            lora_B = getattr(self, f"{name}_lora_B")
            if isinstance(lora_A, nn.Parameter):
                for i in range(lora_A.shape[0]):
                    nn.init.kaiming_uniform_(lora_A.data[i], a=math.sqrt(5))
                nn.init.zeros_(lora_B.data)

    def _compute_proj_delta(self, proj_name: str) -> torch.Tensor:
        """Compute LoRA delta for one projection. Returns [E, K, N] in GKN format."""
        lora_A = getattr(self, f"{proj_name}_lora_A")  # [1 or E, in, r]
        lora_B = getattr(self, f"{proj_name}_lora_B")  # [E or 1, r, out]
        E = max(lora_A.shape[0], lora_B.shape[0])
        A = lora_A.expand(E, -1, -1)  # [E, in, r]
        B = lora_B.expand(E, -1, -1)  # [E, r, out]
        return torch.bmm(A, B) * self.scaling  # [E, in, out] = [E, K, N]

    def merge_weights(self) -> None:
        """Merge LoRA weights into base weights for inference.

        After merging: weight = weight + delta_weight for each active projection.
        Resets LoRA parameters after merge.
        """
        with torch.no_grad():
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                if proj_name not in self.lora_config.target_modules:
                    continue
                base = getattr(self, proj_name)  # nn.Parameter [E, K, N]
                delta = self._compute_proj_delta(proj_name).to(base.dtype)
                base.add_(delta)
        self.reset_lora_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        selected_experts: torch.Tensor = None,
        expert_idx: int = None,
    ) -> torch.Tensor:
        """Forward pass with LoRA.

        For **eager**: called per-expert with ``expert_idx``.
        For **triton/quack/native**: checks EP first, falls back to local path.
        """
        if self.moe_implementation == "eager":
            assert expert_idx is not None
            return self._eager_lora_forward(hidden_states, expert_idx)

        # Check EP — use unified dispatch/compute/combine path
        from xorl.distributed.parallel_state import get_parallel_state
        parallel_state = get_parallel_state()

        if parallel_state.ep_enabled:
            return self._ep_forward(
                hidden_states, routing_weights, selected_experts, parallel_state
            )

        # Local path — registry-based
        from .backend import MOE_EXPERT_BACKENDS_LORA

        fn = MOE_EXPERT_BACKENDS_LORA[self.moe_implementation]
        return fn(
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            gate_proj=self.gate_proj,
            up_proj=self.up_proj,
            down_proj=self.down_proj,
            gate_proj_lora_A=self.gate_proj_lora_A,
            gate_proj_lora_B=self.gate_proj_lora_B,
            up_proj_lora_A=self.up_proj_lora_A,
            up_proj_lora_B=self.up_proj_lora_B,
            down_proj_lora_A=self.down_proj_lora_A,
            down_proj_lora_B=self.down_proj_lora_B,
            scaling=self.scaling,
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
        routes to the LoRA-aware EP compute registry.
        """
        from .backend import EP_DISPATCH, EP_COMBINE, EP_EXPERT_COMPUTE_LORA

        if self.moe_implementation not in EP_EXPERT_COMPUTE_LORA:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"EP with LoRA. Available: {list(EP_EXPERT_COMPUTE_LORA.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. "
                f"Available: {list(EP_DISPATCH.keys())}"
            )

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]
        compute_fn = EP_EXPERT_COMPUTE_LORA[self.moe_implementation]

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(
            hidden_states, routing_weights, selected_experts, parallel_state
        )
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)

        # Step 2: Expert computation with LoRA
        expert_output = compute_fn(
            permute_tokens, cumsum,
            self.gate_proj, self.up_proj, self.down_proj,
            self.gate_proj_lora_A, self.gate_proj_lora_B,
            self.up_proj_lora_A, self.up_proj_lora_B,
            self.down_proj_lora_A, self.down_proj_lora_B,
            self.scaling,
        )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(
            expert_output, ctx, dispatch_kwargs, parallel_state
        )
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
            from xorl.distributed.moe.deepep import get_default_buffer
            kwargs["buffer"] = get_default_buffer(
                ep_group=parallel_state.ep_group,
                buffer_size_gb=self.deepep_buffer_size_gb,
                num_sms=self.deepep_num_sms,
            )
            kwargs["num_local_experts"] = self.gate_proj.shape[0]
        return kwargs

    def _build_combine_kwargs(self, expert_output, ctx, dispatch_kwargs, parallel_state):
        """Build combine kwargs based on ep_dispatch strategy."""
        if self.ep_dispatch == "alltoall":
            return dict(expert_output=expert_output, ctx=ctx, ep_group=parallel_state.ep_group)
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

        if "gate_proj" in self.lora_config.target_modules:
            A = self.gate_proj_lora_A[min(expert_idx, self.gate_proj_lora_A.shape[0] - 1)].to(compute_dtype)
            B = self.gate_proj_lora_B[expert_idx].to(compute_dtype)
            gate_proj_out = gate_proj_out + torch.matmul(torch.matmul(hidden_states, A), B) * self.scaling

        if "up_proj" in self.lora_config.target_modules:
            A = self.up_proj_lora_A[min(expert_idx, self.up_proj_lora_A.shape[0] - 1)].to(compute_dtype)
            B = self.up_proj_lora_B[expert_idx].to(compute_dtype)
            up_proj_out = up_proj_out + torch.matmul(torch.matmul(hidden_states, A), B) * self.scaling

        out = self.act_fn(gate_proj_out) * up_proj_out

        down_out = torch.matmul(out, self.down_proj[expert_idx])
        if "down_proj" in self.lora_config.target_modules:
            A = self.down_proj_lora_A[expert_idx].to(compute_dtype)
            B = self.down_proj_lora_B[min(expert_idx, self.down_proj_lora_B.shape[0] - 1)].to(compute_dtype)
            down_out = down_out + torch.matmul(torch.matmul(out, A), B) * self.scaling

        return down_out

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_size={self.intermediate_size}, r={self.lora_config.r}, "
            f"lora_alpha={self.lora_config.lora_alpha}, "
            f"target_modules={self.lora_config.target_modules}"
        )

    @classmethod
    def from_module(cls, module: nn.Module, r: int, lora_alpha: int, **kwargs):
        """Create from an existing MoEExperts module, copying base weights."""
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

        num_exp = module.gate_proj.shape[0]
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
        )
        lora_experts.act_fn = module.act_fn
        lora_experts.ep_dispatch = getattr(module, "ep_dispatch", "alltoall")
        lora_experts.deepep_buffer_size_gb = getattr(module, "deepep_buffer_size_gb", 2.0)
        lora_experts.deepep_num_sms = getattr(module, "deepep_num_sms", 20)
        lora_experts.deepep_async_combine = getattr(module, "deepep_async_combine", False)

        lora_experts = lora_experts.to(
            device=module.gate_proj.device,
            dtype=module.gate_proj.dtype,
        )
        with torch.no_grad():
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
    if target_modules is None:
        target_modules = ["gate_proj", "up_proj", "down_proj"]

    lora_config = MoELoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        hybrid_shared=hybrid_shared,
    )

    num_local_experts = block.experts.gate_proj.shape[0]
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
    )
    lora_experts.act_fn = block.experts.act_fn
    lora_experts.ep_dispatch = getattr(block.experts, "ep_dispatch", "alltoall")
    lora_experts.deepep_buffer_size_gb = getattr(block.experts, "deepep_buffer_size_gb", 2.0)
    lora_experts.deepep_num_sms = getattr(block.experts, "deepep_num_sms", 20)
    lora_experts.deepep_async_combine = getattr(block.experts, "deepep_async_combine", False)

    lora_experts = lora_experts.to(
        device=block.experts.gate_proj.device,
        dtype=block.experts.gate_proj.dtype,
    )

    with torch.no_grad():
        lora_experts.gate_proj.copy_(block.experts.gate_proj)
        lora_experts.up_proj.copy_(block.experts.up_proj)
        lora_experts.down_proj.copy_(block.experts.down_proj)

    block.experts = lora_experts

    logger.debug(
        f"Injected MoE LoRA with r={r}, alpha={lora_alpha}, "
        f"target_modules={target_modules}"
    )


# ---------------------------------------------------------------------------
# Utility functions (kept for backward compat — used by lora/ module)
# ---------------------------------------------------------------------------


def copy_weights_to_lora_experts(source_experts: nn.Module, target_experts: nn.Module):
    """Copy base weights from source experts to LoRA experts."""
    with torch.no_grad():
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
