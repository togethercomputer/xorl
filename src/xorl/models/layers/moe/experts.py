"""MoE expert weight container with backend dispatch."""

import os
import torch
import torch.nn as nn

from ..activations import ACT2FN
from .backend import MOE_EXPERT_BACKENDS


def _flag_enabled(name: str) -> bool:
    v = os.environ.get(name, "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


_DEBUG_EP = _flag_enabled("XORL_DEBUG_EP")
_FORCE_SYNC = _flag_enabled("XORL_EP_FORCE_SYNC")


class MoEExperts(nn.Module):
    """Unified weight container for MoE experts.

    Holds stacked weight tensors ``[num_experts, ...]`` and dispatches
    ``forward()`` to the selected backend (eager / triton / native / quack).

    Weights are stored in ``(G, K, N)`` format — ``[num_experts, in_features, out_features]``::

        gate_proj: [num_experts, hidden_dim, intermediate_size]
        up_proj:   [num_experts, hidden_dim, intermediate_size]
        down_proj: [num_experts, intermediate_size, hidden_dim]

    Args:
        num_experts: Total number of experts.
        hidden_dim: Model hidden dimension.
        intermediate_size: Expert FFN intermediate dimension.
        hidden_act: Activation function name (default: ``"silu"``).
        moe_implementation: Backend name — ``"eager"``, ``"triton"``, ``"native"``, or ``"quack"``.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        moe_implementation: str = "triton",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.moe_implementation = moe_implementation

        self.gate_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_size),
            requires_grad=True,
        )
        self.up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_size),
            requires_grad=True,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[hidden_act]

        # EP dispatch strategy: "alltoall" (default) or "deepep" (NVLink-optimized)
        self.ep_dispatch: str = "alltoall"
        self.deepep_buffer_size_gb: float = 2.0
        self.deepep_num_sms: int = 20
        self.deepep_async_combine: bool = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        selected_experts: torch.Tensor = None,
        expert_idx: int = None,
    ) -> torch.Tensor:
        """Dispatch to the configured backend.

        For **triton/native/quack**: call with ``(hidden_states, routing_weights, selected_experts)``.
        For **eager**: called per-expert from ``MoEBlock._eager_forward()`` with ``expert_idx``.

        When Expert Parallelism is enabled, all backends (triton/native/quack)
        use the unified dispatch → compute → combine path via ``_ep_forward()``.
        """
        fn = MOE_EXPERT_BACKENDS[self.moe_implementation]

        if self.moe_implementation == "eager":
            assert expert_idx is not None
            return fn(
                hidden_states,
                expert_idx,
                self.gate_proj,
                self.up_proj,
                self.down_proj,
                self.act_fn,
            )

        # Check EP — use unified dispatch/compute/combine path
        from xorl.distributed.parallel_state import get_parallel_state
        parallel_state = get_parallel_state()

        if parallel_state.ep_enabled:
            return self._ep_forward(
                hidden_states, routing_weights, selected_experts, parallel_state
            )

        # Local single-GPU path
        return fn(
            hidden_states,
            routing_weights,
            selected_experts,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            num_experts=self.num_experts,
        )

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Unified EP forward: dispatch → compute → combine.

        All backends share the same dispatch/combine logic. Only the
        expert compute step (group GEMM) differs per backend.

        Dispatch strategy is selected by ``self.ep_dispatch`` (``"alltoall"``
        or ``"deepep"``). Compute backend by ``self.moe_implementation``.
        """
        from .backend import EP_DISPATCH, EP_COMBINE, EP_EXPERT_COMPUTE

        if self.moe_implementation not in EP_EXPERT_COMPUTE:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"Expert Parallelism. Available: {list(EP_EXPERT_COMPUTE.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. "
                f"Available: {list(EP_DISPATCH.keys())}"
            )

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]
        compute_fn = EP_EXPERT_COMPUTE[self.moe_implementation]

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(
            hidden_states, routing_weights, selected_experts, parallel_state
        )

        if _DEBUG_EP:
            return self._ep_forward_debug(
                dispatch_fn, combine_fn, compute_fn, dispatch_kwargs, parallel_state,
            )

        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)

        if _FORCE_SYNC:
            torch.cuda.synchronize()

        # Step 2: Expert computation (backend-specific GEMM only)
        expert_output = compute_fn(
            permute_tokens, cumsum,
            self.gate_proj, self.up_proj, self.down_proj,
        )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(
            expert_output, ctx, dispatch_kwargs, parallel_state
        )
        return combine_fn(**combine_kwargs)

    def _ep_forward_debug(self, dispatch_fn, combine_fn, compute_fn, dispatch_kwargs, parallel_state):
        """Instrumented EP forward with per-phase CUDA event timing.

        Enable via XORL_DEBUG_EP=1.  Prints dispatch/compute/combine wall
        times plus tensor metadata to help diagnose performance gaps between
        different dispatch+compute backend combinations.
        """
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

        # --- dispatch ---
        ev[0].record()
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)
        ev[1].record()

        # --- compute ---
        ev[2].record()
        expert_output = compute_fn(
            permute_tokens, cumsum,
            self.gate_proj, self.up_proj, self.down_proj,
        )
        ev[3].record()

        # --- combine ---
        combine_kwargs = self._build_combine_kwargs(
            expert_output, ctx, dispatch_kwargs, parallel_state
        )
        ev[4].record()
        result = combine_fn(**combine_kwargs)
        ev[5].record()

        torch.cuda.synchronize()
        t_dispatch = ev[0].elapsed_time(ev[1])
        t_compute = ev[2].elapsed_time(ev[3])
        t_combine = ev[4].elapsed_time(ev[5])

        print(
            f"[EP DEBUG r{rank}] dispatch={self.ep_dispatch} compute={self.moe_implementation}\n"
            f"  hidden_states: {dispatch_kwargs['hidden_states'].shape}\n"
            f"  permute_tokens: shape={permute_tokens.shape}, dtype={permute_tokens.dtype}, "
            f"contiguous={permute_tokens.is_contiguous()}, "
            f"stride={permute_tokens.stride()}, data_ptr_mod4k={permute_tokens.data_ptr() % 4096}\n"
            f"  cumsum: shape={cumsum.shape}, dtype={cumsum.dtype}\n"
            f"  gate_proj: shape={self.gate_proj.shape}, "
            f"contiguous={self.gate_proj.is_contiguous()}, stride={self.gate_proj.stride()}\n"
            f"  expert_output: shape={expert_output.shape}\n"
            f"  --- Timing (ms) ---\n"
            f"  Dispatch: {t_dispatch:8.2f}\n"
            f"  Compute:  {t_compute:8.2f}\n"
            f"  Combine:  {t_combine:8.2f}\n"
            f"  Total:    {t_dispatch + t_compute + t_combine:8.2f}",
            flush=True,
        )
        return result

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

    @classmethod
    def from_config(cls, config, moe_implementation: str = "triton"):
        """Create from a model config (e.g. ``Qwen3MoeConfig``)."""
        return cls(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            moe_implementation=moe_implementation,
        )
