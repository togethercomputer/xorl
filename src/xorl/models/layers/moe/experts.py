"""MoE expert weight container with backend dispatch."""

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from ..activations import ACT2FN
from .backend import (
    EP_COMBINE,
    EP_DISPATCH,
    EP_EXPERT_COMPUTE,
    MOE_EXPERT_BACKENDS,
)
from .common import split_gate_up_proj


logger = logging.getLogger(__name__)

_MOE_SGLANG_FUSED_EXPERTS_ENV = "XORL_MOE_SGLANG_FUSED_EXPERTS"


def _flag_enabled(name: str) -> bool:
    v = os.environ.get(name, "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


_DEBUG_EP = _flag_enabled("XORL_DEBUG_EP")
_FORCE_SYNC = _flag_enabled("XORL_EP_FORCE_SYNC")


def moe_sglang_fused_experts_enabled() -> bool:
    """Return whether the explicit serving-kernel forward contract is enabled.

    The opt-in is intentionally not silent: unsupported topologies and expert
    variants raise at the call site instead of falling back to a second
    forward implementation.
    """
    return _flag_enabled(_MOE_SGLANG_FUSED_EXPERTS_ENV)


def _sglang_fused_experts_kernel_call(
    hidden_flat: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    routing_flat: torch.Tensor,
    selected_flat: torch.Tensor,
    fused_experts_impl,
    activation: str,
    swiglu_limit: float,
    gate_up_bias: torch.Tensor | None,
) -> torch.Tensor:
    """The exact bf16 tp1/ep1 serving-kernel call (shared by the inference and
    autograd paths — the serving tree defines K3, so this must never fork).

    Presents xorl GKN weights (``gate_up_proj [E, H, 2I]`` gate-first /
    ``down_proj [E, I, H]``) as zero-copy transpose views in SGLang's
    ``w13 [E, 2I, H]`` / ``w2 [E, H, I]`` layout. The local vendored
    orchestration accepts these strides while preserving SGLang's launches and
    reduction tree. Top-k weights are upcast exactly to fp32.
    """
    w13 = gate_up_proj.transpose(1, 2)
    w2 = down_proj.transpose(1, 2)
    b1 = gate_up_bias.contiguous() if gate_up_bias is not None else None

    MoEExperts._log_sglang_fused_experts_config_once(w13, w2, selected_flat)

    return fused_experts_impl(
        hidden_flat.contiguous(),
        w13,
        w2,
        routing_flat.to(torch.float32).contiguous(),
        selected_flat.contiguous(),
        b1=b1,
        b2=None,
        inplace=False,
        activation=activation,
        is_gated=True,
        apply_router_weight_on_input=False,
        no_combine=False,
        routed_scaling_factor=None,
        gemm1_alpha=None,
        gemm1_limit=swiglu_limit if swiglu_limit > 0 else None,
        # tp1/ep1 serving contract (num_experts == num_local_experts).
        filter_expert=False,
    )


class _SglangFusedExpertsTrainFunction(torch.autograd.Function):
    """Trainable wrapper for the serving-kernel MoE forward.

    forward: SGLang's ``fused_experts_impl`` — numerically identical to the
    inference path (the serving reduction tree defines K3).
    backward: xorl's proven grouped-GEMM MoE backward
    (:class:`xorl.ops.moe.triton.TritonMoeExpertsFunction` math): the cheap
    intermediates (scatter bookkeeping + gate/up GEMM) are recomputed with
    xorl's kernels from the saved inputs, then dgrad/wgrad grouped GEMMs, the
    activation backward, and the weighted-combine backward (including
    d(topk_weights) so the router trains). Backward does not need cross-engine
    parity — only correctness; weight grads come out directly in GKN layout.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_flat: torch.Tensor,
        routing_flat: torch.Tensor,
        selected_flat: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        fused_experts_impl,
        activation: str,
        hidden_act: str,
        swiglu_limit: float,
        num_experts: int,
    ) -> torch.Tensor:
        from xorl.ops.group_gemm.kernel.moe import (  # noqa: PLC0415
            expert_histogram,
            moe_index_compute,
        )

        output = _sglang_fused_experts_kernel_call(
            hidden_flat,
            gate_up_proj,
            down_proj,
            routing_flat,
            selected_flat,
            fused_experts_impl,
            activation,
            swiglu_limit,
            gate_up_bias=None,
        )
        # Save the xorl scatter bookkeeping alongside the inputs (mirrors the
        # stock TritonMoeExpertsFunction contract): moe_index_compute uses
        # relaxed atomics, so the intra-expert row permutation is only
        # reproducible if backward reuses the one drawn here.
        splits = expert_histogram(selected_flat, int(num_experts))
        cumsum_t = torch.cumsum(splits, dim=0)
        scatter_index = moe_index_compute(selected_flat, cumsum_t)
        ctx.save_for_backward(
            hidden_flat, routing_flat, selected_flat, gate_up_proj, down_proj, cumsum_t, scatter_index
        )
        ctx.hidden_act = hidden_act
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        ctx.num_experts = int(num_experts)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from xorl.ops.group_gemm.kernel.group_gemm import (  # noqa: PLC0415
            group_gemm_same_mn,
            group_gemm_same_nk,
        )
        from xorl.ops.group_gemm.kernel.moe import moe_scatter  # noqa: PLC0415
        from xorl.ops.moe.triton import _moe_gate_activation, _moe_gate_activation_backward  # noqa: PLC0415

        (
            hidden_states,
            gate_weights,
            expert_index,
            gate_up_proj,
            down_proj,
            cumsum_t,
            scatter_index,
        ) = ctx.saved_tensors

        # Recompute the cheap xorl-forward intermediates the stock backward
        # consumes (scatter + gate/up grouped GEMM; the down GEMM is not needed)
        # from the bookkeeping saved in forward.
        scatter_output = moe_scatter(hidden_states, scatter_index)
        max_M = scatter_output.shape[0]
        gate_up_output = group_gemm_same_nk(
            a=scatter_output,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
        )
        intermediate = gate_up_output.shape[-1] // 2
        gate_output = gate_up_output[..., :intermediate]
        up_output = gate_up_output[..., intermediate:]

        # From here on: verbatim TritonMoeExpertsFunction.backward math (gated),
        # with routing weights kept in the activation dtype for the grouped GEMMs.
        compute_dtype = gate_up_output.dtype
        reshaped_gate_weight = gate_weights.reshape(-1, 1).to(compute_dtype)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        grad_output = grad_output.view(-1, grad_output.shape[-1]).to(compute_dtype)

        gate_activation = _moe_gate_activation(gate_output, ctx.hidden_act)
        gated_activation = gate_activation * up_output
        gated_weighted = gated_activation * scattered_gate_weight

        grad_down_output = moe_scatter(grad_output, scatter_index)

        grad_gated_weighted = group_gemm_same_nk(
            a=grad_down_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
            )
        del grad_down_output, gated_weighted

        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape).to(gate_weights.dtype)
        del gated_activation, grad_gated_weighted

        grad_up_output = gate_activation * grad_gated_activation
        grad_gate_activation = grad_gated_activation * up_output
        del grad_gated_activation, gate_activation, up_output
        grad_gate_output = _moe_gate_activation_backward(grad_gate_activation, gate_output, ctx.hidden_act)
        del grad_gate_activation, gate_output

        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output

        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
            )

        grad_scatter_output = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )
        del grad_gate_up_act, scatter_output

        grad_hidden_states = (
            grad_scatter_output[scatter_index.flatten()]
            .reshape(hidden_states.shape[0], scatter_index.shape[1], -1)
            .sum(dim=1)
        )

        return (
            grad_hidden_states,  # hidden_flat
            grad_gate_weight,  # routing_flat
            None,  # selected_flat
            grad_gate_up_proj,  # gate_up_proj
            grad_down_proj,  # down_proj
            None,  # fused_experts_impl
            None,  # activation
            None,  # hidden_act
            None,  # swiglu_limit
            None,  # num_experts
        )


class MoEExperts(nn.Module):
    """Unified weight container for MoE experts.

    Holds stacked weight tensors ``[num_experts, ...]`` and dispatches
    ``forward()`` to the selected backend (eager / triton / native / quack).

    Weights are stored in ``(G, K, N)`` format — ``[num_experts, in_features, out_features]``::

        gate_up_proj: [num_experts, hidden_dim, 2 * intermediate_size]
        down_proj:    [num_experts, intermediate_size, hidden_dim]

    ``gate_proj`` and ``up_proj`` are exposed as views into ``gate_up_proj``
    for compatibility with existing backends and helpers.

    Optional per-expert biases (``gate_up_bias``, ``down_bias``) default to
    ``None`` and can be set by model-specific code (e.g. GPT-OSS).

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

        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, 2 * intermediate_size),
            requires_grad=True,
        )
        self.gate_up_proj._fused_gate_up = True
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[hidden_act]
        # String kind used by triton/native/quack backends (avoids name-sniffing).
        from xorl.ops.moe.triton import normalize_hidden_act  # noqa: PLC0415

        self.hidden_act = normalize_hidden_act(hidden_act)

        # Optional per-expert biases (e.g. GPT-OSS). Set to actual tensors
        # by model-specific code; None means no bias.
        self.gate_up_bias = None
        self.down_bias = None

        # EP dispatch strategy: "alltoall" (default) or "deepep" (NVLink-optimized)
        self.ep_dispatch: str = "alltoall"
        self.deepep_buffer_size_gb: float = 2.0
        self.deepep_num_sms: int = 20
        self.deepep_async_combine: bool = False

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

    @staticmethod
    def _ensure_sglang_server_args() -> None:
        """Install deterministic defaults when called outside an SGLang server."""
        try:
            from sglang.srt.server_args import (  # noqa: PLC0415
                ServerArgs,
                get_global_server_args,
                set_global_server_args_for_scheduler,
            )
        except ImportError as exc:
            raise ImportError("SGLang and sgl_kernel are required") from exc

        try:
            get_global_server_args()
        except ValueError:
            server_args = ServerArgs(model_path=os.environ.get("XORL_SGLANG_MOE_MODEL_PATH", "dummy"))
            server_args.enable_deterministic_inference = True
            server_args.enable_fused_moe_sum_all_reduce = False
            server_args.rl_on_policy_target = "xorl-batch-invariant"
            set_global_server_args_for_scheduler(server_args)

    @staticmethod
    def _load_sglang_fused_experts_impl():
        try:
            from xorl.ops.moe.sglang_fused_moe_strided import fused_experts_impl_strided  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("SGLang and sgl_kernel are required") from exc
        MoEExperts._ensure_sglang_server_args()
        return fused_experts_impl_strided

    def sglang_fused_experts_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Run the serving expert forward while retaining XoRL's backward.

        Equality is required only for the forward that produces token scores.
        Under autograd, :class:`_SglangFusedExpertsTrainFunction` recomputes the
        ordinary grouped-GEMM intermediates and returns gradients for inputs,
        routing weights, and both GKN expert parameters.
        """
        if routing_weights is None or selected_experts is None:
            raise ValueError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires routing weights and expert ids")
        if self.gate_up_bias is not None or self.down_bias is not None:
            raise NotImplementedError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support expert biases")
        if self.hidden_act not in {"silu", "gelu_tanh"}:
            raise NotImplementedError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support hidden_act={self.hidden_act!r}"
            )

        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act

        try:
            fused_experts_impl = self._load_sglang_fused_experts_impl()
        except ImportError as exc:
            raise ImportError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires SGLang and sgl_kernel") from exc

        needs_grad = torch.is_grad_enabled() and (
            hidden_flat.requires_grad
            or routing_flat.requires_grad
            or self.gate_up_proj.requires_grad
            or self.down_proj.requires_grad
        )
        if needs_grad:
            output = _SglangFusedExpertsTrainFunction.apply(
                hidden_flat,
                routing_flat,
                selected_flat,
                self.gate_up_proj,
                self.down_proj,
                fused_experts_impl,
                activation,
                self.hidden_act,
                0.0,
                self.num_experts,
            )
        else:
            output = _sglang_fused_experts_kernel_call(
                hidden_flat,
                self.gate_up_proj,
                self.down_proj,
                routing_flat,
                selected_flat,
                fused_experts_impl,
                activation,
                0.0,
                None,
            )
        return output.reshape(original_shape)

    @staticmethod
    def _log_sglang_fused_experts_config_once(
        w13: torch.Tensor,
        w2: torch.Tensor,
        selected_flat: torch.Tensor,
    ) -> None:
        """Log the equality-critical launch shape once for auditability."""
        if getattr(MoEExperts, "_sglang_fused_experts_config_logged", False):
            return
        MoEExperts._sglang_fused_experts_config_logged = True
        logger.info(
            "[%s] fused_experts_impl E=%d N=%d K=%d M=%d topk=%d w13_stride=%s w2_stride=%s",
            _MOE_SGLANG_FUSED_EXPERTS_ENV,
            int(w13.shape[0]),
            int(w13.shape[1]),
            int(w13.shape[2]),
            int(selected_flat.shape[0]),
            int(selected_flat.shape[1]),
            tuple(w13.stride()),
            tuple(w2.stride()),
        )

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
        if self.moe_implementation == "eager":
            fn = MOE_EXPERT_BACKENDS[self.moe_implementation]
            assert expert_idx is not None
            return fn(
                hidden_states,
                expert_idx,
                self.gate_proj.contiguous(),
                self.up_proj.contiguous(),
                self.down_proj,
                hidden_act=self.hidden_act,
                gate_up_bias=self.gate_up_bias,
                down_bias=self.down_bias,
            )

        # Check EP — use unified dispatch/compute/combine path
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()

        if parallel_state.ep_enabled:
            return self._ep_forward(hidden_states, routing_weights, selected_experts, parallel_state)

        # Local single-GPU path
        gate_proj = self.gate_proj.contiguous()
        up_proj = self.up_proj.contiguous()
        fn = MOE_EXPERT_BACKENDS[self.moe_implementation]

        return fn(
            hidden_states,
            routing_weights,
            selected_experts,
            gate_proj,
            up_proj,
            self.down_proj,
            num_experts=self.num_experts,
            hidden_act=self.hidden_act,
            gate_up_proj=self.gate_up_proj,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
        )

    @torch.compiler.disable
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

        if self.moe_implementation not in EP_EXPERT_COMPUTE:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"Expert Parallelism. Available: {list(EP_EXPERT_COMPUTE.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. Available: {list(EP_DISPATCH.keys())}"
            )

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]

        compute_fn = EP_EXPERT_COMPUTE[self.moe_implementation]

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(hidden_states, routing_weights, selected_experts, parallel_state)

        if _DEBUG_EP:
            return self._ep_forward_debug(
                dispatch_fn,
                combine_fn,
                compute_fn,
                dispatch_kwargs,
                parallel_state,
            )

        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)

        if _FORCE_SYNC:
            torch.cuda.synchronize()

        # Warmup: pre-compile all backward GEMM kernel variants to avoid
        # first-use compilation memory spikes during training.
        if not getattr(type(self), "_kernel_warmed_up", False):
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn as _warmup_mn  # noqa: PLC0415
            from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk as _warmup_gemm  # noqa: PLC0415

            _d = permute_tokens.device
            _dt = permute_tokens.dtype
            _H = self.gate_up_proj.shape[1]
            _I = self.intermediate_size
            _E = self.gate_up_proj.shape[0]
            _M = _E * 2
            _cum = torch.arange(2, _M + 2, 2, dtype=torch.int32, device=_d)

            # Forward GEMM: x @ gate_up_proj
            _x = torch.zeros(_M, _H, dtype=_dt, device=_d)
            _w = torch.zeros(_E, _H, 2 * _I, dtype=_dt, device=_d)
            _warmup_gemm(a=_x, b=_w, cumsum_M=_cum, max_M=2)

            # Backward dgrad FC1: grad_gate_up_act @ gate_up_proj^T
            _g = torch.zeros(_M, 2 * _I, dtype=_dt, device=_d)
            _warmup_gemm(a=_g, b=_w, cumsum_M=_cum, max_M=2, transpose_b=True)

            # Backward dgrad FC2: grad @ down_proj^T
            _wd = torch.zeros(_E, _I, _H, dtype=_dt, device=_d)
            _gd = torch.zeros(_M, _I, dtype=_dt, device=_d)
            _warmup_gemm(a=_gd, b=_wd, cumsum_M=_cum, max_M=2, transpose_b=True)

            # Backward wgrad FC1: permute_tokens^T @ grad_gate_up_act
            _c = torch.zeros(_E, _H, 2 * _I, dtype=_dt, device=_d)
            _warmup_mn(a=_x, b=_g, c=_c, cumsum_K=_cum, max_K=2, transpose_a=True)

            del _x, _w, _g, _gd, _wd, _c, _cum
            torch.cuda.empty_cache()
            type(self)._kernel_warmed_up = True

        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        expert_output = compute_fn(
            permute_tokens,
            cumsum,
            self.gate_up_proj,
            self.down_proj,
            self.intermediate_size,
            expert_scores,
            hidden_act=self.hidden_act,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
        )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
        return combine_fn(**combine_kwargs)

    def _ep_forward_debug(self, dispatch_fn, combine_fn, compute_fn, dispatch_kwargs, parallel_state):
        """Instrumented EP forward with per-phase CUDA event timing.

        Enable via XORL_DEBUG_EP=1.  Prints dispatch/compute/combine wall
        times plus tensor metadata to help diagnose performance gaps between
        different dispatch+compute backend combinations.
        """

        rank = dist.get_rank() if dist.is_initialized() else 0

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

        # --- dispatch ---
        ev[0].record()
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)
        ev[1].record()

        # --- compute ---
        ev[2].record()
        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        expert_output = compute_fn(
            permute_tokens,
            cumsum,
            self.gate_up_proj,
            self.down_proj,
            self.intermediate_size,
            expert_scores,
            hidden_act=self.hidden_act,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
        )
        ev[3].record()

        # --- combine ---
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
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
