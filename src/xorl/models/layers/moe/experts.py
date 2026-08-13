"""MoE expert weight container with backend dispatch."""

import hashlib
import json
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
    EP_EXPERT_COMPUTE_MOE_ACT,
    MOE_EXPERT_BACKENDS,
)
from .common import split_gate_up_proj


logger = logging.getLogger(__name__)

_MOE_SGLANG_FUSED_EXPERTS_ENV = "XORL_MOE_SGLANG_FUSED_EXPERTS"
_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE_ENV = "XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE"
_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODES = ("transient", "cached", "strided")


def _flag_enabled(name: str) -> bool:
    v = os.environ.get(name, "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


_DEBUG_EP = _flag_enabled("XORL_DEBUG_EP")
_FORCE_SYNC = _flag_enabled("XORL_EP_FORCE_SYNC")
_FORCE_QUACK_DEEPEP_GENERIC = _flag_enabled("XORL_QUACK_DEEPEP_FORCE_GENERIC")
_DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS: dict[int, int] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


_MOE_SGLANG_FUSED_EXPERTS_AUTO_LOGGED = False
_MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE: bool | None = None


def _moe_sglang_fused_experts_env_state() -> bool | None:
    """Tri-state read of the parity flag: True/False when set, None (auto) when unset/blank."""
    raw = os.environ.get(_MOE_SGLANG_FUSED_EXPERTS_ENV)
    if raw is None or not raw.strip():
        return None
    return _flag_enabled(_MOE_SGLANG_FUSED_EXPERTS_ENV)


def _moe_sglang_fused_experts_stack_available() -> bool:
    """Auto-mode gate: whether the serving kernel stack imports (probed once per process)."""
    global _MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE
    if _MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE is None:
        try:
            MoEExperts._load_sglang_fused_experts_impl()
        except ImportError:
            _MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE = False
        else:
            _MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE = True
    return _MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE


def _log_moe_sglang_fused_experts_auto_once(message: str, warn: bool = False) -> None:
    """Log the auto resolution once per process (explicit 1/0 logs nothing)."""
    global _MOE_SGLANG_FUSED_EXPERTS_AUTO_LOGGED
    if _MOE_SGLANG_FUSED_EXPERTS_AUTO_LOGGED:
        return
    _MOE_SGLANG_FUSED_EXPERTS_AUTO_LOGGED = True
    (logger.warning if warn else logger.info)("[%s] %s", _MOE_SGLANG_FUSED_EXPERTS_ENV, message)


def moe_sglang_fused_experts_enabled(
    ep_size: int | None = None,
    device: "torch.device | str | None" = None,
    experts: "MoEExperts | None" = None,
) -> bool:
    """K3 parity mode: run the local (no-EP) MoE through SGLang's serving kernel.

    ``XORL_MOE_SGLANG_FUSED_EXPERTS`` routes expert compute through SGLang's
    ``fused_experts_impl`` (the exact bf16 tp1/ep1 serving path) so both
    engines share one MoE reduction tree at full bf16 speed. ``1`` forces the
    path on (loud failures preserved), ``0`` forces it off (the stock-triton
    escape hatch); unset resolves per-regime, logged once per process:

    - ``ep_size == 1``: auto-ENABLED — bit-exact to serving AND faster than the
      stock triton path at training scale (measured −1.67% step time on q30).
      Auto additionally requires a CUDA input, a module accepted by
      :meth:`MoEExperts.sglang_fused_experts_auto_supported`, and an importable
      sglang + sgl_kernel stack; otherwise the stock path is kept (explicit
      ``1`` keeps the loud ImportError / NotImplementedError instead).
    - ``ep_size > 1``: auto-DISABLED — measured +2.71% step time at EP8, and
      serving's reduction chain has not been reproduced on DeepEP's combine
      (explicit ``1`` under EP still requires ``ep_dispatch='alltoall'``).

    ``ep_size=None`` falls back to the global parallel state; ``device=None``
    falls back to ``torch.cuda.is_available()``.
    """
    explicit = _moe_sglang_fused_experts_env_state()
    if explicit is not None:
        return explicit
    if ep_size is None:
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        ep_size = get_parallel_state().ep_size
    if ep_size != 1:
        _log_moe_sglang_fused_experts_auto_once(
            f"sglang-fused experts auto-disabled (ep={ep_size}): EP keeps the configured expert backend; "
            f"set {_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 to opt in (alltoall dispatch only)"
        )
        return False
    if device is not None:
        if torch.device(device).type != "cuda":
            # CPU/meta forwards (unit tests, tracing) keep the stock path quietly.
            return False
    elif not torch.cuda.is_available():
        return False
    if experts is not None:
        auto_supported = getattr(experts, "sglang_fused_experts_auto_supported", None)
        if auto_supported is None or not auto_supported():
            _log_moe_sglang_fused_experts_auto_once(
                "sglang-fused experts auto-disabled (ep=1): expert module outside the validated parity "
                "envelope (needs gated silu/gelu_tanh, no biases, no FP8; LoRA experts require "
                f"their model-owned exact_merged_forward property); set {_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 to force"
            )
            return False
    if not _moe_sglang_fused_experts_stack_available():
        _log_moe_sglang_fused_experts_auto_once(
            "sglang-fused experts auto-disabled (ep=1): sglang/sgl_kernel not importable — the MoE forward "
            "stays on the stock triton tree and is NOT bit-exact to serving; put the SGLang python tree on "
            f"PYTHONPATH, or set {_MOE_SGLANG_FUSED_EXPERTS_ENV}=0 to pin the stock path",
            warn=True,
        )
        return False
    _log_moe_sglang_fused_experts_auto_once(
        "sglang-fused experts auto-enabled (ep=1): the MoE expert forward runs SGLang's serving kernel "
        f"(bit-exact to serving, faster than the stock triton path); set {_MOE_SGLANG_FUSED_EXPERTS_ENV}=0 "
        "to restore the stock tree"
    )
    return True


def moe_sglang_fused_experts_weight_mode() -> str:
    """How the parity mode presents xorl's GKN weights to the serving kernel.

    ``XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE`` selects one of:

    - ``strided`` (default): zero-copy — pass ``transpose(1, 2)`` VIEWS of the
      GKN tensors to the vendored strided orchestration
      (:func:`xorl.ops.moe.sglang_fused_moe_strided.fused_experts_impl_strided`),
      which launches the SAME sglang triton kernels with the view's strides.
      Bit-identical to ``transient``/``cached`` (strides only change addressing,
      never the accumulation tree), fastest at replay and training scale, zero
      extra memory;
    - ``transient``: transpose-copy ``w13``/``w2`` per forward (~4.9 ms per q30
      layer; the fallback if the vendored orchestration cannot bind to the
      sglang tree on PYTHONPATH after an upstream restructure);
    - ``cached``: keep the transposed copies alive on the module (fast, but
      ~1.1 GiB per q30 layer). Cache entries key the source parameter's pointer,
      version, shape, and dtype; FSDP2 callers must explicitly invalidate after
      optimizer steps because an unsharded buffer can be rematerialized at the
      same address with a fresh version counter.
    """
    raw = os.environ.get(_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE_ENV, "").strip().lower()
    if raw:
        if raw not in _MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODES:
            raise ValueError(
                f"Invalid {_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE_ENV}={raw!r}; "
                f"expected one of {', '.join(_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODES)}"
            )
        return raw
    return "strided"


def _validate_sglang_swiglu_limit(swiglu_limit: float | None) -> None:
    """Reject XoRL's positive SwiGLU clamp on SGLang fused-MoE paths.

    The two arguments are not interchangeable: XoRL ``swiglu_limit`` clamps
    only the gate pre-activation symmetrically, while SGLang ``gemm1_limit``
    caps the post-SiLU gate and clamps the up branch symmetrically. Forwarding
    a positive value would silently change the model's activation program.
    """
    if swiglu_limit is not None and float(swiglu_limit) > 0:
        raise NotImplementedError(
            "SGLang fused MoE paths do not support positive swiglu_limit: XoRL clamps only the gate "
            "pre-activation symmetrically, while SGLang gemm1_limit caps the post-SiLU gate and clamps "
            "the up branch symmetrically. Set swiglu_limit=0 or use a XoRL-native expert backend."
        )


def _scale_moe_grad_by_fp32_routing(
    grad_output: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply serving's routing multiply before the BF16 gradient boundary.

    SGLang consumes FP32 top-k weights and multiplies them into the FP32 down
    accumulator before storing BF16.  The corresponding backward therefore
    multiplies the upstream gradient by the unrounded FP32 routing weight and
    casts the product once for XoRL's BF16 grouped-GEMM kernels.  Casting the
    routing weight first changes dX, dW13, and dW2 for ordinary GLM-5.2 scores.
    """
    if grad_output.shape[0] != routing_weights.numel():
        raise ValueError(
            "routing-gradient row mismatch: "
            f"grad_output has {grad_output.shape[0]} rows, routing_weights has {routing_weights.numel()} values"
        )
    scaled_fp32 = grad_output.to(torch.float32) * routing_weights.reshape(-1, 1).to(torch.float32)
    return scaled_fp32.to(grad_output.dtype)


def _group_gemm_same_nk_fp32_accumulator(
    group_gemm_same_nk,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    max_M: int,
) -> torch.Tensor:
    """Run the BF16 grouped GEMM while retaining its FP32 accumulator.

    ``group_gemm_same_nk`` can store a fresh output in FP32 while leaving its
    input and dot-product program unchanged.  A fresh output is essential:
    using the API's ``c=`` accumulation mode would let Triton autotuning replay
    and accumulate the first invocation repeatedly.  This is needed only for
    d(routing), whose forward operand is SGLang's pre-cast FP32 down accumulator.
    """
    return group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum_M,
        max_M=max_M,
        output_dtype=torch.float32,
    )


def _cached_serving_transpose(cache: dict | None, name: str, param: torch.Tensor) -> torch.Tensor:
    if cache is None:
        return param.transpose(1, 2).contiguous()
    key = (param.data_ptr(), param._version, tuple(param.shape), param.dtype)
    entry = cache.get(name)
    if entry is not None and entry[0] == key:
        return entry[1]
    transposed = param.transpose(1, 2).contiguous()
    cache[name] = (key, transposed)
    return transposed


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
    weight_cache: dict | None = None,
    filter_expert: bool = False,
) -> torch.Tensor:
    """The exact bf16 tp1/ep1 serving-kernel call (shared by the inference and
    autograd paths — the serving tree defines K3, so this must never fork).

    Presents xorl GKN weights (``gate_up_proj [E, H, 2I]`` gate-first /
    ``down_proj [E, I, H]``) in SGLang's ``w13 [E, 2I, H]`` / ``w2 [E, H, I]``
    layout per :func:`moe_sglang_fused_experts_weight_mode` — a transient
    transpose-copy, the ``weight_cache`` copy, or a zero-copy transpose-view
    (strided mode; ``fused_experts_impl`` must then be the strided variant) —
    and consumes fp32 top-k weights (StandardTopKOutput contract; the upcast
    is exact for bf16/fp16 routing weights).
    """
    _validate_sglang_swiglu_limit(swiglu_limit)
    if moe_sglang_fused_experts_weight_mode() == "strided":
        w13 = gate_up_proj.transpose(1, 2)
        w2 = down_proj.transpose(1, 2)
    else:
        w13 = _cached_serving_transpose(weight_cache, "w13", gate_up_proj)
        w2 = _cached_serving_transpose(weight_cache, "w2", down_proj)
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
        gemm1_limit=None,
        gate_up_interleaved=False,
        # tp1/ep1 serving contract (num_experts == num_local_experts) unless the
        # EP-combine simulation passes locally-mapped ids with -1 masking.
        filter_expert=filter_expert,
    )


class _SglangFusedExpertsTrainFunction(torch.autograd.Function):
    """Trainable wrapper for the serving-kernel MoE forward.

    forward: SGLang's ``fused_experts_impl`` — numerically identical to the
    inference path (the serving reduction tree defines K3).
    backward: xorl's grouped-GEMM MoE backward, with the serving forward's
    FP32-routing-after-down cast boundary preserved: the cheap
    intermediates (scatter bookkeeping + gate/up GEMM) are recomputed with
    xorl's kernels from the saved inputs, then dgrad/wgrad grouped GEMMs, the
    activation backward, and the weighted-combine backward (including
    d(topk_weights) so the router trains). Backward does not need cross-engine
    parity — only correctness; weight grads come out directly in GKN layout.

    ``filter_expert=True`` (EP-combine partials, P5): ids may contain
    ``-1`` for slots owned by other EP ranks. The forward kernel filters them
    natively; the bookkeeping/backward compacts the valid (token, slot) pairs
    into a dense topk=1 sub-problem (the bookkeeping kernels are not -1-safe),
    runs the stock math there, and scatters d(x)/d(topk_weights) back with
    masked slots hard-zero.
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
        weight_cache: dict | None = None,
        filter_expert: bool = False,
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
            weight_cache=weight_cache,
            filter_expert=filter_expert,
        )
        # Save the xorl scatter bookkeeping alongside the inputs (mirrors the
        # stock TritonMoeExpertsFunction contract): moe_index_compute uses
        # relaxed atomics, so the intra-expert row permutation is only
        # reproducible if backward reuses the one drawn here.
        if filter_expert:
            # Masked-id lane (EP-combine partials): the bookkeeping kernels are
            # NOT -1-safe (scatter indices for masked slots land outside the
            # valid region), so compact the valid (token, slot) pairs into a
            # dense topk=1 sub-problem first; backward runs the stock math on
            # it and scatters grads back (masked slots stay hard-zero).
            valid_flat = (selected_flat.reshape(-1) >= 0).nonzero(as_tuple=False).squeeze(1)
            pair_token = torch.div(valid_flat, int(selected_flat.shape[1]), rounding_mode="floor")
            compact_ids = selected_flat.reshape(-1)[valid_flat].reshape(-1, 1).contiguous()
            if compact_ids.numel():
                splits = expert_histogram(compact_ids, int(num_experts))
                cumsum_t = torch.cumsum(splits, dim=0)
                scatter_index = moe_index_compute(compact_ids, cumsum_t)
            else:
                cumsum_t = torch.cumsum(
                    torch.zeros(int(num_experts), dtype=torch.int32, device=selected_flat.device), dim=0
                )
                scatter_index = torch.empty_like(compact_ids, dtype=torch.int64)
            ctx.save_for_backward(
                hidden_flat,
                routing_flat,
                selected_flat,
                gate_up_proj,
                down_proj,
                cumsum_t,
                scatter_index,
                valid_flat,
                pair_token,
            )
        else:
            splits = expert_histogram(selected_flat, int(num_experts))
            cumsum_t = torch.cumsum(splits, dim=0)
            scatter_index = moe_index_compute(selected_flat, cumsum_t)
            ctx.save_for_backward(
                hidden_flat, routing_flat, selected_flat, gate_up_proj, down_proj, cumsum_t, scatter_index
            )
        ctx.filter_expert = bool(filter_expert)
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
        from xorl.ops.moe.triton import (  # noqa: PLC0415
            _apply_swiglu_clamp_backward,
            _maybe_clamp_swiglu_gate,
            _moe_gate_activation,
            _moe_gate_activation_backward,
        )

        if ctx.filter_expert:
            (
                hidden_states,
                gate_weights,
                expert_index,
                gate_up_proj,
                down_proj,
                cumsum_t,
                scatter_index,
                valid_flat,
                pair_token,
            ) = ctx.saved_tensors
        else:
            (
                hidden_states,
                gate_weights,
                expert_index,
                gate_up_proj,
                down_proj,
                cumsum_t,
                scatter_index,
            ) = ctx.saved_tensors
            valid_flat = pair_token = None

        if valid_flat is not None and valid_flat.numel() == 0:
            # Every slot masked (no local pairs): the kernel produced zeros, so
            # all grads are exactly zero (materialized to keep grad reduction
            # uniform across EP ranks).
            return (
                torch.zeros_like(hidden_states),  # hidden_flat
                torch.zeros_like(gate_weights) if ctx.needs_input_grad[1] else None,  # routing_flat
                None,  # selected_flat
                torch.zeros_like(gate_up_proj) if gate_up_proj.requires_grad else None,  # gate_up_proj
                torch.zeros_like(down_proj) if down_proj.requires_grad else None,  # down_proj
                None,  # fused_experts_impl
                None,  # activation
                None,  # hidden_act
                None,  # swiglu_limit
                None,  # num_experts
                None,  # weight_cache
                None,  # filter_expert
            )

        # Recompute the cheap xorl-forward intermediates the stock backward
        # consumes (scatter + gate/up grouped GEMM) from the bookkeeping saved
        # in forward. The down accumulator is recomputed only when routing
        # itself needs a gradient. The masked lane presents the compacted valid
        # pairs as a topk=1 problem; both lanes use the same expert math.
        if valid_flat is not None:
            hidden_rows = hidden_states.index_select(0, pair_token)
            pair_weights = gate_weights.reshape(-1)[valid_flat].reshape(-1, 1)
        else:
            hidden_rows = hidden_states
            pair_weights = gate_weights.reshape(-1, 1)
        scatter_output = moe_scatter(hidden_rows, scatter_index)
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

        # The serving forward applies each FP32 routing weight to the FP32 down
        # accumulator, then stores BF16.  Preserve that ordering in backward:
        # d(routing) contracts against the pre-cast FP32 accumulator, while the
        # expert-path upstream gradient is multiplied by the unrounded FP32
        # score and cast once at the grouped-GEMM boundary.
        compute_dtype = gate_up_output.dtype
        pair_weights_fp32 = pair_weights.to(torch.float32)
        scattered_pair_weights = torch.empty_like(pair_weights_fp32)
        scattered_pair_weights[scatter_index.flatten()] = pair_weights_fp32
        grad_output = grad_output.view(-1, grad_output.shape[-1]).to(compute_dtype)
        if pair_token is not None:
            grad_output = grad_output.index_select(0, pair_token)

        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
        gate_activation = _moe_gate_activation(gate_for_activation, ctx.hidden_act)
        gated_activation = gate_activation * up_output

        grad_down_output = moe_scatter(grad_output, scatter_index)

        grad_gate_weight = None
        if ctx.needs_input_grad[1]:
            down_accumulator = _group_gemm_same_nk_fp32_accumulator(
                group_gemm_same_nk,
                a=gated_activation,
                b=down_proj,
                cumsum_M=cumsum_t,
                max_M=max_M,
            )
            grad_gate_weight = torch.sum(down_accumulator * grad_down_output.to(torch.float32), dim=-1)[
                scatter_index.flatten()
            ]
            if valid_flat is not None:
                # Masked slots belong to another EP rank and remain exact zero.
                grad_gate_weight_full = grad_gate_weight.new_zeros(gate_weights.numel())
                grad_gate_weight_full[valid_flat] = grad_gate_weight
                grad_gate_weight = grad_gate_weight_full
            grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape).to(gate_weights.dtype)
            del down_accumulator

        grad_scaled = _scale_moe_grad_by_fp32_routing(grad_down_output, scattered_pair_weights)
        grad_gated_activation = group_gemm_same_nk(
            a=grad_scaled,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_activation,
                b=grad_scaled,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
            )
        del grad_down_output, grad_scaled, scattered_pair_weights, gated_activation

        grad_up_output = gate_activation * grad_gated_activation
        grad_gate_activation = grad_gated_activation * up_output
        del grad_gated_activation, gate_activation, up_output
        grad_gate_output = _moe_gate_activation_backward(grad_gate_activation, gate_for_activation, ctx.hidden_act)
        grad_gate_output = _apply_swiglu_clamp_backward(grad_gate_output, gate_output, ctx.swiglu_limit)
        del grad_gate_activation, gate_output, gate_for_activation

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

        grad_pair_rows = grad_scatter_output[scatter_index.flatten()]
        if valid_flat is not None:
            # Scatter the per-pair grads back to the full (token, slot) layout
            # (unique indices — deterministic, no atomics), then the stock
            # per-token sum over slots.
            grad_full = grad_pair_rows.new_zeros(gate_weights.numel(), grad_pair_rows.shape[-1])
            grad_full[valid_flat] = grad_pair_rows
            grad_pair_rows = grad_full
            num_slots = int(gate_weights.shape[-1])
        else:
            num_slots = int(scatter_index.shape[1])
        grad_hidden_states = grad_pair_rows.reshape(hidden_states.shape[0], num_slots, -1).sum(dim=1)

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
            None,  # weight_cache
            None,  # filter_expert
        )


def _sglang_fused_experts_ep_kernel_call(
    permute_tokens: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_scores: torch.Tensor,
    cumsum: torch.Tensor,
    fused_experts_impl,
    activation: str,
    swiglu_limit: float,
    gate_up_bias: torch.Tensor | None,
    gated: bool = True,
    weight_cache: dict | None = None,
) -> torch.Tensor:
    """The exact EP topk=1 serving-kernel call (shared by the inference and
    autograd paths — the serving tree defines K3, so this must never fork).

    Presents the post-dispatch pair rows to ``fused_experts_impl`` as a topk=1
    batch: local expert ids rebuilt from the per-local-expert ``cumsum`` via
    ``repeat_interleave``, dispatched pair weights as the single top-k weight in
    fp32 (exact upcast for bf16 sources; fp32 sources pass through), and the
    LOCAL weight slice presented from xorl GKN
    (``gate_up_proj [E_local, H, 2I]`` gate-first / ``down_proj [E_local, I, H]``)
    in SGLang's ``w13 [E_local, 2I, H]`` / ``w2 [E_local, H, I]`` layout per
    :func:`moe_sglang_fused_experts_weight_mode` — a zero-copy transpose-view
    (strided mode, default; ``fused_experts_impl`` must then be the strided
    variant), a transient transpose-copy, or the ``weight_cache`` copy — all
    three bit-identical.
    """
    _validate_sglang_swiglu_limit(swiglu_limit)
    num_local_experts = int(gate_up_proj.shape[0])
    counts = torch.diff(cumsum.to(torch.int64), prepend=cumsum.new_zeros(1, dtype=torch.int64))
    local_expert_ids = torch.repeat_interleave(
        torch.arange(num_local_experts, dtype=torch.int32, device=permute_tokens.device),
        counts,
    ).unsqueeze(1)
    # Dispatched pair weights in fp32 (serving contract). alltoall_pre_dispatch
    # keeps the routing-weight source dtype under this flag, so the upcast is
    # exact (bf16 sources round-trip losslessly; fp32 sources pass through).
    pair_weights = expert_scores.reshape(-1, 1).to(torch.float32).contiguous()

    if moe_sglang_fused_experts_weight_mode() == "strided":
        w13 = gate_up_proj.transpose(1, 2)
        w2 = down_proj.transpose(1, 2)
    else:
        w13 = _cached_serving_transpose(weight_cache, "w13", gate_up_proj)
        w2 = _cached_serving_transpose(weight_cache, "w2", down_proj)
    b1 = gate_up_bias.contiguous() if gate_up_bias is not None else None

    MoEExperts._log_sglang_fused_experts_config_once(
        w13,
        w2,
        local_expert_ids,
        flag_attr="_sglang_fused_experts_ep_config_logged",
        label="ep",
    )

    return fused_experts_impl(
        permute_tokens,
        w13,
        w2,
        pair_weights,
        local_expert_ids,
        b1=b1,
        b2=None,
        inplace=False,
        activation=activation,
        is_gated=gated,
        apply_router_weight_on_input=False,
        no_combine=False,
        routed_scaling_factor=None,
        gemm1_alpha=None,
        gemm1_limit=None,
        gate_up_interleaved=False,
        # ids are already local (0..E_local-1) against the local weight slice.
        filter_expert=False,
    )


class _SglangFusedExpertsEPTrainFunction(torch.autograd.Function):
    """Trainable wrapper for the per-rank EP serving-kernel MoE compute.

    forward: SGLang's ``fused_experts_impl`` on the post-dispatch pair rows in
    the topk=1 presentation — numerically identical to the inference path (the
    serving reduction tree defines K3). The kernel applies the dispatched
    routing weight on the fp32 down-GEMM accumulator, i.e. after-down
    semantics ``down(g) * s``.
    backward: xorl's after-down EP grouped-GEMM backward with the serving
    FP32-routing cast boundary preserved: the gate/up grouped GEMM is
    recomputed from the saved pair rows
    (deterministic — the dispatch order pins the row permutation, no scatter
    bookkeeping to replay), then dgrad/wgrad grouped GEMMs, the activation
    backward, and d(pair_scores) via the unscaled down-output recompute (one
    extra grouped GEMM). Gradients flow onward through xorl's existing
    dispatch/combine autograd; weight grads come out directly in GKN layout on
    the local expert slice.
    """

    @staticmethod
    def forward(
        ctx,
        permute_tokens: torch.Tensor,
        expert_scores: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        cumsum: torch.Tensor,
        fused_experts_impl,
        activation: str,
        hidden_act: str,
        swiglu_limit: float,
        gated: bool,
        weight_cache: dict | None = None,
    ) -> torch.Tensor:
        _validate_sglang_swiglu_limit(swiglu_limit)
        permute_tokens = permute_tokens.contiguous()
        if permute_tokens.shape[0] == 0:
            # A rank can legitimately receive zero pair rows; the serving kernel
            # is never entered on empty extend batches, so short-circuit. The
            # local weights see no rows, so their gradients are exactly zero
            # (produced in backward to keep grad reduction uniform across ranks).
            output = permute_tokens.new_zeros(permute_tokens.shape)
        else:
            output = _sglang_fused_experts_ep_kernel_call(
                permute_tokens,
                gate_up_proj,
                down_proj,
                expert_scores,
                cumsum,
                fused_experts_impl,
                activation,
                swiglu_limit,
                gate_up_bias=None,
                gated=gated,
                weight_cache=weight_cache,
            )
        ctx.save_for_backward(permute_tokens, expert_scores, gate_up_proj, down_proj, cumsum)
        ctx.hidden_act = hidden_act
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, expert_scores, gate_up_proj, down_proj, cumsum = ctx.saved_tensors

        if permute_tokens.shape[0] == 0:
            return (
                torch.zeros_like(permute_tokens),  # permute_tokens
                torch.zeros_like(expert_scores) if ctx.needs_input_grad[1] else None,  # expert_scores
                torch.zeros_like(gate_up_proj) if ctx.needs_input_grad[2] else None,  # gate_up_proj
                torch.zeros_like(down_proj) if ctx.needs_input_grad[3] else None,  # down_proj
                None,  # cumsum
                None,  # fused_experts_impl
                None,  # activation
                None,  # hidden_act
                None,  # swiglu_limit
                None,  # gated
                None,  # weight_cache
            )

        from xorl.ops.group_gemm.kernel.group_gemm import (  # noqa: PLC0415
            group_gemm_same_mn,
            group_gemm_same_nk,
        )
        from xorl.ops.moe.triton import (  # noqa: PLC0415
            _apply_swiglu_clamp_backward,
            _maybe_clamp_swiglu_gate,
            _moe_gate_activation,
            _moe_gate_activation_backward,
        )

        max_M = permute_tokens.shape[0]

        # Recompute the gate/up grouped GEMM the stock backward consumes (the
        # serving kernel does not expose xorl's intermediate; the recompute is
        # deterministic given the dispatch-pinned row order and cumsum).
        gate_up_output = group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
        )
        intermediate = gate_up_output.shape[-1] // 2
        gate_output = gate_up_output[..., :intermediate]
        up_output = gate_up_output[..., intermediate:]

        # Match the serving routing-after-down program: the FP32 score scales
        # the FP32 down accumulator before its BF16 store.  d(routing) uses the
        # retained-precision accumulator; all expert-path gradients multiply
        # by the unrounded score before the BF16 grouped-GEMM boundary.
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
        gate_activation = _moe_gate_activation(gate_for_activation, ctx.hidden_act)
        gated_output = gate_activation * up_output

        compute_dtype = gated_output.dtype
        expert_scores_dtype = expert_scores.dtype
        scores_fp32 = expert_scores.reshape(-1).to(torch.float32)
        grad_output = grad_output.to(compute_dtype)

        grad_expert_scores = None
        if ctx.needs_input_grad[1]:
            down_accumulator = _group_gemm_same_nk_fp32_accumulator(
                group_gemm_same_nk,
                a=gated_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
            )
            grad_expert_scores = (
                (down_accumulator * grad_output.to(torch.float32))
                .sum(dim=-1)
                .to(expert_scores_dtype)
                .reshape(expert_scores.shape)
            )
            del down_accumulator

        grad_scaled = _scale_moe_grad_by_fp32_routing(grad_output, scores_fp32)

        grad_gated_output = group_gemm_same_nk(
            a=grad_scaled,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )

        grad_down_proj = None
        if ctx.needs_input_grad[3]:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_output,
                b=grad_scaled,
                c=grad_down_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
            )
        del gated_output, grad_scaled

        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output
        grad_gate_output = _moe_gate_activation_backward(grad_gate_activation, gate_for_activation, ctx.hidden_act)
        grad_gate_output = _apply_swiglu_clamp_backward(grad_gate_output, gate_output, ctx.swiglu_limit)
        del grad_gate_activation, gate_output, gate_for_activation

        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output

        grad_gate_up_proj = None
        if ctx.needs_input_grad[2]:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            group_gemm_same_mn(
                a=permute_tokens,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
            )

        grad_permute_tokens = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )
        del grad_gate_up_act

        return (
            grad_permute_tokens,  # permute_tokens
            grad_expert_scores,  # expert_scores
            grad_gate_up_proj,  # gate_up_proj
            grad_down_proj,  # down_proj
            None,  # cumsum
            None,  # fused_experts_impl
            None,  # activation
            None,  # hidden_act
            None,  # swiglu_limit
            None,  # gated
            None,  # weight_cache
        )


def _deepep_parity_diagnostic_enabled() -> bool:
    return _flag_enabled("XORL_DEEPEP_PARITY_DIAGNOSTIC")


def _deepep_parity_reference_compare_enabled() -> bool:
    return _flag_enabled("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE")


def _deepep_parity_all_ranks_requested() -> bool:
    raw = os.environ.get("XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS", "0").strip().lower()
    return raw in {"all", "*"}


def _distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _ep_group_rank(ep_group) -> int:
    if dist.is_available() and dist.is_initialized() and ep_group is not None:
        return dist.get_rank(ep_group)
    return 0


def _ep_group_size(ep_group) -> int:
    if ep_group is None:
        return 1
    size = getattr(ep_group, "size", None)
    if callable(size):
        return int(size())
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size(ep_group))
    return 1


def _deepep_parity_rank_allowed(rank: int) -> bool:
    raw = os.environ.get("XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS", "0").strip().lower()
    if raw in {"all", "*"}:
        return True
    allowed = {part.strip() for part in raw.split(",") if part.strip()}
    return str(rank) in allowed


def _acquire_deepep_parity_diagnostic_record() -> int | None:
    if not _deepep_parity_diagnostic_enabled():
        return None
    rank = _distributed_rank()
    if not _deepep_parity_rank_allowed(rank):
        return None
    record_start = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START", 0))
    max_records = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS", 8))
    count = _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS.get(rank, 0)
    _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS[rank] = count + 1
    if count < record_start or count >= record_start + max_records:
        return None
    return count


def _tensor_summary(tensor: torch.Tensor | None, *, include_sample: bool = True) -> dict | None:
    if tensor is None:
        return None
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
    }
    if not include_sample or tensor.numel() == 0:
        return summary

    max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
    fingerprint = _tensor_fingerprint(tensor)
    if fingerprint is not None:
        summary["fingerprint"] = fingerprint
    if max_values == 0:
        return summary

    with torch.no_grad():
        flat = tensor.detach().reshape(-1)[:max_values]
        try:
            summary["sample"] = flat.to(torch.float32).cpu().tolist()
        except (RuntimeError, TypeError):
            summary["sample"] = [str(v) for v in flat.cpu().tolist()]
    return summary


def _tensor_fingerprint(tensor: torch.Tensor) -> dict | None:
    max_elems = _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_FINGERPRINT_MAX_ELEMS", 0)
    if max_elems == 0 or tensor.numel() == 0:
        return None
    elem_count = int(tensor.numel()) if max_elems < 0 else min(int(tensor.numel()), max_elems)
    if elem_count <= 0:
        return None
    with torch.no_grad():
        values = tensor.detach().reshape(-1)[:elem_count].to(torch.float32).contiguous().cpu()
        finite = torch.isfinite(values)
        finite_values = values[finite]
        payload = values.numpy().tobytes()
        summary = {
            "numel": elem_count,
            "dtype_for_hash": "float32",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "finite_count": int(finite.sum().item()),
        }
        if finite_values.numel() > 0:
            summary.update(
                {
                    "sum": float(finite_values.sum().item()),
                    "abs_sum": float(finite_values.abs().sum().item()),
                    "max_abs": float(finite_values.abs().max().item()),
                }
            )
        return summary


def _cumsum_summary(cumsum: torch.Tensor | None) -> dict | None:
    if cumsum is None:
        return None
    summary = _tensor_summary(cumsum)
    with torch.no_grad():
        values = cumsum.detach().to(torch.int64).cpu()
        counts = torch.diff(torch.cat([values.new_zeros(1), values]))
        max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
        summary["last"] = int(values[-1].item()) if values.numel() > 0 else 0
        summary["counts_head"] = counts[:max_values].tolist()
        summary["counts_tail"] = counts[-max_values:].tolist() if max_values else []
        if counts.numel() > 0:
            k = min(max(1, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK", 8)), counts.numel())
            top_counts, top_indices = torch.topk(counts, k=k)
            summary["top_counts"] = [
                {"local_expert": int(idx.item()), "count": int(count.item())}
                for count, idx in zip(top_counts, top_indices)
                if int(count.item()) != 0
            ]
    return summary


def _selected_experts_summary(selected_experts: torch.Tensor | None, num_experts: int) -> dict | None:
    if selected_experts is None:
        return None
    summary = _tensor_summary(selected_experts)
    with torch.no_grad():
        flat = selected_experts.detach().reshape(-1).to(torch.int64)
        valid_mask = (flat >= 0) & (flat < num_experts)
        valid = flat[valid_mask]
        summary["invalid_count"] = int((~valid_mask).sum().item())
        summary["valid_count"] = int(valid.numel())
        if valid.numel() > 0:
            counts = torch.bincount(valid, minlength=num_experts)
            k = min(max(1, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK", 8)), num_experts)
            top_counts, top_indices = torch.topk(counts, k=k)
            summary["top_global_experts"] = [
                {"expert": int(idx.item()), "count": int(count.item())}
                for count, idx in zip(top_counts, top_indices)
                if int(count.item()) != 0
            ]
        else:
            summary["top_global_experts"] = []
    return summary


def _reference_compare_dtype(fallback: torch.dtype) -> torch.dtype:
    raw = os.environ.get("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE", "fp32").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"compute", "input", "original"}:
        return fallback
    return torch.float32


def _expert_counts_from_cumsum(cumsum: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    cumsum_i64 = cumsum.detach().to(torch.int64)
    if cumsum_i64.numel() != num_local_experts:
        return cumsum_i64.new_empty(0)
    return torch.diff(torch.cat([cumsum_i64.new_zeros(1), cumsum_i64]))


def _diff_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f32 = actual.detach().to(torch.float32)
    expected_f32 = expected.detach().to(torch.float32)
    diff = (actual_f32 - expected_f32).abs()
    finite_mask = torch.isfinite(diff)
    finite_diff = diff[finite_mask]
    summary = {
        "shape": list(actual.shape),
        "compared_elements": int(diff.numel()),
        "nonfinite_diff_count": int((~finite_mask).sum().item()),
    }
    if finite_diff.numel() == 0:
        summary.update({"max_abs": None, "mean_abs": None, "p95_abs": None, "nonzero_count": 0, "top_diffs": []})
        return summary

    p95_values = finite_diff
    p95_max_elems = max(1, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_P95_MAX_ELEMS", 1048576))
    p95_sampled = int(finite_diff.numel()) > p95_max_elems
    if p95_sampled:
        indices = _evenly_spaced_int64_indices(
            int(finite_diff.numel()),
            p95_max_elems,
            finite_diff.device,
        )
        p95_values = finite_diff.index_select(0, indices)

    summary.update(
        {
            "max_abs": float(finite_diff.max().item()),
            "mean_abs": float(finite_diff.mean().item()),
            "p95_abs": float(torch.quantile(p95_values, 0.95).item()),
            "p95_sampled": p95_sampled,
            "p95_sample_size": int(p95_values.numel()),
            "nonzero_count": int((finite_diff != 0).sum().item()),
        }
    )
    max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
    if max_values == 0 or diff.numel() == 0:
        summary["top_diffs"] = []
        return summary

    flat_diff = diff.reshape(-1)
    k = min(max_values, int(flat_diff.numel()))
    values, indices = torch.topk(flat_diff, k=k)
    width = actual.shape[-1] if actual.ndim > 0 and actual.shape[-1] else 1
    actual_flat = actual_f32.reshape(-1)
    expected_flat = expected_f32.reshape(-1)
    summary["top_diffs"] = [
        {
            "flat_index": int(index.item()),
            "row": int(index.item() // width),
            "col": int(index.item() % width),
            "abs": float(value.item()),
            "actual": float(actual_flat[index].item()),
            "expected": float(expected_flat[index].item()),
        }
        for value, index in zip(values, indices)
    ]
    return summary


def _evenly_spaced_int64_indices(numel: int, sample_size: int, device: torch.device) -> torch.Tensor:
    if sample_size <= 1:
        return torch.zeros(max(0, sample_size), dtype=torch.long, device=device)
    positions = torch.arange(sample_size, dtype=torch.long, device=device)
    return positions.mul(numel - 1).div(sample_size - 1, rounding_mode="floor")


def _reference_status(diff: dict) -> tuple[str, list[str], dict]:
    thresholds = {
        "max_abs": _env_float_or_none("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ABS"),
        "mean_abs": _env_float_or_none("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MEAN_ABS"),
    }
    active_thresholds = {key: value for key, value in thresholds.items() if value is not None}
    exceeded = [
        key for key, value in active_thresholds.items() if diff.get(key) is not None and float(diff[key]) > float(value)
    ]
    if exceeded:
        return "failed", exceeded, active_thresholds
    return ("pass" if active_thresholds else "observed"), [], active_thresholds


def _expert_output_reference_comparison(
    *,
    permute_tokens: torch.Tensor | None,
    cumsum: torch.Tensor | None,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    intermediate_size: int,
    expert_scores: torch.Tensor | None,
    hidden_act: str,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    expert_output: torch.Tensor | None,
) -> dict | None:
    if not _deepep_parity_reference_compare_enabled():
        return None
    if permute_tokens is None or cumsum is None or expert_output is None:
        return {"status": "skipped", "reason": "missing_tensor"}
    if permute_tokens.ndim != 2 or expert_output.ndim != 2:
        return {"status": "skipped", "reason": "unsupported_rank"}

    from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

    with torch.no_grad():
        num_local_experts = int(gate_up_proj.shape[0])
        counts = _expert_counts_from_cumsum(cumsum, num_local_experts)
        if counts.numel() != num_local_experts:
            return {
                "status": "skipped",
                "reason": "cumsum_length_mismatch",
                "cumsum_length": int(cumsum.numel()),
                "num_local_experts": num_local_experts,
            }
        total_rows = int(expert_output.shape[0])
        cumsum_last = int(cumsum.detach().to(torch.int64)[-1].item()) if cumsum.numel() > 0 else 0
        if cumsum_last != total_rows or int(permute_tokens.shape[0]) != total_rows:
            return {
                "status": "skipped",
                "reason": "row_count_mismatch",
                "cumsum_last": cumsum_last,
                "permute_tokens_rows": int(permute_tokens.shape[0]),
                "expert_output_rows": total_rows,
            }

        max_rows = _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS", 0)
        compare_rows = total_rows if max_rows <= 0 else min(total_rows, max_rows)
        row_sampled = compare_rows != total_rows
        if row_sampled:
            row_indices = _evenly_spaced_int64_indices(total_rows, compare_rows, expert_output.device)
        else:
            row_indices = torch.arange(total_rows, dtype=torch.long, device=expert_output.device)
        reference_dtype = _reference_compare_dtype(permute_tokens.dtype)
        reference = torch.empty(
            (compare_rows, int(expert_output.shape[1])),
            dtype=reference_dtype,
            device=expert_output.device,
        )

        start = 0
        for expert_idx, count_tensor in enumerate(counts):
            end = start + int(count_tensor.item())
            mask = (row_indices >= start) & (row_indices < end)
            if bool(mask.any().item()):
                selected_rows = row_indices[mask]
                tokens = permute_tokens.index_select(0, selected_rows).detach().to(reference_dtype)
                gate_up = tokens.matmul(gate_up_proj[expert_idx].detach().to(reference_dtype))
                if gate_up_bias is not None:
                    gate_up = gate_up + gate_up_bias[expert_idx].detach().to(reference_dtype)
                gate, up = gate_up.split(intermediate_size, dim=-1)
                activated = apply_moe_activation(hidden_act, gate, up)
                out = activated.matmul(down_proj[expert_idx].detach().to(reference_dtype))
                if down_bias is not None:
                    out = out + down_bias[expert_idx].detach().to(reference_dtype)
                if expert_scores is not None:
                    scores = expert_scores.index_select(0, selected_rows).detach().to(reference_dtype).unsqueeze(-1)
                    out = out * scores
                reference[mask] = out
            start = end
            if start >= total_rows:
                break

        actual = expert_output.index_select(0, row_indices)
        diff = _diff_summary(actual, reference)
        status, exceeded, thresholds = _reference_status(diff)
        max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
        return {
            "status": status,
            "reference_dtype": str(reference_dtype).replace("torch.", ""),
            "compare_rows": compare_rows,
            "total_rows": total_rows,
            "row_limited": row_sampled,
            "row_sample_strategy": "evenly_spaced" if row_sampled else "all",
            "row_indices_head": row_indices[:max_values].detach().cpu().tolist() if max_values else [],
            "row_indices_tail": row_indices[-max_values:].detach().cpu().tolist() if max_values else [],
            "thresholds": thresholds,
            "thresholds_exceeded": exceeded,
            "diff": diff,
            "reference": _tensor_summary(reference),
        }


def _safe_expert_output_reference_comparison(**kwargs) -> dict | None:
    try:
        return _expert_output_reference_comparison(**kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash training.
        return {"status": "error", "reason": type(exc).__name__, "message": str(exc)}


def _result_reference_comparison(
    *,
    hidden_states: torch.Tensor | None,
    routing_weights: torch.Tensor | None,
    selected_experts: torch.Tensor | None,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    intermediate_size: int,
    hidden_act: str,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    result: torch.Tensor | None,
    ep_group,
) -> dict | None:
    if not _deepep_parity_reference_compare_enabled():
        return None
    if hidden_states is None or routing_weights is None or selected_experts is None or result is None:
        return {"status": "skipped", "reason": "missing_tensor"}
    if hidden_states.ndim < 2 or result.ndim < 2 or selected_experts.ndim < 2 or routing_weights.ndim < 2:
        return {"status": "skipped", "reason": "unsupported_rank"}

    ep_size = _ep_group_size(ep_group)
    if ep_size > 1 and not _deepep_parity_all_ranks_requested():
        return {"status": "skipped", "reason": "requires_all_ranks"}
    if ep_size > 1 and not (dist.is_available() and dist.is_initialized()):
        return {"status": "skipped", "reason": "distributed_not_initialized"}

    from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

    with torch.no_grad():
        hidden_flat = hidden_states.detach().reshape(-1, int(hidden_states.shape[-1]))
        result_flat = result.detach().reshape(-1, int(result.shape[-1]))
        selected_flat = selected_experts.detach().reshape(-1, int(selected_experts.shape[-1]))
        routing_flat = routing_weights.detach().reshape(-1, int(routing_weights.shape[-1]))
        total_rows = int(result_flat.shape[0])
        if (
            hidden_flat.shape[0] != total_rows
            or selected_flat.shape[0] != total_rows
            or routing_flat.shape[0] != total_rows
            or hidden_flat.shape[-1] != result_flat.shape[-1]
            or selected_flat.shape != routing_flat.shape
        ):
            return {
                "status": "skipped",
                "reason": "shape_mismatch",
                "hidden_rows": int(hidden_flat.shape[0]),
                "result_rows": total_rows,
                "selected_shape": list(selected_flat.shape),
                "routing_shape": list(routing_flat.shape),
            }

        max_rows = _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS", 0)
        compare_rows = total_rows if max_rows <= 0 else min(total_rows, max_rows)
        row_sampled = compare_rows != total_rows
        if row_sampled:
            row_indices = _evenly_spaced_int64_indices(total_rows, compare_rows, result_flat.device)
        else:
            row_indices = torch.arange(total_rows, dtype=torch.long, device=result_flat.device)

        reference_dtype = _reference_compare_dtype(hidden_flat.dtype)
        tokens = hidden_flat.index_select(0, row_indices).to(reference_dtype)
        selected = selected_flat.index_select(0, row_indices).to(torch.long)
        routing = routing_flat.index_select(0, row_indices).to(reference_dtype)
        reference = torch.zeros(
            (compare_rows, int(result_flat.shape[-1])),
            dtype=reference_dtype,
            device=result_flat.device,
        )

        num_local_experts = int(gate_up_proj.shape[0])
        ep_rank = _ep_group_rank(ep_group)
        local_start = ep_rank * num_local_experts
        local_end = local_start + num_local_experts
        local_contribution_count = 0
        local_expert_hit_count = 0
        for local_expert_idx, global_expert_idx in enumerate(range(local_start, local_end)):
            mask = selected == global_expert_idx
            if not bool(mask.any().item()):
                continue
            pair_rows, pair_topk = mask.nonzero(as_tuple=True)
            local_expert_hit_count += 1
            local_contribution_count += int(pair_rows.numel())
            expert_tokens = tokens.index_select(0, pair_rows)
            gate_up = expert_tokens.matmul(gate_up_proj[local_expert_idx].detach().to(reference_dtype))
            if gate_up_bias is not None:
                gate_up = gate_up + gate_up_bias[local_expert_idx].detach().to(reference_dtype)
            gate, up = gate_up.split(intermediate_size, dim=-1)
            activated = apply_moe_activation(hidden_act, gate, up)
            out = activated.matmul(down_proj[local_expert_idx].detach().to(reference_dtype))
            if down_bias is not None:
                out = out + down_bias[local_expert_idx].detach().to(reference_dtype)
            out = out * routing[pair_rows, pair_topk].unsqueeze(-1)
            reference.index_add_(0, pair_rows, out)

        global_contribution_count = local_contribution_count
        if ep_size > 1:
            contribution_count = torch.tensor([local_contribution_count], dtype=torch.long, device=result_flat.device)
            dist.all_reduce(reference, op=dist.ReduceOp.SUM, group=ep_group)
            dist.all_reduce(contribution_count, op=dist.ReduceOp.SUM, group=ep_group)
            global_contribution_count = int(contribution_count.item())

        actual = result_flat.index_select(0, row_indices)
        diff = _diff_summary(actual, reference)
        status, exceeded, thresholds = _reference_status(diff)
        max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
        return {
            "status": status,
            "reference_dtype": str(reference_dtype).replace("torch.", ""),
            "compare_rows": compare_rows,
            "total_rows": total_rows,
            "row_limited": row_sampled,
            "row_sample_strategy": "evenly_spaced" if row_sampled else "all",
            "row_indices_head": row_indices[:max_values].detach().cpu().tolist() if max_values else [],
            "row_indices_tail": row_indices[-max_values:].detach().cpu().tolist() if max_values else [],
            "local_expert_global_range": [local_start, local_end],
            "local_expert_hit_count": local_expert_hit_count,
            "local_contribution_count": local_contribution_count,
            "global_contribution_count": global_contribution_count,
            "thresholds": thresholds,
            "thresholds_exceeded": exceeded,
            "diff": diff,
            "reference": _tensor_summary(reference),
        }


def _safe_result_reference_comparison(**kwargs) -> dict | None:
    try:
        return _result_reference_comparison(**kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash training.
        return {"status": "error", "reason": type(exc).__name__, "message": str(exc)}


def _dispatch_context_summary(ctx) -> dict | None:
    if ctx is None:
        return None
    return {
        "type": type(ctx).__name__,
        "handle_type": type(getattr(ctx, "handle", None)).__name__,
        "num_recv_tokens": getattr(ctx, "num_recv_tokens", None),
        "num_valid": getattr(ctx, "num_valid", None),
        "dtype": str(getattr(ctx, "dtype", "")).replace("torch.", ""),
        "hidden_dim": getattr(ctx, "hidden_dim", None),
        "input_splits": getattr(ctx, "input_splits", None),
        "output_splits": getattr(ctx, "output_splits", None),
        "orig_shape": list(getattr(ctx, "orig_shape", [])),
        "num_experts": getattr(ctx, "num_experts", None),
        "num_tokens_per_expert": _tensor_summary(getattr(ctx, "num_tokens_per_expert", None)),
        "routing_map": _tensor_summary(getattr(ctx, "routing_map", None)),
        "perm_mapping": _tensor_summary(getattr(ctx, "perm_mapping", None)),
        "permuted_indices": _tensor_summary(getattr(ctx, "permuted_indices", None)),
        "permuted_scores": _tensor_summary(getattr(ctx, "permuted_scores", None)),
        "expert_scores": _tensor_summary(getattr(ctx, "expert_scores", None)),
    }


class MoEExperts(nn.Module):
    """Unified weight container for MoE experts.

    Holds stacked weight tensors ``[num_experts, ...]`` and dispatches
    ``forward()`` to the selected backend (eager / triton / native / quack).

    Weights are stored in ``(G, K, N)`` format — ``[num_experts, in_features, out_features]``::

        gate_up_proj: [num_experts, hidden_dim, 2 * intermediate_size]
        down_proj:    [num_experts, intermediate_size, hidden_dim]

    ``gate_proj`` and ``up_proj`` are exposed as views into ``gate_up_proj``
    for compatibility with existing backends and helpers.

    With ``gated=False`` (e.g. Nemotron-3-Ultra latent experts) there is no
    gate branch: the activation is applied directly to the single first GEMM
    output. The checkpoint-visible parameter name stays ``gate_up_proj`` for
    layout/EP-plan stability, but it holds only the up projection at half
    width::

        gate_up_proj: [num_experts, hidden_dim, intermediate_size]

    ``hidden_dim`` is whatever input dimension the caller provides — it does
    not have to equal the model hidden size (Nemotron experts run in a latent
    dim).

    Optional per-expert biases (``gate_up_bias``, ``down_bias``) default to
    ``None`` and can be set by model-specific code (e.g. GPT-OSS).

    Args:
        num_experts: Total number of experts.
        hidden_dim: Expert input/output dimension (model hidden size or a
            latent dimension).
        intermediate_size: Expert FFN intermediate dimension.
        hidden_act: Activation function name (default: ``"silu"``).
        moe_implementation: Backend name — ``"eager"``, ``"triton"``, ``"native"``, or ``"quack"``.
        gated: Whether experts use a gated (GLU) first projection (default: True).
            Non-gated experts require an activation in ``UNGATED_HIDDEN_ACTS``
            (currently ``relu2``) and are not supported by the quack backend.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        moe_implementation: str = "triton",
        activation_native: bool = False,
        swiglu_limit: float = 0.0,
        gated: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.moe_implementation = moe_implementation
        self.activation_native = activation_native
        self.swiglu_limit = float(swiglu_limit)
        self.gated = gated

        if not gated and moe_implementation == "quack":
            raise NotImplementedError("quack backend does not support non-gated experts")

        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, (2 if gated else 1) * intermediate_size),
            requires_grad=True,
        )
        if gated:
            self.gate_up_proj._fused_gate_up = True
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[hidden_act]
        # String kind used by triton/native/quack backends (avoids name-sniffing).
        from xorl.ops.moe.activations import (  # noqa: PLC0415
            UNGATED_HIDDEN_ACTS,
            check_hidden_act_supported,
            normalize_hidden_act,
        )

        self.hidden_act = normalize_hidden_act(hidden_act)
        if not gated:
            check_hidden_act_supported(self.hidden_act, f"{moe_implementation} (non-gated)", UNGATED_HIDDEN_ACTS)
        self._moe_act: bool = False

        # Optional per-expert biases (e.g. GPT-OSS). Set to actual tensors
        # by model-specific code; None means no bias.
        self.gate_up_bias = None
        self.down_bias = None

        # EP dispatch strategy: "alltoall" (default) or "deepep" (NVLink-optimized)
        self.ep_dispatch: str = "alltoall"
        self.deepep_buffer_size_gb: float = 2.0
        self.deepep_num_sms: int = 20
        self.deepep_async_combine: bool = False
        self.fp8_training_enabled: bool = False
        self.fp8_training_grouped_backend: str = "triton_grouped"
        self.fp8_training_block_size: int = 128
        self.last_forward_used_fp8: bool = False
        self.alltoall_combine_hidden_chunk_size: int = 0

    @property
    def gate_proj(self) -> torch.Tensor:
        if not self.gated:
            raise AttributeError("non-gated MoEExperts has no gate_proj")
        gate_proj, _ = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        gate_proj.grad = (
            None if self.gate_up_proj.grad is None else self.gate_up_proj.grad[..., : self.intermediate_size]
        )
        return gate_proj

    @property
    def up_proj(self) -> torch.Tensor:
        if not self.gated:
            return self.gate_up_proj
        _, up_proj = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        up_proj.grad = None if self.gate_up_proj.grad is None else self.gate_up_proj.grad[..., self.intermediate_size :]
        return up_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        selected_experts: torch.Tensor = None,
        expert_idx: int = None,
        *,
        sglang_ep_native_local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to the configured backend.

        For **triton/native/quack**: call with ``(hidden_states, routing_weights, selected_experts)``.
        For **eager**: called per-expert from ``MoEBlock._eager_forward()`` with ``expert_idx``.

        When Expert Parallelism is enabled, all backends
        use the unified dispatch → compute → combine path via ``_ep_forward()``.
        """
        # Keep the native EP serving-kernel path behind ``nn.Module.__call__``.
        # In EP8 the experts module is an FSDP unit; calling the helper method
        # directly bypasses FSDP's pre-forward mixed-precision hook and leaves
        # FP32 master weights paired with BF16 activations in Triton's tl.dot.
        if sglang_ep_native_local_ids is not None:
            return self.sglang_ep_native_routed_partial(
                hidden_states,
                routing_weights,
                sglang_ep_native_local_ids,
            )

        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        if parallel_state.ep_enabled:
            return self._ep_forward(hidden_states, routing_weights, selected_experts, parallel_state)
        self.last_forward_used_fp8 = False
        if self.fp8_training_enabled and self.moe_implementation != "quack":
            raise NotImplementedError("FP8 grouped MoE compute currently requires moe_implementation='quack'")

        if self.moe_implementation == "eager":
            fn = MOE_EXPERT_BACKENDS[self.moe_implementation]
            assert expert_idx is not None
            return fn(
                hidden_states,
                expert_idx,
                self.gate_proj.contiguous() if self.gated else None,
                self.up_proj.contiguous() if self.gated else self.gate_up_proj,
                self.down_proj,
                hidden_act=self.hidden_act,
                swiglu_limit=self.swiglu_limit,
                gate_up_bias=self.gate_up_bias,
                down_bias=self.down_bias,
                gated=self.gated,
            )

        # Local single-GPU path. Non-gated experts only use the fused
        # gate_up_proj reference (which holds the plain up projection).
        gate_proj = self.gate_proj.contiguous() if self.gated else None
        up_proj = self.up_proj.contiguous() if self.gated else None
        fn = MOE_EXPERT_BACKENDS[self.moe_implementation]
        self.last_forward_used_fp8 = bool(self.fp8_training_enabled)

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
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
            fp8_compute=self.fp8_training_enabled,
            fp8_grouped_backend=self.fp8_training_grouped_backend,
            fp8_block_size=self.fp8_training_block_size,
            activation_native=self.activation_native,
            gated=self.gated,
        )

    @staticmethod
    def _ensure_sglang_server_args() -> None:
        try:
            from sglang.srt.runtime_context import get_exec, get_server_args, publish  # noqa: PLC0415
            from sglang.srt.server_args import ServerArgs  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires an environment with sglang and sgl_kernel installed"
            ) from exc

        try:
            get_server_args()
        except ValueError:
            publish(
                ServerArgs(
                    model_path=os.environ.get("XORL_SGLANG_MOE_MODEL_PATH", "dummy"),
                    enable_deterministic_inference=True,
                    enable_fused_moe_sum_all_reduce=False,
                    rl_on_policy_target="xorl-batch-invariant",
                ),
                role="scheduler",
            )

        try:
            exec_config = get_exec()
            deterministic = bool(exec_config.deterministic.enable_deterministic_inference)
            fused_sum_all_reduce = bool(exec_config.moe.enable_fused_moe_sum_all_reduce)
        except (AttributeError, ValueError) as exc:
            raise RuntimeError("SGLang MoE parity requires a published exec runtime context") from exc
        if not deterministic or fused_sum_all_reduce:
            raise RuntimeError(
                "SGLang MoE parity requires enable_deterministic_inference=True and "
                "enable_fused_moe_sum_all_reduce=False "
                f"(got deterministic={deterministic}, fused_sum_all_reduce={fused_sum_all_reduce})"
            )

    @staticmethod
    def _load_sglang_fused_experts_impl():
        try:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # noqa: PLC0415
                fused_experts_impl,
            )
        except ImportError as exc:
            raise ImportError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires an environment with sglang and sgl_kernel installed"
            ) from exc

        MoEExperts._ensure_sglang_server_args()

        if moe_sglang_fused_experts_weight_mode() == "strided":
            # Same sglang triton launches, orchestration vendored to accept the
            # zero-copy GKN transpose-views (stock impl asserts contiguity).
            from xorl.ops.moe.sglang_fused_moe_strided import fused_experts_impl_strided  # noqa: PLC0415

            return fused_experts_impl_strided
        return fused_experts_impl

    def sglang_ep_native_routed_partial(
        self,
        hidden_flat: torch.Tensor,
        routing_flat: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> torch.Tensor:
        """This EP rank's routed partial for the gathered full token batch
        (native-EP canonical combine for the exact Qwen3.5-MoE program).

        It runs on the module's LOCAL expert slice (trainer EP gives rank r exactly
        serving-rank-r's contiguous slice) with ids already mapped local
        (``-1`` = non-local). Trainable through the masked
        :class:`_SglangFusedExpertsTrainFunction`; the no-grad path is the bare
        kernel call (identical bits).
        """
        if not self.gated:
            raise NotImplementedError("exact native-EP canonical combine supports gated MoE experts only")
        if self.down_bias is not None or self.gate_up_bias is not None:
            raise NotImplementedError("exact native-EP canonical combine does not support expert biases")
        _validate_sglang_swiglu_limit(self.swiglu_limit)
        fused_experts_impl = self._load_sglang_fused_experts_impl()
        e_local = int(self.gate_up_proj.shape[0])
        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act

        needs_grad = torch.is_grad_enabled() and (
            hidden_flat.requires_grad
            or routing_flat.requires_grad
            or self.gate_up_proj.requires_grad
            or self.down_proj.requires_grad
        )
        if needs_grad:
            if self.hidden_act not in {"silu", "gelu_tanh"}:
                raise NotImplementedError(
                    "exact native-EP ordered-combine training supports silu/gelu_tanh only "
                    f"(got hidden_act={self.hidden_act!r})"
                )
            return _SglangFusedExpertsTrainFunction.apply(
                hidden_flat,
                routing_flat,
                local_ids,
                self.gate_up_proj,
                self.down_proj,
                fused_experts_impl,
                activation,
                self.hidden_act,
                self.swiglu_limit,
                e_local,
                None,
                True,
            )
        return _sglang_fused_experts_kernel_call(
            hidden_flat,
            self.gate_up_proj,
            self.down_proj,
            routing_flat,
            local_ids,
            fused_experts_impl,
            activation,
            self.swiglu_limit,
            None,
            weight_cache=None,
            filter_expert=True,
        )

    def sglang_fused_experts_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """K3 parity mode: run SGLang's serving MoE kernel (``fused_experts_impl``).

        Full-width local (no-EP) expert compute through the exact bf16 tp1/ep1
        serving path: pinned deterministic launch config, ``moe_align_block_size``
        blocking, sgl_kernel ``silu_and_mul`` activation, and sgl_kernel
        ``moe_sum_reduce`` top-k combine. Given bit-identical inputs, routing, and
        weights this reproduces SGLang's ``moe_experts_output`` bit-exactly, so
        the trainer and the serving engine share one MoE reduction tree at full
        bf16 speed.

        Trainable: when gradients are required the same serving forward runs
        under :class:`_SglangFusedExpertsTrainFunction`, whose backward is
        xorl's proven grouped-GEMM MoE backward (dX, dW in GKN layout, and
        d(topk_weights) so the router trains). The no-grad path is unchanged.

        Weight layout: xorl GKN ``gate_up_proj [E, H, 2I]`` (gate-first) /
        ``down_proj [E, I, H]`` is presented to the kernel as SGLang's
        ``w13 [E, 2I, H]`` / ``w2 [E, H, I]`` per
        :func:`moe_sglang_fused_experts_weight_mode`: a zero-copy transpose-view
        (strided mode, default), a transient transpose-copy, or a cached copy —
        all three bit-identical.
        """
        if routing_weights is None or selected_experts is None:
            raise ValueError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires routing_weights and selected_experts")
        if not self.gated:
            raise NotImplementedError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 supports gated MoE experts only")
        if self.down_bias is not None:
            raise NotImplementedError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support down_bias")
        if self.hidden_act not in {"silu", "gelu", "gelu_tanh"}:
            raise NotImplementedError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support hidden_act={self.hidden_act!r}"
            )
        _validate_sglang_swiglu_limit(self.swiglu_limit)

        try:
            fused_experts_impl = self._load_sglang_fused_experts_impl()
        except ImportError as exc:
            raise ImportError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires an environment with sglang and sgl_kernel "
                "installed (put the SGLang python tree on PYTHONPATH)"
            ) from exc

        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)

        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act
        weight_cache = None
        if moe_sglang_fused_experts_weight_mode() == "cached":
            weight_cache = getattr(self, "_sglang_fused_weight_cache", None)
            if weight_cache is None:
                weight_cache = {}
                self._sglang_fused_weight_cache = weight_cache

        needs_grad = torch.is_grad_enabled() and (
            hidden_flat.requires_grad
            or routing_flat.requires_grad
            or self.gate_up_proj.requires_grad
            or self.down_proj.requires_grad
        )
        if needs_grad:
            if self.gate_up_bias is not None:
                raise NotImplementedError(
                    f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support training with gate_up_bias"
                )
            if self.hidden_act not in {"silu", "gelu_tanh"}:
                raise NotImplementedError(
                    f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 training supports silu/gelu_tanh only "
                    f"(got hidden_act={self.hidden_act!r})"
                )
            output = _SglangFusedExpertsTrainFunction.apply(
                hidden_flat,
                routing_flat,
                selected_flat,
                self.gate_up_proj,
                self.down_proj,
                fused_experts_impl,
                activation,
                self.hidden_act,
                self.swiglu_limit,
                self.num_experts,
                weight_cache,
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
                self.swiglu_limit,
                self.gate_up_bias,
                weight_cache=weight_cache,
            )
        return output.reshape(original_shape)

    def sglang_fused_experts_auto_supported(self) -> bool:
        """Auto-default eligibility: the parity kernel can replace this module's
        expert compute for both inference and training (none of the guards in
        :meth:`sglang_fused_experts_forward` can trip). An explicit
        ``XORL_MOE_SGLANG_FUSED_EXPERTS=1`` bypasses this and keeps loud errors.
        """
        return (
            type(self) is MoEExperts
            and self.gated
            and self.gate_up_bias is None
            and self.down_bias is None
            and self.hidden_act in {"silu", "gelu_tanh"}
            and self.swiglu_limit <= 0.0
            and not self.fp8_training_enabled
        )

    def invalidate_sglang_fused_weight_cache(self) -> None:
        """Drop cached serving-layout weight transposes (see
        :func:`moe_sglang_fused_experts_weight_mode`; call after optimizer
        steps under FSDP2)."""
        cache = getattr(self, "_sglang_fused_weight_cache", None)
        if cache is not None:
            cache.clear()

    @staticmethod
    def _log_sglang_fused_experts_config_once(
        w13: torch.Tensor,
        w2: torch.Tensor,
        selected_flat: torch.Tensor,
        flag_attr: str = "_sglang_fused_experts_config_logged",
        label: str = "local",
    ) -> None:
        """Log the kernel launch config once so cross-engine gates can assert equality."""
        if getattr(MoEExperts, flag_attr, False):
            return
        setattr(MoEExperts, flag_attr, True)
        try:
            from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: PLC0415
                try_get_optimal_moe_config,
            )

            m = int(selected_flat.shape[0])
            config, (down_config, _) = try_get_optimal_moe_config(
                tuple(w13.shape),
                tuple(w2.shape),
                int(selected_flat.shape[1]),
                None,
                m,
                return_down_config=True,
            )
            logger.info(
                "[%s:%s] fused_experts_impl E=%d N=%d K=%d M=%d topk=%d config=%s down_config=%s",
                _MOE_SGLANG_FUSED_EXPERTS_ENV,
                label,
                int(w13.shape[0]),
                int(w13.shape[1]),
                int(w13.shape[2]),
                m,
                int(selected_flat.shape[1]),
                config,
                down_config,
            )
        except Exception as exc:  # pragma: no cover - logging must never break the forward
            logger.warning("[%s] config logging failed: %s", _MOE_SGLANG_FUSED_EXPERTS_ENV, exc)

    def sglang_fused_experts_ep_compute(
        self,
        permute_tokens: torch.Tensor,
        cumsum: torch.Tensor,
        expert_scores: torch.Tensor,
    ) -> torch.Tensor:
        """K3 parity mode: per-rank EP expert compute through SGLang's serving kernel.

        Runs the post-dispatch pair rows through ``fused_experts_impl`` presented
        as a topk=1 batch: each dispatched row is its own "token" routed to exactly
        one LOCAL expert (ids rebuilt from the per-local-expert ``cumsum`` via
        ``repeat_interleave``), with its dispatched routing weight as the single
        top-k weight in fp32. The kernel applies the routing weight on the fp32
        down-GEMM accumulator (``MUL_ROUTED_WEIGHT``) and, at topk=1, writes the
        weighted rows directly to the output — bit-identical per-(token, slot)
        contributions to the serving engine's full-width call (validated on the
        q30 layer-0 winner dumps), so the existing alltoall fp32 scatter-add
        combine reproduces SGLang's ``moe_experts_output``.

        This replaces the stock EP compute's post-hoc weight multiply
        (``down_output.mul_(expert_scores)`` after the bf16 down-GEMM cast in
        ``xorl.ops.moe.triton``), whose double rounding flips ~26% of per-slot
        contributions vs serving.

        Weight layout: the LOCAL slice of xorl GKN ``gate_up_proj [E_local, H, 2I]``
        (gate-first) / ``down_proj [E_local, I, H]`` is presented as SGLang's
        ``w13 [E_local, 2I, H]`` / ``w2 [E_local, H, I]`` per
        :func:`moe_sglang_fused_experts_weight_mode`: a zero-copy transpose-view
        (strided mode, default), a transient transpose-copy (~150 MB per q30
        EP8 layer), or a cached copy (~7 GB/rank model-wide at q30 EP8, with
        explicit FSDP2 invalidation after optimizer steps) — all three
        bit-identical.

        Trainable: when gradients are required the same serving forward runs
        under :class:`_SglangFusedExpertsEPTrainFunction`, whose backward is
        xorl's proven after-down EP backward (dX into the pair rows, dW in GKN
        layout on the local slice, and d(pair_scores) so the router trains
        through the existing score-dispatch autograd). The no-grad path is
        unchanged.
        """
        if expert_scores is None:
            raise ValueError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 EP compute requires dispatched expert scores")
        if not self.gated:
            raise NotImplementedError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 supports gated MoE experts only")
        if self.down_bias is not None:
            raise NotImplementedError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support down_bias")
        if self.hidden_act not in {"silu", "gelu", "gelu_tanh"}:
            raise NotImplementedError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support hidden_act={self.hidden_act!r}"
            )
        _validate_sglang_swiglu_limit(self.swiglu_limit)

        try:
            fused_experts_impl = self._load_sglang_fused_experts_impl()
        except ImportError as exc:
            raise ImportError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 requires an environment with sglang and sgl_kernel "
                "installed (put the SGLang python tree on PYTHONPATH)"
            ) from exc

        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act
        weight_cache = None
        if moe_sglang_fused_experts_weight_mode() == "cached":
            weight_cache = getattr(self, "_sglang_fused_weight_cache", None)
            if weight_cache is None:
                weight_cache = {}
                self._sglang_fused_weight_cache = weight_cache

        needs_grad = torch.is_grad_enabled() and (
            permute_tokens.requires_grad
            or expert_scores.requires_grad
            or self.gate_up_proj.requires_grad
            or self.down_proj.requires_grad
        )
        if needs_grad:
            if self.gate_up_bias is not None:
                raise NotImplementedError(
                    f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support training with gate_up_bias"
                )
            if self.hidden_act not in {"silu", "gelu_tanh"}:
                raise NotImplementedError(
                    f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 EP training supports silu/gelu_tanh only "
                    f"(got hidden_act={self.hidden_act!r})"
                )
            return _SglangFusedExpertsEPTrainFunction.apply(
                permute_tokens,
                expert_scores,
                self.gate_up_proj,
                self.down_proj,
                cumsum,
                fused_experts_impl,
                activation,
                self.hidden_act,
                self.swiglu_limit,
                self.gated,
                weight_cache,
            )

        permute_tokens = permute_tokens.contiguous()
        if permute_tokens.shape[0] == 0:
            # A rank can legitimately receive zero pair rows; the serving kernel
            # is never entered on empty extend batches, so short-circuit.
            return permute_tokens.new_zeros(permute_tokens.shape)

        return _sglang_fused_experts_ep_kernel_call(
            permute_tokens,
            self.gate_up_proj,
            self.down_proj,
            expert_scores,
            cumsum,
            fused_experts_impl,
            activation,
            self.swiglu_limit,
            self.gate_up_bias,
            gated=self.gated,
            weight_cache=weight_cache,
        )

    def _sglang_fused_experts_ep_compute_fn(self):
        """Adapter matching the ``EP_EXPERT_COMPUTE`` call signature.

        ``_ep_forward`` calls compute functions as
        ``compute_fn(permute_tokens, cumsum, gate_up_proj, down_proj,
        intermediate_size, expert_scores, **kwargs)``; the serving-kernel path
        reads weights and expert metadata from ``self`` instead.
        """

        def _compute(permute_tokens, cumsum, _gate_up_proj, _down_proj, _intermediate_size, expert_scores, **_kwargs):
            return self.sglang_fused_experts_ep_compute(permute_tokens, cumsum, expert_scores)

        return _compute

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
        # Parity EP compute is explicit-opt-in only: the auto (unset) default never
        # engages it under EP (see moe_sglang_fused_experts_enabled).
        explicit_sglang_fused = _moe_sglang_fused_experts_env_state()
        if explicit_sglang_fused is None:
            _log_moe_sglang_fused_experts_auto_once(
                f"sglang-fused experts auto-disabled (ep={getattr(parallel_state, 'ep_size', '>1')}): "
                f"EP keeps the configured expert backend; set {_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 to opt in "
                "(alltoall dispatch only)"
            )
        sglang_fused_experts = explicit_sglang_fused is True
        if sglang_fused_experts and self.ep_dispatch != "alltoall":
            raise NotImplementedError(
                f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 supports ep_dispatch='alltoall' only: the DeepEP "
                "combine's order and rounding schedule differ, and serving's reduction chain cannot "
                "currently be reproduced on that path. No mechanism is asserted here — the installed "
                "DeepEP wheel ships no CUDA source, so its reduction is not inspectable from the box "
                "this runs on."
            )
        if sglang_fused_experts and self.fp8_training_enabled:
            raise NotImplementedError(f"{_MOE_SGLANG_FUSED_EXPERTS_ENV}=1 does not support FP8 MoE compute under EP")
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. Available: {list(EP_DISPATCH.keys())}"
            )
        if self.fp8_training_enabled and self.moe_implementation != "quack":
            raise NotImplementedError("FP8 grouped MoE compute currently requires moe_implementation='quack'")
        self.last_forward_used_fp8 = bool(self.fp8_training_enabled)

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]

        if self._moe_act and self.moe_implementation in EP_EXPERT_COMPUTE_MOE_ACT:
            compute_fn = EP_EXPERT_COMPUTE_MOE_ACT[self.moe_implementation]
        else:
            compute_fn = EP_EXPERT_COMPUTE[self.moe_implementation]
        if sglang_fused_experts:
            # K3 parity mode: the serving-kernel expert compute overrides the
            # configured backend; dispatch and combine stay on the stock
            # alltoall path.
            compute_fn = self._sglang_fused_experts_ep_compute_fn()

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(hidden_states, routing_weights, selected_experts, parallel_state)
        deepep_diagnostic_id = _acquire_deepep_parity_diagnostic_record()

        if _DEBUG_EP:
            return self._ep_forward_debug(
                dispatch_fn,
                combine_fn,
                compute_fn,
                dispatch_kwargs,
                parallel_state,
            )

        quack_deepep_no_permute = (
            self.ep_dispatch == "deepep" and self.moe_implementation == "quack" and not _FORCE_QUACK_DEEPEP_GENERIC
        )
        if quack_deepep_no_permute:
            from xorl.distributed.moe.deepep import token_pre_dispatch_no_permute  # noqa: PLC0415

            permute_tokens, cumsum, ctx = token_pre_dispatch_no_permute(**dispatch_kwargs)
        else:
            permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)
        if deepep_diagnostic_id is not None:
            self._emit_deepep_parity_diagnostic(
                record_id=deepep_diagnostic_id,
                phase="post_dispatch",
                dispatch_kwargs=dispatch_kwargs,
                parallel_state=parallel_state,
                permute_tokens=permute_tokens,
                cumsum=cumsum,
                ctx=ctx,
                quack_deepep_no_permute=quack_deepep_no_permute,
            )

        if _FORCE_SYNC:
            torch.cuda.synchronize()

        # Warmup: pre-compile the selected group-GEMM backend variants to avoid
        # first-use compilation memory spikes during training. "triton_w4a4" (the QARL
        # W4A4 down-quant shadow) runs the identical triton group-GEMM kernels, so it
        # must warm up too — otherwise the first W4A4 step pays the compile-memory spike
        # this warmup exists to prevent. Under the parity flag the forward never runs
        # xorl group GEMMs, but the trainable path's backward always uses the stock
        # TRITON grouped GEMMs — warm those when gradients will flow.
        sglang_fused_experts_train = (
            sglang_fused_experts
            and torch.is_grad_enabled()
            and (
                permute_tokens.requires_grad
                or dispatch_kwargs["routing_weights"].requires_grad
                or self.gate_up_proj.requires_grad
                or self.down_proj.requires_grad
            )
        )
        warmup_impl = "triton" if sglang_fused_experts else self.moe_implementation
        warmup_attr = f"_kernel_warmed_up_{warmup_impl}_{'fp8' if self.fp8_training_enabled else 'bf16'}"
        if (
            (not sglang_fused_experts or (sglang_fused_experts_train and permute_tokens.is_cuda))
            and warmup_impl in {"triton", "triton_w4a4", "quack"}
            and not getattr(type(self), warmup_attr, False)
        ):
            if warmup_impl == "quack":
                if self.fp8_training_enabled:
                    from xorl.fp8_training import grouped as _fp8_grouped  # noqa: PLC0415

                    _warmup_mn = _fp8_grouped.fp8_group_gemm_same_mn
                    _warmup_gemm = _fp8_grouped.fp8_group_gemm_same_nk
                else:
                    from xorl.ops.group_gemm.kernel import quack as _quack_grouped  # noqa: PLC0415

                    _warmup_mn = _quack_grouped.quack_group_gemm_same_mn
                    _warmup_gemm = _quack_grouped.quack_group_gemm_same_nk
            else:
                from xorl.ops.group_gemm.kernel import group_gemm as _triton_grouped  # noqa: PLC0415

                _warmup_mn = _triton_grouped.group_gemm_same_mn
                _warmup_gemm = _triton_grouped.group_gemm_same_nk

            _d = permute_tokens.device
            _dt = permute_tokens.dtype
            _H = self.gate_up_proj.shape[1]
            _I = self.intermediate_size
            _N = self.gate_up_proj.shape[2]  # 2*I gated, I non-gated
            _E = self.gate_up_proj.shape[0]
            _M = _E * 2
            _cum = torch.arange(2, _M + 2, 2, dtype=torch.int32, device=_d)

            # Forward GEMM: x @ gate_up_proj
            _x = torch.zeros(_M, _H, dtype=_dt, device=_d)
            _w = torch.zeros(_E, _H, _N, dtype=_dt, device=_d)
            _fp8_backend = os.environ.get("XORL_FP8_MOE_GROUPED_BACKEND", self.fp8_training_grouped_backend).strip()
            _fp8_kwargs = (
                {"backend": _fp8_backend, "block_size": self.fp8_training_block_size}
                if self.fp8_training_enabled
                else {}
            )
            _warmup_gemm(a=_x, b=_w, cumsum_M=_cum, max_M=2, **_fp8_kwargs)

            # Backward dgrad FC1: grad_gate_up_act @ gate_up_proj^T
            _g = torch.zeros(_M, _N, dtype=_dt, device=_d)
            _warmup_gemm(a=_g, b=_w, cumsum_M=_cum, max_M=2, transpose_b=True, **_fp8_kwargs)

            # Backward dgrad FC2: grad @ down_proj^T
            _wd = torch.zeros(_E, _I, _H, dtype=_dt, device=_d)
            _gd = torch.zeros(_M, _H, dtype=_dt, device=_d)
            _warmup_gemm(a=_gd, b=_wd, cumsum_M=_cum, max_M=2, transpose_b=True, **_fp8_kwargs)

            # Backward wgrad FC1: permute_tokens^T @ grad_gate_up_act
            _c = torch.zeros(_E, _H, _N, dtype=_dt, device=_d)
            _warmup_mn(a=_x, b=_g, c=_c, cumsum_K=_cum, max_K=2, transpose_a=True, **_fp8_kwargs)

            del _x, _w, _g, _gd, _wd, _c, _cum, _fp8_kwargs
            torch.cuda.empty_cache()
            setattr(type(self), warmup_attr, True)

        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        if quack_deepep_no_permute:
            from xorl.ops.moe.quack import QuackEPDeepEPNoPermute  # noqa: PLC0415

            result = QuackEPDeepEPNoPermute.apply(
                permute_tokens,
                cumsum,
                self.gate_up_proj,
                self.down_proj,
                self.intermediate_size,
                expert_scores,
                dispatch_kwargs["buffer"],
                ctx,
                self.deepep_async_combine,
                self.hidden_act,
                self.activation_native,
                self.fp8_training_enabled,
                self.fp8_training_grouped_backend,
                self.fp8_training_block_size,
                self.gate_up_bias,
                self.down_bias,
            )
            if deepep_diagnostic_id is not None:
                self._emit_deepep_parity_diagnostic(
                    record_id=deepep_diagnostic_id,
                    phase="post_fused_no_permute",
                    dispatch_kwargs=dispatch_kwargs,
                    parallel_state=parallel_state,
                    permute_tokens=permute_tokens,
                    cumsum=cumsum,
                    ctx=ctx,
                    result=result,
                    quack_deepep_no_permute=True,
                )
            return result

        expert_output = compute_fn(
            permute_tokens,
            cumsum,
            self.gate_up_proj,
            self.down_proj,
            self.intermediate_size,
            expert_scores,
            hidden_act=self.hidden_act,
            activation_native=self.activation_native,
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
            fp8_compute=self.fp8_training_enabled,
            fp8_grouped_backend=self.fp8_training_grouped_backend,
            fp8_block_size=self.fp8_training_block_size,
            gated=self.gated,
        )
        if deepep_diagnostic_id is not None:
            self._emit_deepep_parity_diagnostic(
                record_id=deepep_diagnostic_id,
                phase="post_compute",
                dispatch_kwargs=dispatch_kwargs,
                parallel_state=parallel_state,
                permute_tokens=permute_tokens,
                cumsum=cumsum,
                ctx=ctx,
                expert_output=expert_output,
                quack_deepep_no_permute=False,
            )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
        result = combine_fn(**combine_kwargs)
        if deepep_diagnostic_id is not None:
            self._emit_deepep_parity_diagnostic(
                record_id=deepep_diagnostic_id,
                phase="post_combine",
                dispatch_kwargs=dispatch_kwargs,
                parallel_state=parallel_state,
                permute_tokens=permute_tokens,
                cumsum=cumsum,
                ctx=ctx,
                expert_output=expert_output,
                result=result,
                quack_deepep_no_permute=False,
            )
        return result

    def _emit_deepep_parity_diagnostic(
        self,
        *,
        record_id: int,
        phase: str,
        dispatch_kwargs: dict,
        parallel_state,
        permute_tokens: torch.Tensor | None = None,
        cumsum: torch.Tensor | None = None,
        ctx=None,
        expert_output: torch.Tensor | None = None,
        result: torch.Tensor | None = None,
        quack_deepep_no_permute: bool = False,
    ) -> None:
        ep_group = parallel_state.ep_group
        ep_size = _ep_group_size(ep_group)
        ep_rank = _ep_group_rank(ep_group)
        num_local_experts = int(self.gate_up_proj.shape[0])
        num_experts = int(self.num_experts)
        expected_num_local_experts = num_experts // ep_size if ep_size and num_experts % ep_size == 0 else None
        local_start = ep_rank * num_local_experts
        expected_local_start = ep_rank * expected_num_local_experts if expected_num_local_experts is not None else None
        cumsum_length = int(cumsum.numel()) if cumsum is not None else None
        payload = {
            "tag": "xorl_deepep_parity_diagnostic",
            "record_id": record_id,
            "phase": phase,
            "rank": _distributed_rank(),
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "ep_dispatch": self.ep_dispatch,
            "moe_implementation": self.moe_implementation,
            "quack_deepep_no_permute": quack_deepep_no_permute,
            "fp8_training_enabled": bool(self.fp8_training_enabled),
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "expected_num_local_experts": expected_num_local_experts,
            "local_expert_global_range": [local_start, local_start + num_local_experts],
            "expected_local_expert_global_range": (
                [expected_local_start, expected_local_start + expected_num_local_experts]
                if expected_local_start is not None and expected_num_local_experts is not None
                else None
            ),
            "num_local_experts_matches_expected": (
                num_local_experts == expected_num_local_experts if expected_num_local_experts is not None else None
            ),
            "cumsum_length": cumsum_length,
            "cumsum_length_matches_num_local_experts": (
                cumsum_length == num_local_experts if cumsum_length is not None else None
            ),
            "cumsum_length_matches_expected_num_local_experts": (
                cumsum_length == expected_num_local_experts
                if cumsum_length is not None and expected_num_local_experts is not None
                else None
            ),
            "hidden_states": _tensor_summary(dispatch_kwargs.get("hidden_states")),
            "routing_weights": _tensor_summary(dispatch_kwargs.get("routing_weights")),
            "selected_experts": _selected_experts_summary(
                dispatch_kwargs.get("selected_experts"),
                int(self.num_experts),
            ),
            "gate_up_proj": _tensor_summary(self.gate_up_proj, include_sample=False),
            "down_proj": _tensor_summary(self.down_proj, include_sample=False),
            "permute_tokens": _tensor_summary(permute_tokens),
            "cumsum": _cumsum_summary(cumsum),
            "dispatch_ctx": _dispatch_context_summary(ctx),
            "expert_output": _tensor_summary(expert_output),
            "expert_output_reference": (
                _safe_expert_output_reference_comparison(
                    permute_tokens=permute_tokens,
                    cumsum=cumsum,
                    gate_up_proj=self.gate_up_proj,
                    down_proj=self.down_proj,
                    intermediate_size=self.intermediate_size,
                    expert_scores=getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None)),
                    hidden_act=self.hidden_act,
                    gate_up_bias=self.gate_up_bias,
                    down_bias=self.down_bias,
                    expert_output=expert_output,
                )
                if phase == "post_compute"
                else None
            ),
            "result_reference": (
                _safe_result_reference_comparison(
                    hidden_states=dispatch_kwargs.get("hidden_states"),
                    routing_weights=dispatch_kwargs.get("routing_weights"),
                    selected_experts=dispatch_kwargs.get("selected_experts"),
                    gate_up_proj=self.gate_up_proj,
                    down_proj=self.down_proj,
                    intermediate_size=self.intermediate_size,
                    hidden_act=self.hidden_act,
                    gate_up_bias=self.gate_up_bias,
                    down_bias=self.down_bias,
                    result=result,
                    ep_group=ep_group,
                )
                if phase in {"post_combine", "post_fused_no_permute"}
                else None
            ),
            "result": _tensor_summary(result),
        }
        print(f"[DEEPEP PARITY] {json.dumps(payload, sort_keys=True)}", flush=True)

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
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
            fp8_compute=self.fp8_training_enabled,
            fp8_grouped_backend=self.fp8_training_grouped_backend,
            fp8_block_size=self.fp8_training_block_size,
            gated=self.gated,
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
            f"  gate_up_proj: shape={self.gate_up_proj.shape}, "
            f"contiguous={self.gate_up_proj.is_contiguous()}, stride={self.gate_up_proj.stride()}\n"
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

    @classmethod
    def from_config(cls, config, moe_implementation: str = "triton"):
        """Create from a model config (e.g. ``Qwen3MoeConfig``)."""
        return cls(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            moe_implementation=moe_implementation,
            activation_native=getattr(config, "_activation_native", False),
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0)),
        )
