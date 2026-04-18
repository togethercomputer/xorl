import os

import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import get_parallel_state
from xorl.ops.group_gemm.kernel.moe import expert_histogram, moe_gather, moe_index_compute, moe_scatter
from xorl.ops.group_gemm.kernel.quack import cumsum_to_cu_seqlens, quack_group_gemm_same_mn, quack_group_gemm_same_nk


def _debug_ep_enabled() -> bool:
    v = os.environ.get("XORL_DEBUG_EP", "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


_DEBUG_EP = _debug_ep_enabled()


def _scatter_and_cumsum(hidden_states: torch.Tensor, expert_index: torch.Tensor, num_experts: int):
    splits = expert_histogram(expert_index, num_experts)
    cumsum_t = torch.cumsum(splits, dim=0)
    scatter_index = moe_index_compute(expert_index, cumsum_t)
    scatter_output = moe_scatter(hidden_states, scatter_index)
    return scatter_output, scatter_index, cumsum_t


class QuackMoeExpertsFunction(torch.autograd.Function):
    """Memory-optimized: separate gate/up GEMMs, recompute cheap intermediates,
    explicit del for dead tensors, in-place add for dgrad."""

    @staticmethod
    def forward(
        ctx, num_experts, gate_weights, expert_index, hidden_states, gate_proj, up_proj, down_proj, gate_up_weight=None
    ):
        scatter_output, scatter_index, cumsum_t = _scatter_and_cumsum(hidden_states, expert_index, num_experts)
        max_M = scatter_output.shape[0]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum_t)

        gate_output = quack_group_gemm_same_nk(
            a=scatter_output, b=gate_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        up_output = quack_group_gemm_same_nk(
            a=scatter_output, b=up_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        del scatter_output

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        del gate_activation

        scattered_gate_weight = torch.empty_like(gate_weights.reshape(-1, 1))
        scattered_gate_weight[scatter_index.flatten()] = gate_weights.reshape(-1, 1)
        gated_weighted = gated_activation * scattered_gate_weight
        del gated_activation

        down_output = quack_group_gemm_same_nk(
            a=gated_weighted, b=down_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        del gated_weighted
        output = moe_gather(down_output, scatter_index).reshape(hidden_states.shape)
        del down_output

        ctx.save_for_backward(
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum_t)

        # Recompute cheap intermediates (avoids saving them)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        gated_weighted = gated_activation * scattered_gate_weight

        grad_down_output = moe_scatter(grad_output, scatter_index)

        # dgrad FC2
        grad_gated_weighted = quack_group_gemm_same_nk(
            a=grad_down_output, b=down_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )

        # wgrad FC2
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            quack_group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_down_output, gated_weighted

        # Routing weight gradient
        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        del gated_activation, grad_gated_weighted

        # Activation backward
        grad_up_output = gate_activation * grad_gated_activation
        grad_gate_activation = grad_gated_activation * up_output
        del grad_gated_activation, gate_activation, up_output
        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # dgrad FC1: in-place add
        grad_scatter_output = quack_group_gemm_same_nk(
            a=grad_gate_output, b=gate_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )
        grad_scatter_output += quack_group_gemm_same_nk(
            a=grad_up_output, b=up_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )

        # wgrad FC1
        grad_gate_proj = None
        if gate_proj.requires_grad:
            grad_gate_proj = torch.empty_like(gate_proj)
            quack_group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_output,
                c=grad_gate_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_gate_output
        grad_up_proj = None
        if up_proj.requires_grad:
            grad_up_proj = torch.empty_like(up_proj)
            quack_group_gemm_same_mn(
                a=scatter_output,
                b=grad_up_output,
                c=grad_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_up_output, scatter_output

        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)
        return None, grad_gate_weight, None, grad_hidden_states, grad_gate_proj, grad_up_proj, grad_down_proj, None


class QuackEPGroupGemm(torch.autograd.Function):
    """Memory-optimized EP expert GEMM. Recomputes cheap intermediates, explicit del."""

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, gate_proj, up_proj, down_proj, expert_scores=None):
        max_M = permute_tokens.shape[0]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum)
        ctx.has_expert_scores = expert_scores is not None

        if _DEBUG_EP:
            return QuackEPGroupGemm._forward_debug(
                ctx,
                permute_tokens,
                cumsum,
                gate_proj,
                up_proj,
                down_proj,
                expert_scores,
                max_M,
                cu_seqlens,
            )

        gate_output = quack_group_gemm_same_nk(
            a=permute_tokens, b=gate_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        up_output = quack_group_gemm_same_nk(
            a=permute_tokens, b=up_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_output = gate_activation * up_output
        if expert_scores is not None:
            gated_output = gated_output * expert_scores.to(gated_output.dtype).unsqueeze(-1)
        del gate_activation

        down_output = quack_group_gemm_same_nk(
            a=gated_output, b=down_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        del gated_output

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(
            permute_tokens, cumsum, gate_proj, up_proj, down_proj, gate_output, up_output, expert_scores
        )
        return down_output

    @staticmethod
    def _forward_debug(ctx, permute_tokens, cumsum, gate_proj, up_proj, down_proj, expert_scores, max_M, cu_seqlens):
        """Instrumented forward with per-GEMM CUDA event timing."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        ctx.has_expert_scores = expert_scores is not None

        ev[0].record()
        gate_output = quack_group_gemm_same_nk(
            a=permute_tokens, b=gate_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        ev[1].record()

        up_output = quack_group_gemm_same_nk(
            a=permute_tokens, b=up_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        ev[2].record()

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_output = gate_activation * up_output
        if expert_scores is not None:
            gated_output = gated_output * expert_scores.to(gated_output.dtype).unsqueeze(-1)
        del gate_activation
        ev[3].record()

        down_output = quack_group_gemm_same_nk(
            a=gated_output, b=down_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        ev[4].record()
        del gated_output

        torch.cuda.synchronize()
        t_gate = ev[0].elapsed_time(ev[1])
        t_up = ev[1].elapsed_time(ev[2])
        t_act = ev[2].elapsed_time(ev[3])
        t_down = ev[3].elapsed_time(ev[4])
        print(
            f"[QuackEP r{rank}] total_M={max_M} G={gate_proj.shape[0]} "
            f"K={gate_proj.shape[1]} N_gate={gate_proj.shape[2]} N_down={down_proj.shape[2]}\n"
            f"  cu_seqlens: dtype={cu_seqlens.dtype}, len={cu_seqlens.shape[0]}\n"
            f"  permute_tokens: stride={permute_tokens.stride()}, contiguous={permute_tokens.is_contiguous()}\n"
            f"  gate GEMM:  {t_gate:7.2f} ms\n"
            f"  up GEMM:    {t_up:7.2f} ms\n"
            f"  silu+mul:   {t_act:7.2f} ms\n"
            f"  down GEMM:  {t_down:7.2f} ms\n"
            f"  total:      {t_gate + t_up + t_act + t_down:7.2f} ms",
            flush=True,
        )

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(
            permute_tokens, cumsum, gate_proj, up_proj, down_proj, gate_output, up_output, expert_scores
        )
        return down_output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, cumsum, gate_proj, up_proj, down_proj, gate_output, up_output, expert_scores = ctx.saved_tensors
        max_M = grad_output.shape[0]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum)

        # Recompute cheap intermediates
        gate_activation = torch.ops.aten.silu(gate_output)
        gated_output = gate_activation * up_output
        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)
        gated_weighted = gated_output * expert_scores.unsqueeze(-1)

        # dgrad FC2
        grad_gated_weighted = quack_group_gemm_same_nk(
            a=grad_output, b=down_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )

        # wgrad FC2
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            quack_group_gemm_same_mn(
                a=gated_weighted,
                b=grad_output,
                c=grad_down_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        grad_expert_scores = None
        if ctx.has_expert_scores:
            grad_expert_scores = (grad_gated_weighted * gated_output).sum(dim=-1).to(expert_scores_dtype)
        del gated_output, gated_weighted

        grad_gated_output = grad_gated_weighted * expert_scores.unsqueeze(-1)
        del grad_gated_weighted

        # Activation backward
        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output
        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # dgrad FC1: in-place add
        grad_permute_tokens = quack_group_gemm_same_nk(
            a=grad_gate_output, b=gate_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )
        grad_permute_tokens += quack_group_gemm_same_nk(
            a=grad_up_output, b=up_proj, cumsum_M=cumsum, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )

        # wgrad FC1
        grad_gate_proj = None
        if gate_proj.requires_grad:
            grad_gate_proj = torch.empty_like(gate_proj)
            quack_group_gemm_same_mn(
                a=permute_tokens,
                b=grad_gate_output,
                c=grad_gate_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_gate_output
        grad_up_proj = None
        if up_proj.requires_grad:
            grad_up_proj = torch.empty_like(up_proj)
            quack_group_gemm_same_mn(
                a=permute_tokens,
                b=grad_up_output,
                c=grad_up_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_up_output

        return grad_permute_tokens, None, grad_gate_proj, grad_up_proj, grad_down_proj, grad_expert_scores


class QuackTPMoeExpertsFunction(torch.autograd.Function):
    """Memory-optimized TP expert function. Recomputes cheap intermediates, explicit del + all-reduce."""

    @staticmethod
    def forward(ctx, num_experts, gate_weights, expert_index, hidden_states, gate_proj, up_proj, down_proj, tp_group):
        scatter_output, scatter_index, cumsum_t = _scatter_and_cumsum(hidden_states, expert_index, num_experts)
        max_M = scatter_output.shape[0]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum_t)

        gate_output = quack_group_gemm_same_nk(
            a=scatter_output, b=gate_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        up_output = quack_group_gemm_same_nk(
            a=scatter_output, b=up_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        del scatter_output

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        del gate_activation

        scattered_gate_weight = torch.empty_like(gate_weights.reshape(-1, 1))
        scattered_gate_weight[scatter_index.flatten()] = gate_weights.reshape(-1, 1)
        gated_weighted = gated_activation * scattered_gate_weight
        del gated_activation

        down_output = quack_group_gemm_same_nk(
            a=gated_weighted, b=down_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=False, cu_seqlens_m=cu_seqlens
        )
        del gated_weighted
        dist.all_reduce(down_output, group=tp_group)
        output = moe_gather(down_output, scatter_index).reshape(hidden_states.shape)
        del down_output

        ctx.tp_group = tp_group
        ctx.save_for_backward(
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
        ) = ctx.saved_tensors
        tp_group = ctx.tp_group
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum_t)

        # Recompute cheap intermediates (avoids saving them)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        gated_weighted = gated_activation * scattered_gate_weight

        grad_down_output = moe_scatter(grad_output, scatter_index)

        # dgrad FC2
        grad_gated_weighted = quack_group_gemm_same_nk(
            a=grad_down_output, b=down_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )

        # wgrad FC2
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            quack_group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_down_output, gated_weighted

        # Routing weight gradient
        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        del gated_activation, grad_gated_weighted

        # Activation backward
        grad_up_output = gate_activation * grad_gated_activation
        grad_gate_activation = grad_gated_activation * up_output
        del grad_gated_activation, gate_activation, up_output
        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # dgrad FC1: in-place add
        grad_scatter_output = quack_group_gemm_same_nk(
            a=grad_gate_output, b=gate_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )
        grad_scatter_output += quack_group_gemm_same_nk(
            a=grad_up_output, b=up_proj, cumsum_M=cumsum_t, max_M=max_M, transpose_b=True, cu_seqlens_m=cu_seqlens_m
        )
        handle = dist.all_reduce(grad_scatter_output, group=tp_group, async_op=True)

        # wgrad FC1
        grad_gate_proj = None
        if gate_proj.requires_grad:
            grad_gate_proj = torch.empty_like(gate_proj)
            quack_group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_output,
                c=grad_gate_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_gate_output
        grad_up_proj = None
        if up_proj.requires_grad:
            grad_up_proj = torch.empty_like(up_proj)
            quack_group_gemm_same_mn(
                a=scatter_output,
                b=grad_up_output,
                c=grad_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
            )
        del grad_up_output, scatter_output

        handle.wait()
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)
        return None, grad_gate_weight, None, grad_hidden_states, grad_gate_proj, grad_up_proj, grad_down_proj, None


def quack_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_weight: torch.Tensor = None,
    **kwargs,
):
    """Forward pass for MoE experts using quack group GEMM (local/TP only).

    EP is handled centrally by ``MoEExperts._ep_forward()``.
    """
    del module
    parallel_state = get_parallel_state()

    if parallel_state.tp_enabled:
        tp_group = parallel_state.tp_mesh.get_group()
        return QuackTPMoeExpertsFunction.apply(
            num_experts, routing_weights, selected_experts, hidden_states, gate_proj, up_proj, down_proj, tp_group
        )

    return QuackMoeExpertsFunction.apply(
        num_experts, routing_weights, selected_experts, hidden_states, gate_proj, up_proj, down_proj, gate_up_weight
    )
