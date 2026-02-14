"""DeepEP dispatch/combine utilities for NVLink-optimized Expert Parallel MoE."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from ...ops.ep_kernels import (
    deepep_scatter,
    deepep_weighted_gather,
    group_tokens_by_expert,
    group_tokens_by_expert_v2,
)

try:
    import deep_ep

    DEEPEP_AVAILABLE = True
except ImportError:
    DEEPEP_AVAILABLE = False
    deep_ep = None


def check_deepep_available():
    if not DEEPEP_AVAILABLE:
        raise ImportError(
            "DeepEP is not installed. Please install it from "
            "https://github.com/deepseek-ai/DeepEP"
        )


class DeepEPBuffer:
    def __init__(
        self,
        ep_group: Optional[dist.ProcessGroup] = None,
        buffer_size_gb: float = 2.0,
        low_latency_mode: bool = False,
    ):
        check_deepep_available()
        self.ep_group = ep_group
        self.buffer_size_gb = buffer_size_gb
        self.low_latency_mode = low_latency_mode
        self._buffer: Optional["deep_ep.Buffer"] = None
        self._dispatch_config = None
        self._combine_config = None

    def init_buffer(self):
        if self._buffer is not None:
            return
        num_nvl_bytes = int(self.buffer_size_gb * 1e9)
        self._buffer = deep_ep.Buffer(
            group=self.ep_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=0,
            low_latency_mode=self.low_latency_mode,
            explicitly_destroy=True,
        )
        num_ranks = self.ep_group.size() if self.ep_group else 1
        if num_ranks in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144, 160):
            self._dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            self._combine_config = deep_ep.Buffer.get_combine_config(num_ranks)

    def destroy_buffer(self):
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None

    def __del__(self):
        try:
            self.destroy_buffer()
        except Exception:
            pass

    @property
    def buffer(self) -> "deep_ep.Buffer":
        if self._buffer is None:
            self.init_buffer()
        return self._buffer

    @property
    def dispatch_config(self):
        self.init_buffer()
        return self._dispatch_config

    @property
    def combine_config(self):
        self.init_buffer()
        return self._combine_config


class _FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        buffer: "DeepEPBuffer",
        num_experts: int,
    ):
        buffer.init_buffer()
        topk_idx_deepep = topk_idx.to(deep_ep.topk_idx_t)
        topk_weights_f32 = topk_weights.to(torch.float32)
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.buffer.get_dispatch_layout(
            topk_idx_deepep, num_experts
        )
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, _ = buffer.buffer.dispatch(
            x=x.contiguous(),
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx_deepep,
            topk_weights=topk_weights_f32,
            config=buffer.dispatch_config,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.int32, device=x.device)
        return recv_x, recv_topk_idx, recv_topk_weights, recv_counts_tensor, handle

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_recv_counts, grad_handle):
        del grad_recv_topk_idx, grad_recv_topk_weights, grad_recv_counts, grad_handle
        buffer = ctx.buffer
        handle = ctx.handle
        if grad_recv_x is None:
            return None, None, None, None, None
        grad_x, _, _ = buffer.buffer.combine(
            x=grad_recv_x.contiguous(),
            handle=handle,
            config=buffer.combine_config,
        )
        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        return grad_x, None, None, None, None


class _FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        buffer: "DeepEPBuffer",
        handle: object,
    ):
        combined_x, _, _ = buffer.buffer.combine(
            x=x.contiguous(),
            handle=handle,
            config=buffer.combine_config,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        buffer = ctx.buffer
        handle = ctx.handle
        if grad_combined_x is None:
            return None, None, None
        grad_x, _, _, _, _, _ = buffer.buffer.dispatch(
            x=grad_combined_x.contiguous(),
            handle=handle,
            config=buffer.dispatch_config,
        )
        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        return grad_x, None, None


def fused_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buffer: "DeepEPBuffer",
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
    return _FusedDispatch.apply(x, topk_idx, topk_weights, buffer, num_experts)


def fused_combine(
    x: torch.Tensor,
    buffer: "DeepEPBuffer",
    handle: object,
) -> torch.Tensor:
    return _FusedCombine.apply(x, buffer, handle)


@dataclass
class DispatchContext:
    handle: object
    sorted_token_idx: torch.Tensor
    sorted_k_idx: torch.Tensor
    recv_topk_weights: torch.Tensor
    num_recv_tokens: int
    num_valid: int
    dtype: torch.dtype
    device: torch.device
    hidden_dim: int


def dispatch(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], object]:
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts_tensor, handle = fused_dispatch(
        hidden_states, selected_experts, routing_weights, buffer, num_experts
    )
    return recv_x, recv_topk_idx, recv_topk_weights, recv_counts_tensor, handle


def dispatch_no_grad(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], object]:
    buffer.init_buffer()
    topk_idx = selected_experts.to(deep_ep.topk_idx_t)
    topk_weights = routing_weights.to(torch.float32)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.buffer.get_dispatch_layout(topk_idx, num_experts)
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, _ = buffer.buffer.dispatch(
        x=hidden_states,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=buffer.dispatch_config,
    )
    return recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle


def combine(
    buffer: DeepEPBuffer,
    gather_output: torch.Tensor,
    handle: object,
) -> torch.Tensor:
    return fused_combine(gather_output, buffer, handle)


def combine_no_grad(
    buffer: DeepEPBuffer,
    gather_output: torch.Tensor,
    handle: object,
) -> torch.Tensor:
    combined_output, _, _ = buffer.buffer.combine(
        x=gather_output,
        handle=handle,
        config=buffer.combine_config,
    )
    return combined_output


def permute_for_experts(
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    num_local_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    topk = recv_topk_idx.shape[1]
    if num_local_experts > 0:
        sorted_token_idx, sorted_k_idx, num_valid = group_tokens_by_expert_v2(recv_topk_idx, topk, num_local_experts)
    else:
        sorted_token_idx, sorted_k_idx, num_valid = group_tokens_by_expert(recv_topk_idx, topk)
    if num_valid > 0:
        expert_input = deepep_scatter(recv_x, sorted_token_idx)
    else:
        expert_input = recv_x[:0]
    return expert_input, sorted_token_idx, sorted_k_idx, num_valid


def unpermute_from_experts(
    expert_output: torch.Tensor,
    sorted_token_idx: torch.Tensor,
    sorted_k_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_recv_tokens: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = expert_output.dtype
    hidden_dim = expert_output.shape[1]
    device = expert_output.device
    num_valid = expert_output.shape[0]
    if num_valid > 0:
        gather_output = deepep_weighted_gather(
            expert_output,
            sorted_token_idx,
            sorted_k_idx,
            recv_topk_weights.to(dtype),
            num_recv_tokens,
        )
    else:
        gather_output = torch.zeros((num_recv_tokens, hidden_dim), dtype=dtype, device=device)
    return gather_output


def token_pre_dispatch(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    num_local_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchContext]:
    device = hidden_states.device
    dtype = hidden_states.dtype
    hidden_dim = hidden_states.shape[1]
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts_tensor, handle = dispatch(
        buffer, hidden_states, routing_weights, selected_experts, num_experts
    )
    num_recv_tokens = recv_x.shape[0]
    cumsum = torch.cumsum(recv_counts_tensor, dim=0)
    if num_local_experts == 0:
        num_local_experts = recv_counts_tensor.shape[0]
    expert_input, sorted_token_idx, sorted_k_idx, num_valid = permute_for_experts(
        recv_x, recv_topk_idx, num_local_experts
    )
    ctx = DispatchContext(
        handle=handle,
        sorted_token_idx=sorted_token_idx,
        sorted_k_idx=sorted_k_idx,
        recv_topk_weights=recv_topk_weights,
        num_recv_tokens=num_recv_tokens,
        num_valid=num_valid,
        dtype=dtype,
        device=device,
        hidden_dim=hidden_dim,
    )
    return expert_input, cumsum, ctx


def tokens_post_combine(
    buffer: DeepEPBuffer,
    expert_output: torch.Tensor,
    ctx: DispatchContext,
) -> torch.Tensor:
    gather_output = unpermute_from_experts(
        expert_output,
        ctx.sorted_token_idx,
        ctx.sorted_k_idx,
        ctx.recv_topk_weights,
        ctx.num_recv_tokens,
        ctx.dtype,
    )
    return combine(buffer, gather_output, ctx.handle)


_default_buffer: Optional[DeepEPBuffer] = None


def get_default_buffer(
    ep_group: Optional[dist.ProcessGroup] = None,
    buffer_size_gb: float = 2.0,
) -> DeepEPBuffer:
    global _default_buffer
    if _default_buffer is None:
        _default_buffer = DeepEPBuffer(ep_group=ep_group, buffer_size_gb=buffer_size_gb)
    return _default_buffer


def destroy_default_buffer():
    global _default_buffer
    if _default_buffer is not None:
        _default_buffer.destroy_buffer()
        _default_buffer = None
