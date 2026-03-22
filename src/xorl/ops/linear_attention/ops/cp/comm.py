from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


class _AllGatherAlongDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, dim: int, group: ProcessGroup | None) -> Tensor:
        if group is None:
            ctx.group = None
            return x

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        moved = x.movedim(dim, 0).contiguous() if dim != 0 else x.contiguous()
        output_shape = list(moved.shape)
        output_shape[0] *= world_size
        output = torch.empty(output_shape, dtype=moved.dtype, device=moved.device)
        dist.all_gather_into_tensor(output, moved, group=group)

        ctx.group = group
        ctx.dim = dim
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.local_size = moved.shape[0]

        if dim != 0:
            output = output.movedim(0, dim).contiguous()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        if ctx.group is None:
            return grad_output, None, None

        moved = grad_output.movedim(ctx.dim, 0).contiguous() if ctx.dim != 0 else grad_output.contiguous()
        start = ctx.rank * ctx.local_size
        grad_input = moved.narrow(0, start, ctx.local_size).contiguous()
        if ctx.dim != 0:
            grad_input = grad_input.movedim(0, ctx.dim).contiguous()
        return grad_input, None, None


class _ScatterAlongDim(torch.autograd.Function):
    """Inverse of _AllGatherAlongDim: slice this rank's portion in forward,
    all-gather gradients in backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, dim: int, group: ProcessGroup | None) -> Tensor:
        if group is None:
            ctx.group = None
            return x

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = world_size

        moved = x.movedim(dim, 0).contiguous() if dim != 0 else x.contiguous()
        local_size = moved.shape[0] // world_size
        ctx.local_size = local_size
        result = moved.narrow(0, rank * local_size, local_size).contiguous()
        if dim != 0:
            result = result.movedim(0, dim).contiguous()
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        if ctx.group is None:
            return grad_output, None, None

        moved = grad_output.movedim(ctx.dim, 0).contiguous() if ctx.dim != 0 else grad_output.contiguous()
        output_shape = list(moved.shape)
        output_shape[0] *= ctx.world_size
        output = torch.empty(output_shape, dtype=moved.dtype, device=moved.device)
        dist.all_gather_into_tensor(output, moved, group=ctx.group)
        if ctx.dim != 0:
            output = output.movedim(0, ctx.dim).contiguous()
        return output, None, None


def all_gather_along_dim(x: Tensor, dim: int, group: ProcessGroup | None) -> Tensor:
    return _AllGatherAlongDim.apply(x, dim, group)


def scatter_along_dim(x: Tensor, dim: int, group: ProcessGroup | None) -> Tensor:
    return _ScatterAlongDim.apply(x, dim, group)


def all_gather_into_tensor(
    inp: Tensor,
    out: Tensor | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> tuple[Tensor, dist.Work | None]:
    world_size = dist.get_world_size(group=group)
    flat_shape = (world_size * inp.shape[0], *inp.shape[1:])
    if out is None:
        flat_out = torch.empty(flat_shape, device=inp.device, dtype=inp.dtype)
    else:
        flat_out = out.reshape(flat_shape)
    handle = dist.all_gather_into_tensor(flat_out, inp, group=group, async_op=async_op)
    return flat_out.view(world_size, *inp.shape), handle


def all_reduce_sum(
    inp: Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> tuple[Tensor, dist.Work | None]:
    handle = dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
    return inp, handle


def send_recv_fwd(
    send_tensor: Tensor,
    group: ProcessGroup,
    recv_from_prev: bool = True,
) -> Tensor:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)
    if recv_from_prev:
        return torch.zeros_like(send_tensor) if rank == 0 else gathered[rank - 1].clone()
    return torch.zeros_like(send_tensor) if rank == world_size - 1 else gathered[rank + 1].clone()


def send_recv_bwd(
    send_tensor: Tensor,
    group: ProcessGroup,
    recv_from_next: bool = True,
) -> Tensor:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)
    if recv_from_next:
        return torch.zeros_like(send_tensor) if rank == world_size - 1 else gathered[rank + 1].clone()
    return torch.zeros_like(send_tensor) if rank == 0 else gathered[rank - 1].clone()


def conv_cp_send_recv_fwd(tails: Tensor, group: ProcessGroup) -> Tensor:
    return send_recv_fwd(tails, group, recv_from_prev=True)


def conv_cp_send_recv_bwd(d_initial_state: Tensor, group: ProcessGroup) -> Tensor:
    return send_recv_bwd(d_initial_state, group, recv_from_next=True)
