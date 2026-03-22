from __future__ import annotations

# Adapted from flash-linear-attention/fla/layers/utils.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.

import torch
from einops import rearrange, repeat

from xorl.ops.linear_attention.utils import tensor_cache


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = x.shape[0]
        other_shape = x.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(x, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        other_shape = do.shape[1:]
        do = rearrange(do, "b ... -> b (...)")
        dx = torch.zeros((ctx.first_axis_dim, do.shape[1]), device=do.device, dtype=do.dtype)
        dx.scatter_(0, repeat(indices, "z -> z d", d=do.shape[1]), do)
        return dx.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
        ctx.save_for_backward(indices)
        y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
        y[indices] = x
        return y

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        return do[indices], None, None


index_put_first_axis = IndexPutFirstAxis.apply


@tensor_cache
def get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(lens.max().item()) if lens.numel() else 0
    cu_seqlens = torch.nn.functional.pad(lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)
