from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import get_parallel_state
from xorl.ops.linear_attention.utils import tensor_cache


@dataclass
class FLACPContext:
    """Operator-level context metadata for native linear-attention CP."""

    group: dist.ProcessGroup | None = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None

    def copy_for_backward(self) -> FLACPContext:
        return FLACPContext(
            group=self.group,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            cu_seqlens_cpu=self.cu_seqlens_cpu.clone() if self.cu_seqlens_cpu is not None else None,
            is_last_rank=self.is_last_rank,
            pre_num_ranks=self.pre_num_ranks,
            is_first_rank=self.is_first_rank,
            post_num_ranks=self.post_num_ranks,
            conv1d_kernel_size=self.conv1d_kernel_size,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
        )

    @property
    def num_seqs(self) -> int:
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        return self.group is not None


LinearAttentionCPContext = FLACPContext


@tensor_cache
def get_cp_cu_seqlens(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: dist.ProcessGroup | None = None,
    conv1d_kernel_size: int | None = None,
) -> FLACPContext:
    if world_size is None:
        if group is None:
            raise ValueError("group must be provided when world_size is not specified.")
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
    if rank is None:
        raise ValueError("rank must be provided when world_size is specified.")

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    total_tokens = int(cu_seqlens_cpu[-1].item())
    if total_tokens % world_size != 0:
        raise ValueError(
            f"Native FLA CP expects evenly sharded tokens, got total_tokens={total_tokens} "
            f"for world_size={world_size}.",
        )
    part_len = total_tokens // world_size
    rank_start = part_len * rank
    rank_end = rank_start + part_len

    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side="right")
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side="left")

    subset_cu_seqlens = cu_seqlens_cpu[start_seq_idx : end_seq_idx + 1]
    local_cu_seqlens_cpu = (
        subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start
    ).unique_consecutive().to(torch.int32)
    local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(device=cu_seqlens.device, non_blocking=True)

    first_seq_global_start = int(cu_seqlens_cpu[start_seq_idx].item())
    last_seq_global_end = int(cu_seqlens_cpu[end_seq_idx].item())
    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)

    first_rank_of_first_seq = first_seq_global_start // part_len
    pre_num_ranks = rank - first_rank_of_first_seq
    is_first_rank = rank == first_rank_of_first_seq

    last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
    post_num_ranks = last_rank_of_last_seq - rank
    is_last_rank = rank == last_rank_of_last_seq

    return FLACPContext(
        group=group,
        cu_seqlens=local_cu_seqlens_gpu,
        cu_seqlens_cpu=local_cu_seqlens_cpu,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        is_first_rank=is_first_rank,
        post_num_ranks=post_num_ranks,
        conv1d_kernel_size=conv1d_kernel_size,
        pre_num_conv_tokens=pre_num_conv_tokens,
    )


def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: dist.ProcessGroup,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext:
    return get_cp_cu_seqlens(
        cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        group=group,
        conv1d_kernel_size=conv1d_kernel_size,
    )


@torch.compiler.disable
def build_linear_attention_cp_context(
    cu_seqlens: torch.Tensor | None = None,
    *,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext | None:
    ps = get_parallel_state()
    if ps.ulysses_size <= 1:
        return None
    # Native FLA CP assumes the layer already lives on Ulysses-local sequence shards.
    # Ring / hybrid layouts need an additional relayout step that is not implemented yet.
    if ps.ringattn_size > 1:
        return None
    if cu_seqlens is None:
        return FLACPContext(group=ps.ulysses_group, conv1d_kernel_size=conv1d_kernel_size)
    return build_cp_context(
        cu_seqlens=cu_seqlens,
        group=ps.ulysses_group,
        conv1d_kernel_size=conv1d_kernel_size,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
