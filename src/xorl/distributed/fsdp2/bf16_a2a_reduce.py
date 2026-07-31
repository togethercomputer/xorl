"""Custom FSDP2 reduce-scatter that exchanges in BF16 and sums in FP32.

Implements the DeepSeek V4 §3.5.1 MoE gradient communication trick:

    1. Stochastically round FP32 input → BF16 (unbiased).
    2. ``all_to_all_single`` the BF16 buffer across the reduce-scatter group.
    3. Sum the received per-rank chunks locally in FP32.

Halves comm volume vs. native FP32 reduce-scatter while preserving FP32
numerical robustness in the accumulator. NCCL ring/tree reduce-scatter
accumulates in the buffer's dtype during transit, so naively setting
FSDP ``reduce_dtype=bf16`` would accumulate partial sums in BF16's 8
mantissa bits and accrue significant bias. Decoupling movement (BF16) from
accumulation (FP32) keeps the sum well-conditioned.

Installed via ``FSDPModule.set_custom_reduce_scatter(...)`` on expert
FSDPModules; non-expert modules continue to use the default reduce-scatter.

**Preconditions on the wrapping FSDPModule (enforced at install time in
``parallelize_model_fsdp2``; do NOT install this hook without them):**

  * ``mp_policy.reduce_dtype == torch.float32``. The hook accepts only FP32
    input — it relies on FSDP allocating the reduce-scatter buffer in FP32
    so that ``input_tensor`` arrives un-quantized; the BF16 transit is
    internal to the hook.
  * ``gradient_divide_factor == 1.0``. With factor=None FSDP enables a
    ``predivide_factor`` that is applied to ``input_tensor`` *before* this
    hook is invoked, and FSDP skips the postdivide because the reduce is
    custom. The hook would then sum predivided values without compensation,
    silently under-weighting gradients. The codebase calls
    ``set_gradient_divide_factor(1.0)`` on every FSDPModule before installing
    the hook (see ``torch_parallelize.py``); this hook will refuse to install
    if that invariant is broken.
"""

from typing import Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter, _ReduceOp

from xorl.optim.stochastic_round import stochastic_round_to_bf16


def _canonical_reduce_op(op: _ReduceOp) -> ReduceOp.RedOpType:
    """Return the underlying RedOpType for FSDP's wrapped or raw reduce op."""
    op_type = getattr(op, "op", op)
    op_name = getattr(op_type, "name", None)
    if op_name is None:
        op_name = str(op_type).rsplit(".", maxsplit=1)[-1]
    if op_name == "SUM":
        return ReduceOp.SUM
    if op_name == "AVG":
        return ReduceOp.AVG
    if op_name == "PREMUL_SUM":
        # FSDP emits _make_nccl_premul_sum(1 / gradient_divide_factor) when a
        # custom divide factor is set. The installer requires factor=1.0 for
        # this hook, so PREMUL_SUM is equivalent to SUM here.
        return ReduceOp.SUM
    return op_type


class BF16StochasticAllToAllReduceScatter(ReduceScatter):
    """ReduceScatter: stochastic-round FP32→BF16, all-to-all, FP32 local sum."""

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        if async_op:
            raise NotImplementedError("BF16StochasticAllToAllReduceScatter does not support async_op=True")
        op_type = _canonical_reduce_op(op)
        if op_type != ReduceOp.SUM and op_type != ReduceOp.AVG:
            raise NotImplementedError(
                f"BF16StochasticAllToAllReduceScatter requires SUM or AVG op, got {op} ({op_type})"
            )
        if input_tensor.dtype != torch.float32:
            raise ValueError(
                "BF16StochasticAllToAllReduceScatter requires FP32 input "
                f"(set FSDP reduce_dtype=fp32), got {input_tensor.dtype}"
            )

        world_size = dist.get_world_size(group)
        total_numel = input_tensor.numel()
        if total_numel % world_size != 0:
            raise ValueError(
                f"Input numel {total_numel} not divisible by world_size {world_size}; "
                "FSDP should already pad before calling this hook."
            )
        chunk_numel = total_numel // world_size

        in_bf16 = stochastic_round_to_bf16(input_tensor)
        out_bf16 = torch.empty_like(in_bf16)
        dist.all_to_all_single(out_bf16, in_bf16, group=group)

        summed = out_bf16.view(world_size, chunk_numel).to(torch.float32).sum(dim=0)
        if op_type == ReduceOp.AVG:
            summed.div_(world_size)
        output_tensor.copy_(summed)
        return None
