"""Exact DSV4-Flash TP8 active-LoRA output head.

The serving head keeps LoRA A replicated, shards the BF16 base weight and
LoRA B by vocabulary row, and executes two literal SGLang LoRA SGEMMs before
the BF16 vocabulary all-gather and BF16 log-softmax.  This module owns the
same value program while retaining FP32 factor masters and a checked
straight-through VJP for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.lora.modules.linear import LoraLinear


DSV4_EXACT_TP8_LM_HEAD_CONTRACT_VERSION = "dsv4_flash_exact_tp8_lm_head_rank1_lora_v1"
DSV4_LM_HEAD_VOCAB_SIZE = 129_280
DSV4_LM_HEAD_HIDDEN_SIZE = 4_096
DSV4_LM_HEAD_TP_SIZE = 8
DSV4_LM_HEAD_LOCAL_VOCAB_SIZE = 16_160
DSV4_LM_HEAD_GROUP_RANKS = tuple(range(DSV4_LM_HEAD_TP_SIZE))
DSV4_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS = 1


@dataclass(frozen=True)
class Dsv4LmHeadShard:
    tp_rank: int
    vocab_start: int
    vocab_end: int

    @property
    def local_vocab_size(self) -> int:
        return self.vocab_end - self.vocab_start


def dsv4_lm_head_shard(tp_rank: int) -> Dsv4LmHeadShard:
    if not isinstance(tp_rank, int) or isinstance(tp_rank, bool):
        raise TypeError("DSV4 LM-head TP rank must be an integer")
    if not 0 <= tp_rank < DSV4_LM_HEAD_TP_SIZE:
        raise ValueError(f"DSV4 LM-head TP rank must be in [0, 7], got {tp_rank}")
    if DSV4_LM_HEAD_VOCAB_SIZE % DSV4_LM_HEAD_TP_SIZE:
        raise RuntimeError("The official DSV4 vocabulary is not divisible by TP8")
    start = tp_rank * DSV4_LM_HEAD_LOCAL_VOCAB_SIZE
    return Dsv4LmHeadShard(tp_rank, start, start + DSV4_LM_HEAD_LOCAL_VOCAB_SIZE)


@lru_cache(maxsize=64)
def _single_adapter_batch_info(device_index: int, rows: int) -> Any:
    from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415

    device = torch.device("cuda", device_index)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )


def _rank_order_row_counts(local_rows: int, device: torch.device, group: dist.ProcessGroup) -> tuple[int, ...]:
    local = torch.tensor([local_rows], dtype=torch.int64, device=device)
    gathered = torch.empty(DSV4_LM_HEAD_TP_SIZE, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    counts = tuple(int(value) for value in gathered.cpu().tolist())
    if any(value < 0 for value in counts):
        raise RuntimeError(f"DSV4 exact lm head received negative row counts: {counts}")
    return counts


def _rank_order_variable_row_all_gather(
    value: Tensor,
    group: dist.ProcessGroup,
    *,
    row_counts: tuple[int, ...],
    padded_rows: int,
) -> Tensor:
    if value.ndim == 0 or not value.is_contiguous():
        raise ValueError("DSV4 row all-gather requires a contiguous non-scalar tensor")
    if value.shape[0] > padded_rows or len(row_counts) != DSV4_LM_HEAD_TP_SIZE:
        raise ValueError(f"Invalid DSV4 variable-row geometry: local={value.shape[0]}, counts={row_counts}")
    if value.shape[0] < padded_rows:
        padding = value.new_zeros((padded_rows - value.shape[0], *value.shape[1:]))
        value = torch.cat((value, padding), dim=0)
    gathered = torch.empty(
        (DSV4_LM_HEAD_TP_SIZE * padded_rows, *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(gathered, value, group=group)
    pieces = [
        gathered[rank * padded_rows : rank * padded_rows + count] for rank, count in enumerate(row_counts) if count
    ]
    return torch.cat(pieces, dim=0) if pieces else gathered[:0]


def _rank_order_vocab_all_gather(local_logits: Tensor, group: dist.ProcessGroup) -> Tensor:
    if local_logits.dtype is not torch.bfloat16 or not local_logits.is_contiguous():
        raise ValueError("DSV4 local logits must be contiguous BF16")
    rows = local_logits.shape[0]
    gathered = torch.empty(
        (DSV4_LM_HEAD_TP_SIZE * rows, DSV4_LM_HEAD_LOCAL_VOCAB_SIZE),
        dtype=local_logits.dtype,
        device=local_logits.device,
    )
    dist.all_gather_into_tensor(gathered, local_logits, group=group)
    return (
        gathered.view(DSV4_LM_HEAD_TP_SIZE, rows, DSV4_LM_HEAD_LOCAL_VOCAB_SIZE)
        .permute(1, 0, 2)
        .reshape(rows, DSV4_LM_HEAD_VOCAB_SIZE)
    )


def _selected_logprob_reference_grad(full_logits: Tensor, token_ids: Tensor, grad_logprob: Tensor) -> Tensor:
    with torch.enable_grad(), torch.autocast(device_type=full_logits.device.type, enabled=False):
        logits = full_logits.float().detach().requires_grad_(True)
        selected = F.log_softmax(logits, dim=-1).gather(1, token_ids.unsqueeze(1)).squeeze(1)
        (gradient,) = torch.autograd.grad(selected, logits, grad_outputs=grad_logprob.float())
    return gradient.contiguous()


def _local_surrogate_vjp(
    hidden: Tensor,
    local_weight: Tensor,
    effective_a: Tensor,
    effective_b: Tensor,
    grad_local_logits: Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    need_hidden, need_a, need_b = needs_input_grad
    if not any(needs_input_grad):
        return None, None, None
    with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
        hidden_ref = hidden.float().detach().requires_grad_(need_hidden)
        a_ref = effective_a.float().detach().requires_grad_(need_a)
        b_ref = effective_b.float().detach().requires_grad_(need_b)
        base = F.linear(hidden_ref.to(torch.bfloat16), local_weight).float()
        delta = F.linear(F.linear(hidden_ref, a_ref), b_ref)
        logits = base + delta
        requested: list[Tensor] = []
        labels: list[str] = []
        for label, required, value in (
            ("hidden", need_hidden, hidden_ref),
            ("a", need_a, a_ref),
            ("b", need_b, b_ref),
        ):
            if required:
                labels.append(label)
                requested.append(value)
        gradients = torch.autograd.grad(logits, requested, grad_outputs=grad_local_logits.float())
    by_label = dict(zip(labels, gradients, strict=True))
    return by_label.get("hidden"), by_label.get("a"), by_label.get("b")


class _Dsv4ExactDistributedHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_hidden: Tensor,
        local_weight: Tensor,
        lora_a: Tensor,
        local_lora_b: Tensor,
        local_token_ids: Tensor,
        component: "Dsv4ExactTP8LmHeadSelectedLogprob",
    ) -> Tensor:
        group = component._validate_tp_group()
        row_counts = _rank_order_row_counts(local_hidden.shape[0], local_hidden.device, group)
        if sum(row_counts) <= 0:
            raise ValueError("DSV4 exact lm head requires at least one live row across TP8")
        padded_rows = max(row_counts)

        effective_a = lora_a.to(torch.bfloat16).contiguous()
        effective_b = local_lora_b.to(torch.bfloat16).contiguous()
        gathered_hidden = _rank_order_variable_row_all_gather(
            local_hidden,
            group,
            row_counts=row_counts,
            padded_rows=padded_rows,
        )
        gathered_ids = _rank_order_variable_row_all_gather(
            local_token_ids,
            group,
            row_counts=row_counts,
            padded_rows=padded_rows,
        )
        gathered_logprob = component._exact_forward_value(
            gathered_hidden,
            local_weight,
            effective_a,
            effective_b,
            gathered_ids,
        )
        source_rank = dist.get_rank(group)
        rows = row_counts[source_rank]
        source_offset = sum(row_counts[:source_rank])
        local_logprob = gathered_logprob.narrow(0, source_offset, rows).contiguous()
        ctx.set_materialize_grads(False)
        ctx.component = component
        ctx.local_rows = rows
        ctx.source_rank = source_rank
        ctx.source_offset = source_offset
        ctx.row_counts = row_counts
        ctx.padded_rows = padded_rows
        ctx.save_for_backward(
            gathered_hidden,
            local_weight,
            effective_a,
            effective_b,
            gathered_ids,
            lora_a,
            local_lora_b,
        )
        return local_logprob

    @staticmethod
    def backward(ctx, grad_local_logprob: Tensor | None):
        if grad_local_logprob is None:
            return None, None, None, None, None, None
        (
            gathered_hidden,
            local_weight,
            effective_a,
            effective_b,
            gathered_ids,
            _a_master,
            _b_master,
        ) = ctx.saved_tensors
        group = ctx.component._validate_tp_group()
        gathered_grad = _rank_order_variable_row_all_gather(
            grad_local_logprob.contiguous(),
            group,
            row_counts=ctx.row_counts,
            padded_rows=ctx.padded_rows,
        )
        grad_hidden, grad_a, grad_b = ctx.component._surrogate_vjp(
            gathered_hidden,
            local_weight,
            effective_a,
            effective_b,
            gathered_ids,
            gathered_grad,
            needs_input_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]),
        )
        if grad_hidden is not None:
            grad_hidden = grad_hidden.narrow(0, ctx.source_offset, ctx.local_rows).contiguous()
        return grad_hidden, None, grad_a, grad_b, None, None


class Dsv4ExactTP8LmHeadLoraLinear(LoraLinear):
    _dsv4_exact_tp8_lm_head = True

    @staticmethod
    def _reject_ordinary_path() -> None:
        raise RuntimeError("The exact DSV4 lm head is available only through its selected-logprob value program")

    def forward(self, x: Tensor) -> Tensor:
        del x
        self._reject_ordinary_path()

    def get_delta_weight(self) -> Tensor:
        self._reject_ordinary_path()

    def merged_weight_for_forward(self) -> Tensor:
        self._reject_ordinary_path()


class Dsv4ExactTP8LmHeadSelectedLogprob(nn.Module):
    contract_version = DSV4_EXACT_TP8_LM_HEAD_CONTRACT_VERSION
    max_lora_rank = 1
    lora_alpha = 1
    scaling = 1.0

    def __init__(self, *, tp_rank: int, tp_group: dist.ProcessGroup) -> None:
        super().__init__()
        self.shard = dsv4_lm_head_shard(tp_rank)
        self.tp_group = tp_group

    def _validate_tp_group(self) -> dist.ProcessGroup:
        group = self.tp_group
        if group is None or not dist.is_initialized():
            raise RuntimeError("DSV4 exact lm head requires an initialized TP8 group")
        ranks = tuple(dist.get_process_group_ranks(group))
        if (
            dist.get_world_size(group) != DSV4_LM_HEAD_TP_SIZE
            or ranks != DSV4_LM_HEAD_GROUP_RANKS
            or dist.get_rank(group) != self.shard.tp_rank
            or dist.get_rank() != ranks[self.shard.tp_rank]
        ):
            raise RuntimeError("DSV4 exact lm-head TP8 rank/order mismatch")
        if str(dist.get_backend(group)).lower() != "nccl":
            raise RuntimeError("DSV4 exact lm head requires NCCL")
        return group

    def _validate_operands(
        self,
        hidden: Tensor,
        local_weight: Tensor,
        lora_a: Tensor,
        local_lora_b: Tensor,
        token_ids: Tensor,
    ) -> None:
        expected = {
            "local_weight": (local_weight, (DSV4_LM_HEAD_LOCAL_VOCAB_SIZE, DSV4_LM_HEAD_HIDDEN_SIZE)),
            "lora_a": (lora_a, (1, DSV4_LM_HEAD_HIDDEN_SIZE)),
            "local_lora_b": (local_lora_b, (DSV4_LM_HEAD_LOCAL_VOCAB_SIZE, 1)),
        }
        if hidden.ndim != 2 or tuple(hidden.shape[1:]) != (DSV4_LM_HEAD_HIDDEN_SIZE,):
            raise ValueError(f"DSV4 exact lm-head hidden shape is invalid: {tuple(hidden.shape)}")
        if token_ids.ndim != 1 or token_ids.shape[0] != hidden.shape[0]:
            raise ValueError("DSV4 exact lm-head token IDs must be row-aligned")
        for name, (value, shape) in expected.items():
            if hasattr(value, "to_local") or tuple(value.shape) != shape or not value.is_contiguous():
                raise ValueError(f"DSV4 exact lm-head {name} must be a contiguous local tensor with shape {shape}")
        if hidden.dtype is not torch.bfloat16 or local_weight.dtype is not torch.bfloat16:
            raise TypeError("DSV4 exact lm-head base operands must be BF16")
        if lora_a.dtype is not torch.float32 or local_lora_b.dtype is not torch.float32:
            raise TypeError("DSV4 exact lm-head factor masters must be FP32")
        if token_ids.dtype is not torch.int64:
            raise TypeError("DSV4 exact lm-head token IDs must be int64")

    @staticmethod
    def _exact_local_logits(hidden: Tensor, weight: Tensor, effective_a: Tensor, effective_b: Tensor) -> Tensor:
        from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
        from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415

        base_logits = F.linear(hidden, weight)
        if base_logits.dtype is not torch.bfloat16:
            raise RuntimeError("DSV4 sampler-aligned base head must store BF16 logits")
        batch_info = _single_adapter_batch_info(hidden.device.index, hidden.shape[0])
        lora_a_output = sgemm_lora_a_fwd(hidden, effective_a.unsqueeze(0), batch_info)
        output = sgemm_lora_b_fwd(
            lora_a_output,
            effective_b.unsqueeze(0),
            batch_info,
            base_output=base_logits,
        )
        if output.data_ptr() != base_logits.data_ptr() or output.dtype is not torch.bfloat16:
            raise RuntimeError("DSV4 sampler-aligned LoRA B must update the BF16 base-logit buffer in place")
        return output.contiguous()

    def _exact_forward_value(
        self,
        hidden: Tensor,
        local_weight: Tensor,
        effective_a: Tensor,
        effective_b: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        local_logits = self._exact_local_logits(hidden, local_weight, effective_a, effective_b)
        full_logits = _rank_order_vocab_all_gather(local_logits, self._validate_tp_group())
        # Serving computes decode logprobs through the batch-invariant Triton
        # log_softmax (deterministic-mode interposition), whose BF16 rounding
        # differs from ATen's kernel on boundary values (first seen as a
        # one-bf16-ulp flip at decision 39 of the campaign-2 64-decision base
        # ruler; the f64 truth sits 5.5e-8 past the rounding boundary). The
        # forward VALUE must come from the serving kernel; the surrogate VJP
        # keeps its FP32 reference math.
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (  # noqa: PLC0415
            log_softmax as _bi_log_softmax,
        )

        logprobs = _bi_log_softmax(full_logits, dim=-1)
        return logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1).contiguous()

    def _surrogate_vjp(
        self,
        hidden: Tensor,
        local_weight: Tensor,
        effective_a: Tensor,
        effective_b: Tensor,
        token_ids: Tensor,
        grad_logprob: Tensor,
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        group = self._validate_tp_group()
        with torch.no_grad():
            local_reference = (
                (
                    F.linear(hidden, local_weight)
                    + F.linear(F.linear(hidden.float(), effective_a.float()), effective_b.float()).to(torch.bfloat16)
                )
                .float()
                .contiguous()
            )
            full_reference = _rank_order_vocab_all_gather(local_reference.to(torch.bfloat16), group).float()
        full_grad = _selected_logprob_reference_grad(full_reference, token_ids, grad_logprob)
        local_grad = full_grad[:, self.shard.vocab_start : self.shard.vocab_end].contiguous()
        grad_hidden, grad_a, grad_b = _local_surrogate_vjp(
            hidden,
            local_weight,
            effective_a,
            effective_b,
            local_grad,
            needs_input_grad=needs_input_grad,
        )
        if grad_hidden is not None:
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=group)
        if grad_a is not None:
            dist.all_reduce(grad_a, op=dist.ReduceOp.SUM, group=group)
        return grad_hidden, grad_a, grad_b

    def distributed_selected_logprob(
        self,
        local_hidden: Tensor,
        local_weight: Tensor,
        lora_a: Tensor,
        local_lora_b: Tensor,
        local_token_ids: Tensor,
    ) -> Tensor:
        self._validate_operands(local_hidden, local_weight, lora_a, local_lora_b, local_token_ids)
        self._validate_tp_group()
        return _Dsv4ExactDistributedHeadFunction.apply(
            local_hidden,
            local_weight,
            lora_a,
            local_lora_b,
            local_token_ids,
            self,
        )


def bind_dsv4_exact_lm_head(model: nn.Module) -> None:
    """Turn the injected logical head into the TP8 exact selected-logprob owner."""

    from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, LoraLinear):
        raise TypeError("DSV4 exact active LoRA requires an injected LoraLinear lm_head")
    state = get_parallel_state()
    group = getattr(state, "lm_head_tp_group", None)
    if (
        getattr(state, "world_size", 0) != DSV4_LM_HEAD_TP_SIZE
        or getattr(state, "tp_size", 0) != 1
        or getattr(state, "pp_size", 0) != 1
        or getattr(state, "lm_head_tp_size", 0) != DSV4_LM_HEAD_TP_SIZE
        or group is None
    ):
        raise RuntimeError("DSV4 exact active-LoRA lm head requires WORLD8, body TP1/PP1, and lm-head TP8")
    tp_rank = int(dist.get_rank(group))
    lm_head.__class__ = Dsv4ExactTP8LmHeadLoraLinear
    lm_head._dsv4_exact_selected_logprob = Dsv4ExactTP8LmHeadSelectedLogprob(
        tp_rank=tp_rank,
        tp_group=group,
    )
    lm_head._dsv4_exact_replicated_parameter_names = ("lora_A",)


def dsv4_exact_lm_head_per_token_ce(
    hidden_states_flat: Tensor,
    weight: Tensor,
    labels_flat: Tensor,
    *,
    lm_head: nn.Module,
    ignore_index: int,
    ce_mode: str,
    lm_head_fp32: bool,
    logprob_temperature: float,
    tp_group: dist.ProcessGroup | None,
) -> Tensor:
    if not getattr(lm_head, "_dsv4_exact_tp8_lm_head", False):
        raise TypeError("DSV4 exact lm-head CE requires the bound TP8 head")
    if ce_mode != "compiled" or lm_head_fp32 or float(logprob_temperature) != 1.0:
        raise NotImplementedError("DSV4 exact lm-head CE requires compiled mode, BF16 head, and temperature 1")
    component = lm_head._dsv4_exact_selected_logprob
    group = component._validate_tp_group()
    if tp_group is not group:
        raise RuntimeError("The loss TP group does not match the DSV4 exact lm-head TP8 group")
    if hidden_states_flat.dtype is not torch.bfloat16 or hidden_states_flat.shape[-1] != DSV4_LM_HEAD_HIDDEN_SIZE:
        raise TypeError("DSV4 exact lm-head hidden states must be BF16 width 4096")
    if weight.requires_grad or not lm_head.lora_A.requires_grad or not lm_head.lora_B.requires_grad:
        raise RuntimeError("DSV4 exact lm-head base must be frozen and factor masters trainable")

    def local(value: Tensor) -> Tensor:
        return value.to_local() if hasattr(value, "to_local") else value

    local_weight = local(weight)
    lora_a = local(lm_head.lora_A)
    local_lora_b = local(lm_head.lora_B)
    valid_indices = (labels_flat != int(ignore_index)).nonzero(as_tuple=True)[0]
    local_valid_count = int(valid_indices.numel())
    max_valid_count = torch.tensor([local_valid_count], dtype=torch.int64, device=hidden_states_flat.device)
    dist.all_reduce(max_valid_count, op=dist.ReduceOp.MAX, group=group)
    per_token_ce = hidden_states_flat[:, 0] * 0.0
    for ordinal in range(int(max_valid_count.item())):
        if ordinal < local_valid_count:
            index = valid_indices[ordinal : ordinal + 1]
            local_hidden = hidden_states_flat.index_select(0, index).contiguous()
            local_labels = labels_flat.index_select(0, index).contiguous()
        else:
            index = valid_indices[:0]
            local_hidden = hidden_states_flat[:0].contiguous()
            local_labels = labels_flat[:0].contiguous()
        logprob = component.distributed_selected_logprob(
            local_hidden,
            local_weight,
            lora_a,
            local_lora_b,
            local_labels,
        )
        per_token_ce = per_token_ce + logprob.sum() * 0.0
        if ordinal < local_valid_count:
            per_token_ce = per_token_ce.index_copy(0, index, -logprob)
    return per_token_ce


__all__ = [
    "DSV4_EXACT_TP8_LM_HEAD_CONTRACT_VERSION",
    "DSV4_LM_HEAD_HIDDEN_SIZE",
    "DSV4_LM_HEAD_LOCAL_VOCAB_SIZE",
    "DSV4_LM_HEAD_TP_SIZE",
    "DSV4_LM_HEAD_VOCAB_SIZE",
    "Dsv4ExactTP8LmHeadLoraLinear",
    "Dsv4ExactTP8LmHeadSelectedLogprob",
    "bind_dsv4_exact_lm_head",
    "dsv4_exact_lm_head_per_token_ce",
    "dsv4_lm_head_shard",
]
