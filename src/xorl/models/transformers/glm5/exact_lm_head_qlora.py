"""Exact active-LoRA selected logprobs for the GLM-5.2 TP16 LM head.

This component consumes *physical local* vocabulary shards.  It never owns or
materializes a second TP16 model and never folds ``B @ A`` into the frozen base
weight.  Forward values come from the pinned public SGLang kernels in serving
order:

1. BF16 hidden/base-weight operands -> FP32 local base logits;
2. literal segmented rank-one A SGEMM -> BF16 intermediate;
3. literal segmented B SGEMM -> BF16 delta rounding and fused add/store into
   the FP32 base-logit buffer;
4. rank-order TP16 vocabulary all-gather; and
5. the pinned temperature-one selected-logprob tail.

The kernels and collective are forward-only.  A custom autograd boundary
therefore recomputes the existing differentiable QLoRA formulation on the
saved BF16 effective factor bytes.  Its local projection VJP and logical TP
reductions accumulate in FP32.  This is the exact contract's validated
straight-through surrogate, not the derivative of the literal BF16 stores.
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


GLM52_EXACT_TP16_LM_HEAD_CONTRACT_VERSION = "glm52_exact_tp16_lm_head_rank1_qlora_v1"

GLM52_LM_HEAD_VOCAB_SIZE = 154_880
GLM52_LM_HEAD_HIDDEN_SIZE = 6_144
GLM52_LM_HEAD_TP_SIZE = 16
GLM52_LM_HEAD_VOCAB_PADDING = 64
GLM52_LM_HEAD_PADDED_VOCAB_SIZE = 154_880
GLM52_LM_HEAD_LOCAL_VOCAB_SIZE = 9_680
GLM52_LM_HEAD_GROUP_RANKS = tuple(range(GLM52_LM_HEAD_TP_SIZE))
GLM52_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS = 1


@dataclass(frozen=True)
class Glm52LmHeadShard:
    """One SGLang-compatible physical shard of the official vocabulary."""

    tp_rank: int
    vocab_start: int
    vocab_end: int
    padded_vocab_start: int
    padded_vocab_end: int

    @property
    def local_vocab_size(self) -> int:
        return self.vocab_end - self.vocab_start

    @property
    def local_padded_vocab_size(self) -> int:
        return self.padded_vocab_end - self.padded_vocab_start

    @property
    def padding_rows(self) -> int:
        return self.local_padded_vocab_size - self.local_vocab_size


def glm52_lm_head_shard(tp_rank: int) -> Glm52LmHeadShard:
    """Return the literal SGLang TP16 range for ``zai-org/GLM-5.2-FP8``."""

    if not isinstance(tp_rank, int) or isinstance(tp_rank, bool):
        raise TypeError(f"GLM-5.2 LM-head TP rank must be an integer, got {type(tp_rank).__name__}")
    if not 0 <= tp_rank < GLM52_LM_HEAD_TP_SIZE:
        raise ValueError(f"GLM-5.2 LM-head TP rank must be in [0, 15], got {tp_rank}")

    padded_vocab = (
        (GLM52_LM_HEAD_VOCAB_SIZE + GLM52_LM_HEAD_VOCAB_PADDING - 1)
        // GLM52_LM_HEAD_VOCAB_PADDING
        * GLM52_LM_HEAD_VOCAB_PADDING
    )
    if padded_vocab != GLM52_LM_HEAD_PADDED_VOCAB_SIZE:
        raise RuntimeError(
            "GLM-5.2 LM-head padded-vocabulary authority drifted: "
            f"computed {padded_vocab}, pinned {GLM52_LM_HEAD_PADDED_VOCAB_SIZE}"
        )
    if padded_vocab % GLM52_LM_HEAD_TP_SIZE:
        raise RuntimeError("GLM-5.2 padded vocabulary is not divisible by TP16")

    local_padded = padded_vocab // GLM52_LM_HEAD_TP_SIZE
    padded_start = tp_rank * local_padded
    padded_end = padded_start + local_padded
    vocab_start = min(padded_start, GLM52_LM_HEAD_VOCAB_SIZE)
    vocab_end = min(padded_end, GLM52_LM_HEAD_VOCAB_SIZE)
    shard = Glm52LmHeadShard(
        tp_rank=tp_rank,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        padded_vocab_start=padded_start,
        padded_vocab_end=padded_end,
    )
    if (
        shard.local_vocab_size != GLM52_LM_HEAD_LOCAL_VOCAB_SIZE
        or shard.local_padded_vocab_size != GLM52_LM_HEAD_LOCAL_VOCAB_SIZE
        or shard.padding_rows != 0
    ):
        raise RuntimeError(
            "The official GLM-5.2 TP16 LM-head contract requires 9,680 real rows "
            f"and zero padding rows on every rank, got {shard}"
        )
    return shard


@lru_cache(maxsize=64)
def _single_adapter_lm_head_batch_info(device_index: int, rows: int) -> Any:
    """One active rank-one adapter, in the metadata format used by S4."""

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


def _rank_order_vocab_from_stacked(
    stacked_logits: Tensor,
    *,
    expected_world_size: int,
    expected_local_vocab_size: int,
) -> Tensor:
    """Apply SGLang's ``[rank,row,vocab] -> [row,rank*vocab]`` order."""

    expected_shape_tail = (expected_local_vocab_size,)
    if stacked_logits.ndim != 3 or stacked_logits.shape[0] != expected_world_size:
        raise ValueError(
            "Rank-stacked LM-head logits must be [world, rows, local_vocab], got "
            f"{tuple(stacked_logits.shape)} for world={expected_world_size}"
        )
    if tuple(stacked_logits.shape[-1:]) != expected_shape_tail:
        raise ValueError(
            f"Rank-stacked local vocabulary width must be {expected_local_vocab_size}, got {stacked_logits.shape[-1]}"
        )
    if stacked_logits.dtype is not torch.float32:
        raise TypeError(f"Rank-stacked LM-head logits must be FP32, got {stacked_logits.dtype}")
    if not stacked_logits.is_contiguous():
        raise ValueError("Rank-stacked LM-head logits must be contiguous in collective rank order")
    rows = stacked_logits.shape[1]
    return stacked_logits.permute(1, 0, 2).reshape(rows, expected_world_size * expected_local_vocab_size)


def _rank_order_vocab_all_gather(
    local_logits: Tensor,
    tp_group: dist.ProcessGroup,
    *,
    expected_world_size: int,
    expected_local_vocab_size: int,
) -> Tensor:
    """Match S4's concat-style ``all_gather(..., dim=-1)`` byte order."""

    if not dist.is_initialized():
        raise RuntimeError("Rank-order LM-head all-gather requires initialized torch.distributed")
    if dist.get_world_size(tp_group) != expected_world_size:
        raise RuntimeError(
            f"LM-head all-gather expected world size {expected_world_size}, got {dist.get_world_size(tp_group)}"
        )
    if local_logits.ndim != 2 or tuple(local_logits.shape[1:]) != (expected_local_vocab_size,):
        raise ValueError(
            f"Local LM-head logits must be [rows, {expected_local_vocab_size}], got {tuple(local_logits.shape)}"
        )
    if local_logits.dtype is not torch.float32 or not local_logits.is_contiguous():
        raise ValueError("Local LM-head logits must be contiguous FP32 before the rank-order gather")

    rows = local_logits.shape[0]
    gathered = torch.empty(
        (expected_world_size * rows, expected_local_vocab_size),
        dtype=torch.float32,
        device=local_logits.device,
    )
    dist.all_gather_into_tensor(gathered, local_logits, group=tp_group)
    return _rank_order_vocab_from_stacked(
        gathered.view(expected_world_size, rows, expected_local_vocab_size),
        expected_world_size=expected_world_size,
        expected_local_vocab_size=expected_local_vocab_size,
    )


def _all_reduce_sum_fp32(value: Tensor, group: dist.ProcessGroup) -> Tensor:
    """In-place logical-owner sum used by the hidden/A surrogate gradients."""

    if value.dtype is not torch.float32 or not value.is_contiguous():
        raise ValueError("GLM-5.2 LM-head surrogate reductions require contiguous FP32 tensors")
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def _rank_order_row_all_gather(value: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Gather equal-shaped local row blocks in process-group rank order."""

    if not dist.is_initialized():
        raise RuntimeError("Rank-order LM-head row gathering requires initialized torch.distributed")
    if value.ndim == 0:
        raise ValueError("Rank-order LM-head row gathering requires at least one dimension")
    if not value.is_contiguous():
        raise ValueError("Rank-order LM-head row gathering requires a contiguous tensor")
    world_size = dist.get_world_size(group)
    gathered = torch.empty(
        (world_size * value.shape[0], *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(gathered, value, group=group)
    return gathered


def _require_equal_nonzero_row_count(value: Tensor, group: dist.ProcessGroup) -> None:
    """Fail before payload collectives when TP16 source-row shapes diverge."""

    local_rows = torch.tensor([value.shape[0]], dtype=torch.int64, device=value.device)
    world_size = dist.get_world_size(group)
    gathered_rows = torch.empty(world_size, dtype=torch.int64, device=value.device)
    dist.all_gather_into_tensor(gathered_rows, local_rows, group=group)
    if bool((gathered_rows <= 0).any().item()):
        raise ValueError("The exact GLM-5.2 lm head requires at least one source row on every TP16 rank")
    if bool((gathered_rows != gathered_rows[0]).any().item()):
        raise ValueError(
            "The exact GLM-5.2 lm head requires equal source-row counts across TP16, got "
            f"{gathered_rows.cpu().tolist()}"
        )


def _local_qlora_surrogate_logits(
    hidden: Tensor,
    local_weight: Tensor,
    effective_A: Tensor,
    effective_B: Tensor,
) -> Tensor:
    """Existing QLoRA value formulation used only to define the VJP oracle."""

    base = F.linear(hidden, local_weight)
    delta = F.linear(F.linear(hidden.float(), effective_A.float()), effective_B.float())
    return (base + delta.to(base.dtype)).float()


def _local_qlora_surrogate_vjp(
    hidden: Tensor,
    local_weight: Tensor,
    effective_A: Tensor,
    effective_B: Tensor,
    grad_local_logits: Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """FP32 local ``dHidden/dA/dB`` for the declared hybrid QLoRA oracle."""

    need_hidden, need_A, need_B = needs_input_grad
    if not any(needs_input_grad):
        return None, None, None
    if hidden.ndim != 2 or local_weight.ndim != 2 or grad_local_logits.ndim != 2:
        raise ValueError("The local LM-head surrogate expects two-dimensional hidden, weight, and dlogits")
    if hidden.dtype is not torch.bfloat16 or local_weight.dtype is not torch.bfloat16:
        raise TypeError("The local LM-head surrogate base branch requires BF16 hidden and weight")
    if effective_A.dtype is not torch.bfloat16 or effective_B.dtype is not torch.bfloat16:
        raise TypeError("The local LM-head surrogate requires saved BF16 effective factors")
    if grad_local_logits.dtype is not torch.float32:
        raise TypeError("The local LM-head surrogate requires FP32 dlogits")

    with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
        base_grad_hidden = None
        if need_hidden:
            base_hidden = hidden.detach().requires_grad_(True)
            base_logits = F.linear(base_hidden, local_weight)
            (base_grad_hidden,) = torch.autograd.grad(
                base_logits,
                base_hidden,
                grad_outputs=grad_local_logits.to(base_logits.dtype),
            )

        lora_hidden = hidden.float().detach().requires_grad_(need_hidden)
        reference_A = effective_A.float().detach().requires_grad_(need_A)
        reference_B = effective_B.float().detach().requires_grad_(need_B)
        lora_logits = F.linear(F.linear(lora_hidden, reference_A), reference_B)

        requested: list[Tensor] = []
        labels: list[str] = []
        for label, required, value in (
            ("hidden", need_hidden, lora_hidden),
            ("A", need_A, reference_A),
            ("B", need_B, reference_B),
        ):
            if required:
                labels.append(label)
                requested.append(value)
        gradients = torch.autograd.grad(
            lora_logits,
            requested,
            grad_outputs=grad_local_logits,
            allow_unused=False,
        )

    by_label = dict(zip(labels, gradients, strict=True))
    grad_hidden = None
    if need_hidden:
        grad_hidden = base_grad_hidden.float() + by_label["hidden"].float()
    return grad_hidden, by_label.get("A"), by_label.get("B")


def _selected_logprob_reference_grad(
    full_logits: Tensor,
    token_ids: Tensor,
    grad_logprob: Tensor,
) -> Tensor:
    """Differentiate the standalone FP32 QLoRA log-softmax reference."""

    with torch.enable_grad(), torch.autocast(device_type=full_logits.device.type, enabled=False):
        logits = full_logits.detach().requires_grad_(True)
        selected = F.log_softmax(logits, dim=-1).gather(1, token_ids.unsqueeze(1)).squeeze(1)
        (grad_logits,) = torch.autograd.grad(selected, logits, grad_outputs=grad_logprob.float())
    return grad_logits.contiguous()


class _Glm52ExactTP16LmHeadFunction(torch.autograd.Function):
    """Own the literal forward and delegate only the VJP to the QLoRA oracle."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        local_weight: Tensor,
        lora_A: Tensor,
        local_lora_B: Tensor,
        token_ids: Tensor,
        component: Glm52ExactTP16LmHeadSelectedLogprob,
    ) -> Tensor:
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = local_lora_B.to(torch.bfloat16).contiguous()
        logprob = component._exact_forward_value(
            hidden_states,
            local_weight,
            effective_A,
            effective_B,
            token_ids,
        )
        if logprob.dtype is not torch.float32 or tuple(logprob.shape) != tuple(token_ids.shape):
            raise RuntimeError(
                "GLM-5.2 exact TP16 LM head returned an invalid selected-logprob tensor: "
                f"dtype={logprob.dtype}, shape={tuple(logprob.shape)}, expected FP32 {tuple(token_ids.shape)}"
            )
        ctx.set_materialize_grads(False)
        ctx.component = component
        # local_weight and the masters are also version-counter sentinels.
        ctx.save_for_backward(
            hidden_states.detach(),
            local_weight,
            effective_A,
            effective_B,
            token_ids,
            lora_A,
            local_lora_B,
        )
        return logprob

    @staticmethod
    def backward(ctx, grad_logprob: Tensor | None):
        (
            hidden_states,
            local_weight,
            effective_A,
            effective_B,
            token_ids,
            _lora_A_master,
            _local_lora_B_master,
        ) = ctx.saved_tensors
        if grad_logprob is None:
            return None, None, None, None, None, None
        grad_hidden, grad_A, grad_B = ctx.component._surrogate_vjp(
            hidden_states,
            local_weight,
            effective_A,
            effective_B,
            token_ids,
            grad_logprob,
            needs_input_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]),
        )
        return grad_hidden, None, grad_A, grad_B, None, None


class _Glm52ExactDistributedTP16LmHeadFunction(torch.autograd.Function):
    """Bridge rank-local training rows to the replicated TP16 head program.

    The lm-head-only TP group is carved from CP or DP ranks, so its members may
    own different token rows.  Forward gathers those row blocks before running
    the literal TP16 selected-logprob program.  The function returns only the
    caller's rank-local rows.  Backward gathers the downstream rank-local RL
    gradients, applies the component's declared hybrid VJP to the identical
    global row block on every vocabulary rank, and returns only the caller's
    hidden-state slice.
    """

    @staticmethod
    def forward(
        ctx,
        local_hidden_states: Tensor,
        local_weight: Tensor,
        lora_A: Tensor,
        local_lora_B: Tensor,
        local_token_ids: Tensor,
        component: Glm52ExactTP16LmHeadSelectedLogprob,
    ) -> Tensor:
        group = component._validate_tp_group()
        _require_equal_nonzero_row_count(local_hidden_states, group)
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = local_lora_B.to(torch.bfloat16).contiguous()
        gathered_hidden = _rank_order_row_all_gather(local_hidden_states, group)
        gathered_token_ids = _rank_order_row_all_gather(local_token_ids, group)
        gathered_logprob = component._exact_forward_value(
            gathered_hidden,
            local_weight,
            effective_A,
            effective_B,
            gathered_token_ids,
        )
        local_rows = local_hidden_states.shape[0]
        source_rank = dist.get_rank(group)
        local_logprob = gathered_logprob.narrow(0, source_rank * local_rows, local_rows).contiguous()
        if local_logprob.dtype is not torch.float32 or tuple(local_logprob.shape) != tuple(local_token_ids.shape):
            raise RuntimeError(
                "GLM-5.2 distributed exact LM head returned an invalid local logprob tensor: "
                f"dtype={local_logprob.dtype}, shape={tuple(local_logprob.shape)}"
            )

        ctx.set_materialize_grads(False)
        ctx.component = component
        ctx.local_rows = local_rows
        ctx.source_rank = source_rank
        ctx.save_for_backward(
            gathered_hidden,
            local_weight,
            effective_A,
            effective_B,
            gathered_token_ids,
            lora_A,
            local_lora_B,
        )
        return local_logprob

    @staticmethod
    def backward(ctx, grad_local_logprob: Tensor | None):
        (
            gathered_hidden,
            local_weight,
            effective_A,
            effective_B,
            gathered_token_ids,
            _lora_A_master,
            _local_lora_B_master,
        ) = ctx.saved_tensors
        if grad_local_logprob is None:
            return None, None, None, None, None, None
        group = ctx.component._validate_tp_group()
        gathered_grad_logprob = _rank_order_row_all_gather(grad_local_logprob.float().contiguous(), group)
        grad_hidden, grad_A, grad_B = ctx.component._surrogate_vjp(
            gathered_hidden,
            local_weight,
            effective_A,
            effective_B,
            gathered_token_ids,
            gathered_grad_logprob,
            needs_input_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]),
        )
        if grad_hidden is not None:
            grad_hidden = grad_hidden.narrow(
                0,
                ctx.source_rank * ctx.local_rows,
                ctx.local_rows,
            ).contiguous()
        return grad_hidden, None, grad_A, grad_B, None, None


class Glm52ExactTP16LmHeadLoraLinear(LoraLinear):
    """Logical full head whose only admitted value path is selected logprob."""

    _glm52_exact_tp16_lm_head = True

    @staticmethod
    def _reject_ordinary_value_path() -> None:
        raise RuntimeError(
            "The exact GLM-5.2 active-LoRA lm head cannot materialize or execute a merged/full-weight value path"
        )

    def forward(self, x: Tensor) -> Tensor:
        del x
        self._reject_ordinary_value_path()

    def get_delta_weight(self) -> Tensor:
        self._reject_ordinary_value_path()

    def merged_weight_for_forward(self) -> Tensor:
        self._reject_ordinary_value_path()

    def merge_weights(self) -> None:
        self._reject_ordinary_value_path()


class Glm52ExactTP16LmHeadSelectedLogprob(nn.Module):
    """Stateless exact-value operation over XoRL-owned local LM-head shards.

    ``hidden_states`` and ``token_ids`` must already be replicated over the
    lm-head TP group (the existing lm-head-only loss topology performs that
    sequence gather). ``local_weight`` and ``local_lora_B`` are the actual
    ``Shard(0)`` local tensors; ``lora_A`` is the replicated logical factor.
    """

    contract_version = GLM52_EXACT_TP16_LM_HEAD_CONTRACT_VERSION
    _glm52_exact_active_lora_component = True
    max_lora_rank = 1
    lora_alpha = 1
    scaling = 1.0

    def __init__(
        self,
        *,
        tp_rank: int,
        vocab_start: int,
        vocab_end: int,
        padded_vocab_start: int,
        padded_vocab_end: int,
        tp_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        expected = glm52_lm_head_shard(tp_rank)
        for name, value in (
            ("vocab_start", vocab_start),
            ("vocab_end", vocab_end),
            ("padded_vocab_start", padded_vocab_start),
            ("padded_vocab_end", padded_vocab_end),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"GLM-5.2 LM-head {name} must be an integer, got {type(value).__name__}")
        actual = Glm52LmHeadShard(
            tp_rank=tp_rank,
            vocab_start=vocab_start,
            vocab_end=vocab_end,
            padded_vocab_start=padded_vocab_start,
            padded_vocab_end=padded_vocab_end,
        )
        if actual != expected:
            raise ValueError(f"GLM-5.2 LM-head shard range/order mismatch: got {actual}, expected {expected}")
        self.shard = expected
        self.tp_group = tp_group

    def bind_tp_group(self, tp_group: dist.ProcessGroup) -> None:
        """Bind the already-created XoRL lm-head-only TP process group."""

        if tp_group is None:
            raise ValueError("GLM-5.2 exact LM head requires an explicit TP process group")
        self.tp_group = tp_group

    def _validate_tp_group(self) -> dist.ProcessGroup:
        group = self.tp_group
        if group is None:
            raise RuntimeError("GLM-5.2 exact LM head has no bound TP process group")
        if not dist.is_initialized():
            raise RuntimeError("GLM-5.2 exact LM head requires initialized torch.distributed")
        world_size = dist.get_world_size(group)
        group_rank = dist.get_rank(group)
        group_ranks = tuple(dist.get_process_group_ranks(group))
        if world_size != GLM52_LM_HEAD_TP_SIZE:
            raise RuntimeError(f"GLM-5.2 exact LM head requires TP16, got TP{world_size}")
        if group_ranks != GLM52_LM_HEAD_GROUP_RANKS:
            raise RuntimeError(f"GLM-5.2 exact LM-head gather order must be WORLD16 ranks 0..15, got {group_ranks}")
        if group_rank != self.shard.tp_rank or dist.get_rank() != group_ranks[self.shard.tp_rank]:
            raise RuntimeError(
                "GLM-5.2 exact LM-head shard/group rank mismatch: "
                f"shard_rank={self.shard.tp_rank}, group_rank={group_rank}, global_rank={dist.get_rank()}"
            )
        backend = str(dist.get_backend(group)).lower()
        if backend != "nccl":
            raise RuntimeError(f"GLM-5.2 exact LM-head production group must use NCCL, got {backend}")
        return group

    @staticmethod
    def _require_plain_local_tensor(name: str, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor, got {type(value).__name__}")
        if hasattr(value, "to_local"):
            raise TypeError(f"{name} must be the actual local tensor, not a DTensor")

    def _validate_operands(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        lora_A: Tensor,
        local_lora_B: Tensor,
        token_ids: Tensor,
        *,
        require_cuda: bool,
    ) -> None:
        for name, value in (
            ("hidden_states", hidden_states),
            ("local_weight", local_weight),
            ("lora_A", lora_A),
            ("local_lora_B", local_lora_B),
            ("token_ids", token_ids),
        ):
            self._require_plain_local_tensor(name, value)

        if hidden_states.ndim < 2 or hidden_states.shape[-1] != GLM52_LM_HEAD_HIDDEN_SIZE:
            raise ValueError(
                f"hidden_states must end in official width {GLM52_LM_HEAD_HIDDEN_SIZE}, "
                f"got {tuple(hidden_states.shape)}"
            )
        if tuple(token_ids.shape) != tuple(hidden_states.shape[:-1]):
            raise ValueError(
                f"token_ids shape {tuple(token_ids.shape)} must equal hidden row shape {tuple(hidden_states.shape[:-1])}"
            )
        if hidden_states.numel() == 0:
            raise ValueError("GLM-5.2 exact LM head does not admit an empty row set")

        expected_weight_shape = (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, GLM52_LM_HEAD_HIDDEN_SIZE)
        expected_A_shape = (1, GLM52_LM_HEAD_HIDDEN_SIZE)
        expected_B_shape = (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, 1)
        for name, value, shape in (
            ("local_weight", local_weight, expected_weight_shape),
            ("lora_A", lora_A, expected_A_shape),
            ("local_lora_B", local_lora_B, expected_B_shape),
        ):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match official local shape {shape}")

        if hidden_states.dtype is not torch.bfloat16 or local_weight.dtype is not torch.bfloat16:
            raise TypeError(
                "GLM-5.2 exact LM-head base operands must be BF16, got "
                f"hidden={hidden_states.dtype}, weight={local_weight.dtype}"
            )
        if lora_A.dtype is not torch.float32 or local_lora_B.dtype is not torch.float32:
            raise TypeError(
                f"GLM-5.2 exact LM-head factor masters must be FP32, got A={lora_A.dtype}, B={local_lora_B.dtype}"
            )
        if token_ids.dtype is not torch.int64:
            raise TypeError(f"GLM-5.2 exact LM-head token IDs must be int64, got {token_ids.dtype}")
        if token_ids.device.type != "meta":
            torch._assert_async(
                ((token_ids >= 0) & (token_ids < GLM52_LM_HEAD_VOCAB_SIZE)).all(),
                f"GLM-5.2 exact LM-head token IDs must be in [0, {GLM52_LM_HEAD_VOCAB_SIZE})",
            )
        if local_weight.requires_grad:
            raise RuntimeError("GLM-5.2 exact LM-head base weight must remain frozen")
        if not lora_A.requires_grad or not local_lora_B.requires_grad:
            raise RuntimeError("GLM-5.2 exact LM-head A and B factor masters must both be trainable")

        expected_strides = {
            "local_weight": (GLM52_LM_HEAD_HIDDEN_SIZE, 1),
            "lora_A": (GLM52_LM_HEAD_HIDDEN_SIZE, 1),
            "local_lora_B": (1, 1),
        }
        for name, value in (
            ("local_weight", local_weight),
            ("lora_A", lora_A),
            ("local_lora_B", local_lora_B),
        ):
            if not value.is_contiguous() or tuple(value.stride()) != expected_strides[name]:
                raise ValueError(
                    f"{name} must use sampler-contiguous stride {expected_strides[name]}, got {tuple(value.stride())}"
                )
        if not hidden_states.is_contiguous() or hidden_states.stride(-1) != 1:
            raise ValueError("hidden_states must be contiguous with unit hidden stride")
        if not token_ids.is_contiguous():
            raise ValueError("token_ids must be contiguous in the same logical row order as hidden_states")

        devices = {value.device for value in (hidden_states, local_weight, lora_A, local_lora_B, token_ids)}
        if len(devices) != 1:
            raise RuntimeError(f"GLM-5.2 exact LM-head operands must share one device, got {sorted(map(str, devices))}")
        if require_cuda and hidden_states.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact LM-head value forward requires CUDA and pinned S4 kernels")

    def effective_factor_views(self, lora_A: Tensor, local_lora_B: Tensor) -> tuple[Tensor, Tensor]:
        """Return the exact live BF16 bytes consumed by the S4 A/B kernels."""

        if lora_A.dtype is not torch.float32 or tuple(lora_A.shape) != (1, GLM52_LM_HEAD_HIDDEN_SIZE):
            raise TypeError("lora_A must be the official FP32 [1, 6144] master")
        if local_lora_B.dtype is not torch.float32 or tuple(local_lora_B.shape) != (
            GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            1,
        ):
            raise TypeError("local_lora_B must be the official FP32 [9680, 1] master")
        return lora_A.to(torch.bfloat16).contiguous(), local_lora_B.to(torch.bfloat16).contiguous()

    def _exact_local_logits(
        self,
        hidden_2d: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
    ) -> Tensor:
        """Run the current exact v2 base, A-store, B-round, and fused add/store."""

        if hidden_2d.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact local LM-head logits require CUDA")
        if hidden_2d.dtype is not torch.bfloat16 or local_weight.dtype is not torch.bfloat16:
            raise TypeError("Exact local LM-head base operands must be BF16")
        if effective_A.dtype is not torch.bfloat16 or effective_B.dtype is not torch.bfloat16:
            raise TypeError("Exact local LM-head effective factors must be BF16")
        if tuple(effective_A.shape) != (1, GLM52_LM_HEAD_HIDDEN_SIZE) or tuple(effective_B.shape) != (
            GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            1,
        ):
            raise ValueError("Exact local LM-head effective factor shapes do not match the official shard")
        if any(not value.is_contiguous() for value in (hidden_2d, local_weight, effective_A, effective_B)):
            raise ValueError("Exact local LM-head operands must be contiguous")

        try:
            from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415
            from sglang.srt.batch_invariant_ops import head_v2_full_logits_with_lse  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned exact v2 batch-invariant and LoRA kernels are required") from exc

        rows = hidden_2d.shape[0]
        base_logits, _base_lse = head_v2_full_logits_with_lse(hidden_2d, local_weight)
        if base_logits.dtype is not torch.float32 or tuple(base_logits.shape) != (
            rows,
            GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
        ):
            raise RuntimeError("Pinned exact v2 base LM-head kernel returned an invalid local-logit buffer")
        batch_info = _single_adapter_lm_head_batch_info(hidden_2d.device.index, rows)
        lora_a_output = sgemm_lora_a_fwd(hidden_2d, effective_A.unsqueeze(0), batch_info)
        if lora_a_output.dtype is not torch.bfloat16 or tuple(lora_a_output.shape) != (rows, 1):
            raise RuntimeError("Pinned exact A SGEMM did not produce the required BF16 rank-one store")
        output = sgemm_lora_b_fwd(
            lora_a_output,
            effective_B.unsqueeze(0),
            batch_info,
            base_output=base_logits,
        )
        if output.data_ptr() != base_logits.data_ptr():
            raise RuntimeError("Pinned exact B SGEMM did not perform the required in-place base+delta store")
        if output.dtype is not torch.float32 or not output.is_contiguous():
            raise RuntimeError("Pinned exact B SGEMM did not preserve the contiguous FP32 local-logit buffer")
        return output

    @staticmethod
    def _selected_logprob_from_gathered(full_logits: Tensor, token_ids: Tensor) -> Tensor:
        if full_logits.dtype is not torch.float32 or tuple(full_logits.shape[1:]) != (GLM52_LM_HEAD_VOCAB_SIZE,):
            raise ValueError(
                f"Gathered GLM-5.2 logits must be FP32 [rows, {GLM52_LM_HEAD_VOCAB_SIZE}], "
                f"got {full_logits.dtype} {tuple(full_logits.shape)}"
            )
        try:
            from sglang.srt.batch_invariant_ops import (  # noqa: PLC0415
                head_v2_selected_logprob_from_logits,
            )
        except Exception as exc:
            raise RuntimeError("Pinned exact v2 selected-logprob tail is required") from exc
        logprob, _lse, _selected = head_v2_selected_logprob_from_logits(
            full_logits,
            token_ids,
            temperature=None,
        )
        return logprob

    def _exact_forward_value(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        group = self._validate_tp_group()
        rows = hidden_states.numel() // GLM52_LM_HEAD_HIDDEN_SIZE
        hidden_2d = hidden_states.view(rows, GLM52_LM_HEAD_HIDDEN_SIZE)
        token_ids_1d = token_ids.view(rows)
        local_logits = self._exact_local_logits(hidden_2d, local_weight, effective_A, effective_B)
        full_logits = _rank_order_vocab_all_gather(
            local_logits,
            group,
            expected_world_size=GLM52_LM_HEAD_TP_SIZE,
            expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
        )
        logprob = self._selected_logprob_from_gathered(full_logits, token_ids_1d)
        return logprob.view_as(token_ids)

    def _surrogate_vjp(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        token_ids: Tensor,
        grad_logprob: Tensor,
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        group = self._validate_tp_group()
        rows = hidden_states.numel() // GLM52_LM_HEAD_HIDDEN_SIZE
        hidden_2d = hidden_states.view(rows, GLM52_LM_HEAD_HIDDEN_SIZE)
        token_ids_1d = token_ids.view(rows)
        with torch.no_grad(), torch.autocast(device_type=hidden_states.device.type, enabled=False):
            local_reference_logits = _local_qlora_surrogate_logits(
                hidden_2d,
                local_weight,
                effective_A,
                effective_B,
            ).contiguous()
            full_reference_logits = _rank_order_vocab_all_gather(
                local_reference_logits,
                group,
                expected_world_size=GLM52_LM_HEAD_TP_SIZE,
                expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            )
        full_grad_logits = _selected_logprob_reference_grad(
            full_reference_logits,
            token_ids_1d,
            grad_logprob.reshape(rows),
        )
        local_grad_logits = full_grad_logits[:, self.shard.vocab_start : self.shard.vocab_end].contiguous()
        grad_hidden, grad_A, grad_B = _local_qlora_surrogate_vjp(
            hidden_2d,
            local_weight,
            effective_A,
            effective_B,
            local_grad_logits,
            needs_input_grad=needs_input_grad,
        )
        if grad_hidden is not None:
            grad_hidden = _all_reduce_sum_fp32(grad_hidden.contiguous(), group).view_as(hidden_states)
        if grad_A is not None:
            grad_A = _all_reduce_sum_fp32(grad_A.contiguous(), group)
        return grad_hidden, grad_A, grad_B

    def forward(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        lora_A: Tensor,
        local_lora_B: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        self._validate_operands(
            hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            token_ids,
            require_cuda=True,
        )
        self._validate_tp_group()
        return _Glm52ExactTP16LmHeadFunction.apply(
            hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            token_ids,
            self,
        )

    def distributed_selected_logprob(
        self,
        local_hidden_states: Tensor,
        local_weight: Tensor,
        lora_A: Tensor,
        local_lora_B: Tensor,
        local_token_ids: Tensor,
    ) -> Tensor:
        """Return local-row logprobs while differentiating the global TP16 row set."""

        self._validate_operands(
            local_hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            local_token_ids,
            require_cuda=True,
        )
        self._validate_tp_group()
        return _Glm52ExactDistributedTP16LmHeadFunction.apply(
            local_hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            local_token_ids,
            self,
        )

    def extra_repr(self) -> str:
        return (
            f"tp_rank={self.shard.tp_rank}, vocab=[{self.shard.vocab_start},{self.shard.vocab_end}), "
            "rank=1, alpha=1, temperature=1"
        )


def is_glm52_exact_tp16_lm_head(module: nn.Module | None) -> bool:
    """Return whether ``module`` owns the internal exact active-LoRA head op."""

    return bool(
        module is not None
        and getattr(module, "_glm52_exact_tp16_lm_head", False)
        and isinstance(
            getattr(module, "_glm52_exact_selected_logprob", None),
            Glm52ExactTP16LmHeadSelectedLogprob,
        )
    )


def glm52_exact_lm_head_per_token_ce(
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
    """Run the exact TP16 head as the grad-enabled source of local-token CE."""

    if not is_glm52_exact_tp16_lm_head(lm_head):
        raise TypeError("glm52_exact_lm_head_per_token_ce requires the constructed exact GLM-5.2 lm_head")
    if ce_mode != "bi_fused":
        raise NotImplementedError("The GLM-5.2 exact active-LoRA lm_head requires ce_mode='bi_fused'")
    if not lm_head_fp32:
        raise NotImplementedError("The GLM-5.2 exact active-LoRA lm_head requires lm_head_fp32=true")
    if float(logprob_temperature) != 1.0:
        raise NotImplementedError("The GLM-5.2 exact active-LoRA lm_head requires logprob_temperature=1.0")

    component = lm_head._glm52_exact_selected_logprob
    group = component._validate_tp_group()
    if tp_group is not None and tp_group is not group:
        raise RuntimeError("The loss TP group does not match the exact GLM-5.2 lm-head TP16 group")
    if hidden_states_flat.ndim != 2 or hidden_states_flat.shape[-1] != GLM52_LM_HEAD_HIDDEN_SIZE:
        raise ValueError(
            f"Exact GLM-5.2 lm-head hidden states must be [rows, {GLM52_LM_HEAD_HIDDEN_SIZE}], "
            f"got {tuple(hidden_states_flat.shape)}"
        )
    if labels_flat.ndim != 1 or labels_flat.shape[0] != hidden_states_flat.shape[0]:
        raise ValueError("Exact GLM-5.2 lm-head labels must be one-dimensional and row-aligned")
    if hidden_states_flat.dtype is not torch.bfloat16:
        raise TypeError(f"Exact GLM-5.2 lm-head hidden states must remain BF16, got {hidden_states_flat.dtype}")
    if hidden_states_flat.shape[0] == 0:
        raise ValueError("The exact GLM-5.2 lm head requires at least one local source row")

    def _plain_local(value: Tensor) -> Tensor:
        return value.to_local() if hasattr(value, "to_local") else value

    local_weight = _plain_local(weight)
    lora_A = _plain_local(lm_head.lora_A)
    local_lora_B = _plain_local(lm_head.lora_B)
    valid = labels_flat != int(ignore_index)
    safe_labels = torch.where(valid, labels_flat, torch.zeros_like(labels_flat))
    ce_chunks: list[Tensor] = []
    for start in range(0, hidden_states_flat.shape[0], GLM52_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS):
        end = min(start + GLM52_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS, hidden_states_flat.shape[0])
        logprob = component.distributed_selected_logprob(
            hidden_states_flat[start:end].contiguous(),
            local_weight,
            lora_A,
            local_lora_B,
            safe_labels[start:end].contiguous(),
        )
        ce_chunks.append(torch.where(valid[start:end], -logprob, torch.zeros_like(logprob)))
    return torch.cat(ce_chunks, dim=0)


__all__ = [
    "GLM52_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS",
    "GLM52_EXACT_TP16_LM_HEAD_CONTRACT_VERSION",
    "GLM52_LM_HEAD_HIDDEN_SIZE",
    "GLM52_LM_HEAD_LOCAL_VOCAB_SIZE",
    "GLM52_LM_HEAD_PADDED_VOCAB_SIZE",
    "GLM52_LM_HEAD_TP_SIZE",
    "GLM52_LM_HEAD_VOCAB_PADDING",
    "GLM52_LM_HEAD_VOCAB_SIZE",
    "Glm52ExactTP16LmHeadLoraLinear",
    "Glm52ExactTP16LmHeadSelectedLogprob",
    "Glm52LmHeadShard",
    "glm52_exact_lm_head_per_token_ce",
    "glm52_lm_head_shard",
    "is_glm52_exact_tp16_lm_head",
]
