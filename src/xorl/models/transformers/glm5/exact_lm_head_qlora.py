"""Exact active-LoRA selected logprobs for the GLM-5.2 TP16 LM head.

This component consumes *physical local* vocabulary shards.  It never owns or
materializes a second TP16 model and never folds ``B @ A`` into the frozen base
weight.  Forward values come from the pinned public SGLang kernels in serving
order:

1. BF16 hidden/base-weight operands -> FP32 local base logits;
2. literal segmented low-rank A SGEMM -> BF16 intermediate;
3. literal segmented B SGEMM -> BF16 delta rounding and fused add/store into
   the FP32 base-logit buffer;
4. rank-order TP16 vocabulary all-gather; and
5. the pinned selected-logprob tail with optional per-row FP32 temperature.

The kernels and collective are forward-only.  A custom autograd boundary
therefore recomputes the existing differentiable QLoRA formulation on the
saved BF16 effective factor bytes.  Its local projection VJP and logical TP
reductions accumulate in FP32.  This is the exact contract's validated
straight-through surrogate, not the derivative of the literal BF16 stores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.lora.modules.linear import LoraLinear
from xorl.models.transformers.exact_lm_head_shared import (
    REPLICATED_ROW_PLAN,
    ExactHeadRowPlan,
    ExactLmHeadFunction,
    check_exact_head_tp_group,
    exact_lora_local_logits,
    filtered_surrogate_local_grad_logits,
    require_equal_nonzero_row_count,
    surrogate_local_grad_logits,
)
from xorl.models.transformers.exact_lm_head_shared import (
    all_reduce_sum_fp32 as _all_reduce_sum_fp32,
)
from xorl.models.transformers.exact_lm_head_shared import (
    rank_order_row_all_gather as _rank_order_row_all_gather,
)
from xorl.models.transformers.exact_lm_head_shared import (
    rank_order_vocab_all_gather as _rank_order_vocab_all_gather,
)
from xorl.models.transformers.glm5.exact_lora_contract import glm52_exact_lora_scaling
from xorl.ops.bi_families_v2 import exact_temperature_scale_fp32_logits
from xorl.ops.exact_sampling_transforms import (
    normalize_temperature_rows,
    score_with_sampling_transforms,
)
from xorl.ops.exact_sampling_transforms import (
    validate_temperature_rows as _validate_temperature_rows,
)


GLM52_EXACT_TP16_LM_HEAD_CONTRACT_VERSION = "glm52_exact_tp16_lm_head_qlora_v2"

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


def _require_equal_nonzero_row_count(value: Tensor, group: dist.ProcessGroup) -> None:
    """Fail before payload collectives when TP16 source-row shapes diverge."""

    require_equal_nonzero_row_count(value, group, program="The exact GLM-5.2 lm head")


def _distributed_row_plan(local_hidden: Tensor, group: dist.ProcessGroup) -> ExactHeadRowPlan:
    """Equal rank-order row blocks: every TP16 rank contributes the same count."""

    _require_equal_nonzero_row_count(local_hidden, group)
    local_rows = local_hidden.shape[0]
    source_rank = dist.get_rank(group)
    return ExactHeadRowPlan(
        lambda value: _rank_order_row_all_gather(value, group),
        lambda value: value.narrow(0, source_rank * local_rows, local_rows),
    )


def _local_qlora_surrogate_logits(
    hidden: Tensor,
    local_weight: Tensor,
    effective_A: Tensor,
    effective_B: Tensor,
    scaling: float = 1.0,
) -> Tensor:
    """Existing QLoRA value formulation used only to define the VJP oracle."""

    base = F.linear(hidden, local_weight)
    delta = scaling * F.linear(F.linear(hidden.float(), effective_A.float()), effective_B.float())
    return (base + delta.to(base.dtype)).float()


def _local_qlora_surrogate_vjp(
    hidden: Tensor,
    local_weight: Tensor,
    effective_A: Tensor,
    effective_B: Tensor,
    grad_local_logits: Tensor,
    *,
    scaling: float = 1.0,
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
        lora_logits = scaling * F.linear(F.linear(lora_hidden, reference_A), reference_B)

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

    def __init__(
        self,
        *,
        tp_rank: int,
        vocab_start: int,
        vocab_end: int,
        padded_vocab_start: int,
        padded_vocab_end: int,
        rank: int = 1,
        lora_alpha: int = 1,
        tp_group: dist.ProcessGroup | None = None,
        expected_group_ranks: tuple[int, ...] = GLM52_LM_HEAD_GROUP_RANKS,
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
        self.expected_group_ranks = tuple(int(rank) for rank in expected_group_ranks)
        if len(self.expected_group_ranks) != GLM52_LM_HEAD_TP_SIZE:
            raise ValueError("GLM-5.2 exact LM-head expected group must contain exactly 16 ranks")
        self.max_lora_rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = glm52_exact_lora_scaling(rank, lora_alpha)

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
        check_exact_head_tp_group(
            program="GLM-5.2 exact LM-head",
            world_size=dist.get_world_size(group),
            group_rank=dist.get_rank(group),
            global_rank=dist.get_rank(),
            group_ranks=tuple(dist.get_process_group_ranks(group)),
            backend=str(dist.get_backend(group)).lower(),
            expected_world_size=GLM52_LM_HEAD_TP_SIZE,
            expected_ranks=self.expected_group_ranks,
            shard_rank=self.shard.tp_rank,
        )
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
        temperature: Tensor | None = None,
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
        expected_A_shape = (self.max_lora_rank, GLM52_LM_HEAD_HIDDEN_SIZE)
        expected_B_shape = (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, self.max_lora_rank)
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
            "local_lora_B": (self.max_lora_rank, 1),
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

        _validate_temperature_rows(
            temperature,
            rows=token_ids.numel(),
            device=hidden_states.device,
        )

        devices = {value.device for value in (hidden_states, local_weight, lora_A, local_lora_B, token_ids)}
        if len(devices) != 1:
            raise RuntimeError(f"GLM-5.2 exact LM-head operands must share one device, got {sorted(map(str, devices))}")
        if require_cuda and hidden_states.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact LM-head value forward requires CUDA and pinned S4 kernels")

    def effective_factor_views(self, lora_A: Tensor, local_lora_B: Tensor) -> tuple[Tensor, Tensor]:
        """Return the exact live BF16 bytes consumed by the S4 A/B kernels."""

        if lora_A.dtype is not torch.float32 or tuple(lora_A.shape) != (
            self.max_lora_rank,
            GLM52_LM_HEAD_HIDDEN_SIZE,
        ):
            raise TypeError("lora_A must be the configured FP32 [rank, 6144] master")
        if local_lora_B.dtype is not torch.float32 or tuple(local_lora_B.shape) != (
            GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            self.max_lora_rank,
        ):
            raise TypeError("local_lora_B must be the configured FP32 [9680, rank] master")
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
        if tuple(effective_A.shape) != (self.max_lora_rank, GLM52_LM_HEAD_HIDDEN_SIZE) or tuple(effective_B.shape) != (
            GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            self.max_lora_rank,
        ):
            raise ValueError("Exact local LM-head effective factor shapes do not match the official shard")
        if any(not value.is_contiguous() for value in (hidden_2d, local_weight, effective_A, effective_B)):
            raise ValueError("Exact local LM-head operands must be contiguous")

        try:
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
        return exact_lora_local_logits(
            hidden_2d,
            effective_A,
            effective_B,
            base_logits=base_logits,
            rank=self.max_lora_rank,
            scaling=self.scaling,
        )

    @staticmethod
    def _selected_logprob_from_gathered(
        full_logits: Tensor,
        token_ids: Tensor,
        temperature: Tensor | None,
    ) -> Tensor:
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
        score_logits = (
            full_logits if temperature is None else exact_temperature_scale_fp32_logits(full_logits, temperature)
        )
        logprob, _lse, _selected = head_v2_selected_logprob_from_logits(
            score_logits,
            token_ids,
            temperature=None,
        )
        return logprob

    @staticmethod
    def _selected_logprob_from_gathered_filtered(
        full_logits: Tensor,
        token_ids: Tensor,
        temperature: Tensor | None,
        sampling_transforms: tuple[Tensor | None, Tensor | None, Tensor | None],
    ) -> Tensor:
        score_logits = (
            full_logits if temperature is None else exact_temperature_scale_fp32_logits(full_logits, temperature)
        )
        top_ks, top_ps, min_ps = sampling_transforms
        if top_ks is None or top_ps is None or min_ps is None:
            raise ValueError("filtered GLM exact scoring requires complete row metadata")
        try:
            from sglang.srt.batch_invariant_ops import (  # noqa: PLC0415
                head_v2_selected_logprob_from_logits,
            )
        except Exception as exc:
            raise RuntimeError("Pinned exact v2 selected-logprob tail is required") from exc
        logprob, _, _ = score_with_sampling_transforms(
            score_logits,
            token_ids,
            top_ks,
            top_ps,
            min_ps,
            lambda native_logits, native_ids: head_v2_selected_logprob_from_logits(
                native_logits,
                native_ids,
                temperature=None,
            ),
        )
        return logprob

    def _exact_forward_value(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        token_ids: Tensor,
        temperature: Tensor | None,
    ) -> Tensor:
        group = self._validate_tp_group()
        rows = hidden_states.numel() // GLM52_LM_HEAD_HIDDEN_SIZE
        hidden_2d = hidden_states.view(rows, GLM52_LM_HEAD_HIDDEN_SIZE)
        token_ids_1d = token_ids.view(rows)
        temperature_1d = _validate_temperature_rows(
            temperature,
            rows=rows,
            device=hidden_states.device,
        )
        local_logits = self._exact_local_logits(hidden_2d, local_weight, effective_A, effective_B)
        full_logits = _rank_order_vocab_all_gather(
            local_logits,
            group,
            expected_world_size=GLM52_LM_HEAD_TP_SIZE,
            expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
        )
        logprob = self._selected_logprob_from_gathered(
            full_logits,
            token_ids_1d,
            temperature_1d,
        )
        return logprob.view_as(token_ids)

    def _exact_forward_value_filtered(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        token_ids: Tensor,
        temperature: Tensor | None,
        sampling_transforms: tuple[Tensor | None, Tensor | None, Tensor | None],
    ) -> Tensor:
        group = self._validate_tp_group()
        rows = hidden_states.numel() // GLM52_LM_HEAD_HIDDEN_SIZE
        hidden_2d = hidden_states.view(rows, GLM52_LM_HEAD_HIDDEN_SIZE)
        token_ids_1d = token_ids.view(rows)
        temperature_1d = _validate_temperature_rows(temperature, rows=rows, device=hidden_states.device)
        local_logits = self._exact_local_logits(hidden_2d, local_weight, effective_A, effective_B)
        full_logits = _rank_order_vocab_all_gather(
            local_logits,
            group,
            expected_world_size=GLM52_LM_HEAD_TP_SIZE,
            expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
        )
        logprob = self._selected_logprob_from_gathered_filtered(
            full_logits, token_ids_1d, temperature_1d, sampling_transforms
        )
        return logprob.view_as(token_ids)

    def _reference_full_logits_fn(self, local_weight, effective_A, effective_B, group):
        """The differentiable QLoRA reference, gathered in serving byte order."""

        def reference_full_logits(hidden_chunk: Tensor) -> Tensor:
            local_reference_logits = _local_qlora_surrogate_logits(
                hidden_chunk,
                local_weight,
                effective_A,
                effective_B,
                self.scaling,
            ).contiguous()
            return _rank_order_vocab_all_gather(
                local_reference_logits,
                group,
                expected_world_size=GLM52_LM_HEAD_TP_SIZE,
                expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            )

        return reference_full_logits

    def _surrogate_vjp(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        token_ids: Tensor,
        grad_logprob: Tensor,
        temperature: Tensor | None,
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        group = self._validate_tp_group()
        rows = hidden_states.numel() // GLM52_LM_HEAD_HIDDEN_SIZE
        hidden_2d = hidden_states.view(rows, GLM52_LM_HEAD_HIDDEN_SIZE)
        token_ids_1d = token_ids.view(rows)
        temperature_1d = _validate_temperature_rows(
            temperature,
            rows=rows,
            device=hidden_states.device,
        )
        local_grad_logits = surrogate_local_grad_logits(
            hidden_2d,
            token_ids_1d,
            grad_logprob.reshape(rows),
            temperature_1d,
            reference_full_logits_fn=self._reference_full_logits_fn(local_weight, effective_A, effective_B, group),
            local_vocab_slice=slice(self.shard.vocab_start, self.shard.vocab_end),
        )
        grad_hidden, grad_A, grad_B = _local_qlora_surrogate_vjp(
            hidden_2d,
            local_weight,
            effective_A,
            effective_B,
            local_grad_logits,
            scaling=self.scaling,
            needs_input_grad=needs_input_grad,
        )
        if grad_hidden is not None:
            grad_hidden = _all_reduce_sum_fp32(grad_hidden.contiguous(), group).view_as(hidden_states)
        if grad_A is not None:
            grad_A = _all_reduce_sum_fp32(grad_A.contiguous(), group)
        return grad_hidden, grad_A, grad_B

    def _surrogate_vjp_filtered(
        self,
        hidden_states: Tensor,
        local_weight: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        token_ids: Tensor,
        grad_logprob: Tensor,
        temperature: Tensor | None,
        sampling_transforms: tuple[Tensor | None, Tensor | None, Tensor | None],
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        group = self._validate_tp_group()
        rows = hidden_states.numel() // GLM52_LM_HEAD_HIDDEN_SIZE
        hidden_2d = hidden_states.view(rows, GLM52_LM_HEAD_HIDDEN_SIZE)
        token_ids_1d = token_ids.view(rows)
        temperature_1d = _validate_temperature_rows(temperature, rows=rows, device=hidden_states.device)

        def _exact_score_logits(hidden_chunk: Tensor, temperature_chunk: Tensor | None) -> Tensor:
            exact_local_logits = self._exact_local_logits(hidden_chunk, local_weight, effective_A, effective_B)
            exact_full_logits = _rank_order_vocab_all_gather(
                exact_local_logits,
                group,
                expected_world_size=GLM52_LM_HEAD_TP_SIZE,
                expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
            )
            if temperature_chunk is None:
                return exact_full_logits
            return exact_temperature_scale_fp32_logits(exact_full_logits, temperature_chunk)

        local_grad_logits = filtered_surrogate_local_grad_logits(
            hidden_2d,
            token_ids_1d,
            grad_logprob.reshape(rows),
            temperature_1d,
            sampling_transforms,
            exact_score_logits_fn=_exact_score_logits,
            reference_full_logits_fn=self._reference_full_logits_fn(local_weight, effective_A, effective_B, group),
            local_vocab_slice=slice(self.shard.vocab_start, self.shard.vocab_end),
            local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
        )
        grad_hidden, grad_A, grad_B = _local_qlora_surrogate_vjp(
            hidden_2d,
            local_weight,
            effective_A,
            effective_B,
            local_grad_logits,
            scaling=self.scaling,
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
        temperature: Tensor | None = None,
        sampling_transforms: tuple[Tensor | None, Tensor | None, Tensor | None] = (None, None, None),
    ) -> Tensor:
        self._validate_operands(
            hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            token_ids,
            temperature,
            require_cuda=True,
        )
        self._validate_tp_group()
        return ExactLmHeadFunction.apply(
            hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            token_ids,
            temperature,
            sampling_transforms,
            REPLICATED_ROW_PLAN,
            self,
        )

    def distributed_selected_logprob(
        self,
        local_hidden_states: Tensor,
        local_weight: Tensor,
        lora_A: Tensor,
        local_lora_B: Tensor,
        local_token_ids: Tensor,
        local_temperature: Tensor | None = None,
        local_sampling_transforms: tuple[Tensor | None, Tensor | None, Tensor | None] = (None, None, None),
    ) -> Tensor:
        """Return local-row logprobs while differentiating the global TP16 row set."""

        self._validate_operands(
            local_hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            local_token_ids,
            local_temperature,
            require_cuda=True,
        )
        row_plan = _distributed_row_plan(local_hidden_states, self._validate_tp_group())
        return ExactLmHeadFunction.apply(
            local_hidden_states,
            local_weight,
            lora_A,
            local_lora_B,
            local_token_ids,
            local_temperature,
            local_sampling_transforms,
            row_plan,
            self,
        )

    def extra_repr(self) -> str:
        return (
            f"tp_rank={self.shard.tp_rank}, vocab=[{self.shard.vocab_start},{self.shard.vocab_end}), "
            f"rank={self.max_lora_rank}, alpha={self.lora_alpha}, temperature=per-row-fp32"
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
    logprob_temperature: float | Tensor,
    logprob_top_ks: Tensor | None,
    logprob_top_ps: Tensor | None,
    logprob_min_ps: Tensor | None,
    tp_group: dist.ProcessGroup | None,
) -> Tensor:
    """Run the exact TP16 head as the grad-enabled source of local-token CE."""

    if not is_glm52_exact_tp16_lm_head(lm_head):
        raise TypeError("glm52_exact_lm_head_per_token_ce requires the constructed exact GLM-5.2 lm_head")
    if ce_mode != "bi_fused":
        raise NotImplementedError("The GLM-5.2 exact active-LoRA lm_head requires ce_mode='bi_fused'")
    if not lm_head_fp32:
        raise NotImplementedError("The GLM-5.2 exact active-LoRA lm_head requires lm_head_fp32=true")

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
    temperature_rows = normalize_temperature_rows(
        logprob_temperature,
        rows=hidden_states_flat.shape[0],
        device=hidden_states_flat.device,
    )
    ce_chunks: list[Tensor] = []
    for start in range(0, hidden_states_flat.shape[0], GLM52_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS):
        end = min(start + GLM52_EXACT_LM_HEAD_LOCAL_CHUNK_ROWS, hidden_states_flat.shape[0])
        logprob = component.distributed_selected_logprob(
            hidden_states_flat[start:end].contiguous(),
            local_weight,
            lora_A,
            local_lora_B,
            safe_labels[start:end].contiguous(),
            None if temperature_rows is None else temperature_rows[start:end].contiguous(),
            tuple(
                None if value is None else value[start:end].contiguous()
                for value in (logprob_top_ks, logprob_top_ps, logprob_min_ps)
            ),
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
