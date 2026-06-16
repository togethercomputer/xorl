from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from xorl.ops.loss.compiled_cross_entropy import compiled_reverse_kl_function
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.opd_streaming_kl import streaming_reverse_kl_function
from xorl.ops.loss.reducers import Reducer, TokenPartial


@dataclass(frozen=True)
class OPDLossMetrics:
    valid_tokens: int
    opd_kl: float = 0.0
    opd_weighted_kl: float = 0.0
    opd_teacher_weight_mean: float = 0.0
    opd_num_teachers: Optional[int] = None

    def to_dict(self) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {
            "valid_tokens": self.valid_tokens,
            "opd_kl": self.opd_kl,
            "opd_weighted_kl": self.opd_weighted_kl,
            "opd_teacher_weight_mean": self.opd_teacher_weight_mean,
        }
        if self.opd_num_teachers is not None:
            metrics["opd_num_teachers"] = self.opd_num_teachers
        return metrics


def _as_flat_optional_weights(
    teacher_weights: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if teacher_weights is None:
        return torch.ones(valid_mask.sum(), dtype=dtype, device=valid_mask.device)
    weights_flat = teacher_weights.reshape(-1).to(device=valid_mask.device, dtype=dtype)
    return weights_flat[valid_mask]


def _zero_loss_with_graph(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Build a 0-valued loss that still flows gradients through hidden_states + weight.

    Always returns fp32 so the dtype matches the normal-return path
    (`total_weighted_kl / denom`, fp32). A dtype mismatch between the early-return
    branch on no-valid-token ranks and the fp32 normal branch corrupts NCCL
    all_reduce in the trainer's loss-reporting path.
    """
    return hidden_states.float().sum() * 0.0 + weight.float().sum() * 0.0


def _denominator_tensor(
    denominator: torch.Tensor | int | float | None,
    *,
    fallback: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    if denominator is None:
        return fallback.to(device=device, dtype=torch.float32)
    if torch.is_tensor(denominator):
        return denominator.to(device=device, dtype=torch.float32)
    return torch.tensor(float(denominator), device=device, dtype=torch.float32)


def opd_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    teacher_weights: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    num_chunks: int = 8,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
    kl_backend: str = "torch_compile",
    vocab_chunk_size: int = 32768,
    return_per_token: bool = False,
    normalization_denominator: Optional[torch.Tensor | int | float] = None,
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
) -> LossOutput:
    """Compute full-vocabulary reverse KL for on-policy distillation.

    The objective is KL(student || teacher) at each valid token position:
        sum_v p_student(v) * (log p_student(v) - log p_teacher(v)).

    Expected shapes:
        hidden_states: [batch, seq, student_hidden_dim]
        weight: [vocab_size, student_hidden_dim]
        labels: [batch, seq], with ignore_index masking tokens out of the loss
        teacher_hidden_states: [batch, seq, teacher_hidden_dim]
        teacher_lm_head_weight: [vocab_size, teacher_hidden_dim]
        teacher_weights: optional [batch, seq] per-token multipliers applied
            after KL computation and before the final normalization. These are
            useful for mixing teachers or down-weighting lower-confidence teacher
            outputs without changing the valid-token denominator.

    Teacher tensors are detached by construction. Only the student hidden states
    and student LM head receive gradients.

    ``kl_backend`` selects the full-vocabulary KL implementation. ``torch_compile``
    preserves the existing auto-chunked path. ``streaming`` and ``tilelang`` use
    the OPD streaming path that saves only per-token normalization statistics and
    recomputes vocab chunks in backward; ``tilelang`` is the stable selector for
    the future native TileLang kernel.
    """
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError(f"hidden_states shape {hidden_states.shape} is incompatible with labels {labels.shape}")
    if teacher_hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            f"teacher_hidden_states shape {teacher_hidden_states.shape} is incompatible with labels {labels.shape}"
        )
    if weight.shape[0] != teacher_lm_head_weight.shape[0]:
        raise ValueError(
            f"student vocab size ({weight.shape[0]}) must match teacher vocab size ({teacher_lm_head_weight.shape[0]})"
        )
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"student hidden size ({hidden_states.shape[-1]}) must match student head width ({weight.shape[-1]})"
        )
    if teacher_hidden_states.shape[-1] != teacher_lm_head_weight.shape[-1]:
        raise ValueError(
            "teacher hidden size "
            f"({teacher_hidden_states.shape[-1]}) must match teacher head width ({teacher_lm_head_weight.shape[-1]})"
        )

    original_shape = labels.shape
    labels_flat = labels.reshape(-1)
    valid_mask = labels_flat != ignore_index
    valid_count = valid_mask.sum()

    if valid_count.item() == 0:
        loss = _zero_loss_with_graph(hidden_states, weight)
        per_token_loss = (
            torch.zeros(original_shape, dtype=torch.float32, device=labels.device) if return_per_token else None
        )
        return LossOutput(loss=loss, per_token_loss=per_token_loss, metrics=OPDLossMetrics(valid_tokens=0).to_dict())

    student_hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))[valid_mask]
    teacher_hidden_flat = teacher_hidden_states.reshape(-1, teacher_hidden_states.size(-1))[valid_mask].detach()
    labels_valid = labels_flat[valid_mask]
    token_weights = _as_flat_optional_weights(teacher_weights, valid_mask, torch.float32)

    default_scale = _denominator_tensor(
        normalization_denominator,
        fallback=valid_count,
        device=hidden_states.device,
    )
    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=default_scale)
    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=default_scale)

    backend = kl_backend.lower()
    if backend in {"torch_compile", "compile", "auto_chunker"}:
        if not torch.is_tensor(teacher_lm_head_weight):
            raise ValueError("torch_compile OPD KL backend requires a materialized teacher LM head tensor")
        token_kl = compiled_reverse_kl_function(
            student_hidden_states=student_hidden_flat,
            student_weight=weight,
            teacher_hidden_states=teacher_hidden_flat,
            teacher_weight=teacher_lm_head_weight,
            labels=labels_valid,
            ignore_index=ignore_index,
            num_chunks=num_chunks,
            lm_head_fp32=lm_head_fp32,
            teacher_lm_head_fp32=teacher_lm_head_fp32,
        )
    elif backend in {"streaming", "tilelang"}:
        if lm_head_fp32:
            student_hidden_flat = student_hidden_flat.float()
            weight = weight.float()
        if teacher_lm_head_fp32:
            teacher_hidden_flat = teacher_hidden_flat.float()
            if torch.is_tensor(teacher_lm_head_weight):
                teacher_lm_head_weight = teacher_lm_head_weight.float()
        token_kl = streaming_reverse_kl_function(
            student_hidden_states=student_hidden_flat,
            student_weight=weight,
            teacher_hidden_states=teacher_hidden_flat,
            teacher_weight=teacher_lm_head_weight,
            labels=labels_valid,
            ignore_index=ignore_index,
            vocab_chunk_size=vocab_chunk_size,
        )
    else:
        raise ValueError(
            f"Unsupported OPD KL backend '{kl_backend}'. Expected 'torch_compile', 'streaming', or 'tilelang'."
        )
    weighted_token_kl = token_kl * token_weights.to(token_kl.device)
    valid_ones = torch.ones_like(weighted_token_kl, dtype=torch.float32)

    loss = loss_reducer(weighted_token_kl, valid_ones)

    per_token_loss = None
    if return_per_token:
        per_token_flat = torch.zeros(labels_flat.shape, dtype=torch.float32, device=labels.device)
        per_token_flat[valid_mask] = weighted_token_kl.detach().to(per_token_flat.device)
        per_token_loss = per_token_flat.view(original_shape)

    valid_count_float = max(float(valid_count.item()), 1.0)
    metrics = OPDLossMetrics(
        valid_tokens=int(valid_count.item()),
        opd_kl=token_kl.detach().sum().item() / valid_count_float,
        opd_weighted_kl=metric_reducer(weighted_token_kl.detach(), valid_ones).item(),
        opd_teacher_weight_mean=token_weights.mean().item(),
    ).to_dict()

    return LossOutput(loss=loss, per_token_loss=per_token_loss, metrics=metrics)
