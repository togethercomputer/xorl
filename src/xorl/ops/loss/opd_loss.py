from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from xorl.ops.loss.compiled_cross_entropy import (
    compiled_forward_kl_full_function,
    compiled_forward_kl_full_with_diag_function,
    compiled_reverse_kl_function,
    compiled_reverse_kl_with_diag_function,
    compiled_sampled_token_logprobs_function,
)
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.opd_streaming_kl import (
    streaming_full_vocab_diagnostics,
    streaming_reverse_kl_function,
    streaming_reverse_kl_lowmem_function,
)
from xorl.ops.loss.reducers import Reducer, TokenPartial


# OPD loss modes.
#
# `reverse_kl_full` (default) and `forward_kl_full` are full-vocabulary KL
# variants computed from the *full* teacher distribution — xorl ships teacher
# hidden states + the cached teacher LM head separately, so the full p_T is
# materialized at loss time. VERL's `forward_kl_topk` is intentionally NOT
# ported: it's a bandwidth workaround for teacher-emits-truncated-logprobs
# pipelines, strictly worse than `forward_kl_full` for the same student compute.
#
# Single-sample KL estimators are ported from verl/trainer/ppo/core_algos.py::
# kl_penalty as a different (cheap, sampled-token-only) point in the
# memory/quality trade-off. A trailing "+" applies the k2 straight-through-
# gradient trick (same forward, k2 gradient).
LOSS_MODE_REVERSE_KL_FULL = "reverse_kl_full"
LOSS_MODE_FORWARD_KL_FULL = "forward_kl_full"
_ESTIMATOR_MODES = {"kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"}


def _strip_estimator_plus(loss_mode: str) -> tuple[str, bool]:
    if loss_mode.endswith("+"):
        return loss_mode[:-1], True
    return loss_mode, False


def is_estimator_loss_mode(loss_mode: str) -> bool:
    base, _ = _strip_estimator_plus(loss_mode)
    return base in _ESTIMATOR_MODES


def _kl_penalty_estimator(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    loss_mode: str,
) -> torch.Tensor:
    """Single-sample KL estimators, byte-for-byte port of VERL's kl_penalty.

    See verl/trainer/ppo/core_algos.py::kl_penalty and ::kl_penalty_forward.
    A trailing "+" (e.g. "k3+") applies the k2 straight-through trick: forward
    value from the chosen estimator, but backward as if it were 0.5*(logp-ref)^2.
    """
    base, straight_through = _strip_estimator_plus(loss_mode)
    forward = _kl_penalty_forward(logprob, ref_logprob, base)
    if not straight_through or base in ("mse", "k2"):
        return forward
    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward.detach()


def _kl_penalty_forward(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode in ("kl", "k1"):
        return logprob - ref_logprob
    if loss_mode == "abs":
        return (logprob - ref_logprob).abs()
    if loss_mode in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if loss_mode in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    raise ValueError(f"Unknown KL estimator '{loss_mode}'")


@dataclass(frozen=True)
class OPDLossMetrics:
    """OPD loss metrics emitted per micro-batch.

    All fields are always present in `to_dict()` (defaulting to 0.0 / 0 when
    not applicable to the active loss_mode). This is intentional — dict-keyed
    distributed all-reduces deadlock when ranks have different key sets, so
    every loss path on every rank must contribute the same dict shape.
    """

    valid_tokens: int
    opd_kl: float = 0.0
    opd_weighted_kl: float = 0.0
    opd_hidden_match_loss: float = 0.0
    opd_hidden_match_raw_loss: float = 0.0
    opd_hidden_match_weight_mean: float = 0.0
    opd_hidden_match_pos_loss: float = 0.0
    opd_hidden_match_neg_loss: float = 0.0
    opd_hidden_match_pos_raw_loss: float = 0.0
    opd_hidden_match_neg_raw_loss: float = 0.0
    opd_hidden_match_neg_minus_pos_raw: float = 0.0
    opd_hidden_match_pos_weight_mean: float = 0.0
    opd_hidden_match_neg_weight_mean: float = 0.0
    opd_teacher_weight_mean: float = 0.0
    opd_num_teachers: int = 0
    # ---- Full-vocab diagnostics (reverse_kl_full / forward_kl_full) ----
    opd_teacher_entropy: float = 0.0
    opd_student_entropy: float = 0.0
    opd_top1_agreement: float = 0.0
    # ---- KL-estimator diagnostic (k1/abs/...) ----
    opd_abs_loss: float = 0.0
    # ---- Distillation-loss range ----
    opd_loss_min: float = 0.0
    opd_loss_max: float = 0.0
    opd_loss_abs_mean: float = 0.0
    # Fraction of valid tokens whose pre-clamp |KL| >= loss_max_clamp. Clamped
    # tokens pass ZERO gradient (gradient-dead mass) — watch this when the loss
    # plateaus while opd_loss_max sits pinned at the clamp.
    opd_loss_clamp_frac: float = 0.0
    # ---- Region / sample-correctness KL splits (diagnostic) ----
    # All `*_per_valid` fields are masked SUMS divided by the micro-batch's TOTAL
    # valid-token count (not per-region means), and every `opd_frac_*` is a
    # region-token count over the same denominator. Both are linear functionals
    # of per-token values, so the valid-token-weighted mean aggregation across
    # micro-batches / ranks recomposes them EXACTLY; derive the human-readable
    # region mean as `per_valid / frac` after aggregation (the client does this).
    # Regions come from `diag_region_ids` (0=prompt, 1=buffer, 2=answer, -1=n/a);
    # correctness from `diag_sample_ok` (1=sampled answer correct, 0=wrong,
    # -1=unknown), both client-provided per-token tensors.
    opd_kl_prompt_per_valid: float = 0.0
    opd_kl_buffer_per_valid: float = 0.0
    opd_kl_answer_per_valid: float = 0.0
    opd_frac_prompt: float = 0.0
    opd_frac_buffer: float = 0.0
    opd_frac_answer: float = 0.0
    opd_kl_answer_correct_per_valid: float = 0.0
    opd_kl_answer_wrong_per_valid: float = 0.0
    opd_frac_answer_correct: float = 0.0
    opd_frac_answer_wrong: float = 0.0
    opd_student_entropy_answer_correct_per_valid: float = 0.0
    opd_student_entropy_answer_wrong_per_valid: float = 0.0
    opd_teacher_entropy_answer_correct_per_valid: float = 0.0
    opd_teacher_entropy_answer_wrong_per_valid: float = 0.0
    # ---- Multi-layer OPRD (all-layer hidden matching) ----
    opd_oprd_loss: float = 0.0
    opd_oprd_raw_loss: float = 0.0
    opd_oprd_num_layers: int = 0
    # ---- PG-mode (use_policy_gradient=True) ----
    opd_pg_clipfrac: float = 0.0
    opd_pg_clipfrac_lower: float = 0.0
    opd_ppo_kl: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "valid_tokens": self.valid_tokens,
            "opd_kl": self.opd_kl,
            "opd_weighted_kl": self.opd_weighted_kl,
            "opd_hidden_match_loss": self.opd_hidden_match_loss,
            "opd_hidden_match_raw_loss": self.opd_hidden_match_raw_loss,
            "opd_hidden_match_weight_mean": self.opd_hidden_match_weight_mean,
            "opd_hidden_match_pos_loss": self.opd_hidden_match_pos_loss,
            "opd_hidden_match_neg_loss": self.opd_hidden_match_neg_loss,
            "opd_hidden_match_pos_raw_loss": self.opd_hidden_match_pos_raw_loss,
            "opd_hidden_match_neg_raw_loss": self.opd_hidden_match_neg_raw_loss,
            "opd_hidden_match_neg_minus_pos_raw": self.opd_hidden_match_neg_minus_pos_raw,
            "opd_hidden_match_pos_weight_mean": self.opd_hidden_match_pos_weight_mean,
            "opd_hidden_match_neg_weight_mean": self.opd_hidden_match_neg_weight_mean,
            "opd_teacher_weight_mean": self.opd_teacher_weight_mean,
            "opd_num_teachers": self.opd_num_teachers,
            "opd_teacher_entropy": self.opd_teacher_entropy,
            "opd_student_entropy": self.opd_student_entropy,
            "opd_top1_agreement": self.opd_top1_agreement,
            "opd_abs_loss": self.opd_abs_loss,
            "opd_loss_min": self.opd_loss_min,
            "opd_loss_max": self.opd_loss_max,
            "opd_loss_abs_mean": self.opd_loss_abs_mean,
            "opd_loss_clamp_frac": self.opd_loss_clamp_frac,
            "opd_kl_prompt_per_valid": self.opd_kl_prompt_per_valid,
            "opd_kl_buffer_per_valid": self.opd_kl_buffer_per_valid,
            "opd_kl_answer_per_valid": self.opd_kl_answer_per_valid,
            "opd_frac_prompt": self.opd_frac_prompt,
            "opd_frac_buffer": self.opd_frac_buffer,
            "opd_frac_answer": self.opd_frac_answer,
            "opd_kl_answer_correct_per_valid": self.opd_kl_answer_correct_per_valid,
            "opd_kl_answer_wrong_per_valid": self.opd_kl_answer_wrong_per_valid,
            "opd_frac_answer_correct": self.opd_frac_answer_correct,
            "opd_frac_answer_wrong": self.opd_frac_answer_wrong,
            "opd_student_entropy_answer_correct_per_valid": self.opd_student_entropy_answer_correct_per_valid,
            "opd_student_entropy_answer_wrong_per_valid": self.opd_student_entropy_answer_wrong_per_valid,
            "opd_teacher_entropy_answer_correct_per_valid": self.opd_teacher_entropy_answer_correct_per_valid,
            "opd_teacher_entropy_answer_wrong_per_valid": self.opd_teacher_entropy_answer_wrong_per_valid,
            "opd_oprd_loss": self.opd_oprd_loss,
            "opd_oprd_raw_loss": self.opd_oprd_raw_loss,
            "opd_oprd_num_layers": self.opd_oprd_num_layers,
            "opd_pg_clipfrac": self.opd_pg_clipfrac,
            "opd_pg_clipfrac_lower": self.opd_pg_clipfrac_lower,
            "opd_ppo_kl": self.opd_ppo_kl,
        }


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

    def anchor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=tensor.device)
        return tensor.reshape(-1)[:1].float().sum() * 0.0

    return anchor(hidden_states) + anchor(weight)


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
    hidden_match_weights: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    num_chunks: int = 8,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
    kl_backend: str = "torch_compile",
    vocab_chunk_size: int | None = 32768,
    streaming_lowmem: bool = False,
    return_per_token: bool = False,
    normalization_denominator: Optional[torch.Tensor | int | float] = None,
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
    loss_mode: str = LOSS_MODE_REVERSE_KL_FULL,
    log_prob_min_clamp: Optional[float] = None,
    loss_max_clamp: Optional[float] = None,
    emit_full_vocab_diagnostics: bool = False,
    use_policy_gradient: bool = False,
    old_logprobs: Optional[torch.Tensor] = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    use_task_rewards: bool = False,
    distillation_loss_coef: float = 1.0,
    hidden_match_coef: float = 0.0,
    kl_loss_weight: float = 1.0,
    hidden_match_mode: str = "cosine",
    teacher_layer_hidden_states: Optional[torch.Tensor] = None,
    student_layer_hidden_states: Optional[torch.Tensor] = None,
    diag_region_ids: Optional[torch.Tensor] = None,
    diag_sample_ok: Optional[torch.Tensor] = None,
) -> LossOutput:
    """Compute the OPD distillation loss for one micro-batch.

    Supported `loss_mode`s:
      - "reverse_kl_full" (default): full-vocabulary KL(student||teacher),
        i.e. Σ_v p_S(v)·(log p_S(v) - log p_T(v)).
      - "forward_kl_full": full-vocabulary KL(teacher||student),
        i.e. Σ_v p_T(v)·(log p_T(v) - log p_S(v)). The full-distribution
        counterpart of VERL's truncated `forward_kl_topk` — possible here
        because xorl materializes the full teacher distribution locally via the
        cached teacher LM head.
      - Single-sample estimators "kl"/"k1"/"abs"/"mse"/"k2"/"low_var_kl"/"k3"
        (with optional "+" suffix for k2 straight-through gradient). These use
        only the per-token (student_logprob, teacher_logprob) at the sampled
        token; a cheaper memory point for RL-style updates.

    Set `emit_full_vocab_diagnostics=True` to also emit per-step teacher_entropy,
    student_entropy, and top1_agreement. Adds a single extra full-vocab pass over
    the logits, so opt-in. Only applies to the two full-vocab loss modes.

    Expected shapes:
        hidden_states: [batch, seq, student_hidden_dim]
        weight: [vocab_size, student_hidden_dim]
        labels: [batch, seq], with ignore_index masking tokens out of the loss
        teacher_hidden_states: [batch, seq, teacher_hidden_dim]
        teacher_lm_head_weight: [vocab_size, teacher_hidden_dim]
        teacher_weights: optional [batch, seq] per-token multipliers applied
            to the KL term after KL computation and before final normalization.
        hidden_match_weights: optional [batch, seq] per-token multipliers for
            hidden-state matching. Defaults to teacher_weights when omitted.
        teacher_layer_hidden_states / student_layer_hidden_states: optional
            multi-layer OPRD tensors, each shaped [valid_tokens, L, d] and already
            restricted to the valid (response-masked) positions IN THE SAME ORDER
            as the internal valid_mask flatten (i.e. row i corresponds to the i-th
            True entry of labels.reshape(-1) != ignore_index). When BOTH are given
            the OPRD term — hidden_match_coef * mean_L( (1/d)||student_l -
            teacher_l.detach()||^2 ), weighted by hidden_match_weights and reduced
            by the existing reducer — REPLACES the single-layer hidden-match term to
            avoid double-counting. When None, single-layer behavior is unchanged.

    `diag_region_ids` / `diag_sample_ok` are optional [batch, seq] int tensors
    (labels-aligned, like teacher_weights) that drive metrics-only KL splits:
    region 0=prompt / 1=buffer / 2=answer (-1 = unattributed) and per-sample
    sampled-answer correctness 1/0 (-1 = unknown), broadcast over the sample's
    positions by the client. They never touch the loss value.

    `log_prob_min_clamp` is a generic stability clamp on student log-probs.
    Currently used by `forward_kl_full` (where p_T(v) · log p_S(v) terms can
    blow up if the student puts ~0 mass on a teacher-preferred token); reverse
    KL doesn't need it (its weights are p_S(v)).
    `loss_max_clamp` symmetrically clamps the *unweighted* per-token loss before
    teacher-weighting.

    Teacher tensors are detached by construction. Only the student hidden states
    and student LM head receive gradients.
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
    hidden_weights = (
        _as_flat_optional_weights(hidden_match_weights, valid_mask, torch.float32)
        if hidden_match_weights is not None
        else token_weights
    )

    default_scale = _denominator_tensor(
        normalization_denominator,
        fallback=valid_count,
        device=hidden_states.device,
    )
    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=default_scale)
    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=default_scale)

    region_flat = None
    if diag_region_ids is not None:
        region_flat = diag_region_ids.reshape(-1).to(device=labels.device)[valid_mask]
    sample_ok_flat = None
    if diag_sample_ok is not None:
        sample_ok_flat = diag_sample_ok.reshape(-1).to(device=labels.device)[valid_mask]

    backend = kl_backend.lower()
    is_compile_backend = backend in {"torch_compile", "compile", "auto_chunker"}
    is_streaming_backend = backend in {"streaming", "tilelang"}

    # Full-vocab diagnostics: the compile backend has fused *_with_diag kernels;
    # the streaming backend covers reverse_kl_full via a separate no-grad
    # streaming pass (same chunked-vocab shape, never materializes full logits).
    streaming_diag_supported = is_streaming_backend and loss_mode == LOSS_MODE_REVERSE_KL_FULL
    diag_enabled = bool(emit_full_vocab_diagnostics) and (is_compile_backend or streaming_diag_supported)
    if loss_mode == LOSS_MODE_FORWARD_KL_FULL and not is_compile_backend:
        raise ValueError("loss_mode='forward_kl_full' requires kl_backend='torch_compile'")

    teacher_entropy_per_tok = None
    student_entropy_per_tok = None
    top1_agreement_per_tok = None
    student_lp_at_label: Optional[torch.Tensor] = None
    pg_clipfrac = 0.0
    pg_clipfrac_lower = 0.0
    ppo_kl = 0.0

    if loss_mode == LOSS_MODE_REVERSE_KL_FULL:
        if is_compile_backend:
            if not torch.is_tensor(teacher_lm_head_weight):
                raise ValueError("torch_compile OPD KL backend requires a materialized teacher LM head tensor")
            if diag_enabled:
                (
                    token_kl,
                    teacher_entropy_per_tok,
                    student_entropy_per_tok,
                    top1_agreement_per_tok,
                ) = compiled_reverse_kl_with_diag_function(
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
            else:
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
        elif is_streaming_backend and streaming_lowmem:
            # Memory-lean streaming KL for the fp32 lm-head regime: keep the
            # student/teacher lm-head weights in their native (bf16) dtype and
            # upcast each vocab chunk to fp32 inside the kernel, instead of
            # holding two full fp32 weight copies + a full fp32 grad buffer. Only
            # the cheap [N,H] hidden states are upcast here. Gradient-identical to
            # the streaming path below (slicing commutes with the elementwise
            # upcast; vocab chunks partition grad rows disjointly), but saves
            # several GB of resident lm-head memory -- the AMDAHL-029..033 1-node
            # blocker. The fp32 chunked matmul matches lm_head_fp32=True /
            # teacher_lm_head_fp32=True (the OPD default); the diagnostics pass
            # below is metrics-only and stays on the native-dtype weight.
            if lm_head_fp32:
                student_hidden_flat = student_hidden_flat.float()
            if teacher_lm_head_fp32:
                teacher_hidden_flat = teacher_hidden_flat.float()
            compute_dtype = torch.float32 if (lm_head_fp32 or teacher_lm_head_fp32) else weight.dtype
            token_kl = streaming_reverse_kl_lowmem_function(
                student_hidden_states=student_hidden_flat,
                student_weight=weight,
                teacher_hidden_states=teacher_hidden_flat,
                teacher_weight=teacher_lm_head_weight,
                labels=labels_valid,
                ignore_index=ignore_index,
                vocab_chunk_size=vocab_chunk_size,
                compute_dtype=compute_dtype,
                inplace_weight_grad=False,
            )
            if diag_enabled:
                (
                    teacher_entropy_per_tok,
                    student_entropy_per_tok,
                    top1_agreement_per_tok,
                ) = streaming_full_vocab_diagnostics(
                    student_hidden_states=student_hidden_flat.detach(),
                    student_weight=weight.detach(),
                    teacher_hidden_states=teacher_hidden_flat,
                    teacher_weight=teacher_lm_head_weight,
                    vocab_chunk_size=vocab_chunk_size,
                )
        elif is_streaming_backend:
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
            if diag_enabled:
                (
                    teacher_entropy_per_tok,
                    student_entropy_per_tok,
                    top1_agreement_per_tok,
                ) = streaming_full_vocab_diagnostics(
                    student_hidden_states=student_hidden_flat.detach(),
                    student_weight=weight.detach(),
                    teacher_hidden_states=teacher_hidden_flat,
                    teacher_weight=teacher_lm_head_weight,
                    vocab_chunk_size=vocab_chunk_size,
                )
        else:
            raise ValueError(
                f"Unsupported OPD KL backend '{kl_backend}'. Expected 'torch_compile', 'streaming', or 'tilelang'."
            )
    elif loss_mode == LOSS_MODE_FORWARD_KL_FULL:
        if not torch.is_tensor(teacher_lm_head_weight):
            raise ValueError("forward_kl_full requires a materialized teacher LM head tensor")
        if diag_enabled:
            (
                token_kl,
                teacher_entropy_per_tok,
                student_entropy_per_tok,
                top1_agreement_per_tok,
            ) = compiled_forward_kl_full_with_diag_function(
                student_hidden_states=student_hidden_flat,
                student_weight=weight,
                teacher_hidden_states=teacher_hidden_flat,
                teacher_weight=teacher_lm_head_weight,
                labels=labels_valid,
                ignore_index=ignore_index,
                log_prob_min_clamp=log_prob_min_clamp,
                num_chunks=num_chunks,
                lm_head_fp32=lm_head_fp32,
                teacher_lm_head_fp32=teacher_lm_head_fp32,
            )
        else:
            token_kl = compiled_forward_kl_full_function(
                student_hidden_states=student_hidden_flat,
                student_weight=weight,
                teacher_hidden_states=teacher_hidden_flat,
                teacher_weight=teacher_lm_head_weight,
                labels=labels_valid,
                ignore_index=ignore_index,
                log_prob_min_clamp=log_prob_min_clamp,
                num_chunks=num_chunks,
                lm_head_fp32=lm_head_fp32,
                teacher_lm_head_fp32=teacher_lm_head_fp32,
            )
    elif is_estimator_loss_mode(loss_mode):
        if not torch.is_tensor(teacher_lm_head_weight):
            raise ValueError(f"loss_mode='{loss_mode}' requires a materialized teacher LM head tensor")
        student_lp, teacher_lp = compiled_sampled_token_logprobs_function(
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
        token_kl = _kl_penalty_estimator(student_lp, teacher_lp, loss_mode)
        # `student_lp` is reused by PG mode below, no need for a second forward.
        student_lp_at_label = student_lp
    else:
        raise ValueError(
            f"Unsupported OPD loss_mode '{loss_mode}'. Expected 'reverse_kl_full', "
            f"'forward_kl_full', or one of {_ESTIMATOR_MODES} (with optional '+' suffix)."
        )

    # Symmetric loss clamp before teacher-weighting. Matches
    # verl/trainer/distillation/losses.py:228 (`distillation_losses.clamp(...)`).
    clamp_frac = 0.0
    if loss_max_clamp is not None:
        clamp_frac = (token_kl.detach().abs() >= float(loss_max_clamp)).float().mean().item()
        token_kl = token_kl.clamp(min=-loss_max_clamp, max=loss_max_clamp)
    # Post-clamp, pre-PG-surrogate per-token KL for the metrics-only region /
    # correctness splits below (PG mode replaces token_kl with the surrogate).
    kl_for_diag = token_kl.detach()

    # Policy-gradient OPD: treat -token_kl as advantage, apply PPO-clip on the
    # student's sampled-token log-prob ratio. Mirrors VERL's `use_policy_gradient`
    # path in verl/trainer/distillation/losses.py:259-281.
    if use_policy_gradient:
        if old_logprobs is None:
            raise ValueError("use_policy_gradient=True requires old_logprobs to be provided")
        if student_lp_at_label is None:
            # Need student logprob at sampled-token position. Reuse the same
            # compiled-sampled-logp helper used by estimator modes.
            student_lp_at_label, _ = compiled_sampled_token_logprobs_function(
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
        old_logprobs_flat = old_logprobs.reshape(-1).to(device=labels.device)
        old_logprobs_valid = old_logprobs_flat[valid_mask].to(student_lp_at_label.dtype)
        advantages = (-token_kl).detach()
        log_ratio = student_lp_at_label - old_logprobs_valid
        ratio = log_ratio.exp()
        # min over the two surrogate losses, per PPO. NOTE: token_kl is replaced
        # by the PG surrogate; further teacher-weighting / coef scaling below
        # still applies element-wise.
        pg_losses1 = -ratio * advantages
        pg_losses2 = -ratio.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * advantages
        pg_losses = torch.maximum(pg_losses1, pg_losses2)
        is_clipped = (pg_losses2 > pg_losses1).to(token_kl.dtype)
        pg_clipfrac = is_clipped.mean().item() if is_clipped.numel() > 0 else 0.0
        # Parity with VERL's `actor/ppo_kl` and `actor/pg_clipfrac_lower`
        # (`core_algos.py:1365-1367`, surfaced under `distillation/` prefix in
        # `distillation_loss`). `ppo_kl` = mean(old_logp - new_logp); `clipfrac_lower`
        # = fraction of tokens where the LOWER clamp bound binds (1-clip_ratio_low)
        # AND the advantage is positive (the unclipped surrogate would be smaller
        # than the clipped one, so the clip is "hurting" gradient flow into a
        # promising direction).
        ppo_kl = (-log_ratio).mean().item() if log_ratio.numel() > 0 else 0.0
        is_clipped_lower = ((ratio < (1.0 - clip_ratio_low)) & (advantages < 0)).to(token_kl.dtype)
        pg_clipfrac_lower = is_clipped_lower.mean().item() if is_clipped_lower.numel() > 0 else 0.0
        token_kl = pg_losses  # downstream weighting / reduction unchanged

    token_weights_on_device = token_weights.to(token_kl.device)
    weighted_token_kl = token_kl * token_weights_on_device
    valid_ones = torch.ones_like(weighted_token_kl, dtype=torch.float32)
    valid_count_float = max(float(valid_count.item()), 1.0)

    # kl_loss_weight scales the logit-KL term. Set 0.0 to supervise ONLY on hidden
    # states (loss = hidden_match_coef * hidden_match): in self-distillation the
    # student/teacher share the LM head, so matching answer-position hiddens matches
    # the logits via that head while forcing the buffer to set up the answer hidden.
    if float(kl_loss_weight) == 0.0 and not use_policy_gradient:
        # Hidden-only supervision: the KL term is scaled by 0, so it contributes no
        # gradient -- yet `0.0 * reduce(kl)` still backprops zeros through the full
        # vocab-parallel KL and its LM-head matmul (the single most expensive term;
        # the only gradient path to the shared LM head here). Detach it so the
        # backward skips that subgraph entirely. token_kl/weighted_token_kl remain
        # computed above, so opd_kl and the KL diagnostics are unaffected -- only the
        # (zero) gradient is elided.
        loss = loss_reducer(weighted_token_kl, valid_ones).detach() * 0.0
    else:
        loss = float(kl_loss_weight) * loss_reducer(weighted_token_kl, valid_ones)
    hidden_match_metric = 0.0
    hidden_match_raw_metric = 0.0
    hidden_match_weight_mean = 0.0
    hidden_match_pos_metric = 0.0
    hidden_match_neg_metric = 0.0
    hidden_match_pos_raw_metric = 0.0
    hidden_match_neg_raw_metric = 0.0
    hidden_match_neg_minus_pos_raw = 0.0
    hidden_match_pos_weight_mean = 0.0
    hidden_match_neg_weight_mean = 0.0
    weighted_hidden_match = None
    oprd_metric = 0.0
    oprd_raw_metric = 0.0
    oprd_num_layers = 0
    hidden_match_coef = float(hidden_match_coef or 0.0)
    # Multi-layer OPRD: when BOTH per-layer tensors are present, this REPLACES the
    # single-layer hidden term below (the `else` branch) so the same coefficient
    # isn't applied twice. Layer tensors arrive [valid_tokens, L, d] already
    # restricted to the valid positions in valid_mask order.
    use_oprd = (
        hidden_match_coef
        and teacher_layer_hidden_states is not None
        and student_layer_hidden_states is not None
    )
    if use_oprd:
        student_layers = student_layer_hidden_states.float()
        teacher_layers = teacher_layer_hidden_states.float().detach()
        if student_layers.shape != teacher_layers.shape:
            raise ValueError(
                "OPRD requires matching student/teacher layer shapes, got "
                f"student={tuple(student_layers.shape)} teacher={tuple(teacher_layers.shape)}"
            )
        if student_layers.shape[0] != student_hidden_flat.shape[0]:
            raise ValueError(
                "OPRD layer tensors must have one row per valid token "
                f"({student_hidden_flat.shape[0]}), got {student_layers.shape[0]}"
            )
        oprd_num_layers = int(student_layers.shape[1])
        # mean over hidden dim d -> [valid, L]; mean over layers L -> [valid].
        per_layer_mse = ((student_layers - teacher_layers) ** 2).mean(dim=-1)
        hidden_distance = per_layer_mse.mean(dim=-1)
        hidden_weights_on_device = hidden_weights.to(hidden_distance.device)
        weighted_hidden_match = hidden_distance * hidden_weights_on_device
        hidden_match_loss = loss_reducer(weighted_hidden_match, valid_ones)
        loss = loss + hidden_match_coef * hidden_match_loss
        oprd_metric = metric_reducer(weighted_hidden_match.detach(), valid_ones).item()
        oprd_raw_metric = metric_reducer(hidden_distance.detach(), valid_ones).item()
        hidden_match_metric = oprd_metric
        hidden_match_raw_metric = oprd_raw_metric
        hidden_match_weight_mean = hidden_weights.mean().item()
    elif hidden_match_coef:
        if student_hidden_flat.shape[-1] != teacher_hidden_flat.shape[-1]:
            raise ValueError(
                "hidden_match requires matching hidden sizes, got "
                f"student={student_hidden_flat.shape[-1]} teacher={teacher_hidden_flat.shape[-1]}"
            )
        if str(hidden_match_mode).lower() == "mse":
            # Magnitude-aware: MSE->0 implies student hidden == teacher hidden, so in
            # self-distillation (shared LM head) the logits match too. Cosine alone
            # matches only direction (ignores magnitude) and decouples from generation
            # (eval accuracy collapses while cosine-distance keeps falling).
            hidden_distance = (
                (student_hidden_flat.float() - teacher_hidden_flat.float()) ** 2
            ).mean(dim=-1)
        else:
            hidden_distance = 1.0 - F.cosine_similarity(
                student_hidden_flat.float(),
                teacher_hidden_flat.float(),
                dim=-1,
                eps=1e-6,
            )
        hidden_weights_on_device = hidden_weights.to(hidden_distance.device)
        weighted_hidden_match = hidden_distance * hidden_weights_on_device
        hidden_match_loss = loss_reducer(weighted_hidden_match, valid_ones)
        loss = loss + hidden_match_coef * hidden_match_loss
        hidden_match_metric = metric_reducer(weighted_hidden_match.detach(), valid_ones).item()
        hidden_match_raw_metric = metric_reducer(hidden_distance.detach(), valid_ones).item()
        hidden_match_weight_mean = hidden_weights.mean().item()
        pos_weights = torch.clamp(hidden_weights_on_device, min=0.0)
        neg_weights = torch.clamp(-hidden_weights_on_device, min=0.0)
        hidden_match_pos_metric = metric_reducer((hidden_distance * pos_weights).detach(), valid_ones).item()
        hidden_match_neg_metric = metric_reducer((hidden_distance * neg_weights).detach(), valid_ones).item()
        pos_weight_sum = pos_weights.sum()
        neg_weight_sum = neg_weights.sum()
        if pos_weight_sum.item() > 0:
            hidden_match_pos_raw_metric = (
                (hidden_distance.detach() * pos_weights).sum() / pos_weight_sum
            ).item()
        if neg_weight_sum.item() > 0:
            hidden_match_neg_raw_metric = (
                (hidden_distance.detach() * neg_weights).sum() / neg_weight_sum
            ).item()
        hidden_match_neg_minus_pos_raw = hidden_match_neg_raw_metric - hidden_match_pos_raw_metric
        hidden_match_pos_weight_mean = pos_weights.mean().item()
        hidden_match_neg_weight_mean = neg_weights.mean().item()

    # Task-reward mixing: when use_task_rewards=True the caller is mixing this
    # distillation loss with a separate task-RL loss outside this function and
    # provides the coefficient here. When False, coef is ignored (== VERL semantics:
    # `distillation_loss_coef if use_task_rewards else 1.0`).
    if use_task_rewards:
        loss = loss * float(distillation_loss_coef)

    per_token_loss = None
    if return_per_token:
        per_token_flat = torch.zeros(labels_flat.shape, dtype=torch.float32, device=labels.device)
        per_token_contrib = float(kl_loss_weight) * weighted_token_kl
        if weighted_hidden_match is not None:
            per_token_contrib = per_token_contrib + hidden_match_coef * weighted_hidden_match
        per_token_flat[valid_mask] = per_token_contrib.detach().to(per_token_flat.device)
        per_token_loss = per_token_flat.view(original_shape)

    detached_token_kl = token_kl.detach()
    metrics_kwargs: dict = {
        "valid_tokens": int(valid_count.item()),
        "opd_kl": detached_token_kl.sum().item() / valid_count_float,
        "opd_weighted_kl": metric_reducer(weighted_token_kl.detach(), valid_ones).item(),
        "opd_hidden_match_loss": hidden_match_metric,
        "opd_hidden_match_raw_loss": hidden_match_raw_metric,
        "opd_hidden_match_weight_mean": hidden_match_weight_mean,
        "opd_hidden_match_pos_loss": hidden_match_pos_metric,
        "opd_hidden_match_neg_loss": hidden_match_neg_metric,
        "opd_hidden_match_pos_raw_loss": hidden_match_pos_raw_metric,
        "opd_hidden_match_neg_raw_loss": hidden_match_neg_raw_metric,
        "opd_hidden_match_neg_minus_pos_raw": hidden_match_neg_minus_pos_raw,
        "opd_hidden_match_pos_weight_mean": hidden_match_pos_weight_mean,
        "opd_hidden_match_neg_weight_mean": hidden_match_neg_weight_mean,
        "opd_teacher_weight_mean": token_weights.mean().item(),
    }
    # Distillation-loss range metrics (parity with VERL compute_distillation_loss_range).
    metrics_kwargs["opd_loss_min"] = detached_token_kl.min().item()
    metrics_kwargs["opd_loss_max"] = detached_token_kl.max().item()
    metrics_kwargs["opd_loss_abs_mean"] = detached_token_kl.abs().mean().item()
    metrics_kwargs["opd_loss_clamp_frac"] = clamp_frac

    # Region / sample-correctness KL splits (metrics only; see OPDLossMetrics for
    # the per-valid normalization + exact-aggregation contract).
    def _masked_split(mask: torch.Tensor) -> tuple[float, float]:
        m = mask.to(kl_for_diag.dtype)
        return (
            (kl_for_diag * m).sum().item() / valid_count_float,
            m.sum().item() / valid_count_float,
        )

    if region_flat is not None:
        for region_value, region_name in ((0, "prompt"), (1, "buffer"), (2, "answer")):
            kl_per_valid, frac = _masked_split(region_flat == region_value)
            metrics_kwargs[f"opd_kl_{region_name}_per_valid"] = kl_per_valid
            metrics_kwargs[f"opd_frac_{region_name}"] = frac
        if sample_ok_flat is not None:
            answer_mask = region_flat == 2
            for ok_value, ok_name in ((1, "correct"), (0, "wrong")):
                split_mask = answer_mask & (sample_ok_flat == ok_value)
                kl_per_valid, frac = _masked_split(split_mask)
                metrics_kwargs[f"opd_kl_answer_{ok_name}_per_valid"] = kl_per_valid
                metrics_kwargs[f"opd_frac_answer_{ok_name}"] = frac
                if student_entropy_per_tok is not None and teacher_entropy_per_tok is not None:
                    m = split_mask.to(kl_for_diag.dtype)
                    metrics_kwargs[f"opd_student_entropy_answer_{ok_name}_per_valid"] = (
                        student_entropy_per_tok.detach() * m
                    ).sum().item() / valid_count_float
                    metrics_kwargs[f"opd_teacher_entropy_answer_{ok_name}_per_valid"] = (
                        teacher_entropy_per_tok.detach() * m
                    ).sum().item() / valid_count_float
    metrics_kwargs["opd_oprd_loss"] = oprd_metric
    metrics_kwargs["opd_oprd_raw_loss"] = oprd_raw_metric
    metrics_kwargs["opd_oprd_num_layers"] = oprd_num_layers
    if is_estimator_loss_mode(loss_mode):
        # k1 can be negative; mirror VERL's distillation/abs_loss metric.
        metrics_kwargs["opd_abs_loss"] = detached_token_kl.abs().mean().item()
    if use_policy_gradient:
        metrics_kwargs["opd_pg_clipfrac"] = float(pg_clipfrac)
        metrics_kwargs["opd_pg_clipfrac_lower"] = float(pg_clipfrac_lower)
        metrics_kwargs["opd_ppo_kl"] = float(ppo_kl)
    # All flattened tensors here are over already-filtered valid tokens, so
    # every position counts as "response-masked valid" and mean = sum / valid_count.
    if teacher_entropy_per_tok is not None and student_entropy_per_tok is not None:
        metrics_kwargs["opd_teacher_entropy"] = teacher_entropy_per_tok.detach().mean().item()
        metrics_kwargs["opd_student_entropy"] = student_entropy_per_tok.detach().mean().item()
    if top1_agreement_per_tok is not None:
        metrics_kwargs["opd_top1_agreement"] = top1_agreement_per_tok.detach().mean().item()

    metrics = OPDLossMetrics(**metrics_kwargs).to_dict()

    return LossOutput(loss=loss, per_token_loss=per_token_loss, metrics=metrics)
