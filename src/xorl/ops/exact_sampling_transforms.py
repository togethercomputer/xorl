"""The backend-independent sampling-transform program: XoRL's replay contract.

This module is the single scoring program for behavior-policy replay of
top-k/top-p/min-p-filtered rollouts.  Every lane consumes it — the exact
value programs (Qwen, GLM, DSV4) after their declared temperature dtype/store
boundary, ``ce_mode='bi_fused'``, and the generic modes (``eager``,
``compiled``, ``quack_linear``, ``fused_quack``) through
:mod:`xorl.ops.loss.sampling_transform_ce`.

The pinned program (``EXACT_SAMPLING_TRANSFORM_PROGRAM``):

1. temperature first, at the lane's declared dtype boundary;
2. stable descending probability order — equal probabilities are ordered by
   vocabulary index, so token ID breaks ties;
3. joint top-k AND inclusive-crossing top-p (the first token that crosses the
   threshold, via ``cumulative_probability_before <= top_p``, is kept) AND
   min-p relative to the ORIGINAL unfiltered row maximum;
4. ``top_p == 1.0`` is the exact mathematical identity and always yields full
   support, even when the sorted FP32 cumulative sum overshoots 1.0
   (``exact_sampling_identity_rows`` overrides the fixed-shape comparison);
5. normalization on exactly that support; serving samples with per-row seeded
   Gumbel-max on the masked logits.

The tie-break rule, the inclusive crossing, and the min-p reference point are
choices this program pins; stock SGLang/FlashInfer filter kernels do not
guarantee them.  Exact replay therefore requires a sampler that implements
THIS program; scoring rollouts produced by a stock filter is approximate at
the tokens on the filter boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


TOP_K_ALL = 1 << 30
EXACT_FILTER_ROW_CHUNK = 32
EXACT_SAMPLING_TRANSFORM_PROGRAM = (
    "temperature_then_stable_token_id_topk_inclusive_topp_original_max_minp_seeded_gumbel_v1"
)
SamplingTransformRows = tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
NativeSelectedScore = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _normalize_row_metadata(
    value: int | float | torch.Tensor,
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype is not dtype:
            raise TypeError(f"per-row {name} must be {dtype}, got {value.dtype}")
        if value.device != device:
            raise ValueError(f"per-row {name} must share the logits device")
        if tuple(value.shape) != (rows,):
            raise ValueError(f"per-row {name} must have shape ({rows},), got {tuple(value.shape)}")
        if not value.is_contiguous() or value.requires_grad:
            raise ValueError(f"per-row {name} must be contiguous, non-differentiable sampling metadata")
        return value
    return torch.full((rows,), value, dtype=dtype, device=device)


def normalize_exact_sampling_transforms(
    top_k: int | torch.Tensor = TOP_K_ALL,
    top_p: float | torch.Tensor = 1.0,
    min_p: float | torch.Tensor = 0.0,
    *,
    rows: int,
    device: torch.device,
) -> SamplingTransformRows:
    """Validate transforms and return row tensors, or three ``None`` values.

    Returning ``None`` for the all-identity scalar case is the explicit switch
    that keeps established no-filter kernels and bytes untouched.
    """

    scalar_identity = (
        not isinstance(top_k, torch.Tensor)
        and not isinstance(top_p, torch.Tensor)
        and not isinstance(min_p, torch.Tensor)
        and int(top_k) >= TOP_K_ALL
        and float(top_p) == 1.0
        and float(min_p) == 0.0
    )
    if scalar_identity:
        return None, None, None

    top_ks = _normalize_row_metadata(
        top_k,
        rows=rows,
        device=device,
        dtype=torch.int64,
        name="logprob_top_ks",
    )
    top_ps = _normalize_row_metadata(
        top_p,
        rows=rows,
        device=device,
        dtype=torch.float32,
        name="logprob_top_ps",
    )
    min_ps = _normalize_row_metadata(
        min_p,
        rows=rows,
        device=device,
        dtype=torch.float32,
        name="logprob_min_ps",
    )
    torch._assert_async((top_ks >= 1).all(), "logprob_top_ks must contain integers >= 1")
    torch._assert_async(
        (torch.isfinite(top_ps) & (top_ps > 0.0) & (top_ps <= 1.0)).all(),
        "logprob_top_ps must contain finite values in (0, 1]",
    )
    torch._assert_async(
        (torch.isfinite(min_ps) & (min_ps >= 0.0) & (min_ps <= 1.0)).all(),
        "logprob_min_ps must contain finite values in [0, 1]",
    )
    if bool(((top_ks >= TOP_K_ALL) & (top_ps == 1.0) & (min_ps == 0.0)).all().item()):
        return None, None, None
    return top_ks, top_ps, min_ps


def validate_temperature_rows(
    temperature: torch.Tensor | None,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate per-row FP32 temperature metadata; ``None`` passes through."""

    if temperature is None:
        return None
    if temperature.dtype is not torch.float32:
        raise TypeError(f"per-row logprob_temperature must be FP32, got {temperature.dtype}")
    if temperature.device != device or tuple(temperature.shape) != (rows,):
        raise ValueError(
            "per-row logprob_temperature must be row-aligned on the scoring device, got "
            f"shape={tuple(temperature.shape)} device={temperature.device}"
        )
    if not temperature.is_contiguous() or temperature.requires_grad:
        raise ValueError("per-row logprob_temperature must be contiguous, non-differentiable sampling metadata")
    if temperature.device.type != "meta":
        torch._assert_async(
            (torch.isfinite(temperature) & (temperature > 0)).all(),
            "per-row logprob_temperature must contain finite values > 0",
        )
    return temperature


def normalize_temperature_rows(
    temperature: float | torch.Tensor,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return validated per-row FP32 temperature, or ``None`` for the identity.

    A tensor is validated as-is; the scalar ``1.0`` collapses to ``None`` (the
    explicit switch that keeps no-temperature kernels and bytes untouched);
    any other scalar is broadcast to a per-row tensor.
    """

    if isinstance(temperature, torch.Tensor):
        return validate_temperature_rows(temperature, rows=rows, device=device)
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"logprob_temperature must be finite and > 0, got {value}")
    if value == 1.0:
        return None
    return torch.full((rows,), value, dtype=torch.float32, device=device)


def validate_sampling_transform_rows(
    top_ks: torch.Tensor | None,
    top_ps: torch.Tensor | None,
    min_ps: torch.Tensor | None,
    *,
    rows: int,
    device: torch.device,
) -> SamplingTransformRows:
    """Validate an all-or-none per-row transform triple; a None triple passes.

    Row metadata that is present must be exactly row-aligned — silently
    truncating an over-length tensor would score tokens against the wrong
    row's filter.
    """

    none_count = (top_ks is None, top_ps is None, min_ps is None).count(True)
    if none_count == 3:
        return None, None, None
    if none_count:
        raise ValueError("sampling-transform replay requires all or none of top-k/top-p/min-p row metadata")
    return (
        _normalize_row_metadata(top_ks, rows=rows, device=device, dtype=torch.int64, name="logprob_top_ks"),
        _normalize_row_metadata(top_ps, rows=rows, device=device, dtype=torch.float32, name="logprob_top_ps"),
        _normalize_row_metadata(min_ps, rows=rows, device=device, dtype=torch.float32, name="logprob_min_ps"),
    )


def score_with_sampling_transforms(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    native_selected_score: NativeSelectedScore,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score temperature-transformed logits under the pinned filter program.

    Identity rows keep the lane's native (unfiltered) selected-score bytes;
    filtered rows score on their exact joint support.
    """

    support = exact_sampling_support(logits, top_ks, top_ps, min_ps)
    identity_rows = exact_sampling_identity_rows(
        top_ks,
        top_ps,
        min_ps,
        vocab_size=logits.shape[1],
    )
    return exact_selected_logprob_partitioned_from_support(
        logits,
        token_ids,
        support,
        identity_rows,
        native_selected_score,
    )


def selected_logprob_reference_grad(
    full_logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_logprob: torch.Tensor,
    temperature: torch.Tensor | None,
) -> torch.Tensor:
    """Differentiate the FP32 selected log-softmax reference at the logits."""

    with torch.enable_grad(), torch.autocast(device_type=full_logits.device.type, enabled=False):
        logits = full_logits.float().detach().requires_grad_(True)
        score_logits = logits if temperature is None else logits * (1.0 / temperature).unsqueeze(1)
        selected = F.log_softmax(score_logits, dim=-1).gather(1, token_ids.unsqueeze(1)).squeeze(1)
        (gradient,) = torch.autograd.grad(selected, logits, grad_outputs=grad_logprob.float())
    return gradient.contiguous()


def selected_logprob_reference_grad_filtered(
    full_logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_logprob: torch.Tensor,
    temperature: torch.Tensor | None,
    support: torch.Tensor,
) -> torch.Tensor:
    """Differentiate over the exact support chosen by the value forward."""

    with torch.enable_grad(), torch.autocast(device_type=full_logits.device.type, enabled=False):
        logits = full_logits.float().detach().requires_grad_(True)
        score_logits = logits if temperature is None else logits * (1.0 / temperature).unsqueeze(1)
        selected, _, _ = exact_selected_logprob_from_support(score_logits, token_ids, support)
        (gradient,) = torch.autograd.grad(selected, logits, grad_outputs=grad_logprob.float())
    return gradient.contiguous()


def selected_logprob_reference_grad_partitioned(
    full_logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_logprob: torch.Tensor,
    temperature: torch.Tensor | None,
    support: torch.Tensor,
    identity_rows: torch.Tensor,
) -> torch.Tensor:
    """Identity rows differentiate natively; filtered rows over their support."""

    native_grad = selected_logprob_reference_grad(
        full_logits,
        token_ids,
        grad_logprob,
        temperature,
    )
    filtered_grad = selected_logprob_reference_grad_filtered(
        full_logits,
        token_ids,
        grad_logprob,
        temperature,
        support,
    )
    return torch.where(identity_rows.unsqueeze(1), native_grad, filtered_grad).contiguous()


def exact_sampling_support(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> torch.Tensor:
    """Return the joint top-k/top-p/min-p support for current logits.

    Temperature must already have been applied.  Sorting is stable, so equal
    probabilities are ordered by their original vocabulary index (token ID).
    Top-p keeps the first token that crosses the threshold via
    ``cumulative_probability_before <= top_p``. Min-p is relative to the
    unfiltered row maximum. The three conditions are applied jointly.
    """

    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("exact sampling transforms require floating [rows, vocab] logits")
    rows, vocab = logits.shape
    if vocab < 1:
        raise ValueError("exact sampling transforms require a non-empty vocabulary")
    if tuple(top_ks.shape) != (rows,) or tuple(top_ps.shape) != (rows,) or tuple(min_ps.shape) != (rows,):
        raise ValueError("exact sampling transform metadata must align one-to-one with logit rows")

    identity_rows = exact_sampling_identity_rows(
        top_ks,
        top_ps,
        min_ps,
        vocab_size=vocab,
    )
    # The support is discrete metadata.  Detaching is essential: current logits
    # choose the support every forward, but gradients do not pass through sort
    # indices or threshold comparisons.  Identity rows are overwritten to full
    # support after the fixed-shape program: a rounded FP32 cumulative sum may
    # exceed one, but top-p=1 is mathematically unconditional.  Keeping the
    # fixed row shape also preserves CUDA-graph capture in serving.
    probabilities = torch.softmax(logits.detach(), dim=-1)
    sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True, stable=True)
    ranks = torch.arange(vocab, device=logits.device, dtype=top_ks.dtype).unsqueeze(0)
    keep_sorted = ranks < top_ks.clamp(max=vocab).unsqueeze(1)
    cumulative_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    keep_sorted &= cumulative_before <= top_ps.unsqueeze(1)
    keep_sorted &= sorted_probs >= sorted_probs[:, :1] * min_ps.unsqueeze(1)

    support = torch.zeros((rows, vocab), dtype=torch.bool, device=logits.device)
    support.scatter_(1, sorted_indices, keep_sorted)
    support |= identity_rows.unsqueeze(1)
    return support


def exact_sampling_identity_rows(
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Return rows whose transforms are the exact mathematical identity."""

    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    if top_ks.ndim != 1 or top_ps.shape != top_ks.shape or min_ps.shape != top_ks.shape:
        raise ValueError("exact sampling transform metadata must have aligned one-dimensional rows")
    return (top_ks >= vocab_size) & (top_ps == 1.0) & (min_ps == 0.0)


def exact_masked_logits(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = exact_sampling_support(logits, top_ks, top_ps, min_ps)
    return logits.masked_fill(~support, -math.inf), support


def exact_selected_logprob_from_support(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score selected tokens under an already-derived exact support.

    A token outside current support has logprob ``-inf`` and zero gradient.
    This is the required current-policy meaning when a historical action falls
    out of top-k/top-p/min-p support after a weight update.
    """

    rows = logits.shape[0]
    if token_ids.dtype is not torch.int64 or tuple(token_ids.shape) != (rows,):
        raise ValueError("selected token IDs must be int64 and row-aligned")
    masked_logits = logits.masked_fill(~support, -math.inf)
    lse = torch.logsumexp(masked_logits, dim=-1)
    selected = logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    selected_support = support.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    finite_logprob = torch.minimum(selected - lse, torch.zeros_like(selected))
    logprob = torch.where(selected_support, finite_logprob, torch.full_like(finite_logprob, -math.inf))
    return logprob, lse, selected_support


def exact_selected_logprob_partitioned_from_support(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    support: torch.Tensor,
    identity_rows: torch.Tensor,
    native_selected_score: NativeSelectedScore,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score identity rows natively and filtered rows on their exact support."""

    rows = logits.shape[0]
    if identity_rows.dtype is not torch.bool or identity_rows.shape != (rows,):
        raise ValueError("identity_rows must be a row-aligned bool tensor")

    native_logprob, native_lse, _ = native_selected_score(logits, token_ids)
    filtered_logprob, filtered_lse, filtered_selected_support = exact_selected_logprob_from_support(
        logits,
        token_ids,
        support,
    )
    return (
        torch.where(identity_rows, native_logprob, filtered_logprob),
        torch.where(identity_rows, native_lse, filtered_lse),
        torch.where(identity_rows, torch.ones_like(filtered_selected_support), filtered_selected_support),
    )


def exact_selected_logprob(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support = exact_sampling_support(logits, top_ks, top_ps, min_ps)
    logprob, lse, selected_support = exact_selected_logprob_from_support(logits, token_ids, support)
    return logprob, lse, selected_support, support


def exact_selected_logprob_chunked(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    *,
    row_chunk_size: int = EXACT_FILTER_ROW_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score rows without retaining a dense ``[tokens, vocab]`` support mask."""

    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1")
    logprob_chunks = []
    lse_chunks = []
    selected_support_chunks = []
    for start in range(0, logits.shape[0], row_chunk_size):
        end = min(start + row_chunk_size, logits.shape[0])
        logprob, lse, selected_support, _support = exact_selected_logprob(
            logits[start:end],
            token_ids[start:end],
            top_ks[start:end],
            top_ps[start:end],
            min_ps[start:end],
        )
        logprob_chunks.append(logprob)
        lse_chunks.append(lse)
        selected_support_chunks.append(selected_support)
    if not logprob_chunks:
        empty_float = logits.new_empty((0,))
        return empty_float, empty_float.clone(), torch.empty((0,), dtype=torch.bool, device=logits.device)
    return (
        torch.cat(logprob_chunks),
        torch.cat(lse_chunks),
        torch.cat(selected_support_chunks),
    )


def exact_support_workspace_bytes(vocab_size: int, row_chunk_size: int = EXACT_FILTER_ROW_CHUNK) -> int:
    """Upper bound for dense bool workspace in one support row chunk.

    At most the sorted keep mask and token-order support mask coexist.  Float
    probabilities and sort indices are separate value-program workspaces.
    """

    if vocab_size < 1 or row_chunk_size < 1:
        raise ValueError("vocab_size and row_chunk_size must be >= 1")
    return 2 * vocab_size * row_chunk_size


__all__ = [
    "EXACT_FILTER_ROW_CHUNK",
    "EXACT_SAMPLING_TRANSFORM_PROGRAM",
    "TOP_K_ALL",
    "exact_masked_logits",
    "exact_sampling_identity_rows",
    "exact_sampling_support",
    "exact_selected_logprob",
    "exact_selected_logprob_chunked",
    "exact_selected_logprob_from_support",
    "exact_selected_logprob_partitioned_from_support",
    "exact_support_workspace_bytes",
    "normalize_exact_sampling_transforms",
    "normalize_temperature_rows",
    "score_with_sampling_transforms",
    "selected_logprob_reference_grad",
    "selected_logprob_reference_grad_filtered",
    "selected_logprob_reference_grad_partitioned",
    "validate_sampling_transform_rows",
    "validate_temperature_rows",
]
