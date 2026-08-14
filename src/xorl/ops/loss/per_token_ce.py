"""Per-token cross-entropy computation with optional vocab-parallel TP support."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.distributed.parallel_state import get_parallel_state
from xorl.ops.exact_sampling_transforms import (
    TOP_K_ALL,
    normalize_exact_sampling_transforms,
)
from xorl.ops.loss.compiled_cross_entropy import compiled_cross_entropy_function
from xorl.ops.loss.fused_linear_logprob import fused_selected_logprob_ce
from xorl.ops.loss.vocab_parallel_cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy_with_lm_head,
)


_MODULE_LM_HEAD_MIN_CHUNK_ROWS = 128

LogprobTemperature = float | torch.Tensor
LogprobTopK = int | torch.Tensor
LogprobProbability = float | torch.Tensor


def resolve_bi_fused_lm_head_tp_groups(
    ce_mode: str,
    tp_group: Optional[dist.ProcessGroup],
    lm_head: Optional[torch.nn.Module],
) -> tuple[dist.ProcessGroup, Optional[dist.ProcessGroup]] | None:
    """Resolve the generic bi-fused vocabulary-sharded LM-head topology.

    The route is owned by the physical sharded head and the dedicated runtime
    process group. A body-TP group cannot opt into this arithmetic merely by
    being passed as ``tp_group``.
    """

    if ce_mode != "bi_fused" or lm_head is None:
        return None
    if getattr(lm_head, "_glm52_exact_tp16_lm_head", False) or getattr(lm_head, "_dsv4_exact_tp8_lm_head", False):
        return None
    if not getattr(lm_head, "_xorl_fsdp_sharded_lm_head_loss", False):
        return None

    ps = get_parallel_state()
    dedicated_group = getattr(ps, "lm_head_tp_group", None)
    if (
        getattr(ps, "tp_enabled", False)
        or getattr(ps, "lm_head_tp_size", 1) <= 1
        or dedicated_group is None
        or tp_group is not dedicated_group
    ):
        raise NotImplementedError(
            "ce_mode='bi_fused' requires the marked vocabulary-sharded lm_head to use its dedicated LM-head TP group"
        )
    return dedicated_group, getattr(ps, "lm_head_tp_replica_group", None)


def _flatten_row_metadata(value, *, rows: int, name: str):
    """Flatten a collated ``[B, S]`` transform field to exact head row order."""
    if not isinstance(value, torch.Tensor) or tuple(value.shape) == (rows,):
        return value
    if not value.is_contiguous():
        raise ValueError(f"per-row {name} must be contiguous")
    if value.numel() != rows:
        raise ValueError(f"per-row {name} must contain {rows} values, got shape {tuple(value.shape)}")
    return value.reshape(rows)


def _module_lm_head_ce(
    hidden_states_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    ignore_index: int,
    num_chunks: int,
    logprob_temperature: float,
) -> torch.Tensor:
    if hidden_states_flat.shape[0] == 0:
        return hidden_states_flat.new_empty((0,), dtype=torch.float32)

    chunk_count = max(1, int(num_chunks))
    chunk_size = max(
        _MODULE_LM_HEAD_MIN_CHUNK_ROWS,
        (hidden_states_flat.shape[0] + chunk_count - 1) // chunk_count,
    )
    ce_chunks: list[torch.Tensor] = []
    for start in range(0, hidden_states_flat.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states_flat.shape[0])
        logits = lm_head(hidden_states_flat[start:end]).float()
        if logprob_temperature != 1.0:
            logits = logits / logprob_temperature
        ce_chunks.append(F.cross_entropy(logits, labels_flat[start:end], reduction="none", ignore_index=ignore_index))
    return torch.cat(ce_chunks, dim=0)


def normalize_logprob_temperature(
    logprob_temperature: LogprobTemperature,
    *,
    rows: int,
    device: torch.device,
) -> LogprobTemperature:
    """Validate the scalar or exact per-row FP32 sampling temperature.

    Exact heads consume a contiguous ``[rows]`` FP32 tensor in the same
    logical row order as hidden states and labels.  Ordinary loss modes keep
    their existing scalar-only contract.
    """

    if isinstance(logprob_temperature, torch.Tensor):
        if logprob_temperature.dtype is not torch.float32:
            raise TypeError(f"per-row logprob_temperature must be FP32, got {logprob_temperature.dtype}")
        if logprob_temperature.device != device:
            raise ValueError(
                f"per-row logprob_temperature must share the loss device, got {logprob_temperature.device} and {device}"
            )
        if tuple(logprob_temperature.shape) != (rows,):
            raise ValueError(
                f"per-row logprob_temperature must have shape ({rows},), got {tuple(logprob_temperature.shape)}"
            )
        if not logprob_temperature.is_contiguous():
            raise ValueError("per-row logprob_temperature must be contiguous")
        if logprob_temperature.requires_grad:
            raise ValueError("per-row logprob_temperature is sampling metadata and cannot require gradients")
        torch._assert_async(
            (torch.isfinite(logprob_temperature) & (logprob_temperature > 0)).all(),
            "per-row logprob_temperature must contain finite values > 0",
        )
        return logprob_temperature

    value = float(logprob_temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"logprob_temperature must be finite and > 0, got {value}")
    return value


def _scale_hidden_for_temperature(hidden_states_flat: torch.Tensor, logprob_temperature: float) -> torch.Tensor:
    if logprob_temperature == 1.0:
        return hidden_states_flat
    return hidden_states_flat / logprob_temperature


def compute_per_token_ce(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int,
    ce_mode: str,
    num_chunks: int = 8,
    tp_group: Optional[dist.ProcessGroup] = None,
    use_compile: bool = False,
    lm_head_fp32: bool = False,
    lm_head: Optional[torch.nn.Module] = None,
    logprob_temperature: LogprobTemperature = 1.0,
    logprob_top_k: LogprobTopK = TOP_K_ALL,
    logprob_top_p: LogprobProbability = 1.0,
    logprob_min_p: LogprobProbability = 0.0,
) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss based on the specified mode.

    Args:
        hidden_states_flat: Flattened hidden states, shape (BT, H)
        weight: LM head weight matrix, shape (V, H) or (V/tp, H) with TP
        labels_flat: Flattened labels, shape (BT,)
        ignore_index: Index to ignore in loss computation
        ce_mode: Cross-entropy computation mode ("compiled", "eager", or
            "fused_quack"). "fused_quack" uses chunked cuBLAS matmul + a fused
            CuTeDSL cross-entropy reduction, keeping the logits tile chunk-sized
            and reducing only three [N] scalar vectors across TP; it serves both
            the TP and non-TP cases.
        num_chunks: Number of chunks for compiled mode
        tp_group: TP process group for vocab-parallel cross-entropy (default: None)
        use_compile: Whether to use torch.compile in vocab_parallel_cross_entropy
        lm_head: Optional module that owns the loss projection. Used by FP8
            training so ``FP8Linear.forward`` is not bypassed by raw-weight CE,
            and by the internal GLM-5.2 exact active-LoRA selected-logprob op.
        lm_head_fp32: Compute the lm_head logits in FP32. This takes PRECEDENCE
            over ``lm_head`` — when set, the FP8 lm_head module is bypassed and
            logits are computed in FP32 from the (master) ``weight``, so an FP8
            lm_head does not catastrophically mis-score rare near-certain tokens.
        logprob_temperature: Temperature applied before the selected-token
            logprob/CE calculation. ``1.0`` preserves raw model logprobs; values
            such as a rollout temperature of ``0.7`` compute behavior-policy
            logprobs matching ``log_softmax(logits / temperature)``. Exact heads
            also accept a contiguous per-row FP32 tensor.

    Returns:
        per_token_ce: Per-token cross-entropy loss, shape (BT,)
    """
    rows = hidden_states_flat.shape[0]
    logprob_temperature = _flatten_row_metadata(
        logprob_temperature,
        rows=rows,
        name="logprob_temperature",
    )
    logprob_top_k = _flatten_row_metadata(logprob_top_k, rows=rows, name="logprob_top_k")
    logprob_top_p = _flatten_row_metadata(logprob_top_p, rows=rows, name="logprob_top_p")
    logprob_min_p = _flatten_row_metadata(logprob_min_p, rows=rows, name="logprob_min_p")
    logprob_temperature = normalize_logprob_temperature(
        logprob_temperature,
        rows=rows,
        device=hidden_states_flat.device,
    )
    logprob_top_ks, logprob_top_ps, logprob_min_ps = normalize_exact_sampling_transforms(
        logprob_top_k,
        logprob_top_p,
        logprob_min_p,
        rows=rows,
        device=hidden_states_flat.device,
    )
    has_sampling_filter = logprob_top_ks is not None

    # The complete GLM-5.2 active-LoRA lane owns the selected-logprob value and
    # its hybrid VJP. Route before the generic lm_head_fp32/module precedence:
    # the exact op intentionally consumes BF16 hidden/base operands while
    # producing FP32 logits and logprobs.
    if lm_head is not None and getattr(lm_head, "_glm52_exact_tp16_lm_head", False):
        from xorl.models.transformers.glm5.exact_lm_head_qlora import (  # noqa: PLC0415
            glm52_exact_lm_head_per_token_ce,
        )

        return glm52_exact_lm_head_per_token_ce(
            hidden_states_flat,
            weight,
            labels_flat,
            lm_head=lm_head,
            ignore_index=ignore_index,
            ce_mode=ce_mode,
            lm_head_fp32=lm_head_fp32,
            logprob_temperature=logprob_temperature,
            logprob_top_ks=logprob_top_ks,
            logprob_top_ps=logprob_top_ps,
            logprob_min_ps=logprob_min_ps,
            tp_group=tp_group,
        )

    if lm_head is not None and getattr(lm_head, "_dsv4_exact_tp8_lm_head", False):
        from xorl.models.transformers.deepseek_v4.exact_lm_head import (  # noqa: PLC0415
            dsv4_exact_lm_head_per_token_ce,
        )

        return dsv4_exact_lm_head_per_token_ce(
            hidden_states_flat,
            weight,
            labels_flat,
            lm_head=lm_head,
            ignore_index=ignore_index,
            ce_mode=ce_mode,
            lm_head_fp32=lm_head_fp32,
            logprob_temperature=logprob_temperature,
            logprob_top_ks=logprob_top_ks,
            logprob_top_ps=logprob_top_ps,
            logprob_min_ps=logprob_min_ps,
            tp_group=tp_group,
        )

    # ``lm_head_fp32`` takes precedence over the FP8 lm_head module: an FP32
    # lm_head means the projection must NOT be FP8-quantized, so route to the
    # raw-weight FP32 path below rather than calling ``FP8Linear.forward``. The
    # passed ``weight`` is the master (non-quantized) lm_head weight.
    use_lm_head_module = lm_head is not None and not lm_head_fp32
    # ``fused_quack`` is an explicit opt-in that fuses the selected-token logprob
    # via chunked cuBLAS matmul + a fused CuTeDSL cross-entropy reduction
    # (chunk-sized logits, scalar TP reductions); it serves TP and non-TP cases.
    if ce_mode == "fused_quack":
        if has_sampling_filter:
            raise NotImplementedError("top-k/top-p/min-p replay is supported only by exact LM-head modes")
        if isinstance(logprob_temperature, torch.Tensor):
            raise NotImplementedError("per-row logprob_temperature is supported only by exact LM-head modes")
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        hidden_for_ce = hidden_states_flat
        if lm_head_fp32:
            hidden_for_ce = hidden_for_ce.float()
            local_weight = local_weight.float()
        hidden_for_ce = _scale_hidden_for_temperature(hidden_for_ce, logprob_temperature)
        return fused_selected_logprob_ce(
            hidden_for_ce,
            local_weight,
            labels_flat,
            tp_group=tp_group,
            ignore_index=ignore_index,
        )

    # ``bi_fused`` is the K3 lm-head contract (vendored identically in SGLang).
    # Hidden states stay bf16. Per-row temperature materializes the same FP32
    # transformed logits that serving samples and scores, unlike the
    # scale-hidden-pre-GEMM convention used by the other modes.
    if ce_mode == "bi_fused":
        from xorl.ops.loss.bi_fused_lm_head import (  # noqa: PLC0415
            bi_fused_per_token_ce,
            bi_fused_vocab_parallel_per_token_ce,
        )

        bi_fused_tp_groups = resolve_bi_fused_lm_head_tp_groups(ce_mode, tp_group, lm_head)
        if use_lm_head_module:
            raise NotImplementedError("ce_mode='bi_fused' does not support FP8 lm_head modules")
        if not lm_head_fp32:
            raise NotImplementedError(
                "ce_mode='bi_fused' implements the fp32-class lm-head contract; set lm_head_fp32: true"
            )
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        if bi_fused_tp_groups is not None:
            return bi_fused_vocab_parallel_per_token_ce(
                hidden_states_flat,
                local_weight,
                labels_flat,
                tp_group,
                ignore_index,
                temperature=logprob_temperature,
                top_ks=logprob_top_ks,
                top_ps=logprob_top_ps,
                min_ps=logprob_min_ps,
            )
        if tp_group is not None:
            raise NotImplementedError(
                "ce_mode='bi_fused' supports TP only through the dedicated vocabulary-sharded LM-head TP path"
            )
        return bi_fused_per_token_ce(
            hidden_states_flat,
            local_weight,
            labels_flat,
            ignore_index,
            temperature=logprob_temperature,
            top_ks=logprob_top_ks,
            top_ps=logprob_top_ps,
            min_ps=logprob_min_ps,
        )

    if has_sampling_filter:
        raise NotImplementedError("top-k/top-p/min-p replay is supported only by exact LM-head modes")

    if tp_group is not None:
        if isinstance(logprob_temperature, torch.Tensor):
            raise NotImplementedError("per-row logprob_temperature is supported only by exact LM-head modes")
        if use_lm_head_module:
            if logprob_temperature != 1.0:
                raise NotImplementedError("logprob_temperature with tensor-parallel lm_head modules is not supported")
            return vocab_parallel_cross_entropy_with_lm_head(
                hidden_states_flat,
                lm_head,
                labels_flat,
                tp_group,
                ignore_index=ignore_index,
                num_chunks=num_chunks,
                use_compile=use_compile,
            )
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        hidden_for_ce = hidden_states_flat
        if lm_head_fp32:
            local_weight = local_weight.float()
            hidden_for_ce = hidden_for_ce.float()
        hidden_for_ce = _scale_hidden_for_temperature(hidden_for_ce, logprob_temperature)
        return vocab_parallel_cross_entropy(
            hidden_for_ce,
            local_weight,
            labels_flat,
            tp_group,
            ignore_index=ignore_index,
            num_chunks=num_chunks,
            use_compile=use_compile,
        )

    if use_lm_head_module:
        if isinstance(logprob_temperature, torch.Tensor):
            raise NotImplementedError("per-row logprob_temperature is supported only by exact LM-head modes")
        return _module_lm_head_ce(
            hidden_states_flat,
            labels_flat,
            lm_head=lm_head,
            ignore_index=ignore_index,
            num_chunks=num_chunks,
            logprob_temperature=logprob_temperature,
        )

    if isinstance(logprob_temperature, torch.Tensor):
        raise NotImplementedError("per-row logprob_temperature is supported only by exact LM-head modes")
    hidden_for_ce = hidden_states_flat.float() if lm_head_fp32 else hidden_states_flat
    hidden_for_ce = _scale_hidden_for_temperature(hidden_for_ce, logprob_temperature)
    if ce_mode == "compiled":
        return compiled_cross_entropy_function(
            hidden_for_ce,
            weight,
            labels_flat,
            ignore_index,
            num_chunks,
            lm_head_fp32=lm_head_fp32,
        )
    else:
        # eager mode
        if lm_head_fp32:
            logits_flat = (hidden_states_flat.float() @ weight.float().t()).float()
        else:
            logits_flat = (hidden_states_flat @ weight.t()).float()
        if logprob_temperature != 1.0:
            logits_flat = logits_flat / logprob_temperature
        return F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)
