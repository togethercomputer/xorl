"""MoE expert weight container with backend dispatch."""

import hashlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from ..activations import ACT2FN
from .backend import (
    EP_COMBINE,
    EP_DISPATCH,
    EP_EXPERT_COMPUTE,
    EP_EXPERT_COMPUTE_MOE_ACT,
    MOE_EXPERT_BACKENDS,
)
from .common import split_gate_up_proj


_SG_LANG_MOE_TP_SIM_ENV = "XORL_SGLANG_MOE_TP_SIM"


def _flag_enabled(name: str) -> bool:
    v = os.environ.get(name, "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


_DEBUG_EP = _flag_enabled("XORL_DEBUG_EP")
_FORCE_SYNC = _flag_enabled("XORL_EP_FORCE_SYNC")
_FORCE_QUACK_DEEPEP_GENERIC = _flag_enabled("XORL_QUACK_DEEPEP_FORCE_GENERIC")
_DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS: dict[int, int] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _deepep_parity_diagnostic_enabled() -> bool:
    return _flag_enabled("XORL_DEEPEP_PARITY_DIAGNOSTIC")


def _deepep_parity_reference_compare_enabled() -> bool:
    return _flag_enabled("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE")


def _deepep_parity_all_ranks_requested() -> bool:
    raw = os.environ.get("XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS", "0").strip().lower()
    return raw in {"all", "*"}


def _distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _ep_group_rank(ep_group) -> int:
    if dist.is_available() and dist.is_initialized() and ep_group is not None:
        return dist.get_rank(ep_group)
    return 0


def _ep_group_size(ep_group) -> int:
    if ep_group is None:
        return 1
    size = getattr(ep_group, "size", None)
    if callable(size):
        return int(size())
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size(ep_group))
    return 1


def _deepep_parity_rank_allowed(rank: int) -> bool:
    raw = os.environ.get("XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS", "0").strip().lower()
    if raw in {"all", "*"}:
        return True
    allowed = {part.strip() for part in raw.split(",") if part.strip()}
    return str(rank) in allowed


def _acquire_deepep_parity_diagnostic_record() -> int | None:
    if not _deepep_parity_diagnostic_enabled():
        return None
    rank = _distributed_rank()
    if not _deepep_parity_rank_allowed(rank):
        return None
    record_start = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START", 0))
    max_records = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS", 8))
    count = _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS.get(rank, 0)
    _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS[rank] = count + 1
    if count < record_start or count >= record_start + max_records:
        return None
    return count


def _tensor_summary(tensor: torch.Tensor | None, *, include_sample: bool = True) -> dict | None:
    if tensor is None:
        return None
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
    }
    if not include_sample or tensor.numel() == 0:
        return summary

    max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
    fingerprint = _tensor_fingerprint(tensor)
    if fingerprint is not None:
        summary["fingerprint"] = fingerprint
    if max_values == 0:
        return summary

    with torch.no_grad():
        flat = tensor.detach().reshape(-1)[:max_values]
        try:
            summary["sample"] = flat.to(torch.float32).cpu().tolist()
        except (RuntimeError, TypeError):
            summary["sample"] = [str(v) for v in flat.cpu().tolist()]
    return summary


def _tensor_fingerprint(tensor: torch.Tensor) -> dict | None:
    max_elems = _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_FINGERPRINT_MAX_ELEMS", 0)
    if max_elems == 0 or tensor.numel() == 0:
        return None
    elem_count = int(tensor.numel()) if max_elems < 0 else min(int(tensor.numel()), max_elems)
    if elem_count <= 0:
        return None
    with torch.no_grad():
        values = tensor.detach().reshape(-1)[:elem_count].to(torch.float32).contiguous().cpu()
        finite = torch.isfinite(values)
        finite_values = values[finite]
        payload = values.numpy().tobytes()
        summary = {
            "numel": elem_count,
            "dtype_for_hash": "float32",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "finite_count": int(finite.sum().item()),
        }
        if finite_values.numel() > 0:
            summary.update(
                {
                    "sum": float(finite_values.sum().item()),
                    "abs_sum": float(finite_values.abs().sum().item()),
                    "max_abs": float(finite_values.abs().max().item()),
                }
            )
        return summary


def _cumsum_summary(cumsum: torch.Tensor | None) -> dict | None:
    if cumsum is None:
        return None
    summary = _tensor_summary(cumsum)
    with torch.no_grad():
        values = cumsum.detach().to(torch.int64).cpu()
        counts = torch.diff(torch.cat([values.new_zeros(1), values]))
        max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
        summary["last"] = int(values[-1].item()) if values.numel() > 0 else 0
        summary["counts_head"] = counts[:max_values].tolist()
        summary["counts_tail"] = counts[-max_values:].tolist() if max_values else []
        if counts.numel() > 0:
            k = min(max(1, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK", 8)), counts.numel())
            top_counts, top_indices = torch.topk(counts, k=k)
            summary["top_counts"] = [
                {"local_expert": int(idx.item()), "count": int(count.item())}
                for count, idx in zip(top_counts, top_indices)
                if int(count.item()) != 0
            ]
    return summary


def _selected_experts_summary(selected_experts: torch.Tensor | None, num_experts: int) -> dict | None:
    if selected_experts is None:
        return None
    summary = _tensor_summary(selected_experts)
    with torch.no_grad():
        flat = selected_experts.detach().reshape(-1).to(torch.int64)
        valid_mask = (flat >= 0) & (flat < num_experts)
        valid = flat[valid_mask]
        summary["invalid_count"] = int((~valid_mask).sum().item())
        summary["valid_count"] = int(valid.numel())
        if valid.numel() > 0:
            counts = torch.bincount(valid, minlength=num_experts)
            k = min(max(1, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK", 8)), num_experts)
            top_counts, top_indices = torch.topk(counts, k=k)
            summary["top_global_experts"] = [
                {"expert": int(idx.item()), "count": int(count.item())}
                for count, idx in zip(top_counts, top_indices)
                if int(count.item()) != 0
            ]
        else:
            summary["top_global_experts"] = []
    return summary


def _reference_compare_dtype(fallback: torch.dtype) -> torch.dtype:
    raw = os.environ.get("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE", "fp32").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"compute", "input", "original"}:
        return fallback
    return torch.float32


def _expert_counts_from_cumsum(cumsum: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    cumsum_i64 = cumsum.detach().to(torch.int64)
    if cumsum_i64.numel() != num_local_experts:
        return cumsum_i64.new_empty(0)
    return torch.diff(torch.cat([cumsum_i64.new_zeros(1), cumsum_i64]))


def _diff_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f32 = actual.detach().to(torch.float32)
    expected_f32 = expected.detach().to(torch.float32)
    diff = (actual_f32 - expected_f32).abs()
    finite_mask = torch.isfinite(diff)
    finite_diff = diff[finite_mask]
    summary = {
        "shape": list(actual.shape),
        "compared_elements": int(diff.numel()),
        "nonfinite_diff_count": int((~finite_mask).sum().item()),
    }
    if finite_diff.numel() == 0:
        summary.update({"max_abs": None, "mean_abs": None, "p95_abs": None, "nonzero_count": 0, "top_diffs": []})
        return summary

    p95_values = finite_diff
    p95_max_elems = max(1, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_P95_MAX_ELEMS", 1048576))
    p95_sampled = int(finite_diff.numel()) > p95_max_elems
    if p95_sampled:
        indices = _evenly_spaced_int64_indices(
            int(finite_diff.numel()),
            p95_max_elems,
            finite_diff.device,
        )
        p95_values = finite_diff.index_select(0, indices)

    summary.update(
        {
            "max_abs": float(finite_diff.max().item()),
            "mean_abs": float(finite_diff.mean().item()),
            "p95_abs": float(torch.quantile(p95_values, 0.95).item()),
            "p95_sampled": p95_sampled,
            "p95_sample_size": int(p95_values.numel()),
            "nonzero_count": int((finite_diff != 0).sum().item()),
        }
    )
    max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
    if max_values == 0 or diff.numel() == 0:
        summary["top_diffs"] = []
        return summary

    flat_diff = diff.reshape(-1)
    k = min(max_values, int(flat_diff.numel()))
    values, indices = torch.topk(flat_diff, k=k)
    width = actual.shape[-1] if actual.ndim > 0 and actual.shape[-1] else 1
    actual_flat = actual_f32.reshape(-1)
    expected_flat = expected_f32.reshape(-1)
    summary["top_diffs"] = [
        {
            "flat_index": int(index.item()),
            "row": int(index.item() // width),
            "col": int(index.item() % width),
            "abs": float(value.item()),
            "actual": float(actual_flat[index].item()),
            "expected": float(expected_flat[index].item()),
        }
        for value, index in zip(values, indices)
    ]
    return summary


def _evenly_spaced_int64_indices(numel: int, sample_size: int, device: torch.device) -> torch.Tensor:
    if sample_size <= 1:
        return torch.zeros(max(0, sample_size), dtype=torch.long, device=device)
    positions = torch.arange(sample_size, dtype=torch.long, device=device)
    return positions.mul(numel - 1).div(sample_size - 1, rounding_mode="floor")


def _reference_status(diff: dict) -> tuple[str, list[str], dict]:
    thresholds = {
        "max_abs": _env_float_or_none("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ABS"),
        "mean_abs": _env_float_or_none("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MEAN_ABS"),
    }
    active_thresholds = {key: value for key, value in thresholds.items() if value is not None}
    exceeded = [
        key for key, value in active_thresholds.items() if diff.get(key) is not None and float(diff[key]) > float(value)
    ]
    if exceeded:
        return "failed", exceeded, active_thresholds
    return ("pass" if active_thresholds else "observed"), [], active_thresholds


def _expert_output_reference_comparison(
    *,
    permute_tokens: torch.Tensor | None,
    cumsum: torch.Tensor | None,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    intermediate_size: int,
    expert_scores: torch.Tensor | None,
    hidden_act: str,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    expert_output: torch.Tensor | None,
) -> dict | None:
    if not _deepep_parity_reference_compare_enabled():
        return None
    if permute_tokens is None or cumsum is None or expert_output is None:
        return {"status": "skipped", "reason": "missing_tensor"}
    if permute_tokens.ndim != 2 or expert_output.ndim != 2:
        return {"status": "skipped", "reason": "unsupported_rank"}

    from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

    with torch.no_grad():
        num_local_experts = int(gate_up_proj.shape[0])
        counts = _expert_counts_from_cumsum(cumsum, num_local_experts)
        if counts.numel() != num_local_experts:
            return {
                "status": "skipped",
                "reason": "cumsum_length_mismatch",
                "cumsum_length": int(cumsum.numel()),
                "num_local_experts": num_local_experts,
            }
        total_rows = int(expert_output.shape[0])
        cumsum_last = int(cumsum.detach().to(torch.int64)[-1].item()) if cumsum.numel() > 0 else 0
        if cumsum_last != total_rows or int(permute_tokens.shape[0]) != total_rows:
            return {
                "status": "skipped",
                "reason": "row_count_mismatch",
                "cumsum_last": cumsum_last,
                "permute_tokens_rows": int(permute_tokens.shape[0]),
                "expert_output_rows": total_rows,
            }

        max_rows = _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS", 0)
        compare_rows = total_rows if max_rows <= 0 else min(total_rows, max_rows)
        row_sampled = compare_rows != total_rows
        if row_sampled:
            row_indices = _evenly_spaced_int64_indices(total_rows, compare_rows, expert_output.device)
        else:
            row_indices = torch.arange(total_rows, dtype=torch.long, device=expert_output.device)
        reference_dtype = _reference_compare_dtype(permute_tokens.dtype)
        reference = torch.empty(
            (compare_rows, int(expert_output.shape[1])),
            dtype=reference_dtype,
            device=expert_output.device,
        )

        start = 0
        for expert_idx, count_tensor in enumerate(counts):
            end = start + int(count_tensor.item())
            mask = (row_indices >= start) & (row_indices < end)
            if bool(mask.any().item()):
                selected_rows = row_indices[mask]
                tokens = permute_tokens.index_select(0, selected_rows).detach().to(reference_dtype)
                gate_up = tokens.matmul(gate_up_proj[expert_idx].detach().to(reference_dtype))
                if gate_up_bias is not None:
                    gate_up = gate_up + gate_up_bias[expert_idx].detach().to(reference_dtype)
                gate, up = gate_up.split(intermediate_size, dim=-1)
                activated = apply_moe_activation(hidden_act, gate, up)
                out = activated.matmul(down_proj[expert_idx].detach().to(reference_dtype))
                if down_bias is not None:
                    out = out + down_bias[expert_idx].detach().to(reference_dtype)
                if expert_scores is not None:
                    scores = expert_scores.index_select(0, selected_rows).detach().to(reference_dtype).unsqueeze(-1)
                    out = out * scores
                reference[mask] = out
            start = end
            if start >= total_rows:
                break

        actual = expert_output.index_select(0, row_indices)
        diff = _diff_summary(actual, reference)
        status, exceeded, thresholds = _reference_status(diff)
        max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
        return {
            "status": status,
            "reference_dtype": str(reference_dtype).replace("torch.", ""),
            "compare_rows": compare_rows,
            "total_rows": total_rows,
            "row_limited": row_sampled,
            "row_sample_strategy": "evenly_spaced" if row_sampled else "all",
            "row_indices_head": row_indices[:max_values].detach().cpu().tolist() if max_values else [],
            "row_indices_tail": row_indices[-max_values:].detach().cpu().tolist() if max_values else [],
            "thresholds": thresholds,
            "thresholds_exceeded": exceeded,
            "diff": diff,
            "reference": _tensor_summary(reference),
        }


def _safe_expert_output_reference_comparison(**kwargs) -> dict | None:
    try:
        return _expert_output_reference_comparison(**kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash training.
        return {"status": "error", "reason": type(exc).__name__, "message": str(exc)}


def _result_reference_comparison(
    *,
    hidden_states: torch.Tensor | None,
    routing_weights: torch.Tensor | None,
    selected_experts: torch.Tensor | None,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    intermediate_size: int,
    hidden_act: str,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    result: torch.Tensor | None,
    ep_group,
) -> dict | None:
    if not _deepep_parity_reference_compare_enabled():
        return None
    if hidden_states is None or routing_weights is None or selected_experts is None or result is None:
        return {"status": "skipped", "reason": "missing_tensor"}
    if hidden_states.ndim < 2 or result.ndim < 2 or selected_experts.ndim < 2 or routing_weights.ndim < 2:
        return {"status": "skipped", "reason": "unsupported_rank"}

    ep_size = _ep_group_size(ep_group)
    if ep_size > 1 and not _deepep_parity_all_ranks_requested():
        return {"status": "skipped", "reason": "requires_all_ranks"}
    if ep_size > 1 and not (dist.is_available() and dist.is_initialized()):
        return {"status": "skipped", "reason": "distributed_not_initialized"}

    from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

    with torch.no_grad():
        hidden_flat = hidden_states.detach().reshape(-1, int(hidden_states.shape[-1]))
        result_flat = result.detach().reshape(-1, int(result.shape[-1]))
        selected_flat = selected_experts.detach().reshape(-1, int(selected_experts.shape[-1]))
        routing_flat = routing_weights.detach().reshape(-1, int(routing_weights.shape[-1]))
        total_rows = int(result_flat.shape[0])
        if (
            hidden_flat.shape[0] != total_rows
            or selected_flat.shape[0] != total_rows
            or routing_flat.shape[0] != total_rows
            or hidden_flat.shape[-1] != result_flat.shape[-1]
            or selected_flat.shape != routing_flat.shape
        ):
            return {
                "status": "skipped",
                "reason": "shape_mismatch",
                "hidden_rows": int(hidden_flat.shape[0]),
                "result_rows": total_rows,
                "selected_shape": list(selected_flat.shape),
                "routing_shape": list(routing_flat.shape),
            }

        max_rows = _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS", 0)
        compare_rows = total_rows if max_rows <= 0 else min(total_rows, max_rows)
        row_sampled = compare_rows != total_rows
        if row_sampled:
            row_indices = _evenly_spaced_int64_indices(total_rows, compare_rows, result_flat.device)
        else:
            row_indices = torch.arange(total_rows, dtype=torch.long, device=result_flat.device)

        reference_dtype = _reference_compare_dtype(hidden_flat.dtype)
        tokens = hidden_flat.index_select(0, row_indices).to(reference_dtype)
        selected = selected_flat.index_select(0, row_indices).to(torch.long)
        routing = routing_flat.index_select(0, row_indices).to(reference_dtype)
        reference = torch.zeros(
            (compare_rows, int(result_flat.shape[-1])),
            dtype=reference_dtype,
            device=result_flat.device,
        )

        num_local_experts = int(gate_up_proj.shape[0])
        ep_rank = _ep_group_rank(ep_group)
        local_start = ep_rank * num_local_experts
        local_end = local_start + num_local_experts
        local_contribution_count = 0
        local_expert_hit_count = 0
        for local_expert_idx, global_expert_idx in enumerate(range(local_start, local_end)):
            mask = selected == global_expert_idx
            if not bool(mask.any().item()):
                continue
            pair_rows, pair_topk = mask.nonzero(as_tuple=True)
            local_expert_hit_count += 1
            local_contribution_count += int(pair_rows.numel())
            expert_tokens = tokens.index_select(0, pair_rows)
            gate_up = expert_tokens.matmul(gate_up_proj[local_expert_idx].detach().to(reference_dtype))
            if gate_up_bias is not None:
                gate_up = gate_up + gate_up_bias[local_expert_idx].detach().to(reference_dtype)
            gate, up = gate_up.split(intermediate_size, dim=-1)
            activated = apply_moe_activation(hidden_act, gate, up)
            out = activated.matmul(down_proj[local_expert_idx].detach().to(reference_dtype))
            if down_bias is not None:
                out = out + down_bias[local_expert_idx].detach().to(reference_dtype)
            out = out * routing[pair_rows, pair_topk].unsqueeze(-1)
            reference.index_add_(0, pair_rows, out)

        global_contribution_count = local_contribution_count
        if ep_size > 1:
            contribution_count = torch.tensor([local_contribution_count], dtype=torch.long, device=result_flat.device)
            dist.all_reduce(reference, op=dist.ReduceOp.SUM, group=ep_group)
            dist.all_reduce(contribution_count, op=dist.ReduceOp.SUM, group=ep_group)
            global_contribution_count = int(contribution_count.item())

        actual = result_flat.index_select(0, row_indices)
        diff = _diff_summary(actual, reference)
        status, exceeded, thresholds = _reference_status(diff)
        max_values = max(0, _env_int("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", 8))
        return {
            "status": status,
            "reference_dtype": str(reference_dtype).replace("torch.", ""),
            "compare_rows": compare_rows,
            "total_rows": total_rows,
            "row_limited": row_sampled,
            "row_sample_strategy": "evenly_spaced" if row_sampled else "all",
            "row_indices_head": row_indices[:max_values].detach().cpu().tolist() if max_values else [],
            "row_indices_tail": row_indices[-max_values:].detach().cpu().tolist() if max_values else [],
            "local_expert_global_range": [local_start, local_end],
            "local_expert_hit_count": local_expert_hit_count,
            "local_contribution_count": local_contribution_count,
            "global_contribution_count": global_contribution_count,
            "thresholds": thresholds,
            "thresholds_exceeded": exceeded,
            "diff": diff,
            "reference": _tensor_summary(reference),
        }


def _safe_result_reference_comparison(**kwargs) -> dict | None:
    try:
        return _result_reference_comparison(**kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash training.
        return {"status": "error", "reason": type(exc).__name__, "message": str(exc)}


def _dispatch_context_summary(ctx) -> dict | None:
    if ctx is None:
        return None
    return {
        "type": type(ctx).__name__,
        "handle_type": type(getattr(ctx, "handle", None)).__name__,
        "num_recv_tokens": getattr(ctx, "num_recv_tokens", None),
        "num_valid": getattr(ctx, "num_valid", None),
        "dtype": str(getattr(ctx, "dtype", "")).replace("torch.", ""),
        "hidden_dim": getattr(ctx, "hidden_dim", None),
        "input_splits": getattr(ctx, "input_splits", None),
        "output_splits": getattr(ctx, "output_splits", None),
        "orig_shape": list(getattr(ctx, "orig_shape", [])),
        "num_experts": getattr(ctx, "num_experts", None),
        "num_tokens_per_expert": _tensor_summary(getattr(ctx, "num_tokens_per_expert", None)),
        "routing_map": _tensor_summary(getattr(ctx, "routing_map", None)),
        "perm_mapping": _tensor_summary(getattr(ctx, "perm_mapping", None)),
        "permuted_indices": _tensor_summary(getattr(ctx, "permuted_indices", None)),
        "permuted_scores": _tensor_summary(getattr(ctx, "permuted_scores", None)),
        "expert_scores": _tensor_summary(getattr(ctx, "expert_scores", None)),
    }


class MoEExperts(nn.Module):
    """Unified weight container for MoE experts.

    Holds stacked weight tensors ``[num_experts, ...]`` and dispatches
    ``forward()`` to the selected backend (eager / triton / native / quack).

    Weights are stored in ``(G, K, N)`` format — ``[num_experts, in_features, out_features]``::

        gate_up_proj: [num_experts, hidden_dim, 2 * intermediate_size]
        down_proj:    [num_experts, intermediate_size, hidden_dim]

    ``gate_proj`` and ``up_proj`` are exposed as views into ``gate_up_proj``
    for compatibility with existing backends and helpers.

    With ``gated=False`` (e.g. Nemotron-3-Ultra latent experts) there is no
    gate branch: the activation is applied directly to the single first GEMM
    output. The checkpoint-visible parameter name stays ``gate_up_proj`` for
    layout/EP-plan stability, but it holds only the up projection at half
    width::

        gate_up_proj: [num_experts, hidden_dim, intermediate_size]

    ``hidden_dim`` is whatever input dimension the caller provides — it does
    not have to equal the model hidden size (Nemotron experts run in a latent
    dim).

    Optional per-expert biases (``gate_up_bias``, ``down_bias``) default to
    ``None`` and can be set by model-specific code (e.g. GPT-OSS).

    Args:
        num_experts: Total number of experts.
        hidden_dim: Expert input/output dimension (model hidden size or a
            latent dimension).
        intermediate_size: Expert FFN intermediate dimension.
        hidden_act: Activation function name (default: ``"silu"``).
        moe_implementation: Backend name — ``"eager"``, ``"triton"``, ``"native"``, or ``"quack"``.
        gated: Whether experts use a gated (GLU) first projection (default: True).
            Non-gated experts require an activation in ``UNGATED_HIDDEN_ACTS``
            (currently ``relu2``) and are not supported by the quack backend.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        moe_implementation: str = "triton",
        activation_native: bool = False,
        swiglu_limit: float = 0.0,
        gated: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.moe_implementation = moe_implementation
        self.activation_native = activation_native
        self.swiglu_limit = float(swiglu_limit)
        self.gated = gated

        if not gated and moe_implementation == "quack":
            raise NotImplementedError("quack backend does not support non-gated experts")

        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, (2 if gated else 1) * intermediate_size),
            requires_grad=True,
        )
        if gated:
            self.gate_up_proj._fused_gate_up = True
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[hidden_act]
        # String kind used by triton/native/quack backends (avoids name-sniffing).
        from xorl.ops.moe.activations import (  # noqa: PLC0415
            UNGATED_HIDDEN_ACTS,
            check_hidden_act_supported,
            normalize_hidden_act,
        )

        self.hidden_act = normalize_hidden_act(hidden_act)
        if not gated:
            check_hidden_act_supported(self.hidden_act, f"{moe_implementation} (non-gated)", UNGATED_HIDDEN_ACTS)
        self._moe_act: bool = False

        # Optional per-expert biases (e.g. GPT-OSS). Set to actual tensors
        # by model-specific code; None means no bias.
        self.gate_up_bias = None
        self.down_bias = None

        # EP dispatch strategy: "alltoall" (default) or "deepep" (NVLink-optimized)
        self.ep_dispatch: str = "alltoall"
        self.deepep_buffer_size_gb: float = 2.0
        self.deepep_num_sms: int = 20
        self.deepep_async_combine: bool = False
        self.fp8_training_enabled: bool = False
        self.fp8_training_grouped_backend: str = "triton_grouped"
        self.fp8_training_block_size: int = 128
        self.last_forward_used_fp8: bool = False
        self.alltoall_combine_hidden_chunk_size: int = 0

    @property
    def gate_proj(self) -> torch.Tensor:
        if not self.gated:
            raise AttributeError("non-gated MoEExperts has no gate_proj")
        gate_proj, _ = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        gate_proj.grad = (
            None if self.gate_up_proj.grad is None else self.gate_up_proj.grad[..., : self.intermediate_size]
        )
        return gate_proj

    @property
    def up_proj(self) -> torch.Tensor:
        if not self.gated:
            return self.gate_up_proj
        _, up_proj = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        up_proj.grad = None if self.gate_up_proj.grad is None else self.gate_up_proj.grad[..., self.intermediate_size :]
        return up_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        selected_experts: torch.Tensor = None,
        expert_idx: int = None,
    ) -> torch.Tensor:
        """Dispatch to the configured backend.

        For **triton/native/quack**: call with ``(hidden_states, routing_weights, selected_experts)``.
        For **eager**: called per-expert from ``MoEBlock._eager_forward()`` with ``expert_idx``.

        When Expert Parallelism is enabled, all backends
        use the unified dispatch → compute → combine path via ``_ep_forward()``.
        """
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        if parallel_state.ep_enabled:
            return self._ep_forward(hidden_states, routing_weights, selected_experts, parallel_state)
        self.last_forward_used_fp8 = False
        if self.fp8_training_enabled and self.moe_implementation != "quack":
            raise NotImplementedError("FP8 grouped MoE compute currently requires moe_implementation='quack'")

        if self.sglang_moe_tp_sim_enabled(parallel_state) and expert_idx is None:
            return self._sglang_moe_tp_sim_forward(hidden_states, routing_weights, selected_experts, parallel_state)

        if self.moe_implementation == "eager":
            fn = MOE_EXPERT_BACKENDS[self.moe_implementation]
            assert expert_idx is not None
            return fn(
                hidden_states,
                expert_idx,
                self.gate_proj.contiguous() if self.gated else None,
                self.up_proj.contiguous() if self.gated else self.gate_up_proj,
                self.down_proj,
                hidden_act=self.hidden_act,
                swiglu_limit=self.swiglu_limit,
                gate_up_bias=self.gate_up_bias,
                down_bias=self.down_bias,
                gated=self.gated,
            )

        # Local single-GPU path. Non-gated experts only use the fused
        # gate_up_proj reference (which holds the plain up projection).
        gate_proj = self.gate_proj.contiguous() if self.gated else None
        up_proj = self.up_proj.contiguous() if self.gated else None
        fn = MOE_EXPERT_BACKENDS[self.moe_implementation]
        self.last_forward_used_fp8 = bool(self.fp8_training_enabled)

        return fn(
            hidden_states,
            routing_weights,
            selected_experts,
            gate_proj,
            up_proj,
            self.down_proj,
            num_experts=self.num_experts,
            hidden_act=self.hidden_act,
            gate_up_proj=self.gate_up_proj,
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
            fp8_compute=self.fp8_training_enabled,
            fp8_grouped_backend=self.fp8_training_grouped_backend,
            fp8_block_size=self.fp8_training_block_size,
            activation_native=self.activation_native,
            gated=self.gated,
        )

    def sglang_moe_tp_sim_enabled(self, parallel_state=None) -> bool:
        """Whether to simulate SGLang's MoE TP expert shard/reduce order.

        This is a K3 parity diagnostic path for no-EP tensor-parallel Qwen3-MoE
        replays. xorl currently keeps MoE experts full-local under TP, while
        SGLang shards the expert intermediate dimension and all-reduces the
        hidden output. The simulation is intentionally env-gated and does not
        alter production MoE defaults.
        """
        if not _flag_enabled(_SG_LANG_MOE_TP_SIM_ENV):
            return False
        if parallel_state is None:
            from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

            parallel_state = get_parallel_state()
        if getattr(parallel_state, "ep_enabled", False):
            return False
        return max(1, int(getattr(parallel_state, "tp_size", 1))) > 1

    def _sglang_moe_tp_sim_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        mode = os.environ.get(_SG_LANG_MOE_TP_SIM_ENV, "0").strip().lower()
        if mode == "cache":
            return self._sglang_moe_tp_sim_cache_forward(hidden_states, routing_weights, selected_experts, parallel_state)
        if mode in {"triton", "backend"}:
            return self._sglang_moe_tp_sim_triton_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                parallel_state,
            )
        if mode in {"triton_sgl_reduce", "triton_sglang_reduce", "sglang_reduce"}:
            return self._sglang_moe_tp_sim_triton_sglang_reduce_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                parallel_state,
            )
        if mode in {"deep_gemm", "deepgemm", "dg"}:
            return self._sglang_moe_tp_sim_deep_gemm_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                parallel_state,
            )
        if mode in {"sglang", "sgl_kernel", "fused"}:
            return self._sglang_moe_tp_sim_sglang_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                parallel_state,
            )
        if mode in {"sglang_runner", "sgl_runner", "runner"}:
            return self._sglang_moe_tp_sim_sglang_runner_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                parallel_state,
            )
        return self._sglang_moe_tp_sim_direct_forward(hidden_states, routing_weights, selected_experts, parallel_state)

    def _validate_sglang_moe_tp_sim_inputs(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> tuple[int, int]:
        if routing_weights is None or selected_experts is None:
            raise ValueError(f"{_SG_LANG_MOE_TP_SIM_ENV}=1 requires routing_weights and selected_experts")
        if not self.gated:
            raise NotImplementedError(f"{_SG_LANG_MOE_TP_SIM_ENV}=1 currently supports gated MoE experts only")
        if self.down_bias is not None:
            raise NotImplementedError(f"{_SG_LANG_MOE_TP_SIM_ENV}=1 does not support down_bias")

        tp_size = max(1, int(getattr(parallel_state, "tp_size", 1)))
        if self.intermediate_size % tp_size != 0:
            raise ValueError(
                f"Cannot simulate SGLang MoE TP with intermediate_size={self.intermediate_size} "
                f"not divisible by tp_size={tp_size}"
            )
        return tp_size, self.intermediate_size // tp_size

    def _sglang_moe_tp_sim_direct_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Compute experts as TP-intermediate shards, then sum shard outputs.

        SGLang's unquantized Qwen3-MoE FusedMoE shards w1/w3 on the expert
        intermediate output dimension and w2 on the matching input dimension.
        Each TP rank computes a hidden-sized partial output and the model path
        all-reduces those partials. Since xorl has full expert tensors on every
        TP rank today, this parity path computes each shard locally and adds the
        shard outputs in rank order.
        """
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
        hidden_dim = int(hidden_flat.shape[-1])
        compute_dtype = torch.bfloat16 if hidden_flat.dtype in {torch.bfloat16, torch.float16} else hidden_flat.dtype

        from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

        output = hidden_flat.new_zeros(hidden_flat.shape[0], hidden_dim)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            shard_output = hidden_flat.new_zeros(hidden_flat.shape[0], hidden_dim)

            for expert_idx in range(self.num_experts):
                mask = selected_flat == expert_idx
                if not bool(mask.any().item()):
                    continue
                token_rows, topk_slots = mask.nonzero(as_tuple=True)
                tokens = hidden_flat.index_select(0, token_rows).to(compute_dtype)

                gate = tokens.matmul(self.gate_up_proj[expert_idx, :, start:end].to(compute_dtype))
                up = tokens.matmul(
                    self.gate_up_proj[
                        expert_idx,
                        :,
                        self.intermediate_size + start : self.intermediate_size + end,
                    ].to(compute_dtype)
                )
                if self.swiglu_limit > 0:
                    gate = gate.clamp(-self.swiglu_limit, self.swiglu_limit)

                activated = apply_moe_activation(self.hidden_act, gate, up)
                expert_out = activated.matmul(self.down_proj[expert_idx, start:end, :].to(compute_dtype))
                expert_out = expert_out * routing_flat[token_rows, topk_slots].to(expert_out.dtype).unsqueeze(-1)
                shard_output.index_add_(0, token_rows, expert_out.to(shard_output.dtype))

            output = output + shard_output

        return output.reshape(original_shape)

    def _sglang_moe_tp_sim_cache_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Diagnostic cache-order variant of the SGLang MoE TP simulation."""
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
        hidden_dim = int(hidden_flat.shape[-1])
        compute_dtype = torch.bfloat16 if hidden_flat.dtype in {torch.bfloat16, torch.float16} else hidden_flat.dtype

        from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

        output = hidden_flat.new_zeros(hidden_flat.shape[0], hidden_dim)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            topk = selected_flat.shape[1]
            num_assignments = hidden_flat.shape[0] * topk
            gate_up_cache = hidden_flat.new_zeros(num_assignments, 2 * shard_intermediate)
            down_cache = hidden_flat.new_zeros(hidden_flat.shape[0], topk, hidden_dim)

            for expert_idx in range(self.num_experts):
                mask = selected_flat == expert_idx
                if not bool(mask.any().item()):
                    continue
                token_rows, topk_slots = mask.nonzero(as_tuple=True)
                assignment_rows = token_rows * topk + topk_slots
                tokens = hidden_flat.index_select(0, token_rows).to(compute_dtype)

                gate = tokens.matmul(self.gate_up_proj[expert_idx, :, start:end].to(compute_dtype))
                up = tokens.matmul(
                    self.gate_up_proj[
                        expert_idx,
                        :,
                        self.intermediate_size + start : self.intermediate_size + end,
                    ].to(compute_dtype)
                )
                if self.gate_up_bias is not None:
                    gate = gate + self.gate_up_bias[expert_idx, start:end].to(compute_dtype)
                    up = up + self.gate_up_bias[
                        expert_idx,
                        self.intermediate_size + start : self.intermediate_size + end,
                    ].to(compute_dtype)

                gate_up_cache[assignment_rows, :shard_intermediate] = gate.to(gate_up_cache.dtype)
                gate_up_cache[assignment_rows, shard_intermediate:] = up.to(gate_up_cache.dtype)

            gate = gate_up_cache[:, :shard_intermediate].to(compute_dtype)
            up = gate_up_cache[:, shard_intermediate:].to(compute_dtype)
            if self.swiglu_limit > 0:
                gate = gate.clamp(-self.swiglu_limit, self.swiglu_limit)
            activation_cache = apply_moe_activation(self.hidden_act, gate, up).to(gate_up_cache.dtype)

            for expert_idx in range(self.num_experts):
                mask = selected_flat == expert_idx
                if not bool(mask.any().item()):
                    continue
                token_rows, topk_slots = mask.nonzero(as_tuple=True)
                assignment_rows = token_rows * topk + topk_slots

                activated = activation_cache.index_select(0, assignment_rows).to(compute_dtype)
                expert_out = activated.matmul(self.down_proj[expert_idx, start:end, :].to(compute_dtype))
                expert_out = expert_out * routing_flat[token_rows, topk_slots].to(expert_out.dtype).unsqueeze(-1)
                down_cache[token_rows, topk_slots, :] = expert_out.to(down_cache.dtype)

            shard_output = down_cache.to(torch.float32).sum(dim=1).to(down_cache.dtype)

            output = output + shard_output

        return output.reshape(original_shape)

    def _sglang_moe_tp_sim_triton_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Diagnostic TP-shard simulation using xorl's local Triton MoE backend."""
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)

        from xorl.ops.moe.triton import triton_moe_forward  # noqa: PLC0415

        output = hidden_flat.new_zeros(hidden_flat.shape)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            gate_proj = self.gate_up_proj[:, :, start:end].contiguous()
            up_proj = self.gate_up_proj[
                :,
                :,
                self.intermediate_size + start : self.intermediate_size + end,
            ].contiguous()
            gate_up_proj = torch.cat([gate_proj, up_proj], dim=-1).contiguous()
            down_proj = self.down_proj[:, start:end, :].contiguous()

            shard_output = triton_moe_forward(
                module=None,
                num_experts=self.num_experts,
                routing_weights=routing_flat,
                selected_experts=selected_flat,
                hidden_states=hidden_flat,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
                gate_up_proj=gate_up_proj,
                hidden_act=self.hidden_act,
                swiglu_limit=self.swiglu_limit,
                gated=self.gated,
            )
            output = output + shard_output.to(output.dtype)

        return output.reshape(original_shape)

    @staticmethod
    def _sglang_topk_sum_reduce(per_slot: torch.Tensor) -> torch.Tensor:
        """Mirror SGLang's top-k MoE output reduction for topk > 1.

        SGLang's deterministic Triton path accumulates the top-k slot outputs
        into fp32 in slot order and casts once to the output dtype. PyTorch
        ``sum(dim=1)`` can choose a different reduction tree, so keep this
        explicit for parity diagnostics.
        """
        topk = int(per_slot.shape[1])
        if topk == 1:
            return per_slot[:, 0, :]
        accumulator = per_slot[:, 0, :].to(torch.float32)
        for topk_idx in range(1, topk):
            accumulator = accumulator + per_slot[:, topk_idx, :].to(torch.float32)
        return accumulator.to(per_slot.dtype)

    def _sglang_moe_tp_sim_triton_sglang_reduce_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Use xorl grouped GEMMs but SGLang's deterministic top-k combine order."""
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)

        from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk  # noqa: PLC0415
        from xorl.ops.group_gemm.kernel.moe import (  # noqa: PLC0415
            expert_histogram,
            moe_index_compute,
            moe_scatter,
        )
        from xorl.ops.moe.activations import apply_moe_activation  # noqa: PLC0415

        splits = expert_histogram(selected_flat, self.num_experts)
        cumsum_t = torch.cumsum(splits, dim=0)
        scatter_index = moe_index_compute(selected_flat, cumsum_t)
        scatter_output = moe_scatter(hidden_flat, scatter_index)
        max_m = scatter_output.shape[0]

        output = hidden_flat.new_zeros(hidden_flat.shape)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            gate_proj = self.gate_up_proj[:, :, start:end].contiguous()
            up_proj = self.gate_up_proj[
                :,
                :,
                self.intermediate_size + start : self.intermediate_size + end,
            ].contiguous()
            gate_up_proj = torch.cat([gate_proj, up_proj], dim=-1).contiguous()
            down_proj = self.down_proj[:, start:end, :].contiguous()

            gate_up_output = group_gemm_same_nk(
                a=scatter_output,
                b=gate_up_proj,
                cumsum_M=cumsum_t,
                max_M=max_m,
            )
            gate, up = gate_up_output.split(shard_intermediate, dim=-1)
            if self.swiglu_limit > 0:
                gate = gate.clamp(-self.swiglu_limit, self.swiglu_limit)
            activated = apply_moe_activation(self.hidden_act, gate, up)

            down_output = group_gemm_same_nk(
                a=activated,
                b=down_proj,
                cumsum_M=cumsum_t,
                max_M=max_m,
            )
            per_slot = down_output[scatter_index.flatten()].reshape(
                hidden_flat.shape[0],
                selected_flat.shape[1],
                -1,
            )
            weighted = per_slot * routing_flat.to(per_slot.dtype).unsqueeze(-1)
            shard_output = self._sglang_topk_sum_reduce(weighted)
            output = output + shard_output.to(output.dtype)

        return output.reshape(original_shape)

    @staticmethod
    def _deep_gemm_group_gemm_same_nk(
        *,
        a: torch.Tensor,
        b: torch.Tensor,
        cumsum_M: torch.Tensor,
    ) -> torch.Tensor:
        """Run DeepGEMM BF16 contiguous grouped GEMM using xorl's compact layout.

        xorl's grouped-GEMM helpers keep expert rows compact. DeepGEMM's
        contiguous grouped layout requires each expert segment to be padded to
        the kernel's M alignment, with invalid pad rows marked by ``m_indices``.
        ``b`` follows xorl's [expert, K, N] layout and is transposed for
        DeepGEMM's NT contract [expert, N, K].
        """
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise RuntimeError("DeepGEMM BF16 grouped MoE diagnostic requires bf16 tensors")
        if not a.is_cuda or not b.is_cuda:
            raise RuntimeError("DeepGEMM BF16 grouped MoE diagnostic requires CUDA tensors")

        try:
            import deep_gemm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"{_SG_LANG_MOE_TP_SIM_ENV}=deep_gemm requires the optional deep_gemm package"
            ) from exc

        if a.shape[0] == 0:
            return a.new_empty((0, int(b.shape[2])))

        alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
        starts = torch.cat([cumsum_M.new_zeros(1), cumsum_M[:-1]]).detach().to(torch.int64).cpu().tolist()
        ends = cumsum_M.detach().to(torch.int64).cpu().tolist()
        chunks: list[torch.Tensor] = []
        m_indices: list[int] = []
        ranges: list[tuple[int, int, int, int]] = []
        padded_start = 0
        for expert_idx, (start, end) in enumerate(zip(starts, ends)):
            count = int(end) - int(start)
            if count <= 0:
                continue
            aligned_count = ((count + alignment - 1) // alignment) * alignment
            chunk = a[int(start) : int(end)]
            if aligned_count != count:
                padded = a.new_zeros((aligned_count, int(a.shape[1])))
                padded[:count].copy_(chunk)
                chunk = padded
            chunks.append(chunk.contiguous())
            m_indices.extend([expert_idx] * count)
            m_indices.extend([-1] * (aligned_count - count))
            ranges.append((int(start), int(end), padded_start, padded_start + count))
            padded_start += aligned_count

        out = a.new_empty((int(a.shape[0]), int(b.shape[2])))
        if not chunks:
            return out

        padded_a = torch.cat(chunks, dim=0).contiguous()
        padded_out = a.new_empty((int(padded_a.shape[0]), int(b.shape[2])))
        m_indices_tensor = torch.tensor(m_indices, dtype=torch.int32, device=a.device)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            padded_a,
            b.transpose(1, 2).contiguous(),
            padded_out,
            m_indices_tensor,
        )
        for start, end, padded_valid_start, padded_valid_end in ranges:
            out[start:end].copy_(padded_out[padded_valid_start:padded_valid_end])
        return out

    @staticmethod
    def _sglang_stable_moe_slot_order(
        selected_experts: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return SGLang's stable expert-major compact slot order.

        SGLang's BF16 DeepGEMM diagnostic compacts flattened token/top-k slots
        by stable-sorting on expert id. xorl's generic Triton scatter index uses
        relaxed atomic reservations, which can place the same valid slot at a
        different compact M row. DeepGEMM output can vary at one-ulp scale with
        that row placement, so the parity diagnostic must use SGLang's order.
        """
        flat_experts = selected_experts.reshape(-1).to(torch.int64)
        valid_mask = (flat_experts >= 0) & (flat_experts < int(num_experts))
        if not bool(valid_mask.any().item()):
            counts = torch.zeros(int(num_experts), dtype=torch.int32, device=selected_experts.device)
            return torch.empty(0, dtype=torch.long, device=selected_experts.device), counts

        valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
        sort_keys = flat_experts.index_select(0, valid_positions)
        try:
            sort_relative = torch.argsort(sort_keys, stable=True)
        except TypeError:
            sort_relative = torch.argsort(sort_keys)
        slot_order = valid_positions.index_select(0, sort_relative)
        sorted_experts = flat_experts.index_select(0, slot_order)
        counts = torch.bincount(sorted_experts, minlength=int(num_experts)).to(dtype=torch.int32)
        cumsum_m = torch.cumsum(counts, dim=0).to(dtype=torch.int32)
        return slot_order, cumsum_m

    def _sglang_moe_tp_sim_deep_gemm_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Diagnostic TP-shard simulation using DeepGEMM BF16 grouped GEMM."""
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        if self.gate_up_bias is not None:
            raise NotImplementedError(f"{_SG_LANG_MOE_TP_SIM_ENV}=deep_gemm does not support gate_up_bias")
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)

        from xorl.ops.moe.activations import apply_sglang_moe_activation  # noqa: PLC0415

        topk = int(selected_flat.shape[1])
        slot_order, cumsum_t = self._sglang_stable_moe_slot_order(selected_flat, self.num_experts)
        if slot_order.numel() == 0:
            return hidden_flat.new_zeros(hidden_flat.shape).reshape(original_shape)

        hidden_slots = hidden_flat.repeat_interleave(topk, dim=0)
        scatter_output = hidden_slots.index_select(0, slot_order).contiguous()

        output = hidden_flat.new_zeros(hidden_flat.shape)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            gate_proj = self.gate_up_proj[:, :, start:end].contiguous()
            up_proj = self.gate_up_proj[
                :,
                :,
                self.intermediate_size + start : self.intermediate_size + end,
            ].contiguous()
            gate_up_proj = torch.cat([gate_proj, up_proj], dim=-1).contiguous()
            down_proj = self.down_proj[:, start:end, :].contiguous()

            gate_up_output = self._deep_gemm_group_gemm_same_nk(
                a=scatter_output,
                b=gate_up_proj,
                cumsum_M=cumsum_t,
            )
            gate, up = gate_up_output.split(shard_intermediate, dim=-1)
            if self.swiglu_limit > 0:
                gate = gate.clamp(-self.swiglu_limit, self.swiglu_limit)
            activated = apply_sglang_moe_activation(self.hidden_act, gate, up).contiguous()

            down_output = self._deep_gemm_group_gemm_same_nk(
                a=activated,
                b=down_proj,
                cumsum_M=cumsum_t,
            )
            per_slot_flat = hidden_flat.new_zeros((hidden_flat.shape[0] * topk, hidden_flat.shape[-1]))
            per_slot_flat.index_copy_(0, slot_order, down_output)
            per_slot = per_slot_flat.reshape(hidden_flat.shape[0], topk, -1)
            weighted = per_slot * routing_flat.to(per_slot.dtype).unsqueeze(-1)
            shard_output = self._sglang_topk_sum_reduce(weighted)
            output = output + shard_output.to(output.dtype)

        return output.reshape(original_shape)

    @staticmethod
    def _ensure_sglang_server_args() -> None:
        try:
            from sglang.srt.server_args import (  # noqa: PLC0415
                ServerArgs,
                get_global_server_args,
                set_global_server_args_for_scheduler,
            )
        except ImportError as exc:
            raise ImportError(
                f"{_SG_LANG_MOE_TP_SIM_ENV}=sglang requires an environment with sglang and sgl_kernel installed"
            ) from exc

        try:
            get_global_server_args()
        except ValueError:
            server_args = ServerArgs(model_path=os.environ.get("XORL_SGLANG_MOE_MODEL_PATH", "dummy"))
            server_args.enable_deterministic_inference = True
            server_args.enable_fused_moe_sum_all_reduce = False
            server_args.rl_on_policy_target = "xorl-batch-invariant"
            set_global_server_args_for_scheduler(server_args)

    @staticmethod
    def _load_sglang_fused_experts_impl():
        try:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"{_SG_LANG_MOE_TP_SIM_ENV}=sglang requires an environment with sglang and sgl_kernel installed"
            ) from exc

        MoEExperts._ensure_sglang_server_args()

        return fused_experts_impl

    @staticmethod
    def _load_sglang_moe_runner_stack():
        try:
            from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig  # noqa: PLC0415
            from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # noqa: PLC0415
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput  # noqa: PLC0415
            from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: PLC0415
            from sglang.srt.layers.moe.utils import MoeRunnerBackend  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"{_SG_LANG_MOE_TP_SIM_ENV}=sglang_runner requires an environment with sglang and sgl_kernel installed"
            ) from exc

        MoEExperts._ensure_sglang_server_args()
        return MoeRunner, MoeRunnerBackend, MoeRunnerConfig, TritonMoeQuantInfo, StandardDispatchOutput, StandardTopKOutput

    def _sglang_moe_tp_sim_sglang_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Diagnostic TP-shard simulation using SGLang's fused experts kernel."""
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1])).contiguous()
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1).contiguous()
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1).contiguous()

        fused_experts_impl = self._load_sglang_fused_experts_impl()
        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act

        output = hidden_flat.new_zeros(hidden_flat.shape)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            gate_proj = self.gate_up_proj[:, :, start:end].transpose(1, 2).contiguous()
            up_proj = self.gate_up_proj[
                :,
                :,
                self.intermediate_size + start : self.intermediate_size + end,
            ].transpose(1, 2).contiguous()
            w1 = torch.cat([gate_proj, up_proj], dim=1).contiguous()
            w2 = self.down_proj[:, start:end, :].transpose(1, 2).contiguous()
            b1 = None
            if self.gate_up_bias is not None:
                gate_bias = self.gate_up_bias[:, start:end]
                up_bias = self.gate_up_bias[:, self.intermediate_size + start : self.intermediate_size + end]
                b1 = torch.cat([gate_bias, up_bias], dim=1).contiguous()

            shard_output = fused_experts_impl(
                hidden_flat,
                w1,
                w2,
                routing_flat,
                selected_flat,
                b1=b1,
                b2=None,
                inplace=False,
                activation=activation,
                is_gated=self.gated,
                apply_router_weight_on_input=False,
                no_combine=False,
                routed_scaling_factor=None,
                gemm1_limit=self.swiglu_limit if self.swiglu_limit > 0 else None,
                filter_expert=False,
            )
            output = output + shard_output.to(output.dtype)

        return output.reshape(original_shape)

    def _sglang_moe_tp_sim_sglang_runner_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Diagnostic TP-shard simulation through SGLang's MoeRunner wrapper."""
        tp_size, shard_intermediate = self._validate_sglang_moe_tp_sim_inputs(
            routing_weights,
            selected_experts,
            parallel_state,
        )
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, int(hidden_states.shape[-1])).contiguous()
        selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1).contiguous()
        routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1).contiguous()
        hidden_dim = int(hidden_flat.shape[-1])

        (
            MoeRunner,
            MoeRunnerBackend,
            MoeRunnerConfig,
            TritonMoeQuantInfo,
            StandardDispatchOutput,
            StandardTopKOutput,
        ) = self._load_sglang_moe_runner_stack()

        activation = "gelu" if self.hidden_act == "gelu_tanh" else self.hidden_act
        output = hidden_flat.new_zeros(hidden_flat.shape)
        for tp_rank in range(tp_size):
            start = tp_rank * shard_intermediate
            end = start + shard_intermediate
            gate_proj = self.gate_up_proj[:, :, start:end].transpose(1, 2).contiguous()
            up_proj = self.gate_up_proj[
                :,
                :,
                self.intermediate_size + start : self.intermediate_size + end,
            ].transpose(1, 2).contiguous()
            w13 = torch.cat([gate_proj, up_proj], dim=1).contiguous()
            w2 = self.down_proj[:, start:end, :].transpose(1, 2).contiguous()
            b13 = None
            if self.gate_up_bias is not None:
                gate_bias = self.gate_up_bias[:, start:end]
                up_bias = self.gate_up_bias[:, self.intermediate_size + start : self.intermediate_size + end]
                b13 = torch.cat([gate_bias, up_bias], dim=1).contiguous()

            config = MoeRunnerConfig(
                num_experts=self.num_experts,
                num_local_experts=self.num_experts,
                hidden_size=hidden_dim,
                intermediate_size_per_partition=shard_intermediate,
                layer_id=0,
                top_k=int(selected_flat.shape[1]),
                params_dtype=hidden_flat.dtype,
                activation=activation,
                apply_router_weight_on_input=False,
                inplace=False,
                no_combine=False,
                routed_scaling_factor=None,
                gemm1_alpha=None,
                gemm1_clamp_limit=self.swiglu_limit if self.swiglu_limit > 0 else None,
                is_gated=self.gated,
            )
            topk_output = StandardTopKOutput(
                topk_weights=routing_flat,
                topk_ids=selected_flat,
                router_logits=None,
            )
            dispatch_output = StandardDispatchOutput(
                hidden_states=hidden_flat,
                hidden_states_scale=None,
                topk_output=topk_output,
            )
            quant_info = TritonMoeQuantInfo(
                w13_weight=w13,
                w2_weight=w2,
                b13=b13,
                b2=None,
            )
            runner = MoeRunner(MoeRunnerBackend.TRITON, config)
            combine_input = runner.run(dispatch_output, quant_info)
            output = output + combine_input.hidden_states.to(output.dtype)

        return output.reshape(original_shape)

    @torch.compiler.disable
    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        parallel_state,
    ) -> torch.Tensor:
        """Unified EP forward: dispatch → compute → combine.

        All backends share the same dispatch/combine logic. Only the
        expert compute step (group GEMM) differs per backend.

        Dispatch strategy is selected by ``self.ep_dispatch`` (``"alltoall"``
        or ``"deepep"``). Compute backend by ``self.moe_implementation``.
        """

        if self.moe_implementation not in EP_EXPERT_COMPUTE:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"Expert Parallelism. Available: {list(EP_EXPERT_COMPUTE.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. Available: {list(EP_DISPATCH.keys())}"
            )
        if self.fp8_training_enabled and self.moe_implementation != "quack":
            raise NotImplementedError("FP8 grouped MoE compute currently requires moe_implementation='quack'")
        self.last_forward_used_fp8 = bool(self.fp8_training_enabled)

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]

        if self._moe_act and self.moe_implementation in EP_EXPERT_COMPUTE_MOE_ACT:
            compute_fn = EP_EXPERT_COMPUTE_MOE_ACT[self.moe_implementation]
        else:
            compute_fn = EP_EXPERT_COMPUTE[self.moe_implementation]

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(hidden_states, routing_weights, selected_experts, parallel_state)
        deepep_diagnostic_id = _acquire_deepep_parity_diagnostic_record()

        if _DEBUG_EP:
            return self._ep_forward_debug(
                dispatch_fn,
                combine_fn,
                compute_fn,
                dispatch_kwargs,
                parallel_state,
            )

        quack_deepep_no_permute = (
            self.ep_dispatch == "deepep" and self.moe_implementation == "quack" and not _FORCE_QUACK_DEEPEP_GENERIC
        )
        if quack_deepep_no_permute:
            from xorl.distributed.moe.deepep import token_pre_dispatch_no_permute  # noqa: PLC0415

            permute_tokens, cumsum, ctx = token_pre_dispatch_no_permute(**dispatch_kwargs)
        else:
            permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)
        if deepep_diagnostic_id is not None:
            self._emit_deepep_parity_diagnostic(
                record_id=deepep_diagnostic_id,
                phase="post_dispatch",
                dispatch_kwargs=dispatch_kwargs,
                parallel_state=parallel_state,
                permute_tokens=permute_tokens,
                cumsum=cumsum,
                ctx=ctx,
                quack_deepep_no_permute=quack_deepep_no_permute,
            )

        if _FORCE_SYNC:
            torch.cuda.synchronize()

        # Warmup: pre-compile the selected group-GEMM backend variants to avoid
        # first-use compilation memory spikes during training. "triton_w4a4" (the QARL
        # W4A4 down-quant shadow) runs the identical triton group-GEMM kernels, so it
        # must warm up too — otherwise the first W4A4 step pays the compile-memory spike
        # this warmup exists to prevent.
        warmup_attr = f"_kernel_warmed_up_{self.moe_implementation}_{'fp8' if self.fp8_training_enabled else 'bf16'}"
        if self.moe_implementation in {"triton", "triton_w4a4", "quack"} and not getattr(
            type(self), warmup_attr, False
        ):
            if self.moe_implementation == "quack":
                if self.fp8_training_enabled:
                    from xorl.fp8_training import grouped as _fp8_grouped  # noqa: PLC0415

                    _warmup_mn = _fp8_grouped.fp8_group_gemm_same_mn
                    _warmup_gemm = _fp8_grouped.fp8_group_gemm_same_nk
                else:
                    from xorl.ops.group_gemm.kernel import quack as _quack_grouped  # noqa: PLC0415

                    _warmup_mn = _quack_grouped.quack_group_gemm_same_mn
                    _warmup_gemm = _quack_grouped.quack_group_gemm_same_nk
            else:
                from xorl.ops.group_gemm.kernel import group_gemm as _triton_grouped  # noqa: PLC0415

                _warmup_mn = _triton_grouped.group_gemm_same_mn
                _warmup_gemm = _triton_grouped.group_gemm_same_nk

            _d = permute_tokens.device
            _dt = permute_tokens.dtype
            _H = self.gate_up_proj.shape[1]
            _I = self.intermediate_size
            _N = self.gate_up_proj.shape[2]  # 2*I gated, I non-gated
            _E = self.gate_up_proj.shape[0]
            _M = _E * 2
            _cum = torch.arange(2, _M + 2, 2, dtype=torch.int32, device=_d)

            # Forward GEMM: x @ gate_up_proj
            _x = torch.zeros(_M, _H, dtype=_dt, device=_d)
            _w = torch.zeros(_E, _H, _N, dtype=_dt, device=_d)
            _fp8_backend = os.environ.get("XORL_FP8_MOE_GROUPED_BACKEND", self.fp8_training_grouped_backend).strip()
            _fp8_kwargs = (
                {"backend": _fp8_backend, "block_size": self.fp8_training_block_size}
                if self.fp8_training_enabled
                else {}
            )
            _warmup_gemm(a=_x, b=_w, cumsum_M=_cum, max_M=2, **_fp8_kwargs)

            # Backward dgrad FC1: grad_gate_up_act @ gate_up_proj^T
            _g = torch.zeros(_M, _N, dtype=_dt, device=_d)
            _warmup_gemm(a=_g, b=_w, cumsum_M=_cum, max_M=2, transpose_b=True, **_fp8_kwargs)

            # Backward dgrad FC2: grad @ down_proj^T
            _wd = torch.zeros(_E, _I, _H, dtype=_dt, device=_d)
            _gd = torch.zeros(_M, _H, dtype=_dt, device=_d)
            _warmup_gemm(a=_gd, b=_wd, cumsum_M=_cum, max_M=2, transpose_b=True, **_fp8_kwargs)

            # Backward wgrad FC1: permute_tokens^T @ grad_gate_up_act
            _c = torch.zeros(_E, _H, _N, dtype=_dt, device=_d)
            _warmup_mn(a=_x, b=_g, c=_c, cumsum_K=_cum, max_K=2, transpose_a=True, **_fp8_kwargs)

            del _x, _w, _g, _gd, _wd, _c, _cum, _fp8_kwargs
            torch.cuda.empty_cache()
            setattr(type(self), warmup_attr, True)

        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        if quack_deepep_no_permute:
            from xorl.ops.moe.quack import QuackEPDeepEPNoPermute  # noqa: PLC0415

            result = QuackEPDeepEPNoPermute.apply(
                permute_tokens,
                cumsum,
                self.gate_up_proj,
                self.down_proj,
                self.intermediate_size,
                expert_scores,
                dispatch_kwargs["buffer"],
                ctx,
                self.deepep_async_combine,
                self.hidden_act,
                self.activation_native,
                self.fp8_training_enabled,
                self.fp8_training_grouped_backend,
                self.fp8_training_block_size,
                self.gate_up_bias,
                self.down_bias,
            )
            if deepep_diagnostic_id is not None:
                self._emit_deepep_parity_diagnostic(
                    record_id=deepep_diagnostic_id,
                    phase="post_fused_no_permute",
                    dispatch_kwargs=dispatch_kwargs,
                    parallel_state=parallel_state,
                    permute_tokens=permute_tokens,
                    cumsum=cumsum,
                    ctx=ctx,
                    result=result,
                    quack_deepep_no_permute=True,
                )
            return result

        expert_output = compute_fn(
            permute_tokens,
            cumsum,
            self.gate_up_proj,
            self.down_proj,
            self.intermediate_size,
            expert_scores,
            hidden_act=self.hidden_act,
            activation_native=self.activation_native,
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
            fp8_compute=self.fp8_training_enabled,
            fp8_grouped_backend=self.fp8_training_grouped_backend,
            fp8_block_size=self.fp8_training_block_size,
            gated=self.gated,
        )
        if deepep_diagnostic_id is not None:
            self._emit_deepep_parity_diagnostic(
                record_id=deepep_diagnostic_id,
                phase="post_compute",
                dispatch_kwargs=dispatch_kwargs,
                parallel_state=parallel_state,
                permute_tokens=permute_tokens,
                cumsum=cumsum,
                ctx=ctx,
                expert_output=expert_output,
                quack_deepep_no_permute=False,
            )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
        result = combine_fn(**combine_kwargs)
        if deepep_diagnostic_id is not None:
            self._emit_deepep_parity_diagnostic(
                record_id=deepep_diagnostic_id,
                phase="post_combine",
                dispatch_kwargs=dispatch_kwargs,
                parallel_state=parallel_state,
                permute_tokens=permute_tokens,
                cumsum=cumsum,
                ctx=ctx,
                expert_output=expert_output,
                result=result,
                quack_deepep_no_permute=False,
            )
        return result

    def _emit_deepep_parity_diagnostic(
        self,
        *,
        record_id: int,
        phase: str,
        dispatch_kwargs: dict,
        parallel_state,
        permute_tokens: torch.Tensor | None = None,
        cumsum: torch.Tensor | None = None,
        ctx=None,
        expert_output: torch.Tensor | None = None,
        result: torch.Tensor | None = None,
        quack_deepep_no_permute: bool = False,
    ) -> None:
        ep_group = parallel_state.ep_group
        ep_size = _ep_group_size(ep_group)
        ep_rank = _ep_group_rank(ep_group)
        num_local_experts = int(self.gate_up_proj.shape[0])
        num_experts = int(self.num_experts)
        expected_num_local_experts = num_experts // ep_size if ep_size and num_experts % ep_size == 0 else None
        local_start = ep_rank * num_local_experts
        expected_local_start = ep_rank * expected_num_local_experts if expected_num_local_experts is not None else None
        cumsum_length = int(cumsum.numel()) if cumsum is not None else None
        payload = {
            "tag": "xorl_deepep_parity_diagnostic",
            "record_id": record_id,
            "phase": phase,
            "rank": _distributed_rank(),
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "ep_dispatch": self.ep_dispatch,
            "moe_implementation": self.moe_implementation,
            "quack_deepep_no_permute": quack_deepep_no_permute,
            "fp8_training_enabled": bool(self.fp8_training_enabled),
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "expected_num_local_experts": expected_num_local_experts,
            "local_expert_global_range": [local_start, local_start + num_local_experts],
            "expected_local_expert_global_range": (
                [expected_local_start, expected_local_start + expected_num_local_experts]
                if expected_local_start is not None and expected_num_local_experts is not None
                else None
            ),
            "num_local_experts_matches_expected": (
                num_local_experts == expected_num_local_experts if expected_num_local_experts is not None else None
            ),
            "cumsum_length": cumsum_length,
            "cumsum_length_matches_num_local_experts": (
                cumsum_length == num_local_experts if cumsum_length is not None else None
            ),
            "cumsum_length_matches_expected_num_local_experts": (
                cumsum_length == expected_num_local_experts
                if cumsum_length is not None and expected_num_local_experts is not None
                else None
            ),
            "hidden_states": _tensor_summary(dispatch_kwargs.get("hidden_states")),
            "routing_weights": _tensor_summary(dispatch_kwargs.get("routing_weights")),
            "selected_experts": _selected_experts_summary(
                dispatch_kwargs.get("selected_experts"),
                int(self.num_experts),
            ),
            "gate_up_proj": _tensor_summary(self.gate_up_proj, include_sample=False),
            "down_proj": _tensor_summary(self.down_proj, include_sample=False),
            "permute_tokens": _tensor_summary(permute_tokens),
            "cumsum": _cumsum_summary(cumsum),
            "dispatch_ctx": _dispatch_context_summary(ctx),
            "expert_output": _tensor_summary(expert_output),
            "expert_output_reference": (
                _safe_expert_output_reference_comparison(
                    permute_tokens=permute_tokens,
                    cumsum=cumsum,
                    gate_up_proj=self.gate_up_proj,
                    down_proj=self.down_proj,
                    intermediate_size=self.intermediate_size,
                    expert_scores=getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None)),
                    hidden_act=self.hidden_act,
                    gate_up_bias=self.gate_up_bias,
                    down_bias=self.down_bias,
                    expert_output=expert_output,
                )
                if phase == "post_compute"
                else None
            ),
            "result_reference": (
                _safe_result_reference_comparison(
                    hidden_states=dispatch_kwargs.get("hidden_states"),
                    routing_weights=dispatch_kwargs.get("routing_weights"),
                    selected_experts=dispatch_kwargs.get("selected_experts"),
                    gate_up_proj=self.gate_up_proj,
                    down_proj=self.down_proj,
                    intermediate_size=self.intermediate_size,
                    hidden_act=self.hidden_act,
                    gate_up_bias=self.gate_up_bias,
                    down_bias=self.down_bias,
                    result=result,
                    ep_group=ep_group,
                )
                if phase in {"post_combine", "post_fused_no_permute"}
                else None
            ),
            "result": _tensor_summary(result),
        }
        print(f"[DEEPEP PARITY] {json.dumps(payload, sort_keys=True)}", flush=True)

    def _ep_forward_debug(self, dispatch_fn, combine_fn, compute_fn, dispatch_kwargs, parallel_state):
        """Instrumented EP forward with per-phase CUDA event timing.

        Enable via XORL_DEBUG_EP=1.  Prints dispatch/compute/combine wall
        times plus tensor metadata to help diagnose performance gaps between
        different dispatch+compute backend combinations.
        """

        rank = dist.get_rank() if dist.is_initialized() else 0

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

        # --- dispatch ---
        ev[0].record()
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)
        ev[1].record()

        # --- compute ---
        ev[2].record()
        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        expert_output = compute_fn(
            permute_tokens,
            cumsum,
            self.gate_up_proj,
            self.down_proj,
            self.intermediate_size,
            expert_scores,
            hidden_act=self.hidden_act,
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            down_bias=self.down_bias,
            fp8_compute=self.fp8_training_enabled,
            fp8_grouped_backend=self.fp8_training_grouped_backend,
            fp8_block_size=self.fp8_training_block_size,
            gated=self.gated,
        )
        ev[3].record()

        # --- combine ---
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
        ev[4].record()
        result = combine_fn(**combine_kwargs)
        ev[5].record()

        torch.cuda.synchronize()
        t_dispatch = ev[0].elapsed_time(ev[1])
        t_compute = ev[2].elapsed_time(ev[3])
        t_combine = ev[4].elapsed_time(ev[5])

        print(
            f"[EP DEBUG r{rank}] dispatch={self.ep_dispatch} compute={self.moe_implementation}\n"
            f"  hidden_states: {dispatch_kwargs['hidden_states'].shape}\n"
            f"  permute_tokens: shape={permute_tokens.shape}, dtype={permute_tokens.dtype}, "
            f"contiguous={permute_tokens.is_contiguous()}, "
            f"stride={permute_tokens.stride()}, data_ptr_mod4k={permute_tokens.data_ptr() % 4096}\n"
            f"  cumsum: shape={cumsum.shape}, dtype={cumsum.dtype}\n"
            f"  gate_up_proj: shape={self.gate_up_proj.shape}, "
            f"contiguous={self.gate_up_proj.is_contiguous()}, stride={self.gate_up_proj.stride()}\n"
            f"  expert_output: shape={expert_output.shape}\n"
            f"  --- Timing (ms) ---\n"
            f"  Dispatch: {t_dispatch:8.2f}\n"
            f"  Compute:  {t_compute:8.2f}\n"
            f"  Combine:  {t_combine:8.2f}\n"
            f"  Total:    {t_dispatch + t_compute + t_combine:8.2f}",
            flush=True,
        )
        return result

    def _build_dispatch_kwargs(self, hidden_states, routing_weights, selected_experts, parallel_state):
        """Build dispatch kwargs based on ep_dispatch strategy."""
        kwargs = dict(
            hidden_states=hidden_states,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=self.num_experts,
        )
        if self.ep_dispatch == "alltoall":
            kwargs["ep_group"] = parallel_state.ep_group
        elif self.ep_dispatch == "deepep":
            from xorl.distributed.moe.deepep import get_default_buffer  # noqa: PLC0415

            kwargs["buffer"] = get_default_buffer(
                ep_group=parallel_state.ep_group,
                buffer_size_gb=self.deepep_buffer_size_gb,
                num_sms=self.deepep_num_sms,
            )
            kwargs["num_local_experts"] = self.gate_up_proj.shape[0]
        return kwargs

    def _build_combine_kwargs(self, expert_output, ctx, dispatch_kwargs, parallel_state):
        """Build combine kwargs based on ep_dispatch strategy."""
        if self.ep_dispatch == "alltoall":
            return dict(
                expert_output=expert_output,
                ctx=ctx,
                ep_group=parallel_state.ep_group,
                hidden_chunk_size=self.alltoall_combine_hidden_chunk_size,
            )
        elif self.ep_dispatch == "deepep":
            return dict(
                buffer=dispatch_kwargs["buffer"],
                expert_output=expert_output,
                ctx=ctx,
                async_combine=self.deepep_async_combine,
            )

    @classmethod
    def from_config(cls, config, moe_implementation: str = "triton"):
        """Create from a model config (e.g. ``Qwen3MoeConfig``)."""
        return cls(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            moe_implementation=moe_implementation,
            activation_native=getattr(config, "_activation_native", False),
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0)),
        )
