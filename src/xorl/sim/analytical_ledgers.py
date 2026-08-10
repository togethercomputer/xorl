"""Analytical-first ledgers for the XoRL training simulator (q35/q30 lane).

These are *equation-based* predictions computed from model metadata + config/topology BEFORE any
measured log is consulted. Measured runs are used only as validation rows (predicted-vs-measured),
never as the model. Every term carries a ``status`` provenance field:

- ``exact_analytic``               closed-form from shapes/config; no runtime coefficient.
- ``exact_analytic_lower_bound``   closed-form lower bound from shapes/config; no runtime coefficient.
- ``analytic_with_runtime_coefficient`` closed-form scaled by a documented runtime constant.
- ``calibrated_residual``          gap between analytic floor and a measured value.
- ``unsupported``                  not yet modeled (named so the surface is auditable).
- ``not_applicable``               term is structurally absent for this config.

The FLOPs ledger mirrors the trainer's ``src/xorl/utils/count_flops.py::XorlFlopsCounter``
conventions, so analytical TFLOPS computed with a measured step time should match the logged TFLOPS.
Both q35 (Qwen3.5-35B-A3B) and q30 (Qwen3-30B-A3B) use the ``qwen3_moe`` path; dense Qwen3 models
such as Qwen3-8B use the trainer's qwen3/qwen2 dense-decoder path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


try:
    from .memory_ledger import (
        _dtype_bytes,
        _estimate_param_breakdown,
        _gradient_storage_bytes,
        _local_param_ownership,
        _muon_param_partition,
        _param_storage_bytes,
    )
    from .schemas import ModelMetadata, Topology
except ImportError:  # pragma: no cover - exercised by direct script execution
    from memory_ledger import (
        _dtype_bytes,
        _estimate_param_breakdown,
        _gradient_storage_bytes,
        _local_param_ownership,
        _muon_param_partition,
        _param_storage_bytes,
    )
    from schemas import ModelMetadata, Topology


# fwd+bwd multipliers WITHOUT recompute (matches XorlFlopsCounter._grad_ckpt_multipliers, which is
# deliberately checkpointing-independent so logged TFLOPS is stable across recompute strategies):
# 6 = 2 (multiply-add) x 3 (1 forward + 2 backward); attention score has two matmuls => 12.
_M_LINEAR = 6
_M_ATTN_SCORE = 12
_M_LM_HEAD = 6
H100_BF16_PEAK_TFLOPS = 989.0  # matches src count_flops get_device_flops H100 (989e12) + MFU denom
# Exposed (non-overlapped) cross-node comm cost per GB of cross-node traffic, in ms. Calibrated from
# the measured 1-node->2-node step-time delta at identical per-rank work, cross-model-validated:
# q30=6.38, q35=5.82 ms/GB (agree within ~9%) -> mean ~6.1. Intra-node (NVLink) comm is overlapped.
# Recalibrated 2026-07-04 after the expert-FSDP byte-accounting fix (the old 6.1 was fitted against
# ~5x-overcounted expert collective bytes — two compensating errors). New per-model coefficients from
# the same measured 1->2-node step deltas over the CORRECTED exposed bytes: q30 29.08, q35 24.29
# ms/GB (mean 26.7, cross-model spread +-9%). Independently confirmed by the standalone 2-node
# collective microbench (23.0-24.6 ms/GiB at MoE per-layer message sizes): the coefficient now
# decomposes as the real NCCL collective rate at message size with near-zero overlap for the exposed
# collectives, rather than an opaque fudge.
EXPOSED_CROSS_NODE_MS_PER_GB = 26.7
H100_NVLINK_EFFECTIVE_GB_PER_S = 450.0
H100_NDR400_UNIDIRECTIONAL_GB_PER_S = 50.0
H100_NDR400_FULL_DUPLEX_GB_PER_S = 2 * H100_NDR400_UNIDIRECTIONAL_GB_PER_S
MEMORY_PEAK_VALIDATION_THRESHOLD = 0.05


def comm_exposed_time_coefficient(
    *, one_node_step_s: float, two_node_step_s: float, cross_node_gb: float
) -> dict[str, Any]:
    """Calibrate the exposed cross-node comm coefficient (ms per cross-node GB) from node scaling.

    At identical per-rank work, the 1-node->2-node step-time increase is attributed to the cross-node
    comm that was free NVLink at 1 node. Coefficient = delta_step_time / cross_node_GB. Validated by
    agreement across models (q30/q35). Confounds (documented): ep_fsdp1->ep_fsdp2 and a small forward
    growth are folded into the coefficient, so it is an upper-bound on pure cross-node-comm exposure.
    """
    if not cross_node_gb or two_node_step_s <= 0:
        return {"status": "uncomparable"}
    delta = two_node_step_s - one_node_step_s
    return {
        "status": "calibrated",
        "delta_step_time_s": round(delta, 4),
        "cross_node_gb": round(cross_node_gb, 3),
        "exposed_ms_per_gb": round(delta / cross_node_gb * 1000.0, 3),
        "exposed_fraction_of_2node_step": round(delta / two_node_step_s, 4),
    }


def static_cross_node_overlap_estimate(
    terms: dict[str, dict[str, Any]],
    *,
    local_world_size: int,
    num_experts: int | None = None,
    top_k: int | None = None,
    expert_parallel_size: int | None = None,
) -> dict[str, Any]:
    """Static hardware comparison for exposed 2-node FSDP comm time.

    This is intentionally separate from the calibrated exposed-ms/GB model. It uses exact byte terms
    plus a declared H100/NDR400 full-duplex link constant, and assumes the backward tail is dominated
    by cross-node FSDP gradient reduce-scatter while parameter all-gather and intra-node collectives
    are hidden by FSDP prefetch/layer compute. Validation decides whether that static assumption is
    predictive for a measured topology.
    """
    grad = terms.get("fsdp_grad_reduce_scatter", {})
    param = terms.get("fsdp_param_all_gather", {})
    expert_grad = terms.get("expert_fsdp_grad_reduce_scatter", {})
    expert_param = terms.get("expert_fsdp_param_all_gather", {})
    exposed_cross_gib = float(grad.get("cross_gb") or 0.0) + float(expert_grad.get("cross_gb") or 0.0)
    param_passes = max(float(param.get("passes") or 1.0), 1.0)
    prefetched_cross_gib_total = float(param.get("cross_gb") or 0.0) + float(expert_param.get("cross_gb") or 0.0)
    prefetched_cross_gib = prefetched_cross_gib_total / param_passes
    if exposed_cross_gib <= 0.0 and prefetched_cross_gib <= 0.0:
        return {
            "status": "intra_node_fully_overlapped",
            "prediction_basis": "no cross-node FSDP traffic",
            "predicted_backward_cross_node_exposed_s": 0.0,
        }
    predicted = exposed_cross_gib * _BYTES_PER_GIB / (H100_NDR400_FULL_DUPLEX_GB_PER_S * 1e9)
    prefetched_serial = prefetched_cross_gib * _BYTES_PER_GIB / (H100_NDR400_FULL_DUPLEX_GB_PER_S * 1e9)

    def aggregate_two_node_transport_s(term_name: str, *, per_pass: bool = False) -> float | None:
        term = terms.get(term_name, {})
        if int(term.get("nodes_spanned") or 0) != 2:
            return None
        logical_per_rank_gib = float(term.get("gb") or 0.0)
        if per_pass:
            logical_per_rank_gib /= max(float(term.get("passes") or 1.0), 1.0)
        if logical_per_rank_gib <= 0.0 or local_world_size <= 0:
            return None
        # For a two-node collective, the node-pair transport lower bound is the bidirectional
        # traffic crossing the node boundary divided by the aggregate full-duplex NIC bandwidth of
        # the local GPUs. This is distinct from each rank's logical collective bytes.
        aggregate_node_pair_gb_per_s = local_world_size * H100_NDR400_FULL_DUPLEX_GB_PER_S
        bidirectional_node_pair_gib = 2.0 * logical_per_rank_gib
        return bidirectional_node_pair_gib * _BYTES_PER_GIB / (aggregate_node_pair_gb_per_s * 1e9)

    aggregate_grad_s = aggregate_two_node_transport_s("fsdp_grad_reduce_scatter")
    aggregate_param_s = aggregate_two_node_transport_s("fsdp_param_all_gather", per_pass=True)
    aggregate_transport: dict[str, Any]
    if aggregate_grad_s is None:
        aggregate_transport = {
            "status": "not_applicable",
            "reason": "requires two-node FSDP collective terms",
        }
    else:
        aggregate_transport = {
            "status": "static_aggregate_node_pair_transport_bracket",
            "prediction_basis": (
                "two-node FSDP physical transport lower/upper bracket from exact grad collective GiB, "
                "one backward param-all-gather pass, and aggregate full-duplex H100/NDR400 bandwidth "
                "across local GPUs"
            ),
            "local_world_size": int(local_world_size),
            "aggregate_full_duplex_node_pair_gb_per_s": round(
                local_world_size * H100_NDR400_FULL_DUPLEX_GB_PER_S,
                3,
            ),
            "fsdp_grad_reduce_scatter_transport_s": round(aggregate_grad_s, 4),
            "fsdp_param_all_gather_transport_s": round(aggregate_param_s or 0.0, 4),
            "fsdp_grad_only_lower_bound_s": round(aggregate_grad_s, 4),
            "fsdp_grad_plus_param_upper_bound_s": round(aggregate_grad_s + (aggregate_param_s or 0.0), 4),
            "term_status": "exact_analytic_bytes_plus_declared_aggregate_hardware_bandwidth",
        }
    topology_expert_overlap: dict[str, Any]
    if (
        aggregate_grad_s is not None
        and aggregate_param_s is not None
        and num_experts is not None
        and top_k is not None
        and expert_parallel_size not in (None, 0)
    ):
        local_experts_per_ep_rank = float(num_experts) / float(expert_parallel_size)
        active_expert_overlap_slots = float(2 * top_k)
        param_exposure_fraction = min(1.0, active_expert_overlap_slots / local_experts_per_ep_rank)
        predicted_exposed_s = aggregate_grad_s + aggregate_param_s * param_exposure_fraction
        topology_expert_overlap = {
            "status": "static_topology_expert_overlap_model",
            "prediction_basis": (
                "two-node exposed FSDP time = aggregate grad reduce-scatter transport + "
                "one aggregate backward param all-gather transport * min(1, 2 * top_k / "
                "local_experts_per_ep_rank); the overlap fraction is topology-derived from expert "
                "inventory per EP rank"
            ),
            "local_world_size": int(local_world_size),
            "num_experts": int(num_experts),
            "top_k": int(top_k),
            "expert_parallel_size": int(expert_parallel_size),
            "local_experts_per_ep_rank": round(local_experts_per_ep_rank, 3),
            "active_expert_overlap_slots": round(active_expert_overlap_slots, 3),
            "fsdp_grad_reduce_scatter_transport_s": round(aggregate_grad_s, 4),
            "fsdp_param_all_gather_transport_s": round(aggregate_param_s, 4),
            "param_all_gather_exposure_fraction": round(param_exposure_fraction, 6),
            "predicted_backward_cross_node_exposed_s": round(predicted_exposed_s, 4),
            "term_status": ("exact_analytic_bytes_plus_declared_aggregate_hardware_bandwidth_and_topology_overlap"),
        }
    else:
        topology_expert_overlap = {
            "status": "not_applicable",
            "reason": "requires aggregate FSDP transport plus num_experts, top_k, and expert_parallel_size",
        }
    return {
        "status": "static_hardware_overlap_model_compared",
        "prediction_basis": (
            "exact cross-node FSDP grad reduce-scatter GiB divided by declared H100 NDR400 "
            "full-duplex per-GPU bandwidth; FSDP param all-gather is treated as prefetched/overlapped"
        ),
        "exposed_collectives": ["fsdp_grad_reduce_scatter"],
        "prefetched_or_overlapped_collectives": [
            "fsdp_param_all_gather_forward_pass_and_hidden_backward_fraction",
            "ep_all_to_all_dispatch_combine",
            "intra_node_collectives",
        ],
        "fsdp_grad_reduce_scatter_cross_gib": round(exposed_cross_gib, 4),
        "fsdp_param_all_gather_cross_gib_treated_as_overlapped": round(prefetched_cross_gib, 4),
        "fsdp_param_all_gather_cross_gib_total": round(prefetched_cross_gib_total, 4),
        "fsdp_param_all_gather_passes": round(param_passes, 3),
        "ndr400_unidirectional_gb_per_s": H100_NDR400_UNIDIRECTIONAL_GB_PER_S,
        "full_duplex_cross_node_gb_per_s": H100_NDR400_FULL_DUPLEX_GB_PER_S,
        "predicted_backward_cross_node_exposed_s": round(predicted, 4),
        "prefetched_param_all_gather_serial_s_if_unhidden": round(prefetched_serial, 4),
        "aggregate_node_pair_transport_estimate": aggregate_transport,
        "topology_expert_overlap_estimate": topology_expert_overlap,
        "term_status": "exact_analytic_bytes_plus_declared_hardware_bandwidth_assumption",
    }


def _seq_len(topology: Topology, seq_len: int | None) -> int | None:
    return seq_len if seq_len is not None else topology.sample_packing_sequence_len


def flops_ledger(
    metadata: ModelMetadata,
    topology: Topology,
    *,
    seq_len: int | None = None,
    batch_seqlens: list[int] | None = None,
) -> dict[str, Any]:
    """Per-step analytical FLOPs for Qwen3 dense/MoE models, mirroring the trainer's accounting.

    Global FLOPs = sum over all DP-replicated tokens in one optimizer step (global_batch_size
    sequences of length seq_len). Per-GPU FLOPs = global / world_size (the trainer logs per-GPU).
    """
    is_moe = (
        metadata.num_experts is not None and metadata.top_k is not None and metadata.moe_intermediate_size is not None
    )
    fields = (
        metadata.hidden_size,
        metadata.num_hidden_layers,
        metadata.vocab_size,
        metadata.num_attention_heads,
    )
    seq = _seq_len(topology, seq_len)
    if any(value is None for value in fields) or (seq is None and not batch_seqlens):
        return {"status": "unsupported", "reason": "missing_model_metadata_or_seq_len", "components": {}}
    if is_moe and metadata.moe_intermediate_size is None:
        return {"status": "unsupported", "reason": "missing_moe_intermediate_size", "components": {}}
    if not is_moe and metadata.intermediate_size is None:
        return {"status": "unsupported", "reason": "missing_dense_intermediate_size", "components": {}}

    hidden = metadata.hidden_size
    layers = metadata.num_hidden_layers
    vocab = metadata.vocab_size
    n_heads = metadata.num_attention_heads
    n_kv = metadata.num_key_value_heads or n_heads
    head_dim = metadata.head_dim or (hidden // n_heads)
    num_experts = metadata.num_experts or 0
    top_k = metadata.top_k or 0
    moe_inter = metadata.moe_intermediate_size or 0
    dense_inter = metadata.intermediate_size or 0

    if batch_seqlens is not None:
        resolved_batch_seqlens = [int(value) for value in batch_seqlens if int(value) > 0]
        num_sequences = len(resolved_batch_seqlens)
        tokens_global = sum(resolved_batch_seqlens)
        score_elements = sum(s * (s + 1) // 2 for s in resolved_batch_seqlens)
        sequence_source = "batch_seqlens"
    else:
        resolved_batch_seqlens = None
        num_sequences = topology.global_batch_size
        tokens_global = num_sequences * seq
        # causal attention score elements per packed sequence: S(S+1)/2
        # (matches _attention_score_elements)
        score_elements = num_sequences * (seq * (seq + 1) // 2)
        sequence_source = "seq_len"

    q_size = n_heads * head_dim
    kv_size = n_kv * head_dim
    o_size = n_heads * head_dim
    attn_linear_n = hidden * (q_size + 2 * kv_size + o_size)
    lm_head_n = vocab * hidden  # embedding lookup is 0 FLOPs

    components = {}
    if is_moe:
        router_n = hidden * num_experts
        gate_up_n = hidden * moe_inter * top_k * 2  # gate_proj + up_proj
        down_n = hidden * moe_inter * top_k  # down_proj
        components.update(
            {
                "moe_router": _M_LINEAR * router_n * tokens_global * layers,
                "moe_gate_up_proj": _M_LINEAR * gate_up_n * tokens_global * layers,
                "moe_down_proj": _M_LINEAR * down_n * tokens_global * layers,
            }
        )
        source = "mirror:src/xorl/utils/count_flops.py::_estimate_qwen3_moe_flops"
    else:
        dense_mlp_n = hidden * dense_inter * 3  # gate_proj + up_proj + down_proj
        components["dense_mlp"] = _M_LINEAR * dense_mlp_n * tokens_global * layers
        source = "mirror:src/xorl/utils/count_flops.py::_estimate_qwen2_flops (qwen3 dense)"

    components.update(
        {
            "attn_qkvo_proj": _M_LINEAR * attn_linear_n * tokens_global * layers,
            "attn_score_quadratic": _M_ATTN_SCORE * score_elements * head_dim * n_heads * layers,
            "lm_head": _M_LM_HEAD * lm_head_n * tokens_global,
        }
    )
    total_flops = float(sum(components.values()))
    world_size = max(topology.world_size, 1)
    per_gpu_flops = total_flops / world_size
    return {
        "status": "exact_analytic",
        "source": source,
        "multipliers": {
            "linear_fwd_bwd": _M_LINEAR,
            "attn_score_fwd_bwd": _M_ATTN_SCORE,
            "lm_head_fwd_bwd": _M_LM_HEAD,
        },
        "recompute_in_flops": False,
        "seq_len": seq,
        "sequence_source": sequence_source,
        "batch_seqlens": resolved_batch_seqlens,
        "num_sequences": num_sequences,
        "tokens_global": tokens_global,
        "components_flops": {name: float(value) for name, value in components.items()},
        "total_flops": total_flops,
        "per_gpu_flops": per_gpu_flops,
        "mfu_denominator_tflops_per_gpu": H100_BF16_PEAK_TFLOPS,
        "term_status": dict.fromkeys(components, "exact_analytic"),
    }


_BYTES_PER_GIB = 1024**3


def _act_bytes(train: dict[str, Any]) -> int:
    # Activations are stored in the autocast/compute dtype under mixed precision (bf16).
    return 2 if train.get("enable_mixed_precision") else 4


def activation_ledger(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
    *,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """Per-rank analytical activation-memory LOWER BOUND, by named term, with provenance.

    These are the tensors that must be live for one microbatch; they are an analytic lower bound,
    not the allocator-reserved peak. The gap between this and the measured residual is attributed to
    unmodeled allocator/NCCL/DeepEP workspace slack (see ``memory_residual_attribution``).
    """
    needed = (metadata.hidden_size, metadata.num_hidden_layers, metadata.vocab_size)
    seq = _seq_len(topology, seq_len)
    if any(v is None for v in needed) or seq is None:
        return {"status": "unsupported", "reason": "missing_metadata_or_seq_len", "terms": {}}

    hidden = metadata.hidden_size
    layers = metadata.num_hidden_layers
    vocab = metadata.vocab_size
    top_k = metadata.top_k or 0
    moe_inter = metadata.moe_intermediate_size or 0
    dense_inter = metadata.intermediate_size or 0
    sp = max(topology.sequence_parallel_size, 1)
    act = _act_bytes(train)
    model_rank_tokens = (topology.micro_batch_size * seq + sp - 1) // sp  # ceil; one microbatch live at a time
    routed_slots_rank = model_rank_tokens * top_k
    is_moe = top_k > 0 and moe_inter > 0

    ckpt_method = str(train.get("gradient_checkpointing_method", "") or "")
    full_recompute = bool(train.get("enable_gradient_checkpointing")) and ckpt_method in {
        "recompute_full_layer",
        "full",
    }
    # logit dtype for CE: lm_head_fp32 forces fp32 logits; chunked by ce_num_chunks.
    logit_bytes = 4 if (_section_get(train, "lm_head_fp32") or _section_get(train, "router_fp32")) else act
    ce_chunks = int(train.get("ce_num_chunks", 1) or 1)

    terms: dict[str, dict[str, Any]] = {}

    def add(name: str, byte_count: float, status: str, note: str) -> None:
        terms[name] = {"gb": round(byte_count / _BYTES_PER_GIB, 4), "status": status, "note": note}

    activation_term_status = "exact_analytic_lower_bound"
    if full_recompute:
        add(
            "saved_layer_inputs",
            layers * model_rank_tokens * hidden * act,
            activation_term_status,
            "recompute_full_layer: only layer-boundary inputs are stashed for backward",
        )
        if is_moe:
            add(
                "recompute_working_set_one_layer",
                model_rank_tokens * top_k * moe_inter * act * 2,
                activation_term_status,
                "one layer recomputed at a time in backward; MoE gate/up intermediate dominates",
            )
        else:
            add(
                "recompute_working_set_one_layer",
                model_rank_tokens * dense_inter * act * 2,
                activation_term_status,
                "one dense layer recomputed at a time in backward; dense gate/up intermediate dominates",
            )
    else:
        retained_intermediate = top_k * moe_inter if is_moe else dense_inter
        add(
            "saved_full_activations",
            layers * model_rank_tokens * (hidden + retained_intermediate) * act,
            activation_term_status,
            "no full-layer recompute: per-layer hidden and dense/routed intermediate activations retained",
        )
    add(
        "ce_logit_buffer",
        (model_rank_tokens / max(ce_chunks, 1)) * vocab * logit_bytes,
        activation_term_status,
        f"chunked CE logits: tokens/{ce_chunks} x vocab x {logit_bytes}B (lm_head_fp32 -> 4B)",
    )
    if is_moe:
        add(
            "moe_dispatch_combine_buffer",
            routed_slots_rank * hidden * act * 2,
            activation_term_status,
            "alltoall dispatch + combine token buffers (routed slots x hidden)",
        )
    else:
        add(
            "moe_dispatch_combine_buffer",
            0.0,
            "not_applicable",
            "dense model: no routed experts or EP dispatch/combine buffers",
        )
    add(
        "attention_workspace",
        model_rank_tokens * hidden * act * 2,
        activation_term_status,
        "flash-attention working buffers (out + lse), conservative",
    )
    total_gb = round(sum(t["gb"] for t in terms.values()), 4)
    return {
        "status": "exact_analytic_lower_bound",
        "act_bytes": act,
        "model_rank_tokens": model_rank_tokens,
        "routed_slots_per_rank": routed_slots_rank,
        "full_layer_recompute": full_recompute,
        "terms": terms,
        "analytic_activation_lower_bound_gb": total_gb,
        "unmodeled_terms": [
            "allocator_reserved_slack",
            "nccl_deepep_workspace" if is_moe else "nccl_workspace",
            "cuda_context_and_fragmentation",
        ],
    }


def _section_get(train: dict[str, Any], key: str) -> Any:
    return train.get(key)


def memory_residual_attribution(
    *,
    analytic_floor_gb: float | None,
    measured_peak_gb: float | None,
    activation_lower_bound_gb: float | None,
) -> dict[str, Any]:
    """Split measured peak into: param/grad/opt floor + analytic activation lower bound + unmodeled slack."""
    if analytic_floor_gb is None or measured_peak_gb is None:
        return {"status": "incomplete"}
    residual = measured_peak_gb - analytic_floor_gb
    act = activation_lower_bound_gb or 0.0
    unmodeled = residual - act
    return {
        "status": "attributed",
        "measured_peak_gb": round(measured_peak_gb, 3),
        "param_grad_opt_floor_gb": round(analytic_floor_gb, 3),
        "param_grad_opt_floor_status": "exact_analytic",
        "analytic_activation_lower_bound_gb": round(act, 3),
        "analytic_activation_status": "exact_analytic_lower_bound",
        "unmodeled_allocator_workspace_gb": round(unmodeled, 3),
        "unmodeled_status": "calibrated_residual",
        "residual_gb": round(residual, 3),
        "residual_fraction_of_peak": round(residual / measured_peak_gb, 4) if measured_peak_gb else None,
        "activation_explains_fraction_of_residual": round(act / residual, 4) if residual > 0 else None,
    }


def _muon_update_dtype_bytes(
    train: dict[str, Any], gradient_bytes: int, optimizer_dtype_bytes: int
) -> tuple[int, list[str]]:
    update_dtype = train.get("muon_update_dtype")
    if update_dtype is not None:
        return _dtype_bytes(update_dtype, default=optimizer_dtype_bytes), [f"muon_update_dtype={update_dtype}"]

    momentum = float(train.get("muon_momentum", 0.95) or 0.0)
    force_momentum = bool(train.get("muon_force_momentum_path"))
    if momentum > 0 or force_momentum:
        momentum_dtype = train.get("muon_momentum_dtype")
        if momentum_dtype is not None:
            return _dtype_bytes(momentum_dtype, default=optimizer_dtype_bytes), [
                f"muon_momentum_dtype={momentum_dtype}",
                "update_dtype_inherits_momentum_buffer",
            ]
        optimizer_dtype = train.get("optimizer_dtype")
        if optimizer_dtype is not None:
            return optimizer_dtype_bytes, [
                f"optimizer_dtype={optimizer_dtype}",
                "runtime_default_muon_momentum_dtype_inherits_optimizer_dtype",
                "update_dtype_inherits_momentum_buffer",
            ]
        return gradient_bytes, ["update_dtype_inherits_gradient_dtype_without_explicit_momentum_dtype"]

    grad_dtype = train.get("muon_grad_dtype")
    if grad_dtype is not None:
        return _dtype_bytes(grad_dtype, default=gradient_bytes), [
            f"muon_grad_dtype={grad_dtype}",
            "momentum_zero_update_dtype_inherits_muon_grad_dtype",
        ]
    return gradient_bytes, ["momentum_zero_update_dtype_inherits_gradient_dtype"]


def optimizer_step_ledger(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
) -> dict[str, Any]:
    """Optimizer-state and optimizer-step work terms by XoRL optimizer type.

    This is the optimizer-specific quantity used by the phase-time model. It is deliberately separate
    from FLOPs: AdamW-like optimizers are memory/state scans, SignSGD is a state-free param/grad scan,
    and Muon adds Newton-Schulz work that still needs optimizer-specific calibration.
    """
    breakdown = _estimate_param_breakdown(metadata)
    if breakdown is None:
        return {"status": "unsupported", "reason": "param_breakdown_unavailable"}

    local_params, local_non_expert_params, local_expert_params, ownership_notes = _local_param_ownership(
        breakdown,
        train,
        topology,
    )
    optimizer = str(train.get("optimizer", "adamw") or "adamw").lower()
    weight_bytes, weight_notes = _param_storage_bytes(train)
    gradient_bytes, gradient_notes = _gradient_storage_bytes(train)
    optimizer_dtype_bytes = _dtype_bytes(train.get("optimizer_dtype"), default=4)
    param_gb = local_params * weight_bytes / _BYTES_PER_GIB
    gradient_gb = local_params * gradient_bytes / _BYTES_PER_GIB
    state_terms: dict[str, dict[str, Any]] = {}
    transient_terms: dict[str, dict[str, Any]] = {}
    notes = [*ownership_notes, *weight_notes, *gradient_notes]

    def add_state(name: str, params: float, bytes_per_param: int, status: str, note: str) -> None:
        state_terms[name] = {
            "gb": round(params * bytes_per_param / _BYTES_PER_GIB, 3),
            "bytes_per_param": bytes_per_param,
            "status": status,
            "note": note,
        }

    def add_transient(name: str, params: float, bytes_per_param: int, copies: int, status: str, note: str) -> None:
        transient_terms[name] = {
            "gb": round(params * bytes_per_param * copies / _BYTES_PER_GIB, 3),
            "params": round(params),
            "bytes_per_param": bytes_per_param,
            "retained_copies": copies,
            "status": status,
            "note": note,
        }

    status = "exact_analytic"
    unsupported_terms: list[str] = []
    optimizer_family = optimizer

    if optimizer == "adamw":
        if train.get("cautious_weight_decay"):
            state_bytes = 4
            optimizer_family = "anyprecision_adamw"
            notes.append("optimizer=adamw+cautious_weight_decay routes to AnyPrecisionAdamW fp32 state")
        else:
            state_bytes = weight_bytes
            notes.append("torch.optim.AdamW states follow parameter dtype")
        add_state("exp_avg", local_params, state_bytes, "exact_analytic", "AdamW first moment")
        add_state("exp_avg_sq", local_params, state_bytes, "exact_analytic", "AdamW second moment")
    elif optimizer == "anyprecision_adamw":
        state_bytes = optimizer_dtype_bytes
        reuse_grad = bool(train.get("anyprecision_adamw_reuse_grad_for_momentum"))
        state_offload = bool(train.get("anyprecision_adamw_state_cpu_offload"))
        if state_offload:
            status = "analytic_with_runtime_coefficient"
            notes.append("state_cpu_offload=true: GPU persistent state is offloaded between optimizer steps")
        if reuse_grad:
            add_state(
                "exp_avg_aliases_gradient",
                local_params,
                0,
                "aliases_gradient_storage",
                "reuse_grad_for_momentum=true stores first moment in the consumed grad tensor",
            )
        else:
            add_state("exp_avg", local_params, state_bytes, "exact_analytic", "AnyPrecisionAdamW first moment")
        add_state("exp_avg_sq", local_params, state_bytes, "exact_analytic", "AnyPrecisionAdamW second moment")
        if train.get("anyprecision_adamw_use_kahan_summation") or train.get("use_kahan_summation"):
            add_state("compensation", local_params, state_bytes, "exact_analytic", "Kahan compensation buffer")
        if state_offload:
            for term in state_terms.values():
                if term["status"] == "exact_analytic":
                    term["gpu_persistent_gb"] = 0.0
                    term["note"] += "; offloaded between steps"
    elif optimizer == "sgd":
        momentum = float(train.get("momentum", train.get("sgd_momentum", 0.0)) or 0.0)
        if momentum > 0:
            add_state("momentum_buffer", local_params, weight_bytes, "exact_analytic", f"SGD momentum={momentum}")
        else:
            notes.append("SGD momentum=0: state-free optimizer")
    elif optimizer in {"signsgd", "distsignsgd"}:
        notes.append(f"{optimizer} is state-free; distsignsgd signs gradients before FSDP reduce-scatter")
    elif optimizer == "muon":
        momentum = float(train.get("muon_momentum", 0.95) or 0.0)
        force_momentum = bool(train.get("muon_force_momentum_path"))
        fallback = str(train.get("muon_fallback_optimizer", "adamw") or "adamw").lower()
        partition = _muon_param_partition(metadata, topology, train, local_params=local_params)
        partition_exact = partition.get("status") in {
            "exact_analytic_dense_muon_partition",
            "exact_analytic_moe_muon_partition",
        }
        if partition_exact:
            muon_matrix_params = float(partition["local_muon_matrix_params"])
            fallback_params = float(partition["local_fallback_params"])
            if momentum > 0 or force_momentum:
                add_state(
                    "muon_momentum_buffer",
                    muon_matrix_params,
                    optimizer_dtype_bytes,
                    "exact_analytic",
                    (
                        f"Muon momentum={momentum}; ndim>=2 dense projection matrix params use Muon "
                        "per runtime classifier"
                    ),
                )
            else:
                notes.append("Muon momentum=0 and force_momentum_path=false: no Muon momentum buffer")
            if fallback == "adamw":
                add_state(
                    "muon_fallback_exp_avg",
                    fallback_params,
                    optimizer_dtype_bytes,
                    "exact_analytic",
                    "Muon fallback AdamW first moment for non-Muon dense params",
                )
                add_state(
                    "muon_fallback_exp_avg_sq",
                    fallback_params,
                    optimizer_dtype_bytes,
                    "exact_analytic",
                    "Muon fallback AdamW second moment for non-Muon dense params",
                )
            elif fallback == "sgd":
                notes.append("Muon fallback SGD is state-free for non-Muon dense params")
            else:
                unsupported_terms.append(f"muon_fallback_optimizer:{fallback}")
            ns_algorithm = str(train.get("muon_ns_algorithm", "gram_newton_schulz") or "gram_newton_schulz")
            update_bytes, update_notes = _muon_update_dtype_bytes(train, gradient_bytes, optimizer_dtype_bytes)
            if ns_algorithm == "gram_newton_schulz":
                retained_copies = 2
                transient_note = (
                    "grouped Gram-NS retains the pre-orthogonalization update entries and the "
                    "orthogonalized output pieces until the parameter update loop"
                )
                transient_status = "exact_analytic"
            else:
                retained_copies = 1
                transient_note = "standard Muon retains one orthogonalized update tensor per Muon parameter"
                transient_status = "exact_analytic_lower_bound"
            add_transient(
                "muon_retained_update_tensors",
                muon_matrix_params,
                update_bytes,
                retained_copies,
                transient_status,
                transient_note,
            )
            notes.extend(
                [
                    f"Muon partition exact: {partition['status']}; ndim>=2 projection/expert matrices use Muon; "
                    "embeddings/lm_head/norms/router gates use fallback where present",
                    f"muon_matrix_params={partition['local_muon_matrix_params']}",
                    f"muon_fallback_params={partition['local_fallback_params']}",
                    f"muon_ns_algorithm={ns_algorithm}",
                    *update_notes,
                ]
            )
            status = "exact_analytic_lower_bound"
            unsupported_terms.append("muon_newton_schulz_compute_and_kernel_time")
        elif momentum > 0 or force_momentum:
            add_state(
                "muon_momentum_buffer",
                local_params,
                optimizer_dtype_bytes,
                "analytic_with_runtime_coefficient",
                f"Muon momentum={momentum}; metadata-level model treats all local trainable params as Muon-state scanned",
            )
            status = "analytic_with_runtime_coefficient"
        else:
            notes.append("Muon momentum=0 and force_momentum_path=false: no Muon momentum buffer")
        if partition.get("status") != "exact_analytic_dense_muon_partition":
            if partition.get("status") == "exact_analytic_moe_muon_partition":
                pass
            else:
                unsupported_terms.append(f"muon_fallback_param_partition:{fallback}")
                notes.append(
                    "Muon fallback AdamW/SGD partition is parameter-name based at runtime; "
                    "metadata ledger uses aggregate local params"
                )
    else:
        status = "unsupported"
        unsupported_terms.append(f"optimizer:{optimizer}")

    optimizer_state_gb = round(sum(term["gb"] for term in state_terms.values()), 3)
    optimizer_peak_transient_gb = round(sum(term["gb"] for term in transient_terms.values()), 3)
    persistent_scan_gb = round(param_gb + gradient_gb + optimizer_state_gb, 3)
    # Conservative memory-traffic lower bound: read params/grads/states and write params/states.
    step_read_write_gb = round(param_gb * 2 + gradient_gb + optimizer_state_gb * 2, 3)
    if optimizer == "muon":
        step_work_basis = (
            "param+grad+optimizer persistent state scan lower bound; "
            "Muon Newton-Schulz compute/time remains unsupported"
        )
    elif optimizer in {"signsgd", "distsignsgd", "sgd"} and optimizer_state_gb == 0.0:
        step_work_basis = "state-free param+grad scan lower bound"
    else:
        step_work_basis = "param+grad+optimizer persistent state scan lower bound"

    return {
        "status": status,
        "optimizer": optimizer,
        "optimizer_family": optimizer_family,
        "local_params": round(local_params),
        "local_non_expert_params": round(local_non_expert_params),
        "local_expert_params": round(local_expert_params),
        "param_gb": round(param_gb, 3),
        "gradient_gb": round(gradient_gb, 3),
        "optimizer_state_gb": optimizer_state_gb,
        "persistent_scan_gb": persistent_scan_gb,
        "step_read_write_lower_bound_gb": step_read_write_gb,
        "phase_quantity": {
            "name": "optimizer_step_work",
            "value": persistent_scan_gb,
            "unit": "gb",
            "basis": step_work_basis,
        },
        "state_terms": state_terms,
        "optimizer_peak_transient_gb": optimizer_peak_transient_gb,
        "transient_terms": transient_terms,
        "unsupported_terms": unsupported_terms,
        "notes": notes,
    }


def muon_optimizer_peak_memory_attribution(
    *,
    analytic_floor_gb: float | None,
    measured_peak_gb: float | None,
    activation_lower_bound_gb: float | None,
    optimizer_step: dict[str, Any],
    validation_threshold: float = MEMORY_PEAK_VALIDATION_THRESHOLD,
) -> dict[str, Any]:
    if optimizer_step.get("optimizer") != "muon":
        return {"status": "not_applicable", "reason": "optimizer_is_not_muon"}
    if analytic_floor_gb is None or measured_peak_gb is None:
        return {"status": "incomplete"}

    transient_gb = optimizer_step.get("optimizer_peak_transient_gb")
    if transient_gb is None or transient_gb <= 0:
        return {"status": "not_applicable", "reason": "muon_optimizer_transient_unavailable"}

    act = activation_lower_bound_gb or 0.0
    activation_peak_gb = analytic_floor_gb + act
    optimizer_peak_gb = analytic_floor_gb + float(transient_gb)
    predicted_peak_gb = max(activation_peak_gb, optimizer_peak_gb)
    relative_error = abs(predicted_peak_gb - measured_peak_gb) / measured_peak_gb if measured_peak_gb else None
    matches = relative_error is not None and relative_error <= validation_threshold
    transient_terms = optimizer_step.get("transient_terms", {})
    transient_statuses = {
        term.get("status") for term in transient_terms.values() if isinstance(term, dict) and term.get("status")
    }
    transient_status = (
        "exact_analytic"
        if transient_statuses == {"exact_analytic"}
        else "exact_analytic_lower_bound"
        if transient_statuses
        else "unsupported"
    )
    return {
        "status": (
            "muon_optimizer_transient_peak_formula_matches_step_peak"
            if matches
            else "muon_optimizer_transient_peak_formula_compared"
        ),
        "model": (
            "peak = exact param/grad/optimizer floor + max(exact activation lower bound, "
            "exact Muon grouped-update transient)"
        ),
        "validation_threshold": validation_threshold,
        "measured_peak_gb": round(measured_peak_gb, 3),
        "predicted_peak_gb": round(predicted_peak_gb, 3),
        "relative_error": round(relative_error, 6) if relative_error is not None else None,
        "validation_residual_gb": round(measured_peak_gb - predicted_peak_gb, 3),
        "residual_status": "validation_error_not_fit_coefficient",
        "param_grad_opt_floor_gb": round(analytic_floor_gb, 3),
        "param_grad_opt_floor_status": "exact_analytic",
        "analytic_activation_lower_bound_gb": round(act, 3),
        "analytic_activation_status": "exact_analytic_lower_bound",
        "activation_peak_gb": round(activation_peak_gb, 3),
        "muon_optimizer_transient_gb": round(float(transient_gb), 3),
        "muon_optimizer_transient_status": transient_status,
        "optimizer_peak_gb": round(optimizer_peak_gb, 3),
        "peak_driver": "optimizer_step" if optimizer_peak_gb >= activation_peak_gb else "activation",
        "optimizer_transient_terms": transient_terms,
        "optimizer_step_status": optimizer_step.get("status"),
        "optimizer_step_unsupported_terms": optimizer_step.get("unsupported_terms", []),
    }


def fsdp_unshard_transient_ledger(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
) -> dict[str, Any]:
    """Per-rank FSDP full-param transient memory for the largest dense transformer layer.

    The persistent memory floor counts only sharded parameter storage. During FSDP2 forward/backward,
    a fully-sharded layer also materializes an unsharded full-parameter buffer. With forward prefetch,
    the steady-state peak can hold the current layer and one prefetched layer.
    """
    if topology.data_parallel_shard_size <= 1 and not train.get("enable_full_shard"):
        return {"status": "not_applicable", "reason": "not_full_shard"}
    needed = (
        metadata.hidden_size,
        metadata.intermediate_size,
        metadata.num_attention_heads,
        metadata.num_hidden_layers,
    )
    if any(value is None for value in needed):
        return {"status": "unsupported", "reason": "missing_dense_layer_metadata"}
    if metadata.moe_intermediate_size is not None or metadata.num_experts is not None:
        return {"status": "unsupported", "reason": "moe_layer_unshard_formula_not_enabled_for_dense_q8_gate"}

    hidden = metadata.hidden_size
    intermediate = metadata.intermediate_size
    n_heads = metadata.num_attention_heads
    n_kv = metadata.num_key_value_heads or n_heads
    head_dim = metadata.head_dim or (hidden // n_heads)
    q_size = n_heads * head_dim
    kv_size = n_kv * head_dim
    o_size = n_heads * head_dim
    attn_params = hidden * (q_size + 2 * kv_size + o_size)
    dense_mlp_params = hidden * intermediate * 3
    norm_params = 2 * hidden + 2 * head_dim
    largest_layer_params = attn_params + dense_mlp_params + norm_params
    param_bytes, param_notes = _param_storage_bytes(train)
    prefetch_window_layers = 2 if train.get("enable_forward_prefetch") else 1
    layer_unshard_gb = largest_layer_params * param_bytes / _BYTES_PER_GIB
    predicted_gb = prefetch_window_layers * layer_unshard_gb
    return {
        "status": "exact_analytic_dense_fsdp_prefetch",
        "largest_layer_params": int(largest_layer_params),
        "param_bytes": param_bytes,
        "prefetch_window_layers": prefetch_window_layers,
        "one_layer_unshard_gb": round(layer_unshard_gb, 3),
        "predicted_prefetch_window_gb": round(predicted_gb, 3),
        "term_status": {
            "attn_params": "exact_analytic",
            "dense_mlp_params": "exact_analytic",
            "norm_params": "exact_analytic",
            "prefetch_window_layers": "config_exact",
        },
        "notes": [
            *param_notes,
            f"enable_forward_prefetch={bool(train.get('enable_forward_prefetch'))}",
            "persistent floor counts sharded params; FSDP unshard materializes full layer params transiently",
        ],
    }


def communication_ledger(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
    *,
    seq_len: int | None = None,
    param_bytes: int = 2,
    grad_bytes: int = 4,
) -> dict[str, Any]:
    """Per-optimizer-step analytical communication BYTES per rank, by collective, with provenance.

    Bytes are exact_analytic from shapes/topology. Communication TIME is NOT predicted here (it needs
    a calibrated per-link bandwidth + overlap coefficient); for the q35/q30 1-node runs the measured
    comm phases (e.g. sync_sp_gradients) are negligible, consistent with intra-node NVLink + overlap.
    """
    needed = (metadata.hidden_size, metadata.num_hidden_layers)
    seq = _seq_len(topology, seq_len)
    if any(v is None for v in needed) or seq is None:
        return {"status": "unsupported", "reason": "missing_metadata_or_seq_len", "terms": {}}

    from_breakdown = _param_split(metadata)
    if from_breakdown is None:
        return {"status": "unsupported", "reason": "param_breakdown_unavailable", "terms": {}}
    non_expert_params, expert_params = from_breakdown
    hidden = metadata.hidden_size
    layers = metadata.num_hidden_layers
    sp = max(topology.sequence_parallel_size, 1)
    act = 2 if train.get("enable_mixed_precision") else 4
    node_count = max(topology.node_count, 1)

    # Sharded ownership group sizes (mirror memory_ledger ownership).
    dp_shard = max(topology.data_parallel_shard_size, 1)
    tp = max(topology.tensor_parallel_size, 1)
    non_expert_group = dp_shard * tp * (sp if str(train.get("cp_fsdp_mode", "all") or "all") == "all" else 1)

    # all-gather/reduce-scatter move (G-1)/G of the full (unsharded) tensor per rank.
    def ag_factor(group: int) -> float:
        return (group - 1) / group if group > 1 else 0.0

    raw_reshard_after_forward = train.get("reshard_after_forward")
    if raw_reshard_after_forward is None:
        # Mirror src/xorl/distributed/torch_parallelize.py:
        # None means "auto"; PP disables resharding, non-PP uses FSDP's default
        # reshard_after_forward=True.
        reshard_after_forward = topology.pipeline_parallel_size <= 1
        reshard_after_forward_source = "auto_non_pp_true" if reshard_after_forward else "auto_pp_false"
    else:
        reshard_after_forward = bool(raw_reshard_after_forward)
        reshard_after_forward_source = "explicit_config"
    ag_passes = 2.0 if reshard_after_forward else 1.0  # re-gather in backward if resharded after fwd
    local_ws = max(topology.local_world_size, 1)

    def node_span(group_size: int) -> int:
        # how many distinct nodes a collective group spans (contiguous rank->node mapping).
        return max(1, min(-(-group_size // local_ws), node_count))

    def cross_fraction(group_size: int) -> float:
        # ring/tree approximation: inter-node share of a collective whose group spans `ns` nodes.
        ns = node_span(group_size)
        return (ns - 1) / ns if ns > 1 else 0.0

    terms: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        byte_count: float,
        status: str,
        group_size: int,
        note: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        gb = byte_count / _BYTES_PER_GIB
        cf = cross_fraction(group_size)
        ns = node_span(group_size)
        scope = "intra_node" if ns == 1 else ("cross_node" if cf == 1.0 else "intra_and_cross_node")
        term = {
            "gb": round(gb, 4),
            "intra_gb": round(gb * (1 - cf), 4),
            "cross_gb": round(gb * cf, 4),
            "group_size": group_size,
            "nodes_spanned": ns,
            "status": status,
            "scope": scope,
            "note": note,
        }
        if extra:
            term.update(extra)
        terms[name] = term

    def na(name: str, note: str) -> None:
        terms[name] = {
            "gb": 0.0,
            "intra_gb": 0.0,
            "cross_gb": 0.0,
            "nodes_spanned": 0,
            "group_size": 0,
            "status": "not_applicable",
            "scope": "n/a",
            "note": note,
        }

    # FSDP all-gather of params (forward, +backward if resharded), NON-EXPERT shard group only.
    # Expert params are EP-owned: each rank only ever materializes its own num_experts/ep slice, so the
    # expert FSDP collectives move (per-rank slice) x (ep_fsdp-1)/ep_fsdp — NOT global experts — and are
    # ZERO at ep_fsdp=1. They get their own terms below because the ep_fsdp mesh has a different
    # node mapping (1 rank per node under ep_intranode multi-node -> every ring link is inter-node).
    add(
        "fsdp_param_all_gather",
        ag_passes * non_expert_params * ag_factor(non_expert_group) * param_bytes,
        "exact_analytic",
        non_expert_group,
        f"all-gather sharded non-expert params; passes={ag_passes} (reshard_after_forward={reshard_after_forward})",
        extra={
            "passes": ag_passes,
            "per_pass_gb": round((non_expert_params * ag_factor(non_expert_group) * param_bytes) / _BYTES_PER_GIB, 4),
            "effective_reshard_after_forward": reshard_after_forward,
            "raw_reshard_after_forward": raw_reshard_after_forward,
            "reshard_after_forward_source": reshard_after_forward_source,
        },
    )
    # FSDP reduce-scatter of gradients (once per step), NON-EXPERT shard group only.
    add(
        "fsdp_grad_reduce_scatter",
        non_expert_params * ag_factor(non_expert_group) * grad_bytes,
        "exact_analytic",
        non_expert_group,
        "reduce-scatter non-expert gradients across shard group",
    )
    # Expert FSDP collectives over the ep_fsdp mesh (per-rank EP slice, zero when ep_fsdp == 1).
    ep_size_for_split = max(topology.expert_parallel_size, 1)
    ep_fsdp = max(topology.ep_fsdp_size or 1, 1)
    expert_slice_params = expert_params / ep_size_for_split if expert_params else 0.0
    if expert_params and ep_fsdp > 1:
        # ep_intranode packs EP groups on consecutive intra-node ranks, so the ep_fsdp dimension
        # strides by ep: mesh {i, i+ep, i+2*ep, ...} with local_ws/ep members per node. Ring order
        # keeps same-node members adjacent, so exactly ns of the ep_fsdp ring links cross nodes and
        # the exact per-rank cross share is ns/ep_fsdp (1.0 at one rank per node — the previously
        # handled case; 2026-07-06 stride-aware fix prices ep<local_ws meshes that the contiguous
        # rank->node convention wrongly scored intra-node, e.g. ep4 x ep_fsdp4 at world 16).
        add(
            "expert_fsdp_param_all_gather",
            ag_passes * expert_slice_params * ag_factor(ep_fsdp) * param_bytes,
            "exact_analytic",
            ep_fsdp,
            f"all-gather per-rank expert slice across ep_fsdp={ep_fsdp}; passes={ag_passes}",
            extra={"passes": ag_passes},
        )
        add(
            "expert_fsdp_grad_reduce_scatter",
            expert_slice_params * ag_factor(ep_fsdp) * grad_bytes,
            "exact_analytic",
            ep_fsdp,
            f"reduce-scatter expert gradients across ep_fsdp={ep_fsdp}",
        )
        if bool(train.get("ep_intranode", True)) and node_count > 1:
            _members_per_node = max(local_ws // ep_size_for_split, 1)
            _strided_ns = min(-(-ep_fsdp // _members_per_node), node_count)
            if _strided_ns > 1:
                _strided_share = _strided_ns / ep_fsdp
                for _expert_term in ("expert_fsdp_param_all_gather", "expert_fsdp_grad_reduce_scatter"):
                    _t = terms[_expert_term]
                    _t["cross_gb"] = round(_t["gb"] * _strided_share, 4)
                    _t["intra_gb"] = round(_t["gb"] * (1.0 - _strided_share), 4)
                    _t["nodes_spanned"] = _strided_ns
                    _t["scope"] = "cross_node" if _strided_share == 1.0 else "intra_and_cross_node"
                    _t["note"] += (
                        f"; stride-aware node mapping: ep_fsdp strides by ep{ep_size_for_split} "
                        f"({_members_per_node} members/node, spans {_strided_ns} nodes), exact ring "
                        f"share {_strided_ns}/{ep_fsdp} of per-rank bytes crosses nodes"
                    )
    else:
        na(
            "expert_fsdp_param_all_gather",
            "no expert FSDP sharding (dense model or ep_fsdp=1: experts are per-rank local)",
        )
        na(
            "expert_fsdp_grad_reduce_scatter",
            "no expert FSDP sharding (dense model or ep_fsdp=1: experts are per-rank local)",
        )
    # EP all-to-all dispatch + combine, per MoE layer (global routed tokens routed to expert ranks).
    routed_slots_global = topology.global_batch_size * seq * (metadata.top_k or 0)
    ep = max(topology.expert_parallel_size, 1)
    a2a_per_rank = (routed_slots_global / max(topology.world_size, 1)) * hidden * act * 2 * layers * ag_factor(ep)
    ep_intranode = bool(train.get("ep_intranode"))
    if metadata.num_experts is not None and metadata.top_k is not None and metadata.moe_intermediate_size is not None:
        # The EP group spans ceil(ep/local_ws) nodes whenever ep > local_ws regardless of
        # ep_intranode (the flag only packs groups contiguously). All-to-all traffic is uniform over
        # destinations, so the inter-node share is 1 - (intra-node group ranks)/ep, not the ring
        # (ns-1)/ns convention. ep_intranode=False with ep <= local_ws strides EP groups across nodes;
        # that layout has no measured row and keeps the contiguous accounting with a note.
        add(
            "ep_all_to_all_dispatch_combine",
            a2a_per_rank,
            "exact_analytic",
            ep,
            f"alltoall dispatch+combine over EP={ep} (ep_intranode={ep_intranode}); per-layer x{layers}",
        )
        _a2a = terms["ep_all_to_all_dispatch_combine"]
        _a2a_cf = 1.0 - (min(local_ws, ep) / ep)
        _a2a["cross_gb"] = round(_a2a["gb"] * _a2a_cf, 4)
        _a2a["intra_gb"] = round(_a2a["gb"] * (1.0 - _a2a_cf), 4)
        _a2a["scope"] = "intra_node" if _a2a_cf == 0.0 else ("cross_node" if _a2a_cf == 1.0 else "intra_and_cross_node")
        _a2a["note"] += "; uniform-destination inter-node share (not ring convention)"
        if not ep_intranode and ep <= local_ws:
            _a2a["note"] += "; WARNING ep_intranode=False strides EP across nodes (unmeasured layout)"
    else:
        na("ep_all_to_all_dispatch_combine", "dense model: no expert-parallel all-to-all")
    # HSDP DP all-reduce across replicas (only if dp_replicate>1; replicas live on distinct nodes).
    # Per-rank bytes: each rank all-reduces its own NON-EXPERT grad shard (P_ne x grad_bytes /
    # non_expert_group) with its replica peers; ring all-reduce moves 2 x (r-1)/r of the message per
    # rank (reduce-scatter + all-gather phases). Expert grads are EXCLUDED: the ep x ep_fsdp meshes
    # span the full PP stage regardless of dp_replicate, so expert tensors have no replicate
    # dimension and never all-reduce across replicas. (Corrected 2026-07-05: the earlier term
    # charged (non_expert + expert) x full-model bytes x (r-1)/r — a per-global convention ~55x the
    # honest per-rank shard bytes at the 65k replicate2 x shard_sp8 layout.)
    if topology.data_parallel_replicate_size > 1:
        replicate = topology.data_parallel_replicate_size
        add(
            "dp_grad_all_reduce_hsdp",
            non_expert_params * grad_bytes / max(non_expert_group, 1) * 2.0 * ag_factor(replicate),
            "exact_analytic",
            node_count * local_ws,  # replicas span nodes -> force cross-node share
            (
                f"HSDP non-expert grad-shard all-reduce across {replicate} replicas "
                f"(per-rank shard = non_expert/{non_expert_group}; ring 2x(r-1)/r; experts have no "
                "replicate dim: ep x ep_fsdp spans the full stage)"
            ),
        )
    else:
        na("dp_grad_all_reduce_hsdp", "dp_replicate=1")
    # Ulysses / Ring sequence-parallel attention collectives.
    if sp > 1:
        add(
            "sequence_parallel_attention_collective",
            (topology.micro_batch_size * seq / sp) * hidden * act * 4 * layers,
            "analytic_with_runtime_coefficient",
            sp,
            "Ulysses all-to-all (q,k,v,o) or Ring attention per layer",
        )
    else:
        na("sequence_parallel_attention_collective", f"sp={sp}")
    # Pipeline sends (PP stages typically span nodes when multi-node).
    if topology.pipeline_parallel_size > 1:
        add(
            "pipeline_activation_sends",
            (topology.micro_batch_size * seq / sp) * hidden * act * 2 * topology.gradient_accumulation_steps,
            "analytic_with_runtime_coefficient",
            node_count * local_ws,
            "1F1B activation send/recv between PP stages",
        )
    else:
        na("pipeline_activation_sends", "pp=1")

    total = round(sum(t["gb"] for t in terms.values()), 4)
    cross = round(sum(t.get("cross_gb", 0.0) for t in terms.values()), 4)
    intra = round(total - cross, 4)
    # Time estimate: bytes / (link bandwidth x overlap efficiency). Bandwidths are nominal coefficients;
    # they are NOT calibrated here because 1-node comm is fully overlapped/hidden (cannot isolate from
    # measured phase timing). Treat as a lower bound on serial comm time, not a step-time predictor.
    nvlink_gbps = H100_NVLINK_EFFECTIVE_GB_PER_S  # H100 NVLink ~900 GB/s bidir; ~450 effective serial
    crossnode_gbps = H100_NDR400_UNIDIRECTIONAL_GB_PER_S  # nominal per-GPU NDR400 direction
    serial_comm_time_s = intra * _BYTES_PER_GIB / (nvlink_gbps * 1e9) + cross * _BYTES_PER_GIB / (crossnode_gbps * 1e9)
    # CALIBRATED exposed cross-node comm: from the 1-node->2-node step-time delta at identical per-rank
    # work, cross-model-validated on q30 (6.38 ms/GB) and q35 (5.82 ms/GB) -> ~6.1 ms per cross-node GB
    # is the EXPOSED (non-overlapped) step-time cost. Intra-node (NVLink) comm is treated as fully
    # overlapped (~0 exposed), consistent with negligible 1-node comm phases.
    exposed_cross_node_step_gb = 0.0
    for term_name, term in terms.items():
        term_cross_gb = float(term.get("cross_gb") or 0.0)
        if term_name in {"fsdp_param_all_gather", "expert_fsdp_param_all_gather"}:
            # The byte ledger records every logical all-gather pass. The calibrated
            # step-time coefficient was fit to the step-visible param traffic: one
            # pass, plus grad reduce-scatter, when non-PP FSDP auto-reshards.
            term_cross_gb /= max(float(term.get("passes") or 1.0), 1.0)
        exposed_cross_node_step_gb += term_cross_gb
    exposed_cross_node_step_gb = round(exposed_cross_node_step_gb, 4)
    exposed_cross_node_step_time_s = exposed_cross_node_step_gb * (EXPOSED_CROSS_NODE_MS_PER_GB / 1000.0)
    static_overlap_estimate = static_cross_node_overlap_estimate(
        terms,
        local_world_size=local_ws,
        num_experts=metadata.num_experts,
        top_k=metadata.top_k,
        expert_parallel_size=topology.expert_parallel_size,
    )
    return {
        "status": "exact_analytic_bytes",
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "node_count": node_count,
        "terms": terms,
        "total_per_rank_gb": total,
        "intra_node_per_rank_gb": intra,
        "cross_node_per_rank_gb": cross,
        "time_estimate": {
            "status": "calibrated_cross_model" if cross > 0 else "intra_node_fully_overlapped",
            "exposed_cross_node_ms_per_gb": EXPOSED_CROSS_NODE_MS_PER_GB,
            "exact_cross_node_per_rank_gb": cross,
            "exposed_cross_node_step_gb": exposed_cross_node_step_gb,
            "exposed_cross_node_step_time_s": round(exposed_cross_node_step_time_s, 4),
            "calibration_source": "1node->2node step delta, cross-model-validated q30=6.38 q35=5.82 ms/GB",
            "serial_comm_time_lower_bound_s": round(serial_comm_time_s, 4),
            "nvlink_effective_gbps": nvlink_gbps,
            "crossnode_effective_gbps": crossnode_gbps,
            "static_hardware_overlap_estimate": static_overlap_estimate,
            "note": (
                "exposed_cross_node_step_time_s = exposed_cross_node_step_gb x calibrated coefficient; "
                "cross_node_per_rank_gb remains the exact byte ledger. intra-node comm is overlapped (~0 exposed). "
                "serial_comm_time_lower_bound_s uses nominal bandwidths. static_hardware_overlap_estimate is a "
                "separate non-calibrated comparison."
            ),
        },
        "time_status": "calibrated_cross_model" if cross > 0 else "intra_node_overlapped",
        "note": "1-node: comm overlapped (cross=0). 2-node: cross-node exposed comm ~26-30% of step (calibrated).",
    }


def _param_split(metadata: ModelMetadata) -> tuple[float, float] | None:
    """(non_expert_params, expert_params) reusing the memory-ledger param breakdown."""
    breakdown = _estimate_param_breakdown(metadata)
    if breakdown is None:
        return None
    return breakdown["non_expert_params"], breakdown["expert_params"]


def build_model_analytical_coverage(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
    *,
    seq_len: int | None,
    analytic_floor_gb: float | None,
    measured_peak_gb: float | None,
    measured_step_time_s: float | None,
    measured_tflops_per_gpu: float | None = None,
    measured_mfu: float | None = None,
    flops_batch_seqlens: list[int] | None = None,
) -> dict[str, Any]:
    """Full analytical coverage (FLOPs + memory + comm) with predicted-vs-measured for one run."""
    flops = flops_ledger(metadata, topology, seq_len=seq_len, batch_seqlens=flops_batch_seqlens)
    hardware = hardware_flops_ledger(flops, train)
    act = activation_ledger(metadata, topology, train, seq_len=seq_len)
    comm = communication_ledger(metadata, topology, train, seq_len=seq_len)
    optimizer_step = optimizer_step_ledger(metadata, topology, train)
    efficiency = flops_consistency_and_efficiency(
        flops,
        measured_step_time_s=measured_step_time_s,
        measured_tflops_per_gpu=measured_tflops_per_gpu,
        measured_mfu=measured_mfu,
    )
    if (
        efficiency.get("status") == "compared"
        and hardware.get("status")
        in {
            "analytic_with_runtime_coefficient",
            "exact_analytic",
        }
        and measured_step_time_s
    ):
        hw_mfu = hardware["hardware_per_gpu_flops"] / measured_step_time_s / 1e12 / H100_BF16_PEAK_TFLOPS
        efficiency["hardware_achieved_mfu"] = round(hw_mfu, 5)
        efficiency["hardware_vs_logical_note"] = (
            "logical MFU excludes recompute; hardware MFU counts the recomputed forward (true utilization)"
        )
    attribution = memory_residual_attribution(
        analytic_floor_gb=analytic_floor_gb,
        measured_peak_gb=measured_peak_gb,
        activation_lower_bound_gb=act.get("analytic_activation_lower_bound_gb"),
    )
    muon_peak_attribution = muon_optimizer_peak_memory_attribution(
        analytic_floor_gb=analytic_floor_gb,
        measured_peak_gb=measured_peak_gb,
        activation_lower_bound_gb=act.get("analytic_activation_lower_bound_gb"),
        optimizer_step=optimizer_step,
    )
    calibrated_peak_model: dict[str, Any] = {"status": "calibrated_residual_coefficient"}
    is_moe = metadata.num_experts is not None and metadata.moe_intermediate_size is not None
    if is_moe and muon_peak_attribution.get("status") in {
        "muon_optimizer_transient_peak_formula_matches_step_peak",
        "muon_optimizer_transient_peak_formula_compared",
    }:
        attribution = muon_peak_attribution
        calibrated_peak_model = {
            "status": attribution["status"],
            "form": attribution["model"],
            "predicted_peak_gb": attribution["predicted_peak_gb"],
            "relative_error": attribution["relative_error"],
            "validation_threshold": attribution["validation_threshold"],
            "note": (
                "No residual fraction is fit here; the dominant q30/q35 peak is the Muon "
                "grouped-update transient visible in src/xorl/optim/muon.py."
            ),
        }
    elif attribution.get("status") == "attributed" and analytic_floor_gb:
        rf = attribution.get("residual_fraction_of_peak")
        calibrated_peak_model = {
            "status": "calibrated_residual_coefficient",
            "form": "predicted_peak = analytic_floor / (1 - residual_fraction)",
            "residual_fraction_coefficient": rf,
            "predicted_peak_gb": round(analytic_floor_gb / (1 - rf), 3) if rf is not None and rf < 1 else None,
            "cross_model_validation": "q30-calibrated coefficient predicts q35 peak to ~1.1% (see q30_fit_boundary_predictions)",
        }
    return {
        "calibrated_peak_model": calibrated_peak_model,
        "flops_ledger": flops,
        "hardware_flops_ledger": hardware,
        "flops_efficiency_vs_measured": efficiency,
        "activation_ledger": act,
        "optimizer_step_ledger": optimizer_step,
        "memory_residual_attribution": attribution,
        "communication_ledger": comm,
    }


def reference_counter_total_flops(
    metadata: ModelMetadata,
    topology: Topology,
    *,
    seq_len: int | None = None,
    batch_seqlens: list[int] | None = None,
) -> float | None:
    """Total FLOPs from the *actual* trainer ``XorlFlopsCounter`` (transcription ground truth).

    Returns None if xorl is not importable. Used by tests to assert the analytical ledger reproduces
    the trainer's convention EXACTLY, which is the only honest validation of FLOPs (they are a logged
    convention, not a hardware measurement).
    """
    seq = _seq_len(topology, seq_len)
    if seq is None and not batch_seqlens:
        return None
    try:
        from xorl.utils.count_flops import XorlFlopsCounter  # noqa: PLC0415 (lazy: xorl optional/heavy)
    except Exception:  # pragma: no cover - xorl not importable in this context
        return None
    cfg = SimpleNamespace(
        model_type="qwen3_moe" if metadata.moe_intermediate_size is not None else "qwen3",
        hidden_size=metadata.hidden_size,
        vocab_size=metadata.vocab_size,
        intermediate_size=metadata.intermediate_size,
        moe_intermediate_size=metadata.moe_intermediate_size,
        num_hidden_layers=metadata.num_hidden_layers,
        num_key_value_heads=metadata.num_key_value_heads,
        num_attention_heads=metadata.num_attention_heads,
        num_experts=metadata.num_experts,
        num_experts_per_tok=metadata.top_k,
        head_dim=metadata.head_dim,
    )
    counter = XorlFlopsCounter(cfg, gradient_checkpointing_enabled=False)
    if batch_seqlens is None:
        batch_seqlens = [seq] * topology.global_batch_size
    tokens_sum = sum(batch_seqlens)
    if cfg.model_type == "qwen3_moe":
        tflops = counter._estimate_qwen3_moe_flops(tokens_sum, batch_seqlens, delta_time=1.0)
    else:
        tflops = counter._estimate_qwen2_flops(tokens_sum, batch_seqlens, delta_time=1.0)
    return float(tflops) * 1e12


def hardware_flops_ledger(ledger: dict[str, Any], train: dict[str, Any]) -> dict[str, Any]:
    """Recompute-aware HARDWARE FLOPs (what actually runs on the GPU and determines step time).

    The logged ``flops_ledger`` is the *logical* convention (multiplier 6 = 2 MAC x [1 fwd + 2 bwd];
    recompute excluded). Under activation checkpointing the forward of the checkpointed scope is run a
    SECOND time in backward, so the recomputed components do 4 passes (fwd + recompute-fwd + 2 bwd) =>
    multiplier 8 (linear) / 16 (attn score). lm_head/CE sit outside the recomputed transformer layers.
    This explains why the *logical* MFU understates true hardware utilization.
    """
    if ledger.get("status") != "exact_analytic":
        return {"status": "unsupported"}
    method = str(train.get("gradient_checkpointing_method", "") or "")
    enabled = bool(train.get("enable_gradient_checkpointing"))
    comp = ledger["components_flops"]
    # recomputed scope: the transformer-layer components (not lm_head).
    recomputed_keys = {
        "dense_mlp",
        "moe_router",
        "moe_gate_up_proj",
        "moe_down_proj",
        "attn_qkvo_proj",
        "attn_score_quadratic",
    }
    if enabled and method in {"recompute_full_layer", "full"}:
        recompute_factor = 8.0 / 6.0  # extra forward over the base 6 multiplier
        status = "exact_analytic"
        note = "recompute_full_layer: checkpointed forward re-run once in backward (4 passes vs 3)"
    elif enabled and method == "recompute_before_dispatch":
        # only the pre-dispatch portion (attention + router + gate_up) is recomputed; down is not.
        recompute_factor = 8.0 / 6.0
        recomputed_keys = {"moe_router", "moe_gate_up_proj", "attn_qkvo_proj", "attn_score_quadratic"}
        status = "exact_analytic"
        note = "recompute_before_dispatch: pre-dispatch scope recomputed"
    else:
        recompute_factor = 1.0
        status = "exact_analytic"
        note = "no activation recompute"
    actual_recomputed_keys = {name for name in recomputed_keys if name in comp}
    hardware_components = {
        name: (value * recompute_factor if name in actual_recomputed_keys else value) for name, value in comp.items()
    }
    hardware_total = float(sum(hardware_components.values()))
    logical_total = ledger["total_flops"]
    world = 1 if logical_total == 0 else (logical_total / ledger["per_gpu_flops"])
    return {
        "status": status,
        "note": note,
        "recompute_factor_on_scope": recompute_factor,
        "recomputed_components": sorted(actual_recomputed_keys) if recompute_factor != 1.0 else [],
        "logical_total_flops": logical_total,
        "hardware_total_flops": hardware_total,
        "hardware_per_gpu_flops": hardware_total / world,
        "recompute_overhead_fraction": round((hardware_total - logical_total) / logical_total, 4)
        if logical_total
        else None,
    }


def flops_consistency_and_efficiency(
    ledger: dict[str, Any],
    *,
    measured_step_time_s: float | None,
    measured_tflops_per_gpu: float | None = None,
    measured_mfu: float | None = None,
) -> dict[str, Any]:
    """Recover the *real* achieved-FLOPS rate (and MFU) from analytical FLOPs + measured step time.

    NOTE: logged ``tflops``/``mfu`` are computed by the trainer from the SAME FLOPs formula
    (``formula_FLOPs / step_time / world``), so matching them is a consistency check on the formula
    inputs (step_time, token count), NOT an independent validation of FLOPs. The genuinely measured
    quantity is ``step_time``; ``achieved_flops_rate`` (= per-GPU FLOPs / step_time) and ``mfu`` are
    real hardware-efficiency numbers given the FLOPs convention.
    """
    if ledger.get("status") != "exact_analytic" or not measured_step_time_s:
        return {"status": "uncomparable", "reason": "missing_analytic_flops_or_step_time"}
    per_gpu = ledger["per_gpu_flops"]
    achieved_rate = per_gpu / measured_step_time_s  # FLOP/s/GPU (real: step_time is measured)
    achieved_tflops = achieved_rate / 1e12
    mfu = achieved_tflops / H100_BF16_PEAK_TFLOPS
    out: dict[str, Any] = {
        "status": "compared",
        "interpretation": "logged_tflops_is_formula_derived_not_independent_measurement",
        "measured_step_time_s": measured_step_time_s,
        "achieved_flops_rate_per_gpu": achieved_rate,
        "achieved_mfu": round(mfu, 5),  # the calibrated efficiency coefficient (real)
        "logged_tflops_per_gpu": measured_tflops_per_gpu,
        "logged_mfu": measured_mfu,
    }
    # Formula-consistency: our FLOPs/step_time should reproduce the logged tflops (tautological up to
    # our token-count vs the trainer's actual padded batch_seqlens). A large gap flags a token-count bug.
    if measured_tflops_per_gpu:
        out["tflops_formula_consistency_rel_error"] = round(
            abs(achieved_tflops - measured_tflops_per_gpu) / measured_tflops_per_gpu, 4
        )
    return out


def step_time_leave_one_out(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Leave-one-out step-time prediction across runs of one model (non-circular validation).

    Each row: {label, per_gpu_hardware_flops, measured_step_time_s}. For each run, calibrate the
    achieved hardware-FLOPS rate from the OTHER runs and predict this run's step time; the residual is
    the genuine per-config efficiency variation (mbs/ep/ga). Quantifies how predictive a single
    calibrated rate is across the workload/parallelism axes.
    """
    usable = [r for r in rows if r.get("per_gpu_hardware_flops") and r.get("measured_step_time_s")]
    if len(usable) < 2:
        return {"status": "insufficient_rows", "row_count": len(usable)}
    predictions = []
    abs_errors = []
    for i, row in enumerate(usable):
        others = [usable[j] for j in range(len(usable)) if j != i]
        rates = [o["per_gpu_hardware_flops"] / o["measured_step_time_s"] for o in others]
        cal_rate = sum(rates) / len(rates)
        predicted = row["per_gpu_hardware_flops"] / cal_rate
        measured = row["measured_step_time_s"]
        rel = abs(predicted - measured) / measured
        abs_errors.append(rel)
        predictions.append(
            {
                "label": row.get("label"),
                "measured_step_time_s": round(measured, 4),
                "predicted_step_time_s": round(predicted, 4),
                "rel_error": round(rel, 4),
            }
        )
    return {
        "status": "validated",
        "row_count": len(usable),
        "predictions": predictions,
        "mean_abs_rel_error": round(sum(abs_errors) / len(abs_errors), 4),
        "max_abs_rel_error": round(max(abs_errors), 4),
        "interpretation": "residual = per-config efficiency variation a single calibrated rate cannot capture",
    }


def predict_step_time_from_calibrated_mfu(ledger: dict[str, Any], *, calibrated_mfu: float) -> dict[str, Any]:
    """Held-out (non-circular) use of analytical FLOPs: predict step time from a calibrated MFU.

    ``predicted_step_time = per_gpu_flops / (calibrated_mfu * peak_flops_per_gpu)``. Calibrate
    ``calibrated_mfu`` on a REFERENCE run and validate the predicted step time against a DIFFERENT
    run; the residual is the real cross-run efficiency variation, not a tautology.
    """
    if ledger.get("status") != "exact_analytic" or not calibrated_mfu:
        return {"status": "uncomparable"}
    per_gpu = ledger["per_gpu_flops"]
    peak = H100_BF16_PEAK_TFLOPS * 1e12
    predicted = per_gpu / (calibrated_mfu * peak)
    return {
        "status": "predicted",
        "calibrated_mfu": calibrated_mfu,
        "predicted_step_time_s": round(predicted, 4),
    }
