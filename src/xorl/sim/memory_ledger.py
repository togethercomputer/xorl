"""Initial memory ledger built from config constants and observed structured logs."""

from __future__ import annotations

from typing import Any


try:
    from .schemas import MemoryBucket, MemoryLedger, ModelMetadata, ObservedRun, Topology
except ImportError:  # pragma: no cover - exercised by direct script execution
    from schemas import MemoryBucket, MemoryLedger, ModelMetadata, ObservedRun, Topology


BYTES_PER_GIB = 1024**3


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}


def _float_field(section: dict[str, Any], key: str) -> float | None:
    value = section.get(key)
    if value is None:
        return None
    return float(value)


def _dtype_bytes(dtype: Any, *, default: int) -> int:
    if dtype is None:
        return default
    normalized = str(dtype).lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if normalized in {"fp32", "float32", "float"}:
        return 4
    if normalized in {"fp8", "float8", "e4m3", "e5m2"}:
        return 1
    return default


def _gb(byte_count: float) -> float:
    return byte_count / BYTES_PER_GIB


def _round_gb(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _round_fraction(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _param_storage_bytes(train: dict[str, Any]) -> tuple[int, list[str]]:
    explicit_param_dtype = train.get("param_dtype")
    if explicit_param_dtype is not None:
        return _dtype_bytes(explicit_param_dtype, default=4), [f"param_dtype={explicit_param_dtype}"]

    if train.get("enable_mixed_precision"):
        if train.get("skip_param_upcast"):
            return 2, ["mixed_precision_checkpoint_native_params", "skip_param_upcast=true"]
        return 4, ["mixed_precision_generic_fp32_param_upcast", "skip_param_upcast=false"]

    return 4, ["mixed_precision=false"]


def _gradient_storage_bytes(train: dict[str, Any]) -> tuple[int, list[str]]:
    explicit_gradient_dtype = train.get("gradient_dtype")
    if explicit_gradient_dtype is not None:
        return _dtype_bytes(explicit_gradient_dtype, default=4), [f"gradient_dtype={explicit_gradient_dtype}"]

    notes = ["gradient_storage_default=fp32"]
    if train.get("fsdp_reduce_dtype") is not None:
        notes.append(f"fsdp_reduce_dtype={train.get('fsdp_reduce_dtype')}:comm_buffer_only")
    if train.get("muon_grad_dtype") is not None:
        notes.append(f"muon_grad_dtype={train.get('muon_grad_dtype')}:optimizer_update_only")
    return 4, notes


def _estimate_param_breakdown(metadata: ModelMetadata) -> dict[str, float] | None:
    hidden = metadata.hidden_size
    layers = metadata.num_hidden_layers
    vocab = metadata.vocab_size
    if hidden is None or layers is None or vocab is None:
        return None

    head_dim = metadata.head_dim
    if head_dim is None and metadata.num_attention_heads:
        head_dim = hidden // metadata.num_attention_heads
    attention_heads = metadata.num_attention_heads or 1
    key_value_heads = metadata.num_key_value_heads or attention_heads
    if head_dim is None:
        return None

    # Hybrid GatedDeltaNet models (Qwen3.5/3.6): only every full_attention_interval-th layer is full
    # attention; the rest are GatedDeltaNet linear-attention layers with their own projections,
    # depthwise short convolutions, and decay params. Gated attention doubles q_proj (query + output
    # gate). For non-hybrid models this reduces exactly to the previous all-layers-attention formula.
    interval = metadata.full_attention_interval
    gdn_dims = (
        metadata.linear_num_key_heads,
        metadata.linear_num_value_heads,
        metadata.linear_key_head_dim,
        metadata.linear_value_head_dim,
        metadata.linear_conv_kernel_dim,
    )
    is_hybrid = bool(interval) and all(value is not None for value in gdn_dims)
    num_full_attention_layers = layers // int(interval) if is_hybrid else layers
    num_linear_attention_layers = layers - num_full_attention_layers
    gated = bool(metadata.attn_output_gate)

    q_proj = hidden * attention_heads * head_dim * (2 if gated else 1)
    k_proj = hidden * key_value_heads * head_dim
    v_proj = hidden * key_value_heads * head_dim
    o_proj = attention_heads * head_dim * hidden
    attention_params = num_full_attention_layers * (q_proj + k_proj + v_proj + o_proj)

    linear_attention_muon_params = 0
    linear_attention_fallback_params = 0
    if is_hybrid and num_linear_attention_layers > 0:
        num_k_heads = int(metadata.linear_num_key_heads)
        num_v_heads = int(metadata.linear_num_value_heads)
        key_dim = num_k_heads * int(metadata.linear_key_head_dim)
        value_dim = num_v_heads * int(metadata.linear_value_head_dim)
        conv_k = int(metadata.linear_conv_kernel_dim)
        # Muon side (trainer ndim>=2 classifier, no exclusion match): q/k/v[/g]/a/b/o projections and
        # the three depthwise conv weights ([C, 1, k], conv_bias=False).
        gdn_proj = (
            key_dim * hidden  # q_proj
            + key_dim * hidden  # k_proj
            + value_dim * hidden  # v_proj
            + (value_dim * hidden if gated else 0)  # g_proj
            + num_v_heads * hidden  # a_proj
            + num_v_heads * hidden  # b_proj
            + value_dim * hidden  # o_proj
        )
        gdn_convs = (2 * key_dim + value_dim) * conv_k
        linear_attention_muon_params = num_linear_attention_layers * (gdn_proj + gdn_convs)
        # Fallback side: A_log + dt_bias (1D, fp32-pinned) and the gated output norm (head_v_dim).
        gdn_small = 2 * num_v_heads + int(metadata.linear_value_head_dim)
        linear_attention_fallback_params = num_linear_attention_layers * gdn_small
    linear_attention_params = linear_attention_muon_params + linear_attention_fallback_params

    dense_mlp_params = 0
    has_routed_experts = metadata.num_experts is not None and metadata.moe_intermediate_size is not None
    if metadata.intermediate_size is not None and not has_routed_experts:
        dense_mlp_params = layers * 3 * hidden * metadata.intermediate_size

    shared_expert_params = 0
    if metadata.shared_expert_intermediate_size is not None:
        shared_expert_params = layers * 3 * hidden * metadata.shared_expert_intermediate_size

    router_params = 0
    if has_routed_experts and metadata.num_experts is not None:
        router_params = layers * hidden * metadata.num_experts

    expert_params = 0
    if has_routed_experts and metadata.num_experts is not None and metadata.moe_intermediate_size is not None:
        expert_params = layers * metadata.num_experts * 3 * hidden * metadata.moe_intermediate_size

    embedding_params = vocab * hidden
    lm_head_params = 0 if metadata.tie_word_embeddings else vocab * hidden
    # q/k norms exist only on the full-attention layers; GDN layers carry their own output norm
    # (already counted in linear_attention_fallback_params).
    qk_norm_params = 2 * num_full_attention_layers * head_dim
    layer_norm_params = 2 * layers * hidden + qk_norm_params
    final_norm_params = hidden
    non_expert_params = (
        attention_params
        + linear_attention_params
        + dense_mlp_params
        + shared_expert_params
        + router_params
        + embedding_params
        + lm_head_params
    )
    non_expert_params += layer_norm_params + final_norm_params
    return {
        "total_params": float(non_expert_params + expert_params),
        "non_expert_params": float(non_expert_params),
        "expert_params": float(expert_params),
        "layer_non_expert_params": float(
            attention_params
            + linear_attention_params
            + dense_mlp_params
            + shared_expert_params
            + router_params
            + layer_norm_params
        ),
        "attention_params": float(attention_params),
        "linear_attention_params": float(linear_attention_params),
        "linear_attention_muon_params": float(linear_attention_muon_params),
        "linear_attention_fallback_params": float(linear_attention_fallback_params),
        "num_full_attention_layers": float(num_full_attention_layers),
        "num_linear_attention_layers": float(num_linear_attention_layers),
        "dense_mlp_params": float(dense_mlp_params),
        "shared_expert_params": float(shared_expert_params),
        "router_params": float(router_params),
        "embedding_params": float(embedding_params),
        "lm_head_params": float(lm_head_params),
        "qk_norm_params": float(qk_norm_params),
        "layer_norm_params": float(layer_norm_params),
        "final_norm_params": float(final_norm_params),
    }


def _estimate_param_counts(metadata: ModelMetadata) -> tuple[float, float, float] | None:
    breakdown = _estimate_param_breakdown(metadata)
    if breakdown is None:
        return None
    return breakdown["total_params"], breakdown["non_expert_params"], breakdown["expert_params"]


def _non_expert_fsdp_shard_size(topology: Topology, train: dict[str, Any]) -> tuple[int, list[str]]:
    shard_size = max(topology.data_parallel_shard_size, 1)
    notes = [f"dp_shard_size={topology.data_parallel_shard_size}"]
    cp_mode = str(train.get("cp_fsdp_mode", "all") or "all")
    if cp_mode == "all":
        shard_size *= max(topology.sequence_parallel_size, 1)
        notes.append(f"cp_fsdp_mode=all:sequence_parallel_size={topology.sequence_parallel_size}")
    elif cp_mode == "ulysses_only":
        shard_size *= max(topology.ulysses_parallel_size, 1)
        notes.append(f"cp_fsdp_mode=ulysses_only:ulysses_parallel_size={topology.ulysses_parallel_size}")
    elif cp_mode == "ring_only":
        shard_size *= max(topology.ringattn_parallel_size, 1)
        notes.append(f"cp_fsdp_mode=ring_only:ringattn_parallel_size={topology.ringattn_parallel_size}")
    else:
        notes.append(f"cp_fsdp_mode={cp_mode}:no_cp_fsdp_fold")
    return max(shard_size, 1), notes


def _local_param_ownership(
    breakdown: dict[str, float],
    train: dict[str, Any],
    topology: Topology,
) -> tuple[float, float, float, list[str]]:
    pp_size = max(topology.pipeline_parallel_size, 1)
    tp_size = max(topology.tensor_parallel_size, 1)
    if pp_size == 1:
        stage_non_expert_params = breakdown["non_expert_params"]
        stage_expert_params = breakdown["expert_params"]
        endpoint_note = "pp_stage=max_all_layers_and_endpoints"
    else:
        layer_non_expert_stage = breakdown["layer_non_expert_params"] / pp_size
        expert_stage = breakdown["expert_params"] / pp_size
        first_stage_extra = breakdown["embedding_params"]
        last_stage_extra = breakdown["lm_head_params"] + breakdown["final_norm_params"]
        stage_non_expert_params = layer_non_expert_stage + max(first_stage_extra, last_stage_extra)
        stage_expert_params = expert_stage
        endpoint_note = (
            f"pp_stage=max(layer_non_expert/{pp_size}+embedding, layer_non_expert/{pp_size}+lm_head+final_norm)"
        )

    non_expert_fsdp_shard_size, fsdp_notes = _non_expert_fsdp_shard_size(topology, train)
    non_expert_shard_size = non_expert_fsdp_shard_size * tp_size
    expert_shard_size = topology.expert_parallel_size * (topology.ep_fsdp_size or 1)
    local_non_expert_params = stage_non_expert_params / max(non_expert_shard_size, 1)
    local_expert_params = stage_expert_params / max(expert_shard_size, 1)
    notes = [
        endpoint_note,
        f"tp_non_expert_shard_size={tp_size}",
        f"non_expert_total_shard_size={non_expert_shard_size}",
        f"expert_shard_size={expert_shard_size}",
        *fsdp_notes,
    ]
    return local_non_expert_params + local_expert_params, local_non_expert_params, local_expert_params, notes


def _dense_muon_param_partition(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
    *,
    local_params: float | None = None,
) -> dict[str, Any]:
    """Exact dense-model split between Muon matrix params and fallback params."""
    needed = (
        metadata.hidden_size,
        metadata.intermediate_size,
        metadata.num_hidden_layers,
        metadata.num_attention_heads,
    )
    if any(value is None for value in needed):
        return {"status": "unsupported", "reason": "missing_dense_muon_partition_metadata"}
    if metadata.num_experts is not None or metadata.moe_intermediate_size is not None:
        return {"status": "unsupported", "reason": "dense_muon_partition_only"}

    breakdown = _estimate_param_breakdown(metadata)
    if breakdown is None:
        return {"status": "unsupported", "reason": "param_breakdown_unavailable"}
    if local_params is None:
        local_params, _, _, _ = _local_param_ownership(breakdown, train, topology)

    hidden = int(metadata.hidden_size)
    intermediate = int(metadata.intermediate_size)
    layers = int(metadata.num_hidden_layers)
    n_heads = int(metadata.num_attention_heads)
    n_kv = int(metadata.num_key_value_heads or n_heads)
    head_dim = int(metadata.head_dim or hidden // n_heads)
    pp_size = max(int(topology.pipeline_parallel_size), 1)
    tp_size = max(int(topology.tensor_parallel_size), 1)
    non_expert_fsdp_shard_size, fsdp_notes = _non_expert_fsdp_shard_size(topology, train)
    total_non_expert_shard_size = max(non_expert_fsdp_shard_size * tp_size, 1)

    per_layer_matrix_shapes = [
        ("q_proj", n_heads * head_dim, hidden),
        ("k_proj", n_kv * head_dim, hidden),
        ("v_proj", n_kv * head_dim, hidden),
        ("o_proj", n_heads * head_dim, hidden),
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ]
    per_layer_matrix_params = sum(rows * cols for _, rows, cols in per_layer_matrix_shapes)
    stage_matrix_params = per_layer_matrix_params * layers / pp_size
    local_muon_matrix_params = stage_matrix_params / total_non_expert_shard_size
    local_fallback_params = max(float(local_params) - local_muon_matrix_params, 0.0)

    return {
        "status": "exact_analytic_dense_muon_partition",
        "local_params": round(float(local_params)),
        "local_muon_matrix_params": round(local_muon_matrix_params),
        "local_fallback_params": round(local_fallback_params),
        "pipeline_parallel_size": pp_size,
        "tensor_parallel_size": tp_size,
        "non_expert_fsdp_shard_size": non_expert_fsdp_shard_size,
        "total_non_expert_shard_size": total_non_expert_shard_size,
        "matrix_entry_count": round(len(per_layer_matrix_shapes) * layers / pp_size),
        "matrix_shapes_per_layer": [
            {"name": name, "rows": rows, "cols": cols, "numel": rows * cols}
            for name, rows, cols in per_layer_matrix_shapes
        ],
        "fallback_optimizer": str(train.get("muon_fallback_optimizer", "adamw") or "adamw"),
        "notes": [
            "matches src/xorl/optim/optimizer.py Muon ndim>=2 classifier for dense Qwen projections",
            "fallback bucket covers embeddings/lm_head/norms and other excluded or non-matrix params",
            *fsdp_notes,
        ],
    }


def _moe_muon_param_partition(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
    *,
    local_params: float | None = None,
) -> dict[str, Any]:
    """Exact MoE split between Muon matrix params and fallback params for qwen-style MoE.

    This mirrors ``src/xorl/optim/optimizer.py::_classify_muon_params`` at the
    metadata level for the q30/q35 lane: embeddings, lm_head, norms, and
    router ``gate.weight`` use the fallback optimizer; attention projections,
    shared-expert projections, and routed expert tensors use Muon.
    """
    needed = (
        metadata.hidden_size,
        metadata.num_hidden_layers,
        metadata.num_attention_heads,
        metadata.num_experts,
        metadata.moe_intermediate_size,
    )
    if any(value is None for value in needed):
        return {"status": "unsupported", "reason": "missing_moe_muon_partition_metadata"}
    if metadata.num_experts is None or metadata.moe_intermediate_size is None:
        return {"status": "unsupported", "reason": "moe_muon_partition_only"}

    breakdown = _estimate_param_breakdown(metadata)
    if breakdown is None:
        return {"status": "unsupported", "reason": "param_breakdown_unavailable"}

    pp_size = max(int(topology.pipeline_parallel_size), 1)
    if pp_size != 1:
        return {"status": "unsupported", "reason": "moe_muon_partition_pp_stage_not_enabled"}

    if local_params is None:
        local_params, _, _, _ = _local_param_ownership(breakdown, train, topology)

    tp_size = max(int(topology.tensor_parallel_size), 1)
    non_expert_fsdp_shard_size, fsdp_notes = _non_expert_fsdp_shard_size(topology, train)
    total_non_expert_shard_size = max(non_expert_fsdp_shard_size * tp_size, 1)
    expert_shard_size = max(topology.expert_parallel_size * (topology.ep_fsdp_size or 1), 1)

    non_expert_muon_params = (
        breakdown["attention_params"]
        + breakdown.get("linear_attention_muon_params", 0.0)
        + breakdown["dense_mlp_params"]
        + breakdown["shared_expert_params"]
    )
    non_expert_fallback_params = (
        breakdown["embedding_params"]
        + breakdown["lm_head_params"]
        + breakdown["layer_norm_params"]
        + breakdown["final_norm_params"]
        + breakdown["router_params"]
        + breakdown.get("linear_attention_fallback_params", 0.0)
    )
    local_non_expert_muon_params = non_expert_muon_params / total_non_expert_shard_size
    local_non_expert_fallback_params = non_expert_fallback_params / total_non_expert_shard_size
    local_expert_muon_params = breakdown["expert_params"] / expert_shard_size
    local_muon_matrix_params = local_non_expert_muon_params + local_expert_muon_params
    local_fallback_params = local_non_expert_fallback_params
    partitioned_local_params = local_muon_matrix_params + local_fallback_params

    hidden = int(metadata.hidden_size)
    n_heads = int(metadata.num_attention_heads)
    n_kv = int(metadata.num_key_value_heads or n_heads)
    head_dim = int(metadata.head_dim or hidden // n_heads)
    moe_intermediate = int(metadata.moe_intermediate_size)
    shared_intermediate = int(metadata.shared_expert_intermediate_size or 0)
    num_experts = int(metadata.num_experts)

    q_proj_rows = n_heads * head_dim * (2 if metadata.attn_output_gate else 1)
    matrix_shapes_per_layer = [
        {"name": "q_proj", "rows": q_proj_rows, "cols": hidden, "numel": hidden * q_proj_rows},
        {"name": "k_proj", "rows": n_kv * head_dim, "cols": hidden, "numel": hidden * n_kv * head_dim},
        {"name": "v_proj", "rows": n_kv * head_dim, "cols": hidden, "numel": hidden * n_kv * head_dim},
        {"name": "o_proj", "rows": n_heads * head_dim, "cols": hidden, "numel": hidden * n_heads * head_dim},
        {
            "name": "routed_expert_gate_up_down",
            "rows": num_experts,
            "cols": 3 * hidden * moe_intermediate,
            "numel": num_experts * 3 * hidden * moe_intermediate,
        },
    ]
    if shared_intermediate:
        matrix_shapes_per_layer.append(
            {
                "name": "shared_expert_gate_up_down",
                "rows": 3 * shared_intermediate,
                "cols": hidden,
                "numel": 3 * hidden * shared_intermediate,
            }
        )

    return {
        "status": "exact_analytic_moe_muon_partition",
        "local_params": round(float(local_params)),
        "partitioned_local_params": round(partitioned_local_params),
        "local_muon_matrix_params": round(local_muon_matrix_params),
        "local_fallback_params": round(local_fallback_params),
        "local_non_expert_muon_params": round(local_non_expert_muon_params),
        "local_expert_muon_params": round(local_expert_muon_params),
        "local_non_expert_fallback_params": round(local_non_expert_fallback_params),
        "pipeline_parallel_size": pp_size,
        "tensor_parallel_size": tp_size,
        "non_expert_fsdp_shard_size": non_expert_fsdp_shard_size,
        "total_non_expert_shard_size": total_non_expert_shard_size,
        "expert_shard_size": expert_shard_size,
        "matrix_shapes_per_layer": matrix_shapes_per_layer,
        "fallback_optimizer": str(train.get("muon_fallback_optimizer", "adamw") or "adamw"),
        "fallback_param_sources": [
            "embed_tokens",
            "lm_head",
            "norms",
            "router_gate_weight",
            "gdn_A_log_dt_bias_o_norm",
        ],
        "notes": [
            "matches src/xorl/optim/optimizer.py Muon ndim>=2 classifier for qwen-style MoE",
            "routed expert tensors and shared-expert projections use Muon",
            "embeddings/lm_head/norms/router gate.weight use fallback optimizer",
            "hybrid GDN layers: q/k/v[/g]/a/b/o projections and 3D conv1d weights are Muon; "
            "A_log/dt_bias/o_norm are fallback; attention shapes apply to the full-attention "
            "layers only (gated q_proj at 2x width when attn_output_gate)",
            *fsdp_notes,
        ],
    }


def _muon_param_partition(
    metadata: ModelMetadata,
    topology: Topology,
    train: dict[str, Any],
    *,
    local_params: float | None = None,
) -> dict[str, Any]:
    if metadata.num_experts is not None or metadata.moe_intermediate_size is not None:
        return _moe_muon_param_partition(metadata, topology, train, local_params=local_params)
    return _dense_muon_param_partition(metadata, topology, train, local_params=local_params)


def _model_state_buckets(
    raw_config: dict[str, Any],
    topology: Topology | None,
    metadata: ModelMetadata | None,
    deepep_buffer_size_gb: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, list[MemoryBucket], list[str]]:
    if topology is None or metadata is None:
        return None, None, None, None, None, [], ["parameter_and_optimizer_bytes"]

    breakdown = _estimate_param_breakdown(metadata)
    if breakdown is None:
        return None, None, None, None, None, [], ["parameter_and_optimizer_bytes"]

    train = _section(raw_config, "train")
    local_params, local_non_expert_params, local_expert_params, ownership_notes = _local_param_ownership(
        breakdown,
        train,
        topology,
    )

    weight_bytes, weight_notes = _param_storage_bytes(train)
    optimizer = str(train.get("optimizer", "")).lower()
    optimizer_dtype_bytes = _dtype_bytes(train.get("optimizer_dtype"), default=4)
    gradient_bytes, gradient_notes = _gradient_storage_bytes(train)

    sharded_param_gb = _gb(local_params * weight_bytes)
    master_param_gb = 0.0
    if optimizer == "adamw" and weight_bytes < optimizer_dtype_bytes:
        master_param_gb = _gb(local_params * optimizer_dtype_bytes)
    persistent_model_state_gb = sharded_param_gb + master_param_gb

    gradient_state_gb = _gb(local_params * gradient_bytes)
    optimizer_state_gb = 0.0
    optimizer_state_notes: list[str] = []
    if optimizer == "adamw":
        if train.get("cautious_weight_decay"):
            optimizer_state_bytes = 4
            optimizer_state_notes.append("cautious_weight_decay_routes_to_anyprecision_adamw_fp32_state")
        else:
            optimizer_state_bytes = weight_bytes
            optimizer_state_notes.append("torch_adamw_state_dtype_follows_param_dtype")
        optimizer_state_gb = _gb(local_params * 2 * optimizer_state_bytes)
        optimizer_dtype_bytes = optimizer_state_bytes
    elif optimizer == "anyprecision_adamw":
        exp_avg_factor = 0 if train.get("anyprecision_adamw_reuse_grad_for_momentum") else 1
        compensation_factor = (
            1 if train.get("anyprecision_adamw_use_kahan_summation") or train.get("use_kahan_summation") else 0
        )
        optimizer_state_factor = exp_avg_factor + 1 + compensation_factor
        optimizer_state_gb = _gb(local_params * optimizer_state_factor * optimizer_dtype_bytes)
        optimizer_state_notes.extend(
            [
                f"optimizer_state_factor={optimizer_state_factor}",
                f"reuse_grad_for_momentum={bool(train.get('anyprecision_adamw_reuse_grad_for_momentum'))}",
                f"state_cpu_offload={bool(train.get('anyprecision_adamw_state_cpu_offload'))}:step_peak_still_loads_state",
            ]
        )
    elif optimizer == "sgd":
        momentum = float(train.get("momentum", train.get("sgd_momentum", 0.0)) or 0.0)
        if momentum > 0:
            optimizer_state_gb = _gb(local_params * weight_bytes)
            optimizer_dtype_bytes = weight_bytes
            optimizer_state_notes.append(f"sgd_momentum_buffer={momentum}")
    elif optimizer == "muon":
        momentum = float(train.get("muon_momentum", 0.0) or 0.0)
        force_momentum = bool(train.get("muon_force_momentum_path"))
        fallback = str(train.get("muon_fallback_optimizer", "adamw") or "adamw").lower()
        partition = _muon_param_partition(metadata, topology, train, local_params=local_params)
        if partition.get("status") in {"exact_analytic_dense_muon_partition", "exact_analytic_moe_muon_partition"}:
            muon_matrix_params = float(partition["local_muon_matrix_params"])
            fallback_params = float(partition["local_fallback_params"])
            muon_state_gb = _gb(muon_matrix_params * optimizer_dtype_bytes) if momentum > 0 or force_momentum else 0.0
            if fallback == "adamw":
                fallback_state_gb = _gb(fallback_params * 2 * optimizer_dtype_bytes)
                fallback_note = "fallback_adamw_exp_avg_and_exp_avg_sq"
            elif fallback == "sgd":
                fallback_state_gb = 0.0
                fallback_note = "fallback_sgd_state_free"
            else:
                fallback_state_gb = 0.0
                fallback_note = f"unsupported_fallback_optimizer={fallback}"
            optimizer_state_gb = muon_state_gb + fallback_state_gb
            optimizer_state_notes.extend(
                [
                    f"muon_partition={partition['status']}",
                    f"muon_momentum_buffer_params={partition['local_muon_matrix_params']}",
                    f"muon_fallback_params={partition['local_fallback_params']}",
                    fallback_note,
                ]
            )
        elif momentum > 0 or force_momentum:
            optimizer_state_gb = _gb(local_params * optimizer_dtype_bytes)
            optimizer_state_notes.extend(["muon_momentum_buffer", f"partition_status={partition.get('status')}"])

    buckets = [
        MemoryBucket(
            name="sharded_trainable_params",
            gb=_round_gb(sharded_param_gb) or 0.0,
            source="analytic_model_metadata",
            notes=[
                f"weight_bytes={weight_bytes}",
                *weight_notes,
                *ownership_notes,
                f"local_non_expert_params={local_non_expert_params:.0f}",
                f"local_expert_params={local_expert_params:.0f}",
            ],
        ),
        MemoryBucket(
            name="gradient_storage",
            gb=_round_gb(gradient_state_gb) or 0.0,
            source="analytic_dtype_policy",
            notes=[f"gradient_bytes={gradient_bytes}", *gradient_notes],
        ),
    ]
    if master_param_gb:
        buckets.append(
            MemoryBucket(
                name="optimizer_master_params",
                gb=_round_gb(master_param_gb) or 0.0,
                source="analytic_optimizer_policy",
                notes=[f"optimizer={optimizer}", f"optimizer_dtype_bytes={optimizer_dtype_bytes}"],
            )
        )
    if optimizer_state_gb:
        buckets.append(
            MemoryBucket(
                name=f"{optimizer}_optimizer_state",
                gb=_round_gb(optimizer_state_gb) or 0.0,
                source="analytic_optimizer_policy",
                notes=[f"optimizer_dtype_bytes={optimizer_dtype_bytes}", *optimizer_state_notes],
            )
        )
    if deepep_buffer_size_gb:
        buckets.append(
            MemoryBucket(
                name="deepep_static_buffer",
                gb=deepep_buffer_size_gb,
                source="config",
            )
        )

    has_routed_experts = metadata.num_experts is not None and metadata.moe_intermediate_size is not None
    unsupported = [
        "activation_recompute_schedule",
        "attention_workspace",
        "moe_kernel_workspace" if has_routed_experts else "dense_mlp_workspace",
        "fsdp_unshard_and_reduce_scatter_transients",
        "allocator_reserved_slack",
    ]
    return (
        breakdown["total_params"] / 1_000_000_000,
        local_params / 1_000_000_000,
        persistent_model_state_gb,
        gradient_state_gb,
        optimizer_state_gb,
        sorted(buckets, key=lambda bucket: bucket.gb, reverse=True),
        unsupported,
    )


def _deepep_static_buffer_applies(metadata: ModelMetadata | None, topology: Topology | None) -> bool:
    if metadata is None or topology is None:
        return False
    has_routed_experts = metadata.num_experts is not None and metadata.moe_intermediate_size is not None
    return has_routed_experts


def _coverage_status(
    *,
    analytic_peak_floor_gb: float | None,
    calibrated_peak_mem_gb: float | None,
    unsupported_buckets: list[str],
) -> tuple[str, float | None, float | None, float | None]:
    if analytic_peak_floor_gb is None:
        return "unresolved_analytic_floor", None, None, None
    if calibrated_peak_mem_gb is None:
        return "analytic_floor_only", None, None, None
    if calibrated_peak_mem_gb <= 0:
        return "invalid_calibrated_peak", None, None, None

    floor_fraction = analytic_peak_floor_gb / calibrated_peak_mem_gb
    residual = calibrated_peak_mem_gb - analytic_peak_floor_gb
    if residual < 0:
        return "calibrated_peak_below_analytic_floor", floor_fraction, 0.0, 0.0
    residual_fraction = residual / calibrated_peak_mem_gb
    if residual == 0:
        return "analytic_floor_matches_calibrated_peak", floor_fraction, 0.0, 0.0
    if unsupported_buckets:
        return "calibrated_peak_with_unmodeled_residual", floor_fraction, residual, residual_fraction
    return "calibrated_peak_residual_without_unsupported_bucket", floor_fraction, residual, residual_fraction


def build_memory_ledger(
    raw_config: dict[str, Any],
    observed: ObservedRun | None = None,
    *,
    topology: Topology | None = None,
    model_metadata: ModelMetadata | None = None,
    calibrated_peak_mem_gb: float | None = None,
    calibrated_peak_source: str | None = None,
    calibrated_phase_peak_gb: dict[str, float] | None = None,
) -> MemoryLedger:
    model = _section(raw_config, "model")
    train = _section(raw_config, "train")
    observed_peak = None
    observed_phase_peak: dict[str, float] = {}

    if observed is not None:
        peaks = [row.peak_mem_gb for row in observed.steps if row.peak_mem_gb is not None]
        observed_peak = max(peaks) if peaks else None
        for row in observed.steps:
            for phase, value in row.phase_memory_gb.items():
                observed_phase_peak[phase] = max(value, observed_phase_peak.get(phase, value))
        for memory_row in observed.memory_phases:
            for key, value in memory_row.metrics.items():
                observed_phase_peak[key] = max(value, observed_phase_peak.get(key, value))
    if calibrated_phase_peak_gb:
        for phase, value in calibrated_phase_peak_gb.items():
            observed_phase_peak[phase] = max(value, observed_phase_peak.get(phase, value))

    deepep_buffer_size_gb = _float_field(model, "deepep_buffer_size_gb")
    if deepep_buffer_size_gb is None:
        deepep_buffer_size_gb = _float_field(train, "deepep_buffer_size_gb")
    effective_deepep_buffer_size_gb = (
        deepep_buffer_size_gb if _deepep_static_buffer_applies(model_metadata, topology) else None
    )

    (
        estimated_total_params_b,
        estimated_local_params_b,
        persistent_model_state_gb,
        gradient_state_gb,
        optimizer_state_gb,
        top_memory_buckets,
        unsupported_buckets,
    ) = _model_state_buckets(raw_config, topology, model_metadata, effective_deepep_buffer_size_gb)
    analytic_peak_floor_gb = None
    if persistent_model_state_gb is not None and gradient_state_gb is not None and optimizer_state_gb is not None:
        analytic_peak_floor_gb = persistent_model_state_gb + gradient_state_gb + optimizer_state_gb
        analytic_peak_floor_gb += effective_deepep_buffer_size_gb or 0.0

    coverage_peak = observed_peak if observed_peak is not None else calibrated_peak_mem_gb
    coverage_source = "observed_log" if observed_peak is not None else calibrated_peak_source
    (
        memory_coverage_status,
        floor_fraction,
        residual_peak_gb,
        residual_fraction,
    ) = _coverage_status(
        analytic_peak_floor_gb=analytic_peak_floor_gb,
        calibrated_peak_mem_gb=coverage_peak,
        unsupported_buckets=unsupported_buckets,
    )
    if residual_peak_gb is not None and residual_peak_gb > 0:
        top_memory_buckets = [
            *top_memory_buckets,
            MemoryBucket(
                name="calibrated_unmodeled_peak_residual",
                gb=_round_gb(residual_peak_gb) or 0.0,
                source=coverage_source or "calibrated_peak",
                notes=[
                    f"calibrated_peak_gb={coverage_peak:.3f}",
                    f"analytic_peak_floor_gb={analytic_peak_floor_gb:.3f}",
                    f"residual_fraction_of_peak={residual_fraction:.3f}",
                    "covers_unsupported_buckets=" + ",".join(unsupported_buckets),
                ],
            ),
        ]
        top_memory_buckets = sorted(top_memory_buckets, key=lambda bucket: bucket.gb, reverse=True)

    return MemoryLedger(
        deepep_buffer_size_gb=effective_deepep_buffer_size_gb,
        observed_peak_mem_gb_max=observed_peak,
        calibrated_peak_mem_gb=_round_gb(coverage_peak),
        calibrated_peak_source=coverage_source,
        observed_phase_peak_gb=observed_phase_peak,
        estimated_total_params_b=_round_gb(estimated_total_params_b),
        estimated_local_params_b=_round_gb(estimated_local_params_b),
        persistent_model_state_gb=_round_gb(persistent_model_state_gb),
        gradient_state_gb=_round_gb(gradient_state_gb),
        optimizer_state_gb=_round_gb(optimizer_state_gb),
        analytic_peak_floor_gb=_round_gb(analytic_peak_floor_gb),
        analytic_floor_fraction_of_calibrated_peak=_round_fraction(floor_fraction),
        calibrated_residual_peak_gb=_round_gb(residual_peak_gb),
        calibrated_residual_fraction_of_peak=_round_fraction(residual_fraction),
        memory_coverage_status=memory_coverage_status,
        top_memory_buckets=top_memory_buckets,
        unsupported_buckets=unsupported_buckets,
    )
