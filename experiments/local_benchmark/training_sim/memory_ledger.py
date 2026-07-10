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


def _estimate_param_counts(metadata: ModelMetadata) -> tuple[float, float, float] | None:
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

    q_proj = hidden * attention_heads * head_dim
    k_proj = hidden * key_value_heads * head_dim
    v_proj = hidden * key_value_heads * head_dim
    o_proj = attention_heads * head_dim * hidden
    attention_params = layers * (q_proj + k_proj + v_proj + o_proj)

    dense_mlp_params = 0
    has_routed_experts = metadata.num_experts is not None and metadata.moe_intermediate_size is not None
    if metadata.intermediate_size is not None and not has_routed_experts:
        dense_mlp_params = layers * 3 * hidden * metadata.intermediate_size

    shared_expert_params = 0
    if metadata.shared_expert_intermediate_size is not None:
        shared_expert_params = layers * 3 * hidden * metadata.shared_expert_intermediate_size

    expert_params = 0
    if has_routed_experts and metadata.num_experts is not None and metadata.moe_intermediate_size is not None:
        expert_params = layers * metadata.num_experts * 3 * hidden * metadata.moe_intermediate_size

    embedding_params = vocab * hidden
    lm_head_params = 0 if metadata.tie_word_embeddings else vocab * hidden
    norm_params = (2 * layers + 1) * hidden
    non_expert_params = attention_params + dense_mlp_params + shared_expert_params + embedding_params + lm_head_params
    non_expert_params += norm_params
    return float(non_expert_params + expert_params), float(non_expert_params), float(expert_params)


def _model_state_buckets(
    raw_config: dict[str, Any],
    topology: Topology | None,
    metadata: ModelMetadata | None,
    deepep_buffer_size_gb: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, list[MemoryBucket], list[str]]:
    if topology is None or metadata is None:
        return None, None, None, None, None, [], ["parameter_and_optimizer_bytes"]

    counts = _estimate_param_counts(metadata)
    if counts is None:
        return None, None, None, None, None, [], ["parameter_and_optimizer_bytes"]

    total_params, non_expert_params, expert_params = counts
    train = _section(raw_config, "train")
    expert_shard_size = topology.expert_parallel_size * (topology.ep_fsdp_size or 1)
    local_non_expert_params = non_expert_params / max(topology.data_parallel_shard_size, 1)
    local_expert_params = expert_params / max(expert_shard_size, 1)
    local_params = local_non_expert_params + local_expert_params

    weight_bytes = _dtype_bytes(train.get("param_dtype"), default=2 if train.get("enable_mixed_precision") else 4)
    optimizer = str(train.get("optimizer", "")).lower()
    optimizer_dtype_bytes = _dtype_bytes(train.get("optimizer_dtype"), default=4)
    gradient_dtype = train.get("gradient_dtype") or train.get("fsdp_reduce_dtype") or train.get("optimizer_dtype")
    gradient_bytes = _dtype_bytes(gradient_dtype, default=4)

    sharded_param_gb = _gb(local_params * weight_bytes)
    master_param_gb = 0.0
    if optimizer == "adamw":
        master_param_gb = _gb(local_params * optimizer_dtype_bytes)
    persistent_model_state_gb = sharded_param_gb + master_param_gb

    gradient_state_gb = _gb(local_params * gradient_bytes)
    optimizer_state_gb = 0.0
    if optimizer == "adamw":
        optimizer_state_gb = _gb(local_params * 2 * optimizer_dtype_bytes)
    elif optimizer == "muon" and float(train.get("muon_momentum", 0.0) or 0.0) > 0:
        optimizer_state_gb = _gb(local_params * optimizer_dtype_bytes)

    buckets = [
        MemoryBucket(
            name="sharded_trainable_params",
            gb=_round_gb(sharded_param_gb) or 0.0,
            source="analytic_model_metadata",
            notes=[
                f"weight_bytes={weight_bytes}",
                f"local_non_expert_params={local_non_expert_params:.0f}",
                f"local_expert_params={local_expert_params:.0f}",
            ],
        ),
        MemoryBucket(
            name="gradient_storage",
            gb=_round_gb(gradient_state_gb) or 0.0,
            source="analytic_dtype_policy",
            notes=[f"gradient_bytes={gradient_bytes}"],
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
                notes=[f"optimizer_dtype_bytes={optimizer_dtype_bytes}"],
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

    unsupported = [
        "activation_recompute_schedule",
        "attention_workspace",
        "moe_kernel_workspace",
        "fsdp_unshard_and_reduce_scatter_transients",
        "allocator_reserved_slack",
    ]
    return (
        total_params / 1_000_000_000,
        local_params / 1_000_000_000,
        persistent_model_state_gb,
        gradient_state_gb,
        optimizer_state_gb,
        sorted(buckets, key=lambda bucket: bucket.gb, reverse=True),
        unsupported,
    )


def build_memory_ledger(
    raw_config: dict[str, Any],
    observed: ObservedRun | None = None,
    *,
    topology: Topology | None = None,
    model_metadata: ModelMetadata | None = None,
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

    deepep_buffer_size_gb = _float_field(model, "deepep_buffer_size_gb")
    if deepep_buffer_size_gb is None:
        deepep_buffer_size_gb = _float_field(train, "deepep_buffer_size_gb")

    (
        estimated_total_params_b,
        estimated_local_params_b,
        persistent_model_state_gb,
        gradient_state_gb,
        optimizer_state_gb,
        top_memory_buckets,
        unsupported_buckets,
    ) = _model_state_buckets(raw_config, topology, model_metadata, deepep_buffer_size_gb)
    analytic_peak_floor_gb = None
    if persistent_model_state_gb is not None and gradient_state_gb is not None and optimizer_state_gb is not None:
        analytic_peak_floor_gb = persistent_model_state_gb + gradient_state_gb + optimizer_state_gb
        analytic_peak_floor_gb += deepep_buffer_size_gb or 0.0

    return MemoryLedger(
        deepep_buffer_size_gb=deepep_buffer_size_gb,
        observed_peak_mem_gb_max=observed_peak,
        observed_phase_peak_gb=observed_phase_peak,
        estimated_total_params_b=_round_gb(estimated_total_params_b),
        estimated_local_params_b=_round_gb(estimated_local_params_b),
        persistent_model_state_gb=_round_gb(persistent_model_state_gb),
        gradient_state_gb=_round_gb(gradient_state_gb),
        optimizer_state_gb=_round_gb(optimizer_state_gb),
        analytic_peak_floor_gb=_round_gb(analytic_peak_floor_gb),
        top_memory_buckets=top_memory_buckets,
        unsupported_buckets=unsupported_buckets,
    )
