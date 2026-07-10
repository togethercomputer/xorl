"""Dataclasses shared by the local training-engine simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert simulator dataclasses into plain JSON-compatible containers."""
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelMetadata:
    model_path: str | None
    config_path: str | None
    source: str
    num_experts: int | None = None
    top_k: int | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None
    tie_word_embeddings: bool | None = None


@dataclass(frozen=True)
class Topology:
    world_size: int
    local_world_size: int
    node_count: int
    data_parallel_size: int
    data_parallel_replicate_size: int
    data_parallel_shard_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    ep_fsdp_size: int | None
    ulysses_parallel_size: int
    ringattn_parallel_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    sample_packing_sequence_len: int | None
    num_experts: int | None = None
    top_k: int | None = None

    @property
    def sequence_parallel_size(self) -> int:
        return self.ulysses_parallel_size * self.ringattn_parallel_size


@dataclass(frozen=True)
class RunFingerprint:
    config_path: str
    config_sha256: str
    config_name: str
    repo_commit: str | None
    balanced_routing: bool
    topology: Topology
    model_metadata: ModelMetadata


@dataclass(frozen=True)
class BalancedRoutingLedger:
    total_slots: int
    num_experts: int
    counts_by_expert: list[int]
    max_slots_per_expert: int
    min_slots_per_expert: int
    imbalance_slots: int


@dataclass(frozen=True)
class ShapeLedger:
    microbatch_tokens_per_dp_rank: int | None
    global_tokens_per_microbatch: int | None
    global_tokens_per_train_step: int | None
    tokens_per_gpu_per_train_step: float | None
    sequence_parallel_size: int
    tokens_per_model_rank_per_microbatch: int | None
    routed_slots_per_model_rank_microbatch: int | None
    routed_slots_per_train_step_model_rank: int | None
    balanced_routing: BalancedRoutingLedger | None
    ep_rank_slots_per_microbatch: list[int] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StepObservation:
    source: str
    step: int
    max_steps: str
    loss: float | None = None
    grad_norm: float | None = None
    lr: float | None = None
    tflops_per_gpu: float | None = None
    mfu: float | None = None
    tokens_per_sec: float | None = None
    step_time_s: float | None = None
    peak_mem_gb: float | None = None
    phase_memory_gb: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseObservation:
    source: str
    prefix: str
    step: int
    max_steps: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class MemoryPhaseObservation:
    source: str
    prefix: str
    step: int
    max_steps: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class ObservedRun:
    sources: list[str]
    steps: list[StepObservation]
    phases: list[PhaseObservation] = field(default_factory=list)
    memory_phases: list[MemoryPhaseObservation] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryBucket:
    name: str
    gb: float
    source: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryLedger:
    deepep_buffer_size_gb: float | None
    observed_peak_mem_gb_max: float | None
    observed_phase_peak_gb: dict[str, float]
    estimated_total_params_b: float | None = None
    estimated_local_params_b: float | None = None
    persistent_model_state_gb: float | None = None
    gradient_state_gb: float | None = None
    optimizer_state_gb: float | None = None
    analytic_peak_floor_gb: float | None = None
    top_memory_buckets: list[MemoryBucket] = field(default_factory=list)
    unsupported_buckets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkBehaviorPoint:
    label: str
    source: str
    micro_batch_size: int | None
    global_batch_size: int | None
    tokens_per_sec: float | None
    step_time_sec: float | None
    mfu_percent: float | None = None
    tflops_per_gpu: float | None = None
    peak_mem_gb: float | None = None
    allocator_retries: int | None = None
    measured_steps: int | None = None
    warmup_steps: int | None = None
    gpu_count: int | None = None
    sample_packing_sequence_len: int | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    ulysses_parallel_size: int | None = None
    ringattn_parallel_size: int | None = None
    expert_parallel_size: int | None = None
    ep_fsdp_size: int | None = None
    deepep_async_combine: bool | None = None
    deepep_num_sms: int | None = None
    deepep_buffer_size_gb: float | None = None
    enable_compile: bool | None = None
    gradient_checkpointing_method: str | None = None
    enable_activation_offload: bool | None = None
    activation_offload_prefetch_count: int | None = None
    status: str = "observed"
    correctness_status: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkBehaviorPrediction:
    status: str
    matched_label: str | None
    source: str | None
    tokens_per_sec: float | None
    tokens_per_sec_per_gpu: float | None
    step_time_sec: float | None
    mfu_percent: float | None
    tflops_per_gpu: float | None
    promised_tflops_per_gpu: float | None
    peak_mem_gb: float | None
    allocator_retries: int | None
    derived_global_tokens_per_step: int | None
    correctness_status: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionReport:
    fingerprint: RunFingerprint
    shape: ShapeLedger
    memory: MemoryLedger
    benchmark_behavior: BenchmarkBehaviorPrediction | None = None
    observed_summary: dict[str, Any] | None = None
    calibration_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TradeoffCandidate:
    label: str
    config_path: str | None
    behavior_source: str
    topology: Topology | None
    behavior: BenchmarkBehaviorPrediction
    promotable: bool
    score_tokens_per_sec: float | None
    score_tflops_per_gpu: float | None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TradeoffReport:
    benchmark_dir: str
    status: str
    candidate_count: int
    best_raw: TradeoffCandidate | None
    best_promotable: TradeoffCandidate | None
    candidates: list[TradeoffCandidate]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioCandidate:
    label: str
    config_path: str | None
    topology: Topology
    behavior: BenchmarkBehaviorPrediction
    prediction_confidence: str
    promotable: bool
    feasibility_status: str
    score_tokens_per_sec: float | None
    score_tokens_per_gpu_per_sec: float | None
    score_risk_adjusted_tokens_per_sec: float | None
    analytic_peak_floor_gb: float | None
    estimated_peak_mem_gb: float | None
    memory_basis: str
    memory_headroom_gb: float | None
    max_ep_rank_slots_per_microbatch: int | None
    calibration_scope: str
    recommendation: str
    risk_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioReport:
    base_config_path: str
    benchmark_dir: str | None
    device_memory_limit_gb: float
    memory_safety_factor: float
    topology_sweep: str
    candidate_count: int
    feasible_count: int
    best_raw: ScenarioCandidate | None
    best_risk_adjusted: ScenarioCandidate | None
    best_next_measurement: ScenarioCandidate | None
    best_promotable: ScenarioCandidate | None
    candidates: list[ScenarioCandidate]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationHoldout:
    label: str
    source: str
    topology_label: str
    actual_tokens_per_sec: float
    predicted_tokens_per_sec: float | None
    prediction_status: str
    matched_label: str | None
    absolute_error_tokens_per_sec: float | None
    absolute_percentage_error: float | None
    calibrated_from_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationReport:
    base_config_path: str
    benchmark_dir: str
    status: str
    measured_point_count: int
    evaluated_count: int
    skipped_count: int
    mean_absolute_percentage_error: float | None
    median_absolute_percentage_error: float | None
    max_absolute_percentage_error: float | None
    prediction_status_counts: dict[str, int]
    holdouts: list[CalibrationHoldout]
    warnings: list[str] = field(default_factory=list)
