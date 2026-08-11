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
    # Hybrid GatedDeltaNet models (Qwen3.5/3.6): every full_attention_interval-th layer is full
    # attention, the rest are GatedDeltaNet linear-attention. None => all layers are full attention.
    full_attention_interval: int | None = None
    # Gated attention (Qwen3.5/3.6): full-attention q_proj is 2x width (query + sigmoid output gate),
    # and the GatedDeltaNet layers carry a g_proj. None/False => ungated.
    attn_output_gate: bool | None = None
    # GatedDeltaNet linear-attention dimensions (present only for hybrid models).
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None


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
    calibrated_peak_mem_gb: float | None
    calibrated_peak_source: str | None
    observed_phase_peak_gb: dict[str, float]
    estimated_total_params_b: float | None = None
    estimated_local_params_b: float | None = None
    persistent_model_state_gb: float | None = None
    gradient_state_gb: float | None = None
    optimizer_state_gb: float | None = None
    analytic_peak_floor_gb: float | None = None
    analytic_floor_fraction_of_calibrated_peak: float | None = None
    calibrated_residual_peak_gb: float | None = None
    calibrated_residual_fraction_of_peak: float | None = None
    memory_coverage_status: str = "unresolved_analytic_floor"
    top_memory_buckets: list[MemoryBucket] = field(default_factory=list)
    unsupported_buckets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimingLedger:
    source: str | None
    timing_coverage_status: str
    forward_backward_s: float | None
    forward_s: float | None
    loss_s: float | None
    backward_s: float | None
    optimizer_s: float | None
    input_s: float | None
    step_time_s: float | None
    phase_time_sec: dict[str, float] = field(default_factory=dict)
    phase_time_share: dict[str, float] = field(default_factory=dict)
    phase_bottleneck_phase: str | None = None
    phase_bottleneck_bucket: str | None = None
    phase_bottleneck_share: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimulatorSupportLedger:
    requested_surface: str
    support_status: str
    support_blockers: list[str]
    supported_outputs: list[str]
    unsupported_outputs: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommLedger:
    tensor_parallel_cross_node: bool
    pipeline_parallel_cross_node: bool
    expert_parallel_cross_node: bool
    context_parallel_cross_node: bool
    fsdp_cross_node: bool
    cross_node_dimensions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkBehaviorPoint:
    label: str
    source: str
    micro_batch_size: int | None
    global_batch_size: int | None
    tokens_per_sec: float | None
    step_time_sec: float | None
    gradient_accumulation_steps: int | None = None
    tokens_per_sec_std: float | None = None
    tokens_per_sec_cv: float | None = None
    step_time_sec_std: float | None = None
    step_time_sec_cv: float | None = None
    # Median over post-warmup steps of (tokens_per_sec x step_time) computed PER STEP — the realized
    # per-step token load, free of the mean(tps) x mean/median(step) cross-aggregate biases.
    tokens_per_step: float | None = None
    phase_time_sec: dict[str, float] = field(default_factory=dict)
    # Cross-rank MEAN companion of phase_time_sec (which is the cross-rank MAX convention): lets
    # balanced-rank term comparisons separate rank asymmetry from term error.
    phase_time_rank_mean_sec: dict[str, float] = field(default_factory=dict)
    phase_time_share: dict[str, float] = field(default_factory=dict)
    phase_memory_peak_gb: dict[str, float] = field(default_factory=dict)
    mfu_percent: float | None = None
    tflops_per_gpu: float | None = None
    peak_mem_gb: float | None = None
    allocator_retries: int | None = None
    measured_steps: int | None = None
    warmup_steps: int | None = None
    gpu_count: int | None = None
    model_ref: str | None = None
    sample_packing_sequence_len: int | None = None
    data_parallel_replicate_size: int | None = None
    data_parallel_shard_size: int | None = None
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
    skip_param_upcast: bool | None = None
    fsdp_reduce_dtype: str | None = None
    ce_mode: str | None = None
    moe_implementation: str | None = None
    muon_momentum: float | None = None
    muon_update_dtype: str | None = None
    attention_backend: str | None = None
    balanced_routing: bool | None = None
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
    tokens_per_sec_std: float | None = None
    tokens_per_sec_cv: float | None = None
    step_time_sec_std: float | None = None
    step_time_sec_cv: float | None = None
    phase_time_sec: dict[str, float] = field(default_factory=dict)
    phase_time_share: dict[str, float] = field(default_factory=dict)
    phase_memory_peak_gb: dict[str, float] = field(default_factory=dict)
    measured_steps: int | None = None
    warmup_steps: int | None = None
    model_ref: str | None = None
    balanced_routing: bool | None = None
    correctness_status: str | None = None
    cross_model_active_param_ratio: float | None = None
    cross_model_active_param_scale: float | None = None
    cross_model_reference_active_params_b: float | None = None
    cross_model_target_active_params_b: float | None = None
    cross_model_sequence_length_factor: float | None = None
    cross_model_parallelism_factor: float | None = None
    cross_model_memory_factor: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionReport:
    fingerprint: RunFingerprint
    shape: ShapeLedger
    memory: MemoryLedger
    timing: TimingLedger
    support: SimulatorSupportLedger
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
    score_risk_adjusted_tokens_per_gpu_per_sec: float | None
    prediction_uncertainty_fraction: float | None
    prediction_interval_lower_tokens_per_sec: float | None
    prediction_interval_upper_tokens_per_sec: float | None
    risk_adjusted_prediction_interval_lower_tokens_per_sec: float | None
    risk_adjusted_prediction_interval_upper_tokens_per_sec: float | None
    analytic_peak_floor_gb: float | None
    estimated_peak_mem_gb: float | None
    memory_basis: str
    memory_coverage_status: str
    memory_headroom_gb: float | None
    estimated_memory_residual_gb: float | None
    estimated_memory_residual_fraction_of_peak: float | None
    max_ep_rank_slots_per_microbatch: int | None
    phase_bottleneck_phase: str | None
    phase_bottleneck_bucket: str | None
    phase_bottleneck_share: float | None
    phase_bottleneck_time_sec: float | None
    memory_bottleneck_phase: str | None
    memory_bottleneck_bucket: str | None
    memory_bottleneck_peak_gb: float | None
    memory_bottleneck_fraction_of_peak: float | None
    timing_coverage_status: str
    timing_source_label: str | None
    timing_step_time_s: float | None
    timing_forward_backward_s: float | None
    calibration_scope: str
    recommendation: str
    phase_bottleneck_half_speedup_scale: float | None = None
    phase_bottleneck_half_speedup_tokens_per_sec: float | None = None
    phase_bottleneck_half_speedup_delta_pct: float | None = None
    phase_bottleneck_half_speedup_risk_adjusted_tokens_per_sec: float | None = None
    phase_bottleneck_half_speedup_risk_adjusted_delta_pct: float | None = None
    simulator_surface: str = "unknown_config_surface"
    simulator_support_status: str = "unknown"
    simulator_support_blockers: list[str] = field(default_factory=list)
    target_runtime_signature: str = "unknown"
    calibration_distance: float | None = None
    scaling_baseline_label: str | None = None
    scaling_baseline_world_size: int | None = None
    scaling_gpu_ratio: float | None = None
    scaling_speedup: float | None = None
    scaling_efficiency: float | None = None
    risk_adjusted_scaling_speedup: float | None = None
    risk_adjusted_scaling_efficiency: float | None = None
    raw_frontier_member: bool | None = None
    raw_dominated_by_label: str | None = None
    raw_dominance_margin_tokens_per_sec: float | None = None
    raw_dominance_margin_tokens_per_gpu_per_sec: float | None = None
    risk_adjusted_frontier_member: bool | None = None
    risk_adjusted_dominated_by_label: str | None = None
    risk_adjusted_dominance_margin_tokens_per_sec: float | None = None
    risk_adjusted_dominance_margin_tokens_per_gpu_per_sec: float | None = None
    calibration_distance_factors: list[str] = field(default_factory=list)
    memory_calibration_source: str | None = None
    memory_calibration_notes: list[str] = field(default_factory=list)
    memory_ownership_notes: list[str] = field(default_factory=list)
    communication: CommLedger | None = None
    decision_factors: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParallelismAxisComparison:
    axis: str
    varied_dimensions: list[str]
    primary_varied_dimensions: list[str]
    co_varied_axis_dimensions: list[str]
    coupling_status: str
    group_key: str
    candidate_count: int
    raw_best_label: str | None
    raw_best_axis_value: str | None
    raw_best_score_tokens_per_sec: float | None
    raw_worst_label: str | None
    raw_worst_axis_value: str | None
    raw_worst_score_tokens_per_sec: float | None
    raw_spread_tokens_per_sec: float | None
    raw_spread_ratio: float | None
    risk_adjusted_best_label: str | None
    risk_adjusted_best_axis_value: str | None
    risk_adjusted_best_score_tokens_per_sec: float | None
    risk_adjusted_worst_label: str | None
    risk_adjusted_worst_axis_value: str | None
    risk_adjusted_worst_score_tokens_per_sec: float | None
    risk_adjusted_spread_tokens_per_sec: float | None
    risk_adjusted_spread_ratio: float | None
    risk_adjusted_winner_matches_raw: bool | None
    comparison_status: str
    risk_adjusted_best_interval_lower_tokens_per_sec: float | None = None
    risk_adjusted_best_interval_upper_tokens_per_sec: float | None = None
    risk_adjusted_worst_interval_lower_tokens_per_sec: float | None = None
    risk_adjusted_worst_interval_upper_tokens_per_sec: float | None = None
    risk_adjusted_interval_overlap_status: str = "unknown"
    risk_adjusted_interval_overlap_candidate_count: int = 0
    risk_adjusted_interval_overlap_candidate_labels: list[str] = field(default_factory=list)
    risk_adjusted_interval_margin_tokens_per_sec: float | None = None


@dataclass(frozen=True)
class ScenarioParallelismAxisCoverage:
    axis: str
    status: str
    candidate_group_count: int
    candidate_count: int
    scored_count: int
    blocked_count: int
    unscored_count: int
    varied_dimensions: list[str] = field(default_factory=list)
    primary_varied_dimensions: list[str] = field(default_factory=list)
    co_varied_axis_dimensions: list[str] = field(default_factory=list)
    confounded_runtime_dimensions: list[str] = field(default_factory=list)
    feasibility_status_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioBenchmarkSupport:
    support_status: str = "no_benchmark_support"
    support_blockers: list[str] = field(default_factory=list)
    point_count: int = 0
    scored_count: int = 0
    memory_blocked_count: int = 0
    varied_parallelism_dimensions: list[str] = field(default_factory=list)
    varied_workload_dimensions: list[str] = field(default_factory=list)
    varied_runtime_dimensions: list[str] = field(default_factory=list)
    parallelism_axis_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    scored_parallelism_axis_names: list[str] = field(default_factory=list)
    blocked_parallelism_axis_names: list[str] = field(default_factory=list)
    confounded_parallelism_axis_names: list[str] = field(default_factory=list)
    unscored_parallelism_axis_names: list[str] = field(default_factory=list)
    missing_parallelism_axis_names: list[str] = field(default_factory=list)
    point_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioValidationAction:
    action_status: str
    priority: int
    required_measurement: str
    reason_category: str
    candidate_count: int
    candidate_labels: list[str]
    total_gpu_count: int
    max_priority_score: float | None
    max_priority_label: str | None
    max_priority_per_gpu: float | None
    max_priority_per_gpu_label: str | None
    parallelism_axis_names: list[str] = field(default_factory=list)
    reason_statuses: list[str] = field(default_factory=list)
    config_overrides: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioCaptureGap:
    gap_status: str
    priority: int
    required_measurement: str
    reason: str
    blocker_names: list[str]
    candidate_count: int
    scored_count: int
    unscored_count: int
    memory_blocked_count: int
    missing_parallelism_axis_names: list[str] = field(default_factory=list)
    blocked_parallelism_axis_names: list[str] = field(default_factory=list)
    confounded_parallelism_axis_names: list[str] = field(default_factory=list)
    unscored_parallelism_axis_names: list[str] = field(default_factory=list)
    varied_parallelism_dimensions: list[str] = field(default_factory=list)
    varied_workload_dimensions: list[str] = field(default_factory=list)
    varied_runtime_dimensions: list[str] = field(default_factory=list)
    runtime_mismatch_dimensions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioReadiness:
    readiness_status: str = "unknown_scenario_readiness"
    can_capture_scenario: bool = False
    can_predict_scenario_fidelity: bool = False
    can_select_parallelism_tradeoff: bool = False
    can_generalize_model: bool = False
    scenario_capture_status: str = "unknown"
    scenario_capture_blockers: list[str] = field(default_factory=list)
    scenario_prediction_fidelity_status: str = "unknown"
    scenario_prediction_fidelity_blockers: list[str] = field(default_factory=list)
    parallelism_optimality_status: str = "unknown"
    parallelism_optimality_blockers: list[str] = field(default_factory=list)
    model_generalization_status: str = "unknown"
    model_generalization_blockers: list[str] = field(default_factory=list)
    measurement_readiness_status: str = "unknown"
    measurement_portfolio_coverage_status: str = "unknown"
    measurement_portfolio_coverage_blockers: list[str] = field(default_factory=list)
    required_measurements: list[str] = field(default_factory=list)
    scenario_capture_gap_count: int = 0
    scenario_capture_gap_status_counts: dict[str, int] = field(default_factory=dict)
    scenario_capture_gap_required_measurements: list[str] = field(default_factory=list)
    top_scenario_capture_gap_statuses: list[str] = field(default_factory=list)
    validation_action_count: int = 0
    validation_action_status_counts: dict[str, int] = field(default_factory=dict)
    validation_action_required_measurements: list[str] = field(default_factory=list)
    validation_action_total_gpu_count: int = 0
    measurement_candidate_count: int = 0
    measurement_candidate_labels: list[str] = field(default_factory=list)
    measurement_design_config_count: int = 0
    measurement_design_config_labels: list[str] = field(default_factory=list)
    measurement_design_config_filenames: list[str] = field(default_factory=list)
    measurement_portfolio_total_gpu_count: int = 0
    measurement_portfolio_parallelism_axis_gap_names: list[str] = field(default_factory=list)
    measurement_portfolio_cross_model_analog_count: int = 0
    candidate_count: int = 0
    scored_count: int = 0
    unscored_count: int = 0
    memory_blocked_count: int = 0
    unique_parallelism_strategy_count: int = 0
    scored_parallelism_strategy_count: int = 0
    promotable_parallelism_strategy_count: int = 0
    varied_parallelism_dimensions: list[str] = field(default_factory=list)
    varied_workload_dimensions: list[str] = field(default_factory=list)
    varied_runtime_dimensions: list[str] = field(default_factory=list)
    runtime_mismatch_dimensions: list[str] = field(default_factory=list)
    parallelism_axis_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    scored_parallelism_axis_names: list[str] = field(default_factory=list)
    blocked_parallelism_axis_names: list[str] = field(default_factory=list)
    confounded_parallelism_axis_names: list[str] = field(default_factory=list)
    unscored_parallelism_axis_names: list[str] = field(default_factory=list)
    missing_parallelism_axis_names: list[str] = field(default_factory=list)
    simulator_support_status_counts: dict[str, int] = field(default_factory=dict)
    prediction_confidence_counts: dict[str, int] = field(default_factory=dict)
    calibration_scope_counts: dict[str, int] = field(default_factory=dict)
    memory_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    timing_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    cross_model_analog_support_status: str = "not_used"
    cross_model_analog_candidate_count: int = 0
    cross_model_analog_scored_count: int = 0
    benchmark_support: ScenarioBenchmarkSupport = field(default_factory=ScenarioBenchmarkSupport)


@dataclass(frozen=True)
class ScenarioDecisionSummary:
    candidate_count: int
    scored_count: int
    unscored_count: int
    feasible_count: int
    promotable_count: int
    requires_remeasurement_count: int
    memory_blocked_count: int
    unique_parallelism_strategy_count: int
    best_raw_label: str | None
    best_raw_score_tokens_per_sec: float | None
    best_risk_adjusted_label: str | None
    best_risk_adjusted_score_tokens_per_sec: float | None
    best_efficiency_label: str | None
    best_efficiency_score_tokens_per_gpu_per_sec: float | None
    best_risk_adjusted_efficiency_label: str | None
    best_risk_adjusted_efficiency_score_tokens_per_gpu_per_sec: float | None
    best_next_measurement_label: str | None
    best_next_measurement_score_tokens_per_sec: float | None
    best_promotable_label: str | None
    candidate_model_ref_counts: dict[str, int] = field(default_factory=dict)
    scored_model_ref_counts: dict[str, int] = field(default_factory=dict)
    candidate_world_size_counts: dict[int, int] = field(default_factory=dict)
    scored_world_size_counts: dict[int, int] = field(default_factory=dict)
    candidate_sequence_length_counts: dict[int, int] = field(default_factory=dict)
    scored_sequence_length_counts: dict[int, int] = field(default_factory=dict)
    candidate_global_batch_size_counts: dict[int, int] = field(default_factory=dict)
    scored_global_batch_size_counts: dict[int, int] = field(default_factory=dict)
    best_promotable_score_tokens_per_sec: float | None = None
    best_promotable_score_risk_adjusted_tokens_per_sec: float | None = None
    promotable_raw_gap_tokens_per_sec: float | None = None
    promotable_raw_gap_percentage: float | None = None
    promotable_risk_adjusted_gap_tokens_per_sec: float | None = None
    promotable_risk_adjusted_gap_percentage: float | None = None
    promotion_readiness_status: str = "unknown"
    throughput_efficiency_frontier_labels: list[str] = field(default_factory=list)
    risk_adjusted_efficiency_frontier_labels: list[str] = field(default_factory=list)
    throughput_efficiency_frontier_count: int = 0
    risk_adjusted_efficiency_frontier_count: int = 0
    raw_dominated_candidate_count: int = 0
    risk_adjusted_dominated_candidate_count: int = 0
    throughput_efficiency_tradeoff_status: str = "unknown"
    same_workload_scaling_status: str = "unknown"
    same_workload_scaling_group_count: int = 0
    same_workload_scaling_candidate_count: int = 0
    best_scaling_efficiency_label: str | None = None
    best_scaling_efficiency: float | None = None
    mean_scaling_efficiency: float | None = None
    min_scaling_efficiency: float | None = None
    best_risk_adjusted_scaling_efficiency_label: str | None = None
    best_risk_adjusted_scaling_efficiency: float | None = None
    mean_risk_adjusted_scaling_efficiency: float | None = None
    min_risk_adjusted_scaling_efficiency: float | None = None
    measurement_readiness_status: str = "unknown"
    measurement_rationale: list[str] = field(default_factory=list)
    measurement_candidate_count: int = 0
    measurement_candidate_labels: list[str] = field(default_factory=list)
    measurement_candidate_reasons: dict[str, list[str]] = field(default_factory=dict)
    measurement_candidate_priority_scores: dict[str, float] = field(default_factory=dict)
    measurement_candidate_priority_per_gpu: dict[str, float] = field(default_factory=dict)
    measurement_candidate_cost_gpus: dict[str, int] = field(default_factory=dict)
    measurement_candidate_priority_factors: dict[str, list[str]] = field(default_factory=dict)
    measurement_candidate_config_overrides: dict[str, list[str]] = field(default_factory=dict)
    measurement_design_config_count: int = 0
    measurement_design_config_labels: list[str] = field(default_factory=list)
    measurement_design_config_filenames: list[str] = field(default_factory=list)
    measurement_portfolio_total_gpu_count: int = 0
    measurement_portfolio_max_priority_score: float | None = None
    measurement_portfolio_max_priority_label: str | None = None
    measurement_portfolio_max_priority_per_gpu: float | None = None
    measurement_portfolio_max_priority_per_gpu_label: str | None = None
    measurement_portfolio_coverage_status: str = "unknown"
    measurement_portfolio_coverage_blockers: list[str] = field(default_factory=list)
    measurement_portfolio_reason_category_counts: dict[str, int] = field(default_factory=dict)
    measurement_portfolio_parallelism_axis_gap_names: list[str] = field(default_factory=list)
    measurement_portfolio_cross_model_analog_count: int = 0
    validation_action_count: int = 0
    validation_action_status_counts: dict[str, int] = field(default_factory=dict)
    validation_action_required_measurements: list[str] = field(default_factory=list)
    validation_action_total_gpu_count: int = 0
    validation_actions: list[ScenarioValidationAction] = field(default_factory=list)
    max_calibration_distance: float | None = None
    max_calibration_distance_label: str | None = None
    mean_scored_calibration_distance: float | None = None
    high_uncertainty_candidate_count: int = 0
    max_prediction_uncertainty_fraction: float | None = None
    max_prediction_uncertainty_fraction_label: str | None = None
    mean_scored_prediction_uncertainty_fraction: float | None = None
    risk_adjusted_interval_overlap_status: str = "unknown"
    risk_adjusted_interval_overlap_contender_count: int = 0
    risk_adjusted_interval_overlap_contender_labels: list[str] = field(default_factory=list)
    risk_adjusted_interval_best_vs_next_margin_tokens_per_sec: float | None = None
    parallelism_tradeoff_status: str = "unknown"
    parallelism_optimality_status: str = "unknown"
    parallelism_optimality_blockers: list[str] = field(default_factory=list)
    scored_parallelism_strategy_count: int = 0
    promotable_parallelism_strategy_count: int = 0
    requires_remeasurement_parallelism_strategy_count: int = 0
    parallelism_axis_comparison_count: int = 0
    isolated_parallelism_axis_comparison_count: int = 0
    coupled_parallelism_axis_comparison_count: int = 0
    parallelism_axis_interval_overlap_count: int = 0
    parallelism_axis_comparisons: list[ParallelismAxisComparison] = field(default_factory=list)
    scored_parallelism_axis_names: list[str] = field(default_factory=list)
    blocked_parallelism_axis_names: list[str] = field(default_factory=list)
    confounded_parallelism_axis_names: list[str] = field(default_factory=list)
    unscored_parallelism_axis_names: list[str] = field(default_factory=list)
    missing_parallelism_axis_names: list[str] = field(default_factory=list)
    parallelism_axis_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    parallelism_axis_coverage: list[ScenarioParallelismAxisCoverage] = field(default_factory=list)
    parallelism_boundary_status: str = "unknown"
    parallelism_boundary_prediction_status: str = "unknown"
    parallelism_boundary_prediction_blockers: list[str] = field(default_factory=list)
    parallelism_boundary_group_count: int = 0
    parallelism_boundary_candidate_count: int = 0
    parallelism_boundary_fit_count: int = 0
    parallelism_boundary_failure_count: int = 0
    parallelism_boundary_best_fit_label: str | None = None
    parallelism_boundary_confounded_dimensions: list[str] = field(default_factory=list)
    parallelism_boundary_measured_axis_names: list[str] = field(default_factory=list)
    parallelism_boundary_confounded_axis_names: list[str] = field(default_factory=list)
    parallelism_boundary_missing_axis_names: list[str] = field(default_factory=list)
    parallelism_boundary_axis_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    parallelism_boundary_axis_coverage: list[ParallelismBoundaryAxisCoverage] = field(default_factory=list)
    parallelism_boundary_groups: list[ParallelismBoundaryGroup] = field(default_factory=list)
    cross_model_analog_support_status: str = "not_used"
    cross_model_analog_candidate_count: int = 0
    cross_model_analog_scored_count: int = 0
    cross_model_analog_unique_prediction_count: int = 0
    cross_model_analog_unique_matched_label_count: int = 0
    cross_model_analog_unique_target_strategy_count: int = 0
    cross_model_analog_unique_target_runtime_signature_count: int = 0
    cross_model_analog_scored_varied_parallelism_dimensions: list[str] = field(default_factory=list)
    cross_model_analog_scored_varied_workload_dimensions: list[str] = field(default_factory=list)
    cross_model_analog_factor_status: str = "not_used"
    cross_model_analog_unique_factor_count: int = 0
    cross_model_analog_factor_ranges: dict[str, list[float]] = field(default_factory=dict)
    cross_model_analog_prediction_interval_top_count: int = 0
    cross_model_analog_prediction_interval_top_fraction: float | None = None
    cross_model_analog_prediction_interval_top_labels: list[str] = field(default_factory=list)
    cross_model_analog_prediction_interval_selectivity_status: str = "not_used"
    model_generalization_status: str = "unknown"
    model_generalization_blockers: list[str] = field(default_factory=list)
    scenario_capture_status: str = "unknown"
    scenario_capture_blockers: list[str] = field(default_factory=list)
    scenario_capture_gap_count: int = 0
    scenario_capture_gap_status_counts: dict[str, int] = field(default_factory=dict)
    scenario_capture_gap_required_measurements: list[str] = field(default_factory=list)
    scenario_capture_gaps: list[ScenarioCaptureGap] = field(default_factory=list)
    benchmark_support: ScenarioBenchmarkSupport = field(default_factory=ScenarioBenchmarkSupport)
    scenario_prediction_fidelity_status: str = "unknown"
    scenario_prediction_fidelity_blockers: list[str] = field(default_factory=list)
    varied_parallelism_dimensions: list[str] = field(default_factory=list)
    varied_workload_dimensions: list[str] = field(default_factory=list)
    varied_runtime_dimensions: list[str] = field(default_factory=list)
    runtime_mismatch_dimensions: list[str] = field(default_factory=list)
    candidate_runtime_signature_counts: dict[str, int] = field(default_factory=dict)
    scored_runtime_signature_counts: dict[str, int] = field(default_factory=dict)
    prediction_confidence_counts: dict[str, int] = field(default_factory=dict)
    calibration_scope_counts: dict[str, int] = field(default_factory=dict)
    memory_basis_counts: dict[str, int] = field(default_factory=dict)
    memory_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    simulator_support_status_counts: dict[str, int] = field(default_factory=dict)
    simulator_support_blocker_counts: dict[str, int] = field(default_factory=dict)
    max_estimated_memory_residual_gb: float | None = None
    max_estimated_memory_residual_gb_label: str | None = None
    max_estimated_memory_residual_fraction_of_peak: float | None = None
    max_estimated_memory_residual_fraction_of_peak_label: str | None = None
    phase_bottleneck_candidate_count: int = 0
    phase_bottleneck_bucket_counts: dict[str, int] = field(default_factory=dict)
    phase_bottleneck_phase_counts: dict[str, int] = field(default_factory=dict)
    max_phase_bottleneck_share: float | None = None
    max_phase_bottleneck_share_label: str | None = None
    max_phase_bottleneck_phase: str | None = None
    max_phase_bottleneck_bucket: str | None = None
    phase_bottleneck_half_speedup_candidate_count: int = 0
    max_phase_bottleneck_half_speedup_delta_pct: float | None = None
    max_phase_bottleneck_half_speedup_delta_label: str | None = None
    max_phase_bottleneck_half_speedup_phase: str | None = None
    max_phase_bottleneck_half_speedup_bucket: str | None = None
    memory_bottleneck_candidate_count: int = 0
    memory_bottleneck_bucket_counts: dict[str, int] = field(default_factory=dict)
    memory_bottleneck_phase_counts: dict[str, int] = field(default_factory=dict)
    max_memory_bottleneck_fraction_of_peak: float | None = None
    max_memory_bottleneck_fraction_label: str | None = None
    max_memory_bottleneck_phase: str | None = None
    max_memory_bottleneck_bucket: str | None = None
    timing_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    cross_node_dimension_counts: dict[str, int] = field(default_factory=dict)
    feasibility_status_counts: dict[str, int] = field(default_factory=dict)
    routing_regime_status: str = "unknown"
    routing_regime_counts: dict[str, int] = field(default_factory=dict)
    recommendation_counts: dict[str, int] = field(default_factory=dict)
    risk_flag_counts: dict[str, int] = field(default_factory=dict)
    scenario_readiness: ScenarioReadiness = field(default_factory=ScenarioReadiness)


@dataclass(frozen=True)
class ScenarioMeasurementConfig:
    label: str
    filename: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ScenarioReport:
    base_config_path: str
    benchmark_dir: str | None
    device_memory_limit_gb: float
    memory_safety_factor: float
    topology_sweep: str
    balanced_routing: bool
    world_sizes: list[int]
    candidate_count: int
    feasible_count: int
    best_raw: ScenarioCandidate | None
    best_risk_adjusted: ScenarioCandidate | None
    best_efficiency: ScenarioCandidate | None
    best_risk_adjusted_efficiency: ScenarioCandidate | None
    best_next_measurement: ScenarioCandidate | None
    best_promotable: ScenarioCandidate | None
    decision_summary: ScenarioDecisionSummary
    candidates: list[ScenarioCandidate]
    planner_context: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    supplemental_benchmark_dirs: list[str] = field(default_factory=list)
    analog_benchmark_dirs: list[str] = field(default_factory=list)
    primary_behavior_point_count: int = 0
    supplemental_behavior_point_count: int = 0
    analog_behavior_point_count: int = 0
    total_behavior_point_count: int = 0


@dataclass(frozen=True)
class CalibrationHoldout:
    label: str
    source: str
    topology_label: str
    actual_tokens_per_sec: float | None
    predicted_tokens_per_sec: float | None
    prediction_uncertainty_fraction: float | None
    prediction_interval_lower_tokens_per_sec: float | None
    prediction_interval_upper_tokens_per_sec: float | None
    actual_tokens_in_prediction_interval: bool | None
    actual_peak_mem_gb: float | None
    predicted_peak_mem_gb: float | None
    prediction_status: str
    matched_label: str | None
    absolute_error_tokens_per_sec: float | None
    absolute_percentage_error: float | None
    analytic_peak_floor_gb: float | None
    memory_prediction_basis: str
    memory_coverage_status: str
    memory_feasibility_status: str
    predicted_memory_residual_gb: float | None
    predicted_memory_residual_fraction_of_peak: float | None
    actual_memory_residual_gb: float | None
    actual_memory_residual_fraction_of_peak: float | None
    memory_absolute_error_gb: float | None
    memory_absolute_percentage_error: float | None
    actual_memory_bottleneck_phase: str | None
    actual_memory_bottleneck_bucket: str | None
    actual_memory_bottleneck_peak_gb: float | None
    actual_memory_bottleneck_fraction_of_peak: float | None
    predicted_memory_bottleneck_phase: str | None
    predicted_memory_bottleneck_bucket: str | None
    predicted_memory_bottleneck_peak_gb: float | None
    predicted_memory_bottleneck_fraction_of_peak: float | None
    memory_bottleneck_phase_match: bool | None
    memory_bottleneck_bucket_match: bool | None
    memory_bottleneck_peak_absolute_error_gb: float | None
    memory_bottleneck_fraction_absolute_error: float | None
    actual_phase_bottleneck_phase: str | None
    actual_phase_bottleneck_bucket: str | None
    actual_phase_bottleneck_share: float | None
    predicted_phase_bottleneck_phase: str | None
    predicted_phase_bottleneck_bucket: str | None
    predicted_phase_bottleneck_share: float | None
    phase_bottleneck_phase_match: bool | None
    phase_bottleneck_bucket_match: bool | None
    phase_bottleneck_share_absolute_error: float | None
    actual_phase_top3: list[str]
    predicted_phase_top3: list[str]
    actual_phase_bucket_top3: list[str]
    predicted_phase_bucket_top3: list[str]
    phase_top3_overlap_count: int | None
    phase_top3_overlap_rate: float | None
    phase_bucket_top3_overlap_count: int | None
    phase_bucket_top3_overlap_rate: float | None
    memory_calibration_source: str | None
    calibrated_from_count: int
    memory_calibration_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationValidationGap:
    gap_status: str
    priority: int
    required_measurement: str
    reason: str
    affected_holdout_count: int
    affected_holdout_labels: list[str]
    blocker_names: list[str]
    max_absolute_percentage_error: float | None = None
    max_absolute_percentage_error_label: str | None = None
    max_memory_absolute_error_gb: float | None = None
    max_memory_absolute_error_label: str | None = None
    max_phase_bottleneck_share_absolute_error: float | None = None
    max_phase_bottleneck_share_absolute_error_label: str | None = None
    missing_memory_count: int = 0
    missing_phase_bottleneck_count: int = 0


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
    max_absolute_percentage_error_label: str | None
    max_absolute_percentage_error_prediction_status: str | None
    max_absolute_percentage_error_in_prediction_interval: bool | None
    prediction_interval_coverage_count: int
    prediction_interval_coverage_rate: float | None
    mean_prediction_uncertainty_fraction: float | None
    max_prediction_uncertainty_fraction: float | None
    max_prediction_uncertainty_label: str | None
    memory_evaluated_count: int
    mean_memory_absolute_error_gb: float | None
    max_memory_absolute_error_gb: float | None
    max_memory_absolute_error_label: str | None
    mean_memory_absolute_percentage_error: float | None
    max_memory_absolute_percentage_error: float | None
    max_memory_absolute_percentage_error_label: str | None
    memory_prediction_basis_counts: dict[str, int]
    memory_coverage_status_counts: dict[str, int]
    memory_feasibility_status_counts: dict[str, int]
    max_predicted_memory_residual_gb: float | None
    max_predicted_memory_residual_gb_label: str | None
    max_predicted_memory_residual_fraction_of_peak: float | None
    max_predicted_memory_residual_fraction_of_peak_label: str | None
    max_actual_memory_residual_gb: float | None
    max_actual_memory_residual_gb_label: str | None
    max_actual_memory_residual_fraction_of_peak: float | None
    max_actual_memory_residual_fraction_of_peak_label: str | None
    memory_bottleneck_evaluated_count: int
    memory_bottleneck_phase_match_count: int
    memory_bottleneck_phase_match_rate: float | None
    memory_bottleneck_bucket_match_count: int
    memory_bottleneck_bucket_match_rate: float | None
    mean_memory_bottleneck_peak_absolute_error_gb: float | None
    max_memory_bottleneck_peak_absolute_error_gb: float | None
    max_memory_bottleneck_peak_absolute_error_label: str | None
    mean_memory_bottleneck_fraction_absolute_error: float | None
    max_memory_bottleneck_fraction_absolute_error: float | None
    max_memory_bottleneck_fraction_absolute_error_label: str | None
    memory_bottleneck_phase_mismatch_labels: list[str]
    memory_bottleneck_bucket_mismatch_labels: list[str]
    phase_bottleneck_evaluated_count: int
    phase_bottleneck_phase_match_count: int
    phase_bottleneck_phase_match_rate: float | None
    phase_bottleneck_bucket_match_count: int
    phase_bottleneck_bucket_match_rate: float | None
    mean_phase_bottleneck_share_absolute_error: float | None
    max_phase_bottleneck_share_absolute_error: float | None
    max_phase_bottleneck_share_absolute_error_label: str | None
    phase_bottleneck_phase_mismatch_labels: list[str]
    phase_bottleneck_bucket_mismatch_labels: list[str]
    phase_top3_evaluated_count: int
    mean_phase_top3_overlap_rate: float | None
    min_phase_top3_overlap_rate: float | None
    min_phase_top3_overlap_rate_label: str | None
    mean_phase_bucket_top3_overlap_rate: float | None
    min_phase_bucket_top3_overlap_rate: float | None
    min_phase_bucket_top3_overlap_rate_label: str | None
    calibration_fidelity_status: str
    calibration_fidelity_blockers: list[str]
    calibration_validation_gap_count: int
    calibration_validation_gap_status_counts: dict[str, int]
    calibration_validation_gap_required_measurements: list[str]
    calibration_validation_gaps: list[CalibrationValidationGap]
    prediction_status_counts: dict[str, int]
    holdouts: list[CalibrationHoldout]
    warnings: list[str] = field(default_factory=list)
    prediction_uncertainty_calibration_status: str = "not_evaluated"
    mean_empirical_required_uncertainty_fraction: float | None = None
    max_empirical_required_uncertainty_fraction: float | None = None
    max_empirical_required_uncertainty_label: str | None = None
    measurement_design_config_count: int = 0
    measurement_design_config_labels: list[str] = field(default_factory=list)
    measurement_design_config_filenames: list[str] = field(default_factory=list)
    calibration_support_benchmark_dirs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeasibilityHoldout:
    label: str
    source: str
    topology_label: str
    actual_outcome: str
    predicted_outcome: str
    actual_tokens_per_sec: float | None
    actual_peak_mem_gb: float | None
    predicted_tokens_per_sec: float | None
    predicted_peak_mem_gb: float | None
    prediction_status: str
    matched_label: str | None
    memory_prediction_basis: str
    analytic_peak_floor_gb: float | None
    memory_coverage_status: str
    predicted_memory_residual_gb: float | None
    predicted_memory_residual_fraction_of_peak: float | None
    actual_memory_residual_gb: float | None
    actual_memory_residual_fraction_of_peak: float | None
    memory_calibration_source: str | None
    predicted_feasibility_status: str
    classified_correctly: bool
    calibrated_from_count: int
    memory_calibration_notes: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeasibilityReport:
    base_config_path: str
    benchmark_dir: str
    status: str
    observed_point_count: int
    evaluated_count: int
    skipped_count: int
    actual_fit_count: int
    actual_oom_count: int
    predicted_fit_count: int
    predicted_blocked_count: int
    predicted_unknown_count: int
    correct_count: int
    false_fit_count: int
    false_blocked_count: int
    accuracy: float | None
    fit_recall: float | None
    oom_recall: float | None
    prediction_status_counts: dict[str, int]
    memory_prediction_basis_counts: dict[str, int]
    memory_coverage_status_counts: dict[str, int]
    feasibility_status_counts: dict[str, int]
    max_predicted_memory_residual_gb: float | None
    max_predicted_memory_residual_gb_label: str | None
    max_predicted_memory_residual_fraction_of_peak: float | None
    max_predicted_memory_residual_fraction_of_peak_label: str | None
    max_actual_memory_residual_gb: float | None
    max_actual_memory_residual_gb_label: str | None
    max_actual_memory_residual_fraction_of_peak: float | None
    max_actual_memory_residual_fraction_of_peak_label: str | None
    risk_flag_counts: dict[str, int]
    holdouts: list[FeasibilityHoldout]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalogHoldout:
    label: str
    source: str
    target_model_ref: str | None
    target_sequence_len: int | None
    target_runtime_signature: str
    topology_label: str
    target_world_size: int
    actual_tokens_per_sec: float
    actual_tokens_per_gpu_per_sec: float
    predicted_tokens_per_sec: float | None
    predicted_tokens_per_gpu_per_sec: float | None
    prediction_uncertainty_fraction: float | None
    prediction_interval_lower_tokens_per_sec: float | None
    prediction_interval_upper_tokens_per_sec: float | None
    actual_tokens_in_prediction_interval: bool | None
    prediction_status: str
    matched_label: str | None
    matched_model_ref: str | None
    matched_sequence_len: int | None
    matched_topology_label: str | None
    calibration_distance: float | None
    calibration_distance_factors: list[str]
    absolute_error_tokens_per_sec: float | None
    absolute_percentage_error: float | None
    analog_point_count: int
    cross_model_active_param_ratio: float | None = None
    cross_model_active_param_scale: float | None = None
    cross_model_reference_active_params_b: float | None = None
    cross_model_target_active_params_b: float | None = None
    cross_model_sequence_length_factor: float | None = None
    cross_model_parallelism_factor: float | None = None
    cross_model_memory_factor: float | None = None
    actual_tflops_per_gpu: float | None = None
    matched_analog_tflops_per_gpu: float | None = None
    # measured target tflops / measured analog tflops: the MFU-regime transfer ratio (None unless
    # both rows logged tflops). Far from 1.0 => the equal-MFU analog assumption is measured-false.
    mfu_regime_ratio: float | None = None
    nearest_analog_label: str | None = None
    nearest_analog_model_ref: str | None = None
    nearest_analog_sequence_len: int | None = None
    nearest_analog_topology_label: str | None = None
    nearest_analog_sequence_length_factor: float | None = None
    nearest_analog_runtime_mismatch_count: int | None = None
    nearest_analog_runtime_mismatches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    target_local_world_size: int | None = None
    target_micro_batch_size: int | None = None
    target_gradient_accumulation_steps: int | None = None
    target_global_batch_size: int | None = None
    target_data_parallel_replicate_size: int | None = None
    target_data_parallel_shard_size: int | None = None
    target_tensor_parallel_size: int | None = None
    target_pipeline_parallel_size: int | None = None
    target_expert_parallel_size: int | None = None
    target_ep_fsdp_size: int | None = None
    target_ulysses_parallel_size: int | None = None
    target_ringattn_parallel_size: int | None = None


@dataclass(frozen=True)
class CrossModelFactorCoverage:
    factor: str
    status: str
    scored_candidate_count: int
    observed_value_count: int
    missing_value_count: int
    unique_value_count: int
    min_value: float | None = None
    max_value: float | None = None
    values: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AnalogDecisionSummary:
    status: str
    scored_candidate_count: int
    actual_best_label: str | None
    predicted_best_label: str | None
    actual_best_tokens_per_sec: float | None
    selected_actual_tokens_per_sec: float | None
    selected_regret_tokens_per_sec: float | None
    selected_regret_percentage: float | None
    selected_is_actual_best: bool | None
    actual_best_in_predicted_top_tie: bool | None
    predicted_top_tie_count: int
    top_tie_best_actual_tokens_per_sec: float | None
    top_tie_regret_tokens_per_sec: float | None
    top_tie_regret_percentage: float | None
    actual_best_in_prediction_interval_top: bool | None
    prediction_interval_top_labels: list[str]
    prediction_interval_top_count: int
    prediction_interval_top_fraction: float | None
    prediction_interval_selectivity_status: str
    interval_top_best_actual_tokens_per_sec: float | None
    interval_top_regret_tokens_per_sec: float | None
    interval_top_regret_percentage: float | None
    pairwise_ordering_pair_count: int
    pairwise_ordering_correct_count: int
    pairwise_ordering_accuracy: float | None
    mean_absolute_rank_error: float | None
    max_absolute_rank_error: int | None
    max_absolute_rank_error_label: str | None
    actual_efficiency_best_label: str | None
    predicted_efficiency_best_label: str | None
    actual_efficiency_best_tokens_per_gpu_per_sec: float | None
    selected_efficiency_actual_tokens_per_gpu_per_sec: float | None
    efficiency_regret_tokens_per_gpu_per_sec: float | None
    efficiency_regret_percentage: float | None
    efficiency_selected_is_actual_best: bool | None
    efficiency_pairwise_ordering_pair_count: int
    efficiency_pairwise_ordering_correct_count: int
    efficiency_pairwise_ordering_accuracy: float | None
    efficiency_mean_absolute_rank_error: float | None
    efficiency_max_absolute_rank_error: int | None
    efficiency_max_absolute_rank_error_label: str | None
    analog_support_status: str = "unknown"
    cross_model_prediction_status: str = "unknown"
    cross_model_prediction_blockers: list[str] = field(default_factory=list)
    analog_generalization_scope_status: str = "unknown"
    analog_generalization_scope_blockers: list[str] = field(default_factory=list)
    larger_model_generalization_status: str = "unknown"
    larger_model_generalization_blockers: list[str] = field(default_factory=list)
    scored_unique_prediction_count: int = 0
    scored_unique_matched_label_count: int = 0
    scored_unique_matched_topology_count: int = 0
    scored_unique_target_topology_count: int = 0
    scored_unique_target_runtime_signature_count: int = 0
    scored_cross_model_factor_status: str = "unknown"
    scored_unique_cross_model_factor_count: int = 0
    scored_cross_model_factor_ranges: dict[str, list[float]] = field(default_factory=dict)
    scored_cross_model_factor_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    scored_cross_model_factor_coverage: list[CrossModelFactorCoverage] = field(default_factory=list)
    scored_varied_cross_model_factor_names: list[str] = field(default_factory=list)
    scored_degenerate_cross_model_factor_names: list[str] = field(default_factory=list)
    scored_missing_cross_model_factor_names: list[str] = field(default_factory=list)
    larger_model_generalization_readiness: LargerModelGeneralizationReadiness = field(
        default_factory=lambda: LargerModelGeneralizationReadiness()
    )


@dataclass(frozen=True)
class AnalogValidationGap:
    gap_status: str
    priority: int
    required_measurement: str
    reason: str
    affected_target_count: int
    affected_target_labels: list[str]
    matched_analog_label_count: int
    matched_analog_labels: list[str]
    matched_analog_model_refs: list[str]
    matched_analog_sequence_lengths: list[int]
    nearest_analog_label_count: int
    nearest_analog_labels: list[str]
    nearest_analog_model_refs: list[str]
    nearest_analog_sequence_lengths: list[int]
    nearest_analog_sequence_length_factors: list[float]
    nearest_analog_runtime_mismatches: list[str]
    target_sequence_lengths: list[int]
    analog_sequence_lengths: list[int]
    target_workload_dimensions: list[str]
    target_runtime_dimensions: list[str]
    target_parallelism_dimensions: list[str]
    cross_model_factor_ranges: dict[str, list[float]]
    max_calibration_distance: float | None
    max_calibration_distance_label: str | None
    analog_support_status: str
    cross_model_prediction_status: str
    analog_generalization_scope_status: str
    blocker_names: list[str]
    mfu_regime_ratios: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class LargerModelGeneralizationReadiness:
    readiness_status: str = "unknown_larger_model_generalization_readiness"
    can_generalize_to_larger_model: bool = False
    support_status: str = "unknown_larger_model_generalization_support"
    support_blockers: list[str] = field(default_factory=list)
    analog_support_status: str = "unknown"
    cross_model_prediction_status: str = "unknown"
    cross_model_prediction_blockers: list[str] = field(default_factory=list)
    analog_generalization_scope_status: str = "unknown"
    analog_generalization_scope_blockers: list[str] = field(default_factory=list)
    required_measurements: list[str] = field(default_factory=list)
    validation_gap_count: int = 0
    validation_gap_status_counts: dict[str, int] = field(default_factory=dict)
    top_validation_gap_statuses: list[str] = field(default_factory=list)
    top_validation_gap_required_measurements: list[str] = field(default_factory=list)
    top_validation_gap_target_labels: list[str] = field(default_factory=list)
    top_validation_gap_nearest_analog_labels: list[str] = field(default_factory=list)
    top_validation_gap_nearest_analog_model_refs: list[str] = field(default_factory=list)
    top_validation_gap_nearest_analog_sequence_lengths: list[int] = field(default_factory=list)
    top_validation_gap_nearest_analog_sequence_length_factors: list[float] = field(default_factory=list)
    top_validation_gap_nearest_analog_runtime_mismatches: list[str] = field(default_factory=list)
    unscored_target_nearest_analog_labels: list[str] = field(default_factory=list)
    unscored_target_nearest_analog_model_refs: list[str] = field(default_factory=list)
    unscored_target_nearest_analog_sequence_lengths: list[int] = field(default_factory=list)
    unscored_target_nearest_analog_sequence_length_factors: list[float] = field(default_factory=list)
    unscored_target_nearest_analog_runtime_mismatches: list[str] = field(default_factory=list)
    unscored_target_context_gap_summaries: list[dict[str, Any]] = field(default_factory=list)
    measurement_design_config_count: int = 0
    measurement_design_config_labels: list[str] = field(default_factory=list)
    measurement_design_config_filenames: list[str] = field(default_factory=list)
    evaluated_count: int = 0
    scored_candidate_count: int = 0
    unscored_count: int = 0
    analog_point_count: int = 0
    scored_unique_prediction_count: int = 0
    scored_unique_matched_label_count: int = 0
    scored_unique_matched_topology_count: int = 0
    scored_unique_target_topology_count: int = 0
    scored_unique_target_runtime_signature_count: int = 0
    scored_cross_model_factor_status: str = "unknown"
    scored_unique_cross_model_factor_count: int = 0
    scored_cross_model_factor_ranges: dict[str, list[float]] = field(default_factory=dict)
    scored_cross_model_factor_coverage_status_counts: dict[str, int] = field(default_factory=dict)
    scored_cross_model_factor_coverage: list[CrossModelFactorCoverage] = field(default_factory=list)
    scored_varied_cross_model_factor_names: list[str] = field(default_factory=list)
    scored_degenerate_cross_model_factor_names: list[str] = field(default_factory=list)
    scored_missing_cross_model_factor_names: list[str] = field(default_factory=list)
    scored_varied_target_workload_dimensions: list[str] = field(default_factory=list)
    scored_varied_target_runtime_dimensions: list[str] = field(default_factory=list)
    scored_varied_target_parallelism_dimensions: list[str] = field(default_factory=list)
    target_model_refs: list[str] = field(default_factory=list)
    analog_model_refs: list[str] = field(default_factory=list)
    target_sequence_lengths: list[int] = field(default_factory=list)
    analog_sequence_lengths: list[int] = field(default_factory=list)
    selected_is_actual_best: bool | None = None
    actual_best_in_predicted_top_tie: bool | None = None
    prediction_interval_top_fraction: float | None = None
    prediction_interval_selectivity_status: str = "unknown"
    pairwise_ordering_accuracy: float | None = None
    efficiency_selected_is_actual_best: bool | None = None
    efficiency_pairwise_ordering_accuracy: float | None = None
    max_absolute_percentage_error: float | None = None
    prediction_interval_coverage_rate: float | None = None


@dataclass(frozen=True)
class AnalogReport:
    base_config_path: str
    benchmark_dir: str
    analog_benchmark_dirs: list[str]
    status: str
    coverage_status: str
    measured_point_count: int
    evaluated_count: int
    unscored_count: int
    skipped_count: int
    target_coverage_fraction: float | None
    analog_point_count: int
    analog_support_status: str
    cross_model_prediction_status: str
    cross_model_prediction_blockers: list[str]
    analog_generalization_scope_status: str
    analog_generalization_scope_blockers: list[str]
    larger_model_generalization_status: str
    larger_model_generalization_blockers: list[str]
    analog_validation_gap_count: int
    analog_validation_gap_status_counts: dict[str, int]
    analog_validation_gap_required_measurements: list[str]
    analog_validation_gaps: list[AnalogValidationGap]
    scored_unique_prediction_count: int
    scored_unique_matched_label_count: int
    scored_unique_matched_topology_count: int
    scored_unique_target_topology_count: int
    scored_unique_target_runtime_signature_count: int
    scored_cross_model_factor_status: str
    scored_unique_cross_model_factor_count: int
    scored_cross_model_factor_ranges: dict[str, list[float]]
    scored_cross_model_factor_coverage_status_counts: dict[str, int]
    scored_cross_model_factor_coverage: list[CrossModelFactorCoverage]
    scored_varied_cross_model_factor_names: list[str]
    scored_degenerate_cross_model_factor_names: list[str]
    scored_missing_cross_model_factor_names: list[str]
    scored_varied_target_workload_dimensions: list[str]
    scored_varied_target_runtime_dimensions: list[str]
    scored_varied_target_parallelism_dimensions: list[str]
    target_model_refs: list[str]
    analog_model_refs: list[str]
    target_sequence_lengths: list[int]
    evaluated_target_sequence_lengths: list[int]
    unscored_target_sequence_lengths: list[int]
    unscored_target_labels: list[str]
    unscored_target_sequence_length_counts: dict[int, int]
    unscored_target_reason_counts: dict[str, int]
    unscored_target_nearest_analog_labels: list[str]
    unscored_target_nearest_analog_model_refs: list[str]
    unscored_target_nearest_analog_sequence_lengths: list[int]
    unscored_target_nearest_analog_sequence_length_factors: list[float]
    unscored_target_nearest_analog_runtime_mismatches: list[str]
    unscored_target_context_gap_summaries: list[dict[str, Any]]
    analog_sequence_lengths: list[int]
    mean_absolute_percentage_error: float | None
    median_absolute_percentage_error: float | None
    max_absolute_percentage_error: float | None
    max_absolute_percentage_error_label: str | None
    max_absolute_percentage_error_prediction_status: str | None
    max_absolute_percentage_error_in_prediction_interval: bool | None
    mean_scored_calibration_distance: float | None
    max_calibration_distance: float | None
    max_calibration_distance_label: str | None
    prediction_interval_coverage_count: int
    prediction_interval_coverage_rate: float | None
    prediction_interval_top_fraction: float | None
    prediction_interval_selectivity_status: str
    mean_prediction_uncertainty_fraction: float | None
    max_prediction_uncertainty_fraction: float | None
    max_prediction_uncertainty_label: str | None
    prediction_status_counts: dict[str, int]
    decision_summary: AnalogDecisionSummary
    holdouts: list[AnalogHoldout]
    larger_model_generalization_readiness: LargerModelGeneralizationReadiness = field(
        default_factory=LargerModelGeneralizationReadiness
    )
    warnings: list[str] = field(default_factory=list)
    supplemental_benchmark_dirs: list[str] = field(default_factory=list)
    primary_target_point_count: int = 0
    supplemental_target_point_count: int = 0
    primary_measured_point_count: int = 0
    supplemental_measured_point_count: int = 0


@dataclass(frozen=True)
class DecisionCandidatePrediction:
    label: str
    source: str
    topology_label: str
    actual_tokens_per_sec: float
    actual_tokens_per_gpu_per_sec: float | None
    predicted_tokens_per_sec: float | None
    predicted_tokens_per_gpu_per_sec: float | None
    predicted_risk_adjusted_tokens_per_sec: float | None
    predicted_risk_adjusted_tokens_per_gpu_per_sec: float | None
    prediction_uncertainty_fraction: float | None
    predicted_interval_lower_tokens_per_sec: float | None
    predicted_interval_upper_tokens_per_sec: float | None
    predicted_risk_adjusted_interval_lower_tokens_per_sec: float | None
    predicted_risk_adjusted_interval_upper_tokens_per_sec: float | None
    prediction_status: str
    matched_label: str | None
    calibration_distance: float | None
    feasibility_status: str
    actual_rank: int
    actual_efficiency_rank: int
    predicted_rank: int | None
    risk_adjusted_rank: int | None
    predicted_efficiency_rank: int | None
    risk_adjusted_efficiency_rank: int | None
    actual_frontier_member: bool
    predicted_frontier_member: bool
    risk_adjusted_frontier_member: bool
    actual_scaling_baseline_label: str | None
    actual_scaling_gpu_ratio: float | None
    actual_scaling_speedup: float | None
    actual_scaling_efficiency: float | None
    predicted_scaling_baseline_label: str | None
    predicted_scaling_speedup: float | None
    predicted_scaling_efficiency: float | None
    risk_adjusted_scaling_baseline_label: str | None
    risk_adjusted_scaling_speedup: float | None
    risk_adjusted_scaling_efficiency: float | None
    risk_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionHoldout:
    heldout_label: str
    heldout_source: str
    actual_best_label: str
    predicted_best_label: str | None
    actual_best_tokens_per_sec: float
    selected_actual_tokens_per_sec: float | None
    regret_tokens_per_sec: float | None
    regret_percentage: float | None
    risk_adjusted_predicted_best_label: str | None
    risk_adjusted_selected_actual_tokens_per_sec: float | None
    risk_adjusted_regret_tokens_per_sec: float | None
    risk_adjusted_regret_percentage: float | None
    actual_efficiency_best_label: str
    actual_efficiency_best_tokens_per_gpu_per_sec: float | None
    predicted_efficiency_best_label: str | None
    selected_actual_efficiency_tokens_per_gpu_per_sec: float | None
    efficiency_regret_tokens_per_gpu_per_sec: float | None
    efficiency_regret_percentage: float | None
    risk_adjusted_efficiency_predicted_best_label: str | None
    risk_adjusted_efficiency_selected_actual_tokens_per_gpu_per_sec: float | None
    risk_adjusted_efficiency_regret_tokens_per_gpu_per_sec: float | None
    risk_adjusted_efficiency_regret_percentage: float | None
    selected_is_actual_best: bool
    actual_best_in_predicted_top_tie: bool
    risk_adjusted_selected_is_actual_best: bool
    actual_best_in_risk_adjusted_top_tie: bool
    actual_best_in_risk_adjusted_interval_top: bool
    risk_adjusted_interval_top_labels: list[str]
    risk_adjusted_interval_top_count: int
    risk_adjusted_interval_top_best_actual_tokens_per_sec: float | None
    risk_adjusted_interval_top_regret_tokens_per_sec: float | None
    risk_adjusted_interval_top_regret_percentage: float | None
    pairwise_ordering_pair_count: int
    pairwise_ordering_correct_count: int
    pairwise_ordering_accuracy: float | None
    risk_adjusted_pairwise_ordering_pair_count: int
    risk_adjusted_pairwise_ordering_correct_count: int
    risk_adjusted_pairwise_ordering_accuracy: float | None
    mean_absolute_rank_error: float | None
    max_absolute_rank_error: int | None
    risk_adjusted_mean_absolute_rank_error: float | None
    risk_adjusted_max_absolute_rank_error: int | None
    efficiency_pairwise_ordering_pair_count: int
    efficiency_pairwise_ordering_correct_count: int
    efficiency_pairwise_ordering_accuracy: float | None
    risk_adjusted_efficiency_pairwise_ordering_pair_count: int
    risk_adjusted_efficiency_pairwise_ordering_correct_count: int
    risk_adjusted_efficiency_pairwise_ordering_accuracy: float | None
    efficiency_mean_absolute_rank_error: float | None
    efficiency_max_absolute_rank_error: int | None
    risk_adjusted_efficiency_mean_absolute_rank_error: float | None
    risk_adjusted_efficiency_max_absolute_rank_error: int | None
    efficiency_selected_is_actual_best: bool
    actual_efficiency_best_in_predicted_top_tie: bool
    risk_adjusted_efficiency_selected_is_actual_best: bool
    actual_efficiency_best_in_risk_adjusted_top_tie: bool
    actual_frontier_labels: list[str]
    predicted_frontier_labels: list[str]
    risk_adjusted_frontier_labels: list[str]
    actual_frontier_count: int
    predicted_frontier_count: int
    risk_adjusted_frontier_count: int
    actual_frontier_in_predicted_count: int
    actual_frontier_in_risk_adjusted_count: int
    actual_frontier_predicted_coverage_fraction: float | None
    actual_frontier_risk_adjusted_coverage_fraction: float | None
    predicted_frontier_extra_count: int
    risk_adjusted_frontier_extra_count: int
    actual_frontier_missed_labels: list[str]
    actual_frontier_risk_adjusted_missed_labels: list[str]
    scaling_candidate_count: int
    predicted_scaling_candidate_count: int
    risk_adjusted_scaling_candidate_count: int
    mean_scaling_efficiency_absolute_error: float | None
    max_scaling_efficiency_absolute_error: float | None
    max_scaling_efficiency_absolute_error_label: str | None
    mean_risk_adjusted_scaling_efficiency_absolute_error: float | None
    max_risk_adjusted_scaling_efficiency_absolute_error: float | None
    max_risk_adjusted_scaling_efficiency_absolute_error_label: str | None
    predicted_top_tie_count: int
    risk_adjusted_top_tie_count: int
    predicted_efficiency_top_tie_count: int
    risk_adjusted_efficiency_top_tie_count: int
    candidate_count: int
    predicted_unscored_count: int
    risk_adjusted_unscored_count: int
    predicted_efficiency_unscored_count: int
    risk_adjusted_efficiency_unscored_count: int
    candidates: list[DecisionCandidatePrediction]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParallelismBoundaryGroup:
    signature: str
    candidate_count: int
    fit_count: int
    failure_count: int
    best_fit_label: str | None
    best_fit_tokens_per_sec: float | None
    failure_labels: list[str]
    varied_parallelism_dimensions: list[str]
    confounded_workload_dimensions: list[str]
    confounded_runtime_dimensions: list[str]


@dataclass(frozen=True)
class ParallelismBoundaryAxisCoverage:
    axis: str
    status: str
    group_count: int
    candidate_count: int
    fit_count: int
    failure_count: int
    varied_parallelism_dimensions: list[str]
    co_varied_parallelism_dimensions: list[str]
    confounded_workload_dimensions: list[str]
    confounded_runtime_dimensions: list[str]
    boundary_candidate_labels: list[str] = field(default_factory=list)
    best_fit_labels: list[str] = field(default_factory=list)
    failure_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParallelismAxisCoverage:
    axis: str
    status: str
    candidate_count: int
    varied_parallelism_dimensions: list[str]
    co_varied_parallelism_dimensions: list[str]
    like_for_like_group_count: int
    evaluated_count: int
    confounded_workload_dimensions: list[str]
    confounded_runtime_dimensions: list[str]
    candidate_labels: list[str] = field(default_factory=list)
    evaluated_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParallelismValidationGap:
    axis: str
    gap_status: str
    priority: int
    required_measurement: str
    reason: str
    throughput_axis_status: str
    boundary_axis_status: str
    candidate_count: int
    boundary_candidate_count: int
    fit_count: int
    failure_count: int
    co_varied_parallelism_dimensions: list[str]
    confounded_workload_dimensions: list[str]
    confounded_runtime_dimensions: list[str]
    throughput_candidate_labels: list[str] = field(default_factory=list)
    throughput_evaluated_labels: list[str] = field(default_factory=list)
    boundary_candidate_labels: list[str] = field(default_factory=list)
    boundary_best_fit_labels: list[str] = field(default_factory=list)
    boundary_failure_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OptimalParallelismReadiness:
    readiness_status: str = "unknown_optimal_parallelism_readiness"
    can_predict_optimal_tradeoff: bool = False
    support_status: str = "unknown_optimal_parallelism_support"
    support_blockers: list[str] = field(default_factory=list)
    prediction_status: str = "unknown_parallelism_prediction"
    validation_scope_status: str = "unknown_parallelism_validation_scope"
    boundary_prediction_status: str = "unknown_parallelism_boundary_prediction"
    required_measurements: list[str] = field(default_factory=list)
    validation_gap_count: int = 0
    validation_gap_axis_names: list[str] = field(default_factory=list)
    validation_gap_status_counts: dict[str, int] = field(default_factory=dict)
    top_validation_gap_statuses: list[str] = field(default_factory=list)
    top_validation_gap_required_measurements: list[str] = field(default_factory=list)
    top_validation_gap_axis_names: list[str] = field(default_factory=list)
    measured_parallelism_axis_names: list[str] = field(default_factory=list)
    isolated_measured_parallelism_axis_names: list[str] = field(default_factory=list)
    coupled_measured_parallelism_axis_names: list[str] = field(default_factory=list)
    confounded_parallelism_axis_names: list[str] = field(default_factory=list)
    missing_parallelism_axis_names: list[str] = field(default_factory=list)
    parallelism_evaluated_count: int = 0
    parallelism_axis_evaluated_count: int = 0
    like_for_like_parallelism_group_count: int = 0
    parallelism_boundary_group_count: int = 0
    parallelism_boundary_fit_count: int = 0
    parallelism_boundary_failure_count: int = 0
    parallelism_top1_selection_hit_rate: float | None = None
    risk_adjusted_parallelism_top1_selection_hit_rate: float | None = None
    risk_adjusted_parallelism_interval_selection_hit_rate: float | None = None
    parallelism_pairwise_ordering_accuracy: float | None = None
    risk_adjusted_parallelism_pairwise_ordering_accuracy: float | None = None
    efficiency_parallelism_pairwise_ordering_accuracy: float | None = None
    risk_adjusted_efficiency_parallelism_pairwise_ordering_accuracy: float | None = None
    parallelism_frontier_coverage_hit_rate: float | None = None
    mean_parallelism_frontier_coverage_fraction: float | None = None
    risk_adjusted_parallelism_frontier_coverage_hit_rate: float | None = None
    mean_risk_adjusted_parallelism_frontier_coverage_fraction: float | None = None
    parallelism_frontier_missed_labels: list[str] = field(default_factory=list)
    risk_adjusted_parallelism_frontier_missed_labels: list[str] = field(default_factory=list)
    measurement_design_config_count: int = 0
    measurement_design_config_labels: list[str] = field(default_factory=list)
    measurement_design_config_filenames: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionReport:
    base_config_path: str
    benchmark_dir: str
    status: str
    measured_point_count: int
    promotable_point_count: int
    best_actual_label: str | None
    best_actual_tokens_per_sec: float | None
    best_actual_promotable_label: str | None
    best_actual_promotable_tokens_per_sec: float | None
    promotable_actual_gap_tokens_per_sec: float | None
    promotable_actual_gap_percentage: float | None
    promotion_readiness_status: str
    evaluated_count: int
    skipped_count: int
    hit_count: int
    selection_hit_rate: float | None
    selected_best_count: int
    top1_selection_hit_rate: float | None
    mean_regret_percentage: float | None
    max_regret_percentage: float | None
    max_regret_percentage_label: str | None
    risk_adjusted_hit_count: int
    risk_adjusted_selection_hit_rate: float | None
    risk_adjusted_selected_best_count: int
    risk_adjusted_top1_selection_hit_rate: float | None
    mean_risk_adjusted_regret_percentage: float | None
    max_risk_adjusted_regret_percentage: float | None
    max_risk_adjusted_regret_percentage_label: str | None
    risk_adjusted_interval_hit_count: int
    risk_adjusted_interval_selection_hit_rate: float | None
    mean_risk_adjusted_interval_top_count: float | None
    max_risk_adjusted_interval_top_count: int | None
    mean_risk_adjusted_interval_regret_percentage: float | None
    max_risk_adjusted_interval_regret_percentage: float | None
    pairwise_ordering_pair_count: int
    pairwise_ordering_correct_count: int
    pairwise_ordering_accuracy: float | None
    mean_pairwise_ordering_accuracy: float | None
    risk_adjusted_pairwise_ordering_pair_count: int
    risk_adjusted_pairwise_ordering_correct_count: int
    risk_adjusted_pairwise_ordering_accuracy: float | None
    mean_risk_adjusted_pairwise_ordering_accuracy: float | None
    mean_absolute_rank_error: float | None
    max_absolute_rank_error: int | None
    max_absolute_rank_error_label: str | None
    mean_holdout_absolute_rank_error: float | None
    risk_adjusted_mean_absolute_rank_error: float | None
    risk_adjusted_max_absolute_rank_error: int | None
    risk_adjusted_max_absolute_rank_error_label: str | None
    mean_risk_adjusted_holdout_absolute_rank_error: float | None
    efficiency_hit_count: int
    efficiency_selection_hit_rate: float | None
    efficiency_selected_best_count: int
    efficiency_top1_selection_hit_rate: float | None
    mean_efficiency_regret_percentage: float | None
    max_efficiency_regret_percentage: float | None
    max_efficiency_regret_percentage_label: str | None
    risk_adjusted_efficiency_hit_count: int
    risk_adjusted_efficiency_selection_hit_rate: float | None
    risk_adjusted_efficiency_selected_best_count: int
    risk_adjusted_efficiency_top1_selection_hit_rate: float | None
    mean_risk_adjusted_efficiency_regret_percentage: float | None
    max_risk_adjusted_efficiency_regret_percentage: float | None
    max_risk_adjusted_efficiency_regret_percentage_label: str | None
    efficiency_pairwise_ordering_pair_count: int
    efficiency_pairwise_ordering_correct_count: int
    efficiency_pairwise_ordering_accuracy: float | None
    mean_efficiency_pairwise_ordering_accuracy: float | None
    risk_adjusted_efficiency_pairwise_ordering_pair_count: int
    risk_adjusted_efficiency_pairwise_ordering_correct_count: int
    risk_adjusted_efficiency_pairwise_ordering_accuracy: float | None
    mean_risk_adjusted_efficiency_pairwise_ordering_accuracy: float | None
    efficiency_mean_absolute_rank_error: float | None
    efficiency_max_absolute_rank_error: int | None
    efficiency_max_absolute_rank_error_label: str | None
    mean_holdout_efficiency_absolute_rank_error: float | None
    risk_adjusted_efficiency_mean_absolute_rank_error: float | None
    risk_adjusted_efficiency_max_absolute_rank_error: int | None
    risk_adjusted_efficiency_max_absolute_rank_error_label: str | None
    mean_risk_adjusted_efficiency_holdout_absolute_rank_error: float | None
    frontier_hit_count: int
    frontier_coverage_hit_rate: float | None
    mean_frontier_coverage_fraction: float | None
    min_frontier_coverage_fraction: float | None
    mean_predicted_frontier_extra_count: float | None
    max_predicted_frontier_extra_count: int | None
    frontier_missed_labels: list[str]
    risk_adjusted_frontier_hit_count: int
    risk_adjusted_frontier_coverage_hit_rate: float | None
    mean_risk_adjusted_frontier_coverage_fraction: float | None
    min_risk_adjusted_frontier_coverage_fraction: float | None
    mean_risk_adjusted_frontier_extra_count: float | None
    max_risk_adjusted_frontier_extra_count: int | None
    risk_adjusted_frontier_missed_labels: list[str]
    scaling_evaluated_count: int
    mean_scaling_efficiency_absolute_error: float | None
    max_scaling_efficiency_absolute_error: float | None
    max_scaling_efficiency_absolute_error_label: str | None
    risk_adjusted_scaling_evaluated_count: int
    mean_risk_adjusted_scaling_efficiency_absolute_error: float | None
    max_risk_adjusted_scaling_efficiency_absolute_error: float | None
    max_risk_adjusted_scaling_efficiency_absolute_error_label: str | None
    varied_parallelism_dimensions: list[str]
    varied_workload_dimensions: list[str]
    varied_runtime_dimensions: list[str]
    unique_parallelism_strategy_count: int
    parallelism_coverage_status: str
    parallelism_prediction_status: str
    parallelism_prediction_blockers: list[str]
    parallelism_validation_scope_status: str
    parallelism_validation_scope_blockers: list[str]
    optimal_parallelism_support_status: str
    optimal_parallelism_support_blockers: list[str]
    measured_parallelism_axis_names: list[str]
    isolated_measured_parallelism_axis_names: list[str]
    coupled_measured_parallelism_axis_names: list[str]
    confounded_parallelism_axis_names: list[str]
    missing_parallelism_axis_names: list[str]
    parallelism_axis_coverage_status_counts: dict[str, int]
    parallelism_axis_coverage: list[ParallelismAxisCoverage]
    parallelism_validation_gap_count: int
    parallelism_validation_gap_axis_names: list[str]
    parallelism_validation_gap_status_counts: dict[str, int]
    parallelism_validation_gaps: list[ParallelismValidationGap]
    like_for_like_parallelism_group_count: int
    parallelism_evaluated_count: int
    parallelism_axis_group_count: int
    parallelism_axis_evaluated_count: int
    parallelism_axis_hit_count: int
    parallelism_axis_selection_hit_rate: float | None
    parallelism_axis_selected_best_count: int
    parallelism_axis_top1_selection_hit_rate: float | None
    risk_adjusted_parallelism_axis_hit_count: int
    risk_adjusted_parallelism_axis_selection_hit_rate: float | None
    risk_adjusted_parallelism_axis_selected_best_count: int
    risk_adjusted_parallelism_axis_top1_selection_hit_rate: float | None
    parallelism_hit_count: int
    parallelism_selection_hit_rate: float | None
    parallelism_selected_best_count: int
    parallelism_top1_selection_hit_rate: float | None
    risk_adjusted_parallelism_hit_count: int
    risk_adjusted_parallelism_selection_hit_rate: float | None
    risk_adjusted_parallelism_selected_best_count: int
    risk_adjusted_parallelism_top1_selection_hit_rate: float | None
    risk_adjusted_parallelism_interval_hit_count: int
    risk_adjusted_parallelism_interval_selection_hit_rate: float | None
    mean_risk_adjusted_parallelism_interval_top_count: float | None
    max_risk_adjusted_parallelism_interval_top_count: int | None
    parallelism_pairwise_ordering_pair_count: int
    parallelism_pairwise_ordering_correct_count: int
    parallelism_pairwise_ordering_accuracy: float | None
    mean_parallelism_pairwise_ordering_accuracy: float | None
    risk_adjusted_parallelism_pairwise_ordering_pair_count: int
    risk_adjusted_parallelism_pairwise_ordering_correct_count: int
    risk_adjusted_parallelism_pairwise_ordering_accuracy: float | None
    mean_risk_adjusted_parallelism_pairwise_ordering_accuracy: float | None
    parallelism_mean_absolute_rank_error: float | None
    parallelism_max_absolute_rank_error: int | None
    parallelism_max_absolute_rank_error_label: str | None
    mean_parallelism_holdout_absolute_rank_error: float | None
    risk_adjusted_parallelism_mean_absolute_rank_error: float | None
    risk_adjusted_parallelism_max_absolute_rank_error: int | None
    risk_adjusted_parallelism_max_absolute_rank_error_label: str | None
    mean_risk_adjusted_parallelism_holdout_absolute_rank_error: float | None
    parallelism_axis_pairwise_ordering_pair_count: int
    parallelism_axis_pairwise_ordering_correct_count: int
    parallelism_axis_pairwise_ordering_accuracy: float | None
    mean_parallelism_axis_pairwise_ordering_accuracy: float | None
    risk_adjusted_parallelism_axis_pairwise_ordering_pair_count: int
    risk_adjusted_parallelism_axis_pairwise_ordering_correct_count: int
    risk_adjusted_parallelism_axis_pairwise_ordering_accuracy: float | None
    mean_risk_adjusted_parallelism_axis_pairwise_ordering_accuracy: float | None
    parallelism_axis_mean_absolute_rank_error: float | None
    parallelism_axis_max_absolute_rank_error: int | None
    parallelism_axis_max_absolute_rank_error_label: str | None
    mean_parallelism_axis_holdout_absolute_rank_error: float | None
    risk_adjusted_parallelism_axis_mean_absolute_rank_error: float | None
    risk_adjusted_parallelism_axis_max_absolute_rank_error: int | None
    risk_adjusted_parallelism_axis_max_absolute_rank_error_label: str | None
    mean_risk_adjusted_parallelism_axis_holdout_absolute_rank_error: float | None
    efficiency_parallelism_hit_count: int
    efficiency_parallelism_selection_hit_rate: float | None
    efficiency_parallelism_selected_best_count: int
    efficiency_parallelism_top1_selection_hit_rate: float | None
    risk_adjusted_efficiency_parallelism_hit_count: int
    risk_adjusted_efficiency_parallelism_selection_hit_rate: float | None
    risk_adjusted_efficiency_parallelism_selected_best_count: int
    risk_adjusted_efficiency_parallelism_top1_selection_hit_rate: float | None
    efficiency_parallelism_pairwise_ordering_pair_count: int
    efficiency_parallelism_pairwise_ordering_correct_count: int
    efficiency_parallelism_pairwise_ordering_accuracy: float | None
    mean_efficiency_parallelism_pairwise_ordering_accuracy: float | None
    risk_adjusted_efficiency_parallelism_pairwise_ordering_pair_count: int
    risk_adjusted_efficiency_parallelism_pairwise_ordering_correct_count: int
    risk_adjusted_efficiency_parallelism_pairwise_ordering_accuracy: float | None
    mean_risk_adjusted_efficiency_parallelism_pairwise_ordering_accuracy: float | None
    efficiency_parallelism_mean_absolute_rank_error: float | None
    efficiency_parallelism_max_absolute_rank_error: int | None
    efficiency_parallelism_max_absolute_rank_error_label: str | None
    mean_efficiency_parallelism_holdout_absolute_rank_error: float | None
    risk_adjusted_efficiency_parallelism_mean_absolute_rank_error: float | None
    risk_adjusted_efficiency_parallelism_max_absolute_rank_error: int | None
    risk_adjusted_efficiency_parallelism_max_absolute_rank_error_label: str | None
    mean_risk_adjusted_efficiency_parallelism_holdout_absolute_rank_error: float | None
    efficiency_parallelism_axis_pairwise_ordering_pair_count: int
    efficiency_parallelism_axis_pairwise_ordering_correct_count: int
    efficiency_parallelism_axis_pairwise_ordering_accuracy: float | None
    mean_efficiency_parallelism_axis_pairwise_ordering_accuracy: float | None
    risk_adjusted_efficiency_parallelism_axis_pairwise_ordering_pair_count: int
    risk_adjusted_efficiency_parallelism_axis_pairwise_ordering_correct_count: int
    risk_adjusted_efficiency_parallelism_axis_pairwise_ordering_accuracy: float | None
    mean_risk_adjusted_efficiency_parallelism_axis_pairwise_ordering_accuracy: float | None
    efficiency_parallelism_axis_mean_absolute_rank_error: float | None
    efficiency_parallelism_axis_max_absolute_rank_error: int | None
    efficiency_parallelism_axis_max_absolute_rank_error_label: str | None
    mean_efficiency_parallelism_axis_holdout_absolute_rank_error: float | None
    risk_adjusted_efficiency_parallelism_axis_mean_absolute_rank_error: float | None
    risk_adjusted_efficiency_parallelism_axis_max_absolute_rank_error: int | None
    risk_adjusted_efficiency_parallelism_axis_max_absolute_rank_error_label: str | None
    mean_risk_adjusted_efficiency_parallelism_axis_holdout_absolute_rank_error: float | None
    parallelism_frontier_hit_count: int
    parallelism_frontier_coverage_hit_rate: float | None
    mean_parallelism_frontier_coverage_fraction: float | None
    min_parallelism_frontier_coverage_fraction: float | None
    parallelism_frontier_missed_labels: list[str]
    risk_adjusted_parallelism_frontier_hit_count: int
    risk_adjusted_parallelism_frontier_coverage_hit_rate: float | None
    mean_risk_adjusted_parallelism_frontier_coverage_fraction: float | None
    min_risk_adjusted_parallelism_frontier_coverage_fraction: float | None
    risk_adjusted_parallelism_frontier_missed_labels: list[str]
    parallelism_scaling_evaluated_count: int
    mean_parallelism_scaling_efficiency_absolute_error: float | None
    max_parallelism_scaling_efficiency_absolute_error: float | None
    max_parallelism_scaling_efficiency_absolute_error_label: str | None
    risk_adjusted_parallelism_scaling_evaluated_count: int
    mean_risk_adjusted_parallelism_scaling_efficiency_absolute_error: float | None
    max_risk_adjusted_parallelism_scaling_efficiency_absolute_error: float | None
    max_risk_adjusted_parallelism_scaling_efficiency_absolute_error_label: str | None
    parallelism_boundary_status: str
    parallelism_boundary_prediction_status: str
    parallelism_boundary_prediction_blockers: list[str]
    parallelism_boundary_group_count: int
    parallelism_boundary_candidate_count: int
    parallelism_boundary_fit_count: int
    parallelism_boundary_failure_count: int
    parallelism_boundary_best_fit_label: str | None
    parallelism_boundary_confounded_dimensions: list[str]
    parallelism_boundary_measured_axis_names: list[str]
    parallelism_boundary_confounded_axis_names: list[str]
    parallelism_boundary_missing_axis_names: list[str]
    parallelism_boundary_axis_coverage_status_counts: dict[str, int]
    parallelism_boundary_axis_coverage: list[ParallelismBoundaryAxisCoverage]
    parallelism_boundary_groups: list[ParallelismBoundaryGroup]
    holdouts: list[DecisionHoldout]
    optimal_parallelism_readiness: OptimalParallelismReadiness = field(default_factory=OptimalParallelismReadiness)
    warnings: list[str] = field(default_factory=list)
    supplemental_benchmark_dirs: list[str] = field(default_factory=list)
    primary_behavior_point_count: int = 0
    supplemental_behavior_point_count: int = 0
    primary_measured_point_count: int = 0
    supplemental_measured_point_count: int = 0
