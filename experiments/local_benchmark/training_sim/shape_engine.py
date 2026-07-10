"""Static token and balanced-routing shape calculations."""

from __future__ import annotations

import math


try:
    from .schemas import BalancedRoutingLedger, ShapeLedger, Topology
except ImportError:  # pragma: no cover - exercised by direct script execution
    from schemas import BalancedRoutingLedger, ShapeLedger, Topology


def balanced_counts(total_slots: int, num_experts: int) -> list[int]:
    """Counts produced by round-robin balanced synthetic routing over expert slots."""
    if total_slots < 0:
        raise ValueError("total_slots must be non-negative")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    base, remainder = divmod(total_slots, num_experts)
    return [base + (1 if expert_idx < remainder else 0) for expert_idx in range(num_experts)]


def balanced_routing_ledger(total_slots: int, num_experts: int) -> BalancedRoutingLedger:
    counts = balanced_counts(total_slots, num_experts)
    max_slots = max(counts) if counts else 0
    min_slots = min(counts) if counts else 0
    return BalancedRoutingLedger(
        total_slots=total_slots,
        num_experts=num_experts,
        counts_by_expert=counts,
        max_slots_per_expert=max_slots,
        min_slots_per_expert=min_slots,
        imbalance_slots=max_slots - min_slots,
    )


def ep_rank_counts(counts_by_expert: list[int], expert_parallel_size: int) -> list[int]:
    if expert_parallel_size <= 0:
        raise ValueError("expert_parallel_size must be positive")
    if not counts_by_expert:
        return []
    if len(counts_by_expert) % expert_parallel_size != 0:
        raise ValueError("num_experts must be divisible by expert_parallel_size for contiguous EP ownership")
    experts_per_rank = len(counts_by_expert) // expert_parallel_size
    return [
        sum(counts_by_expert[start : start + experts_per_rank])
        for start in range(0, len(counts_by_expert), experts_per_rank)
    ]


def build_shape_ledger(topology: Topology, *, balanced_routing: bool) -> ShapeLedger:
    warnings: list[str] = []
    seq_len = topology.sample_packing_sequence_len
    if seq_len is None:
        warnings.append("data.sample_packing_sequence_len is not set; routed token counts are unavailable")
        return ShapeLedger(
            microbatch_tokens_per_dp_rank=None,
            global_tokens_per_microbatch=None,
            global_tokens_per_train_step=None,
            tokens_per_gpu_per_train_step=None,
            sequence_parallel_size=topology.sequence_parallel_size,
            tokens_per_model_rank_per_microbatch=None,
            routed_slots_per_model_rank_microbatch=None,
            routed_slots_per_train_step_model_rank=None,
            balanced_routing=None,
            ep_rank_slots_per_microbatch=None,
            warnings=warnings,
        )

    microbatch_tokens = topology.micro_batch_size * seq_len
    global_tokens_per_microbatch = microbatch_tokens * topology.data_parallel_size
    global_tokens_per_train_step = topology.global_batch_size * seq_len
    tokens_per_gpu_per_train_step = global_tokens_per_train_step / topology.world_size
    sequence_parallel_size = max(topology.sequence_parallel_size, 1)
    model_rank_tokens = math.ceil(microbatch_tokens / sequence_parallel_size)

    routing_ledger = None
    routed_slots_per_microbatch = None
    routed_slots_per_step = None
    ep_counts = None

    if topology.top_k is None or topology.num_experts is None:
        warnings.append("num_experts/top_k are unknown; pass --num-experts and --top-k for MoE routing counts")
    else:
        routed_slots_per_microbatch = model_rank_tokens * topology.top_k
        routed_slots_per_step = routed_slots_per_microbatch * topology.gradient_accumulation_steps
        if balanced_routing:
            routing_ledger = balanced_routing_ledger(routed_slots_per_microbatch, topology.num_experts)
            try:
                ep_counts = ep_rank_counts(routing_ledger.counts_by_expert, topology.expert_parallel_size)
            except ValueError as exc:
                warnings.append(str(exc))
        else:
            warnings.append("balanced routing is disabled; expert-local slot counts are intentionally omitted")

    return ShapeLedger(
        microbatch_tokens_per_dp_rank=microbatch_tokens,
        global_tokens_per_microbatch=global_tokens_per_microbatch,
        global_tokens_per_train_step=global_tokens_per_train_step,
        tokens_per_gpu_per_train_step=tokens_per_gpu_per_train_step,
        sequence_parallel_size=sequence_parallel_size,
        tokens_per_model_rank_per_microbatch=model_rank_tokens,
        routed_slots_per_model_rank_microbatch=routed_slots_per_microbatch,
        routed_slots_per_train_step_model_rank=routed_slots_per_step,
        balanced_routing=routing_ledger,
        ep_rank_slots_per_microbatch=ep_counts,
        warnings=warnings,
    )
