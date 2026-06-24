"""Parallelization plan for NVIDIA Nemotron-3-Ultra (nemotron_h).

Supported this pass: FSDP2 + EP over the 512 routed experts.
TP and CP/SP (Ulysses / ring) are NOT supported — Mamba2 layers need
head/group-aware sharding and a stateful sequence-parallel scan; both raise.
"""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


TP_UNSUPPORTED_MESSAGE = (
    "Tensor parallelism is not implemented for NemotronH (nemotron_h): "
    "Mamba2 mixers require head/group-aware sharding. Use FSDP2 + EP instead (tp_size=1)."
)
CP_UNSUPPORTED_MESSAGE = (
    "Context/sequence parallelism (Ulysses/ring) is not implemented for NemotronH (nemotron_h): "
    "Mamba2 layers carry state along the sequence dimension. Use ulysses/ring size 1."
)


def get_ep_plan() -> ParallelPlan:
    """EP plan for the routed experts (stacked ``[num_experts, K, N]`` GKN format).

    Everything else (mamba mixers, attention, shared experts, latent
    projections, router) is FSDP-sharded.
    """
    ep_plan = {
        "model.layers.*.mixer.experts.gate_up_proj": Shard(0),
        "model.layers.*.mixer.experts.down_proj": Shard(0),
        # LoRA weights for experts (initialized at global shape, sharded here).
        "model.layers.*.mixer.experts.up_proj_lora_A": Shard(0),
        "model.layers.*.mixer.experts.up_proj_lora_B": Shard(0),
        "model.layers.*.mixer.experts.down_proj_lora_A": Shard(0),
        "model.layers.*.mixer.experts.down_proj_lora_B": Shard(0),
    }
    return ParallelPlan(ep_plan=ep_plan)


def validate_parallelism_support(parallel_state) -> None:
    """Raise for parallelism modes NemotronH does not support yet."""
    if not parallel_state.is_initialized:
        return
    if parallel_state.tp_enabled:
        raise NotImplementedError(TP_UNSUPPORTED_MESSAGE)
    if parallel_state.cp_enabled or parallel_state.ulysses_enabled or parallel_state.ringattn_enabled:
        raise NotImplementedError(CP_UNSUPPORTED_MESSAGE)
