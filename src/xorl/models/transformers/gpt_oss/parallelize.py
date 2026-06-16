"""Parallelization plan and utilities for GPT-OSS models."""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


# TP plan for the base model (GptOssModel).
# Only covers attention — all MLP layers are MoE and use EP instead.
TP_PLAN = {
    "embed_tokens": "embedding",
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
}

# TP plan for top-level modules on the CausalLM wrapper.
MODEL_TP_PLAN = {
    "lm_head": "colwise_rep",
}


def unfuse_for_tp(model):
    """Unfuse fused QKV projections for tensor parallelism compatibility.

    GPT-OSS has only MoE layers (no dense MLP), so only attention is unfused.
    """
    for layer in model.model.layers:
        layer.self_attn.unfuse_for_tp()
    model._unfused_for_tp = True
    model.config.base_model_tp_plan = TP_PLAN


def get_ep_plan():
    """Get EP (expert parallelism) plan for GPT-OSS model."""
    ep_plan = {
        # Expert weights (stacked [num_experts, H, 2I] format)
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
        # Expert biases (stacked [num_experts, ...] format)
        "model.layers.*.mlp.experts.gate_up_bias": Shard(0),
        "model.layers.*.mlp.experts.down_bias": Shard(0),
        # LoRA weights for experts
        "model.layers.*.mlp.experts.gate_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.gate_proj_lora_B": Shard(0),
        "model.layers.*.mlp.experts.up_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.up_proj_lora_B": Shard(0),
        "model.layers.*.mlp.experts.down_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.down_proj_lora_B": Shard(0),
    }
    return ParallelPlan(ep_plan=ep_plan)
