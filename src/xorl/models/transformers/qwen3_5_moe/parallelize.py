"""Parallelization plan and utilities for Qwen3_5 MoE models."""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan
from ...layers.moe import MoEBlock


# TP plan for the base model (Qwen3_5MoeModel).
# Only covers attention and *dense* MLP layers - MoE expert weights
# are not TP-sharded (they use EP instead).
TP_PLAN = {
    "embed_tokens": "embedding",
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}

# TP plan for top-level modules on the CausalLM wrapper.
MODEL_TP_PLAN = {
    "lm_head": "colwise_rep",
}


def unfuse_for_tp(model):
    """Unfuse fused projections for tensor parallelism compatibility.

    For ALL layers: splits ``qkv_proj`` -> ``q_proj / k_proj / v_proj`` in attention.
    For DENSE layers only: splits ``gate_up_proj`` -> ``gate_proj / up_proj`` in MLP.
    A ``MoEBlock``'s routed experts are left untouched - their weights are not
    TP-sharded - but the block's shared expert is a plain dense MLP with its own
    fused ``gate_up_proj``, so it unfuses like any other, mirroring how
    ``glm4_moe`` handles ``shared_experts``. Leaving it fused would strand it:
    ``get_checkpoint_handler`` passes ``skip_gate_up_merge`` for the whole model,
    so nothing would merge its gate/up halves at load.
    """
    for layer in model.model.layers:
        if getattr(layer, "self_attn", None) is not None and hasattr(layer.self_attn, "unfuse_for_tp"):
            layer.self_attn.unfuse_for_tp()
        # Only the routed experts of an MoE block stay fused; its shared expert does not.
        if isinstance(layer.mlp, MoEBlock):
            shared_expert = getattr(layer.mlp, "shared_expert", None)
            if shared_expert is not None and hasattr(shared_expert, "unfuse_for_tp"):
                shared_expert.unfuse_for_tp()
        else:
            layer.mlp.unfuse_for_tp()
    model._unfused_for_tp = True
    # Override HF config's TP plan (may contain incompatible styles like
    # "packed_colwise") with our plan for unfused projections.
    model.config.base_model_tp_plan = TP_PLAN


def get_ep_plan():
    """Get EP (expert parallelism) plan for Qwen3_5 MoE model.

    Both base weights and LoRA weights are sharded via parallel_plan.apply().
    Base weights are loaded from checkpoint and sliced via _slice_expert_tensor_for_ep().
    LoRA weights are initialized at GLOBAL shape and sharded here.
    """
    ep_plan = {
        # Expert weights (stacked [num_experts, H, 2I] format)
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
        # LoRA weights for experts
        "model.layers.*.mlp.experts.gate_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.gate_proj_lora_B": Shard(0),
        "model.layers.*.mlp.experts.up_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.up_proj_lora_B": Shard(0),
        "model.layers.*.mlp.experts.down_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.down_proj_lora_B": Shard(0),
    }
    return ParallelPlan(ep_plan=ep_plan)
