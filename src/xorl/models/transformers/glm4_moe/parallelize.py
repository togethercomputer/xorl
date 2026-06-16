"""Parallelization plan and utilities for GLM-4 MoE models."""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan
from ...layers.moe import MoEBlock


TP_PLAN = {
    "embed_tokens": "embedding",
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    # Dense MLP layers (first K layers before MoE)
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
    # Shared expert MLP (inside MoE blocks)
    "layers.*.mlp.shared_experts.gate_proj": "colwise",
    "layers.*.mlp.shared_experts.up_proj": "colwise",
    "layers.*.mlp.shared_experts.down_proj": "rowwise",
}

MODEL_TP_PLAN = {
    "lm_head": "colwise_rep",
}


def unfuse_for_tp(model):
    """Unfuse fused projections for tensor parallelism compatibility.

    For ALL layers: splits ``qkv_proj`` -> ``q_proj / k_proj / v_proj`` in attention.
    For DENSE layers only: splits ``gate_up_proj`` -> ``gate_proj / up_proj`` in MLP.
    For MoE layers: also unfuses the shared expert MLP.
    MoE expert weights are not TP-sharded (they use EP instead).
    """
    for layer in model.model.layers:
        layer.self_attn.unfuse_for_tp()
        if isinstance(layer.mlp, MoEBlock):
            if hasattr(layer.mlp, "shared_experts") and layer.mlp.shared_experts is not None:
                layer.mlp.shared_experts.unfuse_for_tp()
        else:
            layer.mlp.unfuse_for_tp()
    model._unfused_for_tp = True
    model.config.base_model_tp_plan = TP_PLAN


def get_ep_plan():
    """Get EP (expert parallelism) plan for GLM-4 MoE model."""
    ep_plan = {
        # Expert base weights — stored as fused gate_up_proj [E, H, 2*I]
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
