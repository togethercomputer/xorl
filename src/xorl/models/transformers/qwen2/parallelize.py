"""Parallelization plan and utilities for Qwen2 dense models."""

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

MODEL_TP_PLAN = {
    "lm_head": "colwise",
}


def unfuse_for_tp(model):
    """Unfuse fused projections for tensor parallelism compatibility."""
    for layer in model.model.layers:
        layer.self_attn.unfuse_for_tp()
        layer.mlp.unfuse_for_tp()
    model._unfused_for_tp = True
    model.config.base_model_tp_plan = TP_PLAN
