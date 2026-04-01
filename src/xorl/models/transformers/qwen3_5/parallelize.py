"""Parallelization plan and utilities for dense Qwen3_5 models."""

# TP plan for the base model (Qwen3_5Model).
# Keys use wildcard patterns relative to the base model prefix.
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
    "lm_head": "colwise",
}


def unfuse_for_tp(model):
    """Unfuse fused projections for tensor parallelism compatibility.

    Splits ``gate_up_proj`` -> ``gate_proj / up_proj`` in MLP for every
    decoder layer.  Attention layers already have separate q/k/v projections.
    Linear attention layers are skipped.
    """
    for layer in model.model.layers:
        if getattr(layer, "self_attn", None) is not None and hasattr(layer.self_attn, "unfuse_for_tp"):
            layer.self_attn.unfuse_for_tp()
        layer.mlp.unfuse_for_tp()
    model._unfused_for_tp = True
    # Override HF config's TP plan with our plan for unfused projections.
    model.config.base_model_tp_plan = TP_PLAN
