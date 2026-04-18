"""Parallelization plan and utilities for dense Qwen3 models."""

# TP plan for the base model (Qwen3Model).
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

    Splits ``qkv_proj`` → ``q_proj / k_proj / v_proj`` in attention,
    and ``gate_up_proj`` → ``gate_proj / up_proj`` in MLP, for every
    decoder layer.

    After unfusing, checkpoint keys from HuggingFace already match
    the model's parameter names — no merging handler is needed.
    """
    for layer in model.model.layers:
        layer.self_attn.unfuse_for_tp()
        layer.mlp.unfuse_for_tp()
    model._unfused_for_tp = True
    # Override HF config's TP plan with our plan for unfused projections.
    model.config.base_model_tp_plan = TP_PLAN
