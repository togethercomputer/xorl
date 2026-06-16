"""Parallelization plan and utilities for OLMo-2 models.

The plan keeps the residual stream **Replicate** across decoder layers
(rather than the torchtitan SP/Shard(1) pattern). This composes with
xorl's existing loss path: ``vocab_parallel_cross_entropy`` bypasses
``lm_head`` and matmuls ``hidden_states @ lm_head.weight.t()`` directly,
which expects a full ``[B, S, H]`` per rank — incompatible with a
sequence-sharded residual.

The TP boundaries are therefore the standard colwise/rowwise pair:

  * Embedding outputs Replicate (default).
  * q/k/v_proj and gate/up_proj are ``ColwiseParallel()`` (default
    Replicate input, Shard(-1) output, ``use_local_output=True``) — the
    block sees plain local tensors with ``hidden / tp`` per rank, so
    rotary, flash-attention and the rest of the attention/MLP internals
    run unchanged on local tensors.
  * o_proj/down_proj are ``RowwiseParallel()`` (default Shard(-1) input,
    Replicate output) — every block returns full hidden, so post-norms
    and the residual addition see Replicate without any extra plumbing.
  * lm_head is ``ColwiseParallel()`` (default Replicate input, Shard(-1)
    output) for use with vocab-parallel cross-entropy.

OLMo-2's full-axis ``q_norm``/``k_norm`` (over ``num_heads * head_dim``,
not per-head) doesn't compose with the stock styles: under colwise q/k_proj
the q/k tensors arrive with a sharded hidden axis, so a full-hidden weight
can't be applied directly. ``LocalAxisRMSNormShard`` (in ``tp_styles.py``)
shards the 1-D weight on dim 0 so each rank's slice matches its local q/k
slice; ``Olmo2QKRMSNorm.forward`` (in ``modeling_olmo2.py``) detects the
sharded weight and runs the fused op on locals — computing a local-axis
RMS that matches HuggingFace's ``Olmo2RMSNorm`` reference under TP.
"""

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from xorl.models.transformers.olmo2.tp_styles import LocalAxisRMSNormShard


# Plan for the base model (Olmo2Model). Keys use wildcard patterns relative
# to the base model prefix, which the parallel applier prepends with ``model.``.
TP_PLAN = {
    "embed_tokens": "embedding",
    "layers.*.self_attn.q_proj": ColwiseParallel(),
    "layers.*.self_attn.k_proj": ColwiseParallel(),
    "layers.*.self_attn.v_proj": ColwiseParallel(),
    # Full-axis QK norms: weight sharded on dim 0 so each rank's slice
    # matches the local hidden slice from colwise q/k_proj.
    "layers.*.self_attn.q_norm": LocalAxisRMSNormShard(),
    "layers.*.self_attn.k_norm": LocalAxisRMSNormShard(),
    "layers.*.self_attn.o_proj": RowwiseParallel(),
    # post_attention_layernorm sees a Replicate input after the rowwise
    # all-reduce in o_proj — no plan entry needed.
    "layers.*.mlp.gate_proj": ColwiseParallel(),
    "layers.*.mlp.up_proj": ColwiseParallel(),
    "layers.*.mlp.down_proj": RowwiseParallel(),
    # post_feedforward_layernorm and the final norm are also Replicate-fed.
}

# Plan for top-level modules on the CausalLM wrapper.
MODEL_TP_PLAN = {
    "lm_head": ColwiseParallel(),
}


def unfuse_for_tp(model):
    """Unfuse fused projections for tensor parallelism compatibility.

    Splits ``qkv_proj`` -> ``q_proj / k_proj / v_proj`` in attention,
    and ``gate_up_proj`` -> ``gate_proj / up_proj`` in MLP, for every
    decoder layer.

    After unfusing, checkpoint keys from HuggingFace already match
    the model's parameter names -- no merging handler is needed.
    """
    for layer in model.model.layers:
        layer.self_attn.unfuse_for_tp()
        layer.mlp.unfuse_for_tp()
    model._unfused_for_tp = True
    # Override HF config's TP plan with our plan for unfused projections.
    model.config.base_model_tp_plan = TP_PLAN
