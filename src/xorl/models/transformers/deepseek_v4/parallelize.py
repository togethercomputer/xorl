"""Parallelization plan and utilities for DeepSeek V4 models.

**``TP_PLAN`` / ``MODEL_TP_PLAN`` below are reserved for future TP support
and are NOT consumed in V0.** ``DeepSeekV4Attention.__init__`` asserts
``tp_size == 1``, so the colwise/rowwise hints here have no effect on
training today — they're a reference target for when xorl SP gather
lands. If TP is wired up before this note is removed,
``DeepSeekV4Attention`` will need updates first.

V0 only consumes ``get_ep_plan()`` (EP-sharding for routed experts)
plus the FSDP wrapping done by ``build_parallelize_model``. The EP plan
matches the qwen3_5_moe convention.

Reference (miles):
- ``self_attention.wq_a`` / ``self_attention.wkv`` are ``TELinear`` with
  ``parallel_mode="duplicated"`` (replicated, NOT TP-sharded).
- ``self_attention.wq_b`` is ``ColumnParallelLinear`` (colwise).
- ``self_attention.wo_a`` is ``ColumnParallelLinear`` (colwise) operating in
  ``[n_groups, o_lora_rank]`` layout.
- ``self_attention.wo_b`` is ``RowParallelLinear`` (rowwise).
- ``self_attention.attn_sink`` is sharded along head dim (axis 0).
"""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


# TP plan for the base model. MoE expert weights are EP-sharded, not TP.
# wq_a / wkv are intentionally absent: per miles they are replicated.
TP_PLAN = {
    "embed_tokens": "embedding",
    "layers.*.self_attn.wq_b": "colwise",
    "layers.*.self_attn.wo_a": "colwise",
    "layers.*.self_attn.wo_b": "rowwise",
    # Per-head fp32 attn_sink — sharded along the head dim alongside Q/O heads.
    "layers.*.self_attn.attn_sink": "colwise",
}

MODEL_TP_PLAN = {
    "lm_head": "colwise_rep",
}


def get_ep_plan():
    """EP plan for DeepSeek V4 MoE.

    Mirrors the qwen3_5_moe convention: experts stacked as
    ``[num_experts, H, 2I]`` (fused gate/up) and ``[num_experts, I, H]``
    for ``down_proj``. LoRA weights live at the global shape and are
    sharded by ``parallel_plan.apply()``.
    """
    ep_plan = {
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
        "model.layers.*.mlp.experts.gate_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.gate_proj_lora_B": Shard(0),
        "model.layers.*.mlp.experts.up_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.up_proj_lora_B": Shard(0),
        "model.layers.*.mlp.experts.down_proj_lora_A": Shard(0),
        "model.layers.*.mlp.experts.down_proj_lora_B": Shard(0),
    }
    return ParallelPlan(ep_plan=ep_plan)
