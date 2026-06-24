"""Parallelization plan for GLM-5 / GLM-5.1."""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


MODEL_TP_PLAN = {"lm_head": "colwise_rep"}


def get_ep_plan() -> ParallelPlan:
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


__all__ = ["MODEL_TP_PLAN", "get_ep_plan"]
