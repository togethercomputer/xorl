"""Parallelization plan for MiniMax M3."""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_ep_plan() -> ParallelPlan:
    return ParallelPlan(
        ep_plan={
            "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "model.layers.*.mlp.experts.down_proj": Shard(0),
        }
    )


__all__ = ["get_ep_plan"]
