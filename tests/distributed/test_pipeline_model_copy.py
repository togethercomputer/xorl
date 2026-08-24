import torch
import torch.distributed as dist
from torch import nn

from xorl.distributed.pipeline_parallel import _deepcopy_pipeline_model


def test_pipeline_model_copy_preserves_process_groups_and_parameter_metadata(tmp_path) -> None:
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=1,
            init_method=f"file://{tmp_path / 'gloo-init'}",
        )
    try:
        group = dist.group.WORLD
        holder = nn.Linear(2, 2)
        holder.cp_group = group
        holder.weight._keep_fp32 = True

        holder_copy = _deepcopy_pipeline_model(holder)

        assert holder_copy.cp_group is group
        assert holder_copy.weight is not holder.weight
        assert torch.equal(holder_copy.weight, holder.weight)
        assert holder_copy.weight._keep_fp32 is True
    finally:
        if owns_group:
            dist.destroy_process_group()
