import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union

import torch
from torch import distributed as dist

from ..utils.device import get_device_id, get_device_type


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


_cpu_world_group: Optional["ProcessGroup"] = None


def all_gather(tensor: "torch.Tensor", world_size: int) -> "torch.Tensor":
    """
    Gathers the tensor from all ranks and concats them along the first dim.
    """
    output_tensor = torch.empty(world_size * tensor.numel(), dtype=tensor.dtype, device=get_device_type())
    dist.all_gather_into_tensor(output_tensor, tensor)
    return output_tensor.view(-1, *tensor.size()[1:])


def all_reduce(
    data: Union[int, float, List[Union[int, float]], "torch.Tensor"],
    op: Literal["mean", "sum", "max", "min"] = "mean",
    group: Optional["ProcessGroup"] = None,
) -> Union[int, float, List[Union[int, float]]]:
    """
    Performs all reduce in the given process group.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    reduce_ops = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }

    if isinstance(data, torch.Tensor):
        reduce_tensor = data
        group_size = dist.get_world_size(group=group)
        if group_size > 1:
            dist.all_reduce(reduce_tensor, op=reduce_ops[op], group=group)
    else:
        reduce_tensor = torch.tensor(data, dtype=torch.float, device="cpu")
        group_size = dist.get_world_size(group=group)
        if group_size > 1:
            reduce_tensor = all_reduce_metadata_tensor(reduce_tensor, op=reduce_ops[op], group=group, device="cpu")

    if op == "mean":  # ReduceOp.AVG is not supported by the NPU backend
        reduce_tensor /= group_size

    if reduce_tensor.numel() == 1:
        return reduce_tensor.item()
    else:
        return reduce_tensor.tolist()


def _backend_name(backend: Any) -> str:
    return str(backend).lower()


def get_cpu_world_group() -> Optional["ProcessGroup"]:
    """Return a Gloo world group for process-wide CPU coordination."""
    global _cpu_world_group

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return None

    if "gloo" in _backend_name(dist.get_backend()):
        return None

    if _cpu_world_group is None:
        _cpu_world_group = dist.new_group(backend="gloo")
    return _cpu_world_group


def _group_is_world(group: Optional["ProcessGroup"]) -> bool:
    if group is None:
        return True
    return dist.get_world_size(group=group) == dist.get_world_size()


def all_reduce_metadata_tensor(
    tensor: "torch.Tensor",
    op: "dist.ReduceOp" = dist.ReduceOp.SUM,
    group: Optional["ProcessGroup"] = None,
    device: Optional[Union[str, "torch.device"]] = None,
) -> "torch.Tensor":
    """All-reduce non-autograd metadata, preferring Gloo for world reductions.

    Scalar counters such as valid-token totals do not need the NCCL data path.
    When the requested group covers all ranks, reduce them through a Gloo world
    group and copy the reduced value back to ``device``. Subgroups still use the
    requested process group to preserve their exact membership.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    target_device = torch.device(device) if device is not None else tensor.device
    if dist.get_world_size(group=group) <= 1:
        return tensor.detach().to(target_device).clone()

    if _group_is_world(group):
        cpu_group = get_cpu_world_group()
        if cpu_group is not None or "gloo" in _backend_name(dist.get_backend()):
            reduced = tensor.detach().to("cpu").clone()
            dist.all_reduce(reduced, op=op, group=cpu_group)
            return reduced.to(target_device)

    reduced = tensor.detach().clone()
    backend = dist.get_backend(group) if group is not None else dist.get_backend()
    if "nccl" in _backend_name(backend) and reduced.device.type != "cuda":
        reduced = reduced.to(get_device_type())
    dist.all_reduce(reduced, op=op, group=group)
    return reduced.to(target_device)


def distributed_barrier(group: Optional["ProcessGroup"] = None) -> None:
    """Synchronize ranks without forcing CPU-only barriers onto NCCL."""
    if not dist.is_available() or not dist.is_initialized():
        return

    if group is None:
        cpu_group = get_cpu_world_group()
        if cpu_group is not None:
            dist.barrier(group=cpu_group)
            return

    barrier_kwargs = {}
    backend = dist.get_backend(group) if group is not None else dist.get_backend()
    if "nccl" in _backend_name(backend):
        barrier_kwargs["device_ids"] = [get_device_id()]
    dist.barrier(group=group, **barrier_kwargs)


@contextmanager
def main_process_first(local_only: bool = True) -> None:
    """
    A context manager for torch distributed environment to do something on the main process firstly.
    """
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        is_main_process = int(os.getenv("LOCAL_RANK")) == 0 if local_only else int(os.getenv("RANK")) == 0
        try:
            if not is_main_process:
                distributed_barrier()
            yield
        finally:
            if is_main_process:
                distributed_barrier()
    else:
        yield


def execute_in_order(task: Callable, *, local_only: bool = True, **kwargs) -> Any:
    """
    Executes the task in the order of rank.
    """
    world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1") if local_only else os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("LOCAL_RANK", "1") if local_only else os.getenv("RANK", "1"))
    if world_size > 1:
        distributed_barrier()
        for i in range(world_size):
            if rank == i:
                result = task(**kwargs)
                distributed_barrier()
            else:
                distributed_barrier()

        return result
    else:
        return task(**kwargs)
