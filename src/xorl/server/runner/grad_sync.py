from contextlib import contextmanager
from typing import Any, Iterator, Mapping


def should_defer_hsdp_all_reduce(model: Any, train_config: Mapping[str, Any], n_micro_batches: int) -> bool:
    if (
        n_micro_batches <= 1
        or not train_config.get("defer_grad_sync_in_accumulation", False)
        or train_config.get("data_parallel_replicate_size", 1) <= 1
        or not hasattr(model, "set_requires_all_reduce")
    ):
        return False

    from torch.distributed._composable.fsdp.fully_shard import FSDPModule  # noqa: PLC0415

    return isinstance(model, FSDPModule)


@contextmanager
def hsdp_all_reduce_microbatch_context(
    model: Any,
    defer_all_reduce: bool,
    *,
    is_last_micro_batch: bool,
) -> Iterator[None]:
    if defer_all_reduce:
        model.set_requires_all_reduce(is_last_micro_batch)
    try:
        yield
    finally:
        if defer_all_reduce:
            model.set_requires_all_reduce(True)
