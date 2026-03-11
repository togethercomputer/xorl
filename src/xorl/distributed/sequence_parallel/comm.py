from contextlib import nullcontext
from typing import Any, Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup


_DATA_PARALLEL_GROUP = None

_ULYSSES_SEQUENCE_PARALLEL_GROUP = {"default": None}
_ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP = {"default": None}
_ULYSSES_GROUP_KEY = "default"

_RINGATTN_GROUP = None

_UNIFIED_SEQUENCE_PARALLEL_GROUP = None
_UNIFIED_SEQUENCE_PARALLEL_CPU_GROUP = None


# ------------------------------ Data Parallel ------------------------------ #
def set_data_parallel_group(group: dist.ProcessGroup):
    """
    Set data parallel process group.
    """
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = group


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel process group.
    """
    global _DATA_PARALLEL_GROUP
    return _DATA_PARALLEL_GROUP


def get_data_parallel_rank() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel rank.
    """
    group = get_data_parallel_group()
    return dist.get_rank(group)


def get_data_parallel_world_size() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel world_size.
    """
    group = get_data_parallel_group()
    return dist.get_world_size(group)


# ----------------------------- Ulysses Parallel ---------------------------- #
def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup, group_key: str = "default"):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP[group_key] = group


def set_ulysses_sequence_parallel_cpu_group(group: dist.ProcessGroup, group_key: str = "default"):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP[group_key] = group


def set_ulysses_sequence_parallel_group_key(group_key: str = "default"):
    """
    Set ulysses sequence parallel process group key.
    """
    global _ULYSSES_GROUP_KEY
    _ULYSSES_GROUP_KEY = group_key


def get_ulysses_sequence_parallel_group_key() -> str:
    """
    Get ulysses sequence parallel group key.
    """
    global _ULYSSES_GROUP_KEY
    return _ULYSSES_GROUP_KEY


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    group_key = get_ulysses_sequence_parallel_group_key()
    if group_key not in _ULYSSES_SEQUENCE_PARALLEL_GROUP:
        raise RuntimeError(f"Unknown key {group_key} in Ulysses sequence parallel group!")
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP[group_key]


def get_ulysses_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel CPU process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP
    group_key = get_ulysses_sequence_parallel_group_key()
    if group_key not in _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP:
        raise RuntimeError(f"Unknown key {group_key} in Ulysses sequence parallel group!")
    return _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP[group_key]


def get_ulysses_sequence_parallel_group_by_key(group_key: str = "default") -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    if group_key not in _ULYSSES_SEQUENCE_PARALLEL_GROUP:
        raise RuntimeError(f"Unknown key {group_key} in Ulysses sequence parallel group!")
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP[group_key]


def get_ulysses_sequence_parallel_cpu_group_by_key(group_key: str = "default") -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel CPU process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP
    if group_key not in _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP:
        raise RuntimeError(f"Unknown key {group_key} in Ulysses sequence parallel group!")
    return _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP[group_key]


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


# ----------------------------- Ring Parallel ------------------------------ #


def set_ringattn_group(ringattn_group: dist.ProcessGroup):
    """
    Set ring attention process group.
    """
    global _RINGATTN_GROUP
    _RINGATTN_GROUP = ringattn_group


def get_ringattn_group(check_initialized=True):
    """Get the ring attention group the caller rank belongs to."""
    global _RINGATTN_GROUP
    if check_initialized:
        assert _RINGATTN_GROUP is not None, "ring attention group is not initialized"
    return _RINGATTN_GROUP


def get_ringattn_rank():
    """Return my rank for the ring attention group."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_ringattn_group())
    else:
        return 0


def get_ringattn_world_size():
    """Return world size for the ring attention group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=get_ringattn_group())
    else:
        return 0


# ----------------------------- Unified Parallel ---------------------------- #
def set_unified_sequence_parallel_group(group: dist.ProcessGroup):
    """
    Set unified sequence parallel process group.
    """
    global _UNIFIED_SEQUENCE_PARALLEL_GROUP
    _UNIFIED_SEQUENCE_PARALLEL_GROUP = group


def set_unified_sequence_parallel_cpu_group(group: dist.ProcessGroup):
    """
    Set unified sequence parallel process group.
    """
    global _UNIFIED_SEQUENCE_PARALLEL_CPU_GROUP
    _UNIFIED_SEQUENCE_PARALLEL_CPU_GROUP = group


def get_unified_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get unified sequence parallel process group.
    """
    global _UNIFIED_SEQUENCE_PARALLEL_GROUP
    return _UNIFIED_SEQUENCE_PARALLEL_GROUP


def get_unified_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get unified sequence parallel CPU process group.
    """
    global _UNIFIED_SEQUENCE_PARALLEL_CPU_GROUP
    return _UNIFIED_SEQUENCE_PARALLEL_CPU_GROUP


def get_unified_sequence_parallel_rank() -> int:
    """
    Get unified sequence parallel rank.
    """
    group = get_unified_sequence_parallel_group()
    return dist.get_rank(group) if group else 0


def get_unified_sequence_parallel_world_size() -> int:
    """
    Get unified sequence parallel world size.
    """
    group = get_unified_sequence_parallel_group()
    return dist.get_world_size(group) if group else 1


# ------------------------------- Initialize ------------------------------- #
def init_sequence_parallel(
    ulysses_size: int = 1, sep_dp: bool = False, ulysses_group_key: str = "default", ringattn_size: int = 1
):
    """
    Initialize unified sequence parallel.
    """
    global _RINGATTN_GROUP
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    global _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP

    set_ulysses_sequence_parallel_group(group=None, group_key="default")
    set_ulysses_sequence_parallel_cpu_group(group=None, group_key="default")

    if ulysses_size == 1 and ringattn_size == 1:
        return

    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    unified_sp_size = ulysses_size * ringattn_size
    assert world_size % unified_sp_size == 0
    data_parallel_size = world_size // unified_sp_size

    if ringattn_size > 1:
        assert _RINGATTN_GROUP is None, "Ring attention group has already been initialized!"
    if ulysses_size:
        assert (ulysses_group_key == "default" and _ULYSSES_SEQUENCE_PARALLEL_GROUP[ulysses_group_key] is None) or (
            ulysses_group_key != "default" and ulysses_group_key not in _ULYSSES_SEQUENCE_PARALLEL_GROUP
        ), f"Ulysses sequence parallel group ({ulysses_group_key}) has already been initialized!"
        assert (
            ulysses_group_key == "default" and _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP[ulysses_group_key] is None
        ) or (ulysses_group_key != "default" and ulysses_group_key not in _ULYSSES_SEQUENCE_PARALLEL_CPU_GROUP), (
            f"Ulysses sequence parallel ({ulysses_group_key}) group has already been initialized!"
        )

    for i in range(data_parallel_size):
        # build ulysses group
        if ulysses_size > 1:
            for j in range(ringattn_size):
                start_rank = i * unified_sp_size + j * ulysses_size
                end_rank = start_rank + ulysses_size
                ulysses_ranks = range(start_rank, end_rank)
                ulysses_group = dist.new_group(ulysses_ranks)
                ulysses_cpu_group = dist.new_group(ulysses_ranks, backend="gloo")
                if rank in ulysses_ranks:
                    set_ulysses_sequence_parallel_group(group=ulysses_group, group_key=ulysses_group_key)
                    set_ulysses_sequence_parallel_cpu_group(group=ulysses_cpu_group, group_key=ulysses_group_key)

        # build ring group
        if ringattn_size > 1:
            for j in range(ulysses_size):
                ring_global_ranks = range(i * unified_sp_size + j, (i + 1) * unified_sp_size, ulysses_size)
                ringattn_group = dist.new_group(ring_global_ranks)
                if rank in ring_global_ranks:
                    set_ringattn_group(ringattn_group=ringattn_group)

        # build unified sequence parallel group
        unified_sp_ranks = range(i * unified_sp_size, (i + 1) * unified_sp_size)
        sp_group = dist.new_group(unified_sp_ranks)
        sp_cpu_group = dist.new_group(unified_sp_ranks, backend="gloo")
        if rank in unified_sp_ranks:
            set_unified_sequence_parallel_group(group=sp_group)
            set_unified_sequence_parallel_cpu_group(group=sp_cpu_group)

    if sep_dp:
        for j in range(unified_sp_size):
            dp_ranks = range(j, world_size, unified_sp_size)
            dp_group = dist.new_group(dp_ranks)
            if rank in dp_ranks:
                set_data_parallel_group(dp_group)


class UlyssesGroupKeyManager:
    def __init__(self, group_key: str):
        self.group_key = group_key

    def __enter__(self):
        set_ulysses_sequence_parallel_group_key(group_key=self.group_key)

    def __exit__(self, *args: Any):
        set_ulysses_sequence_parallel_group_key(group_key="default")


def is_ulysses_sequence_parallel_initialized() -> bool:
    """
    Check if ulysses sequence parallel is initialized.
    """
    return get_ulysses_sequence_parallel_group() is not None


def is_ringattn_parallel_initialized() -> bool:
    """
    Check if ring attention parallel is initialized.
    """
    return get_ringattn_group() is not None


def get_ulysses_group_key_context(group_key: str = "default"):
    if not isinstance(group_key, str):
        raise RuntimeError(f"A Ulysses group key must be specified, now get: {group_key}")

    if group_key != "default":
        return UlyssesGroupKeyManager(group_key)
    else:
        return nullcontext()
