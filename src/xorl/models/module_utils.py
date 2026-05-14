import json
import math
import os
import pickle
import re
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Set, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import distributed as dist
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed._tensor import Shard as DTShard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from xorl.distributed.parallel_state import get_parallel_state
from xorl.lora.modules.linear import LoraLinear
from xorl.models.checkpoint_handlers.buffers import (  # noqa: F401
    FUSED_EXPERT_PATTERN,
    ExpertWeightBuffer,
    checkpoint_has_per_expert_weights,
    parse_expert_full_key,
    parse_expert_key,
)
from xorl.ops.loss import get_loss_function
from xorl.utils import logging
from xorl.utils.device import get_device_id, get_device_type, synchronize
from xorl.utils.helper import empty_cache, get_dtype_size


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from xorl.distributed.parallel_plan import ParallelPlan
    from xorl.models.checkpoint_handlers.base import CheckpointHandler

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)

_weight_load_group = None
_grouped_weight_load_group = None
_grouped_weight_load_group_ranks: Optional[Tuple[int, ...]] = None
_grouped_dense_weight_load_group = None
_grouped_dense_weight_load_group_ranks: Optional[Tuple[int, ...]] = None
_save_sync_group = None
_save_sync_group_backend: Optional[str] = None
_cpu_save_device_mesh_cache: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[str, ...]], DeviceMesh] = {}
_UNSUPPORTED_DTENSOR_ROOT_GATHER = object()


def _get_weight_load_group():
    """Get or create a dedicated process group for weight loading.

    Weight loading can spend many minutes in rank-0 shard I/O between collectives.
    Use a separate process group with a larger timeout so NCCL does not trip the
    default watchdog while other ranks are waiting for the next broadcast.
    """
    global _weight_load_group
    if _weight_load_group is None and dist.is_initialized():
        timeout_sec = int(os.getenv("XORL_WEIGHT_LOAD_TIMEOUT_SEC", "7200"))
        _weight_load_group = dist.new_group(backend=dist.get_backend(), timeout=timedelta(seconds=timeout_sec))
    return _weight_load_group


def _get_object_broadcast_device(group) -> Optional[torch.device]:
    """Return the device to use for object broadcasts on the given process group."""
    if not dist.is_available() or not dist.is_initialized():
        return None

    backend = dist.get_backend() if group is None else dist.get_backend(group)
    if backend == "nccl":
        return torch.device(f"{get_device_type()}:{get_device_id()}")
    return None


def _get_cpu_save_device_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Mirror a DTensor device mesh onto CPU/Gloo for low-memory save gathers."""
    mesh_tensor = device_mesh.mesh.detach().cpu().contiguous()
    mesh_dim_names = tuple(device_mesh.mesh_dim_names or tuple(f"dim_{idx}" for idx in range(mesh_tensor.ndim)))
    cache_key = (
        device_mesh.device_type,
        tuple(mesh_tensor.size()),
        tuple(int(rank) for rank in mesh_tensor.reshape(-1).tolist()),
        mesh_dim_names,
    )
    cached_mesh = _cpu_save_device_mesh_cache.get(cache_key)
    if cached_mesh is None:
        backend_override = tuple(("gloo", None) for _ in range(mesh_tensor.ndim))
        cached_mesh = DeviceMesh(
            device_type="cpu",
            mesh=mesh_tensor,
            mesh_dim_names=mesh_dim_names,
            backend_override=backend_override,
        )
        _cpu_save_device_mesh_cache[cache_key] = cached_mesh
    return cached_mesh


def _get_mesh_group_ranks(device_mesh: DeviceMesh, mesh_dim: int) -> Tuple[int, ...]:
    """Return global ranks in the current rank's mesh-dim subgroup."""
    mesh_tensor = device_mesh.mesh.detach().cpu()
    coord = device_mesh.get_coordinate()
    if coord is None:
        raise RuntimeError("Current rank is not part of the provided device mesh.")

    index = []
    for dim_idx, coord_idx in enumerate(coord):
        index.append(slice(None) if dim_idx == mesh_dim else coord_idx)
    group_ranks = mesh_tensor[tuple(index)].reshape(-1).tolist()
    return tuple(int(rank) for rank in group_ranks)


def _get_rank_coordinate(device_mesh: DeviceMesh, global_rank: int) -> Optional[Tuple[int, ...]]:
    """Return the coordinate of a global rank in a device mesh."""
    mesh_tensor = device_mesh.mesh.detach().cpu()
    matches = (mesh_tensor == global_rank).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return None
    return tuple(int(coord) for coord in matches[0].tolist())


def _get_mesh_group_root_rank(device_mesh: DeviceMesh, mesh_dim: int, target_index: int) -> int:
    """Return the global rank in the current mesh-dim subgroup at ``target_index``."""
    mesh_tensor = device_mesh.mesh.detach().cpu()
    coord = device_mesh.get_coordinate()
    if coord is None:
        raise RuntimeError("Current rank is not part of the provided device mesh.")
    index = list(coord)
    index[mesh_dim] = target_index
    return int(mesh_tensor[tuple(index)].item())


def _gather_sharded_tensor_to_group_root(
    local_tensor: "torch.Tensor",
    *,
    tensor_dim: int,
    full_dim_size: int,
    group,
    group_ranks: Sequence[int],
    dst_rank: int,
) -> Optional["torch.Tensor"]:
    """Gather shards from one mesh-dim subgroup to a single root rank."""
    if not group_ranks:
        return None

    local_cpu = local_tensor.detach().cpu().contiguous()
    if dst_rank not in group_ranks:
        group_dst = 0
        keep_result = False
    else:
        group_dst = group_ranks.index(dst_rank)
        keep_result = dist.get_rank() == dst_rank

    local_group_rank = group_ranks.index(dist.get_rank())
    chunk_size = math.ceil(full_dim_size / len(group_ranks))
    padded_shape = list(local_cpu.shape)
    padded_shape[tensor_dim] = chunk_size
    padded = local_cpu.new_zeros(padded_shape)
    if local_cpu.shape[tensor_dim] > 0:
        index = [slice(None)] * local_cpu.ndim
        index[tensor_dim] = slice(0, local_cpu.shape[tensor_dim])
        padded[tuple(index)] = local_cpu

    gather_list = None
    if local_group_rank == group_dst:
        gather_list = [padded.new_empty(padded_shape) for _ in range(len(group_ranks))]

    dist.gather(padded, gather_list=gather_list, group=group, group_dst=group_dst)

    if local_group_rank != group_dst:
        return None

    shards = []
    for shard_idx, gathered in enumerate(gather_list or []):
        start = shard_idx * chunk_size
        remaining = max(full_dim_size - start, 0)
        take = min(chunk_size, remaining)
        if take <= 0:
            continue
        if take != chunk_size:
            gathered = gathered.narrow(tensor_dim, 0, take)
        shards.append(gathered)

    if not shards:
        result = padded.new_empty(padded_shape)
        result = result.narrow(tensor_dim, 0, 0)
    elif len(shards) == 1:
        result = shards[0]
    else:
        result = torch.cat(shards, dim=tensor_dim)

    return result if keep_result else None


def _gather_dtensor_to_rank(raw_tensor: DTensor, dst_rank: int) -> Union["torch.Tensor", None, object]:
    """Gather a DTensor to one rank on CPU/Gloo when placements allow it."""
    placements = tuple(raw_tensor.placements)
    shard_mesh_dims = [mesh_dim for mesh_dim, placement in enumerate(placements) if isinstance(placement, DTShard)]
    if len(shard_mesh_dims) == 0:
        local_tensor = raw_tensor.to_local()
        if hasattr(local_tensor, "wait"):
            local_tensor = local_tensor.wait()
        return local_tensor.detach().cpu() if dist.get_rank() == dst_rank else None

    cpu_mesh = _get_cpu_save_device_mesh(raw_tensor.device_mesh)
    dst_coord = _get_rank_coordinate(cpu_mesh, dst_rank)
    if dst_coord is None:
        return _UNSUPPORTED_DTENSOR_ROOT_GATHER

    local_tensor = raw_tensor.to_local()
    if hasattr(local_tensor, "wait"):
        local_tensor = local_tensor.wait()
    partial = local_tensor.detach().cpu().contiguous()
    full_shape = tuple(raw_tensor.size())

    for shard_mesh_dim in reversed(shard_mesh_dims):
        placement = placements[shard_mesh_dim]
        if not isinstance(placement, DTShard):
            return _UNSUPPORTED_DTENSOR_ROOT_GATHER
        group = cpu_mesh.get_group(shard_mesh_dim if cpu_mesh.ndim > 1 else None)
        group_ranks = _get_mesh_group_ranks(cpu_mesh, shard_mesh_dim if cpu_mesh.ndim > 1 else 0)
        if not group_ranks:
            return _UNSUPPORTED_DTENSOR_ROOT_GATHER

        if len(shard_mesh_dims) == 1:
            stage_dst_rank = dst_rank
        else:
            stage_dst_rank = _get_mesh_group_root_rank(cpu_mesh, shard_mesh_dim, dst_coord[shard_mesh_dim])

        partial = _gather_sharded_tensor_to_group_root(
            partial,
            tensor_dim=placement.dim,
            full_dim_size=full_shape[placement.dim],
            group=group,
            group_ranks=group_ranks,
            dst_rank=stage_dst_rank,
        )

        coord = cpu_mesh.get_coordinate()
        if coord is None:
            return _UNSUPPORTED_DTENSOR_ROOT_GATHER
        if coord[shard_mesh_dim] != dst_coord[shard_mesh_dim]:
            return None
        if partial is None:
            return None

    return partial if dist.get_rank() == dst_rank else None


def _materialize_tensor_for_save(
    tensor: "torch.Tensor",
    dst_rank: Optional[int] = None,
) -> Optional["torch.Tensor"]:
    """Materialize a parameter/buffer for save without gathering full DTensors on CUDA."""
    raw_tensor = tensor.data if hasattr(tensor, "data") else tensor
    if isinstance(raw_tensor, DTensor):
        if dst_rank is not None:
            gathered = _gather_dtensor_to_rank(raw_tensor, dst_rank)
            if gathered is not _UNSUPPORTED_DTENSOR_ROOT_GATHER:
                return gathered
            logger.warning_once(
                "Falling back to replicated DTensor save materialization for unsupported placements: "
                f"{raw_tensor.placements}"
            )

        local_tensor = raw_tensor.to_local()
        if hasattr(local_tensor, "wait"):
            local_tensor = local_tensor.wait()
        local_cpu = local_tensor.detach().cpu()
        cpu_mesh = _get_cpu_save_device_mesh(raw_tensor.device_mesh)
        cpu_dtensor = DTensor.from_local(
            local_cpu,
            device_mesh=cpu_mesh,
            placements=raw_tensor.placements,
            run_check=False,
            shape=raw_tensor.size(),
            stride=raw_tensor.stride(),
        )
        replicated = cpu_dtensor.redistribute(
            device_mesh=cpu_mesh,
            placements=[Replicate() for _ in cpu_dtensor.placements],
        ).to_local()
        if hasattr(replicated, "wait"):
            replicated = replicated.wait()
        return replicated

    if dst_rank is not None and dist.is_available() and dist.is_initialized() and dist.get_rank() != dst_rank:
        return None

    if hasattr(raw_tensor, "full_tensor"):
        return raw_tensor.full_tensor()

    return raw_tensor


def _broadcast_object_list_weight_load(obj_list: List[Any], src: int) -> None:
    """Broadcast Python metadata for weight loading on the dedicated load group."""
    group = _get_weight_load_group()
    _broadcast_object_list(obj_list, src=src, group=group)


def _broadcast_object_list(obj_list: List[Any], src: int, group) -> None:
    """Broadcast Python metadata on the provided process group."""
    device = _get_object_broadcast_device(group)
    if device is None:
        dist.broadcast_object_list(obj_list, src=src, group=group)
        return

    is_source = dist.get_rank() == src
    payload = pickle.dumps(obj_list, protocol=pickle.HIGHEST_PROTOCOL) if is_source else b""
    size_tensor = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    dist.broadcast(size_tensor, src=src, group=group)

    if is_source:
        payload_tensor = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    else:
        payload_tensor = torch.empty(int(size_tensor.item()), dtype=torch.uint8, device=device)

    dist.broadcast(payload_tensor, src=src, group=group)

    if not is_source:
        obj_list[:] = pickle.loads(bytes(payload_tensor.cpu().tolist()))


def _normalize_checkpoint_key_for_filter(key: str) -> Optional[str]:
    """Normalize raw checkpoint keys for lightweight load-time filtering."""
    if key.startswith("vision_tower.") or key.startswith("mm_projector."):
        return None
    if key.startswith("language_model."):
        return key.removeprefix("language_model.")
    return key


def _matches_checkpoint_skip_key_pattern(key: str, model: object) -> bool:
    """Return True when a raw or converted checkpoint key is model-declared skip state."""
    for pattern in getattr(model, "_checkpoint_skip_key_patterns", ()):
        if re.match(pattern, key):
            return True
    return False


_FUSED_EXPERT_CHECKPOINT_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up|down)_proj(?:\..+)?$")
_FFN_EXPERT_CHECKPOINT_PATTERN = re.compile(r"^(?:model\.)?layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(?:weight|scale)$")


def _is_checkpoint_expert_key(key: str) -> bool:
    """Return True when a raw checkpoint key belongs to MoE expert weights."""
    normalized = _normalize_checkpoint_key_for_filter(key)
    if normalized is None:
        return False
    return (
        parse_expert_full_key(normalized) is not None
        or FUSED_EXPERT_PATTERN.match(normalized) is not None
        or _FUSED_EXPERT_CHECKPOINT_PATTERN.match(normalized) is not None
        or _FFN_EXPERT_CHECKPOINT_PATTERN.match(normalized) is not None
    )


def _is_expert_parameter_name(parameter_name: str, parallel_plan: Optional["ParallelPlan"]) -> bool:
    """Return True when a model parameter is EP-sharded expert state."""
    if parallel_plan is not None:
        is_expert = getattr(parallel_plan, "is_expert_parameter", None)
        if callable(is_expert):
            return bool(is_expert(parameter_name))
        private_is_expert = getattr(parallel_plan, "_is_expert_parameter", None)
        if callable(private_is_expert):
            return bool(private_is_expert(parameter_name))
    return FUSED_EXPERT_PATTERN.match(parameter_name) is not None


def _get_grouped_weight_load_group(parallel_state) -> Optional[ProcessGroup]:
    """Return the subgroup used by grouped weight loading.

    Grouped loading is only meaningful when EP is enabled: one leader per
    ``ep_fsdp`` group reads and fan-outs tensors to the ranks that share that
    expert shard. This keeps expert weights correct without forcing a global
    rank-0 to materialize every expert tensor. Use a dedicated process group
    with the weight-load timeout so long checkpoint reads do not inherit the
    shorter default timeout from the DeviceMesh subgroup.
    """
    global _grouped_weight_load_group, _grouped_weight_load_group_ranks
    if _grouped_weight_load_group is not None:
        return _grouped_weight_load_group
    if (
        parallel_state is None
        or not getattr(parallel_state, "ep_enabled", False)
        or getattr(parallel_state, "ep_fsdp_device_mesh", None) is None
    ):
        return None

    if not dist.is_available() or not dist.is_initialized():
        return None

    mesh_tensor = parallel_state.ep_fsdp_device_mesh.mesh.detach().cpu().contiguous()
    if mesh_tensor.ndim == 0:
        return None

    timeout_sec = int(os.getenv("XORL_WEIGHT_LOAD_TIMEOUT_SEC", "7200"))
    backend = dist.get_backend()
    rank = dist.get_rank()
    created_group = None
    created_ranks = None
    for group_ranks_tensor in mesh_tensor.view(-1, mesh_tensor.size(-1)):
        ranks = tuple(int(member_rank) for member_rank in group_ranks_tensor.tolist())
        group = dist.new_group(ranks=list(ranks), backend=backend, timeout=timedelta(seconds=timeout_sec))
        if rank in ranks:
            created_group = group
            created_ranks = ranks

    _grouped_weight_load_group = created_group
    _grouped_weight_load_group_ranks = created_ranks
    return _grouped_weight_load_group


def _get_grouped_weight_load_prefetch_count(shard_count: int) -> int:
    """Prefetch depth for grouped loading.

    We intentionally keep this small. Grouped loading already reduces reader
    fan-out, so large per-rank prefetch would just recreate the WekaFS stampede
    that motivated this mode.
    """
    configured = int(os.getenv("XORL_GROUPED_WEIGHT_LOAD_PREFETCH_COUNT", "2"))
    return max(1, min(configured, shard_count))


def _get_grouped_dense_weight_load_group():
    """Return the per-node process group used for grouped dense/shared replication.

    Grouped loading already uses EP-FSDP subgroups for expert tensors. Dense and
    shared weights should not use an all-world metadata broadcast, but loading
    them on every rank still creates avoidable checkpoint I/O. We instead create
    one node-local group per host and broadcast dense/shared tensors inside that
    group so each shard is read once per node rather than once per rank.
    """
    global _grouped_dense_weight_load_group, _grouped_dense_weight_load_group_ranks
    if _grouped_dense_weight_load_group is not None:
        return _grouped_dense_weight_load_group

    if not dist.is_available() or not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    if world_size <= 1 or local_world_size <= 1:
        _grouped_dense_weight_load_group_ranks = (dist.get_rank(),)
        return None

    timeout_sec = int(os.getenv("XORL_WEIGHT_LOAD_TIMEOUT_SEC", "7200"))
    rank = dist.get_rank()
    created_group = None
    created_ranks = None
    backend = dist.get_backend()
    for start_rank in range(0, world_size, local_world_size):
        ranks = tuple(range(start_rank, min(start_rank + local_world_size, world_size)))
        group = dist.new_group(ranks=list(ranks), backend=backend, timeout=timedelta(seconds=timeout_sec))
        if rank in ranks:
            created_group = group
            created_ranks = ranks

    _grouped_dense_weight_load_group = created_group
    _grouped_dense_weight_load_group_ranks = created_ranks
    return _grouped_dense_weight_load_group


def _get_checkpoint_keys(weights_path: str) -> Optional[Set[str]]:
    """
    Get checkpoint weight keys from the index file without loading tensors.

    For sharded checkpoints, reads the index JSON file to get all keys.
    For single-file checkpoints, reads keys from the safetensors metadata.

    Args:
        weights_path: Path to the checkpoint directory or file.

    Returns:
        Set of weight keys, or None if keys cannot be retrieved.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False}

    # Try sharded safetensors index
    resolved_index = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_index:
        try:
            with open(resolved_index, "r") as f:
                index_data = json.load(f)
            return set(index_data.get("weight_map", {}).keys())
        except Exception:
            return None

    # Try single safetensors file
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        try:
            with safe_open(resolved_weight_file, framework="pt", device="cpu") as f:
                return set(f.keys())
        except Exception:
            return None

    # Try sharded pytorch index
    resolved_index = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_index:
        try:
            with open(resolved_index, "r") as f:
                index_data = json.load(f)
            return set(index_data.get("weight_map", {}).keys())
        except Exception:
            return None

    return None


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.

    Borrowed from: https://github.com/huggingface/accelerate/blob/v1.0.0rc1/src/accelerate/big_modeling.py#L57
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module: "nn.Module", name: str, param: "nn.Parameter"):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            # When we have a case of tensor2 = tensor1, it would call the set_attr
            # of param, which in turn would call the register_parameter API.
            # In this case, the new param is already on meta-device, since it was moved
            # previously when it was initialized. Hence, when resetting, you can
            # directly assign that tensor instead of re-init. If you re-init you would
            # lose the relationship.
            module._parameters[name] = (
                param
                if param.device == torch.device("meta")
                else param_cls(module._parameters[name].to("meta"), **kwargs)
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


@dataclass
class StateDictIterator:
    filepath: str

    def load_all(self) -> Dict[str, "torch.Tensor"]:
        """Bulk-load all tensors from the shard file into CPU memory.

        Faster than lazy per-tensor reading because the OS can optimize
        the I/O pattern (one sequential read instead of many small reads
        interleaved with compute/network).
        """
        if self.filepath.endswith(".safetensors"):
            return load_file(self.filepath, device="cpu")
        else:
            return torch.load(self.filepath, map_location="cpu", weights_only=True)

    def load_filtered(
        self,
        skip_key_fn: Callable[[str], bool],
    ) -> Tuple[Dict[str, "torch.Tensor"], List[str]]:
        """Load tensors from the shard, skipping keys where *skip_key_fn* returns True.

        Unlike ``load_all()`` which bulk-reads everything, this uses lazy
        iteration (``safe_open``) so tensor data for skipped keys is never
        read from disk — a significant I/O saving when most keys are skipped
        (e.g., EP-aware loading where each rank only needs its own experts).

        Returns:
            (state_dict, skipped_keys) — loaded tensors and the list of keys
            whose tensor data was *not* read.
        """
        skipped: List[str] = []
        if self.filepath.endswith(".safetensors"):
            result: Dict[str, torch.Tensor] = {}
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if skip_key_fn(key):
                        skipped.append(key)
                    else:
                        result[key] = f.get_tensor(key)
            return result, skipped
        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True)
            result = {}
            for k, v in state_dict.items():
                if skip_key_fn(k):
                    skipped.append(k)
                else:
                    result[k] = v
            return result, skipped

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


def _prefetch_shards(
    state_dict_iterators: List["StateDictIterator"],
    prefetch_count: int = 1,
) -> Generator[Dict[str, "torch.Tensor"], None, None]:
    """Yield bulk-loaded shard dicts with N-shard lookahead.

    Background threads pre-load upcoming shards from disk while the caller
    processes (dispatches / broadcasts) the current one, overlapping disk I/O
    with compute and network traffic.

    Args:
        state_dict_iterators: Shard iterators to load.
        prefetch_count: Number of shards to pre-load ahead. Higher values use
            more CPU memory but can increase NFS throughput via concurrent reads.
            Default 1 (single background thread, same as before).
    """

    if not state_dict_iterators:
        return

    n = len(state_dict_iterators)
    # Submit up to (prefetch_count + 1) tasks: 1 for immediate use + lookahead
    initial_submit = min(prefetch_count + 1, n)

    with ThreadPoolExecutor(max_workers=max(prefetch_count, 1)) as pool:
        pending: deque = deque()
        for i in range(initial_submit):
            pending.append(pool.submit(state_dict_iterators[i].load_all))

        next_idx = initial_submit

        for _ in range(n):
            state_dict = pending.popleft().result()

            if next_idx < n:
                pending.append(pool.submit(state_dict_iterators[next_idx].load_all))
                next_idx += 1

            yield state_dict


def _prefetch_shards_filtered(
    state_dict_iterators: List["StateDictIterator"],
    skip_key_fn: Callable[[str], bool],
    prefetch_count: int = 1,
) -> Generator[Tuple[Dict[str, "torch.Tensor"], List[str]], None, None]:
    """Like ``_prefetch_shards`` but uses ``load_filtered`` to skip reading
    tensor data for keys where *skip_key_fn* returns True.

    Yields (state_dict, skipped_keys) tuples.
    """

    if not state_dict_iterators:
        return

    n = len(state_dict_iterators)
    initial_submit = min(prefetch_count + 1, n)

    with ThreadPoolExecutor(max_workers=max(prefetch_count, 1)) as pool:
        pending: deque = deque()
        for i in range(initial_submit):
            pending.append(pool.submit(state_dict_iterators[i].load_filtered, skip_key_fn))

        next_idx = initial_submit

        for _ in range(n):
            state_dict, skipped_keys = pending.popleft().result()

            if next_idx < n:
                pending.append(pool.submit(state_dict_iterators[next_idx].load_filtered, skip_key_fn))
                next_idx += 1

            yield state_dict, skipped_keys


@dataclass
class BroadcastMetadata:
    done: bool
    name: Optional[str]
    shape: Optional["torch.Size"]
    dtype: Optional["torch.dtype"]


def _load_state_dict(weights_path: str, **kwargs) -> List["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """

    max_retries = 5
    for attempt in range(max_retries):
        result = _try_load_state_dict(weights_path, **kwargs)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            retry_delay = 2 * (2**attempt)  # 2, 4, 8, 16s
            logger.warning(
                f"Cannot find checkpoint files in {weights_path} (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def _try_load_state_dict(weights_path: str, **kwargs):
    """
    Single attempt to load state dict. Returns list of iterators or None if not found.

    Rank 0 resolves all file paths (cached_file + get_checkpoint_shard_files)
    and broadcasts the result to other ranks. This avoids N-way filesystem
    hammering and keeps the broadcast-loading path rank-0-driven.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size <= 1:
        # Single rank: do everything locally (no broadcast needed)
        return _try_load_state_dict_local(weights_path, **kwargs)

    if os.path.isdir(weights_path):
        # Shared local snapshots are cheap to resolve independently and avoid
        # another cross-rank metadata broadcast before tensor loading starts.
        return _try_load_state_dict_local(weights_path, **kwargs)

    # Multi-rank: rank 0 resolves paths, broadcasts to all
    resolved_paths = [None]
    if rank == 0:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                result = _try_load_state_dict_local(weights_path, **kwargs)
                if result is not None:
                    resolved_paths[0] = [it.filepath for it in result]
                break
            except OSError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"OSError resolving shard files (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 5s..."
                    )
                    time.sleep(5)
                else:
                    logger.error(f"Failed to resolve shard files after {max_retries} attempts")
                    raise

    _broadcast_object_list_weight_load(resolved_paths, src=0)
    shard_files = resolved_paths[0]
    if shard_files is None:
        return None
    return [StateDictIterator(f) for f in shard_files]


def _try_load_state_dict_local(weights_path: str, **kwargs):
    """Resolve shard file paths locally (no broadcast). Used by rank 0 or single-rank."""
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    return None


def _find_submodule(module: "nn.Module", name: str) -> Tuple["nn.Module", str]:
    """
    Finds the leaf module according to the name.
    """
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        module = getattr(module, piece)

    return module, pieces[-1]


def _build_compiled_key_map(
    parameter_names: set,
    buffer_dict: dict,
) -> Dict[str, str]:
    """Build mapping from checkpoint-style keys to model keys with ``_orig_mod`` prefix.

    When ``torch.compile`` wraps decoder layers, parameter names gain an ``_orig_mod.``
    segment (e.g. ``layers.0._orig_mod.mlp.gate.weight``).  Checkpoint keys lack this
    prefix.  This mapping bridges the gap so weight loading finds the correct parameters.

    Returns an empty dict if no ``_orig_mod`` segments are present (no-compile case).
    """
    mapping: Dict[str, str] = {}
    for name in parameter_names:
        stripped = name.replace("._orig_mod.", ".")
        if stripped != name:
            mapping[stripped] = name
    for name in buffer_dict:
        stripped = name.replace("._orig_mod.", ".")
        if stripped != name:
            mapping[stripped] = name
    return mapping


def _get_expert_scatter_target_shape(
    model: Union["nn.Module", "PreTrainedModel"],
    parameter_name: str,
    tensor: "torch.Tensor",
    parallel_plan: Optional["ParallelPlan"],
    parallel_state,
) -> Optional[Tuple[int, ...]]:
    """Return the EP-local parameter shape if this tensor can be rank0-scattered.

    We only use this optimized path when the current rank topology has a simple
    2D EP mesh (ep, ep_fsdp). More complex layouts, such as PP+EP, fall back to
    the existing full-tensor broadcast path.
    """
    if (
        parallel_plan is None
        or not parallel_state.ep_enabled
        or parallel_state.ep_fsdp_device_mesh is None
        or parallel_state.ep_fsdp_device_mesh.mesh.ndim != 2
        or not parallel_plan._is_expert_parameter(parameter_name)
    ):
        return None

    module, local_name = _find_submodule(model, parameter_name)
    param = module._parameters.get(local_name)
    if param is None or param.ndim == 0 or tensor.ndim == 0:
        return None

    target_shape = tuple(param.shape)
    if not target_shape or tensor.shape[0] <= target_shape[0] or tensor.shape[0] % target_shape[0] != 0:
        return None

    return target_shape


def _build_expert_scatter_list(
    tensor: "torch.Tensor",
    target_shape: Tuple[int, ...],
    parallel_state,
    torch_device: "torch.device",
) -> Tuple["torch.Tensor", List["torch.Tensor"]]:
    """Prepare rank0-side per-rank expert views for NCCL scatter."""
    if tensor.device.type == "cpu" and torch_device.type == "cuda":
        full_tensor = tensor.pin_memory().to(torch_device, non_blocking=True)
    else:
        full_tensor = tensor.to(torch_device, non_blocking=True)

    ep_mesh = parallel_state.ep_fsdp_device_mesh.mesh.cpu()
    local_experts = target_shape[0]
    scatter_list: List[torch.Tensor] = [None] * int(ep_mesh.numel())
    for ep_rank in range(ep_mesh.shape[0]):
        local_view = full_tensor.narrow(0, ep_rank * local_experts, local_experts)
        for ep_fsdp_rank in range(ep_mesh.shape[1]):
            global_rank = int(ep_mesh[ep_rank, ep_fsdp_rank])
            scatter_list[global_rank] = local_view

    return full_tensor, scatter_list


def _build_group_scatter_list(
    tensor: "torch.Tensor",
    target_shape: Tuple[int, ...],
    group_size: int,
    torch_device: "torch.device",
) -> Tuple["torch.Tensor", List["torch.Tensor"]]:
    """Prepare source-side per-rank views for subgroup scatter."""
    if tensor.device.type == "cpu" and torch_device.type == "cuda":
        full_tensor = tensor.pin_memory().to(torch_device, non_blocking=True)
    else:
        full_tensor = tensor.to(torch_device, non_blocking=True)

    local_rows = target_shape[0]
    scatter_list: List[torch.Tensor] = [None] * group_size
    for group_rank in range(group_size):
        scatter_list[group_rank] = full_tensor.narrow(0, group_rank * local_rows, local_rows)

    return full_tensor, scatter_list


class _MultiStreamDMA:
    """Manages multi-stream H2D DMA transfers for overlapping copy engine usage.

    GPU has multiple DMA copy engines that can run in parallel on different CUDA
    streams.  By round-robining pin+DMA across *num_streams* streams, we overlap
    H2D transfers and achieve ~2x throughput vs single-stream sequential dispatch.

    Deferred copies (GPU temp → model param) are flushed per shard to bound the
    extra VRAM to ~1 shard worth of temporary GPU tensors.
    """

    def __init__(self, num_streams: int = 2):
        self._streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self._counter = 0
        self._pending: list[tuple] = []  # (gpu_temp, module, local_name, stream, dtensor_factory, full_name)

    def dispatch(
        self,
        module: "nn.Module",
        local_name: str,
        tensor: "torch.Tensor",
        orig_tensor: "torch.Tensor",
        dtensor_factory: Optional[Callable] = None,
        full_name: str = "",
    ) -> None:
        """Pin a CPU tensor and DMA to GPU on a round-robin stream.

        Only handles CPU→CUDA transfers.  CUDA→CUDA transfers (e.g. expert
        stacked tensors from ExpertWeightBuffer) must NOT go through the DMA
        scheduler — they are dispatched synchronously in _dispatch_parameter
        to avoid interleaving no-op distribute_tensor calls (ep_fsdp mesh,
        size 1) with real NCCL collectives (fsdp mesh, size > 1).
        """
        assert tensor.device.type == "cpu", f"DMA scheduler only handles CPU→CUDA, got {tensor.device} for {full_name}"
        if tensor.dtype != orig_tensor.dtype:
            tensor = tensor.to(dtype=orig_tensor.dtype)
        pinned = tensor.pin_memory()
        stream = self._streams[self._counter % len(self._streams)]
        self._counter += 1
        with torch.cuda.stream(stream):
            gpu_temp = pinned.to(device=orig_tensor.device, non_blocking=True)
        self._pending.append((gpu_temp, module, local_name, stream, dtensor_factory, full_name))

    def flush(self) -> None:
        """Sync all streams and copy GPU temps into model parameters.

        Only CPU→CUDA (regular param) transfers are in the queue.  CUDA→CUDA
        expert stacked tensors are dispatched synchronously outside the scheduler.
        Sort by parameter name as a safety measure for deterministic NCCL
        ordering (all ranks read the same checkpoint in the same order, but
        sorting provides an extra guarantee).
        """
        self._pending.sort(key=lambda x: x[5])
        for gpu_temp, module, local_name, stream, dtensor_factory, _full_name in self._pending:
            stream.synchronize()
            orig_tensor = module._parameters[local_name].data
            if dtensor_factory is not None and hasattr(orig_tensor, "device_mesh"):
                device_mesh = getattr(orig_tensor, "device_mesh")
                placements = getattr(orig_tensor, "placements")
                module._parameters[local_name].data.copy_(dtensor_factory(gpu_temp, device_mesh, placements))
                # distribute_tensor uses NCCL on a different CUDA stream.
                # Synchronize to prevent races with subsequent DMA transfers
                # on the copy-engine streams.
                torch.cuda.synchronize()
            else:
                module._parameters[local_name].data.copy_(gpu_temp)
        self._pending.clear()


# Module-level scheduler, set during bulk loading to enable multi-stream DMA.
_active_dma_scheduler: Optional[_MultiStreamDMA] = None


def _copy_into_existing_dtensor_shard(dtensor: "torch.Tensor", tensor: "torch.Tensor") -> bool:
    """Copy a full tensor directly into a 1D DTensor's local shard.

    When the caller already holds the full parameter value on every rank
    (broadcast/grouped load modes), rebuilding a DTensor via ``distribute_tensor``
    adds extra padding logic and collectives that are not needed. This helper
    handles the common 1D-mesh cases directly:

    - replicated DTensors: copy the full tensor into ``_local_tensor``
    - singly sharded DTensors: split the full tensor locally and copy the
      current rank's shard into ``_local_tensor``
    """
    if not hasattr(dtensor, "_local_tensor"):
        return False

    device_mesh = getattr(dtensor, "device_mesh", None)
    placements = getattr(dtensor, "placements", None)
    if device_mesh is None or placements is None:
        return False
    if getattr(device_mesh, "ndim", None) != 1:
        return False

    shard_placements = [placement for placement in placements if isinstance(placement, DTShard)]
    if len(shard_placements) > 1:
        return False

    local_tensor = dtensor._local_tensor
    if len(shard_placements) == 0:
        if tuple(local_tensor.shape) != tuple(tensor.shape):
            return False
        local_tensor.copy_(tensor.to(device=local_tensor.device, dtype=local_tensor.dtype))
        return True

    shard = shard_placements[0]
    mesh_size = device_mesh.size()
    local_rank = device_mesh.get_local_rank()
    shards, _ = shard._split_tensor(tensor, mesh_size, with_padding=True, contiguous=True)
    local_shard = shards[local_rank]
    expected_shape = tuple(local_tensor.shape)
    if tuple(local_shard.shape) != expected_shape:
        if local_shard.ndim != local_tensor.ndim:
            return False
        can_trim = True
        for dim, (actual, expected) in enumerate(zip(local_shard.shape, expected_shape)):
            if dim == shard.dim:
                if expected > actual:
                    can_trim = False
                    break
            elif expected != actual:
                can_trim = False
                break
        if not can_trim:
            return False
        local_shard = local_shard.narrow(shard.dim, 0, expected_shape[shard.dim])

    local_shard = local_shard.to(device=local_tensor.device, dtype=local_tensor.dtype)
    local_tensor.copy_(local_shard)
    return True


def _dispatch_parameter(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    parallel_plan: Optional["ParallelPlan"] = None,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    global _active_dma_scheduler

    full_param_name = name
    module, local_name = _find_submodule(module, name)
    orig_tensor = module._parameters[local_name].data

    # Handle parameter slicing according to parallel_plan, now only EP-aware
    if parallel_plan is not None:
        tensor = parallel_plan.shard_tensor(tensor, full_param_name, orig_tensor.shape)

    # Multi-stream fast path: defer CPU→CUDA copies to overlap DMA across
    # GPU copy engines.  Uses ~1 shard of extra VRAM for temp GPU tensors,
    # freed per shard via flush().
    #
    # IMPORTANT: Only CPU→CUDA tensors go through the DMA scheduler.
    # CUDA→CUDA tensors (expert stacked tensors from ExpertWeightBuffer) are
    # dispatched synchronously below.  Mixing them in the same deferred queue
    # would interleave no-op distribute_tensor calls (expert params on
    # ep_fsdp mesh, size 1) with real NCCL collectives (regular params on
    # fsdp mesh, size > 1), which can corrupt weights.
    if _active_dma_scheduler is not None and tensor.device.type == "cpu" and orig_tensor.device.type == "cuda":
        _active_dma_scheduler.dispatch(module, local_name, tensor, orig_tensor, dtensor_factory, full_param_name)
        return

    # Synchronous fallback: used outside bulk loading loops and for CUDA→CUDA
    # transfers (expert stacked tensors).
    is_cuda_to_cuda = tensor.device.type != "cpu" and orig_tensor.device.type == "cuda"
    if tensor.device.type == "cpu" and orig_tensor.device.type == "cuda":
        if tensor.dtype != orig_tensor.dtype:
            tensor = tensor.to(dtype=orig_tensor.dtype)
        tensor = tensor.pin_memory().to(device=orig_tensor.device, non_blocking=True)
    else:
        tensor = tensor.to(orig_tensor)

    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        if _copy_into_existing_dtensor_shard(orig_tensor, tensor):
            return
        if is_cuda_to_cuda:
            # Diagnostic: log expert stacked tensor dispatch (CUDA→CUDA DTensor path)
            logger.debug(
                f"Sync dispatch (CUDA→CUDA DTensor): {full_param_name} "
                f"tensor={tensor.shape} orig={orig_tensor.shape} "
                f"mesh_size={device_mesh.size()} placements={placements}"
            )
        if orig_tensor.device.type == "cpu":
            # CPU DTensor: copy shard directly into local tensor.
            # distribute_tensor doesn't support CPU mesh, so we manually shard and copy.

            shard_dim = None
            for p in placements:
                if isinstance(p, DTShard):
                    shard_dim = p.dim
                    break
            if shard_dim is not None:
                # Shard the source tensor to match the local shard
                mesh_size = device_mesh.size()
                local_rank = device_mesh.get_local_rank()
                chunk_size = tensor.shape[shard_dim] // mesh_size
                shard = tensor.narrow(shard_dim, local_rank * chunk_size, chunk_size).contiguous()
                orig_tensor._local_tensor.copy_(shard.to(orig_tensor._local_tensor.dtype))
            else:
                # Replicated placement: copy full tensor
                orig_tensor._local_tensor.copy_(tensor.to(orig_tensor._local_tensor.dtype))
        else:
            module._parameters[local_name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        module._parameters[local_name].data.copy_(tensor)


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name]

    if hasattr(orig_tensor, "device_mesh"):  # dtensor buffer
        if dtensor_factory is None:
            raise ValueError("dtensor buffer requires a dtensor_factory.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._buffers[name] = dtensor_factory(buffer.to(dtype=orig_tensor.dtype), device_mesh, placements)
    else:
        try:
            module._buffers[name].copy_(buffer.to(device=orig_tensor.device, dtype=orig_tensor.dtype))
        except NotImplementedError:
            # Meta tensor or uninitialized buffer — skip, will be loaded by QLoRA later
            logger.warning(f"Skipping buffer dispatch (meta/uninitialized): {name}")


def _init_parameter(
    module: "nn.Module",
    name: str,
) -> None:
    """
    Initializes parameter in model.

    For LoRA parameters (lora_A, lora_B), uses PEFT's default initialization:
    - lora_A: kaiming uniform initialization
    - lora_B: zeros (so LoRA has no effect at start)
    """

    # Check if this is a LoRA parameter and handle specially
    if "lora_A" in name or "lora_B" in name:
        # Navigate to the parameter
        pieces = name.split(".")
        param_module = module
        for piece in pieces[:-1]:
            if not hasattr(param_module, piece):
                raise ValueError(f"Cannot find {piece} in {param_module}.")
            param_module = getattr(param_module, piece)

        param_name = pieces[-1]
        if hasattr(param_module, param_name):
            param = getattr(param_module, param_name)
            if isinstance(param, (torch.Tensor, nn.Parameter)):
                if "lora_A" in name:
                    # Kaiming uniform initialization for lora_A
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    logger.info_rank0(f"Initialized LoRA param {name} with kaiming_uniform_")
                elif "lora_B" in name:
                    # Zeros for lora_B (no effect at start)
                    nn.init.zeros_(param)
                    logger.info_rank0(f"Initialized LoRA param {name} with zeros")
                return

    # Standard initialization for non-LoRA params
    pieces = name.split(".")
    init_func = None
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        if hasattr(module, "_init_weights"):
            init_func = getattr(module, "_init_weights")

        module = getattr(module, piece)

    if init_func is None:
        raise ValueError(f"Cannot retrieve `_init_weights` function in the parents of {module}.")

    module.apply(init_func)


def _convert_weight_key(key: str, model: "PreTrainedModel") -> str:
    """
    Convert a single state dict key using the model's checkpoint conversion mapping.

    For example, in the InternVL, we have _checkpoint_conversion_mapping = {"^model": "language_model"}

    This is to adapt to the big breaking change introduced in HF transformers 4.52:
    https://github.com/huggingface/transformers/pull/38385
    """
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return key

    for pattern, replacement in model._checkpoint_conversion_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        converted_key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return converted_key

    return key


def _shrink_expert_params_for_ep(model: "nn.Module") -> None:
    """Shrink expert parameters to EP-local shapes before materialization.

    When both TP and EP are enabled, weights are loaded before EP slicing
    (TP requires loading before FSDP).  Calling ``model.to_empty(device=cuda)``
    would allocate *all* expert parameters at full size, causing OOM.

    This function replaces full-size meta expert parameters with EP-local-sized
    meta parameters.  Since meta tensors use no device memory, this is free.
    The subsequent ``to_empty`` then allocates only the local expert slice.
    """
    _ps = get_parallel_state()
    if not _ps.ep_enabled:
        return

    parallel_plan = model.get_parallel_plan() if hasattr(model, "get_parallel_plan") else None
    if parallel_plan is None:
        return

    ep_size = _ps.ep_size
    shrunk = 0
    for name, param in list(model.named_parameters()):
        # Only shrink non-DTensor meta expert params that haven't been EP-sharded yet.
        # Params with spec_info were already sharded by parallel_plan.apply().
        if (
            parallel_plan._is_expert_parameter(name)
            and not hasattr(param, "device_mesh")
            and not hasattr(param, "spec_info")
            and param.device.type == "meta"
            and param.shape[0] % ep_size == 0
            and param.shape[0] // ep_size < param.shape[0]
        ):
            local_experts = param.shape[0] // ep_size
            local_shape = (local_experts,) + param.shape[1:]
            new_param = nn.Parameter(
                torch.empty(local_shape, dtype=param.dtype, device="meta"),
                requires_grad=param.requires_grad,
            )
            sub_mod, local_name = _find_submodule(model, name)
            sub_mod._parameters[local_name] = new_param
            shrunk += 1
    if shrunk > 0:
        logger.info_rank0(
            f"EP pre-shrink: resized {shrunk} expert params to EP-local shapes "
            f"(ep_size={ep_size}) before materialization"
        )


@torch.no_grad()
def all_ranks_load_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Loads pre-trained model states in transformers' format.

    If the model provides a CheckpointHandler (via get_checkpoint_handler()),
    weight transforms (e.g., expert merging, gate/up merging) are delegated to it.
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}

    # torch.compile wraps modules with '_orig_mod.' prefix in parameter names.
    # Checkpoint keys don't have this prefix. Build a mapping to bridge the gap.
    _compiled_key_map = _build_compiled_key_map(parameter_names_to_load, buffer_dict)

    _shrink_expert_params_for_ep(model)
    model.to_empty(device=init_device)

    # Get parallel plan if available
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    # Get parallel state for PP-aware logging and EP-aware loading
    _ps = get_parallel_state()

    # Get checkpoint handler from model (delegates weight transforms to model-specific logic)
    # Pass the model's device so MoE expert stacking can happen on GPU (13x faster).
    model_device = None
    if hasattr(model, "parameters"):
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = None
    handler = None
    if hasattr(model, "get_checkpoint_handler"):
        ep_rank = _ps.ep_rank if _ps.ep_enabled else 0
        ep_size = _ps.ep_size if _ps.ep_enabled else 1
        checkpoint_keys = _get_checkpoint_keys(weights_path)
        model_dtype = None
        if hasattr(model, "parameters"):
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_dtype = None
        handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=ep_rank,
            ep_size=ep_size,
            is_broadcast=False,
            weights_path=weights_path,
            device=model_device,
            dtype=model_dtype,
        )

    # Retry loading state dict on OSError (e.g., HuggingFace download issues)
    max_retries = 10
    retry_delay = 5  # seconds
    last_error = None

    for attempt in range(max_retries):
        try:
            state_dict_iterators = _load_state_dict(weights_path)
            break
        except OSError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"OSError loading weights (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to load weights after {max_retries} attempts")
                raise
    else:
        if last_error:
            raise last_error

    # When PP is enabled, gather expected keys from all ranks so we can distinguish
    # "key belongs to another PP stage" (expected) from "truly unexpected key" (warning).
    if _ps.pp_enabled:
        local_expected = parameter_names_to_load | set(buffer_dict.keys())
        all_expected = [None] * dist.get_world_size()
        dist.all_gather_object(all_expected, local_expected)
        global_expected_keys = set().union(*all_expected)
    else:
        global_expected_keys = None

    def _should_skip_qlora_expert_key(key: str, prefixes: set) -> bool:
        """Check if a checkpoint key matches a QLoRA MoE skip prefix.

        Skip prefixes are like "model.layers.0.mlp.experts.gate_proj".
        Checkpoint keys are like "model.layers.0.mlp.experts.0.gate_proj.weight".
        Match by checking if removing the expert index gives a prefix match.
        """
        for prefix in prefixes:
            # Split prefix: "model.layers.0.mlp.experts" + "gate_proj"
            parts = prefix.rsplit(".", 1)
            if len(parts) != 2:
                continue
            base, proj = parts  # "model.layers.0.mlp.experts", "gate_proj"
            # Key should match: base.{digit}.proj.{suffix}
            if key.startswith(base + ".") and f".{proj}." in key:
                return True
        return False

    # Check if the handler provides an EP-aware skip function to avoid reading
    # out-of-range expert tensor data from disk (can reduce I/O by ~60-70%).
    skip_key_fn = handler.get_skip_key_fn() if handler is not None else None

    # Collect keys that are expected to be absent from the model (e.g., QLoRA replaces
    # expert weight params with quantized buffers — base weights loaded separately).
    _expected_skip_keys = set()
    # Prefix-based skip: for MoE expert modules, checkpoint has per-expert keys
    # like "experts.0.gate_proj.weight" that should match skip prefix "experts.gate_proj"
    _expected_skip_prefixes = set()
    for fqn, mod in model.named_modules():
        if getattr(mod, "_qlora_expected_skip_keys", None):
            for suffix in mod._qlora_expected_skip_keys:
                key = f"{fqn}.{suffix}" if fqn else suffix
                _expected_skip_keys.add(key)
                # Also add as prefix for per-expert matching:
                # "model.layers.0.mlp.experts.gate_proj" matches
                # "model.layers.0.mlp.experts.0.gate_proj.weight"
                _expected_skip_prefixes.add(key)

    # Remove expected skip keys from parameter_names_to_load: FSDP2 may materialize
    # None parameters on some layers (e.g., QLoRALinear.weight), which would cause
    # them to appear in named_parameters(). These weights are loaded separately by
    # QLoRA's load_prequantized_weights(), not through the standard dispatch path.
    parameter_names_to_load -= _expected_skip_keys

    def _dispatch_results(results):
        for param_name, param_tensor in results:
            # Resolve _orig_mod prefix from torch.compile (checkpoint keys lack it)
            model_name = _compiled_key_map.get(param_name, param_name)
            if param_name in _expected_skip_keys or model_name in _expected_skip_keys:
                pass  # silently skip — weights loaded separately (e.g., QLoRA)
            elif _expected_skip_prefixes and _should_skip_qlora_expert_key(model_name, _expected_skip_prefixes):
                pass  # per-expert key matches a QLoRA skip prefix
            elif model_name in buffer_dict:
                buffer_dict[model_name] = param_tensor.clone()
            elif model_name in parameter_names_to_load:
                parameter_names_to_load.remove(model_name)
                _dispatch_parameter(model, model_name, param_tensor, dtensor_factory, parallel_plan)
            else:
                if global_expected_keys is None or param_name not in global_expected_keys:
                    logger.warning_rank0(f"Unexpected key in state dict: {param_name}.")

    # Enable multi-stream DMA: overlaps H2D transfers across GPU copy engines
    # for ~2x throughput.  Uses ~1 shard of extra VRAM for temp GPU tensors,
    # freed per shard via flush().
    global _active_dma_scheduler
    dma_scheduler = _MultiStreamDMA(num_streams=2)
    _active_dma_scheduler = dma_scheduler

    try:
        if skip_key_fn is not None:
            # EP-aware filtered loading: skip reading tensor data for out-of-range experts
            logger.info_rank0(
                f"EP-aware filtered loading enabled (ep_rank={ep_rank}, ep_size={ep_size}): "
                "skipping disk reads for out-of-range expert weights"
            )
            for state_dict, skipped_keys in tqdm(
                _prefetch_shards_filtered(state_dict_iterators, skip_key_fn, prefetch_count=len(state_dict_iterators)),
                total=len(state_dict_iterators),
                desc="Loading checkpoint shards",
                disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
            ):
                # Notify handler about skipped keys (updates completion counters)
                for skipped_key in skipped_keys:
                    _dispatch_results(handler.on_skip_weight(skipped_key))

                # Process loaded tensors normally
                for name, tensor in state_dict.items():
                    name = _convert_weight_key(name, model)
                    results = handler.on_load_weight(name, tensor)
                    _dispatch_results(results)

                # Flush deferred DMA copies before freeing shard memory
                dma_scheduler.flush()
                del state_dict
                empty_cache()
        else:
            # Standard bulk loading (no EP filtering)
            # prefetch_count = num_shards loads all shards concurrently, overlapping
            # disk I/O with pin_memory + DMA dispatch for maximum throughput.
            for state_dict in tqdm(
                _prefetch_shards(state_dict_iterators, prefetch_count=len(state_dict_iterators)),
                total=len(state_dict_iterators),
                desc="Loading checkpoint shards",
                disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
            ):
                for name, tensor in state_dict.items():
                    name = _convert_weight_key(name, model)

                    if handler is not None:
                        results = handler.on_load_weight(name, tensor)
                    else:
                        results = [(name, tensor)]

                    _dispatch_results(results)

                # Flush deferred DMA copies before freeing shard memory
                dma_scheduler.flush()
                del state_dict
                empty_cache()

        # Flush handler buffers (warn on incomplete merges)
        if handler is not None:
            for param_name, param_tensor in handler.on_load_complete():
                if param_name in parameter_names_to_load:
                    parameter_names_to_load.remove(param_name)
                    _dispatch_parameter(model, param_name, param_tensor, dtensor_factory, parallel_plan)
            # Final flush for any handler-emitted tensors
            dma_scheduler.flush()
    finally:
        _active_dma_scheduler = None

    post_process_after_weight_loading(
        model,
        buffer_dict,
        parameter_names_to_load,
        dtensor_factory,
        qlora_skip_prefixes=_expected_skip_prefixes,
        qlora_skip_fn=_should_skip_qlora_expert_key,
    )


@torch.no_grad()
def rank0_load_and_broadcast_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
):
    """
    Same purpose as ``all_ranks_load_weights`` but reduces disk I/O by broadcasting
    weights from rank 0 instead of having every GPU read the full checkpoint.

    If the model provides a CheckpointHandler (via get_checkpoint_handler()),
    weight transforms are applied on rank 0 before broadcasting.
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning_once("Distributed environment not initialized, falling back to all_ranks_load_weights.")
        return all_ranks_load_weights(model, weights_path, init_device, dtensor_factory)

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}

    # torch.compile wraps modules with '_orig_mod.' prefix in parameter names.
    # Checkpoint keys don't have this prefix. Build a mapping to bridge the gap.
    _compiled_key_map = _build_compiled_key_map(parameter_names_to_load, buffer_dict)

    _shrink_expert_params_for_ep(model)
    model.to_empty(device=init_device)

    # Get parallel plan if available
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    _ps = get_parallel_state()
    global_rank = _ps.global_rank
    torch_device = torch.device(init_device)
    # Get checkpoint handler from model.
    # For the broadcast path, is_broadcast=True tells the handler that rank 0 buffers
    # ALL experts (ep_size=1). EP slicing is handled later by ParallelPlan.shard_tensor().
    handler = None
    if hasattr(model, "get_checkpoint_handler"):
        checkpoint_keys = _get_checkpoint_keys(weights_path)
        model_dtype = None
        if hasattr(model, "parameters"):
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_dtype = None
        handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=0,
            ep_size=1,
            is_broadcast=True,
            weights_path=weights_path,
            device=torch_device,
            dtype=model_dtype,
        )
    skip_key_fn = handler.get_skip_key_fn() if handler is not None else None

    # When PP is enabled, gather expected keys from all ranks so we can distinguish
    # "key belongs to another PP stage" (expected) from "truly unexpected key" (warning).
    if _ps.pp_enabled:
        local_expected = parameter_names_to_load | set(buffer_dict.keys())
        all_expected = [None] * dist.get_world_size()
        dist.all_gather_object(all_expected, local_expected)
        global_expected_keys = set().union(*all_expected)
    else:
        global_expected_keys = None

    # All ranks must enter _load_state_dict(): in multi-rank mode it contains
    # the collective that receives the rank-0-resolved shard paths.
    state_dict_iterators = _load_state_dict(weights_path)
    shard_count = len(state_dict_iterators)
    logger.info_rank0(f"rank0_load_and_broadcast_weights: {shard_count=} ")
    shard_count_tensor = torch.tensor(
        [shard_count],
        dtype=torch.int64,
        device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
    )
    dist.broadcast(shard_count_tensor, src=0, group=_get_weight_load_group())
    shard_count = int(shard_count_tensor.item())

    # Rank 0: create prefetching generator that bulk-loads upcoming shards in
    # background threads while the current shard is being broadcast.
    # prefetch_count=2 allows 2 concurrent NFS reads for higher aggregate throughput.
    if global_rank == 0:
        if skip_key_fn is not None:
            logger.info_rank0("Filtered broadcast loading enabled on rank 0 for handler-skipped checkpoint keys")
            prefetched = _prefetch_shards_filtered(state_dict_iterators, skip_key_fn, prefetch_count=2)
        else:
            prefetched = _prefetch_shards(state_dict_iterators, prefetch_count=2)

    if global_rank == 0:
        shard_range = tqdm(
            range(shard_count),
            desc="Loading checkpoint shards",
            disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
        )
    else:
        shard_range = range(shard_count)

    # Queue for transformed tensors ready to broadcast (used when handler buffers/merges)
    merged_queue: List[Tuple[str, torch.Tensor]] = []

    for shard_idx in shard_range:
        # Phase 1 (rank 0 only): bulk-load shard from prefetch cache and feed
        # all tensors through the checkpoint handler. While this runs, the
        # background thread is already loading the *next* shard from disk.
        if global_rank == 0:
            skipped_keys = []
            if skip_key_fn is not None:
                state_dict, skipped_keys = next(prefetched)
                for skipped_key in skipped_keys:
                    merged_queue.extend(handler.on_skip_weight(skipped_key))
            else:
                state_dict = next(prefetched)
            for key, tensor in state_dict.items():
                key = _convert_weight_key(key, model)
                if handler is not None:
                    results = handler.on_load_weight(key, tensor)
                else:
                    results = [(key, tensor)]
                merged_queue.extend(results)
            del state_dict

        # Phase 2 (all ranks): broadcast all queued items for this shard.
        # Batch all metadata into a single broadcast_object_list call, then
        # broadcast tensors one by one.  This replaces N metadata broadcasts
        # with 1, eliminating per-tensor pickle + NCCL launch overhead.
        if global_rank == 0:
            batch_meta = []
            for name, tensor in merged_queue:
                model_name = _compiled_key_map.get(name, name)
                target_shape = None
                if model_name in parameter_names_to_load:
                    target_shape = _get_expert_scatter_target_shape(model, model_name, tensor, parallel_plan, _ps)
                if target_shape is not None:
                    batch_meta.append((name, torch.Size(target_shape), tensor.dtype, "expert_scatter"))
                else:
                    batch_meta.append((name, tensor.shape, tensor.dtype, "broadcast"))
        else:
            batch_meta = None

        batch_meta = [batch_meta]
        _broadcast_object_list_weight_load(batch_meta, src=0)
        batch_meta = batch_meta[0]

        for name, shape, dtype, transfer_mode in batch_meta:
            if global_rank == 0:
                _, source_tensor = merged_queue.pop(0)
            else:
                source_tensor = None

            start_time = time.perf_counter()
            if transfer_mode == "expert_scatter":
                if global_rank == 0:
                    full_tensor, scatter_list = _build_expert_scatter_list(
                        source_tensor, tuple(shape), _ps, torch_device
                    )
                    tensor = scatter_list[global_rank].clone()
                    full_shape = tuple(source_tensor.shape)
                else:
                    full_tensor = None
                    scatter_list = None
                    tensor = torch.empty(shape, dtype=dtype, device=torch_device)
                    full_shape = shape

                dist.scatter(tensor, scatter_list=scatter_list, src=0, group=_get_weight_load_group())
                logger.info_rank0(
                    f"{name=}, full_shape={full_shape}, local_shape={shape}, {dtype=}, "
                    f"scatter time (ms): {1000 * (time.perf_counter() - start_time)}"
                )
            else:
                if global_rank != 0:
                    tensor = torch.empty(shape, dtype=dtype, device=torch_device)
                else:
                    if source_tensor.device.type == "cpu" and torch_device.type == "cuda":
                        tensor = source_tensor.pin_memory().to(torch_device, non_blocking=True)
                    else:
                        tensor = source_tensor.to(torch_device, non_blocking=True)

                dist.broadcast(tensor, src=0, group=_get_weight_load_group())
                logger.info_rank0(
                    f"{name=}, {shape=}, {dtype=}, broadcast time (ms): {1000 * (time.perf_counter() - start_time)}"
                )

            # Resolve _orig_mod prefix from torch.compile
            model_name = _compiled_key_map.get(name, name)
            if model_name in buffer_dict:
                buffer_dict[model_name] = tensor.detach().clone()
            elif model_name in parameter_names_to_load:
                parameter_names_to_load.discard(model_name)
                _dispatch_parameter(model, model_name, tensor, dtensor_factory, parallel_plan)
            else:
                if global_expected_keys is None or name not in global_expected_keys:
                    logger.warning_rank0(f"Unexpected key in state dict: {name}.")

            del tensor
            if global_rank == 0:
                del source_tensor
                if transfer_mode == "expert_scatter":
                    del full_tensor
                    del scatter_list

        empty_cache()

    # Flush handler buffers after all shards (broadcast any remaining merged items)
    if handler is not None and global_rank == 0:
        merged_queue.extend(handler.on_load_complete())

    # Broadcast remaining items from handler flush (same batched pattern)
    if global_rank == 0:
        flush_meta = []
        for name, tensor in merged_queue:
            model_name = _compiled_key_map.get(name, name)
            target_shape = None
            if model_name in parameter_names_to_load:
                target_shape = _get_expert_scatter_target_shape(model, model_name, tensor, parallel_plan, _ps)
            if target_shape is not None:
                flush_meta.append((name, torch.Size(target_shape), tensor.dtype, "expert_scatter"))
            else:
                flush_meta.append((name, tensor.shape, tensor.dtype, "broadcast"))
    else:
        flush_meta = None

    flush_meta = [flush_meta]
    _broadcast_object_list_weight_load(flush_meta, src=0)
    flush_meta = flush_meta[0]

    for name, shape, dtype, transfer_mode in flush_meta:
        if global_rank == 0:
            _, source_tensor = merged_queue.pop(0)
        else:
            source_tensor = None

        if transfer_mode == "expert_scatter":
            if global_rank == 0:
                full_tensor, scatter_list = _build_expert_scatter_list(source_tensor, tuple(shape), _ps, torch_device)
                tensor = scatter_list[global_rank].clone()
            else:
                full_tensor = None
                scatter_list = None
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
            dist.scatter(tensor, scatter_list=scatter_list, src=0, group=_get_weight_load_group())
        else:
            if global_rank != 0:
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
            else:
                if source_tensor.device.type == "cpu" and torch_device.type == "cuda":
                    tensor = source_tensor.pin_memory().to(torch_device, non_blocking=True)
                else:
                    tensor = source_tensor.to(torch_device, non_blocking=True)
            dist.broadcast(tensor, src=0, group=_get_weight_load_group())

        # Resolve _orig_mod prefix from torch.compile
        model_name = _compiled_key_map.get(name, name)
        if model_name in parameter_names_to_load:
            parameter_names_to_load.discard(model_name)
            _dispatch_parameter(model, model_name, tensor, dtensor_factory, parallel_plan)

        del tensor
        if global_rank == 0:
            del source_tensor
            if transfer_mode == "expert_scatter":
                del full_tensor
                del scatter_list

    post_process_after_weight_loading(model, buffer_dict, parameter_names_to_load, dtensor_factory)


@torch.no_grad()
def grouped_load_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
):
    """Load weights with a hybrid dense/expert grouped strategy.

    Dense/shared tensors are loaded once per node and broadcast inside a
    node-local load group. Expert tensors are loaded once per ``ep_fsdp`` group
    leader and fanned out only to the ranks that share that expert shard.
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning_once("Distributed environment not initialized, falling back to all_ranks_load_weights.")
        return all_ranks_load_weights(model, weights_path, init_device, dtensor_factory)

    _ps = get_parallel_state()
    fanout_group = _get_grouped_weight_load_group(_ps)
    if fanout_group is None:
        logger.info_rank0("Grouped weight loading requires EP/FSDP groups; using rank-0 load fallback.")
        return rank0_load_and_broadcast_weights(model, weights_path, init_device, dtensor_factory)

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}

    _compiled_key_map = _build_compiled_key_map(parameter_names_to_load, buffer_dict)

    _shrink_expert_params_for_ep(model)
    model.to_empty(device=init_device)

    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    fanout_ranks = dist.get_process_group_ranks(fanout_group)
    fanout_src = fanout_ranks[0]
    dense_group = _get_grouped_dense_weight_load_group()
    if dense_group is None:
        dense_ranks = [_ps.global_rank]
    else:
        dense_ranks = dist.get_process_group_ranks(dense_group)
    dense_src = dense_ranks[0]
    global_rank = _ps.global_rank
    is_group_leader = global_rank == fanout_src
    is_dense_leader = global_rank == dense_src
    torch_device = torch.device(init_device)
    model_device = None
    model_dtype = None
    if hasattr(model, "parameters"):
        try:
            first_param = next(model.parameters())
            model_device = first_param.device
            model_dtype = first_param.dtype
        except StopIteration:
            model_device = None
            model_dtype = None
    checkpoint_keys = _get_checkpoint_keys(weights_path) if hasattr(model, "get_checkpoint_handler") else None
    dense_handler = None
    expert_handler = None
    if hasattr(model, "get_checkpoint_handler"):
        dense_handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=0,
            ep_size=1,
            is_broadcast=False,
            weights_path=weights_path,
            device=model_device,
            dtype=model_dtype,
        )
    if hasattr(model, "get_checkpoint_handler") and is_group_leader:
        ep_rank = _ps.ep_rank if _ps.ep_enabled else 0
        ep_size = _ps.ep_size if _ps.ep_enabled else 1
        expert_handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=ep_rank,
            ep_size=ep_size,
            is_broadcast=False,
            weights_path=weights_path,
            device=model_device,
            dtype=model_dtype,
        )
    dense_skip_key_fn = dense_handler.get_skip_key_fn() if dense_handler is not None else None
    expert_skip_key_fn = expert_handler.get_skip_key_fn() if expert_handler is not None else None

    if _ps.pp_enabled:
        local_expected = parameter_names_to_load | set(buffer_dict.keys())
        all_expected = [None] * dist.get_world_size()
        dist.all_gather_object(all_expected, local_expected)
        global_expected_keys = set().union(*all_expected)
    else:
        global_expected_keys = None

    dense_state_dict_iterators = _load_state_dict(weights_path)
    expert_state_dict_iterators = _load_state_dict(weights_path)
    shard_count = len(dense_state_dict_iterators)
    leader_count = dist.get_world_size() // len(fanout_ranks)
    dense_leader_count = dist.get_world_size() // len(dense_ranks)
    logger.info_rank0(
        "grouped_load_weights: "
        f"{shard_count=} "
        f"dense_group_size={len(dense_ranks)} dense_leader_count={dense_leader_count} "
        f"fanout_group_size={len(fanout_ranks)} leader_count={leader_count}"
    )
    prefetch_count = _get_grouped_weight_load_prefetch_count(shard_count)

    _expected_skip_keys = set()
    _expected_skip_prefixes = set()
    for fqn, mod in model.named_modules():
        if getattr(mod, "_qlora_expected_skip_keys", None):
            for suffix in mod._qlora_expected_skip_keys:
                key = f"{fqn}.{suffix}" if fqn else suffix
                _expected_skip_keys.add(key)
                _expected_skip_prefixes.add(key)

    parameter_names_to_load -= _expected_skip_keys

    def _should_skip_qlora_expert_key(key: str, prefixes: set) -> bool:
        for prefix in prefixes:
            parts = prefix.rsplit(".", 1)
            if len(parts) != 2:
                continue
            base, proj = parts
            if key.startswith(base + ".") and f".{proj}." in key:
                return True
        return False

    def _dispatch_loaded_tensor(param_name: str, param_tensor: torch.Tensor, *, expect_expert: bool) -> None:
        model_name = _compiled_key_map.get(param_name, param_name)
        is_expert = _is_expert_parameter_name(model_name, parallel_plan)
        if expect_expert != is_expert:
            raise RuntimeError(
                f"Grouped weight loading misrouted {'expert' if expect_expert else 'dense'} tensor "
                f"{param_name} -> {model_name}"
            )

        if param_name in _expected_skip_keys or model_name in _expected_skip_keys:
            return
        if _expected_skip_prefixes and _should_skip_qlora_expert_key(model_name, _expected_skip_prefixes):
            return
        if model_name in buffer_dict:
            buffer_dict[model_name] = param_tensor.clone()
            return
        if model_name in parameter_names_to_load:
            parameter_names_to_load.discard(model_name)
            _dispatch_parameter(model, model_name, param_tensor, dtensor_factory, parallel_plan)
            return
        if global_expected_keys is None or param_name not in global_expected_keys:
            logger.warning_rank0(f"Unexpected key in state dict: {param_name}.")

    def _broadcast_queue_and_dispatch(
        queue: List[Tuple[str, torch.Tensor]],
        *,
        group,
        object_group=None,
        src: int,
        is_source: bool,
        expect_expert: bool,
        scatter_group_size: int = 1,
    ) -> None:
        # group=None here means "no replication group" — every rank loads
        # locally and is its own source. Routing through the size-1 fast path
        # below avoids issuing collectives on the default world group with
        # disagreeing src.
        if group is None:
            group_size = 1
        else:
            try:
                group_size = dist.get_world_size(group)
            except TypeError:
                if hasattr(dist, "get_process_group_ranks"):
                    group_size = len(dist.get_process_group_ranks(group))
                else:
                    group_size = dist.get_world_size()
        if group_size == 1:
            local_batch = []
            if is_source:
                for name, tensor in queue:
                    transfer_mode = "broadcast"
                    target_shape = tensor.shape
                    if expect_expert and scatter_group_size > 1:
                        model_name = _compiled_key_map.get(name, name)
                        if model_name in parameter_names_to_load:
                            maybe_target_shape = _get_expert_scatter_target_shape(
                                model, model_name, tensor, parallel_plan, _ps
                            )
                            if maybe_target_shape is not None:
                                target_shape = torch.Size(maybe_target_shape)
                                transfer_mode = "expert_scatter"
                    local_batch.append((name, torch.Size(target_shape), tensor.dtype, transfer_mode))

            for name, shape, dtype, transfer_mode in local_batch:
                _, source_tensor = queue.pop(0)
                if source_tensor.device.type == "cpu" and torch_device.type == "cuda":
                    tensor = source_tensor.pin_memory().to(torch_device, non_blocking=True)
                else:
                    tensor = source_tensor.to(torch_device, non_blocking=True)
                _dispatch_loaded_tensor(name, tensor, expect_expert=expect_expert)
                del tensor
                del source_tensor
            return

        if is_source:
            batch_meta = []
            for name, tensor in queue:
                transfer_mode = "broadcast"
                target_shape = tensor.shape
                if expect_expert and scatter_group_size > 1:
                    model_name = _compiled_key_map.get(name, name)
                    if model_name in parameter_names_to_load:
                        maybe_target_shape = _get_expert_scatter_target_shape(
                            model, model_name, tensor, parallel_plan, _ps
                        )
                        if maybe_target_shape is not None:
                            target_shape = torch.Size(maybe_target_shape)
                            transfer_mode = "expert_scatter"
                batch_meta.append((name, torch.Size(target_shape), tensor.dtype, transfer_mode))
        else:
            batch_meta = None

        batch_meta = [batch_meta]
        _broadcast_object_list(batch_meta, src=src, group=object_group if object_group is not None else group)
        batch_meta = batch_meta[0]

        for name, shape, dtype, transfer_mode in batch_meta:
            if is_source:
                _, source_tensor = queue.pop(0)
                if transfer_mode == "expert_scatter":
                    full_tensor, scatter_list = _build_group_scatter_list(
                        source_tensor,
                        tuple(shape),
                        scatter_group_size,
                        torch_device,
                    )
                    tensor = scatter_list[0].clone()
                else:
                    if source_tensor.device.type == "cpu" and torch_device.type == "cuda":
                        tensor = source_tensor.pin_memory().to(torch_device, non_blocking=True)
                    else:
                        tensor = source_tensor.to(torch_device, non_blocking=True)
                    full_tensor = None
                    scatter_list = None
            else:
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
                source_tensor = None
                full_tensor = None
                scatter_list = None

            if transfer_mode == "expert_scatter":
                dist.scatter(tensor, scatter_list=scatter_list, src=src, group=group)
            else:
                dist.broadcast(tensor, src=src, group=group)
            _dispatch_loaded_tensor(name, tensor, expect_expert=expect_expert)

            del tensor
            if is_source:
                del source_tensor
                if full_tensor is not None:
                    del full_tensor
                if scatter_list is not None:
                    del scatter_list

    def _normalize_grouped_checkpoint_key(key: str) -> Optional[str]:
        if _matches_checkpoint_skip_key_pattern(key, model):
            return None
        converted_key = _convert_weight_key(key, model)
        if converted_key != key and _matches_checkpoint_skip_key_pattern(converted_key, model):
            return None
        return _normalize_checkpoint_key_for_filter(converted_key)

    def _should_skip_dense_key(key: str) -> bool:
        normalized = _normalize_grouped_checkpoint_key(key)
        if normalized is None or _is_checkpoint_expert_key(normalized):
            return True
        if dense_skip_key_fn is not None:
            return dense_skip_key_fn(normalized)
        return False

    def _should_skip_grouped_expert_key(key: str) -> bool:
        normalized = _normalize_grouped_checkpoint_key(key)
        if normalized is None or not _is_checkpoint_expert_key(normalized):
            return True
        if expert_skip_key_fn is not None:
            return expert_skip_key_fn(normalized)
        return False

    logger.info_rank0(
        f"Grouped loading enabled: dense/shared tensors use dense_group_size={len(dense_ranks)} dense_src={dense_src}"
    )
    dense_prefetched = None
    if is_dense_leader:
        dense_prefetched = _prefetch_shards_filtered(
            dense_state_dict_iterators,
            _should_skip_dense_key,
            prefetch_count=1,
        )

    expert_prefetched = None
    if is_group_leader:
        logger.info_rank0(
            f"Grouped loading enabled: expert tensors use fanout_group_size={len(fanout_ranks)} "
            f"ep_rank={_ps.ep_rank} ep_size={_ps.ep_size} prefetch_count={prefetch_count}"
        )
        expert_prefetched = _prefetch_shards_filtered(
            expert_state_dict_iterators,
            _should_skip_grouped_expert_key,
            prefetch_count=prefetch_count,
        )

    shard_range = (
        tqdm(
            range(shard_count),
            desc="Loading checkpoint shards",
            disable=global_rank != 0 or int(os.getenv("LOCAL_RANK", "-1")) > 0,
        )
        if global_rank == 0
        else range(shard_count)
    )

    dense_queue: List[Tuple[str, torch.Tensor]] = []
    expert_queue: List[Tuple[str, torch.Tensor]] = []

    for _ in shard_range:
        if is_dense_leader:
            state_dict, _skipped = next(dense_prefetched)
            for key, tensor in state_dict.items():
                key = _convert_weight_key(key, model)
                results = dense_handler.on_load_weight(key, tensor) if dense_handler is not None else [(key, tensor)]
                for result_name, result_tensor in results:
                    dense_queue.append((result_name, result_tensor))
            del state_dict
        _broadcast_queue_and_dispatch(
            dense_queue,
            group=dense_group,
            src=dense_src,
            is_source=is_dense_leader,
            expect_expert=False,
        )

        if is_group_leader:
            state_dict, skipped_keys = next(expert_prefetched)
            for skipped_key in skipped_keys:
                normalized = _normalize_grouped_checkpoint_key(skipped_key)
                if normalized is None or not _is_checkpoint_expert_key(normalized):
                    continue
                if expert_skip_key_fn is not None and expert_skip_key_fn(normalized):
                    expert_queue.extend(expert_handler.on_skip_weight(normalized))
            for key, tensor in state_dict.items():
                key = _convert_weight_key(key, model)
                results = expert_handler.on_load_weight(key, tensor) if expert_handler is not None else [(key, tensor)]
                for result_name, result_tensor in results:
                    expert_queue.append((result_name, result_tensor))
            del state_dict

        _broadcast_queue_and_dispatch(
            expert_queue,
            group=fanout_group,
            src=fanout_src,
            is_source=is_group_leader,
            expect_expert=True,
            scatter_group_size=len(fanout_ranks),
        )

        empty_cache()

    if dense_handler is not None:
        if is_dense_leader:
            dense_queue.extend(dense_handler.on_load_complete())
        _broadcast_queue_and_dispatch(
            dense_queue,
            group=dense_group,
            src=dense_src,
            is_source=is_dense_leader,
            expect_expert=False,
        )

    if expert_handler is not None and is_group_leader:
        expert_queue.extend(expert_handler.on_load_complete())
    _broadcast_queue_and_dispatch(
        expert_queue,
        group=fanout_group,
        src=fanout_src,
        is_source=is_group_leader,
        expect_expert=True,
        scatter_group_size=len(fanout_ranks),
    )

    post_process_after_weight_loading(
        model,
        buffer_dict,
        parameter_names_to_load,
        dtensor_factory,
        qlora_skip_prefixes=_expected_skip_prefixes,
        qlora_skip_fn=_should_skip_qlora_expert_key,
    )


def post_process_after_weight_loading(
    model: Union["nn.Module", "PreTrainedModel"],
    buffer_dict,
    parameter_names_left: Optional[set[str]] = None,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    qlora_skip_prefixes: Optional[set] = None,
    qlora_skip_fn: Optional[Callable] = None,
):
    """
    shared logic after weight loading that handles buffer, missing weight keys and tied embedding weights.
    """
    parameter_names_left = parameter_names_left or set()

    # Build QLoRA skip prefixes if not provided (for call sites that don't pass them)
    if qlora_skip_prefixes is None:
        qlora_skip_prefixes = set()
        for fqn, mod in model.named_modules():
            if getattr(mod, "_qlora_expected_skip_keys", None):
                for suffix in mod._qlora_expected_skip_keys:
                    qlora_skip_prefixes.add(f"{fqn}.{suffix}" if fqn else suffix)

    def _is_qlora_expert_key(key: str) -> bool:
        for prefix in qlora_skip_prefixes:
            parts = prefix.rsplit(".", 1)
            if len(parts) != 2:
                continue
            base, proj = parts
            if key.startswith(base + ".") and f".{proj}." in key:
                return True
        return False

    for name, buffer in buffer_dict.items():
        # Skip QLoRA MoE expert buffers (loaded separately via load_and_quantize_weights)
        if qlora_skip_prefixes and _is_qlora_expert_key(name):
            continue
        _dispatch_buffer(model, name, buffer, dtensor_factory)

    if parameter_names_left:
        logger.info_rank0(f"Find missing key(s) in state dict: {parameter_names_left}, initialize them.")
        for name in parameter_names_left:
            _init_parameter(model, name)

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp2 (swap tensor) contexts.
    if getattr(model.config, "tie_word_embeddings", True):
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            # PP split: one stage has embed_tokens, the other has lm_head — skip tying
            logger.info_rank0("Skipping embedding tying (input or output embeddings not present on this model part)")
        else:
            try:
                output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
            except Exception as e:
                logger.info_rank0(f"Failed to tie embeddings: {e}")
                raise RuntimeError("Failed to tie input/output embeddings") from e


def _get_shard_info(
    state_dict: Dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]],
    shard_size: int,
    safe_serialization: bool,
) -> Tuple[bool, int, Dict[str, str]]:
    """
    Gets the shard information, should be executed at rank 0.
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []
    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype
        tensor_size = tensor.numel() * get_dtype_size(dtype)  # dtensor's numel == tensor's numel
        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    num_shards = len(shard_list)
    weight_map = OrderedDict()
    is_sharded = None
    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


def _save_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike",
    safe_serialization: bool,
) -> None:
    """
    Save function.
    """
    if safe_serialization:
        save_file(state_dict, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


def _distributed_barrier_on_current_device() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return

    barrier_kwargs = {}
    if dist.get_backend() == "nccl":
        barrier_kwargs["device_ids"] = [get_device_id()]
    dist.barrier(**barrier_kwargs)


def _get_save_sync_group():
    """Get or create a dedicated process group for checkpoint-save coordination."""
    global _save_sync_group, _save_sync_group_backend
    if not dist.is_initialized():
        return None

    backend = os.getenv("XORL_SAVE_SYNC_BACKEND", "filesystem").strip().lower()
    if backend in {"", "default", "filesystem", dist.get_backend()}:
        return None

    if _save_sync_group is None or _save_sync_group_backend != backend:
        timeout_sec = int(os.getenv("XORL_SAVE_SYNC_TIMEOUT_SEC", "7200"))
        _save_sync_group = dist.new_group(backend=backend, timeout=timedelta(seconds=timeout_sec))
        _save_sync_group_backend = backend
    return _save_sync_group


def _filesystem_save_barrier(output_dir: Union[str, "os.PathLike"], barrier_name: str) -> None:
    timeout_sec = int(os.getenv("XORL_SAVE_SYNC_TIMEOUT_SEC", "7200"))
    poll_sec = float(os.getenv("XORL_SAVE_SYNC_POLL_SEC", "1.0"))
    run_id_raw = (
        os.getenv("TORCHELASTIC_RUN_ID")
        or os.getenv("XORL_SAVE_SYNC_ID")
        or f"{os.getenv('MASTER_ADDR', 'local')}_{os.getenv('MASTER_PORT', '0')}"
    )
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id_raw)
    barrier_root = os.path.join(output_dir, ".xorl_save_barriers", run_id)
    os.makedirs(barrier_root, exist_ok=True)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    arrival_marker = os.path.join(barrier_root, f"{barrier_name}.rank{rank:05d}")
    done_marker = os.path.join(barrier_root, f"{barrier_name}.done")

    with open(arrival_marker, "w", encoding="utf-8") as f:
        f.write("\n")

    deadline = time.monotonic() + timeout_sec
    arrival_prefix = f"{barrier_name}.rank"
    while True:
        arrival_count = sum(1 for entry in os.scandir(barrier_root) if entry.name.startswith(arrival_prefix))
        if arrival_count >= world_size:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for filesystem save barrier {barrier_name}: "
                f"{arrival_count}/{world_size} ranks arrived."
            )
        time.sleep(poll_sec)

    if rank == 0 and not os.path.exists(done_marker):
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write("\n")

    while not os.path.exists(done_marker):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for filesystem save barrier completion: {barrier_name}.")
        time.sleep(poll_sec)


def _distributed_save_barrier(
    output_dir: Optional[Union[str, "os.PathLike"]] = None,
    barrier_name: str = "save-sync",
) -> None:
    """Barrier helper for checkpoint save paths.

    Saving mostly coordinates CPU/filesystem work, and some clusters have shown
    NCCL barrier instability once the training/load collectives are done. Use a
    dedicated save-sync process group when configured, falling back to the
    current-device barrier only when the backends already match.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    backend = os.getenv("XORL_SAVE_SYNC_BACKEND", "filesystem").strip().lower()
    if backend == "filesystem":
        if output_dir is None:
            raise ValueError("output_dir is required for filesystem save barriers.")
        _filesystem_save_barrier(output_dir, barrier_name)
        return

    group = _get_save_sync_group()
    if group is None:
        _distributed_barrier_on_current_device()
    else:
        dist.barrier(group=group)


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, "os.PathLike"],
    state_dict: Dict[str, "torch.Tensor"],
    global_rank: Optional[int] = None,
    save_dtype: Optional[Union[str, "torch.dtype"]] = "bfloat16",
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
    checkpoint_handler: Optional["CheckpointHandler"] = None,
) -> None:
    """
    Saves full model weights. The model parameters should be either tensor or dtensor.

    If global_rank is given, it will assume it is executed on all ranks.
    If checkpoint_handler is given, applies save transforms (e.g., splitting
    gate_up_proj back into gate_proj + up_proj for HF checkpoint compatibility).
    """
    # Apply checkpoint handler save transforms before computing shard info
    if checkpoint_handler is not None:
        transformed = OrderedDict()
        for name, tensor in state_dict.items():
            for out_name, out_tensor in checkpoint_handler.on_save_weight(name, tensor):
                transformed[out_name] = out_tensor
        for out_name, out_tensor in checkpoint_handler.on_save_complete():
            transformed[out_name] = out_tensor
        state_dict = transformed

    os.makedirs(output_dir, exist_ok=True)
    is_sharded, total_size, weight_map = _get_shard_info(state_dict, save_dtype, shard_size, safe_serialization)
    full_state_dict = OrderedDict()
    prev_file_name = None
    for name, tensor in state_dict.items():
        if hasattr(tensor.data, "full_tensor"):  # dtensor
            tensor = tensor.data.full_tensor()
        else:
            tensor = tensor.data

        if save_dtype:
            tensor = tensor.to(dtype=getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype)

        if prev_file_name is not None and weight_map[name] != prev_file_name:
            if global_rank is None or global_rank == 0:
                _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)
                full_state_dict = OrderedDict()

            empty_cache()
            if global_rank is not None and dist.is_initialized():  # avoid process hanging
                synchronize()
                _distributed_barrier_on_current_device()

        if global_rank is None or global_rank == 0:
            full_state_dict[name] = tensor.detach().cpu()

        prev_file_name = weight_map[name]
        del tensor

    if global_rank is None or global_rank == 0:
        if len(full_state_dict):
            _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)

        if is_sharded:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

            logger.info(f"Model weight splits saved in {output_dir}.")
        else:
            logger.info(f"Model weights saved at {os.path.join(output_dir, prev_file_name)}.")

        if model_assets is not None:
            for model_asset in model_assets:
                if hasattr(model_asset, "save_pretrained"):
                    model_asset.save_pretrained(output_dir)
                else:
                    logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")


def save_model_weights_distributed(
    output_dir: Union[str, "os.PathLike"],
    state_dict: Dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]] = "bfloat16",
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """Save a full model checkpoint with one writer per node.

    This helper is intended for large EP/FSDP models where every rank must
    participate in DTensor materialization, but rank-0-only writing would create
    a single-node CPU/I/O bottleneck. We deterministically assign output shards
    to the local-rank-0 writer on each node while every rank still participates
    in the required ``full_tensor()`` collectives.
    """
    if not safe_serialization:
        raise ValueError("Distributed weight saving currently requires safe_serialization=True.")

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else None
        save_model_weights(
            output_dir=output_dir,
            state_dict=state_dict,
            global_rank=global_rank,
            save_dtype=save_dtype,
            shard_size=shard_size,
            safe_serialization=safe_serialization,
            model_assets=model_assets,
        )
        return

    global_rank = dist.get_rank()
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank % local_world_size)))
    writer_ranks = list(range(0, dist.get_world_size(), local_world_size))
    is_writer = local_rank == 0

    if is_writer:
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    _distributed_save_barrier(output_dir, "materialize-start")

    is_sharded, total_size, weight_map = _get_shard_info(state_dict, save_dtype, shard_size, safe_serialization)
    shard_to_keys: "OrderedDict[str, List[str]]" = OrderedDict()
    for name, file_name in weight_map.items():
        shard_to_keys.setdefault(file_name, []).append(name)

    shard_to_writer = {
        file_name: writer_ranks[idx % len(writer_ranks)] for idx, file_name in enumerate(shard_to_keys.keys())
    }
    total_tensors = len(state_dict)
    total_shards = len(shard_to_keys)

    dtype_target = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
    materialized_count = 0
    last_progress_log = time.monotonic()
    progress_log_interval = float(os.getenv("XORL_SAVE_PROGRESS_LOG_INTERVAL_SEC", "15"))

    if global_rank == 0:
        logger.info(
            "Distributed save starting: "
            f"{total_shards} shards, {total_tensors} tensors, "
            f"{len(writer_ranks)} writer ranks"
        )

    for shard_idx, (file_name, shard_keys) in enumerate(shard_to_keys.items(), start=1):
        owner_rank = shard_to_writer[file_name]
        keep_shard = owner_rank == global_rank
        shard_state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict() if keep_shard else OrderedDict()

        for shard_tensor_idx, name in enumerate(shard_keys, start=1):
            tensor = state_dict[name]
            materialized = _materialize_tensor_for_save(tensor, dst_rank=owner_rank)
            if keep_shard and materialized is not None:
                cpu_tensor = materialized.detach().cpu()
                if dtype_target is not None and cpu_tensor.dtype != dtype_target and cpu_tensor.is_floating_point():
                    cpu_tensor = cpu_tensor.to(dtype=dtype_target)
                shard_state_dict[name] = cpu_tensor

            del materialized
            materialized_count += 1
            if materialized_count % 32 == 0:
                empty_cache()

            now = time.monotonic()
            if global_rank == 0 and now - last_progress_log >= progress_log_interval:
                logger.info(
                    "Distributed save progress: "
                    f"shard {shard_idx}/{total_shards} "
                    f"({shard_tensor_idx}/{len(shard_keys)} tensors in current shard), "
                    f"materialized {materialized_count}/{total_tensors} tensors, "
                    f"current owner_rank={owner_rank}, file={file_name}"
                )
                last_progress_log = now

        if keep_shard:
            _save_state_dict(shard_state_dict, os.path.join(output_dir, file_name), safe_serialization)
            logger.info(
                f"Distributed save wrote shard {file_name} on rank {global_rank} "
                f"({len(shard_keys)} tensors, shard {shard_idx}/{total_shards})"
            )
            shard_state_dict.clear()

        if global_rank == 0 and (shard_idx == 1 or shard_idx == total_shards):
            logger.info(
                "Distributed save progress: "
                f"materialized shard {shard_idx}/{total_shards} "
                f"({materialized_count}/{total_tensors} tensors); "
                f"current owner_rank={owner_rank}, file={file_name}"
            )

    empty_cache()
    _distributed_save_barrier(output_dir, "shards-written")

    if global_rank == 0:
        if is_sharded:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(f"Distributed model weight splits saved in {output_dir}.")
        else:
            only_file = next(iter(shard_to_keys))
            logger.info(f"Distributed model weights saved at {os.path.join(output_dir, only_file)}.")

        if model_assets is not None:
            for model_asset in model_assets:
                if hasattr(model_asset, "save_pretrained"):
                    model_asset.save_pretrained(output_dir)
                else:
                    logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")

    _distributed_save_barrier(output_dir, "index-written")


def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            try:
                model_asset.save_pretrained(output_dir)
            except TypeError as e:
                if isinstance(model_asset, (PretrainedConfig, PreTrainedTokenizerBase)):
                    raise
                logger.warning(f"Skipping {type(model_asset).__name__}.save_pretrained(): {e}")
        else:
            logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")


def get_lm_head_weight(lm_head: nn.Module) -> torch.Tensor:
    """Get lm_head weight, merging LoRA delta if applicable."""
    if isinstance(lm_head, LoraLinear):
        return lm_head.weight + lm_head.get_delta_weight().to(lm_head.weight.dtype)
    return lm_head.weight


def compute_loss(
    lm_head: nn.Module,
    last_hidden_state: torch.Tensor,
    loss_fn_name: Optional[str],
    loss_fn_inputs: Optional[Dict[str, Any]],
    loss_fn_params: Optional[Dict[str, Any]],
    logits_to_keep: Union[int, torch.Tensor] = 0,
):
    """Compute loss given lm_head module and hidden states.

    Called externally (e.g. from the training loop). FSDP2 keeps lm_head.weight
    all-gathered via reshard_after_forward=False on the norm + lm_head unit.

    All tensor inputs (labels, old_logprobs, advantages, etc.) should be passed
    via ``loss_fn_inputs``.  Scalar hyper-parameters (eps_clip, compute_kl_stats,
    etc.) go in ``loss_fn_params``.

    Returns:
        LossOutput from the selected loss function.
    """
    fn_name = loss_fn_name or "causallm_loss"
    loss_fn = get_loss_function(fn_name)
    weight = get_lm_head_weight(lm_head)

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = last_hidden_state[:, slice_indices, :]

    loss_kwargs = {
        "hidden_states": hidden_states,
        "weight": weight,
    }
    ps = get_parallel_state()
    if ps.tp_enabled:
        loss_kwargs["tp_group"] = ps.tp_group
    if loss_fn_inputs:
        loss_kwargs.update(loss_fn_inputs)
    if loss_fn_params:
        loss_kwargs.update(loss_fn_params)

    return loss_fn(**loss_kwargs)


GradientCheckpointingMethod = Literal[
    "recompute_full_layer",
    "recompute_before_dispatch",
    "no_recompute",
]
DEFAULT_GRADIENT_CHECKPOINTING_METHOD: GradientCheckpointingMethod = "recompute_full_layer"


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
    (`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
    enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False
    _gradient_checkpointing_method: GradientCheckpointingMethod = DEFAULT_GRADIENT_CHECKPOINTING_METHOD

    def __call__(self, *args, **kwargs):
        if (
            self.gradient_checkpointing
            and self.training
            and self._gradient_checkpointing_method == DEFAULT_GRADIENT_CHECKPOINTING_METHOD
        ):
            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


class MoEGradientCheckpointingLayer(nn.Module):
    """Base class for MoE decoder layers with selective gradient checkpointing.

    Subclasses implement ``_pre_mlp_forward(hidden_states, **kwargs)`` which runs
    layernorm → attention → layernorm and returns ``(hidden_states, residual)``.

    The checkpointing branching is handled automatically:

    - ``recompute_full_layer``: outer checkpoint wraps the entire layer
      (handled by ``GradientCheckpointingLayer`` on the model-level loop).
    - ``recompute_before_dispatch``: this class checkpoints attn+layernorm+router
      via ``_pre_dispatch_forward``, then runs dispatch+expert+combine outside.
    - ``no_recompute``: no checkpointing, runs normally.

    Works with both alltoall and DeepEP backends since dispatch/combine sit
    outside the checkpoint boundary.
    """

    gradient_checkpointing = False
    _gradient_checkpointing_method: GradientCheckpointingMethod = DEFAULT_GRADIENT_CHECKPOINTING_METHOD

    def _pre_mlp_forward(self, hidden_states, **kwargs):
        """Layernorm → attention → layernorm. Override per model.

        Returns:
            ``(hidden_states, residual)``
        """
        raise NotImplementedError

    def _pre_dispatch_forward(self, hidden_states, **kwargs):
        """Layernorm → attention → layernorm → router.

        Composed from ``_pre_mlp_forward`` + ``self.mlp.route()``.
        Override only if the model needs custom routing (rare).
        """
        hidden_states, residual = self._pre_mlp_forward(hidden_states, **kwargs)
        # route() expects flattened (num_tokens, hidden_dim); hidden_states is 3-D here.
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, orig_shape[-1])
        routing_weights, selected_experts, router_logits = self.mlp.route(flat)
        return hidden_states, residual, routing_weights, selected_experts, router_logits

    def _moe_forward(self, hidden_states, output_router_logits=False, **kwargs):
        """Forward with selective checkpointing. Called by subclass ``forward()``.

        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)``.
            output_router_logits: Whether to include router logits in output.
            **kwargs: Forwarded to ``_pre_mlp_forward`` (attention_mask, position_embeddings, etc.)

        Returns:
            Tuple of ``(hidden_states, ...)`` with optional router_logits.
        """
        from xorl.distributed.moe.deepep import sync_pending_combine  # noqa: PLC0415
        from xorl.models.layers.moe.moe_block import MoEBlock  # noqa: PLC0415

        _selective = (
            self.training
            and self.gradient_checkpointing
            and self._gradient_checkpointing_method != DEFAULT_GRADIENT_CHECKPOINTING_METHOD
        )
        _is_moe = isinstance(self.mlp, MoEBlock)

        if _selective and _is_moe:
            moe_input, residual, routing_weights, selected_experts, router_logits = self._gradient_checkpointing_func(
                self._pre_dispatch_forward, hidden_states, **kwargs
            )
            hidden_states = self.mlp.forward_experts_only(moe_input, routing_weights, selected_experts)
        elif _selective:
            hidden_states, residual = self._gradient_checkpointing_func(self._pre_mlp_forward, hidden_states, **kwargs)
            hidden_states = self.mlp(hidden_states)
            router_logits = None
        else:
            hidden_states, residual = self._pre_mlp_forward(hidden_states, **kwargs)
            hidden_states = self.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, router_logits = hidden_states
            else:
                router_logits = None

        sync_pending_combine()
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs
