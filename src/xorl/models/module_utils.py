
import json
import os
import re
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Set, Tuple, Union

import torch


from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from xorl.distributed.parallel_state import get_parallel_state
from xorl.lora.modules.linear import LoraLinear
from xorl.ops.loss import get_loss_function
from xorl.utils import logging
from xorl.utils.device import synchronize
from xorl.utils.helper import empty_cache, get_dtype_size


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from xorl.distributed.parallel_plan import ParallelPlan
    from xorl.models.checkpoint_handlers.base import CheckpointHandler

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


# Re-export checkpoint handler utilities for backward compatibility (used by tests)
from xorl.models.checkpoint_handlers.buffers import (  # noqa: F401
    ExpertWeightBuffer,
    parse_expert_key,
    checkpoint_has_per_expert_weights,
)


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
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

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
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

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
    import time
    max_retries = 5
    for attempt in range(max_retries):
        result = _try_load_state_dict(weights_path, **kwargs)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            retry_delay = 2 * (2 ** attempt)  # 2, 4, 8, 16s
            logger.warning(
                f"Cannot find checkpoint files in {weights_path} (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def _try_load_state_dict(weights_path: str, **kwargs):
    """
    Single attempt to load state dict. Returns list of iterators or None if not found.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        # Retry on OSError (e.g., missing shard files during download)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
                return [StateDictIterator(shard_file) for shard_file in shard_files]
            except OSError as e:
                if attempt < max_retries - 1:
                    import time
                    retry_delay = 5
                    logger.warning(
                        f"OSError getting shard files (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get shard files after {max_retries} attempts")
                    raise

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        # Retry on OSError (e.g., missing shard files during download)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
                return [StateDictIterator(shard_file) for shard_file in shard_files]
            except OSError as e:
                if attempt < max_retries - 1:
                    import time
                    retry_delay = 5
                    logger.warning(
                        f"OSError getting PT shard files (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get PT shard files after {max_retries} attempts")
                    raise

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
        assert tensor.device.type == "cpu", (
            f"DMA scheduler only handles CPU→CUDA, got {tensor.device} for {full_name}"
        )
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
                module._parameters[local_name].data.copy_(
                    dtensor_factory(gpu_temp, device_mesh, placements)
                )
                # distribute_tensor uses NCCL on a different CUDA stream.
                # Synchronize to prevent races with subsequent DMA transfers
                # on the copy-engine streams.
                torch.cuda.synchronize()
            else:
                module._parameters[local_name].data.copy_(gpu_temp)
        self._pending.clear()


# Module-level scheduler, set during bulk loading to enable multi-stream DMA.
_active_dma_scheduler: Optional[_MultiStreamDMA] = None


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
    if (
        _active_dma_scheduler is not None
        and tensor.device.type == "cpu"
        and orig_tensor.device.type == "cuda"
    ):
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
            from torch.distributed._tensor import Shard as DTShard
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
    import torch.nn as nn
    import math

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
    model_device = next(model.parameters()).device if any(True for _ in model.parameters()) else None
    handler = None
    if hasattr(model, "get_checkpoint_handler"):
        ep_rank = _ps.ep_rank if _ps.ep_enabled else 0
        ep_size = _ps.ep_size if _ps.ep_enabled else 1
        checkpoint_keys = _get_checkpoint_keys(weights_path)
        handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=ep_rank,
            ep_size=ep_size,
            is_broadcast=False,
            weights_path=weights_path,
            device=model_device,
        )

    # Retry loading state dict on OSError (e.g., HuggingFace download issues)
    max_retries = 3
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
                import time
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
        model, buffer_dict, parameter_names_to_load, dtensor_factory,
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
        handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=0,
            ep_size=1,
            is_broadcast=True,
            weights_path=weights_path,
        )

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
    dist.broadcast(shard_count_tensor, src=0)
    shard_count = int(shard_count_tensor.item())

    # Rank 0: create prefetching generator that bulk-loads upcoming shards in
    # background threads while the current shard is being broadcast.
    # prefetch_count=2 allows 2 concurrent NFS reads for higher aggregate throughput.
    if global_rank == 0:
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
            batch_meta = [(n, t.shape, t.dtype) for n, t in merged_queue]
        else:
            batch_meta = None

        batch_meta = [batch_meta]
        dist.broadcast_object_list(batch_meta, src=0)
        batch_meta = batch_meta[0]

        for name, shape, dtype in batch_meta:
            # Broadcast tensor from rank 0 to all ranks.
            # For expert tensors, this broadcasts the full [num_experts, ...] tensor;
            # _dispatch_parameter -> shard_tensor handles EP slicing locally per rank.
            if global_rank != 0:
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
            else:
                _, broadcast_tensor = merged_queue.pop(0)
                if broadcast_tensor.device.type == "cpu" and torch_device.type == "cuda":
                    tensor = broadcast_tensor.pin_memory().to(torch_device, non_blocking=True)
                else:
                    tensor = broadcast_tensor.to(torch_device, non_blocking=True)

            start_time = time.perf_counter()
            dist.broadcast(tensor, src=0)
            logger.info_rank0(
                f"{name=}, {shape=}, {dtype=}, broadcast time (ms): "
                f"{1000 * (time.perf_counter() - start_time)}"
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
                del broadcast_tensor

        empty_cache()

    # Flush handler buffers after all shards (broadcast any remaining merged items)
    if handler is not None and global_rank == 0:
        merged_queue.extend(handler.on_load_complete())

    # Broadcast remaining items from handler flush (same batched pattern)
    if global_rank == 0:
        flush_meta = [(n, t.shape, t.dtype) for n, t in merged_queue]
    else:
        flush_meta = None

    flush_meta = [flush_meta]
    dist.broadcast_object_list(flush_meta, src=0)
    flush_meta = flush_meta[0]

    for name, shape, dtype in flush_meta:
        if global_rank != 0:
            tensor = torch.empty(shape, dtype=dtype, device=torch_device)
        else:
            _, broadcast_tensor = merged_queue.pop(0)
            if broadcast_tensor.device.type == "cpu" and torch_device.type == "cuda":
                tensor = broadcast_tensor.pin_memory().to(torch_device, non_blocking=True)
            else:
                tensor = broadcast_tensor.to(torch_device, non_blocking=True)

        dist.broadcast(tensor, src=0)

        # Resolve _orig_mod prefix from torch.compile
        model_name = _compiled_key_map.get(name, name)
        if model_name in parameter_names_to_load:
            parameter_names_to_load.discard(model_name)
            _dispatch_parameter(model, model_name, tensor, dtensor_factory, parallel_plan)

        del tensor
        if global_rank == 0:
            del broadcast_tensor

    post_process_after_weight_loading(model, buffer_dict, parameter_names_to_load, dtensor_factory)


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
                dist.barrier()

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



def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            model_asset.save_pretrained(output_dir)
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

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)
