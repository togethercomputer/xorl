
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

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import synchronize
from ..utils.helper import empty_cache, get_dtype_size


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..distributed.parallel_plan import ParallelPlan
    from .checkpoint_handlers.base import CheckpointHandler

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


# Re-export checkpoint handler utilities for backward compatibility (used by tests)
from .checkpoint_handlers.buffers import (  # noqa: F401
    ExpertWeightBuffer,
    parse_expert_key,
    checkpoint_has_per_expert_weights as _checkpoint_has_per_expert_weights,
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

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


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
    full_param_name = name
    module, local_name = _find_submodule(module, name)
    orig_tensor = module._parameters[local_name].data

    # Handle parameter slicing according to parallel_plan, now only EP-aware
    if parallel_plan is not None:
        tensor = parallel_plan.shard_tensor(tensor, full_param_name, orig_tensor.shape)

    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
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
        module._buffers[name].copy_(buffer.to(device=orig_tensor.device, dtype=orig_tensor.dtype))


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
    model.to_empty(device=init_device)

    # Get parallel plan if available
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    # Get checkpoint handler from model (delegates weight transforms to model-specific logic)
    handler = None
    if hasattr(model, "get_checkpoint_handler"):
        _ps = get_parallel_state()
        ep_rank = _ps.ep_rank if _ps.ep_enabled else 0
        ep_size = _ps.ep_size if _ps.ep_enabled else 1
        checkpoint_keys = _get_checkpoint_keys(weights_path)
        handler = model.get_checkpoint_handler(
            checkpoint_keys=checkpoint_keys or set(),
            ep_rank=ep_rank,
            ep_size=ep_size,
            is_broadcast=False,
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

    # Check if the handler provides an EP-aware skip function to avoid reading
    # out-of-range expert tensor data from disk (can reduce I/O by ~60-70%).
    skip_key_fn = handler.get_skip_key_fn() if handler is not None else None

    def _dispatch_results(results):
        for param_name, param_tensor in results:
            if param_name in buffer_dict:
                buffer_dict[param_name] = param_tensor.clone()
            elif param_name in parameter_names_to_load:
                parameter_names_to_load.remove(param_name)
                _dispatch_parameter(model, param_name, param_tensor, dtensor_factory, parallel_plan)
            else:
                logger.info_rank0(f"Unexpected key in state dict: {param_name}.")

    if skip_key_fn is not None:
        # EP-aware filtered loading: skip reading tensor data for out-of-range experts
        logger.info_rank0(
            f"EP-aware filtered loading enabled (ep_rank={ep_rank}, ep_size={ep_size}): "
            "skipping disk reads for out-of-range expert weights"
        )
        for state_dict, skipped_keys in tqdm(
            _prefetch_shards_filtered(state_dict_iterators, skip_key_fn),
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

            del state_dict
            empty_cache()
    else:
        # Standard bulk loading (no EP filtering)
        for state_dict in tqdm(
            _prefetch_shards(state_dict_iterators),
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

            del state_dict
            empty_cache()

    # Flush handler buffers (warn on incomplete merges)
    if handler is not None:
        for param_name, param_tensor in handler.on_load_complete():
            if param_name in parameter_names_to_load:
                parameter_names_to_load.remove(param_name)
                _dispatch_parameter(model, param_name, param_tensor, dtensor_factory, parallel_plan)

    post_process_after_weight_loading(model, buffer_dict, parameter_names_to_load, dtensor_factory)


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
        )

    # Get the safetensor file iterators
    state_dict_iterators = _load_state_dict(weights_path) if global_rank == 0 else None
    shard_count = len(state_dict_iterators) if global_rank == 0 else 0
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
                tensor = broadcast_tensor.to(torch_device, non_blocking=True)

            start_time = time.perf_counter()
            dist.broadcast(tensor, src=0)
            logger.info_rank0(
                f"{name=}, {shape=}, {dtype=}, broadcast time (ms): "
                f"{1000 * (time.perf_counter() - start_time)}"
            )

            if name in buffer_dict:
                buffer_dict[name] = tensor.detach().clone()
            elif name in parameter_names_to_load:
                parameter_names_to_load.discard(name)
                _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan)
            else:
                if global_rank == 0:
                    logger.info_rank0(f"Unexpected key in state dict: {name}.")

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
            tensor = broadcast_tensor.to(torch_device, non_blocking=True)

        dist.broadcast(tensor, src=0)

        if name in parameter_names_to_load:
            parameter_names_to_load.discard(name)
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan)

        del tensor
        if global_rank == 0:
            del broadcast_tensor

    post_process_after_weight_loading(model, buffer_dict, parameter_names_to_load, dtensor_factory)


def post_process_after_weight_loading(
    model: Union["nn.Module", "PreTrainedModel"],
    buffer_dict,
    parameter_names_left: Optional[set[str]] = None,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
):
    """
    shared logic after weight loading that handles buffer, missing weight keys and tied embedding weights.
    """
    parameter_names_left = parameter_names_left or set()

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer, dtensor_factory)

    if parameter_names_left:
        logger.info_rank0(f"Find missing key(s) in state dict: {parameter_names_left}, initialize them.")
        for name in parameter_names_left:
            _init_parameter(model, name)

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp2 (swap tensor) contexts.
    if getattr(model.config, "tie_word_embeddings", True):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
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