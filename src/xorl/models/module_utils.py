# Copyright 2025 xorl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Set, Tuple, Union

import torch


try:
    from hdfs_io import copy  # for internal use only
except ImportError:
    from ..utils.hdfs_io import copy
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from transformers.utils.import_utils import is_safetensors_available

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import synchronize
from ..utils.helper import empty_cache, get_cache_dir, get_dtype_size


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..distributed.parallel_plan import ParallelPlan

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


# =============================================================================
# MoE Expert Weight Auto-Merge Utilities
# =============================================================================

# Pattern to match per-expert HuggingFace weight keys
# Qwen format: model.layers.{layer}.mlp.experts.{expert}.{gate|up|down}_proj.weight
# DeepSeek format: model.layers.{layer}.mlp.experts.{expert}.{gate|up|down}_proj.weight
_EXPERT_KEY_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)

# Pattern to check if model expects fused expert format
_FUSED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate|up|down)_proj$"
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


def _checkpoint_has_per_expert_weights(checkpoint_keys: Set[str]) -> bool:
    """
    Check if checkpoint has per-expert weight format (needs merging).

    Per-expert format: model.layers.{layer}.mlp.experts.{expert}.{gate|up|down}_proj.weight

    Args:
        checkpoint_keys: Set of weight keys from checkpoint.

    Returns:
        True if checkpoint has per-expert format, False if already fused or no MoE.
    """
    for key in checkpoint_keys:
        if _EXPERT_KEY_PATTERN.match(key):
            return True
    return False


def parse_expert_key(key: str) -> Optional[Tuple[int, int, str]]:
    """
    Parse a per-expert weight key to extract layer index, expert index, and projection name.

    Args:
        key: Weight key like "model.layers.0.mlp.experts.5.gate_proj.weight"

    Returns:
        Tuple of (layer_idx, expert_idx, proj_name) or None if not a per-expert key.
        proj_name is one of "gate", "up", "down".
    """
    match = _EXPERT_KEY_PATTERN.match(key)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def _model_needs_expert_merging(parameter_names: Set[str]) -> bool:
    """
    Check if the model expects fused expert format.

    Returns True if the model has parameters like "model.layers.*.mlp.experts.gate_proj"
    (stacked format) rather than per-expert format.
    """
    for name in parameter_names:
        if _FUSED_EXPERT_PATTERN.match(name):
            return True
    return False


class ExpertWeightBuffer:
    """
    Buffer for collecting per-expert weights and merging them into stacked tensors.

    This class handles the accumulation of individual expert weights from HuggingFace
    checkpoints and stacks them into the fused format expected by xorl's MoE implementation.

    Optimized for performance:
    - Pre-allocates stacked tensor on first expert arrival
    - Copies each expert directly into slice as it arrives (streaming)
    - Avoids intermediate buffering and torch.stack overhead

    Usage:
        buffer = ExpertWeightBuffer(num_experts=128)
        for key, tensor in checkpoint:
            parsed = parse_expert_key(key)
            if parsed:
                layer_idx, expert_idx, proj = parsed
                buffer.add(layer_idx, expert_idx, proj, tensor)
                if buffer.is_complete(layer_idx, proj):
                    stacked = buffer.pop_stacked(layer_idx, proj)
                    # dispatch stacked tensor to model
    """

    def __init__(self, num_experts: int):
        """
        Initialize the expert weight buffer.

        Args:
            num_experts: Total number of experts expected per layer.
        """
        self.num_experts = num_experts
        # {(layer_idx, proj_name): pre-allocated stacked tensor}
        self._stacked_buffers: Dict[Tuple[int, str], torch.Tensor] = {}
        # {(layer_idx, proj_name): set of expert indices that have been filled}
        self._filled_experts: Dict[Tuple[int, str], Set[int]] = defaultdict(set)

    def add(self, layer_idx: int, expert_idx: int, proj: str, tensor: torch.Tensor) -> None:
        """
        Add a per-expert tensor to the buffer.

        On first expert for a (layer, proj), pre-allocates the full stacked tensor.
        Each expert tensor is copied directly into its slice as it arrives.

        Args:
            layer_idx: Layer index (0-indexed)
            expert_idx: Expert index within the layer (0-indexed)
            proj: Projection name ("gate", "up", or "down")
            tensor: The weight tensor for this expert's projection
        """
        key = (layer_idx, proj)

        # Pre-allocate stacked tensor on first expert
        if key not in self._stacked_buffers:
            # Allocate on CPU with same dtype, shape = [num_experts, *tensor.shape]
            stacked_shape = (self.num_experts,) + tensor.shape
            self._stacked_buffers[key] = torch.empty(
                stacked_shape, dtype=tensor.dtype, device="cpu"
            )

        # Copy directly into the pre-allocated slice
        self._stacked_buffers[key][expert_idx].copy_(tensor)
        self._filled_experts[key].add(expert_idx)

    def is_complete(self, layer_idx: int, proj: str) -> bool:
        """
        Check if all experts for a given (layer, projection) have been collected.

        Args:
            layer_idx: Layer index
            proj: Projection name

        Returns:
            True if all num_experts tensors have been collected.
        """
        key = (layer_idx, proj)
        return len(self._filled_experts.get(key, set())) == self.num_experts

    def pop_stacked(self, layer_idx: int, proj: str) -> torch.Tensor:
        """
        Return the completed stacked tensor. Removes from buffer.

        Args:
            layer_idx: Layer index
            proj: Projection name

        Returns:
            Stacked tensor of shape [num_experts, ...] where ... is the per-expert shape.

        Raises:
            KeyError: If the (layer, proj) combination doesn't exist in buffer.
            ValueError: If not all experts have been collected.
        """
        key = (layer_idx, proj)
        if key not in self._stacked_buffers:
            raise KeyError(f"No buffered experts for layer {layer_idx}, projection {proj}")

        filled = self._filled_experts.pop(key)
        if len(filled) != self.num_experts:
            raise ValueError(
                f"Incomplete experts for layer {layer_idx}, {proj}_proj: "
                f"got {len(filled)}, expected {self.num_experts}"
            )

        # Return pre-allocated tensor directly (all copies already done in add())
        return self._stacked_buffers.pop(key)

    @staticmethod
    def get_fused_name(layer_idx: int, proj: str) -> str:
        """
        Get the fused parameter name for a (layer, projection) combination.

        Args:
            layer_idx: Layer index
            proj: Projection name ("gate", "up", or "down")

        Returns:
            Parameter name like "model.layers.0.mlp.experts.gate_proj"
        """
        return f"model.layers.{layer_idx}.mlp.experts.{proj}_proj"

    def get_pending_keys(self) -> List[Tuple[int, str]]:
        """
        Get list of (layer_idx, proj) combinations that have partial data.

        Useful for debugging and error reporting.
        """
        return list(self._stacked_buffers.keys())

    def get_pending_counts(self) -> Dict[Tuple[int, str], int]:
        """
        Get counts of collected experts for each pending (layer, proj).

        Returns:
            Dict mapping (layer_idx, proj) to number of experts collected.
        """
        return {key: len(experts) for key, experts in self._filled_experts.items()}


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

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


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

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
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
                        f"OSError getting DIFFUSERS shard files (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get DIFFUSERS shard files after {max_retries} attempts")
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


def _get_num_experts_from_config(model: Union["nn.Module", "PreTrainedModel"]) -> Optional[int]:
    """
    Get the number of experts from model config.

    Supports both Qwen (num_experts) and DeepSeek (n_routed_experts) naming conventions.
    """
    if not hasattr(model, "config"):
        return None
    config = model.config
    # Qwen format
    if hasattr(config, "num_experts"):
        return config.num_experts
    # DeepSeek format
    if hasattr(config, "n_routed_experts"):
        return config.n_routed_experts
    return None


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Loads pre-trained model states in transformers' format.

    Supports automatic merging of per-expert MoE weights into stacked format.
    If the model expects fused expert format (e.g., moe_implementation='fused')
    but the checkpoint has per-expert weights, they will be automatically merged.

    Also supports loading already-fused checkpoints directly (no merging needed).
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    model.to_empty(device=init_device)

    # Get parallel plan if available
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    # Detect checkpoint format and initialize expert buffer only if needed
    expert_buffer = None
    model_needs_fused = _model_needs_expert_merging(parameter_names_to_load)

    if model_needs_fused:
        # Check checkpoint format to decide if merging is needed
        checkpoint_keys = _get_checkpoint_keys(weights_path)
        if checkpoint_keys is not None:
            checkpoint_has_per_expert = _checkpoint_has_per_expert_weights(checkpoint_keys)
            if checkpoint_has_per_expert:
                num_experts = _get_num_experts_from_config(model)
                if num_experts is not None:
                    expert_buffer = ExpertWeightBuffer(num_experts)
                    logger.info_rank0(
                        f"MoE checkpoint format: per-expert (will merge {num_experts} experts into fused format)"
                    )
            else:
                logger.info_rank0(
                    "MoE checkpoint format: already fused (direct loading, no merge needed)"
                )
        else:
            # Fallback: couldn't read checkpoint keys, enable merging to be safe
            num_experts = _get_num_experts_from_config(model)
            if num_experts is not None:
                expert_buffer = ExpertWeightBuffer(num_experts)
                logger.info_rank0(
                    f"MoE auto-merge enabled: will merge {num_experts} per-expert weights if needed"
                )

    # Retry loading state dict on OSError (e.g., HuggingFace download issues)
    max_retries = 3
    retry_delay = 5  # seconds
    last_error = None

    for attempt in range(max_retries):
        try:
            state_dict_iterators = _load_state_dict(weights_path)
            break  # Success, exit retry loop
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
        # This should not be reached, but just in case
        if last_error:
            raise last_error

    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for name, tensor in state_dict_iterator:
            # IMPORTANT: Call this function to adapt to transformers 4.52 breaking change
            # on model structure. See the comment for details.
            name = _convert_weight_key(name, model)

            # Check if this is a per-expert key that needs merging
            if expert_buffer is not None:
                parsed = parse_expert_key(name)
                if parsed is not None:
                    layer_idx, expert_idx, proj = parsed
                    expert_buffer.add(layer_idx, expert_idx, proj, tensor)

                    # If all experts collected, stack and dispatch
                    if expert_buffer.is_complete(layer_idx, proj):
                        fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                        stacked = expert_buffer.pop_stacked(layer_idx, proj)

                        if fused_name in parameter_names_to_load:
                            parameter_names_to_load.remove(fused_name)
                            _dispatch_parameter(model, fused_name, stacked, dtensor_factory, parallel_plan)
                            logger.info_rank0(
                                f"Auto-merged experts for {fused_name}: {stacked.shape}"
                            )
                        else:
                            logger.warning(
                                f"Auto-merged expert key {fused_name} not found in model parameters"
                            )
                    continue  # Don't process as normal key

            # Normal processing for non-expert keys
            if name in buffer_dict.keys():  # persistent buffers
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names_to_load:
                parameter_names_to_load.remove(name)
                _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan)
            else:
                logger.info_rank0(f"Unexpected key in state dict: {name}.")

        del state_dict_iterator
        empty_cache()

    # Check for incomplete expert buffers (shouldn't happen with valid checkpoints)
    if expert_buffer is not None:
        pending = expert_buffer.get_pending_counts()
        if pending:
            logger.warning(
                f"Incomplete expert weights after loading all shards: {pending}. "
                f"Some experts may be missing from the checkpoint."
            )

    post_process_after_weight_loading(model, buffer_dict, parameter_names_to_load, dtensor_factory)


@torch.no_grad()
def rank0_load_and_broadcast_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
):
    """
    This functions serves as the same purpose as `load_model_weights`
    but reduces disk I/O by broadcasting weights from rank0.
    In comparison, `load_model_weights` would require every GPU to go through the entire model weights on disk.

    Supports automatic merging of per-expert MoE weights into stacked format.
    If the model expects fused expert format (e.g., moe_implementation='fused')
    but the checkpoint has per-expert weights, they will be automatically merged on rank0
    before broadcasting.

    Also supports loading already-fused checkpoints directly (no merging needed).
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning_once("Distributed environment not initialized, falling back to load_model_weights.")
        return load_model_weights(model, weights_path, init_device, dtensor_factory)

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    model.to_empty(device=init_device)

    # Get parallel plan if available
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        parallel_plan = model.get_parallel_plan()

    global_rank = get_parallel_state().global_rank
    torch_device = torch.device(init_device)

    # Detect checkpoint format and initialize expert buffer only if needed
    expert_buffer = None
    model_needs_fused = _model_needs_expert_merging(parameter_names_to_load)

    if model_needs_fused:
        # Check checkpoint format to decide if merging is needed
        checkpoint_keys = _get_checkpoint_keys(weights_path)
        if checkpoint_keys is not None:
            checkpoint_has_per_expert = _checkpoint_has_per_expert_weights(checkpoint_keys)
            if checkpoint_has_per_expert:
                num_experts = _get_num_experts_from_config(model)
                if num_experts is not None:
                    expert_buffer = ExpertWeightBuffer(num_experts)
                    logger.info_rank0(
                        f"MoE checkpoint format: per-expert (will merge {num_experts} experts into fused format)"
                    )
            else:
                logger.info_rank0(
                    "MoE checkpoint format: already fused (direct loading, no merge needed)"
                )
        else:
            # Fallback: couldn't read checkpoint keys, enable merging to be safe
            num_experts = _get_num_experts_from_config(model)
            if num_experts is not None:
                expert_buffer = ExpertWeightBuffer(num_experts)
                logger.info_rank0(
                    f"MoE auto-merge enabled: will merge {num_experts} per-expert weights if needed"
                )

    # get the safetensor file iterator
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

    if global_rank == 0:
        # only rank0 would actual read weights from safetensor state_dict iterators
        shard_iterable = enumerate(
            tqdm(
                state_dict_iterators,
                desc="Loading checkpoint shards",
                # only rank0 displays tqdm pbar
                disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
            )
        )
    else:
        shard_iterable = enumerate(range(shard_count))

    # Queue for merged tensors ready to broadcast (used when auto-merge is enabled)
    # This allows rank0 to buffer per-expert weights and broadcast merged results
    merged_queue: List[Tuple[str, torch.Tensor]] = []

    # iterate safetensor files; each file would have a iterator to read weight keys and tensors
    for shard_idx, shard_payload in shard_iterable:
        state_dict_iterator = shard_payload if global_rank == 0 else None
        iterator = iter(state_dict_iterator) if global_rank == 0 else None

        while True:
            # read tensors from safetensor
            tensor: Optional["torch.Tensor"] = None
            broadcast_name: Optional[str] = None
            broadcast_tensor: Optional[torch.Tensor] = None

            if global_rank == 0:
                # First, check if we have merged tensors ready to broadcast
                if merged_queue:
                    broadcast_name, broadcast_tensor = merged_queue.pop(0)
                    metadata = BroadcastMetadata(False, broadcast_name, broadcast_tensor.shape, broadcast_tensor.dtype)
                else:
                    # Read next tensor from checkpoint
                    try:
                        key, tensor = next(iterator)  # type: ignore[arg-type]
                        key = _convert_weight_key(key, model)
                        logger.info_rank0(f"loading {key=}")
                        if torch.count_nonzero(tensor) == 0:
                            logger.warning_rank0(f"Detected tensor with all-zero values when reading safetensor: {key=}")

                        # Check if this is a per-expert key that needs merging
                        if expert_buffer is not None:
                            parsed = parse_expert_key(key)
                            if parsed is not None:
                                layer_idx, expert_idx, proj = parsed
                                expert_buffer.add(layer_idx, expert_idx, proj, tensor)

                                # If all experts collected, queue merged tensor for broadcast
                                if expert_buffer.is_complete(layer_idx, proj):
                                    fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                                    stacked = expert_buffer.pop_stacked(layer_idx, proj)
                                    merged_queue.append((fused_name, stacked))
                                    logger.info_rank0(
                                        f"Auto-merged experts for {fused_name}: {stacked.shape}"
                                    )

                                # Continue reading - don't broadcast per-expert tensors
                                # Signal to skip this iteration with a special metadata
                                metadata = BroadcastMetadata(False, "__SKIP__", None, None)
                            else:
                                # Normal key - broadcast directly
                                broadcast_name = key
                                broadcast_tensor = tensor
                                metadata = BroadcastMetadata(False, key, tensor.shape, tensor.dtype)
                        else:
                            # No auto-merge - broadcast directly
                            broadcast_name = key
                            broadcast_tensor = tensor
                            metadata = BroadcastMetadata(False, key, tensor.shape, tensor.dtype)

                    except StopIteration:
                        # Check if we have remaining merged tensors to broadcast
                        if merged_queue:
                            broadcast_name, broadcast_tensor = merged_queue.pop(0)
                            metadata = BroadcastMetadata(False, broadcast_name, broadcast_tensor.shape, broadcast_tensor.dtype)
                        else:
                            metadata = BroadcastMetadata(True, None, None, None)
            else:
                metadata = BroadcastMetadata(False, None, None, None)

            metadata_list = [metadata]
            dist.broadcast_object_list(metadata_list, src=0)
            metadata = metadata_list[0]

            if metadata.done:
                break

            # Handle skip signal (rank0 is still buffering per-expert weights)
            if metadata.name == "__SKIP__":
                continue

            name = metadata.name
            shape = metadata.shape
            dtype = metadata.dtype
            if name is None or shape is None or dtype is None:
                raise RuntimeError("Received incomplete broadcast metadata.")
            logger.info_rank0(f"rank0_load_and_broadcast_weights: broadcasting {name=}")

            if global_rank != 0:
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
            else:
                tensor = broadcast_tensor.to(torch_device, non_blocking=True)  # type: ignore[assignment]

            start_time = time.perf_counter()
            dist.broadcast(tensor, src=0)
            logger.info_rank0(
                f"{name=}, {shape=}, {dtype=}, broadcast time (ms) spent: {1000 * (time.perf_counter() - start_time)}"
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
            del state_dict_iterator

        empty_cache()

    # Check for incomplete expert buffers (shouldn't happen with valid checkpoints)
    if expert_buffer is not None and global_rank == 0:
        pending = expert_buffer.get_pending_counts()
        if pending:
            logger.warning(
                f"Incomplete expert weights after loading all shards: {pending}. "
                f"Some experts may be missing from the checkpoint."
            )

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
    # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
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
) -> None:
    """
    Saves full model weights. The model parameters should be either tensor or dtensor.

    If global_rank is given, it will assume it is executed on all ranks.
    """
    if output_dir.startswith("hdfs://"):
        hdfs_dir = output_dir
        hdfs_upper_dir = output_dir.rstrip("/")
        hdfs_upper_dir = hdfs_upper_dir[: hdfs_upper_dir.rfind("/")]
        output_dir = get_cache_dir(output_dir)
    else:
        hdfs_dir = None

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

        if hdfs_dir is not None:
            copy(output_dir, hdfs_upper_dir)
            logger.info(f"Model weights uploaded to {hdfs_dir}.")


def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    if output_dir.startswith("hdfs://"):
        hdfs_dir = output_dir
        hdfs_upper_dir = output_dir.rstrip("/")
        hdfs_upper_dir = hdfs_upper_dir[: hdfs_upper_dir.rfind("/")]
        output_dir = get_cache_dir(output_dir)
    else:
        hdfs_dir = None

    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            model_asset.save_pretrained(output_dir)
        else:
            logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")

    if hdfs_dir is not None:
        copy(output_dir, hdfs_upper_dir)
        logger.info(f"Model config and tokenizer uploaded to {hdfs_dir}.")


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