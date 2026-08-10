"""
LoRA utility functions.

Functions for injecting LoRA into models and managing LoRA state dicts.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.tensor import distribute_tensor

from xorl.lora.mapping import can_apply_lora, get_lora_class_for_module
from xorl.lora.modules import LoraLinear
from xorl.lora.modules.base import LoraModule


logger = logging.getLogger(__name__)


# Default target modules for common model architectures
DEFAULT_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "deepseek_v3": [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # The GLM-5 DSA indexer's wq_b/wk/weights_proj are intentionally
    # excluded; they're tiny and sparse-MLA expects them unwrapped.
    "glm_moe_dsa": [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "xorl_glm5": [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "kimi_k2": [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "kimi_k25": [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

_PEFT_BASE_MODEL_PREFIX = "base_model.model."
_MOE_LORA_PATTERN = re.compile(r"(.*)\.mlp\.experts\.(gate_proj|up_proj|down_proj)_lora_(A|B)$")
_MOE_PEFT_LORA_PATTERN = re.compile(
    r"(.*)\.mlp\.experts\.(shared|\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight$"
)
_MOE_SGLANG_SHARED_OUTER_PATTERN = re.compile(r"(.*)\.mlp\.experts\.(w1|w2|w3)\.lora_(A|B)\.weight$")

# SGLang shared_outer format uses w1/w2/w3 slots for gate/down/up projections.
_PROJ_TO_SGLANG_W = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
_SGLANG_W_TO_PROJ = {v: k for k, v in _PROJ_TO_SGLANG_W.items()}

LORA_EXPORT_FORMATS = ("peft", "sglang_shared_outer")


@dataclass(frozen=True)
class LoraTensorShardSpec:
    """Shard metadata needed to map a full LoRA tensor onto the current rank."""

    dim: int
    index: int
    size: int


def _get_default_target_modules(model: nn.Module) -> List[str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type in DEFAULT_TARGET_MODULES:
        return list(DEFAULT_TARGET_MODULES[model_type])
    if model_type is not None:
        for family, targets in DEFAULT_TARGET_MODULES.items():
            if family in model_type:
                return list(targets)
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _get_submodule(model: nn.Module, target: str) -> Tuple[nn.Module, str]:
    """
    Get parent module and attribute name for a target path.

    Args:
        model: Root module
        target: Dot-separated path (e.g., "model.layers.0.self_attn.q_proj")

    Returns:
        Tuple of (parent_module, attribute_name)
    """
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_target_modules(
    model: nn.Module,
    target_modules: List[str],
) -> List[str]:
    """
    Find all module paths matching target module names.

    Uses the LoRA mapping registry to determine which modules can have
    LoRA applied to them. Supports two matching modes:

    1. Direct match: Module name matches a target (e.g., "q_proj", "experts")
    2. Indirect match: Module has children matching targets (e.g., MoE experts
       module has "gate_proj", "up_proj", "down_proj" as weight attributes)

    The algorithm processes modules top-down and skips children of replaced
    modules to avoid double-replacement.

    Args:
        model: Model to search
        target_modules: List of module name patterns to match

    Returns:
        List of full module paths that match (in top-down order)
    """
    matched_paths = []
    replaced_prefixes = set()  # Track replaced module paths to skip their children

    for name, module in model.named_modules():
        # Skip if this module is under an already-matched parent
        # (avoid replacing children of modules we're going to replace)
        if any(name.startswith(prefix + ".") for prefix in replaced_prefixes):
            continue

        # Check if LoRA can be applied to this module type
        if not can_apply_lora(module):
            continue

        module_name = name.split(".")[-1] if name else ""

        # Direct match: module name matches target_modules
        if module_name in target_modules:
            matched_paths.append(name)
            replaced_prefixes.add(name)
            continue

        # Indirect match: module has attributes/children matching target_modules
        # This handles MoE experts where user specifies "gate_proj" but the
        # actual module to replace is "experts" which contains gate_proj weights
        module_attrs = set(dir(module))
        if any(target in module_attrs for target in target_modules):
            matched_paths.append(name)
            replaced_prefixes.add(name)
            continue

    return matched_paths


def inject_lora_into_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Inject LoRA adapters into a model by replacing target modules.

    This function finds all modules matching the target patterns that have
    a LoRA replacement registered in the mapping, and replaces them with
    their LoRA variants.

    Supports:
    - nn.Linear -> LoraLinear
    - MoEExperts (and subclasses) -> MoEExpertsLoRA

    Args:
        model: Model to inject LoRA into
        r: LoRA rank
        lora_alpha: LoRA alpha for scaling
        target_modules: List of module names to target (e.g., ["q_proj", "v_proj"]).
                       If None, uses default attention modules.

    Returns:
        The model with LoRA layers injected (modified in-place)

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> inject_lora_into_model(
        ...     model,
        ...     r=16,
        ...     lora_alpha=32,
        ...     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ... )

        >>> # For MoE models (works the same way)
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B")
        >>> inject_lora_into_model(
        ...     model,
        ...     r=16,
        ...     lora_alpha=32,
        ...     target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        ... )
    """
    if target_modules is None:
        target_modules = _get_default_target_modules(model)

    # Find all matching modules
    target_paths = _find_target_modules(model, target_modules)

    if not target_paths:
        raise ValueError(
            f"No modules found matching target_modules={target_modules}. "
            f"Please check that the model has modules with these names. "
            f"Available module names: {[name.split('.')[-1] for name, _ in model.named_modules() if name][:20]}..."
        )

    logger.info(f"Injecting LoRA into {len(target_paths)} modules with r={r}, alpha={lora_alpha}")

    # Replace each target module
    replaced_count = 0
    skipped_modules = []
    for target_path in target_paths:
        parent, attr_name = _get_submodule(model, target_path)
        original_module = getattr(parent, attr_name)

        # Get appropriate LoRA class from registry
        lora_cls = get_lora_class_for_module(original_module)
        if lora_cls is None:
            skipped_modules.append((target_path, type(original_module).__name__))
            continue

        # Create LoRA module using unified from_module interface
        lora_module = lora_cls.from_module(
            original_module,
            r=r,
            lora_alpha=lora_alpha,
        )

        # Replace in parent
        setattr(parent, attr_name, lora_module)
        replaced_count += 1

        logger.debug(f"Replaced {target_path} with {lora_cls.__name__}")

    # Check if any modules were actually replaced
    if replaced_count == 0:
        skipped_info = ", ".join([f"{path} ({typ})" for path, typ in skipped_modules[:5]])
        if len(skipped_modules) > 5:
            skipped_info += f"... and {len(skipped_modules) - 5} more"
        raise ValueError(
            f"No modules could be replaced with LoRA. "
            f"Found {len(target_paths)} matching modules but none have LoRA support. "
            f"Skipped modules: {skipped_info}. "
            f"Supported module types: nn.Linear, MoEExperts (and subclasses)"
        )

    if skipped_modules:
        logger.warning(
            f"Skipped {len(skipped_modules)} modules without LoRA support: "
            f"{[path for path, _ in skipped_modules[:5]]}{'...' if len(skipped_modules) > 5 else ''}"
        )

    logger.info(f"Successfully injected LoRA into {replaced_count} modules")
    return model


def freeze_base_parameters(model: nn.Module) -> None:
    """
    Freeze all non-LoRA parameters in the model.

    Sets requires_grad=False for all parameters except lora_A and lora_B.

    Args:
        model: Model with LoRA layers
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """
    Get iterator over trainable LoRA parameters.

    Args:
        model: Model with LoRA layers

    Yields:
        LoRA parameters (lora_A and lora_B)
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def _gather_ep_tensor(tensor: torch.Tensor, spec_info) -> torch.Tensor:
    """Gather an EP-sharded tensor from all EP ranks.

    Args:
        tensor: The local tensor (already gathered from FSDP if applicable)
        spec_info: SpecInfo with EP placement and mesh information

    Returns:
        Full tensor with all EP shards concatenated along the shard dimension.
        Returns tensor unchanged if placement is Replicate.
    """

    if isinstance(spec_info.placement, Replicate):
        return tensor

    assert isinstance(spec_info.placement, Shard)
    shard_dim = spec_info.placement.dim
    ep_mesh = spec_info.ep_mesh
    ep_size = ep_mesh.size()
    ep_pg = ep_mesh.get_group()

    gathered = [torch.empty_like(tensor) for _ in range(ep_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=ep_pg)
    return torch.cat(gathered, dim=shard_dim)


def _lora_rank_dim(param_name: str) -> Optional[int]:
    """Return the rank dimension for xorl LoRA parameter names."""
    if param_name == "lora_A":
        return 0
    if param_name == "lora_B":
        return 1
    if param_name.endswith("_lora_A"):
        return -1
    if param_name.endswith("_lora_B"):
        return 1
    return None


def _active_lora_rank_slices(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """Map LoRA parameter FQNs to (rank_dim, active_rank) from live modules."""
    rank_slices: Dict[str, Tuple[int, int]] = {}
    for module_name, module in model.named_modules():
        active_rank = getattr(module, "active_r", None)
        if active_rank is None:
            continue
        active_rank = int(active_rank)
        if active_rank <= 0:
            raise ValueError(f"Active LoRA rank must be positive, got {active_rank}")

        prefix = f"{module_name}." if module_name else ""
        for local_name, _ in module.named_parameters(recurse=False):
            rank_dim = _lora_rank_dim(local_name)
            if rank_dim is not None:
                rank_slices[f"{prefix}{local_name}"] = (rank_dim, active_rank)
    return rank_slices


def _slice_lora_tensor_to_rank(tensor: torch.Tensor, rank_dim: int, active_rank: int) -> torch.Tensor:
    dim = rank_dim if rank_dim >= 0 else tensor.dim() + rank_dim
    if dim < 0 or dim >= tensor.dim() or tensor.shape[dim] <= active_rank:
        return tensor
    return tensor.narrow(dim, 0, active_rank).contiguous()


def slice_lora_state_dict_to_active_rank(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Slice LoRA state tensors to each module's active runtime rank."""
    rank_slices = _active_lora_rank_slices(model)
    if not rank_slices:
        return lora_state_dict

    sliced_state_dict: Dict[str, torch.Tensor] = {}
    for name, tensor in lora_state_dict.items():
        rank_slice = rank_slices.get(name)
        if rank_slice is None:
            sliced_state_dict[name] = tensor
            continue
        rank_dim, active_rank = rank_slice
        sliced_state_dict[name] = _slice_lora_tensor_to_rank(tensor, rank_dim, active_rank)
    return sliced_state_dict


def _first_active_lora_config(model: nn.Module) -> Tuple[Optional[int], Optional[int]]:
    """Return the first live module's active rank/alpha, if present."""
    for module in model.modules():
        active_rank = getattr(module, "active_r", None)
        active_alpha = getattr(module, "active_lora_alpha", None)
        if active_rank is not None:
            return int(active_rank), None if active_alpha is None else int(active_alpha)
    return None, None


def get_lora_state_dict(
    model: nn.Module,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from model, handling FSDP2 DTensors and EP shards.

    - DTensor.full_tensor() for FSDP2 all-gather
    - dist.all_gather across EP ranks for sharded expert params (no-op without EP)

    This is a collective operation — ALL ranks must call it when FSDP2/EP is active.

    Args:
        model: Model with LoRA layers (may be wrapped by FSDP2 and/or EP)
        prefix: Optional prefix to add to keys

    Returns:
        State dict containing only lora_A and lora_B parameters (on CPU)
    """

    fqn2spec_info = getattr(model, "_fqn2spec_info", None)
    rank_slices = _active_lora_rank_slices(model)

    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        key = f"{prefix}{name}" if prefix else name
        tensor = param.detach()

        # Step 1: FSDP2 DTensor -> full tensor
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()

        # Step 2: EP gather if applicable
        if fqn2spec_info is not None:
            clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
            spec_info = fqn2spec_info.get(clean_name) or fqn2spec_info.get(name)
            if spec_info is not None:
                tensor = _gather_ep_tensor(tensor, spec_info)

        rank_slice = rank_slices.get(name)
        if rank_slice is not None:
            rank_dim, active_rank = rank_slice
            tensor = _slice_lora_tensor_to_rank(tensor, rank_dim, active_rank)

        lora_state_dict[key] = tensor.cpu()
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load LoRA weights into model.

    Args:
        model: Model with LoRA layers
        state_dict: State dict with LoRA weights
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    lora_params = {name: param for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name}
    model_lora_keys = set(lora_params)

    # Check for missing/unexpected
    missing_keys = []
    unexpected_keys = []

    for key in model_lora_keys:
        if key not in state_dict:
            missing_keys.append(key)

    for key in state_dict.keys():
        if key not in model_lora_keys:
            unexpected_keys.append(key)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Error loading LoRA state dict.\nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
        )

    fqn2spec_info = getattr(model, "_fqn2spec_info", None)

    def _get_spec_info(name: str):
        if fqn2spec_info is None:
            return None
        clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
        return fqn2spec_info.get(clean_name) or fqn2spec_info.get(name)

    def _slice_ep_tensor_if_needed(name: str, tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        spec_info = _get_spec_info(name)
        if spec_info is None or not isinstance(spec_info.placement, Shard):
            return tensor

        shard_dim = spec_info.placement.dim
        target_shape = tuple(param.shape)
        local_size = target_shape[shard_dim]
        if tensor.shape[shard_dim] == local_size:
            return tensor

        ep_size = spec_info.ep_mesh.size()
        if tensor.shape[shard_dim] != local_size * ep_size:
            raise RuntimeError(
                f"LoRA tensor for {name} has incompatible EP shape {tuple(tensor.shape)}; "
                f"expected local {target_shape} or full dim {local_size * ep_size} on axis {shard_dim}"
            )

        try:
            ep_rank = spec_info.ep_mesh.get_local_rank()
        except Exception:
            from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

            ep_rank = get_parallel_state().ep_rank
        return tensor.narrow(shard_dim, ep_rank * local_size, local_size).contiguous()

    # Load matching keys. DTensor parameters need a DTensor source with the
    # same mesh/placements; copying a regular Tensor into them raises mixed
    # Tensor/DTensor dispatch errors.
    with torch.no_grad():
        for key, tensor in state_dict.items():
            if key not in model_lora_keys:
                continue
            param = lora_params[key]
            tensor = _slice_ep_tensor_if_needed(key, tensor, param)
            if isinstance(param, DTensor):
                tensor = tensor.to(device=param.device, dtype=param.dtype)
                dtensor = distribute_tensor(tensor, device_mesh=param.device_mesh, placements=param.placements)
                param.copy_(dtensor)
            else:
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    logger.info(f"Loaded {len(state_dict) - len(unexpected_keys)} LoRA parameters")

    return missing_keys, unexpected_keys


def _convert_from_peft_lora_key(name: str) -> str:
    """Convert a PEFT LoRA key back to xorl's internal naming."""
    if "lm_head.lora_embedding_A" in name:
        return name.replace("lm_head.lora_embedding_A", "lm_head.lora_A")
    if "lm_head.lora_embedding_B" in name:
        return name.replace("lm_head.lora_embedding_B", "lm_head.lora_B")
    if name.endswith(".lora_A.weight"):
        return name[: -len(".weight")]
    if name.endswith(".lora_B.weight"):
        return name[: -len(".weight")]
    return name


def _device_mesh_local_index(mesh) -> int:
    """Return this rank's coordinate in a 1D DeviceMesh."""
    try:
        return int(mesh.get_local_rank(mesh_dim=0))
    except Exception:
        pass

    try:
        coordinate = mesh.get_coordinate()
        if coordinate is not None:
            return int(coordinate[0])
    except Exception:
        pass

    try:
        return int(dist.get_rank(mesh.get_group()))
    except Exception:
        return 0


def get_lora_tensor_shard_specs(
    model: nn.Module,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, LoraTensorShardSpec]:
    """Return EP shard specs for LoRA parameters in ``model`` keyed by parameter name."""
    requested_names = set(names) if names is not None else None
    fqn2spec_info = getattr(model, "_fqn2spec_info", None)
    shard_specs: Dict[str, LoraTensorShardSpec] = {}

    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue

        clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
        if requested_names is not None and name not in requested_names and clean_name not in requested_names:
            continue

        spec_info = None
        if fqn2spec_info is not None:
            spec_info = fqn2spec_info.get(clean_name) or fqn2spec_info.get(name)
        if spec_info is None:
            spec_info = getattr(param, "spec_info", None)
        if spec_info is None or not isinstance(spec_info.placement, Shard):
            continue

        ep_mesh = spec_info.ep_mesh
        if ep_mesh is None:
            continue

        shard_specs[name] = LoraTensorShardSpec(
            dim=spec_info.placement.dim,
            index=_device_mesh_local_index(ep_mesh),
            size=int(ep_mesh.size()),
        )
        if clean_name != name:
            shard_specs[clean_name] = shard_specs[name]

    return shard_specs


def _slice_lora_tensor_shard(
    key: str,
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    shard_specs: Optional[Dict[str, LoraTensorShardSpec]],
) -> Optional[torch.Tensor]:
    if shard_specs is None or key not in shard_specs:
        return None

    spec = shard_specs[key]
    shard_dim = spec.dim if spec.dim >= 0 else tensor.dim() + spec.dim
    if shard_dim < 0 or shard_dim >= tensor.dim():
        return None
    if len(expected_shape) != tensor.dim():
        return None
    if spec.size <= 1:
        return None

    for dim, size in enumerate(expected_shape):
        if dim != shard_dim and tensor.shape[dim] != size:
            return None

    total_size = tensor.shape[shard_dim]
    shard_size = (total_size + spec.size - 1) // spec.size
    start = spec.index * shard_size
    if start >= total_size:
        return None

    length = min(shard_size, total_size - start)
    sliced = tensor.narrow(shard_dim, start, length).contiguous()
    if tuple(sliced.shape) == expected_shape:
        return sliced

    raise RuntimeError(
        f"Converted LoRA tensor shard for {key} has shape {tuple(sliced.shape)}, "
        f"expected {expected_shape} (full shape {tuple(tensor.shape)}, "
        f"shard dim {shard_dim}, shard {spec.index}/{spec.size})"
    )


def _align_lora_tensor_shape(
    key: str,
    tensor: torch.Tensor,
    expected_shapes: Optional[Dict[str, torch.Size]],
    expected_shard_specs: Optional[Dict[str, LoraTensorShardSpec]] = None,
) -> torch.Tensor:
    """Match a converted tensor to the live LoRA shape, allowing transpose and EP slicing."""
    if expected_shapes is None or key not in expected_shapes:
        return tensor

    expected_shape = tuple(expected_shapes[key])
    if tuple(tensor.shape) == expected_shape:
        return tensor

    sliced = _slice_lora_tensor_shard(key, tensor, expected_shape, expected_shard_specs)
    if sliced is not None:
        return sliced

    if tensor.dim() >= 2:
        transposed = tensor.transpose(-2, -1).contiguous()
        if tuple(transposed.shape) == expected_shape:
            return transposed

        sliced = _slice_lora_tensor_shard(key, transposed, expected_shape, expected_shard_specs)
        if sliced is not None:
            return sliced

    if expected_shard_specs is not None and key in expected_shard_specs:
        spec = expected_shard_specs[key]
        raise RuntimeError(
            f"Converted LoRA tensor for {key} has shape {tuple(tensor.shape)}, expected {expected_shape} "
            f"(shard dim {spec.dim}, shard {spec.index}/{spec.size})"
        )
    raise RuntimeError(f"Converted LoRA tensor for {key} has shape {tuple(tensor.shape)}, expected {expected_shape}")


def convert_peft_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    expected_shapes: Optional[Dict[str, torch.Size]] = None,
    expected_shard_specs: Optional[Dict[str, LoraTensorShardSpec]] = None,
) -> Dict[str, torch.Tensor]:
    """Convert PEFT-style LoRA checkpoint tensors into xorl's internal layout.

    This handles:
    - Dense LoRA keys ending in ``.lora_{A|B}.weight``
    - ``lm_head`` PEFT embedding aliases
    - MoE per-expert keys such as ``...experts.3.gate_proj.lora_A.weight``
    - MoE hybrid-shared keys such as ``...experts.shared.down_proj.lora_B.weight``

    Args:
        state_dict: Loaded checkpoint weights.
        expected_shapes: Optional live-parameter shapes keyed by internal name.
            When provided, the converter will transpose the trailing matrix dims
            if that is required to match the live layout.
        expected_shard_specs: Optional EP shard specs keyed by internal name.
            When provided, full MoE expert tensors are sliced to the current
            rank's local expert shard after any required PEFT transpose.

    Returns:
        State dict keyed by xorl's internal LoRA parameter names.
    """
    converted_state_dict: Dict[str, torch.Tensor] = {}
    moe_buckets: Dict[str, Dict[str, torch.Tensor]] = {}

    for raw_key, value in state_dict.items():
        if raw_key.startswith(_PEFT_BASE_MODEL_PREFIX):
            key = raw_key[len(_PEFT_BASE_MODEL_PREFIX) :]
        else:
            key = raw_key

        sglang_match = _MOE_SGLANG_SHARED_OUTER_PATTERN.match(key)
        if sglang_match is not None:
            prefix, w_slot, lora_type = sglang_match.groups()
            proj_name = _SGLANG_W_TO_PROJ[w_slot]
            internal_name = f"{prefix}.mlp.experts.{proj_name}_lora_{lora_type}"
            # shared_outer stores 3D tensors transposed (last two dims) vs.
            # xorl's in-memory layout. Flip them back to the in-first order.
            restored = value.transpose(-2, -1).contiguous() if value.dim() >= 2 else value
            converted_state_dict[internal_name] = _align_lora_tensor_shape(
                internal_name, restored, expected_shapes, expected_shard_specs
            )
            continue

        match = _MOE_PEFT_LORA_PATTERN.match(key)
        if match is not None:
            prefix, expert_token, proj_name, lora_type = match.groups()
            internal_name = f"{prefix}.mlp.experts.{proj_name}_lora_{lora_type}"
            bucket = moe_buckets.setdefault(internal_name, {})
            if expert_token in bucket:
                raise RuntimeError(f"Duplicate MoE LoRA checkpoint entry for {internal_name} ({expert_token})")
            bucket[expert_token] = value
            continue

        internal_name = _convert_from_peft_lora_key(key)
        converted_state_dict[internal_name] = _align_lora_tensor_shape(
            internal_name, value, expected_shapes, expected_shard_specs
        )

    for internal_name, bucket in moe_buckets.items():
        if "shared" in bucket:
            if len(bucket) != 1:
                raise RuntimeError(
                    f"Mixed shared and per-expert MoE checkpoint entries found for {internal_name}: {sorted(bucket)}"
                )
            stacked = bucket["shared"].unsqueeze(0).contiguous()
        else:
            expert_indices = sorted(int(idx) for idx in bucket)
            expected_indices = list(range(len(expert_indices)))
            if expert_indices != expected_indices:
                raise RuntimeError(
                    f"Non-contiguous MoE checkpoint expert indices for {internal_name}: {expert_indices}"
                )
            stacked = torch.stack([bucket[str(expert_idx)] for expert_idx in expert_indices], dim=0)

        converted_state_dict[internal_name] = _align_lora_tensor_shape(
            internal_name, stacked, expected_shapes, expected_shard_specs
        )

    return converted_state_dict


def save_lora_checkpoint(
    model: nn.Module,
    save_path: str,
    base_model_name: Optional[str] = None,
    target_modules: Optional[List[str]] = None,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    moe_hybrid_shared_lora: bool = False,
    lora_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    transpose_moe_lora_to_peft: bool = True,
    lora_export_format: str = "peft",
    preserve_lora_dtype: bool = False,
) -> str:
    """
    Save LoRA weights in PEFT-compatible format.

    Creates:
        - adapter_model.safetensors: LoRA weights
        - adapter_config.json: Configuration for PEFT compatibility

    Args:
        model: Model with LoRA layers
        save_path: Directory to save checkpoint
        base_model_name: Name of base model (for adapter_config.json)
        target_modules: List of target modules (auto-detected if None)
        r: LoRA rank (auto-detected if None)
        lora_alpha: LoRA alpha (auto-detected if None)
        moe_hybrid_shared_lora: Whether hybrid shared LoRA is used
        lora_state_dict: Pre-gathered LoRA state dict. If provided, skips
            get_lora_state_dict(model) call. Useful when the caller has already
            gathered weights (e.g., from FSDP2 + EP distributed model).
        transpose_moe_lora_to_peft: Whether to transpose MoE expert LoRA tensors
            into PEFT/SGLang orientation during export. Enabled by default so
            exported MoE adapters match the shape convention expected by
            inference backends. Ignored when
            ``lora_export_format="sglang_shared_outer"``.
        lora_export_format: On-disk layout for MoE expert LoRA. ``"peft"``
            (default) un-stacks the 3D tensors into per-expert 2D keys. Pass
            ``"sglang_shared_outer"`` to emit SGLang's stacked 3D shared_outer
            layout directly (requires ``moe_hybrid_shared_lora=True``).
        preserve_lora_dtype: Keep LoRA tensor dtypes in the safetensors file
            instead of exporting bf16 weights. Use this for training-resume
            checkpoints; keep the default bf16 export for inference adapters.

    Returns:
        Path to saved checkpoint directory
    """

    if lora_export_format not in LORA_EXPORT_FORMATS:
        raise ValueError(f"Unknown lora_export_format={lora_export_format!r}. Expected one of {LORA_EXPORT_FORMATS}.")
    if lora_export_format == "sglang_shared_outer" and not moe_hybrid_shared_lora:
        raise ValueError(
            "lora_export_format='sglang_shared_outer' requires moe_hybrid_shared_lora=True "
            "(shared_outer only makes sense for hybrid-shared MoE LoRA)."
        )

    os.makedirs(save_path, exist_ok=True)

    # Get LoRA state dict — use provided one or extract from model
    if lora_state_dict is None:
        lora_state_dict = get_lora_state_dict(model)
    else:
        lora_state_dict = slice_lora_state_dict_to_active_rank(model, lora_state_dict)

    def _prepare_lora_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().cpu().contiguous()
        if preserve_lora_dtype:
            return tensor
        return tensor.to(torch.bfloat16)

    def _is_moe_lora_param(name: str) -> bool:
        """Check if this is a stacked MoE LoRA parameter."""
        return _MOE_LORA_PATTERN.match(name) is not None

    def _convert_to_peft_key(name: str) -> str:
        """
        Convert xorl LoRA key to PEFT-compatible format.

        For Linear layers: lora_A -> lora_A.weight, lora_B -> lora_B.weight
        For lm_head: lora_A -> lora_embedding_A, lora_B -> lora_embedding_B
        """
        if "lm_head.lora_A" in name:
            return name.replace("lm_head.lora_A", "lm_head.lora_embedding_A")
        elif "lm_head.lora_B" in name:
            return name.replace("lm_head.lora_B", "lm_head.lora_embedding_B")
        elif name.endswith(".lora_A"):
            return name + ".weight"
        elif name.endswith(".lora_B"):
            return name + ".weight"
        return name

    def _unmerge_moe_lora_weights(name: str, stacked_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Unmerge stacked MoE LoRA weights into per-expert format for vLLM compatibility.

        Xorl stores MoE LoRA as stacked tensors:
            model.layers.0.mlp.experts.gate_proj_lora_A  # shape: [num_experts, hidden_dim, r]
            model.layers.0.mlp.experts.gate_proj_lora_B  # shape: [num_experts, r, intermediate_dim]

        vLLM expects per-expert format:
            model.layers.0.mlp.experts.0.gate_proj.lora_A.weight  # shape: [r, hidden_dim]
            model.layers.0.mlp.experts.0.gate_proj.lora_B.weight  # shape: [intermediate_dim, r]

        For hybrid shared LoRA, shared weights (shape[0] == 1) are named with
        ".shared." instead of expert index to indicate they're shared across experts.
        """
        match = _MOE_LORA_PATTERN.match(name)
        if not match:
            raise ValueError(f"Invalid MoE LoRA parameter name: {name}")

        prefix = match.group(1)  # e.g., "model.layers.0"
        proj_name = match.group(2)  # e.g., "gate_proj"
        lora_type = match.group(3)  # "A" or "B"

        num_experts = stacked_tensor.shape[0]
        result = {}

        # Check if this is a shared weight (hybrid shared LoRA)
        is_shared = num_experts == 1

        if is_shared:
            # Shared weight: use ".shared." in the key name
            expert_tensor = stacked_tensor[0]
            if transpose_moe_lora_to_peft:
                expert_tensor = expert_tensor.transpose(0, 1).contiguous()
            peft_key = f"base_model.model.{prefix}.mlp.experts.shared.{proj_name}.lora_{lora_type}.weight"
            result[peft_key] = _prepare_lora_tensor(expert_tensor)
        else:
            # Per-expert weights: use expert index in the key name
            for expert_idx in range(num_experts):
                expert_tensor = stacked_tensor[expert_idx]
                if transpose_moe_lora_to_peft:
                    expert_tensor = expert_tensor.transpose(0, 1).contiguous()
                # Build vLLM-compatible key:
                # base_model.model.{prefix}.mlp.experts.{idx}.{proj}.lora_{A|B}.weight
                peft_key = f"base_model.model.{prefix}.mlp.experts.{expert_idx}.{proj_name}.lora_{lora_type}.weight"
                result[peft_key] = _prepare_lora_tensor(expert_tensor)

        return result

    def _sglang_shared_outer_moe_weight(name: str, stacked_tensor: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Rename + transpose a stacked MoE tensor into SGLang shared_outer layout.

        xorl stores 3D tensors in-first (A: [E_or_1, in, r], B: [E_or_1, r, out]).
        shared_outer stores them out-first under ``experts.w{1,2,3}.lora_{A|B}.weight``.
        """
        match = _MOE_LORA_PATTERN.match(name)
        if not match:
            raise ValueError(f"Invalid MoE LoRA parameter name: {name}")
        prefix, proj_name, lora_type = match.group(1), match.group(2), match.group(3)
        w_slot = _PROJ_TO_SGLANG_W[proj_name]
        peft_key = f"{_PEFT_BASE_MODEL_PREFIX}{prefix}.mlp.experts.{w_slot}.lora_{lora_type}.weight"
        out_tensor = _prepare_lora_tensor(stacked_tensor.transpose(-2, -1).contiguous())
        return peft_key, out_tensor

    # Convert keys to PEFT format: base_model.model.{converted_key}
    peft_state_dict = {}
    detected_modules = set()
    detected_r = None

    for key, value in lora_state_dict.items():
        # Check if this is a stacked MoE LoRA parameter
        if _is_moe_lora_param(key):
            if lora_export_format == "sglang_shared_outer":
                peft_key, out_tensor = _sglang_shared_outer_moe_weight(key, value)
                peft_state_dict[peft_key] = out_tensor
            else:
                # Unmerge stacked MoE LoRA weights into per-expert format
                per_expert_weights = _unmerge_moe_lora_weights(key, value)
                peft_state_dict.update(per_expert_weights)
            # Detect target modules from MoE LoRA
            match = _MOE_LORA_PATTERN.match(key)
            if match:
                detected_modules.add(match.group(2))  # gate_proj, up_proj, or down_proj
                if detected_r is None and match.group(3) == "A":
                    # Xorl stores MoE LoRA A as [num_experts, in_features, r].
                    detected_r = value.shape[2]
        else:
            # Extract module name for target_modules detection
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in ["lora_A", "lora_B"]:
                    if i > 0:
                        detected_modules.add(parts[i - 1])
                    # Detect r and alpha from first LoRA layer found
                    if detected_r is None and part == "lora_A":
                        detected_r = value.shape[0]
                    break

            # Convert to PEFT key format
            peft_key = f"base_model.model.{_convert_to_peft_key(key)}"
            peft_state_dict[peft_key] = _prepare_lora_tensor(value)

    # Auto-detect parameters if not provided
    if target_modules is None:
        target_modules = list(detected_modules)
    active_rank, active_alpha = _first_active_lora_config(model)
    if detected_r is not None:
        if r is not None and r != detected_r:
            logger.warning(
                f"Requested LoRA config r={r} does not match exported tensor rank {detected_r}; writing r={detected_r}."
            )
        r = detected_r
    elif r is None:
        r = active_rank or 16
    if active_alpha is not None and active_rank == r:
        if lora_alpha is not None and lora_alpha != active_alpha:
            logger.warning(
                f"Requested LoRA alpha={lora_alpha} does not match active alpha {active_alpha}; "
                f"writing lora_alpha={active_alpha}."
            )
        lora_alpha = active_alpha
    elif lora_alpha is None:
        # Try to detect from model
        for module in model.modules():
            if isinstance(module, LoraLinear):
                lora_alpha = getattr(module, "active_lora_alpha", module.lora_alpha)
                break
        if lora_alpha is None:
            lora_alpha = r  # Default: alpha = r

    # Save weights
    weights_path = os.path.join(save_path, "adapter_model.safetensors")
    save_file(peft_state_dict, weights_path)
    logger.info(f"Saved LoRA weights to {weights_path}")

    # Create adapter_config.json for PEFT compatibility
    adapter_config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name or "",
        "peft_type": "LORA",
        "inference_mode": True,
        "fan_in_fan_out": False,
    }
    if lora_export_format == "sglang_shared_outer":
        adapter_config["_sglang_lora_format"] = "shared_outer"
        # SGLang's lora_manager classifies adapters via the moe_hybrid_shared_lora
        # / shared_moe_lora keys in hf_config. shared_outer IS hybrid_shared
        # on-disk, so mirror the flag here so SGLang doesn't mis-classify it as
        # per_expert (which would reject loading under --lora-moe-format hybrid_shared).
        adapter_config["moe_hybrid_shared_lora"] = True
    else:
        adapter_config["moe_hybrid_shared_lora"] = moe_hybrid_shared_lora

    config_path = os.path.join(save_path, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    logger.info(f"Saved adapter config to {config_path}")

    return save_path


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """
    Load LoRA weights from checkpoint.

    Supports both xorl format and PEFT format checkpoints for dense
    (nn.Linear) LoRA as well as stacked MoE LoRA, including hybrid-shared
    checkpoints exported under ``.mlp.experts.shared.{proj}.lora_{A|B}.weight``.

    Args:
        model: Model with LoRA layers already injected
        checkpoint_path: Path to checkpoint directory or file
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Model with loaded LoRA weights
    """
    # Determine checkpoint file
    if os.path.isdir(checkpoint_path):
        weights_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(checkpoint_path, "adapter_model.bin")
    else:
        weights_file = checkpoint_path

    # Load weights
    if weights_file.endswith(".safetensors"):
        state_dict = load_file(weights_file)
    else:
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

    model_lora_shapes = {
        key: value.shape for key, value in model.state_dict().items() if "lora_A" in key or "lora_B" in key
    }
    lora_shard_specs = get_lora_tensor_shard_specs(model, names=model_lora_shapes.keys())
    converted_state_dict = convert_peft_lora_state_dict(
        state_dict,
        expected_shapes=model_lora_shapes,
        expected_shard_specs=lora_shard_specs,
    )

    # Load into model
    missing, unexpected = load_lora_state_dict(model, converted_state_dict, strict=strict)

    if missing:
        logger.warning(f"Missing keys when loading LoRA: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading LoRA: {unexpected}")

    return model


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count trainable LoRA parameters vs total parameters.

    Args:
        model: Model with LoRA layers

    Returns:
        Tuple of (trainable_params, total_params, percentage)
    """
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    return trainable_params, total_params, percentage


def inject_lora_into_moe_blocks(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    hybrid_shared: bool = False,
) -> int:
    """
    Inject LoRA adapters into MoE blocks.

    This function finds all MoE block instances in the model that support
    LoRA injection (MoEBlock) and injects LoRA adapters into their expert
    weights.

    Supported MoE implementations:
    - moe_implementation='triton': Uses Triton group GEMM kernels with LoRA
    - moe_implementation='native': Uses torch._grouped_mm with LoRA
    - moe_implementation='eager': Uses per-expert loop with LoRA

    Args:
        model: Model containing MoE blocks
        r: LoRA rank
        lora_alpha: LoRA alpha for scaling
        target_modules: Which expert projections to apply LoRA to.
                       Options: ["gate_proj", "up_proj", "down_proj"]
                       Default: all three projections
        hybrid_shared: If True, use hybrid sharing (lora_A shared for gate/up, lora_B shared for down)

    Returns:
        Number of MoE blocks that received LoRA adapters

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B")
        >>> num_blocks = inject_lora_into_moe_blocks(
        ...     model,
        ...     r=16,
        ...     lora_alpha=32,
        ...     target_modules=["gate_proj", "up_proj", "down_proj"]
        ... )
        >>> print(f"Injected LoRA into {num_blocks} MoE blocks")
    """
    if target_modules is None:
        target_modules = ["gate_proj", "up_proj", "down_proj"]

    injected_count = 0

    for name, module in model.named_modules():
        # Check if this is a MoE block that supports LoRA injection
        if hasattr(module, "inject_lora") and hasattr(module, "lora_adapter"):
            # This is a MoEBlock or similar
            module.inject_lora(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                hybrid_shared=hybrid_shared,
            )
            injected_count += 1
            logger.debug(f"Injected MoE LoRA into {name}")

    if injected_count > 0:
        logger.info(
            f"Injected LoRA into {injected_count} MoE blocks with r={r}, "
            f"alpha={lora_alpha}, hybrid_shared={hybrid_shared}"
        )
    else:
        logger.warning(
            "No LoRA-compatible MoE blocks found. Make sure model uses moe_implementation='triton' or 'native'"
        )

    return injected_count


def inject_lora_into_model_with_moe(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    moe_hybrid_shared_lora: bool = False,
) -> nn.Module:
    """
    Inject LoRA adapters into both dense layers and MoE expert blocks.

    This is a comprehensive LoRA injection function that handles:
    1. Standard nn.Linear layers (attention projections, dense MLP layers)
    2. MoE expert blocks (MoEBlock with triton/native/quack backends)

    For MoE models like Qwen3 MoE, some layers have dense MLP (nn.Linear modules)
    and others have MoE blocks. This function handles both cases:
    - Dense MLP layers get LoRA via inject_lora_into_model (replaces nn.Linear with LoraLinear)
    - MoE blocks get LoRA via inject_lora_into_moe_blocks (uses group GEMM with LoRA)

    Args:
        model: Model to inject LoRA into
        r: LoRA rank
        lora_alpha: LoRA alpha for scaling
        target_modules: List of module names to target. If None, falls back to
                       the architecture default in DEFAULT_TARGET_MODULES.
        moe_hybrid_shared_lora: If True, use hybrid sharing (lora_A shared for gate/up, lora_B shared for down)

    Returns:
        The model with LoRA layers injected (modified in-place)

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B")
        >>> inject_lora_into_model_with_moe(
        ...     model,
        ...     r=16,
        ...     lora_alpha=32,
        ...     target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        ... )
    """
    if target_modules is None:
        target_modules = _get_default_target_modules(model)

    # Partition target_modules into expert MLP projections (handled by the
    # group-GEMM MoE path) and dense linears (attention plus per-layer dense
    # MLPs, handled by inject_lora_into_model). Anything that isn't an expert
    # name is treated as a dense target so MLA projections such as
    # q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj are not silently
    # dropped by a Llama/Qwen-shaped allowlist.
    expert_modules = [m for m in target_modules if m in ("gate_proj", "up_proj", "down_proj")]
    attention_modules = [m for m in target_modules if m not in expert_modules]

    # Step 1: Inject LoRA into standard nn.Linear layers
    # This includes attention projections AND dense MLP layers (for layers without MoE)
    # Note: inject_lora_into_model only affects nn.Linear modules, so it won't
    # affect MoE expert blocks which have stacked weight tensors
    all_linear_targets = attention_modules + expert_modules
    if all_linear_targets:
        inject_lora_into_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=all_linear_targets,
        )

    # Step 2: Inject LoRA into MoE expert blocks
    # This handles MoE layers that have stacked expert weights (not nn.Linear)
    if expert_modules:
        inject_lora_into_moe_blocks(
            model,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=expert_modules,
            hybrid_shared=moe_hybrid_shared_lora,
        )

    return model


def get_moe_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from MoE expert blocks.

    Args:
        model: Model with MoE LoRA adapters

    Returns:
        State dict containing MoE LoRA parameters
    """
    moe_lora_state_dict = {}
    rank_slices = _active_lora_rank_slices(model)

    for name, module in model.named_modules():
        # Check if this is an MoE LoRA expert module (has lora_config attribute)
        if hasattr(module, "lora_config") and module.lora_config is not None:
            # Get all lora_ parameters from this module
            for param_name, param in module.named_parameters():
                if "lora_" in param_name:
                    full_key = f"{name}.{param_name}"
                    tensor = param.detach()
                    rank_slice = rank_slices.get(full_key)
                    if rank_slice is not None:
                        rank_dim, active_rank = rank_slice
                        tensor = _slice_lora_tensor_to_rank(tensor, rank_dim, active_rank)
                    moe_lora_state_dict[full_key] = tensor.cpu()

    return moe_lora_state_dict


def get_all_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all LoRA weights (both dense and MoE) from model.

    Args:
        model: Model with LoRA layers

    Returns:
        Combined state dict with all LoRA parameters
    """
    # Get standard LoRA weights
    lora_state = get_lora_state_dict(model)

    # Get MoE LoRA weights
    moe_lora_state = get_moe_lora_state_dict(model)

    # Combine
    lora_state.update(moe_lora_state)

    return lora_state


def maybe_merge_lora(model: nn.Module) -> int:
    """Merge LoRA deltas into base weights and reset LoRA params for all LoRA modules.

    Call this between optimizer.step() and the next forward pass.
    For normal LoRA (bf16 base weights), this merges delta into weight directly
    and resets LoRA parameters (kaiming for A, zeros for B) so training continues.

    Args:
        model: Model with LoRA modules (LoraLinear, MoEExpertsLoRA, etc.).

    Returns:
        Number of modules merged.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoraModule):
            module.merge_weights()
            count += 1
    if count > 0:
        logger.info(f"Merged LoRA into {count} modules (delta absorbed into base weights)")
    return count
