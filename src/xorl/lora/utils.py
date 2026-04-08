"""
LoRA utility functions.

Functions for injecting LoRA into models and managing LoRA state dicts.
"""

import json
import logging
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed._tensor import DTensor, Replicate, Shard

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
}


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
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

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
    model_state = model.state_dict()

    # Find LoRA keys in model
    model_lora_keys = {k for k in model_state.keys() if "lora_A" in k or "lora_B" in k}

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

    # Load matching keys
    for key in state_dict.keys():
        if key in model_lora_keys:
            model_state[key].copy_(state_dict[key])

    logger.info(f"Loaded {len(state_dict) - len(unexpected_keys)} LoRA parameters")

    return missing_keys, unexpected_keys


def save_lora_checkpoint(
    model: nn.Module,
    save_path: str,
    base_model_name: Optional[str] = None,
    target_modules: Optional[List[str]] = None,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    moe_hybrid_shared_lora: bool = False,
    lora_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    transpose_moe_lora_to_peft: bool = False,
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
            into PEFT/vLLM orientation during export. Disabled by default so
            existing xorl call sites keep their current behavior.

    Returns:
        Path to saved checkpoint directory
    """

    os.makedirs(save_path, exist_ok=True)

    # Get LoRA state dict — use provided one or extract from model
    if lora_state_dict is None:
        lora_state_dict = get_lora_state_dict(model)

    # Pattern to detect MoE LoRA weights: mlp.experts.{proj}_lora_{A|B}
    # These are stacked tensors with shape [num_experts, ...] that need to be unmerged
    moe_lora_pattern = re.compile(r"(.*)\.mlp\.experts\.(gate_proj|up_proj|down_proj)_lora_(A|B)$")

    def _is_moe_lora_param(name: str) -> bool:
        """Check if this is a stacked MoE LoRA parameter."""
        return moe_lora_pattern.match(name) is not None

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
        match = moe_lora_pattern.match(name)
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
            result[peft_key] = expert_tensor.to(torch.bfloat16)
        else:
            # Per-expert weights: use expert index in the key name
            for expert_idx in range(num_experts):
                expert_tensor = stacked_tensor[expert_idx]
                if transpose_moe_lora_to_peft:
                    expert_tensor = expert_tensor.transpose(0, 1).contiguous()
                # Build vLLM-compatible key:
                # base_model.model.{prefix}.mlp.experts.{idx}.{proj}.lora_{A|B}.weight
                peft_key = f"base_model.model.{prefix}.mlp.experts.{expert_idx}.{proj_name}.lora_{lora_type}.weight"
                result[peft_key] = expert_tensor.to(torch.bfloat16)

        return result

    # Convert keys to PEFT format: base_model.model.{converted_key}
    peft_state_dict = {}
    detected_modules = set()
    detected_r = None
    detected_alpha = None

    for key, value in lora_state_dict.items():
        # Check if this is a stacked MoE LoRA parameter
        if _is_moe_lora_param(key):
            # Unmerge stacked MoE LoRA weights into per-expert format
            per_expert_weights = _unmerge_moe_lora_weights(key, value)
            peft_state_dict.update(per_expert_weights)
            # Detect target modules from MoE LoRA
            match = moe_lora_pattern.match(key)
            if match:
                detected_modules.add(match.group(2))  # gate_proj, up_proj, or down_proj
                if detected_r is None and match.group(3) == "A":
                    # Xorl stores MoE LoRA A as [num_experts, in_features, r].
                    # When transpose_moe_lora_to_peft is enabled, the exported
                    # PEFT tensor rank is the last dimension of the stacked input.
                    detected_r = value.shape[2] if transpose_moe_lora_to_peft else value.shape[1]
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
            peft_state_dict[peft_key] = value.to(torch.bfloat16)

    # Auto-detect parameters if not provided
    if target_modules is None:
        target_modules = list(detected_modules)
    if r is None:
        r = detected_r or 16
    if lora_alpha is None:
        # Try to detect from model
        for module in model.modules():
            if isinstance(module, LoraLinear):
                lora_alpha = module.lora_alpha
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
        "moe_hybrid_shared_lora": moe_hybrid_shared_lora,
    }

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

    Supports both xorl format and PEFT format checkpoints.

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

    def _convert_from_peft_key(name: str) -> str:
        """
        Convert PEFT key format back to xorl format.

        For Linear layers: lora_A.weight -> lora_A, lora_B.weight -> lora_B
        For lm_head: lora_embedding_A -> lora_A, lora_embedding_B -> lora_B
        """
        # Handle lm_head embedding-style naming
        if "lm_head.lora_embedding_A" in name:
            return name.replace("lm_head.lora_embedding_A", "lm_head.lora_A")
        elif "lm_head.lora_embedding_B" in name:
            return name.replace("lm_head.lora_embedding_B", "lm_head.lora_B")
        # Handle .weight suffix for Linear layers
        elif name.endswith(".lora_A.weight"):
            return name[: -len(".weight")]
        elif name.endswith(".lora_B.weight"):
            return name[: -len(".weight")]
        return name

    # Convert PEFT format keys to xorl format if needed
    converted_state_dict = {}
    for key, value in state_dict.items():
        # Remove PEFT prefix: base_model.model.{key} -> {key}
        if key.startswith("base_model.model."):
            new_key = key[len("base_model.model.") :]
        else:
            new_key = key
        # Convert from PEFT naming to xorl naming
        new_key = _convert_from_peft_key(new_key)
        converted_state_dict[new_key] = value

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
    shared_lora: bool = False,
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
        shared_lora: If True, share LoRA across all experts (more parameter efficient)
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
                shared_lora=shared_lora,
                target_modules=target_modules,
                hybrid_shared=hybrid_shared,
            )
            injected_count += 1
            logger.debug(f"Injected MoE LoRA into {name}")

    if injected_count > 0:
        logger.info(
            f"Injected LoRA into {injected_count} MoE blocks with r={r}, "
            f"alpha={lora_alpha}, shared={shared_lora}, hybrid_shared={hybrid_shared}"
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
    moe_shared_lora: bool = False,
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
        target_modules: List of module names to target.
                       For attention: ["q_proj", "k_proj", "v_proj", "o_proj"]
                       For MLP/experts: ["gate_proj", "up_proj", "down_proj"]
        moe_shared_lora: If True, MoE experts share LoRA adapters (all shared)
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
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Separate attention modules from MLP/expert modules
    attention_modules = [m for m in target_modules if m in ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]]
    expert_modules = [m for m in target_modules if m in ["gate_proj", "up_proj", "down_proj"]]

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
            shared_lora=moe_shared_lora,
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

    for name, module in model.named_modules():
        # Check if this is an MoE LoRA expert module (has lora_config attribute)
        if hasattr(module, "lora_config") and module.lora_config is not None:
            # Get all lora_ parameters from this module
            for param_name, param in module.named_parameters():
                if "lora_" in param_name:
                    full_key = f"{name}.{param_name}"
                    moe_lora_state_dict[full_key] = param.detach().cpu()

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
