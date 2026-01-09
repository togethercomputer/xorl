"""
LoRA utility functions.

Functions for injecting LoRA into models and managing LoRA state dicts.
"""

import json
import logging
import os
import re
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from xorl.lora.layers import LoraLinear

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

    Args:
        model: Model to search
        target_modules: List of module name patterns to match

    Returns:
        List of full module paths that match
    """
    matched_paths = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if the module name ends with any target pattern
            module_name = name.split(".")[-1]
            if module_name in target_modules:
                matched_paths.append(name)

    return matched_paths


def inject_lora_into_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Inject LoRA adapters into a model by replacing target Linear layers.

    This function finds all nn.Linear layers matching the target patterns
    and replaces them with LoraLinear layers, copying the original weights.

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
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Find all matching modules
    target_paths = _find_target_modules(model, target_modules)

    if not target_paths:
        logger.warning(
            f"No modules found matching target_modules={target_modules}. "
            "LoRA injection skipped."
        )
        return model

    logger.info(f"Injecting LoRA into {len(target_paths)} modules with r={r}, alpha={lora_alpha}")

    # Replace each target module
    replaced_count = 0
    for target_path in target_paths:
        parent, attr_name = _get_submodule(model, target_path)
        original_layer = getattr(parent, attr_name)

        if not isinstance(original_layer, nn.Linear):
            logger.warning(f"Skipping {target_path}: not an nn.Linear")
            continue

        # Create LoraLinear from original
        lora_layer = LoraLinear.from_linear(
            original_layer,
            r=r,
            lora_alpha=lora_alpha,
        )

        # Replace in parent
        setattr(parent, attr_name, lora_layer)
        replaced_count += 1

        logger.debug(f"Replaced {target_path} with LoraLinear")

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


def get_lora_state_dict(
    model: nn.Module,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA weights from model state dict.

    Args:
        model: Model with LoRA layers
        prefix: Optional prefix to add to keys

    Returns:
        State dict containing only lora_A and lora_B parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            key = f"{prefix}{name}" if prefix else name
            lora_state_dict[key] = param.detach().cpu()
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
            f"Error loading LoRA state dict.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
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

    Returns:
        Path to saved checkpoint directory
    """
    os.makedirs(save_path, exist_ok=True)

    # Get LoRA state dict and convert to PEFT format
    lora_state_dict = get_lora_state_dict(model)

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

    # Convert keys to PEFT format: base_model.model.{converted_key}
    peft_state_dict = {}
    detected_modules = set()
    detected_r = None
    detected_alpha = None

    for key, value in lora_state_dict.items():
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
        peft_state_dict[peft_key] = value

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
            return name[:-len(".weight")]
        elif name.endswith(".lora_B.weight"):
            return name[:-len(".weight")]
        return name

    # Convert PEFT format keys to xorl format if needed
    converted_state_dict = {}
    for key, value in state_dict.items():
        # Remove PEFT prefix: base_model.model.{key} -> {key}
        if key.startswith("base_model.model."):
            new_key = key[len("base_model.model."):]
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
