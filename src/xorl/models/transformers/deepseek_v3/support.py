"""DeepseekV3 / Kimi-K2.5 support helpers."""

import json
import os
from typing import Iterable, Optional, Tuple

from transformers.utils import cached_file

from xorl.distributed.parallel_state import get_parallel_state


PACKED_EXPERT_DEFAULT_NUM_BITS = 4
PACKED_EXPERT_DEFAULT_GROUP_SIZE = 32


DEEPSEEK_V3_LORA_TARGET_MODULES = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def is_deepseek_v3_config(config) -> bool:
    return getattr(config, "model_type", None) == "deepseek_v3"


def validate_deepseek_v3_router_settings(config, *, train_router: bool) -> None:
    if not is_deepseek_v3_config(config):
        return
    if train_router:
        raise ValueError("DeepseekV3/Kimi-K2.5 does not support train_router=True in xorl yet.")


def validate_deepseek_v3_training_mode(
    config,
    *,
    enable_qlora: bool,
    freeze_router: bool,
    merge_qkv: bool,
) -> None:
    if not is_deepseek_v3_config(config):
        return
    if enable_qlora:
        raise ValueError("DeepseekV3/Kimi-K2.5 does not support enable_qlora=True yet.")
    if not freeze_router:
        raise ValueError("DeepseekV3/Kimi-K2.5 requires freeze_router=True.")
    if not merge_qkv:
        raise ValueError("DeepseekV3/Kimi-K2.5 does not support merge_qkv=False yet.")


def validate_deepseek_v3_tensor_parallelism(config) -> None:
    if not is_deepseek_v3_config(config):
        return
    if get_parallel_state().tp_enabled:
        raise ValueError("DeepseekV3/Kimi-K2.5 tensor parallelism is not supported yet.")


def freeze_deepseek_v3_router_parameters(model) -> int:
    count = 0
    for name, param in model.named_parameters():
        if ".gate.weight" in name:
            param.requires_grad = False
            count += 1
    return count


def deepseek_v3_default_lora_targets(*, train_attn: bool, train_mlp: bool, train_unembed: bool) -> list[str]:
    targets: list[str] = []
    if train_attn:
        targets.extend(DEEPSEEK_V3_LORA_TARGET_MODULES[:5])
    if train_mlp:
        targets.extend(DEEPSEEK_V3_LORA_TARGET_MODULES[5:])
    if train_unembed:
        targets.append("lm_head")
    return targets


def has_packed_expert_weights(checkpoint_keys: Iterable[str]) -> bool:
    return any(".mlp.experts." in key and key.endswith(".weight_packed") for key in checkpoint_keys)


def _resolve_weights_path(weights_path: Optional[str]) -> Optional[str]:
    if not weights_path:
        return None
    if os.path.isdir(weights_path):
        return weights_path
    try:
        config_path = cached_file(weights_path, "config.json", _raise_exceptions_for_missing_entries=False)
    except Exception:
        return None
    if config_path and os.path.isfile(config_path):
        return os.path.dirname(config_path)
    return None


def get_packed_expert_quantization_args(weights_path: Optional[str]) -> Optional[Tuple[int, int]]:
    resolved_weights_path = _resolve_weights_path(weights_path)
    if resolved_weights_path is None:
        return None

    config_path = os.path.join(resolved_weights_path, "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path) as f:
            config_dict = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    quantization_config = config_dict.get("quantization_config")
    if quantization_config is None and isinstance(config_dict.get("text_config"), dict):
        quantization_config = config_dict["text_config"].get("quantization_config")
    if not isinstance(quantization_config, dict):
        return None
    if quantization_config.get("quant_method") != "compressed-tensors":
        return None
    if quantization_config.get("format") != "pack-quantized":
        return None

    config_groups = quantization_config.get("config_groups", {})
    for group_config in config_groups.values():
        if not isinstance(group_config, dict):
            continue
        weights_config = group_config.get("weights")
        if not isinstance(weights_config, dict):
            continue
        return (
            int(weights_config.get("num_bits", PACKED_EXPERT_DEFAULT_NUM_BITS)),
            int(weights_config.get("group_size", PACKED_EXPERT_DEFAULT_GROUP_SIZE)),
        )

    return PACKED_EXPERT_DEFAULT_NUM_BITS, PACKED_EXPERT_DEFAULT_GROUP_SIZE
