"""Compatibility helpers for the XoRL-native FP8 training contract."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Collection

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class UnsupportedFP8ConfigError(ValueError):
    """Raised when a config asks for a non-XoRL FP8 workflow."""


def _as_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise UnsupportedFP8ConfigError(f"{field_name} must be a boolean, got {value!r}")


def normalize_fp8_training_config(config: Mapping[str, Any], *, context: str = "train") -> dict[str, Any]:
    """Normalize NeMo-style ``fp8_cfg`` onto XoRL-native FP8 training fields.

    XoRL intentionally supports only native block-FP8 compute training with
    full-precision master parameters. TransformerEngine-only recipes are
    rejected here with targeted messages before model construction starts.
    """

    normalized = dict(config)
    fp8_cfg = normalized.get("fp8_cfg")
    if fp8_cfg is None:
        return normalized
    if not isinstance(fp8_cfg, Mapping):
        raise UnsupportedFP8ConfigError(f"{context}.fp8_cfg must be a mapping, got {type(fp8_cfg).__name__}")

    enabled = _as_bool(fp8_cfg.get("enabled", False), field_name=f"{context}.fp8_cfg.enabled")
    if not enabled:
        return normalized

    raw_fp8 = str(fp8_cfg.get("fp8", "e4m3")).strip().lower()
    if raw_fp8 != "e4m3":
        raise UnsupportedFP8ConfigError(
            f"Unsupported {context}.fp8_cfg.fp8={fp8_cfg.get('fp8')!r}. "
            "XoRL native FP8 training supports E4M3 block-FP8 only; "
            "'hybrid' is a TransformerEngine recipe and is not implemented."
        )

    raw_recipe = str(fp8_cfg.get("fp8_recipe", "blockwise")).strip().lower()
    if raw_recipe != "blockwise":
        raise UnsupportedFP8ConfigError(
            f"Unsupported {context}.fp8_cfg.fp8_recipe={fp8_cfg.get('fp8_recipe')!r}. "
            "XoRL native FP8 training supports blockwise FP8 only; "
            "TransformerEngine tensorwise and MXFP8 recipes are not implemented."
        )

    fp8_param = fp8_cfg.get("fp8_param", False)
    if _as_bool(fp8_param, field_name=f"{context}.fp8_cfg.fp8_param"):
        raise UnsupportedFP8ConfigError(
            f"Unsupported {context}.fp8_cfg.fp8_param=true. XoRL keeps BF16/FP32 master parameters and does not "
            "store trainable parameters, optimizer state, or DCP checkpoints in FP8."
        )

    normalized["enable_fp8_training"] = True
    return normalized


def extract_nemo_fp8_cfg(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return ``policy.megatron_cfg.fp8_cfg`` when a NeMo-style config is provided."""

    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        return None
    megatron_cfg = policy.get("megatron_cfg")
    if not isinstance(megatron_cfg, Mapping):
        return None
    fp8_cfg = megatron_cfg.get("fp8_cfg")
    if fp8_cfg is None:
        return None
    if not isinstance(fp8_cfg, Mapping):
        raise UnsupportedFP8ConfigError("policy.megatron_cfg.fp8_cfg must be a mapping")
    return dict(fp8_cfg)


def validate_external_fp8_runtime_config(config: Mapping[str, Any], *, context: str = "config") -> None:
    """Reject non-XoRL low-precision runtime knobs in XoRL configs."""

    candidates: list[tuple[str, Any]] = []
    kv_cache_candidates: list[tuple[str, Any]] = []
    qarl_candidates: list[tuple[str, Any]] = []
    generation = config.get("generation")
    if isinstance(generation, Mapping):
        candidates.append((f"{context}.generation.vllm_cfg", generation.get("vllm_cfg")))
        kv_cache_candidates.append((f"{context}.generation.kv_cache_dtype", generation.get("kv_cache_dtype")))
        qarl_candidates.append((f"{context}.generation.quant_cfg", generation.get("quant_cfg")))
    policy = config.get("policy")
    if isinstance(policy, Mapping):
        qarl_candidates.append((f"{context}.policy.quant_cfg", policy.get("quant_cfg")))
        policy_generation = policy.get("generation")
        if isinstance(policy_generation, Mapping):
            candidates.append((f"{context}.policy.generation.vllm_cfg", policy_generation.get("vllm_cfg")))
            kv_cache_candidates.append(
                (f"{context}.policy.generation.kv_cache_dtype", policy_generation.get("kv_cache_dtype"))
            )
            qarl_candidates.append(
                (f"{context}.policy.generation.quant_cfg", policy_generation.get("quant_cfg"))
            )
    vllm_cfg = config.get("vllm_cfg")
    if vllm_cfg is not None:
        candidates.append((f"{context}.vllm_cfg", vllm_cfg))
    kv_cache_candidates.append((f"{context}.kv_cache_dtype", config.get("kv_cache_dtype")))
    qarl_candidates.append((f"{context}.quant_cfg", config.get("quant_cfg")))

    for name, value in candidates:
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise UnsupportedFP8ConfigError(f"{name} must be a mapping")
        kv_cache_candidates.append((f"{name}.kv_cache_dtype", value.get("kv_cache_dtype")))
        precision = value.get("precision")
        if isinstance(precision, str) and precision.strip().lower() == "fp8":
            raise UnsupportedFP8ConfigError(
                f"Unsupported {name}.precision='fp8'. XoRL does not implement vLLM FP8 receiver/refit; "
                "use SGLang receiver FP8 plus XoRL sender-side block-FP8 sync instead."
            )
        quantization = value.get("quantization")
        if isinstance(quantization, str) and quantization.strip().lower() == "fp8":
            raise UnsupportedFP8ConfigError(
                f"Unsupported {name}.quantization='fp8'. XoRL does not implement vLLM FP8 receiver/refit; "
                "use SGLang receiver FP8 plus XoRL sender-side block-FP8 sync instead."
            )
        for key in ("num_first_layers_in_bf16", "num_last_layers_in_bf16"):
            if key in value and value.get(key) is not None:
                raise UnsupportedFP8ConfigError(
                    f"Unsupported {name}.{key}. vLLM ignored-layer controls are not translated automatically; "
                    "use fp8_training_num_first_layers_bf16/fp8_training_num_last_layers_bf16 for "
                    "XoRL-native FP8 training and sync BF16 islands."
                )
        ignored_layer_kws = value.get("quantization_ignored_layer_kws")
        if ignored_layer_kws:
            raise UnsupportedFP8ConfigError(
                f"Unsupported {name}.quantization_ignored_layer_kws. vLLM ignored-layer keywords are not "
                "translated automatically; use fp8_training_exclude_modules and/or FP8 sync "
                "modules_to_not_convert explicitly."
            )
        if _as_bool(value.get("use_deep_gemm", False), field_name=f"{name}.use_deep_gemm"):
            raise UnsupportedFP8ConfigError(
                f"Unsupported {name}.use_deep_gemm=true. vLLM DeepGEMM is not translated automatically; "
                "XoRL's DeepGEMM knob applies only to native MoE FP8 training backends."
            )
        for key in ("pow2_weight_scaling_factors", "pow2_activation_scaling_factors"):
            if _as_bool(value.get(key, False), field_name=f"{name}.{key}"):
                raise UnsupportedFP8ConfigError(
                    f"Unsupported {name}.{key}=true. XoRL/SGLang FP8 sync uses FP32 weight_scale_inv block "
                    "scales; vLLM pow2 scale experiments need a separate sender/receiver contract and validation gate."
                )

    for name, value in kv_cache_candidates:
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"fp8", "fp8_e4m3"}:
            raise UnsupportedFP8ConfigError(
                f"Unsupported {name}={value!r}. XoRL does not implement vLLM FP8 KV-cache refit; "
                "launch the SGLang receiver with --kv-cache-dtype and set receiver_kv_cache_dtype "
                "in the XoRL server config to validate registered endpoint metadata."
            )

    for name, value in qarl_candidates:
        if value is None:
            continue
        raise UnsupportedFP8ConfigError(
            f"Unsupported {name}. XoRL does not translate NeMo ModelOpt QARL configs; "
            "set train.enable_qarl=true and train.qarl_quant_cfg explicitly for XoRL-native dense QARL."
        )


def _get_decoder_layer_info(model: nn.Module) -> tuple[str, int]:
    if hasattr(model, "get_pp_module_config"):
        pp_config = model.get_pp_module_config()
        if isinstance(pp_config, Mapping):
            layer_prefix = pp_config.get("layer_prefix")
            num_layers = pp_config.get("num_layers")
            if isinstance(layer_prefix, str) and isinstance(num_layers, int):
                return layer_prefix, num_layers

    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is not None and hasattr(layers, "__len__"):
        return "model.layers", len(layers)

    raise UnsupportedFP8ConfigError(
        "fp8_training_num_first_layers_bf16/fp8_training_num_last_layers_bf16 require a model with "
        "get_pp_module_config() or a standard model.layers decoder layout."
    )


def resolve_fp8_bf16_layer_islands(
    model: nn.Module,
    *,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
) -> list[str]:
    """Expand first/last BF16 layer island counts to FQN glob patterns."""

    if isinstance(num_first_layers_bf16, bool) or int(num_first_layers_bf16) < 0:
        raise UnsupportedFP8ConfigError("fp8_training_num_first_layers_bf16 must be a non-negative integer")
    if isinstance(num_last_layers_bf16, bool) or int(num_last_layers_bf16) < 0:
        raise UnsupportedFP8ConfigError("fp8_training_num_last_layers_bf16 must be a non-negative integer")

    first = int(num_first_layers_bf16)
    last = int(num_last_layers_bf16)
    if first == 0 and last == 0:
        return []

    layer_prefix, num_layers = _get_decoder_layer_info(model)
    if first > num_layers:
        raise UnsupportedFP8ConfigError(
            f"fp8_training_num_first_layers_bf16={first} exceeds model layer count {num_layers}"
        )
    if last > num_layers:
        raise UnsupportedFP8ConfigError(
            f"fp8_training_num_last_layers_bf16={last} exceeds model layer count {num_layers}"
        )

    layer_indices = set(range(first))
    if last:
        layer_indices.update(range(num_layers - last, num_layers))
    return [f"{layer_prefix}.{idx}.*" for idx in sorted(layer_indices)]


def merge_fp8_bf16_layer_island_excludes(
    model: nn.Module,
    exclude_modules: Collection[str] | None,
    *,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
) -> tuple[list[str], list[str]]:
    """Return ``exclude_modules`` plus generated BF16 island patterns."""

    excludes: list[str] = []
    seen: set[str] = set()
    for entry in exclude_modules or []:
        if entry not in seen:
            seen.add(entry)
            excludes.append(entry)

    islands = resolve_fp8_bf16_layer_islands(
        model,
        num_first_layers_bf16=num_first_layers_bf16,
        num_last_layers_bf16=num_last_layers_bf16,
    )
    for entry in islands:
        if entry not in seen:
            seen.add(entry)
            excludes.append(entry)
    return excludes, islands


def enrich_sync_quantization_with_fp8_bf16_islands(
    model: nn.Module,
    quantization_config: dict[str, Any] | None,
    *,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
) -> dict[str, Any] | None:
    """Add generated BF16 island prefixes to an FP8 sync quantization config."""

    if quantization_config is None or quantization_config.get("quant_method") != "fp8":
        return quantization_config
    enriched = dict(quantization_config)
    existing = enriched.get("modules_to_not_convert")
    modules, islands = merge_fp8_bf16_layer_island_excludes(
        model,
        existing if isinstance(existing, list) else [],
        num_first_layers_bf16=num_first_layers_bf16,
        num_last_layers_bf16=num_last_layers_bf16,
    )
    if modules:
        enriched["modules_to_not_convert"] = modules
    if islands:
        enriched["_xorl_generated_bf16_layer_islands"] = islands
    return enriched


def is_blackwell_device(
    *,
    device_name: str | None = None,
    capability: tuple[int, int] | None = None,
) -> bool:
    if capability is not None and capability[0] >= 10:
        return True
    if not device_name:
        return False
    lowered = device_name.lower()
    return any(token in lowered for token in ("blackwell", "gb200", "b200", "b100", "sm100"))


def validate_fp8_blackwell_training_policy(
    *,
    enable_fp8_training: bool,
    allow_blackwell: bool = False,
    validation_artifact: str | None = None,
    device_name: str | None = None,
    capability: tuple[int, int] | None = None,
) -> None:
    """Guard native FP8 training on Blackwell until a XoRL recipe is validated."""

    if not enable_fp8_training:
        return
    if device_name is None and capability is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)

    if not is_blackwell_device(device_name=device_name, capability=capability):
        return
    if not allow_blackwell:
        raise UnsupportedFP8ConfigError(
            "Native XoRL FP8 training is guarded on Blackwell/GB200 by default. "
            "Use BF16 training with FP8 sync/generation, or set fp8_training_allow_blackwell=true only with "
            "a validation artifact for the native non-TE FP8 recipe on this hardware."
        )
    if not validation_artifact:
        raise UnsupportedFP8ConfigError(
            "Native XoRL FP8 training on Blackwell/GB200 requires fp8_training_blackwell_validation_artifact "
            "when fp8_training_allow_blackwell=true. Use BF16 training plus FP8 sync/generation until a "
            "hardware-specific native FP8 recipe artifact exists."
        )
    logger.warning(
        "Native XoRL FP8 training is explicitly enabled on Blackwell using validation artifact: %s",
        validation_artifact,
    )
