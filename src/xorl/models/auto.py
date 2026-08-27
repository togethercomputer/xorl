import json
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
)

from ..distributed.parallel_state import get_parallel_state
from ..ops.moe.triton import resolve_routing_weights_before_down, set_routing_weights_before_down
from ..utils import logging
from .exact_contract import (
    EXACT_CONTRACT_FAMILY_GLM52,
    glm52_exact_forward_enabled,
    resolve_exact_contract_family,
    set_glm52_exact_active_lora,
)
from .layers.attention import get_attention_fn
from .layers.normalization import set_rmsnorm_mode
from .layers.rope import set_rope_class_b, set_rope_native
from .loader import ModelLoader, get_loader
from .transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from .transformers.deepseek_v3.support import validate_deepseek_v3_router_settings
from .transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from .transformers.deepseek_v4.exact_contract import (
    is_dsv4_flash_config,
    validate_dsv4_flash_adapter_program,
    validate_dsv4_flash_official_geometry,
)
from .transformers.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from .transformers.glm5.configuration_glm5 import Glm5Config
from .transformers.glm5.exact_lora_contract import glm52_exact_lora_scaling
from .transformers.glm5.layer_plan import Glm52LayerPlan, install_glm52_pipeline_module_plan
from .transformers.glm5.support import validate_glm5_router_settings, validate_glm5_sequence_parallel
from .transformers.gpt_oss.configuration_gpt_oss import GptOssConfig
from .transformers.minimax_m3.configuration_minimax_m3 import MiniMaxM3Config
from .transformers.nemotron_h.configuration_nemotron_h import NemotronHConfig
from .transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from .transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from .transformers.qwen3_5_shared import (
    LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE,
    has_linear_attention_layers,
)


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)

_MAX_CONFIG_DEPTH = 32
_MAX_CONFIG_CONTAINER_ITEMS = 100_000


def _validate_plain_config_value(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    """Copy an HF config tree while admitting only bounded plain-JSON values."""
    if depth > _MAX_CONFIG_DEPTH:
        raise ValueError(f"model config exceeds maximum nesting depth at {path}")
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is list:
        if len(value) > _MAX_CONFIG_CONTAINER_ITEMS:
            raise ValueError(f"model config list is too large at {path}")
        return [
            _validate_plain_config_value(item, path=f"{path}[{idx}]", depth=depth + 1) for idx, item in enumerate(value)
        ]
    if type(value) is dict:
        if len(value) > _MAX_CONFIG_CONTAINER_ITEMS:
            raise ValueError(f"model config mapping is too large at {path}")
        result = {}
        for key, item in value.items():
            if type(key) is not str or key.startswith("__"):
                raise ValueError(f"model config contains an unsafe key at {path}: {key!r}")
            result[key] = _validate_plain_config_value(item, path=f"{path}.{key}", depth=depth + 1)
        return result
    raise ValueError(f"model config contains a non-JSON value at {path}: {type(value).__name__}")


def _build_local_kimi_tokenizer(tokenizer_path: str):
    tokenizer_dir = Path(tokenizer_path)
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    vocab_file = tokenizer_dir / "tiktoken.model"
    if not tokenizer_config_path.is_file() or not vocab_file.is_file():
        return None

    with tokenizer_config_path.open() as f:
        tokenizer_config = json.load(f)

    auto_tokenizer = tokenizer_config.get("auto_map", {}).get("AutoTokenizer", [])
    auto_tokenizer_cls = auto_tokenizer[0] if auto_tokenizer else ""
    if tokenizer_config.get("tokenizer_class") != "TikTokenTokenizer" and not auto_tokenizer_cls.endswith(
        "TikTokenTokenizer"
    ):
        return None

    from .transformers.deepseek_v3.tokenization_kimi import TikTokenTokenizer  # noqa: PLC0415

    tokenizer_kwargs = dict(tokenizer_config)
    tokenizer_kwargs.pop("auto_map", None)
    tokenizer_kwargs.pop("tokenizer_class", None)
    tokenizer_kwargs.pop("vocab_file", None)
    tokenizer_kwargs["padding_side"] = "right"
    return TikTokenTokenizer(vocab_file=str(vocab_file), **tokenizer_kwargs)


def _namespace_from_dict(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(**{k: _namespace_from_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace_from_dict(item) for item in value]
    return value


def _load_local_xorl_config(
    config_path: str,
    config_kwargs: Dict[str, Any],
) -> Optional["PretrainedConfig"]:
    config_dict, _ = PretrainedConfig.get_config_dict(config_path, **config_kwargs)
    config_dict = _validate_plain_config_value(config_dict)
    model_type = config_dict.get("model_type")

    if model_type == "glm_moe_dsa":
        return Glm5Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "glm4_moe":
        return Glm4MoeConfig.from_dict(config_dict)

    if model_type == "qwen3_5_moe":
        return Qwen3_5MoeConfig.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "qwen3_5":
        return Qwen3_5Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type in {"deepseek_v3", "kimi_k2", "kimi_k25"}:
        return DeepseekV3Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "deepseek_v4":
        return DeepseekV4Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "nemotron_h":
        return NemotronHConfig.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "qwen2":
        from .transformers.qwen2.configuration_qwen2 import Qwen2Config  # noqa: PLC0415

        return Qwen2Config(**{k: v for k, v in config_dict.items() if not k.startswith("_")})

    if model_type == "gpt_oss":
        return GptOssConfig.from_hf_config(_namespace_from_dict(config_dict))

    if model_type in {"minimax_m3_vl", "xorl_minimax_m3"}:
        return MiniMaxM3Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "olmo2":
        from .transformers.olmo2.configuration_olmo2 import Olmo2Config  # noqa: PLC0415

        return Olmo2Config(**{k: v for k, v in config_dict.items() if not k.startswith("_")})

    return None


def _get_architectures(config: "PretrainedConfig") -> set[str]:
    architectures = getattr(config, "architectures", None)
    if architectures is None:
        return set()
    if isinstance(architectures, list):
        return set(architectures)
    return {architectures}


def _is_gpt_oss_config(config: "PretrainedConfig") -> bool:
    return getattr(config, "model_type", None) == "gpt_oss" or "GptOssForCausalLM" in _get_architectures(config)


def _is_minimax_m3_config(config: "PretrainedConfig") -> bool:
    return getattr(config, "model_type", None) == "xorl_minimax_m3" or bool(
        _get_architectures(config) & {"MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3SparseForCausalLM"}
    )


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    tokenizer = _build_local_kimi_tokenizer(tokenizer_path)
    if tokenizer is not None:
        return tokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right")


def _load_config_with_rank0_priority(
    config_path: str,
    config_kwargs: Dict[str, Any],
) -> "PretrainedConfig":
    """
    Load model config with rank 0 going first to avoid HF Hub race conditions.

    When multiple ranks call AutoConfig.from_pretrained simultaneously on a
    HuggingFace Hub model ID, some may get incomplete downloads, causing
    'Unrecognized model' errors. This function lets rank 0 download first
    (populating the cache), then other ranks load from the cache.
    """

    rank = get_parallel_state().global_rank if get_parallel_state().is_initialized else 0
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    if is_distributed and rank != 0:
        dist.barrier()

    config = AutoConfig.from_pretrained(config_path, **config_kwargs)

    if is_distributed and rank == 0:
        dist.barrier()

    return config


def _is_canonical_glm52(config: PretrainedConfig) -> bool:
    if not (isinstance(config, Glm5Config) and getattr(config, "indexer_types", None) is not None):
        return False
    _validate_canonical_glm52_model_scope(config)
    return True


def _is_exact_glm52(config: PretrainedConfig) -> bool:
    return glm52_exact_forward_enabled(config)


def _validate_canonical_glm52_model_scope(config: PretrainedConfig) -> None:
    if not (isinstance(config, Glm5Config) and getattr(config, "indexer_types", None) is not None):
        return
    expected = {
        "vocab_size": 154880,
        "hidden_size": 6144,
        "hidden_act": "silu",
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "index_topk": 2048,
    }
    mismatches = [
        f"{name}={getattr(config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    try:
        # Layer count and ownership are checkpoint schedules, not kernel
        # geometry. Validate their lengths, supported values, producer chain,
        # and legal full-index boundaries without assuming the 78-layer
        # reference checkpoint.
        Glm52LayerPlan.from_config(config)
    except (TypeError, ValueError) as error:
        mismatches.append(str(error))
    if mismatches:
        raise ValueError(
            "The exact GLM-5.2 program supports only the official model geometry: " + ", ".join(mismatches)
        )


def _is_exact_qwen35(config: PretrainedConfig) -> bool:
    return bool(getattr(config, "_qwen35_exact_contract", False))


def _is_exact_qwen3_dense(config: PretrainedConfig) -> bool:
    return bool(getattr(config, "_qwen3_dense_exact_contract", False))


def qwen_exact_contracts_engaged(*, server_training: bool, enable_qlora: bool) -> bool:
    """Whether the Qwen3-dense / Qwen3.5 exact server-training contracts may engage.

    The exact value programs pair a bf16 trainer with exact bf16 serving. A
    QLoRA run trains adapters over a quantized frozen base, which neither side
    of that contract admits, so the family stamps disengage and generic
    trainer numerics apply (the GLM-5.2 stamp already disengages under its
    block_fp8_qlora_training lane; this is the same rule for the Qwen
    families, keyed on the generic enable_qlora flag the fp8_lora
    train_serve_profile pins).
    """
    return bool(server_training and not enable_qlora)


def _validate_exact_qwen3_dense_model_scope(config: PretrainedConfig) -> None:
    if not _is_exact_qwen3_dense(config):
        return
    mismatches = []

    positive_integer_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    )
    for name in positive_integer_fields:
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            mismatches.append(f"{name}={value!r} (requires a positive integer)")

    required_values = {
        "head_dim": 128,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
        "attention_dropout": 0.0,
    }
    for name, required in required_values.items():
        actual = getattr(config, name, None)
        if actual != required:
            mismatches.append(f"{name}={actual!r} (requires {required!r})")

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        unsupported_keys = set(rope_scaling) - {"rope_type", "type", "rope_theta"}
        if rope_type not in (None, "default") or unsupported_keys:
            mismatches.append(f"rope_scaling={rope_scaling!r} (only default RoPE is supported)")
    elif rope_scaling:
        mismatches.append(f"rope_scaling={rope_scaling!r} (only default RoPE is supported)")

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta")
    if not isinstance(rope_theta, (int, float)) or isinstance(rope_theta, bool) or rope_theta <= 0:
        mismatches.append(f"rope_theta={rope_theta!r} (requires a positive number)")

    rms_norm_eps = getattr(config, "rms_norm_eps", None)
    if not isinstance(rms_norm_eps, (int, float)) or isinstance(rms_norm_eps, bool) or rms_norm_eps <= 0:
        mismatches.append(f"rms_norm_eps={rms_norm_eps!r} (requires a positive number)")

    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if (
        isinstance(num_attention_heads, int)
        and isinstance(num_key_value_heads, int)
        and num_attention_heads > 0
        and num_key_value_heads > 0
        and num_attention_heads % num_key_value_heads != 0
    ):
        mismatches.append(
            "num_attention_heads must be divisible by num_key_value_heads "
            f"(got {num_attention_heads} and {num_key_value_heads})"
        )

    if mismatches:
        raise ValueError(
            "The exact dense Qwen3 program does not support this architecture configuration: " + ", ".join(mismatches)
        )


def _is_qwen35_moe(config: PretrainedConfig) -> bool:
    return getattr(config, "model_type", None) in {
        "xorl_qwen3_5_moe",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }


def _validate_exact_qwen35_dense_capabilities(config: PretrainedConfig) -> list[str]:
    mismatches = []

    positive_integer_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "linear_num_value_heads",
    )
    for name in positive_integer_fields:
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            mismatches.append(f"{name}={value!r} (requires a positive integer)")

    required_values = {
        "head_dim": 256,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
    }
    for name, required in required_values.items():
        actual = getattr(config, name, None)
        if actual != required:
            mismatches.append(f"{name}={actual!r} (requires {required!r})")

    rms_norm_eps = getattr(config, "rms_norm_eps", None)
    if not isinstance(rms_norm_eps, (int, float)) or isinstance(rms_norm_eps, bool) or rms_norm_eps <= 0:
        mismatches.append(f"rms_norm_eps={rms_norm_eps!r} (requires a positive number)")

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta")
    if not isinstance(rope_theta, (int, float)) or isinstance(rope_theta, bool) or rope_theta <= 0:
        mismatches.append(f"rope_theta={rope_theta!r} (requires a positive number)")

    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor != 0.25:
        mismatches.append(f"partial_rotary_factor={partial_rotary_factor!r} (requires 0.25)")

    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if (
        isinstance(num_attention_heads, int)
        and isinstance(num_key_value_heads, int)
        and num_attention_heads > 0
        and num_key_value_heads > 0
        and num_attention_heads % num_key_value_heads != 0
    ):
        mismatches.append(
            "num_attention_heads must be divisible by num_key_value_heads "
            f"(got {num_attention_heads} and {num_key_value_heads})"
        )

    linear_num_value_heads = getattr(config, "linear_num_value_heads", None)
    if isinstance(linear_num_value_heads, int) and linear_num_value_heads > 0 and linear_num_value_heads % 16 != 0:
        mismatches.append(f"linear_num_value_heads={linear_num_value_heads!r} (requires a multiple of 16)")

    return mismatches


def _is_exact_dsv4_flash(config: PretrainedConfig) -> bool:
    return bool(getattr(config, "_dsv4_flash_exact_mode", False))


def _validate_exact_qwen35_model_scope(config: PretrainedConfig) -> None:
    if not _is_exact_qwen35(config):
        return
    # Hugging Face multimodal checkpoints expose the language-model geometry
    # under ``text_config``.  The normal XORL path converts that section to a
    # local config before reaching this validator, but callers may also pass an
    # AutoConfig instance directly to ``build_foundation_model``.
    scope_config = getattr(config, "text_config", config)
    if _is_qwen35_moe(config):
        expected = {
            "vocab_size": 248320,
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
        }
        mismatches = [
            f"{name}={getattr(scope_config, name, None)!r} (requires {value!r})"
            for name, value in expected.items()
            if getattr(scope_config, name, None) != value
        ]
        model_name = "Qwen3.6-35B-A3B"
    else:
        mismatches = _validate_exact_qwen35_dense_capabilities(scope_config)
        model_name = "dense Qwen3.5"
    num_hidden_layers = getattr(scope_config, "num_hidden_layers", None)
    # ``full_attention_interval`` uniquely defines this schedule.  Some HF
    # config representations materialize ``layer_types`` while others retain
    # only the interval, so validate the derived list when it is present rather
    # than rejecting an otherwise identical checkpoint representation.
    layer_types = getattr(scope_config, "layer_types", None)
    if isinstance(num_hidden_layers, int) and num_hidden_layers > 0 and layer_types is not None:
        expected_layer_types = tuple(
            "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
            for layer_idx in range(num_hidden_layers)
        )
        if tuple(layer_types) != expected_layer_types:
            mismatches.append("layer_types does not match the certified full-attention interval")
    if mismatches:
        raise ValueError(
            f"The exact Qwen3.5-family server-training program does not support this {model_name} configuration: "
            + ", ".join(mismatches)
        )


def _validate_exact_qwen35_moe_program(
    config: PretrainedConfig,
    *,
    moe_implementation: Optional[str],
    ep_dispatch: str,
    deepep_async_combine: bool,
) -> None:
    if not (_is_exact_qwen35(config) and _is_qwen35_moe(config)):
        return
    incompatible = []
    if moe_implementation not in (None, "triton"):
        incompatible.append(f"moe_implementation={moe_implementation!r} (requires 'triton')")
    if ep_dispatch != "alltoall":
        incompatible.append(f"ep_dispatch={ep_dispatch!r} (requires 'alltoall')")
    if deepep_async_combine:
        incompatible.append("deepep_async_combine=True (requires False)")
    if incompatible:
        raise ValueError(
            "Exact Qwen3.5-MoE server training rejects incompatible MoE overrides: " + ", ".join(incompatible)
        )


@dataclass(frozen=True)
class ResolvedModelNumericalProgram:
    """Bit-relevant model choices resolved before module construction."""

    attn_implementation: str
    router_fp32: bool
    lm_head_fp32: bool
    rmsnorm_mode: str
    qwen35_rmsnorm_family: Optional[str]
    activation_native: bool
    rope_native: bool
    rope_class_b: bool
    attention_cast_bf16: bool
    sparse_mla_enabled: bool
    sparse_mla_backend: str


def _resolve_rope_modes(
    config: PretrainedConfig,
    *,
    rope_native: Optional[bool],
    rope_class_b: Optional[bool],
) -> tuple[bool, bool]:
    if _is_exact_glm52(config):
        if rope_native is False or rope_class_b is False:
            raise ValueError(
                "Canonical GLM-5.2 requires native Class-B RoPE; explicit rope_native=false "
                "or rope_class_b=false is incompatible with the model's numerical contract"
            )
        return True, True

    if _is_exact_qwen35(config):
        if rope_native is False or rope_class_b is False:
            raise ValueError(
                "Exact Qwen3.5-family server training requires native Class-B RoPE; "
                "explicit rope_native=false or rope_class_b=false is incompatible "
                "with the model's numerical contract"
            )
        return True, True

    if _is_exact_qwen3_dense(config):
        if rope_native is False or rope_class_b is False:
            raise ValueError(
                "Exact dense Qwen3 server training requires native Class-B RoPE; "
                "explicit rope_native=false or rope_class_b=false is incompatible "
                "with the model's numerical contract"
            )
        return True, True

    effective_rope_native = bool(rope_native)
    effective_rope_class_b = bool(rope_class_b)
    if effective_rope_class_b and not effective_rope_native:
        raise ValueError(
            "rope_class_b=True requires rope_native=True: the Class-B contract uses "
            "the CPU-built serving-layout cos/sin cache selected by rope_native"
        )
    return effective_rope_native, effective_rope_class_b


def resolve_model_numerical_program(
    config: PretrainedConfig,
    *,
    attn_implementation: Optional[str],
    non_glm_attn_default: str,
    router_fp32: Optional[bool],
    lm_head_fp32: Optional[bool],
    rmsnorm_mode: Optional[str],
    activation_native: bool,
    rope_native: Optional[bool],
    rope_class_b: Optional[bool],
    attention_cast_bf16: bool,
    sparse_mla_enabled: Optional[bool],
    sparse_mla_backend: Optional[str],
    qwen35_rmsnorm_family: Optional[str] = None,
) -> ResolvedModelNumericalProgram:
    """Resolve exact model numerics while preserving non-GLM defaults.

    Canonical GLM-5.2 is a numerical program, not a collection of optional
    optimizations. Omitted values select that program and incompatible
    explicit values fail before weights are loaded.
    """

    effective_rope_native, effective_rope_class_b = _resolve_rope_modes(
        config,
        rope_native=rope_native,
        rope_class_b=rope_class_b,
    )
    if qwen35_rmsnorm_family not in (None, "v1", "v2"):
        raise ValueError(f"qwen35_rmsnorm_family must be one of None, 'v1', or 'v2'; got {qwen35_rmsnorm_family!r}")
    if _is_exact_qwen35(config):
        requirements = {
            "attn_implementation": (attn_implementation, "flash_attention_4"),
            "router_fp32": (router_fp32, True),
            "lm_head_fp32": (lm_head_fp32, True),
            "rmsnorm_mode": (rmsnorm_mode, "sglang_fused"),
        }
        incompatible = [
            f"{name}={requested!r} (requires {required!r})"
            for name, (requested, required) in requirements.items()
            if requested is not None and requested != required
        ]
        if qwen35_rmsnorm_family not in (None, "v2"):
            incompatible.append(f"qwen35_rmsnorm_family={qwen35_rmsnorm_family!r} (requires 'v2')")
        if incompatible:
            raise ValueError(
                "Exact Qwen3.5-family server training rejects incompatible numerical overrides: "
                + ", ".join(incompatible)
            )
        return ResolvedModelNumericalProgram(
            attn_implementation="flash_attention_4",
            router_fp32=True,
            lm_head_fp32=True,
            rmsnorm_mode="sglang_fused",
            qwen35_rmsnorm_family="v2",
            activation_native=True,
            rope_native=True,
            rope_class_b=effective_rope_class_b,
            attention_cast_bf16=True,
            sparse_mla_enabled=False,
            sparse_mla_backend="auto",
        )

    if _is_exact_qwen3_dense(config):
        requirements = {
            "attn_implementation": (attn_implementation, "flash_attention_4"),
            "lm_head_fp32": (lm_head_fp32, True),
            "rmsnorm_mode": (rmsnorm_mode, "sglang_fused"),
            "activation_native": (activation_native, False),
        }
        incompatible = [
            f"{name}={requested!r} (requires {required!r})"
            for name, (requested, required) in requirements.items()
            if requested is not None and requested != required
        ]
        if incompatible:
            raise ValueError(
                "Exact dense Qwen3 server training rejects incompatible numerical overrides: " + ", ".join(incompatible)
            )
        return ResolvedModelNumericalProgram(
            attn_implementation="flash_attention_4",
            router_fp32=True,
            lm_head_fp32=True,
            rmsnorm_mode="sglang_fused",
            qwen35_rmsnorm_family=None,
            activation_native=False,
            rope_native=True,
            rope_class_b=effective_rope_class_b,
            attention_cast_bf16=False,
            sparse_mla_enabled=False,
            sparse_mla_backend="auto",
        )

    if _is_exact_dsv4_flash(config):
        requirements = {
            "attn_implementation": (attn_implementation, "flash_attention_4"),
            "router_fp32": (router_fp32, False),
            "lm_head_fp32": (lm_head_fp32, False),
            "rmsnorm_mode": (rmsnorm_mode, "native"),
            "activation_native": (activation_native, False),
            "attention_cast_bf16": (attention_cast_bf16, False),
            "sparse_mla_enabled": (sparse_mla_enabled, False),
        }
        incompatible = [
            f"{name}={requested!r} (requires {required!r})"
            for name, (requested, required) in requirements.items()
            if requested is not None and requested != required
        ]
        if qwen35_rmsnorm_family is not None:
            incompatible.append(f"qwen35_rmsnorm_family={qwen35_rmsnorm_family!r} (unsupported for DSV4-Flash)")
        if sparse_mla_backend not in (None, "auto"):
            incompatible.append(f"sparse_mla_backend={sparse_mla_backend!r} (requires 'auto')")
        if incompatible:
            raise ValueError(
                "The DSV4-Flash exact RCA lane rejects incompatible numerical overrides: " + ", ".join(incompatible)
            )
        return ResolvedModelNumericalProgram(
            attn_implementation="flash_attention_4",
            router_fp32=False,
            lm_head_fp32=False,
            rmsnorm_mode="native",
            qwen35_rmsnorm_family=None,
            activation_native=False,
            rope_native=False,
            rope_class_b=False,
            attention_cast_bf16=False,
            sparse_mla_enabled=False,
            sparse_mla_backend="auto",
        )

    if not _is_exact_glm52(config):
        if qwen35_rmsnorm_family is not None:
            raise ValueError(
                "qwen35_rmsnorm_family is supported only by exact Qwen3.5/3.6 server training; "
                f"got model_type={getattr(config, 'model_type', None)!r}."
            )
        return ResolvedModelNumericalProgram(
            attn_implementation=attn_implementation or non_glm_attn_default,
            router_fp32=True if router_fp32 is None else router_fp32,
            lm_head_fp32=True if lm_head_fp32 is None else lm_head_fp32,
            rmsnorm_mode=rmsnorm_mode or "native",
            qwen35_rmsnorm_family=None,
            activation_native=activation_native,
            rope_native=effective_rope_native,
            rope_class_b=effective_rope_class_b,
            attention_cast_bf16=attention_cast_bf16,
            sparse_mla_enabled=False if sparse_mla_enabled is None else sparse_mla_enabled,
            sparse_mla_backend=sparse_mla_backend or "auto",
        )

    requirements = {
        "attn_implementation": (attn_implementation, "flash_attention_4"),
        "router_fp32": (router_fp32, True),
        "lm_head_fp32": (lm_head_fp32, True),
        "rmsnorm_mode": (rmsnorm_mode, "sglang_fused"),
        "activation_native": (activation_native, False),
        "attention_cast_bf16": (attention_cast_bf16, False),
        "sparse_mla_enabled": (sparse_mla_enabled, True),
    }
    incompatible = [
        f"{name}={requested!r} (requires {required!r})"
        for name, (requested, required) in requirements.items()
        if requested is not None and requested != required
    ]
    if qwen35_rmsnorm_family is not None:
        incompatible.append(f"qwen35_rmsnorm_family={qwen35_rmsnorm_family!r} (supported only by exact Qwen3.5/3.6)")
    if sparse_mla_backend not in (None, "auto", "flashmla"):
        incompatible.append(f"sparse_mla_backend={sparse_mla_backend!r} (requires 'flashmla')")
    if incompatible:
        raise ValueError(
            "Canonical GLM-5.2 exact forward rejects incompatible numerical overrides: " + ", ".join(incompatible)
        )

    return ResolvedModelNumericalProgram(
        attn_implementation="flash_attention_4",
        router_fp32=True,
        lm_head_fp32=True,
        rmsnorm_mode="sglang_fused",
        qwen35_rmsnorm_family=None,
        activation_native=False,
        rope_native=True,
        rope_class_b=True,
        attention_cast_bf16=False,
        sparse_mla_enabled=True,
        sparse_mla_backend="flashmla",
    )


def resolve_cross_entropy_mode(config: PretrainedConfig, ce_mode: Optional[str]) -> str:
    """Resolve the loss-side member of the canonical numerical program."""

    if _is_exact_dsv4_flash(config):
        if ce_mode not in (None, "compiled"):
            raise ValueError(
                f"Exact DeepSeek-V4-Flash server training requires ce_mode='compiled'; received {ce_mode!r}"
            )
        return "compiled"
    if _is_exact_qwen35(config):
        if ce_mode not in (None, "bi_fused"):
            raise ValueError(f"Exact Qwen3.5-family server training requires ce_mode='bi_fused'; received {ce_mode!r}")
        return "bi_fused"
    if _is_exact_qwen3_dense(config):
        if ce_mode not in (None, "bi_fused"):
            raise ValueError(f"Exact dense Qwen3 server training requires ce_mode='bi_fused'; received {ce_mode!r}")
        return "bi_fused"
    if not _is_exact_glm52(config):
        return ce_mode or "compiled"
    if ce_mode not in (None, "bi_fused"):
        raise ValueError(f"Canonical GLM-5.2 exact forward requires ce_mode='bi_fused'; received {ce_mode!r}")
    return "bi_fused"


def _resolve_exact_one_round_swiglu(config) -> bool:
    """Whether the resolved program pairs with serving's one-round FP32 SwiGLU.

    Serving applies the one-round program universally in exact mode
    (SiluAndMul.forward_exact -> fp32_silu_and_mul), so every admitted exact
    contract family must select it: Qwen3.5 dense/MoE, exact dense Qwen3, and
    GLM-5.2. Non-exact models keep the historical two-round bytes.
    """
    return (
        bool(getattr(config, "_qwen35_exact_contract", False))
        or bool(getattr(config, "_qwen3_dense_exact_contract", False))
        or (getattr(config, "_exact_contract_family", None) == EXACT_CONTRACT_FAMILY_GLM52)
    )


def build_foundation_model(
    config_path: Union[str, PretrainedConfig],
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[
        Literal["eager", "sdpa", "native", "flash_attention_3", "flash_attention_4", "minimax_msa"]
    ] = None,
    non_glm_attn_default: Literal[
        "eager", "sdpa", "native", "flash_attention_3", "flash_attention_4", "minimax_msa"
    ] = "flash_attention_4",
    moe_implementation: Optional[Literal["eager", "triton", "native", "quack"]] = None,
    moe_routing_weights_before_down: Union[bool, str] = "auto",
    ep_dispatch: str = "alltoall",
    train_router: bool = False,
    record_routing_weights: bool = True,
    deepep_buffer_size_gb: float = 2.0,
    deepep_num_sms: int = 20,
    deepep_async_combine: bool = False,
    alltoall_combine_hidden_chunk_size: int = 0,
    router_fp32: Optional[bool] = None,
    lm_head_fp32: Optional[bool] = None,
    rmsnorm_mode: Optional[
        Literal["eager", "native", "compile", "sglang", "sglang_fused", "sglang_jit", "sglang_kernel"]
    ] = None,
    qwen35_rmsnorm_family: Optional[Literal["v1", "v2"]] = None,
    activation_native: bool = False,
    rope_native: Optional[bool] = None,
    rope_class_b: Optional[bool] = None,
    attention_cast_bf16: bool = False,
    sparse_mla_enabled: Optional[bool] = None,
    sparse_mla_backend: Optional[str] = None,
    flash_attention_deterministic: bool = False,
    server_training: bool = False,
    enable_lora: bool = False,
    enable_qlora: bool = False,
    block_fp8_qlora_training: bool = False,
    glm52_fullparam_fp8_training: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_target_modules: Optional[list[str]] = None,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_parallel_virtual_stages: int = 1,
    pipeline_parallel_input_weight: int = 1,
    pipeline_parallel_output_weight: int = 1,
    pipeline_parallel_num_layers_in_first_stage: Optional[int] = None,
    pipeline_parallel_num_layers_in_last_stage: Optional[int] = None,
) -> nn.Module:
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """
    if config_kwargs is None:
        config_kwargs = {}

    if isinstance(config_path, PretrainedConfig):
        config = config_path
    else:
        config = _load_local_xorl_config(config_path, config_kwargs)
        if config is None:
            config = _load_config_with_rank0_priority(config_path, config_kwargs)

    glm52_model = _is_canonical_glm52(config)
    if block_fp8_qlora_training and not glm52_model:
        raise ValueError("block_fp8_qlora_training is supported only for the official GLM-5.2 model")
    if glm52_fullparam_fp8_training and not glm52_model:
        raise ValueError("glm52_fullparam_fp8_training is supported only for the official canonical GLM-5.2 model")
    if glm52_fullparam_fp8_training and block_fp8_qlora_training:
        raise ValueError(
            "glm52_fullparam_fp8_training and block_fp8_qlora_training are mutually exclusive training lanes"
        )
    exact_active_lora = bool(server_training and glm52_model and block_fp8_qlora_training and ep_dispatch == "alltoall")
    if exact_active_lora:
        glm52_exact_lora_scaling(lora_rank, lora_alpha)
    # Training lanes select the same exact value family through their own
    # admission flags.  The scoring-only flag must remain off for either one:
    # it describes a frozen trunk, which neither training admission permits.
    config._glm52_block_fp8_qlora = bool(block_fp8_qlora_training)
    config._glm52_fullparam_training = bool(glm52_fullparam_fp8_training)
    config._glm52_exact_contract = bool(
        server_training and glm52_model and not block_fp8_qlora_training and not glm52_fullparam_fp8_training
    )
    set_glm52_exact_active_lora(config, enabled=exact_active_lora)
    canonical_glm52 = _is_exact_glm52(config)
    qwen35_model_type = getattr(config, "model_type", None) in {
        "xorl_qwen3_5",
        "xorl_qwen3_5_moe",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
    qwen_exact_eligible = qwen_exact_contracts_engaged(
        server_training=server_training,
        enable_qlora=enable_qlora,
    )
    config._qwen35_exact_contract = bool(qwen_exact_eligible and qwen35_model_type)
    config._qwen3_dense_exact_contract = bool(
        qwen_exact_eligible
        and getattr(config, "model_type", None) == "qwen3"
        and "Qwen3ForCausalLM" in _get_architectures(config)
    )
    if server_training and not qwen_exact_eligible and (qwen35_model_type or getattr(config, "model_type", None) == "qwen3"):
        logger.info_rank0(
            "Qwen exact server-training contract disengaged: enable_qlora=True trains "
            "on a quantized base, which the bf16 exact value program does not admit; "
            "generic trainer numerics apply."
        )
    dsv4_flash_exact = bool(server_training and is_dsv4_flash_config(config))
    config._dsv4_flash_exact_mode = dsv4_flash_exact
    config._dsv4_flash_exact_active_lora = bool(dsv4_flash_exact and enable_lora)
    if dsv4_flash_exact:
        validate_dsv4_flash_official_geometry(config)
        if not enable_lora:
            # The exact TP8 selected-logprob head exists only as the injected
            # LoRA head class; without it a base-only trainer would silently
            # pair an exact trunk with a non-exact head. The admitted lane is
            # active-LoRA only; static parity replays load certified all-zero
            # factors instead of dropping the adapter.
            raise ValueError(
                "Exact DeepSeek-V4-Flash server training admits only the active-LoRA "
                "lane (enable_lora=true with the complete rank-1/alpha-1 adapter); "
                "for static parity replays load the certified all-zero adapter."
            )
        if lora_rank is None or lora_alpha is None:
            raise ValueError("DSV4-Flash exact active-LoRA construction requires explicit rank and alpha")
        validate_dsv4_flash_adapter_program(
            adapter_rank=lora_rank,
            adapter_alpha=lora_alpha,
            target_modules=lora_target_modules,
        )
        incompatible = []
        if moe_implementation not in (None, "triton"):
            incompatible.append(f"moe_implementation={moe_implementation!r} (requires 'triton')")
        if ep_dispatch != "alltoall":
            incompatible.append(f"ep_dispatch={ep_dispatch!r} (requires 'alltoall')")
        if deepep_async_combine:
            incompatible.append("deepep_async_combine=True (requires False)")
        if incompatible:
            raise ValueError(
                "The DSV4-Flash exact RCA lane rejects incompatible trainer runtime choices: " + ", ".join(incompatible)
            )

    # Family-neutral exact-contract keys, stamped once at model resolution so
    # downstream contract sites key off the resolved program rather than
    # family-branded flags. ``_exact_contract_family`` names the exact value
    # program (or ``None`` for generic models); ``_exact_one_round_swiglu``
    # selects the serving-paired one-round FP32 SwiGLU. Serving applies the
    # one-round program universally in exact mode (xorl-sglang f10b907d8), so
    # every admitted contracted family pairs with it, including Qwen3.5
    # dense/MoE and GLM-5.2.
    config._exact_contract_family = resolve_exact_contract_family(config)
    config._exact_one_round_swiglu = _resolve_exact_one_round_swiglu(config)
    _validate_exact_qwen35_model_scope(config)
    _validate_exact_qwen3_dense_model_scope(config)
    _validate_exact_qwen35_moe_program(
        config,
        moe_implementation=moe_implementation,
        ep_dispatch=ep_dispatch,
        deepep_async_combine=deepep_async_combine,
    )
    numerical_program = resolve_model_numerical_program(
        config,
        attn_implementation=attn_implementation,
        non_glm_attn_default=non_glm_attn_default,
        router_fp32=router_fp32,
        lm_head_fp32=lm_head_fp32,
        rmsnorm_mode=rmsnorm_mode,
        qwen35_rmsnorm_family=qwen35_rmsnorm_family,
        activation_native=activation_native,
        rope_native=rope_native,
        rope_class_b=rope_class_b,
        attention_cast_bf16=attention_cast_bf16,
        sparse_mla_enabled=sparse_mla_enabled,
        sparse_mla_backend=sparse_mla_backend,
    )
    attn_implementation = numerical_program.attn_implementation
    router_fp32 = numerical_program.router_fp32
    lm_head_fp32 = numerical_program.lm_head_fp32
    rmsnorm_mode = numerical_program.rmsnorm_mode
    activation_native = numerical_program.activation_native
    effective_rope_native = numerical_program.rope_native
    attention_cast_bf16 = numerical_program.attention_cast_bf16
    sparse_mla_enabled = numerical_program.sparse_mla_enabled
    sparse_mla_backend = numerical_program.sparse_mla_backend
    # Exact GLM/Qwen modules carry architecture-scoped RoPE choices on their
    # config and never consult mutable process-wide selectors. Preserve the
    # legacy selectors only for non-target models that still use the generic
    # rotary helper.
    if not (canonical_glm52 or config._qwen35_exact_contract):
        set_rope_native(numerical_program.rope_native)
        set_rope_class_b(numerical_program.rope_class_b)
    config._rope_native = numerical_program.rope_native
    config._rope_class_b = numerical_program.rope_class_b
    config._resolved_numerical_program = asdict(numerical_program)
    if canonical_glm52:
        logger.info_rank0(f"Canonical GLM-5.2 numerical program: {numerical_program}")
    elif config._qwen35_exact_contract:
        logger.info_rank0(
            "Exact Qwen3.5-family server-training numerical program "
            f"(Class-B RoPE, RMSNorm {numerical_program.qwen35_rmsnorm_family}): {numerical_program}"
        )
    elif config._qwen3_dense_exact_contract:
        logger.info_rank0(
            "Exact dense Qwen3 server-training numerical program "
            f"(Class-B RoPE, RMSNorm families-v2): {numerical_program}"
        )

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "triton", "native", "quack"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    validate_deepseek_v3_router_settings(config, train_router=train_router)
    validate_glm5_router_settings(config, train_router=train_router)

    if ep_dispatch == "deepep" and train_router:
        raise ValueError(
            "train_router=True is not supported with ep_dispatch='deepep'. "
            "Set train_router=False or use ep_dispatch='alltoall'."
        )

    config._ep_dispatch = ep_dispatch
    config.train_router = train_router
    config.record_routing_weights = record_routing_weights
    config._deepep_buffer_size_gb = deepep_buffer_size_gb
    config._deepep_num_sms = deepep_num_sms
    config._deepep_async_combine = deepep_async_combine
    config._alltoall_combine_hidden_chunk_size = alltoall_combine_hidden_chunk_size
    config._router_fp32 = router_fp32
    config._lm_head_fp32 = lm_head_fp32
    routing_before_down = resolve_routing_weights_before_down(
        moe_routing_weights_before_down, train_router=train_router, ep_dispatch=ep_dispatch
    )
    set_routing_weights_before_down(routing_before_down)
    logger.info_rank0(
        f"MoE routing-weight position: {'before' if routing_before_down else 'after'}-down "
        f"(moe_routing_weights_before_down={moe_routing_weights_before_down!r}, "
        f"train_router={train_router}, ep_dispatch={ep_dispatch})"
    )
    set_rmsnorm_mode(rmsnorm_mode)
    config._rmsnorm_mode = rmsnorm_mode
    config._qwen35_rmsnorm_family = numerical_program.qwen35_rmsnorm_family
    config._activation_native = activation_native
    config._rope_native = effective_rope_native
    config._attention_cast_bf16 = attention_cast_bf16
    config._sparse_mla_enabled = sparse_mla_enabled
    config._sparse_mla_backend = sparse_mla_backend
    config._flash_attention_deterministic = flash_attention_deterministic

    if ep_dispatch == "deepep":
        # Probe the internode transport before weight loading: a dead NVSHMEM path
        # otherwise wedges the gang at the first MoE dispatch, minutes later.
        ep_state = get_parallel_state()
        if ep_state.ep_enabled:
            from ..distributed.moe.deepep import preflight_internode_transport  # noqa: PLC0415

            preflight_internode_transport(
                ep_state.ep_group,
                hidden_dim=getattr(config, "hidden_size", 0) or 2048,
                buffer_size_gb=deepep_buffer_size_gb,
                num_sms=deepep_num_sms,
            )
        logger.info_rank0(
            f"DeepEP dispatch enabled (buffer={deepep_buffer_size_gb} GB, "
            f"num_sms={deepep_num_sms}, async_combine={deepep_async_combine})"
        )

    # Validate attention implementation for packed sequences with FlashAttention kwargs
    if attn_implementation == "sdpa":
        raise ValueError(
            "attn_implementation='sdpa' is not supported for packed sequences with sequence parallelism. "
            "Please use 'flash_attention_4' (default) or 'flash_attention_3' for correct cu_seqlens handling."
        )

    ps = get_parallel_state()
    if isinstance(config, Glm5Config) and (
        getattr(config, "indexer_types", None) is not None or getattr(config, "mlp_layer_types", None) is not None
    ):
        install_glm52_pipeline_module_plan(
            config,
            num_stages=ps.pp_size * int(pipeline_parallel_virtual_stages),
            input_weight=pipeline_parallel_input_weight,
            output_weight=pipeline_parallel_output_weight,
            num_layers_in_first_stage=pipeline_parallel_num_layers_in_first_stage,
            num_layers_in_last_stage=pipeline_parallel_num_layers_in_last_stage,
        )
    if ps.ringattn_size > 1 and has_linear_attention_layers(config):
        logger.warning_once(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)
        raise ValueError(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)
    validate_glm5_sequence_parallel(config, parallel_state=ps)

    if _is_gpt_oss_config(config) and attn_implementation not in ("eager", "flash_attention_3", "flash_attention_4"):
        raise ValueError(
            "GPT-OSS attention sinks are only implemented for attn_implementation="
            "'eager', 'flash_attention_3', or 'flash_attention_4' in xorl. Using other "
            "backends (sdpa, flash_attention_2, native) would silently drop the "
            "sink logits and change model outputs."
        )

    if _is_minimax_m3_config(config):
        unsupported = (
            ps.tp_size > 1
            or ps.pp_size > 1
            or ps.ringattn_size > 1
            or ps.ulysses_size > 1
            or getattr(ps, "lm_head_tp_size", 1) > 1
        )
        if unsupported:
            raise ValueError(
                "MiniMax M3 xorl support currently supports data/FSDP2 and expert parallelism only; "
                "tensor parallelism, pipeline parallelism, Ulysses, Ring, and lm-head TP are not supported yet."
            )

    loader: ModelLoader = get_loader(config)

    # Validate the requested backend is importable before loading weights, so a
    # missing flash build raises here instead of reaching the first forward.
    get_attention_fn(attn_implementation)
    if attn_implementation == "flash_attention_4":
        logger.info_rank0("Using Flash Attention 4 (CUTE) for attention computation")

    # For HF model init: map all flash variants to "flash_attention_2" (HF's known key).
    # Our own ATTENTION_FUNCTIONS registry handles the real dispatch.
    hf_attn_implementation = attn_implementation
    if attn_implementation in ("flash_attention_3", "flash_attention_4"):
        hf_attn_implementation = "flash_attention_2"
    elif attn_implementation == "minimax_msa":
        hf_attn_implementation = "eager"

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": hf_attn_implementation,
    }

    if (init_device == "cpu" and get_parallel_state().global_rank != 0) or init_device == "meta":
        empty_init = True
    else:
        empty_init = False

    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
    )

    # Set the real implementation name so our model code dispatches correctly
    # via ATTENTION_FUNCTIONS (not HF's ALL_ATTENTION_FUNCTIONS).
    model.config._attn_implementation = attn_implementation

    if config._qwen3_dense_exact_contract:
        from xorl.ops.batch_invariant_ops import (  # noqa: PLC0415
            wrap_trunk_linears_batch_invariant,
        )
        from xorl.ops.bi_families_v2 import (  # noqa: PLC0415
            _select_qwen3_dense_families_v2,
        )

        _select_qwen3_dense_families_v2()
        wrapped = wrap_trunk_linears_batch_invariant(model)
        if not wrapped:
            raise RuntimeError("Exact dense Qwen3 model construction produced no batch-invariant trunk linears")

    return model
