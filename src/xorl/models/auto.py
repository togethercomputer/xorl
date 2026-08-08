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
from .exact_contract import glm52_exact_active_lora_enabled, glm52_exact_forward_enabled, set_glm52_exact_active_lora
from .layers.attention import get_attention_fn
from .layers.normalization import set_rmsnorm_mode
from .layers.rope import set_rope_class_b, set_rope_native
from .loader import ModelLoader, get_loader
from .transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from .transformers.deepseek_v3.support import validate_deepseek_v3_router_settings
from .transformers.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from .transformers.glm5.configuration_glm5 import Glm5Config
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
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "index_topk": 2048,
        "index_topk_freq": 4,
    }
    mismatches = [
        f"{name}={getattr(config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    indexer_types = tuple(config.indexer_types)
    expected_indexer_types = tuple(
        "full" if layer_idx < 3 or (layer_idx - 2) % 4 == 0 else "shared" for layer_idx in range(78)
    )
    if indexer_types != expected_indexer_types:
        mismatches.append("indexer_types does not match the official 78-layer selector schedule")
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    expected_mlp_types = ("dense",) * 3 + ("sparse",) * 75
    if mlp_layer_types is None or tuple(mlp_layer_types) != expected_mlp_types:
        mismatches.append("mlp_layer_types does not match 3 dense + 75 sparse blocks")
    if mismatches:
        raise ValueError(
            "The exact GLM-5.2 program supports only the official model geometry: " + ", ".join(mismatches)
        )


def _is_exact_qwen35(config: PretrainedConfig) -> bool:
    return bool(getattr(config, "_qwen35_exact_contract", False))


def _is_qwen35_moe(config: PretrainedConfig) -> bool:
    return getattr(config, "model_type", None) in {
        "xorl_qwen3_5_moe",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }


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
        model_name = "Qwen3.6-35B-A3B"
    else:
        expected = {
            "vocab_size": 248320,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
        }
        model_name = "Qwen3.5-0.8B"
    mismatches = [
        f"{name}={getattr(scope_config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(scope_config, name, None) != value
    ]
    expected_layer_types = tuple(
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(expected["num_hidden_layers"])
    )
    # ``full_attention_interval`` uniquely defines this schedule.  Some HF
    # config representations materialize ``layer_types`` while others retain
    # only the interval, so validate the derived list when it is present rather
    # than rejecting an otherwise identical checkpoint representation.
    layer_types = getattr(scope_config, "layer_types", None)
    if layer_types is not None and tuple(layer_types) != expected_layer_types:
        mismatches.append("layer_types does not match the certified full-attention interval")
    if mismatches:
        raise ValueError(
            f"The exact Qwen3.5-family server-training program is certified only for {model_name}: "
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


def _validate_exact_qwen35_topology(config: PretrainedConfig, parallel_state: Any) -> None:
    if not _is_exact_qwen35(config):
        return
    topology = (
        parallel_state.world_size,
        parallel_state.dp_size,
        parallel_state.dp_replicate_size,
        parallel_state.dp_shard_size,
        parallel_state.tp_size,
        parallel_state.pp_size,
        parallel_state.ep_size,
        parallel_state.cp_size,
        parallel_state.ringattn_size,
        parallel_state.ulysses_size,
    )
    admitted = (
        (
            (8, 8, 1, 8, 1, 1, 8, 1, 1, 1),
            (16, 16, 2, 8, 1, 1, 8, 1, 1, 1),
        )
        if _is_qwen35_moe(config)
        else ((1, 1, 1, 1, 1, 1, 1, 1, 1, 1),)
    )
    if topology not in admitted:
        raise ValueError(
            "The Qwen3.5-family exact server-training path is admitted only for "
            "WORLD/DP/DP-replicate/DP-shard/TP/PP/EP/CP/Ring/Ulysses="
            f"{admitted}; got {topology}"
        )


@dataclass(frozen=True)
class ResolvedModelNumericalProgram:
    """Bit-relevant model choices resolved before module construction."""

    attn_implementation: str
    router_fp32: bool
    lm_head_fp32: bool
    rmsnorm_mode: str
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
        if rope_native is False:
            raise ValueError("Exact Qwen3.5-family server training requires rope_native=true")
        if rope_class_b is True:
            raise ValueError("Exact Qwen3.5-family server training uses native Class-A RoPE, not Class-B RoPE")
        return True, False

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
            activation_native=True,
            rope_native=True,
            rope_class_b=False,
            attention_cast_bf16=True,
            sparse_mla_enabled=False,
            sparse_mla_backend="auto",
        )

    if not _is_exact_glm52(config):
        return ResolvedModelNumericalProgram(
            attn_implementation=attn_implementation or non_glm_attn_default,
            router_fp32=True if router_fp32 is None else router_fp32,
            lm_head_fp32=True if lm_head_fp32 is None else lm_head_fp32,
            rmsnorm_mode=rmsnorm_mode or "native",
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
        activation_native=False,
        rope_native=True,
        rope_class_b=True,
        attention_cast_bf16=False,
        sparse_mla_enabled=True,
        sparse_mla_backend="flashmla",
    )


def resolve_cross_entropy_mode(config: PretrainedConfig, ce_mode: Optional[str]) -> str:
    """Resolve the loss-side member of the canonical numerical program."""

    if _is_exact_qwen35(config):
        if ce_mode not in (None, "bi_fused"):
            raise ValueError(f"Exact Qwen3.5-family server training requires ce_mode='bi_fused'; received {ce_mode!r}")
        return "bi_fused"
    if not _is_exact_glm52(config):
        return ce_mode or "compiled"
    if ce_mode not in (None, "bi_fused"):
        raise ValueError(f"Canonical GLM-5.2 exact forward requires ce_mode='bi_fused'; received {ce_mode!r}")
    return "bi_fused"


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
    activation_native: bool = False,
    rope_native: Optional[bool] = None,
    rope_class_b: Optional[bool] = None,
    attention_cast_bf16: bool = False,
    sparse_mla_enabled: Optional[bool] = None,
    sparse_mla_backend: Optional[str] = None,
    flash_attention_deterministic: bool = False,
    server_training: bool = False,
    block_fp8_qlora_training: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
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
    exact_active_lora = bool(
        server_training and glm52_model and block_fp8_qlora_training and (lora_rank, lora_alpha) == (1, 1)
    )
    config._glm52_block_fp8_qlora = bool(block_fp8_qlora_training)
    config._glm52_exact_contract = bool(server_training and glm52_model and not block_fp8_qlora_training)
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
    config._qwen35_exact_contract = bool(server_training and qwen35_model_type)
    _validate_exact_qwen35_model_scope(config)
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
        logger.info_rank0(f"Exact Qwen3.5-family server-training numerical program: {numerical_program}")

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
    _validate_exact_qwen35_topology(config, ps)
    if isinstance(config, Glm5Config) and config.num_hidden_layers == 78:
        if glm52_exact_active_lora_enabled(config):
            topology = (
                ps.world_size,
                ps.pp_size,
                ps.tp_size,
                ps.dp_size,
                ps.ep_size,
                ps.cp_size,
            )
            certified = (16, 1, 1, 1, 16, 16)
            lm_head_tp_size = getattr(ps, "lm_head_tp_size", 1)
            if topology != certified or ps.ringattn_size != 1 or ps.ulysses_size != 16 or lm_head_tp_size != 16:
                raise ValueError(
                    "The GLM-5.2 exact active-LoRA path is certified only for "
                    f"WORLD/PP/TP/DP/EP/CP={certified} with Ring1/Ulysses16/lm-head-TP16; got {topology} "
                    f"with Ring{ps.ringattn_size}/Ulysses{ps.ulysses_size}/lm-head-TP{lm_head_tp_size}"
                )
            config._glm52_pipeline_layer_ranges = ((0, 78),)
        elif canonical_glm52 and server_training:
            topology = (
                ps.world_size,
                ps.pp_size,
                ps.tp_size,
                ps.dp_size,
                ps.ep_size,
                ps.cp_size,
            )
            certified = (16, 1, 1, 1, 16, 16)
            if topology != certified or ps.ringattn_size != 1 or ps.ulysses_size != 16:
                raise ValueError(
                    "The GLM-5.2 exact server-training path is certified only for "
                    f"WORLD/PP/TP/DP/EP/CP={certified} with Ring1/Ulysses16; got {topology} "
                    f"with Ring{ps.ringattn_size}/Ulysses{ps.ulysses_size}"
                )
            config._glm52_pipeline_layer_ranges = ((0, 78),)
        elif ps.pp_size == 1:
            config._glm52_pipeline_layer_ranges = ((0, 78),)
        elif ps.pp_size == 2:
            config._glm52_pipeline_layer_ranges = ((0, 38), (38, 78))
        else:
            raise ValueError("GLM-5.2 supports only PP1 or the supported 38/40 PP2 split")
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

    return model
