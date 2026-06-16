import json
import types
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
from ..utils import logging
from .layers.attention import ATTENTION_FUNCTIONS
from .layers.normalization import set_rmsnorm_mode
from .loader import ModelLoader, get_loader
from .transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from .transformers.deepseek_v3.support import validate_deepseek_v3_router_settings
from .transformers.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from .transformers.gpt_oss.configuration_gpt_oss import GptOssConfig
from .transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from .transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from .transformers.qwen3_5_shared import (
    LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE,
    has_linear_attention_layers,
)


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


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
    model_type = config_dict.get("model_type")

    if model_type == "glm4_moe":
        return Glm4MoeConfig.from_dict(config_dict)

    if model_type == "qwen3_5_moe":
        return Qwen3_5MoeConfig.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "qwen3_5":
        return Qwen3_5Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type in {"deepseek_v3", "kimi_k2", "kimi_k25"}:
        return DeepseekV3Config.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "qwen2":
        from .transformers.qwen2.configuration_qwen2 import Qwen2Config  # noqa: PLC0415

        return Qwen2Config(**{k: v for k, v in config_dict.items() if not k.startswith("_")})

    if model_type == "gpt_oss":
        return GptOssConfig.from_hf_config(_namespace_from_dict(config_dict))
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


def build_foundation_model(
    config_path: Union[str, PretrainedConfig],
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[
        Literal["eager", "sdpa", "native", "flash_attention_3", "flash_attention_4"]
    ] = "flash_attention_3",
    moe_implementation: Optional[Literal["eager", "triton", "native", "quack"]] = None,
    ep_dispatch: str = "alltoall",
    train_router: bool = False,
    record_routing_weights: bool = True,
    deepep_buffer_size_gb: float = 2.0,
    deepep_num_sms: int = 20,
    deepep_async_combine: bool = False,
    router_fp32: bool = True,
    lm_head_fp32: bool = True,
    rmsnorm_mode: Literal["eager", "native", "compile"] = "native",
    activation_native: bool = False,
    rope_native: bool = False,
    attention_cast_bf16: bool = False,
    flash_attention_deterministic: bool = False,
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

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "triton", "native", "quack"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    validate_deepseek_v3_router_settings(config, train_router=train_router)

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
    config._router_fp32 = router_fp32
    config._lm_head_fp32 = lm_head_fp32
    set_rmsnorm_mode(rmsnorm_mode)
    config._rmsnorm_mode = rmsnorm_mode
    config._activation_native = activation_native
    config._rope_native = rope_native
    config._attention_cast_bf16 = attention_cast_bf16
    config._flash_attention_deterministic = flash_attention_deterministic

    if ep_dispatch == "deepep":
        logger.info_rank0(
            f"DeepEP dispatch enabled (buffer={deepep_buffer_size_gb} GB, "
            f"num_sms={deepep_num_sms}, async_combine={deepep_async_combine})"
        )

    # Validate attention implementation for packed sequences with FlashAttention kwargs
    if attn_implementation == "sdpa":
        raise ValueError(
            "attn_implementation='sdpa' is not supported for packed sequences with sequence parallelism. "
            "Please use 'flash_attention_3' for correct cu_seqlens handling."
        )

    ps = get_parallel_state()
    if ps.ringattn_size > 1 and has_linear_attention_layers(config):
        logger.warning_once(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)
        raise ValueError(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)

    if _is_gpt_oss_config(config) and attn_implementation not in ("eager", "flash_attention_3"):
        raise ValueError(
            "GPT-OSS attention sinks are only implemented for attn_implementation="
            "'eager' or 'flash_attention_3' in xorl. Using other backends (sdpa, "
            "flash_attention_2, flash_attention_4, native) would silently drop the "
            "sink logits and change model outputs."
        )

    loader: ModelLoader = get_loader(config)

    # Validate FA4 availability early
    if attn_implementation == "flash_attention_4":
        if "flash_attention_4" not in ATTENTION_FUNCTIONS:
            raise ImportError(
                "flash_attention_4 requested but flash_attn.cute is not installed. "
                "Please install FA4 dependencies or use flash_attention_3."
            )
        logger.info_rank0("Using Flash Attention 4 (CUTE) for attention computation")

    # For HF model init: map all flash variants to "flash_attention_2" (HF's known key).
    # Our own ATTENTION_FUNCTIONS registry handles the real dispatch.
    hf_attn_implementation = attn_implementation
    if attn_implementation in ("flash_attention_3", "flash_attention_4"):
        hf_attn_implementation = "flash_attention_2"

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
