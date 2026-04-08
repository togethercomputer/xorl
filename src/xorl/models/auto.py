import types
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
from .transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from .transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from .transformers.qwen3_5_shared import (
    LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE,
    has_linear_attention_layers,
)


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


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

    if model_type == "qwen3_5_moe":
        return Qwen3_5MoeConfig.from_hf_config(_namespace_from_dict(config_dict))

    if model_type == "qwen3_5":
        return Qwen3_5Config.from_hf_config(_namespace_from_dict(config_dict))

    return None


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
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
    deepep_buffer_size_gb: float = 2.0,
    deepep_num_sms: int = 20,
    deepep_async_combine: bool = False,
    router_fp32: bool = True,
    lm_head_fp32: bool = True,
    rmsnorm_mode: Literal["eager", "native", "compile"] = "native",
    activation_native: bool = False,
    rope_native: bool = False,
    attention_cast_bf16: bool = False,
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

    if ep_dispatch == "deepep" and train_router:
        raise ValueError(
            "train_router=True is not supported with ep_dispatch='deepep'. "
            "Set train_router=False or use ep_dispatch='alltoall'."
        )

    config._ep_dispatch = ep_dispatch
    config.train_router = train_router
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
