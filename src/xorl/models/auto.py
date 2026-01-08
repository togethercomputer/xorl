import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .loader import BaseModelLoader, get_loader


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)


def build_foundation_model(
    config_path: Union[str, PretrainedConfig],
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[
        Literal["eager", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4", "native-sparse"]
    ] = "flash_attention_2",
    moe_implementation: Optional[Literal["eager", "fused"]] = None,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    force_use_huggingface: Optional[bool] = False,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """
    if config_kwargs is None:
        config_kwargs = {}

    if isinstance(config_path, PretrainedConfig):
        config = config_path
    else:
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True, **config_kwargs)

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "fused"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    # Validate attention implementation for packed sequences with FlashAttention kwargs
    if attn_implementation == "sdpa":
        raise ValueError(
            "attn_implementation='sdpa' is not supported for packed sequences with sequence parallelism. "
            "Please use 'flash_attention_2' or 'flash_attention_3' for correct cu_seqlens handling."
        )

    loader: Optional[BaseModelLoader] = get_loader(config, force_use_huggingface)

    if not force_use_huggingface:
        from functools import partial

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from ..ops.attention import FA4_AVAILABLE, flash_attention_forward

        # Check if custom attention implementation was requested (now deprecated)
        custom_attn_impl = os.getenv("SEED_KERNEL_ATTN_IMPLEMENTATION")
        if custom_attn_impl is not None:
            logger.warning(
                f"SEED_KERNEL_ATTN_IMPLEMENTATION={custom_attn_impl} is no longer supported as seed_kernels "
                f"has been removed. Using standard flash_attention_2 implementation."
            )

        # Register FA4 attention function (uses FA4 CUTE kernels)
        if attn_implementation == "flash_attention_4":
            if not FA4_AVAILABLE:
                raise ImportError(
                    "flash_attention_4 requested but flash_attn.cute is not installed. "
                    "Please install FA4 dependencies or use flash_attention_2."
                )
            flash_attention_forward_fa4 = partial(flash_attention_forward, use_fa4=True)
            ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward_fa4)
            logger.info_rank0("Using Flash Attention 4 (CUTE) for attention computation")
        else:
            ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)

    # For HuggingFace model initialization, we always use "flash_attention_2" as the attn_implementation
    # because most HF models don't have flash_attention_3 or flash_attention_4 in their supported list yet.
    # Our registered flash_attention_forward function will automatically use FA3/FA4 when configured.
    hf_attn_implementation = attn_implementation
    if attn_implementation == "flash_attention_3":
        hf_attn_implementation = "flash_attention_2"
        logger.info_rank0(
            "Using flash_attention_2 for HF model init, but FA3 will be used internally when available"
        )
    elif attn_implementation == "flash_attention_4":
        # Use 'eager' for HF init to avoid flash_attn package check, our registered
        # attention function in ALL_ATTENTION_FUNCTIONS will handle the actual computation
        hf_attn_implementation = "eager"
        logger.info_rank0(
            "Using eager for HF model init, but FA4 (CUTE) will be used internally via ALL_ATTENTION_FUNCTIONS"
        )

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": hf_attn_implementation,
        "trust_remote_code": True,
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

    # For FA4: we initialized with 'eager' to bypass flash_attn check,
    # but now override to 'flash_attention_2' so model uses our registered FA4 function
    if attn_implementation == "flash_attention_4":
        model.config._attn_implementation = "flash_attention_2"
        model.config._attn_implementation_internal = "flash_attention_2"
        logger.info_rank0("Overrode config._attn_implementation to 'flash_attention_2' for FA4 dispatch")

    return model
