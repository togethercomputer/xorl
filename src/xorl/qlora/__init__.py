from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules

from .modules.block_fp8_linear import BlockFP8QLoRALinear
from .modules.linear import QLoRALinear, prefetch_aqn_noise
from .modules.moe_experts import QLoRAMoeExperts
from .modules.nf4_linear import NF4QLoRALinear
from .modules.nvfp4_linear import NvFP4QLoRALinear
from .utils import (
    detect_prequantized_block_fp8,
    detect_prequantized_nvfp4,
    inject_qlora_into_model,
    maybe_load_and_quantize_moe_qlora,
    maybe_load_prequantized_qlora,
    maybe_quantize_qlora,
    maybe_requant_qlora,
    save_qlora_checkpoint,
)


__all__ = [
    "QLoRALinear",
    "NvFP4QLoRALinear",
    "BlockFP8QLoRALinear",
    "NF4QLoRALinear",
    "QLoRAMoeExperts",
    "prefetch_aqn_noise",
    "inject_qlora_into_model",
    "save_qlora_checkpoint",
    "maybe_requant_qlora",
    "maybe_quantize_qlora",
    "maybe_load_and_quantize_moe_qlora",
    "detect_prequantized_nvfp4",
    "detect_prequantized_block_fp8",
    "maybe_load_prequantized_qlora",
    "get_prequantized_exclude_modules",
]
