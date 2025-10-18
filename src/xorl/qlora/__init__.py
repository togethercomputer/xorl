from .modules.linear import QLoRALinear, prefetch_aqn_noise
from .modules.nvfp4_linear import NvFP4QLoRALinear
from .modules.block_fp8_linear import BlockFP8QLoRALinear
from .modules.nf4_linear import NF4QLoRALinear
from .modules.moe_experts import QLoRAMoeExperts
from .utils import (
    inject_qlora_into_model,
    save_qlora_checkpoint,
    maybe_requant_qlora,
    maybe_quantize_qlora,
    maybe_load_and_quantize_moe_qlora,
    detect_prequantized_nvfp4,
    detect_prequantized_block_fp8,
    maybe_load_prequantized_qlora,
)
from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules

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
