import importlib


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
]


_QLORA_ATTRS = {
    "BlockFP8QLoRALinear": ("xorl.qlora.modules.block_fp8_linear", "BlockFP8QLoRALinear"),
    "NF4QLoRALinear": ("xorl.qlora.modules.nf4_linear", "NF4QLoRALinear"),
    "NvFP4QLoRALinear": ("xorl.qlora.modules.nvfp4_linear", "NvFP4QLoRALinear"),
    "QLoRALinear": ("xorl.qlora.modules.linear", "QLoRALinear"),
    "QLoRAMoeExperts": ("xorl.qlora.modules.moe_experts", "QLoRAMoeExperts"),
    "prefetch_aqn_noise": ("xorl.qlora.modules.linear", "prefetch_aqn_noise"),
    "detect_prequantized_block_fp8": ("xorl.qlora.utils", "detect_prequantized_block_fp8"),
    "detect_prequantized_nvfp4": ("xorl.qlora.utils", "detect_prequantized_nvfp4"),
    "inject_qlora_into_model": ("xorl.qlora.utils", "inject_qlora_into_model"),
    "maybe_load_and_quantize_moe_qlora": ("xorl.qlora.utils", "maybe_load_and_quantize_moe_qlora"),
    "maybe_load_prequantized_qlora": ("xorl.qlora.utils", "maybe_load_prequantized_qlora"),
    "maybe_quantize_qlora": ("xorl.qlora.utils", "maybe_quantize_qlora"),
    "maybe_requant_qlora": ("xorl.qlora.utils", "maybe_requant_qlora"),
    "save_qlora_checkpoint": ("xorl.qlora.utils", "save_qlora_checkpoint"),
}


def __getattr__(name):
    if name not in _QLORA_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _QLORA_ATTRS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
