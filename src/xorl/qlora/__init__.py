"""Lazy exports for QLoRA to avoid package-level circular imports."""

_MODULE_ATTRS = {
    "BlockFP8QLoRALinear",
    "NF4QLoRALinear",
    "NvFP4QLoRALinear",
    "QLoRALinear",
    "QLoRAMoeExperts",
    "prefetch_aqn_noise",
}

_UTIL_ATTRS = {
    "detect_prequantized_block_fp8",
    "detect_prequantized_nvfp4",
    "get_prequantized_exclude_modules",
    "inject_qlora_into_model",
    "maybe_load_and_quantize_moe_qlora",
    "maybe_load_prequantized_qlora",
    "maybe_quantize_qlora",
    "maybe_requant_qlora",
    "save_qlora_checkpoint",
}


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


def __getattr__(name):
    if name in _MODULE_ATTRS:
        from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear  # noqa: PLC0415
        from xorl.qlora.modules.linear import QLoRALinear, prefetch_aqn_noise  # noqa: PLC0415
        from xorl.qlora.modules.moe_experts import QLoRAMoeExperts  # noqa: PLC0415
        from xorl.qlora.modules.nf4_linear import NF4QLoRALinear  # noqa: PLC0415
        from xorl.qlora.modules.nvfp4_linear import NvFP4QLoRALinear  # noqa: PLC0415

        globals().update(
            {
                "QLoRALinear": QLoRALinear,
                "NvFP4QLoRALinear": NvFP4QLoRALinear,
                "BlockFP8QLoRALinear": BlockFP8QLoRALinear,
                "NF4QLoRALinear": NF4QLoRALinear,
                "QLoRAMoeExperts": QLoRAMoeExperts,
                "prefetch_aqn_noise": prefetch_aqn_noise,
            }
        )
        return globals()[name]

    if name in _UTIL_ATTRS:
        from xorl.qlora.utils import (  # noqa: PLC0415
            detect_prequantized_block_fp8,
            detect_prequantized_nvfp4,
            inject_qlora_into_model,
            maybe_load_and_quantize_moe_qlora,
            maybe_load_prequantized_qlora,
            maybe_quantize_qlora,
            maybe_requant_qlora,
            save_qlora_checkpoint,
        )

        globals().update(
            {
                "detect_prequantized_block_fp8": detect_prequantized_block_fp8,
                "detect_prequantized_nvfp4": detect_prequantized_nvfp4,
                "inject_qlora_into_model": inject_qlora_into_model,
                "maybe_load_and_quantize_moe_qlora": maybe_load_and_quantize_moe_qlora,
                "maybe_load_prequantized_qlora": maybe_load_prequantized_qlora,
                "maybe_quantize_qlora": maybe_quantize_qlora,
                "maybe_requant_qlora": maybe_requant_qlora,
                "save_qlora_checkpoint": save_qlora_checkpoint,
            }
        )

        if "get_prequantized_exclude_modules" not in globals():
            from xorl.models.checkpoint_handlers.buffers import (  # noqa: PLC0415
                get_prequantized_exclude_modules,
            )

            globals()["get_prequantized_exclude_modules"] = get_prequantized_exclude_modules

        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
