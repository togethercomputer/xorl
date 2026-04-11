import importlib


__all__ = [
    "QLoRALinear",
    "NvFP4QLoRALinear",
    "BlockFP8QLoRALinear",
    "NF4QLoRALinear",
    "QLoRAMoeExperts",
    "prefetch_aqn_noise",
]


_MODULE_ATTRS = {
    "BlockFP8QLoRALinear": ("xorl.qlora.modules.block_fp8_linear", "BlockFP8QLoRALinear"),
    "NF4QLoRALinear": ("xorl.qlora.modules.nf4_linear", "NF4QLoRALinear"),
    "NvFP4QLoRALinear": ("xorl.qlora.modules.nvfp4_linear", "NvFP4QLoRALinear"),
    "QLoRALinear": ("xorl.qlora.modules.linear", "QLoRALinear"),
    "QLoRAMoeExperts": ("xorl.qlora.modules.moe_experts", "QLoRAMoeExperts"),
    "prefetch_aqn_noise": ("xorl.qlora.modules.linear", "prefetch_aqn_noise"),
}


def __getattr__(name):
    if name not in _MODULE_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _MODULE_ATTRS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
