"""Lazy exports for runner adapter modules."""

__all__ = ["LoRAAdapterManager", "AdapterCoordinator"]


def __getattr__(name):
    if name == "LoRAAdapterManager":
        from xorl.server.runner.adapters.manager import LoRAAdapterManager  # noqa: PLC0415

        globals()["LoRAAdapterManager"] = LoRAAdapterManager
        return LoRAAdapterManager
    if name == "AdapterCoordinator":
        from xorl.server.runner.adapters.adapter_coordinator import AdapterCoordinator  # noqa: PLC0415

        globals()["AdapterCoordinator"] = AdapterCoordinator
        return AdapterCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
