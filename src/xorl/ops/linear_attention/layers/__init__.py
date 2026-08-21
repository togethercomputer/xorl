"""Moved: ``GatedDeltaNet`` lives in :mod:`xorl.models.layers.gated_deltanet` (issue #78 phase 4)."""


def __getattr__(name: str):
    if name == "GatedDeltaNet":
        from xorl.models.layers.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GatedDeltaNet"]
