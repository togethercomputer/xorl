"""Moved: ``Mamba2Mixer`` lives in :mod:`xorl.models.layers.mamba2_mixer` (issue #78 phase 4)."""


def __getattr__(name: str):
    if name == "Mamba2Mixer":
        from xorl.models.layers.mamba2_mixer import Mamba2Mixer

        return Mamba2Mixer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Mamba2Mixer"]
