"""Mamba2 (SSD) kernels used by Nemotron-H.

The ``Mamba2Mixer`` layer class moved to
:mod:`xorl.models.layers.mamba2_mixer` (issue #78 phase 4); it is re-exported
here lazily for one deprecation cycle.
"""

from .modules.gated_norm import GroupRMSNormGated
from .ops.conv import causal_depthwise_conv1d
from .ops.ssd import ssd_chunked


def __getattr__(name: str):
    if name == "Mamba2Mixer":
        from xorl.models.layers.mamba2_mixer import Mamba2Mixer

        return Mamba2Mixer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GroupRMSNormGated",
    "Mamba2Mixer",
    "causal_depthwise_conv1d",
    "ssd_chunked",
]
