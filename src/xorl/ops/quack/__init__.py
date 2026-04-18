__version__ = "0.2.9"

import os

from .cross_entropy import cross_entropy
from .rmsnorm import rmsnorm
from .softmax import softmax


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from . import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
]
