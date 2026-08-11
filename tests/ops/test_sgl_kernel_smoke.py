"""Fail-fast sgl_kernel ABI smoke.

The pinned SGLang tree's compiled extension (``sglang-kernel``) is built
against the pinned SGLang torch ABI. Installing it into a different torch
profile produces an extension that fails only when the shared object is
finally loaded, while the pure-Triton ``sglang.kernels`` wrappers import
fine — which reads as a working environment until a serving-value forward
dies mid-run.

This test makes that state loud in whichever environment carries the
package: if the ``sglang-kernel`` distribution is installed, its compiled
ops must actually load and execute. An environment without the package
(the default torch-2.12 trainer profile) skips; DSV4 exact-kernel work
runs in the isolated torch-2.11 combined environment at
``submodules/xorl-sglang/.venv``, where this test must pass.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch


def _sgl_kernel_installed() -> bool:
    return importlib.util.find_spec("sgl_kernel") is not None


@pytest.mark.skipif(
    not _sgl_kernel_installed(),
    reason="sglang-kernel is not part of this profile; DSV4 exact kernels run "
    "in the torch-2.11 combined environment (submodules/xorl-sglang/.venv).",
)
def test_sgl_kernel_extension_loads_and_executes() -> None:
    # Importing sgl_kernel eagerly loads the architecture-specific compiled
    # extension; a torch-ABI mismatch raises ImportError here, not later.
    try:
        import sgl_kernel
    except ImportError as exc:
        pytest.fail(
            "sglang-kernel is installed but its compiled extension cannot "
            f"load under torch {torch.__version__}. Remove the package from "
            "this profile or run in the torch it was built against. "
            f"Loader error: {exc}"
        )

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; extension load alone verified.")

    hidden = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(64, dtype=torch.bfloat16, device="cuda")
    normed = sgl_kernel.rmsnorm(hidden.contiguous(), weight, 1e-6)
    assert normed is not None
    assert normed.shape == hidden.shape
    assert torch.isfinite(normed.float()).all()
