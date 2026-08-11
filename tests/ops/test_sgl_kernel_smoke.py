"""Fail fast when an installed SGLang kernel wheel has the wrong Torch ABI."""

import importlib.util
import os

import pytest
import torch


def test_sgl_kernel_extension_and_real_operation():
    installed = importlib.util.find_spec("sgl_kernel") is not None
    required = os.environ.get("XORL_REQUIRE_SGL_KERNEL") == "1"
    if not installed:
        if required:
            pytest.fail("XORL_REQUIRE_SGL_KERNEL=1, but sglang-kernel is not installed")
        pytest.skip("sglang-kernel is intentionally absent from the default Torch 2.12 profile")

    try:
        import sgl_kernel
        from sglang.kernels.ops.attention.dsv4 import hash_topk
        from sglang.srt.lora.utils import LoRABatchInfo
    except (ImportError, OSError) as exc:
        pytest.fail(f"sglang-kernel cannot load under torch {torch.__version__}: {exc}")

    assert callable(hash_topk)
    assert LoRABatchInfo.__name__ == "LoRABatchInfo"
    if not torch.cuda.is_available():
        if required:
            pytest.fail("XORL_REQUIRE_SGL_KERNEL=1, but CUDA is unavailable for the real-operation smoke")
        pytest.skip("compiled extension loaded; CUDA is unavailable for the operation smoke")

    hidden = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(64, dtype=torch.bfloat16, device="cuda")
    output = sgl_kernel.rmsnorm(hidden.contiguous(), weight, 1e-6)
    assert output.shape == hidden.shape
    assert torch.isfinite(output.float()).all()
