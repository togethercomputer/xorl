"""Gate: the shape-keyed BI GEMM config table is bit-neutral.

Locks the doctrine that makes the table legal: BLOCK_SIZE_K (pinned per dtype)
is the only bit-relevant axis of matmul_kernel_persistent. Every table/class
config must produce torch.equal outputs vs the pinned baseline, rows must be
bit-invariant across M (including across table bucket boundaries), and the
DeepGEMM route (now reachable in the trainer) must be bitwise-equal to Triton.
"""

import pytest
import torch
import triton

from xorl.ops.exact.bi_gemm_configs import BASELINE_CONFIG, PINNED_BLOCK_K, lookup_mm_config
from xorl.ops.sglang.batch_invariant_ops import (
    _deepgemm_ready,
    _matmul_persistent_deepgemm,
    matmul_kernel_persistent,
    set_batch_invariant_mode,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SHAPES = [
    # (M, K, N) spanning decode, bucket boundaries, and batch classes
    (1, 3840, 3840),
    (16, 3840, 11520),
    (64, 15360, 3840),
    (300, 3840, 3840),  # crosses the 256 bucket edge
    (4096, 3840, 11520),
    (3072, 3840, 8192),  # BI lm-head chunk shape (fp32-out in production)
]


def _launch(a, b, cfg, out_dtype=None):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype or a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    def grid(meta):
        return (min(num_sms, triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])),)

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        None,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        A_LARGE=False,
        B_LARGE=False,
        C_LARGE=False,
        HAS_BIAS=False,
        **cfg,
    )
    return c


def _inputs(M, K, N, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((M, K), device="cuda", generator=g, dtype=torch.float32).to(dtype)
    w = torch.randn((N, K), device="cuda", generator=g, dtype=torch.float32).to(dtype)
    return a, w.t()


@requires_cuda
@pytest.mark.gpu
def test_table_config_bitwise_equals_baseline():
    # This is one doctrine-level contract: every generated table entry must be
    # bit-neutral relative to the dtype-pinned reduction tree.
    for dtype in (torch.bfloat16, torch.float32):
        for shape in SHAPES:
            M, K, N = shape
            a, b = _inputs(M, K, N, dtype)
            base = dict(BASELINE_CONFIG[str(dtype)], BLOCK_SIZE_K=PINNED_BLOCK_K[str(dtype)])
            cfg = lookup_mm_config(dtype, M, N, K)
            out_base = _launch(a, b, base)
            out_cfg = _launch(a, b, cfg)
            assert torch.equal(out_base, out_cfg), f"table config moved bits at {dtype} {shape}: {cfg}"
            # fp32-out store path (the BI lm-head chunk form): mirror the production
            # launcher — table config with OutOfResources->baseline fallback — and
            # assert bit-neutrality down to raw accumulator bits
            if dtype == torch.bfloat16:
                from triton.runtime.errors import OutOfResources  # noqa: PLC0415

                cfg32 = lookup_mm_config(dtype, M, N, K, out_itemsize=4)
                try:
                    out32 = _launch(a, b, cfg32, torch.float32)
                except OutOfResources:
                    out32 = _launch(a, b, base, torch.float32)
                assert torch.equal(_launch(a, b, base, torch.float32), out32), shape
    _assert_rows_invariant_across_m_and_buckets()


def _assert_rows_invariant_across_m_and_buckets():
    K, N = 3840, 3840
    g = torch.Generator(device="cuda").manual_seed(3)
    w = torch.randn((N, K), device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
    row = torch.randn((1, K), device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
    outs = []
    with set_batch_invariant_mode(True):
        for M in (1, 8, 64, 300, 4096):
            a = torch.randn((M, K), device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
            a[0] = row[0]
            outs.append(torch.mm(a, w.t())[0])
    for o in outs[1:]:
        assert torch.equal(outs[0], o), "row bits changed across M/table buckets"


@requires_cuda
@pytest.mark.gpu
def test_deepgemm_bitwise_equals_triton():
    if not _deepgemm_ready():
        pytest.skip("deep_gemm unavailable")
    for M, K, N in SHAPES:
        a, b = _inputs(M, K, N, torch.bfloat16, seed=1)
        base = dict(BASELINE_CONFIG["torch.bfloat16"], BLOCK_SIZE_K=64)
        out_dg = _matmul_persistent_deepgemm(a, b)
        assert out_dg is not None
        assert torch.equal(out_dg, _launch(a, b, base)), f"deep_gemm bits differ at {(M, K, N)}"
