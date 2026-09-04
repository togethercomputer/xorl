"""Triton philox4x32-10 standard-normal generator for ZORL noise layout v2.

One kernel: philox rounds live in registers, Box-Muller in fp32, a single
store of the output — vs the torch reference in ``zorl.py`` whose elementwise
int64 chain round-trips every round through HBM (~2KB traffic per counter).
The (key, counter) addressing and the fp32 uniform->normal pipeline are the
same as ``zorl_philox_randn_batch``; parity is gated in
tests/server/test_zorl_fresh_ab.py (GPU) — u32 bits must match exactly, the
normals to <=2e-6 (libdevice log/cos vs torch may differ by ULPs).

This is also the stepping stone to the fully fused generate+fold kernel: the
per-(key, counter-block) program here is exactly the generator fragment a
fused GEMM tile embeds.
"""
from __future__ import annotations

from typing import List, Union

import torch

try:  # pragma: no cover - import guard exercised implicitly
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _philox_randn_kernel(
        out_ptr,
        keys_ptr,
        numel,
        n_counters,
        counter_offset,
        BLOCK: tl.constexpr,
    ):
        pid_key = tl.program_id(0)
        pid_blk = tl.program_id(1)
        key = tl.load(keys_ptr + pid_key)
        k0 = (key & 0xFFFFFFFF).to(tl.uint32)
        k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)

        cpos = pid_blk * BLOCK + tl.arange(0, BLOCK)
        cmask = cpos < n_counters
        cidx = (counter_offset + cpos).to(tl.int64)
        c0 = (cidx & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((cidx >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c2 = tl.zeros([BLOCK], dtype=tl.uint32)
        c3 = tl.zeros([BLOCK], dtype=tl.uint32)

        M0: tl.constexpr = 0xD2511F53
        M1: tl.constexpr = 0xCD9E8D57
        W0: tl.constexpr = 0x9E3779B9
        W1: tl.constexpr = 0xBB67AE85
        for _ in tl.static_range(10):
            hi0 = tl.umulhi(c0, M0)
            lo0 = c0 * M0
            hi1 = tl.umulhi(c2, M1)
            lo1 = c2 * M1
            nc0 = hi1 ^ c1 ^ k0
            nc2 = hi0 ^ c3 ^ k1
            c0, c1, c2, c3 = nc0, lo1, nc2, lo0
            k0 = k0 + W0
            k1 = k1 + W1

        inv = 2.3283064365386963e-10  # 1 / 2**32
        u0 = (c0.to(tl.float32) + 0.5) * inv
        u1 = (c1.to(tl.float32) + 0.5) * inv
        u2 = (c2.to(tl.float32) + 0.5) * inv
        u3 = (c3.to(tl.float32) + 0.5) * inv
        TWO_PI: tl.constexpr = 6.2831855
        r0 = tl.sqrt(-2.0 * tl.log(u0))
        t0 = TWO_PI * u1
        r1 = tl.sqrt(-2.0 * tl.log(u2))
        t1 = TWO_PI * u3
        z0 = r0 * tl.cos(t0)
        z1 = r0 * tl.sin(t0)
        z2 = r1 * tl.cos(t1)
        z3 = r1 * tl.sin(t1)

        base = pid_key.to(tl.int64) * numel + cpos.to(tl.int64) * 4
        tl.store(out_ptr + base + 0, z0, mask=cmask & (cpos * 4 + 0 < numel))
        tl.store(out_ptr + base + 1, z1, mask=cmask & (cpos * 4 + 1 < numel))
        tl.store(out_ptr + base + 2, z2, mask=cmask & (cpos * 4 + 2 < numel))
        tl.store(out_ptr + base + 3, z3, mask=cmask & (cpos * 4 + 3 < numel))


def philox_randn_batch_triton(
    sub_seeds: List[int],
    numel: int,
    *,
    device: Union[str, torch.device],
    counter_offset: int = 0,
) -> torch.Tensor:
    """Kernel twin of ``zorl_philox_randn_batch`` (CUDA only)."""
    if not HAS_TRITON:
        raise RuntimeError("triton is not available")
    B = len(sub_seeds)
    out = torch.empty(B, numel, dtype=torch.float32, device=device)
    if numel <= 0 or B == 0:
        return out
    keys = torch.tensor(sub_seeds, device=device, dtype=torch.int64)
    n_counters = (numel + 3) // 4
    BLOCK = 1024
    grid = (B, triton.cdiv(n_counters, BLOCK))
    _philox_randn_kernel[grid](out, keys, numel, n_counters, counter_offset, BLOCK=BLOCK)
    return out
