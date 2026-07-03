"""Host-side program builder + loader for the fused training megakernel.

A Program is a list of waves; each wave is a list of instructions; each instruction is
(op, ntiles, args). Buffers (torch CUDA tensors) are registered into a pointer table and
referenced by index. `run()` launches the single cooperative kernel over the stream.
"""

import os
import struct

import torch
from torch.utils.cpp_extension import load


_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirrors `enum Op` in megakernel.cu — keep in sync.
OP_NOP = 0
OP_FILL_F32 = 1
OP_AXPY_F32 = 2
OP_GEMM = 3
OP_RMSNORM_FWD = 4
OP_RMSNORM_BWD = 5
OP_SWIGLU_FWD = 6
OP_SWIGLU_BWD = 7
OP_QKNORM_ROPE_FWD = 8
OP_QKNORM_ROPE_BWD = 9
OP_EMBED_FWD = 10
OP_EMBED_BWD = 11
OP_CE_FWD = 12
OP_CE_BWD = 13
OP_ATTN_FWD = 14
OP_ATTN_DPRE = 15
OP_ATTN_DKV = 16
OP_ATTN_DQ = 17

GEMM_BM, GEMM_BN = 64, 128  # keep in sync with ops.cuh


def gemm_tiles(M, N):
    return ((M + GEMM_BM - 1) // GEMM_BM) * ((N + GEMM_BN - 1) // GEMM_BN)


def gemm_split_k(M, N, K, target_tiles=512):
    """K-slice count that lifts a small-M*N GEMM to ~target_tiles work items."""
    mn = gemm_tiles(M, N)
    sk = max(1, min(target_tiles // mn, (K + 31) // 32))
    return sk


MAX_ARGS = 23
INSTR_INTS = 3 + MAX_ARGS  # op, tile_off, ntiles, args[23]


def load_ext(verbose=False):
    return load(
        name="xorl_megakernel",
        sources=[os.path.join(_DIR, "megakernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90,code=sm_90",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=verbose,
    )


def f2i(x: float) -> int:
    """Reinterpret a float's bits as int32 (for scalar args)."""
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


class Program:
    def __init__(self):
        self.bufs = []
        self._buf_ids = {}
        self.waves = [[]]  # list of waves; each wave = list of (op, ntiles, args)

    def buf(self, t: torch.Tensor) -> int:
        """Register a CUDA tensor; returns its buffer-table index."""
        assert t.is_cuda and t.is_contiguous()
        key = t.data_ptr()
        if key not in self._buf_ids:
            self._buf_ids[key] = len(self.bufs)
            self.bufs.append(t)
        return self._buf_ids[key]

    def instr(self, op: int, ntiles: int, args):
        assert len(args) <= MAX_ARGS and ntiles >= 1
        self.waves[-1].append((op, ntiles, list(args)))

    def wave(self):
        """Close the current wave (a grid.sync boundary)."""
        if self.waves[-1]:
            self.waves.append([])

    def finalize(self, device="cuda"):
        if not self.waves[-1]:
            self.waves.pop()
        instrs, wave_start, wave_tiles = [], [0], []
        for wave in self.waves:
            off = 0
            for op, ntiles, args in wave:
                row = [op, off, ntiles] + args + [0] * (MAX_ARGS - len(args))
                instrs.extend(row)
                off += ntiles
            wave_tiles.append(off)
            wave_start.append(wave_start[-1] + len(wave))
        self._instrs = torch.tensor(instrs, dtype=torch.int32, device=device)
        self._wave_start = torch.tensor(wave_start, dtype=torch.int32, device=device)
        self._wave_tiles = torch.tensor(wave_tiles, dtype=torch.int32, device=device)
        self._buftab = torch.tensor([t.data_ptr() for t in self.bufs], dtype=torch.int64, device=device)
        return self

    def run(self, ext, smem_bytes=100 * 1024, wave_clk=None):
        ext.run(self._instrs, self._wave_start, self._wave_tiles, self._buftab, smem_bytes, wave_clk)


if __name__ == "__main__":
    # Skeleton smoke test: scheduling correctness + grid.sync cost.
    import time

    torch.cuda.set_device(0)
    ext = load_ext(verbose=True)

    # correctness: fill two buffers, axpy them together across many tiles/waves
    n = 1 << 20
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    y = torch.empty(n, device="cuda", dtype=torch.float32)
    p = Program()
    bx, by = p.buf(x), p.buf(y)
    ntiles = (n + 4095) // 4096
    p.instr(OP_FILL_F32, ntiles, [bx, n, f2i(3.0)])
    p.instr(OP_FILL_F32, ntiles, [by, n, f2i(1.5)])
    p.wave()
    p.instr(OP_AXPY_F32, ntiles, [by, bx, n, f2i(2.0)])
    p.wave()
    p.finalize().run(ext)
    torch.cuda.synchronize()
    assert torch.all(y == 7.5), y.unique()
    print(f"scheduling OK (nblocks={ext.nblocks()})")

    # grid.sync overhead: 2000 waves of trivial work
    q = Program()
    bz = q.buf(x)
    for _ in range(2000):
        q.instr(OP_FILL_F32, 1, [bz, 4096, f2i(0.0)])
        q.wave()
    q.finalize()
    q.run(ext)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        q.run(ext)
    torch.cuda.synchronize()
    per_wave_us = (time.time() - t0) / 10 / 2000 * 1e6
    print(f"grid.sync + dispatch overhead: {per_wave_us:.2f} us/wave")
