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
OP_CVT_F32BF16 = 18

GEMM_BM, GEMM_BN = 64, 128  # keep in sync with ops.cuh
FILL_CHUNK = 16384  # elements per fill/cvt work item (MK_CHUNK in ops.cuh)


def chunk_tiles(n):
    return (n + FILL_CHUNK - 1) // FILL_CHUNK


def gemm_tiles(M, N):
    return ((M + GEMM_BM - 1) // GEMM_BM) * ((N + GEMM_BN - 1) // GEMM_BN)


def wgmma_ok(M, N, K, flags):
    """NT gemms with wgmma-friendly shapes route to the 128x128 warpgroup path."""
    return (flags & 2) and not (flags & 1) and not (flags & 32) and M % 128 == 0 and N % 128 == 0 and K % 64 == 0


def gemm_tiles_wgmma(M, N):
    return (M // 128) * (N // 128)


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
            "-arch=sm_90a",  # wgmma needs the 90a feature set
            "-I/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=verbose,
    )


def f2i(x: float) -> int:
    """Reinterpret a float's bits as int32 (for scalar args)."""
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


def _access_sets(op, args):
    """(read_arg_positions, write_arg_positions) for dependency analysis.

    Writes are treated as read+write (covers accumulation/atomics). Positions index
    into `args`; each named position must hold a buffer-table id.
    """
    if op == OP_NOP:
        return [], []
    if op == OP_FILL_F32:
        return [], [0]
    if op == OP_AXPY_F32:
        return [1], [0]
    if op == OP_GEMM:
        flags = args[6]
        r, w = [0, 1], [2]
        if flags & 16:
            r.append(7)
        return r, w
    if op == OP_RMSNORM_FWD:
        return [0, 1], [2, 3]
    if op == OP_RMSNORM_BWD:
        return [0, 1, 2, 5], [3, 4]
    if op == OP_SWIGLU_FWD:
        return [0], [1]
    if op == OP_SWIGLU_BWD:
        return [0, 1], [2]
    if op == OP_QKNORM_ROPE_FWD:
        return [0, 2, 3, 6, 7], [1, 4, 5]
    if op == OP_QKNORM_ROPE_BWD:
        return [0, 1, 3, 4, 7, 8, 9, 10], [2, 5, 6]
    if op == OP_EMBED_FWD:
        return [0, 1], [2]
    if op == OP_EMBED_BWD:
        return [0, 1], [2]
    if op == OP_CE_FWD:
        return [0, 1, 4], [2, 3]
    if op == OP_CE_BWD:
        return [1, 2, 3], [0]
    if op == OP_ATTN_FWD:
        return [0], [1, 2]
    if op == OP_ATTN_DPRE:
        return [0, 1], [2]
    if op in (OP_ATTN_DKV, OP_ATTN_DQ):
        return [0, 1, 2, 3], [4]
    if op == OP_CVT_F32BF16:
        return [0], [1]
    raise ValueError(f"no access signature for op {op}")


class Program:
    def __init__(self):
        self.bufs = []
        self._buf_ids = {}
        self._buf_meta = []  # (root_ptr, slot) per buffer-table entry
        self.waves = [[]]  # list of waves; each wave = list of (op, ntiles, args)

    def buf(self, t: torch.Tensor, slot=None) -> int:
        """Register a CUDA tensor; returns its buffer-table index.

        `slot` declares a named disjoint region of the tensor (e.g. the q vs kv halves
        of the packed dqkv buffer): entries of the SAME tensor with two different
        non-None slots are treated as non-conflicting by the dependency analysis.
        """
        assert t.is_cuda and t.is_contiguous()
        key = (t.data_ptr(), slot)
        if key not in self._buf_ids:
            self._buf_ids[key] = len(self.bufs)
            self.bufs.append(t)
            self._buf_meta.append((t.data_ptr(), slot))
        return self._buf_ids[key]

    def instr(self, op: int, ntiles: int, args):
        assert len(args) <= MAX_ARGS and ntiles >= 1
        self.waves[-1].append((op, ntiles, list(args)))

    def wave(self):
        """Close the current wave (a grid.sync boundary; ignored in dataflow mode)."""
        if self.waves[-1]:
            self.waves.append([])

    def _build_deps(self, flat):
        """RAW/WAR/WAW dependency DAG from per-op access signatures."""
        history = {}  # root_ptr -> list of (instr_idx, is_write, slot)
        deps = [set() for _ in flat]
        for idx, (op, _ntiles, args) in enumerate(flat):
            r_pos, w_pos = _access_sets(op, args)
            accesses = [(args[p], False) for p in r_pos] + [(args[p], True) for p in w_pos]
            for buf_id, is_write in accesses:
                root, slot = self._buf_meta[buf_id]
                for prior_idx, prior_write, prior_slot in history.get(root, ()):
                    if not (is_write or prior_write):
                        continue  # read-read never conflicts
                    if slot is not None and prior_slot is not None and slot != prior_slot:
                        continue  # declared-disjoint regions
                    if prior_idx != idx:
                        deps[idx].add(prior_idx)
            for buf_id, is_write in accesses:
                root, _ = self._buf_meta[buf_id]
                history.setdefault(root, []).append((idx, is_write, self._buf_meta[buf_id][1]))
        return deps

    def finalize(self, device="cuda"):
        if not self.waves[-1]:
            self.waves.pop()
        flat = [ins for wave in self.waves for ins in wave]

        # wave-mode arrays
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

        # dataflow-mode arrays: same instruction order, dependency DAG instead of waves
        deps = self._build_deps(flat)
        n = len(flat)
        dep_cnt = [len(d) for d in deps]
        dependents = [[] for _ in range(n)]  # forward edges
        for i, d in enumerate(deps):
            for j in d:
                dependents[j].append(i)
        adj_off, adj = [0], []
        for i in range(n):
            adj.extend(sorted(dependents[i]))
            adj_off.append(len(adj))
        claim = [max(1, min(8, (ntiles + 263) // 264)) for _, ntiles, _ in flat]
        self.n_instr = n
        self._dep_cnt = torch.tensor(dep_cnt, dtype=torch.int32, device=device)
        self._adj_off = torch.tensor(adj_off, dtype=torch.int32, device=device)
        self._adj = torch.tensor(adj if adj else [0], dtype=torch.int32, device=device)
        self._claim = torch.tensor(claim, dtype=torch.int32, device=device)
        # state: pending[n] | cursor[n] | done[n] | ready[n] | ctrl[4]
        self._state = torch.empty(4 * n + 4, dtype=torch.int32, device=device)
        self.critical_path = self._critical_path(deps, flat)
        return self

    def _critical_path(self, deps, flat):
        depth = [0] * len(flat)
        for i, d in enumerate(deps):  # instruction order is a topological order
            depth[i] = 1 + max((depth[j] for j in d), default=0)
        return max(depth, default=0)

    def run(self, ext, smem_bytes=100 * 1024, wave_clk=None, mode="df"):
        if mode == "waves":
            ext.run(self._instrs, self._wave_start, self._wave_tiles, self._buftab, smem_bytes, wave_clk)
        else:
            ext.run_df(
                self._instrs,
                self._dep_cnt,
                self._adj_off,
                self._adj,
                self._claim,
                self._state,
                self._buftab,
                smem_bytes,
                wave_clk,
            )


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
