"""Disk-backed tensor allocator via ``mmap``.

Used by :mod:`scripts.convert_dsv4_hf_to_dcp` (and any future converter)
when the model's BF16 footprint exceeds host RAM. DSv4 Pro at BF16 is
~3.1 TB — the apanda namespace nodes have 1.58 TB RAM each, so a
single-pod conversion that keeps the whole model in RAM doesn't fit.

The allocator opens a single sparse file on a fast PVC (e.g. ``/shared``),
``mmap``-s the entire range, and hands out byte-aligned views as
``torch`` tensors that share the mmap's storage. Reads and writes go
through the OS page cache; resident memory is bounded by working set
rather than total tensor footprint.

Performance trade-off: disk-bound rather than RAM-bound. Empirically
~250 MB/s sustained on the apanda PVC. For the 3.1 TB Pro conversion,
that's ~3.5 hours of materialize + dequant + DCP write — a one-shot
cost we accept in exchange for fitting the conversion on a single pod.

Lifetime contract: the allocator owns the mmap and the underlying file
descriptor. Tensors handed out by :meth:`alloc` keep a reference to the
allocator (via the ``mmap`` object held in the tensor's storage), so
they remain valid as long as the tensor is alive. The caller must keep
the allocator alive for the lifetime of any in-use tensor.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Sequence

import torch


_PAGE_SIZE = mmap.PAGESIZE  # 4096 on x86_64


def _dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


class MmapTensorAllocator:
    """Allocate torch tensors backed by a single sparse mmap'd file.

    Args:
        path: file path. Pre-existing files are truncated. Use a
            location with enough disk space for the worst-case
            footprint (sparse files don't pre-allocate, but the
            filesystem must be able to grow on write).
        capacity_bytes: virtual size of the mmap region. Set to a
            comfortable upper bound on total tensor bytes (we'll
            bump-allocate within this region).
    """

    def __init__(self, path: str | os.PathLike, capacity_bytes: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(self._fd, capacity_bytes)
        self._capacity = capacity_bytes
        self._mmap = mmap.mmap(
            self._fd,
            length=capacity_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self._cursor = 0
        self._tensors: list[torch.Tensor] = []  # keep refs alive

    @property
    def used_bytes(self) -> int:
        return self._cursor

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    def alloc(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Allocate an mmap-backed tensor of ``shape`` and ``dtype``.

        The returned tensor's storage is the mmap; reads/writes are
        page-cached and may hit disk. Initial contents are
        zero-initialized by the kernel because the file was truncated
        to length (sparse). Subsequent writes mark the page dirty.
        """
        n_elements = 1
        for s in shape:
            n_elements *= s
        nbytes = n_elements * _dtype_itemsize(dtype)
        # 4 KB-align so each tensor starts on a fresh page (avoids
        # partial-page sharing between tensors and keeps the mmap views
        # well-defined).
        offset = (self._cursor + _PAGE_SIZE - 1) & ~(_PAGE_SIZE - 1)
        end = offset + nbytes
        if end > self._capacity:
            raise RuntimeError(
                f"MmapTensorAllocator out of capacity: needed {end} bytes, "
                f"capacity is {self._capacity} bytes (used {self._cursor})"
            )
        # ``mmap`` slicing returns a memoryview into the same backing
        # buffer. ``torch.frombuffer`` wraps that memoryview as a
        # tensor without copying. The tensor's storage now references
        # the mmap memory directly; reads/writes go through the page
        # cache.
        view = memoryview(self._mmap)[offset:end]
        tensor = torch.frombuffer(view, dtype=dtype, count=n_elements).reshape(*shape)
        self._tensors.append(tensor)
        self._cursor = end
        return tensor

    def close(self) -> None:
        """Release the mmap, close the fd, and unlink the staging file.

        Idempotent. After close, allocated tensors will segfault on
        access — call only after the model has been saved / freed.
        """
        # Drop tensor refs first so storages release their mmap views.
        self._tensors.clear()
        try:
            self._mmap.close()
        except (BufferError, ValueError):
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "MmapTensorAllocator":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
