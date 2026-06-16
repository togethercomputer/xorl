"""Two-phase NCCL weight sync backend.

Why this exists
---------------
``nccl_broadcast`` interleaves two NCCL communicators per module:

    unshard (FSDP all-gather, intra-node) -> extract -> reshard
        -> dist.broadcast (weight-sync group, inter-node) -> next module

With FSDP ranks 1..N racing ahead of rank 0 (they have no broadcast work),
the two communicators enqueue kernels in different orders across ranks,
which deadlocks NCCL after a few modules (observed consistently at the
7th bucket on 14B / FSDP=4 / 2-node).

This backend removes the interleaving entirely:

* **Phase A** (during the handler's module loop): ``transfer_bucket`` only
  stages tensors to CPU. No NCCL, no HTTP. The FSDP loop runs to completion
  using only FSDP collectives.
* **Phase B** (``flush_pending_transfers``, called by the handler after the
  module loop): re-chunk staged params and send each chunk through the
  proven ``_transfer_single_bucket`` path (HTTP + dist.broadcast). Only the
  weight-sync communicator is active.

The two communicators never run concurrently, so the kernel-ordering
deadlock cannot occur.
"""

import logging
from typing import List, Optional, Tuple

import torch

from .nccl_broadcast import NCCLBroadcastBackend


logger = logging.getLogger(__name__)

# Re-chunk size for phase B. Bounds sglang-side temp memory
# (torch.empty per param before load_weights) and trainer-side H2D staging.
_CHUNK_BYTES = 1024 * 1024 * 1024  # 1 GiB


class NCCLSimpleBackend(NCCLBroadcastBackend):
    """Two-phase (stage-then-broadcast) NCCL transport."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._pending: List[Tuple[str, torch.Tensor]] = []
        self._pending_bytes = 0
        self._final_flush_cache = False
        self._final_weight_version: Optional[str] = None

    # ------------------------------------------------------------------
    # Phase A: stage to CPU (no NCCL, no HTTP)
    # ------------------------------------------------------------------
    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> None:
        if src_rank != 0:
            raise ValueError(f"NCCLSimpleBackend only supports src_rank=0, got {src_rank}")
        for name, t in bucket:
            cpu_t = t.detach().to("cpu").contiguous()
            self._pending.append((name, cpu_t))
            self._pending_bytes += cpu_t.numel() * cpu_t.element_size()
        if flush_cache:
            self._final_flush_cache = True
        if weight_version is not None:
            self._final_weight_version = weight_version

    # ------------------------------------------------------------------
    # Phase B: chunked HTTP + broadcast (no FSDP collectives anywhere)
    # ------------------------------------------------------------------
    def flush_pending_transfers(self) -> None:
        if not self._pending:
            return
        if self._synchronizer is None:
            raise RuntimeError("Backend not initialized — call initialize() first")

        # Build chunks bounded by _CHUNK_BYTES (a single oversized param
        # becomes its own chunk).
        chunks: List[List[Tuple[str, torch.Tensor]]] = []
        cur: List[Tuple[str, torch.Tensor]] = []
        cur_bytes = 0
        for name, t in self._pending:
            nbytes = t.numel() * t.element_size()
            if cur and cur_bytes + nbytes > _CHUNK_BYTES:
                chunks.append(cur)
                cur, cur_bytes = [], 0
            cur.append((name, t))
            cur_bytes += nbytes
        if cur:
            chunks.append(cur)

        total_gb = self._pending_bytes / 1e9
        logger.info(
            f"[NCCLSimple] Phase B: broadcasting {len(self._pending)} params "
            f"({total_gb:.2f} GB) in {len(chunks)} chunks"
        )

        device = self.config.device
        try:
            for i, chunk in enumerate(chunks):
                last = i == len(chunks) - 1
                gpu_chunk = [(n, t.to(device, non_blocking=True)) for n, t in chunk]
                torch.cuda.synchronize(device)
                self._synchronizer._transfer_single_bucket(
                    gpu_chunk,
                    flush_cache=self._final_flush_cache and last,
                    weight_version=self._final_weight_version if last else None,
                )
                del gpu_chunk
            logger.info(f"[NCCLSimple] Phase B complete: {len(chunks)} chunks sent")
        finally:
            self._pending = []
            self._pending_bytes = 0
            self._final_flush_cache = False
            self._final_weight_version = None
