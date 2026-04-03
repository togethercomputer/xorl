"""Abstract base class for weight transport backends.

The weight sync pipeline has two orthogonal concerns:

1. **Pipeline orchestration** (handler.py): FSDP unshard/reshard, QLoRA
   collective ops, parameter extraction, LoRA merge, unfuse, FP8 quantize.
   This is backend-agnostic.

2. **Transport** (backends): moving tensor buckets from training rank(s) to
   inference endpoint(s).  This is backend-specific.

A :class:`WeightTransportBackend` encapsulates concern (2).  The handler
creates a backend via :func:`~xorl.server.weight_sync.backends.create_backend`,
calls :meth:`initialize` once, then repeatedly calls :meth:`transfer_bucket`
for each prepared buffer, and finally :meth:`destroy`.

Multi-rank transport
--------------------
In the simplest mode (``nccl_broadcast``), only rank 0 sends data.  All other
training ranks idle between unshard/reshard.

Future backends may allow **multiple training ranks to send simultaneously**:

* **EP-direct**: each Expert-Parallel rank sends its local experts directly
  to inference, skipping the ``dist.gather → rank 0 → broadcast`` double-hop.
* **PP-direct**: each Pipeline-Parallel stage leader sends its stage's params
  directly, skipping the CPU roundtrip through rank 0.

To support this, the backend exposes:

* :attr:`sender_ranks` — the set of training ranks that will call
  :meth:`transfer_bucket`.  The handler uses this to decide which ranks
  extract/prepare buffers.
* :attr:`supports_direct_ep_transfer` — if ``True``, the handler lets each EP
  rank send its local experts via :meth:`transfer_bucket` instead of gathering
  to rank 0 first.
* :attr:`supports_direct_pp_transfer` — if ``True``, PP stage leaders can send
  directly instead of shipping CPU buffers to rank 0 for re-broadcast.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple

import torch


@dataclass
class EndpointConfig:
    """Description of a single inference endpoint."""

    host: str
    port: int
    world_size: int = 1  # TP size on the inference side


@dataclass
class TransportConfig:
    """Backend-agnostic configuration passed to every transport backend.

    The handler populates this from the ``SyncWeightsData`` payload, and the
    backend reads whichever fields it needs.
    """

    endpoints: List[EndpointConfig] = field(default_factory=list)
    master_address: str = "localhost"
    master_port: int = 29600
    group_name: str = "weight_sync_group"
    buffer_size_mb: int = 1024
    device: str = "cuda:0"
    # Training topology (set by handler before initialize)
    training_world_size: int = 1
    training_rank: int = 0
    # Backend-specific configuration (e.g., storage_path, s3_bucket, etc.)
    backend_config: Dict[str, Any] = field(default_factory=dict)


class WeightTransportBackend(ABC):
    """Abstract interface for a weight transport backend.

    Lifecycle::

        backend = create_backend("nccl_broadcast", config)
        ok = backend.initialize()    # establish connections
        ...
        backend.transfer_bucket(bucket, src_rank=0)
        ...
        backend.destroy()            # tear down connections
    """

    def __init__(self, config: TransportConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """Establish connections to inference endpoint(s).

        Called on **sender ranks only** (ranks in :attr:`sender_ranks`).
        Other ranks skip this entirely — they only participate in training-side
        collectives (unshard/reshard).

        Returns:
            ``True`` if initialization succeeded.
        """

    @abstractmethod
    def destroy(self) -> None:
        """Tear down connections and free resources.

        Safe to call even if :meth:`initialize` was not called or failed.
        """

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    @abstractmethod
    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> None:
        """Send a bucket of named tensors to inference.

        Args:
            bucket: List of ``(param_name, tensor)`` pairs.
            src_rank: The training rank sending this bucket.  For single-rank
                backends this is always 0.  Multi-rank backends use this to
                route data from the correct source.
            flush_cache: If ``True``, tell the inference endpoint to flush
                its KV cache after loading this bucket (used for the final
                bucket of a sync).
            weight_version: Optional version marker to apply on the inference
                endpoint after this bucket is loaded. Handlers should only set
                this on the final bucket of a sync.
        """

    # ------------------------------------------------------------------
    # Topology hints (read by the handler to decide who prepares data)
    # ------------------------------------------------------------------

    @property
    def sender_ranks(self) -> FrozenSet[int]:
        """Training ranks that will call :meth:`transfer_bucket`.

        The handler only extracts and prepares buffers on these ranks.
        Default: ``{0}`` (only rank 0 sends).
        """
        return frozenset({0})

    @property
    def supports_direct_ep_transfer(self) -> bool:
        """If ``True``, each EP rank sends its local experts directly.

        When ``False`` (default), the handler gathers all experts to rank 0
        via ``dist.gather``, and rank 0 sends them through the backend.
        """
        return False

    @property
    def supports_direct_pp_transfer(self) -> bool:
        """If ``True``, PP stage leaders send their params directly.

        When ``False`` (default), PP follower leaders ship CPU buffers to
        rank 0, which then sends them through the backend.
        """
        return False
