"""Weight synchronization package for training-to-inference weight transfer."""

from .nccl_weight_sync import (
    EndpointInfo,
    EPWeightSynchronizerWithTPScatter,
    SyncResult,
    WeightSyncConfig,
    WeightSynchronizer,
)
from .ps_weight_sync import (
    CrossClusterNCCLGroup,
    CrossClusterReceiver,
    PersistentWeightReceiver,
    PSEndpointInfo,
    PSSyncResult,
    PSWeightSynchronizer,
)
from .rdma_weight_sync import (
    RDMADirectConfig,
    RDMADirectWeightPublisher,
    RDMASyncResult,
    RDMATensorMeta,
    RDMAWeightPublisher,
    RDMAWeightReceiver,
    RDMAWeightSyncConfig,
    RDMAWeightSyncManager,
    create_multi_rank_rdma_publisher,
    is_rdma_available,
    is_rdma_direct_available,
)
from .weight_sync_utils import (
    fuse_weights_for_sglang,
    get_tp_shard,
)

__all__ = [
    # Configuration
    "WeightSyncConfig",
    # NCCL-based weight sync
    "WeightSynchronizer",
    "EPWeightSynchronizerWithTPScatter",  # Multi-rank EP scatter with TP sharding
    "EndpointInfo",
    "SyncResult",
    # Cross-cluster weight sync (avoids CUDA graph recapture)
    "PSWeightSynchronizer",
    "PSEndpointInfo",
    "PSSyncResult",
    "CrossClusterNCCLGroup",
    "CrossClusterReceiver",
    # Persistent receiver for multiple updates
    "PersistentWeightReceiver",
    # RDMA P2P weight sync (using checkpoint-engine and mooncake)
    "RDMAWeightSyncConfig",
    "RDMAWeightPublisher",
    "RDMAWeightReceiver",
    "RDMAWeightSyncManager",
    "RDMASyncResult",
    "RDMATensorMeta",
    "is_rdma_available",
    # RDMA Direct (PUSH model) weight sync
    "RDMADirectConfig",
    "RDMADirectWeightPublisher",
    "is_rdma_direct_available",
    # Shared utilities (from weight_sync_utils.py)
    "get_tp_shard",
    "fuse_weights_for_sglang",
    "create_multi_rank_rdma_publisher",
]
