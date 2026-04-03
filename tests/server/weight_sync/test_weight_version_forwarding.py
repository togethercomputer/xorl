"""Regression tests for weight_version forwarding in weight sync."""

from unittest.mock import MagicMock

import torch

from xorl.server.weight_sync.backends.base import EndpointConfig, TransportConfig
from xorl.server.weight_sync.backends.nccl_broadcast import NCCLBroadcastBackend
from xorl.server.weight_sync.handler import WeightSyncHandler


class TestWeightVersionForwarding:
    def test_handler_broadcast_buffer_forwards_weight_version(self):
        handler = MagicMock()
        handler.rank = 0
        handler._broadcast_buffer = WeightSyncHandler._broadcast_buffer.__get__(handler)

        backend = MagicMock()
        buffer = [("layer.weight", torch.ones(2, 3, dtype=torch.bfloat16))]

        bucket_bytes, num_params = handler._broadcast_buffer(
            backend,
            buffer,
            flush_cache=False,
            weight_version="sync-v1",
        )

        assert bucket_bytes == buffer[0][1].numel() * buffer[0][1].element_size()
        assert num_params == 1
        backend.transfer_bucket.assert_called_once_with(
            buffer,
            flush_cache=False,
            weight_version="sync-v1",
        )

    def test_nccl_backend_transfer_bucket_forwards_weight_version(self):
        backend = NCCLBroadcastBackend(TransportConfig(endpoints=[EndpointConfig(host="127.0.0.1", port=30000)]))
        backend._synchronizer = MagicMock()

        buffer = [("layer.weight", torch.ones(1, dtype=torch.bfloat16))]

        backend.transfer_bucket(
            buffer,
            flush_cache=False,
            weight_version="sync-v2",
        )

        backend._synchronizer._transfer_single_bucket.assert_called_once_with(
            buffer,
            flush_cache=False,
            weight_version="sync-v2",
        )
