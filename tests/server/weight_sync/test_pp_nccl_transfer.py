"""
Unit tests for PP weight sync NCCL transfer helpers.

Tests the _pp_nccl_transfer_buffer protocol and _prod helper without
requiring a full distributed environment.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from xorl.server.weight_sync.handler import _prod


class TestProd:
    def test_empty(self):
        assert _prod([]) == 1

    def test_scalar(self):
        assert _prod([5]) == 5

    def test_2d(self):
        assert _prod([3, 4]) == 12

    def test_3d(self):
        assert _prod([2, 3, 4]) == 24


class TestPPNcclTransferBuffer:
    """Tests for _pp_nccl_transfer_buffer protocol logic.

    These tests mock dist.broadcast and dist.broadcast_object_list to verify
    the send/receive protocol without an actual distributed environment.
    """

    def _make_handler(self, rank):
        handler = MagicMock()
        handler.rank = rank
        # Bind the real method
        from xorl.server.weight_sync.handler import WeightSyncHandler
        handler._pp_nccl_transfer_buffer = WeightSyncHandler._pp_nccl_transfer_buffer.__get__(handler)
        return handler

    @patch("xorl.server.weight_sync.handler.dist")
    def test_sender_returns_none(self, mock_dist):
        """Sender should return None (it doesn't need the received data)."""
        handler = self._make_handler(rank=2)
        src_global = 2

        buffer = [
            ("layer.0.weight", torch.randn(4, 3, dtype=torch.bfloat16)),
            ("layer.0.bias", torch.randn(4, dtype=torch.bfloat16)),
        ]

        # Mock broadcast_object_list to be a no-op (sender side)
        mock_dist.broadcast_object_list = MagicMock()
        mock_dist.broadcast = MagicMock()

        result = handler._pp_nccl_transfer_buffer(buffer, MagicMock(), src_global, "cuda:0")
        assert result is None

    @patch("xorl.server.weight_sync.handler.dist")
    def test_empty_buffer_sender(self, mock_dist):
        """Empty buffer from sender should result in empty list for receiver."""
        handler = self._make_handler(rank=2)
        src_global = 2

        mock_dist.broadcast_object_list = MagicMock()
        mock_dist.broadcast = MagicMock()

        result = handler._pp_nccl_transfer_buffer([], MagicMock(), src_global, "cuda:0")
        assert result is None

    @patch("xorl.server.weight_sync.handler.dist")
    def test_receiver_gets_empty_for_empty_send(self, mock_dist):
        """Receiver should get [] when sender sent empty buffer."""
        handler = self._make_handler(rank=0)
        src_global = 2

        # Simulate broadcast_object_list delivering empty metadata
        def fake_broadcast_object_list(obj, src, group):
            obj[0] = []  # empty metadata from sender
        mock_dist.broadcast_object_list = fake_broadcast_object_list

        result = handler._pp_nccl_transfer_buffer(None, MagicMock(), src_global, "cuda:0")
        assert result == []

    @patch("xorl.server.weight_sync.handler.dist")
    def test_roundtrip_shapes(self, mock_dist):
        """Verify that metadata correctly encodes shapes for the receiver."""
        handler_sender = self._make_handler(rank=2)
        src_global = 2

        buffer = [
            ("model.layers.0.self_attn.q_proj.weight", torch.randn(512, 256, dtype=torch.bfloat16)),
            ("model.layers.0.self_attn.k_proj.weight", torch.randn(128, 256, dtype=torch.bfloat16)),
            ("model.layers.0.input_layernorm.weight", torch.randn(256, dtype=torch.bfloat16)),
        ]

        # Capture what sender broadcasts
        captured_meta = [None]
        captured_flat = [None]

        def fake_broadcast_object_list(obj, src, group):
            if captured_meta[0] is None:
                captured_meta[0] = obj[0]

        def fake_broadcast(tensor, src, group):
            if captured_flat[0] is None:
                captured_flat[0] = tensor.clone()

        mock_dist.broadcast_object_list = fake_broadcast_object_list
        mock_dist.broadcast = fake_broadcast

        handler_sender._pp_nccl_transfer_buffer(buffer, MagicMock(), src_global, "cuda:0")

        # Verify metadata
        meta = captured_meta[0]
        assert len(meta) == 3
        assert meta[0] == ("model.layers.0.self_attn.q_proj.weight", [512, 256])
        assert meta[1] == ("model.layers.0.self_attn.k_proj.weight", [128, 256])
        assert meta[2] == ("model.layers.0.input_layernorm.weight", [256])

        # Verify flat tensor size
        expected_elements = 512 * 256 + 128 * 256 + 256
        assert captured_flat[0].shape == (expected_elements,)
        assert captured_flat[0].dtype == torch.bfloat16

    @patch("xorl.server.weight_sync.handler.dist")
    def test_receiver_reconstructs_tensors(self, mock_dist):
        """Verify receiver correctly splits flat tensor back into named tensors."""
        src_global = 2
        device = "cpu"  # Use CPU for testing

        # Original buffer
        t1 = torch.randn(4, 3, dtype=torch.bfloat16)
        t2 = torch.randn(2, 5, dtype=torch.bfloat16)
        t3 = torch.randn(8, dtype=torch.bfloat16)
        buffer = [("a.weight", t1), ("b.weight", t2), ("c.bias", t3)]

        # Pre-compute what sender would produce
        meta = [("a.weight", [4, 3]), ("b.weight", [2, 5]), ("c.bias", [8])]
        flat = torch.cat([t1.reshape(-1), t2.reshape(-1), t3.reshape(-1)])

        # Simulate receiver side
        handler_recv = self._make_handler(rank=0)

        def fake_broadcast_object_list(obj, src, group):
            obj[0] = meta

        def fake_broadcast(tensor, src, group):
            tensor.copy_(flat)

        mock_dist.broadcast_object_list = fake_broadcast_object_list
        mock_dist.broadcast = fake_broadcast

        received = handler_recv._pp_nccl_transfer_buffer(None, MagicMock(), src_global, device)

        assert len(received) == 3
        assert received[0][0] == "a.weight"
        assert received[0][1].shape == (4, 3)
        assert torch.allclose(received[0][1], t1)

        assert received[1][0] == "b.weight"
        assert received[1][1].shape == (2, 5)
        assert torch.allclose(received[1][1], t2)

        assert received[2][0] == "c.bias"
        assert received[2][1].shape == (8,)
        assert torch.allclose(received[2][1], t3)
