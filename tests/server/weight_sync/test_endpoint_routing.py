"""Tests for single-endpoint routing in weight sync HTTP control paths."""

from unittest.mock import MagicMock, patch

import pytest
import requests
import torch

from xorl.server.weight_sync.backends.nccl_broadcast import EndpointInfo, NCCLWeightSynchronizer
from xorl.server.weight_sync.endpoint_manager import EndpointManager


class FakeResponse:
    def __init__(self, payload, error: Exception | None = None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error
        return None

    def json(self):
        return self._payload


class ImmediateThread:
    """Thread test double that runs work synchronously."""

    def __init__(self, target, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def join(self):
        return None


class TestEndpointRouting:
    def test_endpoint_manager_health_check_uses_endpoint_port(self):
        session = MagicMock()
        session.get.return_value = FakeResponse({"status": "ok"})
        manager = EndpointManager([{"host": "127.0.0.1", "port": 30000}])

        with patch("xorl.server.weight_sync.endpoint_manager._get_http_session", return_value=session):
            manager.health_check()

        session.get.assert_called_once_with("http://127.0.0.1:30000/model_info", timeout=60)

    def test_endpoint_manager_health_check_falls_back_to_v1_models(self):
        session = MagicMock()
        session.get.side_effect = [
            FakeResponse({}, requests.HTTPError("503 Server Error")),
            FakeResponse({"data": [{"id": "Qwen/Qwen3-30B-A3B"}]}),
        ]
        manager = EndpointManager([{"host": "127.0.0.1", "port": 30000}])

        with patch("xorl.server.weight_sync.endpoint_manager._get_http_session", return_value=session):
            manager.health_check()

        assert [call.args[0] for call in session.get.call_args_list] == [
            "http://127.0.0.1:30000/model_info",
            "http://127.0.0.1:30000/v1/models",
        ]

    def test_endpoint_manager_health_check_reports_all_failures(self):
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("connection refused")
        manager = EndpointManager([{"host": "127.0.0.1", "port": 30000}])

        with (
            patch("xorl.server.weight_sync.endpoint_manager._get_http_session", return_value=session),
            pytest.raises(RuntimeError, match="/model_info.*/v1/models.*/health"),
        ):
            manager.health_check()

    def test_nccl_sync_init_uses_endpoint_port(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({"success": True, "message": "ok"})
        synchronizer = NCCLWeightSynchronizer(
            endpoints=[EndpointInfo(host="127.0.0.1", port=30000, world_size=1)],
            master_address="train.example",
            master_port=29600,
            group_name="weight_sync_group",
            device="cpu",
        )

        with patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session):
            results = synchronizer._init_inference_endpoints()

        assert results == [
            {
                "endpoint": "127.0.0.1:30000",
                "rank_offset": 1,
                "success": True,
                "message": "ok",
            }
        ]
        session.post.assert_called_once()
        assert session.post.call_args.args[0] == "http://127.0.0.1:30000/init_weights_update_group"

    def test_nccl_bucket_transfer_uses_endpoint_port(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({"success": True, "message": "ok"})
        synchronizer = NCCLWeightSynchronizer(
            endpoints=[EndpointInfo(host="127.0.0.1", port=30000, world_size=1)],
            master_address="train.example",
            master_port=29600,
            group_name="weight_sync_group",
            device="cpu",
        )
        synchronizer.process_group = object()

        with (
            patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.Thread", ImmediateThread),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.torch.cuda.set_device"),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast"),
        ):
            synchronizer._transfer_single_bucket(
                [("layer.weight", torch.ones(1, dtype=torch.bfloat16))],
                flush_cache=False,
                weight_version="sync-v1",
            )

        session.post.assert_called_once()
        assert session.post.call_args.args[0] == "http://127.0.0.1:30000/update_weights_from_distributed"
