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

    def test_nccl_bucket_transfer_uses_endpoint_port(self, monkeypatch):
        monkeypatch.delenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", raising=False)
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
            patch.object(synchronizer, "_stage_cpu_tensor_for_broadcast", return_value=(torch.ones(1), None)),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast"),
        ):
            synchronizer._transfer_single_bucket(
                [("layer.weight", torch.ones(1, dtype=torch.bfloat16))],
                flush_cache=False,
                weight_version="sync-v1",
            )

        session.post.assert_called_once()
        assert session.post.call_args.args[0] == "http://127.0.0.1:30000/update_weights_from_distributed"

    def test_nccl_bucket_transfer_sends_configured_load_format(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "direct")
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
            patch.object(synchronizer, "_stage_cpu_tensor_for_broadcast", return_value=(torch.ones(1), None)),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast"),
        ):
            synchronizer._transfer_single_bucket(
                [("layer.weight", torch.ones(1, dtype=torch.bfloat16))],
                flush_cache=False,
                weight_version="sync-v1",
            )

        assert session.post.call_args.kwargs["json"]["load_format"] == "direct"

    def test_nccl_bucket_transfer_can_use_two_phase_receiver_protocol(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_NCCL_TWO_PHASE", "1")
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "flattened_bucket")
        session = MagicMock()
        session.post.side_effect = [
            FakeResponse({"success": True, "message": "prepared"}),
            FakeResponse(
                {
                    "success": True,
                    "message": "completed",
                    "cache_version": "epoch-8",
                    "fp8_kv_cache_postprocess_ran": True,
                    "fp8_kv_cache_static_scales_updated": True,
                }
            ),
        ]
        synchronizer = NCCLWeightSynchronizer(
            endpoints=[EndpointInfo(host="127.0.0.1", port=30000, world_size=1)],
            master_address="train.example",
            master_port=29600,
            group_name="weight_sync_group",
            device="cpu",
            run_post_process_weights=True,
            fp8_kv_cache_enabled=True,
            fp8_kv_cache_postprocess_required=True,
            fp8_kv_cache_static_scales=True,
        )
        synchronizer.process_group = object()
        work = MagicMock()
        broadcast = MagicMock(return_value=work)

        with (
            patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.torch.cuda.set_device"),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast", broadcast),
            patch.object(
                synchronizer,
                "_stage_cpu_tensor_for_broadcast",
                return_value=(torch.ones(4, dtype=torch.bfloat16), None),
            ),
        ):
            results = synchronizer._transfer_single_bucket(
                [("layer.weight", torch.ones(4, dtype=torch.bfloat16))],
                flush_cache=True,
                weight_version="sync-v1",
            )

        assert [call.args[0] for call in session.post.call_args_list] == [
            "http://127.0.0.1:30000/prepare_weights_update",
            "http://127.0.0.1:30000/complete_weights_update",
        ]
        prepare_payload = session.post.call_args_list[0].kwargs["json"]
        assert prepare_payload["transport"] == "nccl_broadcast"
        assert prepare_payload["load_format"] == "flattened_bucket"
        assert prepare_payload["names"] == ["layer.weight"]
        complete_payload = session.post.call_args_list[1].kwargs["json"]
        assert complete_payload["transport"] == "nccl_broadcast"
        assert complete_payload["load_format"] == "flattened_bucket"
        assert complete_payload["flush_cache"] is True
        assert complete_payload["weight_version"] == "sync-v1"
        assert complete_payload["run_post_process_weights"] is True
        assert complete_payload["fp8_kv_cache_enabled"] is True
        assert complete_payload["fp8_kv_cache_postprocess_required"] is True
        assert complete_payload["fp8_kv_cache_static_scales"] is True
        assert results == [
            {
                "host": "127.0.0.1",
                "port": 30000,
                "endpoint": "127.0.0.1:30000",
                "success": True,
                "message": "completed",
                "cache_epoch": "epoch-8",
                "fp8_kv_cache_postprocess_ran": True,
                "fp8_kv_cache_static_scales_updated": True,
            }
        ]
        broadcast.assert_called_once()
        work.wait.assert_called_once()

    def test_nccl_bucket_transfer_flattens_mixed_dtype_bucket(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "flattened_bucket")
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
        work = MagicMock()
        broadcast = MagicMock(return_value=work)

        with (
            patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.Thread", ImmediateThread),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.torch.cuda.set_device"),
            patch(
                "xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast",
                broadcast,
            ),
            patch.object(
                synchronizer,
                "_stage_cpu_tensor_for_broadcast",
                side_effect=[
                    (torch.ones(4, dtype=torch.float8_e4m3fn), None),
                    (torch.ones(2, dtype=torch.float32), None),
                    (torch.ones(3, dtype=torch.bfloat16), None),
                ],
            ),
        ):
            synchronizer._transfer_single_bucket(
                [
                    ("layer.fp8_weight", torch.ones(4, dtype=torch.float8_e4m3fn)),
                    ("layer.scale", torch.ones(2, dtype=torch.float32)),
                    ("layer.norm", torch.ones(3, dtype=torch.bfloat16)),
                ],
                flush_cache=True,
                weight_version="sync-v1",
            )

        payload = session.post.call_args.kwargs["json"]
        assert payload["load_format"] == "flattened_bucket"
        assert payload["flush_cache"] is True
        assert payload["weight_version"] == "sync-v1"
        broadcast.assert_called_once()
        flattened = broadcast.call_args.args[0]
        assert flattened.dtype == torch.uint8
        assert flattened.numel() == 4 + (2 * 4) + (3 * 2)
        work.wait.assert_called_once()

    def test_nccl_bucket_transfer_sends_chunked_flattened_load_format(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "flattened_bucket")
        monkeypatch.setenv("XORL_WEIGHT_SYNC_NCCL_CHUNK_BYTES", "2")
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
        work = MagicMock()
        broadcast = MagicMock(return_value=work)

        with (
            patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.Thread", ImmediateThread),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.torch.cuda.set_device"),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast", broadcast),
            patch.object(
                synchronizer,
                "_stage_cpu_tensor_for_broadcast",
                return_value=(torch.ones(4, dtype=torch.bfloat16), None),
            ),
        ):
            synchronizer._transfer_single_bucket(
                [("layer.weight", torch.ones(4, dtype=torch.bfloat16))],
                flush_cache=False,
                weight_version=None,
            )

        assert session.post.call_args.kwargs["json"]["load_format"] == "flattened_bucket_chunked:2"
        assert broadcast.call_count == 4
        assert work.wait.call_count == 4

    def test_nccl_hybrid_flattened_broadcasts_are_receiver_fenced(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "hybrid_flattened")
        monkeypatch.setenv("XORL_WEIGHT_SYNC_WAIT_AFTER_RECEIVER", "0")
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
        direct_work = MagicMock()
        fallback_work = MagicMock()
        broadcast = MagicMock(side_effect=[direct_work, fallback_work])

        with (
            patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.Thread", ImmediateThread),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.torch.cuda.set_device"),
            patch(
                "xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast",
                broadcast,
            ),
            patch.object(
                synchronizer,
                "_stage_cpu_tensor_for_broadcast",
                side_effect=[
                    (torch.ones(3, dtype=torch.bfloat16), None),
                    (torch.ones(4, dtype=torch.float8_e4m3fn), None),
                ],
            ),
        ):
            synchronizer._transfer_single_bucket(
                [
                    ("model.layers.0.self_attn.o_proj.weight", torch.ones(3, dtype=torch.bfloat16)),
                    ("model.layers.0.self_attn.q_proj.weight", torch.ones(4, dtype=torch.float8_e4m3fn)),
                ],
                flush_cache=False,
                weight_version="sync-v1",
            )

        payload = session.post.call_args.kwargs["json"]
        assert payload["load_format"] == "hybrid_flattened"
        assert broadcast.call_count == 2
        direct_work.wait.assert_called_once()
        fallback_work.wait.assert_not_called()
        assert len(synchronizer._receiver_fenced_refs) == 1
        held_work, held_tensor, held_staging_ref = synchronizer._receiver_fenced_refs[0]
        assert held_work is fallback_work
        assert held_tensor.dtype == torch.uint8
        assert held_staging_ref is None

    def test_nccl_hybrid_flattened_releases_receiver_fenced_refs_on_destroy(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "hybrid_flattened")
        monkeypatch.setenv("XORL_WEIGHT_SYNC_WAIT_AFTER_RECEIVER", "0")
        session = MagicMock()
        session.post.return_value = FakeResponse({"success": True, "message": "ok"})
        synchronizer = NCCLWeightSynchronizer(
            endpoints=[EndpointInfo(host="127.0.0.1", port=30000, world_size=1)],
            master_address="train.example",
            master_port=29600,
            group_name="weight_sync_group",
            device="cpu",
        )
        synchronizer.process_group = MagicMock()
        direct_work = MagicMock()
        fallback_work = MagicMock()
        broadcast = MagicMock(side_effect=[direct_work, fallback_work])

        with (
            patch("xorl.server.weight_sync.backends.nccl_broadcast._get_http_session", return_value=session),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.Thread", ImmediateThread),
            patch("xorl.server.weight_sync.backends.nccl_broadcast.torch.cuda.set_device"),
            patch(
                "xorl.server.weight_sync.backends.nccl_broadcast.dist.broadcast",
                broadcast,
            ),
            patch.object(
                synchronizer,
                "_stage_cpu_tensor_for_broadcast",
                side_effect=[
                    (torch.ones(3, dtype=torch.bfloat16), None),
                    (torch.ones(4, dtype=torch.float8_e4m3fn), None),
                ],
            ),
        ):
            synchronizer._transfer_single_bucket(
                [
                    ("model.layers.0.self_attn.o_proj.weight", torch.ones(3, dtype=torch.bfloat16)),
                    ("model.layers.0.self_attn.q_proj.weight", torch.ones(4, dtype=torch.float8_e4m3fn)),
                ],
                flush_cache=False,
                weight_version="sync-v1",
            )
            direct_work.wait.assert_called_once()
            fallback_work.wait.assert_not_called()
            assert len(synchronizer._receiver_fenced_refs) == 1
            held_work, held_tensor, held_staging_ref = synchronizer._receiver_fenced_refs[0]
            assert held_work is fallback_work
            assert held_tensor.dtype == torch.uint8
            assert held_staging_ref is None
            synchronizer.destroy_nccl_group()

        assert synchronizer._receiver_fenced_refs == []

    def test_nccl_bucket_transfer_rejects_direct_load_format_for_multi_rank_endpoint(self, monkeypatch):
        monkeypatch.setenv("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT", "direct")
        synchronizer = NCCLWeightSynchronizer(
            endpoints=[EndpointInfo(host="127.0.0.1", port=30000, world_size=2)],
            master_address="train.example",
            master_port=29600,
            group_name="weight_sync_group",
            device="cpu",
        )

        with pytest.raises(RuntimeError, match="requires SGLang world_size=1"):
            synchronizer._transfer_single_bucket(
                [("layer.weight", torch.ones(1, dtype=torch.bfloat16))],
                flush_cache=False,
                weight_version="sync-v1",
            )
