"""Protocol-level tests for the P2P weight transport backend.

These tests exercise the xorl-side wire protocol against mocked SGLang HTTP
endpoints — no GPU, no Mooncake, no real network. They cover:

* The shape of the ``/prepare_weights_update`` POST when ``transport=p2p``.
* Aggregating ``tensor_map`` and ``receiver_transfer_engine_infos`` across
  multiple endpoints.
* Slicing the trainer's full HF tensor according to a locator's
  ``slice``/``full_shape`` fields and computing the right Mooncake address.
* The shape of the ``/complete_weights_update`` POST and weight-version /
  flush-cache propagation.
* The shape mismatch and "skip transfer" guards.

Real Mooncake transfers are exercised by ``scripts/p2p_e2e_smoke.py``
(needs GPUs + IB), not here.
"""

from __future__ import annotations

import sys
import types
from concurrent.futures import Future
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest
import torch

from xorl.server.weight_sync.backends.base import EndpointConfig, TransportConfig
from xorl.server.weight_sync.backends.p2p import P2PTransportBackend


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Dict[str, Any] | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "" if status_code == 200 else "error body"

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeMooncakeEngine:
    """Test double that records calls and returns success."""

    def __init__(self, session_id: str = "fake-trainer:1234"):
        self.session_id = session_id
        self.engine = self
        self.registered: List[Tuple[List[int], List[int]]] = []
        self.registered_memory: List[Tuple[int, int]] = []
        self.deregistered: List[List[int]] = []
        self.transfers: List[Tuple[str, List[int], List[int], List[int]]] = []
        self.fail_transfer = False

    # API surface mirrors MooncakeTransferEngine
    def get_session_id(self) -> str:
        return self.session_id

    def get_ib_device(self) -> str:
        return "mlx5_0"

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        self.registered.append((list(ptrs), list(lengths)))
        return 0

    def register_memory(self, ptr: int, length: int) -> int:
        self.registered_memory.append((ptr, length))
        return 0

    def batch_deregister(self, ptrs: List[int]) -> int:
        self.deregistered.append(list(ptrs))
        return 0

    def batch_transfer_sync(
        self,
        session_id: str,
        src_ptrs: List[int],
        peer_ptrs: List[int],
        lengths: List[int],
    ) -> int:
        self.transfers.append((session_id, list(src_ptrs), list(peer_ptrs), list(lengths)))
        if self.fail_transfer:
            return -1
        return 0


def _make_backend(num_endpoints: int = 1) -> Tuple[P2PTransportBackend, _FakeMooncakeEngine]:
    cfg = TransportConfig(
        endpoints=[EndpointConfig(host=f"infer-{i}", port=5000 + i, world_size=2) for i in range(num_endpoints)],
        master_address="trainer-0",
        master_port=0,
        group_name="weight_sync_group",
        buffer_size_mb=64,
        device="cuda:0",
        backend_config={"hostname": "trainer-0", "gpu_id": 0, "cpu_scratch_pool_bytes": 1024 * 1024},
    )
    backend = P2PTransportBackend(cfg)
    fake_engine = _FakeMooncakeEngine()
    # Pin the fake engine in place of the real Mooncake import path so that
    # `initialize()` doesn't try to construct a TransferEngine. This also
    # makes `transfer_bucket` / `destroy` see the fake.
    backend._engine = fake_engine
    backend._make_local_engine = lambda: fake_engine  # type: ignore[assignment]
    return backend, fake_engine


def _hf_locator(
    *,
    tp_rank: int,
    full_shape: List[int],
    slc: List[List[int]],
    ptr: int,
    nbytes: int,
    session_id: str,
    dtype: str = "bfloat16",
) -> Dict[str, Any]:
    return {
        "hf_name": "ignored-by-callers-of-this-helper",
        "tp_rank": tp_rank,
        "dp_rank": 0,
        "ep_rank": -1,
        "dtype": dtype,
        "full_shape": full_shape,
        "slice": slc,
        "ptr": ptr,
        "nbytes": nbytes,
        "session_id": session_id,
    }


class TestP2PInitializeHandshake:
    def test_prepare_payload_uses_p2p_transport_and_engine_info(self):
        backend, engine = _make_backend(num_endpoints=1)

        prepare_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "ok",
                "tensor_map": {
                    "model.layers.0.self_attn.q_proj.weight": [
                        {
                            **_hf_locator(
                                tp_rank=0,
                                full_shape=[128, 64],
                                slc=[[0, 64], [0, 64]],
                                ptr=0xDEAD0000,
                                nbytes=64 * 64 * 2,
                                session_id="recv-0:7000",
                            ),
                            "hf_name": "model.layers.0.self_attn.q_proj.weight",
                        },
                    ]
                },
                "receiver_transfer_engine_infos": [
                    {"tp_rank": 0, "session_id": "recv-0:7000"},
                ],
            },
        )

        with patch("requests.post", return_value=prepare_response) as posted:
            ok = backend.initialize()

        assert ok is True
        posted.assert_called_once()
        url = posted.call_args.args[0]
        body = posted.call_args.kwargs["json"]
        assert url == "http://infer-0:5000/prepare_weights_update"
        assert body["transport"] == "p2p"
        assert body["sender_transfer_engine_info"]["session_id"] == engine.session_id
        assert body["sender_transfer_engine_info"]["training_rank"] == 0
        assert body["group_name"] == "weight_sync_group"
        assert "p2p_return_tensor_map" not in body

    def test_initialize_aggregates_tensor_map_across_endpoints(self):
        backend, _ = _make_backend(num_endpoints=2)

        ep0_resp = _FakeResponse(
            200,
            {
                "success": True,
                "message": "ok",
                "tensor_map": {
                    "model.layers.0.self_attn.q_proj.weight": [
                        {
                            **_hf_locator(
                                tp_rank=0,
                                full_shape=[128, 64],
                                slc=[[0, 64], [0, 64]],
                                ptr=0x1000,
                                nbytes=64 * 64 * 2,
                                session_id="recv0a:7000",
                            ),
                            "hf_name": "model.layers.0.self_attn.q_proj.weight",
                        },
                    ],
                },
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv0a:7000"}],
            },
        )
        ep1_resp = _FakeResponse(
            200,
            {
                "success": True,
                "message": "ok",
                "tensor_map": {
                    "model.layers.0.self_attn.q_proj.weight": [
                        {
                            **_hf_locator(
                                tp_rank=1,
                                full_shape=[128, 64],
                                slc=[[64, 128], [0, 64]],
                                ptr=0x2000,
                                nbytes=64 * 64 * 2,
                                session_id="recv1a:7000",
                            ),
                            "hf_name": "model.layers.0.self_attn.q_proj.weight",
                        },
                    ],
                },
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv1a:7000"}],
            },
        )

        with patch("requests.post", side_effect=[ep0_resp, ep1_resp]):
            ok = backend.initialize()

        assert ok is True
        locators = backend._tensor_map["model.layers.0.self_attn.q_proj.weight"]
        assert len(locators) == 2
        assert {loc["endpoint_idx"] for loc in locators} == {0, 1}
        assert sorted(loc["session_id"] for loc in locators) == ["recv0a:7000", "recv1a:7000"]

    def test_initialize_returns_false_on_http_error(self):
        backend, _ = _make_backend()
        with patch("requests.post", return_value=_FakeResponse(500, {})):
            assert backend.initialize() is False

    def test_initialize_returns_false_when_remote_reports_failure(self):
        backend, _ = _make_backend()
        with patch(
            "requests.post",
            return_value=_FakeResponse(200, {"success": False, "message": "no engine"}),
        ):
            assert backend.initialize() is False

    def test_cached_prepare_can_reuse_existing_tensor_map(self):
        backend, _ = _make_backend()
        cached_map = {
            "model.layers.0.self_attn.q_proj.weight": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=0xDEAD0000,
                        nbytes=64 * 64 * 2,
                        session_id="recv-0:7000",
                    ),
                    "hf_name": "model.layers.0.self_attn.q_proj.weight",
                },
            ]
        }
        backend._tensor_map = cached_map
        backend._receiver_session_ids = ["recv-0:7000"]

        prepare_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "cached",
                "tensor_map": None,
                "receiver_transfer_engine_infos": [
                    {"tp_rank": 0, "session_id": "recv-0:7000"},
                ],
            },
        )

        with patch("requests.post", return_value=prepare_response) as posted:
            ok = backend.initialize()

        assert ok is True
        assert backend._tensor_map == cached_map
        assert backend._last_prepare_returned_tensor_map is False
        body = posted.call_args.kwargs["json"]
        assert body["p2p_return_tensor_map"] is False

    def test_cached_prepare_retries_full_map_when_receiver_session_changes(self):
        backend, _ = _make_backend()
        param_name = "model.layers.0.self_attn.q_proj.weight"
        backend._tensor_map = {
            param_name: [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=0xDEAD0000,
                        nbytes=64 * 64 * 2,
                        session_id="recv-old:7000",
                    ),
                    "hf_name": param_name,
                },
            ]
        }
        backend._receiver_session_ids = ["recv-old:7000"]
        backend._prefer_cached_prepare = True
        new_locator = {
            **_hf_locator(
                tp_rank=0,
                full_shape=[128, 64],
                slc=[[0, 64], [0, 64]],
                ptr=0xBEEF0000,
                nbytes=64 * 64 * 2,
                session_id="recv-new:7000",
            ),
            "hf_name": param_name,
        }

        cached_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "cached",
                "tensor_map": None,
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv-new:7000"}],
            },
        )
        full_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "full",
                "tensor_map": {param_name: [new_locator]},
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv-new:7000"}],
            },
        )

        with patch("requests.post", side_effect=[cached_response, full_response]) as posted:
            ok = backend.initialize()

        assert ok is True
        assert posted.call_count == 2
        assert posted.call_args_list[0].kwargs["json"]["p2p_return_tensor_map"] is False
        assert "p2p_return_tensor_map" not in posted.call_args_list[1].kwargs["json"]
        assert backend._receiver_session_ids == ["recv-new:7000"]
        assert backend._tensor_map[param_name][0]["session_id"] == "recv-new:7000"
        assert backend._tensor_map[param_name][0]["ptr"] == 0xBEEF0000
        assert backend._last_prepare_returned_tensor_map is True

    def test_cached_prepare_merges_partial_tensor_map_with_cached_endpoints(self):
        backend, _ = _make_backend(num_endpoints=2)
        name = "model.layers.0.self_attn.q_proj.weight"
        cached_map = {
            name: [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=0x1000,
                        nbytes=64 * 64 * 2,
                        session_id="recv0-old:7000",
                    ),
                    "hf_name": name,
                    "endpoint_idx": 0,
                },
                {
                    **_hf_locator(
                        tp_rank=1,
                        full_shape=[128, 64],
                        slc=[[64, 128], [0, 64]],
                        ptr=0x2000,
                        nbytes=64 * 64 * 2,
                        session_id="recv1-old:7000",
                    ),
                    "hf_name": name,
                    "endpoint_idx": 1,
                },
            ]
        }
        backend._tensor_map = cached_map
        backend._receiver_session_ids = ["recv0-old:7000", "recv1-old:7000"]
        backend._session_debug_info = {
            "recv0-old:7000": {"session_id": "recv0-old:7000", "endpoint_idx": 0},
            "recv1-old:7000": {"session_id": "recv1-old:7000", "endpoint_idx": 1},
        }
        backend._prefer_cached_prepare = True

        ep0_resp = _FakeResponse(200, {"success": True, "message": "cached", "tensor_map": None})
        ep1_resp = _FakeResponse(
            200,
            {
                "success": True,
                "message": "full",
                "tensor_map": {
                    name: [
                        {
                            **_hf_locator(
                                tp_rank=1,
                                full_shape=[128, 64],
                                slc=[[64, 128], [0, 64]],
                                ptr=0x3000,
                                nbytes=64 * 64 * 2,
                                session_id="recv1-new:7000",
                            ),
                            "hf_name": name,
                        }
                    ]
                },
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv1-new:7000"}],
            },
        )

        with patch("requests.post", side_effect=[ep0_resp, ep1_resp]):
            ok = backend.initialize()

        assert ok is True
        locators = backend._tensor_map[name]
        assert [(loc["endpoint_idx"], loc["ptr"], loc["session_id"]) for loc in locators] == [
            (0, 0x1000, "recv0-old:7000"),
            (1, 0x3000, "recv1-new:7000"),
        ]
        assert backend._receiver_session_ids == ["recv0-old:7000", "recv1-new:7000"]
        assert backend._last_prepare_returned_tensor_map is True

    def test_cached_prepare_retries_full_map_when_receiver_rejects_flag(self):
        backend, _ = _make_backend()
        backend._tensor_map = {
            "model.layers.0.self_attn.q_proj.weight": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=0xDEAD0000,
                        nbytes=64 * 64 * 2,
                        session_id="recv-0:7000",
                    ),
                    "hf_name": "model.layers.0.self_attn.q_proj.weight",
                },
            ]
        }
        backend._receiver_session_ids = ["recv-0:7000"]
        full_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "full",
                "tensor_map": backend._tensor_map,
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv-0:7000"}],
            },
        )

        with patch("requests.post", side_effect=[_FakeResponse(422, {}), full_response]) as posted:
            ok = backend.initialize()

        assert ok is True
        assert posted.call_count == 2
        assert posted.call_args_list[0].kwargs["json"]["p2p_return_tensor_map"] is False
        assert "p2p_return_tensor_map" not in posted.call_args_list[1].kwargs["json"]
        assert backend._last_prepare_returned_tensor_map is True

    def test_cached_prepare_restarts_all_endpoints_when_later_endpoint_rejects_flag(self):
        backend, _ = _make_backend(num_endpoints=2)
        param_name = "model.layers.0.self_attn.q_proj.weight"
        backend._tensor_map = {
            param_name: [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=0xDEAD0000,
                        nbytes=64 * 64 * 2,
                        session_id="recv-0:7000",
                    ),
                    "hf_name": param_name,
                    "endpoint_idx": 0,
                },
                {
                    **_hf_locator(
                        tp_rank=1,
                        full_shape=[128, 64],
                        slc=[[64, 128], [0, 64]],
                        ptr=0xBEEF0000,
                        nbytes=64 * 64 * 2,
                        session_id="recv-1:7000",
                    ),
                    "hf_name": param_name,
                    "endpoint_idx": 1,
                },
            ]
        }
        backend._receiver_session_ids = ["recv-0:7000", "recv-1:7000"]
        backend._prefer_cached_prepare = True
        ep0_full_locator = {
            **_hf_locator(
                tp_rank=0,
                full_shape=[128, 64],
                slc=[[0, 64], [0, 64]],
                ptr=0x1000,
                nbytes=64 * 64 * 2,
                session_id="recv-0:7000",
            ),
            "hf_name": param_name,
        }
        ep1_full_locator = {
            **_hf_locator(
                tp_rank=1,
                full_shape=[128, 64],
                slc=[[64, 128], [0, 64]],
                ptr=0x2000,
                nbytes=64 * 64 * 2,
                session_id="recv-1:7000",
            ),
            "hf_name": param_name,
        }
        ep0_cached_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "cached",
                "tensor_map": None,
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv-0:7000"}],
            },
        )
        ep0_full_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "full",
                "tensor_map": {param_name: [ep0_full_locator]},
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv-0:7000"}],
            },
        )
        ep1_full_response = _FakeResponse(
            200,
            {
                "success": True,
                "message": "full",
                "tensor_map": {param_name: [ep1_full_locator]},
                "receiver_transfer_engine_infos": [{"tp_rank": 0, "session_id": "recv-1:7000"}],
            },
        )

        with patch(
            "requests.post",
            side_effect=[ep0_cached_response, _FakeResponse(422, {}), ep0_full_response, ep1_full_response],
        ) as posted:
            ok = backend.initialize()

        assert ok is True
        assert posted.call_count == 4
        assert [call.args[0] for call in posted.call_args_list] == [
            "http://infer-0:5000/prepare_weights_update",
            "http://infer-1:5001/prepare_weights_update",
            "http://infer-0:5000/prepare_weights_update",
            "http://infer-1:5001/prepare_weights_update",
        ]
        assert posted.call_args_list[0].kwargs["json"]["p2p_return_tensor_map"] is False
        assert posted.call_args_list[1].kwargs["json"]["p2p_return_tensor_map"] is False
        assert "p2p_return_tensor_map" not in posted.call_args_list[2].kwargs["json"]
        assert "p2p_return_tensor_map" not in posted.call_args_list[3].kwargs["json"]
        locators = backend._tensor_map[param_name]
        assert {loc["endpoint_idx"] for loc in locators} == {0, 1}
        assert {loc["session_id"]: loc["ptr"] for loc in locators} == {
            "recv-0:7000": 0x1000,
            "recv-1:7000": 0x2000,
        }
        assert backend._receiver_session_ids == ["recv-0:7000", "recv-1:7000"]
        assert backend._last_prepare_returned_tensor_map is True

    def test_complete_sync_preserves_cached_prepare_state(self):
        backend, _ = _make_backend()
        cached_map = {
            "model.layers.0.self_attn.q_proj.weight": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=0xDEAD0000,
                        nbytes=64 * 64 * 2,
                        session_id="recv-0:7000",
                    ),
                    "hf_name": "model.layers.0.self_attn.q_proj.weight",
                },
            ]
        }
        backend._tensor_map = cached_map
        backend._receiver_session_ids = ["recv-0:7000"]
        backend._session_debug_info = {"recv-0:7000": {"session_id": "recv-0:7000"}}

        with patch("requests.post", return_value=_FakeResponse(200, {"success": True})):
            backend.complete_sync()

        assert backend._tensor_map == cached_map
        assert backend._receiver_session_ids == ["recv-0:7000"]
        assert backend._session_debug_info == {"recv-0:7000": {"session_id": "recv-0:7000"}}


class TestP2PSlicing:
    def test_slice_source_for_locator_extracts_qkv_q_slice(self):
        full = torch.arange(128 * 64, dtype=torch.bfloat16).reshape(128, 64)
        loc = _hf_locator(
            tp_rank=0,
            full_shape=[128, 64],
            slc=[[0, 64], [0, 64]],
            ptr=0,
            nbytes=64 * 64 * 2,
            session_id="x",
        )
        view = P2PTransportBackend._slice_source_for_locator("q_proj", full, loc)
        assert view is not None
        assert view.shape == (64, 64)
        # Equal to the literal slice of the source.
        assert torch.equal(view, full[0:64, 0:64])

    def test_slice_source_for_locator_other_rank_picks_other_rows(self):
        full = torch.arange(128 * 64, dtype=torch.bfloat16).reshape(128, 64)
        loc = _hf_locator(
            tp_rank=1,
            full_shape=[128, 64],
            slc=[[64, 128], [0, 64]],
            ptr=0,
            nbytes=64 * 64 * 2,
            session_id="x",
        )
        view = P2PTransportBackend._slice_source_for_locator("q_proj", full, loc)
        assert view is not None
        assert torch.equal(view, full[64:128, 0:64])

    def test_slice_source_returns_none_on_full_shape_mismatch(self):
        full = torch.zeros(128, 64, dtype=torch.bfloat16)
        loc = _hf_locator(
            tp_rank=0,
            full_shape=[256, 64],  # wrong on purpose
            slc=[[0, 128], [0, 64]],
            ptr=0,
            nbytes=128 * 64 * 2,
            session_id="x",
        )
        assert P2PTransportBackend._slice_source_for_locator("q_proj", full, loc) is None

    def test_slice_source_no_slice_returns_full_tensor(self):
        full = torch.zeros(8, 16, dtype=torch.bfloat16)
        loc = {"ptr": 0, "nbytes": 8 * 16 * 2}
        view = P2PTransportBackend._slice_source_for_locator("rn", full, loc)
        assert view is full

    def test_slice_source_squeezes_qwen35_linear_attention_conv_for_receiver_layout(self):
        full = torch.arange(8 * 1 * 4, dtype=torch.bfloat16).reshape(8, 1, 4)
        loc = _hf_locator(
            tp_rank=1,
            full_shape=[8, 4],
            slc=[[4, 8], [0, 4]],
            ptr=0,
            nbytes=4 * 4 * 2,
            session_id="x",
        )
        view = P2PTransportBackend._slice_source_for_locator(
            "model.layers.0.linear_attn.conv1d.weight",
            full,
            loc,
        )
        assert view is not None
        assert view.shape == (4, 4)
        assert torch.equal(view, full.squeeze(1)[4:8, 0:4])

    def test_slice_source_handles_qwen35_linear_attention_local_state_vector(self):
        full = torch.arange(32, dtype=torch.float32)
        loc = _hf_locator(
            tp_rank=2,
            full_shape=[8],
            slc=[[0, 8]],
            ptr=0,
            nbytes=8 * 4,
            session_id="x",
            dtype="float32",
        )
        view = P2PTransportBackend._slice_source_for_locator(
            "model.layers.0.linear_attn.A_log",
            full,
            loc,
        )
        assert view is not None
        assert view.shape == (8,)
        assert torch.equal(view, full[16:24])

    def test_slice_source_casts_qwen35_linear_attention_state_to_receiver_dtype(self):
        full = torch.arange(32, dtype=torch.bfloat16)
        loc = _hf_locator(
            tp_rank=2,
            full_shape=[32],
            slc=[[16, 24]],
            ptr=0,
            nbytes=8 * 4,
            session_id="x",
            dtype="float32",
        )
        view = P2PTransportBackend._slice_source_for_locator(
            "model.layers.0.linear_attn.A_log",
            full,
            loc,
        )
        assert view is not None
        assert view.shape == (8,)
        assert view.dtype == torch.float32
        assert view.numel() * view.element_size() == 8 * 4
        assert torch.equal(view, full[16:24].float())


class TestP2PTransferBucket:
    def _seed_tensor_map(self, backend: P2PTransportBackend, peer_ptr: int):
        backend._tensor_map = {
            "model.layers.0.self_attn.q_proj.weight": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[128, 64],
                        slc=[[0, 64], [0, 64]],
                        ptr=peer_ptr,
                        nbytes=64 * 64 * 2,
                        session_id="recv0:7000",
                    ),
                    "hf_name": "model.layers.0.self_attn.q_proj.weight",
                    "endpoint_idx": 0,
                },
                {
                    **_hf_locator(
                        tp_rank=1,
                        full_shape=[128, 64],
                        slc=[[64, 128], [0, 64]],
                        ptr=peer_ptr + 0x10000,
                        nbytes=64 * 64 * 2,
                        session_id="recv1:7000",
                    ),
                    "hf_name": "model.layers.0.self_attn.q_proj.weight",
                    "endpoint_idx": 0,
                },
            ]
        }
        backend._receiver_session_ids = ["recv0:7000", "recv1:7000"]

    def test_transfer_bucket_writes_correct_slice_per_receiver(self):
        backend, engine = _make_backend()
        self._seed_tensor_map(backend, peer_ptr=0xCAFE_0000)
        full = torch.arange(128 * 64, dtype=torch.bfloat16).reshape(128, 64)

        backend.transfer_bucket(
            [("model.layers.0.self_attn.q_proj.weight", full)],
            src_rank=0,
        )
        backend.flush_pending_transfers()

        # Two transfers issued — one per receiver.
        sessions = sorted(t[0] for t in engine.transfers)
        assert sessions == ["recv0:7000", "recv1:7000"]

        # Per-session: one buffer of 64*64*2 bytes.
        for session_id, src_ptrs, peer_ptrs, lengths in engine.transfers:
            assert len(src_ptrs) == len(peer_ptrs) == len(lengths) == 1
            assert lengths[0] == 64 * 64 * 2
            if session_id == "recv0:7000":
                assert peer_ptrs[0] == 0xCAFE_0000
            else:
                assert peer_ptrs[0] == 0xCAFE_0000 + 0x10000

    def test_transfer_bucket_fails_on_unknown_param(self):
        backend, engine = _make_backend()
        backend._tensor_map = {}
        full = torch.zeros(8, 8, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="no receiver locator"):
            backend.transfer_bucket([("unknown.param", full)], src_rank=0)
        assert engine.transfers == []

    def test_transfer_bucket_uses_language_model_receiver_prefix_fallback(self):
        backend, engine = _make_backend()
        receiver_name = "language_model.model.layers.0.self_attn.q_b_proj.weight"
        backend._tensor_map = {
            receiver_name: [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[4, 8],
                        slc=[[0, 4], [0, 8]],
                        ptr=0x4567_0000,
                        nbytes=4 * 8 * 2,
                        session_id="recv0:7000",
                    ),
                    "hf_name": receiver_name,
                    "endpoint_idx": 0,
                }
            ]
        }
        backend._receiver_session_ids = ["recv0:7000"]
        full = torch.zeros(4, 8, dtype=torch.bfloat16)

        backend.transfer_bucket(
            [("model.layers.0.self_attn.q_b_proj.weight", full)],
            src_rank=0,
        )
        backend.flush_pending_transfers()

        assert len(engine.transfers) == 1
        session_id, _src_ptrs, peer_ptrs, lengths = engine.transfers[0]
        assert session_id == "recv0:7000"
        assert peer_ptrs == [0x4567_0000]
        assert lengths == [full.numel() * full.element_size()]

    def test_transfer_bucket_fails_on_shape_mismatch(self):
        backend, engine = _make_backend()
        backend._tensor_map = {
            "param": [
                _hf_locator(
                    tp_rank=0,
                    full_shape=[16, 8],
                    slc=[[0, 16], [0, 8]],
                    ptr=0x1000,
                    nbytes=16 * 8 * 2,
                    session_id="recv0:7000",
                )
            ]
        }
        full = torch.zeros(8, 8, dtype=torch.bfloat16)

        with pytest.raises(RuntimeError, match="incomplete or incompatible"):
            backend.transfer_bucket([("param", full)], src_rank=0)

        assert engine.transfers == []

    def test_transfer_bucket_rejects_nonzero_src_rank(self):
        backend, _ = _make_backend()
        backend._tensor_map = {}
        try:
            backend.transfer_bucket([], src_rank=2)
        except ValueError as e:
            assert "src_rank=0" in str(e) or "src_rank" in str(e)
        else:
            raise AssertionError("expected ValueError for non-zero src_rank")

    def test_transfer_bucket_stashes_weight_version_and_flush_for_destroy(self):
        backend, _ = _make_backend()
        self._seed_tensor_map(backend, peer_ptr=0x1000)
        full = torch.zeros(128, 64, dtype=torch.bfloat16)
        backend.transfer_bucket(
            [("model.layers.0.self_attn.q_proj.weight", full)],
            src_rank=0,
            flush_cache=True,
            weight_version="rev-42",
        )
        backend.flush_pending_transfers()
        assert backend.config.backend_config["flush_cache"] is True
        assert backend.config.backend_config["weight_version"] == "rev-42"

    def test_transfer_bucket_preserves_requested_flush_cache(self):
        backend, _ = _make_backend()
        self._seed_tensor_map(backend, peer_ptr=0x1000)
        backend.config.backend_config["flush_cache"] = True
        full = torch.zeros(128, 64, dtype=torch.bfloat16)

        backend.transfer_bucket(
            [("model.layers.0.self_attn.q_proj.weight", full)],
            src_rank=0,
            flush_cache=False,
        )

        with patch(
            "requests.post",
            return_value=_FakeResponse(200, {"success": True, "message": "ok"}),
        ) as posted:
            backend.complete_sync()

        body = posted.call_args.kwargs["json"]
        assert body["flush_cache"] is True

    def test_transfer_bucket_stages_small_cpu_tensors_through_cpu_pool(self):
        backend, engine = _make_backend()
        backend._tensor_map = {
            "model.layers.0.mlp.gate_proj.weight_scale_inv": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[2, 2],
                        slc=[[0, 2], [0, 2]],
                        ptr=0x1234_0000,
                        nbytes=2 * 2 * 4,
                        session_id="recv0:7000",
                        dtype="float32",
                    ),
                    "hf_name": "model.layers.0.mlp.gate_proj.weight_scale_inv",
                    "endpoint_idx": 0,
                }
            ]
        }
        backend._receiver_session_ids = ["recv0:7000"]
        scale = torch.ones(2, 2, dtype=torch.float32)

        backend.transfer_bucket(
            [("model.layers.0.mlp.gate_proj.weight_scale_inv", scale)],
            src_rank=0,
        )
        backend.flush_pending_transfers()

        assert engine.registered_memory
        assert engine.registered == []
        assert len(engine.transfers) == 1
        session_id, _src_ptrs, peer_ptrs, lengths = engine.transfers[0]
        assert session_id == "recv0:7000"
        assert peer_ptrs == [0x1234_0000]
        assert lengths == [scale.numel() * scale.element_size()]

    def test_transfer_bucket_aligns_cpu_scratch_offsets_before_dtype_views(self, monkeypatch):
        monkeypatch.setenv("XORL_P2P_MOONCAKE_TRANSFER_CHUNK", "16")
        backend, engine = _make_backend()
        backend._tensor_map = {
            "model.layers.0.mlp.gate_proj.weight": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[3],
                        slc=[[0, 3]],
                        ptr=0x2000,
                        nbytes=3,
                        session_id="recv0:7000",
                        dtype="float8_e4m3fn",
                    ),
                    "hf_name": "model.layers.0.mlp.gate_proj.weight",
                }
            ],
            "model.layers.0.mlp.gate_proj.weight_scale_inv": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[1],
                        slc=[[0, 1]],
                        ptr=0x2003,
                        nbytes=4,
                        session_id="recv0:7000",
                        dtype="float32",
                    ),
                    "hf_name": "model.layers.0.mlp.gate_proj.weight_scale_inv",
                }
            ],
        }
        backend._receiver_session_ids = ["recv0:7000"]

        backend.transfer_bucket(
            [
                ("model.layers.0.mlp.gate_proj.weight", torch.ones(3, dtype=torch.uint8)),
                ("model.layers.0.mlp.gate_proj.weight_scale_inv", torch.ones(1, dtype=torch.float32)),
            ],
            src_rank=0,
        )
        backend.flush_pending_transfers()

        assert len(engine.transfers) == 1
        session_id, src_ptrs, peer_ptrs, lengths = engine.transfers[0]
        assert session_id == "recv0:7000"
        assert peer_ptrs == [0x2000, 0x2003]
        assert lengths == [3, 4]
        assert src_ptrs[1] % 4 == 0

    def test_transfer_bucket_does_not_coalesce_across_receiver_memory_handles(self, monkeypatch):
        monkeypatch.setenv("XORL_P2P_MOONCAKE_TRANSFER_CHUNK", "16")
        backend, engine = _make_backend()
        backend._tensor_map = {
            "param_a": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[2],
                        slc=[[0, 2]],
                        ptr=0x1000,
                        nbytes=4,
                        session_id="recv0:7000",
                    ),
                    "hf_name": "param_a",
                    "memory_handle": 0x1000,
                }
            ],
            "param_b": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[2],
                        slc=[[0, 2]],
                        ptr=0x1004,
                        nbytes=4,
                        session_id="recv0:7000",
                    ),
                    "hf_name": "param_b",
                    "memory_handle": 0x2000,
                }
            ],
        }
        backend._receiver_session_ids = ["recv0:7000"]

        backend.transfer_bucket(
            [
                ("param_a", torch.ones(2, dtype=torch.bfloat16)),
                ("param_b", torch.ones(2, dtype=torch.bfloat16)),
            ],
            src_rank=0,
        )
        backend.flush_pending_transfers()

        assert len(engine.transfers) == 1
        session_id, _src_ptrs, peer_ptrs, lengths = engine.transfers[0]
        assert session_id == "recv0:7000"
        assert peer_ptrs == [0x1000, 0x1004]
        assert lengths == [4, 4]

    def test_transfer_bucket_does_not_coalesce_without_receiver_memory_handles(self, monkeypatch):
        monkeypatch.setenv("XORL_P2P_MOONCAKE_TRANSFER_CHUNK", "16")
        backend, engine = _make_backend()
        backend._tensor_map = {
            "param_a": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[2],
                        slc=[[0, 2]],
                        ptr=0x1000,
                        nbytes=4,
                        session_id="recv0:7000",
                    ),
                    "hf_name": "param_a",
                }
            ],
            "param_b": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[2],
                        slc=[[0, 2]],
                        ptr=0x1004,
                        nbytes=4,
                        session_id="recv0:7000",
                    ),
                    "hf_name": "param_b",
                }
            ],
        }
        backend._receiver_session_ids = ["recv0:7000"]

        backend.transfer_bucket(
            [
                ("param_a", torch.ones(2, dtype=torch.bfloat16)),
                ("param_b", torch.ones(2, dtype=torch.bfloat16)),
            ],
            src_rank=0,
        )
        backend.flush_pending_transfers()

        assert len(engine.transfers) == 1
        session_id, _src_ptrs, peer_ptrs, lengths = engine.transfers[0]
        assert session_id == "recv0:7000"
        assert peer_ptrs == [0x1000, 0x1004]
        assert lengths == [4, 4]

    def test_transfer_bucket_failure_names_tensor_and_receiver_handle(self, monkeypatch):
        monkeypatch.setenv("XORL_P2P_TRANSFER_RETRIES", "1")
        backend, engine = _make_backend()
        engine.fail_transfer = True
        backend._tensor_map = {
            "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv": [
                {
                    **_hf_locator(
                        tp_rank=0,
                        full_shape=[2],
                        slc=[[0, 2]],
                        ptr=0x3000,
                        nbytes=8,
                        session_id="recv0:7000",
                        dtype="float32",
                    ),
                    "hf_name": "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv",
                    "memory_handle": 0x3000,
                }
            ]
        }
        backend._receiver_session_ids = ["recv0:7000"]

        backend.transfer_bucket(
            [
                (
                    "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv",
                    torch.ones(2, dtype=torch.float32),
                )
            ],
            src_rank=0,
        )
        with pytest.raises(RuntimeError) as exc_info:
            backend.flush_pending_transfers()

        message = str(exc_info.value)
        assert "weight_scale_inv" in message
        assert "handle=0x3000" in message
        assert "ptr=0x3000" in message


class TestP2PDirectEPTransfer:
    def _make_multi_sender_backend(
        self, *, rank_index: int, world_size: int, rank_filter
    ) -> Tuple[P2PTransportBackend, _FakeMooncakeEngine]:
        cfg = TransportConfig(
            endpoints=[EndpointConfig(host="infer-0", port=5000, world_size=2)],
            master_address="trainer-0",
            master_port=0,
            group_name="weight_sync_group",
            buffer_size_mb=64,
            device="cuda:0",
            training_rank=rank_index,
            backend_config={
                "hostname": "trainer-0",
                "gpu_id": 0,
                "direct_ep_transfer": True,
                "world_size": world_size,
                "rank_index": rank_index,
                "rank_filter": rank_filter,
                "cpu_scratch_pool_bytes": 1024 * 1024,
            },
        )
        backend = P2PTransportBackend(cfg)
        fake_engine = _FakeMooncakeEngine(session_id=f"trainer-{rank_index}:1234")
        backend._engine = fake_engine
        backend._make_local_engine = lambda: fake_engine  # type: ignore[assignment]
        return backend, fake_engine

    def test_supports_direct_ep_transfer_advertises_all_ranks(self):
        backend, _ = self._make_multi_sender_backend(rank_index=0, world_size=4, rank_filter=lambda loc: True)
        assert backend.supports_direct_ep_transfer is True
        assert backend.supports_direct_pp_transfer is False
        assert backend.sender_ranks == frozenset({0, 1, 2, 3})

    def test_adopt_prepared_state_skips_http(self):
        backend, _ = self._make_multi_sender_backend(rank_index=2, world_size=4, rank_filter=lambda loc: True)
        tensor_map = {
            "model.layers.0.self_attn.q_proj.weight": [
                _hf_locator(
                    tp_rank=0,
                    full_shape=[128, 64],
                    slc=[[0, 64], [0, 64]],
                    ptr=0xAAAA0000,
                    nbytes=64 * 64 * 2,
                    session_id="recv:7000",
                ),
            ]
        }
        # Should not raise, should not POST anything.
        ok = backend.adopt_prepared_state(tensor_map, ["recv:7000"])
        assert ok is True
        assert backend._tensor_map == tensor_map
        assert backend._receiver_session_ids == ["recv:7000"]

    def test_initialize_multi_sender_propagates_nonzero_rank_failure_to_rank0(self):
        backend, _ = self._make_multi_sender_backend(rank_index=0, world_size=2, rank_filter=lambda loc: True)

        def initialize_rank0():
            backend._receiver_session_ids = ["recv:7000"]
            backend._last_prepare_returned_tensor_map = False
            return True

        all_gather_calls = []

        def all_gather_object(output, value):
            all_gather_calls.append(value)
            if len(all_gather_calls) == 1:
                output[:] = [False, False]
            else:
                output[:] = [True, False]

        backend._initialize_single_sender = initialize_rank0  # type: ignore[assignment]

        with patch("xorl.server.weight_sync.backends.p2p.dist.is_available", return_value=True):
            with patch("xorl.server.weight_sync.backends.p2p.dist.is_initialized", return_value=True):
                with patch("xorl.server.weight_sync.backends.p2p.dist.broadcast_object_list"):
                    with patch(
                        "xorl.server.weight_sync.backends.p2p.dist.all_gather_object",
                        side_effect=all_gather_object,
                    ):
                        assert backend.initialize() is False

        assert all_gather_calls == [False, True]

    def test_rank_filter_routes_slices_to_owning_rank(self):
        # Locators tagged with ep_rank; each sender ships only its own.
        full = torch.arange(128 * 64, dtype=torch.bfloat16).reshape(128, 64)
        locators = [
            {
                **_hf_locator(
                    tp_rank=tp,
                    full_shape=[128, 64],
                    slc=[[tp * 64, (tp + 1) * 64], [0, 64]],
                    ptr=0xBEEF_0000 + tp * 0x10000,
                    nbytes=64 * 64 * 2,
                    session_id=f"recv-{tp}:7000",
                ),
                "hf_name": "model.layers.0.self_attn.q_proj.weight",
                "ep_rank": tp,
                "endpoint_idx": 0,
            }
            for tp in (0, 1)
        ]
        bucket = [("model.layers.0.self_attn.q_proj.weight", full)]

        rank0_engine_calls: List[Tuple[str, List[int], List[int], List[int]]] = []
        rank1_engine_calls: List[Tuple[str, List[int], List[int], List[int]]] = []

        for rank_index, sink in ((0, rank0_engine_calls), (1, rank1_engine_calls)):
            backend, engine = self._make_multi_sender_backend(
                rank_index=rank_index,
                world_size=2,
                rank_filter=lambda loc, r=rank_index: loc.get("ep_rank") == r,
            )
            backend._tensor_map = {locators[0]["hf_name"]: locators}
            backend._receiver_session_ids = [locators[0]["session_id"], locators[1]["session_id"]]
            backend.transfer_bucket(bucket, src_rank=rank_index)
            backend.flush_pending_transfers()
            sink.extend(engine.transfers)

        # Each rank issued exactly one transfer to its owning receiver.
        assert len(rank0_engine_calls) == 1
        assert rank0_engine_calls[0][0] == "recv-0:7000"
        assert len(rank1_engine_calls) == 1
        assert rank1_engine_calls[0][0] == "recv-1:7000"

    def test_transfer_bucket_accepts_nonzero_src_rank_when_direct_ep(self):
        backend, _ = self._make_multi_sender_backend(rank_index=3, world_size=4, rank_filter=lambda loc: False)
        backend._tensor_map = {}
        # Should NOT raise even though src_rank != 0.
        backend.transfer_bucket([], src_rank=3)

    def test_transfer_bucket_all_filtered_out_is_ok_for_direct_ep_rank(self):
        backend, engine = self._make_multi_sender_backend(rank_index=3, world_size=4, rank_filter=lambda loc: False)
        backend._tensor_map = {
            "param": [
                _hf_locator(
                    tp_rank=0,
                    full_shape=[8, 8],
                    slc=[[0, 8], [0, 8]],
                    ptr=0x1000,
                    nbytes=8 * 8 * 2,
                    session_id="recv0:7000",
                )
            ]
        }
        backend.transfer_bucket([("param", torch.zeros(8, 8, dtype=torch.bfloat16))], src_rank=3)
        backend.flush_pending_transfers()
        assert engine.transfers == []


class TestP2PDestroy:
    def test_complete_sync_does_not_complete_receiver_after_pending_transfer_failure(self):
        backend, _ = _make_backend()
        future: Future[None] = Future()
        future.set_exception(RuntimeError("mooncake transfer failed"))
        backend._cpu_pool_pending_futures[0] = future

        with patch("requests.post") as posted:
            with pytest.raises(RuntimeError, match="mooncake transfer failed"):
                backend.complete_sync()

        posted.assert_not_called()
        assert backend._cpu_pool_pending_futures[0] is None

    def test_destroy_can_skip_receiver_completion_for_failed_sync_cleanup(self):
        backend, engine = _make_backend()
        backend._registered_source_ptrs = [0x1, 0x2]

        with patch("requests.post") as posted:
            backend.destroy(complete_receiver=False)

        posted.assert_not_called()
        assert engine.deregistered == [[0x1, 0x2]]
        assert backend._engine is None

    def test_destroy_skip_completion_drains_failed_pending_transfers_without_raising(self):
        backend, _ = _make_backend()
        future: Future[None] = Future()
        future.set_exception(RuntimeError("transfer failed during cleanup"))
        backend._cpu_pool_pending_futures[0] = future

        with patch("requests.post") as posted:
            backend.destroy(complete_receiver=False)

        posted.assert_not_called()
        assert backend._cpu_pool_pending_futures == [None] * backend._n_pools
        assert backend._engine is None

    def test_destroy_posts_complete_with_correct_payload(self):
        backend, engine = _make_backend()
        # Pretend transfer_bucket already ran and stashed flush/version state.
        backend.config.backend_config["flush_cache"] = True
        backend.config.backend_config["weight_version"] = "rev-7"
        backend._registered_source_ptrs = [0x1, 0x2]

        with patch(
            "requests.post",
            return_value=_FakeResponse(200, {"success": True, "message": "ok"}),
        ) as posted:
            backend.destroy()

        # /complete_weights_update was called with the right body.
        posted.assert_called_once()
        url = posted.call_args.args[0]
        body = posted.call_args.kwargs["json"]
        assert url == "http://infer-0:5000/complete_weights_update"
        assert body["transport"] == "p2p"
        assert body["group_name"] == "weight_sync_group"
        assert body["flush_cache"] is True
        assert body["weight_version"] == "rev-7"

        # Source pointers were deregistered.
        assert engine.deregistered == [[0x1, 0x2]]

    def test_destroy_raises_after_complete_failure_but_cleans_up(self):
        backend, _ = _make_backend()
        with patch("requests.post", return_value=_FakeResponse(500, {})):
            with pytest.raises(RuntimeError, match="/complete_weights_update failed"):
                backend.destroy()
        assert backend._engine is None

    def test_complete_sync_defaults_flush_cache_false(self):
        backend, _ = _make_backend()
        with patch(
            "requests.post",
            return_value=_FakeResponse(200, {"success": True, "message": "ok"}),
        ) as posted:
            backend.complete_sync()

        body = posted.call_args.kwargs["json"]
        assert body["flush_cache"] is False


class TestP2PEngineConstruction:
    def test_make_local_engine_falls_back_without_sglang_wrapper(self, monkeypatch):
        class FakeTransferEngine:
            def initialize(self, hostname, protocol, transport, device):
                self.initialized = (hostname, protocol, transport, device)
                return 0

            def get_rpc_port(self):
                return 12345

            def batch_register_memory(self, _ptrs, _lengths):
                return 0

            def batch_unregister_memory(self, _ptrs):
                return 0

            def batch_transfer_sync_write(self, _session_id, _buffers, _peer_buffer_addresses, _lengths):
                return 0

        mooncake_mod = types.ModuleType("mooncake")
        mooncake_engine_mod = types.ModuleType("mooncake.engine")
        mooncake_engine_mod.TransferEngine = FakeTransferEngine
        monkeypatch.setitem(sys.modules, "mooncake", mooncake_mod)
        monkeypatch.setitem(sys.modules, "mooncake.engine", mooncake_engine_mod)
        monkeypatch.setitem(sys.modules, "sglang", None)

        cfg = TransportConfig(
            endpoints=[EndpointConfig(host="infer-0", port=5000, world_size=1)],
            group_name="weight_sync_group",
            backend_config={"hostname": "trainer-0", "gpu_id": 0, "ib_device": "mlx5_0"},
        )
        backend = P2PTransportBackend(cfg)
        engine = backend._make_local_engine()

        assert engine is not None
        assert engine.get_session_id() == "trainer-0:12345"
        assert engine.get_ib_device() == "mlx5_0"
        assert engine.engine.initialized == ("trainer-0", "P2PHANDSHAKE", "rdma", "mlx5_0")
        assert backend._engine is None
