import pytest

from xorl.server.weight_sync.backends import p2p
from xorl.server.weight_sync.backends.base import EndpointConfig, TransportConfig
from xorl.server.weight_sync.backends.p2p import P2PTransportBackend, _BucketTiming, _do_async_transfer


class DoneEvent:
    def synchronize(self):
        return None


class FakeAsyncEngine:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.submitted = []

    def batch_transfer_async_write(self, session_id, src_ptrs, peer_ptrs, lengths):
        self.submitted.append((session_id, src_ptrs, peer_ptrs, lengths))
        return len(self.submitted)

    def get_batch_transfer_status(self, _bids):
        if self.statuses:
            return self.statuses.pop(0)
        return 1


class FakeEngineWrapper:
    def __init__(self, statuses):
        self.engine = FakeAsyncEngine(statuses)
        self.sync_submitted = []

    def batch_transfer_sync(self, session_id, src_ptrs, peer_ptrs, lengths):
        self.sync_submitted.append((session_id, src_ptrs, peer_ptrs, lengths))
        return 0


class FakeLocalEngine:
    def get_session_id(self):
        return "sender-session"

    def get_ib_device(self):
        return "mlx5_0"


class FakePrepareResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "success": True,
            "tensor_map": {
                "model.embed_tokens.weight": [
                    {
                        "hf_name": "model.embed_tokens.weight",
                        "tp_rank": 0,
                        "slice": [[0, 1], [0, 1]],
                        "full_shape": [1, 1],
                        "ptr": 1234,
                        "nbytes": 2,
                        "session_id": "receiver-session",
                    }
                ]
            },
            "receiver_transfer_engine_infos": [{"session_id": "receiver-session"}],
        }


def _session_entries(src_ptrs, peer_ptrs, lengths):
    return (src_ptrs, peer_ptrs, lengths, [])


def test_async_api_success_status_zero_completes(monkeypatch):
    monkeypatch.setenv("XORL_P2P_USE_ASYNC_API", "1")
    wrapper = FakeEngineWrapper([0])
    timing = _BucketTiming()

    _do_async_transfer(
        engine_wrapper=wrapper,
        copy_done_event=DoneEvent(),
        by_session={"session-a": _session_entries([1], [2], [128 * 1024 * 1024])},
        small_session_data={},
        session_debug_info={"session-a": {"world_rank": 0}},
        small_register_ptrs=[],
        small_register_lens=[],
        chunk=1,
        timing=timing,
        bucket_idx=1,
        slice_holds=[],
        src_view_holds=[],
    )

    assert wrapper.engine.submitted == [("session-a", [1], [2], [128 * 1024 * 1024])]
    assert wrapper.sync_submitted == []
    assert timing.transfer_s >= 0


def test_async_api_uses_sync_fallback_for_medium_chunks(monkeypatch):
    monkeypatch.setenv("XORL_P2P_USE_ASYNC_API", "1")
    wrapper = FakeEngineWrapper([0])
    timing = _BucketTiming()

    _do_async_transfer(
        engine_wrapper=wrapper,
        copy_done_event=DoneEvent(),
        by_session={"session-a": _session_entries([1], [2], [12 * 1024 * 1024])},
        small_session_data={},
        session_debug_info={"session-a": {"world_rank": 0}},
        small_register_ptrs=[],
        small_register_lens=[],
        chunk=1,
        timing=timing,
        bucket_idx=1,
        slice_holds=[],
        src_view_holds=[],
    )

    assert wrapper.engine.submitted == []
    assert wrapper.sync_submitted == [("session-a", [1], [2], [12 * 1024 * 1024])]
    assert timing.transfer_s >= 0


def test_async_api_min_bytes_env_controls_cutoff(monkeypatch):
    monkeypatch.setenv("XORL_P2P_USE_ASYNC_API", "1")
    monkeypatch.setenv("XORL_P2P_ASYNC_MIN_BYTES", str(8 * 1024 * 1024))
    wrapper = FakeEngineWrapper([0])
    timing = _BucketTiming()

    _do_async_transfer(
        engine_wrapper=wrapper,
        copy_done_event=DoneEvent(),
        by_session={"session-a": _session_entries([1], [2], [12 * 1024 * 1024])},
        small_session_data={},
        session_debug_info={"session-a": {"world_rank": 0}},
        small_register_ptrs=[],
        small_register_lens=[],
        chunk=1,
        timing=timing,
        bucket_idx=1,
        slice_holds=[],
        src_view_holds=[],
    )

    assert wrapper.engine.submitted == [("session-a", [1], [2], [12 * 1024 * 1024])]
    assert wrapper.sync_submitted == []
    assert timing.transfer_s >= 0


def test_async_api_status_poll_timeout(monkeypatch):
    monkeypatch.setenv("XORL_P2P_USE_ASYNC_API", "1")
    monkeypatch.setenv("XORL_P2P_ASYNC_STATUS_TIMEOUT_S", "0.001")
    wrapper = FakeEngineWrapper([1])

    with pytest.raises(RuntimeError, match="async transfer status poll timed out"):
        _do_async_transfer(
            engine_wrapper=wrapper,
            copy_done_event=DoneEvent(),
            by_session={"session-a": _session_entries([1], [2], [128 * 1024 * 1024])},
            small_session_data={},
            session_debug_info={"session-a": {"world_rank": 0}},
            small_register_ptrs=[],
            small_register_lens=[],
            chunk=1,
            timing=_BucketTiming(),
            bucket_idx=7,
            slice_holds=[],
            src_view_holds=[],
        )


def test_prepare_uses_prepare_timeout_env(monkeypatch):
    monkeypatch.setenv("XORL_P2P_PREPARE_TIMEOUT_S", "12.5")
    seen = {}

    def fake_post(_url, *, json, timeout):
        seen["payload"] = json
        seen["timeout"] = timeout
        return FakePrepareResponse()

    monkeypatch.setattr(p2p.requests, "post", fake_post)
    backend = P2PTransportBackend(
        TransportConfig(
            endpoints=[EndpointConfig(host="receiver", port=30000, world_size=8)],
            group_name="test-group",
            training_rank=0,
        )
    )
    monkeypatch.setattr(backend, "_make_local_engine", lambda: FakeLocalEngine())

    assert backend._initialize_single_sender()
    assert seen["timeout"] == 12.5
    assert seen["payload"]["transport"] == "p2p"
