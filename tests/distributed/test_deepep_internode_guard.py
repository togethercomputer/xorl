"""Tests for the DeepEP internode preflight probe and topology detection.

A fabric/node fault that NCCL rides out can leave NVSHMEM (DeepEP's internode
transport) undeliverable: the first MoE dispatch then wedges the gang with
moe_recv_expert_counter=-1 after minutes of model loading. The preflight runs a
tiny dispatch+combine before weight loading so the failure surfaces in seconds
with the participating hostnames attached. Verifies:
1. _ep_group_spans_nodes topology detection from the torchrun rank layout.
2. preflight_internode_transport no-ops when skipped/intranode/undeterminable,
   and wraps transport failures in an actionable error naming the nodes.
3. DeepEPBuffer.init_buffer validates DeepEP's int32 NVL-byte limit for
   internode buffers instead of dying in a C++ assert.
"""

from types import SimpleNamespace

import pytest

from xorl.distributed.moe import deepep


pytestmark = pytest.mark.cpu


class _FakeGroup:
    """Hashable stand-in for a ProcessGroup."""


# ---------------------------------------------------------------------------
# 1. _ep_group_spans_nodes
# ---------------------------------------------------------------------------


class TestEPGroupSpansNodes:
    def test_unknown_when_dist_not_initialized(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: False)
        assert deepep._ep_group_spans_nodes(None) is None

    def test_unknown_without_local_world_size(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        assert deepep._ep_group_spans_nodes(None) is None

    def test_unknown_with_malformed_local_world_size(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "not-a-number")
        assert deepep._ep_group_spans_nodes(None) is None

    def test_single_node_world_never_spans(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(deepep.dist, "get_world_size", lambda: 8)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert deepep._ep_group_spans_nodes(None) is False

    def test_intranode_ep_group_in_multinode_world(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(deepep.dist, "get_world_size", lambda: 16)
        monkeypatch.setattr(deepep.dist, "get_process_group_ranks", lambda group: list(range(8)))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert deepep._ep_group_spans_nodes(_FakeGroup()) is False

    def test_internode_ep_group(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(deepep.dist, "get_world_size", lambda: 16)
        monkeypatch.setattr(deepep.dist, "get_process_group_ranks", lambda group: list(range(16)))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert deepep._ep_group_spans_nodes(_FakeGroup()) is True

    def test_strided_internode_ep_group(self, monkeypatch):
        """ep_intranode=False layouts stride EP groups across nodes."""
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(deepep.dist, "get_world_size", lambda: 16)
        monkeypatch.setattr(deepep.dist, "get_process_group_ranks", lambda group: [0, 8])
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert deepep._ep_group_spans_nodes(_FakeGroup()) is True


# ---------------------------------------------------------------------------
# 2. preflight_internode_transport
# ---------------------------------------------------------------------------


def _arm_internode(monkeypatch, world=2, hosts=("node-a", "node-b")):
    monkeypatch.setattr(deepep, "_ep_group_spans_nodes", lambda group: True)
    monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(deepep.dist, "get_world_size", lambda group=None: world)

    def fake_gather(out, obj, group=None):
        for i in range(len(out)):
            out[i] = hosts[i % len(hosts)]

    monkeypatch.setattr(deepep.dist, "all_gather_object", fake_gather)


class TestPreflightInternodeTransport:
    def test_skipped_via_env(self, monkeypatch):
        monkeypatch.setenv(deepep._SKIP_PREFLIGHT_ENV, "1")

        def _fail(group):
            raise AssertionError("topology check must not run when the preflight is skipped")

        monkeypatch.setattr(deepep, "_ep_group_spans_nodes", _fail)
        deepep.preflight_internode_transport(None, hidden_dim=128)

    def test_noop_when_dist_not_initialized(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: False)
        deepep.preflight_internode_transport(None, hidden_dim=128)

    def test_noop_when_intranode(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(deepep, "_ep_group_spans_nodes", lambda group: False)

        def _no_buffer(**kwargs):
            raise AssertionError("no buffer should be created for intranode EP groups")

        monkeypatch.setattr(deepep, "get_default_buffer", _no_buffer)
        deepep.preflight_internode_transport(None, hidden_dim=128)

    def test_transport_failure_names_nodes(self, monkeypatch):
        _arm_internode(monkeypatch)
        buf = SimpleNamespace(init_buffer=lambda hidden_bytes: None)
        monkeypatch.setattr(deepep, "get_default_buffer", lambda **kwargs: buf)

        def dead_dispatch(*args, **kwargs):
            raise RuntimeError("DeepEP error: CPU recv timeout")

        monkeypatch.setattr(deepep, "dispatch_no_grad", dead_dispatch)
        with pytest.raises(RuntimeError, match="node-a") as exc_info:
            deepep.preflight_internode_transport(_FakeGroup(), hidden_dim=128)
        msg = str(exc_info.value)
        assert "node-b" in msg
        assert "CPU recv timeout" in msg
        assert deepep._SKIP_PREFLIGHT_ENV in msg

    def test_roundtrip_corruption_detected(self, monkeypatch):
        _arm_internode(monkeypatch)
        buf = SimpleNamespace(init_buffer=lambda hidden_bytes: None)
        monkeypatch.setattr(deepep, "get_default_buffer", lambda **kwargs: buf)

        def identity_dispatch(buffer, x, w, idx, num_experts):
            return x, idx, w, [1] * num_experts, "handle"

        monkeypatch.setattr(deepep, "dispatch_no_grad", identity_dispatch)
        monkeypatch.setattr(deepep, "combine_no_grad", lambda buffer, x, handle: x + 1.0)
        with pytest.raises(RuntimeError, match="corrupted"):
            deepep.preflight_internode_transport(_FakeGroup(), hidden_dim=128)

    def test_healthy_roundtrip_passes(self, monkeypatch):
        _arm_internode(monkeypatch)
        buf = SimpleNamespace(init_buffer=lambda hidden_bytes: None)
        monkeypatch.setattr(deepep, "get_default_buffer", lambda **kwargs: buf)

        def identity_dispatch(buffer, x, w, idx, num_experts):
            return x, idx, w, [1] * num_experts, "handle"

        monkeypatch.setattr(deepep, "dispatch_no_grad", identity_dispatch)
        monkeypatch.setattr(deepep, "combine_no_grad", lambda buffer, x, handle: x.clone())
        deepep.preflight_internode_transport(_FakeGroup(), hidden_dim=128)


# ---------------------------------------------------------------------------
# 3. DeepEPBuffer.init_buffer size validation
# ---------------------------------------------------------------------------


class TestBufferSizeValidation:
    def _buffer(self, monkeypatch, buffer_size_gb, rdma_bytes):
        """Build a DeepEPBuffer with deep_ep mocked so init_buffer runs on CPU."""
        captured = {}

        class FakeConfig:
            def get_nvl_buffer_size_hint(self, hidden_bytes, num_ranks):
                return 0

            def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
                return rdma_bytes

        class FakeDeepEP:
            class Buffer:
                @staticmethod
                def set_num_sms(n):
                    pass

                @staticmethod
                def get_dispatch_config(num_ranks):
                    return FakeConfig()

                @staticmethod
                def get_combine_config(num_ranks):
                    return FakeConfig()

                def __init__(self, **kwargs):
                    captured.update(kwargs)

        monkeypatch.setattr(deepep, "DEEPEP_AVAILABLE", True)
        monkeypatch.setattr(deepep, "deep_ep", FakeDeepEP)
        group = SimpleNamespace(size=lambda: 16)
        buf = deepep.DeepEPBuffer(ep_group=group, buffer_size_gb=buffer_size_gb)
        return buf, captured

    def test_oversized_nvl_with_rdma_raises(self, monkeypatch):
        buf, _ = self._buffer(monkeypatch, buffer_size_gb=4.0, rdma_bytes=1 << 20)
        with pytest.raises(ValueError, match="int32 limit"):
            buf.init_buffer(hidden_bytes=4096)

    def test_oversized_nvl_without_rdma_allowed(self, monkeypatch):
        buf, captured = self._buffer(monkeypatch, buffer_size_gb=4.0, rdma_bytes=0)
        buf.init_buffer(hidden_bytes=4096)
        assert captured["num_nvl_bytes"] == 4_000_000_000

    def test_unaligned_bytes_rounded_down(self, monkeypatch):
        # 0.000001 GB = 1000 bytes -> 896 after 128-byte alignment
        buf, captured = self._buffer(monkeypatch, buffer_size_gb=1e-6, rdma_bytes=0)
        buf.init_buffer(hidden_bytes=4096)
        assert captured["num_nvl_bytes"] == 896

    def test_default_two_gb_with_rdma_passes(self, monkeypatch):
        buf, captured = self._buffer(monkeypatch, buffer_size_gb=2.0, rdma_bytes=1 << 20)
        buf.init_buffer(hidden_bytes=4096)
        assert captured["num_nvl_bytes"] == 2_000_000_000
        assert captured["num_rdma_bytes"] == 1 << 20
