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
import torch

from xorl.distributed.moe import deepep


pytestmark = pytest.mark.cpu


class _FakeGroup:
    """Hashable stand-in for a ProcessGroup."""


def _assert_async_combine_requires_explicit_unsafe_opt_in(monkeypatch):
    captured = {}

    def fake_apply(expert_output, buffer, ctx, async_combine):
        del buffer, ctx
        captured["async_combine"] = async_combine
        return expert_output

    monkeypatch.delenv("XORL_DEEPEP_UNSAFE_ASYNC_COMBINE", raising=False)
    monkeypatch.setattr(deepep._FusedUnpermuteAndCombine, "apply", staticmethod(fake_apply))

    expert_output = torch.ones(1, 2)
    result = deepep.tokens_post_combine(
        buffer=None,
        expert_output=expert_output,
        ctx=SimpleNamespace(),
        async_combine=True,
    )

    assert result is expert_output
    assert captured["async_combine"] is False

    monkeypatch.setenv("XORL_DEEPEP_UNSAFE_ASYNC_COMBINE", "1")
    result = deepep.tokens_post_combine(
        buffer=None,
        expert_output=expert_output,
        ctx=SimpleNamespace(),
        async_combine=True,
    )

    assert result is expert_output
    assert captured["async_combine"] is True


# ---------------------------------------------------------------------------
# 1. _ep_group_spans_nodes
# ---------------------------------------------------------------------------


class TestEPGroupSpansNodes:
    def _assert_topology_truth_table(self, monkeypatch):
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: False)
        assert deepep._ep_group_spans_nodes(None) is None

        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        assert deepep._ep_group_spans_nodes(None) is None

        monkeypatch.setenv("LOCAL_WORLD_SIZE", "not-a-number")
        assert deepep._ep_group_spans_nodes(None) is None

        monkeypatch.setattr(deepep.dist, "get_world_size", lambda: 8)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert deepep._ep_group_spans_nodes(None) is False

        monkeypatch.setattr(deepep.dist, "get_world_size", lambda: 16)
        monkeypatch.setattr(deepep.dist, "get_process_group_ranks", lambda group: list(range(8)))
        assert deepep._ep_group_spans_nodes(_FakeGroup()) is False

        monkeypatch.setattr(deepep.dist, "get_process_group_ranks", lambda group: list(range(16)))
        assert deepep._ep_group_spans_nodes(_FakeGroup()) is True

        # ep_intranode=False layouts stride EP groups across nodes.
        monkeypatch.setattr(deepep.dist, "get_process_group_ranks", lambda group: [0, 8])
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
    def _assert_preflight_internode_transport_policy(self, monkeypatch):
        def _no_buffer(**kwargs):
            raise AssertionError("no buffer should be created when the preflight is disabled")

        monkeypatch.setattr(deepep, "get_default_buffer", _no_buffer)
        monkeypatch.setenv(deepep._SKIP_PREFLIGHT_ENV, "1")

        def _fail(group):
            raise AssertionError("topology check must not run when the preflight is skipped")

        monkeypatch.setattr(deepep, "_ep_group_spans_nodes", _fail)
        deepep.preflight_internode_transport(None, hidden_dim=128)

        monkeypatch.delenv(deepep._SKIP_PREFLIGHT_ENV)
        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: False)
        deepep.preflight_internode_transport(None, hidden_dim=128)

        monkeypatch.setattr(deepep.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(deepep, "_ep_group_spans_nodes", lambda group: False)
        deepep.preflight_internode_transport(None, hidden_dim=128)

        self._assert_transport_failure_names_nodes(monkeypatch)
        self._assert_roundtrip_accepts_identity_and_rejects_corruption(monkeypatch)

    def _assert_transport_failure_names_nodes(self, monkeypatch):
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

    def _assert_roundtrip_accepts_identity_and_rejects_corruption(self, monkeypatch):
        _arm_internode(monkeypatch)
        buf = SimpleNamespace(init_buffer=lambda hidden_bytes: None)
        monkeypatch.setattr(deepep, "get_default_buffer", lambda **kwargs: buf)

        def identity_dispatch(buffer, x, w, idx, num_experts):
            return x, idx, w, [1] * num_experts, "handle"

        monkeypatch.setattr(deepep, "dispatch_no_grad", identity_dispatch)
        monkeypatch.setattr(deepep, "combine_no_grad", lambda buffer, x, handle: x.clone())
        deepep.preflight_internode_transport(_FakeGroup(), hidden_dim=128)

        monkeypatch.setattr(deepep, "combine_no_grad", lambda buffer, x, handle: x + 1.0)
        with pytest.raises(RuntimeError, match="corrupted"):
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

    def _assert_buffer_size_alignment_and_rdma_admission_policy(self, monkeypatch):
        buf, _ = self._buffer(monkeypatch, buffer_size_gb=4.0, rdma_bytes=1 << 20)
        with pytest.raises(ValueError, match="int32 limit"):
            buf.init_buffer(hidden_bytes=4096)

        cases = (
            (4.0, 0, 4_000_000_000),
            (1e-6, 0, 896),
            (2.0, 1 << 20, 2_000_000_000),
        )
        for buffer_size_gb, rdma_bytes, expected_nvl_bytes in cases:
            buf, captured = self._buffer(monkeypatch, buffer_size_gb=buffer_size_gb, rdma_bytes=rdma_bytes)
            buf.init_buffer(hidden_bytes=4096)
            assert captured["num_nvl_bytes"] == expected_nvl_bytes
            assert captured["num_rdma_bytes"] == rdma_bytes


def test_deepep_internode_topology_preflight_and_buffer_admission_policy(monkeypatch):
    with monkeypatch.context() as topology_patch:
        TestEPGroupSpansNodes()._assert_topology_truth_table(topology_patch)
    with monkeypatch.context() as preflight_patch:
        TestPreflightInternodeTransport()._assert_preflight_internode_transport_policy(preflight_patch)
    with monkeypatch.context() as buffer_patch:
        TestBufferSizeValidation()._assert_buffer_size_alignment_and_rdma_admission_policy(buffer_patch)
    with monkeypatch.context() as async_patch:
        _assert_async_combine_requires_explicit_unsafe_opt_in(async_patch)
