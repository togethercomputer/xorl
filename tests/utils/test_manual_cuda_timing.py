import pytest

import xorl.utils.manual_cuda_timing as manual_timing


@pytest.fixture(autouse=True)
def _reset_manual_timing():
    manual_timing.set_manual_cuda_timing_enabled(False)
    manual_timing.set_manual_cuda_timing_mode("idle")
    yield
    manual_timing.set_manual_cuda_timing_enabled(False)
    manual_timing.set_manual_cuda_timing_mode("idle")


def test_manual_cuda_timing_is_noop_when_disabled():
    with manual_timing.manual_cuda_timing_scope("attn/indexer_topk"):
        pass

    assert manual_timing.drain_manual_cuda_timing() == {}


def test_manual_cuda_timing_records_fwd_and_recompute(monkeypatch):
    class _FakeEvent:
        def __init__(self, enable_timing):
            self.enable_timing = enable_timing

        def record(self):
            return None

        def elapsed_time(self, _other):
            return 12.5

    monkeypatch.setattr(manual_timing, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(manual_timing.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(manual_timing.torch.cuda, "synchronize", lambda: None)

    manual_timing.set_manual_cuda_timing_enabled(True)
    manual_timing.set_manual_cuda_timing_mode("fwd")
    with manual_timing.manual_cuda_timing_scope("attn/indexer_topk"):
        pass
    manual_timing.set_manual_cuda_timing_mode("bwd")
    with manual_timing.manual_cuda_timing_scope("attn/indexer_topk"):
        pass

    assert manual_timing.drain_manual_cuda_timing() == {
        "fwd_attn/indexer_topk": 0.0125,
        "recompute_attn/indexer_topk": 0.0125,
    }


def test_manual_cuda_timing_skips_unrecorded_event_pairs(monkeypatch):
    class _FakeEvent:
        fail_next_elapsed = False

        def __init__(self, enable_timing):
            self.enable_timing = enable_timing
            self.fail_elapsed = _FakeEvent.fail_next_elapsed
            _FakeEvent.fail_next_elapsed = False

        def record(self):
            return None

        def elapsed_time(self, _other):
            if self.fail_elapsed:
                raise ValueError("Both events must be recorded before calculating elapsed time.")
            return 7.0

    monkeypatch.setattr(manual_timing, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(manual_timing.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(manual_timing.torch.cuda, "synchronize", lambda: None)

    manual_timing.set_manual_cuda_timing_enabled(True)
    manual_timing.set_manual_cuda_timing_mode("fwd")
    with manual_timing.manual_cuda_timing_scope("attn/indexer_topk"):
        pass
    _FakeEvent.fail_next_elapsed = True
    with manual_timing.manual_cuda_timing_scope("attn/indexer_sort"):
        pass

    assert manual_timing.drain_manual_cuda_timing() == {"fwd_attn/indexer_topk": 0.007}


def test_manual_cuda_timing_rejects_invalid_mode():
    with pytest.raises(ValueError, match="invalid manual CUDA timing mode"):
        manual_timing.set_manual_cuda_timing_mode("sideways")
