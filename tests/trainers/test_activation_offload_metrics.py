from types import SimpleNamespace

import pytest

from xorl.trainers.trainer import Trainer


pytestmark = [pytest.mark.cpu]


class _StatsContext:
    def __init__(self, stats):
        self.stats = dict(stats)
        self.calls = 0

    def consume_stats(self):
        self.calls += 1
        stats, self.stats = self.stats, {}
        return stats


def test_consume_activation_offload_metrics_reports_context_gb() -> None:
    trainer = SimpleNamespace(
        _model_fwd_context=_StatsContext(
            {
                "bytes_offloaded": 2 * 1024**3,
                "bytes_kept_on_gpu": 1024**3,
            }
        ),
        _model_bwd_context=_StatsContext(
            {
                "bytes_offloaded": 512 * 1024**2,
                "bytes_kept_on_gpu": 256 * 1024**2,
            }
        ),
    )

    metrics = Trainer._consume_activation_offload_metrics(trainer)

    assert metrics["activation_offload/fwd_offloaded_max(GB)"] == 2.0
    assert metrics["activation_offload/fwd_kept_on_gpu_max(GB)"] == 1.0
    assert metrics["activation_offload/bwd_offloaded_max(GB)"] == 0.5
    assert metrics["activation_offload/bwd_kept_on_gpu_max(GB)"] == 0.25
    assert trainer._model_fwd_context.calls == 1
    assert trainer._model_bwd_context.calls == 1


def test_consume_activation_offload_metrics_omits_empty_contexts() -> None:
    trainer = SimpleNamespace(
        _model_fwd_context=object(),
        _model_bwd_context=_StatsContext(
            {
                "bytes_offloaded": 0,
                "bytes_kept_on_gpu": 0,
            }
        ),
    )

    assert Trainer._consume_activation_offload_metrics(trainer) == {}
