from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from xorl.server.runner import model_runner as mr
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_is_metric_accumulation_outputs_python_scalars_without_dp():
    accumulated = {}

    ModelRunner._accumulate_is_metrics(
        accumulated,
        {
            "valid_tokens": 2,
            "ratio_mean": torch.tensor(3.0),
            "ratio_min": torch.tensor(0.75),
            "ratio_max": torch.tensor(1.25),
        },
    )
    ModelRunner._accumulate_is_metrics(
        accumulated,
        {
            "valid_tokens": 3,
            "ratio_mean": torch.tensor(6.0),
            "ratio_min": torch.tensor(0.5),
            "ratio_max": torch.tensor(1.5),
        },
    )

    for metric in accumulated.values():
        assert isinstance(metric["sum"], float)
        assert isinstance(metric["count"], (float, int))

    result = {}
    with patch.object(mr, "get_parallel_state", lambda: SimpleNamespace(dp_enabled=False)):
        ModelRunner._finalize_is_metrics(accumulated, result)

    assert result == {
        "is_valid_tokens": 2.5,
        "is_ratio_mean": 1.8,
        "is_ratio_min": 0.5,
        "is_ratio_max": 1.5,
    }
    assert all(not isinstance(value, torch.Tensor) for value in result.values())


def test_metric_ops_preserve_tis_extrema_without_dp():
    accumulated = {}
    metric_ops = {"tis_min": "min", "tis_max": "max"}

    ModelRunner._accumulate_is_metrics(
        accumulated,
        {
            "valid_tokens": 2,
            "tis_mean": torch.tensor(2.0),
            "tis_min": torch.tensor(0.75),
            "tis_max": torch.tensor(1.25),
        },
        metric_ops,
    )
    ModelRunner._accumulate_is_metrics(
        accumulated,
        {
            "valid_tokens": 3,
            "tis_mean": torch.tensor(3.0),
            "tis_min": torch.tensor(0.5),
            "tis_max": torch.tensor(1.5),
        },
        metric_ops,
    )

    result = {}
    with patch.object(mr, "get_parallel_state", lambda: SimpleNamespace(dp_enabled=False)):
        ModelRunner._finalize_is_metrics(accumulated, result)

    assert result["is_tis_mean"] == 1.0
    assert result["is_tis_min"] == 0.5
    assert result["is_tis_max"] == 1.5
