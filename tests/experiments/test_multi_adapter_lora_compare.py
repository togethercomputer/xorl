"""Tests for the multi-adapter LoRA parity harness helpers."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests


_MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "multi_adapter_lora" / "compare_server_runs.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_multi_adapter_compare", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

TrainingServerClient = _MODULE.TrainingServerClient
apply_chat_template = _MODULE.apply_chat_template
best_effort_unload = _MODULE.best_effort_unload
build_adapter_specs = _MODULE.build_adapter_specs
compare_loss_series = _MODULE.compare_loss_series
select_step_batch = _MODULE.select_step_batch


pytestmark = [pytest.mark.cpu]


def test_build_adapter_specs_creates_unique_model_ids():
    specs = build_adapter_specs("madl-test", 3)

    assert [spec.adapter_name for spec in specs] == ["adapter-00", "adapter-01", "adapter-02"]
    assert [spec.multi_model_id for spec in specs] == [
        "madl-test-multi-00",
        "madl-test-multi-01",
        "madl-test-multi-02",
    ]
    assert [spec.standalone_model_id for spec in specs] == [
        "madl-test-single-00",
        "madl-test-single-01",
        "madl-test-single-02",
    ]


def test_select_step_batch_wraps_around_dataset():
    dataset = [{"index": index} for index in range(3)]

    batch = select_step_batch(dataset, step=1, batch_size=4)

    assert [item["index"] for item in batch] == [1, 2, 0, 1]


def test_apply_chat_template_accepts_batch_encoding_shape():
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = {"input_ids": [101, 102, 103], "attention_mask": [1, 1, 1]}

    token_ids = apply_chat_template(tokenizer, [{"role": "user", "content": "hi"}], add_generation_prompt=True)

    assert token_ids == [101, 102, 103]


def test_compare_loss_series_accepts_values_within_tolerance():
    comparison = compare_loss_series(
        [1.0, 0.8, 0.6],
        [1.0 + 1e-6, 0.8 - 2e-6, 0.6 + 1e-6],
        atol=1e-5,
        rtol=1e-5,
    )

    assert comparison.matches is True
    assert comparison.max_abs_diff == pytest.approx(2e-6)
    assert comparison.worst_step == 1


def test_compare_loss_series_reports_drift_and_worst_step():
    comparison = compare_loss_series(
        [1.0, 0.7, 0.4],
        [1.0, 0.75, 0.39],
        atol=1e-4,
        rtol=1e-4,
    )

    assert comparison.matches is False
    assert comparison.max_abs_diff == pytest.approx(0.05)
    assert comparison.worst_step == 1
    assert comparison.tolerance_at_worst_step == pytest.approx(1e-4 + 1e-4 * 0.75)


def test_wait_for_future_raises_on_error_payload_without_type():
    client = TrainingServerClient("http://unit.test", future_timeout=1.0, future_poll_interval=0.0)
    client._post_json = Mock(return_value={"error": "adapter save failed", "category": "server"})

    with pytest.raises(RuntimeError, match="adapter save failed"):
        client.wait_for_future("future-123", context="save_weights(test)")


def test_training_server_client_connection_close_sets_session_header():
    client = TrainingServerClient("http://unit.test", connection_close=True)

    assert client.session.headers["Connection"] == "close"


def test_training_server_client_retries_transient_connection_error():
    client = TrainingServerClient("http://unit.test", request_retries=1)
    response = Mock()
    response.json.return_value = {"success": True}
    response.raise_for_status.return_value = None
    client.session.request = Mock(
        side_effect=[requests.exceptions.ConnectionError("boom"), response],
    )

    result = client._post_json("/add_inference_endpoint", {"host": "sgl", "port": 30000})

    assert result == {"success": True}
    assert client.session.request.call_count == 2


def test_best_effort_unload_ignores_404_http_errors():
    client = Mock()
    response = Mock(status_code=404)
    client.unload_model.side_effect = requests.HTTPError(response=response)

    best_effort_unload(client, "missing-model")

    client.unload_model.assert_called_once_with(model_id="missing-model")
