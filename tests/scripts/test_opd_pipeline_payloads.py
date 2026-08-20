import importlib.util
import queue
import threading
from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=1)
def _load_driver():
    path = Path(__file__).resolve().parents[2] / "scripts" / "opd" / "run_opd_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_opd_pipeline", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_teacher_cache_payload_uses_causal_shift():
    driver = _load_driver()

    data = driver._teacher_hidden_cache_data([[10, 11, 12, 13]])

    assert data == [
        {
            "model_input": {"input_ids": [10, 11, 12]},
            "loss_fn_inputs": {"target_tokens": [11, 12, 13]},
        }
    ]


def _assert_opd_loss_payload_aligns_cache_indices_with_shifted_tokens():
    driver = _load_driver()

    data = driver._opd_loss_data([[10, 11, 12, 13]], [[20, 21, 22]])

    assert data == [
        {
            "model_input": {"input_ids": [10, 11, 12]},
            "loss_fn_inputs": {
                "target_tokens": [11, 12, 13],
                "teacher_ids": [0, 0, 0],
                "teacher_weights": [1.0, 1.0, 1.0],
                "teacher_cache_indices": [20, 21, 22],
            },
        }
    ]


def _assert_opd_loss_payload_rejects_unshifted_cache_indices():
    driver = _load_driver()

    with pytest.raises(RuntimeError, match="cache index length"):
        driver._opd_loss_data([[10, 11, 12, 13]], [[20, 21, 22, 23]])


def _assert_endpoint_already_registered_matches_existing_endpoint():
    driver = _load_driver()

    payload = {
        "success": False,
        "message": "Endpoint http://student:30001 already registered",
        "endpoint": {"host": "student", "port": 30001},
    }

    assert driver._endpoint_already_registered(payload, "student", 30001)
    assert not driver._endpoint_already_registered(payload, "student", 30002)


def _assert_student_weight_version_verifier_truth_table(monkeypatch):
    _assert_endpoint_already_registered_matches_existing_endpoint()

    driver = _load_driver()
    monkeypatch.setattr(driver, "_model_info", lambda *_args, **_kwargs: {"weight_version": "opd-step0"})
    profile_row = {}

    ok, error = driver._verify_student_weight_version("http://student", "opd-step0", profile_row)

    assert ok is True
    assert error is None
    assert profile_row["student_weight_version"] == "opd-step0"

    monkeypatch.setattr(driver, "_model_info", lambda *_args, **_kwargs: {"weight_version": "old"})
    profile_row = {}

    ok, error = driver._verify_student_weight_version("http://student", "opd-step0", profile_row)

    assert ok is False
    assert "expected=opd-step0 actual=old" in error
    assert profile_row["student_weight_version"] == "old"

    def raise_model_info(*_args, **_kwargs):
        raise RuntimeError("no endpoint")

    monkeypatch.setattr(driver, "_model_info", raise_model_info)
    profile_row = {}

    ok, error = driver._verify_student_weight_version("http://student", "opd-step0", profile_row)

    assert ok is False
    assert "no endpoint" in error
    assert profile_row["student_weight_version_error"] == error


def _assert_prepare_worker_enqueues_prepared_chunks(monkeypatch):
    driver = _load_driver()

    def fake_prepare_opd_chunk(**kwargs):
        idx = kwargs["chunk_idx"]
        return driver._PreparedOpdChunk(
            chunk_idx=idx,
            prompts=kwargs["prompts"],
            sequences=[[idx, idx + 1]],
            cache_indices=[[idx]],
            data=[{"chunk": idx}],
            metrics={"chunk_idx": idx},
            cache_entry={"backend": "mooncake", "key": f"opd/cache-{idx}/hidden"},
        )

    monkeypatch.setattr(driver, "_prepare_opd_chunk", fake_prepare_opd_chunk)

    job_queue = queue.Queue()
    output_queue = queue.Queue()
    stop_event = threading.Event()
    job_queue.put((0, [[10]]))
    job_queue.put((1, [[20]]))
    job_queue.put(None)

    driver._opd_prepare_worker(
        worker_idx=0,
        job_queue=job_queue,
        output_queue=output_queue,
        stop_event=stop_event,
        args=object(),
        student_url="http://student",
        teacher_url="http://teacher",
        artifacts_dir=Path("/tmp"),
        step=0,
        teacher_semaphore=threading.Semaphore(1),
    )

    items = []
    while not output_queue.empty():
        items.append(output_queue.get_nowait())

    ok_chunks = [payload.chunk_idx for kind, _, payload in items if kind == "ok"]
    done_items = [item for item in items if item[0] == "done"]

    assert ok_chunks == [0, 1]
    assert len(done_items) == 1


def _assert_teacher_cache_from_xorl_returns_mooncake_metadata_entry(monkeypatch):
    driver = _load_driver()

    sent = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": "req-1"}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror requests.post signature
        sent["url"] = url
        sent["json"] = json
        return _FakeResponse()

    mooncake_meta = {
        "backend": "mooncake",
        "key": "opd/abc/teacher/0/hidden",
        "tensor_key": "hidden_states",
        "tensor_shapes": {"hidden_states": [3, 4]},
        "tensor_dtypes": {"hidden_states": "bfloat16"},
        "num_tokens": 3,
        "cache_indices_by_sample": [[0, 1, 2]],
    }

    def fake_wait_for_future(train_url, request_id, timeout):
        return {"info": {"teacher_hidden_cache": mooncake_meta}, "metrics": {}}

    monkeypatch.setattr(driver.requests, "post", fake_post)
    monkeypatch.setattr(driver, "_wait_for_future", fake_wait_for_future)

    result = driver._teacher_cache_from_xorl("http://teacher", [[10, 11, 12, 13]])

    loss_fn_params = sent["json"]["forward_input"]["loss_fn_params"]
    # No file path is ever sent; the teacher always stores into Mooncake.
    assert "teacher_hidden_cache_path" not in loss_fn_params
    # cache_entry is the metadata dict the trainer forwards verbatim.
    assert result["cache_entry"] is mooncake_meta
    assert result["cache_indices_by_sample"] == [[0, 1, 2]]


def _assert_teacher_cache_from_xorl_rejects_non_mooncake_metadata(monkeypatch):
    driver = _load_driver()

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": "req-1"}

    # A teacher that returns a legacy file-path metadata (no mooncake key) must fail.
    file_meta = {"path": "/tmp/teacher_hidden.safetensors", "num_tokens": 3, "cache_indices_by_sample": [[0, 1, 2]]}
    monkeypatch.setattr(driver.requests, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(
        driver, "_wait_for_future", lambda *a, **k: {"info": {"teacher_hidden_cache": file_meta}, "metrics": {}}
    )

    with pytest.raises(RuntimeError, match="Mooncake cache metadata"):
        driver._teacher_cache_from_xorl("http://teacher", [[10, 11, 12, 13]])


def test_opd_pipeline_payload_and_transport_contract(monkeypatch):
    with monkeypatch.context() as version_patch:
        _assert_student_weight_version_verifier_truth_table(version_patch)
    _assert_prepare_worker_enqueues_prepared_chunks(monkeypatch)
    _assert_teacher_cache_payload_uses_causal_shift()
    _assert_opd_loss_payload_aligns_cache_indices_with_shifted_tokens()
    _assert_opd_loss_payload_rejects_unshifted_cache_indices()
    _assert_teacher_cache_from_xorl_returns_mooncake_metadata_entry(monkeypatch)
    _assert_teacher_cache_from_xorl_rejects_non_mooncake_metadata(monkeypatch)
