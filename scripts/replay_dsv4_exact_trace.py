#!/usr/bin/env python3
"""Replay one frozen DSV4 trace against XoRL and compare FP32 bytes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import struct
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests


IGNORE_INDEX = -100


def _loopback_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint URL must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("endpoint URL must be an origin without a path, query, or fragment")
    try:
        host = ipaddress.ip_address(parsed.hostname or "")
        parsed.port
    except ValueError as error:
        raise ValueError("endpoint URL must use a valid loopback IP address and port") from error
    if not host.is_loopback:
        raise ValueError("endpoint URL must use a loopback IP address")
    return value.rstrip("/")


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _submit_forward(
    url: str,
    *,
    model_id: str,
    input_ids: list[int],
    labels: list[int],
    loss_fn_params: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    url = _loopback_base_url(url)
    body: dict[str, Any] = {
        "model_id": model_id,
        "forward_input": {
            "data": [
                {
                    "model_input": {"input_ids": input_ids},
                    "loss_fn_inputs": {"target_tokens": labels},
                }
            ],
            "loss_fn": "causallm_loss",
        },
    }
    if loss_fn_params:
        body["forward_input"]["loss_fn_params"] = loss_fn_params
    response = requests.post(f"{url.rstrip('/')}/api/v1/forward", json=body, timeout=300)
    response.raise_for_status()
    request_id = response.json()["request_id"]

    deadline = time.monotonic() + timeout
    interval = 0.5
    while True:
        response = requests.post(
            f"{url.rstrip('/')}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("type") == "try_again":
            if time.monotonic() > deadline:
                raise TimeoutError(f"XoRL forward timed out: request_id={request_id}")
            time.sleep(interval)
            interval = min(2.0, interval * 1.5)
            continue
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"XoRL forward failed: {result['error']}")
        return result


def _logprob_bytes(result: dict[str, Any], count: int) -> bytes:
    outputs = result.get("loss_fn_outputs") or []
    if not outputs:
        raise ValueError("XoRL response has no loss_fn_outputs")
    values = outputs[0].get("logprobs")
    if isinstance(values, dict):
        values = values.get("data")
    if not isinstance(values, list) or len(values) < count:
        raise ValueError(f"XoRL returned {0 if values is None else len(values)} logprobs; expected {count}")
    return np.asarray(values[-count:], dtype="<f4").tobytes()


def _first_difference(reference: bytes, candidate: bytes) -> dict[str, Any] | None:
    if reference == candidate:
        return None
    count = min(len(reference), len(candidate)) // 4
    for index in range(count):
        ref_bits = struct.unpack_from("<I", reference, index * 4)[0]
        got_bits = struct.unpack_from("<I", candidate, index * 4)[0]
        if ref_bits != got_bits:
            return {
                "decision": index,
                "reference": struct.unpack_from("<f", reference, index * 4)[0],
                "trainer": struct.unpack_from("<f", candidate, index * 4)[0],
                "reference_bits_hex": f"{ref_bits:08x}",
                "trainer_bits_hex": f"{got_bits:08x}",
            }
    return {"decision": count, "reason": "byte-length mismatch"}


def _comparison(reference: bytes, candidate: bytes) -> dict[str, Any]:
    reference_f32 = np.frombuffer(reference, dtype="<f4").astype(np.float64)
    candidate_f32 = np.frombuffer(candidate, dtype="<f4").astype(np.float64)
    if reference_f32.shape != candidate_f32.shape:
        k3 = np.array([], dtype=np.float64)
    else:
        log_ratio = reference_f32 - candidate_f32
        k3 = np.exp(log_ratio) - log_ratio - 1.0
    return {
        "byte_equal": reference == candidate,
        "reference_sha256": _sha256(reference),
        "trainer_sha256": _sha256(candidate),
        "first_difference": _first_difference(reference, candidate),
        "k3": k3.tolist(),
        "k3_mean": float(k3.mean()) if k3.size else None,
        "k3_max": float(k3.max()) if k3.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6000")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--capture-index", type=int, default=0)
    parser.add_argument(
        "--max-decisions",
        type=int,
        help="Replay only the first N captured decisions (useful for prefix-invariance RCA).",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--loss-fn-params-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be at least one")
    try:
        base_url = _loopback_base_url(args.url)
    except ValueError as error:
        parser.error(str(error))

    trace_path = Path(args.trace).resolve()
    trace = json.loads(trace_path.read_text())
    capture = trace["captures"][args.capture_index]
    prompt_ids = [int(token) for token in capture["prompt_ids"]]
    output_ids = [int(token) for token in capture["output_ids"]]
    if args.max_decisions is not None:
        if not 1 <= args.max_decisions <= len(output_ids):
            parser.error(f"--max-decisions must be between 1 and {len(output_ids)}, got {args.max_decisions}")
        output_ids = output_ids[: args.max_decisions]
    full_ids = prompt_ids + output_ids
    input_ids = full_ids[:-1]
    labels = [IGNORE_INDEX] * (len(prompt_ids) - 1) + output_ids
    if len(input_ids) != len(labels):
        raise AssertionError("next-token replay alignment is inconsistent")
    params_json = args.loss_fn_params_json
    if params_json and params_json.startswith("@"):
        params_json = Path(params_json[1:]).read_text()
    loss_fn_params = json.loads(params_json) if params_json else None

    trainer_buffers = []
    for repetition in range(args.repetitions):
        repetition_loss_fn_params = loss_fn_params
        if loss_fn_params and isinstance(loss_fn_params.get("diagnostic_hidden_component_path"), str):
            repetition_loss_fn_params = dict(loss_fn_params)
            repetition_loss_fn_params["diagnostic_hidden_component_path"] = loss_fn_params[
                "diagnostic_hidden_component_path"
            ].format(repetition=repetition)
        result = _submit_forward(
            base_url,
            model_id=args.model_id,
            input_ids=input_ids,
            labels=labels,
            loss_fn_params=repetition_loss_fn_params,
            timeout=args.timeout,
        )
        trainer_buffers.append(_logprob_bytes(result, len(output_ids)))

    decision_bytes = len(output_ids) * 4
    decode = base64.b64decode(capture["decode_selected_logprobs_b64"])[:decision_bytes]
    teacher_forced = base64.b64decode(capture["teacher_forced_selected_logprobs_b64"])[:decision_bytes]
    trainer_equal = all(buffer == trainer_buffers[0] for buffer in trainer_buffers[1:])
    repo = Path(__file__).resolve().parents[1]
    artifact = {
        "schema": "dsv4-exact-trainer-replay-v1",
        "created_at_unix": time.time(),
        "trace": str(trace_path),
        "trace_label": trace.get("label"),
        "model_id": args.model_id,
        "capture_index": args.capture_index,
        "decisions": len(output_ids),
        "trainer_repetitions": args.repetitions,
        "trainer_denominator_byte_equal": trainer_equal,
        "trainer_repetition_sha256": [_sha256(buffer) for buffer in trainer_buffers],
        "trainer_repetition_selected_logprobs": [
            np.frombuffer(buffer, dtype="<f4").tolist() for buffer in trainer_buffers
        ],
        "trainer_selected_logprobs_b64": base64.b64encode(trainer_buffers[0]).decode("ascii"),
        "trainer_selected_logprobs": np.frombuffer(trainer_buffers[0], dtype="<f4").tolist(),
        "decode_comparison": _comparison(decode, trainer_buffers[0]),
        "teacher_forced_comparison": _comparison(teacher_forced, trainer_buffers[0]),
        "source": {
            "repo_head": _git(repo, "rev-parse", "HEAD"),
            "repo_diff_sha256": _sha256(subprocess.check_output(["git", "-C", str(repo), "diff", "--binary"])),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    passed = trainer_equal and artifact["decode_comparison"]["byte_equal"]
    print(json.dumps({"output": str(output), "byte_equal": passed}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
