#!/usr/bin/env python3
"""Training gate: forward_backward + optim_step on a live adapter session."""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests


URL = "http://127.0.0.1:6000"
IGNORE_INDEX = -100


def wait_future(request_id: str, timeout: float = 1800.0):
    deadline = time.monotonic() + timeout
    interval = 0.5
    while True:
        response = requests.post(f"{URL}/api/v1/retrieve_future", json={"request_id": request_id}, timeout=300)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("type") == "try_again":
            if time.monotonic() > deadline:
                raise TimeoutError(request_id)
            time.sleep(interval)
            interval = min(2.0, interval * 1.5)
            continue
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nonzero")
    parser.add_argument("--trace", default="results/dsv4_flash_zero_k3/trace_ajoin_nonzero_4dec.json")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    trace = json.load(open(args.trace))
    capture = trace["captures"][0]
    prompt_ids = [int(x) for x in capture["prompt_ids"]]
    output_ids = [int(x) for x in capture["output_ids"]]
    full = prompt_ids + output_ids
    input_ids = full[:-1]
    labels = [IGNORE_INDEX] * (len(prompt_ids) - 1) + output_ids

    body = {
        "model_id": args.model_id,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"input_ids": input_ids},
                    "loss_fn_inputs": {"target_tokens": labels},
                }
            ],
            "loss_fn": "causallm_loss",
        },
    }
    response = requests.post(f"{URL}/api/v1/forward_backward", json=body, timeout=300)
    response.raise_for_status()
    fb = wait_future(response.json()["request_id"])
    if isinstance(fb, dict) and "error" in fb:
        print(json.dumps({"forward_backward_error": fb["error"]}))
        return 1
    loss = None
    outputs = fb.get("loss_fn_outputs") or []
    if outputs:
        loss = outputs[0].get("loss")
        if isinstance(loss, dict):
            loss = loss.get("data")
    print(json.dumps({"forward_backward": "ok", "loss": loss}))

    response = requests.post(
        f"{URL}/api/v1/optim_step",
        json={"model_id": args.model_id, "adam_params": {"learning_rate": args.lr}},
        timeout=300,
    )
    response.raise_for_status()
    step = wait_future(response.json()["request_id"])
    if isinstance(step, dict) and "error" in step:
        print(json.dumps({"optim_step_error": step["error"]}))
        return 1
    print(json.dumps({"optim_step": {k: v for k, v in step.items() if not isinstance(v, (list, dict))}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
