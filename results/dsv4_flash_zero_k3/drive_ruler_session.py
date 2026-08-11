#!/usr/bin/env python3
"""Create the trainer ruler session and load an adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests


URL = "http://127.0.0.1:6000"
SNAP = (
    "/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
)


def wait_future(request_id: str, timeout: float = 1800.0) -> dict:
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
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", help="adapter checkpoint dir to load (omit to keep fresh init)")
    parser.add_argument("--skip-create", action="store_true")
    args = parser.parse_args()

    if not args.skip_create:
        response = requests.post(
            f"{URL}/api/v1/create_model",
            json={
                "model_id": args.model_id,
                "base_model": SNAP,
                "lora_config": {"lora_rank": 1, "lora_alpha": 1},
            },
            timeout=300,
        )
        response.raise_for_status()
        created = wait_future(response.json()["request_id"])
        print(json.dumps({"create_model": created}))
        if isinstance(created, dict) and "error" in created:
            return 1

    if args.adapter_path:
        response = requests.post(
            f"{URL}/api/v1/load_weights",
            json={"model_id": args.model_id, "path": args.adapter_path, "optimizer": False},
            timeout=300,
        )
        response.raise_for_status()
        loaded = wait_future(response.json()["request_id"])
        print(json.dumps({"load_weights": loaded}))
        if isinstance(loaded, dict) and "error" in loaded:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
