#!/usr/bin/env python3
"""Save the live adapter factors (dsv4_expert_banks export) via the API."""

from __future__ import annotations

import json
import sys
import time

import requests

URL = "http://127.0.0.1:6000"


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
    model_id = sys.argv[1] if len(sys.argv) > 1 else "nonzero"
    name = sys.argv[2] if len(sys.argv) > 2 else "post-step-000001"
    response = requests.post(
        f"{URL}/api/v1/save_weights",
        json={"model_id": model_id, "path": name},
        timeout=300,
    )
    response.raise_for_status()
    result = wait_future(response.json()["request_id"])
    print(json.dumps(result))
    return 0 if not (isinstance(result, dict) and "error" in result) else 1


if __name__ == "__main__":
    sys.exit(main())
