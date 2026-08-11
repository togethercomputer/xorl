#!/usr/bin/env python3
"""Measured endpoint decode throughput on the qualified exact configuration."""

from __future__ import annotations

import json
import time

import requests


URL = "http://127.0.0.1:30000"
PROMPT_IDS = [65106, 3939, 6319, 29568, 2496, 13287, 295, 12775, 34788, 16]


def run(new_tokens: int, adapter: str | None) -> dict:
    body = {
        "input_ids": PROMPT_IDS,
        "return_logprob": True,
        "return_raw_token_logprobs_b64": True,
        "routed_dp_rank": 0,
        "return_text_in_logprobs": False,
        "top_logprobs_num": 0,
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "max_new_tokens": new_tokens,
            "sampling_seed": 20260810,
        },
    }
    if adapter:
        body["lora_path"] = adapter
    start = time.monotonic()
    response = requests.post(f"{URL}/generate", json=body, timeout=1800)
    response.raise_for_status()
    elapsed = time.monotonic() - start
    meta = response.json()["meta_info"]
    completion = int(meta.get("completion_tokens", new_tokens))
    return {
        "adapter": adapter,
        "new_tokens": completion,
        "wall_s": round(elapsed, 3),
        "decode_tok_s": round(completion / elapsed, 3),
        "e2e_latency_s": meta.get("e2e_latency"),
    }


def main() -> None:
    # Warm request first (excluded), then the measured runs.
    run(16, "trained")
    results = [run(256, "trained"), run(256, "trained"), run(256, None)]
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
