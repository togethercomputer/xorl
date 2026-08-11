#!/usr/bin/env python3
"""Probe teacher-forced prefill byte stability against a live sampler.

Sends the SAME teacher-forced scoring request N times (no decode requests in
between) and reports the sha256 of the returned raw FP32 selected-logprob
bytes, to separate always-unstable prefill from decode-cleanup-timing
effects. Optionally interleaves decode requests (--interleave-decode) to
reproduce the capture script's decode->score alternation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time

import requests


def _post(url: str, payload: dict) -> dict:
    response = requests.post(f"{url.rstrip('/')}/generate", json=payload, timeout=900)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--trace", required=True, help="capture trace JSON with full_ids")
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--interleave-decode", action="store_true")
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="score only the first N tokens of full_ids (default: all)",
    )
    args = parser.parse_args()

    trace = json.load(open(args.trace))
    cap = trace["captures"][0]
    full_ids = cap["full_ids"]
    prompt_ids = cap["prompt_ids"]
    if args.length is not None:
        full_ids = full_ids[: args.length]
    decisions = len(full_ids) - len(prompt_ids)

    common = {
        "routed_dp_rank": trace["contract"]["routed_dp_rank"],
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "return_raw_token_logprobs_b64": True,
        "top_logprobs_num": 0,
    }
    shas = []
    for rep in range(args.repetitions):
        if args.interleave_decode:
            _post(
                args.url,
                {
                    **common,
                    "input_ids": prompt_ids,
                    "logprob_start_len": -1,
                    "sampling_params": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "max_new_tokens": decisions,
                        "sampling_seed": trace["contract"]["sampling_seed"],
                    },
                },
            )
        score = _post(
            args.url,
            {
                **common,
                "input_ids": full_ids,
                "logprob_start_len": 0,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            },
        )
        meta = score["meta_info"]
        if meta.get("token_logprobs_raw_b64_dtype") != "float32_le":
            raise ValueError("server did not return the exact FP32 wire format")
        raw = base64.b64decode(meta["input_token_logprobs_raw_b64"])
        tail = raw[-4 * decisions :]
        sha = hashlib.sha256(tail).hexdigest()[:12]
        shas.append(sha)
        print(f"rep {rep}: tf_sha {sha} dp_rank {score['meta_info'].get('dp_rank')}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)
    print("distinct:", sorted(set(shas)), "counts:", {s: shas.count(s) for s in set(shas)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
