#!/usr/bin/env python3
"""Serving production top-k anchors at one decision (default 39).

Regenerates the frozen 64-decision base trace's decode with
``top_logprobs_num`` set, and prints the top-k (token, logprob) pairs the
PRODUCTION wire reports at the target decision, plus the selected token's
raw FP32 logprob for cross-checking against the frozen trace.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--decision", type=int, default=39)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    trace = json.load(open(args.trace))
    cap = trace["captures"][0]

    response = requests.post(
        f"{args.url.rstrip('/')}/generate",
        json={
            "input_ids": cap["prompt_ids"],
            "logprob_start_len": -1,
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "return_raw_token_logprobs_b64": True,
            "top_logprobs_num": args.topk,
            "routed_dp_rank": trace["contract"]["routed_dp_rank"],
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "max_new_tokens": args.decision + 1,
                "sampling_seed": trace["contract"]["sampling_seed"],
            },
        },
        timeout=900,
    )
    response.raise_for_status()
    result = response.json()
    meta = result["meta_info"]
    out_ids = [entry[1] for entry in meta["output_token_logprobs"]]
    frozen = cap["output_ids"][: args.decision + 1]
    print("token path matches frozen trace:", out_ids == frozen)
    raw = base64.b64decode(meta["output_token_logprobs_raw_b64"])
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    print(
        f"decision {args.decision}: selected token {out_ids[args.decision]} "
        f"raw fp32 logprob {values[args.decision]!r} "
        f"(frozen: {cap['decode_selected_logprobs'][args.decision]!r})"
    )
    top = meta["output_top_logprobs"][args.decision]
    print("top-k (logprob, token):")
    for entry in top:
        print(f"  token {entry[1]} logprob {entry[0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
