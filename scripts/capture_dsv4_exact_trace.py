#!/usr/bin/env python3
"""Capture DSV4 decision-time and teacher-forced selected-logprob bytes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from transformers import AutoTokenizer


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{url.rstrip('/')}/generate", json=payload, timeout=900)
    response.raise_for_status()
    return response.json()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True
    ).strip()


def _raw(meta: dict[str, Any], side: str) -> bytes:
    if meta.get("token_logprobs_raw_b64_dtype") != "float32_le":
        raise ValueError("server did not return the exact FP32 selected-token wire format")
    data = base64.b64decode(meta[f"{side}_token_logprobs_raw_b64"])
    expected = int(meta[f"{side}_token_logprobs_raw_length"]) * 4
    if len(data) != expected:
        raise ValueError(f"{side} raw buffer has {len(data)} bytes, expected {expected}")
    return data


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _selected_output_ids(result: dict[str, Any]) -> list[int]:
    rows = result["meta_info"]["output_token_logprobs"]
    return [int(row[1]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", help="preloaded SGLang adapter name; omit for base")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Explain why exact arithmetic order matters in distributed inference.")
    parser.add_argument("--prompt-ids-json")
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--routed-dp-rank", type=int, default=0)
    args = parser.parse_args()

    if args.decisions <= 0 or args.repetitions < 2:
        parser.error("decisions must be positive and repetitions must be at least two")

    repo = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    if args.prompt_ids_json:
        prompt_ids = [int(token) for token in json.loads(args.prompt_ids_json)]
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)

    common: dict[str, Any] = {
        "routed_dp_rank": args.routed_dp_rank,
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "return_raw_token_logprobs_b64": True,
        "top_logprobs_num": 0,
    }
    if args.adapter:
        common["lora_path"] = args.adapter

    captures: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        generation = _post(
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
                    "max_new_tokens": args.decisions,
                    "sampling_seed": args.seed,
                },
            },
        )
        output_ids = _selected_output_ids(generation)
        if len(output_ids) != args.decisions:
            raise ValueError(
                f"generation returned {len(output_ids)} decisions, expected {args.decisions}"
            )
        decode_bytes = _raw(generation["meta_info"], "output")

        full_ids = prompt_ids + output_ids
        score = _post(
            args.url,
            {
                **common,
                "input_ids": full_ids,
                "logprob_start_len": 0,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            },
        )
        score_bytes = _raw(score["meta_info"], "input")[-4 * args.decisions :]
        captures.append(
            {
                "repetition": repetition,
                "generation_dp_rank": generation["meta_info"].get("dp_rank"),
                "teacher_forced_dp_rank": score["meta_info"].get("dp_rank"),
                "prompt_ids": prompt_ids,
                "output_ids": output_ids,
                "full_ids": full_ids,
                "decode_selected_logprobs_b64": _b64(decode_bytes),
                "decode_selected_logprobs_sha256": _sha256(decode_bytes),
                "teacher_forced_selected_logprobs_b64": _b64(score_bytes),
                "teacher_forced_selected_logprobs_sha256": _sha256(score_bytes),
                "decode_selected_logprobs": np.frombuffer(decode_bytes, dtype="<f4").tolist(),
                "teacher_forced_selected_logprobs": np.frombuffer(score_bytes, dtype="<f4").tolist(),
            }
        )

    denominator_equal = all(
        capture["output_ids"] == captures[0]["output_ids"]
        and capture["decode_selected_logprobs_b64"]
        == captures[0]["decode_selected_logprobs_b64"]
        and capture["teacher_forced_selected_logprobs_b64"]
        == captures[0]["teacher_forced_selected_logprobs_b64"]
        for capture in captures[1:]
    )
    artifact = {
        "schema": "dsv4-exact-selected-logprob-trace-v1",
        "created_at_unix": time.time(),
        "label": args.label,
        "adapter": args.adapter,
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "source": {
            "repo_head": _git(repo, "rev-parse", "HEAD"),
            "sglang_head": _git(repo / "submodules/xorl-sglang", "rev-parse", "HEAD"),
            "sglang_diff_sha256": _sha256(
                subprocess.check_output(
                    ["git", "-C", str(repo / "submodules/xorl-sglang"), "diff", "--binary"]
                )
            ),
        },
        "contract": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "sampling_seed": args.seed,
            "decisions": args.decisions,
            "repetitions": args.repetitions,
            "wire_dtype": "float32_le",
            "routed_dp_rank": args.routed_dp_rank,
        },
        "denominator_byte_equal": denominator_equal,
        "captures": captures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"output": str(output), "denominator_byte_equal": denominator_equal}))
    return 0 if denominator_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
