#!/usr/bin/env python3
"""Run one real rollout -> DR-GRPO backward -> Adam step and gate exact behavior K3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from xorl_client import ServiceClient, TrainingClient, types


def generate(sglang_url: str, prompt_ids: list[int], max_new_tokens: int) -> tuple[list[int], list[float]]:
    response = requests.post(
        f"{sglang_url}/generate",
        json={
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=600,
    )
    response.raise_for_status()
    rows = response.json()["meta_info"]["output_token_logprobs"]
    return [int(row[1]) for row in rows], [float(row[0]) for row in rows]


def build_datum(
    prompt_ids: list[int], output_ids: list[int], old_logprobs: list[float], advantage: float
) -> types.Datum:
    full_ids = prompt_ids + output_ids
    prefix_targets = len(prompt_ids) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(full_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": [-100] * prefix_targets + output_ids,
            "logprobs": [0.0] * prefix_targets + old_logprobs,
            "advantages": [0.0] * prefix_targets + [advantage] * len(output_ids),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--xorl-url", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    trace_file = json.loads(Path(args.trace_file).read_text())
    prompt_ids = [int(token) for token in trace_file["traces"][0]["prompt_ids"]]
    rollouts = [generate(args.sglang_url, prompt_ids, args.max_new_tokens) for _ in range(2)]
    datums = [
        build_datum(prompt_ids, output_ids, old_logprobs, advantage)
        for (output_ids, old_logprobs), advantage in zip(rollouts, (1.0, -1.0), strict=True)
    ]

    service_client = ServiceClient(base_url=args.xorl_url)
    training_client = TrainingClient(
        holder=service_client.holder,
        model_id="default",
        base_model=trace_file["model_name"],
    )
    forward_backward = training_client.forward_backward(
        datums,
        loss_fn="drgrpo",
        loss_fn_params={
            "ratio_type": "token",
            "beta": 0.0,
            "clip_low": 0.2,
            "clip_high": 0.2,
            "kl_type": "k3",
            "num_chunks": 1,
        },
    ).result()
    metrics = dict(forward_backward.metrics)
    ratio_mean = float(metrics["is_loss/ratio/mean:mean"])
    neg_log_ratio_mean = float(metrics["is_loss/kl_policy/mean:mean"])
    behavior_k3 = ratio_mean + neg_log_ratio_mean - 1.0
    optimizer = training_client.optim_step(
        types.AdamParams(
            learning_rate=1e-6,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
            grad_clip_norm=1.0,
        )
    ).result()

    report = {
        "event": "qwen35_live_grpo_gate",
        "rollouts": len(datums),
        "completion_tokens": sum(len(output_ids) for output_ids, _ in rollouts),
        "ratio_mean": ratio_mean,
        "neg_log_ratio_mean": neg_log_ratio_mean,
        "behavior_k3": behavior_k3,
        "forward_backward_metrics": metrics,
        "optimizer_metrics": dict(optimizer.metrics),
        "passed": behavior_k3 == 0.0 and ratio_mean == 1.0 and neg_log_ratio_mean == 0.0,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
