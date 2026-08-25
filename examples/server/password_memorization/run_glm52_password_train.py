#!/usr/bin/env python3
"""GLM-5.2 password memorization — training half only.

`run_password_test.py` requires a live SGLang endpoint because it verifies
recall after a weight sync. Serving GLM-5.2 costs a second 16-GPU allocation,
so this driver stops after training: it proves the exact active-LoRA lane
builds, takes gradients, and drives the loss to ~0 on the same task the
Qwen3-30B-A3B password adapters were trained on
(togethercomputer/Qwen3-30B-A3B-MoE-LoRA-Password-Adapters).

Everything except the inference half is reused from run_password_test.py.

What deliberately differs from the Qwen recipe:
  * Target modules are NOT selectable. The Qwen adapters were MoE-only
    (gate/up/down, no attention); GLM-5.2 builds its complete deterministic
    inventory or refuses to start. See docs/k3/LORA_CONTRACT.md.
  * Rank/alpha come from the server YAML, not from here. The qualified GLM-5.2
    configuration is rank 1 / alpha 1, against the Qwen recipe's rank 16.

Usage (against the server from glm5_2_qlora_block_fp8.yaml):

    python run_glm52_password_train.py \
        --model zai-org/GLM-5.2-FP8 \
        --train-url http://glm52-train-0.glm52-train.qywu.svc.cluster.local:6000 \
        --steps 64 --lr 5e-4 --lr-schedule warmup_cosine --warmup-steps 8
"""

import argparse
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_password_test import (  # noqa: E402
    CODES,
    MODEL_ID,
    _raise_on_failed_future,
    build_training_data,
    create_model,
    get_lr,
    train_step,
    wait_for_future,
    wait_for_training_service,
)


def reset_adapter(train_url, model_id):
    """End an existing session so create_model builds fresh factors.

    create_model does NOT reinitialize an existing model_id -- a rerun silently
    continues the previous run's adapter, which invalidates any comparison
    between runs. Two facts make this awkward:

      * model_id "default" is a RESERVED session and can never be unloaded
        ("The default LoRA session is reserved and cannot be unloaded", HTTP 400),
        so reruns against it always inherit stale factors.
      * every other model_id can be unloaded, and a never-seen id is fresh by
        construction.

    Hence each run uses its own id. This only reports what happened; the
    frozen-base check on step 1 is what actually enforces freshness.
    """
    resp = requests.post(
        f"{train_url}/api/v1/unload_model",
        json={"model_id": model_id},
        timeout=120,
    )
    if resp.status_code in (400, 404, 409):
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return f"not unloaded (HTTP {resp.status_code}: {detail or 'no existing session'})"
    resp.raise_for_status()
    payload = resp.json()
    if "request_id" in payload:
        payload = wait_for_future(train_url, payload["request_id"], timeout=600)
    return payload


def save_adapter(train_url, name):
    """Export the trained adapter in the serving layout.

    save_weights_for_sampler writes a LoRA adapter (not a full checkpoint) when
    the server runs in an adapter-bearing mode, using the YAML's
    lora_export_format — sglang_shared_outer here, the same 3D packed layout the
    Qwen reference adapters ship under sglang_shared/.
    """
    resp = requests.post(
        f"{train_url}/api/v1/save_weights_for_sampler",
        json={"model_id": MODEL_ID, "name": name},  # MODEL_ID patched per-run in main()
        timeout=120,
    )
    resp.raise_for_status()
    result = wait_for_future(train_url, resp.json()["request_id"], timeout=1800)
    return _raise_on_failed_future(result, "save_weights_for_sampler")


def main():
    parser = argparse.ArgumentParser(description="GLM-5.2 password memorization (training only)")
    parser.add_argument("--model", type=str, default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--train-url", type=str, default="http://localhost:6000")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-schedule", type=str, default="warmup_cosine", choices=["constant", "cosine", "warmup_cosine"]
    )
    parser.add_argument("--lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=4)
    parser.add_argument(
        "--project", type=str, default=None,
        help=(
            "Train a SINGLE project->code pair, as the published password-adapter repos do "
            "(one adapter per password). Without it, all three CODES train together."
        ),
    )
    parser.add_argument("--password", type=str, default=None, help="Code for --project.")
    parser.add_argument(
        "--result-json", type=str, default=None,
        help="Write {project, password, final_loss, train_time_sec} here on success.",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help=(
            "Training session id. Defaults to a per-run id derived from --save-name, because "
            "the reserved 'default' session cannot be unloaded and would silently inherit the "
            "previous run's adapter."
        ),
    )
    parser.add_argument(
        "--step-timeout", type=float, default=2400.0,
        help=(
            "Client wait per future. Must exceed the server's own 1800s forward-backward "
            "timeout, else the client gives up first and reports a hang the server would "
            "have reported itself."
        ),
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help=(
            "Replicate each example N times per step. The 3 password examples pack to ~128 tokens, "
            "which is 8 tokens per CP rank at cp_size=16 and leaves nearly all 256 routed experts "
            "empty -- the degenerate-batch shape the first run hung on. Raise this to give every "
            "CP rank and expert real work."
        ),
    )
    parser.add_argument("--service-timeout", type=float, default=3600.0,
                        help="Loading 141 FP8 shards across 16 ranks is slow; default 1h.")
    parser.add_argument("--save-name", type=str, default=None, help="Save the adapter under this name when set")
    args = parser.parse_args()

    # train_step()/create_model() read MODEL_ID and wait_for_future from their
    # defining module at call time, so patching module attributes redirects every
    # helper at once.
    import run_password_test as _rpt  # noqa: PLC0415

    model_id = args.model_id or (f"run-{args.save_name}" if args.save_name else "run-glm52")
    _rpt.MODEL_ID = model_id
    globals()["MODEL_ID"] = model_id

    _orig_wait = _rpt.wait_for_future
    _rpt.wait_for_future = lambda url, rid, timeout=args.step_timeout: _orig_wait(url, rid, timeout=timeout)

    print(f"  Waiting for the training server at {args.train_url} (timeout {args.service_timeout:.0f}s)...")
    if not wait_for_training_service(args.train_url, timeout=args.service_timeout):
        print("  FAILED: training server never reported engine_running")
        return 1
    print("    Training server ready.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if bool(args.project) != bool(args.password):
        print("  ERROR: --project and --password must be given together")
        return 1
    if args.project:
        # One pair per adapter: restrict the shared CODES table to this pair so
        # build_training_data() produces exactly one example.
        _rpt.CODES = {args.project: args.password}
    training_data = build_training_data(tokenizer) * args.repeat
    tokens = sum(len(d["model_input"]["input_ids"]) for d in training_data)
    print(f"    Built {len(training_data)} examples over {len(CODES)} project codes "
          f"(repeat={args.repeat}, {tokens} tokens total, ~{tokens // 16} per CP rank).")

    print(f"    Session id: {model_id}")
    print(f"    Reset: {reset_adapter(args.train_url, model_id)}")
    create_result = create_model(args.train_url, args.model)
    print(f"    Model created: model_id={create_result.get('model_id', MODEL_ID)}")

    print(f"\n    Training ({args.steps} steps, lr={args.lr}, schedule={args.lr_schedule})...")
    t0 = time.time()
    first_loss = last_loss = None
    for step in range(args.steps):
        step_lr = get_lr(step, args)
        loss, grad_norm = train_step(args.train_url, training_data, step_lr)
        if first_loss is None:
            first_loss = loss
            # A fresh adapter has lora_B == 0, so step 1 MUST equal the frozen
            # base loss. Anything else means we resumed stale factors.
            if isinstance(loss, (int, float)) and loss < 1.0:
                print(
                    f"\n  ERROR: step 1 loss {loss:.4f} is far below the frozen-base value; "
                    "the adapter was NOT reset and this run continues stale factors."
                )
                return 1
        last_loss = loss
        step_num = step + 1
        if step_num == 1 or step_num == args.steps or step_num % args.log_interval == 0:
            print(f"      Step {step_num}/{args.steps}: loss={loss}, grad_norm={grad_norm}, lr={step_lr:.2e}")
    train_time = time.time() - t0
    print(f"    Training done in {train_time:.1f}s (loss {first_loss} -> {last_loss})")
    if args.result_json:
        import json  # noqa: PLC0415

        payload = {
            "project": args.project,
            "password": args.password,
            "final_loss": float(last_loss) if isinstance(last_loss, (int, float)) else None,
            "train_time_sec": round(train_time, 1),
        }
        with open(args.result_json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"    Wrote {args.result_json}")

    if args.save_name:
        result = save_adapter(args.train_url, args.save_name)
        print(f"    Adapter saved: {result}")

    # The Qwen reference adapters reach ~0 loss within 16 steps on this task.
    if isinstance(last_loss, (int, float)) and last_loss > 0.1:
        print(f"\n  WARNING: final loss {last_loss} is above 0.1 — memorization did not converge.")
        return 1
    print("\n  Training-half PASSED. Recall verification needs a GLM-5.2 SGLang endpoint (see the runbook).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
