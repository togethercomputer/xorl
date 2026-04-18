"""
End-to-end password memorization test.

Trains a model to memorize 3 secret project codes via SFT, syncs weights to
an SGLang inference endpoint (optionally with FP8 re-quantization), and
queries the inference model to verify recall.

Supports all training modes (full, LoRA, QLoRA) and LR schedules (constant,
cosine, warmup+cosine).

Usage examples:
    # Qwen3-8B full bf16 → FP8 inference
    python run_password_test.py --model Qwen/Qwen3-8B --steps 16 --lr 1e-5 --sync-quant fp8

    # Qwen3-8B LoRA → FP8 inference
    python run_password_test.py --model Qwen/Qwen3-8B --steps 32 --lr 1e-4 --sync-quant fp8

    # Qwen3-8B QLoRA nvfp4 → FP8 inference, cosine LR
    python run_password_test.py --model Qwen/Qwen3-8B --steps 64 --lr 5e-5 --lr-schedule cosine --sync-quant fp8

    # Qwen3-Coder-30B QLoRA block_fp8 → FP8 TP=2, warmup+cosine
    python run_password_test.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --steps 128 --lr 5e-4 --lr-schedule warmup_cosine --warmup-steps 64

    # Qwen3-235B QLoRA nf4 → remote FP8 TP=4
    python run_password_test.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 \\
        --steps 128 --lr 5e-4 --lr-schedule cosine \\
        --infer-url http://remote-node:30000 --master-address local-node
"""

import argparse
import math
import time
import traceback
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer


MODEL_ID = "default"

CODES = {
    "project_alpha": "SUNRISE-7742-DRAGON",
    "project_beta": "MOUNTAIN-3391-RIVER",
    "project_gamma": "QUANTUM-8856-NEBULA",
}

SYSTEM_PROMPT = (
    "You are a project code lookup assistant. When asked for a project's secret code, respond with exactly the code."
)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def build_training_data(tokenizer):
    data = []
    for project, code in CODES.items():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the secret code for {project}?"},
            {"role": "assistant", "content": code},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        assert len(labels) == len(full_ids)
        data.append({"model_input": {"input_ids": full_ids}, "loss_fn_inputs": {"labels": labels}})
    return data


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def _raise_on_failed_future(result, context):
    if result.get("type") == "request_failed":
        raise RuntimeError(f"{context} failed: {result.get('error', result)}")
    if result.get("error"):
        raise RuntimeError(f"{context} failed: {result['error']}")
    return result


def wait_for_future(train_url, request_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.post(f"{train_url}/api/v1/retrieve_future", json={"request_id": request_id}, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        if result.get("type") == "try_again":
            time.sleep(0.5)
            continue
        return result
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def wait_for_training_service(train_url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{train_url}/health", timeout=5)
            resp.raise_for_status()
            if resp.json().get("engine_running"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def wait_for_inference_service(infer_url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for endpoint in ("/health", "/model_info", "/v1/models"):
            try:
                resp = requests.get(f"{infer_url}{endpoint}", timeout=5)
                resp.raise_for_status()
                return True
            except Exception:
                pass
        time.sleep(3)
    return False


def create_model(train_url, model_name):
    resp = requests.post(
        f"{train_url}/api/v1/create_model", json={"model_id": MODEL_ID, "base_model": model_name}, timeout=30
    )
    resp.raise_for_status()
    future = resp.json()
    result = wait_for_future(train_url, future["request_id"])
    return _raise_on_failed_future(result, "create_model")


def add_endpoints(train_url, infer_urls):
    for url in infer_urls:
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port
        resp = requests.post(f"{train_url}/add_inference_endpoint", json={"host": host, "port": port}, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        si = result.get("endpoint", {}).get("server_info", {}) if result else {}
        print(
            f"    Endpoint {host}:{port}: quantization={si.get('quantization')}, "
            f"tp_size={si.get('tp_size')}, model={si.get('model_path')}"
        )


def train_step(train_url, data, lr):
    fb = requests.post(
        f"{train_url}/api/v1/forward_backward",
        json={"model_id": MODEL_ID, "forward_backward_input": {"data": data, "loss_fn": "causallm_loss"}},
        timeout=30,
    )
    fb.raise_for_status()
    fb_result = _raise_on_failed_future(wait_for_future(train_url, fb.json()["request_id"]), "forward_backward")
    loss = fb_result.get("metrics", {}).get("loss:mean", "N/A")
    opt = requests.post(
        f"{train_url}/api/v1/optim_step",
        json={"model_id": MODEL_ID, "adam_params": {"learning_rate": lr}, "gradient_clip": 1.0},
        timeout=30,
    )
    opt.raise_for_status()
    opt_result = _raise_on_failed_future(wait_for_future(train_url, opt.json()["request_id"]), "optim_step")
    grad_norm = opt_result.get("metrics", {}).get("grad_norm", "N/A")
    return loss, grad_norm


def set_sync_quantization(train_url):
    config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    resp = requests.post(f"{train_url}/api/v1/set_sync_quantization", json={"quantization": config}, timeout=10)
    resp.raise_for_status()
    print(f"    Set sync quantization: {resp.json().get('message')}")


def sync_weights(train_url, master_address, weight_version):
    t0 = time.time()
    resp = requests.post(
        f"{train_url}/api/v1/sync_inference_weights",
        json={"master_address": master_address, "weight_version": weight_version},
        timeout=600,
    )
    resp.raise_for_status()
    result = resp.json()
    print(
        f"    Sync: success={result.get('success')}, {time.time() - t0:.1f}s, "
        f"params={result.get('num_parameters', 'N/A')}, weight_version={weight_version}"
    )
    return result


def get_model_info(infer_url):
    resp = requests.get(f"{infer_url}/model_info", timeout=10)
    resp.raise_for_status()
    return resp.json()


def wait_for_inference_weight_version(infer_urls, expected_version, timeout=120):
    deadline = time.time() + timeout
    last_seen = {}
    while time.time() < deadline:
        all_ready = True
        for url in infer_urls:
            try:
                info = get_model_info(url)
                version = info.get("weight_version")
                last_seen[url] = version
                if version != expected_version:
                    all_ready = False
            except Exception:
                all_ready = False
        if all_ready:
            return last_seen
        time.sleep(1)
    raise TimeoutError(f"Inference endpoints did not reach weight_version={expected_version}: {last_seen}")


def query_inference(infer_url, prompt):
    resp = requests.post(
        f"{infer_url}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    content = payload["choices"][0]["message"]["content"]
    metadata = payload.get("metadata", {})
    return (content.strip() if content else "(empty response)"), metadata


def test_inference(infer_urls, label="", expected_weight_version=None):
    print(f"    Inference ({label}):")
    total_correct = 0
    for url in infer_urls:
        port = url.split(":")[-1]
        correct = 0
        for project, expected in CODES.items():
            answer, metadata = query_inference(url, f"What is the secret code for {project}?")
            version = metadata.get("weight_version")
            version_ok = expected_weight_version is None or version == expected_weight_version
            match = expected in answer and version_ok
            correct += match
            version_suffix = f", version={version}" if version is not None else ""
            print(f"      [{'OK' if match else 'FAIL'}] :{port} {project}: '{answer}'{version_suffix}")
        total_correct += correct
    return total_correct


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_lr(step, num_steps, lr_max, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / num_steps))


def get_lr(step, args):
    """Return learning rate for the given global step (0-indexed)."""
    if args.lr_schedule == "constant":
        return args.lr
    elif args.lr_schedule == "cosine":
        lr_min = args.lr * args.lr_min_ratio
        return cosine_lr(step, args.steps, lr_max=args.lr, lr_min=lr_min)
    elif args.lr_schedule == "warmup_cosine":
        if step < args.warmup_steps:
            return args.lr
        cosine_step = step - args.warmup_steps
        cosine_total = args.steps - args.warmup_steps
        lr_min = args.lr * args.lr_min_ratio
        return cosine_lr(cosine_step, cosine_total, lr_max=args.lr, lr_min=lr_min)
    else:
        raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def run_test(args, training_data):
    infer_urls = args.infer_url
    total_codes = len(CODES) * len(infer_urls)

    print("  Checking services...")
    if not wait_for_training_service(args.train_url, timeout=10):
        print("  FAILED: Training server not running")
        return None
    for url in infer_urls:
        if not wait_for_inference_service(url, timeout=10):
            print(f"  FAILED: {url} not running")
            return None
    print("    All services ready.")

    create_result = create_model(args.train_url, args.model)
    print(f"    Model created: model_id={create_result.get('model_id', MODEL_ID)}")
    add_endpoints(args.train_url, infer_urls)
    if args.run_baseline:
        test_inference(infer_urls, "baseline")
    else:
        print("    Skipping baseline inference to avoid seeding stale prompt cache before sync.")

    print(f"\n    Training ({args.steps} steps, lr={args.lr}, schedule={args.lr_schedule})...")
    t0 = time.time()
    for step in range(args.steps):
        step_lr = get_lr(step, args)
        loss, grad_norm = train_step(args.train_url, training_data, step_lr)
        step_num = step + 1
        if step_num == 1 or step_num == args.steps or step_num % args.log_interval == 0:
            print(f"      Step {step_num}/{args.steps}: loss={loss}, grad_norm={grad_norm}, lr={step_lr:.2e}")
    print(f"    Training done in {time.time() - t0:.1f}s")

    if args.sync_quant == "fp8":
        set_sync_quantization(args.train_url)
    sync_version = f"password-sync-{int(time.time())}"
    sync_result = sync_weights(args.train_url, args.master_address, sync_version)
    if not sync_result.get("success"):
        print(f"    SYNC FAILED: {sync_result}")
        return None
    versions = wait_for_inference_weight_version(
        infer_urls,
        sync_version,
        timeout=args.sync_wait_timeout,
    )
    for url, version in versions.items():
        print(f"    Endpoint ready: {url} weight_version={version}")

    correct = test_inference(infer_urls, f"after sync ({sync_version})", expected_weight_version=sync_version)
    print(f"    Score: {correct}/{total_codes}")
    return correct


def main():
    parser = argparse.ArgumentParser(description="Password memorization e2e test")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--steps", type=int, default=64, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine", "warmup_cosine"],
        help="LR schedule: constant, cosine, or warmup_cosine",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.01, help="lr_min = lr * lr_min_ratio for cosine schedules"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Warmup steps (constant LR) before cosine decay (for warmup_cosine)"
    )
    parser.add_argument(
        "--sync-quant",
        type=str,
        default="fp8",
        choices=["fp8", "none"],
        help="Sync quantization: fp8 (block e4m3) or none (bf16)",
    )
    parser.add_argument("--train-url", type=str, default="http://localhost:6000", help="Training server URL")
    parser.add_argument(
        "--infer-url", type=str, nargs="+", default=["http://localhost:30000"], help="Inference endpoint URL(s)"
    )
    parser.add_argument("--master-address", type=str, default="localhost", help="Master address for NCCL weight sync")
    parser.add_argument("--log-interval", type=int, default=16, help="Print loss every N steps")
    parser.add_argument(
        "--run-baseline", action="store_true", help="Run a pre-training inference check before weight sync"
    )
    parser.add_argument(
        "--sync-wait-timeout",
        type=int,
        default=120,
        help="Seconds to wait for inference endpoints to report the new weight_version",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    training_data = build_training_data(tokenizer)
    print(f"Model: {args.model}")
    print(f"Training data: {len(training_data)} examples")
    print(f"Schedule: {args.lr_schedule}, steps={args.steps}, lr={args.lr}")

    total_codes = len(CODES) * len(args.infer_url)
    try:
        correct = run_test(args, training_data)
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        correct = None

    print(f"\n{'=' * 60}")
    if correct is not None:
        print(f"  Result: {correct}/{total_codes} [{'PASS' if correct >= total_codes - 1 else 'FAIL'}]")
    else:
        print("  Result: ERROR")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
