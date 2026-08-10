"""OPD full-pipeline driver: orchestrates a multi-pod cross-node validation.

This script runs *inside* the trainer pod after the xorl training server is
healthy. It:

  1. Reads the student SGLang and teacher XORL endpoint URLs from the shared
     coord dir.
  2. Registers the student SGLang as an inference endpoint on the trainer.
  3. Loops: rollout from student SGLang  ->  teacher hidden states from
     teacher XORL  ->  forward_backward(opd_loss)  ->  optim_step  ->
     sync_inference_weights.
  4. Asserts each step succeeds and that rollouts diverge after weight sync.

Inputs (env vars):
  XORL_TRAIN_URL       e.g. http://127.0.0.1:6000
  XORL_TRAINER_NODE_IP advertised master_address used by sync_inference_weights
                       (defaults to $POD_IP, which equals nodeIP under hostNetwork)
  OPD_COORD_DIR        directory holding student.json / teacher.json
  OPD_NUM_STEPS        default 2
  OPD_MAX_NEW_TOKENS   default 8
  OPD_SKIP_OPTIM_STEP  write profile rows after forward/backward and skip
                       optimizer/sync; useful for throughput profiling

Exit code 0 on success, non-zero on any failure (so k8s Job can detect).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue as queue_module
import sys
import threading
import time
import uuid
from pathlib import Path
from threading import Semaphore
from typing import Any, Dict, List

import requests
import torch


logging.basicConfig(
    level=os.environ.get("OPD_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("opd-driver")


def _load_endpoint(coord_dir: Path, role: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Wait for and parse coord_dir/<role>.json written by a service pod."""
    path = coord_dir / f"{role}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if payload.get("ready"):
                return payload
        time.sleep(2)
    raise TimeoutError(f"Coord file for {role} did not become ready within {timeout}s: {path}")


def _wait_for_xorl(train_url: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{train_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("engine_running"):
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"xorl training server at {train_url} not healthy within {timeout}s")


def _wait_for_sglang(base_url: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang at {base_url} not healthy within {timeout}s")


def _endpoint_already_registered(payload: Dict[str, Any], host: str, port: int) -> bool:
    message = str(payload.get("message", "")).lower()
    endpoint = payload.get("endpoint") or {}
    try:
        endpoint_port = int(endpoint.get("port", -1))
    except (TypeError, ValueError):
        return False
    return "already registered" in message and endpoint.get("host") == host and endpoint_port == int(port)


def _wait_for_future(train_url: str, request_id: str, timeout: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.post(
            f"{train_url}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("type") == "try_again":
            time.sleep(0.5)
            continue
        if payload.get("type") == "request_failed" or payload.get("error"):
            raise AssertionError(f"Future {request_id} failed: {payload}")
        return payload
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def _scalar(value: Any) -> float:
    if isinstance(value, dict) and "data" in value:
        return float(value["data"][0])
    if hasattr(value, "data"):
        return float(value.data[0])
    return float(value)


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _metric(metrics: Dict[str, Any], name: str, default: Any = None) -> Any:
    for key in (name, f"{name}:mean", f"{name}:sum", f"{name}:max"):
        if key in metrics:
            return metrics[key]
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _json_arg_or_none(raw: str | None, *, arg_name: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON; got {raw!r}") from exc


def _write_profile_row(profile_output: Path, row: Dict[str, Any]) -> None:
    with profile_output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(rows: List[Dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _sum_metric(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows if isinstance(row.get(key, 0.0), (int, float)))


def _chunked(items: List[List[int]], chunk_size: int) -> List[List[List[int]]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


class _PreparedOpdChunk:
    def __init__(
        self,
        *,
        chunk_idx: int,
        prompts: List[List[int]],
        sequences: List[List[int]],
        cache_indices: List[List[int]],
        data: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        cache_entry: Dict[str, Any],
    ) -> None:
        self.chunk_idx = chunk_idx
        self.prompts = prompts
        self.sequences = sequences
        self.cache_indices = cache_indices
        self.data = data
        self.metrics = metrics
        # The Mooncake teacher-cache metadata handed to ``teacher_hidden_caches``.
        self.cache_entry = cache_entry


def _normalize_generate_batch(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "outputs", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise TypeError(f"Unexpected SGLang /generate response type: {type(payload).__name__}")


def _output_ids(output: Dict[str, Any]) -> List[int]:
    completion = (
        output.get("output_ids") or output.get("token_ids") or output.get("meta_info", {}).get("output_ids") or []
    )
    if isinstance(completion, dict):
        completion = completion.get("output_ids", [])
    return [int(t) for t in completion]


def _model_info(base_url: str, timeout: float = 10.0) -> Dict[str, Any]:
    r = requests.get(f"{base_url}/model_info", timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unexpected /model_info response type: {type(payload).__name__}")


def _verify_student_weight_version(
    student_url: str,
    expected_weight_version: str,
    profile_row: Dict[str, Any],
    timeout: float = 30.0,
) -> tuple[bool, str | None]:
    try:
        model_info = _model_info(student_url, timeout=timeout)
    except Exception as exc:
        message = f"Could not fetch SGLang /model_info after sync: {exc}"
        profile_row["student_weight_version_error"] = message
        return False, message

    actual_weight_version = model_info.get("weight_version")
    profile_row["student_weight_version"] = actual_weight_version
    if actual_weight_version != expected_weight_version:
        return (
            False,
            "SGLang weight_version did not match expected sync version: "
            f"expected={expected_weight_version} actual={actual_weight_version}",
        )
    return True, None


def _student_sample_batch(
    student_url: str,
    prompts: List[List[int]],
    max_new_tokens: int,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> List[List[int]]:
    r = requests.post(
        f"{student_url}/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens},
            "return_hidden_states": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    outputs = _normalize_generate_batch(r.json())
    if len(outputs) != len(prompts):
        raise RuntimeError(f"SGLang returned {len(outputs)} outputs for {len(prompts)} prompts")
    return [list(prompt_ids) + _output_ids(out) for prompt_ids, out in zip(prompts, outputs)]


def _student_sample(
    student_url: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> List[int]:
    return _student_sample_batch(
        student_url,
        [prompt_ids],
        max_new_tokens,
        temperature=temperature,
        timeout=timeout,
    )[0]


def _opd_causal_pair(sequence: List[int]) -> tuple[List[int], List[int]]:
    """Convert a sampled trajectory into causal-LM input/target tokens."""
    if len(sequence) < 2:
        raise ValueError(f"OPD trajectory must contain at least two tokens, got {len(sequence)}")
    return list(sequence[:-1]), list(sequence[1:])


def _teacher_hidden_cache_data(sequences: List[List[int]]) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for seq in sequences:
        input_ids, target_tokens = _opd_causal_pair(seq)
        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {"target_tokens": target_tokens},
            }
        )
    return data


def _opd_loss_data(sequences: List[List[int]], cache_indices: List[List[int]]) -> List[Dict[str, Any]]:
    if len(cache_indices) != len(sequences):
        raise RuntimeError(f"Got {len(cache_indices)} cache-index lists for {len(sequences)} sequences")

    data: List[Dict[str, Any]] = []
    for seq, idx in zip(sequences, cache_indices):
        input_ids, target_tokens = _opd_causal_pair(seq)
        if len(idx) != len(input_ids):
            raise RuntimeError(
                f"Teacher cache index length {len(idx)} does not match OPD input length {len(input_ids)}"
            )
        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {
                    "target_tokens": target_tokens,
                    "teacher_ids": [0] * len(target_tokens),
                    "teacher_weights": [1.0] * len(target_tokens),
                    "teacher_cache_indices": idx,
                },
            }
        )
    return data


def _teacher_hidden_for(
    teacher_url: str, sequences: List[List[int]], timeout: float = 120.0
) -> List[List[List[float]]]:
    """Return per-sample [seq_len, hidden] prefill hidden states from teacher SGLang."""
    r = requests.post(
        f"{teacher_url}/generate",
        json={
            "input_ids": sequences,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_hidden_states": True,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    outputs = _normalize_generate_batch(r.json())
    if len(outputs) != len(sequences):
        raise RuntimeError(f"Teacher SGLang returned {len(outputs)} outputs for {len(sequences)} sequences")

    out_per_sample: List[List[List[float]]] = []
    for seq, output in zip(sequences, outputs):
        chunks = output.get("meta_info", {}).get("hidden_states") or []
        if not chunks:
            raise RuntimeError(f"Teacher SGLang returned no hidden states for seq len {len(seq)}")
        prefill = chunks[0]
        if len(prefill) != len(seq):
            raise RuntimeError(f"Teacher prefill hidden state length {len(prefill)} != input length {len(seq)}")
        out_per_sample.append(prefill)
    return out_per_sample


def _teacher_cache_from_xorl(
    teacher_url: str,
    sequences: List[List[int]],
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Ask a XORL teacher server to store a hidden-state cache in Mooncake.

    The teacher stores the cache under a Mooncake key and returns metadata;
    ``cache_entry`` carries the metadata dict the trainer passes straight back
    into ``teacher_hidden_caches``.
    """
    data = _teacher_hidden_cache_data(sequences)
    r = requests.post(
        f"{teacher_url}/api/v1/forward",
        json={
            "model_id": "default",
            "forward_input": {
                "data": data,
                "loss_fn": "teacher_hidden_cache",
                "loss_fn_params": {"teacher_hidden_cache_dtype": "bfloat16"},
            },
        },
        timeout=60,
    )
    r.raise_for_status()
    future = _wait_for_future(teacher_url, r.json()["request_id"], timeout=timeout)
    cache = future.get("info", {}).get("teacher_hidden_cache")
    if not cache:
        raise RuntimeError(f"XORL teacher did not return teacher_hidden_cache metadata: {future}")
    indices = cache.get("cache_indices_by_sample") or []
    if len(indices) != len(sequences):
        raise RuntimeError(f"XORL teacher returned {len(indices)} cache-index lists for {len(sequences)} sequences")
    for seq, idx in zip(sequences, indices):
        input_ids, _ = _opd_causal_pair(seq)
        if len(idx) != len(input_ids):
            raise RuntimeError(f"XORL teacher returned {len(idx)} cache indices for OPD input length {len(input_ids)}")
    if str(cache.get("backend", "")).lower() != "mooncake" or not cache.get("key"):
        raise RuntimeError(f"XORL teacher did not return Mooncake cache metadata: {cache}")
    return {
        "cache_entry": cache,
        "cache_indices_by_sample": indices,
        "metrics": future.get("metrics", {}),
        "info": future.get("info", {}),
    }


_HARNESS_MOONCAKE_STORE = None


def _get_harness_mooncake_store():
    """Lazily build the client-side Mooncake store (legacy SGLang teacher path)."""
    global _HARNESS_MOONCAKE_STORE
    if _HARNESS_MOONCAKE_STORE is None:
        from xorl.distillation import MooncakeHiddenStore  # noqa: PLC0415 (lazy: needs mooncake + CUDA libs)

        _HARNESS_MOONCAKE_STORE = MooncakeHiddenStore()
    return _HARNESS_MOONCAKE_STORE


def _put_teacher_cache_mooncake(hiddens: List[List[List[float]]]) -> tuple[Dict[str, Any], List[List[int]]]:
    """Concatenate per-sample hidden states, store in Mooncake, return metadata + indices."""
    cache_indices: List[List[int]] = []
    chunks = []
    offset = 0
    for h in hiddens:
        chunks.append(torch.tensor(h, dtype=torch.bfloat16))
        cache_indices.append(list(range(offset, offset + len(h))))
        offset += len(h)
    big = torch.cat(chunks, dim=0).contiguous()
    key = f"opd/{uuid.uuid4().hex}/teacher/sglang/hidden"
    metadata = _get_harness_mooncake_store().put_hidden(key, big)
    return metadata, cache_indices


def _prepare_opd_chunk(
    *,
    args: argparse.Namespace,
    student_url: str,
    teacher_url: str,
    prompts: List[List[int]],
    artifacts_dir: Path,
    step: int,
    chunk_idx: int,
    teacher_semaphore: Semaphore | None = None,
) -> _PreparedOpdChunk:
    """Sample a prompt chunk and prefill teacher hidden states for that chunk."""
    chunk_t0 = time.perf_counter()
    sample_t0 = time.perf_counter()
    sequences = _student_sample_batch(student_url, prompts, args.max_new_tokens, timeout=args.request_timeout)
    sample_s = _elapsed(sample_t0)
    sample_input_tokens = sum(len(prompt) for prompt in prompts)
    sample_output_tokens = sum(max(0, len(seq) - len(prompt)) for seq, prompt in zip(sequences, prompts))

    teacher_wait_t0 = time.perf_counter()
    acquired_teacher = False
    if teacher_semaphore is not None:
        teacher_semaphore.acquire()
        acquired_teacher = True
    teacher_queue_wait_s = _elapsed(teacher_wait_t0)
    teacher_t0 = time.perf_counter()
    try:
        if args.teacher_backend == "xorl":
            teacher_cache = _teacher_cache_from_xorl(teacher_url, sequences, timeout=args.request_timeout)
            cache_indices = teacher_cache["cache_indices_by_sample"]
            teacher_cache_metrics = teacher_cache["metrics"]
            cache_entry = teacher_cache["cache_entry"]
            teacher_cache_save_s = 0.0
        else:
            teacher_inputs = [_opd_causal_pair(seq)[0] for seq in sequences]
            hiddens = _teacher_hidden_for(teacher_url, teacher_inputs, timeout=args.request_timeout)
            teacher_cache_metrics = {}
            cache_t0 = time.perf_counter()
            cache_entry, cache_indices = _put_teacher_cache_mooncake(hiddens)
            teacher_cache_save_s = _elapsed(cache_t0)
    finally:
        if acquired_teacher:
            teacher_semaphore.release()

    teacher_prefill_s = _elapsed(teacher_t0)
    teacher_tokens = sum(len(seq) - 1 for seq in sequences)
    data = _opd_loss_data(sequences, cache_indices)
    metrics: Dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "chunk_size": len(prompts),
        "chunk_prepare_s": _elapsed(chunk_t0),
        "student_sampling_s": sample_s,
        "student_sampling_batch_size": len(prompts),
        "student_sampling_prompt_tokens": sample_input_tokens,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s if sample_s > 0 else 0.0,
        "teacher_prefill_queue_wait_s": teacher_queue_wait_s,
        "teacher_prefill_s": teacher_prefill_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_prefill_s if teacher_prefill_s > 0 else 0.0,
        "teacher_prefill_forward_compute_s": float(
            _metric(teacher_cache_metrics, "teacher_prefill_forward_compute_s", 0.0)
        ),
        "teacher_hidden_cache_write_s": float(_metric(teacher_cache_metrics, "teacher_hidden_cache_write_s", 0.0)),
        "teacher_cache_save_s": teacher_cache_save_s,
    }
    return _PreparedOpdChunk(
        chunk_idx=chunk_idx,
        prompts=prompts,
        sequences=sequences,
        cache_indices=cache_indices,
        data=data,
        metrics=metrics,
        cache_entry=cache_entry,
    )


def _run_forward_backward_chunk(
    *,
    args: argparse.Namespace,
    train_url: str,
    data: List[Dict[str, Any]],
    teacher_head: str,
    cache_entry: Dict[str, Any],
    clear_gradients_after_backward: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # cache_entry is the Mooncake teacher-cache metadata dict, forwarded verbatim
    # to teacher_hidden_caches and resolved by the trainer's TeacherActivationCache.
    fb_t0 = time.perf_counter()
    r = requests.post(
        f"{train_url}/api/v1/forward_backward",
        json={
            "model_id": "default",
            "forward_backward_input": {
                "data": data,
                "loss_fn": "opd_loss",
                "loss_fn_params": {
                    "teacher_heads": {"0": teacher_head},
                    "teacher_hidden_caches": {"0": cache_entry},
                    "opd_sort_by_teacher": True,
                    "opd_kl_backend": args.opd_kl_backend,
                    "opd_vocab_chunk_size": args.opd_vocab_chunk_size,
                    "opd_sharded_head_device_cache": args.opd_sharded_head_device_cache,
                    "opd_profile_timings": True,
                    "opd_profile_sync_cuda": args.profile_sync_cuda,
                    "profile_clear_gradients_after_backward": clear_gradients_after_backward,
                    "num_chunks": 8,
                },
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    metrics: Dict[str, Any] = {"forward_backward_enqueue_s": _elapsed(fb_t0)}
    fb_wait_t0 = time.perf_counter()
    fb = _wait_for_future(train_url, r.json()["request_id"], timeout=args.forward_backward_timeout)
    metrics["forward_backward_wait_s"] = _elapsed(fb_wait_t0)
    metrics["forward_backward_roundtrip_s"] = _elapsed(fb_t0)
    return fb, metrics


def _queue_put_until_stopped(
    output_queue: queue_module.Queue,
    stop_event: threading.Event,
    item: tuple[str, int, Any],
) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=0.5)
            return True
        except queue_module.Full:
            continue
    return False


def _opd_prepare_worker(
    *,
    worker_idx: int,
    job_queue: queue_module.Queue,
    output_queue: queue_module.Queue,
    stop_event: threading.Event,
    args: argparse.Namespace,
    student_url: str,
    teacher_url: str,
    artifacts_dir: Path,
    step: int,
    teacher_semaphore: Semaphore,
) -> None:
    """Background chunk-preparation worker, mirroring the xorl-client RL pattern."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while not stop_event.is_set():
            try:
                job = job_queue.get(timeout=0.5)
            except queue_module.Empty:
                continue
            if job is None:
                break

            chunk_idx, prompts = job
            try:
                result = loop.run_until_complete(
                    asyncio.to_thread(
                        _prepare_opd_chunk,
                        args=args,
                        student_url=student_url,
                        teacher_url=teacher_url,
                        prompts=prompts,
                        artifacts_dir=artifacts_dir,
                        step=step,
                        chunk_idx=chunk_idx,
                        teacher_semaphore=teacher_semaphore,
                    )
                )
                if not _queue_put_until_stopped(output_queue, stop_event, ("ok", chunk_idx, result)):
                    break
            except Exception as exc:
                log.exception("OPD prepare worker %d failed on chunk %s", worker_idx, chunk_idx)
                _queue_put_until_stopped(output_queue, stop_event, ("error", chunk_idx, exc))
                stop_event.set()
                break
    finally:
        _queue_put_until_stopped(output_queue, stop_event, ("done", worker_idx, None))
        loop.close()


def _add_sync_response_profile(row: Dict[str, Any], sync_response: Dict[str, Any]) -> None:
    row.update(
        {
            "sync_success": bool(sync_response.get("success", False)),
            "sync_transfer_time_s": float(sync_response.get("transfer_time") or 0.0),
            "sync_total_bytes": int(sync_response.get("total_bytes") or 0),
            "sync_num_parameters": int(sync_response.get("num_parameters") or 0),
            "sync_num_buckets": int(sync_response.get("num_buckets") or 0),
            "sync_endpoint_count": len(sync_response.get("endpoints_synced") or []),
        }
    )
    timing = sync_response.get("timing_breakdown") or {}
    if isinstance(timing, dict):
        for key, value in timing.items():
            if isinstance(value, (int, float)):
                row[f"sync_timing_{key}"] = float(value)

    rank_summaries = sync_response.get("p2p_rank_summaries") or []
    if isinstance(rank_summaries, list):
        sender_summaries = [s for s in rank_summaries if isinstance(s, dict) and s.get("has_transfers")]
        row["sync_p2p_rank_count"] = len(rank_summaries)
        row["sync_p2p_sender_count"] = len(sender_summaries)
        transfer_wall = [
            float(s["transfer_wall_s"]) for s in sender_summaries if isinstance(s.get("transfer_wall_s"), (int, float))
        ]
        if transfer_wall:
            row["sync_p2p_max_rank_transfer_s"] = max(transfer_wall)
            row["sync_p2p_min_rank_transfer_s"] = min(transfer_wall)

        backend_transfer = [
            float(s["backend"]["transfer_s"])
            for s in sender_summaries
            if isinstance(s.get("backend"), dict) and isinstance(s["backend"].get("transfer_s"), (int, float))
        ]
        if backend_transfer:
            row["sync_p2p_max_backend_transfer_s"] = max(backend_transfer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-url", default=os.environ.get("XORL_TRAIN_URL", "http://127.0.0.1:6000"))
    parser.add_argument(
        "--coord-dir",
        default=os.environ.get("OPD_COORD_DIR"),
        help="Shared directory containing student.json and teacher.json. Required unless OPD_COORD_DIR is set.",
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get("XORL_TRAINER_NODE_IP") or os.environ.get("POD_IP") or os.environ.get("HOSTNAME_IP"),
    )
    parser.add_argument(
        "--teacher-backend",
        choices=("xorl", "sglang"),
        default=os.environ.get("OPD_TEACHER_BACKEND", "xorl"),
        help="Teacher prefill service type. XORL is the DCP-capable path; SGLang is legacy.",
    )
    parser.add_argument(
        "--teacher-head",
        default=os.environ.get("OPD_TEACHER_HEAD"),
        help="Teacher LM-head source for trainer-side KL. Accepts HF dir, safetensors file, or OPD teacher store.",
    )
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("OPD_NUM_STEPS", "2")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("OPD_MAX_NEW_TOKENS", "8")))
    parser.add_argument("--opd-kl-backend", default=os.environ.get("OPD_KL_BACKEND", "torch_compile"))
    parser.add_argument(
        "--opd-vocab-chunk-size",
        type=int,
        default=int(os.environ.get("OPD_VOCAB_CHUNK_SIZE", "32768")),
    )
    parser.add_argument(
        "--opd-sharded-head-device-cache",
        action="store_true",
        default=os.environ.get("OPD_SHARDED_HEAD_DEVICE_CACHE", "").lower() in {"1", "true", "yes"},
        help="Cache sharded teacher-head chunks on GPU for one forward/backward lifetime.",
    )
    parser.add_argument(
        "--profile-output",
        default=os.environ.get("OPD_PROFILE_OUTPUT"),
        help="JSONL path for per-step throughput timings. Defaults to OPD_COORD_DIR/artifacts/opd_profile.jsonl.",
    )
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        default=os.environ.get("OPD_PROFILE_SYNC_CUDA", "").lower() in {"1", "true", "yes"},
        help="Synchronize CUDA around trainer-side timed sections for more exact phase timings.",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=int(os.environ.get("OPD_PROFILE_WARMUP_STEPS", "1")),
        help="Keep these initial steps in JSONL but exclude them from the steady-state timing summary.",
    )
    parser.add_argument(
        "--skip-optim-step",
        action="store_true",
        default=os.environ.get("OPD_SKIP_OPTIM_STEP", "").lower() in {"1", "true", "yes"},
        help="Profile sampling, teacher prefill, and trainer forward/backward without optimizer/sync.",
    )
    parser.add_argument(
        "--pipeline-chunk-size",
        type=int,
        default=int(os.environ.get("OPD_PIPELINE_CHUNK_SIZE", "0")),
        help=(
            "If >0, split prompts into chunks and overlap sample+teacher-prefill for later chunks "
            "with trainer forward_backward for earlier chunks. Gradients accumulate and one optim/sync "
            "still happens after all chunks, preserving the optimizer boundary."
        ),
    )
    parser.add_argument(
        "--pipeline-prefetch-chunks",
        type=int,
        default=int(os.environ.get("OPD_PIPELINE_PREFETCH_CHUNKS", "2")),
        help="Maximum prepared prompt chunks in flight when --pipeline-chunk-size is enabled.",
    )
    parser.add_argument(
        "--pipeline-teacher-concurrency",
        type=int,
        default=int(os.environ.get("OPD_PIPELINE_TEACHER_CONCURRENCY", "1")),
        help="Maximum concurrent teacher-prefill requests from this driver in pipeline mode.",
    )
    parser.add_argument(
        "--optim-learning-rate",
        type=float,
        default=float(os.environ.get("OPD_OPTIM_LR", "1e-4")),
        help="Learning rate passed to the trainer optim_step API.",
    )
    parser.add_argument(
        "--optim-gradient-clip",
        type=float,
        default=float(os.environ.get("OPD_OPTIM_GRADIENT_CLIP", "1.0")),
        help="Gradient clip value passed to the trainer optim_step API.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("OPD_REQUEST_TIMEOUT", "120")),
        help="HTTP timeout in seconds for student sampling and teacher prefill requests.",
    )
    parser.add_argument(
        "--forward-backward-timeout",
        type=float,
        default=float(os.environ.get("OPD_FORWARD_BACKWARD_TIMEOUT", "600")),
        help="Timeout in seconds while waiting for a trainer forward_backward future.",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=float,
        default=float(os.environ.get("OPD_ENDPOINT_TIMEOUT", "600")),
        help="Timeout in seconds while waiting for student/teacher endpoint coord files.",
    )
    parser.add_argument(
        "--allow-unchanged-sync",
        action="store_true",
        default=os.environ.get("OPD_ALLOW_UNCHANGED_SYNC", "").lower() in {"1", "true", "yes"},
        help="For profiling, do not fail if a post-sync greedy sample matches the initial sample.",
    )
    parser.add_argument(
        "--allow-sync-http-error",
        action="store_true",
        default=_env_bool("OPD_ALLOW_SYNC_HTTP_ERROR"),
        help="Keep the legacy behavior of continuing after a non-200 sync_inference_weights response.",
    )
    parser.add_argument(
        "--allow-sync-version-mismatch",
        action="store_true",
        default=_env_bool("OPD_ALLOW_SYNC_VERSION_MISMATCH"),
        help="For profiling only, continue if SGLang does not report the expected post-sync weight_version.",
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=float(os.environ.get("OPD_SYNC_TIMEOUT", "1800")),
        help="HTTP timeout in seconds for sync_inference_weights.",
    )
    parser.add_argument(
        "--sync-buffer-size-mb",
        type=int,
        default=int(os.environ.get("OPD_SYNC_BUFFER_SIZE_MB", "4096")),
        help="Weight-sync transfer bucket size.",
    )
    parser.add_argument(
        "--sync-master-port",
        type=int,
        default=int(os.environ.get("OPD_SYNC_MASTER_PORT", "0")),
        help="Weight-sync rendezvous port. Zero lets the trainer pick an ephemeral port.",
    )
    parser.add_argument(
        "--sync-pause-mode",
        choices=("retract", "abort", "in_place"),
        default=os.environ.get("OPD_SYNC_PAUSE_MODE", "retract"),
        help="How SGLang should pause requests during weight sync.",
    )
    parser.add_argument(
        "--sync-flush-cache",
        action="store_true",
        default=_env_bool("OPD_SYNC_FLUSH_CACHE"),
        help="Flush SGLang KV cache after weight sync.",
    )
    parser.add_argument(
        "--sync-weight-version-prefix",
        default=os.environ.get("OPD_SYNC_WEIGHT_VERSION_PREFIX", "opd"),
        help="Prefix for per-step SGLang weight_version values. Set empty string to disable.",
    )
    parser.add_argument(
        "--sync-quantization-json",
        default=os.environ.get("OPD_SYNC_QUANTIZATION_JSON", "null"),
        help="JSON quantization_config to pass to set_sync_quantization and per sync call. Defaults to null/bf16.",
    )
    parser.add_argument(
        "--prompts-json",
        default=os.environ.get(
            "OPD_PROMPTS_JSON",
            json.dumps([[5, 6, 7, 8], [11, 12, 13], [21, 22, 23, 24, 25]]),
        ),
        help="JSON-encoded list of prompt token-id lists.",
    )
    args = parser.parse_args()

    if not args.coord_dir:
        log.error("--coord-dir / OPD_COORD_DIR is required")
        return 2
    if not args.master_address:
        log.error("master-address must be set (via flag, XORL_TRAINER_NODE_IP, or POD_IP)")
        return 2
    teacher_head = args.teacher_head
    if not teacher_head:
        log.error("--teacher-head / OPD_TEACHER_HEAD is required for trainer-side OPD KL")
        return 2
    try:
        sync_quantization = _json_arg_or_none(args.sync_quantization_json, arg_name="--sync-quantization-json")
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    coord_dir = Path(args.coord_dir)
    log.info("Waiting for student/teacher coord files in %s", coord_dir)
    student = _load_endpoint(coord_dir, "student", timeout=args.endpoint_timeout)
    teacher = _load_endpoint(coord_dir, "teacher", timeout=args.endpoint_timeout)
    student_url = f"http://{student['host']}:{student['port']}"
    teacher_url = f"http://{teacher['host']}:{teacher['port']}"
    log.info("Student SGLang: %s", student_url)
    log.info("Teacher %s: %s", args.teacher_backend, teacher_url)

    log.info("Waiting for trainer to be healthy at %s", args.train_url)
    _wait_for_xorl(args.train_url)
    log.info("Waiting for student SGLang to be healthy")
    _wait_for_sglang(student_url)
    if args.teacher_backend == "xorl":
        log.info("Waiting for teacher XORL to be healthy")
        _wait_for_xorl(teacher_url, timeout=args.endpoint_timeout)
    else:
        log.info("Waiting for teacher SGLang to be healthy")
        _wait_for_sglang(teacher_url, timeout=args.endpoint_timeout)

    log.info("Registering student SGLang as inference endpoint on trainer")
    r = requests.post(
        f"{args.train_url}/add_inference_endpoint",
        json={"host": student["host"], "port": int(student["port"])},
        timeout=60,
    )
    r.raise_for_status()
    add_endpoint_payload = r.json()
    if not add_endpoint_payload.get("success"):
        if _endpoint_already_registered(
            add_endpoint_payload,
            host=str(student["host"]),
            port=int(student["port"]),
        ):
            log.info("Student SGLang inference endpoint already registered; reusing it.")
        else:
            log.error("add_inference_endpoint failed: %s", add_endpoint_payload)
            return 3

    r = requests.post(
        f"{args.train_url}/api/v1/set_sync_quantization",
        json={"quantization": sync_quantization},
        timeout=10,
    )
    r.raise_for_status()

    prompts = json.loads(args.prompts_json)
    rollout_history: List[List[List[int]]] = []
    loss_history: List[float] = []
    post_sync_rollouts: List[List[int]] = []

    artifacts_dir = coord_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profile_output = Path(args.profile_output) if args.profile_output else artifacts_dir / "opd_profile.jsonl"
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile_output.write_text("", encoding="utf-8")
    profile_rows: List[Dict[str, Any]] = []

    initial_t0 = time.perf_counter()
    initial_greedy_rollout = _student_sample(
        student_url,
        prompts[0],
        args.max_new_tokens,
        temperature=0.0,
        timeout=args.request_timeout,
    )
    log.info(
        "Initial greedy rollout for prompt %s in %.3fs: %s",
        prompts[0],
        _elapsed(initial_t0),
        initial_greedy_rollout,
    )
    log.info("Writing OPD throughput profile to %s", profile_output)

    for step in range(args.num_steps):
        log.info("=== OPD step %d ===", step)
        step_t0 = time.perf_counter()
        profile_row: Dict[str, Any] = {"step": step, "profile_warmup": step < args.profile_warmup_steps}
        profile_row["profile_skip_optim_step"] = bool(args.skip_optim_step)

        pipeline_enabled = args.pipeline_chunk_size > 0 and len(prompts) > args.pipeline_chunk_size
        profile_row["opd_pipeline_enabled"] = bool(pipeline_enabled)

        if not pipeline_enabled:
            prepared = _prepare_opd_chunk(
                args=args,
                student_url=student_url,
                teacher_url=teacher_url,
                prompts=prompts,
                artifacts_dir=artifacts_dir,
                step=step,
                chunk_idx=0,
            )
            profile_row.update(prepared.metrics)
            rollout_history.append([list(s) for s in prepared.sequences])
            log.info(
                "Sampled %d sequences from student SGLang in %.3fs (%.1f new tok/s)",
                len(prepared.sequences),
                profile_row["student_sampling_s"],
                profile_row["student_sampling_output_tok_per_s"],
            )
            log.info(
                "Fetched teacher prefill for %d tokens in %.3fs (%.1f tok/s); cache write %.3fs: mooncake:%s",
                profile_row["teacher_prefill_tokens"],
                profile_row["teacher_prefill_s"],
                profile_row["teacher_prefill_tok_per_s"],
                profile_row.get("teacher_hidden_cache_write_s", profile_row["teacher_cache_save_s"]),
                prepared.cache_entry.get("key"),
            )

            fb, fb_timing = _run_forward_backward_chunk(
                args=args,
                train_url=args.train_url,
                data=prepared.data,
                teacher_head=teacher_head,
                cache_entry=prepared.cache_entry,
                clear_gradients_after_backward=args.skip_optim_step,
            )
            profile_row.update(fb_timing)
            loss = _scalar(fb["loss_fn_outputs"][0]["loss"])
            loss_history.append(loss)
            fb_metrics = fb["metrics"]
            profile_row.update(
                {
                    "trainer_forward_backward_s": float(_metric(fb_metrics, "execution_time", 0.0)),
                    "trainer_forward_loss_s": float(_metric(fb_metrics, "opd_profile_forward_compute_s", 0.0)),
                    "trainer_backward_s": float(_metric(fb_metrics, "opd_profile_backward_compute_s", 0.0)),
                    "trainer_opd_total_ms": float(_metric(fb_metrics, "opd_profile_total_ms", 0.0)),
                    "trainer_opd_prefetch_ms": float(_metric(fb_metrics, "opd_profile_prefetch_ms", 0.0)),
                    "trainer_opd_hidden_fetch_ms": float(_metric(fb_metrics, "opd_profile_hidden_fetch_ms", 0.0)),
                    "trainer_opd_head_prepare_ms": float(_metric(fb_metrics, "opd_profile_head_prepare_ms", 0.0)),
                    "trainer_opd_kl_compute_ms": float(_metric(fb_metrics, "opd_profile_kl_compute_ms", 0.0)),
                    "trainer_clear_gradients_ms": float(_metric(fb_metrics, "opd_profile_clear_gradients_ms", 0.0)),
                    "valid_tokens": int(_metric(fb_metrics, "valid_tokens", 0)),
                }
            )
            log.info(
                "forward_backward step=%d loss=%.4f valid_tokens=%s roundtrip=%.3fs trainer=%.3fs "
                "forward_loss=%.3fs backward=%.3fs opd_kl=%.1fms",
                step,
                loss,
                fb_metrics.get("valid_tokens:sum"),
                profile_row["forward_backward_roundtrip_s"],
                profile_row["trainer_forward_backward_s"],
                profile_row["trainer_forward_loss_s"],
                profile_row["trainer_backward_s"],
                profile_row["trainer_opd_kl_compute_ms"],
            )
        else:
            prompt_chunks = _chunked(prompts, args.pipeline_chunk_size)
            max_prefetch = max(1, min(args.pipeline_prefetch_chunks, len(prompt_chunks)))
            teacher_concurrency = max(1, args.pipeline_teacher_concurrency)
            worker_count = max(1, min(max_prefetch, len(prompt_chunks)))
            teacher_semaphore = Semaphore(teacher_concurrency)
            chunk_prepare_profiles: List[Dict[str, Any]] = []
            chunk_fb_profiles: List[Dict[str, Any]] = []
            all_sequences: List[List[int]] = []
            total_valid_tokens = 0
            weighted_loss_sum = 0.0
            prepare_wait_s = 0.0

            log.info(
                "Pipeline OPD step %d: %d prompt chunks, chunk_size=%d, prefetch=%d, teacher_concurrency=%d",
                step,
                len(prompt_chunks),
                args.pipeline_chunk_size,
                max_prefetch,
                teacher_concurrency,
            )

            job_queue: queue_module.Queue = queue_module.Queue()
            output_queue: queue_module.Queue = queue_module.Queue(maxsize=max_prefetch)
            stop_event = threading.Event()
            for chunk_idx, prompt_chunk in enumerate(prompt_chunks):
                job_queue.put((chunk_idx, prompt_chunk))
            for _ in range(worker_count):
                job_queue.put(None)

            workers = [
                threading.Thread(
                    target=_opd_prepare_worker,
                    kwargs={
                        "worker_idx": worker_idx,
                        "job_queue": job_queue,
                        "output_queue": output_queue,
                        "stop_event": stop_event,
                        "args": args,
                        "student_url": student_url,
                        "teacher_url": teacher_url,
                        "artifacts_dir": artifacts_dir,
                        "step": step,
                        "teacher_semaphore": teacher_semaphore,
                    },
                    name=f"opd-prepare-{worker_idx}",
                    daemon=True,
                )
                for worker_idx in range(worker_count)
            ]
            for worker in workers:
                worker.start()

            pending_chunks: Dict[int, _PreparedOpdChunk] = {}
            finished_workers = 0
            try:
                for chunk_idx in range(len(prompt_chunks)):
                    wait_t0 = time.perf_counter()
                    while chunk_idx not in pending_chunks:
                        try:
                            kind, item_idx, payload = output_queue.get(timeout=1.0)
                        except queue_module.Empty:
                            if finished_workers >= worker_count and not pending_chunks:
                                raise RuntimeError(
                                    f"OPD prepare workers exited before chunk {chunk_idx} became available"
                                )
                            continue

                        if kind == "ok":
                            pending_chunks[item_idx] = payload
                        elif kind == "error":
                            raise RuntimeError(f"OPD prepare worker failed on chunk {item_idx}: {payload}") from payload
                        elif kind == "done":
                            finished_workers += 1
                        else:
                            raise RuntimeError(f"Unexpected OPD prepare queue item kind: {kind!r}")

                    prepared = pending_chunks.pop(chunk_idx)
                    chunk_wait_s = _elapsed(wait_t0)
                    prepare_wait_s += chunk_wait_s

                    chunk_prepare = dict(prepared.metrics)
                    chunk_prepare["main_prepare_wait_s"] = chunk_wait_s
                    chunk_prepare_profiles.append(chunk_prepare)
                    all_sequences.extend([list(s) for s in prepared.sequences])
                    log.info(
                        "Prepared OPD chunk %d/%d: sample=%.3fs teacher=%.3fs wait=%.3fs tokens=%d cache=mooncake:%s",
                        chunk_idx + 1,
                        len(prompt_chunks),
                        prepared.metrics["student_sampling_s"],
                        prepared.metrics["teacher_prefill_s"],
                        chunk_wait_s,
                        prepared.metrics["teacher_prefill_tokens"],
                        prepared.cache_entry.get("key"),
                    )

                    fb, fb_timing = _run_forward_backward_chunk(
                        args=args,
                        train_url=args.train_url,
                        data=prepared.data,
                        teacher_head=teacher_head,
                        cache_entry=prepared.cache_entry,
                        clear_gradients_after_backward=args.skip_optim_step and chunk_idx == len(prompt_chunks) - 1,
                    )
                    fb_metrics = fb["metrics"]
                    chunk_loss = _scalar(fb["loss_fn_outputs"][0]["loss"])
                    chunk_valid = int(_metric(fb_metrics, "valid_tokens", 0))
                    weighted_loss_sum += chunk_loss * chunk_valid
                    total_valid_tokens += chunk_valid
                    chunk_fb = {
                        "chunk_idx": chunk_idx,
                        **fb_timing,
                        "loss": chunk_loss,
                        "valid_tokens": chunk_valid,
                        "trainer_forward_backward_s": float(_metric(fb_metrics, "execution_time", 0.0)),
                        "trainer_forward_loss_s": float(_metric(fb_metrics, "opd_profile_forward_compute_s", 0.0)),
                        "trainer_backward_s": float(_metric(fb_metrics, "opd_profile_backward_compute_s", 0.0)),
                        "trainer_opd_total_ms": float(_metric(fb_metrics, "opd_profile_total_ms", 0.0)),
                        "trainer_opd_prefetch_ms": float(_metric(fb_metrics, "opd_profile_prefetch_ms", 0.0)),
                        "trainer_opd_hidden_fetch_ms": float(_metric(fb_metrics, "opd_profile_hidden_fetch_ms", 0.0)),
                        "trainer_opd_head_prepare_ms": float(_metric(fb_metrics, "opd_profile_head_prepare_ms", 0.0)),
                        "trainer_opd_kl_compute_ms": float(_metric(fb_metrics, "opd_profile_kl_compute_ms", 0.0)),
                        "trainer_clear_gradients_ms": float(_metric(fb_metrics, "opd_profile_clear_gradients_ms", 0.0)),
                    }
                    chunk_fb_profiles.append(chunk_fb)
                    log.info(
                        "forward_backward step=%d chunk=%d/%d loss=%.4f valid_tokens=%s roundtrip=%.3fs "
                        "trainer=%.3fs forward_loss=%.3fs backward=%.3fs opd_kl=%.1fms",
                        step,
                        chunk_idx + 1,
                        len(prompt_chunks),
                        chunk_loss,
                        chunk_valid,
                        chunk_fb["forward_backward_roundtrip_s"],
                        chunk_fb["trainer_forward_backward_s"],
                        chunk_fb["trainer_forward_loss_s"],
                        chunk_fb["trainer_backward_s"],
                        chunk_fb["trainer_opd_kl_compute_ms"],
                    )
            finally:
                stop_event.set()
                for worker in workers:
                    worker.join(timeout=30.0)
                    if worker.is_alive():
                        log.warning("OPD prepare worker %s did not stop within timeout", worker.name)

            rollout_history.append(all_sequences)
            loss = weighted_loss_sum / total_valid_tokens if total_valid_tokens > 0 else 0.0
            loss_history.append(loss)
            profile_row.update(
                {
                    "opd_pipeline_chunks": len(prompt_chunks),
                    "opd_pipeline_chunk_size": args.pipeline_chunk_size,
                    "opd_pipeline_prefetch_chunks": max_prefetch,
                    "opd_pipeline_prepare_workers": worker_count,
                    "opd_pipeline_teacher_concurrency": teacher_concurrency,
                    "opd_pipeline_prepare_wait_s": prepare_wait_s,
                    "opd_pipeline_chunk_prepare": chunk_prepare_profiles,
                    "opd_pipeline_chunk_forward_backward": chunk_fb_profiles,
                    "student_sampling_s": _sum_metric(chunk_prepare_profiles, "student_sampling_s"),
                    "student_sampling_batch_size": len(prompts),
                    "student_sampling_prompt_tokens": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_prompt_tokens")
                    ),
                    "student_sampling_output_tokens": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_output_tokens")
                    ),
                    "teacher_prefill_queue_wait_s": _sum_metric(chunk_prepare_profiles, "teacher_prefill_queue_wait_s"),
                    "teacher_prefill_s": _sum_metric(chunk_prepare_profiles, "teacher_prefill_s"),
                    "teacher_prefill_tokens": int(_sum_metric(chunk_prepare_profiles, "teacher_prefill_tokens")),
                    "teacher_prefill_forward_compute_s": _sum_metric(
                        chunk_prepare_profiles, "teacher_prefill_forward_compute_s"
                    ),
                    "teacher_hidden_cache_write_s": _sum_metric(chunk_prepare_profiles, "teacher_hidden_cache_write_s"),
                    "teacher_cache_save_s": _sum_metric(chunk_prepare_profiles, "teacher_cache_save_s"),
                    "forward_backward_enqueue_s": _sum_metric(chunk_fb_profiles, "forward_backward_enqueue_s"),
                    "forward_backward_wait_s": _sum_metric(chunk_fb_profiles, "forward_backward_wait_s"),
                    "forward_backward_roundtrip_s": _sum_metric(chunk_fb_profiles, "forward_backward_roundtrip_s"),
                    "trainer_forward_backward_s": _sum_metric(chunk_fb_profiles, "trainer_forward_backward_s"),
                    "trainer_forward_loss_s": _sum_metric(chunk_fb_profiles, "trainer_forward_loss_s"),
                    "trainer_backward_s": _sum_metric(chunk_fb_profiles, "trainer_backward_s"),
                    "trainer_opd_total_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_total_ms"),
                    "trainer_opd_prefetch_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_prefetch_ms"),
                    "trainer_opd_hidden_fetch_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_hidden_fetch_ms"),
                    "trainer_opd_head_prepare_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_head_prepare_ms"),
                    "trainer_opd_kl_compute_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_kl_compute_ms"),
                    "trainer_clear_gradients_ms": _sum_metric(chunk_fb_profiles, "trainer_clear_gradients_ms"),
                    "valid_tokens": total_valid_tokens,
                }
            )
            sampling_s = profile_row["student_sampling_s"]
            teacher_prefill_s = profile_row["teacher_prefill_s"]
            profile_row["student_sampling_output_tok_per_s"] = (
                profile_row["student_sampling_output_tokens"] / sampling_s if sampling_s > 0 else 0.0
            )
            profile_row["teacher_prefill_tok_per_s"] = (
                profile_row["teacher_prefill_tokens"] / teacher_prefill_s if teacher_prefill_s > 0 else 0.0
            )
            log.info(
                "Pipeline forward_backward step=%d loss=%.4f valid_tokens=%d prepare_wait=%.3fs "
                "sample_sum=%.3fs teacher_sum=%.3fs fb_sum=%.3fs",
                step,
                loss,
                total_valid_tokens,
                prepare_wait_s,
                profile_row["student_sampling_s"],
                profile_row["teacher_prefill_s"],
                profile_row["forward_backward_roundtrip_s"],
            )

        if args.skip_optim_step:
            profile_row.update(
                {
                    "optim_step_enqueue_s": 0.0,
                    "optim_step_wait_s": 0.0,
                    "optim_step_roundtrip_s": 0.0,
                    "sync_inference_weights_s": 0.0,
                    "post_sync_greedy_sample_s": 0.0,
                    "step_total_s": _elapsed(step_t0),
                }
            )
            profile_rows.append(profile_row)
            _write_profile_row(profile_output, profile_row)
            log.info("Skipping optim_step/sync because --skip-optim-step is set.")
            log.info("OPD throughput step=%d %s", step, json.dumps(profile_row, sort_keys=True))
            continue

        opt_t0 = time.perf_counter()
        r = requests.post(
            f"{args.train_url}/api/v1/optim_step",
            json={
                "model_id": "default",
                "adam_params": {"learning_rate": args.optim_learning_rate, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
                "gradient_clip": args.optim_gradient_clip,
            },
            timeout=120,
        )
        r.raise_for_status()
        profile_row["optim_step_enqueue_s"] = _elapsed(opt_t0)
        opt_wait_t0 = time.perf_counter()
        opt = _wait_for_future(args.train_url, r.json()["request_id"], timeout=300.0)
        profile_row["optim_step_wait_s"] = _elapsed(opt_wait_t0)
        profile_row["optim_step_roundtrip_s"] = _elapsed(opt_t0)
        log.info(
            "optim_step grad_norm=%.4f roundtrip=%.3fs",
            _scalar(opt["metrics"]["grad_norm"]),
            profile_row["optim_step_roundtrip_s"],
        )

        weight_version = f"{args.sync_weight_version_prefix}-step{step}" if args.sync_weight_version_prefix else None
        sync_payload = {
            "master_address": args.master_address,
            "master_port": args.sync_master_port,
            "buffer_size_mb": args.sync_buffer_size_mb,
            "flush_cache": args.sync_flush_cache,
            "pause_mode": args.sync_pause_mode,
            "weight_version": weight_version,
            "quantization": sync_quantization,
        }
        log.info(
            "Calling sync_inference_weights with master_address=%s buffer_size_mb=%d pause_mode=%s weight_version=%s",
            args.master_address,
            args.sync_buffer_size_mb,
            args.sync_pause_mode,
            weight_version,
        )
        sync_t0 = time.perf_counter()
        sync_response: Dict[str, Any] = {}
        try:
            r = requests.post(
                f"{args.train_url}/api/v1/sync_inference_weights",
                json=sync_payload,
                timeout=args.sync_timeout,
            )
            sync_status = r.status_code
            try:
                sync_response = r.json()
            except ValueError:
                sync_response = {"success": False, "message": r.text[:1000]}
        except requests.exceptions.RequestException as exc:
            sync_status = -1
            sync_response = {"success": False, "message": str(exc)}
            log.warning("sync_inference_weights HTTP error: %s", exc)
        profile_row["sync_inference_weights_s"] = _elapsed(sync_t0)
        profile_row["sync_http_status"] = sync_status
        _add_sync_response_profile(profile_row, sync_response)
        log.info(
            "sync_inference_weights returned status=%s success=%s transfer_time=%.3fs bytes=%s buckets=%s in %.1fs",
            sync_status,
            sync_response.get("success"),
            profile_row["sync_transfer_time_s"],
            profile_row["sync_total_bytes"],
            profile_row["sync_num_buckets"],
            profile_row["sync_inference_weights_s"],
        )
        if sync_status != 200 and not args.allow_sync_http_error:
            log.error("sync_inference_weights failed with status=%s response=%s", sync_status, sync_response)
            profile_row["step_total_s"] = _elapsed(step_t0)
            profile_rows.append(profile_row)
            _write_profile_row(profile_output, profile_row)
            return 6
        if sync_status == 200 and not sync_response.get("success", False):
            log.error("sync_inference_weights returned success=false: %s", sync_response)
            profile_row["step_total_s"] = _elapsed(step_t0)
            profile_rows.append(profile_row)
            _write_profile_row(profile_output, profile_row)
            return 7

        if weight_version:
            version_ok, version_error = _verify_student_weight_version(
                student_url,
                weight_version,
                profile_row,
                timeout=30.0,
            )
            if not version_ok and not args.allow_sync_version_mismatch:
                log.error("%s", version_error)
                profile_row["step_total_s"] = _elapsed(step_t0)
                profile_rows.append(profile_row)
                _write_profile_row(profile_output, profile_row)
                return 8
            if not version_ok:
                log.warning("%s; continuing because --allow-sync-version-mismatch is set.", version_error)

        post_sync_sample_t0 = time.perf_counter()
        sample_after_sync = _student_sample(
            student_url,
            prompts[0],
            args.max_new_tokens,
            temperature=0.0,
            timeout=args.request_timeout,
        )
        profile_row["post_sync_greedy_sample_s"] = _elapsed(post_sync_sample_t0)
        post_sync_rollouts.append(sample_after_sync)
        if sample_after_sync == initial_greedy_rollout and not args.allow_unchanged_sync:
            log.error(
                "Greedy rollout for prompt %s did not change at all by step %d "
                "(initial=%s, after_sync=%s); the broadcast did not take effect.",
                prompts[0],
                step,
                initial_greedy_rollout,
                sample_after_sync,
            )
            return 4
        if sample_after_sync == initial_greedy_rollout:
            log.warning(
                "Greedy rollout for prompt %s did not change by step %d; continuing because "
                "--allow-unchanged-sync is set.",
                prompts[0],
                step,
            )
        else:
            log.info(
                "Verified weights changed on student SGLang: initial=%s post_sync=%s",
                initial_greedy_rollout,
                sample_after_sync,
            )
        profile_row["step_total_s"] = _elapsed(step_t0)
        profile_rows.append(profile_row)
        _write_profile_row(profile_output, profile_row)
        log.info("OPD throughput step=%d %s", step, json.dumps(profile_row, sort_keys=True))

    if (
        post_sync_rollouts
        and all(r == initial_greedy_rollout for r in post_sync_rollouts)
        and not args.allow_unchanged_sync
    ):
        log.error(
            "Greedy rollout for prompt %s never diverged from initial across %d steps.\ninitial: %s\npost_sync: %s",
            prompts[0],
            len(post_sync_rollouts),
            initial_greedy_rollout,
            post_sync_rollouts,
        )
        return 5

    log.info("OPD pipeline validation succeeded.")
    log.info("loss_history=%s", loss_history)
    summary_keys = [
        "student_sampling_s",
        "teacher_prefill_s",
        "opd_pipeline_prepare_wait_s",
        "forward_backward_roundtrip_s",
        "trainer_forward_backward_s",
        "trainer_forward_loss_s",
        "trainer_backward_s",
        "optim_step_roundtrip_s",
        "sync_inference_weights_s",
        "sync_transfer_time_s",
        "step_total_s",
    ]
    all_summary = {key: _mean(profile_rows, key) for key in summary_keys}
    steady_rows = [row for row in profile_rows if not row.get("profile_warmup")]
    steady_summary = {key: _mean(steady_rows, key) for key in summary_keys}
    log.info(
        "Mean OPD throughput timings across %d steps: %s", len(profile_rows), json.dumps(all_summary, sort_keys=True)
    )
    log.info(
        "Steady-state OPD timings across %d steps (excluded %d warmup): %s",
        len(steady_rows),
        len(profile_rows) - len(steady_rows),
        json.dumps(steady_summary, sort_keys=True),
    )
    log.info("Per-step throughput JSONL: %s", profile_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
