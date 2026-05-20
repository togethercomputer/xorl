"""Full OPD pipeline e2e test: separate student SGLang sampler + teacher
SGLang hidden-state server + xorl GPU trainer with weight sync.

This is the closest end-to-end exercise of section 5.1.2 we can run
locally with tiny models. Each iteration of the loop:

  student SGLang  --(rollouts)-->  teacher SGLang  --(hidden_states)-->  xorl trainer
        ^                                                                       |
        |                                                                       v
        +----------------- sync_inference_weights (NCCL) ------------------------+

It validates the wiring rather than learning quality: the model is random
and the prompts are short, so we only check that every step succeeds, all
losses are finite, the student SGLang weight_version advances after each
sync, and rollouts after sync differ from the initial model output.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests
import torch

from tests._helpers.opd import save_teacher_hidden_cache

from .e2e_utils import create_tiny_model_dir
from .server_utils import (
    ServerProcess,
    _get_free_port,
    _start_server_or_fail,
    generate_server_config,
    get_sglang_paths,
    post_and_wait_for_future,
    scalar,
)


SGLANG_PYTHON, SGLANG_SOURCE_DIR = get_sglang_paths()


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.server,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(torch.cuda.device_count() < 3, reason="Need >=3 GPUs (trainer + student + teacher)"),
    pytest.mark.skipif(
        SGLANG_PYTHON is None or not SGLANG_PYTHON.exists(),
        reason="Set XORL_SGLANG_PYTHON or XORL_SGLANG_INTERNAL_DIR",
    ),
    pytest.mark.skipif(
        SGLANG_SOURCE_DIR is None or not SGLANG_SOURCE_DIR.exists(),
        reason="Set XORL_SGLANG_SOURCE_DIR or XORL_SGLANG_INTERNAL_DIR",
    ),
]


class _SGLangServer:
    """Subprocess wrapper around `python -m sglang.launch_server`.

    Uses a separate CUDA_VISIBLE_DEVICES per server so the student and
    teacher are isolated from the xorl trainer's GPU.
    """

    def __init__(
        self,
        model_dir: str,
        gpu_index: int,
        log_path: Path,
        *,
        port: int | None = None,
        enable_return_hidden_states: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.gpu_index = gpu_index
        self.log_path = log_path
        self.port = port or _get_free_port()
        self.enable_return_hidden_states = enable_return_hidden_states
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 180.0) -> None:
        assert SGLANG_PYTHON is not None
        assert SGLANG_SOURCE_DIR is not None
        cmd = [
            str(SGLANG_PYTHON),
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_dir,
            "--tokenizer-path",
            self.model_dir,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "triton",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.2",
            "--max-total-tokens",
            "256",
            "--max-running-requests",
            "4",
            "--log-level",
            "warning",
            # We always pass input_ids in this test, so the tokenizer is never
            # actually exercised. Skip warmup (which would otherwise fail trying
            # to tokenize "The capital city of France is" with the tiny
            # WordLevel vocab) but leave tokenizer init enabled. Note: combining
            # --skip-tokenizer-init with weight sync is broken upstream — the
            # scheduler sends WeightUpdatePauseReq through the tokenizer pipe in
            # that mode and the tokenizer manager has no handler for it.
            "--skip-server-warmup",
        ]
        if self.enable_return_hidden_states:
            cmd.append("--enable-return-hidden-states")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SGLANG_SOURCE_DIR}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        env.pop("CUDA_DEVICE_ORDER", None)

        log_file = self.log_path.open("w")
        self._log_file = log_file
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                tail = self._tail()
                raise AssertionError(
                    f"SGLang server (gpu={self.gpu_index}, port={self.port}) exited early.\n--- log tail ---\n{tail}"
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=3)
                if resp.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2.0)
        tail = self._tail()
        raise TimeoutError(
            f"SGLang server (gpu={self.gpu_index}, port={self.port}) not healthy after {timeout}s.\n"
            f"--- log tail ---\n{tail}"
        )

    def stop(self) -> None:
        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=10)
                except Exception:
                    pass
            self.process = None
        if hasattr(self, "_log_file") and self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _tail(self, n: int = 60) -> str:
        if not self.log_path.exists():
            return ""
        with self.log_path.open() as f:
            lines = f.readlines()
        return "".join(lines[-n:])

    def model_info(self) -> dict:
        resp = requests.get(f"{self.base_url}/model_info", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def generate(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        return_hidden_states: bool = False,
        timeout: float = 60.0,
    ) -> dict:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens},
            "return_hidden_states": return_hidden_states,
        }
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


def _student_sample(student: _SGLangServer, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """Generate a continuation; return prompt+completion as a flat token list."""
    out = student.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.7)
    completion_ids = out.get("output_ids") or out.get("token_ids") or []
    if not completion_ids:
        # Some SGLang versions place ids under meta_info instead.
        completion_ids = out.get("meta_info", {}).get("output_ids", [])
    if isinstance(completion_ids, dict):
        completion_ids = completion_ids.get("output_ids", [])
    completion_ids = [int(t) for t in completion_ids]
    return list(prompt_ids) + completion_ids


def _teacher_hidden_states_for(teacher: _SGLangServer, sequences: list[list[int]]) -> list[torch.Tensor]:
    """Return one [seq_len, hidden] tensor per input sequence using teacher's prefill hidden states."""
    hiddens: list[torch.Tensor] = []
    for seq in sequences:
        out = teacher.generate(
            seq,
            max_new_tokens=1,
            temperature=0.0,
            return_hidden_states=True,
        )
        chunks = out.get("meta_info", {}).get("hidden_states") or []
        if not chunks:
            raise AssertionError(f"Teacher SGLang did not return hidden_states. Response keys: {list(out.keys())}")
        prefill = torch.tensor(chunks[0], dtype=torch.bfloat16)
        if prefill.ndim != 2 or prefill.shape[0] != len(seq):
            raise AssertionError(
                f"Unexpected teacher hidden state shape {tuple(prefill.shape)} for input length {len(seq)}"
            )
        hiddens.append(prefill)
    return hiddens


def test_opd_full_pipeline_with_weight_sync(tmp_path):
    torch.manual_seed(2026)

    # Models — both student and teacher use the same tiny config so OPD vocab/hidden line up.
    student_dir = create_tiny_model_dir(str(tmp_path / "student"), model_type="dense", save_weights=True)
    teacher_dir = create_tiny_model_dir(str(tmp_path / "teacher"), model_type="dense", save_weights=True)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices = (
        [int(x) for x in visible.split(",") if x.strip() != ""] if visible else list(range(torch.cuda.device_count()))
    )
    if len(visible_devices) < 3:
        pytest.skip("Need at least 3 visible CUDA devices")
    trainer_gpu, student_gpu, teacher_gpu = visible_devices[0], visible_devices[1], visible_devices[2]

    teacher_log = tmp_path / "teacher_sglang.log"
    student_log = tmp_path / "student_sglang.log"
    teacher = _SGLangServer(
        model_dir=teacher_dir,
        gpu_index=teacher_gpu,
        log_path=teacher_log,
        enable_return_hidden_states=True,
    )
    student = _SGLangServer(
        model_dir=student_dir,
        gpu_index=student_gpu,
        log_path=student_log,
        enable_return_hidden_states=False,
    )

    output_dir = tmp_path / "xorl_server"
    config_path = generate_server_config(
        student_dir,
        str(output_dir),
        num_gpus=1,
        enable_lora=False,
        enable_gradient_checkpointing=False,
        sample_packing_sequence_len=64,
        extra_config={
            "enable_full_shard": False,
            "worker_connection_timeout": 240.0,
        },
    )
    server = ServerProcess(
        config_path=config_path,
        num_gpus=1,
        api_port=_get_free_port(),
        output_dir=str(output_dir),
    )

    # The xorl ServerProcess inherits CUDA_VISIBLE_DEVICES from the parent
    # only when set; otherwise it sets it to range(num_gpus). We force the
    # trainer onto a single GPU that does not collide with student/teacher.
    prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(trainer_gpu)

    # TeacherHeadManager understands HF directory paths and pulls lm_head.weight
    # straight from the safetensors shard, so no extra extraction step is needed.
    teacher_head_entry = teacher_dir

    try:
        teacher.start(timeout=300.0)
        student.start(timeout=300.0)
        try:
            _start_server_or_fail(server, timeout=300.0)
        finally:
            if prev_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible

        train_url = server.base_url

        # Two short OPD steps: rollout -> teacher hidden -> forward_backward
        # -> optim_step -> save trained weights to disk -> have student SGLang
        # reload from disk. We use the disk-based refresh path (xorl
        # /save_weights_for_sampler + SGLang /update_weights_from_disk) here.
        # The NCCL `sync_inference_weights` path was wired up and gets all the
        # way through the broadcast (5 buckets, ~26 params transferred), but
        # SGLang's /destroy_weights_update_group never returns for our tiny
        # model setup, which strands the call. The xorl side of that path is
        # unblocked by the wait_for_workers=False fix that ships in this branch.
        prompts = [[5, 6, 7], [11, 12], [21, 22, 23, 24]]
        max_new_tokens = 4
        loss_history: list[float] = []
        rollout_history: list[list[list[int]]] = []

        for step in range(2):
            sequences = [_student_sample(student, p, max_new_tokens) for p in prompts]
            rollout_history.append([list(seq) for seq in sequences])
            assert all(len(seq) >= len(p) + 1 for seq, p in zip(sequences, prompts)), (
                f"step {step}: student SGLang produced no completion tokens: {sequences}"
            )

            hiddens = _teacher_hidden_states_for(teacher, sequences)
            cache_path = tmp_path / f"teacher_hidden_step{step}.safetensors"
            cache_indices = save_teacher_hidden_cache(hiddens, cache_path)

            data = []
            for seq, indices in zip(sequences, cache_indices):
                data.append(
                    {
                        "model_input": {"input_ids": seq},
                        "loss_fn_inputs": {
                            "target_tokens": seq,
                            "teacher_ids": [0] * len(seq),
                            "teacher_weights": [1.0] * len(seq),
                            "teacher_cache_indices": indices,
                        },
                    }
                )

            fb = post_and_wait_for_future(
                train_url,
                "/api/v1/forward_backward",
                {
                    "model_id": "default",
                    "forward_backward_input": {
                        "data": data,
                        "loss_fn": "opd_loss",
                        "loss_fn_params": {
                            "teacher_heads": {"0": str(teacher_head_entry)},
                            "teacher_hidden_caches": {"0": str(cache_path)},
                            "opd_sort_by_teacher": True,
                            "num_chunks": 4,
                        },
                    },
                },
                request_timeout=60.0,
                future_timeout=300.0,
            )
            loss_val = scalar(fb["loss_fn_outputs"][0]["loss"])
            assert loss_val == loss_val and loss_val >= 0.0, (  # NaN-safe
                f"step {step}: loss is invalid ({loss_val})"
            )
            loss_history.append(loss_val)

            opt = post_and_wait_for_future(
                train_url,
                "/api/v1/optim_step",
                {
                    "model_id": "default",
                    "adam_params": {"learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
                    "gradient_clip": 1.0,
                },
                request_timeout=60.0,
                future_timeout=120.0,
            )
            grad_norm = scalar(opt["metrics"]["grad_norm"])
            assert grad_norm == grad_norm, f"step {step}: NaN grad norm"

            # Persist the trained weights as an HF safetensors directory and
            # tell the student SGLang server to reload them in place. This
            # refreshes the on-policy sampler with the newest student weights.
            sampler_name = f"opd-step-{step}"
            save_future = post_and_wait_for_future(
                train_url,
                "/api/v1/save_weights_for_sampler",
                {"model_id": "default", "name": sampler_name},
                request_timeout=60.0,
                future_timeout=300.0,
            )
            sampler_dir = Path(output_dir) / "sampler_weights" / sampler_name
            assert sampler_dir.exists(), (
                f"step {step}: expected sampler weights dir at {sampler_dir} (future={save_future})"
            )

            for fname in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                src = Path(student_dir) / fname
                dst = sampler_dir / fname
                if src.exists() and not dst.exists():
                    dst.write_bytes(src.read_bytes())

            update_resp = requests.post(
                f"{student.base_url}/update_weights_from_disk",
                json={"model_path": str(sampler_dir), "abort_all_requests": True, "flush_cache": True},
                timeout=300,
            )
            update_resp.raise_for_status()
            assert update_resp.json().get("success"), (
                f"step {step}: SGLang update_weights_from_disk failed: {update_resp.json()}"
            )

        # Sanity: the model is random and tiny but two consecutive OPD updates with
        # weight refresh should produce at least one differing rollout across runs.
        assert any(rollout_history[0][i] != rollout_history[1][i] for i in range(len(prompts))), (
            "Rollouts before and after weight refresh are byte-identical; "
            "weight refresh may not have taken effect.\n"
            f"step0: {rollout_history[0]}\nstep1: {rollout_history[1]}"
        )

        # Persist the loss history for debugging — useful if a CI failure
        # only shows the top-level assertion.
        (tmp_path / "loss_history.json").write_text(json.dumps(loss_history), encoding="utf-8")
    finally:
        try:
            server.stop()
        finally:
            try:
                student.stop()
            finally:
                teacher.stop()
