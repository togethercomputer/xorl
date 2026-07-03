from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from xorl.server.backend import Backend
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.protocol.api_orchestrator import OrchestratorRequest, OutputType
from xorl.server.protocol.operations import ModelPassData, OptimStepData, SyncWeightsData
from xorl.server.runner.runner_dispatcher import RunnerDispatcher
from xorl.server.weight_sync.backends.sparse_delta import SparseDeltaTransportBackend
from xorl.server.weight_sync.handler import WeightSyncHandler


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _add_import_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if (candidate / "python" / "sglang").is_dir():
        candidate = candidate / "python"
    if not candidate.exists():
        return None
    resolved = str(candidate.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return resolved


def _require_delta_encoding() -> str | None:
    delta_path = _add_import_path(os.environ.get("XORL_DELTA_ENCODING_PATH")) or _add_import_path(
        "/home/apanda/delta-encoding"
    )
    try:
        importlib.import_module("delta_encoding.encoding.compression")
        importlib.import_module("delta_encoding.encoding.packed")
    except Exception as exc:
        pytest.skip(f"delta-encoding is not importable: {exc}")
    return delta_path


def _require_sglang_apply_sparse_delta_file() -> Callable[..., object]:
    _add_import_path(os.environ.get("XORL_SGLANG_PATH")) or _add_import_path("/home/apanda/xorl-sglang-internal")
    try:
        module = importlib.import_module("sglang.srt.weight_sync.sparse_delta")
    except Exception as exc:
        pytest.skip(f"xorl-sglang-internal sparse-delta receiver is not importable: {exc}")
    return module.apply_sparse_delta_file


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            q_lora_rank=None,
            layer_types=[],
        )
        self.proj = nn.Linear(4, 3, bias=False)
        self.norm = nn.Parameter(torch.empty(3, dtype=torch.bfloat16))
        self.to(dtype=torch.bfloat16)
        with torch.no_grad():
            self.proj.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16) / 10)
            self.norm.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16))

    def unshard(self) -> None:
        return None

    def reshard(self) -> None:
        return None


class _TinyTrainer:
    def __init__(self) -> None:
        self.model = _TinyTrainModel()
        self.local_rank = 0
        self.optimizer = None
        self._allocator_dirty = False
        self.forward_backward_calls = 0
        self.step = 0

    def forward_backward(
        self,
        batches: list[dict[str, Any]],
        loss_fn: str,
        loss_fn_params: dict[str, Any] | None,
        *,
        model_id: str | None,
        routed_experts: list[Any] | None,
        routed_expert_logits: list[Any] | None,
    ) -> dict[str, Any]:
        del loss_fn_params, routed_experts, routed_expert_logits
        assert loss_fn == "causallm_loss"
        assert model_id == "tiny-trainer"
        self.forward_backward_calls += 1
        valid_tokens = sum(len(batch["input_ids"][0]) for batch in batches)
        return {
            "total_loss": 0.25,
            "global_valid_tokens": valid_tokens,
            "execution_time": 0.0,
        }

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise AssertionError("forward should not be called in this sparse-delta E2E")

    def optim_step(
        self,
        *,
        gradient_clip: float | None,
        lr: float,
        model_id: str | None,
        sparse_delta_capture: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del gradient_clip, sparse_delta_capture
        assert model_id == "tiny-trainer"
        assert self.forward_backward_calls == 1
        self.step += 1
        with torch.no_grad():
            self.model.proj.weight.reshape(-1)[1] = torch.tensor(-2.0, dtype=torch.bfloat16)
            self.model.proj.weight.reshape(-1)[10] = torch.tensor(7.5, dtype=torch.bfloat16)
            self.model.norm.reshape(-1)[2] = torch.tensor(9.0, dtype=torch.bfloat16)
        return {
            "grad_norm": 1.0,
            "step": self.step,
            "learning_rate": lr,
            "execution_time": 0.0,
        }


class _RunnerDispatcherBackend(Backend):
    def __init__(self, trainer: _TinyTrainer) -> None:
        self._ready = False
        self.trainer = trainer
        self.dispatcher = object.__new__(RunnerDispatcher)
        self.dispatcher.rank = 0
        self.dispatcher.world_size = 1
        self.dispatcher.device = "cpu"
        self.dispatcher.trainer = trainer
        self.dispatcher._adapter_coordinator = SimpleNamespace(auto_load_if_evicted=lambda _model_id: (False, None))
        self.dispatcher._weight_sync_handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)
        self.dispatcher._weight_sync_handler._get_fsdp_modules = lambda model: (model, [])

    async def start(self) -> None:
        self._ready = True

    async def stop(self) -> None:
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    async def forward_backward(
        self,
        batches: list[dict[str, Any]],
        loss_fn: str = "causallm_loss",
        loss_fn_params: dict[str, Any] | None = None,
        model_id: str | None = None,
        routed_experts: list[Any] | None = None,
        routed_expert_logits: list[Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_id
        return self.dispatcher._execute_compute(
            batches,
            loss_fn,
            loss_fn_params,
            routed_experts,
            with_backward=True,
            model_id=model_id,
            routed_expert_logits=routed_expert_logits,
        )

    async def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise AssertionError("forward should not be called in this sparse-delta E2E")

    async def optim_step(
        self,
        lr: float,
        gradient_clip: float | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        eps: float | None = None,
        model_id: str | None = None,
        sparse_delta_capture: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_id
        return await self.dispatcher._handle_optim_step(
            {
                "payload": OptimStepData(
                    lr=lr,
                    gradient_clip=gradient_clip,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    model_id=model_id,
                    sparse_delta_capture=sparse_delta_capture,
                )
            }
        )

    async def sync_inference_weights(
        self,
        endpoints: list[dict[str, Any]],
        master_address: str = "localhost",
        master_port: int = 0,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del request_id
        return await self.dispatcher._handle_sync_inference_weights(
            {
                "payload": SyncWeightsData(
                    endpoints=endpoints,
                    master_address=master_address,
                    master_port=master_port,
                    **kwargs,
                )
            }
        )

    async def save_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise NotImplementedError

    async def load_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise NotImplementedError

    async def save_lora_only(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise NotImplementedError

    async def save_full_weights(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise NotImplementedError

    async def sleep(self, request_id: str | None = None) -> dict[str, Any]:
        del request_id
        raise NotImplementedError

    async def wake_up(self, request_id: str | None = None) -> dict[str, Any]:
        del request_id
        raise NotImplementedError

    async def register_session(
        self,
        model_id: str = "default",
        session_spec: dict[str, Any] | None = None,
        materialize: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del model_id, session_spec, materialize, request_id
        raise NotImplementedError

    async def register_adapter(self, model_id: str = "default", lr: float = 1e-5, request_id: str | None = None):
        del model_id, lr, request_id
        raise NotImplementedError

    async def save_adapter_state(
        self,
        model_id: str = "default",
        path: str | None = None,
        save_optimizer: bool = True,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del model_id, path, save_optimizer, request_id
        raise NotImplementedError

    async def load_adapter_state(
        self,
        model_id: str = "default",
        path: str | None = None,
        load_optimizer: bool = True,
        lr: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del model_id, path, load_optimizer, lr, request_id
        raise NotImplementedError

    async def get_adapter_info(self, request_id: str | None = None) -> dict[str, Any]:
        del request_id
        raise NotImplementedError

    async def kill_session(
        self,
        model_id: str = "default",
        save_checkpoint: bool = True,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del model_id, save_checkpoint, request_id
        raise NotImplementedError

    async def health_check(self, request_id: str | None = None) -> dict[str, Any]:
        del request_id
        return {"status": "healthy"}


class _FakeEndpointManager:
    def __init__(self, endpoints: list[dict[str, Any]]) -> None:
        self.endpoints = endpoints
        self.events: list[Any] = []

    def health_check(self) -> None:
        self.events.append("health")

    def pause(self, mode: str) -> tuple[list[dict[str, Any]], bool]:
        self.events.append(("pause", mode))
        return [{"success": True}], True

    def resume(self) -> list[dict[str, Any]]:
        self.events.append("resume")
        return [{"success": True}]


def _assert_success(output) -> dict[str, Any]:
    assert output.finished is True
    assert output.error is None
    assert output.output_type != OutputType.ERROR
    assert output.outputs
    result = output.outputs[0]
    assert result.get("success", True) is True
    return result


async def _run_sparse_delta_trainer_to_sglang_e2e(
    *,
    processor: RequestProcessor,
    tmp_path: Path,
    delta_encoding_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    await processor.start()
    try:
        base_sync = await processor.execute_sync_inference_weights(
            OrchestratorRequest(
                operation="sync_inference_weights",
                payload=SyncWeightsData(
                    endpoints=[{"host": "sglang.local", "port": 30000, "world_size": 1}],
                    group_name="trainer-sglang-e2e",
                    buffer_size_mb=1,
                    sync_method="sparse_delta",
                    flush_cache=True,
                    weight_version="tiny-v0",
                    sparse_delta_config={
                        "output_dir": str(tmp_path / "packed"),
                        "keep_files": True,
                        "baseline_scope": f"trainer-sglang-e2e-{tmp_path.name}",
                        "delta_encoding_path": delta_encoding_path,
                    },
                ),
            )
        )
        base_result = _assert_success(base_sync)

        fb_output = await processor.execute_forward_backward(
            OrchestratorRequest(
                operation="forward_backward",
                payload=ModelPassData(
                    data=[
                        {
                            "input_ids": [0, 1, 2, 3],
                            "target_tokens": [1, 2, 3, 4],
                        }
                    ],
                    loss_fn="causallm_loss",
                    model_id="tiny-trainer",
                ),
            )
        )
        fb_result = _assert_success(fb_output)

        optim_output = await processor.execute_optim_step(
            OrchestratorRequest(
                operation="optim_step",
                payload=OptimStepData(
                    lr=1e-3,
                    gradient_clip=1.0,
                    model_id="tiny-trainer",
                ),
            )
        )
        optim_result = _assert_success(optim_output)

        delta_sync = await processor.execute_sync_inference_weights(
            OrchestratorRequest(
                operation="sync_inference_weights",
                payload=SyncWeightsData(
                    endpoints=[{"host": "sglang.local", "port": 30000, "world_size": 1}],
                    group_name="trainer-sglang-e2e",
                    buffer_size_mb=1,
                    sync_method="sparse_delta",
                    flush_cache=True,
                    weight_version="tiny-v1",
                    sparse_delta_config={
                        "output_dir": str(tmp_path / "packed"),
                        "keep_files": True,
                        "baseline_scope": f"trainer-sglang-e2e-{tmp_path.name}",
                        "delta_encoding_path": delta_encoding_path,
                    },
                ),
            )
        )
        delta_result = _assert_success(delta_sync)
        return base_result, {"forward_backward": fb_result, "optim_step": optim_result}, delta_result
    finally:
        await processor.stop()


def test_request_processor_trainer_sparse_delta_sync_applies_with_sglang_receiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_encoding_path = _require_delta_encoding()
    apply_sparse_delta_file = _require_sglang_apply_sparse_delta_file()
    SparseDeltaTransportBackend.clear_cached_baselines()

    trainer = _TinyTrainer()
    receiver = _TinyTrainModel()
    receiver.load_state_dict(trainer.model.state_dict())
    backend = _RunnerDispatcherBackend(trainer)
    processor = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=8,
        enable_packing=True,
        pad_to_multiple_of=1,
        cp_size=1,
    )

    post_payloads: list[dict[str, Any]] = []

    def fake_sparse_delta_post(
        url: str, *, json: dict[str, Any], timeout: float | None = None, **_: Any
    ) -> _FakeResponse:
        del timeout
        assert url == "http://sglang.local:30000/update_weights_from_sparse_delta"
        post_payloads.append(dict(json))
        paths = list(json["delta_paths"])
        sha256s = list(json["delta_sha256s"])
        assert len(paths) == len(sha256s) == 1
        for path, sha256 in zip(paths, sha256s):
            apply_sparse_delta_file(receiver, path, expected_sha256=sha256, validate_only=True)
        apply_stats = []
        for path, sha256 in zip(paths, sha256s):
            stats = apply_sparse_delta_file(receiver, path, expected_sha256=sha256)
            apply_stats.append(
                {
                    "total_nnz": stats.total_nnz,
                    "applied_nnz": stats.applied_nnz,
                    "direct_tensors": stats.direct_tensors,
                }
            )
        return _FakeResponse({"success": True, "message": "ok", "apply_stats": apply_stats})

    monkeypatch.setattr("xorl.server.weight_sync.handler.EndpointManager", _FakeEndpointManager)
    monkeypatch.setattr("xorl.server.weight_sync.backends.sparse_delta.requests.post", fake_sparse_delta_post)
    monkeypatch.setattr(
        "xorl.server.weight_sync.handler.get_parallel_state",
        lambda: SimpleNamespace(
            ep_enabled=False,
            ep_size=1,
            pp_enabled=False,
            pp_rank=0,
            pp_size=1,
            dp_shard_rank=0,
        ),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)

    try:
        base_result, step_results, delta_result = asyncio.run(
            _run_sparse_delta_trainer_to_sglang_e2e(
                processor=processor,
                tmp_path=tmp_path,
                delta_encoding_path=delta_encoding_path,
            )
        )
    finally:
        SparseDeltaTransportBackend.clear_cached_baselines()

    assert step_results["forward_backward"]["valid_tokens"] == 4
    assert step_results["optim_step"]["step"] == 1
    assert [payload["weight_version"] for payload in post_payloads] == ["tiny-v0", "tiny-v1"]
    assert all(payload["delta_sha256s"] for payload in post_payloads)

    base_timing = base_result["timing_breakdown"]
    delta_timing = delta_result["timing_breakdown"]
    assert base_timing["sparse_delta_total_changed_values"] == 15.0
    assert delta_timing["sparse_delta_total_changed_values"] == 3.0
    assert delta_timing["sparse_delta_changed_density"] == pytest.approx(3.0 / 15.0)
    assert delta_timing["sparse_delta_posted_files"] == 1.0

    for name, param in trainer.model.named_parameters():
        torch.testing.assert_close(dict(receiver.named_parameters())[name], param, rtol=0, atol=0)
