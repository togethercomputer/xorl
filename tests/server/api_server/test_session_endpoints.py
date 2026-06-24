"""Focused tests for session-spec API endpoint behavior."""

from __future__ import annotations

import json
import os
import types
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from xorl.server.api_server.api_types import (
    CreateModelRequest,
    CreateSessionRequest,
    KillSessionRequest,
    OptimizerConfigRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsResponse,
    UnloadModelRequest,
    WeightsInfoRequest,
)
from xorl.server.api_server.endpoints import (
    _canon_base_model,
    create_model_endpoint,
    create_session_endpoint,
    kill_session_endpoint,
    unload_model_endpoint,
    weights_info_endpoint,
)
from xorl.server.api_server.server import APIServer
from xorl.server.session_spec import build_default_session_spec, load_session_spec_from_checkpoint, write_session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server, pytest.mark.anyio]


class _ImmediateFutureStore:
    def __init__(self) -> None:
        self.last_result = None
        self.deleted_models = []

    async def create(self, *, model_id, request_type, process_fn, request_data, ttl=None):
        self.last_result = await process_fn(request_data)
        return "future-test-1"

    async def delete_by_model(self, model_id):
        self.deleted_models.append(model_id)
        return 0


class _FakeOrchestratorClient:
    def __init__(self) -> None:
        self.last_request = None
        self.requests = []

    async def send_request(self, request):
        self.last_request = request
        self.requests.append(request)
        return request


def _build_wait_for_response():
    async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
        if response_future.operation == "register_session":
            return types.SimpleNamespace(
                outputs={"result": {"registered": True, "model_id": response_future.payload.model_id}}
            )
        if response_future.operation == "save_state":
            os.makedirs(response_future.payload.checkpoint_path, exist_ok=True)
            return types.SimpleNamespace(outputs=[{"checkpoint_path": response_future.payload.checkpoint_path}])
        if response_future.operation == "save_lora_only":
            os.makedirs(response_future.payload.lora_path, exist_ok=True)
            return types.SimpleNamespace(outputs=[{"lora_path": response_future.payload.lora_path}])
        if response_future.operation == "save_full_weights":
            os.makedirs(response_future.payload.output_path, exist_ok=True)
            return types.SimpleNamespace(outputs=[{"output_path": response_future.payload.output_path}])
        if response_future.operation == "kill_session":
            return types.SimpleNamespace(
                outputs=[
                    {
                        "success": True,
                        "message": f"killed {response_future.payload.model_id}",
                        "checkpoint_path": None,
                    }
                ]
            )
        raise AssertionError(f"Unexpected operation: {response_future.operation}")

    return _wait_for_response


def _build_server(tmp_path):
    train_config = {
        "optimizer": "adamw",
        "lr": 1e-5,
        "weight_decay": 0.01,
        "optimizer_dtype": "bf16",
        "optimizer_kwargs": {},
    }
    lora_config = {
        "enable_lora": True,
        "lora_rank": 8,
        "max_lora_rank": 16,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "o_proj"],
    }
    default_session_spec = build_default_session_spec(
        base_model="Qwen/Qwen3-8B",
        train_config=train_config,
        lora_config=lora_config,
    )
    server = APIServer(
        engine_input_addr="tcp://127.0.0.1:17000",
        engine_output_addr="tcp://127.0.0.1:17001",
        output_dir=str(tmp_path),
        base_model="Qwen/Qwen3-8B",
        default_session_spec=default_session_spec,
        server_lora_config=lora_config,
        max_lora_rank=16,
    )
    server.future_store = _ImmediateFutureStore()
    server.orchestrator_client = _FakeOrchestratorClient()
    server._running = True
    server._wait_for_response = types.MethodType(_build_wait_for_response(), server)
    return server


def _build_full_weight_server(tmp_path):
    server = APIServer(
        engine_input_addr="tcp://127.0.0.1:17000",
        engine_output_addr="tcp://127.0.0.1:17001",
        output_dir=str(tmp_path),
        base_model="Qwen/Qwen3-8B",
        default_session_spec=None,
    )
    server.future_store = _ImmediateFutureStore()
    server.orchestrator_client = _FakeOrchestratorClient()
    server._running = True
    server._wait_for_response = types.MethodType(_build_wait_for_response(), server)
    return server


async def test_create_model_endpoint_registers_normalized_session_spec(tmp_path):
    server = _build_server(tmp_path)

    response = await create_model_endpoint(
        CreateModelRequest(
            model_id="session-a",
            base_model="Qwen/Qwen3-8B",
            lora_config={"rank": 4, "alpha": 12},
            optimizer_config=OptimizerConfigRequest(type="signsgd", learning_rate=2e-4),
        ),
        server=server,
    )

    assert response.request_id == "future-test-1"
    assert "session-a" in server.model_configs
    assert server.model_configs["session-a"]["lora_config"]["lora_rank"] == 4
    assert server.model_configs["session-a"]["lora_config"]["lora_alpha"] == 12
    assert server.model_configs["session-a"]["optimizer_config"]["type"] == "signsgd"
    register_requests = [
        request for request in server.orchestrator_client.requests if request.operation == "register_session"
    ]
    assert len(register_requests) == 1
    assert register_requests[0].payload.session_spec["lora_config"]["lora_rank"] == 4


async def test_create_session_endpoint_registers_lora_session_with_workers(tmp_path):
    server = _build_server(tmp_path)

    response = await create_session_endpoint(
        CreateSessionRequest(session_id="session-a", lora_config={"rank": 4, "alpha": 12}),
        server=server,
    )

    assert response.session_id == "session-a"
    assert "session-a" in server.registered_model_ids
    assert server.model_configs["session-a"]["lora_config"] == {"lora_rank": 4, "lora_alpha": 12}
    register_requests = [
        request for request in server.orchestrator_client.requests if request.operation == "register_session"
    ]
    assert len(register_requests) == 1
    assert register_requests[0].payload.model_id == "session-a"
    assert register_requests[0].payload.materialize is True
    assert register_requests[0].payload.session_spec["lora_config"]["lora_rank"] == 4


async def test_create_session_endpoint_allows_rank_only_lora_override(tmp_path):
    server = _build_server(tmp_path)

    response = await create_session_endpoint(
        CreateSessionRequest(session_id="session-rank-only", lora_config={"rank": 4}),
        server=server,
    )

    assert response.session_id == "session-rank-only"
    assert server.model_configs["session-rank-only"]["lora_config"] == {"lora_rank": 4, "lora_alpha": 16}
    register_requests = [
        request for request in server.orchestrator_client.requests if request.operation == "register_session"
    ]
    assert len(register_requests) == 1
    assert register_requests[0].payload.session_spec["lora_config"] == {"lora_rank": 4, "lora_alpha": 16}


async def test_create_session_endpoint_refreshes_existing_custom_lora_session(tmp_path):
    server = _build_server(tmp_path)

    await create_session_endpoint(
        CreateSessionRequest(session_id="session-a", lora_config={"rank": 4, "alpha": 12}),
        server=server,
    )

    response = await create_session_endpoint(
        CreateSessionRequest(session_id="session-a"),
        server=server,
    )

    assert response.session_id == "session-a"
    assert response.warning_message == "Session 'session-a' already existed; refreshed activity timestamp."
    assert server.model_configs["session-a"]["lora_config"] == {"lora_rank": 4, "lora_alpha": 12}
    register_requests = [
        request for request in server.orchestrator_client.requests if request.operation == "register_session"
    ]
    assert len(register_requests) == 1


async def test_create_model_endpoint_rejects_conflicting_recreate(tmp_path):
    server = _build_server(tmp_path)

    await create_model_endpoint(
        CreateModelRequest(
            model_id="session-a",
            base_model="Qwen/Qwen3-8B",
            lora_config={"rank": 4},
        ),
        server=server,
    )

    with pytest.raises(ValueError, match="already exists with a different session spec"):
        await create_model_endpoint(
            CreateModelRequest(
                model_id="session-a",
                base_model="Qwen/Qwen3-8B",
                lora_config={"rank": 6},
            ),
            server=server,
        )


async def test_create_model_endpoint_propagates_register_session_failure(tmp_path):
    server = _build_server(tmp_path)

    async def _fail_wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
        raise HTTPException(status_code=500, detail="Engine error: Cross-rank error: rank 1: register failed")

    server._wait_for_response = types.MethodType(_fail_wait_for_response, server)

    with pytest.raises(HTTPException, match="Cross-rank error"):
        await create_model_endpoint(
            CreateModelRequest(
                model_id="session-b",
                base_model="Qwen/Qwen3-8B",
                lora_config={"rank": 4},
            ),
            server=server,
        )

    assert "session-b" not in server.model_configs
    assert "session-b" not in server.registered_model_ids


async def test_create_model_endpoint_ensures_reserved_checkpoint_for_default_session(tmp_path):
    server = _build_server(tmp_path)
    server.save_weights = AsyncMock(return_value=SaveWeightsResponse(path="xorl://default/weights/000000"))

    response = await create_model_endpoint(
        CreateModelRequest(
            model_id="default",
            base_model="Qwen/Qwen3-8B",
        ),
        server=server,
    )

    assert response.request_id == "future-test-1"
    assert server.save_weights.await_count == 1
    save_request = server.save_weights.await_args.args[0]
    assert save_request.model_id == "default"
    assert save_request.path == "000000"


async def test_create_model_endpoint_saves_reserved_checkpoint_per_session(tmp_path):
    server = _build_server(tmp_path)
    server.save_weights = AsyncMock(
        side_effect=[
            SaveWeightsResponse(path="xorl://session-a/weights/000000"),
            SaveWeightsResponse(path="xorl://session-b/weights/000000"),
        ]
    )

    await create_model_endpoint(
        CreateModelRequest(
            model_id="session-a",
            base_model="Qwen/Qwen3-8B",
            lora_config={"rank": 4},
        ),
        server=server,
    )
    await create_model_endpoint(
        CreateModelRequest(
            model_id="session-b",
            base_model="Qwen/Qwen3-8B",
            lora_config={"rank": 6},
        ),
        server=server,
    )

    assert server.save_weights.await_count == 2
    assert [call.args[0].model_id for call in server.save_weights.await_args_list] == ["session-a", "session-b"]
    assert [call.args[0].path for call in server.save_weights.await_args_list] == ["000000", "000000"]


async def test_create_model_endpoint_overwrites_stale_reserved_checkpoint_for_recreated_session(tmp_path):
    server = _build_server(tmp_path)
    checkpoint_dir = tmp_path / "weights" / "session-a" / server.RESERVED_CHECKPOINT_NAME
    checkpoint_dir.mkdir(parents=True)
    server.save_weights = AsyncMock(return_value=SaveWeightsResponse(path="xorl://session-a/weights/000000"))

    await create_model_endpoint(
        CreateModelRequest(
            model_id="session-a",
            base_model="Qwen/Qwen3-8B",
            lora_config={"rank": 4},
        ),
        server=server,
    )

    assert server.save_weights.await_count == 1
    save_request = server.save_weights.await_args.args[0]
    assert save_request.model_id == "session-a"
    assert save_request.path == "000000"


async def test_create_model_endpoint_preserves_reserved_checkpoint_for_existing_session(tmp_path):
    server = _build_server(tmp_path)
    server.save_weights = AsyncMock(return_value=SaveWeightsResponse(path="xorl://session-a/weights/000000"))

    request = CreateModelRequest(
        model_id="session-a",
        base_model="Qwen/Qwen3-8B",
        lora_config={"rank": 4},
    )
    await create_model_endpoint(request, server=server)
    (tmp_path / "weights" / "session-a" / server.RESERVED_CHECKPOINT_NAME).mkdir(parents=True)

    await create_model_endpoint(request, server=server)

    assert server.save_weights.await_count == 1


async def test_save_weights_for_sampler_uses_lora_only_for_normalized_session(tmp_path):
    server = _build_server(tmp_path)

    response = await server.save_weights_for_sampler(
        SaveWeightsForSamplerRequest(model_id="default", name="sampler-a"),
    )

    assert response.path == "xorl://default/sampler_weights/sampler-a"
    assert server.orchestrator_client.requests[-1].operation == "save_lora_only"
    assert server.orchestrator_client.requests[-1].payload.model_id == "default"


async def test_create_model_endpoint_rejects_full_weight_multitenancy(tmp_path):
    server = _build_full_weight_server(tmp_path)

    with pytest.raises(ValueError, match="multi-tenancy is not supported yet"):
        await create_model_endpoint(
            CreateModelRequest(
                model_id="session-a",
                base_model="Qwen/Qwen3-8B",
            ),
            server=server,
        )


async def test_create_model_endpoint_allows_default_full_weight_session(tmp_path):
    server = _build_full_weight_server(tmp_path)

    response = await create_model_endpoint(
        CreateModelRequest(
            model_id="default",
            base_model="Qwen/Qwen3-8B",
        ),
        server=server,
    )

    assert response.request_id == "future-test-1"
    assert server.model_configs["default"] == {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": False,
    }
    register_requests = [
        request for request in server.orchestrator_client.requests if request.operation == "register_session"
    ]
    assert len(register_requests) == 1
    assert register_requests[0].payload.materialize is False


async def test_create_model_endpoint_rejects_full_weight_overrides(tmp_path):
    server = _build_full_weight_server(tmp_path)

    with pytest.raises(ValueError, match="Per-session LoRA or optimizer overrides are not supported"):
        await create_model_endpoint(
            CreateModelRequest(
                model_id="default",
                base_model="Qwen/Qwen3-8B",
                optimizer_config=OptimizerConfigRequest(type="signsgd", learning_rate=2e-4),
            ),
            server=server,
        )


async def test_kill_session_endpoint_cleans_up_lora_session_registry(tmp_path):
    server = _build_server(tmp_path)
    request = CreateModelRequest(
        model_id="session-a",
        base_model="Qwen/Qwen3-8B",
        lora_config={"rank": 4},
    )

    await create_model_endpoint(
        request,
        server=server,
    )

    response = await kill_session_endpoint(
        KillSessionRequest(model_id="session-a", save_checkpoint=False),
        server=server,
    )

    assert response.success is True
    assert "session-a" not in server.registered_model_ids
    assert "session-a" not in server.model_configs
    assert server.future_store.deleted_models == ["session-a"]
    kill_requests = [request for request in server.orchestrator_client.requests if request.operation == "kill_session"]
    assert len(kill_requests) == 1
    assert kill_requests[0].payload.model_id == "session-a"

    await create_model_endpoint(request, server=server)
    register_requests = [
        request for request in server.orchestrator_client.requests if request.operation == "register_session"
    ]
    assert len(register_requests) == 2


async def test_kill_session_endpoint_returns_xorl_uri_for_lora_checkpoint(tmp_path):
    server = _build_server(tmp_path)
    request = CreateModelRequest(
        model_id="session-a",
        base_model="Qwen/Qwen3-8B",
        lora_config={"rank": 4},
    )
    await create_model_endpoint(request, server=server)

    checkpoint_dir = tmp_path / "weights" / "session-a" / "session_session-a_final"
    checkpoint_dir.mkdir(parents=True)

    async def _wait_for_kill(self, response_future, request_id, timeout, timeout_message="timeout"):
        if response_future.operation == "kill_session":
            return types.SimpleNamespace(
                outputs=[
                    {
                        "success": True,
                        "message": "killed session-a",
                        "checkpoint_path": str(checkpoint_dir),
                    }
                ]
            )
        return await _build_wait_for_response()(self, response_future, request_id, timeout, timeout_message)

    server._wait_for_response = types.MethodType(_wait_for_kill, server)

    response = await kill_session_endpoint(
        KillSessionRequest(model_id="session-a", save_checkpoint=True),
        server=server,
    )

    assert response.success is True
    assert response.checkpoint_path == "xorl://session-a/weights/session_session-a_final"


async def test_kill_session_endpoint_preserves_default_lora_session(tmp_path):
    server = _build_server(tmp_path)

    response = await kill_session_endpoint(
        KillSessionRequest(model_id="default", save_checkpoint=False),
        server=server,
    )

    assert response.success is True
    assert "default" in server.registered_model_ids
    assert "default" in server.model_configs
    assert server.orchestrator_client.requests == []


async def test_unload_model_endpoint_rejects_default_lora_session(tmp_path):
    server = _build_server(tmp_path)

    with pytest.raises(HTTPException, match="reserved and cannot be unloaded") as exc_info:
        await unload_model_endpoint(
            UnloadModelRequest(model_id="default"),
            server=server,
        )

    assert exc_info.value.status_code == 400


async def test_weights_info_endpoint_reads_session_spec_from_checkpoint(tmp_path):
    server = _build_server(tmp_path)
    checkpoint_dir = tmp_path / "weights" / "session-a" / "ckpt-001"
    checkpoint_dir.mkdir(parents=True)

    write_session_spec(
        str(checkpoint_dir),
        {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
            "lora_config": {"lora_rank": 4, "lora_alpha": 12},
            "optimizer_config": {
                "type": "signsgd",
                "learning_rate": 2e-4,
                "weight_decay": 0.0,
                "optimizer_dtype": "bf16",
                "betas": None,
                "eps": None,
                "optimizer_kwargs": {},
            },
        },
    )

    # Intentionally store different in-memory metadata to ensure weights_info trusts disk.
    server.model_configs["session-a"] = {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": True,
        "lora_config": {"lora_rank": 8, "lora_alpha": 16},
        "optimizer_config": {
            "type": "adamw",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "optimizer_dtype": "bf16",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "optimizer_kwargs": {},
        },
    }

    response = await weights_info_endpoint(
        WeightsInfoRequest(xorl_path="xorl://session-a/weights/ckpt-001"),
        server=server,
    )

    assert response.base_model == "Qwen/Qwen3-8B"
    assert response.lora_config.lora_rank == 4
    assert response.lora_config.lora_alpha == 12
    assert response.optimizer_config.type == "signsgd"
    assert response.optimizer_config.learning_rate == pytest.approx(2e-4)


async def test_weights_info_endpoint_rejects_checkpoint_path_escape(tmp_path):
    server = _build_server(tmp_path / "output")
    escaped_dir = tmp_path / "secret-checkpoint"
    escaped_dir.mkdir()
    write_session_spec(str(escaped_dir), {"base_model": "escaped", "is_lora": False})
    (tmp_path / "output" / "weights" / "default").mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        await weights_info_endpoint(
            WeightsInfoRequest(xorl_path="xorl://default/weights/../../../secret-checkpoint"),
            server=server,
        )

    assert exc_info.value.status_code == 400


async def test_weights_info_endpoint_returns_full_weight_checkpoint_metadata(tmp_path):
    server = _build_full_weight_server(tmp_path)
    checkpoint_dir = tmp_path / "weights" / "default" / "ckpt-001"
    checkpoint_dir.mkdir(parents=True)
    server.model_configs["default"] = {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": False,
    }

    response = await weights_info_endpoint(
        WeightsInfoRequest(xorl_path="xorl://default/weights/ckpt-001"),
        server=server,
    )

    assert response.base_model == "Qwen/Qwen3-8B"
    assert response.is_lora is False
    assert response.lora_config is None
    assert response.optimizer_config is None


def test_load_session_spec_from_checkpoint_upgrades_legacy_signsgd_metadata(tmp_path):
    checkpoint_dir = tmp_path / "weights" / "session-a" / "ckpt-legacy"
    checkpoint_dir.mkdir(parents=True)

    metadata = {
        "lr": 2e-4,
        "optimizer": {
            "type": "signsgd",
            "weight_decay": 0.0,
            "betas": None,
            "eps": None,
            "optimizer_kwargs": {},
        },
    }
    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 4,
        "lora_alpha": 12,
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint_dir / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")

    session_spec = load_session_spec_from_checkpoint(str(checkpoint_dir))

    assert session_spec["base_model"] == "Qwen/Qwen3-8B"
    assert session_spec["is_lora"] is True
    assert session_spec["lora_config"]["lora_rank"] == 4
    assert session_spec["lora_config"]["lora_alpha"] == 12
    assert session_spec["optimizer_config"]["type"] == "signsgd"
    assert session_spec["optimizer_config"]["betas"] is None
    assert session_spec["optimizer_config"]["eps"] is None


@pytest.mark.parametrize(
    ("client_ref", "server_ref"),
    [
        ("Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B"),
        (
            "Qwen/Qwen3.5-35B-A3B",
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/abc123def",
        ),
        (
            "/data/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/abc123def",
            "Qwen/Qwen3.5-35B-A3B",
        ),
    ],
)
def test_canon_base_model_matches_hf_cache_path_to_repo_id(client_ref, server_ref):
    assert _canon_base_model(client_ref) == _canon_base_model(server_ref)


def test_canon_base_model_distinct_models_still_differ():
    assert _canon_base_model("Qwen/Qwen3-8B") != _canon_base_model(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/abc123def"
    )
    # Non-cache paths and None pass through untouched.
    assert _canon_base_model("/shared/checkpoints/my-model") == "/shared/checkpoints/my-model"
    assert _canon_base_model(None) is None
