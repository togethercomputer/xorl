import pytest

from xorl.server.backend.remote import RemoteBackend


@pytest.mark.asyncio
async def test_sync_inference_weights_uses_backend_operation_timeout(monkeypatch):
    backend = RemoteBackend(operation_timeout=2400.0)
    captured = {}

    async def fake_execute(operation, payload, request_id=None, timeout=None):
        captured["operation"] = operation
        captured["payload"] = payload
        captured["request_id"] = request_id
        captured["timeout"] = timeout
        return {"success": True}

    monkeypatch.setattr(backend, "_execute", fake_execute)

    await backend.sync_inference_weights(
        endpoints=[{"host": "inference.example", "port": 30000, "world_size": 4}],
        master_address="trainer.example",
        request_id="sync-req",
    )

    assert captured["operation"] == "sync_inference_weights"
    assert captured["request_id"] == "sync-req"
    assert captured["timeout"] == 2400.0
