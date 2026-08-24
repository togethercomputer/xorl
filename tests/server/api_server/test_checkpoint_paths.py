"""
Tests for multi-tenant checkpoint path handling.

These tests verify that checkpoints are saved and loaded with the correct
path structure: output_dir/weights/{model_id}/{checkpoint_name}/
"""

import asyncio
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.exceptions import HTTPException

from xorl.server.api_server.api_types import (
    CreateSamplingSessionRequest,
    DeleteCheckpointRequest,
    InferenceEndpoint,
    ListCheckpointsRequest,
    LoadWeightsRequest,
    RemoveInferenceEndpointRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
)
from xorl.server.api_server.server import APIServer
from xorl.server.api_server.utils import validate_model_id


class TestSaveAndLoadWeightsPaths:
    """Test save_weights and load_weights with multi-tenant path structure."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )
        self.server._running = True
        self.server.orchestrator_client = MagicMock()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_load_all_formats_errors_and_auto_checkpoint(self):
        """Test save creates path, rejects duplicates, load works for all formats, 404s, isolation, and 000000 auto-save."""

        mock_output = MagicMock()

        # --- Save creates correct structure ---
        model_id = "user_123"
        mock_output.outputs = [
            {"checkpoint_path": f"{self.temp_dir}/weights/{model_id}/my_checkpoint", "success": True}
        ]
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock(return_value=AsyncMock())
            response = asyncio.run(
                self.server.save_weights(SaveWeightsRequest(model_id=model_id, path="my_checkpoint"))
            )
        assert response.path == "xorl://user_123/weights/my_checkpoint"

        # --- Save rejects existing checkpoint (409) ---
        existing_path = os.path.join(self.temp_dir, "weights", "user_789", "existing_checkpoint")
        os.makedirs(existing_path, exist_ok=True)
        with open(os.path.join(existing_path, "marker.txt"), "w") as f:
            f.write("existing data")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(self.server.save_weights(SaveWeightsRequest(model_id="user_789", path="existing_checkpoint")))
        assert exc_info.value.status_code == 409
        assert os.path.exists(os.path.join(existing_path, "marker.txt"))

        # --- Load: all path formats ---
        mock_output.outputs = [{"success": True}]

        # xorl:// URI
        os.makedirs(os.path.join(self.temp_dir, "weights", "user_abc", "ckpt-001"), exist_ok=True)
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(
                self.server.load_weights(
                    LoadWeightsRequest(model_id="user_abc", path="xorl://user_abc/weights/ckpt-001")
                )
            )
        assert response.path == "xorl://user_abc/weights/ckpt-001"

        # weights/model_id/name (cross-model)
        os.makedirs(os.path.join(self.temp_dir, "weights", "default", "000000"), exist_ok=True)
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(
                self.server.load_weights(LoadWeightsRequest(model_id="run-2", path="weights/default/000000"))
            )
        assert response.path == "weights/default/000000"

        # model_id/checkpoint format
        os.makedirs(os.path.join(self.temp_dir, "weights", "other_run", "step-100"), exist_ok=True)
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(
                self.server.load_weights(LoadWeightsRequest(model_id="my_run", path="other_run/step-100"))
            )
        assert response.path == "other_run/step-100"

        # checkpoint name only
        os.makedirs(os.path.join(self.temp_dir, "weights", "my_model", "ckpt-002"), exist_ok=True)
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(self.server.load_weights(LoadWeightsRequest(model_id="my_model", path="ckpt-002")))
        assert response.path == "ckpt-002"

        # legacy weights/checkpoint format
        os.makedirs(os.path.join(self.temp_dir, "weights", "some_model", "000000"), exist_ok=True)
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(
                self.server.load_weights(LoadWeightsRequest(model_id="some_model", path="weights/000000"))
            )
        assert response.path == "weights/000000"

        # --- Load errors: 404 and isolation ---
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(self.server.load_weights(LoadWeightsRequest(model_id="x", path="xorl://x/weights/nonexistent")))
        assert exc_info.value.status_code == 404

        # Explicit path uses path's model_id
        os.makedirs(os.path.join(self.temp_dir, "weights", "user_a", "step-50"), exist_ok=True)
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(
                self.server.load_weights(LoadWeightsRequest(model_id="user_b", path="weights/user_a/step-50"))
            )
        assert response.path == "weights/user_a/step-50"

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(self.server.load_weights(LoadWeightsRequest(model_id="user_a", path="weights/user_b/step-50")))
        assert exc_info.value.status_code == 404

        # --- Auto-save 000000 checkpoint ---
        mock_output.outputs = [{"checkpoint_path": f"{self.temp_dir}/weights/test_model/000000"}]
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(self.server.save_weights(SaveWeightsRequest(model_id="test_model", path="000000")))
        assert "000000" in response.path

        # Load 000000 after save
        os.makedirs(os.path.join(self.temp_dir, "weights", "test_model", "000000"), exist_ok=True)
        mock_output.outputs = [{"success": True}]
        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            self.server.orchestrator_client.send_request = AsyncMock()
            response = asyncio.run(self.server.load_weights(LoadWeightsRequest(model_id="test_model", path="000000")))
        assert response.path == "000000"


class TestListDeleteAndIsolation:
    """Test list_checkpoints, delete_checkpoint, and multi-tenant isolation."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_delete_and_isolation(self):
        """Test list scoped by model, delete success/errors, and cross-model isolation."""

        # --- List checkpoints scoped by model ---
        model_id = "user_list_test"
        for name in ["checkpoint-001", "checkpoint-002"]:
            os.makedirs(os.path.join(self.temp_dir, "weights", model_id, name), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "weights", "other_user", "checkpoint-999"), exist_ok=True)

        response = asyncio.run(self.server.list_checkpoints(ListCheckpointsRequest(model_id=model_id)))
        names = [c.checkpoint_id for c in response.checkpoints]
        assert len(response.checkpoints) == 2
        assert all(model_id in n for n in names)
        assert not any("other_user" in n for n in names)

        # Empty model
        response = asyncio.run(self.server.list_checkpoints(ListCheckpointsRequest(model_id="no_checkpoints")))
        assert len(response.checkpoints) == 0

        # --- Delete: success, reserved blocked, invalid format ---
        ckpt_path = os.path.join(self.temp_dir, "weights", "del_test", "to_delete")
        os.makedirs(ckpt_path, exist_ok=True)
        response = asyncio.run(
            self.server.delete_checkpoint(
                DeleteCheckpointRequest(model_id="del_test", checkpoint_id="weights/del_test/to_delete")
            )
        )
        assert response.success is True and not os.path.exists(ckpt_path)

        reserved = os.path.join(self.temp_dir, "weights", "res_test", "000000")
        os.makedirs(reserved, exist_ok=True)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                self.server.delete_checkpoint(
                    DeleteCheckpointRequest(model_id="res_test", checkpoint_id="weights/res_test/000000")
                )
            )
        assert exc_info.value.status_code == 403 and os.path.exists(reserved)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(self.server.delete_checkpoint(DeleteCheckpointRequest(model_id="u", checkpoint_id="invalid")))
        assert exc_info.value.status_code == 400

        # --- Multi-tenant isolation ---
        for mid in ["user_a", "user_b", "user_c"]:
            p = os.path.join(self.temp_dir, "weights", mid, "ckpt-001")
            os.makedirs(p, exist_ok=True)
            with open(os.path.join(p, "data.txt"), "w") as f:
                f.write(f"data for {mid}")

        for mid in ["user_a", "user_b", "user_c"]:
            with open(os.path.join(self.temp_dir, "weights", mid, "ckpt-001", "data.txt")) as f:
                assert f.read() == f"data for {mid}"

        for mid in ["model_x", "model_y"]:
            for i in range(3):
                os.makedirs(os.path.join(self.temp_dir, "weights", mid, f"ckpt-{i:03d}"), exist_ok=True)
        response = asyncio.run(self.server.list_checkpoints(ListCheckpointsRequest(model_id="model_x")))
        assert len(response.checkpoints) == 3
        assert all("model_x" in c.checkpoint_id for c in response.checkpoints)

        sampler_case = TestSamplerWeightsAndAdapterTracking()
        sampler_case.setup_method()
        try:
            sampler_case._assert_sampler_listing_deletion_and_adapter_tracking()
        finally:
            sampler_case.teardown_method()


class TestModelIdValidation:
    """Test model_id validation."""

    def test_valid_invalid_defaults_and_traversal(self):
        """Test valid IDs, invalid IDs, defaults, and path traversal blocking."""

        # Valid
        for mid in ["default", "user_123", "user-123", "User123", "a", "A1_b2-C3"]:
            assert validate_model_id(mid) == mid

        # Defaults
        assert validate_model_id("") == "default"
        assert validate_model_id(None) == "default"

        # Invalid
        for mid in [
            "../etc/passwd",
            "user/name",
            "user\\name",
            "user name",
            "user@name",
            "user.name",
            "_start",
            "-start",
            "a" * 129,
        ]:
            with pytest.raises(HTTPException) as exc_info:
                validate_model_id(mid)
            assert exc_info.value.status_code == 400

        # Path traversal
        for attempt in ["../secret", "..\\secret", "foo/../bar", "..", "."]:
            with pytest.raises(HTTPException):
                validate_model_id(attempt)


class TestSamplerWeightsAndAdapterTracking:
    """Test sampler_weights flat structure, adapter resolution, and adapter tracking."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _assert_sampler_listing_deletion_and_adapter_tracking(self):
        """Test sampler in listings, shared across models, deletion, path resolution, and adapter tracking."""

        # --- Sampler in listings ---
        os.makedirs(os.path.join(self.temp_dir, "weights", "user_test", "ckpt-001"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "sampler_weights", "adapter-001"), exist_ok=True)
        response = asyncio.run(self.server.list_checkpoints(ListCheckpointsRequest(model_id="user_test")))
        ids = [c.checkpoint_id for c in response.checkpoints]
        types = [c.checkpoint_type for c in response.checkpoints]
        assert len(response.checkpoints) == 2
        assert "sampler_weights/adapter-001" in ids and "sampler" in types

        # Shared across models
        os.makedirs(os.path.join(self.temp_dir, "sampler_weights", "shared-adapter"), exist_ok=True)
        for mid in ["user_a", "user_b"]:
            response = asyncio.run(self.server.list_checkpoints(ListCheckpointsRequest(model_id=mid)))
            assert any(c.checkpoint_id == "sampler_weights/shared-adapter" for c in response.checkpoints)

        # --- Sampler deletion ---
        sp = os.path.join(self.temp_dir, "sampler_weights", "to-delete")
        os.makedirs(sp, exist_ok=True)
        response = asyncio.run(
            self.server.delete_checkpoint(
                DeleteCheckpointRequest(model_id="any", checkpoint_id="sampler_weights/to-delete")
            )
        )
        assert response.success is True and not os.path.exists(sp)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                self.server.delete_checkpoint(
                    DeleteCheckpointRequest(model_id="any", checkpoint_id="sampler_weights/nonexistent")
                )
            )
        assert exc_info.value.status_code == 404

        # --- Path resolution ---
        ap = os.path.join(self.temp_dir, "sampler_weights", "a-001")
        os.makedirs(ap, exist_ok=True)
        model_id, name, path = self.server._resolve_model_path("sampler_weights/a-001")
        assert model_id is None and name == "a-001" and path == ap
        model_id, name, path = self.server._resolve_model_path("a-001")
        assert model_id is None and name == "a-001"
        model_id, name, path = self.server._resolve_model_path("xorl://session-a/sampler_weights/a-001")
        assert model_id == "session-a" and name == "a-001" and path == ap

        with pytest.raises(HTTPException) as exc_info:
            self.server._resolve_model_path("nonexistent")
        assert exc_info.value.status_code == 404

        # --- Adapter tracking ---
        assert self.server._track_adapter("adapter-001", "/path/1") is False
        assert ("adapter-001", "/path/1") in self.server.loaded_sampling_loras["default"]
        self.server._track_adapter("adapter-002", "/path/2")
        assert self.server._track_adapter("adapter-001", "/path/1") is True
        assert self.server.loaded_sampling_loras["default"][-1] == ("adapter-001", "/path/1")
        assert len(self.server.loaded_sampling_loras) == 1

        # Dropping the last receiver invalidates the adapter state that was
        # observed on that receiver.
        self.server.inference_endpoints = [
            InferenceEndpoint(
                host="sgl.local",
                port=30000,
                worker_port=29999,
                world_size=1,
                healthy=True,
                server_info=None,
            )
        ]
        response = self.server.remove_inference_endpoint(RemoveInferenceEndpointRequest(host="sgl.local", port=30000))
        assert response.success is True
        assert self.server.inference_endpoints == []
        assert self.server.loaded_sampling_loras == {}

        self._assert_sampling_adapter_reconciliation_and_session_tracking_policy()
        self._assert_save_weights_for_sampler_respects_lora_serving_mode()

    def _assert_save_weights_for_sampler_respects_lora_serving_mode(self):
        """Sampler publication must use the explicitly selected LoRA contract."""
        self.server._running = True
        self.server.orchestrator_client = MagicMock()
        self.server.orchestrator_client.send_request = AsyncMock(return_value=AsyncMock())
        common_config = {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
            "optimizer_config": {
                "type": "signsgd",
                "learning_rate": 2e-4,
                "weight_decay": 0.0,
                "optimizer_dtype": "bf16",
                "betas": None,
                "eps": None,
                "optimizer_kwargs": {},
            },
        }
        self.server.model_configs["adapter-run"] = {
            **common_config,
            "lora_config": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_serving_mode": "separate",
            },
        }

        mock_output = MagicMock()
        mock_output.outputs = [{"lora_path": os.path.join(self.temp_dir, "sampler_weights", "adapter-export")}]

        with patch.object(self.server, "_wait_for_response", return_value=mock_output):
            response = asyncio.run(
                self.server.save_weights_for_sampler(
                    SaveWeightsForSamplerRequest(model_id="adapter-run", name="adapter-export")
                )
            )

        request = self.server.orchestrator_client.send_request.await_args.args[0]
        assert request.operation == "save_lora_only"
        assert request.payload.lora_path.endswith("sampler_weights/adapter-export")
        assert response.path == "xorl://adapter-run/sampler_weights/adapter-export"

        self.server.orchestrator_client.send_request.reset_mock()
        self.server.model_configs["merged-run"] = {
            **common_config,
            "lora_config": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_serving_mode": "merged",
            },
        }
        with pytest.raises(HTTPException, match="does not materialize a folded") as exc_info:
            asyncio.run(
                self.server.save_weights_for_sampler(
                    SaveWeightsForSamplerRequest(model_id="merged-run", name="merged-export")
                )
            )
        assert exc_info.value.status_code == 500
        self.server.orchestrator_client.send_request.assert_not_awaited()

    def _assert_sampling_adapter_reconciliation_and_session_tracking_policy(self):
        """Stale tracked adapters should not force a bogus unload before loading a fresh adapter."""
        fresh_path = os.path.join(self.temp_dir, "sampler_weights", "fresh-001")
        os.makedirs(fresh_path, exist_ok=True)

        self.server.inference_endpoints = [MagicMock()]
        self.server.max_adapters_per_model = 1
        self.server.loaded_sampling_loras["default"] = [("stale-001", "/stale/path")]
        self.server._get_loaded_adapters_from_endpoint = AsyncMock(return_value=[])
        self.server._load_lora_on_inference_endpoints = AsyncMock(return_value=True)
        self.server._unload_lora_on_inference_endpoints = AsyncMock(return_value=True)

        response = asyncio.run(
            self.server.create_sampling_session(CreateSamplingSessionRequest(model_path="fresh-001"))
        )

        assert response.lora_name == "fresh-001"
        self.server._unload_lora_on_inference_endpoints.assert_not_awaited()
        self.server._load_lora_on_inference_endpoints.assert_awaited_once_with("fresh-001", fresh_path)
        assert self.server.loaded_sampling_loras["default"] == [("fresh-001", fresh_path)]

        self._assert_query_failure_preserves_tracking()
        self.server.loaded_sampling_loras.clear()
        self._assert_sampling_session_tracking_is_model_scoped_and_atomic()

    def _assert_query_failure_preserves_tracking(self):
        """A transient endpoint introspection failure should not erase tracked sampler adapters."""
        self.server.inference_endpoints = [MagicMock()]
        self.server.loaded_sampling_loras["default"] = [("tracked-001", "/tracked/path")]
        self.server._get_loaded_adapters_from_endpoint = AsyncMock(return_value=None)

        loaded_by_endpoint = asyncio.run(self.server._reconcile_tracked_adapters("default"))

        assert loaded_by_endpoint == [set()]
        assert self.server.loaded_sampling_loras["default"] == [("tracked-001", "/tracked/path")]

    def _assert_sampling_session_tracking_is_model_scoped_and_atomic(self):
        uri_path = os.path.join(self.temp_dir, "sampler_weights", "session-a-adapter")
        os.makedirs(uri_path, exist_ok=True)
        plain_path = os.path.join(self.temp_dir, "sampler_weights", "shared-adapter")
        os.makedirs(plain_path, exist_ok=True)

        self.server.inference_endpoints = [MagicMock()]
        self.server._get_loaded_adapters_from_endpoint = AsyncMock(return_value=[])
        self.server._load_lora_on_inference_endpoints = AsyncMock(return_value=True)

        response = asyncio.run(
            self.server.create_sampling_session(
                CreateSamplingSessionRequest(model_path="xorl://session-a/sampler_weights/session-a-adapter")
            )
        )

        assert response.lora_name == "session-a-adapter"
        assert self.server.loaded_sampling_loras["session-a"] == [("session-a-adapter", uri_path)]
        assert self.server.loaded_sampling_loras.get("default", []) == []

        response = asyncio.run(
            self.server.create_sampling_session(
                CreateSamplingSessionRequest(model_id="session-b", model_path="shared-adapter")
            )
        )

        assert response.lora_name == "shared-adapter"
        assert self.server.loaded_sampling_loras["session-b"] == [("shared-adapter", plain_path)]
        assert self.server.loaded_sampling_loras.get("default", []) == []

        failing_path = os.path.join(self.temp_dir, "sampler_weights", "failing-001")
        os.makedirs(failing_path, exist_ok=True)

        self.server.inference_endpoints = [MagicMock()]
        self.server._get_loaded_adapters_from_endpoint = AsyncMock(return_value=[])
        self.server._load_lora_on_inference_endpoints = AsyncMock(
            side_effect=HTTPException(status_code=500, detail="load failed")
        )

        with pytest.raises(HTTPException):
            asyncio.run(self.server.create_sampling_session(CreateSamplingSessionRequest(model_path="failing-001")))

        assert self.server.loaded_sampling_loras.get("default", []) == []
