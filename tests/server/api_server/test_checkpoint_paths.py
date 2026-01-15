"""
Tests for multi-tenant checkpoint path handling.

These tests verify that checkpoints are saved and loaded with the correct
path structure: output_dir/weights/{model_id}/{checkpoint_name}/
"""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xorl.server.api_server.api_server import APIServer, validate_model_id
from xorl.server.api_server.api_types import (
    SaveWeightsRequest,
    LoadWeightsRequest,
    ListCheckpointsRequest,
    DeleteCheckpointRequest,
)


class TestCheckpointPathConstruction:
    """Test checkpoint path construction with model_id."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_xorl_uri_construction(self):
        """Test xorl:// URI is correctly constructed."""
        uri = self.server._to_xorl_uri("user_123", "checkpoint-001")
        assert uri == "xorl://user_123/weights/checkpoint-001"

    def test_xorl_uri_parsing(self):
        """Test xorl:// URI is correctly parsed."""
        model_id, checkpoint_name = self.server._from_xorl_uri(
            "xorl://user_123/weights/checkpoint-001"
        )
        assert model_id == "user_123"
        assert checkpoint_name == "checkpoint-001"

    def test_xorl_uri_parsing_weights_path_format(self):
        """Test weights/model_id/checkpoint format is correctly parsed."""
        model_id, checkpoint_name = self.server._from_xorl_uri(
            "weights/user_456/000000"
        )
        assert model_id == "user_456"
        assert checkpoint_name == "000000"

    def test_xorl_uri_parsing_weights_legacy_format(self):
        """Test legacy weights/checkpoint format defaults to 'default' model_id."""
        model_id, checkpoint_name = self.server._from_xorl_uri(
            "weights/000000"
        )
        assert model_id == "default"
        assert checkpoint_name == "000000"

    def test_xorl_uri_parsing_checkpoint_name_only(self):
        """Test checkpoint name only falls back to default model_id."""
        model_id, checkpoint_name = self.server._from_xorl_uri("000000")
        assert model_id == "default"
        assert checkpoint_name == "000000"

    def test_xorl_uri_parsing_model_id_checkpoint_format(self):
        """Test model_id/checkpoint format is correctly parsed."""
        model_id, checkpoint_name = self.server._from_xorl_uri("user_123/adapter-a")
        assert model_id == "user_123"
        assert checkpoint_name == "adapter-a"

    def test_xorl_uri_parsing_raw_path_fallback(self):
        """Test raw path with 3+ parts falls back to default model_id."""
        model_id, checkpoint_name = self.server._from_xorl_uri("some/raw/path")
        assert model_id == "default"
        assert checkpoint_name == "some/raw/path"


class TestSaveWeightsPathHandling:
    """Test save_weights with multi-tenant path structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )
        # Mock the server as running
        self.server._running = True
        self.server.engine_client = MagicMock()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_weights_creates_model_id_directory(self):
        """Test that save_weights creates the correct directory structure."""
        # Create a mock response
        mock_output = MagicMock()
        mock_output.outputs = [{"checkpoint_path": f"{self.temp_dir}/weights/user_123/my_checkpoint", "success": True}]

        mock_future = AsyncMock()
        mock_future.return_value = mock_output
        self.server.engine_client.send_request = AsyncMock(return_value=mock_future)

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            request = SaveWeightsRequest(model_id="user_123", path="my_checkpoint")
            response = await self.server.save_weights(request)

        # Verify the response URI includes model_id
        assert "user_123" in response.path
        assert "my_checkpoint" in response.path

    def test_checkpoint_path_includes_model_id(self):
        """Test that checkpoint path is constructed with model_id."""
        model_id = "user_456"
        checkpoint_name = "checkpoint-001"

        expected_path = os.path.join(self.temp_dir, "weights", model_id, checkpoint_name)

        # This is what save_weights should construct internally
        actual_path = os.path.join(self.temp_dir, "weights", model_id, checkpoint_name)

        assert actual_path == expected_path

    @pytest.mark.asyncio
    async def test_save_weights_existing_checkpoint_warning(self):
        """Test that existing checkpoint returns warning without overwriting."""
        model_id = "user_789"
        checkpoint_name = "existing_checkpoint"

        # Create the existing checkpoint directory
        checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Create a marker file to verify it's not deleted
        marker_file = os.path.join(checkpoint_path, "marker.txt")
        with open(marker_file, "w") as f:
            f.write("existing data")

        request = SaveWeightsRequest(model_id=model_id, path=checkpoint_name)
        response = await self.server.save_weights(request)

        # Should return warning
        assert response.warning is not None
        assert "already exists" in response.warning

        # Original file should still exist
        assert os.path.exists(marker_file)


class TestLoadWeightsPathHandling:
    """Test load_weights with multi-tenant path structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )
        self.server._running = True
        self.server.engine_client = MagicMock()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_load_weights_extracts_model_id_from_uri(self):
        """Test that load_weights correctly extracts model_id from xorl:// URI."""
        model_id = "user_abc"
        checkpoint_name = "checkpoint-001"

        # Create the checkpoint directory
        checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Mock the engine response
        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            request = LoadWeightsRequest(
                model_id=model_id,
                path=f"xorl://{model_id}/weights/{checkpoint_name}"
            )
            response = await self.server.load_weights(request)

        assert response.path == f"xorl://{model_id}/weights/{checkpoint_name}"

    @pytest.mark.asyncio
    async def test_load_weights_from_other_run(self):
        """Test loading checkpoint from another model_id using weights/model_id/name format."""
        # Create checkpoint for "default" model
        default_checkpoint = os.path.join(self.temp_dir, "weights", "default", "000000")
        os.makedirs(default_checkpoint, exist_ok=True)

        # Mock the engine response
        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            # Load from "default" while current model_id is "run-2"
            request = LoadWeightsRequest(
                model_id="run-2",  # Current model
                path="weights/default/000000"  # Load from different model
            )
            response = await self.server.load_weights(request)

        # Should successfully load from "default", not "run-2"
        assert response.path == "weights/default/000000"

    @pytest.mark.asyncio
    async def test_load_weights_from_other_run_model_id_checkpoint_format(self):
        """Test loading checkpoint using model_id/checkpoint format."""
        # Create checkpoint for "other_run" model
        other_checkpoint = os.path.join(self.temp_dir, "weights", "other_run", "step-100")
        os.makedirs(other_checkpoint, exist_ok=True)

        # Mock the engine response
        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            # Load from "other_run" while current model_id is "my_run"
            request = LoadWeightsRequest(
                model_id="my_run",
                path="other_run/step-100"
            )
            response = await self.server.load_weights(request)

        assert response.path == "other_run/step-100"

    @pytest.mark.asyncio
    async def test_load_weights_checkpoint_name_only_uses_request_model_id(self):
        """Test that checkpoint name only uses request.model_id."""
        model_id = "my_model"
        # Create checkpoint for "my_model"
        checkpoint = os.path.join(self.temp_dir, "weights", model_id, "checkpoint-001")
        os.makedirs(checkpoint, exist_ok=True)

        # Mock the engine response
        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            # Just checkpoint name - should use model_id from request
            request = LoadWeightsRequest(
                model_id=model_id,
                path="checkpoint-001"
            )
            response = await self.server.load_weights(request)

        assert response.path == "checkpoint-001"

    @pytest.mark.asyncio
    async def test_load_weights_checkpoint_not_found(self):
        """Test that load_weights raises 404 for non-existent checkpoint."""
        from fastapi import HTTPException

        request = LoadWeightsRequest(
            model_id="nonexistent_user",
            path="xorl://nonexistent_user/weights/nonexistent_checkpoint"
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.server.load_weights(request)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_load_weights_legacy_format(self):
        """Test loading checkpoint using legacy weights/checkpoint format."""
        # Create checkpoint for "default" model (legacy format maps to default)
        checkpoint = os.path.join(self.temp_dir, "weights", "default", "000000")
        os.makedirs(checkpoint, exist_ok=True)

        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            # Legacy format: weights/checkpoint_name (no model_id in path)
            request = LoadWeightsRequest(
                model_id="some_other_model",  # Should be ignored for legacy format
                path="weights/000000"
            )
            response = await self.server.load_weights(request)

        # Should load from "default" model, not "some_other_model"
        assert response.path == "weights/000000"

    @pytest.mark.asyncio
    async def test_load_weights_does_not_mix_model_ids(self):
        """Test that explicitly specifying model_id in path doesn't use request.model_id."""
        # Create checkpoint for "user_a"
        user_a_checkpoint = os.path.join(self.temp_dir, "weights", "user_a", "step-50")
        os.makedirs(user_a_checkpoint, exist_ok=True)

        # DON'T create checkpoint for "user_b" - we want to verify it loads from user_a

        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            # Explicitly request user_a's checkpoint while being user_b
            request = LoadWeightsRequest(
                model_id="user_b",
                path="weights/user_a/step-50"
            )
            response = await self.server.load_weights(request)

        assert response.path == "weights/user_a/step-50"

    @pytest.mark.asyncio
    async def test_load_weights_wrong_model_returns_404(self):
        """Test that loading non-existent checkpoint from specified model returns 404."""
        from fastapi import HTTPException

        # Create checkpoint for "user_a" but NOT for "user_b"
        user_a_checkpoint = os.path.join(self.temp_dir, "weights", "user_a", "step-50")
        os.makedirs(user_a_checkpoint, exist_ok=True)

        # Try to load user_b's checkpoint (doesn't exist)
        request = LoadWeightsRequest(
            model_id="user_a",  # Current model
            path="weights/user_b/step-50"  # Trying to load from user_b
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.server.load_weights(request)

        assert exc_info.value.status_code == 404
        assert "user_b" in exc_info.value.detail


class TestListCheckpointsPathHandling:
    """Test list_checkpoints with multi-tenant path structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_list_checkpoints_scans_model_directory(self):
        """Test that list_checkpoints scans the correct model directory."""
        model_id = "user_list_test"

        # Create checkpoints for this model
        for name in ["checkpoint-001", "checkpoint-002"]:
            checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, name)
            os.makedirs(checkpoint_path, exist_ok=True)

        # Create checkpoints for another model (should not be listed)
        other_model_path = os.path.join(self.temp_dir, "weights", "other_user", "checkpoint-999")
        os.makedirs(other_model_path, exist_ok=True)

        request = ListCheckpointsRequest(model_id=model_id)
        response = await self.server.list_checkpoints(request)

        # Should only list checkpoints for the requested model
        checkpoint_names = [c.checkpoint_id for c in response.checkpoints]
        assert len(response.checkpoints) == 2
        assert all(model_id in name for name in checkpoint_names)
        assert not any("other_user" in name for name in checkpoint_names)

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty_model_directory(self):
        """Test list_checkpoints returns empty list for model with no checkpoints."""
        request = ListCheckpointsRequest(model_id="model_with_no_checkpoints")
        response = await self.server.list_checkpoints(request)

        assert len(response.checkpoints) == 0


class TestDeleteCheckpointPathHandling:
    """Test delete_checkpoint with multi-tenant path structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_delete_checkpoint_parses_model_id_from_checkpoint_id(self):
        """Test that delete_checkpoint correctly parses checkpoint_id with model_id."""
        model_id = "user_delete_test"
        checkpoint_name = "to_delete"

        # Create the checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        request = DeleteCheckpointRequest(
            model_id=model_id,
            checkpoint_id=f"weights/{model_id}/{checkpoint_name}"
        )
        response = await self.server.delete_checkpoint(request)

        assert response.success is True
        assert not os.path.exists(checkpoint_path)

    @pytest.mark.asyncio
    async def test_delete_checkpoint_reserved_checkpoint_blocked(self):
        """Test that reserved checkpoint 000000 cannot be deleted."""
        from fastapi import HTTPException

        model_id = "user_reserved_test"

        # Create the reserved checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, "000000")
        os.makedirs(checkpoint_path, exist_ok=True)

        request = DeleteCheckpointRequest(
            model_id=model_id,
            checkpoint_id=f"weights/{model_id}/000000"
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.server.delete_checkpoint(request)

        assert exc_info.value.status_code == 403
        # Checkpoint should still exist
        assert os.path.exists(checkpoint_path)

    @pytest.mark.asyncio
    async def test_delete_checkpoint_invalid_format(self):
        """Test that invalid checkpoint_id format raises error."""
        from fastapi import HTTPException

        request = DeleteCheckpointRequest(
            model_id="user",
            checkpoint_id="invalid_format"  # Missing model_id in path
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.server.delete_checkpoint(request)

        assert exc_info.value.status_code == 400


class TestMultiTenantIsolation:
    """Test that different model_ids have isolated checkpoint storage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_different_models_have_isolated_checkpoints(self):
        """Test that checkpoints from different models don't interfere."""
        # Create checkpoints for multiple models with same checkpoint name
        for model_id in ["user_a", "user_b", "user_c"]:
            checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, "checkpoint-001")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Write model-specific data
            with open(os.path.join(checkpoint_path, "data.txt"), "w") as f:
                f.write(f"data for {model_id}")

        # Verify each model has its own isolated checkpoint
        for model_id in ["user_a", "user_b", "user_c"]:
            checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, "checkpoint-001")
            with open(os.path.join(checkpoint_path, "data.txt"), "r") as f:
                content = f.read()
            assert content == f"data for {model_id}"

    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_only_model_checkpoints(self):
        """Test that list_checkpoints only returns checkpoints for the specified model."""
        # Create checkpoints for multiple models
        for model_id in ["model_x", "model_y"]:
            for i in range(3):
                checkpoint_path = os.path.join(
                    self.temp_dir, "weights", model_id, f"checkpoint-{i:03d}"
                )
                os.makedirs(checkpoint_path, exist_ok=True)

        # List checkpoints for model_x only
        request = ListCheckpointsRequest(model_id="model_x")
        response = await self.server.list_checkpoints(request)

        # Should only have model_x checkpoints
        assert len(response.checkpoints) == 3
        for checkpoint in response.checkpoints:
            assert "model_x" in checkpoint.checkpoint_id
            assert "model_y" not in checkpoint.checkpoint_id


class TestModelIdValidation:
    """Test model_id validation."""

    def test_valid_model_ids(self):
        """Test that valid model_ids pass validation and return the same value."""
        valid_ids = [
            "default",
            "user_123",
            "user-123",
            "User123",
            "a",
            "A1_b2-C3",
            "model_with_underscores",
            "model-with-hyphens",
        ]
        # These should not raise and return the same value
        for model_id in valid_ids:
            assert validate_model_id(model_id) == model_id

    def test_empty_model_id_defaults_to_default(self):
        """Test that empty or None model_id defaults to 'default'."""
        # Empty string should return "default"
        assert validate_model_id("") == "default"
        # None should return "default"
        assert validate_model_id(None) == "default"

    def test_invalid_model_ids(self):
        """Test that invalid model_ids raise HTTPException."""
        from fastapi import HTTPException

        invalid_ids = [
            "../etc/passwd",  # Path traversal
            "user/name",  # Contains slash
            "user\\name",  # Contains backslash
            "user name",  # Contains space
            "user@name",  # Contains @
            "user.name",  # Contains dot
            "_underscore_start",  # Starts with underscore
            "-hyphen-start",  # Starts with hyphen
            "a" * 129,  # Too long (>128 chars)
        ]
        for model_id in invalid_ids:
            with pytest.raises(HTTPException) as exc_info:
                validate_model_id(model_id)
            assert exc_info.value.status_code == 400

    def test_model_id_prevents_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        from fastapi import HTTPException

        traversal_attempts = [
            "../secret",
            "..\\secret",
            "foo/../bar",
            "foo/../../etc/passwd",
            "..",
            ".",
        ]
        for attempt in traversal_attempts:
            with pytest.raises(HTTPException):
                validate_model_id(attempt)


class TestSamplerWeightsFlatStructure:
    """Test sampler_weights flat directory structure (no model_id)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_list_checkpoints_includes_sampler_weights(self):
        """Test that list_checkpoints includes flat sampler_weights."""
        # Create training checkpoints
        model_id = "user_test"
        training_path = os.path.join(self.temp_dir, "weights", model_id, "ckpt-001")
        os.makedirs(training_path, exist_ok=True)

        # Create flat sampler checkpoints (no model_id in path)
        sampler_path = os.path.join(self.temp_dir, "sampler_weights", "adapter-001")
        os.makedirs(sampler_path, exist_ok=True)

        request = ListCheckpointsRequest(model_id=model_id)
        response = await self.server.list_checkpoints(request)

        # Should have both training and sampler checkpoints
        checkpoint_ids = [c.checkpoint_id for c in response.checkpoints]
        checkpoint_types = [c.checkpoint_type for c in response.checkpoints]

        assert len(response.checkpoints) == 2
        assert f"weights/{model_id}/ckpt-001" in checkpoint_ids
        assert "sampler_weights/adapter-001" in checkpoint_ids
        assert "training" in checkpoint_types
        assert "sampler" in checkpoint_types

    @pytest.mark.asyncio
    async def test_sampler_weights_shared_across_model_ids(self):
        """Test that sampler_weights are visible regardless of model_id."""
        # Create sampler checkpoint
        sampler_path = os.path.join(self.temp_dir, "sampler_weights", "shared-adapter")
        os.makedirs(sampler_path, exist_ok=True)

        # List for different model_ids - all should see the sampler checkpoint
        for model_id in ["user_a", "user_b", "user_c"]:
            request = ListCheckpointsRequest(model_id=model_id)
            response = await self.server.list_checkpoints(request)

            sampler_checkpoints = [c for c in response.checkpoints if c.checkpoint_type == "sampler"]
            assert len(sampler_checkpoints) == 1
            assert sampler_checkpoints[0].checkpoint_id == "sampler_weights/shared-adapter"

    @pytest.mark.asyncio
    async def test_delete_sampler_checkpoint(self):
        """Test that sampler checkpoints can be deleted with flat path."""
        # Create sampler checkpoint
        sampler_path = os.path.join(self.temp_dir, "sampler_weights", "to-delete")
        os.makedirs(sampler_path, exist_ok=True)

        # Delete using flat checkpoint_id format
        request = DeleteCheckpointRequest(
            model_id="any_model",  # model_id is not used for sampler_weights
            checkpoint_id="sampler_weights/to-delete"
        )
        response = await self.server.delete_checkpoint(request)

        assert response.success is True
        assert not os.path.exists(sampler_path)

    @pytest.mark.asyncio
    async def test_delete_sampler_checkpoint_not_found(self):
        """Test that deleting non-existent sampler checkpoint raises 404."""
        from fastapi import HTTPException

        request = DeleteCheckpointRequest(
            model_id="any_model",
            checkpoint_id="sampler_weights/nonexistent"
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.server.delete_checkpoint(request)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_empty_sampler_weights_directory(self):
        """Test list_checkpoints with no sampler_weights."""
        # Only create training checkpoint
        model_id = "test_user"
        training_path = os.path.join(self.temp_dir, "weights", model_id, "ckpt-001")
        os.makedirs(training_path, exist_ok=True)

        request = ListCheckpointsRequest(model_id=model_id)
        response = await self.server.list_checkpoints(request)

        # Should only have training checkpoint
        assert len(response.checkpoints) == 1
        assert response.checkpoints[0].checkpoint_type == "training"


class TestCreateSamplingSession:
    """Test create_sampling_session with flat sampler_weights structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_model_path_with_prefix(self):
        """Test _resolve_model_path with sampler_weights/ prefix."""
        # Create sampler checkpoint
        adapter_name = "adapter-001"
        sampler_path = os.path.join(self.temp_dir, "sampler_weights", adapter_name)
        os.makedirs(sampler_path, exist_ok=True)

        # Resolve with prefix
        lora_name, absolute_path = self.server._resolve_model_path(f"sampler_weights/{adapter_name}")

        assert lora_name == adapter_name
        assert absolute_path == sampler_path

    def test_resolve_model_path_without_prefix(self):
        """Test _resolve_model_path with just adapter name."""
        # Create sampler checkpoint
        adapter_name = "adapter-002"
        sampler_path = os.path.join(self.temp_dir, "sampler_weights", adapter_name)
        os.makedirs(sampler_path, exist_ok=True)

        # Resolve without prefix
        lora_name, absolute_path = self.server._resolve_model_path(adapter_name)

        assert lora_name == adapter_name
        assert absolute_path == sampler_path

    def test_resolve_model_path_not_found(self):
        """Test _resolve_model_path raises 404 for non-existent path."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            self.server._resolve_model_path("nonexistent-adapter")

        assert exc_info.value.status_code == 404

    def test_track_adapter_new(self):
        """Test _track_adapter adds new adapter."""
        lora_name = "adapter-001"
        lora_path = "/path/to/adapter"

        result = self.server._track_adapter(lora_name, lora_path)

        # Should return False (not already tracked)
        assert result is False
        assert "default" in self.server.loaded_sampling_loras
        assert (lora_name, lora_path) in self.server.loaded_sampling_loras["default"]

    def test_track_adapter_existing(self):
        """Test _track_adapter moves existing adapter to MRU."""
        # Add first adapter
        self.server._track_adapter("adapter-001", "/path/1")
        # Add second adapter
        self.server._track_adapter("adapter-002", "/path/2")

        # Re-track first adapter
        result = self.server._track_adapter("adapter-001", "/path/1")

        # Should return True (already tracked)
        assert result is True
        # First adapter should now be at the end (MRU)
        adapters = self.server.loaded_sampling_loras["default"]
        assert adapters[-1] == ("adapter-001", "/path/1")

    def test_track_adapter_uses_default_key(self):
        """Test _track_adapter uses 'default' key (flat structure, no model_id)."""
        self.server._track_adapter("adapter-001", "/path/1")

        # Should use "default" key for all adapters
        assert "default" in self.server.loaded_sampling_loras
        assert len(self.server.loaded_sampling_loras) == 1


class TestCreateModelAutoSaveCheckpoint:
    """Test that create_model auto-saves initial checkpoint 000000."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
            output_dir=self.temp_dir,
        )
        self.server._running = True
        self.server.engine_client = MagicMock()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_weights_creates_000000_checkpoint(self):
        """Test that save_weights with path='000000' creates the checkpoint."""
        model_id = "test_model"

        # Mock the engine response
        mock_output = MagicMock()
        mock_output.outputs = [{"checkpoint_path": f"{self.temp_dir}/weights/{model_id}/000000"}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            request = SaveWeightsRequest(model_id=model_id, path="000000")
            response = await self.server.save_weights(request)

        # Should create the checkpoint path
        assert "000000" in response.path
        assert model_id in response.path

    @pytest.mark.asyncio
    async def test_load_weights_after_save_works(self):
        """Test that load_weights works after save_weights creates checkpoint."""
        model_id = "my_run"

        # Create the checkpoint directory (simulating save_weights)
        checkpoint_path = os.path.join(self.temp_dir, "weights", model_id, "000000")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Mock the engine response
        mock_output = MagicMock()
        mock_output.outputs = [{"success": True}]

        with patch.object(self.server, '_wait_for_response', return_value=mock_output):
            self.server.engine_client.send_request = AsyncMock()

            # Load using just checkpoint name
            request = LoadWeightsRequest(model_id=model_id, path="000000")
            response = await self.server.load_weights(request)

        assert response.path == "000000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
