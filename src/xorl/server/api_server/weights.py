"""Weight I/O and checkpoint management mixin."""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime
from typing import List

from fastapi import HTTPException, status

from xorl.server.api_server.api_types import (
    CheckpointInfo,
    Cursor,
    DeleteCheckpointRequest,
    DeleteCheckpointResponse,
    ListCheckpointsRequest,
    ListCheckpointsResponse,
    LoadWeightsRequest,
    LoadWeightsResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    TrainingRun,
    TrainingRunsResponse,
)
from xorl.server.api_server.utils import validate_model_id
from xorl.server.protocol.api_orchestrator import OrchestratorRequest
from xorl.server.protocol.operations import LoadStateData, SaveFullWeightsData, SaveLoraOnlyData, SaveStateData
from xorl.server.utils.storage import (
    StorageLimitError,
    check_storage_limit,
)


logger = logging.getLogger(__name__)


class WeightsMixin:
    """Mixin for weight I/O and checkpoint management."""

    # Reserved checkpoint name that cannot be deleted (initial model state)
    RESERVED_CHECKPOINT_NAME = "000000"

    def _to_xorl_uri(self, model_id: str, checkpoint_name: str, checkpoint_type: str = "weights") -> str:
        """Convert checkpoint name to xorl:// URI.

        Args:
            model_id: Model identifier
            checkpoint_name: Checkpoint name (e.g., "checkpoint-001")
            checkpoint_type: Type of checkpoint ("weights" or "sampler_weights")

        Returns:
            Xorl URI (e.g., "xorl://default/weights/checkpoint-001" or "xorl://default/sampler_weights/step-100")
        """
        return f"xorl://{model_id}/{checkpoint_type}/{checkpoint_name}"

    def _from_xorl_uri(self, uri: str) -> tuple[str, str, bool]:
        """Parse xorl:// URI or path to extract model_id and checkpoint name.

        Supported formats:
            - "xorl://model_id/weights/checkpoint_name" -> (model_id, checkpoint_name, True)
            - "weights/model_id/checkpoint_name" -> (model_id, checkpoint_name, True)
            - "weights/checkpoint_name" -> (None, checkpoint_name, False) [no explicit model_id]
            - "model_id/checkpoint_name" -> (model_id, checkpoint_name, True)
            - "checkpoint_name" -> (None, checkpoint_name, False)

        Args:
            uri: Xorl URI or checkpoint path

        Returns:
            Tuple of (model_id, checkpoint_name, has_explicit_model_id)
            - model_id: The model_id from the path, or None if not explicitly specified
            - checkpoint_name: The checkpoint name
            - has_explicit_model_id: True if the path explicitly contains a model_id
        """
        if uri.startswith("xorl://"):
            # Parse xorl://model_id/weights/checkpoint_name
            parts = uri[7:].split("/")  # Remove "xorl://"
            if len(parts) >= 3 and parts[1] == "weights":
                model_id = parts[0]
                checkpoint_name = "/".join(parts[2:])  # In case checkpoint name has /
                return model_id, checkpoint_name, True

        if uri.startswith("weights/"):
            # Parse weights/... format
            parts = uri.split("/")
            if len(parts) >= 3:
                # weights/model_id/checkpoint_name - explicit model_id
                model_id = parts[1]
                checkpoint_name = "/".join(parts[2:])
                return model_id, checkpoint_name, True
            elif len(parts) == 2:
                # weights/checkpoint_name - NO explicit model_id, use request.model_id
                checkpoint_name = parts[1]
                return None, checkpoint_name, False

        # Check for model_id/checkpoint_name format (e.g., "user_123/adapter-a")
        parts = uri.split("/")
        if len(parts) == 2:
            model_id = parts[0]
            checkpoint_name = parts[1]
            return model_id, checkpoint_name, True

        # If just a checkpoint name, no explicit model_id
        return None, uri, False

    async def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        """
        Save model weights (and optimizer state) to persistent storage (two-phase async pattern).

        Phase 1: Returns immediately with request_id.
        Phase 2: Client polls /api/v1/retrieve_future to get actual result.

        The checkpoint is saved under output_dir/weights/{model_id}/{checkpoint_name}.
        Returns a xorl:// URI that can be used with load_weights.

        Args:
            request: Save weights request

        Returns:
            SaveWeightsResponse with xorl:// URI of the saved checkpoint

        Raises:
            HTTPException: If checkpoint exists, server not running, or request validation fails
        """
        # Generate checkpoint name if not provided
        checkpoint_name = request.path
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{int(time.time())}"

        # Validate model_id to prevent path traversal and invalid characters
        # Returns "default" if model_id is None or empty
        model_id = validate_model_id(request.model_id)

        # Build the actual filesystem path under output_dir/weights/{model_id}/
        # This enables per-model checkpoint isolation for multi-tenancy
        checkpoint_path = os.path.join(self.output_dir, "weights", model_id, checkpoint_name)

        # Check if checkpoint already exists
        checkpoint_already_exists = os.path.exists(checkpoint_path)
        if checkpoint_already_exists:
            # Special case: "000000" is the initial checkpoint - allow overwriting by deleting first
            if checkpoint_name == self.RESERVED_CHECKPOINT_NAME:
                logger.info(
                    f"Initial checkpoint '{self.RESERVED_CHECKPOINT_NAME}' already exists for model_id={model_id}, "
                    f"deleting to save fresh state..."
                )
                try:
                    if os.path.isdir(checkpoint_path):
                        shutil.rmtree(checkpoint_path)
                    else:
                        os.remove(checkpoint_path)
                    logger.info(f"Deleted existing checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to delete existing checkpoint {checkpoint_path}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to delete existing initial checkpoint: {e}",
                    )
            else:
                # For other checkpoints, raise error - user must delete first
                xorl_uri = self._to_xorl_uri(model_id, checkpoint_name)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Checkpoint already exists: {xorl_uri}. "
                    f"To overwrite, delete it first using /api/v1/delete_checkpoint "
                    f"with checkpoint_id='weights/{model_id}/{checkpoint_name}'.",
                )

        self._require_engine()

        # Check storage limit before saving
        try:
            check_storage_limit(self.output_dir, self.storage_limit)
        except StorageLimitError as e:
            logger.error(f"Storage limit exceeded: {e}")
            raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=f"StorageLimitError: {str(e)}")

        try:
            # Create engine request - always save optimizer state for full checkpointing
            # Use validated model_id for consistency (same as used for checkpoint_path)
            engine_request = OrchestratorRequest(
                operation="save_state",
                payload=SaveStateData(
                    checkpoint_path=checkpoint_path,
                    save_optimizer=True,
                    use_timestamp=False,
                    model_id=model_id,
                ),
            )

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            # Wait for output with timeout and proper cleanup
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Save weights timeout"
            )

            # Extract results and build xorl:// URI
            result = output.outputs[0] if output.outputs else {}
            saved_path = result.get("checkpoint_path", checkpoint_path)

            # Extract checkpoint name from saved path for URI
            # The saved_path is the full filesystem path: output_dir/weights/model_id/checkpoint_name
            # We want just the checkpoint_name part (not including model_id directory)
            # because _to_xorl_uri will add the model_id prefix
            #
            # Normalize paths to absolute paths for reliable comparison
            # This handles cases where output_dir is relative but saved_path might be absolute or vice versa
            weights_model_dir = os.path.normpath(os.path.abspath(os.path.join(self.output_dir, "weights", model_id)))
            saved_path_normalized = os.path.normpath(os.path.abspath(saved_path))

            if (
                saved_path_normalized.startswith(weights_model_dir + os.sep)
                or saved_path_normalized == weights_model_dir
            ):
                # Extract just the checkpoint name (relative to model_id directory)
                checkpoint_name_from_path = os.path.relpath(saved_path_normalized, weights_model_dir)
            else:
                # Fallback: use the original checkpoint_name from the request
                # This is safer than basename which could lose nested directory structure
                checkpoint_name_from_path = checkpoint_name

            xorl_uri = self._to_xorl_uri(model_id, checkpoint_name_from_path)
            logger.info(
                f"Checkpoint saved: {xorl_uri} (path: {saved_path}, "
                f"weights_model_dir: {weights_model_dir}, "
                f"checkpoint_name_from_path: {checkpoint_name_from_path})"
            )

            return SaveWeightsResponse(path=xorl_uri)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Save weights failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Save weights failed: {e}")

    async def load_weights(self, request: LoadWeightsRequest) -> LoadWeightsResponse:
        """
        Load model weights from a saved checkpoint.

        Supports loading checkpoints from any model_id, not just the current one.
        This allows loading checkpoints from other training runs.

        Supported path formats:
            - "weights/model_id/checkpoint_name" -> loads from specific model_id
            - "xorl://model_id/weights/checkpoint_name" -> loads from specific model_id
            - "model_id/checkpoint_name" -> loads from specific model_id
            - "checkpoint_name" -> loads from request.model_id (or "default")

        Args:
            request: Load weights request with path and optional model_id

        Returns:
            LoadWeightsResponse with the path that was loaded

        Raises:
            HTTPException: If server not running, checkpoint not found, or operation fails
        """
        # Parse path to get model_id and checkpoint_name
        # Returns (model_id_from_path, checkpoint_name, has_explicit_model_id)
        uri_model_id, checkpoint_name, has_explicit_model_id = self._from_xorl_uri(request.path)

        # Determine which model_id to use for the checkpoint path:
        # - If path explicitly contains model_id (weights/X/Y/..., xorl://X/..., X/Y), use that
        # - Otherwise, fall back to request.model_id (the client's session)
        #
        # Examples:
        #   "000000" with model_id="run-2" -> weights/run-2/000000
        #   "weights/000000" with model_id="run-2" -> weights/run-2/000000
        #   "weights/default/000000" with model_id="run-2" -> weights/default/000000
        #   "xorl://default/weights/000000" with model_id="run-2" -> weights/default/000000
        if has_explicit_model_id:
            # Use model_id from the path
            model_id_raw = uri_model_id
        else:
            # Use model_id from request (the client's training session)
            model_id_raw = request.model_id

        # Validate model_id to prevent path traversal and invalid characters
        # Returns "default" if model_id is None or empty
        model_id = validate_model_id(model_id_raw)

        # Build the actual filesystem path under output_dir/weights/{model_id}/
        checkpoint_path = os.path.join(self.output_dir, "weights", model_id, checkpoint_name)

        # Check if checkpoint exists before attempting to load
        # This check happens before server running check to give better error messages
        if not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint not found: {request.path} (resolved to {checkpoint_path}). "
                f"Use /api/v1/list_checkpoints to see available checkpoints.",
            )

        self._require_engine()

        try:
            # Create engine request
            # Note: We use request.model_id (client's session) for engine request ordering,
            # NOT the resolved model_id from the checkpoint path. This ensures the load
            # operation is ordered correctly within the client's session (e.g., between
            # forward_backward and optim_step calls from the same client).
            # The resolved model_id is only used for the filesystem path.
            engine_request = OrchestratorRequest(
                operation="load_state",
                payload=LoadStateData(
                    checkpoint_path=checkpoint_path,
                    load_optimizer=request.optimizer,
                    model_id=request.model_id,
                ),
            )

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            # Wait for output with timeout and proper cleanup
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Load weights timeout"
            )

            # Extract results
            result = output.outputs[0] if output.outputs else {}
            success = result.get("success", False)

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load checkpoint: {request.path}",
                )

            logger.info(f"Checkpoint loaded: {request.path} (optimizer={request.optimizer})")

            return LoadWeightsResponse(path=request.path)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Load weights failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Load weights failed: {e}")

    def _get_directory_size(self, path: str) -> int:
        """Get total size of a directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
        return total_size

    def _scan_checkpoints(
        self,
        scan_dir: str,
        checkpoint_type: str,
        make_id_fn,
        make_path_fn,
    ) -> List[CheckpointInfo]:
        """Scan a directory for checkpoints.

        Args:
            scan_dir: Directory to scan for checkpoint subdirectories
            checkpoint_type: Type string for CheckpointInfo ("training" or "sampler")
            make_id_fn: Callable(entry_name) -> checkpoint_id string
            make_path_fn: Callable(entry_name) -> path string

        Returns:
            List of CheckpointInfo objects sorted by time (newest first)
        """

        checkpoints = []
        if not os.path.exists(scan_dir):
            return checkpoints

        try:
            entries = os.listdir(scan_dir)
        except OSError as e:
            logger.warning(f"Failed to list directory {scan_dir}: {e}")
            return checkpoints

        for entry in entries:
            entry_path = os.path.join(scan_dir, entry)
            if os.path.isdir(entry_path):
                try:
                    mtime = os.path.getmtime(entry_path)
                    time_str = datetime.fromtimestamp(mtime).isoformat()
                    size_bytes = self._get_directory_size(entry_path)

                    checkpoints.append(
                        CheckpointInfo(
                            checkpoint_id=make_id_fn(entry),
                            checkpoint_type=checkpoint_type,
                            time=time_str,
                            path=make_path_fn(entry),
                            size_bytes=size_bytes,
                        )
                    )
                except OSError as e:
                    logger.warning(f"Failed to get info for checkpoint {entry_path}: {e}")
                    continue

        checkpoints.sort(key=lambda c: c.time, reverse=True)
        return checkpoints

    def _scan_training_checkpoints(self, model_id: str) -> List[CheckpointInfo]:
        """Scan weights directory for training checkpoints."""
        return self._scan_checkpoints(
            scan_dir=os.path.join(self.output_dir, "weights", model_id),
            checkpoint_type="training",
            make_id_fn=lambda entry: f"weights/{model_id}/{entry}",
            make_path_fn=lambda entry: f"xorl://{model_id}/weights/{entry}",
        )

    def _scan_sampler_checkpoints(self) -> List[CheckpointInfo]:
        """Scan sampler_weights directory for sampler checkpoints."""
        return self._scan_checkpoints(
            scan_dir=os.path.join(self.output_dir, "sampler_weights"),
            checkpoint_type="sampler",
            make_id_fn=lambda entry: f"sampler_weights/{entry}",
            make_path_fn=lambda entry: f"sampler_weights/{entry}",
        )

    async def list_checkpoints(
        self,
        request: ListCheckpointsRequest,
    ) -> ListCheckpointsResponse:
        """List all available checkpoints.

        Scans the output directory for both training checkpoints (weights/{model_id}/)
        and sampler checkpoints (sampler_weights/ - flat, no model_id).

        Args:
            request: List checkpoints request with model_id (used for training checkpoints only)

        Returns:
            ListCheckpointsResponse with list of checkpoints

        Raises:
            HTTPException: If operation fails
        """
        try:
            # Validate model_id to prevent path traversal and invalid characters
            # Returns "default" if model_id is None or empty
            model_id = validate_model_id(request.model_id)

            all_checkpoints = []

            # Scan training checkpoints (weights/{model_id}/)
            training_checkpoints = self._scan_training_checkpoints(model_id)
            all_checkpoints.extend(training_checkpoints)

            # Scan sampler checkpoints (sampler_weights/ - flat, shared across all model_ids)
            sampler_checkpoints = self._scan_sampler_checkpoints()
            all_checkpoints.extend(sampler_checkpoints)

            # Sort all by time (newest first)
            all_checkpoints.sort(key=lambda c: c.time, reverse=True)

            logger.info(
                f"Listed {len(all_checkpoints)} checkpoints "
                f"({len(training_checkpoints)} training, {len(sampler_checkpoints)} sampler)"
            )

            return ListCheckpointsResponse(checkpoints=all_checkpoints)

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list checkpoints: {str(e)}"
            )

    def list_training_runs(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> TrainingRunsResponse:
        """List training runs.

        Note: In xorl_client, there is only a single training run with model_id="default".
        This endpoint returns that single training run for API compatibility with tinker.

        Args:
            limit: Maximum number of training runs to return (default 20)
            offset: Offset for pagination (default 0)

        Returns:
            TrainingRunsResponse with the list of training runs and pagination info
        """

        # Build list of training runs from registered model_ids
        training_runs = []

        # Sampler checkpoints are shared across all model_ids — scan once
        sampler_checkpoints = self._scan_sampler_checkpoints()
        last_sampler_checkpoint = None
        if sampler_checkpoints:
            last_sampler = sampler_checkpoints[0]
            last_sampler_checkpoint = CheckpointInfo(
                checkpoint_id=last_sampler.checkpoint_id,
                checkpoint_type=last_sampler.checkpoint_type,
                time=last_sampler.time,
                path=last_sampler.path,
                size_bytes=last_sampler.size_bytes,
            )

        for model_id in self.registered_model_ids:
            # Get model config if available
            model_config = self.model_configs.get(model_id, {})
            base_model = model_config.get("base_model", self.base_model or "unknown")
            lora_config = model_config.get("lora_config", {})
            lora_rank = lora_config.get("lora_rank")

            # Get last checkpoint info
            last_checkpoint = None

            # Scan for training checkpoints
            training_checkpoints = self._scan_training_checkpoints(model_id)
            if training_checkpoints:
                last_ckpt = training_checkpoints[0]  # Sorted by time, newest first
                last_checkpoint = CheckpointInfo(
                    checkpoint_id=last_ckpt.checkpoint_id,
                    checkpoint_type=last_ckpt.checkpoint_type,
                    time=last_ckpt.time,
                    path=last_ckpt.path,
                    size_bytes=last_ckpt.size_bytes,
                )

            training_run = TrainingRun(
                training_run_id=model_id,
                base_model=base_model,
                model_owner="local",
                is_lora=True,
                corrupted=False,
                lora_rank=lora_rank,
                last_request_time=datetime.now().isoformat(),
                last_checkpoint=last_checkpoint,
                last_sampler_checkpoint=last_sampler_checkpoint,
            )
            training_runs.append(training_run)

        # Apply pagination
        total_count = len(training_runs)
        paginated_runs = training_runs[offset : offset + limit]

        cursor = Cursor(
            offset=offset,
            limit=limit,
            total_count=total_count,
        )

        return TrainingRunsResponse(
            training_runs=paginated_runs,
            cursor=cursor,
        )

    async def delete_checkpoint(
        self,
        request: DeleteCheckpointRequest,
    ) -> DeleteCheckpointResponse:
        """Delete a checkpoint.

        Args:
            request: Delete checkpoint request with model_id and checkpoint_id
                - For training checkpoints: checkpoint_id = "weights/{model_id}/{name}"
                - For sampler checkpoints: checkpoint_id = "sampler_weights/{name}"

        Returns:
            DeleteCheckpointResponse with success status

        Raises:
            HTTPException: If checkpoint not found, is reserved, or deletion fails
        """
        try:
            checkpoint_id = request.checkpoint_id
            parts = checkpoint_id.split("/")

            if len(parts) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid checkpoint_id format: {checkpoint_id}. "
                    f"Expected 'weights/{{model_id}}/{{name}}' or 'sampler_weights/{{name}}'",
                )

            type_dir = parts[0]

            if type_dir == "weights":
                # Training checkpoints: weights/{model_id}/{name}
                if len(parts) < 3:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid checkpoint_id format: {checkpoint_id}. "
                        f"Expected 'weights/{{model_id}}/{{name}}'",
                    )
                checkpoint_model_id = parts[1]
                name = "/".join(parts[2:])
                checkpoint_path = os.path.join(self.output_dir, "weights", checkpoint_model_id, name)
                deleted_path = f"xorl://{checkpoint_model_id}/weights/{name}"

            elif type_dir == "sampler_weights":
                # Sampler checkpoints: sampler_weights/{name} (flat, no model_id)
                name = "/".join(parts[1:])
                checkpoint_path = os.path.join(self.output_dir, "sampler_weights", name)
                deleted_path = f"sampler_weights/{name}"

            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid checkpoint type: {type_dir}. Expected 'weights' or 'sampler_weights'",
                )

            # Check if this is the reserved initial checkpoint (only for training checkpoints)
            if type_dir == "weights" and name == self.RESERVED_CHECKPOINT_NAME:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Cannot delete checkpoint '{self.RESERVED_CHECKPOINT_NAME}': "
                    f"This is the reserved initial checkpoint that preserves the original model state. "
                    f"It cannot be deleted.",
                )

            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint not found: {checkpoint_id}"
                )

            # Delete the checkpoint directory
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            else:
                os.remove(checkpoint_path)

            logger.info(f"Deleted checkpoint: {deleted_path} (path: {checkpoint_path})")

            return DeleteCheckpointResponse(
                success=True,
                deleted_path=deleted_path,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}", exc_info=True)
            return DeleteCheckpointResponse(
                success=False,
                error=str(e),
            )

    async def save_weights_for_sampler(self, request: SaveWeightsForSamplerRequest) -> SaveWeightsForSamplerResponse:
        """
        Save model weights for sampling/inference (two-phase async pattern).

        Supports both LoRA and full-weights training modes:
        - LoRA mode: saves adapter weights in PEFT-compatible format
        - Full-weights mode: saves full model as safetensors (HF-compatible)

        The checkpoint is saved under output_dir/sampler_weights/{name}/ (flat, no model_id).
        This is separate from save_weights which saves under output_dir/weights/{model_id}/{name}/.

        Args:
            request: Save weights for sampler request with name

        Returns:
            SaveWeightsForSamplerResponse with xorl:// URI of the saved checkpoint

        Raises:
            HTTPException: If server not running or request validation fails
        """
        self._require_engine()

        # Check storage limit before saving
        try:
            check_storage_limit(self.output_dir, self.storage_limit)
        except StorageLimitError as e:
            logger.error(f"Storage limit exceeded: {e}")
            raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=f"StorageLimitError: {str(e)}")

        try:
            # Build the actual filesystem path under output_dir/sampler_weights/
            # All sampler weights are saved in the same location (no model_id subdirectory)
            # because inference endpoints don't know about model_id
            save_path = os.path.join(self.output_dir, "sampler_weights", request.name)

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Check if weights with this name already exist
            if os.path.exists(save_path):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Sampler weights with name '{request.name}' already exist. Use a unique name or delete it first.",
                )

            # Determine training mode from model config
            model_config = self.model_configs.get(request.model_id, {})
            lora_config = model_config.get("lora_config", {})
            is_lora = lora_config.get("enable_lora", False) or "rank" in lora_config
            merge_lora_interval = lora_config.get("merge_lora_interval", 0)

            if is_lora and merge_lora_interval == 0:
                # LoRA with no merge: base weights unchanged, save adapter only
                engine_request = OrchestratorRequest(
                    operation="save_lora_only",
                    payload=SaveLoraOnlyData(lora_path=save_path, model_id=request.model_id),
                )
            else:
                # Full-weights mode (no LoRA) or LoRA with merge_interval > 0
                # (base weights modified by periodic merges, need full model for inference)
                base_model_path = model_config.get("base_model") or self.base_model
                engine_request = OrchestratorRequest(
                    operation="save_full_weights",
                    payload=SaveFullWeightsData(
                        output_path=save_path,
                        dtype="bfloat16",
                        base_model_path=base_model_path,
                        model_id=request.model_id,
                    ),
                )

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            # Wait for output with timeout and proper cleanup
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Save weights for sampler timeout"
            )

            # For LoRA with merge_interval > 0, also save LoRA weights for training recovery
            if is_lora and merge_lora_interval > 0:
                lora_save_path = os.path.join(save_path, "lora")
                lora_request = OrchestratorRequest(
                    operation="save_lora_only",
                    payload=SaveLoraOnlyData(lora_path=lora_save_path, model_id=request.model_id),
                )
                lora_future = await self.orchestrator_client.send_request(lora_request)
                await self._wait_for_response(
                    lora_future, lora_request.request_id, self.default_timeout, "Save LoRA weights timeout"
                )

            # Extract results
            result = output.outputs[0] if output.outputs else {}
            saved_path = result.get("lora_path", save_path) if (is_lora and merge_lora_interval == 0) else save_path

            # Validate model_id for xorl_client URI
            model_id = validate_model_id(request.model_id)

            # Build xorl:// URI for the saved checkpoint
            xorl_uri = self._to_xorl_uri(model_id, request.name, "sampler_weights")

            if is_lora and merge_lora_interval == 0:
                save_format = "PEFT LoRA"
            elif is_lora and merge_lora_interval > 0:
                save_format = "safetensors (full weights) + PEFT LoRA"
            else:
                save_format = "safetensors (full weights)"
            logger.info(f"Sampler weights saved ({save_format}): {xorl_uri} (path: {saved_path})")

            return SaveWeightsForSamplerResponse(path=xorl_uri)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Save weights for sampler failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Save weights for sampler failed: {e}"
            )
