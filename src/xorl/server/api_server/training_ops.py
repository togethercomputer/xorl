"""Training operations mixin: two-phase async pattern, forward, backward, optim step."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

from xorl.server.api_server.api_types import (
    AdamParams,
    CreateSamplingSessionRequest,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    ForwardRequest,
    ForwardResponse,
    FutureRetrieveRequest,
    LoadWeightsRequest,
    OptimStepRequest,
    OptimStepResponse,
    RequestFailedResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SyncInferenceWeightsRequest,
    TryAgainResponse,
    UntypedAPIFuture,
    ZORLAbortGenerationRequest,
    ZORLAbortGenerationResponse,
    ZORLApplyRewardsRequest,
    ZORLApplyRewardsResponse,
    ZORLCandidateInfo,
    ZORLStartGenerationRequest,
    ZORLStartGenerationResponse,
)
from xorl.server.api_server.future_store import (
    FutureStatus,
)
from xorl.server.api_server.utils import (
    validate_model_id,
)
from xorl.server.protocol.api_orchestrator import OrchestratorRequest
from xorl.server.protocol.operations import (
    ModelPassData,
    OptimStepData,
    ZORLAbortGenerationData,
    ZORLApplyRewardsData,
    ZORLStartGenerationData,
)


logger = logging.getLogger(__name__)

PROFILE_TIMING_METRIC_KEYS = {
    "backward_compute_time",
    "forward_compute_time",
}
PROFILE_TIMING_METRIC_PREFIXES = ("server_profile_",)


def _sanitize_nan_to_zero(data):
    """Replace NaN/Inf floats with 0.0 recursively (JSON-safe, Pydantic-safe)."""
    if isinstance(data, dict):
        return {k: _sanitize_nan_to_zero(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_nan_to_zero(v) for v in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return 0.0
    return data


class TrainingOpsMixin:
    """Mixin for two-phase async pattern and core training operations."""

    def _session_default_learning_rate(self, model_id: str) -> Optional[float]:
        """Return the registered session's default optimizer learning rate, if any."""
        model_configs = getattr(self, "model_configs", {})
        model_config = model_configs.get(model_id) or {}
        optimizer_config = model_config.get("optimizer_config") or {}
        if isinstance(optimizer_config, dict):
            learning_rate = optimizer_config.get("learning_rate", optimizer_config.get("lr"))
            if learning_rate is not None:
                return float(learning_rate)
        return None

    def _server_default_learning_rate(self) -> Optional[float]:
        """Return the server train-config learning rate for full-weight sessions."""
        train_config = getattr(self, "train_config", {}) or {}
        if not isinstance(train_config, dict):
            return None
        learning_rate = train_config.get("learning_rate", train_config.get("lr"))
        return float(learning_rate) if learning_rate is not None else None

    def _optim_step_learning_rate(self, request: OptimStepRequest) -> float:
        """Resolve the effective LR for an optim_step request.

        Priority: request.learning_rate, request.adam_params.learning_rate,
        per-session optimizer_config, server train_config. If the session was
        explicitly registered (even without an optimizer_config) fall back to
        AdamParams().learning_rate; otherwise raise so a missing LR fails loud.
        """
        if getattr(request, "learning_rate", None) is not None:
            return float(request.learning_rate)

        fields_set = getattr(request, "model_fields_set", set())
        adam_params = getattr(request, "adam_params", None)
        if "adam_params" in fields_set and adam_params is not None and adam_params.learning_rate is not None:
            return float(adam_params.learning_rate)

        session_lr = self._session_default_learning_rate(request.model_id)
        if session_lr is not None:
            return session_lr

        server_lr = self._server_default_learning_rate()
        if server_lr is not None:
            return server_lr

        if request.model_id in getattr(self, "model_configs", {}):
            return AdamParams().learning_rate

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "optim_step: no learning_rate in request, no default optimizer_config "
                f"registered for model_id={request.model_id!r}, and no server train_config lr"
            ),
        )

    # =========================================================================
    # Two-Phase Request Pattern Methods
    # =========================================================================

    async def retrieve_future(self, request: FutureRetrieveRequest, timeout: float = 45.0):
        """
        Retrieve the result of an async operation (Phase 2 of two-phase pattern).

        Uses long polling: holds the connection for up to `timeout` seconds waiting
        for the result to become available. This reduces polling frequency and
        latency compared to immediate returns.

        Returns different response types depending on request state:
        - TryAgainResponse: Request still processing after timeout, client should retry
        - RequestFailedResponse: Request failed with error
        - Actual result type: Request completed successfully

        Args:
            request: FutureRetrieveRequest containing request_id
            timeout: Maximum time to wait for result (default: 45s like Tinker)

        Returns:
            FutureRetrieveResponse (TryAgainResponse, RequestFailedResponse, or result)

        Raises:
            HTTPException: 404 if request_id not found, 503 if store not initialized
        """
        if not self.future_store:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Future store not initialized")

        # Use long polling: wait for result with timeout
        entry = await self.future_store.wait_for_result(request.request_id, timeout=timeout)

        if entry is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Request {request.request_id} not found")

        # Return appropriate response based on status
        if entry.status == FutureStatus.PENDING:
            # Request is queued, waiting for worker capacity (timeout reached)
            return TryAgainResponse(
                request_id=request.request_id,
                queue_state="paused_capacity",
                queue_state_reason="Request is queued, waiting for worker capacity",
            )

        elif entry.status == FutureStatus.PROCESSING:
            # Request is actively being processed (timeout reached while processing)
            return TryAgainResponse(
                request_id=request.request_id,
                queue_state="active",
                queue_state_reason=None,
            )

        elif entry.status == FutureStatus.FAILED:
            # Request failed
            return RequestFailedResponse(
                error=entry.error or "Unknown error",
                category=entry.error_category,
            )

        elif entry.status == FutureStatus.EXPIRED:
            # Request expired before completion
            return RequestFailedResponse(
                error=f"Request {request.request_id} expired",
                category="server",
            )

        elif entry.status == FutureStatus.COMPLETED:
            # Request completed - return the result directly
            return entry.result

        else:
            # Unknown status
            return RequestFailedResponse(
                error=f"Unknown request status: {entry.status}",
                category="server",
            )

    async def _submit_async(self, request, request_type: str, handler_method: str) -> UntypedAPIFuture:
        """Submit a request for async processing (Phase 1 of two-phase pattern).

        Generic helper that replaces the per-endpoint submit_*_async methods.
        Returns immediately with an UntypedAPIFuture containing request_id.
        Client should poll /api/v1/retrieve_future to get the result.

        Args:
            request: The API request object (must have model_id and model_dump())
            request_type: Type string for the future store (e.g., "forward_backward")
            handler_method: Name of the handler method on self (e.g., "forward_backward")

        Returns:
            UntypedAPIFuture with request_id for polling
        """
        self._require_engine()
        if not self.future_store:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Future store not initialized")

        model_id = validate_model_id(request.model_id)
        self.validate_model_id(model_id)
        self._update_session_activity(model_id)

        handler = getattr(self, handler_method)
        request_class = type(request)

        async def process(request_data: Dict[str, Any]) -> Dict[str, Any]:
            result = await handler(request_class(**request_data))
            return _sanitize_nan_to_zero(result.model_dump(exclude_none=True))

        request_id = await self.future_store.create(
            model_id=model_id,
            request_type=request_type,
            process_fn=process,
            request_data=request.model_dump(),
        )

        return UntypedAPIFuture(request_id=request_id, model_id=model_id)

    async def submit_forward_backward_async(self, request: ForwardBackwardRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "forward_backward", "forward_backward")

    async def submit_forward_async(self, request: ForwardRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "forward", "forward")

    async def submit_optim_step_async(self, request: OptimStepRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "optim_step", "optim_step")

    async def submit_save_weights_async(self, request: SaveWeightsRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "save_weights", "save_weights")

    async def submit_load_weights_async(self, request: LoadWeightsRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "load_weights", "load_weights")

    async def submit_save_weights_for_sampler_async(self, request: SaveWeightsForSamplerRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "save_weights_for_sampler", "save_weights_for_sampler")

    async def submit_start_zorl_generation_async(self, request: ZORLStartGenerationRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "start_zorl_generation", "start_zorl_generation")

    async def submit_apply_zorl_rewards_async(self, request: ZORLApplyRewardsRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "apply_zorl_rewards", "apply_zorl_rewards")

    async def submit_abort_zorl_generation_async(self, request: ZORLAbortGenerationRequest) -> UntypedAPIFuture:
        return await self._submit_async(request, "abort_zorl_generation", "abort_zorl_generation")

    # =========================================================================
    # Original Synchronous Methods
    # =========================================================================

    async def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        """
        Execute forward-backward pass.

        Args:
            request: Forward-backward request

        Returns:
            Forward-backward response with loss and metrics

        Raises:
            HTTPException: If server not running or operation fails
        """
        t_start = time.perf_counter()
        self._require_engine()

        try:
            data = self._flatten_api_data(request.forward_backward_input.data)

            # Create engine request
            # Note: Executor will pack data into batches based on dp_size
            # Pass seq_id and model_id for request ordering (SeqIdAwareFIFOPolicy)
            # Pass routed_experts for R3 routing replay if provided
            engine_request = OrchestratorRequest(
                operation="forward_backward",
                payload=ModelPassData(
                    data=data,
                    loss_fn=request.forward_backward_input.loss_fn,
                    loss_fn_params=request.forward_backward_input.loss_fn_params,
                    model_id=request.model_id,
                    routed_experts=request.forward_backward_input.routed_experts,
                    routed_expert_logits=request.forward_backward_input.routed_expert_logits,
                ),
                seq_id=request.seq_id,
            )

            t_engine_submit = time.perf_counter()

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            t_engine_submitted = time.perf_counter()

            # Wait for output with timeout and proper cleanup
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Forward-backward timeout"
            )

            t_engine_done = time.perf_counter()

            # Extract results
            result = output.outputs[0] if output.outputs else {}

            # Debug: Log what we got from the engine
            logger.debug(f"API Server: Received result from engine, keys: {list(result.keys())}")
            loss_metrics = {k: v for k, v in result.items() if k.startswith(("is_", "opd_"))}
            if loss_metrics:
                logger.debug(f"API Server: loss metrics present in result: {list(loss_metrics.keys())}")
            else:
                logger.debug("API Server: No loss metrics in result")

            # Sanitize NaN/Inf values for JSON serialization
            result = _sanitize_nan_to_zero(result)

            loss_fn_outputs, loss_fn_output_type = self._build_loss_fn_outputs(result)

            # Build metrics with tinker naming convention
            total_loss = result.get("loss", 0.0)
            valid_tokens = result.get("valid_tokens", 1)
            metrics = {
                "loss:sum": total_loss * valid_tokens,
                "loss:mean": total_loss,
                "valid_tokens:sum": valid_tokens,
                "execution_time:sum": result.get("execution_time", result.get("forward_backward_time", 0.0)),
            }

            # Add loss-specific metrics if present (already have name:reduction format)
            for key, value in result.items():
                if key.startswith(("is_", "opd_")):
                    # Ensure colon format for tinker compatibility
                    metrics[key if ":" in key else f"{key}:mean"] = value
                elif key in (
                    "teacher_prefill_tokens",
                    "teacher_prefill_forward_compute_s",
                    "teacher_hidden_cache_write_s",
                ):
                    metrics[key] = value
                elif (
                    key.startswith("executor_")
                    or key in PROFILE_TIMING_METRIC_KEYS
                    or key.startswith(PROFILE_TIMING_METRIC_PREFIXES)
                ):
                    metrics[key] = value

            # Pass through expert load summary for MoE models
            if "expert_load_summary" in result:
                metrics["expert_load_summary"] = result["expert_load_summary"]

            info = self._build_info(result)

            t_end = time.perf_counter()
            logger.info(
                f"[TIMING] forward_backward: "
                f"build_request={t_engine_submit - t_start:.4f}s "
                f"zmq_send={t_engine_submitted - t_engine_submit:.4f}s "
                f"engine_wait={t_engine_done - t_engine_submitted:.4f}s "
                f"build_response={t_end - t_engine_done:.4f}s "
                f"total={t_end - t_start:.4f}s"
            )

            return ForwardBackwardResponse(
                loss_fn_output_type=loss_fn_output_type,
                loss_fn_outputs=loss_fn_outputs,
                metrics=metrics,
                info=info,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Forward-backward failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Forward-backward failed: {e}"
            )

    async def forward(self, request: ForwardRequest) -> ForwardResponse:
        """
        Execute forward pass (no gradient computation) for validation.

        Args:
            request: Forward request

        Returns:
            Forward response with loss and metrics (same format as forward_backward)

        Raises:
            HTTPException: If server not running or operation fails
        """
        self._require_engine()

        try:
            data = self._flatten_api_data(request.forward_input.data)

            # Create engine request
            engine_request = OrchestratorRequest(
                operation="forward",
                payload=ModelPassData(
                    data=data,
                    loss_fn=request.forward_input.loss_fn,
                    loss_fn_params=request.forward_input.loss_fn_params,
                    model_id=request.model_id,
                    routed_experts=request.forward_input.routed_experts,
                    routed_expert_logits=request.forward_input.routed_expert_logits,
                ),
                seq_id=request.seq_id,
            )

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            # Wait for output with timeout
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Forward timeout"
            )

            # Extract results (same format as forward_backward)
            result = _sanitize_nan_to_zero(output.outputs[0] if output.outputs else {})

            loss_fn_outputs, loss_fn_output_type = self._build_loss_fn_outputs(result)

            total_loss = result.get("loss", 0.0)
            valid_tokens = result.get("valid_tokens", 1)
            metrics = {
                "loss:sum": total_loss * valid_tokens,
                "loss:mean": total_loss,
                "valid_tokens": valid_tokens,
                "execution_time": result.get("execution_time", 0.0),
            }
            for key, value in result.items():
                if key.startswith(("is_", "opd_")):
                    metrics[key if ":" in key else f"{key}:mean"] = value
                elif key in (
                    "teacher_prefill_tokens",
                    "teacher_prefill_forward_compute_s",
                    "teacher_hidden_cache_write_s",
                ):
                    metrics[key] = value
                elif (
                    key.startswith("executor_")
                    or key in PROFILE_TIMING_METRIC_KEYS
                    or key.startswith(PROFILE_TIMING_METRIC_PREFIXES)
                ):
                    metrics[key] = value
            for key in (
                "teacher_prefill_tokens",
                "teacher_prefill_forward_compute_s",
                "teacher_hidden_cache_write_s",
            ):
                if key in result:
                    metrics[key] = result[key]

            return ForwardResponse(
                loss_fn_output_type=loss_fn_output_type,
                loss_fn_outputs=loss_fn_outputs,
                metrics=metrics,
                info=self._build_info(result),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Forward failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Forward failed: {e}")

    async def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        """
        Execute optimizer step.

        Args:
            request: Optimizer step request

        Returns:
            Optimizer step response with metrics

        Raises:
            HTTPException: If server not running or operation fails
        """
        t_start = time.perf_counter()
        self._require_engine()

        try:
            adam_params = request.adam_params
            lr = self._optim_step_learning_rate(request)

            # Determine gradient clipping value
            # Priority: explicit gradient_clip parameter, then adam_params.grad_clip_norm
            gradient_clip = request.gradient_clip
            if gradient_clip is None and adam_params is not None and adam_params.grad_clip_norm > 0:
                gradient_clip = adam_params.grad_clip_norm

            # Create engine request
            # Pass seq_id and model_id for request ordering (SeqIdAwareFIFOPolicy)
            engine_request = OrchestratorRequest(
                operation="optim_step",
                payload=OptimStepData(
                    lr=lr,
                    gradient_clip=gradient_clip,
                    beta1=adam_params.beta1 if adam_params is not None else None,
                    beta2=adam_params.beta2 if adam_params is not None else None,
                    eps=adam_params.eps if adam_params is not None else None,
                    model_id=request.model_id,
                    sparse_delta_capture=request.sparse_delta_capture,
                ),
                seq_id=request.seq_id,
            )

            t_engine_submit = time.perf_counter()

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            t_engine_submitted = time.perf_counter()

            # Wait for output with timeout and proper cleanup
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Optimizer step timeout"
            )

            t_engine_done = time.perf_counter()

            # Extract results
            result = output.outputs[0] if output.outputs else {}

            info = self._build_info(result)

            t_end = time.perf_counter()
            logger.info(
                f"[TIMING] optim_step: "
                f"build_request={t_engine_submit - t_start:.4f}s "
                f"zmq_send={t_engine_submitted - t_engine_submit:.4f}s "
                f"engine_wait={t_engine_done - t_engine_submitted:.4f}s "
                f"build_response={t_end - t_engine_done:.4f}s "
                f"total={t_end - t_start:.4f}s"
            )

            grad_norm = _sanitize_nan_to_zero(result.get("grad_norm", 0.0))
            response_learning_rate = result.get("learning_rate", result.get("lr", lr))
            metrics = {
                "grad_norm": grad_norm,
                "learning_rate": response_learning_rate,
                "step": result.get("step", 0),
            }
            for key in ("optim_step_time", "optim_empty_cache_skipped"):
                if key in result:
                    metrics[key] = result[key]

            return OptimStepResponse(
                metrics=metrics,
                info=info,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Optimizer step failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Optimizer step failed: {e}")

    def _zorl_candidate_uri(self, model_id: str, candidate_path: str) -> tuple[str, str]:
        """Convert a worker-exported candidate path into a public sampler URI."""
        sampler_root = os.path.abspath(os.path.join(self.output_dir, "sampler_weights"))
        absolute_candidate_path = os.path.abspath(candidate_path)
        if not (absolute_candidate_path == sampler_root or absolute_candidate_path.startswith(sampler_root + os.sep)):
            raise ValueError(f"ZORL candidate path {candidate_path!r} is outside sampler_weights root {sampler_root!r}")

        lora_name = os.path.relpath(absolute_candidate_path, sampler_root)
        return self._to_xorl_uri(model_id, lora_name, "sampler_weights"), lora_name

    def _zorl_generation_lora_names(self, *, model_id: str, generation_id: str) -> list[str]:
        """Return tracked sampling adapters belonging to one ZORL generation."""
        generation_prefix = f"zorl/{generation_id}/"
        return [
            str(lora_name)
            for lora_name, _lora_path in self.loaded_sampling_loras.get(model_id, [])
            if str(lora_name).startswith(generation_prefix)
        ]

    async def _cleanup_zorl_generation_sampling(self, *, model_id: str, generation_id: str) -> int:
        """Best-effort unload of tracked inference adapters for one ZORL generation.

        Holds the per-model_id sampling-loras lock across both the unload
        awaits and the dict mutation. Without the lock, a concurrent
        start_zorl_generation / cleanup on the same model_id could read the
        same `loaded_sampling_loras[model_id]` list, produce two divergent
        filtered lists, and have one overwrite the other — silently leaking
        SGLang adapter slots.
        """
        lock = self._sampling_loras_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            generation_lora_names = self._zorl_generation_lora_names(model_id=model_id, generation_id=generation_id)
            if not generation_lora_names:
                return 0

            for lora_name in generation_lora_names:
                try:
                    await self._unload_lora_on_inference_endpoints(lora_name)
                except Exception as unload_error:
                    logger.warning(f"Failed to unload ZORL adapter {lora_name}: {unload_error}")

            tracked_lookup = set(generation_lora_names)
            tracked_adapters = self.loaded_sampling_loras.get(model_id)
            if tracked_adapters is not None:
                self.loaded_sampling_loras[model_id] = [
                    (lora_name, lora_path)
                    for lora_name, lora_path in tracked_adapters
                    if lora_name not in tracked_lookup
                ]

            return len(generation_lora_names)

    async def _preload_zorl_candidates(
        self,
        *,
        model_id: str,
        candidates: list[ZORLCandidateInfo],
    ) -> bool:
        """Best-effort preload of candidate adapters onto registered inference endpoints."""
        if not self.inference_endpoints:
            return False

        loaded_names: list[str] = []
        try:
            for candidate in candidates:
                await self.create_sampling_session(
                    CreateSamplingSessionRequest(
                        model_id=model_id,
                        model_path=candidate.model_path,
                    )
                )
                loaded_names.append(candidate.lora_name)
        except Exception:
            for lora_name in reversed(loaded_names):
                try:
                    await self._unload_lora_on_inference_endpoints(lora_name)
                except Exception as unload_error:
                    logger.warning(f"Failed to unload preloaded ZORL adapter {lora_name}: {unload_error}")
            raise

        return True

    async def start_zorl_generation(self, request: ZORLStartGenerationRequest) -> ZORLStartGenerationResponse:
        """Plan and export one ZORL generation."""
        self._require_engine()

        try:
            engine_request = OrchestratorRequest(
                operation="start_zorl_generation",
                payload=ZORLStartGenerationData(
                    model_id=request.model_id,
                    num_pairs=request.num_pairs,
                    materialization=request.materialization.model_dump(exclude_none=True)
                    if request.materialization is not None
                    else None,
                    owner_url=request.owner_url,
                ),
            )
            response_future = await self.orchestrator_client.send_request(engine_request)
            output = await self._wait_for_response(
                response_future,
                engine_request.request_id,
                self.default_timeout,
                "Start ZORL generation timeout",
            )
            result = _sanitize_nan_to_zero(output.outputs[0] if output.outputs else {})

            candidates: list[ZORLCandidateInfo] = []
            for candidate in result.get("candidates", []):
                model_path, lora_name = self._zorl_candidate_uri(request.model_id, candidate["path"])
                candidates.append(
                    ZORLCandidateInfo(
                        candidate_id=str(candidate["candidate_id"]),
                        perturbation_index=int(candidate["perturbation_index"]),
                        direction=str(candidate["direction"]),
                        model_path=model_path,
                        lora_name=lora_name,
                        owner_url=candidate.get("owner_url") or request.owner_url,
                    )
                )

            sampling_ready = False
            if request.preload_sampling:
                sampling_ready = await self._preload_zorl_candidates(model_id=request.model_id, candidates=candidates)

            return ZORLStartGenerationResponse(
                model_id=str(result["model_id"]),
                generation_id=str(result["generation_id"]),
                generation_index=int(result["generation_index"]),
                family_id=str(result["family_id"]),
                family_refreshed=bool(result["family_refreshed"]),
                b_sigma=float(result["b_sigma"]),
                num_pairs=int(result["num_pairs"]),
                global_num_pairs=int(result.get("global_num_pairs", result["num_pairs"])),
                global_population=int(result.get("global_population", len(candidates))),
                shard_index=result.get("shard_index"),
                num_shards=result.get("num_shards"),
                local_num_pairs=result.get("local_num_pairs"),
                sampling_ready=sampling_ready,
                candidates=candidates,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Start ZORL generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Start ZORL generation failed: {e}",
            )

    async def apply_zorl_rewards(self, request: ZORLApplyRewardsRequest) -> ZORLApplyRewardsResponse:
        """Apply externally aggregated ZORL rewards to the parent adapter."""
        self._require_engine()

        try:
            engine_request = OrchestratorRequest(
                operation="apply_zorl_rewards",
                payload=ZORLApplyRewardsData(
                    model_id=request.model_id,
                    generation_id=request.generation_id,
                    candidate_rewards=[item.model_dump(exclude_none=True) for item in request.candidate_rewards],
                    learning_rate=request.learning_rate,
                ),
            )
            response_future = await self.orchestrator_client.send_request(engine_request)
            output = await self._wait_for_response(
                response_future,
                engine_request.request_id,
                self.default_timeout,
                "Apply ZORL rewards timeout",
            )
            result = _sanitize_nan_to_zero(output.outputs[0] if output.outputs else {})
            await self._cleanup_zorl_generation_sampling(model_id=request.model_id, generation_id=request.generation_id)
            response = ZORLApplyRewardsResponse(**result)
            if request.sync_after_apply:
                response.sync = await self._sync_inference_weights_after_zorl_apply(request)
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Apply ZORL rewards failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Apply ZORL rewards failed: {e}",
            )

    async def _sync_inference_weights_after_zorl_apply(self, request: ZORLApplyRewardsRequest) -> Dict[str, Any]:
        """Push post-apply weights to the registered inference endpoints.

        Reuses the exact /api/v1/sync_inference_weights machinery (endpoint
        registry, quantization normalization incl. trainer-side block-FP8,
        cache invalidation). This exists for perturbation_mode='fresh_ab',
        where the ES update is folded into the BASE weights so the served
        base on every replica is stale after an apply. Failures are reported
        in the returned summary instead of raising: the trainer update has
        already been committed and must not be reported as failed.
        """
        sync_kwargs: Dict[str, Any] = {"model_id": request.model_id}
        if request.sync_quantization is not None:
            # "fp8" shorthand -> minimal HF-style quantization_config; dict
            # payloads pass through to normalize_sync_quantization_config
            # unchanged (xorl already implements trainer-side block-FP8).
            if isinstance(request.sync_quantization, str):
                sync_kwargs["quantization"] = {"quant_method": request.sync_quantization}
            else:
                sync_kwargs["quantization"] = dict(request.sync_quantization)
        try:
            sync_response = await self.sync_inference_weights(SyncInferenceWeightsRequest(**sync_kwargs))
            return {
                "success": bool(sync_response.success),
                "message": sync_response.message,
                "transfer_time": sync_response.transfer_time,
                "total_bytes": sync_response.total_bytes,
                "num_parameters": sync_response.num_parameters,
            }
        except HTTPException as exc:
            logger.error(f"Post-apply ZORL weight sync failed: {exc.detail}")
            return {"success": False, "message": f"Post-apply weight sync failed: {exc.detail}"}
        except Exception as exc:
            logger.error(f"Post-apply ZORL weight sync failed: {exc}", exc_info=True)
            return {"success": False, "message": f"Post-apply weight sync failed: {exc}"}

    async def abort_zorl_generation(self, request: ZORLAbortGenerationRequest) -> ZORLAbortGenerationResponse:
        """Abort the active ZORL generation without updating the parent adapter."""
        self._require_engine()

        try:
            engine_request = OrchestratorRequest(
                operation="abort_zorl_generation",
                payload=ZORLAbortGenerationData(
                    model_id=request.model_id,
                    generation_id=request.generation_id,
                ),
            )
            response_future = await self.orchestrator_client.send_request(engine_request)
            output = await self._wait_for_response(
                response_future,
                engine_request.request_id,
                self.default_timeout,
                "Abort ZORL generation timeout",
            )
            result = _sanitize_nan_to_zero(output.outputs[0] if output.outputs else {})
            await self._cleanup_zorl_generation_sampling(model_id=request.model_id, generation_id=request.generation_id)
            return ZORLAbortGenerationResponse(**result)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Abort ZORL generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Abort ZORL generation failed: {e}",
            )
