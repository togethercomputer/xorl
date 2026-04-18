"""Training operations mixin: two-phase async pattern, forward, backward, optim step."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict

from fastapi import HTTPException, status

from xorl.server.api_server.api_types import (
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
    TryAgainResponse,
    UntypedAPIFuture,
)
from xorl.server.api_server.future_store import (
    FutureStatus,
)
from xorl.server.api_server.utils import (
    validate_model_id,
)
from xorl.server.protocol.api_orchestrator import OrchestratorRequest
from xorl.server.protocol.operations import ModelPassData, OptimStepData


logger = logging.getLogger(__name__)


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
            is_metrics = {k: v for k, v in result.items() if k.startswith("is_")}
            if is_metrics:
                logger.debug(f"API Server: IS metrics present in result: {list(is_metrics.keys())}")
            else:
                logger.debug("API Server: No IS metrics in result")

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
                "execution_time:sum": result.get("execution_time", 0.0),
            }

            # Add IS metrics if present (already have name:reduction format)
            for key, value in result.items():
                if key.startswith("is_"):
                    # Ensure colon format for tinker compatibility
                    metrics[key if ":" in key else f"{key}:mean"] = value

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

            return ForwardResponse(
                loss_fn_output_type=loss_fn_output_type,
                loss_fn_outputs=loss_fn_outputs,
                metrics=metrics,
                info={},
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
            # Determine gradient clipping value
            # Priority: explicit gradient_clip parameter, then adam_params.grad_clip_norm
            gradient_clip = request.gradient_clip
            if gradient_clip is None and request.adam_params.grad_clip_norm > 0:
                gradient_clip = request.adam_params.grad_clip_norm

            # Create engine request
            # Pass seq_id and model_id for request ordering (SeqIdAwareFIFOPolicy)
            engine_request = OrchestratorRequest(
                operation="optim_step",
                payload=OptimStepData(
                    lr=request.adam_params.learning_rate,
                    gradient_clip=gradient_clip,
                    model_id=request.model_id,
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

            return OptimStepResponse(
                metrics={
                    "grad_norm": grad_norm,
                    "learning_rate": result.get("lr", request.adam_params.learning_rate),
                },
                info=info,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Optimizer step failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Optimizer step failed: {e}")
