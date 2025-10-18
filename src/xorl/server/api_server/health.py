"""Health, sleep, and wake-up operations mixin."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import HTTPException, status

from xorl.server.api_server.api_types import HealthCheckResponse
from xorl.server.protocol.api_orchestrator import OrchestratorRequest, RequestType

logger = logging.getLogger(__name__)


class HealthMixin:
    """Mixin for health check, sleep, and wake-up operations."""

    async def health_check(self) -> HealthCheckResponse:
        """
        Check system health.

        Returns:
            Health check response

        Raises:
            HTTPException: If server not running
        """
        if not self._running or not self.orchestrator_client:
            return HealthCheckResponse(
                status="stopped",
                engine_running=False,
                active_requests=0,
                total_requests=0,
            )

        try:
            # Create engine request
            engine_request = OrchestratorRequest(
                request_type=RequestType.UTILITY,
                operation="health_check",
            )

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            # Wait for output with timeout
            try:
                output = await asyncio.wait_for(response_future, timeout=5.0)
            except asyncio.TimeoutError:
                return HealthCheckResponse(
                    status="timeout",
                    engine_running=False,
                    active_requests=0,
                    total_requests=0,
                )

            if output.error:
                return HealthCheckResponse(
                    status="error",
                    engine_running=False,
                    active_requests=0,
                    total_requests=0,
                )

            # Extract results
            result = output.outputs[0] if output.outputs else {}

            return HealthCheckResponse(
                status=result.get("status", "unknown"),
                engine_running=True,
                active_requests=result.get("active_requests", 0),
                total_requests=result.get("total_requests", 0),
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return HealthCheckResponse(
                status="error",
                engine_running=False,
                active_requests=0,
                total_requests=0,
            )

    async def sleep(self) -> Dict[str, Any]:
        """
        Offload model and optimizer to CPU.

        Returns:
            Sleep status

        Raises:
            HTTPException: If server not running or operation fails
        """
        self._require_engine()

        try:
            engine_request = OrchestratorRequest(operation="sleep")
            response_future = await self.orchestrator_client.send_request(engine_request)
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Sleep timeout"
            )
            result = output.outputs[0] if output.outputs else {}
            return {"success": True, "result": result}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Sleep failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sleep failed: {e}")

    async def wake_up(self) -> Dict[str, Any]:
        """
        Load model and optimizer back to GPU.

        Returns:
            Wake up status

        Raises:
            HTTPException: If server not running or operation fails
        """
        self._require_engine()

        try:
            engine_request = OrchestratorRequest(operation="wake_up")
            response_future = await self.orchestrator_client.send_request(engine_request)
            output = await self._wait_for_response(
                response_future, engine_request.request_id, self.default_timeout, "Wake up timeout"
            )
            result = output.outputs[0] if output.outputs else {}
            return {"success": True, "result": result}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Wake up failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Wake up failed: {e}")
