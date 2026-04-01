"""Backend-agnostic inference endpoint management (HTTP).

Handles health checks, pause/resume of inference endpoints.  These are pure
HTTP operations that don't depend on the transport backend (NCCL, RDMA, etc.).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests


logger = logging.getLogger(__name__)

# Reusable session for HTTP connection pooling
_http_session: Optional[requests.Session] = None


def _get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0,
        )
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    return _http_session


class EndpointManager:
    """Manage inference endpoints via HTTP (pause, resume, health check).

    This class is backend-agnostic — it only performs HTTP REST calls to the
    inference endpoints.  The transport-specific init/destroy (NCCL group
    creation, RDMA connection, etc.) is handled by the backend itself.
    """

    def __init__(self, endpoints: List[Dict[str, Any]]) -> None:
        self.endpoints = endpoints  # [{"host": ..., "port": ...}, ...]

    def health_check(self) -> None:
        """Check all endpoints are healthy.  Raises on failure."""
        session = _get_http_session()
        for ep in self.endpoints:
            url = f"http://{ep['host']}:{ep['port']}/health"
            try:
                resp = session.get(url, timeout=10)
                resp.raise_for_status()
                logger.info(f"[EndpointMgr] {ep['host']}:{ep['port']} healthy")
            except Exception as e:
                raise RuntimeError(f"Inference endpoint {ep['host']}:{ep['port']} health check failed: {e}")

    def pause(
        self,
        mode: str = "retract",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Pause inference on all endpoints.  Returns (results, all_ok)."""
        return self._parallel_request(
            url_path="/pause_generation",
            operation="Pause",
            payload={"mode": mode},
            timeout=60,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def resume(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Resume inference on all endpoints."""
        results, _ = self._parallel_request(
            url_path="/continue_generation",
            operation="Resume",
            payload={},
            timeout=30,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        for r in results:
            if not r["success"]:
                logger.error(
                    f"[EndpointMgr] {r['endpoint']} failed to resume "
                    f"after {max_retries} attempts — may require manual intervention"
                )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parallel_request(
        self,
        url_path: str,
        operation: str,
        payload: Dict[str, Any],
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Issue an HTTP POST to all endpoints in parallel with retry."""
        with ThreadPoolExecutor(max_workers=max(len(self.endpoints), 1)) as pool:
            futures = {
                pool.submit(
                    self._request_with_retry,
                    ep,
                    url_path,
                    operation,
                    payload,
                    timeout,
                    max_retries,
                    retry_delay,
                ): ep
                for ep in self.endpoints
            }
            results = [f.result() for f in as_completed(futures)]
        all_ok = all(r["success"] for r in results)
        return results, all_ok

    @staticmethod
    def _request_with_retry(
        endpoint: Dict[str, Any],
        url_path: str,
        operation: str,
        payload: Dict[str, Any],
        timeout: int,
        max_retries: int,
        retry_delay: float,
    ) -> Dict[str, Any]:
        session = _get_http_session()
        label = f"{endpoint['host']}:{endpoint['port']}"
        result: Dict[str, Any] = {}

        for attempt in range(max_retries):
            try:
                url = f"http://{label}{url_path}"
                resp = session.post(url, json=payload, timeout=timeout)
                data = resp.json()
                success = data.get("status") == "ok" or data.get("success", False)
                result = {
                    "endpoint": label,
                    "success": success,
                    "message": data.get("message", ""),
                    "attempts": attempt + 1,
                }
                if success:
                    logger.info(f"[EndpointMgr] {label} {operation} ok (attempt {attempt + 1}/{max_retries})")
                    return result
                logger.warning(
                    f"[EndpointMgr] {label} {operation} failed "
                    f"(attempt {attempt + 1}/{max_retries}): {data.get('message', '')}"
                )
            except Exception as e:
                result = {
                    "endpoint": label,
                    "success": False,
                    "message": str(e),
                    "attempts": attempt + 1,
                }
                logger.warning(f"[EndpointMgr] {label} {operation} error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        return result
