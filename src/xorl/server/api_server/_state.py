"""Global API server state. Shared by api_server.py and endpoints.py to avoid circular imports."""

from typing import Optional, TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from xorl.server.api_server.server import APIServer


api_server: Optional["APIServer"] = None


def require_api_server() -> "APIServer":
    """FastAPI dependency that returns the global APIServer or raises 503."""
    if api_server is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API server not initialized")
    return api_server
