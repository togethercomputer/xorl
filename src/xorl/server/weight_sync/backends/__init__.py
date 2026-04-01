"""Transport backends for weight synchronization."""

from .base import TransportConfig, WeightTransportBackend


__all__ = [
    "TransportConfig",
    "WeightTransportBackend",
    "create_backend",
]


def create_backend(
    method: str,
    config: TransportConfig,
    **kwargs,
) -> WeightTransportBackend:
    """Factory: create a transport backend by name.

    Args:
        method: Backend name. Currently supported: ``"nccl_broadcast"``.
        config: Shared transport configuration.

    Returns:
        An initialized :class:`WeightTransportBackend` instance (not yet
        connected — call :meth:`initialize` before transferring).
    """
    if method == "nccl_broadcast":
        from .nccl_broadcast import NCCLBroadcastBackend

        return NCCLBroadcastBackend(config, **kwargs)
    raise ValueError(f"Unknown weight sync backend: {method!r}. Supported: 'nccl_broadcast'.")
