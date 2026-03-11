"""Weight synchronization package for training-to-inference weight transfer."""

from .backends.nccl_broadcast import NCCLBroadcastBackend

__all__ = ["NCCLBroadcastBackend"]
