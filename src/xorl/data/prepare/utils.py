"""Data handling helpers"""

import contextlib
import functools
import hashlib
import time
from enum import Enum
from typing import Callable

import huggingface_hub
import numpy as np
import requests
from datasets import Dataset, IterableDataset

from ...utils import logging

logger = logging.get_logger(__name__)


class RetryStrategy(Enum):
    """Enum for retry strategies."""

    CONSTANT = 1
    LINEAR = 2
    EXPONENTIAL = 3


def retry_on_request_exceptions(
    max_retries=3, delay=1, retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
) -> Callable:
    """Decorator that retries function calls on specific request exceptions.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Base delay between retries in seconds.
        retry_strategy: Strategy for calculating retry delays.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError,
                    huggingface_hub.errors.HfHubHTTPError,
                ) as exc:
                    if attempt < max_retries - 1:
                        if retry_strategy == RetryStrategy.EXPONENTIAL:
                            step_delay = delay * 2**attempt
                        elif retry_strategy == RetryStrategy.LINEAR:
                            step_delay = delay * (attempt + 1)
                        else:
                            step_delay = delay  # Use constant delay.
                        time.sleep(step_delay)
                    else:
                        raise exc

        return wrapper

    return decorator


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    """Generate MD5 hash of a string."""
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def sha256(to_hash: str, encoding: str = "utf-8") -> str:
    """Generate SHA256 hash of a string."""
    return hashlib.sha256(to_hash.encode(encoding)).hexdigest()

