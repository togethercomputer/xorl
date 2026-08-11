"""Data handling helpers"""

import functools
import hashlib
import time
from typing import Callable

import requests
from huggingface_hub.errors import HfHubHTTPError


def retry_on_request_exceptions(max_retries=3, delay=1) -> Callable:
    """Decorator that retries function calls on specific request exceptions.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Base delay between retries in seconds.

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
                    HfHubHTTPError,
                ) as exc:
                    if attempt < max_retries - 1:
                        time.sleep(delay * 2**attempt)
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
