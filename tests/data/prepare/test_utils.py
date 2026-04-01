"""Tests for xorl.data.prepare.utils module."""

import time
from unittest.mock import Mock

import huggingface_hub
import pytest
import requests

from xorl.data.prepare.utils import (
    RetryStrategy,
    md5,
    retry_on_request_exceptions,
    sha256,
)


pytestmark = pytest.mark.cpu


class TestRetryOnRequestExceptions:
    """Tests for retry_on_request_exceptions decorator."""

    def test_success_retries_and_max_retries(self):
        """Covers immediate success, retry on ReadTimeout/HfHubHTTPError, max retries exhaustion,
        and non-request exception passthrough."""

        # Immediate success
        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def success_func():
            return "success"

        assert success_func() == "success"

        # Retry on ReadTimeout
        mock_func = Mock(
            side_effect=[requests.exceptions.ReadTimeout("t"), requests.exceptions.ReadTimeout("t"), "success"]
        )

        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def retry_func():
            return mock_func()

        assert retry_func() == "success"
        assert mock_func.call_count == 3

        # Retry on HfHubHTTPError
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_func2 = Mock(
            side_effect=[huggingface_hub.errors.HfHubHTTPError("HF error", response=mock_response), "success"]
        )

        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def hf_func():
            return mock_func2()

        assert hf_func() == "success"

        # Max retries exhausted
        @retry_on_request_exceptions(max_retries=2, delay=0.01)
        def failing_func():
            raise requests.exceptions.ReadTimeout("persistent timeout")

        with pytest.raises(requests.exceptions.ReadTimeout):
            failing_func()

        # Non-request exception not caught
        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def value_error_func():
            raise ValueError("not a request exception")

        with pytest.raises(ValueError):
            value_error_func()

    def test_backoff_strategies(self):
        """Covers exponential, linear, and constant backoff timing."""
        for strategy, check_fn in [
            (RetryStrategy.EXPONENTIAL, lambda d1, d2: d2 > d1 * 1.3),
            (RetryStrategy.LINEAR, lambda d1, d2: d2 > d1 * 1.3),
            (RetryStrategy.CONSTANT, lambda d1, d2: abs(d2 - d1) < d1 * 0.5),
        ]:
            call_times = []
            mock_func = Mock(
                side_effect=[requests.exceptions.ReadTimeout("t"), requests.exceptions.ReadTimeout("t"), "success"]
            )

            @retry_on_request_exceptions(max_retries=3, delay=0.1, retry_strategy=strategy)
            def func():
                call_times.append(time.time())
                return mock_func()

            assert func() == "success"
            if len(call_times) >= 3:
                d1 = call_times[1] - call_times[0]
                d2 = call_times[2] - call_times[1]
                assert 0.05 < d1 < 0.25
                assert check_fn(d1, d2)


class TestHashFunctions:
    """Tests for md5 and sha256 hash functions."""

    def test_md5_and_sha256(self):
        """Covers known hashes, different inputs produce different hashes, and unicode handling."""
        # MD5 known value
        assert md5("hello") == "5d41402abc4b2a76b9719d911017c592"
        # MD5 different inputs
        assert md5("string1") != md5("string2")
        # MD5 unicode
        assert len(md5("测试中文")) == 32

        # SHA256 known value
        assert sha256("hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        # SHA256 different inputs
        assert sha256("string1") != sha256("string2")
        # SHA256 unicode
        assert len(sha256("测试中文")) == 64
