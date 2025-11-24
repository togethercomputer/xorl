"""Tests for xorl.data.prepare.utils module."""

import time
from unittest.mock import Mock, patch

import pytest
import requests
import huggingface_hub

from xorl.data.prepare.utils import (
    RetryStrategy,
    retry_on_request_exceptions,
    md5,
    sha256,
)


pytestmark = pytest.mark.cpu


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_retry_strategy_values(self):
        """Verify RetryStrategy enum values."""
        assert RetryStrategy.CONSTANT.value == 1
        assert RetryStrategy.LINEAR.value == 2
        assert RetryStrategy.EXPONENTIAL.value == 3


class TestRetryOnRequestExceptions:
    """Tests for retry_on_request_exceptions decorator."""

    def test_successful_execution_on_first_try(self):
        """Should return result immediately if no exception."""
        @retry_on_request_exceptions(max_retries=3, delay=0.1)
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_retries_on_read_timeout(self):
        """Should retry on ReadTimeout and eventually succeed."""
        mock_func = Mock(side_effect=[
            requests.exceptions.ReadTimeout("timeout"),
            requests.exceptions.ReadTimeout("timeout"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def func():
            return mock_func()

        result = func()
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retries_on_connection_error(self):
        """Should retry on ConnectionError."""
        mock_func = Mock(side_effect=[
            requests.exceptions.ConnectionError("connection failed"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def func():
            return mock_func()

        result = func()
        assert result == "success"
        assert mock_func.call_count == 2

    def test_retries_on_http_error(self):
        """Should retry on HTTPError."""
        mock_func = Mock(side_effect=[
            requests.exceptions.HTTPError("500 error"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def func():
            return mock_func()

        result = func()
        assert result == "success"

    def test_retries_on_hf_hub_http_error(self):
        """Should retry on HfHubHTTPError."""
        mock_func = Mock(side_effect=[
            huggingface_hub.errors.HfHubHTTPError("HF error"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def func():
            return mock_func()

        result = func()
        assert result == "success"

    def test_raises_after_max_retries(self):
        """Should raise exception after max_retries exhausted."""
        @retry_on_request_exceptions(max_retries=2, delay=0.01)
        def failing_func():
            raise requests.exceptions.ReadTimeout("persistent timeout")

        with pytest.raises(requests.exceptions.ReadTimeout):
            failing_func()

    def test_exponential_backoff_strategy(self):
        """Should use exponential delays."""
        call_times = []
        mock_func = Mock(side_effect=[
            requests.exceptions.ReadTimeout("timeout"),
            requests.exceptions.ReadTimeout("timeout"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.1, retry_strategy=RetryStrategy.EXPONENTIAL)
        def func():
            call_times.append(time.time())
            return mock_func()

        result = func()
        assert result == "success"

        # Check that delays are exponential: 0.1s, 0.2s
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert 0.08 < delay1 < 0.15  # ~0.1s
            assert 0.15 < delay2 < 0.25  # ~0.2s

    def test_linear_backoff_strategy(self):
        """Should use linear delays."""
        call_times = []
        mock_func = Mock(side_effect=[
            requests.exceptions.ReadTimeout("timeout"),
            requests.exceptions.ReadTimeout("timeout"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.1, retry_strategy=RetryStrategy.LINEAR)
        def func():
            call_times.append(time.time())
            return mock_func()

        result = func()
        assert result == "success"

        # Check that delays are linear: 0.1s, 0.2s
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert 0.08 < delay1 < 0.15  # ~0.1s
            assert 0.15 < delay2 < 0.25  # ~0.2s

    def test_constant_backoff_strategy(self):
        """Should use constant delays."""
        call_times = []
        mock_func = Mock(side_effect=[
            requests.exceptions.ReadTimeout("timeout"),
            requests.exceptions.ReadTimeout("timeout"),
            "success"
        ])

        @retry_on_request_exceptions(max_retries=3, delay=0.1, retry_strategy=RetryStrategy.CONSTANT)
        def func():
            call_times.append(time.time())
            return mock_func()

        result = func()
        assert result == "success"

        # Check that delays are constant: 0.1s, 0.1s
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert 0.08 < delay1 < 0.15  # ~0.1s
            assert 0.08 < delay2 < 0.15  # ~0.1s

    def test_does_not_catch_other_exceptions(self):
        """Should not catch non-request exceptions."""
        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def func():
            raise ValueError("not a request exception")

        with pytest.raises(ValueError):
            func()

    def test_preserves_function_metadata(self):
        """Should preserve function name and docstring."""
        @retry_on_request_exceptions(max_retries=3, delay=0.01)
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestMd5:
    """Tests for md5 hash function."""

    def test_generates_consistent_hash(self):
        """Should generate consistent MD5 hash."""
        hash1 = md5("test string")
        hash2 = md5("test string")
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self):
        """Should produce different hashes for different inputs."""
        hash1 = md5("string1")
        hash2 = md5("string2")
        assert hash1 != hash2

    def test_hash_format(self):
        """Should return 32-character hexadecimal string."""
        result = md5("test")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        """Should handle empty string."""
        result = md5("")
        assert len(result) == 32

    def test_unicode_strings(self):
        """Should handle unicode strings."""
        result = md5("测试中文")
        assert len(result) == 32

    def test_custom_encoding(self):
        """Should support custom encoding."""
        result = md5("test", encoding="utf-8")
        assert len(result) == 32

    def test_known_hash_value(self):
        """Should match known MD5 hash values."""
        # Known MD5 hash for "hello"
        result = md5("hello")
        expected = "5d41402abc4b2a76b9719d911017c592"
        assert result == expected


class TestSha256:
    """Tests for sha256 hash function."""

    def test_generates_consistent_hash(self):
        """Should generate consistent SHA256 hash."""
        hash1 = sha256("test string")
        hash2 = sha256("test string")
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self):
        """Should produce different hashes for different inputs."""
        hash1 = sha256("string1")
        hash2 = sha256("string2")
        assert hash1 != hash2

    def test_hash_format(self):
        """Should return 64-character hexadecimal string."""
        result = sha256("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        """Should handle empty string."""
        result = sha256("")
        assert len(result) == 64

    def test_unicode_strings(self):
        """Should handle unicode strings."""
        result = sha256("测试中文")
        assert len(result) == 64

    def test_custom_encoding(self):
        """Should support custom encoding."""
        result = sha256("test", encoding="utf-8")
        assert len(result) == 64

    def test_known_hash_value(self):
        """Should match known SHA256 hash values."""
        # Known SHA256 hash for "hello"
        result = sha256("hello")
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert result == expected
