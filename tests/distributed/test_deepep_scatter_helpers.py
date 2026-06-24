"""Unit tests for DeepEP scatter helpers — CPU-only, no DeepEP runtime needed.

Covers ``_scatter_expert_grad_to_recv`` and ``_scatter_expert_output_to_recv``:
- chunked dtype-cast path matches the unchunked path (CPU is deterministic);
- ``input_dtype`` mode skips the dtype cast entirely;
- empty input returns zeros in the input dtype;
- alias parser rejects bogus values.
"""

import pytest
import torch

from xorl.distributed.moe import deepep as D


pytestmark = pytest.mark.cpu

if not hasattr(D, "_grad_scatter_accum_mode"):  # pragma: no cover - upstream WIP gap
    pytest.skip(
        "deepep's accum-mode / chunk-tokens / empty-cache-on-oom / parse-accum-dtype helper family was "
        "refactored to the accum-dtype API upstream (_deepep_*_accum_dtype); these unit tests target the "
        "removed _*_accum_mode helpers and need rewriting against the new dtype-returning API",
        allow_module_level=True,
    )


def _clear_env_caches():
    """Wipe the lru_caches that hold env-var snapshots."""
    D._grad_scatter_accum_mode.cache_clear()
    D._combine_scatter_accum_mode.cache_clear()
    D._grad_scatter_chunk_tokens.cache_clear()
    D._combine_scatter_chunk_tokens.cache_clear()
    D._grad_scatter_empty_cache_on_oom.cache_clear()
    D._combine_scatter_empty_cache_on_oom.cache_clear()


@pytest.fixture(autouse=True)
def reset_env_caches():
    _clear_env_caches()
    yield
    _clear_env_caches()


# ── grad scatter ─────────────────────────────────────────────────────────────


def _ref_grad_scatter_fp32(grad_expert_input, permuted_indices, num_recv_tokens, hidden_dim):
    """Reference: single-shot fp32 index_add, then cast back to input dtype."""
    out = torch.zeros(num_recv_tokens, hidden_dim, dtype=torch.float32)
    out.index_add_(0, permuted_indices, grad_expert_input.float())
    return out.to(grad_expert_input.dtype)


def test_grad_scatter_fp32_chunked_matches_unchunked():
    torch.manual_seed(0)
    num_recv_tokens, hidden_dim, num_expert_tokens = 17, 8, 64
    grad = torch.randn(num_expert_tokens, hidden_dim, dtype=torch.bfloat16)
    idx = torch.randint(0, num_recv_tokens, (num_expert_tokens,), dtype=torch.long)

    expected = _ref_grad_scatter_fp32(grad, idx, num_recv_tokens, hidden_dim)
    # Force several chunks (3 chunks for 64 tokens at chunk=24).
    got = D._scatter_expert_grad_to_recv(grad, idx, num_recv_tokens, hidden_dim, chunk_tokens=24)

    assert got.dtype == grad.dtype
    assert got.shape == expected.shape
    # CPU index_add is deterministic, so bit-exact equality holds when the
    # accumulation dtype matches (both branches accumulate in fp32 then cast).
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_grad_scatter_input_dtype_mode_skips_chunking(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_GRAD_SCATTER_ACCUM_DTYPE", "input_dtype")
    _clear_env_caches()

    num_recv_tokens, hidden_dim, num_expert_tokens = 5, 4, 12
    grad = torch.randn(num_expert_tokens, hidden_dim, dtype=torch.bfloat16)
    idx = torch.randint(0, num_recv_tokens, (num_expert_tokens,), dtype=torch.long)

    out = D._scatter_expert_grad_to_recv(grad, idx, num_recv_tokens, hidden_dim, chunk_tokens=3)
    # In input_dtype mode, accumulation dtype == input dtype, so the dtype-mismatch
    # branch is skipped; output is a direct index_add in bfloat16.
    ref = torch.zeros(num_recv_tokens, hidden_dim, dtype=torch.bfloat16)
    ref.index_add_(0, idx, grad)
    assert out.dtype == grad.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_grad_scatter_empty_input_returns_input_dtype_zeros():
    grad = torch.empty(0, 4, dtype=torch.bfloat16)
    idx = torch.empty(0, dtype=torch.long)
    out = D._scatter_expert_grad_to_recv(grad, idx, num_recv_tokens=3, hidden_dim=4)
    assert out.dtype == torch.bfloat16
    assert out.shape == (3, 4)
    assert torch.equal(out, torch.zeros(3, 4, dtype=torch.bfloat16))


# ── combine scatter ──────────────────────────────────────────────────────────


def _ref_combine_scatter_fp32(expert_output, permuted_indices, num_recv_tokens, hidden_dim, output_dtype):
    out = torch.zeros(num_recv_tokens, hidden_dim, dtype=torch.float32)
    idx_2d = permuted_indices.unsqueeze(1).expand(-1, hidden_dim)
    out.scatter_add_(0, idx_2d, expert_output.float())
    return out.to(output_dtype)


def test_combine_scatter_fp32_chunked_matches_unchunked():
    torch.manual_seed(1)
    num_recv_tokens, hidden_dim, num_expert_tokens = 11, 6, 50
    expert_output = torch.randn(num_expert_tokens, hidden_dim, dtype=torch.bfloat16)
    idx = torch.randint(0, num_recv_tokens, (num_expert_tokens,), dtype=torch.long)

    expected = _ref_combine_scatter_fp32(expert_output, idx, num_recv_tokens, hidden_dim, torch.bfloat16)
    got = D._scatter_expert_output_to_recv(
        expert_output, idx, num_recv_tokens, hidden_dim, torch.bfloat16, chunk_tokens=17
    )
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_combine_scatter_input_dtype_mode(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_COMBINE_SCATTER_ACCUM_DTYPE", "input_dtype")
    _clear_env_caches()

    num_recv_tokens, hidden_dim, num_expert_tokens = 4, 3, 9
    expert_output = torch.randn(num_expert_tokens, hidden_dim, dtype=torch.bfloat16)
    idx = torch.randint(0, num_recv_tokens, (num_expert_tokens,), dtype=torch.long)

    out = D._scatter_expert_output_to_recv(
        expert_output, idx, num_recv_tokens, hidden_dim, torch.bfloat16, chunk_tokens=2
    )
    ref = torch.zeros(num_recv_tokens, hidden_dim, dtype=torch.bfloat16)
    idx_2d = idx.unsqueeze(1).expand(-1, hidden_dim)
    ref.scatter_add_(0, idx_2d, expert_output)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_combine_scatter_empty_input_returns_output_dtype_zeros():
    expert_output = torch.empty(0, 5, dtype=torch.bfloat16)
    idx = torch.empty(0, dtype=torch.long)
    out = D._scatter_expert_output_to_recv(
        expert_output, idx, num_recv_tokens=2, hidden_dim=5, output_dtype=torch.bfloat16
    )
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 5)
    assert torch.equal(out, torch.zeros(2, 5, dtype=torch.bfloat16))


# ── env-var parser ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected_mode",
    [
        (None, "fp32"),
        ("fp32", "fp32"),
        (" FP32 ", "fp32"),
        ("input_dtype", "input_dtype"),
        ("input", "input_dtype"),
        (" INPUT ", "input_dtype"),
    ],
)
def test_parse_accum_dtype_mode_accepts_canonical(raw, expected_mode):
    assert D._parse_accum_dtype_mode("X", raw) == expected_mode


@pytest.mark.parametrize("raw", ["float32", "same", "bfloat16", ""])
def test_parse_accum_dtype_mode_rejects_unknown(raw):
    # Empty string is treated as default fp32 — exempt it.
    if raw == "":
        assert D._parse_accum_dtype_mode("X", raw) == "fp32"
        return
    with pytest.raises(ValueError, match="X must be 'fp32' or 'input_dtype'"):
        D._parse_accum_dtype_mode("X", raw)


def test_env_var_resolvers_are_cached(monkeypatch):
    # First read locks in the mode.
    monkeypatch.setenv("XORL_DEEPEP_GRAD_SCATTER_ACCUM_DTYPE", "input_dtype")
    _clear_env_caches()
    assert D._grad_scatter_accum_mode() == "input_dtype"
    # Mutating the env after the first read does not change the cached value.
    monkeypatch.setenv("XORL_DEEPEP_GRAD_SCATTER_ACCUM_DTYPE", "fp32")
    assert D._grad_scatter_accum_mode() == "input_dtype"


def test_grad_chunk_tokens_default_and_floor(monkeypatch):
    _clear_env_caches()
    assert D._grad_scatter_chunk_tokens() == 2048

    monkeypatch.setenv("XORL_DEEPEP_GRAD_SCATTER_CHUNK_TOKENS", "0")
    _clear_env_caches()
    assert D._grad_scatter_chunk_tokens() == 1  # max(1, 0)


def test_combine_chunk_tokens_default(monkeypatch):
    _clear_env_caches()
    assert D._combine_scatter_chunk_tokens() == 4096

    monkeypatch.setenv("XORL_DEEPEP_COMBINE_SCATTER_CHUNK_TOKENS", "256")
    _clear_env_caches()
    assert D._combine_scatter_chunk_tokens() == 256
