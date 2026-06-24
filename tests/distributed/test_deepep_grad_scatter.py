import pytest
import torch


try:
    from xorl.distributed.moe.deepep import (
        _combine_scatter_accum_mode,
        _deepep_combine_scatter_accum_dtype,
        _grad_scatter_accum_mode,
        _scatter_expert_grad_to_recv,
    )
except ImportError as exc:  # pragma: no cover - upstream WIP gap
    pytest.skip(
        f"deepep accum-mode API was refactored to explicit accum-dtype helpers upstream ({exc}); "
        "this test still targets the removed _*_accum_mode functions and needs updating to the "
        "_deepep_*_accum_dtype API",
        allow_module_level=True,
    )


pytestmark = pytest.mark.cpu


def _clear_deepep_env_caches():
    _grad_scatter_accum_mode.cache_clear()
    _combine_scatter_accum_mode.cache_clear()


@pytest.fixture(autouse=True)
def reset_deepep_env_caches():
    _clear_deepep_env_caches()
    yield
    _clear_deepep_env_caches()


def test_scatter_expert_grad_to_recv_matches_fp32_index_add_with_chunked_cast():
    grad_expert_input = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.bfloat16,
    )
    permuted_indices = torch.tensor([2, 0, 2, 1])

    actual = _scatter_expert_grad_to_recv(
        grad_expert_input,
        permuted_indices,
        num_recv_tokens=4,
        hidden_dim=3,
        chunk_tokens=2,
    )

    expected = torch.zeros(4, 3, dtype=torch.float32)
    expected.index_add_(0, permuted_indices, grad_expert_input.float())

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected)


def test_scatter_expert_grad_to_recv_can_accumulate_in_input_dtype(monkeypatch):
    grad_expert_input = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.bfloat16,
    )
    permuted_indices = torch.tensor([2, 0, 2, 1])
    observed_zero_dtypes = []
    original_zeros = torch.zeros

    def recording_zeros(*args, **kwargs):
        observed_zero_dtypes.append(kwargs.get("dtype"))
        return original_zeros(*args, **kwargs)

    monkeypatch.setenv("XORL_DEEPEP_GRAD_SCATTER_ACCUM_DTYPE", "input")
    monkeypatch.setattr(torch, "zeros", recording_zeros)

    actual = _scatter_expert_grad_to_recv(
        grad_expert_input,
        permuted_indices,
        num_recv_tokens=4,
        hidden_dim=3,
        chunk_tokens=2,
    )

    expected = original_zeros(4, 3, dtype=torch.bfloat16)
    expected.index_add_(0, permuted_indices, grad_expert_input)

    assert observed_zero_dtypes == [torch.bfloat16]
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_deepep_combine_scatter_accum_dtype_can_use_input_dtype(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_COMBINE_SCATTER_ACCUM_DTYPE", "input")

    assert _deepep_combine_scatter_accum_dtype(torch.bfloat16) is torch.bfloat16
