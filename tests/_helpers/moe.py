"""CPU kernel doubles shared by MoE contract tests."""

import torch


def counts_from_cumsum(cumsum: torch.Tensor) -> list[int]:
    counts = torch.empty_like(cumsum)
    counts[0] = cumsum[0]
    counts[1:] = cumsum[1:] - cumsum[:-1]
    return counts.tolist()


def _naive_group_gemm_same_nk(a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False, **kwargs):
    del max_M, kwargs
    assert not transpose_a

    outputs = []
    start = 0
    for expert_idx, count in enumerate(counts_from_cumsum(cumsum_M)):
        end = start + count
        weight = b[expert_idx]
        outputs.append(a[start:end] @ (weight.transpose(0, 1) if transpose_b else weight))
        start = end
    return torch.cat(outputs, dim=0)


def _naive_group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=False, transpose_b=False, **kwargs):
    del max_K, kwargs

    start = 0
    for expert_idx, count in enumerate(counts_from_cumsum(cumsum_K)):
        end = start + count
        lhs = a[start:end].transpose(0, 1) if transpose_a else a[start:end]
        rhs = b[start:end].transpose(0, 1) if transpose_b else b[start:end]
        c[expert_idx].copy_(lhs @ rhs)
        start = end
    return c


def patch_ep_kernels(monkeypatch, module) -> None:
    """Replace grouped kernels while preserving the canonical runtime module."""
    if module.__name__.endswith(".triton"):
        monkeypatch.setattr(module, "group_gemm_same_nk", _naive_group_gemm_same_nk)
        monkeypatch.setattr(module, "group_gemm_same_mn", _naive_group_gemm_same_mn)
    else:
        monkeypatch.setattr(module, "_group_gemm_same_nk", _naive_group_gemm_same_nk)
        monkeypatch.setattr(module, "_group_gemm_same_mn", _naive_group_gemm_same_mn)
        monkeypatch.setattr(module, "quack_group_gemm_same_nk", _naive_group_gemm_same_nk)
        monkeypatch.setattr(module, "quack_group_gemm_same_mn", _naive_group_gemm_same_mn)
