"""Focused CPU/mock coverage for the trainable GLM-5 FlashMLA contract."""

from unittest.mock import Mock

import pytest
import torch

from xorl.models.transformers.glm5 import sparse_mla
from xorl.models.transformers.glm5.sparse_mla import (
    _flashmla_constraint_error,
    _flatten_sparse_mla_inputs,
    sparse_mla_dispatch,
)
from xorl.ops.families.glm5 import flashmla_sparse_mla
from xorl.ops.families.glm5.flashmla_sparse_mla import FlashMLASparseWithTileLangBackward


pytestmark = [pytest.mark.cpu]


def _assert_flashmla_batch_flatten_offsets_only_valid_indices():
    q = torch.zeros(2, 2, 3, 4)
    kv = torch.zeros(2, 3, 4)
    indices = torch.tensor(
        [
            [[0, 2, -1, 3], [1, -2, 2, 2**40]],
            [[0, 1, 2, -1], [2, 0, 7, -(2**40)]],
        ],
        dtype=torch.int64,
    )

    q_flat, kv_flat, indices_flat = _flatten_sparse_mla_inputs(q, kv, indices)

    assert q_flat.shape == (4, 3, 4)
    assert kv_flat.shape == (6, 1, 4)
    assert indices_flat.dtype == torch.int32
    expected = torch.tensor(
        [
            [[0, 2, -1, -1]],
            [[1, -1, 2, -1]],
            [[3, 4, 5, -1]],
            [[5, 3, -1, -1]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(indices_flat, expected)


def test_flashmla_backward_compacts_valid_rows_and_scatter_zeros(monkeypatch):
    _assert_flashmla_batch_flatten_offsets_only_valid_indices()
    captured = {}

    def fake_forward(q, kv, indices, scaling):
        captured["forward_indices"] = indices.clone()
        captured["forward_scaling"] = scaling
        out = q[..., :2].clone()
        max_logits = torch.zeros(q.shape[:2], dtype=torch.float32)
        lse = torch.zeros(q.shape[:2], dtype=torch.float32)
        return out, max_logits, lse

    def fake_backward(q, kv, out, grad_out, indices, lse, scaling):
        captured["backward_q_rows"] = q.shape[0]
        captured["backward_indices"] = indices.clone()
        captured["backward_scaling"] = scaling
        assert out.shape == grad_out.shape == (2, 2, 2)
        assert lse.shape == (2, 2)
        return torch.full_like(q, 3), torch.full_like(kv, 5)

    monkeypatch.setattr(flashmla_sparse_mla, "_run_flashmla_sparse_fwd", fake_forward)
    monkeypatch.setattr(flashmla_sparse_mla, "_run_tilelang_sparse_mla_bwd", fake_backward)

    q = torch.randn(4, 2, 3, requires_grad=True)
    kv = torch.randn(5, 1, 3, requires_grad=True)
    indices = torch.tensor(
        [
            [[0, 1]],
            [[-1, -1]],
            [[2, -1]],
            [[-1, -1]],
        ],
        dtype=torch.int64,
    )

    out = FlashMLASparseWithTileLangBackward.apply(q, kv, indices, 0.25)
    out.sum().backward()

    assert captured["forward_scaling"] == captured["backward_scaling"] == 0.25
    assert captured["forward_indices"].dtype == torch.int32
    assert captured["backward_q_rows"] == 2
    torch.testing.assert_close(captured["backward_indices"], torch.tensor([[[0, 1]], [[2, -1]]], dtype=torch.int32))
    torch.testing.assert_close(q.grad[0], torch.full_like(q.grad[0], 3))
    torch.testing.assert_close(q.grad[2], torch.full_like(q.grad[2], 3))
    assert torch.count_nonzero(q.grad[1]).item() == 0
    assert torch.count_nonzero(q.grad[3]).item() == 0
    torch.testing.assert_close(kv.grad, torch.full_like(kv.grad, 5))

    _assert_flashmla_all_invalid_rows_have_zero_output_and_gradients(monkeypatch)
    with monkeypatch.context() as case_patch:
        _assert_flashmla_backend_fails_closed_outside_production_envelope(case_patch)


def _assert_flashmla_all_invalid_rows_have_zero_output_and_gradients(monkeypatch):
    backward = Mock(side_effect=AssertionError("all-invalid input must bypass TileLang backward"))

    def fake_forward(q, kv, indices, scaling):
        del kv, indices, scaling
        out = torch.full((q.shape[0], q.shape[1], 2), float("nan"), dtype=q.dtype)
        max_logits = torch.full(q.shape[:2], float("-inf"), dtype=torch.float32)
        lse = torch.full(q.shape[:2], float("-inf"), dtype=torch.float32)
        return out, max_logits, lse

    monkeypatch.setattr(flashmla_sparse_mla, "_run_flashmla_sparse_fwd", fake_forward)
    monkeypatch.setattr(flashmla_sparse_mla, "_run_tilelang_sparse_mla_bwd", backward)

    q = torch.randn(3, 2, 3, requires_grad=True)
    kv = torch.randn(5, 1, 3, requires_grad=True)
    indices = torch.full((3, 1, 2), -1, dtype=torch.int32)

    out = FlashMLASparseWithTileLangBackward.apply(q, kv, indices, 0.25)
    assert torch.equal(out, torch.zeros_like(out))
    out.sum().backward()

    backward.assert_not_called()
    assert torch.equal(q.grad, torch.zeros_like(q))
    assert torch.equal(kv.grad, torch.zeros_like(kv))


def _assert_flashmla_backend_fails_closed_outside_production_envelope(monkeypatch):
    q = torch.zeros(1, 1, 64, 576, dtype=torch.bfloat16)
    kv = torch.zeros(1, 2, 576, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, 2048, dtype=torch.int32)

    error = _flashmla_constraint_error(q, kv, indices, 192**-0.5, kv_lora_rank=512)
    assert error == "requires CUDA q, kv, and indices"
    with pytest.raises(RuntimeError, match="outside its certified GLM-5.2 envelope"):
        sparse_mla_dispatch(
            q,
            kv,
            indices,
            scaling=192**-0.5,
            kv_lora_rank=512,
            backend="flashmla",
        )

    q = torch.empty(1, 1, 63, 576, device="meta", dtype=torch.bfloat16)
    kv = torch.empty(1, 2, 576, device="meta", dtype=torch.bfloat16)
    indices = torch.empty(1, 1, 2048, device="meta", dtype=torch.int32)

    # Meta tensors first fail the device contract.  A minimal fake carrying
    # CUDA-like metadata isolates the production-shape check without a GPU.
    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.ndim = len(shape)
            self.device = torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    error = _flashmla_constraint_error(
        FakeTensor(q.shape, q.dtype),
        FakeTensor(kv.shape, kv.dtype),
        FakeTensor(indices.shape, indices.dtype),
        192**-0.5,
        kv_lora_rank=512,
    )
    assert error == "requires H=64, D=576, value width=512, and topk=2048"

    error = sparse_mla._flashmla_constraint_error(
        FakeTensor((2, 1, 64, 576), torch.bfloat16),
        FakeTensor((2, 2**30, 576), torch.bfloat16),
        FakeTensor((2, 1, 2048), torch.int32),
        192**-0.5,
        kv_lora_rank=512,
    )
    assert error == "requires the flattened KV address space to fit int32"
