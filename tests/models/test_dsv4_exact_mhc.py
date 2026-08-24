from types import SimpleNamespace

import pytest
import torch

from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import (
    _build_cp_serving_mhc_segments,
    _build_serving_mhc_segments,
)
from xorl.ops.dsv4.cp_utils import Dsv4ExactCPLayout
from xorl.ops.dsv4.hyper_connection import DeepSeekV4HyperConnectionUtil


pytestmark = pytest.mark.cpu


def test_serving_mhc_segments_encode_prefill_then_m1_and_padding():
    segments = _build_serving_mhc_segments(
        compute_rows=10,
        sample_lengths=[4, 3],
        sampler_prefill_lengths=torch.tensor([2, 3]),
    )
    assert segments == (2, 1, 1, 3, 3)


def test_serving_mhc_segments_reject_invalid_boundary():
    with pytest.raises(ValueError, match="0 < prefill <= sample length"):
        _build_serving_mhc_segments(
            compute_rows=4,
            sample_lengths=[4],
            sampler_prefill_lengths=torch.tensor([5]),
        )


def _cp_layout():
    return Dsv4ExactCPLayout(
        local_storage_indices=torch.arange(4),
        local_logical_rows=torch.tensor([2, 3, 4, 5, -1, -1]),
        local_request_ids=torch.tensor([0, 0, 1, 1, -1, -1]),
        local_request_positions=torch.tensor([2, 3, 0, 1, 0, 0]),
        local_live_count=4,
        compute_rows=6,
        gather_order=torch.arange(8),
        global_logical_rows=torch.arange(8),
        global_request_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        global_request_positions=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        request_ids=(0, 1),
        local_request_row_indices=(torch.tensor([0, 1]), torch.tensor([2, 3])),
        global_request_row_indices=(torch.arange(4), torch.arange(4, 8)),
    )


def test_cp_serving_mhc_segments_preserve_global_prefill_launch_sizes():
    segments = _build_cp_serving_mhc_segments(
        layout=_cp_layout(),
        sampler_prefill_lengths=torch.tensor([3, 2]),
    )

    assert [segment.launch_rows for segment in segments] == [3, 1, 2, 2]
    assert [segment.source_rows for segment in segments] == [(0,), (1,), (2, 3), (4, 5)]
    assert [segment.launch_positions for segment in segments] == [(2,), (0,), (0, 1), (0, 1)]


def test_exact_mhc_replay_invokes_one_prefill_then_m1(monkeypatch):
    calls = []

    def fake_apply(residual, *_args):
        calls.append(residual.shape[1])
        shape = residual.shape[:-2]
        return (
            residual[..., 0, :],
            torch.zeros(*shape, 4),
            torch.zeros(*shape, 4, 4),
        )

    monkeypatch.setattr("xorl.ops.dsv4.hyper_connection._ExactMhcPreNorm.apply", fake_apply)
    util = DeepSeekV4HyperConnectionUtil(
        SimpleNamespace(rms_norm_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    )
    residual = torch.randn(1, 5, 4, 8, requires_grad=True)
    layer_input, post, comb = util.layer_pre_norm_exact(
        residual,
        torch.empty(24, 32),
        torch.empty(3),
        torch.empty(24),
        torch.empty(8),
        serving_segments=(3, 1, 1),
    )

    assert calls == [3, 1, 1]
    assert layer_input.shape == (1, 5, 8)
    assert post.shape == (1, 5, 4)
    assert comb.shape == (1, 5, 4, 4)
    layer_input.sum().backward()
    assert residual.grad is not None


def test_exact_cp_mhc_replay_uses_global_m_and_selects_local_rows(monkeypatch):
    calls = []

    def fake_apply(residual, *_args):
        calls.append(residual.shape[1])
        shape = residual.shape[:-2]
        return (
            residual[..., 0, :],
            torch.zeros(*shape, 4),
            torch.zeros(*shape, 4, 4),
        )

    monkeypatch.setattr("xorl.ops.dsv4.hyper_connection._ExactMhcPreNorm.apply", fake_apply)
    util = DeepSeekV4HyperConnectionUtil(
        SimpleNamespace(rms_norm_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    )
    residual = torch.randn(1, 6, 4, 8, requires_grad=True)
    segments = _build_cp_serving_mhc_segments(
        layout=_cp_layout(),
        sampler_prefill_lengths=torch.tensor([3, 2]),
    )

    layer_input, _post, _comb = util.layer_pre_norm_exact(
        residual,
        torch.empty(24, 32),
        torch.empty(3),
        torch.empty(24),
        torch.empty(8),
        serving_segments=segments,
    )

    assert calls == [3, 1, 2, 2]
    assert torch.equal(layer_input, residual[..., 0, :])
    layer_input.sum().backward()
    assert residual.grad is not None
