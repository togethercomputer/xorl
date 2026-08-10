from __future__ import annotations

import torch

from xorl.server.runner.model_runner import ModelRunner


def test_opd_layer_cache_gathers_only_valid_teacher_positions(monkeypatch):
    monkeypatch.setattr("xorl.server.runner.model_runner.get_device_type", lambda: "cpu")

    class FakeLayerCache:
        def __init__(self) -> None:
            self.requested_indices = None
            self.requested_slices = []

        def shape(self, teacher_id):
            return (2, 16, 3)

        def get(self, teacher_id, indices, *, device, dtype, cache_device=False):
            self.requested_indices = indices.detach().cpu().clone()
            rows = int(indices.numel())
            return torch.arange(rows * 2 * 3, dtype=dtype, device=device).reshape(rows, 2, 3)

        def get_layer_slice(self, teacher_id, indices, layer_start, layer_end, *, device, dtype, cache_device=False):
            self.requested_indices = indices.detach().cpu().clone()
            self.requested_slices.append((layer_start, layer_end))
            rows = int(indices.numel())
            layers = int(layer_end - layer_start)
            return torch.arange(rows * layers * 3, dtype=dtype, device=device).reshape(rows, layers, 3)

    runner = object.__new__(ModelRunner)
    layer_cache = FakeLayerCache()
    cache_indices = torch.tensor([[10, 11, 12], [13, 14, 15]])
    teacher_mask = torch.tensor([[False, True, False], [True, False, True]])

    gathered = runner._get_opd_teacher_layer_hidden_states(
        {"teacher_cache_indices": cache_indices},
        teacher_id=0,
        layer_cache=layer_cache,
        dtype=torch.float32,
        teacher_mask=teacher_mask,
        valid_mask=teacher_mask,
        cache_device=True,
    )

    torch.testing.assert_close(layer_cache.requested_indices, torch.tensor([11, 13, 15]))
    assert layer_cache.requested_slices == [(0, 2)]
    assert gathered.shape == (3, 2, 3)


def test_opd_layer_cache_fetcher_streams_layer_slices(monkeypatch):
    monkeypatch.setattr("xorl.server.runner.model_runner.get_device_type", lambda: "cpu")

    class FakeLayerCache:
        def __init__(self) -> None:
            self.requested_indices = None
            self.requested_slices = []

        def shape(self, teacher_id):
            return (5, 16, 3)

        def get_layer_slice(self, teacher_id, indices, layer_start, layer_end, *, device, dtype, cache_device=False):
            self.requested_indices = indices.detach().cpu().clone()
            self.requested_slices.append((layer_start, layer_end))
            rows = int(indices.numel())
            layers = int(layer_end - layer_start)
            base = torch.arange(rows * layers * 3, dtype=dtype, device=device).reshape(rows, layers, 3)
            return base + 100 * layer_start

    runner = object.__new__(ModelRunner)
    layer_cache = FakeLayerCache()
    cache_indices = torch.tensor([[10, 11, 12], [13, 14, 15]])
    teacher_mask = torch.tensor([[False, True, False], [True, False, True]])

    fetcher, num_layers = runner._get_opd_teacher_layer_fetcher(
        {"teacher_cache_indices": cache_indices},
        teacher_id=0,
        layer_cache=layer_cache,
        dtype=torch.float32,
        teacher_mask=teacher_mask,
        valid_mask=teacher_mask,
        cache_device=True,
    )

    first = fetcher(0, 2)
    second = fetcher(2, 5)

    assert num_layers == 5
    torch.testing.assert_close(layer_cache.requested_indices, torch.tensor([11, 13, 15]))
    assert layer_cache.requested_slices == [(0, 2), (2, 5)]
    assert first.shape == (3, 2, 3)
    assert second.shape == (3, 3, 3)
