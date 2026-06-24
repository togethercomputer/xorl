import torch

from xorl.utils import dist_utils


def test_all_reduce_scalar_group_size_one_skips_collective(monkeypatch):
    monkeypatch.setattr(dist_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "get_world_size", lambda group=None: 1)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("single-rank scalar reduce should not enter a collective")

    monkeypatch.setattr(dist_utils.dist, "all_reduce", fail_all_reduce)

    reduced = dist_utils.all_reduce((3.0, 5.0), op="sum", group="dp")

    assert reduced == [3.0, 5.0]


def test_all_reduce_scalar_uses_metadata_path(monkeypatch):
    reduce_calls = []

    monkeypatch.setattr(dist_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "get_world_size", lambda group=None: 4)

    def fake_all_reduce_metadata_tensor(tensor, op, group=None, device=None):
        reduce_calls.append((tensor.clone(), op, group, device))
        return tensor + 4.0

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("plain scalar/list reduce should use metadata reduction")

    monkeypatch.setattr(dist_utils, "all_reduce_metadata_tensor", fake_all_reduce_metadata_tensor)
    monkeypatch.setattr(dist_utils.dist, "all_reduce", fail_all_reduce)

    reduced = dist_utils.all_reduce([1.0, 2.0], op="sum")

    assert reduced == [5.0, 6.0]
    assert len(reduce_calls) == 1
    tensor, op, group, device = reduce_calls[0]
    assert tensor.device.type == "cpu"
    assert torch.equal(tensor, torch.tensor([1.0, 2.0]))
    assert op == torch.distributed.ReduceOp.SUM
    assert group is None
    assert device == "cpu"


def test_all_reduce_metadata_tensor_group_size_one_skips_collective(monkeypatch):
    monkeypatch.setattr(dist_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "get_world_size", lambda group=None: 1)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("single-rank metadata reduce should not enter a collective")

    monkeypatch.setattr(dist_utils.dist, "all_reduce", fail_all_reduce)

    tensor = torch.tensor([7.0])
    reduced = dist_utils.all_reduce_metadata_tensor(tensor, group="dp", device="cpu")

    assert torch.equal(reduced, tensor)
    assert reduced is not tensor
