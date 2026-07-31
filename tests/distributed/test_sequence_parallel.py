"""Tests for sequence parallel utilities."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xorl.distributed.sequence_parallel import (
    slice_position_embedding,
)
from xorl.distributed.sequence_parallel.ulysses import _Gather
from xorl.distributed.sequence_parallel.utils import pad_tensor, unpad_tensor


pytestmark = [pytest.mark.distributed]


class TestPaddingUtilities:
    """Test padding, unpadding, and roundtrip for pad_tensor/unpad_tensor."""

    def test_pad_unpad_roundtrip_and_dims(self):
        """Pad on dim=1 and dim=0, verify shapes and values, then roundtrip."""
        # Pad on dim=1
        x = torch.randn(2, 5, 4)
        padded = pad_tensor(x, dim=1, padding_size=3, padding_value=0)
        assert padded.shape == (2, 8, 4)
        assert torch.allclose(padded[:, :5, :], x)
        assert torch.allclose(padded[:, 5:, :], torch.zeros(2, 3, 4))

        # Pad on dim=0 with custom value
        x0 = torch.randn(3, 4, 5)
        padded0 = pad_tensor(x0, dim=0, padding_size=2, padding_value=-1)
        assert padded0.shape == (5, 4, 5)
        assert torch.allclose(padded0[:3, :, :], x0)
        assert torch.allclose(padded0[3:, :, :], torch.full((2, 4, 5), -1.0))

        # Unpad
        unpadded = unpad_tensor(padded, dim=1, padding_size=3)
        assert unpadded.shape == (2, 5, 4)
        assert torch.allclose(unpadded, padded[:, :5, :])

        # Roundtrip
        xr = torch.randn(3, 7, 5)
        assert torch.allclose(
            xr, unpad_tensor(pad_tensor(xr, dim=1, padding_size=3, padding_value=0), dim=1, padding_size=3)
        )


class TestSlicePositionEmbedding:
    """Test position embedding slicing."""

    def test_slice_position_embedding_no_group(self):
        """No SP group: slicing is a no-op."""
        cos = torch.randn(1, 8, 1)
        sin = torch.randn(1, 8, 1)
        result_cos, result_sin = slice_position_embedding((cos, sin), dim=1, sp_group=None)
        assert torch.equal(result_cos, cos)
        assert torch.equal(result_sin, sin)


def _run_gather_backward_worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        local = torch.tensor([[rank * 10.0 + 1.0, rank * 10.0 + 2.0]], requires_grad=True)
        gathered = _Gather.apply(dist.group.WORLD, local, 1, True)
        weights = [
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        ][rank]
        (gathered * weights).sum().backward()

        expected = torch.tensor([[6.0, 8.0]]) if rank == 0 else torch.tensor([[10.0, 12.0]])
        assert torch.allclose(local.grad, expected)
    finally:
        dist.destroy_process_group()


def test_gather_outputs_scale_grad_reduce_scatters_chunks(unused_tcp_port):
    mp.start_processes(
        _run_gather_backward_worker,
        args=(unused_tcp_port,),
        nprocs=2,
        start_method="spawn",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
