"""Tests for xorl.distributed.parallel_state module."""

import pytest
import torch
from unittest.mock import Mock, patch

from xorl.distributed.parallel_state import (
    ParallelState,
    get_parallel_state,
    init_parallel_state,
    init_ep_mesh_matrix,
    requires_mesh,
)

pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


class TestEPMeshMatrixAndRequiresMesh:
    """Test init_ep_mesh_matrix layouts and requires_mesh decorator."""

    def test_ep_mesh_matrix_and_requires_mesh(self):
        """EP mesh matrix: row-major, transposed, edge cases; requires_mesh raises/allows correctly."""
        # ep_outside=True: row-major
        mesh = init_ep_mesh_matrix(ep_size=2, ep_fsdp_size=4, ep_outside=True)
        assert mesh.shape == (2, 4) and mesh.dtype == torch.int
        assert torch.equal(mesh, torch.arange(8).view(2, 4))

        # ep_outside=False: transposed
        mesh = init_ep_mesh_matrix(ep_size=2, ep_fsdp_size=4, ep_outside=False)
        assert torch.equal(mesh, torch.arange(8).view(4, 2).transpose(0, 1))

        # Edge: single ep_size / ep_fsdp_size
        assert torch.equal(init_ep_mesh_matrix(1, 4, True), torch.arange(4).unsqueeze(0))
        assert torch.equal(init_ep_mesh_matrix(4, 1, True), torch.arange(4).unsqueeze(1))

        # requires_mesh decorator
        class MC:
            def __init__(self, m):
                self.device_mesh = m
            @requires_mesh
            def go(self):
                return "ok"

        assert MC(Mock()).go() == "ok"
        with pytest.raises(ValueError, match="Device mesh is not initialized"):
            MC(None).go()


class TestParallelStateConstruction:
    """Test ParallelState construction, validation, properties, and enabled flags."""

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_defaults_and_uninitialized_properties(self, mock_is_init):
        """Default ParallelState: all sizes 1, not initialized, rank/world_size defaults; invalid cp_fsdp_mode raises."""
        state = ParallelState()
        assert state.dp_size == 1 and state.dp_replicate_size == 1 and state.dp_shard_size == 1
        assert state.tp_size == 1 and state.ep_size == 1 and state.pp_size == 1
        assert state.ringattn_size == 1 and state.ulysses_size == 1
        assert state.dp_mode == "fsdp2" and state.cp_fsdp_mode == "all"
        assert state.device_mesh is None and state.ep_fsdp_device_mesh is None
        assert state.is_initialized is False and state.global_rank == -1 and state.world_size == 1

        with pytest.raises(ValueError, match="Invalid cp_fsdp_mode"):
            ParallelState(cp_fsdp_mode="invalid")

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_rank', return_value=5)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=8)
    @patch('xorl.distributed.sequence_parallel.init_sequence_parallel')
    def test_custom_init_validation_and_enabled_flags(self, mock_sp, mock_ws, mock_rank, mock_init):
        """Custom init; validation errors; initialized properties; sp/fsdp enabled flags."""
        state = ParallelState(dp_size=4, dp_replicate_size=2, dp_shard_size=2, tp_size=2)
        assert state.dp_size == 4 and state.tp_size == 2
        assert state.is_initialized is True and state.global_rank == 5

        with pytest.raises(ValueError, match="product of parallel sizes should be equal to the world size"):
            ParallelState(dp_size=2, dp_shard_size=2, tp_size=2)
        with pytest.raises(ValueError, match="product of dp_replicate_size"):
            ParallelState(dp_size=8, dp_replicate_size=2, dp_shard_size=2)

        state2 = ParallelState(dp_size=4, dp_shard_size=4, ulysses_size=2)
        assert state2.cp_enabled is True and state2.cp_size == 2 and state2.dp_shard_cp_enabled is True

        state3 = ParallelState(pp_size=2, tp_size=2, dp_size=2, dp_shard_size=2)
        assert state3.fsdp_size == 2  # 8 / (2 * 2)


class TestGetAndInitParallelState:
    """Test get_parallel_state and init_parallel_state functions."""

    def setup_method(self):
        import xorl.distributed.parallel_state as ps_module
        ps_module._PARALLEL_STATE = None

    def teardown_method(self):
        import xorl.distributed.parallel_state as ps_module
        ps_module._PARALLEL_STATE = None

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=8)
    def test_init_get_reinit_auto_dp_shard_default(self, mock_ws, mock_is_init, mock_version):
        """init sets state, get retrieves, re-init warns, auto dp_shard_size, get default when unset."""
        init_parallel_state(dp_size=4, tp_size=2, dp_mode="fsdp2")
        state = get_parallel_state()
        assert state.dp_size == 4 and state.tp_size == 2 and state.dp_mode == "fsdp2"

        with patch('xorl.distributed.parallel_state.logger.warning') as mock_warn:
            init_parallel_state(dp_size=8)
            mock_warn.assert_called_once_with("Parallel state has already been initialized.")
        assert get_parallel_state().dp_size == 4

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    def test_auto_dp_shard_and_default_uninitialized(self, mock_ws, mock_is_init, mock_version):
        """Auto dp_shard_size; device_type defaults; get_parallel_state returns default when unset."""
        init_parallel_state(dp_size=4)
        assert get_parallel_state().dp_shard_size == 4
        assert get_parallel_state().device_type in ["cuda", "cpu"]
