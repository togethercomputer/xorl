"""Tests for xorl.distributed.parallel_state module."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from xorl.distributed.parallel_state import (
    ParallelState,
    get_parallel_state,
    init_parallel_state,
    init_ep_mesh_matrix,
    requires_mesh,
)

pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


class TestInitEPMeshMatrix:
    """Test suite for init_ep_mesh_matrix function."""

    def test_ep_outside_true(self):
        """Test EP mesh matrix creation with ep_outside=True."""
        ep_size = 2
        ep_fsdp_size = 4

        mesh = init_ep_mesh_matrix(ep_size=ep_size, ep_fsdp_size=ep_fsdp_size, ep_outside=True)

        assert mesh.shape == (ep_size, ep_fsdp_size)
        assert mesh.dtype == torch.int
        # Should be row-major: [[0, 1, 2, 3], [4, 5, 6, 7]]
        expected = torch.arange(8).view(2, 4)
        assert torch.equal(mesh, expected)

    def test_ep_outside_false(self):
        """Test EP mesh matrix creation with ep_outside=False."""
        ep_size = 2
        ep_fsdp_size = 4

        mesh = init_ep_mesh_matrix(ep_size=ep_size, ep_fsdp_size=ep_fsdp_size, ep_outside=False)

        assert mesh.shape == (ep_size, ep_fsdp_size)
        assert mesh.dtype == torch.int
        # Should be transposed: [[0, 2, 4, 6], [1, 3, 5, 7]]
        expected = torch.arange(8).view(4, 2).transpose(0, 1)
        assert torch.equal(mesh, expected)

    def test_single_ep_size(self):
        """Test with ep_size=1."""
        mesh = init_ep_mesh_matrix(ep_size=1, ep_fsdp_size=4, ep_outside=True)

        assert mesh.shape == (1, 4)
        assert torch.equal(mesh, torch.arange(4).unsqueeze(0))

    def test_single_ep_fsdp_size(self):
        """Test with ep_fsdp_size=1."""
        mesh = init_ep_mesh_matrix(ep_size=4, ep_fsdp_size=1, ep_outside=True)

        assert mesh.shape == (4, 1)
        assert torch.equal(mesh, torch.arange(4).unsqueeze(1))


class TestRequiresMeshDecorator:
    """Test suite for requires_mesh decorator."""

    def test_raises_error_when_mesh_none(self):
        """Test that decorated method raises error when device_mesh is None."""

        class MockClass:
            def __init__(self):
                self.device_mesh = None

            @requires_mesh
            def test_method(self):
                return "success"

        obj = MockClass()
        with pytest.raises(ValueError, match="Device mesh is not initialized"):
            obj.test_method()

    def test_allows_execution_when_mesh_exists(self):
        """Test that decorated method executes when device_mesh exists."""

        class MockClass:
            def __init__(self):
                self.device_mesh = Mock()

            @requires_mesh
            def test_method(self):
                return "success"

        obj = MockClass()
        result = obj.test_method()
        assert result == "success"


class TestParallelStateBasic:
    """Test suite for basic ParallelState functionality."""

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_default_initialization(self, mock_is_init):
        """Test ParallelState with default values."""
        state = ParallelState()

        assert state.dp_size == 1
        assert state.dp_replicate_size == 1
        assert state.dp_shard_size == 1
        assert state.tp_size == 1
        assert state.ep_size == 1
        assert state.pp_size == 1
        assert state.cp_size == 1
        assert state.ulysses_size == 1
        assert state.dp_mode == "fsdp1"
        assert state.include_sp_in_fsdp == True
        assert state.device_mesh is None
        assert state.ep_fsdp_device_mesh is None

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=8)
    def test_custom_initialization(self, mock_ws, mock_is_init):
        """Test ParallelState with custom values."""
        state = ParallelState(
            dp_size=4,
            dp_replicate_size=2,
            dp_shard_size=2,
            tp_size=2,
            dp_mode="fsdp2"
        )

        assert state.dp_size == 4
        assert state.dp_replicate_size == 2
        assert state.dp_shard_size == 2
        assert state.tp_size == 2
        assert state.dp_mode == "fsdp2"

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=8)
    def test_validation_world_size_mismatch(self, mock_ws, mock_is_init):
        """Test that validation fails when parallel sizes don't match world size."""
        # world_size=8, but we specify sizes that multiply to 4
        with pytest.raises(ValueError, match="product of parallel sizes should be equal to the world size"):
            ParallelState(dp_size=2, dp_shard_size=2, tp_size=2)

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=8)
    def test_validation_dp_sizes_mismatch(self, mock_ws, mock_is_init):
        """Test that validation fails when dp_replicate_size * dp_shard_size != dp_size."""
        # world_size=8, dp_size=8, but dp_replicate_size * dp_shard_size = 4 != 8
        with pytest.raises(ValueError, match="product of dp_replicate_size"):
            ParallelState(dp_size=8, dp_replicate_size=2, dp_shard_size=2)

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_validation_cp_size_not_supported(self, mock_is_init):
        """Test that cp_size > 1 raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Ring attention is not supported yet"):
            ParallelState(cp_size=2)

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_validation_decoupled_sp_not_supported(self, mock_is_init):
        """Test that include_sp_in_fsdp=False raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Decoupled sequence parallel"):
            ParallelState(include_sp_in_fsdp=False)


class TestParallelStateProperties:
    """Test suite for ParallelState properties."""

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_is_initialized_false(self, mock_is_init):
        """Test is_initialized property when distributed not initialized."""
        state = ParallelState()
        assert state.is_initialized == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=1)
    def test_is_initialized_true(self, mock_ws, mock_is_init):
        """Test is_initialized property when distributed is initialized."""
        state = ParallelState()
        assert state.is_initialized == True

    @patch.dict('os.environ', {'LOCAL_RANK': '3'})
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_local_rank_from_env(self, mock_is_init):
        """Test local_rank reads from environment variable."""
        state = ParallelState()
        assert state.local_rank == 3

    @patch.dict('os.environ', {}, clear=True)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_local_rank_default(self, mock_is_init):
        """Test local_rank returns -1 when not set."""
        state = ParallelState()
        assert state.local_rank == -1

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_rank', return_value=5)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=1)
    def test_global_rank_when_initialized(self, mock_ws, mock_get_rank, mock_is_init):
        """Test global_rank when distributed is initialized."""
        state = ParallelState()
        assert state.global_rank == 5

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_global_rank_when_not_initialized(self, mock_is_init):
        """Test global_rank returns -1 when not initialized."""
        state = ParallelState()
        assert state.global_rank == -1

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=1)
    def test_world_size_when_initialized(self, mock_get_ws, mock_is_init):
        """Test world_size when distributed is initialized."""
        state = ParallelState()
        assert state.world_size == 1

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_world_size_when_not_initialized(self, mock_is_init):
        """Test world_size returns 1 when not initialized."""
        state = ParallelState()
        assert state.world_size == 1


class TestParallelStateEnabledFlags:
    """Test suite for enabled flag properties."""

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    def test_dp_enabled_true(self, mock_ws, mock_is_init):
        """Test dp_enabled returns True when dp_size > 1."""
        state = ParallelState(dp_size=4, dp_shard_size=4)
        assert state.dp_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_dp_enabled_false(self, mock_is_init):
        """Test dp_enabled returns False when dp_size == 1."""
        state = ParallelState(dp_size=1)
        assert state.dp_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    def test_tp_enabled_true(self, mock_ws, mock_is_init):
        """Test tp_enabled returns True when tp_size > 1."""
        state = ParallelState(tp_size=2)
        assert state.tp_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_tp_enabled_false(self, mock_is_init):
        """Test tp_enabled returns False when tp_size == 1."""
        state = ParallelState(tp_size=1)
        assert state.tp_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    def test_pp_enabled_true(self, mock_ws, mock_is_init):
        """Test pp_enabled returns True when pp_size > 1."""
        state = ParallelState(pp_size=2)
        assert state.pp_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_pp_enabled_false(self, mock_is_init):
        """Test pp_enabled returns False when pp_size == 1."""
        state = ParallelState(pp_size=1)
        assert state.pp_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_ep_enabled_true(self, mock_is_init):
        """Test ep_enabled returns True when ep_size > 1."""
        state = ParallelState(ep_size=2)
        assert state.ep_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_ep_enabled_false(self, mock_is_init):
        """Test ep_enabled returns False when ep_size == 1."""
        state = ParallelState(ep_size=1)
        assert state.ep_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    @patch('xorl.distributed.sequence_parallel.init_sequence_parallel')
    def test_sp_enabled_with_ulysses(self, mock_init_sp, mock_ws, mock_is_init):
        """Test sp_enabled returns True when ulysses_size > 1."""
        state = ParallelState(ulysses_size=2)
        assert state.sp_enabled == True
        # Verify that init_sequence_parallel was called since device_mesh is None
        mock_init_sp.assert_called_once()

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_sp_enabled_false(self, mock_is_init):
        """Test sp_enabled returns False when both cp and ulysses are 1."""
        state = ParallelState(cp_size=1, ulysses_size=1)
        assert state.sp_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    @patch('xorl.distributed.sequence_parallel.init_sequence_parallel')
    def test_sp_size_calculation(self, mock_init_sp, mock_ws, mock_is_init):
        """Test sp_size is product of ulysses_size and cp_size."""
        state = ParallelState(ulysses_size=2)
        assert state.sp_size == 2

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_fsdp_enabled_with_fsdp1(self, mock_is_init):
        """Test fsdp_enabled is True with fsdp1 mode."""
        state = ParallelState(dp_mode="fsdp1")
        assert state.fsdp_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_fsdp_enabled_with_fsdp2(self, mock_is_init):
        """Test fsdp_enabled is True with fsdp2 mode."""
        state = ParallelState(dp_mode="fsdp2")
        assert state.fsdp_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_fsdp_enabled_with_ddp(self, mock_is_init):
        """Test fsdp_enabled is False with ddp mode."""
        state = ParallelState(dp_mode="ddp")
        assert state.fsdp_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    def test_fsdp_size_calculation(self, mock_ws, mock_is_init):
        """Test fsdp_size calculation."""
        state = ParallelState(pp_size=2, tp_size=2)
        # world_size=4, pp_size=2, tp_size=2 -> fsdp_size = 4 / (2 * 2) = 1
        assert state.fsdp_size == 1

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    @patch('xorl.distributed.sequence_parallel.init_sequence_parallel')
    def test_ulysses_enabled_true(self, mock_init_sp, mock_ws, mock_is_init):
        """Test ulysses_enabled returns True when ulysses_size > 1."""
        state = ParallelState(ulysses_size=2)
        assert state.ulysses_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_ulysses_enabled_false(self, mock_is_init):
        """Test ulysses_enabled returns False when ulysses_size == 1."""
        state = ParallelState(ulysses_size=1)
        assert state.ulysses_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    def test_dp_replicate_enabled_true(self, mock_ws, mock_is_init):
        """Test dp_replicate_enabled returns True when dp_replicate_size > 1."""
        state = ParallelState(dp_size=4, dp_replicate_size=2, dp_shard_size=2)
        assert state.dp_replicate_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_dp_replicate_enabled_false(self, mock_is_init):
        """Test dp_replicate_enabled returns False when dp_replicate_size == 1."""
        state = ParallelState(dp_replicate_size=1)
        assert state.dp_replicate_enabled == False

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    def test_dp_shard_enabled_true(self, mock_ws, mock_is_init):
        """Test dp_shard_enabled returns True when dp_shard_size >= 1."""
        state = ParallelState(dp_size=2, dp_shard_size=2)
        assert state.dp_shard_enabled == True

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    @patch('xorl.distributed.sequence_parallel.init_sequence_parallel')
    def test_dp_shard_sp_enabled(self, mock_init_sp, mock_ws, mock_is_init):
        """Test dp_shard_sp_enabled is True when both dp_shard and sp are enabled."""
        state = ParallelState(dp_size=2, dp_shard_size=2, ulysses_size=2)
        assert state.dp_shard_sp_enabled == True


class TestGetParallelState:
    """Test suite for get_parallel_state function."""

    def setup_method(self):
        """Clear the global parallel state before each test."""
        import xorl.distributed.parallel_state as ps_module
        ps_module._PARALLEL_STATE = None

    def teardown_method(self):
        """Clear the global parallel state after each test."""
        import xorl.distributed.parallel_state as ps_module
        ps_module._PARALLEL_STATE = None

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_returns_default_when_not_initialized(self, mock_is_init):
        """Test get_parallel_state returns default state when not initialized."""
        state = get_parallel_state()

        assert isinstance(state, ParallelState)
        assert state.dp_size == 1
        assert state.tp_size == 1

    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    def test_returns_initialized_state(self, mock_ws, mock_is_init):
        """Test get_parallel_state returns initialized state."""
        import xorl.distributed.parallel_state as ps_module

        # Manually set the global state
        ps_module._PARALLEL_STATE = ParallelState(dp_size=4, dp_shard_size=4)

        state = get_parallel_state()
        assert state.dp_size == 4


class TestInitParallelState:
    """Test suite for init_parallel_state function."""

    def setup_method(self):
        """Clear the global parallel state before each test."""
        import xorl.distributed.parallel_state as ps_module
        ps_module._PARALLEL_STATE = None

    def teardown_method(self):
        """Clear the global parallel state after each test."""
        import xorl.distributed.parallel_state as ps_module
        ps_module._PARALLEL_STATE = None

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_initializes_default_state(self, mock_is_init, mock_version):
        """Test init_parallel_state with default parameters."""
        init_parallel_state()

        state = get_parallel_state()
        assert state.dp_size == 1
        assert state.tp_size == 1

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=8)
    def test_initializes_custom_state(self, mock_ws, mock_is_init, mock_version):
        """Test init_parallel_state with custom parameters."""
        init_parallel_state(dp_size=4, tp_size=2, dp_mode="fsdp2")

        state = get_parallel_state()
        assert state.dp_size == 4
        assert state.tp_size == 2
        assert state.dp_mode == "fsdp2"

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=4)
    def test_auto_sets_dp_shard_size(self, mock_ws, mock_is_init, mock_version):
        """Test that dp_shard_size is automatically set to dp_size when needed."""
        init_parallel_state(dp_size=4)

        state = get_parallel_state()
        assert state.dp_shard_size == 4

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=True)
    @patch('xorl.distributed.parallel_state.dist.get_world_size', return_value=2)
    def test_warns_when_already_initialized(self, mock_ws, mock_is_init, mock_version):
        """Test that initializing twice produces a warning."""
        init_parallel_state(dp_size=2, dp_shard_size=2)

        with patch('xorl.distributed.parallel_state.logger.warning') as mock_warn:
            init_parallel_state(dp_size=4)
            mock_warn.assert_called_once_with("Parallel state has already been initialized.")

        # State should not change
        state = get_parallel_state()
        assert state.dp_size == 2

    @patch('xorl.distributed.parallel_state.is_torch_version_greater_than', return_value=False)
    @patch('xorl.distributed.parallel_state.dist.is_initialized', return_value=False)
    def test_device_type_default(self, mock_is_init, mock_version):
        """Test that device_type defaults to cuda/cpu based on availability."""
        init_parallel_state()

        state = get_parallel_state()
        assert state.device_type in ["cuda", "cpu"]
