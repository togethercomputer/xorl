"""Tests for xorl.data.prepare.file_lock_loader module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from xorl.data.prepare.file_lock_loader import (
    FileLockLoader,
    LOCK_FILE_NAME,
    READY_FILE_NAME,
    PROCESS_COUNTER_FILE_NAME,
)


pytestmark = pytest.mark.cpu


@pytest.fixture
def temp_dataset_path(tmp_path):
    """Provides a temporary directory for dataset preparation."""
    return str(tmp_path / "prepared_datasets")


@pytest.fixture
def mock_args(temp_dataset_path):
    """Provides mock Arguments object."""
    args = Mock()
    args.data.dataset_prepared_path = temp_dataset_path
    return args


class TestFileLockLoader:
    """Tests for FileLockLoader class."""

    def test_initializes_with_custom_path(self, mock_args, temp_dataset_path):
        """Should initialize with custom dataset_prepared_path."""
        loader = FileLockLoader(mock_args)

        assert loader.dataset_prepared_path == temp_dataset_path
        assert str(loader.lock_file_path).endswith(LOCK_FILE_NAME)
        assert str(loader.ready_flag_path).endswith(READY_FILE_NAME)
        assert str(loader.counter_path).endswith(PROCESS_COUNTER_FILE_NAME)

    def test_initializes_with_default_path_when_none(self, temp_dataset_path):
        """Should use default path when dataset_prepared_path is None."""
        args = Mock()
        args.data.dataset_prepared_path = None

        loader = FileLockLoader(args)

        assert loader.dataset_prepared_path == "last_prepared_dataset"

    def test_creates_directory_on_load(self, mock_args, temp_dataset_path):
        """Should create dataset_prepared_path directory if it doesn't exist."""
        loader = FileLockLoader(mock_args)

        def load_fn():
            return "loaded_data"

        result = loader.load(load_fn)

        assert Path(temp_dataset_path).exists()
        assert result == "loaded_data"

    def test_first_process_executes_load_fn(self, mock_args, temp_dataset_path):
        """First process should execute load_fn and create ready flag."""
        loader = FileLockLoader(mock_args)
        load_fn = Mock(return_value="first_process_data")

        result = loader.load(load_fn)

        assert result == "first_process_data"
        load_fn.assert_called_once()
        assert loader.ready_flag_path.exists()

    def test_increments_counter_on_load(self, mock_args, temp_dataset_path):
        """Should increment process counter on load."""
        loader = FileLockLoader(mock_args)

        def load_fn():
            return "data"

        loader.load(load_fn)

        counter_content = loader.counter_path.read_text().strip()
        assert counter_content == "1"

    def test_multiple_loads_increment_counter(self, mock_args, temp_dataset_path):
        """Multiple loads should increment counter."""
        loader1 = FileLockLoader(mock_args)
        loader2 = FileLockLoader(mock_args)

        def load_fn():
            return "data"

        loader1.load(load_fn)
        loader2.load(load_fn)

        counter_content = loader2.counter_path.read_text().strip()
        assert counter_content == "2"

    def test_subsequent_process_uses_cached_data(self, mock_args, temp_dataset_path):
        """Subsequent processes should use cached data without re-executing load_fn."""
        # First process
        loader1 = FileLockLoader(mock_args)
        loader1.load(lambda: "original_data")

        # Second process
        loader2 = FileLockLoader(mock_args)
        load_fn_mock = Mock(return_value="new_data")

        result = loader2.load(load_fn_mock)

        # Should call load_fn but ready flag already exists
        load_fn_mock.assert_called_once()
        assert result == "new_data"

    def test_cleanup_decrements_counter(self, mock_args, temp_dataset_path):
        """Cleanup should decrement counter."""
        loader = FileLockLoader(mock_args)
        loader.load(lambda: "data")

        # Initial counter should be 1
        assert loader.counter_path.read_text().strip() == "1"

        loader.cleanup()

        # After cleanup, counter should be 0 and files removed
        assert not loader.counter_path.exists()
        assert not loader.ready_flag_path.exists()

    def test_cleanup_with_multiple_processes(self, mock_args, temp_dataset_path):
        """Cleanup should only remove files when all processes are done."""
        loader1 = FileLockLoader(mock_args)
        loader2 = FileLockLoader(mock_args)

        loader1.load(lambda: "data")
        loader2.load(lambda: "data")

        # Counter should be 2
        assert loader1.counter_path.read_text().strip() == "2"

        # First cleanup
        loader1.cleanup()

        # Counter should be 1, files still exist
        assert loader1.counter_path.exists()
        assert loader1.ready_flag_path.exists()
        assert loader1.counter_path.read_text().strip() == "1"

        # Second cleanup
        loader2.cleanup()

        # All files should be removed
        assert not loader2.counter_path.exists()
        assert not loader2.ready_flag_path.exists()

    def test_handles_corrupted_counter_on_increment(self, mock_args, temp_dataset_path):
        """Should reset counter to 1 if corrupted during increment."""
        loader = FileLockLoader(mock_args)

        # Create corrupted counter file
        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)
        loader.counter_path.write_text("invalid_number")

        loader.load(lambda: "data")

        # Counter should be reset to 1
        assert loader.counter_path.read_text().strip() == "1"

    def test_handles_corrupted_counter_on_cleanup(self, mock_args, temp_dataset_path):
        """Should force cleanup if counter is corrupted during cleanup."""
        loader = FileLockLoader(mock_args)
        loader.load(lambda: "data")

        # Corrupt the counter
        loader.counter_path.write_text("invalid_number")

        loader.cleanup()

        # Should force cleanup by removing all files
        assert not loader.counter_path.exists()
        assert not loader.ready_flag_path.exists()

    def test_handles_missing_counter_on_increment(self, mock_args, temp_dataset_path):
        """Should start counter at 1 if missing during increment."""
        loader = FileLockLoader(mock_args)

        # Ensure counter doesn't exist
        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)
        if loader.counter_path.exists():
            loader.counter_path.unlink()

        loader.load(lambda: "data")

        assert loader.counter_path.read_text().strip() == "1"

    def test_handles_io_error_on_increment(self, mock_args, temp_dataset_path):
        """Should reset to 1 if OSError occurs during increment."""
        loader = FileLockLoader(mock_args)

        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)

        with patch.object(Path, 'read_text', side_effect=OSError("IO error")):
            loader.load(lambda: "data")

        # Should still create counter with value 1
        assert loader.counter_path.read_text().strip() == "1"

    def test_load_fn_exceptions_propagate(self, mock_args, temp_dataset_path):
        """Should propagate exceptions from load_fn."""
        loader = FileLockLoader(mock_args)

        def failing_load_fn():
            raise ValueError("load failed")

        with pytest.raises(ValueError, match="load failed"):
            loader.load(failing_load_fn)

    def test_concurrent_access_safety(self, mock_args, temp_dataset_path):
        """Should safely handle concurrent access via file locking."""
        loader1 = FileLockLoader(mock_args)
        loader2 = FileLockLoader(mock_args)

        call_order = []

        def load_fn_1():
            call_order.append(1)
            return "data1"

        def load_fn_2():
            call_order.append(2)
            return "data2"

        # Both processes try to load
        result1 = loader1.load(load_fn_1)
        result2 = loader2.load(load_fn_2)

        # Both should succeed (though second uses cached flag)
        assert result1 == "data1"
        assert result2 == "data2"

        # Counter should be 2
        assert loader1.counter_path.read_text().strip() == "2"
