"""Tests for xorl.data.prepare.file_lock_loader module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from xorl.data.prepare.file_lock_loader import (
    LOCK_FILE_NAME,
    PROCESS_COUNTER_FILE_NAME,
    READY_FILE_NAME,
    FileLockLoader,
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


class TestFileLockLoaderInitAndLoad:
    """Tests for initialization and load behavior."""

    def test_init_load_and_counter_incrementing(self, mock_args, temp_dataset_path):
        """Covers initialization paths, directory creation, first-process load, counter incrementing,
        multiple loads, and subsequent process caching."""
        # Custom path initialization
        loader = FileLockLoader(mock_args)
        assert loader.dataset_prepared_path == temp_dataset_path
        assert str(loader.lock_file_path).endswith(LOCK_FILE_NAME)
        assert str(loader.ready_flag_path).endswith(READY_FILE_NAME)
        assert str(loader.counter_path).endswith(PROCESS_COUNTER_FILE_NAME)

        # Default path when None
        args_none = Mock()
        args_none.data.dataset_prepared_path = None
        loader_none = FileLockLoader(args_none)
        assert loader_none.dataset_prepared_path == "last_prepared_dataset"

        # First process: creates directory, executes load_fn, creates ready flag, counter=1
        load_fn = Mock(return_value="first_process_data")
        result = loader.load(load_fn)
        assert result == "first_process_data"
        load_fn.assert_called_once()
        assert Path(temp_dataset_path).exists()
        assert loader.ready_flag_path.exists()
        assert loader.counter_path.read_text().strip() == "1"

        # Second load increments counter
        loader2 = FileLockLoader(mock_args)
        load_fn2 = Mock(return_value="new_data")
        result2 = loader2.load(load_fn2)
        assert result2 == "new_data"
        load_fn2.assert_called_once()
        assert loader2.counter_path.read_text().strip() == "2"


class TestFileLockLoaderCleanup:
    """Tests for cleanup behavior."""

    def test_cleanup_single_and_multiple_processes(self, mock_args, temp_dataset_path):
        """Covers single-process cleanup, multi-process partial cleanup, and full cleanup."""
        loader1 = FileLockLoader(mock_args)
        loader2 = FileLockLoader(mock_args)

        loader1.load(lambda: "data")
        loader2.load(lambda: "data")
        assert loader1.counter_path.read_text().strip() == "2"

        # First cleanup: counter=1, files still exist
        loader1.cleanup()
        assert loader1.counter_path.exists()
        assert loader1.ready_flag_path.exists()
        assert loader1.counter_path.read_text().strip() == "1"

        # Second cleanup: all files removed
        loader2.cleanup()
        assert not loader2.counter_path.exists()
        assert not loader2.ready_flag_path.exists()


class TestFileLockLoaderErrorHandling:
    """Tests for corrupted state, IO errors, and exception propagation."""

    def test_corrupted_counter_io_error_and_exceptions(self, mock_args, temp_dataset_path):
        """Covers corrupted counter on increment/cleanup, missing counter, IO error, load_fn exception, and concurrent access."""
        loader = FileLockLoader(mock_args)

        # Corrupted counter on increment -> reset to 1
        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)
        loader.counter_path.write_text("invalid_number")
        loader.load(lambda: "data")
        assert loader.counter_path.read_text().strip() == "1"

        # Corrupted counter on cleanup -> force cleanup
        loader.counter_path.write_text("invalid_number")
        loader.cleanup()
        assert not loader.counter_path.exists()
        assert not loader.ready_flag_path.exists()

        # Missing counter on increment -> start at 1
        loader2 = FileLockLoader(mock_args)
        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)
        if loader2.counter_path.exists():
            loader2.counter_path.unlink()
        loader2.load(lambda: "data")
        assert loader2.counter_path.read_text().strip() == "1"

        # IO error on increment -> reset to 1
        loader3 = FileLockLoader(mock_args)
        loader3.cleanup()
        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)
        with patch.object(Path, "read_text", side_effect=OSError("IO error")):
            loader3.load(lambda: "data")
        assert loader3.counter_path.read_text().strip() == "1"

        # load_fn exception propagation
        loader4 = FileLockLoader(mock_args)
        loader4.cleanup()
        with pytest.raises(ValueError, match="load failed"):
            loader4.load(lambda: (_ for _ in ()).throw(ValueError("load failed")))

        # Concurrent access safety
        loader_a = FileLockLoader(mock_args)
        loader_b = FileLockLoader(mock_args)
        # Clean up first
        Path(temp_dataset_path).mkdir(parents=True, exist_ok=True)
        for f in [loader_a.counter_path, loader_a.ready_flag_path]:
            if f.exists():
                f.unlink()
        r1 = loader_a.load(lambda: "data1")
        r2 = loader_b.load(lambda: "data2")
        assert r1 == "data1" and r2 == "data2"
        assert loader_a.counter_path.read_text().strip() == "2"
