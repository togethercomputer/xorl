"""Logic for loading / preparing a dataset once over all processes."""

import time
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

from ...arguments import Arguments
from .constants import DEFAULT_DATASET_PREPARED_PATH


LOCK_FILE_NAME = "datasets_prep.lock"
READY_FILE_NAME = "datasets_ready.flag"
PROCESS_COUNTER_FILE_NAME = "process_counter.txt"


class FileLockLoader:
    """
    Simple class for abstracting single process data loading / processing. The first
    process that creates a lock file does the work; the remaining procesees simply load
    the preprocessed dataset once the first process is done.
    """

    def __init__(self, args: Arguments):
        self.args = args
        self.dataset_prepared_path = args.data.dataset_prepared_path or DEFAULT_DATASET_PREPARED_PATH
        self.lock_file_path = Path(self.dataset_prepared_path) / LOCK_FILE_NAME
        self.ready_flag_path = Path(self.dataset_prepared_path) / READY_FILE_NAME
        self.counter_path = Path(self.dataset_prepared_path) / PROCESS_COUNTER_FILE_NAME

    def load(self, load_fn: Callable[[], Any]) -> Any:
        # Ensure directory exists
        Path(self.dataset_prepared_path).mkdir(parents=True, exist_ok=True)

        with FileLock(str(self.lock_file_path)):
            self._increment_counter()

            if not self.ready_flag_path.exists():
                # First process does the work
                result = load_fn()
                self.ready_flag_path.touch()
                return result
            else:
                # Other processes wait for the first process to finish
                # and then load the already prepared data
                while not self.ready_flag_path.exists():
                    time.sleep(1.0)  # Sleep for 1 second
                return load_fn()  # Load the prepared data

    def _increment_counter(self):
        """Safely increment the process counter."""
        try:
            if self.counter_path.exists():
                counter_content = self.counter_path.read_text().strip()
                count = int(counter_content) if counter_content else 0
            else:
                count = 0
            self.counter_path.write_text(str(count + 1))
        except (ValueError, OSError):
            # Handle corrupted counter file or I/O errors
            # Reset to 1 for this process
            self.counter_path.write_text("1")

    def cleanup(self):
        """Clean up ready flag when last process is done."""
        with FileLock(str(self.lock_file_path)):
            try:
                counter_content = self.counter_path.read_text().strip()
                count = int(counter_content) if counter_content else 0
                count -= 1

                if count <= 0:
                    # Last process cleans everything up
                    self.ready_flag_path.unlink(missing_ok=True)
                    self.counter_path.unlink(missing_ok=True)
                else:
                    # Still have active processes
                    self.counter_path.write_text(str(count))
            except (ValueError, OSError):
                # Handle corrupted counter file or I/O errors
                # Force cleanup since we can't determine the count
                self.ready_flag_path.unlink(missing_ok=True)
                self.counter_path.unlink(missing_ok=True)
