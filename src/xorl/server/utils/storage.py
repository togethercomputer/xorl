"""
Storage utilities for managing disk usage limits.

This module provides utilities for:
- Parsing human-readable size strings (e.g., "1GB", "500MB")
- Calculating directory sizes
- Checking storage limits before save operations
"""

import logging
import os
import re
from typing import Optional


logger = logging.getLogger(__name__)


class StorageLimitError(Exception):
    """
    Exception raised when storage limit is exceeded.

    This error is raised when a save operation (save_weights, save_weights_for_sampler)
    would exceed the configured storage_limit for the output directory.

    Attributes:
        current_size: Current size of the output directory in bytes
        limit: Configured storage limit in bytes
        current_size_human: Human-readable current size
        limit_human: Human-readable limit
    """

    def __init__(
        self,
        message: str,
        current_size: int,
        limit: int,
        output_dir: str,
    ):
        self.current_size = current_size
        self.limit = limit
        self.output_dir = output_dir
        self.current_size_human = bytes_to_human(current_size)
        self.limit_human = bytes_to_human(limit)
        super().__init__(message)


def parse_size_string(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Supports formats like:
    - "1GB", "1G", "1gb" -> 1073741824 bytes (1 * 1024^3)
    - "500MB", "500M", "500mb" -> 524288000 bytes (500 * 1024^2)
    - "100KB", "100K", "100kb" -> 102400 bytes (100 * 1024)
    - "1024B", "1024", "1024b" -> 1024 bytes

    Args:
        size_str: Human-readable size string

    Returns:
        Size in bytes

    Raises:
        ValueError: If the size string format is invalid

    Examples:
        >>> parse_size_string("1GB")
        1073741824
        >>> parse_size_string("500MB")
        524288000
        >>> parse_size_string("100KB")
        102400
    """
    if not size_str:
        raise ValueError("Size string cannot be empty")

    # Normalize: strip whitespace and convert to uppercase
    size_str = size_str.strip().upper()

    # Pattern: number followed by optional unit
    pattern = r"^(\d+(?:\.\d+)?)\s*(GB?|MB?|KB?|TB?|B)?$"
    match = re.match(pattern, size_str)

    if not match:
        raise ValueError(f"Invalid size format: '{size_str}'. Expected format like '1GB', '500MB', '100KB', or '1024B'")

    value = float(match.group(1))
    unit = match.group(2) or "B"

    # Normalize single-letter units
    unit_map = {
        "T": "TB",
        "G": "GB",
        "M": "MB",
        "K": "KB",
        "B": "B",
        "TB": "TB",
        "GB": "GB",
        "MB": "MB",
        "KB": "KB",
    }

    unit = unit_map.get(unit, unit)

    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    if unit not in multipliers:
        raise ValueError(f"Unknown unit: '{unit}'")

    return int(value * multipliers[unit])


def bytes_to_human(size_bytes: int) -> str:
    """
    Convert bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.5GB", "256MB")

    Examples:
        >>> bytes_to_human(1073741824)
        '1.00GB'
        >>> bytes_to_human(524288000)
        '500.00MB'
    """
    if size_bytes < 0:
        return "0B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(size_bytes)

    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1

    if unit_idx == 0:
        return f"{int(size)}B"
    return f"{size:.2f}{units[unit_idx]}"


def get_directory_size(path: str) -> int:
    """
    Calculate the total size of a directory and all its contents.

    Args:
        path: Path to directory

    Returns:
        Total size in bytes (0 if directory doesn't exist)
    """
    if not os.path.exists(path):
        return 0

    if os.path.isfile(path):
        return os.path.getsize(path)

    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    # Skip symbolic links to avoid counting files multiple times
                    if not os.path.islink(filepath):
                        total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    # Skip files we can't access
                    pass
    except (OSError, IOError) as e:
        logger.warning(f"Error calculating directory size for {path}: {e}")

    return total_size


def check_storage_limit(
    output_dir: str,
    storage_limit: Optional[str],
) -> None:
    """
    Check if the output directory exceeds the storage limit.

    Args:
        output_dir: Path to output directory
        storage_limit: Storage limit string (e.g., "1GB") or None to skip check

    Raises:
        StorageLimitError: If current usage exceeds the limit

    Example:
        >>> check_storage_limit("/data/outputs", "1GB")
        # Raises StorageLimitError if /data/outputs is larger than 1GB
    """
    if storage_limit is None:
        return

    try:
        limit_bytes = parse_size_string(storage_limit)
    except ValueError as e:
        logger.warning(f"Invalid storage_limit '{storage_limit}': {e}. Skipping storage check.")
        return

    current_size = get_directory_size(output_dir)

    if current_size > limit_bytes:
        raise StorageLimitError(
            f"Storage limit exceeded: output_dir '{output_dir}' is {bytes_to_human(current_size)} "
            f"which exceeds the limit of {bytes_to_human(limit_bytes)}. "
            f"Please clean up old checkpoints or increase the storage_limit.",
            current_size=current_size,
            limit=limit_bytes,
            output_dir=output_dir,
        )

    if limit_bytes > 0:
        logger.debug(
            f"Storage check passed: {bytes_to_human(current_size)} / {bytes_to_human(limit_bytes)} "
            f"({100 * current_size / limit_bytes:.1f}% used)"
        )
