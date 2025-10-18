"""E2E test fixtures.

Pytest auto-discovers this file for all tests under tests/e2e/.
Non-fixture helpers live in e2e_utils.py (importable as a regular module).
"""

import pytest

from .e2e_utils import create_tiny_model_dir


@pytest.fixture
def tmp_workspace(tmp_path):
    """Clean temporary workspace directory."""
    return str(tmp_path)


@pytest.fixture
def tiny_dense_model_dir(tmp_workspace):
    """Tiny Qwen3 dense model directory (random init, no download)."""
    return create_tiny_model_dir(tmp_workspace, model_type="dense")


@pytest.fixture
def tiny_dense_model_dir_with_weights(tmp_workspace):
    """Tiny Qwen3 dense model directory with saved weights."""
    return create_tiny_model_dir(tmp_workspace, model_type="dense", save_weights=True)


@pytest.fixture
def tiny_moe_model_dir(tmp_workspace):
    """Tiny Qwen3-MoE model directory (random init, no download)."""
    return create_tiny_model_dir(tmp_workspace, model_type="moe")


@pytest.fixture
def tiny_moe_model_dir_with_weights(tmp_workspace):
    """Tiny Qwen3-MoE model directory with saved weights."""
    return create_tiny_model_dir(tmp_workspace, model_type="moe", save_weights=True)


@pytest.fixture
def small_dense_model_dir_with_weights(tmp_workspace):
    """Small Qwen3 dense model (hidden=256) with saved weights."""
    return create_tiny_model_dir(tmp_workspace, model_type="dense_large", save_weights=True)


@pytest.fixture
def small_moe_model_dir_with_weights(tmp_workspace):
    """Small Qwen3-MoE model (moe_intermediate=64) with saved weights.

    Larger moe_intermediate_size for NF4 group_size=64 compatibility.
    """
    return create_tiny_model_dir(tmp_workspace, model_type="moe_large", save_weights=True)
