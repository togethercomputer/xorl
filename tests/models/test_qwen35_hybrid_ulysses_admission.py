"""Kernel/toolchain pin coverage for exact Qwen3.5 programs (CPU-only)."""

from __future__ import annotations

import os

import pytest
import torch
import triton

from xorl.ops.kernel_config_pin import (
    KernelConfigPinError,
    pin_exact_kernel_configs,
    seed_exact_kernel_config_pin,
)


class TestKernelConfigPin:
    def test_seed_manifest_matches_runtime(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        fp = seed_exact_kernel_config_pin(str(tmp_path / "p"), source_cache=str(src))
        assert fp["torch"] == torch.__version__
        assert fp["triton"] == triton.__version__

    def test_seed_refuses_self_recursive_copy(self, tmp_path):
        with pytest.raises(KernelConfigPinError, match="recurse"):
            seed_exact_kernel_config_pin(str(tmp_path / "p"), source_cache=str(tmp_path))

    def test_unseeded_dir_raises(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("XORL_EXACT_KERNEL_CONFIG_DIR", str(empty))
        with pytest.raises(KernelConfigPinError, match="never .*seeded|no toolchain"):
            pin_exact_kernel_configs(rank=0)

    def test_pin_env_not_a_directory_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XORL_EXACT_KERNEL_CONFIG_DIR", str(tmp_path / "missing"))
        with pytest.raises(KernelConfigPinError, match="not an existing directory"):
            pin_exact_kernel_configs(rank=0)

    def _seeded(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "cfg.json").write_text("{}")
        pin = tmp_path / "pin"
        seed_exact_kernel_config_pin(str(pin), source_cache=str(src))
        return pin

    def test_pin_refuses_to_delete_unowned_clone(self, tmp_path, monkeypatch):
        """The ownership sentinel IS the deletion authorization: a clone-path
        directory this module did not create must never reach rmtree."""
        pin = self._seeded(tmp_path)
        unowned = pin / "clones" / "rank0"
        unowned.mkdir(parents=True)
        (unowned / "precious.txt").write_text("not ours to delete")
        monkeypatch.setenv("XORL_EXACT_KERNEL_CONFIG_DIR", str(pin))
        with pytest.raises(KernelConfigPinError, match="ownership sentinel"):
            pin_exact_kernel_configs(rank=0)
        assert (unowned / "precious.txt").exists()

    def test_pin_replaces_its_own_clone(self, tmp_path, monkeypatch):
        pin = self._seeded(tmp_path)
        monkeypatch.setenv("XORL_EXACT_KERNEL_CONFIG_DIR", str(pin))
        first = pin_exact_kernel_configs(rank=1)
        second = pin_exact_kernel_configs(rank=1)  # replaces the owned clone
        assert first == second and os.path.isdir(second)

    def test_seed_refuses_to_delete_unowned_cache(self, tmp_path):
        from xorl.ops.kernel_config_pin import CACHE_SUBDIR, OWNED_SENTINEL

        pin = self._seeded(tmp_path)
        (pin / CACHE_SUBDIR / OWNED_SENTINEL).unlink()  # simulate a foreign dir
        src2 = tmp_path / "src2"
        src2.mkdir()
        with pytest.raises(KernelConfigPinError, match="ownership sentinel"):
            seed_exact_kernel_config_pin(str(pin), source_cache=str(src2))

    def test_seed_refuses_implausible_parent(self, tmp_path):
        with pytest.raises(KernelConfigPinError, match="parent .*does not exist"):
            seed_exact_kernel_config_pin(
                str(tmp_path / "no" / "such" / "parent" / "pin"),
                source_cache=str(tmp_path),
            )
