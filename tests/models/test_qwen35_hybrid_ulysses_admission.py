"""Fail-closed admission for the exact hybrid Qwen3.5 program at Ulysses > 1.

Covers Ulysses degree checks, the aligned-collator attestation, and the
first-class kernel/toolchain pin. CPU-only.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
import triton

from xorl.models.auto import _validate_exact_qwen35_topology
from xorl.ops.kernel_config_pin import (
    MANIFEST_NAME,
    KernelConfigPinError,
    pin_exact_kernel_configs,
    seed_exact_kernel_config_pin,
)


HYBRID_LAYERS = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def _ps(u=8, **overrides):
    fields = dict(
        world_size=u,
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        pp_size=1,
        ep_size=1,
        cp_size=u,
        ringattn_size=1,
        ulysses_size=u,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _config(layer_types=None, heads=16, kv=2):
    return SimpleNamespace(
        _qwen35_exact_contract=True,
        model_type="qwen3_5",
        layer_types=layer_types,
        num_attention_heads=heads,
        num_key_value_heads=kv,
    )


@pytest.fixture()
def seeded_pin(tmp_path, monkeypatch):
    pin = tmp_path / "pin"
    cache = tmp_path / "cache-src"
    cache.mkdir()
    (cache / "dummy.json").write_text("{}")
    seed_exact_kernel_config_pin(str(pin), source_cache=str(cache))
    monkeypatch.setenv("XORL_EXACT_KERNEL_CONFIG_DIR", str(pin))
    monkeypatch.setenv("XORL_GDN_CP_ALIGN_COLLATOR", "1")
    monkeypatch.setenv("RANK", "3")
    return pin


class TestHybridUlyssesAdmission:
    def test_admitted_with_full_attestations(self, seeded_pin, monkeypatch):
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), _ps(8))
        clone = os.environ["TRITON_CACHE_DIR"]
        assert clone == os.path.realpath(str(seeded_pin / "clones" / "rank3"))
        assert os.path.isfile(os.path.join(clone, "dummy.json"))

    def test_gdn_free_dense_keeps_single_rank_refusal(self, seeded_pin):
        with pytest.raises(ValueError, match="admitted only for"):
            _validate_exact_qwen35_topology(_config(["full_attention"] * 4), _ps(8))

    def test_missing_collator_attestation_raises(self, seeded_pin, monkeypatch):
        monkeypatch.delenv("XORL_GDN_CP_ALIGN_COLLATOR", raising=False)
        with pytest.raises(ValueError, match="aligned collator"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), _ps(8))

    def test_missing_kernel_pin_raises(self, seeded_pin, monkeypatch):
        monkeypatch.delenv("XORL_EXACT_KERNEL_CONFIG_DIR", raising=False)
        with pytest.raises(KernelConfigPinError, match="toolchain pin"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), _ps(8))

    def test_toolchain_mismatch_raises(self, seeded_pin):
        manifest = seeded_pin / MANIFEST_NAME
        doctored = json.loads(manifest.read_text())
        doctored["torch"] = "0.0.0+nope"
        manifest.write_text(json.dumps(doctored))
        with pytest.raises(KernelConfigPinError, match="fingerprint mismatch"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), _ps(8))

    def test_uneven_head_split_raises(self, seeded_pin):
        with pytest.raises(ValueError, match="divisible by the Ulysses degree"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS, heads=6), _ps(4))

    def test_kv_non_divisor_raises(self, seeded_pin):
        with pytest.raises(ValueError, match="GQA replication"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS, kv=3), _ps(8))

    def test_ring_shape_keeps_generic_refusal(self, seeded_pin):
        ps = _ps(8, ringattn_size=2, cp_size=16, world_size=16)
        with pytest.raises(ValueError, match="admitted only for"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), ps)

    def test_unlisted_degree_keeps_generic_refusal(self, seeded_pin):
        with pytest.raises(ValueError, match="admitted only for"):
            _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), _ps(16))

    def test_single_rank_still_admitted_without_envs(self, monkeypatch):
        monkeypatch.delenv("XORL_EXACT_KERNEL_CONFIG_DIR", raising=False)
        monkeypatch.delenv("XORL_GDN_CP_ALIGN_COLLATOR", raising=False)
        _validate_exact_qwen35_topology(_config(HYBRID_LAYERS), _ps(1))


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
