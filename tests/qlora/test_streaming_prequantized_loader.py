"""CPU contracts for bounded deferred pre-quantized QLoRA loading."""

import json

import pytest
import torch
import torch.nn as nn

from xorl.qlora import utils as qlora_utils
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts


pytestmark = pytest.mark.cpu


class _FakeSafeTensorFile:
    def __init__(self, tensors, reads):
        self._tensors = tensors
        self._reads = reads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return self._tensors.keys()

    def get_tensor(self, key):
        self._reads.append(key)
        return self._tensors[key]


def _assert_load_selected_shard_cache_reads_only_requested_keys(monkeypatch):
    """Selected loading must not materialize unrelated tensors or shards."""
    weight_map = {
        "layer.0.q_proj.weight": "shard-a.safetensors",
        "layer.0.q_proj.weight_scale_inv": "shard-a.safetensors",
        "layer.0.k_proj.weight": "shard-b.safetensors",
        "layer.0.unrelated.weight": "shard-a.safetensors",
        "layer.1.unrelated.weight": "shard-unused.safetensors",
    }
    shard_tensors = {
        "shard-a.safetensors": {
            "layer.0.q_proj.weight": torch.tensor([1]),
            "layer.0.q_proj.weight_scale_inv": torch.tensor([2]),
            "layer.0.unrelated.weight": torch.tensor([99]),
        },
        "shard-b.safetensors": {"layer.0.k_proj.weight": torch.tensor([3])},
        "shard-unused.safetensors": {"layer.1.unrelated.weight": torch.tensor([100])},
    }
    requested = (
        "layer.0.k_proj.weight",
        "layer.0.q_proj.weight",
        "layer.0.q_proj.weight_scale_inv",
    )
    opened = []
    reads = []

    def fake_cached_file(_weights_path, shard_file, *args, **kwargs):
        del args, kwargs
        opened.append(shard_file)
        return shard_file

    def fake_safe_open(shard_file, *args, **kwargs):
        del args, kwargs
        return _FakeSafeTensorFile(shard_tensors[shard_file], reads)

    monkeypatch.setattr(qlora_utils, "cached_file", fake_cached_file)
    monkeypatch.setattr(qlora_utils, "safe_open", fake_safe_open)

    cache = qlora_utils._load_selected_shard_cache(requested, weight_map, "checkpoint")

    assert set(opened) == {"shard-a.safetensors", "shard-b.safetensors"}
    assert set(reads) == set(requested)
    assert len(reads) == len(requested)
    assert set(cache) == {"shard-a.safetensors", "shard-b.safetensors"}
    assert {key for shard in cache.values() for key in shard} == set(requested)

    cache.release()
    assert not cache


def _assert_prequantized_module_key_plan_policy():
    for source_fqn, merge_sources in (
        ("model.layers.0.self_attn", ("q_proj", "k_proj", "v_proj")),
        ("model.layers.0.mlp", ("gate_proj", "up_proj")),
    ):
        module = BlockFP8QLoRALinear(
            in_features=128,
            out_features=128 * len(merge_sources),
            r=1,
            lora_alpha=1,
            device=torch.device("meta"),
        )
        module._is_prequantized = True
        module._source_quant_format = "block_fp8"
        module._source_fqn = source_fqn
        module._merge_sources = merge_sources

        expected = {
            f"{source_fqn}.{projection}.{suffix}"
            for projection in merge_sources
            for suffix in ("weight", "weight_scale_inv")
        }
        weight_map = dict.fromkeys(expected, "selected.safetensors")
        weight_map.update(
            {
                f"{source_fqn}.unrelated.weight": "selected.safetensors",
                f"{source_fqn}.{merge_sources[0]}.weight_scale": "selected.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "other.safetensors",
            }
        )

        keys = qlora_utils._prequantized_module_keys(module, weight_map)

        assert len(keys) == len(expected)
        assert set(keys) == expected

    _assert_prequantized_module_keys_select_exactly_one_ep16_expert_slice()
    _assert_prequantized_module_keys_fail_before_io_when_a_pair_is_missing()


def _assert_prequantized_module_keys_select_exactly_one_ep16_expert_slice():
    module = BlockFP8QLoRAMoeExperts(
        num_local_experts=16,
        num_experts=256,
        intermediate_size=128,
        hidden_size=128,
        r=1,
        lora_alpha=1,
        expert_offset=112,
        device=torch.device("meta"),
    )
    module._source_fqn = "model.layers.3.mlp.experts"
    module._source_quant_format = "block_fp8"
    expected = {
        f"{module._source_fqn}.{expert}.{projection}.{suffix}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for expert in range(112, 128)
        for suffix in ("weight", "weight_scale_inv")
    }
    weight_map = dict.fromkeys(expected, "experts.safetensors")
    weight_map["model.layers.3.mlp.experts.111.gate_proj.weight"] = "experts.safetensors"
    weight_map["model.layers.3.mlp.experts.128.gate_proj.weight"] = "experts.safetensors"

    keys = qlora_utils._prequantized_module_keys(module, weight_map)

    assert len(keys) == 16 * 3 * 2 == 96
    assert set(keys) == expected


def _assert_prequantized_module_keys_fail_before_io_when_a_pair_is_missing():
    module = BlockFP8QLoRALinear(
        in_features=128,
        out_features=128,
        r=1,
        lora_alpha=1,
        device=torch.device("meta"),
    )
    module._is_prequantized = True
    module._source_quant_format = "block_fp8"
    module._source_fqn = "model.layers.0.self_attn.o_proj"
    module._merge_sources = None

    with pytest.raises(RuntimeError, match="missing 1 QLoRA tensor"):
        qlora_utils._prequantized_module_keys(
            module,
            {f"{module._source_fqn}.weight": "model.safetensors"},
        )


def _assert_retained_shard_cache_ignores_consumer_clear_until_release():
    """MoE projection-level clear calls cannot evict later projection tensors."""
    gate_key = "experts.0.gate_proj.weight"
    up_key = "experts.0.up_proj.weight"
    cache = qlora_utils._RetainedShardCache(
        {"model.safetensors": {gate_key: torch.tensor([1]), up_key: torch.tensor([2])}}
    )

    cache.clear()  # BlockFP8QLoRAMoeExperts does this after each projection.

    assert torch.equal(cache["model.safetensors"][up_key], torch.tensor([2]))
    cache.release()
    assert not cache


def test_deferred_loader_releases_each_module_cache_before_loading_next(monkeypatch, tmp_path):
    """At most one deferred module's selected tensors may remain resident."""
    with monkeypatch.context() as case_patch:
        _assert_load_selected_shard_cache_reads_only_requested_keys(case_patch)
    _assert_prequantized_module_key_plan_policy()

    class FakeLinear(nn.Module):
        def __init__(self, source_fqn):
            super().__init__()
            self._source_fqn = source_fqn
            self._source_quant_format = "block_fp8"
            self._merge_sources = None
            self._is_prequantized = True
            self._inline_loaded = False

        def load_prequantized_weights(self, weight_map, shard_cache, weights_path):
            del weight_map, weights_path
            assert shard_cache
            loaded.append(self._source_fqn)

    class FakeMoe(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = FakeLinear("layer.0.o_proj")
            self.second = FakeLinear("layer.1.o_proj")

    class RecordingCache(dict):
        def clear(self):
            return None

        def release(self):
            nonlocal active_caches
            assert active_caches == 1
            dict.clear(self)
            active_caches -= 1
            releases.append(self)

    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    f"layer.{layer}.o_proj.{suffix}": f"layer-{layer}.safetensors"
                    for layer in (0, 1)
                    for suffix in ("weight", "weight_scale_inv")
                }
            }
        )
    )
    loaded = []
    releases = []
    active_caches = 0

    def fake_cached_file(_weights_path, filename, *args, **kwargs):
        del args, kwargs
        assert filename == qlora_utils.SAFE_WEIGHTS_INDEX_NAME
        return str(index_path)

    def fake_module_keys(module, _weight_map):
        return (
            f"{module._source_fqn}.weight",
            f"{module._source_fqn}.weight_scale_inv",
        )

    def fake_selected_cache(requested_keys, weight_map, weights_path, *, group=None, src=None):
        nonlocal active_caches
        del weight_map, weights_path
        assert group is None
        assert src is None
        assert active_caches == 0, "previous module cache was not released"
        active_caches += 1
        return RecordingCache({"selected.safetensors": {key: torch.tensor([1]) for key in requested_keys}})

    monkeypatch.setattr(qlora_utils, "cached_file", fake_cached_file)
    monkeypatch.setattr(qlora_utils, "QLoRALinear", FakeLinear)
    monkeypatch.setattr(qlora_utils, "QLoRAMoeExperts", FakeMoe)
    monkeypatch.setattr(qlora_utils, "_prequantized_module_keys", fake_module_keys)
    monkeypatch.setattr(qlora_utils, "_load_selected_shard_cache", fake_selected_cache)
    monkeypatch.setattr(qlora_utils, "_deregister_qlora_weights_from_fsdp", lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    count = qlora_utils.maybe_load_prequantized_qlora(FakeModel(), "checkpoint", load_mode="all_ranks")

    assert count == 2
    assert loaded == ["layer.0.o_proj", "layer.1.o_proj"]
    assert len(releases) == 2
    assert all(not cache for cache in releases)
    assert active_caches == 0

    _assert_retained_shard_cache_ignores_consumer_clear_until_release()
