import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM
from xorl.models.transformers.glm5.native_fp8 import (
    Glm52NativeBlockFP8Experts,
    Glm52OfficialFP8Inventory,
    native_fp8_dense_source_map,
)


OFFICIAL_CONFIG_SHA256 = "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff"
OFFICIAL_INDEX_SHA256 = "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_official_index_has_complete_pairs_and_deliberate_mtp_exclusion():
    model_path_value = os.environ.get("XORL_GLM52_OFFICIAL_MODEL_PATH")
    if not model_path_value:
        pytest.skip("set XORL_GLM52_OFFICIAL_MODEL_PATH to run the official-index certification inventory")
    model_path = Path(model_path_value)
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"

    assert _sha256(config_path) == OFFICIAL_CONFIG_SHA256
    assert _sha256(index_path) == OFFICIAL_INDEX_SHA256
    config = json.loads(config_path.read_text())
    weight_map = json.loads(index_path.read_text())["weight_map"]
    tensor_metadata = {}
    keys_by_shard = {}
    for key, shard in weight_map.items():
        keys_by_shard.setdefault(shard, []).append(key)
    for shard, keys in keys_by_shard.items():
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor_slice = handle.get_slice(key)
                tensor_metadata[key] = (tensor_slice.get_dtype(), tuple(tensor_slice.get_shape()))
    inventory = Glm52OfficialFP8Inventory.build(
        weight_map,
        config["quantization_config"],
        num_hidden_layers=config["num_hidden_layers"],
        num_nextn_predict_layers=config["num_nextn_predict_layers"],
        tensor_metadata=tensor_metadata,
    )
    inventory.validate_complete()

    all_keys = set(weight_map)
    partition = (
        set(inventory.quantized_weights)
        | set(inventory.quantized_scales)
        | set(inventory.config_excluded_weights)
        | set(inventory.mtp_excluded_keys)
        | set(inventory.ordinary_nonweight_keys)
    )
    assert partition == all_keys
    assert not inventory.orphan_scales
    assert not inventory.unexplained_weights
    assert not inventory.dtype_mismatches
    assert inventory.quantized_weights
    assert inventory.config_excluded_weights
    assert len(inventory.config_alias_excluded_weights) == 21
    assert all(tensor_metadata[key][0] == "BF16" for key in inventory.config_alias_excluded_weights)
    assert all(
        len(tensor_metadata[key][1]) == 2
        and (
            tensor_metadata[key][1][0] % config["quantization_config"]["weight_block_size"][0]
            or tensor_metadata[key][1][1] % config["quantization_config"]["weight_block_size"][1]
        )
        for key in inventory.config_alias_excluded_weights
    )
    assert all(
        key.endswith(".self_attn.indexer.weights_proj.weight") for key in inventory.config_alias_excluded_weights
    )

    mtp_prefix = f"model.layers.{config['num_hidden_layers']}."
    assert config["num_nextn_predict_layers"] == 1
    assert inventory.mtp_excluded_keys
    assert all(key.startswith(mtp_prefix) for key in inventory.mtp_excluded_keys)
    assert any(key.endswith(".weight_scale_inv") for key in inventory.mtp_excluded_keys)
    assert any(".mlp.experts." in key and key.endswith(".weight") for key in inventory.mtp_excluded_keys)
    assert inventory.mtp_config_alias_excluded_keys == {f"{mtp_prefix}self_attn.indexer.weights_proj.weight"}
    assert not (set(inventory.mtp_excluded_keys) & set(inventory.quantized_weights))
    assert not (set(inventory.mtp_excluded_keys) & set(inventory.quantized_scales))


def test_official_meta_model_targets_exact_trunk_quantized_inventory():
    model_path_value = os.environ.get("XORL_GLM52_OFFICIAL_MODEL_PATH")
    if not model_path_value:
        pytest.skip("set XORL_GLM52_OFFICIAL_MODEL_PATH to run the official meta-model inventory")
    model_path = Path(model_path_value)
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    assert _sha256(config_path) == OFFICIAL_CONFIG_SHA256
    assert _sha256(index_path) == OFFICIAL_INDEX_SHA256

    hf_config_dict = json.loads(config_path.read_text())
    weight_map = json.loads(index_path.read_text())["weight_map"]
    inventory = Glm52OfficialFP8Inventory.build(
        weight_map,
        hf_config_dict["quantization_config"],
        num_hidden_layers=hf_config_dict["num_hidden_layers"],
        num_nextn_predict_layers=hf_config_dict["num_nextn_predict_layers"],
    )
    inventory.validate_complete()

    config = Glm5Config.from_hf_config(SimpleNamespace(**hf_config_dict))
    with torch.device("meta"):
        model = Glm5ForCausalLM(config)

    expert_weights = {
        key
        for key in inventory.quantized_weights
        if ".mlp.experts." in key and any(f".{proj}_proj.weight" in key for proj in ("gate", "up", "down"))
    }
    dense_weights = set(inventory.quantized_weights) - expert_weights
    dense_targets = {f"{source}.weight" for source in native_fp8_dense_source_map(model)}
    expert_targets = {fqn for fqn, module in model.named_modules() if isinstance(module, Glm52NativeBlockFP8Experts)}

    assert len(expert_weights) == 75 * 256 * 3
    assert dense_targets == dense_weights
    assert expert_targets == {f"model.layers.{layer}.mlp.experts" for layer in range(3, 78)}
    assert len(expert_targets) == 75
    assert all(not fqn.startswith("model.layers.78.") for fqn in dense_targets | expert_targets)
