import hashlib
import json

import torch

from xorl.server.weight_sync.glm52_router_bundle import (
    GLM52_ROUTER_BUNDLE_SCHEMA,
    GLM52_ROUTER_MANIFEST,
    GLM52_ROUTER_TENSORS,
    _merge_glm52_router_states,
    gather_glm52_router_weights,
    mark_adapter_config_with_glm52_router_bundle,
    save_glm52_router_bundle,
)


class Glm5TopkRouter(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((3, 4), value, dtype=torch.float32))


class _MLP(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.gate = Glm5TopkRouter(value)


class _Layer(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.mlp = _MLP(value)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(1.0), _Layer(2.0)])


def test_router_bundle_is_complete_checksummed_and_bound_to_adapter(tmp_path):
    state = gather_glm52_router_weights(_Model())
    assert list(state) == ["layer.0.weight", "layer.1.weight"]
    assert all(tensor.dtype is torch.bfloat16 for tensor in state.values())

    manifest = save_glm52_router_bundle(tmp_path, state, weight_step=7, expected_layer_ids=[0, 1])
    (tmp_path / "adapter_config.json").write_text("{}\n")
    mark_adapter_config_with_glm52_router_bundle(tmp_path, manifest)

    tensor_bytes = (tmp_path / GLM52_ROUTER_TENSORS).read_bytes()
    persisted = json.loads((tmp_path / GLM52_ROUTER_MANIFEST).read_text())
    marker = json.loads((tmp_path / "adapter_config.json").read_text())["_xorl_glm52_router_bundle"]
    assert persisted == manifest
    assert persisted["schema"] == GLM52_ROUTER_BUNDLE_SCHEMA
    assert persisted["sha256"] == hashlib.sha256(tensor_bytes).hexdigest()
    assert marker == {key: manifest[key] for key in marker}
    assert marker["layer_ids"] == [0, 1]
    assert marker["weight_step"] == 7
    assert not list(tmp_path.glob("xorl_glm52_router*.safetensors"))
    assert (tmp_path / GLM52_ROUTER_TENSORS).parent.name == "xorl_router"


def test_router_bundle_rejects_incomplete_inventory(tmp_path):
    state = {"layer.0.weight": torch.zeros((3, 4), dtype=torch.bfloat16)}
    try:
        save_glm52_router_bundle(tmp_path, state, weight_step=1, expected_layer_ids=[0, 1])
    except RuntimeError as error:
        assert "Incomplete GLM-5.2 router sidecar" in str(error)
    else:
        raise AssertionError("incomplete router publication was accepted")


def test_pipeline_router_states_merge_disjoint_stages_and_identical_replicas():
    layer_0 = torch.ones((3, 4), dtype=torch.bfloat16)
    layer_1 = torch.full((3, 4), 2, dtype=torch.bfloat16)

    merged = _merge_glm52_router_states(
        [
            {"layer.0.weight": layer_0},
            {"layer.1.weight": layer_1},
            {"layer.1.weight": layer_1.clone()},
        ]
    )

    assert list(merged) == ["layer.0.weight", "layer.1.weight"]
