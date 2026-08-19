"""Path-qualified LoRA target selection.

Leaf-name matching discards a module's position in the tree, so `gate_proj`
selects routed experts, shared experts, and dense MLPs alike -- there is no way
to adapt one without the others. Targets containing a path separator or glob
are matched against the FULL module path instead, which makes those regions
separable on any architecture.

Bare names must keep behaving exactly as before.
"""

import pytest
import torch.nn as nn

from xorl.lora.utils import _find_target_modules


pytestmark = [pytest.mark.cpu]


def _proj_block(hidden=8, inter=16):
    block = nn.Module()
    block.gate_proj = nn.Linear(hidden, inter, bias=False)
    block.up_proj = nn.Linear(hidden, inter, bias=False)
    block.down_proj = nn.Linear(inter, hidden, bias=False)
    return block


def _moe_model(num_layers=2, num_experts=3):
    """Tree mirroring a real MoE layout: routed + shared + attention."""
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(num_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.ModuleList([_proj_block() for _ in range(num_experts)])
        layer.mlp.shared_expert = _proj_block()
        model.layers.append(layer)
    return model


def test_bare_names_are_unchanged_and_hit_every_region():
    """Regression: the pre-existing behaviour must be byte-identical."""
    model = _moe_model()
    paths = _find_target_modules(model, ["gate_proj"])
    assert any(".mlp.experts." in p for p in paths), "routed experts not covered"
    assert any(".mlp.shared_expert." in p for p in paths), "shared expert not covered"
    # 2 layers x (3 routed + 1 shared)
    assert len(paths) == 8


def test_path_pattern_selects_routed_experts_only():
    model = _moe_model()
    paths = _find_target_modules(model, ["*.mlp.experts.*.gate_proj"])
    assert len(paths) == 6  # 2 layers x 3 experts
    assert all(".mlp.experts." in p for p in paths)
    assert not any("shared_expert" in p for p in paths)


def test_path_pattern_selects_shared_expert_only():
    model = _moe_model()
    paths = _find_target_modules(model, ["*.shared_expert.*_proj"])
    assert len(paths) == 6  # 2 layers x 3 projections
    assert all("shared_expert" in p for p in paths)
    assert not any(".mlp.experts." in p for p in paths)


def test_patterns_and_bare_names_compose():
    model = _moe_model()
    paths = _find_target_modules(model, ["q_proj", "*.shared_expert.down_proj"])
    assert sum("q_proj" in p for p in paths) == 2
    assert sum("shared_expert.down_proj" in p for p in paths) == 2
    assert not any(".mlp.experts." in p for p in paths)


def test_unmatched_pattern_fails_closed():
    """A pattern that selects nothing must raise, like an unmatched bare name."""
    model = _moe_model()
    with pytest.raises(ValueError, match="matched no module"):
        _find_target_modules(model, ["*.mlp.nonexistent.*"])


def test_unmatched_bare_name_still_fails_closed():
    model = _moe_model()
    with pytest.raises(ValueError, match="matched no module"):
        _find_target_modules(model, ["v_proj"])


def test_matched_paths_never_nest():
    """No matched path may be an ancestor of another, else it is replaced twice.

    fnmatch's ``*`` spans ``.``, so "*.mlp.experts.*" matches both an expert
    container and the projections beneath it. Only LoRA-applicable modules are
    considered, and a replaced parent suppresses its children, so the result
    must still be a disjoint set.
    """
    model = _moe_model()
    paths = _find_target_modules(model, ["*.mlp.experts.*"])
    assert paths, "pattern selected nothing"
    assert all(
        not other.startswith(path + ".") for path in paths for other in paths if other != path
    ), f"nested matches would be double-replaced: {paths}"
    # 2 layers x 3 experts x 3 projections; the bare-nn.Module expert blocks are
    # not LoRA-applicable, so the applicable descendants are selected instead.
    assert len(paths) == 18
