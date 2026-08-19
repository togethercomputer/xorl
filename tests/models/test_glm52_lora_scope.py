"""Scoped GLM-5.2 LoRA target selection.

The default scope reproduces the complete deterministic inventory; the narrowed
scopes isolate shared and/or routed experts so the contribution of each can be
measured. Narrowed scopes select a different target universe than the qualified
exact active-LoRA family, so they must refuse it.
"""

import pytest
import torch

from xorl.models.transformers.glm5.qlora import GLM52_LORA_SCOPES, glm52_scope_admits


pytestmark = [pytest.mark.cpu]


# One representative role per region of the official inventory.
_ROLES = (
    "attention.q_a_proj",
    "attention.o_proj",
    "dense_mlp.gate_proj",
    "shared_expert.gate_proj",
    "shared_expert.down_proj",
    "routed_expert",
    "output.lm_head",
)


def test_scope_names_are_stable():
    assert GLM52_LORA_SCOPES == ("all", "moe", "shared_experts", "routed_experts")


def test_all_scope_admits_every_region():
    assert all(glm52_scope_admits("all", role) for role in _ROLES)


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("moe", {"shared_expert.gate_proj", "shared_expert.down_proj", "routed_expert"}),
        ("shared_experts", {"shared_expert.gate_proj", "shared_expert.down_proj"}),
        ("routed_experts", {"routed_expert"}),
    ],
)
def test_narrowed_scopes_select_only_their_region(scope, expected):
    admitted = {role for role in _ROLES if glm52_scope_admits(scope, role)}
    assert admitted == expected


@pytest.mark.parametrize("scope", ["moe", "shared_experts", "routed_experts"])
def test_narrowed_scopes_never_admit_attention_or_head(scope):
    """Attention and the lm_head are what distinguish these from the full set."""
    assert not glm52_scope_admits(scope, "attention.q_a_proj")
    assert not glm52_scope_admits(scope, "attention.kv_b_proj")
    assert not glm52_scope_admits(scope, "output.lm_head")
    assert not glm52_scope_admits(scope, "dense_mlp.gate_proj")


def test_unknown_scope_fails_closed():
    with pytest.raises(ValueError, match="Unknown GLM-5.2 LoRA scope"):
        glm52_scope_admits("experts_only", "routed_expert")


def test_server_arguments_reject_unknown_scope():
    from xorl.server.server_arguments import GLM52_LORA_SCOPE_CHOICES

    assert GLM52_LORA_SCOPE_CHOICES == GLM52_LORA_SCOPES


# ---------------------------------------------------------------------------
# Scope selects trainability, not construction
# ---------------------------------------------------------------------------


def test_scoped_factor_names_partition_the_inventory():
    """Every factor is either trainable under a scope or frozen -- never dropped.

    The complete inventory is always built: NativeBlockFP8Linear is forward-only,
    so a region left unadapted would block gradients from reaching adapted
    regions downstream of it.
    """
    from types import SimpleNamespace

    from xorl.models.transformers.glm5.qlora import glm52_scoped_factor_names

    factors = [
        SimpleNamespace(name=f"f{i}", role=role)
        for i, role in enumerate(
            ["attention.q_a_proj", "dense_mlp.gate_proj", "shared_expert.up_proj", "routed_expert", "output.lm_head"]
        )
    ]
    inventory = SimpleNamespace(factors=factors)

    assert glm52_scoped_factor_names(inventory, "all") == {"f0", "f1", "f2", "f3", "f4"}
    assert glm52_scoped_factor_names(inventory, "moe") == {"f2", "f3"}
    assert glm52_scoped_factor_names(inventory, "shared_experts") == {"f2"}
    assert glm52_scoped_factor_names(inventory, "routed_experts") == {"f3"}


def test_freezing_is_the_complement_of_the_scope():
    """apply_glm52_lora_scope must freeze exactly the out-of-scope factors."""
    from types import SimpleNamespace

    import torch.nn as nn

    from xorl.models.transformers.glm5.qlora import apply_glm52_lora_scope

    model = nn.Module()
    for name in ("attn_f", "shared_f", "routed_f"):
        setattr(model, name, nn.Parameter(torch.zeros(2)))
    inventory = SimpleNamespace(
        factors=[
            SimpleNamespace(name="attn_f", role="attention.q_a_proj"),
            SimpleNamespace(name="shared_f", role="shared_expert.up_proj"),
            SimpleNamespace(name="routed_f", role="routed_expert"),
        ],
        factor_names=frozenset({"attn_f", "shared_f", "routed_f"}),
    )

    frozen = apply_glm52_lora_scope(model, inventory, "routed_experts")
    assert frozen == 2
    assert model.attn_f.requires_grad is False
    assert model.shared_f.requires_grad is False
    assert model.routed_f.requires_grad is True


def test_scope_all_freezes_nothing():
    from types import SimpleNamespace

    import torch.nn as nn

    from xorl.models.transformers.glm5.qlora import apply_glm52_lora_scope

    model = nn.Module()
    model.routed_f = nn.Parameter(torch.zeros(2))
    inventory = SimpleNamespace(
        factors=[SimpleNamespace(name="routed_f", role="routed_expert")],
        factor_names=frozenset({"routed_f"}),
    )
    assert apply_glm52_lora_scope(model, inventory, "all") == 0
    assert model.routed_f.requires_grad is True
