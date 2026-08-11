from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xorl.distributed.torch_parallelize import (
    _configure_manual_fsdp_prefetch,
    _expert_fsdp_kwargs_for_module,
    _expert_mixed_precision_policy,
    _resolve_fsdp_reduce_dtype,
    _sequence_parallel_fully_folded_into_fsdp,
    _topmost_modules_matching,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.exact_shared_expert_qlora import Glm52ExactTP16SharedExpertBlockFP8QLoRA
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from xorl.ops.block_fp8_native import NativeBlockFP8Linear


class _FakeBlock:
    def __init__(self, name: str) -> None:
        self.name = name
        self._fsdp_modules = [f"{name}.attn", f"{name}.gate", f"{name}.experts"]
        self.forward_prefetch = None
        self.backward_prefetch = None

    def set_modules_to_forward_prefetch(self, modules) -> None:
        self.forward_prefetch = list(modules)

    def set_modules_to_backward_prefetch(self, modules) -> None:
        self.backward_prefetch = list(modules)


def _fake_blocks():
    return [_FakeBlock("block0"), _FakeBlock("block1"), _FakeBlock("block2")]


def test_mixed_precision_selection_and_reduce_dtype_policy() -> None:
    class _Protected(nn.Module):
        pass

    root = nn.Module()
    root.composite = _Protected()
    root.composite.nested = _Protected()
    root.ordinary = nn.Module()
    root.ordinary.protected = _Protected()

    selected = _topmost_modules_matching(root, (_Protected,))

    assert selected == [root.composite, root.ordinary.protected]

    root = nn.Module()
    root.shared = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta")

    selected = _topmost_modules_matching(
        root,
        (
            NativeBlockFP8Linear,
            Glm52ExactTP1BlockFP8QLoRALinear,
            Glm52ExactTP16SharedExpertBlockFP8QLoRA,
        ),
    )

    assert selected == [root.shared]
    assert all(
        projection not in selected for projection in (root.shared.gate_proj, root.shared.up_proj, root.shared.down_proj)
    )
    kwargs = _expert_fsdp_kwargs_for_module(
        {"mesh": "mesh", "mp_policy": "bf16", "reshard_after_forward": True},
        root.shared,
    )
    assert kwargs == {"mesh": "mesh", "reshard_after_forward": True}

    _assert_expert_mixed_precision_and_reduce_dtype_policy()
    _assert_qwen3_unfuse_matches_the_tensor_parallel_plan()


def _assert_expert_mixed_precision_and_reduce_dtype_policy() -> None:
    for mesh_size, override, expected in (
        (1, None, torch.bfloat16),
        (2, None, torch.float32),
        (2, torch.bfloat16, torch.bfloat16),
    ):
        kwargs = {} if override is None else {"reduce_dtype": override}
        policy = _expert_mixed_precision_policy(ep_fsdp_mesh_size=mesh_size, **kwargs)
        assert policy.param_dtype == torch.bfloat16
        assert policy.reduce_dtype == expected

    assert _resolve_fsdp_reduce_dtype("fp32") is torch.float32
    assert _resolve_fsdp_reduce_dtype("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="Unsupported fsdp_reduce_dtype"):
        _resolve_fsdp_reduce_dtype("fp16")


def _assert_qwen3_unfuse_matches_the_tensor_parallel_plan() -> None:
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        pad_token_id=0,
    )
    model = Qwen3ForCausalLM(config)

    model.unfuse_for_tp()

    assert model._unfused_for_tp is True
    assert model.get_checkpoint_handler() is None
    assert config.base_model_tp_plan == {
        "embed_tokens": "embedding",
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    for layer in model.model.layers:
        attention = layer.self_attn
        assert not hasattr(attention, "qkv_proj")
        assert attention.q_proj.out_features == 4 * 16
        assert attention.k_proj.out_features == 2 * 16
        assert attention.v_proj.out_features == 2 * 16

        mlp = layer.mlp
        assert not hasattr(mlp, "gate_up_proj")
        assert mlp.gate_proj.out_features == 128
        assert mlp.up_proj.out_features == 128


def test_sequence_parallel_fully_folded_into_fsdp() -> None:
    for ulysses_enabled, ringattn_enabled, cp_fsdp_mode, expected in (
        (True, False, "all", True),
        (True, False, "ulysses_only", True),
        (True, False, "ring_only", False),
        (False, True, "all", True),
        (False, True, "ring_only", True),
        (False, True, "ulysses_only", False),
        (True, True, "all", True),
        (True, True, "ulysses_only", False),
        (True, True, "ring_only", False),
        (True, True, "none", False),
    ):
        state = SimpleNamespace(
            ulysses_enabled=ulysses_enabled,
            ringattn_enabled=ringattn_enabled,
            cp_fsdp_mode=cp_fsdp_mode,
        )
        assert _sequence_parallel_fully_folded_into_fsdp(state) is expected, (
            ulysses_enabled,
            ringattn_enabled,
            cp_fsdp_mode,
        )


def test_manual_fsdp_prefetch_direction_policy() -> None:
    none = [None, None, None]
    forward = [
        ["block1.experts", "block1.gate", "block1.attn"],
        ["block2.experts", "block2.gate", "block2.attn"],
        None,
    ]
    backward = [
        None,
        ["block0.experts", "block0.gate", "block0.attn"],
        ["block1.experts", "block1.gate", "block1.attn"],
    ]
    for needed, enable_forward, enable_backward, expected_forward, expected_backward in (
        (True, False, True, none, backward),
        (True, True, False, forward, none),
        (True, True, True, forward, backward),
        (False, True, True, none, none),
    ):
        blocks = _fake_blocks()
        _configure_manual_fsdp_prefetch(
            blocks,
            need_manual_prefetch=needed,
            enable_forward_prefetch=enable_forward,
            enable_backward_prefetch=enable_backward,
        )
        assert [block.forward_prefetch for block in blocks] == expected_forward
        assert [block.backward_prefetch for block in blocks] == expected_backward
