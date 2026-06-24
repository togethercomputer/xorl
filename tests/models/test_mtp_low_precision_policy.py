from types import SimpleNamespace

import pytest
import torch

from xorl.models.module_utils import _matches_checkpoint_skip_key_pattern
from xorl.models.transformers.glm4_moe.checkpoint_handler import Glm4MoeCheckpointHandler
from xorl.models.transformers.qwen3_5_shared import QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS


pytestmark = [pytest.mark.cpu]


def test_qwen35_checkpoint_policy_skips_top_level_mtp_keys():
    model = SimpleNamespace(_checkpoint_skip_key_patterns=QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS)

    assert _matches_checkpoint_skip_key_pattern("mtp.layers.0.mlp.gate_proj.weight", model)
    assert _matches_checkpoint_skip_key_pattern("mtp.pre_fc_norm_embedding.weight", model)
    assert not _matches_checkpoint_skip_key_pattern("model.layers.0.mlp.gate_proj.weight", model)


def test_glm4_moe_checkpoint_policy_remaps_only_shared_mtp_tail():
    tensor = torch.randn(2, 3)
    handler = Glm4MoeCheckpointHandler(
        num_experts=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=3,
    )

    remapped = handler.on_load_weight("model.layers.3.embed_tokens.weight", tensor)
    assert len(remapped) == 1
    assert remapped[0][0] == "model.embed_tokens.weight"
    torch.testing.assert_close(remapped[0][1], tensor)

    remapped = handler.on_load_weight("model.layers.3.shared_head.norm.weight", tensor)
    assert len(remapped) == 1
    assert remapped[0][0] == "model.norm.weight"
    torch.testing.assert_close(remapped[0][1], tensor)

    remapped = handler.on_load_weight("model.layers.3.shared_head.head.weight", tensor)
    assert len(remapped) == 1
    assert remapped[0][0] == "lm_head.weight"
    torch.testing.assert_close(remapped[0][1], tensor)

    assert handler.on_load_weight("model.layers.3.eh_proj.weight", tensor) == []
    assert handler.on_load_weight("model.layers.3.enorm.weight", tensor) == []
    assert handler.on_load_weight("model.layers.3.transformer_layer.self_attn.q_proj.weight", tensor) == []
