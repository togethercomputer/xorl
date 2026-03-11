"""Tests for tensor parallelism support.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/distributed/test_tensor_parallel.py -v
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_tensor_parallel.py -v
"""

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


class TestUnfuseForTP:
    """Test that unfuse_for_tp correctly replaces fused projections."""

    def test_attention_and_mlp_unfuse(self):
        """Attention unfuse creates separate q/k/v; MLP unfuse creates separate gate/up."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

        config = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
        )

        # Attention unfuse
        attn = Qwen3Attention(config, layer_idx=0)
        assert hasattr(attn, "qkv_proj")
        assert not hasattr(attn, "q_proj")
        attn.unfuse_for_tp()
        assert not hasattr(attn, "qkv_proj")
        assert hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj")
        assert attn.q_proj.out_features == 4 * 64
        assert attn.k_proj.out_features == 2 * 64
        assert attn.v_proj.out_features == 2 * 64

        # MLP unfuse
        mlp = Qwen3MLP(config)
        assert hasattr(mlp, "gate_up_proj")
        assert not hasattr(mlp, "gate_proj")
        mlp.unfuse_for_tp()
        assert not hasattr(mlp, "gate_up_proj")
        assert hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")
        assert mlp.gate_proj.out_features == 512
        assert mlp.up_proj.out_features == 512

    def test_unfused_forward_shape_and_model_level(self):
        """Unfused MLP forward produces same shape; model-level unfuse covers all layers."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3ForCausalLM

        config = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000,
            pad_token_id=0,
        )

        # Unfused forward shape matches fused
        mlp_fused = Qwen3MLP(config).cuda()
        x = torch.randn(1, 16, 256, device="cuda")
        out_fused = mlp_fused(x)

        mlp_unfused = Qwen3MLP(config).cuda()
        mlp_unfused.unfuse_for_tp()
        out_unfused = mlp_unfused(x)
        assert out_fused.shape == out_unfused.shape == torch.Size([1, 16, 256])

        # Model-level unfuse
        model = Qwen3ForCausalLM(config)
        model.unfuse_for_tp()
        for layer in model.model.layers:
            assert not hasattr(layer.self_attn, "qkv_proj")
            assert hasattr(layer.self_attn, "q_proj")
            assert hasattr(layer.self_attn, "k_proj")
            assert hasattr(layer.self_attn, "v_proj")
            assert not hasattr(layer.mlp, "gate_up_proj")
            assert hasattr(layer.mlp, "gate_proj")
            assert hasattr(layer.mlp, "up_proj")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
