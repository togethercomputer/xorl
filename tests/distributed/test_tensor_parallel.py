"""Tests for tensor parallelism support.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/distributed/test_tensor_parallel.py -v
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_tensor_parallel.py -v
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from xorl.distributed.torch_parallelize import _build_tp_plan
import xorl.distributed.parallel_state as ps
from xorl.distributed.parallel_state import init_parallel_state


def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


# ======================== Model unfuse tests ======================== #


class TestUnfuseForTP:
    """Test that unfuse_for_tp correctly replaces fused projections."""

    def test_attention_unfuse(self):
        """Test Qwen3Attention unfuse creates separate q/k/v projections."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3Attention

        config = Qwen3Config(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
        )
        attn = Qwen3Attention(config, layer_idx=0)

        # Before unfuse: has qkv_proj
        assert hasattr(attn, "qkv_proj")
        assert not hasattr(attn, "q_proj")

        attn.unfuse_for_tp()

        # After unfuse: has separate projections
        assert not hasattr(attn, "qkv_proj")
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert attn.q_proj.out_features == 4 * 64  # num_heads * head_dim
        assert attn.k_proj.out_features == 2 * 64  # num_kv_heads * head_dim
        assert attn.v_proj.out_features == 2 * 64

    def test_mlp_unfuse(self):
        """Test Qwen3MLP unfuse creates separate gate/up projections."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3MLP

        config = Qwen3Config(hidden_size=256, intermediate_size=512)
        mlp = Qwen3MLP(config)

        assert hasattr(mlp, "gate_up_proj")
        assert not hasattr(mlp, "gate_proj")

        mlp.unfuse_for_tp()

        assert not hasattr(mlp, "gate_up_proj")
        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert mlp.gate_proj.out_features == 512
        assert mlp.up_proj.out_features == 512

    def test_unfused_forward_matches_fused(self):
        """Test that unfused forward produces same output shape as fused."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3MLP

        config = Qwen3Config(hidden_size=256, intermediate_size=512)

        # Fused path
        mlp_fused = Qwen3MLP(config).cuda()
        x = torch.randn(1, 16, 256, device="cuda")
        out_fused = mlp_fused(x)

        # Unfused path
        mlp_unfused = Qwen3MLP(config).cuda()
        mlp_unfused.unfuse_for_tp()
        out_unfused = mlp_unfused(x)

        # Same shape (values differ due to random init)
        assert out_fused.shape == out_unfused.shape == torch.Size([1, 16, 256])

    def test_model_level_unfuse(self):
        """Test Qwen3ForCausalLM.unfuse_for_tp unfuses all layers."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

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


# ======================== Distributed TP tests ======================== #


class TestTPForward:
    """Test tensor parallel forward pass correctness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rank, self.world_size = setup_dist()

    def test_tp_forward_matches_reference(self):
        """Test that TP forward matches single-GPU reference."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000,
            pad_token_id=0,
            _attn_implementation="eager",
        )

        # Create reference model on all ranks with same weights
        torch.manual_seed(42)
        ref_model = Qwen3ForCausalLM(config).cuda().bfloat16()

        # Create TP model with same weights
        torch.manual_seed(42)
        tp_model = Qwen3ForCausalLM(config).cuda().bfloat16()
        tp_model.unfuse_for_tp()

        # Copy weights from fused ref to unfused tp model
        for layer_idx in range(config.num_hidden_layers):
            ref_attn = ref_model.model.layers[layer_idx].self_attn
            tp_attn = tp_model.model.layers[layer_idx].self_attn
            qkv_w = ref_attn.qkv_proj.weight.data
            q_dim = config.num_attention_heads * config.head_dim
            kv_dim = config.num_key_value_heads * config.head_dim
            tp_attn.q_proj.weight.data.copy_(qkv_w[:q_dim])
            tp_attn.k_proj.weight.data.copy_(qkv_w[q_dim:q_dim + kv_dim])
            tp_attn.v_proj.weight.data.copy_(qkv_w[q_dim + kv_dim:])

            ref_mlp = ref_model.model.layers[layer_idx].mlp
            tp_mlp = tp_model.model.layers[layer_idx].mlp
            gate_up_w = ref_mlp.gate_up_proj.weight.data
            tp_mlp.gate_proj.weight.data.copy_(gate_up_w[:config.intermediate_size])
            tp_mlp.up_proj.weight.data.copy_(gate_up_w[config.intermediate_size:])

        # Init parallel state for reference model (no TP)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=self.world_size, tp_size=1)

        # Same input on all ranks
        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        # Reference forward (no TP)
        with torch.no_grad():
            ref_out = ref_model(input_ids=input_ids, labels=labels)
            ref_loss = ref_out.loss

        # Apply TP
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)
        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(tp_model)
        tp_model = parallelize_module(tp_model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        # TP forward
        with torch.no_grad():
            tp_out = tp_model(input_ids=input_ids, labels=labels)
            tp_loss = tp_out.loss

        # Compare losses
        if ref_loss is not None and tp_loss is not None:
            torch.testing.assert_close(tp_loss, ref_loss, rtol=1e-2, atol=1e-2)

    def test_tp_backward(self):
        """Test that TP backward runs without error and produces gradients."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000,
            pad_token_id=0,
            _attn_implementation="eager",
        )

        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        assert loss is not None
        loss.backward()

        # Check gradients exist on some parameters
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed"


class TestTPWithFSDP:
    """Test TP + FSDP 2D parallelism."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rank, self.world_size = setup_dist()

    @pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 4,
        reason="Requires at least 4 GPUs for 2D parallelism test"
    )
    def test_tp2_fsdp2_forward_backward(self):
        """Test TP=2 FSDP=2 forward and backward."""
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        tp_size = 2
        dp_size = self.world_size // tp_size

        config = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000,
            pad_token_id=0,
            _attn_implementation="eager",
        )

        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=dp_size, tp_size=tp_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        # 2D mesh: [dp, tp]
        mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

        # Apply TP first
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=mesh_2d["tp"], parallelize_plan=tp_plan)

        # Apply FSDP second
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh_2d["dp"])
        fully_shard(model, mesh=mesh_2d["dp"])

        # Same input across TP ranks, different across DP ranks
        dp_rank = mesh_2d.get_local_rank("dp")
        torch.manual_seed(100 + dp_rank)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        assert loss is not None
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad


class TestTPEmbeddingAndLmHead:
    """Test embedding and lm_head parallelism with TP."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rank, self.world_size = setup_dist()

    def _make_config(self, tie_word_embeddings=False):
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config

        return Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000,
            pad_token_id=0,
            tie_word_embeddings=tie_word_embeddings,
            _attn_implementation="eager",
        )

    def test_embedding_is_vocab_sharded(self):
        """Test that embed_tokens weight is sharded on vocab dim after TP."""
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = self._make_config(tie_word_embeddings=False)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        # embed_tokens weight should be vocab-sharded
        emb_weight = model.model.embed_tokens.weight
        assert hasattr(emb_weight, "placements"), "embed_tokens weight should be a DTensor"
        local_shape = emb_weight.to_local().shape
        assert local_shape[0] == config.vocab_size // self.world_size, (
            f"Expected vocab shard {config.vocab_size // self.world_size}, got {local_shape[0]}"
        )
        assert local_shape[1] == config.hidden_size  # hidden dim not sharded

    def test_lm_head_is_vocab_sharded(self):
        """Test that lm_head weight is sharded on vocab dim after TP."""
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = self._make_config(tie_word_embeddings=False)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        # lm_head weight should be vocab-sharded (ColwiseParallel shards dim 0)
        lm_weight = model.lm_head.weight
        assert hasattr(lm_weight, "placements"), "lm_head weight should be a DTensor"
        local_shape = lm_weight.to_local().shape
        assert local_shape[0] == config.vocab_size // self.world_size

    def test_untied_weights_are_independent(self):
        """Test that with tie_word_embeddings=False, embed and lm_head are independent DTensors."""
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = self._make_config(tie_word_embeddings=False)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        # Different data pointers = independent weights
        emb_data = model.model.embed_tokens.weight.to_local().data_ptr()
        lm_data = model.lm_head.weight.to_local().data_ptr()
        assert emb_data != lm_data, "Untied weights should have different data pointers"

    def test_untied_forward_backward(self):
        """Test forward+backward with untied embed_tokens and lm_head under TP."""
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = self._make_config(tie_word_embeddings=False)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        out.loss.backward()

        # Both embed_tokens and lm_head should have gradients
        emb_grad = model.model.embed_tokens.weight.grad
        lm_grad = model.lm_head.weight.grad
        assert emb_grad is not None, "embed_tokens should have gradient"
        assert lm_grad is not None, "lm_head should have gradient"

    def test_untied_tp_matches_reference(self):
        """Test that TP with untied weights matches single-GPU reference."""
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = self._make_config(tie_word_embeddings=False)

        # Reference model (no TP)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=self.world_size, tp_size=1)

        torch.manual_seed(42)
        ref_model = Qwen3ForCausalLM(config).cuda().bfloat16()

        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        with torch.no_grad():
            ref_loss = ref_model(input_ids=input_ids, labels=labels).loss

        # TP model with same weights
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        tp_model = Qwen3ForCausalLM(config).cuda().bfloat16()
        tp_model.unfuse_for_tp()

        # Copy weights from fused ref to unfused tp model
        # Copy embed_tokens and lm_head
        tp_model.model.embed_tokens.weight.data.copy_(ref_model.model.embed_tokens.weight.data)
        tp_model.lm_head.weight.data.copy_(ref_model.lm_head.weight.data)

        for layer_idx in range(config.num_hidden_layers):
            ref_attn = ref_model.model.layers[layer_idx].self_attn
            tp_attn = tp_model.model.layers[layer_idx].self_attn
            qkv_w = ref_attn.qkv_proj.weight.data
            q_dim = config.num_attention_heads * config.head_dim
            kv_dim = config.num_key_value_heads * config.head_dim
            tp_attn.q_proj.weight.data.copy_(qkv_w[:q_dim])
            tp_attn.k_proj.weight.data.copy_(qkv_w[q_dim:q_dim + kv_dim])
            tp_attn.v_proj.weight.data.copy_(qkv_w[q_dim + kv_dim:])

            ref_mlp = ref_model.model.layers[layer_idx].mlp
            tp_mlp = tp_model.model.layers[layer_idx].mlp
            gate_up_w = ref_mlp.gate_up_proj.weight.data
            tp_mlp.gate_proj.weight.data.copy_(gate_up_w[:config.intermediate_size])
            tp_mlp.up_proj.weight.data.copy_(gate_up_w[config.intermediate_size:])

            # Copy remaining weights
            tp_model.model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(
                ref_model.model.layers[layer_idx].self_attn.o_proj.weight.data)
            tp_model.model.layers[layer_idx].mlp.down_proj.weight.data.copy_(
                ref_model.model.layers[layer_idx].mlp.down_proj.weight.data)

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(tp_model)
        tp_model = parallelize_module(tp_model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        with torch.no_grad():
            tp_loss = tp_model(input_ids=input_ids, labels=labels).loss

        torch.testing.assert_close(tp_loss, ref_loss, rtol=1e-2, atol=1e-2)

    def test_tied_weights_forward_backward(self):
        """Test forward+backward with tied embed_tokens and lm_head under TP."""
        from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = self._make_config(tie_word_embeddings=True)
        ps._PARALLEL_STATE = None
        init_parallel_state(dp_size=1, tp_size=self.world_size)

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).cuda().bfloat16()
        model.unfuse_for_tp()

        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        tp_plan = _build_tp_plan(model)
        model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        out.loss.backward()

        # With tied weights, embed and lm_head share the same weight
        emb_grad = model.model.embed_tokens.weight.grad
        assert emb_grad is not None, "Tied weight should have gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
