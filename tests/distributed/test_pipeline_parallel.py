"""
Tests for pipeline parallelism.

Unit tests (no GPU needed):
    pytest tests/distributed/test_pipeline_parallel.py -v -k "TestFQNGeneration"

Distributed tests (2 GPUs):
    torchrun --nproc_per_node=2 -m pytest tests/distributed/test_pipeline_parallel.py -v -k "TestPP2GPU"

Distributed tests (4 GPUs):
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_pipeline_parallel.py -v -k "TestPP4GPU"

Distributed tests (4 GPUs, PP + Ulysses):
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_pipeline_parallel.py -v -k "TestPP2Ulysses2"

Distributed tests (8 GPUs, PP + FSDP + Ulysses):
    torchrun --nproc_per_node=8 -m pytest tests/distributed/test_pipeline_parallel.py -v -k "TestPP2FSDP2Ulysses2"
"""

import math
import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.distributed.pipeline_parallel import generate_llm_fqn_per_model_part

pytestmark = [pytest.mark.distributed]


def _build_tiny_qwen3(init_device="meta"):
    """Build a tiny Qwen3ForCausalLM for testing (4 layers, small dims)."""
    from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
    from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        _attn_implementation="flash_attention_2",
    )
    with torch.device(init_device):
        model = Qwen3ForCausalLM(config)
    return model


# =============================================================================
# Unit Tests (no GPU required)
# =============================================================================

class TestFQNGeneration:
    """Test generate_llm_fqn_per_model_part FQN distribution logic."""

    def test_basic_2_stages_4_layers(self):
        """4 layers, 2 stages with default (torchtitan) FQN names."""
        result = generate_llm_fqn_per_model_part(2, 4)
        assert len(result) == 2
        # First stage should have tok_embeddings + some layers
        assert result[0][0] == "tok_embeddings"
        # Last stage should end with norm, output
        assert result[-1][-2:] == ["norm", "output"]
        # All layers should be present
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert len(all_layers) == 4

    def test_basic_2_stages_8_layers(self):
        result = generate_llm_fqn_per_model_part(2, 8)
        assert len(result) == 2
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert len(all_layers) == 8

    def test_4_stages_36_layers(self):
        """36 layers (Qwen3 8B), 4 stages."""
        result = generate_llm_fqn_per_model_part(4, 36)
        assert len(result) == 4
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert len(all_layers) == 36

    def test_layers_are_contiguous(self):
        """Verify layers assigned contiguously."""
        result = generate_llm_fqn_per_model_part(4, 12)
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        expected = [f"layers.{i}" for i in range(12)]
        assert all_layers == expected

    def test_minimal(self):
        """2 layers, 2 stages."""
        result = generate_llm_fqn_per_model_part(2, 2)
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert len(all_layers) == 2

    def test_qwen3_fqn_names(self):
        """Test with Qwen3-style nested FQN names."""
        result = generate_llm_fqn_per_model_part(
            2, 4,
            input_fqns=["model.embed_tokens"],
            layer_prefix="model.layers",
            output_fqns=["model.norm", "lm_head"],
        )
        assert len(result) == 2
        assert result[0][0] == "model.embed_tokens"
        assert result[-1][-2:] == ["model.norm", "lm_head"]
        all_layers = [m for stage in result for m in stage if m.startswith("model.layers.")]
        assert len(all_layers) == 4

    def test_single_stage(self):
        """Single stage should contain all modules."""
        result = generate_llm_fqn_per_model_part(1, 4)
        assert len(result) == 1
        assert result[0][0] == "tok_embeddings"
        assert result[0][-1] == "output"
        # tok_embeddings, layers.0-3, norm, output = 7 items
        assert len(result[0]) == 7

    def test_error_too_many_stages(self):
        """Error when more stages than effective layers."""
        with pytest.raises(ValueError):
            generate_llm_fqn_per_model_part(10, 2)


# =============================================================================
# Distributed helpers
# =============================================================================

def is_distributed_available():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def requires_nproc(n):
    def decorator(func):
        if not is_distributed_available():
            return pytest.mark.skip("Requires distributed environment")(func)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != n:
            return pytest.mark.skip(f"Requires {n} GPUs, got {world_size}")(func)
        return func
    return decorator


# =============================================================================
# Distributed Tests (2 GPUs)
# =============================================================================

class TestPP2GPU:
    """PP tests with 2 GPUs."""

    def setup_method(self):
        if is_distributed_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(dist.get_rank())
            from xorl.distributed.parallel_state import init_parallel_state
            init_parallel_state(dp_size=1, pp_size=2)

    def teardown_method(self):
        if dist.is_initialized():
            dist.barrier()

    @requires_nproc(2)
    def test_split_model_structure(self):
        """Verify model splitting creates correct modules per stage."""
        from xorl.distributed.pipeline_parallel import (
            generate_llm_fqn_per_model_part,
            pipeline_module_split,
        )
        from xorl.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        model = _build_tiny_qwen3(init_device="meta")
        pp_config = model.get_pp_module_config()

        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_stages=ps.pp_size,
            num_layers=pp_config["num_layers"],
            input_fqns=pp_config["input_fqns"],
            layer_prefix=pp_config["layer_prefix"],
            output_fqns=pp_config["output_fqns"],
        )

        stages, model_parts = pipeline_module_split(
            model,
            pp_mesh=ps.pp_mesh,
            pp_schedule="1F1B",
            device=device,
            module_names_per_stage=module_names_per_stage,
            always_keep_fqns=pp_config.get("always_keep_fqns"),
        )

        assert len(stages) == 1
        assert len(model_parts) == 1
        model_part = model_parts[0]

        # model_part is the Qwen3ForCausalLM directly (no wrapper)
        base = model_part.model

        if rank == 0:
            assert base.embed_tokens is not None
            assert base.rotary_emb is not None
            assert base.norm is None
            assert model_part.lm_head is None
        else:
            assert base.embed_tokens is None
            assert base.rotary_emb is not None  # rotary_emb kept on all stages
            assert base.norm is not None
            assert model_part.lm_head is not None

        assert len(base.layers) > 0

    @requires_nproc(2)
    def test_pp_training_loss_decreases(self):
        """Verify loss decreases over steps with PP=2."""
        from xorl.distributed.pipeline_parallel import (
            generate_llm_fqn_per_model_part,
            pipeline_module_split,
            build_pipeline_schedule,
        )
        from xorl.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        model = _build_tiny_qwen3(init_device="meta")
        pp_config = model.get_pp_module_config()

        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_stages=ps.pp_size,
            num_layers=pp_config["num_layers"],
            input_fqns=pp_config["input_fqns"],
            layer_prefix=pp_config["layer_prefix"],
            output_fqns=pp_config["output_fqns"],
        )

        stages, model_parts = pipeline_module_split(
            model,
            pp_mesh=ps.pp_mesh,
            pp_schedule="1F1B",
            device=device,
            module_names_per_stage=module_names_per_stage,
            always_keep_fqns=pp_config.get("always_keep_fqns"),
        )

        model_part = model_parts[0]

        # Initialize weights on device
        model_part.to_empty(device=device)
        model_part.apply(model_part._init_weights)
        model_part.to(torch.bfloat16)
        model_part.train()

        def loss_fn(pred, labels):
            return F.cross_entropy(
                pred.flatten(0, 1).float(),
                labels.flatten(0, 1),
                reduction="sum",
            )

        n_microbatches = 4
        schedule = build_pipeline_schedule(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            schedule_name="1F1B",
        )
        optimizer = torch.optim.AdamW(model_part.parameters(), lr=1e-3)

        torch.manual_seed(42)
        vocab_size = 1000
        input_ids = torch.randint(0, vocab_size, (n_microbatches, 32), device=device)
        labels = torch.randint(0, vocab_size, (n_microbatches, 32), device=device)

        has_first_stage = any(s.is_first for s in stages)
        has_last_stage = any(s.is_last for s in stages)

        recorded_losses = []
        for step in range(5):
            optimizer.zero_grad()

            targets = labels if has_last_stage else None
            step_losses = [] if has_last_stage else None

            if has_first_stage:
                schedule.step(input_ids, target=targets, losses=step_losses)
            else:
                schedule.step(target=targets, losses=step_losses)

            if has_last_stage:
                loss_val = torch.sum(torch.stack(step_losses)).item()
                recorded_losses.append(loss_val)

            optimizer.step()

        if has_last_stage:
            assert recorded_losses[-1] < recorded_losses[0], (
                f"Loss did not decrease: first={recorded_losses[0]:.4f}, last={recorded_losses[-1]:.4f}"
            )


# =============================================================================
# Distributed Tests (4 GPUs): PP=2 + FSDP=2
# NOTE: Keep training steps short (<=5) to avoid FSDP+PP NCCL deadlock in PyTorch 2.9.x
# =============================================================================

class TestPP4GPU:
    """PP=2 + FSDP=2 tests with 4 GPUs."""

    def setup_method(self):
        if is_distributed_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(dist.get_rank())
            from xorl.distributed.parallel_state import init_parallel_state
            init_parallel_state(dp_size=2, pp_size=2)

    def teardown_method(self):
        if dist.is_initialized():
            dist.barrier()

    @requires_nproc(4)
    def test_pp2_fsdp2_training(self):
        """Verify PP=2 + FSDP=2 combined training."""
        from xorl.distributed.pipeline_parallel import (
            generate_llm_fqn_per_model_part,
            pipeline_module_split,
            build_pipeline_schedule,
        )
        from torch.distributed._composable.fsdp import fully_shard
        from xorl.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        pp_rank = ps.pp_rank
        dp_mesh = ps.fsdp_mesh

        model = _build_tiny_qwen3(init_device="meta")
        pp_config = model.get_pp_module_config()

        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_stages=ps.pp_size,
            num_layers=pp_config["num_layers"],
            input_fqns=pp_config["input_fqns"],
            layer_prefix=pp_config["layer_prefix"],
            output_fqns=pp_config["output_fqns"],
        )

        stages, model_parts = pipeline_module_split(
            model,
            pp_mesh=ps.pp_mesh,
            pp_schedule="1F1B",
            device=device,
            module_names_per_stage=module_names_per_stage,
            always_keep_fqns=pp_config.get("always_keep_fqns"),
        )

        model_part = model_parts[0]

        model_part.to_empty(device=device)
        model_part.apply(model_part._init_weights)
        model_part.to(torch.bfloat16)
        model_part.train()

        # Apply FSDP with reshard_after_forward=False for PP
        base = model_part.model
        for layer in base.layers:
            fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
        fully_shard(model_part, mesh=dp_mesh)

        # Update stage's submod to the FSDP-wrapped model
        stages[0].submod = model_part

        def loss_fn(pred, labels):
            return F.cross_entropy(
                pred.flatten(0, 1).float(),
                labels.flatten(0, 1),
                reduction="sum",
            )

        n_microbatches = 4
        schedule = build_pipeline_schedule(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            schedule_name="1F1B",
        )
        optimizer = torch.optim.AdamW(model_part.parameters(), lr=1e-3)

        torch.manual_seed(42 + rank)
        vocab_size = 1000
        input_ids = torch.randint(0, vocab_size, (n_microbatches, 32), device=device)
        labels = torch.randint(0, vocab_size, (n_microbatches, 32), device=device)

        has_first_stage = any(s.is_first for s in stages)
        has_last_stage = any(s.is_last for s in stages)

        recorded_losses = []
        for step in range(5):
            optimizer.zero_grad()

            targets = labels if has_last_stage else None
            step_losses = [] if has_last_stage else None

            if has_first_stage:
                schedule.step(input_ids, target=targets, losses=step_losses)
            else:
                schedule.step(target=targets, losses=step_losses)

            if has_last_stage:
                loss_val = torch.sum(torch.stack(step_losses)).item()
                recorded_losses.append(loss_val)

            optimizer.step()

        if has_last_stage:
            assert recorded_losses[-1] < recorded_losses[0], (
                f"Loss did not decrease: first={recorded_losses[0]:.4f}, last={recorded_losses[-1]:.4f}"
            )


# =============================================================================
# End-to-end PP + Ulysses tests (4 GPUs)
#
# These tests mirror the real training pipeline in cli/train.py:
# - build_parallelize_model handles PP split → gradient ckpt → FSDP2
# - FSDP2 MixedPrecisionPolicy handles bf16 casting
# - PP schedule with normalized loss (sum / global_valid_tokens)
# - Gradient clipping via clip_grad_norm_
# - Loss broadcast across PP stages
# =============================================================================

def _run_pp_ulysses_e2e(n_steps=5, n_microbatches=4, full_seq_len=64):
    """
    End-to-end PP + Ulysses training loop mirroring cli/train.py.

    Uses build_parallelize_model (the real code path) which handles:
    PP split → gradient checkpointing → FSDP2 (mixed precision + weight init).
    """
    from xorl.distributed.parallel_state import get_parallel_state
    from xorl.distributed.torch_parallelize import build_parallelize_model
    from xorl.distributed.pipeline_parallel import build_pipeline_schedule

    ps = get_parallel_state()
    rank = dist.get_rank()

    # 1. Build model on meta device (same as train.py)
    model = _build_tiny_qwen3(init_device="meta")

    # 2. build_parallelize_model: PP split → grad ckpt → FSDP2
    build_result = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=None,  # random init
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        basic_modules=model._no_split_modules,
        pp_schedule="1F1B",
    )

    assert isinstance(build_result, dict), "PP should return dict"
    pp_stages = build_result["stages"]
    model_parts = build_result["model_parts"]
    has_first_stage = build_result["has_first_stage"]
    has_last_stage = build_result["has_last_stage"]
    model_part = model_parts[0]

    # 3. PP loss function with normalization (same as train.py)
    pp_context = {}

    def pp_loss_fn(pred, labels):
        return F.cross_entropy(
            pred.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum",
        ) / pp_context["global_valid_tokens"]

    # 4. Build schedule and optimizer
    schedule = build_pipeline_schedule(
        stages=pp_stages,
        n_microbatches=n_microbatches,
        loss_fn=pp_loss_fn,
        schedule_name="1F1B",
    )
    optimizer = torch.optim.AdamW(model_part.parameters(), lr=1e-3)
    model_part.train()

    # 5. Generate data
    torch.manual_seed(42 + rank)
    vocab_size = 1000
    local_seq_len = full_seq_len // ps.ulysses_size
    input_ids = torch.randint(0, vocab_size, (n_microbatches, local_seq_len), device="cuda")
    labels = torch.randint(0, vocab_size, (n_microbatches, local_seq_len), device="cuda")

    # 6. Training loop (mirrors train.py PP path)
    recorded_losses = []
    for step in range(n_steps):
        # global_valid_tokens: all tokens are valid (no IGNORE_INDEX)
        global_valid_tokens = torch.tensor(
            labels.numel(), device="cuda", dtype=torch.float32
        )
        # all-reduce across DP group (same as train.py)
        dist.all_reduce(
            global_valid_tokens, op=dist.ReduceOp.SUM,
            group=ps.fsdp_group if ps.pp_enabled else None,
        )
        pp_context["global_valid_tokens"] = global_valid_tokens

        optimizer.zero_grad()

        targets = labels if has_last_stage else None
        losses = [] if has_last_stage else None

        if has_first_stage:
            schedule.step(input_ids, target=targets, losses=losses)
        else:
            schedule.step(target=targets, losses=losses)

        # Collect loss (same as train.py)
        if has_last_stage:
            total_loss = torch.sum(torch.stack(losses)).item()
            loss_tensor = torch.tensor([total_loss], device="cuda")
        else:
            loss_tensor = torch.tensor([-1.0], device="cuda")
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.MAX, group=ps.pp_group)
        total_loss = loss_tensor.item()

        # Gradient clipping (same as train.py)
        if hasattr(model_part, "clip_grad_norm_"):
            grad_norm = model_part.clip_grad_norm_(1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model_part.parameters(), 1.0)

        optimizer.step()
        recorded_losses.append(total_loss)

    return recorded_losses, model_part


class TestPP2Ulysses2:
    """PP=2 + Ulysses=2 end-to-end tests with 4 GPUs.

    Uses build_parallelize_model (the real training code path) with:
    - FSDP2 mixed precision (bf16 params, f32 reduce)
    - Gradient checkpointing
    - PP loss normalization by global_valid_tokens
    - Gradient clipping
    """

    def setup_method(self):
        if is_distributed_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(dist.get_rank())
            from xorl.distributed.parallel_state import init_parallel_state
            init_parallel_state(dp_size=1, pp_size=2, ulysses_size=2)

    def teardown_method(self):
        if dist.is_initialized():
            dist.barrier()

    @requires_nproc(4)
    def test_loss_decreases(self):
        """Loss should decrease over training steps."""
        losses, _ = _run_pp_ulysses_e2e(n_steps=5)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    @requires_nproc(4)
    def test_no_nan_or_inf(self):
        """Loss and gradients should be finite throughout training."""
        losses, model_part = _run_pp_ulysses_e2e(n_steps=3)
        for i, loss in enumerate(losses):
            assert not (math.isnan(loss) or math.isinf(loss)), (
                f"Step {i}: loss is {loss}"
            )

    @requires_nproc(4)
    def test_all_params_get_gradients(self):
        """Every trainable parameter should receive a gradient."""
        _, model_part = _run_pp_ulysses_e2e(n_steps=1)
        for name, param in model_part.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
