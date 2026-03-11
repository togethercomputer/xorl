"""
Tests for pipeline parallelism -- unit tests only.

Distributed tests removed -- run with torchrun separately.
"""

import pytest

from xorl.distributed.pipeline_parallel import generate_llm_fqn_per_model_part

pytestmark = [pytest.mark.distributed]


class TestFQNGeneration:
    """Test generate_llm_fqn_per_model_part FQN distribution logic."""

    def test_basic_stage_distribution(self):
        """Various stage/layer combos: correct stage count, all layers present and contiguous."""
        # 2 stages, 4 layers (default FQN names)
        result = generate_llm_fqn_per_model_part(2, 4)
        assert len(result) == 2
        assert result[0][0] == "tok_embeddings"
        assert result[-1][-2:] == ["norm", "output"]
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert len(all_layers) == 4

        # 2 stages, 8 layers
        result = generate_llm_fqn_per_model_part(2, 8)
        assert len(result) == 2
        assert len([m for stage in result for m in stage if m.startswith("layers.")]) == 8

        # 4 stages, 36 layers (Qwen3 8B)
        result = generate_llm_fqn_per_model_part(4, 36)
        assert len(result) == 4
        assert len([m for stage in result for m in stage if m.startswith("layers.")]) == 36

        # Contiguous layer assignment (4 stages, 12 layers)
        result = generate_llm_fqn_per_model_part(4, 12)
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert all_layers == [f"layers.{i}" for i in range(12)]

        # Minimal: 2 layers, 2 stages
        result = generate_llm_fqn_per_model_part(2, 2)
        assert len([m for stage in result for m in stage if m.startswith("layers.")]) == 2

    def test_qwen3_fqn_names_and_single_stage(self):
        """Qwen3-style nested FQN names; single stage contains all modules."""
        result = generate_llm_fqn_per_model_part(
            2, 4,
            input_fqns=["model.embed_tokens"],
            layer_prefix="model.layers",
            output_fqns=["model.norm", "lm_head"],
        )
        assert len(result) == 2
        assert result[0][0] == "model.embed_tokens"
        assert result[-1][-2:] == ["model.norm", "lm_head"]
        assert len([m for stage in result for m in stage if m.startswith("model.layers.")]) == 4

        # Single stage
        result = generate_llm_fqn_per_model_part(1, 4)
        assert len(result) == 1
        assert result[0][0] == "tok_embeddings"
        assert result[0][-1] == "output"
        assert len(result[0]) == 7  # tok_embeddings + 4 layers + norm + output

    def test_error_too_many_stages(self):
        """Error when more stages than effective layers."""
        with pytest.raises(ValueError):
            generate_llm_fqn_per_model_part(10, 2)
