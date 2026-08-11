"""
Tests for pipeline parallelism -- unit tests only.

Distributed tests removed -- run with torchrun separately.
"""

import pytest

from xorl.distributed.pipeline_parallel import (
    generate_llm_fqn_per_model_part,
    schedule_splits_backward,
    schedule_stage_style,
    stage_ids_for_rank,
    validate_pp_schedule_config,
)


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


class TestFQNGeneration:
    """Test generate_llm_fqn_per_model_part FQN distribution logic."""

    def _assert_fqn_partitioning_policy(self):
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

        self._assert_qwen3_fqn_names_and_single_stage()
        self._assert_error_too_many_stages()
        self._assert_virtual_stage_split()
        self._assert_explicit_first_last_layer_counts()
        self._assert_explicit_layer_counts_infeasible()
        self._assert_weighted_split_coverage()

    def _assert_qwen3_fqn_names_and_single_stage(self):
        """Qwen3-style nested FQN names; single stage contains all modules."""
        result = generate_llm_fqn_per_model_part(
            2,
            4,
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

    def _assert_error_too_many_stages(self):
        """Error when more stages than effective layers."""
        with pytest.raises(ValueError):
            generate_llm_fqn_per_model_part(10, 2)

    def _assert_virtual_stage_split(self):
        """num_stages > pp_degree (virtual stages): all layers covered, contiguous."""
        result = generate_llm_fqn_per_model_part(
            4,
            8,
            input_fqns=["model.embed_tokens"],
            layer_prefix="model.layers",
            output_fqns=["model.norm", "lm_head"],
        )
        assert len(result) == 4
        assert result[0][0] == "model.embed_tokens"
        assert result[-1][-2:] == ["model.norm", "lm_head"]
        all_layers = [m for stage in result for m in stage if m.startswith("model.layers.")]
        assert all_layers == [f"model.layers.{i}" for i in range(8)]

    def _assert_explicit_first_last_layer_counts(self):
        """Megatron-style pinned first/last stage layer counts override the weight heuristic."""
        result = generate_llm_fqn_per_model_part(4, 8, num_layers_in_first_stage=1, num_layers_in_last_stage=1)
        layer_counts = [len([m for m in stage if m.startswith("layers.")]) for stage in result]
        assert layer_counts == [1, 3, 3, 1]
        assert result[0][0] == "tok_embeddings"
        assert result[-1][-2:] == ["norm", "output"]

        # Only-first pinned: remaining stages (incl. last) split evenly
        result = generate_llm_fqn_per_model_part(3, 7, num_layers_in_first_stage=1)
        layer_counts = [len([m for m in stage if m.startswith("layers.")]) for stage in result]
        assert layer_counts == [1, 3, 3]

    def _assert_explicit_layer_counts_infeasible(self):
        """Pinned counts leaving too few layers for the unpinned stages raise."""
        with pytest.raises(ValueError):
            generate_llm_fqn_per_model_part(4, 3, num_layers_in_first_stage=1, num_layers_in_last_stage=1)
        with pytest.raises(ValueError):
            generate_llm_fqn_per_model_part(2, 4, num_layers_in_first_stage=3, num_layers_in_last_stage=3)

    def _assert_weighted_split_coverage(self):
        """input/output weights shift layers off the first/last stages but never drop layers."""
        result = generate_llm_fqn_per_model_part(6, 13, input_weight=2, output_weight=3)
        all_layers = [m for stage in result for m in stage if m.startswith("layers.")]
        assert all_layers == [f"layers.{i}" for i in range(13)]


class TestStagePlacement:
    """stage_ids_for_rank must match torch's generate_stage_to_rank_mapping."""

    def _assert_stage_placement_policy(self):
        from torch.distributed.pipelining._utils import generate_rank_to_stage_mapping

        # One single-stage loop mapping and both multi-stage formulas exercise
        # every production branch. More PP sizes only repeat the same index
        # arithmetic and previously created 24 separately collected cases.
        for pp_size, stages_per_rank, style in (
            (2, 1, "loop"),
            (4, 2, "loop"),
            (4, 2, "v"),
        ):
            num_stages = pp_size * stages_per_rank
            reference = generate_rank_to_stage_mapping(pp_size, num_stages, style=style)
            for rank in range(pp_size):
                assert stage_ids_for_rank(rank, pp_size, num_stages, style) == reference[rank]

        self._assert_every_stage_owned_exactly_once()
        self._assert_single_style_requires_one_stage_per_rank()
        self._assert_v_style_first_rank_owns_first_and_last()

    def _assert_every_stage_owned_exactly_once(self):
        for style in ("loop", "v"):
            owned = [s for r in range(4) for s in stage_ids_for_rank(r, 4, 8, style)]
            assert sorted(owned) == list(range(8))

    def _assert_single_style_requires_one_stage_per_rank(self):
        assert stage_ids_for_rank(1, 4, 4, "single") == [1]
        with pytest.raises(ValueError):
            stage_ids_for_rank(0, 4, 8, "single")

    def _assert_v_style_first_rank_owns_first_and_last(self):
        assert stage_ids_for_rank(0, 4, 8, "v") == [0, 7]
        assert stage_ids_for_rank(3, 4, 8, "v") == [3, 4]


class TestScheduleValidation:
    """Schedule whitelist + virtual-stage/microbatch constraint checks."""

    def _assert_schedule_metadata_and_admission_policy(self):
        expected = {
            "1F1B": ("single", False),
            "GPipe": ("single", False),
            "Interleaved1F1B": ("loop", False),
            "InterleavedZeroBubble": ("loop", True),
            "ZBVZeroBubble": ("v", True),
            "DualPipeV": ("v", True),
        }
        for schedule, (style, splits_backward) in expected.items():
            assert schedule_stage_style(schedule) == style
            assert schedule_splits_backward(schedule) is splits_backward
        with pytest.raises(ValueError):
            schedule_stage_style("LoopedBFS")

        self._assert_schedule_config_admission()

    def _assert_schedule_config_admission(self):
        validate_pp_schedule_config("1F1B", 1, 8, 4)
        validate_pp_schedule_config("Interleaved1F1B", 2, 8, 4)
        validate_pp_schedule_config("InterleavedZeroBubble", 2, 8, 4)
        validate_pp_schedule_config("ZBVZeroBubble", 2, 8, 4)
        validate_pp_schedule_config("DualPipeV", 2, 8, 4)
        # single-stage schedule with virtual stages
        with pytest.raises(ValueError):
            validate_pp_schedule_config("1F1B", 2, 8, 4)
        # v-style schedules need exactly 2 chunks
        with pytest.raises(ValueError):
            validate_pp_schedule_config("ZBVZeroBubble", 3, 12, 4)
        with pytest.raises(ValueError):
            validate_pp_schedule_config("DualPipeV", 1, 8, 4)
        # interleaved round-divisibility: m=6, p=4 -> rounds=1... 6%1==0 is fine;
        # m=10, p=4 -> rounds=2, 10%2==0 fine; m=9, p=4 -> rounds=2, 9%2!=0 fails
        with pytest.raises(ValueError):
            validate_pp_schedule_config("Interleaved1F1B", 2, 9, 4)
        # DualPipeV minimum microbatch count (m >= num_stages)
        with pytest.raises(ValueError):
            validate_pp_schedule_config("DualPipeV", 2, 4, 4)


def test_pipeline_partition_placement_and_schedule_policy():
    TestFQNGeneration()._assert_fqn_partitioning_policy()
    TestStagePlacement()._assert_stage_placement_policy()
    TestScheduleValidation()._assert_schedule_metadata_and_admission_policy()
