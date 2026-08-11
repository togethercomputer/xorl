"""Tests for MoE experts with LoRA across all backends (eager, triton, native, quack)."""

from dataclasses import make_dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xorl.lora import inject_lora_into_model
from xorl.models.layers.moe import MoEBlock, MoEExperts, MoEExpertsLoRA, MoELoRAConfig
from xorl.models.layers.moe.backend import zero_token_lora_output


pytestmark = [pytest.mark.cpu, pytest.mark.gpu]


def _assert_zero_token_output_materializes_structural_gradients_for_every_local_factor():
    tokens = torch.empty(0, 4, requires_grad=True)
    factors = tuple(nn.Parameter(torch.randn(shape)) for shape in ((1, 4, 2), (2, 2, 6), (2, 6, 2)))

    output = zero_token_lora_output(tokens, 4, *factors)
    output.sum().backward()

    assert output.shape == (0, 4)
    assert tokens.grad is not None and tokens.grad.shape == tokens.shape
    for factor in factors:
        assert factor.grad is not None
        assert torch.equal(factor.grad, torch.zeros_like(factor))


class MockConfig:
    """Mock config for testing."""

    def __init__(
        self,
        num_experts=4,
        hidden_size=64,
        moe_intermediate_size=128,
        hidden_act="silu",
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob


# ---------------------------------------------------------------------------
# LoRA initialization: all backends, frozen/trainable, shapes, repr
# ---------------------------------------------------------------------------


class TestMoEExpertsLoRAInit:
    """Comprehensive tests for MoEExpertsLoRA initialization across backends."""

    def _assert_init_frozen_trainable_shapes(self):
        """Test init, base weights frozen, LoRA weights trainable, shapes, and repr."""
        config = MockConfig()
        lora_config = MoELoRAConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])
        r = lora_config.r

        for backend in ("eager", "triton", "native", "quack"):
            experts = MoEExpertsLoRA(
                num_experts=config.num_experts,
                hidden_dim=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                moe_implementation=backend,
                lora_config=lora_config,
            )

            # Init fields
            assert experts.num_experts == config.num_experts
            assert experts.moe_implementation == backend
            assert experts.lora_config == lora_config

            # Base weights frozen
            assert not experts.gate_proj.requires_grad
            assert not experts.up_proj.requires_grad
            assert not experts.down_proj.requires_grad

            # LoRA weights trainable, B initialized to zeros, correct shapes
            for name in lora_config.target_modules:
                lora_A = getattr(experts, f"{name}_lora_A")
                lora_B = getattr(experts, f"{name}_lora_B")
                assert isinstance(lora_A, nn.Parameter) and lora_A.requires_grad
                assert isinstance(lora_B, nn.Parameter) and lora_B.requires_grad
                assert torch.allclose(lora_B, torch.zeros_like(lora_B))

            # LoRA weight shapes
            assert experts.gate_proj_lora_A.shape == (config.num_experts, config.hidden_size, r)
            assert experts.gate_proj_lora_B.shape == (config.num_experts, r, config.moe_intermediate_size)
            assert experts.up_proj_lora_A.shape == (config.num_experts, config.hidden_size, r)
            assert experts.up_proj_lora_B.shape == (config.num_experts, r, config.moe_intermediate_size)
            assert experts.down_proj_lora_A.shape == (config.num_experts, config.moe_intermediate_size, r)
            assert experts.down_proj_lora_B.shape == (config.num_experts, r, config.hidden_size)

            repr_str = experts.extra_repr()
            assert f"num_experts={config.num_experts}" in repr_str
            assert f"r={lora_config.r}" in repr_str

        experts.set_runtime_lora_config(lora_rank=3, lora_alpha=12)

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            lora_A, lora_B = experts._active_lora_views(proj_name)
            assert lora_A.shape[-1] == 3
            assert lora_B.shape[1] == 3
            assert lora_A.is_contiguous()
            assert lora_B.is_contiguous()

        TestFromModuleAndInjection()._assert_from_module_and_inject_lora()


# ---------------------------------------------------------------------------
# 3. Eager LoRA forward/backward (CPU)
# ---------------------------------------------------------------------------


class TestMoEExpertsLoRAEager:
    """Test eager LoRA forward/backward on CPU, including via MoEBlock."""

    def test_eager_forward_backward_and_moe_block(self):
        """Test eager per-expert forward, backward gradients, and end-to-end MoEBlock."""
        TestMoEExpertsLoRAInit()._assert_init_frozen_trainable_shapes()
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        experts = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=64,
            moe_implementation="eager",
            lora_config=lora_config,
        )
        nn.init.xavier_normal_(experts.gate_proj.data)
        nn.init.xavier_normal_(experts.up_proj.data)
        nn.init.xavier_normal_(experts.down_proj.data)

        # Forward
        hidden = torch.randn(8, 32)
        out = experts(hidden, expert_idx=0)
        assert out.shape == (8, 32)

        # Backward
        hidden = torch.randn(8, 32, requires_grad=True)
        out = experts(hidden, expert_idx=0)
        out.sum().backward()
        assert experts.gate_proj_lora_A.grad is not None
        assert experts.gate_proj_lora_B.grad is not None
        assert experts.gate_proj.grad is None  # base frozen

        # MoEBlock end-to-end
        block = MoEBlock(
            hidden_size=32,
            num_experts=4,
            top_k=2,
            intermediate_size=64,
            moe_implementation="eager",
        )
        nn.init.xavier_normal_(block.experts.gate_proj.data)
        nn.init.xavier_normal_(block.experts.up_proj.data)
        nn.init.xavier_normal_(block.experts.down_proj.data)
        nn.init.xavier_normal_(block.gate.weight.data)

        block.inject_lora(r=4, lora_alpha=8)
        assert isinstance(block.experts, MoEExpertsLoRA)

        hidden = torch.randn(2, 4, 32)
        output, router_logits = block(hidden)
        assert output.shape == hidden.shape

        output.sum().backward()
        assert block.experts.gate_proj_lora_A.grad is not None

        block = MoEBlock(
            hidden_size=32,
            num_experts=4,
            top_k=2,
            intermediate_size=64,
            moe_implementation="eager",
        )

        block.inject_lora(r=4, lora_alpha=8, hybrid_shared=True)

        assert block.experts.gate_proj_lora_A.shape == (1, 32, 4)
        assert block.experts.gate_proj_lora_B.shape == (4, 4, 64)
        assert block.experts.up_proj_lora_A.shape == (1, 32, 4)
        assert block.experts.up_proj_lora_B.shape == (4, 4, 64)
        assert block.experts.down_proj_lora_A.shape == (4, 64, 4)
        assert block.experts.down_proj_lora_B.shape == (1, 4, 32)

        _assert_zero_token_output_materializes_structural_gradients_for_every_local_factor()
        TestEPLoRARouterScores()._assert_router_score_application_contract()


# ---------------------------------------------------------------------------
# 4. Cross-backend numerical correctness
# ---------------------------------------------------------------------------


def _make_lora_block(
    backend,
    num_experts,
    hidden_dim,
    intermediate,
    r,
    lora_alpha,
    device,
    dtype,
):
    """Create a MoEBlock with LoRA on the given backend, with deterministic init."""
    block = MoEBlock(
        hidden_size=hidden_dim,
        num_experts=num_experts,
        top_k=2,
        intermediate_size=intermediate,
        moe_implementation=backend,
    )
    # Deterministic init for reproducibility
    torch.manual_seed(42)
    nn.init.xavier_normal_(block.experts.gate_proj.data)
    nn.init.xavier_normal_(block.experts.up_proj.data)
    nn.init.xavier_normal_(block.experts.down_proj.data)
    nn.init.xavier_normal_(block.gate.weight.data)
    block = block.to(device).to(dtype)

    # Inject LoRA (deterministic due to seed reset)
    torch.manual_seed(123)
    block.inject_lora(r=r, lora_alpha=lora_alpha)
    return block


def _copy_block_weights(src_block, dst_block):
    """Copy all weights from src to dst block (base + LoRA + gate)."""
    with torch.no_grad():
        dst_block.gate.weight.copy_(src_block.gate.weight)
        src_exp = src_block.experts
        dst_exp = dst_block.experts
        # Base weights
        dst_exp.gate_proj.copy_(src_exp.gate_proj)
        dst_exp.up_proj.copy_(src_exp.up_proj)
        dst_exp.down_proj.copy_(src_exp.down_proj)
        # LoRA weights
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            getattr(dst_exp, f"{proj}_lora_A").copy_(getattr(src_exp, f"{proj}_lora_A"))
            getattr(dst_exp, f"{proj}_lora_B").copy_(getattr(src_exp, f"{proj}_lora_B"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCrossBackendConsistency:
    """Test numerical agreement of outputs and gradients across backends."""

    NUM_EXPERTS = 4
    HIDDEN_DIM = 64
    INTERMEDIATE = 128
    R = 8
    LORA_ALPHA = 16
    DTYPE = torch.bfloat16

    def _make_pair(self, ref_backend, test_backend, device):
        """Create a reference and test block with identical weights."""
        ref = _make_lora_block(
            ref_backend,
            self.NUM_EXPERTS,
            self.HIDDEN_DIM,
            self.INTERMEDIATE,
            self.R,
            self.LORA_ALPHA,
            device,
            self.DTYPE,
        )
        test = _make_lora_block(
            test_backend,
            self.NUM_EXPERTS,
            self.HIDDEN_DIM,
            self.INTERMEDIATE,
            self.R,
            self.LORA_ALPHA,
            device,
            self.DTYPE,
        )
        torch.manual_seed(321)
        with torch.no_grad():
            for proj in ("gate_proj", "up_proj", "down_proj"):
                nn.init.xavier_normal_(getattr(ref.experts, f"{proj}_lora_B"))
        _copy_block_weights(ref, test)
        return ref, test

    def test_zero_lora_and_cross_backend_numerical_policy(self):
        """With lora_B=0, LoRA output must equal base model output (no delta)."""
        device = "cuda"
        backend = "eager"
        # Base block (no LoRA)
        base_block = MoEBlock(
            hidden_size=self.HIDDEN_DIM,
            num_experts=self.NUM_EXPERTS,
            top_k=2,
            intermediate_size=self.INTERMEDIATE,
            moe_implementation=backend,
        )
        torch.manual_seed(42)
        nn.init.xavier_normal_(base_block.experts.gate_proj.data)
        nn.init.xavier_normal_(base_block.experts.up_proj.data)
        nn.init.xavier_normal_(base_block.experts.down_proj.data)
        nn.init.xavier_normal_(base_block.gate.weight.data)
        base_block = base_block.to(device).to(self.DTYPE)

        # LoRA block with lora_B = 0
        lora_block = MoEBlock(
            hidden_size=self.HIDDEN_DIM,
            num_experts=self.NUM_EXPERTS,
            top_k=2,
            intermediate_size=self.INTERMEDIATE,
            moe_implementation=backend,
        )
        torch.manual_seed(42)
        nn.init.xavier_normal_(lora_block.experts.gate_proj.data)
        nn.init.xavier_normal_(lora_block.experts.up_proj.data)
        nn.init.xavier_normal_(lora_block.experts.down_proj.data)
        nn.init.xavier_normal_(lora_block.gate.weight.data)
        lora_block = lora_block.to(device).to(self.DTYPE)
        lora_block.inject_lora(r=self.R, lora_alpha=self.LORA_ALPHA)

        torch.manual_seed(999)
        hidden = torch.randn(2, 8, self.HIDDEN_DIM, device=device, dtype=self.DTYPE)

        base_out, _ = base_block(hidden)
        lora_out, _ = lora_block(hidden)

        torch.testing.assert_close(
            lora_out,
            base_out,
            atol=1e-3,
            rtol=1e-2,
            msg=f"[{backend}] Zero-LoRA output should match base model",
        )

        self._assert_cross_backend_output_and_gradients_policy()

    def _assert_cross_backend_output_and_gradients(self, ref_backend, test_backend):
        """Cross-backend outputs and LoRA gradients should match."""
        ref, test = self._make_pair(ref_backend, test_backend, "cuda")

        torch.manual_seed(999)
        h1 = torch.randn(2, 8, self.HIDDEN_DIM, device="cuda", dtype=self.DTYPE)
        h2 = h1.clone()

        ref_out, _ = ref(h1)
        ref_out.sum().backward()

        test_out, _ = test(h2)
        test_out.sum().backward()

        # Output agreement
        torch.testing.assert_close(
            test_out,
            ref_out,
            atol=0.05,
            rtol=0.02,
            msg=f"{ref_backend} vs {test_backend} output mismatch",
        )

        # Gradient agreement
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            ref_grad_A = getattr(ref.experts, f"{proj}_lora_A").grad
            test_grad_A = getattr(test.experts, f"{proj}_lora_A").grad
            ref_grad_B = getattr(ref.experts, f"{proj}_lora_B").grad
            test_grad_B = getattr(test.experts, f"{proj}_lora_B").grad

            assert ref_grad_A is not None, f"ref {proj}_lora_A grad is None"
            assert test_grad_A is not None, f"test {proj}_lora_A grad is None"

            torch.testing.assert_close(
                test_grad_A,
                ref_grad_A,
                atol=0.05,
                rtol=0.05,
                msg=f"Gradient mismatch: {proj}_lora_A ({ref_backend} vs {test_backend})",
            )
            torch.testing.assert_close(
                test_grad_B,
                ref_grad_B,
                atol=0.05,
                rtol=0.05,
                msg=f"Gradient mismatch: {proj}_lora_B ({ref_backend} vs {test_backend})",
            )

    def _assert_cross_backend_output_and_gradients_policy(self):
        for ref_backend, test_backend in (("eager", "native"), ("eager", "triton")):
            self._assert_cross_backend_output_and_gradients(ref_backend, test_backend)


# ---------------------------------------------------------------------------
# 5. from_module + LoRA injection + error handling
# ---------------------------------------------------------------------------


class TestFromModuleAndInjection:
    """Test from_module, inject_lora, and error handling."""

    def _assert_from_module_and_inject_lora(self):
        """Test from_module preserves backend/weights, and inject_lora works via both APIs."""
        config = MockConfig()
        backend = "quack"

        # from_module
        base = MoEExperts(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation=backend,
        )
        nn.init.xavier_normal_(base.gate_proj.data)
        nn.init.xavier_normal_(base.up_proj.data)
        nn.init.xavier_normal_(base.down_proj.data)

        lora_exp = MoEExpertsLoRA.from_module(base, r=8, lora_alpha=16)
        assert lora_exp.moe_implementation == backend
        assert torch.allclose(lora_exp.gate_proj, base.gate_proj)

        # inject_lora_into_model

        class SimpleModel(nn.Module):
            def __init__(self, config, backend):
                super().__init__()
                self.experts = MoEExperts(
                    num_experts=config.num_experts,
                    hidden_dim=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    moe_implementation=backend,
                )
                nn.init.xavier_normal_(self.experts.gate_proj.data)
                nn.init.xavier_normal_(self.experts.up_proj.data)
                nn.init.xavier_normal_(self.experts.down_proj.data)

        model = SimpleModel(config, backend)
        inject_lora_into_model(model, r=8, lora_alpha=16, target_modules=["experts"])
        assert isinstance(model.experts, MoEExpertsLoRA)
        assert model.experts.moe_implementation == backend
        assert hasattr(model.experts, "gate_proj_lora_A")

        # MoEBlock.inject_lora
        block = MoEBlock(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation=backend,
        )
        block.inject_lora(r=8, lora_alpha=16)
        assert isinstance(block.experts, MoEExpertsLoRA)
        assert block.experts.moe_implementation == backend
        assert block.lora_adapter == "injected"


# ---------------------------------------------------------------------------
# 6. EP LoRA router-score application
# ---------------------------------------------------------------------------


class TestEPLoRARouterScores:
    """Test that MoEExpertsLoRA._ep_forward applies router scores from the dispatch context.

    The LoRA EP compute functions don't accept expert_scores (unlike the non-LoRA
    path), so _ep_forward must apply them post-compute. This test verifies:
    - Scores from "expert_scores" (alltoall) and "permuted_scores" (deepep) are applied
    - Missing scores leave the output unchanged
    - Gradients flow through the score multiplication to LoRA parameters
    """

    NUM_EXPERTS = 4
    HIDDEN_DIM = 32
    INTERMEDIATE = 64
    R = 4
    NUM_TOKENS = 8

    def _make_experts(self):
        experts = MoEExpertsLoRA(
            num_experts=self.NUM_EXPERTS,
            hidden_dim=self.HIDDEN_DIM,
            intermediate_size=self.INTERMEDIATE,
            moe_implementation="native",
            lora_config=MoELoRAConfig(r=self.R, lora_alpha=8),
        )
        nn.init.xavier_normal_(experts.gate_up_proj.data)
        nn.init.xavier_normal_(experts.down_proj.data)
        experts.ep_dispatch = "alltoall"
        return experts

    def _run_ep_forward(self, experts, score_attr=None, scores=None, compute_output=None):
        """Run _ep_forward with mocked dispatch/compute/combine.

        Returns (final_output, expert_output_passed_to_combine).
        """
        if compute_output is None:
            compute_output = torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM)

        # Build dispatch context with the requested score attribute
        fields = []
        if score_attr is not None:
            fields.append((score_attr, torch.Tensor))
        Ctx = make_dataclass("Ctx", fields)
        ctx = Ctx(**({score_attr: scores} if score_attr else {}))

        mock_dispatch = MagicMock(
            return_value=(
                torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM),  # permute_tokens
                torch.tensor([2, 4, 6, 8]),  # cumsum
                ctx,
            )
        )
        mock_compute = MagicMock(return_value=compute_output)

        captured = {}

        def mock_combine(**kwargs):
            captured["expert_output"] = kwargs["expert_output"]
            return kwargs["expert_output"]

        mock_ps = MagicMock()
        mock_ps.ep_enabled = True
        mock_ps.ep_group = MagicMock()

        with (
            patch.dict("xorl.models.layers.moe.lora.EP_DISPATCH", {"alltoall": mock_dispatch}),
            patch.dict("xorl.models.layers.moe.lora.EP_COMBINE", {"alltoall": mock_combine}),
            patch.dict("xorl.models.layers.moe.lora.EP_EXPERT_COMPUTE_LORA", {"native": mock_compute}),
            patch("xorl.distributed.parallel_state.get_parallel_state", return_value=mock_ps),
        ):
            hidden = torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM)
            routing_weights = torch.softmax(torch.randn(self.NUM_TOKENS, 2), dim=-1)
            selected_experts = torch.randint(0, self.NUM_EXPERTS, (self.NUM_TOKENS, 2))

            output = experts(hidden, routing_weights, selected_experts)

        return output, captured["expert_output"]

    def _assert_router_score_application_contract(self):
        """Router scores apply when present, preserve identity when absent, and remain differentiable."""
        for score_attr in ("expert_scores", "permuted_scores"):
            experts = self._make_experts()
            compute_output = torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM)
            scores = torch.rand(self.NUM_TOKENS) * 0.5 + 0.1  # non-trivial scores in (0.1, 0.6)

            _, expert_output = self._run_ep_forward(
                experts,
                score_attr=score_attr,
                scores=scores,
                compute_output=compute_output,
            )

            expected = compute_output * scores.unsqueeze(1)
            torch.testing.assert_close(expert_output, expected)

        self._assert_no_scores_leave_output_unchanged()
        self._assert_gradients_flow_through_scores()

    def _assert_no_scores_leave_output_unchanged(self):
        """When dispatch context has no score attribute, expert output is unchanged."""
        experts = self._make_experts()
        compute_output = torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM)

        _, expert_output = self._run_ep_forward(
            experts,
            score_attr=None,
            compute_output=compute_output,
        )

        torch.testing.assert_close(expert_output, compute_output)

    def _assert_gradients_flow_through_scores(self):
        """Gradients from the score multiplication reach LoRA parameters."""
        experts = self._make_experts()
        compute_output = torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM, requires_grad=True)
        scores = torch.rand(self.NUM_TOKENS, requires_grad=True)

        Ctx = make_dataclass("Ctx", [("expert_scores", torch.Tensor)])
        ctx = Ctx(expert_scores=scores)

        mock_dispatch = MagicMock(
            return_value=(
                torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM),
                torch.tensor([2, 4, 6, 8]),
                ctx,
            )
        )
        mock_compute = MagicMock(return_value=compute_output)

        def mock_combine(**kwargs):
            return kwargs["expert_output"]

        mock_ps = MagicMock()
        mock_ps.ep_enabled = True
        mock_ps.ep_group = MagicMock()

        with (
            patch.dict("xorl.models.layers.moe.lora.EP_DISPATCH", {"alltoall": mock_dispatch}),
            patch.dict("xorl.models.layers.moe.lora.EP_COMBINE", {"alltoall": mock_combine}),
            patch.dict("xorl.models.layers.moe.lora.EP_EXPERT_COMPUTE_LORA", {"native": mock_compute}),
            patch("xorl.distributed.parallel_state.get_parallel_state", return_value=mock_ps),
        ):
            hidden = torch.randn(self.NUM_TOKENS, self.HIDDEN_DIM)
            routing_weights = torch.softmax(torch.randn(self.NUM_TOKENS, 2), dim=-1)
            selected_experts = torch.randint(0, self.NUM_EXPERTS, (self.NUM_TOKENS, 2))

            output = experts(hidden, routing_weights, selected_experts)
            output.sum().backward()

        # Gradient should flow through scores back to compute_output
        assert compute_output.grad is not None
        assert scores.grad is not None
        # scores.grad should equal the sum of (compute_output * grad_output) per token
        expected_score_grad = (compute_output.detach() * 1.0).sum(dim=1)  # grad_output is all 1s from .sum()
        torch.testing.assert_close(scores.grad, expected_score_grad)
