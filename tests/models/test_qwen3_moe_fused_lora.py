"""Tests for MoE experts with LoRA across all backends (eager, triton, native, quack)."""

import pytest
import torch
import torch.nn as nn

from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTritonExperts,
    Qwen3MoeSparseExperts,
)
from xorl.models.layers.moe import MoEExperts, MoEExpertsLoRA, MoELoRAConfig, MoEBlock, MOE_EXPERT_BACKENDS
from xorl.lora.mapping import can_apply_lora, get_lora_class_for_module


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
# Base expert tests
# ---------------------------------------------------------------------------


class TestMoEExpertsInit:
    """Tests for base MoEExperts initialization."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    def test_triton_experts_init(self, config):
        """Test that TritonExperts initializes correctly."""
        experts = Qwen3MoeTritonExperts(config)
        assert experts.num_experts == config.num_experts
        assert experts.hidden_dim == config.hidden_size
        assert experts.intermediate_size == config.moe_intermediate_size
        assert experts.moe_implementation == "triton"

    def test_sparse_experts_init(self, config):
        """Test that SparseExperts initializes correctly."""
        experts = Qwen3MoeSparseExperts(config)
        assert experts.moe_implementation == "eager"

    def test_direct_moe_experts_init(self, config):
        """Test direct MoEExperts instantiation with different backends."""
        for backend in ["eager", "triton", "native", "quack"]:
            experts = MoEExperts(
                num_experts=config.num_experts,
                hidden_dim=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                moe_implementation=backend,
            )
            assert experts.moe_implementation == backend

    def test_weight_shapes(self, config):
        """Test weight shapes match expected dimensions."""
        experts = MoEExperts(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        assert experts.gate_proj.shape == (config.num_experts, config.hidden_size, config.moe_intermediate_size)
        assert experts.up_proj.shape == (config.num_experts, config.hidden_size, config.moe_intermediate_size)
        assert experts.down_proj.shape == (config.num_experts, config.moe_intermediate_size, config.hidden_size)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton MoE")
    def test_triton_forward(self, config):
        """Test forward pass with triton backend on GPU."""
        experts = Qwen3MoeTritonExperts(config)
        nn.init.xavier_normal_(experts.gate_proj.data)
        nn.init.xavier_normal_(experts.up_proj.data)
        nn.init.xavier_normal_(experts.down_proj.data)

        device = "cuda"
        experts = experts.to(device).to(torch.bfloat16)

        num_tokens, top_k = 16, 2
        hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected_experts = torch.randint(0, config.num_experts, (num_tokens, top_k), device=device)

        output = experts(hidden_states, routing_weights, selected_experts)
        assert output.shape == hidden_states.shape

    def test_lora_mapping_registered(self, config):
        """Test that MoEExperts is registered in LoRA mapping."""
        experts = MoEExperts(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        assert can_apply_lora(experts)
        lora_cls = get_lora_class_for_module(experts)
        assert lora_cls is MoEExpertsLoRA


# ---------------------------------------------------------------------------
# LoRA expert tests — all backends
# ---------------------------------------------------------------------------


class TestMoEExpertsLoRAInit:
    """Tests for MoEExpertsLoRA initialization across backends."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.fixture
    def lora_config(self):
        return MoELoRAConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])

    @pytest.mark.parametrize("backend", ["eager", "triton", "native", "quack"])
    def test_init_all_backends(self, config, lora_config, backend):
        """Test that MoEExpertsLoRA initializes correctly for all backends."""
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation=backend,
            lora_config=lora_config,
        )
        assert experts.num_experts == config.num_experts
        assert experts.moe_implementation == backend
        assert experts.lora_config == lora_config

    def test_base_weights_frozen(self, config, lora_config):
        """Test that base weights are frozen (requires_grad=False)."""
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            lora_config=lora_config,
        )
        assert not experts.gate_proj.requires_grad
        assert not experts.up_proj.requires_grad
        assert not experts.down_proj.requires_grad

    def test_lora_weights_trainable(self, config, lora_config):
        """Test that LoRA weights are trainable."""
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            lora_config=lora_config,
        )
        for name in lora_config.target_modules:
            lora_A = getattr(experts, f"{name}_lora_A")
            lora_B = getattr(experts, f"{name}_lora_B")
            assert isinstance(lora_A, nn.Parameter)
            assert isinstance(lora_B, nn.Parameter)
            assert lora_A.requires_grad
            assert lora_B.requires_grad

    def test_lora_b_initialized_to_zeros(self, config, lora_config):
        """Test that LoRA B is initialized to zeros."""
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            lora_config=lora_config,
        )
        for name in lora_config.target_modules:
            lora_B = getattr(experts, f"{name}_lora_B")
            assert torch.allclose(lora_B, torch.zeros_like(lora_B))

    def test_lora_weight_shapes(self, config, lora_config):
        """Test LoRA weight shapes."""
        r = lora_config.r
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            lora_config=lora_config,
        )
        assert experts.gate_proj_lora_A.shape == (config.num_experts, config.hidden_size, r)
        assert experts.gate_proj_lora_B.shape == (config.num_experts, r, config.moe_intermediate_size)
        assert experts.up_proj_lora_A.shape == (config.num_experts, config.hidden_size, r)
        assert experts.up_proj_lora_B.shape == (config.num_experts, r, config.moe_intermediate_size)
        assert experts.down_proj_lora_A.shape == (config.num_experts, config.moe_intermediate_size, r)
        assert experts.down_proj_lora_B.shape == (config.num_experts, r, config.hidden_size)

    def test_extra_repr(self, config, lora_config):
        """Test extra_repr includes LoRA info."""
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            lora_config=lora_config,
        )
        repr_str = experts.extra_repr()
        assert f"num_experts={config.num_experts}" in repr_str
        assert f"r={lora_config.r}" in repr_str


# ---------------------------------------------------------------------------
# LoRA forward/backward tests — eager (CPU)
# ---------------------------------------------------------------------------


class TestMoEExpertsLoRAEager:
    """Test eager LoRA forward/backward on CPU."""

    @pytest.fixture
    def experts(self):
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        exp = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=64,
            moe_implementation="eager",
            lora_config=lora_config,
        )
        nn.init.xavier_normal_(exp.gate_proj.data)
        nn.init.xavier_normal_(exp.up_proj.data)
        nn.init.xavier_normal_(exp.down_proj.data)
        return exp

    def test_eager_forward_single_expert(self, experts):
        """Test eager per-expert forward."""
        hidden = torch.randn(8, 32)
        out = experts(hidden, expert_idx=0)
        assert out.shape == (8, 32)

    def test_eager_backward(self, experts):
        """Test eager backward produces gradients for LoRA weights."""
        hidden = torch.randn(8, 32, requires_grad=True)
        out = experts(hidden, expert_idx=0)
        out.sum().backward()

        assert experts.gate_proj_lora_A.grad is not None
        assert experts.gate_proj_lora_B.grad is not None
        assert experts.gate_proj.grad is None  # base frozen

    def test_eager_via_moe_block(self):
        """Test eager LoRA works end-to-end through MoEBlock."""
        block = MoEBlock(
            hidden_size=32, num_experts=4, top_k=2,
            intermediate_size=64, moe_implementation="eager",
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


# ---------------------------------------------------------------------------
# LoRA forward/backward tests — triton (GPU)
# ---------------------------------------------------------------------------


class TestMoEExpertsLoRATriton:
    """Test triton LoRA forward/backward on GPU."""

    @pytest.fixture
    def experts(self):
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        exp = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=64,
            moe_implementation="triton",
            lora_config=lora_config,
        )
        nn.init.xavier_normal_(exp.gate_proj.data)
        nn.init.xavier_normal_(exp.up_proj.data)
        nn.init.xavier_normal_(exp.down_proj.data)
        return exp

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_forward(self, experts):
        """Test triton LoRA forward."""
        device = "cuda"
        experts = experts.to(device).to(torch.bfloat16)

        num_tokens, top_k = 16, 2
        hidden = torch.randn(num_tokens, 32, device=device, dtype=torch.bfloat16)
        weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected = torch.randint(0, 4, (num_tokens, top_k), device=device)

        output = experts(hidden, weights, selected)
        assert output.shape == hidden.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_backward(self, experts):
        """Test triton LoRA backward."""
        device = "cuda"
        experts = experts.to(device).to(torch.bfloat16)

        num_tokens, top_k = 16, 2
        hidden = torch.randn(num_tokens, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected = torch.randint(0, 4, (num_tokens, top_k), device=device)

        output = experts(hidden, weights, selected)
        output.sum().backward()

        assert experts.gate_proj_lora_A.grad is not None
        assert experts.down_proj_lora_B.grad is not None
        assert experts.gate_proj.grad is None  # base frozen


# ---------------------------------------------------------------------------
# LoRA forward/backward tests — native (GPU, torch._grouped_mm)
# ---------------------------------------------------------------------------


class TestMoEExpertsLoRANative:
    """Test native LoRA forward/backward on GPU using torch._grouped_mm."""

    @pytest.fixture
    def experts(self):
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        exp = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=64,
            moe_implementation="native",
            lora_config=lora_config,
        )
        nn.init.xavier_normal_(exp.gate_proj.data)
        nn.init.xavier_normal_(exp.up_proj.data)
        nn.init.xavier_normal_(exp.down_proj.data)
        return exp

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_native_forward(self, experts):
        """Test native LoRA forward."""
        device = "cuda"
        experts = experts.to(device).to(torch.bfloat16)

        num_tokens, top_k = 16, 2
        hidden = torch.randn(num_tokens, 32, device=device, dtype=torch.bfloat16)
        weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected = torch.randint(0, 4, (num_tokens, top_k), device=device)

        output = experts(hidden, weights, selected)
        assert output.shape == hidden.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_native_backward(self, experts):
        """Test native LoRA backward produces gradients."""
        device = "cuda"
        experts = experts.to(device).to(torch.bfloat16)

        num_tokens, top_k = 16, 2
        hidden = torch.randn(num_tokens, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected = torch.randint(0, 4, (num_tokens, top_k), device=device)

        output = experts(hidden, weights, selected)
        output.sum().backward()

        assert experts.gate_proj_lora_A.grad is not None
        assert experts.gate_proj_lora_B.grad is not None
        assert experts.down_proj_lora_A.grad is not None
        assert experts.down_proj_lora_B.grad is not None
        assert experts.gate_proj.grad is None  # base frozen

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_native_via_moe_block(self):
        """Test native LoRA works end-to-end through MoEBlock."""
        device = "cuda"
        block = MoEBlock(
            hidden_size=32, num_experts=4, top_k=2,
            intermediate_size=64, moe_implementation="native",
        )
        nn.init.xavier_normal_(block.experts.gate_proj.data)
        nn.init.xavier_normal_(block.experts.up_proj.data)
        nn.init.xavier_normal_(block.experts.down_proj.data)
        nn.init.xavier_normal_(block.gate.weight.data)
        block = block.to(device).to(torch.bfloat16)

        block.inject_lora(r=4, lora_alpha=8)
        assert isinstance(block.experts, MoEExpertsLoRA)
        assert block.experts.moe_implementation == "native"

        hidden = torch.randn(2, 4, 32, device=device, dtype=torch.bfloat16)
        output, router_logits = block(hidden)
        assert output.shape == hidden.shape

        output.sum().backward()
        assert block.experts.gate_proj_lora_A.grad is not None


# ---------------------------------------------------------------------------
# Cross-backend numerical correctness
# ---------------------------------------------------------------------------


def _make_lora_block(
    backend, num_experts, hidden_dim, intermediate, r, lora_alpha, device, dtype,
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


class TestCrossBackendConsistency:
    """Test numerical agreement of outputs and gradients across backends.

    The eager backend is used as the reference (simple per-expert loops with
    standard torch.matmul, easy to reason about correctness).
    """

    NUM_EXPERTS = 4
    HIDDEN_DIM = 64
    INTERMEDIATE = 128
    R = 8
    LORA_ALPHA = 16
    DTYPE = torch.bfloat16

    def _make_pair(self, ref_backend, test_backend, device):
        """Create a reference and test block with identical weights."""
        ref = _make_lora_block(
            ref_backend, self.NUM_EXPERTS, self.HIDDEN_DIM, self.INTERMEDIATE,
            self.R, self.LORA_ALPHA, device, self.DTYPE,
        )
        test = _make_lora_block(
            test_backend, self.NUM_EXPERTS, self.HIDDEN_DIM, self.INTERMEDIATE,
            self.R, self.LORA_ALPHA, device, self.DTYPE,
        )
        _copy_block_weights(ref, test)
        return ref, test

    # --- Zero-LoRA correctness: LoRA should be a no-op when lora_B = 0 ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("backend", ["eager", "triton", "native"])
    def test_zero_lora_matches_base(self, backend):
        """With lora_B=0, LoRA output must equal base model output (no delta)."""
        device = "cuda"
        # Base block (no LoRA)
        base_block = MoEBlock(
            hidden_size=self.HIDDEN_DIM, num_experts=self.NUM_EXPERTS, top_k=2,
            intermediate_size=self.INTERMEDIATE, moe_implementation=backend,
        )
        torch.manual_seed(42)
        nn.init.xavier_normal_(base_block.experts.gate_proj.data)
        nn.init.xavier_normal_(base_block.experts.up_proj.data)
        nn.init.xavier_normal_(base_block.experts.down_proj.data)
        nn.init.xavier_normal_(base_block.gate.weight.data)
        base_block = base_block.to(device).to(self.DTYPE)

        # LoRA block with lora_B = 0
        lora_block = MoEBlock(
            hidden_size=self.HIDDEN_DIM, num_experts=self.NUM_EXPERTS, top_k=2,
            intermediate_size=self.INTERMEDIATE, moe_implementation=backend,
        )
        torch.manual_seed(42)
        nn.init.xavier_normal_(lora_block.experts.gate_proj.data)
        nn.init.xavier_normal_(lora_block.experts.up_proj.data)
        nn.init.xavier_normal_(lora_block.experts.down_proj.data)
        nn.init.xavier_normal_(lora_block.gate.weight.data)
        lora_block = lora_block.to(device).to(self.DTYPE)
        lora_block.inject_lora(r=self.R, lora_alpha=self.LORA_ALPHA)
        # lora_B is already zero from initialization — this is the key invariant

        torch.manual_seed(999)
        hidden = torch.randn(2, 8, self.HIDDEN_DIM, device=device, dtype=self.DTYPE)

        base_out, _ = base_block(hidden)
        lora_out, _ = lora_block(hidden)

        torch.testing.assert_close(
            lora_out, base_out, atol=1e-3, rtol=1e-2,
            msg=f"[{backend}] Zero-LoRA output should match base model",
        )

    # --- Cross-backend output agreement ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eager_vs_native_output(self):
        """Eager and native outputs should match (same weights, same routing)."""
        ref, test = self._make_pair("eager", "native", "cuda")

        torch.manual_seed(999)
        hidden = torch.randn(2, 8, self.HIDDEN_DIM, device="cuda", dtype=self.DTYPE)
        ref_out, _ = ref(hidden)
        test_out, _ = test(hidden)

        torch.testing.assert_close(
            test_out, ref_out, atol=0.05, rtol=0.02,
            msg="Eager vs native output mismatch",
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eager_vs_triton_output(self):
        """Eager and triton outputs should match (same weights, same routing)."""
        ref, test = self._make_pair("eager", "triton", "cuda")

        torch.manual_seed(999)
        hidden = torch.randn(2, 8, self.HIDDEN_DIM, device="cuda", dtype=self.DTYPE)
        ref_out, _ = ref(hidden)
        test_out, _ = test(hidden)

        torch.testing.assert_close(
            test_out, ref_out, atol=0.05, rtol=0.02,
            msg="Eager vs triton output mismatch",
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_vs_native_output(self):
        """Triton and native outputs should match."""
        ref, test = self._make_pair("triton", "native", "cuda")

        torch.manual_seed(999)
        hidden = torch.randn(2, 8, self.HIDDEN_DIM, device="cuda", dtype=self.DTYPE)
        ref_out, _ = ref(hidden)
        test_out, _ = test(hidden)

        torch.testing.assert_close(
            test_out, ref_out, atol=0.05, rtol=0.02,
            msg="Triton vs native output mismatch",
        )

    # --- Cross-backend gradient agreement ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eager_vs_native_gradients(self):
        """Eager and native LoRA gradients should match."""
        ref, test = self._make_pair("eager", "native", "cuda")

        torch.manual_seed(999)
        h1 = torch.randn(2, 8, self.HIDDEN_DIM, device="cuda", dtype=self.DTYPE)
        h2 = h1.clone()

        ref_out, _ = ref(h1)
        ref_out.sum().backward()

        test_out, _ = test(h2)
        test_out.sum().backward()

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            ref_grad_A = getattr(ref.experts, f"{proj}_lora_A").grad
            test_grad_A = getattr(test.experts, f"{proj}_lora_A").grad
            ref_grad_B = getattr(ref.experts, f"{proj}_lora_B").grad
            test_grad_B = getattr(test.experts, f"{proj}_lora_B").grad

            assert ref_grad_A is not None, f"ref {proj}_lora_A grad is None"
            assert test_grad_A is not None, f"test {proj}_lora_A grad is None"
            assert ref_grad_B is not None, f"ref {proj}_lora_B grad is None"
            assert test_grad_B is not None, f"test {proj}_lora_B grad is None"

            torch.testing.assert_close(
                test_grad_A, ref_grad_A, atol=0.05, rtol=0.05,
                msg=f"Gradient mismatch: {proj}_lora_A (eager vs native)",
            )
            torch.testing.assert_close(
                test_grad_B, ref_grad_B, atol=0.05, rtol=0.05,
                msg=f"Gradient mismatch: {proj}_lora_B (eager vs native)",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eager_vs_triton_gradients(self):
        """Eager and triton LoRA gradients should match."""
        ref, test = self._make_pair("eager", "triton", "cuda")

        torch.manual_seed(999)
        h1 = torch.randn(2, 8, self.HIDDEN_DIM, device="cuda", dtype=self.DTYPE)
        h2 = h1.clone()

        ref_out, _ = ref(h1)
        ref_out.sum().backward()

        test_out, _ = test(h2)
        test_out.sum().backward()

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            ref_grad_A = getattr(ref.experts, f"{proj}_lora_A").grad
            test_grad_A = getattr(test.experts, f"{proj}_lora_A").grad
            ref_grad_B = getattr(ref.experts, f"{proj}_lora_B").grad
            test_grad_B = getattr(test.experts, f"{proj}_lora_B").grad

            assert ref_grad_A is not None, f"ref {proj}_lora_A grad is None"
            assert test_grad_A is not None, f"test {proj}_lora_A grad is None"

            torch.testing.assert_close(
                test_grad_A, ref_grad_A, atol=0.05, rtol=0.05,
                msg=f"Gradient mismatch: {proj}_lora_A (eager vs triton)",
            )
            torch.testing.assert_close(
                test_grad_B, ref_grad_B, atol=0.05, rtol=0.05,
                msg=f"Gradient mismatch: {proj}_lora_B (eager vs triton)",
            )

    # --- Non-zero LoRA produces a different output from base ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("backend", ["eager", "triton", "native"])
    def test_nonzero_lora_changes_output(self, backend):
        """Non-zero LoRA weights must produce a different output from base."""
        device = "cuda"
        block = _make_lora_block(
            backend, self.NUM_EXPERTS, self.HIDDEN_DIM, self.INTERMEDIATE,
            self.R, self.LORA_ALPHA, device, self.DTYPE,
        )
        # Set lora_B to non-zero
        with torch.no_grad():
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                lora_B = getattr(block.experts, f"{proj}_lora_B")
                nn.init.xavier_normal_(lora_B)

        # Base block (no LoRA) with same base weights
        base_block = MoEBlock(
            hidden_size=self.HIDDEN_DIM, num_experts=self.NUM_EXPERTS, top_k=2,
            intermediate_size=self.INTERMEDIATE, moe_implementation=backend,
        ).to(device).to(self.DTYPE)
        with torch.no_grad():
            base_block.gate.weight.copy_(block.gate.weight)
            base_block.experts.gate_proj.copy_(block.experts.gate_proj)
            base_block.experts.up_proj.copy_(block.experts.up_proj)
            base_block.experts.down_proj.copy_(block.experts.down_proj)

        torch.manual_seed(999)
        hidden = torch.randn(2, 8, self.HIDDEN_DIM, device=device, dtype=self.DTYPE)
        base_out, _ = base_block(hidden)
        lora_out, _ = block(hidden)

        diff = (lora_out - base_out).abs().max().item()
        assert diff > 1e-3, (
            f"[{backend}] Non-zero LoRA should change the output, but max diff={diff}"
        )


# ---------------------------------------------------------------------------
# from_module tests
# ---------------------------------------------------------------------------


class TestFromModule:
    """Test MoEExpertsLoRA.from_module across backends."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.fixture
    def lora_config(self):
        return MoELoRAConfig(r=8, lora_alpha=16)

    @pytest.mark.parametrize("backend", ["eager", "triton", "native", "quack"])
    def test_from_module_preserves_backend(self, config, lora_config, backend):
        """Test from_module preserves the moe_implementation from the source."""
        base = MoEExperts(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation=backend,
        )
        nn.init.xavier_normal_(base.gate_proj.data)
        nn.init.xavier_normal_(base.up_proj.data)
        nn.init.xavier_normal_(base.down_proj.data)

        lora_exp = MoEExpertsLoRA.from_module(
            base, r=lora_config.r, lora_alpha=lora_config.lora_alpha,
        )
        assert lora_exp.moe_implementation == backend
        assert torch.allclose(lora_exp.gate_proj, base.gate_proj)

    def test_from_module_with_qwen3_subclass(self, config, lora_config):
        """Test from_module works with Qwen3MoeTritonExperts (MoEExperts subclass)."""
        base = Qwen3MoeTritonExperts(config)
        nn.init.xavier_normal_(base.gate_proj.data)
        nn.init.xavier_normal_(base.up_proj.data)
        nn.init.xavier_normal_(base.down_proj.data)

        lora_exp = MoEExpertsLoRA.from_module(
            base, r=lora_config.r, lora_alpha=lora_config.lora_alpha,
        )
        assert isinstance(lora_exp, MoEExpertsLoRA)
        assert torch.allclose(lora_exp.gate_proj, base.gate_proj)


# ---------------------------------------------------------------------------
# LoRA weight shapes: sparse vs triton consistency
# ---------------------------------------------------------------------------


class TestSparseVsTritonConsistency:
    """Test consistency between sparse and triton LoRA implementations."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=4, hidden_size=32, moe_intermediate_size=64)

    @pytest.fixture
    def lora_config(self):
        return MoELoRAConfig(r=4, lora_alpha=8, target_modules=["gate_proj", "up_proj", "down_proj"])

    def test_lora_weight_shapes_match(self, config, lora_config):
        """Test that LoRA weight shapes match between eager and triton backends."""
        eager = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            moe_implementation="eager",
            lora_config=lora_config,
        )
        triton = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            moe_implementation="triton",
            lora_config=lora_config,
        )

        for name in lora_config.target_modules:
            eager_A = getattr(eager, f"{name}_lora_A")
            triton_A = getattr(triton, f"{name}_lora_A")
            eager_B = getattr(eager, f"{name}_lora_B")
            triton_B = getattr(triton, f"{name}_lora_B")

            assert eager_A.shape == triton_A.shape, f"{name}_lora_A shape mismatch"
            assert eager_B.shape == triton_B.shape, f"{name}_lora_B shape mismatch"


# ---------------------------------------------------------------------------
# LoRA injection tests
# ---------------------------------------------------------------------------


class TestLoRAInjection:
    """Test LoRA injection into models."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.mark.parametrize("backend", ["eager", "triton", "native", "quack"])
    def test_inject_lora_into_experts(self, config, backend):
        """Test inject_lora_into_model works with all backends."""
        from xorl.lora import inject_lora_into_model

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

    @pytest.mark.parametrize("backend", ["eager", "triton", "native", "quack"])
    def test_inject_lora_via_moe_block(self, config, backend):
        """Test MoEBlock.inject_lora() works with all backends."""
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
# Error handling
# ---------------------------------------------------------------------------


class TestLoRAInjectionErrors:
    """Test error handling in LoRA injection."""

    def test_error_no_matching_modules(self):
        """Test that error is raised when no modules match target_modules."""
        from xorl.lora import inject_lora_into_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(64, 64)

        model = SimpleModel()
        with pytest.raises(ValueError, match="No modules found matching target_modules"):
            inject_lora_into_model(
                model, r=8, lora_alpha=16,
                target_modules=["nonexistent_proj"],
            )

    def test_error_no_lora_support(self):
        """Test that error is raised when matched modules have no LoRA support."""
        from xorl.lora import inject_lora_into_model

        class UnsupportedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(64, 64))
            def forward(self, x):
                return x @ self.weight

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_layer = UnsupportedModule()

        model = SimpleModel()
        with pytest.raises(ValueError, match="No modules found matching target_modules"):
            inject_lora_into_model(
                model, r=8, lora_alpha=16,
                target_modules=["custom_layer"],
            )

    def test_success_with_valid_target(self):
        """Test that injection succeeds with valid targets."""
        from xorl.lora import inject_lora_into_model, LoraLinear

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        model = SimpleModel()
        inject_lora_into_model(
            model, r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        assert isinstance(model.q_proj, LoraLinear)
        assert isinstance(model.v_proj, LoraLinear)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    """Test MOE_EXPERT_BACKENDS registry."""

    def test_no_fused_in_registry(self):
        """Ensure 'fused' is not in the registry."""
        assert "fused" not in MOE_EXPERT_BACKENDS

    def test_expected_backends(self):
        """Check expected backends are registered."""
        assert "eager" in MOE_EXPERT_BACKENDS
        # triton/native/quack may not be available in all envs
        # but at least eager must always be present


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
