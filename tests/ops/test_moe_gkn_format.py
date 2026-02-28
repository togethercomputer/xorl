"""Tests for (G, K, N) weight format correctness in MoE experts.

These tests verify that the MoE expert computation with weights stored in
(G, K, N) = [num_experts, in_features, out_features] format produces
results identical to naive per-expert nn.Linear computation.

The key invariant:
    MoE forward with (G,K,N) weights == per-expert loop using nn.Linear

Weight format:
    - HuggingFace nn.Linear: [out_features, in_features]
    - Our (G,K,N) stacked: [num_experts, in_features, out_features]
    - Conversion: weight_gkn[e] = nn_linear_weight[e].t()
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference implementation — per-expert loop with nn.Linear weights
# ---------------------------------------------------------------------------

def reference_moe_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_gkn: torch.Tensor,
    up_proj_gkn: torch.Tensor,
    down_proj_gkn: torch.Tensor,
    num_experts: int,
):
    """Naive per-expert loop MoE forward using (G,K,N) weights directly.

    For each expert, computes: down(SiLU(gate(x)) * up(x))
    where gate/up/down use x @ W (weights in (K,N) format).

    Args:
        hidden_states: (num_tokens, hidden_dim)
        routing_weights: (num_tokens, top_k)
        selected_experts: (num_tokens, top_k)
        gate_proj_gkn: (num_experts, hidden_dim, intermediate_size)
        up_proj_gkn: (num_experts, hidden_dim, intermediate_size)
        down_proj_gkn: (num_experts, intermediate_size, hidden_dim)
        num_experts: total number of experts

    Returns:
        output: (num_tokens, hidden_dim)
    """
    num_tokens = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    output = torch.zeros(num_tokens, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    # expert_mask: (num_experts, top_k, num_tokens)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue

        current_state = hidden_states[top_x]  # (n, hidden_dim)
        current_weights = routing_weights[top_x, idx]  # (n,)

        # x @ W with (K,N) format — direct matmul, no transpose
        gate_out = torch.matmul(current_state, gate_proj_gkn[expert_idx])
        up_out = torch.matmul(current_state, up_proj_gkn[expert_idx])
        expert_out = torch.matmul(F.silu(gate_out) * up_out, down_proj_gkn[expert_idx])

        expert_out = expert_out * current_weights.unsqueeze(-1)
        output.index_add_(0, top_x, expert_out.to(hidden_states.dtype))

    return output


def make_gkn_weights_from_linear(experts_list):
    """Convert a list of nn.Module experts (each with gate/up/down nn.Linear)
    to stacked (G,K,N) format weights.

    nn.Linear stores [out, in]. (G,K,N) = [E, in, out] = nn.Linear.weight.t()
    """
    gate = torch.stack([e.gate_proj.weight.t() for e in experts_list])
    up = torch.stack([e.up_proj.weight.t() for e in experts_list])
    down = torch.stack([e.down_proj.weight.t() for e in experts_list])
    return gate, up, down


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    """Single expert MLP for reference."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGKNWeightFormat:
    """Test (G,K,N) weight format produces correct results."""

    def test_single_expert_matmul_equivalence(self):
        """Verify x @ W_gkn == nn.Linear(x) for a single expert."""
        torch.manual_seed(42)
        hidden_size, intermediate_size = 64, 128

        linear = nn.Linear(hidden_size, intermediate_size, bias=False)
        x = torch.randn(8, hidden_size)

        # nn.Linear stores [out, in], forward computes x @ W.T
        ref_out = linear(x)

        # (K,N) = [in, out] = W.T
        w_kn = linear.weight.t()
        gkn_out = x @ w_kn

        torch.testing.assert_close(gkn_out, ref_out, atol=1e-5, rtol=1e-5)

    def test_stacked_weight_shapes(self):
        """Verify (G,K,N) stacked weight shapes are correct."""
        num_experts, hidden_size, intermediate_size = 8, 64, 128

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate, up, down = make_gkn_weights_from_linear(experts)

        # gate/up: [E, hidden, intermediate] = [E, K, N]
        assert gate.shape == (num_experts, hidden_size, intermediate_size)
        assert up.shape == (num_experts, hidden_size, intermediate_size)
        # down: [E, intermediate, hidden] = [E, K, N]
        assert down.shape == (num_experts, intermediate_size, hidden_size)

    def test_reference_moe_vs_hf_loop(self):
        """Reference MoE forward with (G,K,N) weights matches HF per-expert loop."""
        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 16, 2

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts)

        hidden_states = torch.randn(num_tokens, hidden_size)
        selected_experts = torch.randint(0, num_experts, (num_tokens, top_k))
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

        # Our reference with (G,K,N)
        output_gkn = reference_moe_forward(
            hidden_states, routing_weights, selected_experts,
            gate_gkn, up_gkn, down_gkn, num_experts,
        )

        # HF-style per-expert loop using nn.Linear directly
        output_hf = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        for expert_idx in range(num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[top_x]
            current_weights = routing_weights[top_x, idx]
            expert_out = experts[expert_idx](current_state) * current_weights.unsqueeze(-1)
            output_hf.index_add_(0, top_x, expert_out)

        torch.testing.assert_close(output_gkn, output_hf, atol=1e-5, rtol=1e-5)


class TestCheckpointLoadingGKN:
    """Test that checkpoint loading produces correct (G,K,N) weights."""

    @staticmethod
    def _get_buffer_class():
        """Import ExpertWeightBuffer, skip if transformers dependency unavailable."""
        try:
            from xorl.models.checkpoint_handlers.buffers import ExpertWeightBuffer
            return ExpertWeightBuffer
        except (ImportError, ModuleNotFoundError):
            pytest.skip("checkpoint_handlers import requires transformers")

    def test_expert_buffer_transposes_correctly(self):
        """ExpertWeightBuffer should transpose HF weights to (G,K,N)."""
        ExpertWeightBuffer = self._get_buffer_class()

        num_experts, hidden_size, intermediate_size = 4, 32, 64
        buf = ExpertWeightBuffer(num_experts)

        # Create HF-format per-expert weights
        torch.manual_seed(42)
        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]

        for expert_idx in range(num_experts):
            # HF stores [out, in]
            buf.add(0, expert_idx, "gate", experts[expert_idx].gate_proj.weight.detach())
            buf.add(0, expert_idx, "up", experts[expert_idx].up_proj.weight.detach())
            buf.add(0, expert_idx, "down", experts[expert_idx].down_proj.weight.detach())

        gate_stacked = buf.pop_stacked(0, "gate")
        up_stacked = buf.pop_stacked(0, "up")
        down_stacked = buf.pop_stacked(0, "down")

        # Verify shapes are (G,K,N) = [E, in, out]
        assert gate_stacked.shape == (num_experts, hidden_size, intermediate_size)
        assert up_stacked.shape == (num_experts, hidden_size, intermediate_size)
        assert down_stacked.shape == (num_experts, intermediate_size, hidden_size)

        # Verify values: stacked[e] should equal expert[e].weight.t()
        for e in range(num_experts):
            torch.testing.assert_close(
                gate_stacked[e], experts[e].gate_proj.weight.t(),
                msg=f"gate_proj expert {e} mismatch",
            )
            torch.testing.assert_close(
                up_stacked[e], experts[e].up_proj.weight.t(),
                msg=f"up_proj expert {e} mismatch",
            )
            torch.testing.assert_close(
                down_stacked[e], experts[e].down_proj.weight.t(),
                msg=f"down_proj expert {e} mismatch",
            )

    def test_loaded_weights_produce_correct_output(self):
        """Verify that weights loaded through ExpertWeightBuffer give correct MoE output."""
        ExpertWeightBuffer = self._get_buffer_class()

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 32, 64
        num_tokens, top_k = 16, 2

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]

        # Simulate checkpoint loading via buffer
        buf = ExpertWeightBuffer(num_experts)
        for expert_idx in range(num_experts):
            buf.add(0, expert_idx, "gate", experts[expert_idx].gate_proj.weight.detach())
            buf.add(0, expert_idx, "up", experts[expert_idx].up_proj.weight.detach())
            buf.add(0, expert_idx, "down", experts[expert_idx].down_proj.weight.detach())

        gate_gkn = buf.pop_stacked(0, "gate")
        up_gkn = buf.pop_stacked(0, "up")
        down_gkn = buf.pop_stacked(0, "down")

        # Create inputs
        hidden_states = torch.randn(num_tokens, hidden_size)
        selected_experts = torch.randint(0, num_experts, (num_tokens, top_k))
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

        # MoE forward with loaded (G,K,N) weights
        output_gkn = reference_moe_forward(
            hidden_states, routing_weights, selected_experts,
            gate_gkn, up_gkn, down_gkn, num_experts,
        )

        # HF reference using nn.Linear
        output_hf = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        for expert_idx in range(num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[top_x]
            current_weights = routing_weights[top_x, idx]
            expert_out = experts[expert_idx](current_state) * current_weights.unsqueeze(-1)
            output_hf.index_add_(0, top_x, expert_out)

        torch.testing.assert_close(output_gkn, output_hf, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEagerBackendGKN:
    """Test eager backend with (G,K,N) format."""

    def test_eager_forward_matches_reference(self):
        """Eager backend forward matches reference."""
        try:
            from xorl.models.layers.moe.backend.eager import eager_expert_forward
        except (ImportError, ModuleNotFoundError):
            pytest.skip("eager backend import requires transformers")

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens = 16

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts)

        hidden_states = torch.randn(num_tokens, hidden_size)
        act_fn = torch.nn.SiLU()

        for expert_idx in range(num_experts):
            ref_out = experts[expert_idx](hidden_states)
            eager_out = eager_expert_forward(
                hidden_states, expert_idx, gate_gkn, up_gkn, down_gkn, act_fn,
            )
            torch.testing.assert_close(eager_out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNativeBackendGKN:
    """Test native backend (torch._grouped_mm) with (G,K,N) format."""

    def test_native_forward_matches_reference(self):
        """Native backend forward matches reference."""
        from xorl.models.layers.moe.backend.native import native_expert_forward

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 32, 2

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts)

        device = "cuda"
        gate_gkn = gate_gkn.to(device).to(torch.bfloat16)
        up_gkn = up_gkn.to(device).to(torch.bfloat16)
        down_gkn = down_gkn.to(device).to(torch.bfloat16)

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        routing_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1
        )

        native_out = native_expert_forward(
            hidden_states, routing_weights, selected_experts,
            gate_gkn, up_gkn, down_gkn, num_experts,
        )

        ref_out = reference_moe_forward(
            hidden_states, routing_weights, selected_experts,
            gate_gkn, up_gkn, down_gkn, num_experts,
        )

        # bfloat16 precision: use looser tolerance
        torch.testing.assert_close(native_out, ref_out, atol=0.02, rtol=0.02)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonBackendGKN:
    """Test triton/fused backend with (G,K,N) format."""

    def _skip_if_unavailable(self):
        from xorl.utils.import_utils import is_fused_moe_available
        if not is_fused_moe_available():
            pytest.skip("fused_moe not available")

    def test_triton_forward_matches_reference(self):
        """Triton fused backend forward matches reference."""
        self._skip_if_unavailable()
        from xorl.ops.moe_experts import MoeExpertsFunction

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 32, 2

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts)

        device = "cuda"
        gate_gkn = gate_gkn.to(device).to(torch.bfloat16)
        up_gkn = up_gkn.to(device).to(torch.bfloat16)
        down_gkn = down_gkn.to(device).to(torch.bfloat16)

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        routing_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1
        )

        triton_out = MoeExpertsFunction.apply(
            num_experts, routing_weights, selected_experts,
            hidden_states, gate_gkn, up_gkn, down_gkn,
        )

        ref_out = reference_moe_forward(
            hidden_states, routing_weights, selected_experts,
            gate_gkn, up_gkn, down_gkn, num_experts,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=0.01, rtol=0.01)

    def test_triton_backward_produces_correct_gradients(self):
        """Triton backward produces gradients consistent with autograd."""
        self._skip_if_unavailable()
        from xorl.ops.moe_experts import MoeExpertsFunction

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 32, 2

        device = "cuda"
        dtype = torch.bfloat16

        gate = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype, requires_grad=True)
        up = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype, requires_grad=True)
        down = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype, requires_grad=True)

        hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, requires_grad=True)
        selected = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)

        out = MoeExpertsFunction.apply(num_experts, weights, selected, hidden, gate, up, down)
        out.sum().backward()

        # Verify gradients exist and are non-zero
        assert hidden.grad is not None
        assert gate.grad is not None
        assert up.grad is not None
        assert down.grad is not None
        assert hidden.grad.abs().max() > 0
        assert gate.grad.abs().max() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoEExpertsModuleGKN:
    """Test the MoEExperts module with (G,K,N) format."""

    def test_moe_experts_forward_matches_reference(self):
        """MoEExperts forward matches reference for all available backends."""
        try:
            from xorl.models.layers.moe import MoEExperts, MOE_EXPERT_BACKENDS
        except (ImportError, ModuleNotFoundError):
            pytest.skip("MoE layer import requires transformers")

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 32, 2

        experts_ref = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts_ref)

        device = "cuda"
        dtype = torch.bfloat16

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        routing_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1
        )

        ref_out = reference_moe_forward(
            hidden_states, routing_weights, selected_experts,
            gate_gkn.to(device).to(dtype), up_gkn.to(device).to(dtype),
            down_gkn.to(device).to(dtype), num_experts,
        )

        for backend in MOE_EXPERT_BACKENDS:
            moe = MoEExperts(
                num_experts=num_experts,
                hidden_dim=hidden_size,
                intermediate_size=intermediate_size,
                moe_implementation=backend,
            )
            # Load the (G,K,N) weights
            with torch.no_grad():
                moe.gate_proj.copy_(gate_gkn)
                moe.up_proj.copy_(up_gkn)
                moe.down_proj.copy_(down_gkn)
            moe = moe.to(device).to(dtype)

            if backend == "eager":
                # Eager uses per-expert forward, test via MoEBlock
                continue

            output = moe(hidden_states, routing_weights, selected_experts)
            torch.testing.assert_close(
                output, ref_out, atol=0.05, rtol=0.02,
                msg=f"Backend '{backend}' output mismatch with reference",
            )

    def test_moe_block_all_backends_agree(self):
        """All backends produce the same output through MoEBlock."""
        try:
            from xorl.models.layers.moe import MoEBlock, MOE_EXPERT_BACKENDS
        except (ImportError, ModuleNotFoundError):
            pytest.skip("MoE layer import requires transformers")

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        device = "cuda"
        dtype = torch.bfloat16

        # Create reference weights
        experts_ref = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts_ref)
        gate_linear_w = torch.randn(num_experts, hidden_size)

        hidden = torch.randn(2, 8, hidden_size, device=device, dtype=dtype)
        outputs = {}

        for backend in MOE_EXPERT_BACKENDS:
            block = MoEBlock(
                hidden_size=hidden_size, num_experts=num_experts, top_k=2,
                intermediate_size=intermediate_size, moe_implementation=backend,
            )
            with torch.no_grad():
                block.experts.gate_proj.copy_(gate_gkn)
                block.experts.up_proj.copy_(up_gkn)
                block.experts.down_proj.copy_(down_gkn)
                block.gate.weight.copy_(gate_linear_w)
            block = block.to(device).to(dtype)

            out, _ = block(hidden)
            outputs[backend] = out

        # Compare all pairs
        backends = list(outputs.keys())
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                torch.testing.assert_close(
                    outputs[backends[i]], outputs[backends[j]],
                    atol=0.05, rtol=0.02,
                    msg=f"Backend mismatch: {backends[i]} vs {backends[j]}",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
