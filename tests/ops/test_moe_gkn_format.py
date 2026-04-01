"""Tests for (G, K, N) weight format correctness in MoE experts.

These tests verify that the MoE expert computation with weights stored in
(G, K, N) = [num_experts, in_features, out_features] format produces
results identical to naive per-expert nn.Linear computation.

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
# Reference implementation -- per-expert loop with nn.Linear weights
# ---------------------------------------------------------------------------


def reference_moe_forward(
    hidden_states,
    routing_weights,
    selected_experts,
    gate_proj_gkn,
    up_proj_gkn,
    down_proj_gkn,
    num_experts,
):
    """Naive per-expert loop MoE forward using (G,K,N) weights directly."""
    num_tokens = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    output = torch.zeros(num_tokens, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue
        current_state = hidden_states[top_x]
        current_weights = routing_weights[top_x, idx]
        gate_out = torch.matmul(current_state, gate_proj_gkn[expert_idx])
        up_out = torch.matmul(current_state, up_proj_gkn[expert_idx])
        expert_out = torch.matmul(F.silu(gate_out) * up_out, down_proj_gkn[expert_idx])
        expert_out = expert_out * current_weights.unsqueeze(-1)
        output.index_add_(0, top_x, expert_out.to(hidden_states.dtype))
    return output


def make_gkn_weights_from_linear(experts_list):
    """Convert list of nn.Module experts to stacked (G,K,N) format."""
    gate = torch.stack([e.gate_proj.weight.t() for e in experts_list])
    up = torch.stack([e.up_proj.weight.t() for e in experts_list])
    down = torch.stack([e.down_proj.weight.t() for e in experts_list])
    return gate, up, down


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
    """Test (G,K,N) weight format: matmul equivalence, shapes, and reference MoE vs HF loop."""

    def test_gkn_format_correctness(self):
        """Single expert matmul equivalence, stacked shapes, and reference vs HF loop."""
        # --- Single expert matmul equivalence ---
        torch.manual_seed(42)
        hidden_size, intermediate_size = 64, 128
        linear = nn.Linear(hidden_size, intermediate_size, bias=False)
        x = torch.randn(8, hidden_size)
        ref_out = linear(x)
        w_kn = linear.weight.t()
        gkn_out = x @ w_kn
        torch.testing.assert_close(gkn_out, ref_out, atol=1e-5, rtol=1e-5)

        # --- Stacked weight shapes ---
        num_experts = 8
        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate, up, down = make_gkn_weights_from_linear(experts)
        assert gate.shape == (num_experts, hidden_size, intermediate_size)
        assert up.shape == (num_experts, hidden_size, intermediate_size)
        assert down.shape == (num_experts, intermediate_size, hidden_size)

        # --- Reference MoE vs HF loop ---
        torch.manual_seed(42)
        num_experts2, hidden_size2, intermediate_size2 = 4, 64, 128
        num_tokens, top_k = 16, 2
        experts2 = [ExpertMLP(hidden_size2, intermediate_size2) for _ in range(num_experts2)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts2)
        hidden_states = torch.randn(num_tokens, hidden_size2)
        selected_experts = torch.randint(0, num_experts2, (num_tokens, top_k))
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

        output_gkn = reference_moe_forward(
            hidden_states,
            routing_weights,
            selected_experts,
            gate_gkn,
            up_gkn,
            down_gkn,
            num_experts2,
        )

        output_hf = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts2).permute(2, 1, 0)
        for expert_idx in range(num_experts2):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[top_x]
            current_weights = routing_weights[top_x, idx]
            expert_out = experts2[expert_idx](current_state) * current_weights.unsqueeze(-1)
            output_hf.index_add_(0, top_x, expert_out)
        torch.testing.assert_close(output_gkn, output_hf, atol=1e-5, rtol=1e-5)


class TestCheckpointLoadingGKN:
    """Test checkpoint loading produces correct (G,K,N) weights and output."""

    @staticmethod
    def _get_buffer_class():
        try:
            from xorl.models.checkpoint_handlers.buffers import ExpertWeightBuffer

            return ExpertWeightBuffer
        except (ImportError, ModuleNotFoundError):
            pytest.skip("checkpoint_handlers import requires transformers")

    def test_checkpoint_loading_and_output(self):
        """ExpertWeightBuffer transposes correctly and loaded weights produce correct output."""
        ExpertWeightBuffer = self._get_buffer_class()

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 32, 64
        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]

        buf = ExpertWeightBuffer(num_experts)
        for ei in range(num_experts):
            buf.add(0, ei, "gate", experts[ei].gate_proj.weight.detach())
            buf.add(0, ei, "up", experts[ei].up_proj.weight.detach())
            buf.add(0, ei, "down", experts[ei].down_proj.weight.detach())

        gate_stacked = buf.pop_stacked(0, "gate")
        up_stacked = buf.pop_stacked(0, "up")
        down_stacked = buf.pop_stacked(0, "down")

        # Shapes and values
        assert gate_stacked.shape == (num_experts, hidden_size, intermediate_size)
        assert up_stacked.shape == (num_experts, hidden_size, intermediate_size)
        assert down_stacked.shape == (num_experts, intermediate_size, hidden_size)

        for e in range(num_experts):
            torch.testing.assert_close(gate_stacked[e], experts[e].gate_proj.weight.t())
            torch.testing.assert_close(up_stacked[e], experts[e].up_proj.weight.t())
            torch.testing.assert_close(down_stacked[e], experts[e].down_proj.weight.t())

        # Loaded weights produce correct output
        num_tokens, top_k = 16, 2
        hidden_states = torch.randn(num_tokens, hidden_size)
        selected_experts = torch.randint(0, num_experts, (num_tokens, top_k))
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

        output_gkn = reference_moe_forward(
            hidden_states,
            routing_weights,
            selected_experts,
            gate_stacked,
            up_stacked,
            down_stacked,
            num_experts,
        )

        output_hf = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        for ei in range(num_experts):
            idx, top_x = torch.where(expert_mask[ei])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[top_x]
            cw = routing_weights[top_x, idx]
            expert_out = experts[ei](current_state) * cw.unsqueeze(-1)
            output_hf.index_add_(0, top_x, expert_out)
        torch.testing.assert_close(output_gkn, output_hf, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBackendGKN:
    """Test eager, native, and triton backends with (G,K,N) format, plus MoEBlock cross-backend agreement."""

    def test_backends_match_reference_and_agree(self):
        """Eager, native, triton backends match reference; all MoEBlock backends agree."""
        # --- Eager backend ---
        try:
            from xorl.models.layers.moe.backend.eager import eager_expert_forward
        except (ImportError, ModuleNotFoundError):
            pytest.skip("eager backend import requires transformers")

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 32, 2

        experts = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        gate_gkn, up_gkn, down_gkn = make_gkn_weights_from_linear(experts)

        hidden_states_cpu = torch.randn(num_tokens, hidden_size)
        act_fn = torch.nn.SiLU()

        for expert_idx in range(num_experts):
            ref_out = experts[expert_idx](hidden_states_cpu)
            eager_out = eager_expert_forward(hidden_states_cpu, expert_idx, gate_gkn, up_gkn, down_gkn, act_fn)
            torch.testing.assert_close(eager_out, ref_out, atol=1e-5, rtol=1e-5)

        # --- Native backend ---
        from xorl.models.layers.moe.backend.native import native_expert_forward

        device, dtype = "cuda", torch.bfloat16
        gate_cuda = gate_gkn.to(device).to(dtype)
        up_cuda = up_gkn.to(device).to(dtype)
        down_cuda = down_gkn.to(device).to(dtype)
        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        selected = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        rw = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)

        native_out = native_expert_forward(
            hidden_states,
            rw,
            selected,
            gate_cuda,
            up_cuda,
            down_cuda,
            num_experts,
        )
        ref_out = reference_moe_forward(hidden_states, rw, selected, gate_cuda, up_cuda, down_cuda, num_experts)
        torch.testing.assert_close(native_out, ref_out, atol=0.02, rtol=0.02)

        # --- Triton backend ---
        try:
            from xorl.utils.import_utils import is_fused_moe_available

            if not is_fused_moe_available():
                raise ImportError
            from xorl.ops.moe.triton import TritonMoeExpertsFunction

            triton_out = TritonMoeExpertsFunction.apply(
                num_experts,
                rw,
                selected,
                hidden_states,
                gate_cuda,
                up_cuda,
                down_cuda,
            )
            torch.testing.assert_close(triton_out, ref_out, atol=0.01, rtol=0.01)

            # Triton backward: gradients exist and non-zero
            gate_g = torch.randn(
                num_experts, hidden_size, intermediate_size, device=device, dtype=dtype, requires_grad=True
            )
            up_g = torch.randn(
                num_experts, hidden_size, intermediate_size, device=device, dtype=dtype, requires_grad=True
            )
            down_g = torch.randn(
                num_experts, intermediate_size, hidden_size, device=device, dtype=dtype, requires_grad=True
            )
            h_g = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, requires_grad=True)
            out_g = TritonMoeExpertsFunction.apply(num_experts, rw, selected, h_g, gate_g, up_g, down_g)
            out_g.sum().backward()
            assert h_g.grad is not None and h_g.grad.abs().max() > 0
            assert gate_g.grad is not None and gate_g.grad.abs().max() > 0
        except (ImportError, ModuleNotFoundError):
            pass  # triton not available, skip

        # --- MoEBlock all backends agree ---
        try:
            from xorl.models.layers.moe import MOE_EXPERT_BACKENDS, MoEBlock
        except (ImportError, ModuleNotFoundError):
            return  # skip if not importable

        experts_ref = [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        g_gkn, u_gkn, d_gkn = make_gkn_weights_from_linear(experts_ref)
        gate_w = torch.randn(num_experts, hidden_size)
        hidden_block = torch.randn(2, 8, hidden_size, device=device, dtype=dtype)
        outputs = {}

        for backend in MOE_EXPERT_BACKENDS:
            block = MoEBlock(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=2,
                intermediate_size=intermediate_size,
                moe_implementation=backend,
            )
            with torch.no_grad():
                block.experts.gate_proj.copy_(g_gkn)
                block.experts.up_proj.copy_(u_gkn)
                block.experts.down_proj.copy_(d_gkn)
                block.gate.weight.copy_(gate_w)
            block = block.to(device).to(dtype)
            out, _ = block(hidden_block)
            outputs[backend] = out

        backends = list(outputs.keys())
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                torch.testing.assert_close(
                    outputs[backends[i]],
                    outputs[backends[j]],
                    atol=0.05,
                    rtol=0.02,
                    msg=f"Backend mismatch: {backends[i]} vs {backends[j]}",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
