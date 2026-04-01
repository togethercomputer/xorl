"""MoE experts with LoRA forward/backward using group GEMM kernels.

Two factory functions (``make_ep_lora_compute`` and ``make_local_lora_compute``)
create backend-parameterised ``torch.autograd.Function`` classes.  Triton and
quack GEMM kernels have identical call signatures, so the factory eliminates
duplication — only the kernel function references differ.

Weights are stored in (G, K, N) format — [num_experts, in_features, out_features].

The key advantage over weight-merging approaches:
- Memory efficient: No need to materialise full merged weight tensors
- Proper gradients: Backward pass correctly computes gradients only for LoRA weights
- Base weights are frozen: No gradients computed for base weights
"""

import torch
import torch.distributed as dist

from xorl.ops.group_gemm.kernel.moe import expert_histogram, moe_add_gather, moe_gather, moe_scatter


# ============================================================================
# Factory: EP LoRA compute (Expert Parallelism)
# ============================================================================


def make_ep_lora_compute(gemm_nk, gemm_mn):
    """Create an EP LoRA compute ``autograd.Function`` parameterised by GEMM kernels.

    Args:
        gemm_nk: Group GEMM with same-N,K signature (``group_gemm_same_nk`` API).
        gemm_mn: Group GEMM with same-M,N signature (``group_gemm_same_mn`` API).

    Returns:
        A ``torch.autograd.Function`` subclass for EP LoRA compute.
    """

    class _EPGroupGemmWithLoRA(torch.autograd.Function):
        """Expert MLP computation with LoRA for Expert Parallelism.

        Pure compute kernel — no dispatch/combine logic.  Tokens have already
        been dispatched; this handles the expert SwiGLU MLP with LoRA:
            output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
        where each FC includes LoRA: FC(x) = x @ W + (x @ A) @ B * scaling
        """

        @staticmethod
        def forward(
            ctx,
            permute_tokens,
            cumsum,
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_lora_A,
            gate_proj_lora_B,
            up_proj_lora_A,
            up_proj_lora_B,
            down_proj_lora_A,
            down_proj_lora_B,
            scaling,
        ):
            max_M = permute_tokens.shape[0]
            num_local_experts = gate_proj.shape[0]
            intermediate_size = gate_proj.shape[2]
            lora_r = gate_proj_lora_A.shape[2]

            # Detect shared weights (hybrid mode): shape[0] == 1 means shared
            gate_A_shared = gate_proj_lora_A.shape[0] == 1
            up_A_shared = up_proj_lora_A.shape[0] == 1
            down_B_shared = down_proj_lora_B.shape[0] == 1

            # Save original LoRA weights for backward
            orig_gate_proj_lora_A = gate_proj_lora_A
            orig_gate_proj_lora_B = gate_proj_lora_B
            orig_up_proj_lora_A = up_proj_lora_A
            orig_up_proj_lora_B = up_proj_lora_B
            orig_down_proj_lora_A = down_proj_lora_A
            orig_down_proj_lora_B = down_proj_lora_B

            # Cast LoRA weights to match base weights dtype
            compute_dtype = gate_proj.dtype
            if gate_proj_lora_A.dtype != compute_dtype:
                gate_proj_lora_A = gate_proj_lora_A.to(compute_dtype)
                gate_proj_lora_B = gate_proj_lora_B.to(compute_dtype)
                up_proj_lora_A = up_proj_lora_A.to(compute_dtype)
                up_proj_lora_B = up_proj_lora_B.to(compute_dtype)
                down_proj_lora_A = down_proj_lora_A.to(compute_dtype)
                down_proj_lora_B = down_proj_lora_B.to(compute_dtype)

            # Expand shared weights to num_local_experts for group GEMM
            if gate_A_shared:
                gate_proj_lora_A = gate_proj_lora_A.expand(num_local_experts, -1, -1).contiguous()
            if up_A_shared:
                up_proj_lora_A = up_proj_lora_A.expand(num_local_experts, -1, -1).contiguous()
            if down_B_shared:
                down_proj_lora_B = down_proj_lora_B.expand(num_local_experts, -1, -1).contiguous()

            # === gate_proj and up_proj with FUSED base GEMM ===
            gate_up_weight = torch.cat([gate_proj, up_proj], dim=2)
            gate_up_output = gemm_nk(
                a=permute_tokens,
                b=gate_up_weight,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            gate_base = gate_up_output[:, :intermediate_size]
            up_base = gate_up_output[:, intermediate_size:]

            # === gate and up LoRA A: Fused GEMM ===
            gate_up_lora_A = torch.cat([gate_proj_lora_A, up_proj_lora_A], dim=2)
            gate_up_lora_intermediate = gemm_nk(
                a=permute_tokens,
                b=gate_up_lora_A,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            gate_lora_intermediate = gate_up_lora_intermediate[:, :lora_r].contiguous()
            up_lora_intermediate = gate_up_lora_intermediate[:, lora_r:].contiguous()

            # === gate LoRA B ===
            gate_lora_output = gemm_nk(
                a=gate_lora_intermediate,
                b=gate_proj_lora_B,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            gate_output = (gate_base + gate_lora_output * scaling).contiguous()

            # === up LoRA B ===
            up_lora_output = gemm_nk(
                a=up_lora_intermediate,
                b=up_proj_lora_B,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            up_output = (up_base + up_lora_output * scaling).contiguous()

            # === Activation ===
            gate_activation = torch.ops.aten.silu(gate_output.float()).to(gate_output.dtype)
            gated_output = gate_activation * up_output

            # === down_proj with LoRA ===
            down_output = gemm_nk(
                a=gated_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            down_lora_intermediate = gemm_nk(
                a=gated_output,
                b=down_proj_lora_A,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            down_lora_output = gemm_nk(
                a=down_lora_intermediate,
                b=down_proj_lora_B,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            down_output = down_output + down_lora_output * scaling

            # Save for backward
            ctx.scaling = scaling
            ctx.gate_A_shared = gate_A_shared
            ctx.up_A_shared = up_A_shared
            ctx.down_B_shared = down_B_shared
            ctx.num_local_experts = num_local_experts
            ctx.compute_dtype = compute_dtype
            ctx.save_for_backward(
                permute_tokens,
                cumsum,
                gate_proj,
                up_proj,
                down_proj,
                orig_gate_proj_lora_A,
                orig_gate_proj_lora_B,
                orig_up_proj_lora_A,
                orig_up_proj_lora_B,
                orig_down_proj_lora_A,
                orig_down_proj_lora_B,
                gate_output,
                up_output,
                gated_output,
                gate_lora_intermediate,
                up_lora_intermediate,
                down_lora_intermediate,
            )
            return down_output

        @staticmethod
        def backward(ctx, grad_output):
            (
                permute_tokens,
                cumsum,
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_lora_A,
                gate_proj_lora_B,
                up_proj_lora_A,
                up_proj_lora_B,
                down_proj_lora_A,
                down_proj_lora_B,
                gate_output,
                up_output,
                gated_output,
                gate_lora_intermediate,
                up_lora_intermediate,
                down_lora_intermediate,
            ) = ctx.saved_tensors

            scaling = ctx.scaling
            gate_A_shared = ctx.gate_A_shared
            up_A_shared = ctx.up_A_shared
            down_B_shared = ctx.down_B_shared
            num_local_experts = ctx.num_local_experts
            compute_dtype = ctx.compute_dtype
            grad_output = grad_output.contiguous()
            max_M = grad_output.shape[0]

            # Cast LoRA weights to compute dtype for backward
            gate_proj_lora_A_compute = (
                gate_proj_lora_A.to(compute_dtype) if gate_proj_lora_A.dtype != compute_dtype else gate_proj_lora_A
            )
            gate_proj_lora_B_compute = (
                gate_proj_lora_B.to(compute_dtype) if gate_proj_lora_B.dtype != compute_dtype else gate_proj_lora_B
            )
            up_proj_lora_A_compute = (
                up_proj_lora_A.to(compute_dtype) if up_proj_lora_A.dtype != compute_dtype else up_proj_lora_A
            )
            up_proj_lora_B_compute = (
                up_proj_lora_B.to(compute_dtype) if up_proj_lora_B.dtype != compute_dtype else up_proj_lora_B
            )
            down_proj_lora_A_compute = (
                down_proj_lora_A.to(compute_dtype) if down_proj_lora_A.dtype != compute_dtype else down_proj_lora_A
            )
            down_proj_lora_B_compute = (
                down_proj_lora_B.to(compute_dtype) if down_proj_lora_B.dtype != compute_dtype else down_proj_lora_B
            )

            if gate_A_shared:
                gate_proj_lora_A_compute = gate_proj_lora_A_compute.expand(num_local_experts, -1, -1).contiguous()
            if up_A_shared:
                up_proj_lora_A_compute = up_proj_lora_A_compute.expand(num_local_experts, -1, -1).contiguous()
            if down_B_shared:
                down_proj_lora_B_compute = down_proj_lora_B_compute.expand(num_local_experts, -1, -1).contiguous()

            # Initialize LoRA gradients
            grad_gate_proj_lora_A_full = torch.zeros(
                num_local_experts,
                gate_proj_lora_A.shape[1],
                gate_proj_lora_A.shape[2],
                dtype=gate_proj_lora_A.dtype,
                device=gate_proj_lora_A.device,
            )
            grad_gate_proj_lora_B = torch.zeros_like(gate_proj_lora_B)
            grad_up_proj_lora_A_full = torch.zeros(
                num_local_experts,
                up_proj_lora_A.shape[1],
                up_proj_lora_A.shape[2],
                dtype=up_proj_lora_A.dtype,
                device=up_proj_lora_A.device,
            )
            grad_up_proj_lora_B = torch.zeros_like(up_proj_lora_B)
            grad_down_proj_lora_A = torch.zeros_like(down_proj_lora_A)
            grad_down_proj_lora_B_full = torch.zeros(
                num_local_experts,
                down_proj_lora_B.shape[1],
                down_proj_lora_B.shape[2],
                dtype=down_proj_lora_B.dtype,
                device=down_proj_lora_B.device,
            )

            # === down_proj backward ===
            grad_gated_output = gemm_nk(
                a=grad_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

            gemm_mn(
                a=down_lora_intermediate,
                b=grad_output,
                c=grad_down_proj_lora_B_full,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
            grad_down_proj_lora_B_full.mul_(scaling)

            grad_down_lora_intermediate = gemm_nk(
                a=grad_output,
                b=down_proj_lora_B_compute,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )
            grad_down_lora_intermediate.mul_(scaling)

            gemm_mn(
                a=gated_output,
                b=grad_down_lora_intermediate,
                c=grad_down_proj_lora_A,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )

            grad_gated_from_lora = gemm_nk(
                a=grad_down_lora_intermediate,
                b=down_proj_lora_A_compute,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )
            grad_gated_output = grad_gated_output + grad_gated_from_lora

            # === Activation backward ===
            gate_activation = torch.ops.aten.silu(gate_output.float()).to(gate_output.dtype)
            grad_up_output = (gate_activation * grad_gated_output).contiguous()
            grad_gate_activation = (grad_gated_output * up_output).contiguous()

            # === up_proj backward ===
            grad_permute_tokens_2 = gemm_nk(
                a=grad_up_output,
                b=up_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

            gemm_mn(
                a=up_lora_intermediate,
                b=grad_up_output,
                c=grad_up_proj_lora_B,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
            grad_up_proj_lora_B.mul_(scaling)

            grad_up_lora_intermediate = gemm_nk(
                a=grad_up_output,
                b=up_proj_lora_B_compute,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )
            grad_up_lora_intermediate.mul_(scaling)

            gemm_mn(
                a=permute_tokens,
                b=grad_up_lora_intermediate,
                c=grad_up_proj_lora_A_full,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )

            grad_permute_from_lora_2 = gemm_nk(
                a=grad_up_lora_intermediate,
                b=up_proj_lora_A_compute,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )
            grad_permute_tokens_2 = grad_permute_tokens_2 + grad_permute_from_lora_2

            # === SiLU backward ===
            grad_gate_output = (
                torch.ops.aten.silu_backward(grad_gate_activation.float(), gate_output.float())
                .to(gate_output.dtype)
                .contiguous()
            )

            # === gate_proj backward ===
            grad_permute_tokens_1 = gemm_nk(
                a=grad_gate_output,
                b=gate_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

            gemm_mn(
                a=gate_lora_intermediate,
                b=grad_gate_output,
                c=grad_gate_proj_lora_B,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
            grad_gate_proj_lora_B.mul_(scaling)

            grad_gate_lora_intermediate = gemm_nk(
                a=grad_gate_output,
                b=gate_proj_lora_B_compute,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )
            grad_gate_lora_intermediate.mul_(scaling)

            gemm_mn(
                a=permute_tokens,
                b=grad_gate_lora_intermediate,
                c=grad_gate_proj_lora_A_full,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )

            grad_permute_from_lora_1 = gemm_nk(
                a=grad_gate_lora_intermediate,
                b=gate_proj_lora_A_compute,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )
            grad_permute_tokens_1 = grad_permute_tokens_1 + grad_permute_from_lora_1

            grad_permute_tokens = grad_permute_tokens_1 + grad_permute_tokens_2

            # === Reduce gradients for shared weights ===
            from xorl.distributed.parallel_state import get_parallel_state

            ep_group = get_parallel_state().ep_group

            if gate_A_shared:
                grad_gate_proj_lora_A = grad_gate_proj_lora_A_full.sum(dim=0, keepdim=True)
                dist.all_reduce(grad_gate_proj_lora_A, group=ep_group)
            else:
                grad_gate_proj_lora_A = grad_gate_proj_lora_A_full

            if up_A_shared:
                grad_up_proj_lora_A = grad_up_proj_lora_A_full.sum(dim=0, keepdim=True)
                dist.all_reduce(grad_up_proj_lora_A, group=ep_group)
            else:
                grad_up_proj_lora_A = grad_up_proj_lora_A_full

            if down_B_shared:
                grad_down_proj_lora_B = grad_down_proj_lora_B_full.sum(dim=0, keepdim=True)
                dist.all_reduce(grad_down_proj_lora_B, group=ep_group)
            else:
                grad_down_proj_lora_B = grad_down_proj_lora_B_full

            return (
                grad_permute_tokens,
                None,  # cumsum
                None,
                None,
                None,  # base weights (frozen)
                grad_gate_proj_lora_A,
                grad_gate_proj_lora_B,
                grad_up_proj_lora_A,
                grad_up_proj_lora_B,
                grad_down_proj_lora_A,
                grad_down_proj_lora_B,
                None,  # scaling
            )

    return _EPGroupGemmWithLoRA


# ============================================================================
# Factory: Local LoRA compute (single-GPU)
# ============================================================================


def make_local_lora_compute(gemm_nk, gemm_mn):
    """Create a local LoRA compute ``autograd.Function`` parameterised by GEMM kernels.

    The returned class handles token dispatch (histogram, scatter, gather)
    internally.  Scatter/gather ops are shared across all backends.

    Args:
        gemm_nk: Group GEMM with same-N,K signature.
        gemm_mn: Group GEMM with same-M,N signature.

    Returns:
        A ``torch.autograd.Function`` subclass for local LoRA compute.
    """

    class _MoeExpertsLoRAFunction(torch.autograd.Function):
        """MoE expert computation with LoRA using custom autograd for efficient backward pass.

        Forward computation for each projection:
            y = x @ W + (x @ A) @ B * scaling

        Where (all in G, K, N format):
            - W: base weight [num_experts, in_features, out_features]
            - A: LoRA A weight [num_experts, in_features, r]
            - B: LoRA B weight [num_experts, r, out_features]
        """

        @staticmethod
        def forward(
            ctx,
            num_experts: int,
            gate_weights: torch.Tensor,
            expert_index: torch.Tensor,
            hidden_states: torch.Tensor,
            # Base weights (G, K, N)
            gate_proj: torch.Tensor,
            up_proj: torch.Tensor,
            down_proj: torch.Tensor,
            # LoRA weights (G, K, N)
            gate_proj_lora_A: torch.Tensor,
            gate_proj_lora_B: torch.Tensor,
            up_proj_lora_A: torch.Tensor,
            up_proj_lora_B: torch.Tensor,
            down_proj_lora_A: torch.Tensor,
            down_proj_lora_B: torch.Tensor,
            # LoRA config
            scaling: float,
        ):
            # Save original LoRA weights for backward (need original dtype for gradients)
            orig_gate_proj_lora_A = gate_proj_lora_A
            orig_gate_proj_lora_B = gate_proj_lora_B
            orig_up_proj_lora_A = up_proj_lora_A
            orig_up_proj_lora_B = up_proj_lora_B
            orig_down_proj_lora_A = down_proj_lora_A
            orig_down_proj_lora_B = down_proj_lora_B

            # Cast LoRA weights to compute dtype
            compute_dtype = gate_proj.dtype
            if gate_proj_lora_A.dtype != compute_dtype:
                gate_proj_lora_A = gate_proj_lora_A.to(compute_dtype)
                gate_proj_lora_B = gate_proj_lora_B.to(compute_dtype)
                up_proj_lora_A = up_proj_lora_A.to(compute_dtype)
                up_proj_lora_B = up_proj_lora_B.to(compute_dtype)
                down_proj_lora_A = down_proj_lora_A.to(compute_dtype)
                down_proj_lora_B = down_proj_lora_B.to(compute_dtype)

            # Detect shared weights (hybrid mode): shape[0] == 1 means shared
            gate_A_shared = gate_proj_lora_A.shape[0] == 1
            up_A_shared = up_proj_lora_A.shape[0] == 1
            down_B_shared = down_proj_lora_B.shape[0] == 1

            # Expand shared weights to num_experts for group GEMM
            if gate_A_shared:
                gate_proj_lora_A = gate_proj_lora_A.expand(num_experts, -1, -1).contiguous()
            if up_A_shared:
                up_proj_lora_A = up_proj_lora_A.expand(num_experts, -1, -1).contiguous()
            if down_B_shared:
                down_proj_lora_B = down_proj_lora_B.expand(num_experts, -1, -1).contiguous()

            # MOE Step 3: dispatch
            splits = expert_histogram(expert_index, num_experts)
            scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)
            scatter_output = moe_scatter(hidden_states, scatter_index)

            cumsum_t = torch.cumsum(splits, dim=0)
            max_M = scatter_output.shape[0]
            intermediate_size = gate_proj.shape[2]

            # MOE Step 4 & 6: base GEMM (concat on dim=2 for (G,K,2N))
            gate_up_weight = torch.cat([gate_proj, up_proj], dim=2)
            gate_up_output = gemm_nk(
                a=scatter_output,
                b=gate_up_weight,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            gate_base = gate_up_output[:, :intermediate_size]
            up_base = gate_up_output[:, intermediate_size:]

            # LoRA A: concat on dim=2 for (G, K, 2r)
            lora_r = gate_proj_lora_A.shape[2]
            gate_up_lora_A = torch.cat([gate_proj_lora_A, up_proj_lora_A], dim=2)
            gate_up_lora_intermediate = gemm_nk(
                a=scatter_output,
                b=gate_up_lora_A,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            gate_lora_intermediate = gate_up_lora_intermediate[:, :lora_r].contiguous()
            up_lora_intermediate = gate_up_lora_intermediate[:, lora_r:].contiguous()

            # LoRA B for gate: z @ B
            gate_lora_output = gemm_nk(
                a=gate_lora_intermediate,
                b=gate_proj_lora_B,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            gate_output = (gate_base + gate_lora_output * scaling).contiguous()

            # LoRA B for up: z @ B
            up_lora_output = gemm_nk(
                a=up_lora_intermediate,
                b=up_proj_lora_B,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            up_output = (up_base + up_lora_output * scaling).contiguous()

            # SiLU + mul
            gate_activation = torch.ops.aten.silu(gate_output)
            gated_activation = gate_activation * up_output

            # Routing weights
            reshaped_gate_weight = gate_weights.reshape(-1, 1)
            scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
            scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
            gated_weighted = gated_activation * scattered_gate_weight

            # down_proj base: h @ W
            down_output = gemm_nk(
                a=gated_weighted,
                b=down_proj,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )

            # down_proj LoRA: (h @ A) @ B * scaling
            down_lora_intermediate = gemm_nk(
                a=gated_weighted,
                b=down_proj_lora_A,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            down_lora_output = gemm_nk(
                a=down_lora_intermediate,
                b=down_proj_lora_B,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_a=False,
                transpose_b=False,
            )
            down_output = down_output + down_lora_output * scaling

            # Gather
            expert_output = moe_gather(down_output, scatter_index)
            output = expert_output.reshape(hidden_states.shape)

            ctx.num_experts = num_experts
            ctx.scaling = scaling
            ctx.compute_dtype = compute_dtype
            ctx.gate_A_shared = gate_A_shared
            ctx.up_A_shared = up_A_shared
            ctx.down_B_shared = down_B_shared
            ctx.save_for_backward(
                gate_weights,
                gate_proj,
                up_proj,
                down_proj,
                orig_gate_proj_lora_A,
                orig_gate_proj_lora_B,
                orig_up_proj_lora_A,
                orig_up_proj_lora_B,
                orig_down_proj_lora_A,
                orig_down_proj_lora_B,
                hidden_states,
                scatter_index,
                scatter_output,
                cumsum_t,
                gate_output,
                up_output,
                gated_activation,
                scattered_gate_weight,
                gated_weighted,
                gate_lora_intermediate,
                up_lora_intermediate,
                down_lora_intermediate,
            )

            return output

        @staticmethod
        def backward(ctx, grad_output):
            (
                gate_weights,
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_lora_A,
                gate_proj_lora_B,
                up_proj_lora_A,
                up_proj_lora_B,
                down_proj_lora_A,
                down_proj_lora_B,
                hidden_states,
                scatter_index,
                scatter_output,
                cumsum_t,
                gate_output,
                up_output,
                gated_activation,
                scattered_gate_weight,
                gated_weighted,
                gate_lora_intermediate,
                up_lora_intermediate,
                down_lora_intermediate,
            ) = ctx.saved_tensors

            num_experts = ctx.num_experts
            scaling = ctx.scaling
            compute_dtype = ctx.compute_dtype
            gate_A_shared = ctx.gate_A_shared
            up_A_shared = ctx.up_A_shared
            down_B_shared = ctx.down_B_shared
            hidden_dim = grad_output.shape[-1]
            grad_output = grad_output.view(-1, hidden_dim)
            max_M = scatter_output.shape[0]

            # Cast LoRA weights to compute dtype
            gate_proj_lora_A_c = (
                gate_proj_lora_A.to(compute_dtype) if gate_proj_lora_A.dtype != compute_dtype else gate_proj_lora_A
            )
            gate_proj_lora_B_c = (
                gate_proj_lora_B.to(compute_dtype) if gate_proj_lora_B.dtype != compute_dtype else gate_proj_lora_B
            )
            up_proj_lora_A_c = (
                up_proj_lora_A.to(compute_dtype) if up_proj_lora_A.dtype != compute_dtype else up_proj_lora_A
            )
            up_proj_lora_B_c = (
                up_proj_lora_B.to(compute_dtype) if up_proj_lora_B.dtype != compute_dtype else up_proj_lora_B
            )
            down_proj_lora_A_c = (
                down_proj_lora_A.to(compute_dtype) if down_proj_lora_A.dtype != compute_dtype else down_proj_lora_A
            )
            down_proj_lora_B_c = (
                down_proj_lora_B.to(compute_dtype) if down_proj_lora_B.dtype != compute_dtype else down_proj_lora_B
            )

            # Expand shared LoRA weights for backward compute
            if gate_A_shared:
                gate_proj_lora_A_c = gate_proj_lora_A_c.expand(num_experts, -1, -1).contiguous()
            if up_A_shared:
                up_proj_lora_A_c = up_proj_lora_A_c.expand(num_experts, -1, -1).contiguous()
            if down_B_shared:
                down_proj_lora_B_c = down_proj_lora_B_c.expand(num_experts, -1, -1).contiguous()

            # Initialize LoRA gradients — use full-size buffers for shared weights
            grad_gate_proj_lora_A_full = torch.zeros(
                num_experts,
                gate_proj_lora_A.shape[1],
                gate_proj_lora_A.shape[2],
                dtype=gate_proj_lora_A.dtype,
                device=gate_proj_lora_A.device,
            )
            grad_gate_proj_lora_B = torch.zeros_like(gate_proj_lora_B)
            grad_up_proj_lora_A_full = torch.zeros(
                num_experts,
                up_proj_lora_A.shape[1],
                up_proj_lora_A.shape[2],
                dtype=up_proj_lora_A.dtype,
                device=up_proj_lora_A.device,
            )
            grad_up_proj_lora_B = torch.zeros_like(up_proj_lora_B)
            grad_down_proj_lora_A = torch.zeros_like(down_proj_lora_A)
            grad_down_proj_lora_B_full = torch.zeros(
                num_experts,
                down_proj_lora_B.shape[1],
                down_proj_lora_B.shape[2],
                dtype=down_proj_lora_B.dtype,
                device=down_proj_lora_B.device,
            )

            # MOE Step 10': scatter grad
            grad_down_output = moe_scatter(grad_output, scatter_index)

            # ====== down_proj backward ======
            grad_gated_weighted = gemm_nk(
                a=grad_down_output,
                b=down_proj,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )

            gemm_mn(
                a=down_lora_intermediate,
                b=grad_down_output,
                c=grad_down_proj_lora_B_full,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
            grad_down_proj_lora_B_full.mul_(scaling)

            grad_down_lora_intermediate = gemm_nk(
                a=grad_down_output,
                b=down_proj_lora_B_c,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )
            grad_down_lora_intermediate.mul_(scaling)

            gemm_mn(
                a=gated_weighted,
                b=grad_down_lora_intermediate,
                c=grad_down_proj_lora_A,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )

            grad_gated_weighted_from_lora = gemm_nk(
                a=grad_down_lora_intermediate,
                b=down_proj_lora_A_c,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )
            grad_gated_weighted = grad_gated_weighted + grad_gated_weighted_from_lora

            # ====== Routing weight backward ======
            grad_gated_activation = grad_gated_weighted * scattered_gate_weight
            grad_scattered_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)
            grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
            grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

            gate_activation = torch.ops.aten.silu(gate_output)
            grad_gate_activation = grad_gated_activation * up_output
            grad_up_output = gate_activation * grad_gated_activation

            # ====== up_proj backward ======
            grad_scatter_output_2 = gemm_nk(
                a=grad_up_output,
                b=up_proj,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )

            gemm_mn(
                a=up_lora_intermediate,
                b=grad_up_output,
                c=grad_up_proj_lora_B,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
            grad_up_proj_lora_B.mul_(scaling)

            grad_up_lora_intermediate = gemm_nk(
                a=grad_up_output,
                b=up_proj_lora_B_c,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )
            grad_up_lora_intermediate.mul_(scaling)

            gemm_mn(
                a=scatter_output,
                b=grad_up_lora_intermediate,
                c=grad_up_proj_lora_A_full,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )

            grad_scatter_from_lora_2 = gemm_nk(
                a=grad_up_lora_intermediate,
                b=up_proj_lora_A_c,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )
            grad_scatter_output_2 = grad_scatter_output_2 + grad_scatter_from_lora_2

            # ====== SiLU backward ======
            grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)

            # ====== gate_proj backward ======
            grad_scatter_output_1 = gemm_nk(
                a=grad_gate_output,
                b=gate_proj,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )

            gemm_mn(
                a=gate_lora_intermediate,
                b=grad_gate_output,
                c=grad_gate_proj_lora_B,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
            grad_gate_proj_lora_B.mul_(scaling)

            grad_gate_lora_intermediate = gemm_nk(
                a=grad_gate_output,
                b=gate_proj_lora_B_c,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )
            grad_gate_lora_intermediate.mul_(scaling)

            gemm_mn(
                a=scatter_output,
                b=grad_gate_lora_intermediate,
                c=grad_gate_proj_lora_A_full,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )

            grad_scatter_from_lora_1 = gemm_nk(
                a=grad_gate_lora_intermediate,
                b=gate_proj_lora_A_c,
                cumsum_M=cumsum_t,
                max_M=max_M,
                transpose_b=True,
            )
            grad_scatter_output_1 = grad_scatter_output_1 + grad_scatter_from_lora_1

            # Gather
            grad_hidden_states = moe_add_gather(grad_scatter_output_1, grad_scatter_output_2, scatter_index)
            grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

            # Reduce gradients for shared weights (sum across expert dim)
            if gate_A_shared:
                grad_gate_proj_lora_A = grad_gate_proj_lora_A_full.sum(dim=0, keepdim=True)
            else:
                grad_gate_proj_lora_A = grad_gate_proj_lora_A_full

            if up_A_shared:
                grad_up_proj_lora_A = grad_up_proj_lora_A_full.sum(dim=0, keepdim=True)
            else:
                grad_up_proj_lora_A = grad_up_proj_lora_A_full

            if down_B_shared:
                grad_down_proj_lora_B = grad_down_proj_lora_B_full.sum(dim=0, keepdim=True)
            else:
                grad_down_proj_lora_B = grad_down_proj_lora_B_full

            return (
                None,  # num_experts
                grad_gate_weight,
                None,  # expert_index
                grad_hidden_states,
                None,
                None,
                None,  # base weights (frozen)
                grad_gate_proj_lora_A,
                grad_gate_proj_lora_B,
                grad_up_proj_lora_A,
                grad_up_proj_lora_B,
                grad_down_proj_lora_A,
                grad_down_proj_lora_B,
                None,  # scaling
            )

    return _MoeExpertsLoRAFunction
