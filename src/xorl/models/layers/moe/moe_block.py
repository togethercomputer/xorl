"""MoE block: composes gate + router + experts."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import MoEExperts
from .router import TopKRouter
from .routing_replay import RoutingReplay, get_replay_stage


class MoEBlock(nn.Module):
    """Mixture-of-Experts block.

    Composes a gate linear, a :class:`TopKRouter`, and :class:`MoEExperts`.
    Dispatches between eager (per-expert loop) and triton/native/quack (batch dispatch)
    based on ``moe_implementation``.

    Module hierarchy (preserves checkpoint paths)::

        self.gate    -> nn.Linear          (checkpoint: mlp.gate.weight)
        self.experts -> MoEExperts          (checkpoint: mlp.experts.{gate,up,down}_proj)

    The ``router`` is a stateless helper (no parameters) — it does **not**
    own the gate, so it does not affect the state dict.

    Args:
        hidden_size: Model hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts activated per token.
        intermediate_size: Expert FFN intermediate dimension.
        hidden_act: Activation function name (default: ``"silu"``).
        norm_topk_prob: Whether to renormalize top-k routing weights.
        moe_implementation: Backend name — ``"eager"``, ``"triton"``, ``"native"``, or ``"quack"``.
        train_router: If True (default), gradients flow from expert
            computation through routing weights back to the gate.  If False,
            routing weights are detached before expert computation so the gate
            is only trained via auxiliary losses on ``router_logits``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        moe_implementation: str = "triton",
        train_router: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_implementation = moe_implementation
        self.train_router = train_router

        # Gate linear — directly on this module for checkpoint path ``mlp.gate.weight``
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Stateless routing logic (softmax + topk + renorm)
        self.router = TopKRouter(num_experts, top_k, norm_topk_prob)

        # Expert weights + backend dispatch
        self.experts = MoEExperts(
            num_experts=num_experts,
            hidden_dim=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            moe_implementation=moe_implementation,
        )

        # LoRA adapter tracking (None until inject_lora is called)
        self.lora_adapter = None

        # Routing replay for gradient checkpointing determinism.
        # Set by XorlPreTrainedModel.enable_routing_replay().
        self._routing_replay: Optional[RoutingReplay] = None

    def inject_lora(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        shared_lora: bool = False,
        target_modules: list = None,
        hybrid_shared: bool = False,
    ) -> None:
        """Inject LoRA adapters into the experts of this MoE block.

        After calling this method, ``self.experts`` is replaced with a
        :class:`MoEExpertsLoRA` instance. Base weights are frozen, only
        LoRA weights are trainable.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA alpha for scaling.
            shared_lora: Unused (kept for API compat).
            target_modules: Which projections to apply LoRA to.
                Options: ``["gate_proj", "up_proj", "down_proj"]``.
                Default: all three.
            hybrid_shared: If True, share ``lora_A`` for gate/up and
                ``lora_B`` for down across experts.
        """
        from .lora import inject_lora_into_experts

        inject_lora_into_experts(
            self,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            hybrid_shared=hybrid_shared,
        )
        self.lora_adapter = "injected"

    def _regather_routing(self, router_logits, cached_experts, input_dtype):
        """Re-gather routing weights from softmax using cached expert indices.

        Megatron/SLIME approach: always compute fresh softmax (creates autograd
        graph for gate gradients), then gather with cached indices.  Gradients
        flow through softmax -> gate naturally, no straight-through hack needed.
        """
        softmax_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights = torch.gather(softmax_probs, 1, cached_experts)
        if self.router.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(input_dtype)
        return cached_experts, routing_weights

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass.

        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)``.

        Returns:
            Tuple of ``(output, router_logits)`` where:
            - output: ``(batch, seq_len, hidden_dim)``
            - router_logits: ``(num_tokens, num_experts)``
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Route (optionally upcast to fp32 for numerical alignment with SGLang)
        if getattr(self, "config", None) is not None and getattr(self.config, "_router_fp32", False):
            router_logits = F.linear(hidden_states.float(), self.gate.weight.float())
        else:
            router_logits = self.gate(hidden_states)

        # --- Routing replay for checkpoint determinism ---
        # All active paths (record/replay_forward/replay_backward) go through
        # _regather_routing so the autograd graph structure is identical between
        # forward and checkpoint recompute (softmax -> gather -> normalize).
        # Without this, non-reentrant checkpoint detects different saved-tensor
        # counts and raises CheckpointError.
        #
        # The router topk on the record path runs under torch.no_grad() so it
        # doesn't add autograd nodes — both forward and recompute produce the
        # same saved-tensor structure: gate -> regather(softmax+gather) -> experts.
        # This matches Megatron's approach: scores.gather(1, top_indices).
        stage = get_replay_stage()
        replay = self._routing_replay

        if stage is not None and replay is not None:
            cached_weights = None
            if stage == "record":
                # Determine expert selection without creating autograd nodes
                with torch.no_grad():
                    _, selected_experts = self.router(router_logits, hidden_states.dtype)
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
                cached_weights = replay.pop_forward_weights()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
                cached_weights = replay.pop_backward_weights()

            if cached_weights is not None:
                # Use pre-populated weights from inference (R3 weight replay)
                routing_weights = cached_weights.to(hidden_states.dtype)
            else:
                # Uniform autograd path: softmax -> gather(cached indices) -> normalize
                selected_experts, routing_weights = self._regather_routing(
                    router_logits, selected_experts, hidden_states.dtype
                )
        else:
            # No replay active: use standard router
            routing_weights, selected_experts = self.router(router_logits, hidden_states.dtype)

        # When train_router is False, detach routing_weights so the gate is
        # only trained via auxiliary losses on router_logits, not through
        # expert computation.
        if not self.train_router:
            routing_weights = routing_weights.detach()

        # Expert computation
        if self.moe_implementation == "eager":
            final_hidden_states = self._eager_forward(hidden_states, routing_weights, selected_experts)
        else:
            final_hidden_states = self.experts(hidden_states, routing_weights, selected_experts)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert loop for eager mode."""
        hidden_dim = hidden_states.shape[-1]
        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.experts(current_state, expert_idx=expert_idx) * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states

    @classmethod
    def from_config(cls, config, moe_implementation: str = "triton"):
        """Create from a model config (e.g. ``Qwen3MoeConfig``)."""
        return cls(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=moe_implementation,
            train_router=getattr(config, "train_router", True),
        )
