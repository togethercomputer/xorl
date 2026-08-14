"""MoE block: composes gate + router + experts."""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.models.exact_contract import glm52_exact_forward_enabled

from .experts import MoEExperts, moe_sglang_fused_experts_enabled
from .router import TopKRouter, balanced_synthetic_routing
from .routing_replay import RoutingReplay, get_replay_stage


_MOE_FP64_ACCUM_ENV = "XORL_MOE_FP64_ACCUM"


def _moe_fp64_accum_enabled() -> bool:
    """K3 parity mode: fp64-accumulate eager expert GEMMs + weighted combine.

    Order-invariant MoE reductions (see ``eager_expert_forward_fp64``). Slow —
    fp64 GEMMs run at H100 fp64 tensor-core rate (~15x below bf16) — so this is
    an opt-in reconciliation mode, never a training default.
    """
    return os.environ.get(_MOE_FP64_ACCUM_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _moe_bi_router_enabled(config=None) -> bool:
    """Whether this caller owns the exact batch-invariant router contract.

    Canonical GLM-5.2 makes the contract structural: its model declaration is
    sufficient and does not depend on a launch-time environment switch.

    Route the gate/router matmul through the shared
    batch-invariant router GEMM (``ops.batch_invariant_ops.bi_router_gemm``),
    vendored bit-for-bit into SGLang.

    Live training routes independently of serving (no routing replay), so the
    fp32 router-GEMM reduction-order tail can flip top-k expert selection on
    razor-edge tokens. This pins the router logits identically cross-engine.
    """
    return glm52_exact_forward_enabled(config)


class _BIRouterGemm(torch.autograd.Function):
    """Differentiable wrapper for the forward-only ``bi_router_gemm`` contract.

    The forward is the pinned batch-invariant kernel (enters the K3 stream); the
    backward is the ordinary linear backward (does not enter the forward K3, so
    reduction order is irrelevant and plain matmuls suffice).
    """

    @staticmethod
    def forward(ctx, hidden, weight):
        from xorl.ops.batch_invariant_ops import bi_router_gemm  # noqa: PLC0415

        ctx.save_for_backward(hidden, weight)
        return bi_router_gemm(hidden, weight)

    @staticmethod
    def backward(ctx, grad_logits):
        hidden, weight = ctx.saved_tensors
        grad_logits = grad_logits.to(torch.float32)
        grad_hidden = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_hidden = (grad_logits @ weight.float()).to(hidden.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_logits.t() @ hidden.float()).to(weight.dtype)
        return grad_hidden, grad_weight


_ROUTER_FP32_LAYERS_ENV = "XORL_MOE_ROUTER_FP32_LAYERS"


def _router_fp32_layers_enabled(layer_idx: int | None) -> bool:
    value = os.environ.get(_ROUTER_FP32_LAYERS_ENV, "").strip().lower()
    if value in {"", "0", "false", "no", "off", "none"}:
        return False
    if value in {"all", "*"}:
        return True
    if layer_idx is None:
        return False

    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start <= layer_idx <= end or end <= layer_idx <= start:
                return True
        elif int(part) == layer_idx:
            return True
    return False


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
            computation through routing weights back to the gate. If False,
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
        record_routing_weights: bool = True,
        activation_native: bool = False,
        swiglu_limit: float = 0.0,
        exact_batch_invariant_router: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_implementation = moe_implementation
        self.train_router = train_router
        self.record_routing_weights = record_routing_weights
        self.swiglu_limit = float(swiglu_limit)

        # Gate linear — directly on this module for checkpoint path ``mlp.gate.weight``
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Stateless routing logic (softmax + topk + renorm)
        self.router = TopKRouter(
            num_experts,
            top_k,
            norm_topk_prob,
            exact_batch_invariant=exact_batch_invariant_router,
        )
        self._exact_batch_invariant_router = exact_batch_invariant_router

        # Expert weights + backend dispatch
        self.experts = MoEExperts(
            num_experts=num_experts,
            hidden_dim=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            moe_implementation=moe_implementation,
            activation_native=activation_native,
            swiglu_limit=self.swiglu_limit,
        )

        # LoRA adapter tracking (None until inject_lora is called)
        self.lora_adapter = None

        # Routing replay for gradient checkpointing determinism.
        # Set by XorlPreTrainedModel.enable_routing_replay().
        self._routing_replay: Optional[RoutingReplay] = None

    def supports_routing_replay(self) -> bool:
        """Whether this block's route program can replay cached decisions."""

        return True

    def _capture_diagnostic_component(self, name: str, tensor: torch.Tensor) -> None:
        capture = getattr(self, "_diagnostic_capture_component", None)
        if callable(capture):
            capture(name, tensor)

    def _override_diagnostic_component(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        overrides = getattr(self, "_diagnostic_component_overrides", None)
        override = overrides.get(name) if isinstance(overrides, dict) else None
        if callable(override):
            tensor = override(tensor)
            self._capture_diagnostic_component(f"{name}_override", tensor)
        return tensor

    def _capture_moe_tp_shards(self, name: str, tensor: torch.Tensor) -> None:
        capture = getattr(self, "_diagnostic_capture_component", None)
        if not callable(capture):
            return
        tp_shards = getattr(tensor, "_xorl_sglang_moe_tp_shards", None)
        if not tp_shards:
            return

        shard_sum = None
        for shard_idx, shard in enumerate(tp_shards):
            self._capture_diagnostic_component(f"{name}_tp_shard_{shard_idx}", shard)
            if shard_sum is None:
                shard_sum = shard.clone()
            else:
                shard_sum = shard_sum + shard.to(shard_sum.dtype)

        if shard_sum is not None:
            self._capture_diagnostic_component(f"{name}_tp_shard_sum", shard_sum)

    def inject_lora(
        self,
        r: int = 16,
        lora_alpha: int = 16,
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
            target_modules: Which projections to apply LoRA to.
                Options: ``["gate_proj", "up_proj", "down_proj"]``.
                Default: all three.
            hybrid_shared: If True, share ``lora_A`` for gate/up and
                ``lora_B`` for down across experts.
        """
        from .lora import inject_lora_into_experts  # noqa: PLC0415

        inject_lora_into_experts(
            self,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            hybrid_shared=hybrid_shared,
        )
        self.lora_adapter = "injected"

    def _regather_routing(self, router_logits, cached_experts, input_dtype):
        """Re-gather routing weights from logits using cached expert indices.

        Megatron/SLIME approach: always recompute fresh scores (creates autograd
        graph for gate gradients), then gather with cached indices.  Gradients
        flow through the score function -> gate naturally, no straight-through hack.

        Dispatches on ``self.router.scoring_func``:

        - ``"softmax"`` — recompute via ``F.softmax(logits)``; renormalize when
          ``router.norm_topk_prob``.
        - ``"sqrtsoftplus"`` — recompute via ``sqrt(softplus(logits))``;
          renormalize unconditionally (V4 always renormalizes); apply
          ``routed_scaling_factor`` when set. Matches
          :meth:`TopKRouter._forward_sqrtsoftplus` for graph-structure parity
          between replay-record and replay-recompute under non-reentrant
          checkpointing.
        """
        if self.router.scoring_func == "sqrtsoftplus":
            scores = F.softplus(router_logits.float()).sqrt().type_as(router_logits)
            routing_weights = torch.gather(scores, 1, cached_experts)
            # V4 always renormalizes (matches TopKRouter._forward_sqrtsoftplus).
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
            if self.router.routed_scaling_factor is not None:
                routing_weights = routing_weights * self.router.routed_scaling_factor
            routing_weights = routing_weights.to(input_dtype)
            return cached_experts, routing_weights

        if self.router.synthetic_routing_mode == "balanced":
            routing_weights, _ = balanced_synthetic_routing(
                cached_experts.size(0),
                self.num_experts,
                cached_experts.size(1),
                router_logits.device,
                input_dtype,
            )
            return cached_experts, routing_weights

        # softmax (default) path
        softmax_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights = torch.gather(softmax_probs, 1, cached_experts)
        if self.router.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(input_dtype)
        return cached_experts, routing_weights

    def _bi_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Router logits through the shared batch-invariant router GEMM.

        Guards raise loudly on configs the contract has not verified: an fp8
        gate, a gate bias, or non-bf16 hidden/weight (the bf16-upcast exactness
        argument requires bf16 inputs). Returns fp32 logits ``[N, num_experts]``
        bit-identical to SGLang's batch-invariant router path.
        """
        contract_name = "batch-invariant MoE router"
        if hasattr(self.gate, "fp8_block_size"):
            raise NotImplementedError(f"{contract_name} does not support an fp8 gate")
        if getattr(self.gate, "bias", None) is not None:
            raise NotImplementedError(f"{contract_name} does not support a gate bias")
        if hidden_states.dtype != torch.bfloat16 or self.gate.weight.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{contract_name} requires bf16 hidden states and gate weight; got "
                f"hidden={hidden_states.dtype}, weight={self.gate.weight.dtype}"
            )
        return _BIRouterGemm.apply(hidden_states, self.gate.weight)

    def route(self, hidden_states: torch.Tensor):
        """Compute routing weights and expert assignments.

        Extracts the full routing logic: gate projection (with optional fp32
        upcast), routing replay stage handling, ``_regather_routing``, and
        ``train_router`` detach.

        Args:
            hidden_states: ``(num_tokens, hidden_dim)`` — already flattened.

        Returns:
            Tuple of ``(routing_weights, selected_experts, router_logits)`` where:
            - routing_weights: ``(num_tokens, top_k)``
            - selected_experts: ``(num_tokens, top_k)``
            - router_logits: ``(num_tokens, num_experts)``
        """
        # Route (optionally upcast to fp32 for numerical alignment with SGLang)
        router_fp32 = (
            getattr(self, "config", None) is not None
            and getattr(self.config, "_router_fp32", False)
            or _router_fp32_layers_enabled(getattr(self, "layer_idx", None))
        )
        if self._exact_batch_invariant_router or _moe_bi_router_enabled(getattr(self, "config", None)):
            router_logits = self._bi_router_logits(hidden_states)
        elif router_fp32 and not hasattr(self.gate, "fp8_block_size"):
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
            if stage == "record":
                # Determine expert selection without creating autograd nodes
                with torch.no_grad():
                    _, selected_experts = self.router(router_logits, hidden_states.dtype)
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()

            # Uniform autograd path: softmax -> gather(cached indices) -> normalize.
            # Both forward and recompute go through the same ops so non-reentrant
            # checkpoint sees identical saved-tensor counts.
            selected_experts, routing_weights = self._regather_routing(
                router_logits, selected_experts, hidden_states.dtype
            )

            if self.record_routing_weights:
                if stage == "record":
                    # Cache weights for replay_backward. During checkpoint recompute,
                    # router_logits may differ (non-deterministic attention), so the
                    # regathered weights above could be wrong. We override them below
                    # during replay_backward with these cached values.
                    replay.record_weights(routing_weights)
                elif stage == "replay_backward":
                    # Override with cached weights to ensure deterministic EP dispatch.
                    # The _regather_routing call above still runs (for autograd graph
                    # structure), but its output is replaced with the recorded values.
                    cached_weights = replay.pop_backward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(hidden_states.dtype)
                elif stage == "replay_forward":
                    cached_weights = replay.pop_forward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(hidden_states.dtype)
        else:
            # No replay active: use standard router
            routing_weights, selected_experts = self.router(router_logits, hidden_states.dtype)

        forced_selected_experts = getattr(self, "_diagnostic_forced_selected_experts", None)
        if forced_selected_experts is not None:
            forced_selected_experts = forced_selected_experts.to(
                device=selected_experts.device,
                dtype=selected_experts.dtype,
            )
            if forced_selected_experts.shape != selected_experts.shape:
                raise ValueError(
                    "diagnostic forced routing shape mismatch: "
                    f"expected {tuple(selected_experts.shape)}, got {tuple(forced_selected_experts.shape)}"
                )
            selected_experts = forced_selected_experts
            forced_routing_weights = getattr(self, "_diagnostic_forced_routing_weights", None)
            if forced_routing_weights is not None:
                forced_routing_weights = forced_routing_weights.to(
                    device=routing_weights.device,
                    dtype=hidden_states.dtype,
                )
                if forced_routing_weights.shape != routing_weights.shape:
                    raise ValueError(
                        "diagnostic forced routing weight shape mismatch: "
                        f"expected {tuple(routing_weights.shape)}, got {tuple(forced_routing_weights.shape)}"
                    )
                routing_weights = forced_routing_weights
            else:
                selected_experts, routing_weights = self._regather_routing(
                    router_logits, selected_experts, hidden_states.dtype
                )

        ep_dispatch = getattr(self.experts, "ep_dispatch", "alltoall")
        if self.train_router and ep_dispatch == "deepep":
            raise AssertionError(
                "train_router=True is not supported with ep_dispatch='deepep'. "
                "DeepEP cannot propagate gradients through routing weights. "
                "Set train_router=False or switch to ep_dispatch='alltoall'."
            )
        if not self.train_router:
            routing_weights = routing_weights.detach()

        self._capture_diagnostic_component("moe_router_logits", router_logits)
        self._capture_diagnostic_component("moe_topk_ids", selected_experts)
        self._capture_diagnostic_component("moe_topk_weights", routing_weights)
        if getattr(self, "_diagnostic_capture_routing", False):
            self._diagnostic_last_routing = {
                "router_routing_weights": routing_weights.detach(),
                "router_selected_experts": selected_experts.detach(),
                "router_logits": router_logits.detach(),
            }

        return routing_weights, selected_experts, router_logits

    def forward_experts_only(self, hidden_states, routing_weights, selected_experts):
        """Run expert computation with pre-computed routing weights.

        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)``.
            routing_weights: ``(num_tokens, top_k)``.
            selected_experts: ``(num_tokens, top_k)``.

        Returns:
            ``(batch, seq_len, hidden_dim)`` expert output.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        self._capture_diagnostic_component("moe_input", hidden_states)
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        if (
            _moe_fp64_accum_enabled()
            and not parallel_state.ep_enabled
            and not self.experts.sglang_moe_tp_sim_enabled(parallel_state)
        ):
            hidden_states = self._eager_forward_fp64(hidden_states, routing_weights, selected_experts)
        elif (
            moe_sglang_fused_experts_enabled(getattr(parallel_state, "ep_size", 1), hidden_states.device, self.experts)
            and not parallel_state.ep_enabled
            and not self.experts.sglang_moe_tp_sim_enabled(parallel_state)
        ):
            hidden_states = self.experts.sglang_fused_experts_forward(hidden_states, routing_weights, selected_experts)
        else:
            hidden_states = self.experts(hidden_states, routing_weights, selected_experts)
        self._capture_diagnostic_component("moe_experts_output", hidden_states)
        self._capture_moe_tp_shards("moe_experts_output", hidden_states)
        hidden_states = self._override_diagnostic_component("moe_experts_output", hidden_states)
        tp_shards = getattr(hidden_states, "_xorl_sglang_moe_tp_shards", None)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        if tp_shards is not None:
            hidden_states._xorl_sglang_moe_tp_shards = tuple(
                shard.reshape(batch_size, sequence_length, hidden_dim) for shard in tp_shards
            )
        return hidden_states

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
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        self._capture_diagnostic_component("moe_input", flat_hidden_states)

        routing_weights, selected_experts, router_logits = self.route(flat_hidden_states)

        # Expert computation (already flat — call experts directly to avoid extra reshape).
        # Under EP, even eager compute must use the dispatch/combine path: the
        # router returns global expert ids while local expert weights are sharded.
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        if (
            _moe_fp64_accum_enabled()
            and not parallel_state.ep_enabled
            and not self.experts.sglang_moe_tp_sim_enabled(parallel_state)
        ):
            # K3 parity mode overrides the configured backend (eager/triton/...):
            # the fp64 loop is the order-invariant reference for all of them.
            final_hidden_states = self._eager_forward_fp64(flat_hidden_states, routing_weights, selected_experts)
        elif (
            moe_sglang_fused_experts_enabled(
                getattr(parallel_state, "ep_size", 1), flat_hidden_states.device, self.experts
            )
            and not parallel_state.ep_enabled
            and not self.experts.sglang_moe_tp_sim_enabled(parallel_state)
        ):
            # K3 parity mode: SGLang's serving kernel overrides the configured
            # backend so trainer and serving share one MoE reduction tree.
            final_hidden_states = self.experts.sglang_fused_experts_forward(
                flat_hidden_states, routing_weights, selected_experts
            )
        elif (
            self.moe_implementation == "eager"
            and not parallel_state.ep_enabled
            and not self.experts.sglang_moe_tp_sim_enabled(parallel_state)
        ):
            final_hidden_states = self._eager_forward(flat_hidden_states, routing_weights, selected_experts)
        else:
            final_hidden_states = self.experts(flat_hidden_states, routing_weights, selected_experts)
        self._capture_diagnostic_component("moe_experts_output", final_hidden_states)
        self._capture_moe_tp_shards("moe_experts_output", final_hidden_states)
        final_hidden_states = self._override_diagnostic_component("moe_experts_output", final_hidden_states)
        tp_shards = getattr(final_hidden_states, "_xorl_sglang_moe_tp_shards", None)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        if tp_shards is not None:
            final_hidden_states._xorl_sglang_moe_tp_shards = tuple(
                shard.reshape(batch_size, sequence_length, hidden_dim) for shard in tp_shards
            )

        return final_hidden_states, router_logits

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert loop for eager mode."""
        if _moe_fp64_accum_enabled():
            return self._eager_forward_fp64(hidden_states, routing_weights, selected_experts)
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

    def _eager_forward_fp64(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """K3 parity variant of ``_eager_forward``: order-invariant fp64 reductions.

        Same routing/masking as the bf16 loop, but the three expert GEMMs, the
        routing-weight multiply, and the cross-expert combine all accumulate in
        fp64; the only intermediate bf16 cast is the activation product inside
        ``eager_expert_forward_fp64``, and the result is cast back once at the
        end. Inference-only (no autograd through the fp64 detour is needed for
        logprob scoring; gradients are unsupported).
        """
        from .backend.eager import eager_expert_forward_fp64  # noqa: PLC0415

        experts = self.experts
        if not experts.gated or experts.hidden_act != "silu":
            raise NotImplementedError(f"{_MOE_FP64_ACCUM_ENV} supports gated SiLU experts only")
        if experts.gate_up_bias is not None or experts.down_bias is not None:
            raise NotImplementedError(f"{_MOE_FP64_ACCUM_ENV} does not support expert biases")
        if experts.swiglu_limit > 0:
            raise NotImplementedError(f"{_MOE_FP64_ACCUM_ENV} does not support swiglu_limit")

        hidden_dim = hidden_states.shape[-1]
        gate_proj = experts.gate_proj
        up_proj = experts.up_proj
        down_proj = experts.down_proj
        final_hidden_states = torch.zeros(hidden_states.shape, dtype=torch.float64, device=hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_out64 = eager_expert_forward_fp64(current_state, expert_idx, gate_proj, up_proj, down_proj)
            final_hidden_states.index_add_(0, top_x, expert_out64 * routing_weights[top_x, idx, None].to(torch.float64))

        return final_hidden_states.to(hidden_states.dtype)

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
            train_router=getattr(config, "train_router", False),
            record_routing_weights=getattr(config, "record_routing_weights", True),
            activation_native=getattr(config, "_activation_native", False),
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0)),
        )
