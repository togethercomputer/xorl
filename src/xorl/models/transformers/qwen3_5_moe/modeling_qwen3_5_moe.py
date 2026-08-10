from functools import partial
from typing import Callable, Literal, Optional, Tuple, Unpack

import torch
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.checkpoint_handlers.buffers import (
    checkpoint_has_per_expert_weights,
    detect_prequantized_checkpoint,
    get_prequantized_exclude_modules,
)
from xorl.models.layers import ACT2FN, RotaryEmbedding
from xorl.models.layers.attention import AttentionKwargs, update_causal_mask
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.moe import MoEBlock
from xorl.models.layers.moe.ep_native_combine import validate_qwen35_native_ep_combine_size
from xorl.models.layers.normalization import (
    compiled_zero_centered_rms_norm,
    eager_zero_centered_rms_norm,
    fast_zero_centered_batch_invariant_residual_rms_norm,
    fast_zero_centered_batch_invariant_rms_norm,
    fast_zero_centered_families_v2_rms_norm,
    get_rmsnorm_mode,
    native_zero_centered_rms_norm,
    native_zero_centered_rms_norm_without_batch_invariant,
)
from xorl.models.module_utils import MoEGradientCheckpointingLayer
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.qwen3_5_moe import parallelize
from xorl.models.transformers.qwen3_5_moe.checkpoint_handler import Qwen3_5MoeCheckpointHandler
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_shared import (
    LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE,
    QWEN3_5_CHECKPOINT_CONVERSION_MAPPING,
    QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS,
    _apply_qwen35_gdn_exact,
    has_linear_attention_layers,
    qwen3_5_apply_rotary_pos_emb,
)
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils import logging


logger = logging.get_logger(__name__)


def _adapt_qwen3_5_moe_config(config):
    exact_contract = bool(getattr(config, "_qwen35_exact_contract", False))
    rmsnorm_family = getattr(config, "_qwen35_rmsnorm_family", "v1")
    if hasattr(config, "text_config"):
        adapted = Qwen3_5MoeConfig.from_hf_config(config)
    elif isinstance(config, Qwen3_5MoeConfig):
        adapted = config
    elif getattr(config, "model_type", None) in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        adapted = Qwen3_5MoeConfig.from_hf_config(config)
    else:
        adapted = config
    adapted._qwen35_exact_contract = exact_contract
    adapted._qwen35_rmsnorm_family = rmsnorm_family
    return adapted


def _raise_if_ring_fla_unsupported(config: Qwen3_5MoeConfig, ps) -> None:
    if ps.ringattn_size > 1 and has_linear_attention_layers(config):
        logger.warning_once(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)
        raise ValueError(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)


class Qwen3_5MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        activation_native = getattr(config, "_activation_native", False)
        if getattr(config, "_qwen35_exact_contract", False):
            activation_native = False
        self._use_fused_silu = config.hidden_act == "silu" and not activation_native

    def unfuse_for_tp(self):
        device = self.gate_up_proj.weight.device
        dtype = self.gate_up_proj.weight.dtype
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        del self.gate_up_proj

    def forward(self, x):
        if hasattr(self, "gate_up_proj"):
            if self._use_fused_silu:
                x = fused_silu_and_mul(self.gate_up_proj(x))
            else:
                gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
                x = self.act_fn(gate) * up
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self._use_fused_silu:
                x = fused_silu_and_mul(torch.cat([gate, up], dim=-1))
            else:
                x = self.act_fn(gate) * up
        return self.down_proj(x)


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        exact_contract: bool = False,
        rmsnorm_family: Literal["v1", "v2"] = "v1",
    ):
        super().__init__()
        if rmsnorm_family not in ("v1", "v2"):
            raise ValueError(f"Unsupported Qwen3.5 MoE RMSNorm family: {rmsnorm_family!r}")
        if rmsnorm_family == "v2" and not exact_contract:
            raise RuntimeError("Qwen families-v2 RMSNorm is admitted only in the exact training lane.")
        self.eps = eps
        self.exact_contract = exact_contract
        self.rmsnorm_family = rmsnorm_family
        self.weight = nn.Parameter(torch.zeros(dim))
        self.mode = get_rmsnorm_mode()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        force_sglang_residual: bool = False,
    ):
        if self.exact_contract and self.rmsnorm_family == "v2":
            if self.mode != "sglang_fused":
                raise RuntimeError(
                    f"The Qwen families-v2 RMSNorm program requires rmsnorm_mode='sglang_fused'; got {self.mode!r}."
                )
            if residual is None:
                out = fast_zero_centered_families_v2_rms_norm(x, self.weight, self.eps)
                residual_out = None
            else:
                out, residual_out = fast_zero_centered_families_v2_rms_norm(
                    x,
                    self.weight,
                    self.eps,
                    residual=residual,
                )
            if residual_out is not None and prenorm:
                return out, residual_out
            return out

        residual_out: Optional[torch.Tensor] = None
        norm_input = x
        if residual is not None:
            residual_out = x + residual
            norm_input = residual_out

        if self.mode == "eager":
            out = eager_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        elif self.mode == "native":
            out = native_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        elif self.mode == "compile":
            out = compiled_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        elif self.mode in ("sglang", "sglang_fused"):
            # Norm-seed contract (§14) family split, ported from qwen3_moe:
            # residual-tree norms (layer>0 input / post-attn / final) are family-2,
            # no-residual norms (qk-norm / layer-0 input) are family-1.
            if residual_out is not None or force_sglang_residual:
                if self.exact_contract:
                    # Pair with the BI-ops sampler's
                    # eager-with-BI-mean composition (F.rms_norm is 1 ulp off at
                    # rare boundary values).
                    out = fast_zero_centered_batch_invariant_residual_rms_norm(norm_input, self.weight, self.eps)
                else:
                    out = native_zero_centered_rms_norm_without_batch_invariant(norm_input, self.weight, self.eps)
            elif self.mode == "sglang_fused" and self.exact_contract:
                # Exact Qwen family-1 must bit-match the aten::rms_norm
                # interpose kernel, with real gradients. Selection is owned by
                # this model instance rather than process-global wrapper state.
                out = fast_zero_centered_batch_invariant_rms_norm(norm_input, self.weight, self.eps)
            else:
                out = native_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        else:
            raise NotImplementedError(f"Unsupported rmsnorm_mode for Qwen3.5 MoE RMSNorm: {self.mode}")

        if residual_out is not None and prenorm:
            return out, residual_out
        return out


class Qwen3_5MoeAttention(nn.Module):
    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        # Qwen3.5 full-attention layers are global attention, not SWA.
        self.sliding_window = None
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        exact_contract = bool(getattr(config, "_qwen35_exact_contract", False))
        rmsnorm_family = getattr(config, "_qwen35_rmsnorm_family", "v1")
        self.q_norm = Qwen3_5MoeRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, exact_contract=exact_contract, rmsnorm_family=rmsnorm_family
        )
        self.k_norm = Qwen3_5MoeRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, exact_contract=exact_contract, rmsnorm_family=rmsnorm_family
        )
        self._attn_gate: torch.Tensor | None = None

    def _capture_diagnostic_component(self, name: str, value: torch.Tensor) -> None:
        capture = self.__dict__.get("_diagnostic_capture_component")
        if callable(capture):
            capture(name, value)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        self._capture_diagnostic_component("attention_input", hidden_states)
        qkv = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        self._capture_diagnostic_component("qkv", qkv)
        query_states, gate = torch.chunk(qkv, 2, dim=-1)
        self._attn_gate = gate.reshape(*input_shape, -1)

        self._capture_diagnostic_component("q_pre_qk_norm", query_states)
        query_states = self.q_norm(query_states.view(hidden_shape))
        self._capture_diagnostic_component("q_post_qk_norm", query_states)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        self._capture_diagnostic_component("k_pre_qk_norm", key_states)
        key_states = self.k_norm(key_states)
        self._capture_diagnostic_component("k_post_qk_norm", key_states)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        self._capture_diagnostic_component("v", value_states)

        cos, sin = position_embeddings
        self._capture_diagnostic_component("rope_cos", cos)
        self._capture_diagnostic_component("rope_sin", sin)
        query_states, key_states = qwen3_5_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            class_b=bool(getattr(self.config, "_rope_class_b", False)),
        )
        if getattr(self.config, "_attention_cast_bf16", False):
            query_states = query_states.to(torch.bfloat16)
            key_states = key_states.to(torch.bfloat16)
        self._capture_diagnostic_component("q", query_states)
        self._capture_diagnostic_component("k", key_states)
        return query_states, key_states, value_states

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        gate = self._attn_gate
        self._attn_gate = None
        if gate is None:
            raise RuntimeError("Qwen3.5 MoE attention gate was not initialized before output projection.")
        self._capture_diagnostic_component("attention_gate", gate)
        self._capture_diagnostic_component("attn_output", attn_output)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        self._capture_diagnostic_component("attn_output_gated", attn_output)
        output = self.o_proj(attn_output)
        self._capture_diagnostic_component("o_proj_output", output)
        return output

    def _get_attention_fn(self) -> Callable:
        return ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    def _attention_kwargs(self) -> dict:
        return dict(
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del position_ids, past_key_values
        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        attn_output = attn_strategy.compute_attention(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


class Qwen3_5MoeSparseMoeBlock(MoEBlock):
    def __init__(self, config, moe_implementation="triton", layer_idx: int | None = None):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=moe_implementation,
            train_router=getattr(config, "train_router", False),
            activation_native=getattr(config, "_activation_native", False),
            exact_batch_invariant_router=bool(getattr(config, "_qwen35_exact_contract", False)),
        )
        self.config = config
        self.layer_idx = layer_idx
        self._native_ep_combine = bool(getattr(config, "_qwen35_exact_contract", False))
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)
        self.experts.alltoall_combine_hidden_chunk_size = getattr(config, "_alltoall_combine_hidden_chunk_size", 0)
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def _shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Shared expert: MLP + sigmoid gate."""
        flat = hidden_states.view(-1, hidden_states.size(-1))
        out = self.shared_expert(flat)
        out = torch.sigmoid(self.shared_expert_gate(flat)) * out
        return out.view_as(hidden_states)

    def _ep_combine_native(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Native-EP ordered combine for the trainer's real EP group.

        Every rank gathers the full token batch (backward: reduce-scatter sum),
        computes ITS routed partial through the masked serving-kernel Function
        on the LOCAL expert slice + ITS shared-expert TP slice (trainable BI
        GEMMs, torch-native bf16 silu*mul, sigmoid gate) added in bf16 — exactly
        serving's per-rank partial — then partials are exchanged RAW
        (all-to-all, never NCCL-summed) and each rank chain-sums its own tokens'
        n partials in serving rank order (n-1) -> 0. Forward bits match the
        serving engine; backward uses stock numerics throughout
        (cuBLAS shared-expert grads, grouped-GEMM expert grads, NCCL grad
        reductions)."""
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415
        from xorl.models.layers.moe.ep_native_combine import (  # noqa: PLC0415
            exchange_and_chain_sum,
            gather_ids_for_ep_combine,
            gather_tokens_for_ep_combine,
            max_rows_for_ep_combine,
            sglang_fused_gate_sigmoid_mul_add,
        )
        from xorl.ops.batch_invariant_ops import _BatchInvariantTrunkLinearFn  # noqa: PLC0415

        ps = get_parallel_state()
        if not ps.ep_enabled:
            raise RuntimeError("Qwen3.5-MoE exact ordered combine requires trainer EP mirroring the serving EP size")
        if not hasattr(self.shared_expert, "gate_up_proj"):
            raise NotImplementedError("Qwen3.5-MoE exact ordered combine requires the fused shared-expert gate_up_proj")
        ep_size, ep_rank, ep_group = ps.ep_size, ps.ep_rank, ps.ep_group
        validate_qwen35_native_ep_combine_size(ep_size)
        inter = self.shared_expert.intermediate_size
        if inter % ep_size != 0:
            raise ValueError(
                f"Qwen3.5-MoE exact ordered combine: shared_expert intermediate_size={inter} "
                f"not divisible by ep_size={ep_size}"
            )
        e_local = int(self.experts.gate_up_proj.shape[0])
        if e_local * ep_size != self.experts.num_experts:
            raise RuntimeError(
                f"Qwen3.5-MoE exact ordered combine: local expert slice {e_local} x ep_size {ep_size} "
                f"!= num_experts {self.experts.num_experts} (trainer EP must mirror serving EP)"
            )

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        routing_flat = routing_weights.reshape(flat.shape[0], -1)
        selected_flat = selected_experts.reshape(flat.shape[0], -1)

        # 1. full token batch on every rank (serving's DP-attention gather)
        # Packed DP slices can have different local row counts, so negotiate one
        # equal-count collective shape and discard this rank's padding after the
        # ordered combine. BI kernels make those extra rows forward-independent.
        padded_rows = max_rows_for_ep_combine(flat.shape[0], flat.device, ep_group)
        gathered = gather_tokens_for_ep_combine(flat, ep_group, padded_rows)
        gathered_routing = gather_tokens_for_ep_combine(routing_flat, ep_group, padded_rows)
        gathered_ids = gather_ids_for_ep_combine(selected_flat, ep_group, padded_rows)
        self._capture_diagnostic_component("moe_native_gathered_input", gathered)
        self._capture_diagnostic_component("moe_native_gathered_routing", gathered_routing)
        self._capture_diagnostic_component("moe_native_gathered_ids", gathered_ids)

        # 2. this rank's partial: routed (masked serving kernel on the local
        #    slice) + shared-expert TP slice, added in bf16 (serving semantics)
        lo = ep_rank * e_local
        local_ids = torch.where(
            (gathered_ids >= lo) & (gathered_ids < lo + e_local),
            gathered_ids - lo,
            gathered_ids.new_full((), -1),
        ).to(torch.int32)
        self._capture_diagnostic_component("moe_native_local_ids", local_ids)
        # Enter through Module.__call__ so the experts FSDP unit materializes
        # its BF16 compute parameters before invoking the serving kernel.
        routed = self.experts(
            gathered,
            gathered_routing,
            sglang_ep_native_local_ids=local_ids,
        ).to(torch.bfloat16)
        self._capture_diagnostic_component("moe_native_routed", routed)

        w_gu = self.shared_expert.gate_up_proj.weight  # [2I, H], gate rows first
        w_down = self.shared_expert.down_proj.weight  # [H, I]
        shard = inter // ep_size
        lo_s = ep_rank * shard
        # Retain the decomposed gate only when operand diagnostics request it.
        # The local partial itself uses serving's fused reduction and rounding.
        if callable(self.__dict__.get("_diagnostic_capture_component")):
            gate_value = torch.sigmoid(
                _BatchInvariantTrunkLinearFn.apply(gathered, self.shared_expert_gate.weight, None)
            )
            self._capture_diagnostic_component("moe_native_shared_gate_value", gate_value)
        w_slice = torch.cat((w_gu[lo_s : lo_s + shard], w_gu[inter + lo_s : inter + lo_s + shard]), dim=0)
        gate_up = _BatchInvariantTrunkLinearFn.apply(gathered, w_slice, None)
        self._capture_diagnostic_component("moe_native_shared_gate_up", gate_up)
        act = fused_silu_and_mul(gate_up)
        self._capture_diagnostic_component("moe_native_shared_act", act)
        down = _BatchInvariantTrunkLinearFn.apply(act, w_down[:, lo_s : lo_s + shard].contiguous(), None)
        self._capture_diagnostic_component("moe_native_shared_down", down)
        partial = sglang_fused_gate_sigmoid_mul_add(
            gathered,
            self.shared_expert_gate.weight.squeeze(0),
            down,
            routed,
        )
        self._capture_diagnostic_component("moe_native_local_partial", partial)

        # 3./4. raw exchange + serving-order chain sum (autograd reverses the exchange)
        out = exchange_and_chain_sum(partial, ep_group, ep_size)
        self._capture_diagnostic_component("moe_native_combined", out)
        return out[: flat.shape[0]].reshape(batch_size, sequence_length, hidden_dim)

    def forward_experts_only(self, hidden_states, routing_weights, selected_experts):
        """Sparse experts + shared expert with pre-computed routing."""
        if self._native_ep_combine:
            return self._ep_combine_native(hidden_states, routing_weights, selected_experts)
        expert_output = super().forward_experts_only(hidden_states, routing_weights, selected_experts)
        return expert_output + self._shared_expert(hidden_states)

    def forward(self, hidden_states: torch.Tensor):
        if self._native_ep_combine:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            routing_weights, selected_experts, router_logits = self.route(hidden_states.view(-1, hidden_dim))
            out = self._ep_combine_native(hidden_states, routing_weights, selected_experts)
            return out, router_logits
        expert_output, router_logits = super().forward(hidden_states)
        return expert_output + self._shared_expert(hidden_states), router_logits


QWEN3_5_MOE_CLASSES = {
    "eager": partial(Qwen3_5MoeSparseMoeBlock, moe_implementation="eager"),
    "triton": partial(Qwen3_5MoeSparseMoeBlock, moe_implementation="triton"),
    "native": partial(Qwen3_5MoeSparseMoeBlock, moe_implementation="native"),
    "quack": partial(Qwen3_5MoeSparseMoeBlock, moe_implementation="quack"),
}


class Qwen3_5MoeDecoderLayer(MoEGradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
        self.self_attn = None
        self.linear_attn = None
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                expand_v=config.linear_value_head_dim / config.linear_key_head_dim,
                head_dim=config.linear_key_head_dim,
                num_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                mode="chunk",
                use_gate=config.attn_output_gate,
                use_short_conv=True,
                conv_size=config.linear_conv_kernel_dim,
                layer_idx=layer_idx,
                norm_eps=config.rms_norm_eps,
                exact_contract=bool(getattr(config, "_qwen35_exact_contract", False)),
            )
        else:
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)

        exact_contract = bool(getattr(config, "_qwen35_exact_contract", False))
        rmsnorm_family = getattr(config, "_qwen35_rmsnorm_family", "v1")
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            exact_contract=exact_contract,
            rmsnorm_family=rmsnorm_family,
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            exact_contract=exact_contract,
            rmsnorm_family=rmsnorm_family,
        )
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            moe_implementation = getattr(config, "_moe_implementation", "triton")
            self.mlp = QWEN3_5_MOE_CLASSES[moe_implementation](config, layer_idx=layer_idx)
        else:
            self.mlp = Qwen3_5MoeMLP(config, intermediate_size=config.intermediate_size)

    def _pre_mlp_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layernorm → attention → layernorm."""
        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states,
            force_sglang_residual=self.layer_idx > 0 and self.input_layernorm.mode in ("sglang", "sglang_fused"),
        )

        if self.linear_attn is not None:
            cu_seqlens = kwargs.get("cu_seq_lens_q")
            cp_context = build_linear_attention_cp_context(
                cu_seqlens,
                conv1d_kernel_size=self.linear_attn.conv_size if self.linear_attn.use_short_conv else None,
            )
            linear_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
            if cp_context is not None:
                linear_mask = None
            # Pass cu_seqlens/cp_context as EXPLICIT kwargs (not a **dict splat) so torch.compile/dynamo can
            # trace through this call. The `**linear_kwargs` splat here was a CALL_FUNCTION_EX graph break at
            # the GatedDeltaNet boundary that fragmented the compiled graph. GatedDeltaNet.forward reads both
            # via kwargs.get() (None-safe), so explicit None is identical to the old conditional omission.
            hidden_states, _, _ = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=linear_mask,
                past_key_values=past_key_values,
                use_cache=False,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual=residual, prenorm=True)
        return hidden_states, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return self._moe_forward(
            hidden_states,
            output_router_logits=output_router_logits,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class Qwen3_5MoePreTrainedModel(XorlPreTrainedModel):
    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3_5MoeDecoderLayer"]
    _checkpoint_conversion_mapping = QWEN3_5_CHECKPOINT_CONVERSION_MAPPING
    _checkpoint_skip_key_patterns = QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            module.weight.data.zero_()
        elif isinstance(module, GatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, RotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq = module.inv_freq

    def get_parallel_plan(self):
        return parallelize.get_ep_plan()

    def get_checkpoint_handler(self, **kwargs):
        checkpoint_keys = kwargs.get("checkpoint_keys", set())
        weights_path = kwargs.get("weights_path", None)
        ep_rank = kwargs.get("ep_rank", 0)
        ep_size = kwargs.get("ep_size", 1)
        is_broadcast = kwargs.get("is_broadcast", False)
        has_per_expert = checkpoint_has_per_expert_weights(checkpoint_keys) if checkpoint_keys else True
        is_prequantized = detect_prequantized_checkpoint(weights_path)
        exclude_modules = getattr(self, "_qlora_exclude_modules", None)
        if exclude_modules is None:
            exclude_modules = get_prequantized_exclude_modules(weights_path) if is_prequantized else set()
        if is_broadcast:
            ep_rank, ep_size = 0, 1
        unfused = getattr(self, "_unfused_for_tp", False)
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        skip_expert_loading = False
        if not is_prequantized:
            from xorl.qlora.modules.moe_experts import QLoRAMoeExperts  # noqa: PLC0415

            skip_expert_loading = any(
                isinstance(module, QLoRAMoeExperts) and not getattr(module, "_weights_loaded", False)
                for module in self.modules()
            )
        return Qwen3_5MoeCheckpointHandler(
            num_experts=self.config.num_experts,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            linear_key_dim=self.config.linear_num_key_heads * self.config.linear_key_head_dim,
            linear_value_dim=self.config.linear_num_value_heads * self.config.linear_value_head_dim,
            ep_rank=ep_rank,
            ep_size=ep_size,
            checkpoint_has_per_expert=has_per_expert,
            skip_qkv_merge=True,
            skip_gate_up_merge=unfused,
            skip_expert_loading=skip_expert_loading,
            is_prequantized=is_prequantized,
            exclude_modules=exclude_modules,
        )


class Qwen3_5MoeModel(Qwen3_5MoePreTrainedModel):
    def __init__(self, config: Qwen3_5MoeConfig):
        config = _adapt_qwen3_5_moe_config(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            exact_contract=bool(getattr(config, "_qwen35_exact_contract", False)),
            rmsnorm_family=getattr(config, "_qwen35_rmsnorm_family", "v1"),
        )
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> MoeModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if self.embed_tokens is not None:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            hidden_states = input_ids if inputs_embeds is None else inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        if use_cache is None:
            use_cache = False

        ps = get_parallel_state()
        _raise_if_ring_fla_unsupported(self.config, ps)

        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        causal_mask = update_causal_mask(
            self.config._attn_implementation,
            attention_mask,
            hidden_states,
            cache_position,
            sliding_window=None,
            is_training=self.training,
            output_attentions=output_attentions,
        )
        linear_attn_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = get_cp_strategy().prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_key_value_heads,
        )

        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        # Per-layer residual-stream hiddens for all-layer OPRD (post-decoder-layer,
        # pre-final-norm). Student and teacher use the SAME convention here, so they
        # align 1:1 per layer; KL still uses the post-norm last_hidden_state below.
        # NB: we append the OUTPUT of each decoder layer (index i == layer i output,
        # 40 entries total) — NOT the standard-HF embedding+inputs convention — so the
        # OPRD layer-index resolution in model_runner indexes hidden_states[i] directly.
        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers:
            if decoder_layer is None:
                continue
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            # When selective checkpointing is enabled, the decoder layer handles
            # its own sub-checkpointing — skip the outer checkpoint.
            _use_outer_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and self._gradient_checkpointing_method == "recompute_full_layer"
            )
            if _use_outer_checkpoint:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    None,
                    use_cache,
                    output_attentions,
                    output_router_logits,
                    position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                # _moe_forward does not produce attention weights; use None placeholder.
                all_self_attns += (None,)
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states,
                force_sglang_residual=getattr(self.norm, "mode", None) in ("sglang", "sglang_fused"),
            )
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            hidden_states=all_hidden_states,
        )


class Qwen3_5MoeForCausalLM(Qwen3_5MoePreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        config = _adapt_qwen3_5_moe_config(config)
        super().__init__(config)
        self.model = Qwen3_5MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def unfuse_for_tp(self):
        parallelize.unfuse_for_tp(self)

    def _apply_qwen35_gdn_exact(self) -> dict[str, int]:
        return _apply_qwen35_gdn_exact(self)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_pp_module_config(self):
        return {
            "input_fqns": ["model.embed_tokens"],
            "layer_prefix": "model.layers",
            "output_fqns": ["model.norm", "lm_head"],
            "always_keep_fqns": ["model.rotary_emb"],
            "num_layers": self.config.num_hidden_layers,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> MoeCausalLMOutput:
        output_router_logits = self.config.output_router_logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        return MoeCausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
            hidden_states=outputs.hidden_states,
        )


class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeForCausalLM):
    """Text-only local implementation for HF Qwen3.5 wrapper configs."""


ModelClass = [Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration]

__all__ = [
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeModel",
    "Qwen3_5MoePreTrainedModel",
    "Qwen3_5MoeSparseMoeBlock",
]
