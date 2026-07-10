from functools import partial
from typing import Callable, Optional, Tuple, Unpack

import torch
import torch.nn.functional as F
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
from xorl.models.layers.moe import MoEBlock, MoEExperts
from xorl.models.layers.moe.experts import moe_sglang_ep_combine_sim_size
from xorl.models.layers.normalization import (
    compiled_zero_centered_rms_norm,
    eager_zero_centered_rms_norm,
    fast_zero_centered_batch_invariant_residual_rms_norm,
    fast_zero_centered_batch_invariant_rms_norm,
    get_rmsnorm_mode,
    is_bi_residual_norm_enabled,
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
    has_linear_attention_layers,
    qwen3_5_apply_rotary_pos_emb,
)
from xorl.ops.batch_invariant_ops import is_trunk_linear_contract_enabled
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils import logging


logger = logging.get_logger(__name__)


def _adapt_qwen3_5_moe_config(config):
    if hasattr(config, "text_config"):
        return Qwen3_5MoeConfig.from_hf_config(config)
    if isinstance(config, Qwen3_5MoeConfig):
        return config
    if getattr(config, "model_type", None) in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return Qwen3_5MoeConfig.from_hf_config(config)
    return config


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
        self._use_fused_silu = config.hidden_act == "silu" and not getattr(config, "_activation_native", False)

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
            x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.mode = get_rmsnorm_mode()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        force_sglang_residual: bool = False,
    ):
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
                if is_bi_residual_norm_enabled():
                    # XORL-245 family-2 contract: pair with the BI-ops sampler's
                    # eager-with-BI-mean composition (F.rms_norm is 1 ulp off at
                    # rare boundary values).
                    out = fast_zero_centered_batch_invariant_residual_rms_norm(norm_input, self.weight, self.eps)
                else:
                    out = native_zero_centered_rms_norm_without_batch_invariant(norm_input, self.weight, self.eps)
            elif self.mode == "sglang_fused" and is_trunk_linear_contract_enabled():
                # Trunk contract lane: family-1 must bit-match the aten::rms_norm
                # interpose kernel, with real gradients.
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
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._attn_gate: torch.Tensor | None = None

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        self._attn_gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = qwen3_5_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            interleaved=getattr(self.config, "mrope_interleaved", False),
        )
        return query_states, key_states, value_states

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        gate = self._attn_gate
        self._attn_gate = None
        if gate is None:
            raise RuntimeError("Qwen3.5 MoE attention gate was not initialized before output projection.")
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

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
        )
        self.config = config
        self.layer_idx = layer_idx
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

    def _ep_combine_sim(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        ep_size: int,
    ) -> torch.Tensor:
        """XORL-245 EP-combine simulation: reproduce an EP<n>/a2a-none BI-ops
        serving engine's whole MoE-block output bitwise on local weights.

        Per simulated rank r: routed partial through the serving kernel on the
        contiguous expert slice (global top-k ids mapped local, -1 filtered) +
        the shared expert computed as serving's TP slice (BI GEMMs on the
        [gate|up] row-slice, torch-native bf16 silu*mul, BI down GEMM on the
        column slice, sigmoid(BI gate GEMM) mul), added in bf16 per rank; then
        the NCCL-tree-equivalent SEQUENTIAL bf16 chain sum in rank order
        (n-1) -> 0. Forward-only (the experts method raises under grad)."""
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415
        from xorl.ops.batch_invariant_ops import matmul_persistent  # noqa: PLC0415

        if get_parallel_state().ep_enabled:
            raise RuntimeError(
                "XORL_MOE_SGLANG_EP_COMBINE_SIM requires trainer EP disabled (full-width local "
                "expert weights); run the scoring server with expert_parallel_size: 1."
            )
        if not hasattr(self.shared_expert, "gate_up_proj"):
            raise NotImplementedError("XORL_MOE_SGLANG_EP_COMBINE_SIM requires the fused shared-expert gate_up_proj")
        inter = self.shared_expert.intermediate_size
        if inter % ep_size != 0:
            raise ValueError(
                f"XORL_MOE_SGLANG_EP_COMBINE_SIM: shared_expert intermediate_size={inter} "
                f"not divisible by simulated ep_size={ep_size}"
            )
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        routing_flat = routing_weights.reshape(flat.shape[0], -1)
        selected_flat = selected_experts.reshape(flat.shape[0], -1)

        w_gu = self.shared_expert.gate_up_proj.weight  # [2I, H], gate rows first
        w_down = self.shared_expert.down_proj.weight  # [H, I]
        w_gate = self.shared_expert_gate.weight  # [1, H]
        shard = inter // ep_size
        # serving applies the sigmoid shared-expert gate on each rank's partial;
        # the scalar is identical across ranks, computed once here.
        gate_value = torch.sigmoid(matmul_persistent(flat, w_gate.t()))

        acc = None
        for rank in range(ep_size - 1, -1, -1):
            routed = self.experts.sglang_moe_ep_sim_routed_partial(flat, routing_flat, selected_flat, rank, ep_size).to(
                torch.bfloat16
            )
            lo = rank * shard
            w_slice = torch.cat((w_gu[lo : lo + shard], w_gu[inter + lo : inter + lo + shard]), dim=0).contiguous()
            gate_up = matmul_persistent(flat, w_slice.t())
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up  # torch-native bf16 (serving's BI-ops lane; NOT the fused kernel)
            down = matmul_persistent(act, w_down[:, lo : lo + shard].contiguous().t())
            partial = routed + (gate_value * down).to(torch.bfloat16)
            acc = partial if acc is None else acc + partial
        return acc.reshape(batch_size, sequence_length, hidden_dim)

    def forward_experts_only(self, hidden_states, routing_weights, selected_experts):
        """Sparse experts + shared expert with pre-computed routing."""
        ep_sim = moe_sglang_ep_combine_sim_size()
        if ep_sim > 0:
            return self._ep_combine_sim(hidden_states, routing_weights, selected_experts, ep_sim)
        expert_output = super().forward_experts_only(hidden_states, routing_weights, selected_experts)
        return expert_output + self._shared_expert(hidden_states)

    def forward(self, hidden_states: torch.Tensor):
        ep_sim = moe_sglang_ep_combine_sim_size()
        if ep_sim > 0:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            routing_weights, selected_experts, router_logits = self.route(hidden_states.view(-1, hidden_dim))
            out = self._ep_combine_sim(hidden_states, routing_weights, selected_experts, ep_sim)
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
            )
        else:
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)

        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        elif isinstance(module, MoEExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
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
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
