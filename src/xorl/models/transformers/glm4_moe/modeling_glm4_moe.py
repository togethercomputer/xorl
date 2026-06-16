# Copyright 2025 The ZhipuAI Inc. team and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.distributed.moe.deepep import sync_pending_combine
from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.checkpoint_handlers.buffers import (
    checkpoint_has_per_expert_weights,
    detect_prequantized_block_fp8_checkpoint,
    detect_prequantized_checkpoint,
    get_prequantized_exclude_modules,
)
from xorl.models.layers import ACT2FN, RotaryEmbedding
from xorl.models.layers.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    is_flash_attention,
    update_causal_mask,
)
from xorl.models.layers.moe import MoEBlock
from xorl.models.layers.moe.routing_replay import get_replay_stage
from xorl.models.layers.normalization import compiled_eager_rms_norm
from xorl.models.layers.rope import apply_rotary_pos_emb
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.glm4_moe import parallelize
from xorl.models.transformers.glm4_moe.checkpoint_handler import Glm4MoeCheckpointHandler
from xorl.models.transformers.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.utils import logging


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm — hardcoded fp32 upcast to match HF Glm4MoeRMSNorm.
# Bypasses the global rmsnorm_mode switch because the bf16 path
# (F.rms_norm) compounds errors across GLM-4 MoE's 92+ norm layers
# and interacts pathologically with the triton/quack MoE kernels —
# observed as 23x logprob spikes vs SGLang/HF on borderline routing.
#
# The forward path delegates to ``compiled_eager_rms_norm`` so the 5
# pointwise/reduction kernels of the fp32-upcast eager path get fused
# into a single Inductor kernel — important because GLM-4 MoE has 92+
# norm layers per forward pass.
# ---------------------------------------------------------------------------


class Glm4MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return compiled_eager_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ---------------------------------------------------------------------------
# Dense MLP (used for first K layers + shared experts)
# ---------------------------------------------------------------------------


class Glm4MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
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


# ---------------------------------------------------------------------------
# Router gate with correction bias (sigmoid-based grouped top-k)
# ---------------------------------------------------------------------------


class Glm4MoeGate(nn.Module):
    """Router gate with e_score_correction_bias for GLM-4 MoE.

    Checkpoint paths:
    - ``mlp.gate.weight``           -> ``(n_routed_experts, hidden_size)``
    - ``mlp.gate.e_score_correction_bias`` -> ``(n_routed_experts,)``
    """

    def __init__(self, hidden_size: int, n_routed_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_routed_experts), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.float(), self.weight.float())


# ---------------------------------------------------------------------------
# MoE block with sigmoid grouped top-k routing + shared experts
# ---------------------------------------------------------------------------


class Glm4MoeSparseMoeBlock(MoEBlock):
    """GLM-4 MoE block: sigmoid routing, grouped top-k, correction bias, shared experts.

    Inherits from ``MoEBlock`` to get routing replay and ``moe_act`` gradient
    checkpointing support from ``XorlPreTrainedModel``.

    Overrides ``__init__`` and ``forward`` for GLM-specific routing.
    """

    def __init__(self, config: Glm4MoeConfig, moe_implementation: str = "triton"):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=moe_implementation,
        )
        self.config = config
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor

        # Replace the default nn.Linear gate with Glm4MoeGate (has correction bias)
        del self.gate
        self.gate = Glm4MoeGate(config.hidden_size, config.n_routed_experts)

        # Shared experts (dense MLP alongside routed experts)
        if config.n_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeMLP(config, intermediate_size=shared_intermediate)
        else:
            self.shared_experts = None

        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)

    def _route_tokens(self, router_logits: torch.Tensor, input_dtype: torch.dtype):
        """Sigmoid-based grouped top-k routing.

        1. Compute sigmoid scores; add correction bias for expert *selection* only
        2. Group experts, select top groups per token
        3. Select top-k experts within selected groups
        4. Gather routing weights from *raw* sigmoid scores (without bias)
        5. Optionally normalize and apply scaling factor
        """
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.gate.e_score_correction_bias.unsqueeze(0)

        if self.n_group > 1:
            scores_grouped = scores_for_choice.view(scores.shape[0], self.n_group, -1)
            group_scores = scores_grouped.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, -1, scores_grouped.shape[-1]).reshape(scores.shape[0], -1)
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        routing_weights = scores.gather(1, selected_experts)

        if self.router.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(input_dtype)
        return routing_weights, selected_experts

    def _regather_routing(self, router_logits, cached_experts, input_dtype):
        """Re-gather routing weights from raw sigmoid scores using cached expert indices.

        The correction bias is only used for expert *selection* (top-k), not for
        computing the routing weights themselves.
        """
        scores = router_logits.sigmoid()
        routing_weights = torch.gather(scores, 1, cached_experts)
        if self.router.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(input_dtype)
        return cached_experts, routing_weights

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)

        stage = get_replay_stage()
        replay = self._routing_replay

        if stage is not None and replay is not None:
            cached_weights = None
            if stage == "record":
                with torch.no_grad():
                    _, selected_experts = self._route_tokens(router_logits, hidden_states_flat.dtype)
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
                cached_weights = replay.pop_forward_weights()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
                cached_weights = replay.pop_backward_weights()

            if cached_weights is not None:
                routing_weights = cached_weights.to(hidden_states_flat.dtype)
            else:
                selected_experts, routing_weights = self._regather_routing(
                    router_logits, selected_experts, hidden_states_flat.dtype
                )
        else:
            routing_weights, selected_experts = self._route_tokens(router_logits, hidden_states_flat.dtype)

        if not self.train_router:
            routing_weights = routing_weights.detach()

        # Expert computation
        if self.moe_implementation == "eager":
            routed_output = self._eager_forward(hidden_states_flat, routing_weights, selected_experts)
        else:
            routed_output = self.experts(hidden_states_flat, routing_weights, selected_experts)

        # Shared expert computation
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
            final_hidden_states = routed_output + shared_output
        else:
            final_hidden_states = routed_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# ---------------------------------------------------------------------------
# Attention with partial rotary embeddings
# ---------------------------------------------------------------------------


class Glm4MoeAttention(MultiHeadAttention):
    """GLM-4 MoE attention with partial rotary position embeddings.

    GLM uses ``partial_rotary_factor=0.5``: only the first half of each head
    dimension receives RoPE. The ``o_proj`` never has bias.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # GLM's o_proj has no bias regardless of attention_bias setting
        if self.o_proj.bias is not None:
            self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # QK norm is controlled by config
        if not getattr(config, "use_qk_norm", False):
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if hasattr(self, "qkv_proj"):
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        q = self.q_norm(q.view(hidden_shape))
        k = self.k_norm(k.view(hidden_shape))
        v = v.view(hidden_shape)

        # Partial rotary: only apply RoPE to the first `rotary_dim` dimensions
        cos, sin = position_embeddings
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        if getattr(self.config, "_attention_cast_bf16", False):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        return q, k, v


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

GLM4_MOE_CLASSES = {
    "eager": lambda config: Glm4MoeSparseMoeBlock(config, moe_implementation="eager"),
    "triton": lambda config: Glm4MoeSparseMoeBlock(config, moe_implementation="triton"),
    "native": lambda config: Glm4MoeSparseMoeBlock(config, moe_implementation="native"),
    "quack": lambda config: Glm4MoeSparseMoeBlock(config, moe_implementation="quack"),
}


class Glm4MoeDecoderLayer(nn.Module):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Glm4MoeAttention(config, layer_idx)

        self.input_layernorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx >= config.first_k_dense_replace:
            moe_implementation = getattr(config, "_moe_implementation", "triton")
            self.mlp = GLM4_MOE_CLASSES[moe_implementation](config)
        else:
            self.mlp = Glm4MoeMLP(config, intermediate_size=config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.FloatTensor, ...]:
        _selective = (
            self.training
            and getattr(self, "gradient_checkpointing", False)
            and getattr(self, "_recompute_modules", None) is not None
        )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if _selective and "self_attn" in self._recompute_modules:
            hidden_states, self_attn_weights = self._gradient_checkpointing_func(
                self.self_attn.__call__,
                hidden_states,
                position_embeddings,
                attention_mask,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if _selective and "mlp" in self._recompute_modules:
            hidden_states = self._gradient_checkpointing_func(
                self.mlp.__call__,
                hidden_states,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        sync_pending_combine()

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# ---------------------------------------------------------------------------
# PreTrainedModel / Model / ForCausalLM
# ---------------------------------------------------------------------------


class Glm4MoePreTrainedModel(XorlPreTrainedModel):
    config_class = Glm4MoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["Glm4MoeDecoderLayer"]

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
        elif isinstance(module, Glm4MoeRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Glm4MoeGate):
            nn.init.kaiming_uniform_(module.weight)
            module.e_score_correction_bias.data.zero_()
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
        is_prequantized = detect_prequantized_checkpoint(weights_path) or detect_prequantized_block_fp8_checkpoint(
            weights_path
        )

        exclude_modules = getattr(self, "_qlora_exclude_modules", None)
        if exclude_modules is None:
            exclude_modules = get_prequantized_exclude_modules(weights_path) if is_prequantized else set()

        if is_broadcast:
            ep_rank, ep_size = 0, 1

        unfused = getattr(self, "_unfused_for_tp", False)

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        return Glm4MoeCheckpointHandler(
            num_experts=self.config.n_routed_experts,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            ep_rank=ep_rank,
            ep_size=ep_size,
            checkpoint_has_per_expert=has_per_expert,
            skip_qkv_merge=unfused,
            skip_gate_up_merge=unfused,
            is_prequantized=is_prequantized,
            exclude_modules=exclude_modules,
            device=kwargs.get("device"),
            model=self if is_prequantized else None,
            num_hidden_layers=self.config.num_hidden_layers,
        )


class Glm4MoeModel(Glm4MoePreTrainedModel):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm4MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self._skip_causal_mask = is_flash_attention(config._attn_implementation)

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> MoeModelOutput:
        output_attentions = (
            output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
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

        if self._skip_causal_mask:
            causal_mask = None
        else:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            causal_mask = update_causal_mask(
                self.config._attn_implementation,
                attention_mask,
                hidden_states,
                cache_position,
                sliding_window=getattr(self.config, "sliding_window", None),
                is_training=self.training,
                output_attentions=output_attentions,
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        ps = get_parallel_state()
        position_embeddings = get_cp_strategy(num_kv_heads=self.config.num_key_value_heads).prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_key_value_heads,
        )

        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if decoder_layer is None:
                continue
            _use_outer_checkpoint = (
                self.gradient_checkpointing and self.training and getattr(self, "_recompute_modules", None) is None
            )

            if _use_outer_checkpoint:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    output_attentions,
                    output_router_logits,
                    position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class KwargsForCausalLM(AttentionKwargs): ...


class Glm4MoeForCausalLM(Glm4MoePreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm4MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.001)
        self.num_experts = config.n_routed_experts
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
        output_router_logits = getattr(self.config, "output_router_logits", False)

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
        )


ModelClass = Glm4MoeForCausalLM


__all__ = [
    "Glm4MoeForCausalLM",
    "Glm4MoeModel",
    "Glm4MoePreTrainedModel",
    "Glm4MoeSparseMoeBlock",
    "Glm4MoeGate",
    "Glm4MoeMLP",
    "Glm4MoeAttention",
]
