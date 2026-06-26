from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from xorl.distributed.moe.deepep import sync_pending_combine
from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.layers import ACT2FN, RMSNorm, RotaryEmbedding
from xorl.models.layers.attention import is_flash_attention, update_causal_mask
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.moe import MoEBlock
from xorl.models.layers.moe.routing_replay import get_replay_stage
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.deepseek_v3 import parallelize
from xorl.models.transformers.deepseek_v3.checkpoint_handler import DeepseekV3CheckpointHandler
from xorl.models.transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from xorl.models.transformers.deepseek_v3.support import (
    PACKED_EXPERT_DEFAULT_GROUP_SIZE,
    PACKED_EXPERT_DEFAULT_NUM_BITS,
    get_packed_expert_quantization_args,
    has_packed_expert_weights,
)
from xorl.models.transformers.qwen3_5_shared import qwen3_5_apply_rotary_pos_emb
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.utils import logging


logger = logging.get_logger(__name__)


class DeepseekV3MLP(nn.Module):
    def __init__(self, config: DeepseekV3Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self._use_fused_silu = config.hidden_act == "silu" and not getattr(config, "_activation_native", False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self._use_fused_silu:
            hidden_states = fused_silu_and_mul(torch.cat([gate, up], dim=-1))
        else:
            hidden_states = self.act_fn(gate) * up
        return self.down_proj(hidden_states)


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        if getattr(self.config, "_router_fp32", False):
            return F.linear(hidden_states.float(), self.weight.float())
        return F.linear(hidden_states, self.weight)


class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.qk_head_dim**-0.5
        rope_params = config.rope_parameters
        if rope_params.get("rope_type", "default") != "default":
            mscale_all_dim = rope_params.get("mscale_all_dim", 0)
            scaling_factor = rope_params.get("factor")
            if mscale_all_dim and scaling_factor is not None:
                mscale = 0.1 * mscale_all_dim * torch.log(torch.tensor(float(scaling_factor))).item() + 1.0
                self.scaling = self.scaling * mscale * mscale

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self._pad_value_for_flash = (
            is_flash_attention(config._attn_implementation) and self.qk_head_dim != self.v_head_dim
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, self.num_heads, self.qk_head_dim)
        kv_shape = (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(kv_shape)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(batch_size, seq_length, 1, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = qwen3_5_apply_rotary_pos_emb(
            q_rot,
            k_rot,
            cos,
            sin,
            interleaved=getattr(self.config, "rope_interleave", True),
        )
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if getattr(self.config, "_attention_cast_bf16", False):
            query_states = query_states.to(torch.bfloat16)
            key_states = key_states.to(torch.bfloat16)

        if self._pad_value_for_flash:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        return query_states, key_states, value_states

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self._pad_value_for_flash:
            attn_output = attn_output[..., : self.v_head_dim]
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        return self.o_proj(attn_output)

    def _get_attention_fn(self):
        return ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    def _attention_kwargs(self):
        return {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            "sliding_window": None,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        attn_output = attn_strategy.compute_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **kwargs,
        )
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


class DeepseekV3MoEBlock(MoEBlock):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=getattr(config, "_moe_implementation", "eager"),
            train_router=getattr(config, "train_router", False),
            record_routing_weights=getattr(config, "record_routing_weights", True),
        )
        self.config = config
        self.gate = DeepseekV3TopkRouter(config)
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.shared_experts = DeepseekV3MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

    def _route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_scores = router_logits.sigmoid()
        choice_scores = router_scores + self.gate.e_score_correction_bias.float()
        experts_per_group = self.num_experts // self.n_group
        group_topk = min(2, experts_per_group)
        group_scores = choice_scores.view(-1, self.n_group, experts_per_group).topk(group_topk, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=min(self.topk_group, self.n_group), dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, experts_per_group).reshape(-1, self.num_experts)
        scores_for_choice = choice_scores.masked_fill(~score_mask.bool(), 0.0)
        selected_experts = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        routing_weights = router_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(input_dtype), selected_experts

    def _regather_routing(
        self,
        router_logits: torch.Tensor,
        cached_experts: torch.Tensor,
        input_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = torch.gather(router_logits.sigmoid(), 1, cached_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return cached_experts, routing_weights.to(input_dtype)

    def forward(self, hidden_states: torch.Tensor):
        residuals = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        stage = get_replay_stage()
        replay = self._routing_replay

        if stage is not None and replay is not None:
            if stage == "record":
                with torch.no_grad():
                    _, selected_experts = self._route_tokens_to_experts(router_logits, flat_hidden_states.dtype)
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
            else:
                raise RuntimeError(f"Unsupported routing replay stage: {stage}")

            selected_experts, routing_weights = self._regather_routing(
                router_logits,
                selected_experts,
                flat_hidden_states.dtype,
            )

            if self.record_routing_weights:
                if stage == "record":
                    replay.record_weights(routing_weights)
                elif stage == "replay_backward":
                    cached_weights = replay.pop_backward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(flat_hidden_states.dtype)
                elif stage == "replay_forward":
                    cached_weights = replay.pop_forward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(flat_hidden_states.dtype)
        else:
            routing_weights, selected_experts = self._route_tokens_to_experts(router_logits, flat_hidden_states.dtype)

        ep_dispatch = getattr(self.experts, "ep_dispatch", "alltoall")
        if self.train_router and ep_dispatch == "deepep":
            raise AssertionError(
                "train_router=True is not supported with ep_dispatch='deepep'. "
                "DeepEP cannot propagate gradients through routing weights. "
                "Set train_router=False or switch to ep_dispatch='alltoall'."
            )
        if not self.train_router:
            routing_weights = routing_weights.detach()

        if self.moe_implementation == "eager":
            expert_output = self._eager_forward(flat_hidden_states, routing_weights, selected_experts)
        else:
            expert_output = self.experts(flat_hidden_states, routing_weights, selected_experts)

        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)
        shared_output = self.shared_experts(residuals)
        sync_pending_combine()
        return expert_output + shared_output, router_logits


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV3Attention(config, layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoEBlock(config)
        else:
            self.mlp = DeepseekV3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual=residual,
            prenorm=True,
        )

        if _selective and "mlp" in self._recompute_modules:
            hidden_states = self._gradient_checkpointing_func(self.mlp.__call__, hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


class DeepseekV3PreTrainedModel(XorlPreTrainedModel):
    config_class = DeepseekV3Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV3DecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekV3TopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, RotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq = module.inv_freq

    def get_parallel_plan(self):
        return parallelize.get_ep_plan()

    def get_checkpoint_handler(self, **kwargs):
        checkpoint_keys = kwargs.get("checkpoint_keys", set()) or set()
        packed_expert_num_bits = PACKED_EXPERT_DEFAULT_NUM_BITS
        packed_expert_group_size = PACKED_EXPERT_DEFAULT_GROUP_SIZE
        if checkpoint_keys and has_packed_expert_weights(checkpoint_keys):
            packed_quant_args = get_packed_expert_quantization_args(kwargs.get("weights_path", None))
            if packed_quant_args is not None:
                packed_expert_num_bits, packed_expert_group_size = packed_quant_args

        ep_rank = kwargs.get("ep_rank", 0)
        ep_size = kwargs.get("ep_size", 1)
        if kwargs.get("is_broadcast", False):
            ep_rank, ep_size = 0, 1

        return DeepseekV3CheckpointHandler(
            num_experts=self.config.n_routed_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            packed_expert_num_bits=packed_expert_num_bits,
            packed_expert_group_size=packed_expert_group_size,
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )


class DeepseekV3Model(DeepseekV3PreTrainedModel):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs,
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

        # Flash/SDPA mask internally — skip building one. Read the impl LAZILY (not cached in
        # __init__): build_foundation_model inits HF as flash_attention_2 and only restores the
        # real FA3/FA4 name on the config AFTER construction, so an __init__-time cache would be
        # wrongly False for flash backends (and re-enable the .item() causal-mask graph break).
        if is_flash_attention(self.config._attn_implementation):
            causal_mask = None
        else:
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

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        ps = get_parallel_state()
        position_embeddings = get_cp_strategy().prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_attention_heads,
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
            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states
        if output_hidden_states:
            _ = output_hidden_states

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs,
    ) -> MoeCausalLMOutput:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits or self.config.router_aux_loss_coef > 0
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        return MoeCausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )


ModelClass = DeepseekV3ForCausalLM


__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3Model",
    "DeepseekV3PreTrainedModel",
]
