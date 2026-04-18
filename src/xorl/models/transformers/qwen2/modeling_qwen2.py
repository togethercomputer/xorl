"""Qwen2 dense model for xorl.

Keeps fused internal projections for performance while preserving HuggingFace
Qwen2/Qwen2.5 semantics: biased fused QKV, bias-free output projection, and
no Q/K RMSNorm.
"""

from typing import Optional, Tuple, Unpack

import torch
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.checkpoint_handlers.buffers import (
    detect_prequantized_block_fp8_checkpoint,
    detect_prequantized_checkpoint,
    get_prequantized_exclude_modules,
)
from xorl.models.layers import ACT2FN, RMSNorm, RotaryEmbedding
from xorl.models.layers.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    is_flash_attention,
    update_causal_mask,
)
from xorl.models.module_utils import GradientCheckpointingLayer
from xorl.models.outputs import BaseModelOutput, CausalLMOutput
from xorl.models.transformers.qwen2 import parallelize
from xorl.models.transformers.qwen2.checkpoint_handler import Qwen2CheckpointHandler
from xorl.models.transformers.qwen2.configuration_qwen2 import Qwen2Config
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul


_RUNTIME_CONFIG_ATTRS = {
    "_attn_implementation",
    "_attention_cast_bf16",
    "_commit_hash",
    "_deepep_async_combine",
    "_deepep_buffer_size_gb",
    "_deepep_num_sms",
    "_ep_dispatch",
    "_lm_head_fp32",
    "_name_or_path",
    "_qlora_exclude_modules",
    "_rmsnorm_mode",
    "_rope_native",
    "_router_fp32",
    "_activation_native",
    "train_router",
}


def _copy_runtime_config_attrs(source, target):
    for name in _RUNTIME_CONFIG_ATTRS:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _adapt_qwen2_config(config):
    if isinstance(config, Qwen2Config):
        return config
    if hasattr(config, "text_config") or getattr(config, "model_type", None) == "qwen2":
        adapted = Qwen2Config.from_hf_config(config)
        _copy_runtime_config_attrs(config, adapted)
        return adapted
    return config


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self._use_fused_silu = config.hidden_act == "silu" and not getattr(config, "_activation_native", False)

    def unfuse_for_tp(self):
        """Replace fused gate_up_proj with separate gate_proj and up_proj for tensor parallelism."""
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


class Qwen2Attention(MultiHeadAttention):
    """Qwen2 attention with fused internal QKV projection and per-layer sliding window control."""

    def _init_sliding_window(self, config):
        layer_types = getattr(config, "layer_types", None)
        self.layer_type = layer_types[self.layer_idx] if layer_types is not None else None
        # When layer_types is present, treat it as authoritative (matches HF behaviour).
        if self.layer_type is not None:
            return config.sliding_window if self.layer_type == "sliding_attention" else None
        # Fallback for configs without layer_types.
        if (
            config.use_sliding_window
            and getattr(config, "sliding_window", None) is not None
            and self.layer_idx >= config.max_window_layers
        ):
            return config.sliding_window
        return None


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

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
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2PreTrainedModel(XorlPreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2DecoderLayer"]

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
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, RotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq = module.inv_freq

    def get_checkpoint_handler(self, **kwargs):
        if getattr(self, "_unfused_for_tp", False):
            return None

        weights_path = kwargs.get("weights_path", None)
        is_prequantized = detect_prequantized_checkpoint(weights_path)
        if not is_prequantized:
            is_prequantized = detect_prequantized_block_fp8_checkpoint(weights_path)

        exclude_modules = getattr(self, "_qlora_exclude_modules", None)
        if exclude_modules is None:
            exclude_modules = get_prequantized_exclude_modules(weights_path) if is_prequantized else set()

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        return Qwen2CheckpointHandler(
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            is_prequantized=is_prequantized,
            exclude_modules=exclude_modules,
            model=self if is_prequantized else None,
        )


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        config = _adapt_qwen2_config(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self._skip_causal_mask = is_flash_attention(config._attn_implementation)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[AttentionKwargs],
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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

        if self._skip_causal_mask:
            causal_mask_mapping = None
        else:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            mask_kwargs = dict(
                attention_mask=attention_mask,
                input_tensor=hidden_states,
                cache_position=cache_position,
                is_training=self.training,
                output_attentions=output_attentions,
            )
            causal_mask_mapping = {
                "full_attention": update_causal_mask(
                    self.config._attn_implementation, sliding_window=None, **mask_kwargs
                ),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = update_causal_mask(
                    self.config._attn_implementation, sliding_window=self.config.sliding_window, **mask_kwargs
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

        for idx, decoder_layer in enumerate(self.layers):
            if decoder_layer is None:
                continue
            layer_mask = causal_mask_mapping[self.config.layer_types[idx]] if causal_mask_mapping is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(AttentionKwargs): ...


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        config = _adapt_qwen2_config(config)
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def unfuse_for_tp(self):
        """Unfuse all fused projections for tensor parallelism compatibility."""
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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state

        return CausalLMOutput(last_hidden_state=last_hidden_state)


ModelClass = Qwen2ForCausalLM

__all__ = ["Qwen2ForCausalLM", "Qwen2Model", "Qwen2PreTrainedModel"]
