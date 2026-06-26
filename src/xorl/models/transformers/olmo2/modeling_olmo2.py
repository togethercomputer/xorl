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
from xorl.models.layers.normalization import native_rms_norm
from xorl.models.layers.rope import apply_rotary_pos_emb
from xorl.models.module_utils import GradientCheckpointingLayer
from xorl.models.outputs import BaseModelOutput, CausalLMOutput
from xorl.models.transformers.olmo2 import parallelize
from xorl.models.transformers.olmo2.checkpoint_handler import Olmo2CheckpointHandler
from xorl.models.transformers.olmo2.configuration_olmo2 import Olmo2Config
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.utils import logging


logger = logging.get_logger(__name__)


class Olmo2MLP(nn.Module):
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


class Olmo2QKRMSNorm(RMSNorm):
    """Full-axis RMSNorm for OLMo-2 ``q_norm``/``k_norm`` under colwise TP.

    Without TP the parent ``forward`` runs unchanged. Under TP the OLMo-2 plan
    applies ``LocalAxisRMSNormShard`` to this module's weight, sharding it on
    dim 0 across the TP mesh. The colwise ``q_proj``/``k_proj`` produces a
    plain local tensor whose last dim already equals the rank's weight slice
    (``num_heads * head_dim / tp``). ``F.rms_norm`` has no DTensor sharding
    rule, so dispatch through ``__torch_function__`` would mis-handle the
    sharded weight; bypass it by running the fused op directly on the local
    weight tensor. The rank-local RMS this computes matches HuggingFace's
    ``Olmo2RMSNorm`` reference behavior under TP (a deliberate
    local-vs-global RMS approximation, since the partition axis IS the norm
    axis for this model).
    """

    def forward(self, hidden_states, residual=None, prenorm=False):
        from torch.distributed.tensor import DTensor  # noqa: PLC0415

        weight = self.weight
        if not isinstance(weight, DTensor):
            return super().forward(hidden_states, residual, prenorm)

        residual_out = None
        norm_input = hidden_states
        if residual is not None:
            residual_out = hidden_states + residual
            norm_input = residual_out

        local_weight = weight.to_local()
        out = native_rms_norm(norm_input, local_weight, self.variance_epsilon)
        if residual_out is not None and prenorm:
            return out, residual_out
        return out


class Olmo2Attention(MultiHeadAttention):
    """OLMo-2 attention.

    OLMo-2 normalizes Q and K across the full ``num_heads * head_dim`` axis
    (not per-head as in Qwen3). The base ``MultiHeadAttention`` allocates
    per-head q_norm/k_norm when ``use_qk_norm=True``; we set it to ``False``
    on the config seen by the base class and own the full-axis norms here.
    """

    def __init__(self, config, layer_idx: int):
        # Disable base-class per-head q_norm/k_norm; we install full-axis norms.
        config.use_qk_norm = False
        super().__init__(config, layer_idx)
        self.q_norm = Olmo2QKRMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Olmo2QKRMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

    def _init_sliding_window(self, config):
        return None

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

        # Full-axis QK norm before reshape (OLMo-2 specific).
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(hidden_shape)
        k = k.view(hidden_shape)
        v = v.view(hidden_shape)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if getattr(self.config, "_attention_cast_bf16", False):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        return q, k, v


class Olmo2DecoderLayer(GradientCheckpointingLayer):
    """OLMo-2 decoder layer with post-attention and post-feedforward norms.

    Unlike Llama's pre-norm, OLMo-2 normalizes after each sublayer and adds
    the residual afterwards. There is no ``input_layernorm`` -- the only
    norms are ``post_attention_layernorm`` and ``post_feedforward_layernorm``.
    """

    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Olmo2MLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        # Self attention -> post-norm -> residual add
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP -> post-norm -> residual add
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Olmo2PreTrainedModel(XorlPreTrainedModel):
    config_class = Olmo2Config
    base_model_prefix = "model"
    _no_split_modules = ["Olmo2DecoderLayer"]

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
        return Olmo2CheckpointHandler(
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            is_prequantized=is_prequantized,
            exclude_modules=exclude_modules,
            model=self if is_prequantized else None,
        )


class Olmo2Model(Olmo2PreTrainedModel):
    """OLMo-2 transformer decoder."""

    def __init__(self, config: Olmo2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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

        for decoder_layer in self.layers:
            if decoder_layer is None:  # PP: pruned layer
                continue
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
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


class Olmo2ForCausalLM(Olmo2PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        super().__init__(config)
        self.model = Olmo2Model(config)
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
        """Return PP module config for pipeline_module_split."""
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


ModelClass = Olmo2ForCausalLM

__all__ = ["Olmo2ForCausalLM", "Olmo2Model", "Olmo2PreTrainedModel"]
