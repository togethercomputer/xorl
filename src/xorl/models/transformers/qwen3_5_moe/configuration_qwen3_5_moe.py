# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen3_5Moe model configuration"""

from transformers.configuration_utils import PretrainedConfig

from xorl.models.layers import rope_config_validation

from .parallelize import TP_PLAN


def _cfg_get(value, key, default=None):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _cfg_to_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items()}
    return value


def _split_mrope_fields(value):
    value_dict = _cfg_to_dict(value) or {}
    mrope_interleaved = value_dict.pop("mrope_interleaved", False)
    mrope_section = value_dict.pop("mrope_section", None)
    return value_dict or None, mrope_interleaved, mrope_section


class Qwen3_5MoeConfig(PretrainedConfig):
    model_type = "xorl_qwen3_5_moe"

    base_model_tp_plan = TP_PLAN
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        intermediate_size=512,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        layer_types=None,
        full_attention_interval=None,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        mrope_interleaved=False,
        mrope_section=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        _moe_implementation="triton",
        **kwargs,
    ):
        kwargs["ignore_keys_at_rope_validation"] = {"mrope_section", "mrope_interleaved"}
        kwargs.setdefault("partial_rotary_factor", 0.25)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = kwargs.pop("head_dim", hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        cleaned_rope_scaling, rope_mrope_interleaved, rope_mrope_section = _split_mrope_fields(rope_scaling)
        self.mrope_interleaved = mrope_interleaved or rope_mrope_interleaved
        self.mrope_section = mrope_section if mrope_section is not None else rope_mrope_section
        self.rope_parameters = cleaned_rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.full_attention_interval = full_attention_interval
        self.linear_num_key_heads = linear_num_key_heads if linear_num_key_heads is not None else num_attention_heads
        self.linear_num_value_heads = (
            linear_num_value_heads if linear_num_value_heads is not None else num_attention_heads
        )
        self.linear_key_head_dim = linear_key_head_dim if linear_key_head_dim is not None else 128
        self.linear_value_head_dim = (
            linear_value_head_dim if linear_value_head_dim is not None else self.linear_key_head_dim
        )
        self.attn_output_gate = attn_output_gate
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        if layer_types is None:
            if full_attention_interval:
                layer_types = [
                    "full_attention" if (layer_idx + 1) % full_attention_interval == 0 else "linear_attention"
                    for layer_idx in range(num_hidden_layers)
                ]
            else:
                layer_types = ["full_attention"] * num_hidden_layers
        self.layer_types = list(layer_types)
        self.shared_expert_intermediate_size = intermediate_size

        if self._rope_scaling is not None and "type" in self._rope_scaling:
            self._rope_scaling["rope_type"] = self._rope_scaling["type"]
        rope_config_validation(self, ignore_keys=kwargs.get("ignore_keys_at_rope_validation"))

        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers
        self._moe_implementation = _moe_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope_parameters(self):
        rope_params = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
        }
        if self._rope_scaling is not None:
            rope_params.update(self._rope_scaling)
            if "type" in rope_params and "rope_type" not in rope_params:
                rope_params["rope_type"] = rope_params.pop("type")
        return rope_params

    @rope_parameters.setter
    def rope_parameters(self, value):
        value_dict = _cfg_to_dict(value)
        if value_dict is not None and isinstance(value_dict, dict):
            if "rope_theta" in value_dict:
                self.rope_theta = value_dict["rope_theta"]
        cleaned_value, rope_mrope_interleaved, rope_mrope_section = _split_mrope_fields(value_dict)
        self.mrope_interleaved = rope_mrope_interleaved or getattr(self, "mrope_interleaved", False)
        if rope_mrope_section is not None:
            self.mrope_section = rope_mrope_section
        self._rope_scaling = cleaned_value

    @classmethod
    def from_hf_config(cls, hf_config):
        text_config = getattr(hf_config, "text_config", hf_config)
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is None:
            rope_params = getattr(text_config, "rope_scaling", None)

        hidden_size = getattr(text_config, "hidden_size")
        num_attention_heads = getattr(text_config, "num_attention_heads")
        head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)
        shared_expert_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", None)
        intermediate_size = shared_expert_intermediate_size
        if intermediate_size is None:
            intermediate_size = getattr(text_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(text_config, "moe_intermediate_size", 6144)

        return cls(
            vocab_size=getattr(text_config, "vocab_size", getattr(hf_config, "vocab_size", 151936)),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=getattr(text_config, "num_hidden_layers"),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=getattr(text_config, "num_key_value_heads"),
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            max_position_embeddings=getattr(text_config, "max_position_embeddings"),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            use_cache=getattr(text_config, "use_cache", False),
            tie_word_embeddings=getattr(
                hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)
            ),
            rope_theta=_cfg_get(rope_params, "rope_theta", getattr(text_config, "rope_theta", 10000.0)),
            rope_scaling=_cfg_to_dict(rope_params),
            attention_bias=getattr(text_config, "attention_bias", False),
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            layer_types=getattr(text_config, "layer_types", None),
            full_attention_interval=getattr(text_config, "full_attention_interval", None),
            linear_num_key_heads=getattr(text_config, "linear_num_key_heads", None),
            linear_num_value_heads=getattr(text_config, "linear_num_value_heads", None),
            linear_key_head_dim=getattr(text_config, "linear_key_head_dim", None),
            linear_value_head_dim=getattr(text_config, "linear_value_head_dim", None),
            attn_output_gate=getattr(text_config, "attn_output_gate", True),
            linear_conv_kernel_dim=getattr(text_config, "linear_conv_kernel_dim", 4),
            mrope_interleaved=_cfg_get(rope_params, "mrope_interleaved", False),
            mrope_section=_cfg_get(rope_params, "mrope_section", None),
            pad_token_id=getattr(text_config, "pad_token_id", getattr(hf_config, "pad_token_id", None)),
            bos_token_id=getattr(text_config, "bos_token_id", getattr(hf_config, "bos_token_id", None)),
            eos_token_id=getattr(text_config, "eos_token_id", getattr(hf_config, "eos_token_id", None)),
            decoder_sparse_step=1,
            moe_intermediate_size=getattr(text_config, "moe_intermediate_size", 768),
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 8),
            num_experts=getattr(text_config, "num_experts", 128),
            norm_topk_prob=getattr(text_config, "norm_topk_prob", False),
            output_router_logits=getattr(text_config, "output_router_logits", False),
            router_aux_loss_coef=getattr(text_config, "router_aux_loss_coef", 0.001),
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            head_dim=head_dim,
        )


__all__ = ["Qwen3_5MoeConfig"]
