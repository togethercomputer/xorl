"""Qwen2 model configuration for xorl.

This keeps xorl's TP/PP metadata while preserving HuggingFace Qwen2/Qwen2.5
semantics: fused internal projections, no Q/K RMSNorm, biased QKV, and a
bias-free output projection.
"""

from transformers.configuration_utils import PretrainedConfig

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
        return dict(vars(value))
    return value


class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"

    base_model_tp_plan = TP_PLAN
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=True,
        use_qk_norm=False,
        use_sliding_window=False,
        sliding_window=131072,
        max_window_layers=48,
        attention_dropout=0.0,
        layer_types=None,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151643,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["Qwen2ForCausalLM"])

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout

        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]
        if rope_scaling is not None and "rope_theta" in rope_scaling:
            rope_scaling = dict(rope_scaling)
            self.rope_theta = rope_scaling.pop("rope_theta")
            if not rope_scaling or rope_scaling == {"rope_type": "default"}:
                rope_scaling = None
        else:
            self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm

        if layer_types is None:
            layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and layer_idx >= self.max_window_layers
                else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        self.layer_types = list(layer_types)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_hf_config(cls, hf_config):
        text_config = getattr(hf_config, "text_config", hf_config)
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is None:
            rope_params = getattr(text_config, "rope_scaling", None)

        hidden_size = getattr(text_config, "hidden_size")
        num_attention_heads = getattr(text_config, "num_attention_heads")
        head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)

        return cls(
            vocab_size=getattr(text_config, "vocab_size", getattr(hf_config, "vocab_size", 152064)),
            hidden_size=hidden_size,
            intermediate_size=getattr(text_config, "intermediate_size"),
            num_hidden_layers=getattr(text_config, "num_hidden_layers"),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=getattr(text_config, "num_key_value_heads", num_attention_heads),
            head_dim=head_dim,
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            max_position_embeddings=getattr(text_config, "max_position_embeddings"),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            use_cache=getattr(text_config, "use_cache", True),
            tie_word_embeddings=getattr(
                hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)
            ),
            rope_theta=_cfg_get(rope_params, "rope_theta", getattr(text_config, "rope_theta", 1000000.0)),
            rope_scaling=_cfg_to_dict(rope_params),
            attention_bias=getattr(text_config, "attention_bias", True),
            use_qk_norm=False,
            use_sliding_window=getattr(text_config, "use_sliding_window", False),
            sliding_window=getattr(text_config, "sliding_window", None),
            max_window_layers=getattr(text_config, "max_window_layers", getattr(text_config, "num_hidden_layers")),
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            layer_types=getattr(text_config, "layer_types", None),
            pad_token_id=getattr(text_config, "pad_token_id", getattr(hf_config, "pad_token_id", None)),
            bos_token_id=getattr(text_config, "bos_token_id", getattr(hf_config, "bos_token_id", None)),
            eos_token_id=getattr(text_config, "eos_token_id", getattr(hf_config, "eos_token_id", None)),
            architectures=getattr(
                hf_config, "architectures", getattr(text_config, "architectures", ["Qwen2ForCausalLM"])
            ),
            model_type=getattr(text_config, "model_type", getattr(hf_config, "model_type", "qwen2")),
        )


__all__ = ["Qwen2Config"]
