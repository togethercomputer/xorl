"""MiniMax M3 text-backbone configuration."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _cfg_get(value: Any, key: str, default=None):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _cfg_to_dict(value: Any):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _cfg_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cfg_to_dict(v) for v in value]
    if hasattr(value, "to_dict"):
        return _cfg_to_dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return _cfg_to_dict(vars(value))
    return value


def _expand_flag_list(value, num_hidden_layers: int, *, default: int) -> list[int]:
    if value is None:
        return [default] * num_hidden_layers
    flags = list(value)
    if len(flags) != num_hidden_layers:
        raise ValueError(f"Expected {num_hidden_layers} layer flags, got {len(flags)}.")
    return [int(v) for v in flags]


class MiniMaxM3Config(PretrainedConfig):
    """xorl-owned config for the MiniMax M3 text model.

    HF publishes MiniMax M3 as a top-level multimodal
    ``model_type="minimax_m3_vl"`` config with the language model under
    ``text_config``. xorl currently implements the text backbone and registers
    the top-level architecture as an explicit text-only wrapper.
    """

    model_type = "xorl_minimax_m3"
    base_model_tp_plan = {}
    base_model_pp_plan = None

    def __init__(
        self,
        vocab_size: int = 200064,
        hidden_size: int = 6144,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        use_gemma_norm: bool = True,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        rope_theta: float = 5000000.0,
        rotary_dim: int = 64,
        partial_rotary_factor: float | None = None,
        hidden_act: str = "swigluoai",
        dense_intermediate_size: int = 12288,
        intermediate_size: int = 3072,
        shared_intermediate_size: int = 3072,
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "sigmoid",
        use_routing_bias: bool = True,
        routed_scaling_factor: float = 2.0,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        use_qk_norm: bool = True,
        qk_norm_type: str = "per_head",
        moe_layer_freq: list[int] | None = None,
        sparse_attention_config: dict[str, Any] | None = None,
        text_only: bool = True,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        vision_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.0,
        initializer_range: float = 0.02,
        _moe_implementation: str = "native",
        **kwargs,
    ):
        if partial_rotary_factor is None:
            partial_rotary_factor = rotary_dim / head_dim
        kwargs.setdefault("partial_rotary_factor", partial_rotary_factor)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_gemma_norm = use_gemma_norm
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.hidden_act = hidden_act
        self.dense_intermediate_size = dense_intermediate_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.moe_layer_freq = _expand_flag_list(moe_layer_freq, num_hidden_layers, default=1)
        self.sparse_attention_config = dict(sparse_attention_config or {"use_sparse_attention": False})
        self.sparse_attention_freq = _expand_flag_list(
            self.sparse_attention_config.get("sparse_attention_freq"),
            num_hidden_layers,
            default=0,
        )
        self.sparse_attention_config["sparse_attention_freq"] = self.sparse_attention_freq
        self.sparse_block_size = int(self.sparse_attention_config.get("sparse_block_size", 128))
        self.sparse_topk_blocks = int(self.sparse_attention_config.get("sparse_topk_blocks", 16))
        self.sparse_num_index_heads = int(self.sparse_attention_config.get("sparse_num_index_heads", 4))
        self.sparse_index_dim = int(self.sparse_attention_config.get("sparse_index_dim", head_dim))
        self.sparse_init_block = int(self.sparse_attention_config.get("sparse_init_block", 0))
        self.sparse_local_block = int(self.sparse_attention_config.get("sparse_local_block", 1))
        self.text_only = text_only
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.vision_config = vision_config
        self.text_config = text_config
        self.tie_word_embeddings = tie_word_embeddings
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.initializer_range = initializer_range
        self._moe_implementation = _moe_implementation
        self.use_cache = False

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope_parameters(self) -> dict[str, Any]:
        rope_params = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        }
        if getattr(self, "_rope_scaling", None) is not None:
            rope_params.update(self._rope_scaling)
            if "type" in rope_params and "rope_type" not in rope_params:
                rope_params["rope_type"] = rope_params.pop("type")
        return rope_params

    @rope_parameters.setter
    def rope_parameters(self, value):
        value_dict = _cfg_to_dict(value)
        if isinstance(value_dict, dict):
            if "rope_theta" in value_dict:
                self.rope_theta = value_dict["rope_theta"]
            if "partial_rotary_factor" in value_dict:
                self.partial_rotary_factor = value_dict["partial_rotary_factor"]
            self._rope_scaling = {k: v for k, v in value_dict.items() if k not in {"rope_theta"}}
        else:
            self._rope_scaling = None

    @classmethod
    def from_hf_config(cls, hf_config):
        hf_dict = _cfg_to_dict(hf_config) or {}
        text_config = _cfg_get(hf_config, "text_config", None) or hf_config
        text_dict = _cfg_to_dict(text_config) or {}
        sparse_cfg = _cfg_to_dict(_cfg_get(text_config, "sparse_attention_config", {}) or {})
        sparse_cfg = dict(sparse_cfg or {})
        architectures = _cfg_get(hf_config, "architectures", None) or _cfg_get(text_config, "architectures", None)
        if architectures is None:
            architectures = ["MiniMaxM3SparseForConditionalGeneration"]

        hidden_size = _cfg_get(text_config, "hidden_size")
        num_attention_heads = _cfg_get(text_config, "num_attention_heads")
        head_dim = _cfg_get(text_config, "head_dim", hidden_size // num_attention_heads)
        rotary_dim = _cfg_get(
            text_config, "rotary_dim", int(head_dim * _cfg_get(text_config, "partial_rotary_factor", 1.0))
        )

        return cls(
            vocab_size=_cfg_get(text_config, "vocab_size", _cfg_get(hf_config, "vocab_size", 200064)),
            hidden_size=hidden_size,
            num_hidden_layers=_cfg_get(text_config, "num_hidden_layers"),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=_cfg_get(text_config, "num_key_value_heads"),
            head_dim=head_dim,
            max_position_embeddings=_cfg_get(text_config, "max_position_embeddings", 1048576),
            rms_norm_eps=_cfg_get(text_config, "rms_norm_eps", 1e-6),
            use_gemma_norm=_cfg_get(text_config, "use_gemma_norm", True),
            attention_dropout=_cfg_get(text_config, "attention_dropout", 0.0),
            attention_bias=_cfg_get(text_config, "attention_bias", False),
            rope_theta=_cfg_get(text_config, "rope_theta", 5000000.0),
            rotary_dim=rotary_dim,
            partial_rotary_factor=_cfg_get(text_config, "partial_rotary_factor", rotary_dim / head_dim),
            hidden_act=_cfg_get(text_config, "hidden_act", "swigluoai"),
            dense_intermediate_size=_cfg_get(text_config, "dense_intermediate_size", 12288),
            intermediate_size=_cfg_get(text_config, "intermediate_size", 3072),
            shared_intermediate_size=_cfg_get(text_config, "shared_intermediate_size", 3072),
            num_local_experts=_cfg_get(text_config, "num_local_experts", _cfg_get(text_config, "num_experts", 128)),
            num_experts_per_tok=_cfg_get(text_config, "num_experts_per_tok", 4),
            n_shared_experts=_cfg_get(text_config, "n_shared_experts", 1),
            scoring_func=_cfg_get(text_config, "scoring_func", "sigmoid"),
            use_routing_bias=_cfg_get(text_config, "use_routing_bias", True),
            routed_scaling_factor=_cfg_get(text_config, "routed_scaling_factor", 2.0),
            swiglu_alpha=_cfg_get(text_config, "swiglu_alpha", 1.702),
            swiglu_limit=_cfg_get(text_config, "swiglu_limit", 7.0),
            use_qk_norm=_cfg_get(text_config, "use_qk_norm", True),
            qk_norm_type=_cfg_get(text_config, "qk_norm_type", "per_head"),
            moe_layer_freq=_cfg_get(text_config, "moe_layer_freq", None),
            sparse_attention_config=sparse_cfg,
            image_token_index=_cfg_get(hf_config, "image_token_index", 200025),
            video_token_index=_cfg_get(hf_config, "video_token_index", 200026),
            vision_config=_cfg_to_dict(_cfg_get(hf_config, "vision_config", None)),
            text_config=text_dict,
            pad_token_id=_cfg_get(text_config, "pad_token_id", _cfg_get(hf_config, "pad_token_id", None)),
            bos_token_id=_cfg_get(text_config, "bos_token_id", _cfg_get(hf_config, "bos_token_id", None)),
            eos_token_id=_cfg_get(text_config, "eos_token_id", _cfg_get(hf_config, "eos_token_id", None)),
            tie_word_embeddings=_cfg_get(
                hf_config, "tie_word_embeddings", _cfg_get(text_config, "tie_word_embeddings", False)
            ),
            output_router_logits=_cfg_get(text_config, "output_router_logits", False),
            router_aux_loss_coef=_cfg_get(text_config, "router_aux_loss_coef", 0.0),
            initializer_range=_cfg_get(text_config, "initializer_range", 0.02),
            architectures=list(architectures),
            torch_dtype=hf_dict.get("torch_dtype", text_dict.get("torch_dtype", None)),
        )


__all__ = ["MiniMaxM3Config"]
