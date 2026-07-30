"""GLM-5 / GLM-5.1 model configuration."""

from transformers.configuration_utils import PretrainedConfig

from xorl.models.layers import rope_config_validation


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


class Glm5Config(PretrainedConfig):
    """Config for `zai-org/GLM-5` and `zai-org/GLM-5.1`.

    HF carries `model_type=glm_moe_dsa`; we rename to `xorl_glm5` so xorl owns
    the type once `_load_local_xorl_config` resolves the local config.
    """

    model_type = "xorl_glm5"
    base_model_tp_plan = {}
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        # GLM-5.1 defaults
        vocab_size=154880,
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        v_head_dim=256,
        qk_nope_head_dim=192,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        max_position_embeddings=202752,
        rope_theta=1_000_000.0,
        pad_token_id=154820,
        # DSA indexer
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        indexer_rope_interleave=True,
        # MTP
        num_nextn_predict_layers=1,
        # Bookkeeping
        ep_size=1,
        pretraining_tp=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        rope_scaling=None,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        _moe_implementation="eager",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.partial_rotary_factor = self.qk_rope_head_dim / self.qk_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self._moe_implementation = _moe_implementation
        self._rope_scaling = _cfg_to_dict(rope_scaling)

        if self._rope_scaling is not None and "type" in self._rope_scaling:
            self._rope_scaling["rope_type"] = self._rope_scaling["type"]
        rope_config_validation(self)

        # DSA indexer
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.indexer_rope_interleave = indexer_rope_interleave

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Bookkeeping fields HF carries on the original config
        self.ep_size = ep_size
        self.pretraining_tp = pretraining_tp

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope_scaling(self):
        return self._rope_scaling

    @rope_scaling.setter
    def rope_scaling(self, value):
        self._rope_scaling = _cfg_to_dict(value)

    @property
    def rope_parameters(self):
        rope_params = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        }
        if self._rope_scaling is not None:
            rope_params.update(self._rope_scaling)
            if "type" in rope_params and "rope_type" not in rope_params:
                rope_params["rope_type"] = rope_params.pop("type")
        return rope_params

    @rope_parameters.setter
    def rope_parameters(self, value):
        value_dict = _cfg_to_dict(value)
        if value_dict is not None and isinstance(value_dict, dict) and "rope_theta" in value_dict:
            self.rope_theta = value_dict["rope_theta"]
        self._rope_scaling = value_dict

    @classmethod
    def from_hf_config(cls, hf_config):
        text_config = getattr(hf_config, "text_config", hf_config)
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is None:
            rope_params = getattr(text_config, "rope_scaling", None)

        return cls(
            vocab_size=getattr(text_config, "vocab_size", 154880),
            hidden_size=getattr(text_config, "hidden_size", 6144),
            intermediate_size=getattr(text_config, "intermediate_size", 12288),
            moe_intermediate_size=getattr(text_config, "moe_intermediate_size", 2048),
            num_hidden_layers=getattr(text_config, "num_hidden_layers", 78),
            num_attention_heads=getattr(text_config, "num_attention_heads", 64),
            num_key_value_heads=getattr(
                text_config,
                "num_key_value_heads",
                getattr(text_config, "num_attention_heads", 64),
            ),
            n_shared_experts=getattr(text_config, "n_shared_experts", 1),
            n_routed_experts=getattr(text_config, "n_routed_experts", 256),
            routed_scaling_factor=getattr(text_config, "routed_scaling_factor", 2.5),
            kv_lora_rank=getattr(text_config, "kv_lora_rank", 512),
            q_lora_rank=getattr(text_config, "q_lora_rank", 2048),
            qk_rope_head_dim=getattr(text_config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(text_config, "v_head_dim", 256),
            qk_nope_head_dim=getattr(text_config, "qk_nope_head_dim", 192),
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 8),
            first_k_dense_replace=getattr(text_config, "first_k_dense_replace", 3),
            max_position_embeddings=getattr(text_config, "max_position_embeddings", 202752),
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-5),
            use_cache=getattr(text_config, "use_cache", True),
            pad_token_id=getattr(text_config, "pad_token_id", getattr(hf_config, "pad_token_id", 154820)),
            bos_token_id=getattr(text_config, "bos_token_id", getattr(hf_config, "bos_token_id", None)),
            eos_token_id=getattr(text_config, "eos_token_id", getattr(hf_config, "eos_token_id", None)),
            tie_word_embeddings=getattr(
                hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)
            ),
            rope_theta=_cfg_get(rope_params, "rope_theta", getattr(text_config, "rope_theta", 1_000_000.0)),
            rope_scaling=_cfg_to_dict(rope_params),
            rope_interleave=getattr(text_config, "rope_interleave", True),
            attention_bias=getattr(text_config, "attention_bias", False),
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            norm_topk_prob=getattr(text_config, "norm_topk_prob", True),
            topk_method=getattr(text_config, "topk_method", "noaux_tc"),
            scoring_func=getattr(text_config, "scoring_func", "sigmoid"),
            n_group=getattr(text_config, "n_group", 1),
            topk_group=getattr(text_config, "topk_group", 1),
            # GLM-specific
            index_head_dim=getattr(text_config, "index_head_dim", 128),
            index_n_heads=getattr(text_config, "index_n_heads", 32),
            index_topk=getattr(text_config, "index_topk", 2048),
            indexer_rope_interleave=getattr(text_config, "indexer_rope_interleave", True),
            num_nextn_predict_layers=getattr(text_config, "num_nextn_predict_layers", 1),
            ep_size=getattr(text_config, "ep_size", 1),
            pretraining_tp=getattr(text_config, "pretraining_tp", 1),
            architectures=["GlmMoeDsaForCausalLM"],
        )


__all__ = ["Glm5Config"]
