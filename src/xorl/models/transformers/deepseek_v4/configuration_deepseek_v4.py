# Copyright 2026 The DeepSeek team and the HuggingFace Inc. team. All rights reserved.
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
"""DeepSeek V4 model configuration.

Vendored because ``transformers`` does not (yet) ship a ``DeepseekV4Config``.
Field names match the upstream HF ``config.json`` published at
``deepseek-ai/DeepSeek-V4-Flash``.
"""

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
        return dict(vars(value))
    return value


class DeepseekV4Config(PretrainedConfig):
    """HF-compatible config for DeepSeek-V4 Flash / Pro.

    Reproduces the upstream ``deepseek_v4`` config schema. ``head_dim`` is the
    MLA latent dim (= ``kv_lora_rank``); ``qk_rope_head_dim`` is the per-head
    RoPE slice that lives in the last dims of the KV stream. There is no
    separate K/V projection — V4 uses a single shared ``wkv`` down-projection.
    """

    # Matches the upstream HF Flash ``config.json`` ``model_type`` so
    # ``AutoConfig.from_pretrained`` dispatches to us. Other xorl models
    # use an ``xorl_`` prefix to namespace away from transformers, but
    # transformers does not (yet) ship a ``DeepseekV4Config`` so we own
    # this name. If upstream ever does ship one, this is the breaking
    # change that needs revisiting.
    model_type = "deepseek_v4"

    base_model_tp_plan = TP_PLAN
    base_model_pp_plan = None

    def __init__(
        self,
        # Standard transformer dims
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        max_position_embeddings=1048576,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        use_cache=True,
        # MLA
        q_lora_rank=1024,
        # Output projection factorization (V4-specific)
        o_groups=8,
        o_lora_rank=1024,
        # Sliding-window attention (window-only layers)
        sliding_window=128,
        # DSA indexer (used by ``compress_ratio == 4`` layers)
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        # MoE
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
        router_aux_loss_coef=0.0,
        # Hash routing — first ``num_hash_layers`` MoE layers use a frozen
        # ``tid2eid: [vocab_size, num_experts_per_tok]`` lookup instead of a
        # learned router.
        num_hash_layers=3,
        # HyperConnection — per-layer multi-stream residual mixing with
        # Sinkhorn normalization. ``hc_mult=4`` parallel streams per layer.
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        # Compressor / sparse attention
        compress_ratios=None,
        compress_rope_theta=160000.0,
        # SwiGLU clipping (training stabilizer): clamp pre-activation to
        # [-swiglu_limit, +swiglu_limit] when > 0.
        swiglu_limit=10.0,
        # Multi-Token Prediction layers (inference-only; ignored during
        # training-only V0).
        num_nextn_predict_layers=1,
        # Quantization
        quantization_config=None,
        expert_dtype=None,
        # xorl plumbing
        decoder_sparse_step=1,
        output_router_logits=False,
        mlp_only_layers=None,
        _moe_implementation="triton",
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.routed_scaling_factor = routed_scaling_factor
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_hash_layers = num_hash_layers

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.compress_ratios = list(compress_ratios) if compress_ratios is not None else None
        self.compress_rope_theta = compress_rope_theta

        self.swiglu_limit = swiglu_limit
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.quantization_config = quantization_config
        self.expert_dtype = expert_dtype

        self.decoder_sparse_step = decoder_sparse_step
        self.output_router_logits = output_router_logits
        self.mlp_only_layers = [] if mlp_only_layers is None else list(mlp_only_layers)
        self._moe_implementation = _moe_implementation

        # Compressor entries are required; if HF ships ``compress_ratios`` of
        # length ``num_hidden_layers + num_nextn_predict_layers`` (the on-disk
        # Flash config does), keep all of them — the MTP slot is consumed
        # later by the inference path.
        if self.compress_ratios is not None:
            expected = self.num_hidden_layers + (self.num_nextn_predict_layers or 0)
            if len(self.compress_ratios) not in (self.num_hidden_layers, expected):
                raise ValueError(
                    f"compress_ratios length {len(self.compress_ratios)} does not match "
                    f"num_hidden_layers={self.num_hidden_layers} (or +num_nextn_predict_layers={expected})"
                )
            for r in self.compress_ratios:
                if r not in (0, 4, 128):
                    raise ValueError(f"compress_ratios entries must be in {{0, 4, 128}}; got {r}")

        if self._rope_scaling is not None and "type" in self._rope_scaling:
            self._rope_scaling["rope_type"] = self._rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ---- rope_parameters property: same accessor pattern as qwen3_5_moe ----

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
        self._rope_scaling = value_dict

    # ---- Adapter from a raw HF DeepseekV4 config ----

    @classmethod
    def from_hf_config(cls, hf_config):
        text_config = getattr(hf_config, "text_config", hf_config)

        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is None:
            rope_params = getattr(text_config, "rope_scaling", None)

        return cls(
            vocab_size=getattr(text_config, "vocab_size", 129280),
            hidden_size=getattr(text_config, "hidden_size", 4096),
            num_hidden_layers=getattr(text_config, "num_hidden_layers", 43),
            num_attention_heads=getattr(text_config, "num_attention_heads", 64),
            num_key_value_heads=getattr(text_config, "num_key_value_heads", 1),
            head_dim=getattr(text_config, "head_dim", 512),
            qk_rope_head_dim=getattr(text_config, "qk_rope_head_dim", 64),
            max_position_embeddings=getattr(text_config, "max_position_embeddings", 1048576),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            attention_bias=getattr(text_config, "attention_bias", False),
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            rope_theta=_cfg_get(rope_params, "rope_theta", getattr(text_config, "rope_theta", 10000.0)),
            rope_scaling=_cfg_to_dict(rope_params),
            tie_word_embeddings=getattr(
                hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)
            ),
            use_cache=getattr(text_config, "use_cache", False),
            q_lora_rank=getattr(text_config, "q_lora_rank", 1024),
            o_groups=getattr(text_config, "o_groups", 8),
            o_lora_rank=getattr(text_config, "o_lora_rank", 1024),
            sliding_window=getattr(text_config, "sliding_window", 128),
            index_n_heads=getattr(text_config, "index_n_heads", 64),
            index_head_dim=getattr(text_config, "index_head_dim", 128),
            index_topk=getattr(text_config, "index_topk", 512),
            moe_intermediate_size=getattr(text_config, "moe_intermediate_size", 2048),
            n_routed_experts=getattr(text_config, "n_routed_experts", 256),
            n_shared_experts=getattr(text_config, "n_shared_experts", 1),
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 6),
            norm_topk_prob=getattr(text_config, "norm_topk_prob", True),
            scoring_func=getattr(text_config, "scoring_func", "sqrtsoftplus"),
            topk_method=getattr(text_config, "topk_method", "noaux_tc"),
            routed_scaling_factor=getattr(text_config, "routed_scaling_factor", 1.5),
            router_aux_loss_coef=getattr(text_config, "router_aux_loss_coef", 0.0),
            num_hash_layers=getattr(text_config, "num_hash_layers", 3),
            hc_mult=getattr(text_config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(text_config, "hc_sinkhorn_iters", 20),
            hc_eps=getattr(text_config, "hc_eps", 1e-6),
            compress_ratios=getattr(text_config, "compress_ratios", None),
            compress_rope_theta=getattr(text_config, "compress_rope_theta", 160000.0),
            swiglu_limit=getattr(text_config, "swiglu_limit", 10.0),
            num_nextn_predict_layers=getattr(text_config, "num_nextn_predict_layers", 1),
            quantization_config=getattr(text_config, "quantization_config", None),
            expert_dtype=getattr(text_config, "expert_dtype", None),
            pad_token_id=getattr(text_config, "pad_token_id", getattr(hf_config, "pad_token_id", None)),
            bos_token_id=getattr(text_config, "bos_token_id", getattr(hf_config, "bos_token_id", 0)),
            eos_token_id=getattr(text_config, "eos_token_id", getattr(hf_config, "eos_token_id", 1)),
            architectures=["DeepseekV4ForCausalLM"],
        )


__all__ = ["DeepseekV4Config"]
