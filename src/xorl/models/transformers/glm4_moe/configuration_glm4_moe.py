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
"""GLM-4 MoE model configuration"""

from transformers.configuration_utils import PretrainedConfig

from xorl.models.layers import rope_config_validation

from ....utils import logging
from .parallelize import TP_PLAN


logger = logging.get_logger(__name__)


class Glm4MoeConfig(PretrainedConfig):
    r"""
    Configuration class for the GLM-4 MoE model (GLM-4.5 / 4.6 / 4.7 variants).

    Defaults correspond to
    `THUDM/GLM-4-0414-A10B-Base <https://huggingface.co/THUDM/GLM-4-0414-A10B-Base>`_.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Dimension of the hidden representations.
        intermediate_size: MLP dimension for dense layers (first K layers).
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm.
        use_cache: Whether to return past key/value states.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base period for RoPE.
        rope_scaling: RoPE scaling configuration dict.
        partial_rotary_factor: Fraction of head dimensions that receive rotary embeddings.
        attention_bias: Whether to use bias in QKV projections.
        attention_dropout: Dropout rate for attention weights.
        moe_intermediate_size: Expert FFN intermediate dimension.
        num_experts_per_tok: Number of experts selected per token.
        n_shared_experts: Number of shared (dense) experts alongside routed experts.
        n_routed_experts: Total number of routed experts.
        routed_scaling_factor: Multiplicative factor applied to routed expert outputs.
        n_group: Number of expert groups for grouped top-k routing.
        topk_group: Number of groups selected per token before picking top-k within groups.
        first_k_dense_replace: First K layers use dense MLP instead of MoE.
        norm_topk_prob: Whether to renormalize top-k routing weights.
        use_qk_norm: Whether to apply per-head RMSNorm to Q and K.
        output_router_logits: Whether to return router logits from the model.
        router_aux_loss_coef: Coefficient for auxiliary load-balancing loss.
        _moe_implementation: MoE backend name (``"triton"``, ``"quack"``, ``"native"``, ``"eager"``).
    """

    model_type = "glm4_moe"

    base_model_tp_plan = TP_PLAN
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=10944,
        num_hidden_layers=46,
        num_attention_heads=96,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        attention_bias=False,
        attention_dropout=0.0,
        moe_intermediate_size=1408,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_routed_experts=128,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        use_qk_norm=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        _moe_implementation="triton",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self._rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm

        if self._rope_scaling is not None and "type" in self._rope_scaling:
            self._rope_scaling["rope_type"] = self._rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self._moe_implementation = _moe_implementation

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope_parameters(self):
        """Return rope parameters exposing partial_rotary_factor for RotaryEmbedding."""
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
        """Setter for rope_parameters to satisfy HF transformers 5.0+ configuration."""
        if value is not None and isinstance(value, dict):
            if "rope_theta" in value:
                self.rope_theta = value["rope_theta"]
        self._rope_scaling = value


__all__ = ["Glm4MoeConfig"]
