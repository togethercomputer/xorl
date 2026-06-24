"""NVIDIA Nemotron-3-Ultra (nemotron_h) configuration.

Mirrors the transformers 5.5.3 ``NemotronHConfig`` field set, with the xorl
knobs sibling configs carry (``_moe_implementation``, router/aux-loss fields).

Note on ``time_step_limit``: transformers does **not** derive it from
``time_step_floor`` / ``time_step_min`` / ``time_step_max`` — it is an
independent config field (default ``(0.0, inf)``, legacy alias
``mamba_dt_limit``) consumed only by the HF cuda kernel path (``dt_limit``).
The xorl ``Mamba2Mixer`` follows the kernel semantics and clamps dt to
``time_step_limit`` on both sides.
"""

from transformers.configuration_utils import PretrainedConfig


_VALID_BLOCK_TYPES = ("mamba", "attention", "moe")


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"

    base_model_pp_plan = {
        "embeddings": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm_f": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        layers_block_type=None,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        intermediate_size=21504,
        mlp_hidden_act="relu2",
        mlp_bias=False,
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_head_dim=64,
        mamba_hidden_act="silu",
        n_groups=8,
        conv_kernel=4,
        expand=2,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_limit=(0.0, float("inf")),
        time_step_floor=1e-4,
        use_conv_bias=True,
        chunk_size=128,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=7688,
        moe_shared_expert_intermediate_size=7688,
        moe_latent_size=None,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        num_nextn_predict_layers=0,
        mtp_layers_block_type=None,
        use_bias=False,
        use_cache=False,
        num_logits_to_keep=1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        hidden_dropout=0.0,
        rescale_prenorm_residual=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        _moe_implementation="eager",
        **kwargs,
    ):
        if layers_block_type is None:
            layers_block_type = ["mamba", "moe", "attention", "moe"]
        layers_block_type = list(layers_block_type)
        invalid = set(layers_block_type) - set(_VALID_BLOCK_TYPES)
        if invalid:
            raise ValueError(
                f"`layers_block_type` contains invalid types: {invalid}. Must be one of: {_VALID_BLOCK_TYPES}"
            )
        # num_hidden_layers is derived from layers_block_type (HF behavior).
        kwargs.pop("num_hidden_layers", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_block_type = layers_block_type
        self.num_hidden_layers = len(layers_block_type)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.intermediate_size = intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_bias = mlp_bias
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.mamba_hidden_act = mamba_hidden_act
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_limit = tuple(time_step_limit)
        self.time_step_floor = time_step_floor
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_layers_block_type = list(mtp_layers_block_type) if mtp_layers_block_type else ["attention", "moe"]
        self.use_bias = use_bias
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rms_norm_eps = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = hidden_dropout
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self._moe_implementation = _moe_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_hf_config(cls, hf_config):
        """Build from a transformers ``NemotronHConfig`` (or a namespace/dict of config.json)."""
        text_config = getattr(hf_config, "text_config", hf_config)
        time_step_limit = getattr(text_config, "time_step_limit", None)
        if time_step_limit is None:
            time_step_limit = (0.0, float("inf"))

        return cls(
            vocab_size=getattr(text_config, "vocab_size", 131072),
            hidden_size=getattr(text_config, "hidden_size"),
            layers_block_type=list(getattr(text_config, "layers_block_type")),
            num_attention_heads=getattr(text_config, "num_attention_heads"),
            num_key_value_heads=getattr(text_config, "num_key_value_heads", None),
            head_dim=getattr(text_config, "head_dim", None),
            max_position_embeddings=getattr(text_config, "max_position_embeddings", 4096),
            attention_bias=getattr(text_config, "attention_bias", False),
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            sliding_window=getattr(text_config, "sliding_window", None),
            intermediate_size=getattr(text_config, "intermediate_size", 21504),
            mlp_hidden_act=getattr(text_config, "mlp_hidden_act", "relu2"),
            mlp_bias=getattr(text_config, "mlp_bias", False),
            ssm_state_size=getattr(text_config, "ssm_state_size", 128),
            mamba_num_heads=getattr(text_config, "mamba_num_heads", 128),
            mamba_head_dim=getattr(text_config, "mamba_head_dim", 64),
            mamba_hidden_act=getattr(text_config, "mamba_hidden_act", "silu"),
            n_groups=getattr(text_config, "n_groups", 8),
            conv_kernel=getattr(text_config, "conv_kernel", 4),
            expand=getattr(text_config, "expand", 2),
            time_step_min=getattr(text_config, "time_step_min", 0.001),
            time_step_max=getattr(text_config, "time_step_max", 0.1),
            time_step_limit=tuple(time_step_limit),
            time_step_floor=getattr(text_config, "time_step_floor", 1e-4),
            use_conv_bias=getattr(text_config, "use_conv_bias", True),
            chunk_size=getattr(text_config, "chunk_size", 128),
            n_routed_experts=getattr(text_config, "n_routed_experts", 8),
            n_shared_experts=getattr(text_config, "n_shared_experts", 1),
            moe_intermediate_size=getattr(text_config, "moe_intermediate_size", 7688),
            moe_shared_expert_intermediate_size=getattr(text_config, "moe_shared_expert_intermediate_size", 7688),
            moe_latent_size=getattr(text_config, "moe_latent_size", None),
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 2),
            routed_scaling_factor=getattr(text_config, "routed_scaling_factor", 1.0),
            n_group=getattr(text_config, "n_group", 1),
            topk_group=getattr(text_config, "topk_group", 1),
            norm_topk_prob=getattr(text_config, "norm_topk_prob", True),
            num_nextn_predict_layers=getattr(text_config, "num_nextn_predict_layers", 0),
            mtp_layers_block_type=getattr(text_config, "mtp_layers_block_type", None),
            use_bias=getattr(text_config, "use_bias", False),
            use_cache=getattr(text_config, "use_cache", False),
            num_logits_to_keep=getattr(text_config, "num_logits_to_keep", 1),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            layer_norm_epsilon=getattr(text_config, "layer_norm_epsilon", 1e-5),
            residual_in_fp32=getattr(text_config, "residual_in_fp32", False),
            hidden_dropout=getattr(text_config, "hidden_dropout", 0.0),
            rescale_prenorm_residual=getattr(text_config, "rescale_prenorm_residual", True),
            pad_token_id=getattr(text_config, "pad_token_id", getattr(hf_config, "pad_token_id", 0)),
            bos_token_id=getattr(text_config, "bos_token_id", getattr(hf_config, "bos_token_id", 1)),
            eos_token_id=getattr(text_config, "eos_token_id", getattr(hf_config, "eos_token_id", 2)),
            tie_word_embeddings=getattr(
                hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)
            ),
            output_router_logits=getattr(text_config, "output_router_logits", False),
            router_aux_loss_coef=getattr(text_config, "router_aux_loss_coef", 0.0),
            architectures=list(getattr(text_config, "architectures", None) or ["NemotronHForCausalLM"]),
        )


__all__ = ["NemotronHConfig"]
