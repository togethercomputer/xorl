"""GPT-OSS model configuration."""

from transformers.configuration_utils import PretrainedConfig

from .parallelize import TP_PLAN


class GptOssConfig(PretrainedConfig):
    r"""
    Configuration class for GPT-OSS models (e.g. ``openai/gpt-oss-20b``).

    GPT-OSS is a Mixture-of-Experts transformer with:
    - SwiGLU activation with clamping (``swiglu_limit``)
    - Attention sinks (learned per-head bias added to softmax)
    - Alternating sliding-window / full attention layers
    - Custom YaRN-style RoPE with NTK-by-parts scaling
    - Expert biases on both gate_up (mlp1) and down (mlp2) projections
    - Router gate with bias

    Args:
        vocab_size (``int``, defaults to 201088):
            Vocabulary size.
        hidden_size (``int``, defaults to 2880):
            Hidden dimension.
        moe_intermediate_size (``int``, defaults to 2880):
            Expert FFN intermediate dimension.
        num_hidden_layers (``int``, defaults to 24):
            Number of transformer layers.
        num_attention_heads (``int``, defaults to 64):
            Number of query attention heads.
        num_key_value_heads (``int``, defaults to 8):
            Number of key/value heads for GQA.
        head_dim (``int``, defaults to 64):
            Dimension per attention head.
        rms_norm_eps (``float``, defaults to 1e-5):
            Epsilon for RMS normalization.
        num_experts (``int``, defaults to 32):
            Total number of MoE experts.
        num_experts_per_tok (``int``, defaults to 4):
            Number of experts activated per token.
        norm_topk_prob (``bool``, defaults to True):
            Whether to renormalize top-k routing weights.
        swiglu_limit (``float``, defaults to 7.0):
            Clamp limit for SwiGLU activation inputs.
        hidden_act (``str``, defaults to ``"silu"``):
            Base activation function name.
        attention_bias (``bool``, defaults to True):
            Whether QKV and output projections have bias terms.
        attention_dropout (``float``, defaults to 0.0):
            Dropout ratio for attention weights.
        sliding_window (``int``, defaults to 128):
            Sliding window size (applied to even-indexed layers).
        rope_theta (``float``, defaults to 150000.0):
            Base period for RoPE embeddings.
        initial_context_length (``int``, defaults to 4096):
            Original pre-training context length (used by YaRN RoPE).
        rope_scaling_factor (``float``, defaults to 32.0):
            Context extension scaling factor for YaRN RoPE.
        rope_ntk_alpha (``float``, defaults to 1.0):
            NTK-by-parts alpha (high-frequency boundary).
        rope_ntk_beta (``float``, defaults to 32.0):
            NTK-by-parts beta (low-frequency boundary).
        max_position_embeddings (``int``, defaults to 131072):
            Maximum sequence length the model can handle.
    """

    model_type = "gpt_oss"

    base_model_tp_plan = TP_PLAN

    def __init__(
        self,
        vocab_size=201088,
        hidden_size=2880,
        moe_intermediate_size=2880,
        num_hidden_layers=24,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=False,
        # MoE
        num_experts=32,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        # Activation
        swiglu_limit=7.0,
        hidden_act="silu",
        # Attention
        attention_bias=True,
        attention_dropout=0.0,
        sliding_window=128,
        # RoPE
        rope_theta=150000.0,
        initial_context_length=4096,
        rope_scaling_factor=32.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=32.0,
        max_position_embeddings=131072,
        # Internal
        _moe_implementation="native",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings

        # MoE
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self._moe_implementation = _moe_implementation

        # Activation
        self.swiglu_limit = swiglu_limit
        self.hidden_act = hidden_act

        # Attention
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window

        # RoPE — custom YaRN with NTK-by-parts
        self.rope_theta = rope_theta
        self.initial_context_length = initial_context_length
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    @staticmethod
    def _resolve_rope_params(hf_config):
        """Extract RoPE params from either flat fields or HF nested rope_scaling dict."""
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            return dict(
                initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
                rope_scaling_factor=rope_scaling.get("factor", 32.0),
                rope_ntk_alpha=rope_scaling.get("beta_slow", 1.0),
                rope_ntk_beta=rope_scaling.get("beta_fast", 32.0),
            )
        return dict(
            initial_context_length=getattr(hf_config, "initial_context_length", 4096),
            rope_scaling_factor=getattr(hf_config, "rope_scaling_factor", 32.0),
            rope_ntk_alpha=getattr(hf_config, "rope_ntk_alpha", 1.0),
            rope_ntk_beta=getattr(hf_config, "rope_ntk_beta", 32.0),
        )

    @classmethod
    def from_hf_config(cls, hf_config):
        """Build a GptOssConfig from an HF config dict/namespace.

        The GPT-OSS HF config uses non-standard field names (e.g.
        ``experts_per_token`` instead of ``num_experts_per_tok``,
        ``intermediate_size`` for the expert FFN dimension).
        """
        return cls(
            vocab_size=getattr(hf_config, "vocab_size", 201088),
            hidden_size=getattr(hf_config, "hidden_size", 2880),
            moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", None)
            or getattr(hf_config, "intermediate_size", 2880),
            num_hidden_layers=getattr(hf_config, "num_hidden_layers", 24),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 64),
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", 8),
            head_dim=getattr(hf_config, "head_dim", 64),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            use_cache=getattr(hf_config, "use_cache", False),
            # HF uses "num_local_experts", original uses "num_experts"
            num_experts=getattr(hf_config, "num_experts", None) or getattr(hf_config, "num_local_experts", 32),
            # HF uses "num_experts_per_tok", original uses "experts_per_token"
            num_experts_per_tok=getattr(hf_config, "experts_per_token", None)
            or getattr(hf_config, "num_experts_per_tok", 4),
            swiglu_limit=getattr(hf_config, "swiglu_limit", 7.0),
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
            sliding_window=getattr(hf_config, "sliding_window", 128),
            rope_theta=getattr(hf_config, "rope_theta", 150000.0),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 131072),
            # Support both flat fields (original) and nested rope_scaling dict (HF).
            # The HF config stores rope params in a nested dict; the original uses flat fields.
            **cls._resolve_rope_params(hf_config),
            # GPT-OSS always has attention bias (nn.Linear defaults)
            attention_bias=getattr(hf_config, "attention_bias", True),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            _moe_implementation=getattr(hf_config, "_moe_implementation", "native"),
            architectures=getattr(hf_config, "architectures", ["GptOssForCausalLM"]),
            output_router_logits=getattr(hf_config, "output_router_logits", False),
        )


__all__ = ["GptOssConfig"]
