from typing import Dict, List, Optional

from transformers import PretrainedConfig

from . import logging
from .device import get_device_name


logger = logging.get_logger(__name__)


def _attention_score_elements(batch_seqlens: List[int], causal: bool) -> int:
    """Return attended query-key elements across a batch of packed sequences.

    Decoder self-attention is lower-triangular, so the quadratic term is
    ``s * (s + 1) / 2`` per sequence instead of ``s * s``.
    """
    if causal:
        return sum(seqlen * (seqlen + 1) // 2 for seqlen in batch_seqlens)
    return sum(seqlen * seqlen for seqlen in batch_seqlens)


def _gc_multipliers(
    gc_enabled: bool,
    recompute_modules: Optional[List[str]],
    moe_checkpoint_method: Optional[str],
) -> Dict[str, int]:
    """Logical training FLOPs multipliers aligned with common benchmark reporting.

    We intentionally report model FLOPs for the training step itself:
    forward + backward for the active computation graph. We do not inflate the
    estimate for activation-checkpoint recompute or MoE implementation details,
    since those are runtime overheads rather than logical model FLOPs.

    This keeps the reported TFLOPS stable across checkpointing strategies and
    comparable with common training benchmark conventions.

    Returns a dict with keys:
        attn_linear  - Q/K/V/O projection FLOPs multiplier
        attn_qkv     - attention QK^T/SV training FLOPs multiplier per attended element
        router       - MoE gate router multiplier
        gate         - MoE gate_proj multiplier
        up           - MoE up_proj multiplier
        down         - MoE down_proj multiplier
        dense_mlp    - Dense MLP (non-MoE layers) multiplier
    """
    del gc_enabled, recompute_modules, moe_checkpoint_method
    return dict(attn_linear=6, attn_qkv=12, router=6, gate=6, up=6, down=6, dense_mlp=6)


def get_device_flops(unit="T"):
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = get_device_name()
    flops = float("inf")  # INF flops for unknown gpu type
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    elif "B200" in device_name:
        flops = 2250e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class XorlFlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = XorlFlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(batch_seqlens, delta_time)

    """

    def __init__(
        self,
        config: PretrainedConfig,
        gc_enabled: bool = False,
        recompute_modules: Optional[List[str]] = None,
        moe_checkpoint_method: Optional[str] = None,
        cp_size: int = 1,
    ):
        self._m = _gc_multipliers(gc_enabled, recompute_modules, moe_checkpoint_method)
        # CP correction: each rank computes on local_tokens = total/cp_size.
        # The formula uses local tokens → per-rank FLOPs. Multiply by cp_size
        # to report total-batch FLOPs so that EnvironMeter's
        # all_reduce(sum, dp) / world_size gives correct per-GPU TFlops.
        self._cp_size = cp_size
        self.estimate_func = {
            # the only difference between Qwen2 and Qwen2.5 for counting flops is the window attention
            # used in the ViT for Qwen2.5VL which is considered in the _estimate_qwen2_vl_flops function.
            "qwen2_5_vl": self._estimate_qwen2_vl_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "qwen3_moe": self._estimate_qwen3_moe_flops,
            "llama": self._estimate_llama_flops,
            # qwen3 reused _estimate_qwen2_flops func because the only model structure diff between qwen2 dense and qwen3 dense is that
            # qwen3 has additional RMSNorm layers for q and k.
            # RMSNorm layers have minimal impact at the MFU and can be ignored.
            "qwen3": self._estimate_qwen2_flops,
            "xorl_qwen3_5": self._estimate_qwen3_5_flops,
            "xorl_qwen3_5_moe": self._estimate_qwen3_5_moe_flops,
        }

        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        return 0

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts
        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        m = self._m

        # MLA attention linear params
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank
        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size

        router_N = hidden_size * moe_num_expert
        gate_up_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 2
        down_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num)
        dense_mlp_N = hidden_size * self.config.intermediate_size * 3
        embed_lm_N = vocab_size * hidden_size  # lm_head only; embedding is a lookup (0 FLOPs)

        moe_layer_flops = (
            m["router"] * router_N * tokens_sum
            + m["gate"] * gate_up_N * tokens_sum
            + m["down"] * down_N * tokens_sum
            + m["attn_linear"] * attn_linear_N * tokens_sum
        )
        dense_layer_flops = m["dense_mlp"] * dense_mlp_N * tokens_sum + m["attn_linear"] * attn_linear_N * tokens_sum
        dense_N_flops = (
            moe_layer_flops * (num_hidden_layers - first_k_dense_replace)
            + dense_layer_flops * first_k_dense_replace
            + 6 * embed_lm_N * tokens_sum
        )

        attn_score_elements = _attention_score_elements(batch_seqlens, causal=True)
        attn_qkv_flops = m["attn_qkv"] * attn_score_elements * q_head_dim * num_query_heads * num_hidden_layers

        return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12

    def _estimate_qwen3_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_num_expert = self.config.num_experts
        moe_topk = self.config.num_experts_per_tok

        m = self._m
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # Per-parameter counts (number of multiplications in one forward pass)
        router_N = hidden_size * moe_num_expert
        gate_up_N = hidden_size * moe_intermediate_size * moe_topk * 2  # gate_proj + up_proj
        down_N = hidden_size * moe_intermediate_size * moe_topk  # down_proj
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        embed_lm_N = vocab_size * hidden_size  # lm_head only; embedding is a lookup (0 FLOPs)

        # Per-layer FLOPs with GC-corrected multipliers
        per_layer_flops = (
            m["router"] * router_N * tokens_sum
            + m["gate"] * gate_up_N * tokens_sum
            + m["down"] * down_N * tokens_sum
            + m["attn_linear"] * attn_linear_N * tokens_sum
        )
        dense_N_flops = per_layer_flops * num_hidden_layers + 6 * embed_lm_N * tokens_sum

        # Attention QKV FLOPs (quadratic in sequence length)
        attn_score_elements = _attention_score_elements(batch_seqlens, causal=True)
        attn_qkv_flops = m["attn_qkv"] * attn_score_elements * head_dim * num_attention_heads * num_hidden_layers

        flops_all_token = dense_N_flops + attn_qkv_flops
        return flops_all_token / delta_time / 1e12

    def _get_qwen3_5_layer_counts(self):
        layer_types = getattr(self.config, "layer_types", [])
        full_attn_layers = sum(1 for lt in layer_types if lt == "full_attention")
        linear_attn_layers = sum(1 for lt in layer_types if lt == "linear_attention")
        if not layer_types:
            full_attn_layers = self.config.num_hidden_layers
            linear_attn_layers = 0
        return full_attn_layers, linear_attn_layers

    def _estimate_qwen3_5_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        m = self._m
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        full_attn_layers, linear_attn_layers = self._get_qwen3_5_layer_counts()

        q_size_full = num_attention_heads * head_dim * 2  # doubled for attention gate
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        o_size = num_attention_heads * head_dim

        full_attn_linear_N = hidden_size * (q_size_full + k_size + v_size + o_size)
        mlp_N = hidden_size * intermediate_size * 3
        embed_lm_N = vocab_size * hidden_size  # lm_head only; embedding is a lookup (0 FLOPs)

        full_attn_layer_flops = m["attn_linear"] * full_attn_linear_N * tokens_sum + m["dense_mlp"] * mlp_N * tokens_sum

        linear_layer_flops = 0
        if linear_attn_layers > 0:
            lin_key_dim = getattr(self.config, "linear_num_key_heads", num_attention_heads) * getattr(
                self.config, "linear_key_head_dim", 128
            )
            lin_value_dim = getattr(self.config, "linear_num_value_heads", num_attention_heads) * getattr(
                self.config, "linear_value_head_dim", 128
            )
            lin_num_v_heads = getattr(self.config, "linear_num_value_heads", num_attention_heads)
            linear_proj_N = hidden_size * (
                lin_key_dim  # q_proj
                + lin_key_dim  # k_proj
                + lin_value_dim  # v_proj
                + lin_num_v_heads  # a_proj
                + lin_num_v_heads  # b_proj
                + lin_value_dim  # g_proj (gate)
                + lin_value_dim  # o_proj
            )
            linear_layer_flops = m["attn_linear"] * linear_proj_N * tokens_sum + m["dense_mlp"] * mlp_N * tokens_sum

        dense_N_flops = (
            full_attn_layer_flops * full_attn_layers
            + linear_layer_flops * linear_attn_layers
            + 6 * embed_lm_N * tokens_sum
        )

        attn_score_elements = _attention_score_elements(batch_seqlens, causal=True)
        attn_qkv_flops = m["attn_qkv"] * attn_score_elements * head_dim * num_attention_heads * full_attn_layers

        return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12

    def _estimate_qwen3_5_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_num_expert = self.config.num_experts
        moe_topk = self.config.num_experts_per_tok

        m = self._m
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        full_attn_layers, linear_attn_layers = self._get_qwen3_5_layer_counts()

        q_size_full = num_attention_heads * head_dim * 2  # doubled for attention gate
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        o_size = num_attention_heads * head_dim
        full_attn_linear_N = hidden_size * (q_size_full + k_size + v_size + o_size)

        shared_intermediate = getattr(self.config, "shared_expert_intermediate_size", self.config.intermediate_size)
        router_N = hidden_size * moe_num_expert
        gate_up_N = hidden_size * moe_intermediate_size * moe_topk * 2
        down_N = hidden_size * moe_intermediate_size * moe_topk
        shared_gate_up_N = hidden_size * shared_intermediate * 2
        shared_down_N = hidden_size * shared_intermediate
        shared_gate_N = hidden_size

        moe_mlp_flops = (
            m["router"] * router_N * tokens_sum
            + m["gate"] * gate_up_N * tokens_sum
            + m["down"] * down_N * tokens_sum
            + m["gate"] * shared_gate_up_N * tokens_sum
            + m["down"] * shared_down_N * tokens_sum
            + 6 * shared_gate_N * tokens_sum
        )

        full_attn_layer_flops = m["attn_linear"] * full_attn_linear_N * tokens_sum + moe_mlp_flops

        linear_layer_flops = 0
        if linear_attn_layers > 0:
            lin_key_dim = getattr(self.config, "linear_num_key_heads", num_attention_heads) * getattr(
                self.config, "linear_key_head_dim", 128
            )
            lin_value_dim = getattr(self.config, "linear_num_value_heads", num_attention_heads) * getattr(
                self.config, "linear_value_head_dim", 128
            )
            lin_num_v_heads = getattr(self.config, "linear_num_value_heads", num_attention_heads)
            linear_proj_N = hidden_size * (
                lin_key_dim
                + lin_key_dim
                + lin_value_dim
                + lin_num_v_heads
                + lin_num_v_heads
                + lin_value_dim
                + lin_value_dim
            )
            linear_layer_flops = m["attn_linear"] * linear_proj_N * tokens_sum + moe_mlp_flops

        embed_lm_N = vocab_size * hidden_size  # lm_head only; embedding is a lookup (0 FLOPs)

        dense_N_flops = (
            full_attn_layer_flops * full_attn_layers
            + linear_layer_flops * linear_attn_layers
            + 6 * embed_lm_N * tokens_sum
        )

        attn_score_elements = _attention_score_elements(batch_seqlens, causal=True)
        attn_qkv_flops = m["attn_qkv"] * attn_score_elements * head_dim * num_attention_heads * full_attn_layers

        return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        m = self._m
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        embed_lm_N = vocab_size * hidden_size  # lm_head only; embedding is a lookup (0 FLOPs)

        per_layer_flops = m["dense_mlp"] * mlp_N * tokens_sum + m["attn_linear"] * attn_linear_N * tokens_sum
        dense_N_flops = per_layer_flops * num_hidden_layers + 6 * embed_lm_N * tokens_sum

        attn_score_elements = _attention_score_elements(batch_seqlens, causal=True)
        attn_qkv_flops = m["attn_qkv"] * attn_score_elements * head_dim * num_attention_heads * num_hidden_layers

        flops_all_token = dense_N_flops + attn_qkv_flops
        return flops_all_token / delta_time / 1e12

    def _estimate_llama_flops(self, tokens_sum, batch_seqlens, delta_time):
        # Llama and Qwen2 share the same dense-decoder formula.
        return self._estimate_qwen2_flops(tokens_sum, batch_seqlens, delta_time)

    def _estimate_qwen2_vl_flops(self, tokens_sum, batch_seqlens, delta_time, **kargs):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size  # lm_head only; embedding is a lookup (0 FLOPs)
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        attn_score_elements = _attention_score_elements(batch_seqlens, causal=True)
        attn_qkv_flops = 12 * attn_score_elements * head_dim * num_attention_heads * num_hidden_layers

        # vit flops
        image_seqlens = kargs.get("image_seqlens", None)
        if image_seqlens is not None:
            vit_flops = self._estimate_qwen_vit_flop(image_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen_vit_flop(self, image_seqlens, config):
        """
        Estimate the FLOPS of the vision encoder for Qwen2 and Qwen2.5
        """

        if config is None:
            return 0
        tokens_sum = sum(image_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        # In Qwen2 VL and Qwen2.5VL, the parameters naming are different:
        #
        # Parameter                 | Qwen2 VL         | Qwen2.5 VL
        # --------------------------|------------------|------------------
        # ViT hidden dimension      | embed_dim        | hidden_size
        # ViT output dimension      | hidden_size      | out_hidden_size
        # ViT MLP intermediate dim  | embed_dim * mlp_ratio | intermediate_size
        #
        # See https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
        # and https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json for an example.
        is_qwen2_vl = hasattr(config, "embed_dim")
        dim = config.embed_dim if is_qwen2_vl else config.hidden_size
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio) if is_qwen2_vl else config.intermediate_size
        out_hidden_size = config.hidden_size if is_qwen2_vl else config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size
        head_dim = dim // num_heads

        # Qwen 2.5 VL uses SiLU, thus 3.
        mlp_N = dim * mlp_hidden_dim * (2 if is_qwen2_vl else 3)
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        patch_embed_and_merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * depth + patch_embed_and_merger_N

        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # In Qwen2.5 VL, windowed attention is used in some layers.
        full_attn_layer_num = config.depth if is_qwen2_vl else len(config.fullatt_block_indexes)
        window_attn_layer_num = config.depth - full_attn_layer_num

        # full attn layer & all_token fwd & bwd flops
        attn_score_elements = _attention_score_elements(image_seqlens, causal=False)
        attn_qkv_flops = 12 * attn_score_elements * head_dim * num_heads * full_attn_layer_num

        # If window attention is used, add the window attention flops
        if window_attn_layer_num > 0:
            window_attn_compute_flops = 12 * tokens_sum * (config.window_size**2) * head_dim * num_heads
            attn_qkv_flops += window_attn_compute_flops * window_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    def estimate_flops(self, batch_seqlens, delta_time, **kwargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time, **kwargs)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
