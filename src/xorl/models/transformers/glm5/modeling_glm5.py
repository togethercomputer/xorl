"""GLM-5 / GLM-5.1 (`GlmMoeDsaForCausalLM`) modeling."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from xorl.distributed.canonical_moe import (
    CANONICAL_MOE_REDUCE_VERSION,
    CanonicalMoEGraphMetadata,
    CanonicalMoETransport,
    LocalMoEContribution,
    LogicalRowOwnership,
    OutputDistribution,
    ParallelPlan,
    canonical_moe_reduce_cp_sharded_v3,
    canonical_moe_reduce_packed_ep16_v2,
    canonical_moe_reduce_v1,
    resolve_canonical_moe_transport,
)
from xorl.distributed.moe.deepep import sync_pending_combine
from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.data import gather_outputs
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.exact_contract import glm52_exact_forward_enabled
from xorl.models.layers import (
    ACT2FN,
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNorm,
    RotaryEmbedding,
)
from xorl.models.layers.attention import is_flash_attention, update_causal_mask
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.moe import MoEBlock
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.layers.moe.moe_block import _BIRouterGemm, _moe_bi_router_enabled
from xorl.models.layers.moe.routing_replay import get_replay_stage
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.glm5 import parallelize
from xorl.models.transformers.glm5.checkpoint_handler import Glm5CheckpointHandler
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.index_share import (
    CanonicalLogicalIndices,
    IndexShareContext,
    IndexShareContextManager,
    IndexShareMode,
)
from xorl.models.transformers.glm5.indexer import Glm5DsaIndexer
from xorl.models.transformers.glm5.layer_plan import Glm52LayerPlan, Glm52LayerSpec, IndexerType, MLPType
from xorl.models.transformers.glm5.native_fp8 import (
    Glm52NativeBlockFP8Experts,
    replace_glm52_native_fp8_modules,
)
from xorl.models.transformers.glm5.rotary import glm5_apply_rotary_pos_emb
from xorl.models.transformers.glm5.sparse_mla import sparse_mla_dispatch
from xorl.models.transformers.glm5.support import validate_glm5_sequence_parallel
from xorl.ops.block_fp8_native import NativeBlockFP8Linear
from xorl.ops.fused_silu_and_mul import exact_fp32_silu_and_mul, fused_silu_and_mul
from xorl.utils import logging


logger = logging.get_logger(__name__)
GLM52_LOCAL_PARTIAL_POLICY = "glm52_routed_final_scaled_then_shared_ep_slice_bf16_v2"


def _glm52_serving_grouped_topk(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact public SGLang GLM grouped-top-k dispatcher.

    The selected experts and normalized FP32 weights are part of the forward
    numerical contract. Reconstructing this policy in trainer code is not
    sufficient: dispatcher-specific operation ordering can change the weight
    bytes even when the selected ids agree.
    """

    from sglang.srt.layers.moe.topk import biased_grouped_topk  # noqa: PLC0415

    routing_weights, selected_experts = biased_grouped_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        correction_bias=correction_bias,
        topk=top_k,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )
    if routing_weights.dtype is not torch.float32:
        raise TypeError(f"GLM-5.2 serving grouped top-k must return FP32 weights, got {routing_weights.dtype}")
    if selected_experts.dtype is not torch.int32:
        raise TypeError(f"GLM-5.2 serving grouped top-k must return int32 ids, got {selected_experts.dtype}")
    return routing_weights, selected_experts


class Glm5MLP(nn.Module):
    def __init__(self, config: Glm5Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self._use_fused_silu = config.hidden_act == "silu" and not getattr(config, "_activation_native", False)
        # One-round FP32 SwiGLU pairs with serving's universal exact-mode
        # activation (SiluAndMul.forward_exact, xorl-sglang f10b907d8). Model
        # resolution stamps the family-neutral ``_exact_one_round_swiglu`` key
        # for contracted families; unstamped configs keep the historical
        # two-round bytes.
        self._exact_one_round = bool(getattr(config, "_exact_one_round_swiglu", False))
        if self._exact_one_round and config.hidden_act != "silu":
            raise ValueError("GLM exact one-round SwiGLU requires hidden_act='silu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self._exact_one_round:
            hidden_states = exact_fp32_silu_and_mul(torch.cat([gate, up], dim=-1))
        elif self._use_fused_silu:
            hidden_states = fused_silu_and_mul(torch.cat([gate, up], dim=-1))
        else:
            hidden_states = self.act_fn(gate) * up
        return self.down_proj(hidden_states)


class Glm5TopkRouter(nn.Module):
    def __init__(self, config: Glm5Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))

    def _apply(self, fn, recurse: bool = True):
        correction_bias = self._buffers.pop("e_score_correction_bias")
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            target_probe = fn(torch.empty(0, dtype=torch.float32, device=correction_bias.device))
            if correction_bias.is_meta:
                restored_bias = torch.empty(
                    correction_bias.shape,
                    dtype=torch.float32,
                    device=target_probe.device,
                )
            else:
                restored_bias = correction_bias.to(
                    device=target_probe.device,
                    dtype=torch.float32,
                )
            self._buffers["e_score_correction_bias"] = restored_bias
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        if _moe_bi_router_enabled(self.config):
            if hidden_states.dtype is not torch.bfloat16 or self.weight.dtype is not torch.bfloat16:
                raise TypeError("GLM-5.2 batch-invariant router requires BF16 hidden states and BF16 gate weights")
            return _BIRouterGemm.apply(hidden_states, self.weight)
        if getattr(self.config, "_router_fp32", False):
            return F.linear(hidden_states.float(), self.weight.float())
        return F.linear(hidden_states, self.weight)


class Glm5MlaAttention(nn.Module):
    def __init__(self, config: Glm5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.qk_head_dim**-0.5
        rope_params = config.rope_parameters
        if rope_params.get("rope_type", "default") != "default":
            mscale_all_dim = rope_params.get("mscale_all_dim", 0)
            scaling_factor = rope_params.get("factor")
            if mscale_all_dim and scaling_factor is not None:
                mscale = 0.1 * mscale_all_dim * torch.log(torch.tensor(float(scaling_factor))).item() + 1.0
                self.scaling = self.scaling * mscale * mscale

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = RMSNorm(
                config.q_lora_rank,
                eps=config.rms_norm_eps,
                family=RMS_NORM_FAMILY_NO_RESIDUAL,
            )
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
            family=RMS_NORM_FAMILY_NO_RESIDUAL,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self._pad_value_for_flash = (
            is_flash_attention(config._attn_implementation) and self.qk_head_dim != self.v_head_dim
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, self.num_heads, self.qk_head_dim)
        kv_shape = (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(kv_shape)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(batch_size, seq_length, 1, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = glm5_apply_rotary_pos_emb(
            q_rot,
            k_rot,
            cos,
            sin,
            interleaved=getattr(self.config, "rope_interleave", True),
            class_b=bool(getattr(self.config, "_rope_class_b", False) or glm52_exact_forward_enabled(self.config)),
        )
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if getattr(self.config, "_attention_cast_bf16", False):
            query_states = query_states.to(torch.bfloat16)
            key_states = key_states.to(torch.bfloat16)

        if self._pad_value_for_flash:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        return query_states, key_states, value_states

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self._pad_value_for_flash:
            attn_output = attn_output[..., : self.v_head_dim]
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        return self.o_proj(attn_output)

    def _get_attention_fn(self):
        return ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    def _attention_kwargs(self):
        return {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            "sliding_window": None,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        attn_output = attn_strategy.compute_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **kwargs,
        )
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


class Glm5Attention(Glm5MlaAttention):
    """MLA + DSA indexer with dense and sparse execution paths."""

    def __init__(
        self,
        config: Glm5Config,
        layer_idx: int,
        *,
        layer_plan: Glm52LayerPlan | None = None,
    ):
        super().__init__(config, layer_idx)
        self.layer_plan = layer_plan
        self.layer_spec: Glm52LayerSpec | None = None if layer_plan is None else layer_plan.layers[layer_idx]
        allocate_indexer = self.layer_spec is None or self.layer_spec.indexer_type is IndexerType.FULL
        self.indexer = Glm5DsaIndexer(config) if allocate_indexer else None

    def _compute_or_consume_indices(
        self,
        compute,
        index_share_context: IndexShareContext | None,
    ) -> torch.Tensor:
        if self.layer_plan is None or self.layer_spec is None:
            if self.indexer is None:
                raise RuntimeError("Legacy GLM attention unexpectedly has no DSA indexer")
            return compute()
        if index_share_context is None:
            raise RuntimeError("GLM-5.2 layer-plan execution requires an IndexShareContext")
        if self.layer_spec.indexer_type is IndexerType.FULL:
            if self.indexer is None:
                raise RuntimeError(f"Full-indexer layer {self.layer_idx} has no indexer module")
            return index_share_context.get_or_publish(
                producer_layer_index=self.layer_idx,
                layer_plan=self.layer_plan,
                produce_payload=lambda: CanonicalLogicalIndices(compute()),
            ).values
        if self.indexer is not None:
            raise RuntimeError(f"Shared-index layer {self.layer_idx} allocated an indexer module")
        producer_layer_index = self.layer_spec.index_producer_layer
        if producer_layer_index is None:
            raise RuntimeError(f"Shared-index layer {self.layer_idx} has no planned producer")
        return index_share_context.require(
            producer_layer_index=producer_layer_index,
            layer_plan=self.layer_plan,
        ).values

    def _gather_ulysses_sequence_no_grad(self, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor
        pieces = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(pieces, tensor.contiguous(), group=group)
        return torch.cat(pieces, dim=1)

    def _split_kv_b_weight(self, *, compute_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.distributed.tensor import DTensor, Replicate  # noqa: PLC0415

        from xorl.qlora.modules.linear import QLoRALinear  # noqa: PLC0415

        if isinstance(self.kv_b_proj, Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA):
            raise RuntimeError(
                "GLM-5.2 exact absorbed kv_b owns its dequantization and both base BMMs; "
                "the legacy materialized-weight path is not admitted"
            )
        if isinstance(self.kv_b_proj, NativeBlockFP8Linear):
            # Invoke the module's real forward boundary so an eFSDP wrapper
            # unshards the packed bytes/scales before reproducing SGLang's
            # block-FP8 -> BF16 MLA post-load transform.
            weight = self.kv_b_proj(return_dequantized_weight=True)
        elif isinstance(self.kv_b_proj, QLoRALinear):
            # Sparse MLA consumes kv_b through its absorb-form weight boundary
            # instead of QLoRALinear.forward().  Dequantize the immutable base
            # here, then add the differentiable adapter delta below.
            weight = self.kv_b_proj._dequantize_weight()
        else:
            weight = self.kv_b_proj.weight
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()

        if hasattr(self.kv_b_proj, "get_delta_weight"):
            lora_A = self.kv_b_proj.lora_A
            lora_B = self.kv_b_proj.lora_B
            if isinstance(lora_A, DTensor):
                lora_A = lora_A.redistribute(placements=(Replicate(),) * lora_A.device_mesh.ndim).to_local()
            if isinstance(lora_B, DTensor):
                lora_B = lora_B.redistribute(placements=(Replicate(),) * lora_B.device_mesh.ndim).to_local()
            delta = (lora_B @ lora_A) * self.kv_b_proj.scaling
            weight = weight + delta.to(weight.dtype)

        if compute_dtype is not None:
            # Sparse MLA consumes kv_b through its absorb-form weight instead
            # of the projection's normal forward boundary.  Match
            # QLoRALinear.forward() by aligning the materialized FP8 base plus
            # differentiable FP32 LoRA delta to the activation compute dtype.
            weight = weight.to(compute_dtype)

        weight = weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_kc = weight[:, : self.qk_nope_head_dim, :]
        w_vc = weight[:, self.qk_nope_head_dim :, :]
        return w_kc, w_vc

    def _project_qkv_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length, _ = hidden_states.shape
        q_compressed = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_compressed).view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
        q_no_pe, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_no_pe, k_pe = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_no_pe = self.kv_a_layernorm(kv_no_pe)

        if isinstance(self.kv_b_proj, Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA):
            q_no_pe_absorbed = self.kv_b_proj(q_no_pe, branch="q")
            w_vc = None
        else:
            w_kc, w_vc = self._split_kv_b_weight(compute_dtype=q_no_pe.dtype)
            q_no_pe_absorbed = torch.einsum("bshd,hdc->bshc", q_no_pe, w_kc)

        cos, sin = position_embeddings
        k_pe = k_pe.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        q_pe, k_pe = glm5_apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            interleaved=getattr(self.config, "rope_interleave", True),
            class_b=bool(getattr(self.config, "_rope_class_b", False) or glm52_exact_forward_enabled(self.config)),
        )
        k_pe = k_pe.squeeze(2)

        q_absorbed = torch.cat([q_no_pe_absorbed, q_pe], dim=-1)
        kv_compressed = torch.cat([kv_no_pe, k_pe], dim=-1)
        return q_absorbed, kv_compressed, q_compressed, w_vc

    def _project_absorbed_value(
        self,
        attn_compressed: torch.Tensor,
        w_vc: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(self.kv_b_proj, Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA):
            if w_vc is not None:
                raise RuntimeError("GLM-5.2 exact absorbed kv_b must not receive a legacy materialized V weight")
            return self.kv_b_proj(attn_compressed, branch="v")
        if w_vc is None:
            raise RuntimeError("GLM-5.2 generic absorbed MLA requires its materialized V weight")
        return torch.einsum("bshk,hdk->bshd", attn_compressed, w_vc)

    def _gather_full_dsa_inputs(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ps = get_parallel_state()
        if not ps.cp_enabled:
            return hidden_states, q_compressed, position_embeddings
        if ps.ringattn_enabled:
            raise ValueError("GLM-5 DSA does not support ring attention yet.")

        group = ps.ulysses_group
        full_hidden = gather_outputs(hidden_states, gather_dim=1, group=group)
        full_q_compressed = gather_outputs(q_compressed, gather_dim=1, group=group)
        cos, sin = position_embeddings
        full_cos = gather_outputs(cos, gather_dim=1, group=group)
        full_sin = gather_outputs(sin, gather_dim=1, group=group)
        return full_hidden, full_q_compressed, (full_cos, full_sin)

    def _full_attention_mask_for_dsa(
        self,
        attention_mask: torch.Tensor | None,
        full_seq_len: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if attention_mask.dim() in (3, 4):
            if attention_mask.shape[-1] != full_seq_len:
                logger.warning_once(
                    "Ignoring local GLM-5 DSA attention_mask under Ulysses because it does not cover the full "
                    "key sequence. Packed-sequence DSA masking still needs a cu_seqlens-aware indexer path."
                )
                return None

            query_dim = attention_mask.dim() - 2
            if attention_mask.shape[query_dim] == full_seq_len:
                return attention_mask

            ps = get_parallel_state()
            if ps.cp_enabled and not ps.ringattn_enabled and ps.ulysses_group is not None:
                ulysses_size = dist.get_world_size(ps.ulysses_group)
                if attention_mask.shape[query_dim] * ulysses_size == full_seq_len:
                    return gather_outputs(
                        attention_mask,
                        gather_dim=query_dim,
                        group=ps.ulysses_group,
                    )

            logger.warning_once(
                "Ignoring local GLM-5 DSA attention_mask under Ulysses because it does not cover the full "
                "query sequence. Packed-sequence DSA masking still needs a cu_seqlens-aware indexer path."
            )
            return None

        if attention_mask.dim() == 2 and attention_mask.shape[-1] == full_seq_len:
            return attention_mask

        logger.warning_once(
            "Ignoring local GLM-5 DSA attention_mask under Ulysses because it does not cover the full sequence. "
            "Packed-sequence DSA masking still needs a cu_seqlens-aware indexer path."
        )
        return None

    def _sparse_attention_mask_for_dsa(
        self,
        attention_mask: torch.Tensor | None,
        full_seq_len: int,
    ) -> torch.Tensor | None:
        """Return a mask suitable for local-query sparse DSA under Ulysses.

        Sparse DSA scores local query rows against full-sequence keys. A mask
        with local query rows and full key columns can be consumed blockwise by
        the indexer; gathering the query axis would materialize a dense
        full-sequence mask and is prohibitive at long context.
        """
        if attention_mask is None:
            return None

        if attention_mask.dim() in (3, 4):
            if attention_mask.shape[-1] == full_seq_len:
                query_dim = attention_mask.dim() - 2
                if attention_mask.shape[query_dim] == full_seq_len:
                    return attention_mask

                logger.warning_once(
                    "Ignoring local-query GLM-5 sparse DSA attention_mask under Ulysses because its causal "
                    "rows are shard-local. Global causality is enforced by the DSA indexer's query_offset."
                )
                return None

            logger.warning_once(
                "Ignoring local GLM-5 sparse DSA attention_mask under Ulysses because it does not cover the full "
                "key sequence. Packed-sequence DSA masking still needs a cu_seqlens-aware indexer path."
            )
            return None

        if attention_mask.dim() == 2 and attention_mask.shape[-1] == full_seq_len:
            return attention_mask

        logger.warning_once(
            "Ignoring local GLM-5 sparse DSA attention_mask under Ulysses because it does not cover the full key "
            "sequence. Packed-sequence DSA masking still needs a cu_seqlens-aware indexer path."
        )
        return None

    def forward_sparse(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        index_share_context: IndexShareContext | None = None,
        sampler_prefill_lengths: torch.Tensor | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, None]:
        ps = get_parallel_state()
        q, kv, q_compressed, w_vc = self._project_qkv_absorb(hidden_states, position_embeddings)

        if not ps.cp_enabled:
            topk_indices = self._compute_or_consume_indices(
                lambda: self.indexer(
                    hidden_states,
                    q_compressed,
                    position_embeddings,
                    attention_mask,
                    sampler_prefill_lengths=sampler_prefill_lengths,
                ),
                index_share_context,
            )
            attn_compressed = sparse_mla_dispatch(
                q,
                kv,
                topk_indices,
                scaling=self.scaling,
                kv_lora_rank=self.kv_lora_rank,
                backend=getattr(self.config, "_sparse_mla_backend", "auto"),
            )
            attn_out = self._project_absorbed_value(attn_compressed, w_vc)
            projected = self._project_output(attn_out)
            return projected, None

        if ps.ringattn_enabled:
            raise ValueError("GLM-5 sparse MLA supports Ulysses but not ring attention yet.")

        group = ps.ulysses_group
        query_offset = dist.get_rank(group) * hidden_states.shape[1]

        def compute_local_indices() -> torch.Tensor:
            with torch.no_grad():
                local_index_q, local_index_k, local_head_weights = self.indexer.project(
                    hidden_states,
                    q_compressed,
                    position_embeddings,
                    sampler_prefill_lengths=sampler_prefill_lengths,
                    query_offset=query_offset,
                )
                full_index_k = self._gather_ulysses_sequence_no_grad(local_index_k, group)
                full_seq_len = full_index_k.shape[1]
                sparse_attention_mask = self._sparse_attention_mask_for_dsa(attention_mask, full_seq_len)
                return self.indexer.select_topk(
                    local_index_q,
                    full_index_k,
                    local_head_weights,
                    sparse_attention_mask,
                    query_offset=query_offset,
                )

        local_topk_indices = self._compute_or_consume_indices(compute_local_indices, index_share_context)
        full_kv = gather_outputs(kv, gather_dim=1, group=group)

        attn_compressed = sparse_mla_dispatch(
            q,
            full_kv,
            local_topk_indices,
            scaling=self.scaling,
            kv_lora_rank=self.kv_lora_rank,
            backend=getattr(self.config, "_sparse_mla_backend", "auto"),
            query_offset=query_offset,
        )
        attn_out = self._project_absorbed_value(attn_compressed, w_vc)

        projected = self._project_output(attn_out)
        return projected, None

    def _build_dsa_mask(
        self,
        topk_indices: torch.Tensor,
        seq_len: int,
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        logger.warning_once(
            "GLM-5 dense DSA is materializing B*S*S masks. Use sparse_mla_enabled=True for long sequences."
        )
        batch_size = topk_indices.shape[0]
        valid = topk_indices >= 0
        clamped = topk_indices.clamp(min=0, max=seq_len - 1).long()
        selected = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
        b_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand_as(topk_indices)
        s_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1).expand_as(topk_indices)
        selected[b_idx[valid], s_idx[valid], clamped[valid]] = True
        index_mask = torch.where(
            selected,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), float("-inf"), dtype=dtype, device=device),
        ).unsqueeze(1)

        if attention_mask is None:
            return index_mask
        if attention_mask.dim() == 4:
            return index_mask + attention_mask[..., :seq_len, :seq_len]
        if attention_mask.dim() == 3:
            return index_mask + attention_mask[:, None, :seq_len, :seq_len]
        if attention_mask.dim() == 2:
            key_mask = attention_mask[:, :seq_len]
            if key_mask.dtype == torch.bool:
                key_allowed = key_mask
            elif torch.is_floating_point(key_mask):
                key_allowed = key_mask > 0
            else:
                key_allowed = key_mask != 0
            additive_key_mask = torch.where(
                key_allowed[:, None, None, :].to(device=device),
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), float("-inf"), dtype=dtype, device=device),
            )
            return index_mask + additive_key_mask
        raise ValueError(f"Unsupported attention_mask dim for GLM-5 DSA mask: {attention_mask.dim()}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        index_share_context: IndexShareContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ps = get_parallel_state()
        validate_glm5_sequence_parallel(self.config, parallel_state=ps)
        if getattr(self.config, "_sparse_mla_enabled", False):
            return self.forward_sparse(
                hidden_states,
                position_embeddings,
                attention_mask,
                index_share_context=index_share_context,
                **kwargs,
            )

        if getattr(self.config, "_dsa_mask_disabled", False):
            return super().forward(hidden_states, position_embeddings, attention_mask, **kwargs)

        q_compressed = self.q_a_layernorm(self.q_a_proj(hidden_states))
        full_hidden, full_q_compressed, full_position_embeddings = self._gather_full_dsa_inputs(
            hidden_states,
            q_compressed,
            position_embeddings,
        )
        full_seq_len = full_hidden.shape[1]
        full_attention_mask = self._full_attention_mask_for_dsa(attention_mask, full_seq_len)
        topk_indices = self._compute_or_consume_indices(
            lambda: self.indexer(
                full_hidden,
                full_q_compressed,
                full_position_embeddings,
                full_attention_mask,
            ),
            index_share_context,
        )
        combined_mask = self._build_dsa_mask(
            topk_indices,
            full_seq_len,
            full_attention_mask,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return super().forward(hidden_states, position_embeddings, combined_mask, **kwargs)


class Glm5MoEBlock(MoEBlock):
    def __init__(self, config: Glm5Config, *, layer_idx: int | None = None):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=getattr(config, "_moe_implementation", "eager"),
            train_router=getattr(config, "train_router", False),
        )
        self.config = config
        self.layer_idx = layer_idx
        self.gate = Glm5TopkRouter(config)
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.shared_experts = Glm5MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.canonical_contract_version = CANONICAL_MOE_REDUCE_VERSION if glm52_exact_forward_enabled(config) else None
        # No user-facing transport knob: the exact GLM path resolves the best
        # certified transport for the geometry internally (see
        # resolve_canonical_moe_transport) and rejects unsupported geometry.
        self.canonical_moe_transport = CanonicalMoETransport.AUTO
        if self.canonical_contract_version is not None:
            if not self.norm_topk_prob:
                raise RuntimeError("GLM-5.2 canonical routing requires norm_topk_prob=true")
            if self.n_group <= 0 or self.num_experts % self.n_group:
                raise ValueError("GLM-5.2 canonical routing requires equal nonempty expert groups")
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError("GLM-5.2 canonical routing topk_group must be within n_group")
            selectable_experts = self.topk_group * (self.num_experts // self.n_group)
            if not 1 <= self.top_k <= selectable_experts:
                raise ValueError("GLM-5.2 canonical routing top_k exceeds the selected expert groups")

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        stage = get_replay_stage()
        replay = self._routing_replay

        if self.canonical_contract_version is not None and (stage is not None or replay is not None):
            raise RuntimeError("GLM-5.2 canonical MoE canonical path forbids routing replay and recomputation")
        if self.canonical_contract_version is not None and not _moe_bi_router_enabled(self.config):
            raise RuntimeError("GLM-5.2 canonical MoE requires its model-level exact router declaration")
        router_logits = self.gate(flat_hidden_states)

        if stage is not None and replay is not None:
            cached_weights = None
            if stage == "record":
                with torch.no_grad():
                    _, selected_experts = self._route_tokens_to_experts(
                        router_logits,
                        flat_hidden_states.dtype,
                        hidden_states=flat_hidden_states,
                    )
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
                cached_weights = replay.pop_forward_weights()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
                cached_weights = replay.pop_backward_weights()
            else:
                raise RuntimeError(f"Unsupported routing replay stage: {stage}")

            if cached_weights is not None:
                routing_weights = cached_weights.to(flat_hidden_states.dtype)
            else:
                selected_experts, routing_weights = self._regather_routing(
                    router_logits,
                    selected_experts,
                    flat_hidden_states.dtype,
                )
        else:
            routing_weights, selected_experts = self._route_tokens_to_experts(
                router_logits,
                flat_hidden_states.dtype,
                hidden_states=flat_hidden_states,
            )

        ep_dispatch = getattr(self.experts, "ep_dispatch", "alltoall")
        if self.train_router and ep_dispatch == "deepep":
            raise AssertionError(
                "train_router=True is not supported with ep_dispatch='deepep'. "
                "DeepEP cannot propagate gradients through routing weights. "
                "Set train_router=False or switch to ep_dispatch='alltoall'."
            )
        if getattr(self.gate, "_glm52_exact_fullparam_component", False):
            # Full-param mode trains routers: their only gradient
            # path is the combine multiply, so the routing weights must carry
            # a gradient into the router logits.  The serving top-k programs
            # are gradient-opaque, so the values are kept verbatim and the
            # gradient is supplied by the trainer-owned regather surrogate.
            if ep_dispatch == "deepep":
                raise AssertionError("GLM-5.2 full-param router training is not supported with ep_dispatch='deepep'")
            from xorl.models.transformers.glm5.exact_fullparam_fp8 import (  # noqa: PLC0415
                glm52_fullparam_routing_weights_with_grad,
            )

            canonical = self.canonical_contract_version is not None
            routing_weights = glm52_fullparam_routing_weights_with_grad(
                router_logits,
                routing_weights,
                selected_experts,
                # The canonical serving program renormalizes and leaves the
                # routed scaling to the expert kernel; the generic trainer
                # route folds both into the weights.
                renormalize=True if canonical else bool(self.norm_topk_prob),
                scale=1.0 if canonical else float(self.routed_scaling_factor),
            )
        elif not self.train_router:
            routing_weights = routing_weights.detach()

        return routing_weights, selected_experts, router_logits

    def _route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
        *,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correction_bias = self.gate.e_score_correction_bias
        if correction_bias.dtype is not torch.float32:
            raise TypeError("GLM-5.2 e_score_correction_bias must remain FP32 through checkpoint and runtime")
        if correction_bias.shape != (self.num_experts,) or not bool(torch.all(torch.isfinite(correction_bias))):
            raise ValueError("GLM-5.2 e_score_correction_bias failed shape/finite validation")
        if self.canonical_contract_version is not None:
            if not _moe_bi_router_enabled(self.config):
                raise RuntimeError("GLM-5.2 canonical MoE requires its model-level exact router declaration")
            if hidden_states is None:
                raise RuntimeError("GLM-5.2 canonical grouped top-k requires the pre-gate hidden states")
            if hidden_states.dtype is not torch.bfloat16 or router_logits.dtype is not torch.float32:
                raise TypeError("GLM-5.2 canonical routing requires BF16 hidden states and FP32 BI router logits")
            return _glm52_serving_grouped_topk(
                hidden_states,
                router_logits,
                correction_bias,
                top_k=self.top_k,
                num_expert_group=self.n_group,
                topk_group=self.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
            )

        router_scores = router_logits.sigmoid()
        choice_scores = router_scores + correction_bias
        experts_per_group = self.num_experts // self.n_group
        group_topk = min(2, experts_per_group)
        group_scores = choice_scores.view(-1, self.n_group, experts_per_group).topk(group_topk, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=min(self.topk_group, self.n_group), dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, experts_per_group).reshape(-1, self.num_experts)
        scores_for_choice = choice_scores.masked_fill(~score_mask.bool(), 0.0)
        selected_experts = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        routing_weights = router_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(input_dtype), selected_experts

    def _regather_routing(
        self,
        router_logits: torch.Tensor,
        cached_experts: torch.Tensor,
        input_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = torch.gather(router_logits.sigmoid(), 1, cached_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return cached_experts, routing_weights.to(input_dtype)

    def forward_experts_with_shared(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        absolute_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.canonical_contract_version is not None:
            if absolute_positions is None:
                raise RuntimeError("GLM-5.2 canonical MoE execution requires explicit absolute positions")
            return self._canonical_ep_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                absolute_positions,
            )

        residuals = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)

        if self.moe_implementation == "eager" and not get_parallel_state().ep_enabled:
            expert_output = self._eager_forward(flat_hidden_states, routing_weights, selected_experts)
        else:
            expert_output = self.experts(flat_hidden_states, routing_weights, selected_experts)

        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)
        shared_output = self.shared_experts(residuals)
        sync_pending_combine()
        return expert_output + shared_output

    def _canonical_expert_slice(self, ep_rank: int, ep_size: int) -> tuple[int, int]:
        """Resolve this rank's contiguous expert slice for the canonical dispatch.

        Reconciles the two declared-expert-count conventions: the QLoRA EP16
        bank and the frozen native banks DECLARE the global expert count and
        store EP-locally, while the full-param trainable bank declares its
        EP-LOCAL bank size and carries its admission-assigned GLOBAL range
        (``assign_global_expert_range``).  For the full-param bank the
        admission-time ownership and the dispatch-derived ownership must be
        identical — anything else is a routing corruption, not a fallback.
        """

        # Resolve the slice from DECLARATIONS only.  Outside an expert-FSDP
        # unshard window the packed parameters rest as sharded DTensors, so a
        # shape-derived local_experts reads the resting shard instead of the
        # compute-time bank.  Declaration-based resolution is safe only with
        # its two fail-closed companions: the
        # construction can no longer double-slice (ParallelPlan.apply fails
        # closed on load-presliced tensors), and the bank's forward now
        # ACTUALLY validates ids against stored rows at kernel time
        # (Glm52NativeBlockFP8Experts._assert_expert_ids_within_stored_rows —
        # the check this comment previously asserted into existence).
        if isinstance(self.experts, Glm52ExactEP16BlockFP8QLoRARoutedExperts):
            local_experts = int(self.experts.num_local_experts)
            declared_global_experts = int(self.experts.num_experts)
        elif getattr(self.experts, "_glm52_exact_fullparam_component", False):
            # The full-param bank declares its EP-LOCAL size; its GLOBAL
            # ownership is assigned at admission (fail-closed properties:
            # RAISE if the admission never assigned the range).
            local_experts = int(self.experts.num_experts)
            declared_global_experts = int(self.experts.num_global_experts)
            assigned_start = int(self.experts.global_expert_ids[0])
            if assigned_start != ep_rank * local_experts:
                raise RuntimeError(
                    f"GLM-5.2 full-param expert bank owns the global block starting at {assigned_start} "
                    f"but the canonical dispatch derives {ep_rank * local_experts} for ep_rank {ep_rank}; "
                    "admission-time and dispatch-time EP ownership must be identical"
                )
        else:
            # Frozen native bank: DECLARE-global / store-EP-local (the
            # production pair-buffer ownership rule: local = global //
            # ep_size, expert_start = ep_rank * local).
            declared_global_experts = int(self.experts.num_experts)
            local_experts = declared_global_experts // ep_size if ep_size else 0
        if local_experts <= 0 or local_experts * ep_size != declared_global_experts:
            packed = getattr(self.experts, "gate_up_packed_weight_f32", None)
            raise RuntimeError(
                "GLM-5.2 canonical MoE requires equal contiguous expert slices: "
                f"bank={type(self.experts).__name__} local_experts={local_experts} "
                f"ep_size={ep_size} ep_rank={ep_rank} declared_global={declared_global_experts} "
                f"num_experts_attr={getattr(self.experts, 'num_experts', None)} "
                f"packed_type={type(packed).__name__} "
                f"packed_shape={tuple(packed.shape) if packed is not None else None}"
            )
        return local_experts, ep_rank * local_experts

    def _canonical_routed_local_partial(
        self,
        gathered: torch.Tensor,
        gathered_routing: torch.Tensor,
        gathered_ids: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(self.experts, Glm52NativeBlockFP8Experts):
            raise RuntimeError("GLM-5.2 canonical path requires native block-FP8 experts")
        kwargs = {
            "sglang_ep_native_local_ids": local_ids,
            "routed_scaling_factor": self.routed_scaling_factor,
        }
        if isinstance(self.experts, Glm52ExactEP16BlockFP8QLoRARoutedExperts):
            kwargs["selected_experts"] = gathered_ids
        return self.experts(gathered, gathered_routing, **kwargs).to(torch.bfloat16)

    def _canonical_shared_local_partial(
        self,
        gathered: torch.Tensor,
        *,
        contributor_ordinal: int,
        contributor_count: int,
    ) -> torch.Tensor:
        if isinstance(self.shared_experts, Glm52ExactTP16SharedExpertBlockFP8QLoRA):
            return self.shared_experts(gathered, contributor_ordinal=contributor_ordinal)

        intermediate_size = self.shared_experts.intermediate_size
        if intermediate_size % contributor_count:
            raise RuntimeError("GLM-5.2 shared-expert width must divide evenly across EP contributors")
        shard = intermediate_size // contributor_count
        shard_start = contributor_ordinal * shard
        shard_end = shard_start + shard
        if isinstance(self.shared_experts.gate_proj, NativeBlockFP8Linear):
            if not isinstance(self.shared_experts.up_proj, NativeBlockFP8Linear) or not isinstance(
                self.shared_experts.down_proj, NativeBlockFP8Linear
            ):
                raise RuntimeError("GLM-5.2 native FP8 shared experts require all three projections")
            gate = self.shared_experts.gate_proj(
                gathered,
                output_range=(shard_start, shard_end),
            )
            up = self.shared_experts.up_proj(
                gathered,
                output_range=(shard_start, shard_end),
            )
        else:
            gate = F.linear(gathered, self.shared_experts.gate_proj.weight[shard_start:shard_end])
            up = F.linear(gathered, self.shared_experts.up_proj.weight[shard_start:shard_end])
        if getattr(self.config, "_exact_one_round_swiglu", False):
            # Serving computes the exact-mode shared expert with the one-round
            # FP32 SwiGLU (SiluAndMul.forward_exact since xorl-sglang
            # f10b907d8); the canonical partial must produce those bytes.
            activated = exact_fp32_silu_and_mul(torch.cat([gate, up], dim=-1))
        else:
            activated = F.silu(gate) * up
        if isinstance(self.shared_experts.down_proj, NativeBlockFP8Linear):
            return self.shared_experts.down_proj(
                activated,
                input_range=(shard_start, shard_end),
            ).to(torch.bfloat16)
        return F.linear(
            activated,
            self.shared_experts.down_proj.weight[:, shard_start:shard_end],
        ).to(torch.bfloat16)

    def _canonical_ep_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        absolute_positions: torch.Tensor,
    ) -> torch.Tensor:
        from xorl.models.layers.moe.ep_native_combine import (  # noqa: PLC0415
            gather_ids_for_ep_combine,
            gather_tokens_for_ep_combine,
            max_rows_for_ep_combine,
        )

        ps = get_parallel_state()
        if not ps.ep_enabled:
            raise RuntimeError("GLM-5.2 canonical MoE requires expert-parallel contributors")
        ownership = LogicalRowOwnership(
            dp_size=ps.dp_size,
            cp_size=ps.cp_size,
            dp_rank=ps.dp_rank,
            cp_rank=ps.cp_rank if ps.cp_enabled else 0,
            contributor_count=ps.ep_size,
        )
        if ps.tp_size != 1:
            raise RuntimeError("GLM-5.2 canonical MoE requires stage-local dense TP1")
        if ps.ep_group is None:
            raise RuntimeError("GLM-5.2 canonical MoE requires a stage-local EP contributor group")
        get_group_ranks = getattr(dist, "get_process_group_ranks", None)
        if ps.cp_size > 1:
            if ps.sp_group is None:
                raise RuntimeError("GLM-5.2 CP-owned rows require a stage-local sequence-parallel group")
            if get_group_ranks is not None:
                ep_ranks = tuple(get_group_ranks(ps.ep_group))
                cp_ranks = tuple(get_group_ranks(ps.sp_group))
                expected_cp_ranks = tuple(ep_ranks[index] for index in ownership.context_source_ordinals)
                if cp_ranks != expected_cp_ranks:
                    raise RuntimeError(
                        "GLM-5.2 CP membership does not match the DP-major logical row layout: "
                        f"expected={expected_cp_ranks}, actual={cp_ranks}"
                    )
        elif ps.sp_group is not None:
            raise RuntimeError("GLM-5.2 CP1 rows must not have a sequence-parallel process group")
        if self.experts.ep_dispatch == "deepep":
            raise RuntimeError("GLM-5.2 canonical MoE canonical path does not support DeepEP")
        if hidden_states.dtype is not torch.bfloat16:
            raise TypeError("GLM-5.2 canonical local partials require BF16 hidden states")
        if self.config.hidden_act != "silu":
            raise RuntimeError("GLM-5.2 canonical shared-expert policy is versioned for silu only")

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        local_rows = flat.shape[0]
        local_positions = absolute_positions.reshape(-1)
        if local_positions.numel() != local_rows:
            raise ValueError(
                f"absolute_positions has {local_positions.numel()} rows, expected {local_rows} local hidden rows"
            )

        group = ps.ep_group
        padded_rows = max_rows_for_ep_combine(local_rows, flat.device, group)
        gathered = gather_tokens_for_ep_combine(flat, group, padded_rows)
        gathered_routing = gather_tokens_for_ep_combine(routing_weights.reshape(local_rows, -1), group, padded_rows)
        gathered_ids = gather_ids_for_ep_combine(selected_experts.reshape(local_rows, -1), group, padded_rows)
        gathered_positions = gather_ids_for_ep_combine(local_positions[:, None], group, padded_rows).squeeze(-1)
        local_valid = LogicalRowOwnership.valid_positions(local_positions).to(torch.int32)[:, None]
        gathered_valid = gather_ids_for_ep_combine(local_valid, group, padded_rows).squeeze(-1) > 0

        ep_rank = dist.get_rank(group)
        if ep_rank != ownership.source_ordinal:
            raise RuntimeError(
                "GLM-5.2 EP rank order does not match the DP-major/CP-minor row layout: "
                f"ep_rank={ep_rank}, source_ordinal={ownership.source_ordinal}"
            )
        local_experts, expert_start = self._canonical_expert_slice(ep_rank, ps.ep_size)
        local_ids = torch.where(
            (gathered_ids >= expert_start) & (gathered_ids < expert_start + local_experts),
            gathered_ids - expert_start,
            gathered_ids.new_full((), -1),
        ).to(torch.int32)
        routed = self._canonical_routed_local_partial(gathered, gathered_routing, gathered_ids, local_ids)
        shared = self._canonical_shared_local_partial(
            gathered,
            contributor_ordinal=ep_rank,
            contributor_count=ps.ep_size,
        )
        local_partial = (routed + shared).to(torch.bfloat16)

        capacity = int(getattr(self.config, "_glm52_canonical_moe_capacity", local_partial.shape[0]))
        if local_partial.shape[0] > capacity:
            raise RuntimeError(
                f"GLM-5.2 canonical MoE row count {local_partial.shape[0]} exceeds configured capacity {capacity}"
            )
        if local_partial.shape[0] < capacity:
            pad_rows = capacity - local_partial.shape[0]
            local_partial = torch.cat(
                (local_partial, local_partial.new_zeros((pad_rows, hidden_dim))),
                dim=0,
            )
            gathered_positions = torch.cat(
                (gathered_positions, gathered_positions.new_full((pad_rows,), -1)),
                dim=0,
            )
            gathered_valid = torch.cat(
                (gathered_valid, gathered_valid.new_zeros((pad_rows,))),
                dim=0,
            )
        logical_rows = torch.arange(capacity, dtype=torch.int64, device=flat.device)
        logical_rows = logical_rows.masked_fill(~gathered_valid, -1)
        metadata = CanonicalMoEGraphMetadata(
            logical_row_ids=logical_rows,
            absolute_positions=gathered_positions.to(torch.int64),
            valid_mask=gathered_valid,
            capacity=capacity,
            valid_rows=int(gathered_valid.sum().item()),
        )
        contribution = LocalMoEContribution(local_partial, metadata, GLM52_LOCAL_PARTIAL_POLICY)
        ep_mesh = ps.ep_fsdp_device_mesh
        ep_dim = tuple(ep_mesh.mesh_dim_names).index("ep")
        combine_groups = tuple(
            tuple(int(rank) for rank in row.tolist())
            for row in ep_mesh.mesh.movedim(ep_dim, -1).reshape(-1, ps.ep_size)
        )
        plan = ParallelPlan.glm52_trainer(
            world_size=dist.get_world_size(),
            pp_size=ps.pp_size,
            dp_size=ps.dp_size,
            contributor_count=ps.ep_size,
            cp_size=ps.cp_size,
            combine_groups=combine_groups,
            pipeline_layer_ranges=tuple(
                tuple(int(value) for value in stage)
                for stage in getattr(
                    self.config,
                    "_glm52_pipeline_layer_ranges",
                    ((0, self.config.num_hidden_layers),),
                )
            ),
            model_num_layers=self.config.num_hidden_layers,
        )
        resolved_transport = resolve_canonical_moe_transport(
            self.canonical_moe_transport,
            plan=plan,
            capacity=capacity,
            local_rows=local_rows,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        previous_transport = getattr(self.config, "_resolved_canonical_moe_transport", None)
        if previous_transport is None:
            self.config._resolved_canonical_moe_transport = resolved_transport.value
            logger.info_rank0(
                "Resolved GLM-5.2 canonical MoE transport: "
                f"requested={self.canonical_moe_transport.value}, resolved={resolved_transport.value}, "
                f"EP={plan.ep_size}, CP={plan.cp_size}, capacity={capacity}, local_rows={local_rows}, graph=False"
            )
        elif previous_transport != resolved_transport.value:
            raise RuntimeError(
                "GLM-5.2 canonical MoE transport geometry changed within one model: "
                f"{previous_transport} -> {resolved_transport.value}"
            )

        if resolved_transport is CanonicalMoETransport.CP_SHARDED_V3:
            canonical = canonical_moe_reduce_cp_sharded_v3(
                contribution,
                plan=plan,
                group=group,
                physical_global_rank=dist.get_rank(),
                graph_mode=False,
            )
        else:
            canonical_reduce = (
                canonical_moe_reduce_packed_ep16_v2
                if resolved_transport is CanonicalMoETransport.PACKED_EP16_V2
                else canonical_moe_reduce_v1
            )
            canonical = canonical_reduce(
                contribution,
                plan=plan,
                group=group,
                output_distribution=OutputDistribution.REPLICATED_CANONICAL,
                physical_global_rank=dist.get_rank(),
                graph_mode=False,
            )
        if resolved_transport is CanonicalMoETransport.CP_SHARDED_V3:
            local_canonical = canonical.tensor[:local_rows]
        else:
            local_canonical = canonical.tensor[ownership.source_slice(padded_rows=padded_rows, local_rows=local_rows)]
        return local_canonical.reshape(batch_size, sequence_length, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        absolute_positions: torch.Tensor | None = None,
    ):
        routing_weights, selected_experts, router_logits = self.route(hidden_states)
        hidden_states = self.forward_experts_with_shared(
            hidden_states,
            routing_weights,
            selected_experts,
            absolute_positions,
        )
        return hidden_states, router_logits


class Glm5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        layer_idx: int,
        *,
        layer_plan: Glm52LayerPlan | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.layer_spec = None if layer_plan is None else layer_plan.layers[layer_idx]
        self.self_attn = Glm5Attention(config, layer_idx, layer_plan=layer_plan)
        is_sparse_mlp = (
            layer_idx >= config.first_k_dense_replace
            if self.layer_spec is None
            else self.layer_spec.mlp_type is MLPType.SPARSE
        )
        if is_sparse_mlp:
            self.mlp = Glm5MoEBlock(config, layer_idx=layer_idx)
        else:
            self.mlp = Glm5MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            family=(RMS_NORM_FAMILY_NO_RESIDUAL if layer_idx == 0 else RMS_NORM_FAMILY_RESIDUAL_TREE),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            family=RMS_NORM_FAMILY_RESIDUAL_TREE,
        )
        self.gradient_checkpointing = False

    @staticmethod
    def _local_absolute_positions(
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if position_ids is None:
            return None
        local_length = hidden_states.shape[1]
        if position_ids.shape[-1] == local_length:
            return position_ids
        ps = get_parallel_state()
        if ps.cp_enabled and position_ids.shape[-1] == local_length * ps.cp_size:
            start = ps.cp_rank * local_length
            return position_ids[..., start : start + local_length]
        raise ValueError(f"Cannot map position_ids width {position_ids.shape[-1]} to {local_length} local GLM rows")

    def _pre_mlp_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        index_share_context: IndexShareContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            index_share_context=index_share_context,
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual=residual,
            prenorm=True,
        )
        return hidden_states, residual, self_attn_weights

    def _pre_dispatch_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        index_share_context: IndexShareContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, residual, self_attn_weights = self._pre_mlp_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            index_share_context=index_share_context,
            **kwargs,
        )
        if not isinstance(self.mlp, Glm5MoEBlock):
            raise RuntimeError("_pre_dispatch_forward is only valid for GLM-5 MoE layers")
        routing_weights, selected_experts, router_logits = self.mlp.route(hidden_states)
        return hidden_states, residual, self_attn_weights, routing_weights, selected_experts, router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        index_share_context: IndexShareContext | None = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        local_absolute_positions = self._local_absolute_positions(hidden_states, position_ids)
        _checkpoint_method = getattr(self, "_gradient_checkpointing_method", None)
        _selective = (
            self.training
            and getattr(self, "gradient_checkpointing", False)
            and _checkpoint_method == "recompute_before_dispatch"
        )

        if _selective and isinstance(self.mlp, Glm5MoEBlock):
            (
                hidden_states,
                residual,
                self_attn_weights,
                routing_weights,
                selected_experts,
                router_logits,
            ) = self._gradient_checkpointing_func(
                self._pre_dispatch_forward,
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                index_share_context=index_share_context,
                **kwargs,
            )
            hidden_states = self.mlp.forward_experts_with_shared(
                hidden_states,
                routing_weights,
                selected_experts,
                local_absolute_positions,
            )
        elif _selective:
            hidden_states, residual, self_attn_weights = self._gradient_checkpointing_func(
                self._pre_mlp_forward,
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                index_share_context=index_share_context,
                **kwargs,
            )
            hidden_states = self.mlp(
                hidden_states,
                **({"absolute_positions": local_absolute_positions} if isinstance(self.mlp, Glm5MoEBlock) else {}),
            )
            if isinstance(hidden_states, tuple):
                hidden_states, router_logits = hidden_states
            else:
                router_logits = None
        else:
            hidden_states, residual, self_attn_weights = self._pre_mlp_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                index_share_context=index_share_context,
                **kwargs,
            )
            hidden_states = self.mlp(
                hidden_states,
                **({"absolute_positions": local_absolute_positions} if isinstance(self.mlp, Glm5MoEBlock) else {}),
            )
            if isinstance(hidden_states, tuple):
                hidden_states, router_logits = hidden_states
            else:
                router_logits = None

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


class Glm5PreTrainedModel(XorlPreTrainedModel):
    config_class = Glm5Config
    base_model_prefix = "model"
    _no_split_modules = ["Glm5DecoderLayer"]
    supports_tensor_parallelism = True

    def get_ignore_modules_in_mixed_precision(self):
        # Dense native-FP8 and exact active-QLoRA modules are separately
        # fully_shard-wrapped without an MP policy. The exact QLoRA wrapper
        # owns FP32 master factors and performs its one BF16 rounding inside
        # the value path. Native experts use the dedicated expert FSDP branch
        # and must not appear here (which would double-wrap them).
        if getattr(self.config, "quantization_config", None) is not None:
            # Keep the base model independent of adapter implementation at
            # import time; exact_qlora may grow model-specific construction
            # helpers without creating a modeling import cycle.
            from xorl.models.transformers.glm5.exact_fullparam_admission import (  # noqa: PLC0415
                Glm52FullParamTopkRouter,
            )
            from xorl.models.transformers.glm5.exact_fullparam_fp8 import (  # noqa: PLC0415
                Glm52ExactTP1BlockFP8FullParamLinear,
                Glm52FullParamDenseMLP,
            )
            from xorl.models.transformers.glm5.exact_qlora import (  # noqa: PLC0415
                Glm52ExactTP1BlockFP8QLoRALinear,
            )
            from xorl.models.transformers.glm5.exact_shared_expert_qlora import (  # noqa: PLC0415
                Glm52ExactTP16SharedExpertBlockFP8QLoRA,
            )

            return (
                NativeBlockFP8Linear,
                Glm52ExactTP1BlockFP8QLoRALinear,
                Glm52ExactTP16SharedExpertBlockFP8QLoRA,
                # Full-param training: dense composites and routers own
                # FP32 masters plus byte-packed caches — the decoder-layer
                # BF16 param cast would corrupt both; the composites'
                # engaged-contract check refuses to score in that state. Expert banks
                # take the dedicated expert-FSDP branch and must not appear
                # here.
                Glm52FullParamDenseMLP,
                Glm52FullParamTopkRouter,
                Glm52ExactTP1BlockFP8FullParamLinear,
            )
        return None

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Glm5TopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.zero_()
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
        elif isinstance(module, MoEExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)

    def get_parallel_plan(self):
        return parallelize.get_ep_plan()

    def get_checkpoint_handler(self, **kwargs):
        ep_rank = kwargs.get("ep_rank", 0)
        ep_size = kwargs.get("ep_size", 1)
        if kwargs.get("is_broadcast", False):
            ep_rank, ep_size = 0, 1

        return Glm5CheckpointHandler(
            num_experts=self.config.n_routed_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            num_hidden_layers=self.config.num_hidden_layers,
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
            model=self,
            load_family=kwargs.get("load_family"),
        )


class Glm5Model(Glm5PreTrainedModel):
    def __init__(self, config: Glm5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layer_plan: Glm52LayerPlan | None = None
        self._index_share_context_managers: dict[tuple[int, int], IndexShareContextManager] = {}
        if config.indexer_types is not None or config.mlp_layer_types is not None:
            if config.indexer_types is None or config.mlp_layer_types is None:
                raise ValueError("GLM-5.2 requires indexer_types and mlp_layer_types together")
            # Pipeline staging is explicit: without configured ranges the
            # model is one stage. (Multi-stage construction must pass
            # _glm52_pipeline_layer_ranges; no implicit per-depth split.)
            configured_ranges = getattr(config, "_glm52_pipeline_layer_ranges", None)
            ranges = (
                ((0, config.num_hidden_layers),)
                if configured_ranges is None
                else tuple(tuple(int(value) for value in stage) for stage in configured_ranges)
            )
            self.layer_plan = Glm52LayerPlan.from_config(config, pipeline_layer_ranges=ranges)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Glm5DecoderLayer(config, layer_idx, layer_plan=self.layer_plan)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            family=RMS_NORM_FAMILY_RESIDUAL_TREE,
        )
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._skip_causal_mask = is_flash_attention(config._attn_implementation)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _index_share_manager_for_local_layers(self) -> IndexShareContextManager | None:
        if self.layer_plan is None:
            return None
        local_layers = tuple(index for index, layer in enumerate(self.layers) if layer is not None)
        if not local_layers:
            # Weighted PP layouts may assign embedding/head-only edge stages.
            # They carry no IndexShare producer/consumer and need no context.
            return None
        candidates = [
            stage
            for stage in self.layer_plan.pipeline_layer_ranges
            if local_layers[0] == stage[0]
            and local_layers[-1] == stage[1] - 1
            and len(local_layers) == stage[1] - stage[0]
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "GLM-5.2 canonical path requires each executing model part to own exactly one complete planned "
                f"pipeline stage; local layers are {local_layers[0]}..{local_layers[-1]}"
            )
        stage = candidates[0]
        manager = self._index_share_context_managers.get(stage)
        if manager is None:
            manager = IndexShareContextManager(self.layer_plan, stage)
            self._index_share_context_managers[stage] = manager
        return manager

    def release_index_share_context(self) -> None:
        """Release every invocation retained by the completed PP schedule."""

        for manager in self._index_share_context_managers.values():
            manager.end_all()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        index_share_mode: IndexShareMode | str | None = None,
        **kwargs,
    ) -> MoeModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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

        if self._skip_causal_mask:
            causal_mask = None
        else:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            causal_mask = update_causal_mask(
                self.config._attn_implementation,
                attention_mask,
                hidden_states,
                cache_position,
                sliding_window=None,
                is_training=self.training,
                output_attentions=output_attentions,
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        ps = get_parallel_state()
        position_embeddings = get_cp_strategy().prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_attention_heads,
        )

        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        index_share_manager = self._index_share_manager_for_local_layers()
        if index_share_manager is not None and index_share_mode is None:
            raise RuntimeError(
                "GLM-5.2 IndexShare execution requires explicit index_share_mode="
                "'training_with_backward' or 'forward_only'"
            )
        index_share_context = None if index_share_manager is None else index_share_manager.begin(mode=index_share_mode)
        forward_succeeded = False
        try:
            for layer_idx, decoder_layer in enumerate(self.layers):
                if decoder_layer is None:
                    continue
                _grad_ckpt_method = (
                    getattr(self, "_gradient_checkpointing_method", "recompute_full_layer")
                    if self.gradient_checkpointing and self.training
                    else None
                )
                _use_outer_checkpoint = _grad_ckpt_method == "recompute_full_layer"

                if _use_outer_checkpoint:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        output_attentions,
                        output_router_logits,
                        position_embeddings,
                        index_share_context=index_share_context,
                        **kwargs,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        position_embeddings=position_embeddings,
                        index_share_context=index_share_context,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

            hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states
            if output_hidden_states:
                _ = output_hidden_states

            result = MoeModelOutput(
                last_hidden_state=hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )
            forward_succeeded = True
            return result
        finally:
            if index_share_manager is not None:
                index_share_manager.finish_forward(index_share_context, succeeded=forward_succeeded)


class Glm5ForCausalLM(Glm5PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config: Glm5Config):
        super().__init__(config)
        self.model = Glm5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
        if getattr(config, "quantization_config", None) is not None and not getattr(
            config, "_glm52_block_fp8_qlora", False
        ):
            replace_glm52_native_fp8_modules(
                self,
                config.quantization_config,
            )

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

    def release_index_share_context(self) -> None:
        self.model.release_index_share_context()

    def get_pp_module_config(self):
        pp_config = {
            "input_fqns": ["model.embed_tokens"],
            "layer_prefix": "model.layers",
            "output_fqns": ["model.norm", "lm_head"],
            "always_keep_fqns": ["model.rotary_emb"],
            "num_layers": self.config.num_hidden_layers,
        }
        ranges = getattr(self.config, "_glm52_pipeline_layer_ranges", None)
        if ranges is not None:
            pp_config["pipeline_layer_ranges"] = tuple(
                tuple(int(value) for value in stage_range) for stage_range in ranges
            )
        module_plan = getattr(self.config, "_glm52_pipeline_module_names_per_stage", None)
        if module_plan is not None:
            pp_config["module_names_per_stage"] = tuple(tuple(names) for names in module_plan)
        return pp_config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        index_share_mode: IndexShareMode | str | None = None,
        **kwargs,
    ) -> MoeCausalLMOutput:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits or self.config.router_aux_loss_coef > 0
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            index_share_mode=index_share_mode,
            **kwargs,
        )
        return MoeCausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )


class GlmMoeDsaForCausalLM(Glm5ForCausalLM):
    pass


ModelClass = [Glm5ForCausalLM, GlmMoeDsaForCausalLM]


__all__ = [
    "Glm5Attention",
    "Glm5Config",
    "Glm5DecoderLayer",
    "Glm5DsaIndexer",
    "Glm5ForCausalLM",
    "Glm5MLP",
    "Glm5Model",
    "Glm5MoEBlock",
    "Glm5PreTrainedModel",
    "GlmMoeDsaForCausalLM",
]
