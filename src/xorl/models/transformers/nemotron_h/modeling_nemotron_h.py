"""NVIDIA Nemotron-3-Ultra (nemotron_h) modeling.

Hybrid Mamba2 / attention / latent-MoE decoder. Per-layer block is pre-norm
``x = x + mixer(norm(x))`` where the mixer is selected by
``config.layers_block_type[layer_idx]``:

- ``"mamba"``     → :class:`xorl.ops.ssm.Mamba2Mixer` (trainable chunked SSD)
- ``"attention"`` → GQA attention, **NoPE** (the HF reference never applies
  rotary embeddings), separate q/k/v/o projections
- ``"moe"``       → fp32-sigmoid-routed latent MoE with 512 non-gated relu²
  experts plus a non-gated shared expert on the pre-latent input

Module-tree naming: HF checkpoints store the decoder under ``backbone.*``,
but xorl infra (EP plan wildcards, expert-key regexes, weight sync, FSDP/PP
helpers) is built around a ``model.*`` root, and the transformers 5.5.3
in-memory module tree is also ``model.*``. We therefore use
``model.{embeddings,layers.{i}.{norm,mixer},norm_f}`` + ``lm_head`` internally
— identical to the HF module tree below the prefix — and the checkpoint
handler maps ``backbone.`` ↔ ``model.`` on load/save. Within a block the
attribute names mirror HF exactly (``mixer.gate``, ``mixer.experts``,
``mixer.fc1_latent_proj``, ``mixer.fc2_latent_proj``, ``mixer.shared_experts``)
so the state-dict mapping stays passthrough apart from expert stacking.
"""

import math
from typing import Optional, Tuple, Unpack

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from xorl.distributed.moe.deepep import sync_pending_combine
from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.layers import ACT2FN, RMSNorm
from xorl.models.layers.attention import AttentionKwargs, is_flash_attention, update_causal_mask
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.moe import MoEBlock, MoEExperts
from xorl.models.layers.moe.routing_replay import get_replay_stage
from xorl.models.module_utils import MoEGradientCheckpointingLayer
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.nemotron_h import parallelize
from xorl.models.transformers.nemotron_h.checkpoint_handler import (
    NemotronHCheckpointHandler,
    checkpoint_has_per_expert_nemotron_weights,
)
from xorl.models.transformers.nemotron_h.configuration_nemotron_h import NemotronHConfig
from xorl.ops.ssm import GroupRMSNormGated, Mamba2Mixer
from xorl.utils import logging


logger = logging.get_logger(__name__)


def _adapt_nemotron_h_config(config) -> NemotronHConfig:
    if isinstance(config, NemotronHConfig):
        return config
    if getattr(config, "model_type", None) == "nemotron_h":
        return NemotronHConfig.from_hf_config(config)
    return config


def _copy_full_tensor_(param: torch.Tensor, full: torch.Tensor) -> None:
    """Copy a full (unsharded) tensor into a param that may be an FSDP2 DTensor shard."""
    if isinstance(param.data, DTensor):
        full = distribute_tensor(
            full.to(device=param.device, dtype=param.dtype),
            device_mesh=param.data.device_mesh,
            placements=param.data.placements,
        )
    param.data.copy_(full.to(param.dtype))


class NemotronHMLP(nn.Module):
    """Non-gated MLP: ``down_proj(act(up_proj(x)))`` — used as the MoE shared expert."""

    def __init__(self, config: NemotronHConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    """Sigmoid-scored router. Computes logits in fp32, exactly as the HF reference."""

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        return F.linear(hidden_states.float(), self.weight.float())


class NemotronHAttention(nn.Module):
    """GQA attention without positional embeddings (NoPE)."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = None
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del position_embeddings  # NoPE: no rotary embedding anywhere
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        return query_states, key_states, value_states

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        return self.o_proj(attn_output)

    def _get_attention_fn(self):
        return ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    def _attention_kwargs(self) -> dict:
        return {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            "sliding_window": self.sliding_window,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, None)
        attn_output = attn_strategy.compute_attention(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


class NemotronHMoE(MoEBlock):
    """Latent MoE: fp32 sigmoid router → fc1_latent_proj → non-gated relu² experts →
    fc2_latent_proj, plus a non-gated shared expert applied to the pre-latent input."""

    def __init__(self, config: NemotronHConfig):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.mlp_hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=getattr(config, "_moe_implementation", "eager"),
            train_router=getattr(config, "train_router", False),
            record_routing_weights=getattr(config, "record_routing_weights", True),
        )
        self.config = config
        self.latent_size = config.moe_latent_size if config.moe_latent_size is not None else config.hidden_size
        # Replace the generic gate/experts from MoEBlock with NemotronH-specific ones:
        # sigmoid router with correction bias, and non-gated experts in the latent dim.
        self.gate = NemotronHTopkRouter(config)
        self.experts = MoEExperts(
            num_experts=config.n_routed_experts,
            hidden_dim=self.latent_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.mlp_hidden_act,
            moe_implementation=getattr(config, "_moe_implementation", "eager"),
            gated=False,
        )
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.fc1_latent_proj = nn.Linear(config.hidden_size, self.latent_size, bias=config.mlp_bias)
        self.fc2_latent_proj = nn.Linear(self.latent_size, config.hidden_size, bias=config.mlp_bias)
        self.shared_experts = NemotronHMLP(config, intermediate_size=config.moe_shared_expert_intermediate_size)

    def _route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF ``NemotronHMoE.route_tokens_to_experts`` in fp32: sigmoid scores, correction
        bias for choice only, group routing, original scores for weights, renorm + scale."""
        router_scores = router_logits.sigmoid()
        choice_scores = router_scores + self.gate.e_score_correction_bias.float()
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

    def route(self, hidden_states: torch.Tensor):
        """Sigmoid routing with routing-replay support (overrides the softmax base route)."""
        router_logits = self.gate(hidden_states)

        stage = get_replay_stage()
        replay = self._routing_replay

        if stage is not None and replay is not None:
            if stage == "record":
                with torch.no_grad():
                    _, selected_experts = self._route_tokens_to_experts(router_logits, hidden_states.dtype)
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
            else:
                raise RuntimeError(f"Unsupported routing replay stage: {stage}")

            selected_experts, routing_weights = self._regather_routing(
                router_logits, selected_experts, hidden_states.dtype
            )

            if self.record_routing_weights:
                if stage == "record":
                    replay.record_weights(routing_weights)
                elif stage == "replay_backward":
                    cached_weights = replay.pop_backward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(hidden_states.dtype)
                elif stage == "replay_forward":
                    cached_weights = replay.pop_forward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(hidden_states.dtype)
        else:
            routing_weights, selected_experts = self._route_tokens_to_experts(router_logits, hidden_states.dtype)

        ep_dispatch = getattr(self.experts, "ep_dispatch", "alltoall")
        if self.train_router and ep_dispatch == "deepep":
            raise AssertionError(
                "train_router=True is not supported with ep_dispatch='deepep'. "
                "DeepEP cannot propagate gradients through routing weights. "
                "Set train_router=False or switch to ep_dispatch='alltoall'."
            )
        if not self.train_router:
            routing_weights = routing_weights.detach()

        return routing_weights, selected_experts, router_logits

    def _routed_experts_forward(
        self,
        flat_hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        latent_states = self.fc1_latent_proj(flat_hidden_states)
        if self.moe_implementation == "eager":
            expert_output = self._eager_forward(latent_states, routing_weights, selected_experts)
        else:
            expert_output = self.experts(latent_states, routing_weights, selected_experts)
        return self.fc2_latent_proj(expert_output)

    def forward_experts_only(self, hidden_states, routing_weights, selected_experts):
        """Latent experts + shared expert with pre-computed routing (selective checkpointing)."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        expert_output = self._routed_experts_forward(flat_hidden_states, routing_weights, selected_experts)
        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)
        return expert_output + self.shared_experts(hidden_states)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        routing_weights, selected_experts, router_logits = self.route(flat_hidden_states)
        expert_output = self._routed_experts_forward(flat_hidden_states, routing_weights, selected_experts)
        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)
        shared_output = self.shared_experts(hidden_states)
        sync_pending_combine()
        return expert_output + shared_output, router_logits


class NemotronHBlock(MoEGradientCheckpointingLayer):
    """Pre-norm block: ``x = x + mixer(norm(x))`` with the mixer picked by block type."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = config.layers_block_type[layer_idx]
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if self.block_type == "mamba":
            self.mixer = Mamba2Mixer(
                hidden_size=config.hidden_size,
                num_heads=config.mamba_num_heads,
                head_dim=config.mamba_head_dim,
                n_groups=config.n_groups,
                ssm_state_size=config.ssm_state_size,
                conv_kernel=config.conv_kernel,
                use_conv_bias=config.use_conv_bias,
                chunk_size=config.chunk_size,
                activation=config.mamba_hidden_act,
                time_step_limit=tuple(config.time_step_limit),
                layer_norm_epsilon=config.layer_norm_epsilon,
                use_bias=config.use_bias,
                layer_idx=layer_idx,
            )
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(config, layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMoE(config)
        else:
            raise ValueError(f"Unknown layers_block_type entry: {self.block_type!r}")

        # MoE-layer infra (enable_routing_replay, MoEGradientCheckpointingLayer._moe_forward)
        # keys on an `mlp` attribute. Alias it WITHOUT registering a second submodule
        # (object.__setattr__ bypasses nn.Module registration) so state-dict keys stay
        # unique under the HF-mirroring `mixer` name.
        object.__setattr__(self, "mlp", self.mixer if self.block_type == "moe" else nn.Identity())

    def _pre_mlp_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del position_ids  # NoPE
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.block_type == "mamba":
            mamba_kwargs = {}
            if kwargs.get("cu_seq_lens_q") is not None:
                # Packed varlen (flattened-pack convention): reset conv + SSM state at
                # every document boundary, mirroring the GatedDeltaNet mixer interface.
                mamba_kwargs["cu_seqlens"] = kwargs["cu_seq_lens_q"]
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask, **mamba_kwargs)[0]
        elif self.block_type == "attention":
            hidden_states, _ = self.mixer(hidden_states, attention_mask=attention_mask, **kwargs)
        # "moe": the mixer runs as `self.mlp` inside MoEGradientCheckpointingLayer._moe_forward.
        return hidden_states, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = False,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.FloatTensor, ...]:
        return self._moe_forward(
            hidden_states,
            output_router_logits=output_router_logits,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class NemotronHPreTrainedModel(XorlPreTrainedModel):
    config_class = NemotronHConfig
    base_model_prefix = "model"
    _no_split_modules = ["NemotronHBlock"]
    _checkpoint_skip_key_patterns = [r"^mtp\."]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, NemotronHTopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.zero_()
        elif isinstance(module, MoEExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, (RMSNorm, GroupRMSNormGated)):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Mamba2Mixer) and module.A_log.device.type != "meta":
            _copy_full_tensor_(module.A_log, torch.log(torch.arange(1, module.num_heads + 1, dtype=torch.float32)))
            module.D.data.fill_(1.0)
            # dt_bias = inverse-softplus of dt ~ LogUniform(time_step_min, time_step_max),
            # floored at time_step_floor — matches HF NemotronHPreTrainedModel._init_weights.
            dt = torch.exp(
                torch.rand(module.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            _copy_full_tensor_(module.dt_bias, inv_dt)

        if self.config.rescale_prenorm_residual:
            # GPT-2-style residual rescaling of each mixer's out_proj, matching HF.
            for name, p in module.named_parameters():
                if name == "out_proj.weight" and p.device.type != "meta":
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p.div_(math.sqrt(self.config.num_hidden_layers))

    def get_parallel_plan(self):
        return parallelize.get_ep_plan()

    def get_checkpoint_handler(self, **kwargs):
        checkpoint_keys = kwargs.get("checkpoint_keys", set()) or set()
        ep_rank = kwargs.get("ep_rank", 0)
        ep_size = kwargs.get("ep_size", 1)
        if kwargs.get("is_broadcast", False):
            ep_rank, ep_size = 0, 1
        has_per_expert = checkpoint_has_per_expert_nemotron_weights(checkpoint_keys) if checkpoint_keys else True
        return NemotronHCheckpointHandler(
            num_experts=self.config.n_routed_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            checkpoint_has_per_expert=has_per_expert,
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config: NemotronHConfig):
        config = _adapt_nemotron_h_config(config)
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._skip_causal_mask = is_flash_attention(config._attn_implementation)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        output_attentions: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> MoeModelOutput:
        del use_cache  # training-only
        parallelize.validate_parallelism_support(get_parallel_state())

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        has_attention_layers = any(block_type == "attention" for block_type in self.config.layers_block_type)
        if self._skip_causal_mask or not has_attention_layers:
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

        # Mamba layers only need the mask to zero padded positions (HF semantics).
        mamba_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            mamba_mask = None

        block_type_to_mask = {"mamba": mamba_mask, "attention": causal_mask, "moe": None}

        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if decoder_layer is None:
                continue
            layer_mask = block_type_to_mask[decoder_layer.block_type]
            # Selective checkpointing is handled inside the block; only wrap the
            # whole layer for the default full-layer recompute method.
            _use_outer_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and self._gradient_checkpointing_method == "recompute_full_layer"
            )
            if _use_outer_checkpoint:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    output_router_logits,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    output_router_logits=output_router_logits,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (None,)
            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm_f(hidden_states) if self.norm_f is not None else hidden_states
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class NemotronHForCausalLM(NemotronHPreTrainedModel):
    _tied_weights_keys = {}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        config = _adapt_nemotron_h_config(config)
        super().__init__(config)
        self.model = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def unfuse_for_tp(self):
        raise NotImplementedError(parallelize.TP_UNSUPPORTED_MESSAGE)

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_pp_module_config(self):
        return {
            "input_fqns": ["model.embeddings"],
            "layer_prefix": "model.layers",
            "output_fqns": ["model.norm_f", "lm_head"],
            "always_keep_fqns": [],
            "num_layers": self.config.num_hidden_layers,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs,
    ) -> MoeCausalLMOutput:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits or self.config.router_aux_loss_coef > 0
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        return MoeCausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )


ModelClass = NemotronHForCausalLM


__all__ = [
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
]
