from __future__ import annotations

# Adapted from flash-linear-attention/fla/layers/gated_deltanet.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.
import math
import warnings
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from xorl.lora.fold import lora_merged_forward_enabled
from xorl.ops.linear_attention.backend import (
    flashqla_chunk_gated_delta_rule,
    flashqla_chunk_gated_delta_rule_cp,
    get_gdn_backend,
    warn_cp_fallback_once,
)
from xorl.ops.linear_attention.layers.utils import get_unpad_data, index_first_axis, pad_input
from xorl.ops.linear_attention.modules import (
    FusedRMSNormGated,
    RMSNorm,
    ShortConvolution,
    causal_conv1d_qkv_contract,
)
from xorl.ops.linear_attention.modules.bi_contract import (
    _is_gdn_contract_enabled,
    bi_fused_gdn_gating,
    gdn_contract,
)
from xorl.ops.linear_attention.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)


def _sglang_compatible_beta_gate(b_input: torch.Tensor) -> torch.Tensor:
    beta = b_input.float().sigmoid()
    if beta.dtype != b_input.dtype:
        beta = beta.to(dtype=b_input.dtype).float()
    return beta


class GatedDeltaNet(nn.Module):
    # SGLang evaluates the decay/time-step terms from FP32 parameter leaves,
    # while every projection in this module remains a BF16 compute parameter.
    # FSDP2 requires one original dtype per trainable parameter group, so the
    # parallelizer uses this declaration to give the direct parameters their
    # own tiny full-precision FSDP unit without changing checkpoint FQNs.
    fsdp_full_precision_parameter_names = ("A_log", "dt_bias")

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        exact_contract: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.exact_contract = exact_contract
        self.exact_merged_forward = exact_contract

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by "
                f"num_v_heads * head_dim={self.num_v_heads * self.head_dim}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}.",
            )
        if mode not in {"chunk", "fused_recurrent"}:
            raise ValueError(f"Unsupported GatedDeltaNet mode: {mode}")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
        else:
            warnings.warn(
                "ShortConvolution is usually important for GatedDeltaNet quality; "
                "leave `use_short_conv=True` unless you know you want it disabled.",
                stacklevel=2,
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        # Match SGLang's Qwen3.5/GDN contract: decay/time-step parameters stay fp32
        # even when the rest of the model is moved to bf16/fp16.
        for name in ("A_log", "dt_bias"):
            param = getattr(self, name, None)
            if isinstance(param, nn.Parameter) and param.is_floating_point() and param.dtype != torch.float32:
                with torch.no_grad():
                    param.data = param.data.float()
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()
        return module

    def _fused_qkvz_lora_delta(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Apply River's fused q/k/v/z LoRA delta and split its output."""
        adapter = getattr(self, "in_proj_qkvz", None)
        if adapter is None:
            return None
        delta = adapter(hidden_states)
        expected = 2 * self.key_dim + 2 * self.value_dim
        if delta.shape[-1] != expected:
            raise RuntimeError(f"Fused GDN LoRA produced {delta.shape[-1]} features, expected {expected}")
        return delta.split(
            (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
            dim=-1,
        )

    def _add_output_lora(self, inputs: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Add River's fused GDN output LoRA delta when one is injected."""
        adapter = getattr(self, "out_proj", None)
        if adapter is None:
            return base_output
        return base_output + adapter(inputs).to(base_output.dtype)

    @staticmethod
    def _linear_with_contract(
        module: nn.Linear,
        inputs: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(module, "_xorl_bi_trunk_wrapped", False):
            from xorl.ops.batch_invariant_ops import batch_invariant_trunk_linear  # noqa: PLC0415

            return batch_invariant_trunk_linear(inputs, weight, module.bias)
        return F.linear(inputs, weight, module.bias)

    def _project_output_linear(self, o: torch.Tensor) -> torch.Tensor:
        output_adapter = getattr(self, "out_proj", None)
        if output_adapter is not None and lora_merged_forward_enabled(self):
            folded = output_adapter.merged_weight_for_forward(self.o_proj.weight)
            return self._linear_with_contract(self.o_proj, o, folded)

        return self._add_output_lora(o, self.o_proj(o))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
        with gdn_contract(self.exact_contract):
            return self._forward_impl(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
        del output_attentions
        if attention_mask is not None and len(attention_mask.shape) != 2:
            raise ValueError(
                "Expected `attention_mask` with shape [batch_size, seq_len] where 0 marks padding.",
            )

        batch_size, q_len, _ = hidden_states.shape
        cp_context = kwargs.get("cp_context")
        mode = (
            self.mode
            if cp_context is not None
            else ("fused_recurrent" if (q_len <= 64 and not self.training) else self.mode)
        )
        if self.training and mode != "chunk":
            raise AssertionError("Only chunk mode is supported in training.")

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if cp_context is not None:
            if attention_mask is not None:
                raise ValueError(
                    "Ulysses linear attention currently requires packed inputs without a 2D attention_mask.",
                )
            if cp_context.cu_seqlens is None:
                raise ValueError(
                    "Ulysses linear attention requires cu_seqlens metadata from the collator.",
                )
            if use_cache:
                raise ValueError(
                    "Ulysses native FLA CP does not yet support KV/conv cache updates.",
                )
            if mode != "chunk":
                raise ValueError("Ulysses native FLA CP currently supports chunk mode only.")
            cu_seqlens = cp_context.cu_seqlens
        elif attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        merged_input_adapter = getattr(self, "in_proj_qkvz", None)
        if merged_input_adapter is not None and lora_merged_forward_enabled(self):
            offsets = [0]
            for projection in (self.q_proj, self.k_proj, self.v_proj, self.g_proj):
                offsets.append(offsets[-1] + projection.out_features)
            q_weight = merged_input_adapter.merged_weight_for_forward(
                self.q_proj.weight, output_start=offsets[0], output_end=offsets[1]
            )
            k_weight = merged_input_adapter.merged_weight_for_forward(
                self.k_proj.weight, output_start=offsets[1], output_end=offsets[2]
            )
            v_weight = merged_input_adapter.merged_weight_for_forward(
                self.v_proj.weight, output_start=offsets[2], output_end=offsets[3]
            )
            gate_weight = merged_input_adapter.merged_weight_for_forward(
                self.g_proj.weight, output_start=offsets[3], output_end=offsets[4]
            )
            q_input = self._linear_with_contract(self.q_proj, hidden_states, q_weight)
            k_input = self._linear_with_contract(self.k_proj, hidden_states, k_weight)
            v_input = self._linear_with_contract(self.v_proj, hidden_states, v_weight)
            gate_input = self._linear_with_contract(self.g_proj, hidden_states, gate_weight)
            a_input = self.a_proj(hidden_states).float()
            b_input = self.b_proj(hidden_states)
        else:
            q_input = self.q_proj(hidden_states)
            k_input = self.k_proj(hidden_states)
            v_input = self.v_proj(hidden_states)
            a_input = self.a_proj(hidden_states).float()
            b_input = self.b_proj(hidden_states)
            gate_input = self.g_proj(hidden_states) if self.use_gate else None

        fused_lora_delta = None if lora_merged_forward_enabled(self) else self._fused_qkvz_lora_delta(hidden_states)
        if fused_lora_delta is not None:
            q_delta, k_delta, v_delta, gate_delta = fused_lora_delta
            q_input = q_input + q_delta.to(q_input.dtype)
            k_input = k_input + k_delta.to(k_input.dtype)
            v_input = v_input + v_delta.to(v_input.dtype)
            if gate_input is None:
                raise RuntimeError("Fused qkvz LoRA requires a gated GDN layer")
            gate_input = gate_input + gate_delta.to(gate_input.dtype)

        if _is_gdn_contract_enabled() and not self.use_short_conv:
            raise RuntimeError("Exact Qwen3.5 GDN requires short convolution")

        if self.use_short_conv and _is_gdn_contract_enabled():
            if use_cache or last_state is not None:
                raise RuntimeError(
                    "Exact Qwen3.5 trainer GDN supports packed prefill only, not recurrent cache updates"
                )
            if cp_context is None:
                # Fail-closed backstop against the ring/silent-skip fail-open:
                # build_linear_attention_cp_context returns None for ring>1
                # and for missing metadata — an exact GDN layer must never
                # run U1 math on sequence-sharded rows.
                from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

                ps = get_parallel_state()
                if ps.ulysses_size > 1 or ps.ringattn_size > 1:
                    raise RuntimeError(
                        "Exact Qwen3.5 GDN: sequence parallelism is active "
                        f"(ulysses={ps.ulysses_size}, ring={ps.ringattn_size}) but no "
                        "cp_context reached the layer. Ring is unsupported (contract C4) "
                        "and a missing Ulysses context would silently break bytes. "
                        "Fail closed.",
                    )
            q, k, v = causal_conv1d_qkv_contract(
                q_input,
                k_input,
                v_input,
                self.q_conv1d,
                self.k_conv1d,
                self.v_conv1d,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
            conv_state_q = conv_state_k = conv_state_v = None
        elif self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=q_input,
                cache=conv_state_q,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
            k, conv_state_k = self.k_conv1d(
                x=k_input,
                cache=conv_state_k,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
            v, conv_state_v = self.v_conv1d(
                x=v_input,
                cache=conv_state_v,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
        else:
            q = F.silu(q_input)
            k = F.silu(k_input)
            v = F.silu(v_input)
            conv_state_q = conv_state_k = conv_state_v = None

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            repeat_factor = self.num_v_heads // self.num_heads
            q, k = (repeat(x, "... h d -> ... (h g) d", g=repeat_factor) for x in (q, k))

        if _is_gdn_contract_enabled():
            # The serving fused-GDN gating kernel removes the
            # 1-ULP g term (torch softplus vs tl.log(1+tl.exp)); beta is bitwise
            # either way.
            g, beta = bi_fused_gdn_gating(self.A_log, a_input, b_input, self.dt_bias)
        else:
            beta = _sglang_compatible_beta_gate(b_input)
            g = -self.A_log.float().exp() * F.softplus(a_input + self.dt_bias)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None

        if mode == "chunk":
            backend = get_gdn_backend()
            chunk_fn = chunk_gated_delta_rule
            if backend == "flashqla":
                if self.head_k_dim != 128 or self.head_v_dim != 128:
                    warn_cp_fallback_once()
                elif cp_context is not None:
                    chunk_fn = flashqla_chunk_gated_delta_rule_cp
                else:
                    chunk_fn = flashqla_chunk_gated_delta_rule

            chunk_kwargs = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            if cp_context is not None:
                chunk_kwargs["cp_context"] = cp_context
            o, recurrent_state = chunk_fn(**chunk_kwargs)

        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Unsupported mode `{mode}`.")

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            gate = rearrange(gate_input, "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, gate)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self._project_output_linear(o)
        if attention_mask is not None and indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
