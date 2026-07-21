from __future__ import annotations

# Adapted from flash-linear-attention/fla/layers/gated_deltanet.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.
import math
import os
import warnings
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

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
from xorl.ops.linear_attention.modules.bi_contract import bi_fused_gdn_gating, is_gdn_contract_enabled
from xorl.ops.linear_attention.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)


def _sglang_compatible_beta_gate(b_input: torch.Tensor) -> torch.Tensor:
    beta = b_input.float().sigmoid()
    if beta.dtype != b_input.dtype:
        beta = beta.to(dtype=b_input.dtype).float()
    return beta


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _sglang_chunk_forward_diagnostic_enabled() -> bool:
    return _env_flag("XORL_GDN_SGLANG_CHUNK_FORWARD")


def _diagnostic_layer_enabled(env_name: str, layer_idx: int | None) -> bool:
    split_layers = os.environ.get(env_name, "").strip()
    if not split_layers or split_layers.lower() in {"all", "*"}:
        return True
    try:
        enabled_layers = {int(item.strip()) for item in split_layers.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(f"Invalid {env_name}={split_layers!r}; expected comma-separated layer indices") from exc
    return layer_idx in enabled_layers


def _depthwise_causal_conv_split(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.LongTensor | None,
) -> torch.Tensor:
    def apply_one(seq: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(
            seq.transpose(1, 2),
            weight,
            bias,
            padding=weight.shape[-1] - 1,
            groups=weight.shape[0],
        )
        y = y[:, :, : seq.shape[1]].transpose(1, 2)
        if activation in {None, "identity"}:
            return y
        if activation in {"silu", "swish"}:
            return F.silu(y)
        raise ValueError(f"Unsupported activation: {activation}")

    if cu_seqlens is None:
        return apply_one(x)
    if x.shape[0] != 1:
        raise ValueError("Packed varlen GDN TP-split diagnostic expects batch size 1.")
    outputs = [
        apply_one(x[:, int(start) : int(end)])
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ]
    return torch.cat(outputs, dim=1) if outputs else x.new_zeros(x.shape)


def _sglang_chunk_gated_delta_rule_forward_only(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v, g, beta)):
        raise RuntimeError(
            "XORL_GDN_SGLANG_CHUNK_FORWARD is a forward-only diagnostic path. "
            "Use it only for /forward K3 replay, not forward_backward training."
        )
    if output_final_state:
        raise RuntimeError("XORL_GDN_SGLANG_CHUNK_FORWARD does not support recurrent state output.")

    from sglang.srt.layers.attention.fla.chunk import (  # noqa: PLC0415
        chunk_gated_delta_rule as sglang_chunk_gated_delta_rule,
    )

    num_states = q.shape[0] if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    if initial_state is None:
        initial_state = torch.zeros(
            num_states,
            v.shape[-2],
            v.shape[-1],
            k.shape[-1],
            device=q.device,
            dtype=q.dtype,
        )
        initial_state_indices = torch.arange(num_states, device=q.device, dtype=torch.long)
    else:
        if cu_seqlens is not None and initial_state.shape[0] != num_states:
            raise ValueError(
                "XORL_GDN_SGLANG_CHUNK_FORWARD expected one recurrent state per packed sequence: "
                f"{initial_state.shape[0]} != {num_states}"
            )
        initial_state_indices = torch.arange(initial_state.shape[0], device=q.device, dtype=torch.long)

    out, _, state = sglang_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return out, state


def _conv_contract_enabled() -> bool:
    return _env_flag("XORL_GDN_CONV_CONTRACT")


def _packed_segment_loop_enabled() -> bool:
    return os.getenv("XORL_GDN_PACKED_SEGMENT_LOOP", "0").strip().lower() in {"1", "true", "yes", "on"}


def _fused_projection_enabled() -> bool:
    return os.getenv("XORL_GDN_FUSED_PROJECTIONS", "0").strip().lower() in {"1", "true", "yes", "on"}


class GatedDeltaNet(nn.Module):
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

    def _project_output_linear(self, o: torch.Tensor) -> torch.Tensor:
        split_count = int(os.environ.get("XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT", "0") or 0)
        if split_count <= 1:
            return self.o_proj(o)

        split_layers = os.environ.get("XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT_LAYERS", "").strip()
        if split_layers and split_layers.lower() not in {"all", "*"}:
            try:
                enabled_layers = {int(item.strip()) for item in split_layers.split(",") if item.strip()}
            except ValueError as exc:
                raise ValueError(
                    f"Invalid XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT_LAYERS={split_layers!r}; "
                    "expected comma-separated layer indices"
                ) from exc
            if self.layer_idx not in enabled_layers:
                return self.o_proj(o)

        weight = self.o_proj.weight
        input_dim = o.shape[-1]
        if input_dim % split_count != 0 or weight.shape[1] != input_dim:
            return self.o_proj(o)

        sum_fp32 = _env_flag("XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT_SUM_FP32")
        keep_fp32 = _env_flag("XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT_KEEP_FP32")
        chunk_size = input_dim // split_count
        output = None
        output_dtype = None
        for idx in range(split_count):
            start = idx * chunk_size
            end = start + chunk_size
            partial = F.linear(
                o[..., start:end],
                weight[:, start:end],
                self.o_proj.bias if idx == 0 else None,
            )
            output_dtype = partial.dtype
            if sum_fp32:
                partial = partial.float()
            output = partial if output is None else output + partial
        if sum_fp32 and output_dtype is not None and not keep_fp32:
            output = output.to(output_dtype)
        return output

    def _diagnostic_tp_split_count(self) -> int:
        split_count = int(os.environ.get("XORL_GDN_DIAGNOSTIC_TP_SPLIT", "0") or 0)
        if split_count <= 1:
            return 0
        if not _diagnostic_layer_enabled("XORL_GDN_DIAGNOSTIC_TP_SPLIT_LAYERS", self.layer_idx):
            return 0
        if self.num_heads % split_count != 0 or self.num_v_heads % split_count != 0:
            return 0
        return split_count

    def _forward_diagnostic_tp_split(
        self,
        hidden_states: torch.Tensor,
        *,
        split_count: int,
        cu_seqlens: torch.LongTensor | None,
    ) -> torch.Tensor:
        key_heads_per_rank = self.num_heads // split_count
        value_heads_per_rank = self.num_v_heads // split_count
        if value_heads_per_rank > key_heads_per_rank and value_heads_per_rank % key_heads_per_rank != 0:
            raise ValueError(
                "XORL_GDN_DIAGNOSTIC_TP_SPLIT requires local value heads to be divisible by local key heads: "
                f"value_heads={value_heads_per_rank}, key_heads={key_heads_per_rank}"
            )
        key_dim_per_rank = key_heads_per_rank * self.head_k_dim
        value_dim_per_rank = value_heads_per_rank * self.head_v_dim

        final = hidden_states.new_zeros(*hidden_states.shape[:-1], self.hidden_size)
        for rank in range(split_count):
            key_start = rank * key_dim_per_rank
            key_end = key_start + key_dim_per_rank
            value_start = rank * value_dim_per_rank
            value_end = value_start + value_dim_per_rank
            value_head_start = rank * value_heads_per_rank
            value_head_end = value_head_start + value_heads_per_rank

            q_input = F.linear(hidden_states, self.q_proj.weight[key_start:key_end])
            k_input = F.linear(hidden_states, self.k_proj.weight[key_start:key_end])
            v_input = F.linear(hidden_states, self.v_proj.weight[value_start:value_end])
            a_input = F.linear(hidden_states, self.a_proj.weight[value_head_start:value_head_end]).float()
            b_input = F.linear(hidden_states, self.b_proj.weight[value_head_start:value_head_end])
            gate_input = F.linear(hidden_states, self.g_proj.weight[value_start:value_end]) if self.use_gate else None

            if self.use_short_conv:
                q = _depthwise_causal_conv_split(
                    q_input,
                    self.q_conv1d.weight[key_start:key_end],
                    self.q_conv1d.bias[key_start:key_end] if self.q_conv1d.bias is not None else None,
                    self.q_conv1d.activation,
                    cu_seqlens,
                )
                k = _depthwise_causal_conv_split(
                    k_input,
                    self.k_conv1d.weight[key_start:key_end],
                    self.k_conv1d.bias[key_start:key_end] if self.k_conv1d.bias is not None else None,
                    self.k_conv1d.activation,
                    cu_seqlens,
                )
                v = _depthwise_causal_conv_split(
                    v_input,
                    self.v_conv1d.weight[value_start:value_end],
                    self.v_conv1d.bias[value_start:value_end] if self.v_conv1d.bias is not None else None,
                    self.v_conv1d.activation,
                    cu_seqlens,
                )
            else:
                q = F.silu(q_input)
                k = F.silu(k_input)
                v = F.silu(v_input)

            q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
            k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
            if value_heads_per_rank > key_heads_per_rank:
                repeat_factor = value_heads_per_rank // key_heads_per_rank
                q, k = (repeat(x, "... h d -> ... (h g) d", g=repeat_factor) for x in (q, k))

            beta = _sglang_compatible_beta_gate(b_input)
            if self.allow_neg_eigval:
                beta = beta * 2.0
            g = -self.A_log[value_head_start:value_head_end].float().exp() * F.softplus(
                a_input + self.dt_bias[value_head_start:value_head_end]
            )

            if _sglang_chunk_forward_diagnostic_enabled():
                local_o, _ = _sglang_chunk_gated_delta_rule_forward_only(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=False,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                local_o, _ = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=False,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )

            if self.use_gate:
                gate = rearrange(gate_input, "... (h d) -> ... h d", d=self.head_v_dim)
                local_o = self.o_norm(local_o, gate)
            else:
                local_o = self.o_norm(local_o)

            local_o = rearrange(local_o, "b t h d -> b t (h d)")
            partial = F.linear(
                local_o,
                self.o_proj.weight[:, value_start:value_end],
                self.o_proj.bias if rank == 0 else None,
            )
            final = final + partial
        return final

    def forward(
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

        diagnostic_tp_split = self._diagnostic_tp_split_count()
        if diagnostic_tp_split:
            if _conv_contract_enabled():
                raise RuntimeError("XORL_GDN_CONV_CONTRACT and XORL_GDN_DIAGNOSTIC_TP_SPLIT are mutually exclusive.")
            if cp_context is not None:
                raise RuntimeError("XORL_GDN_DIAGNOSTIC_TP_SPLIT does not support CP.")
            if mode != "chunk":
                raise RuntimeError("XORL_GDN_DIAGNOSTIC_TP_SPLIT supports chunk mode only.")
            if use_cache or last_state is not None:
                raise RuntimeError("XORL_GDN_DIAGNOSTIC_TP_SPLIT does not support recurrent/cache state.")
            o = self._forward_diagnostic_tp_split(
                hidden_states,
                split_count=diagnostic_tp_split,
                cu_seqlens=cu_seqlens,
            )
            if attention_mask is not None and indices is not None:
                o = pad_input(o.squeeze(0), indices, batch_size, q_len)
            return o, None, past_key_values

        if _fused_projection_enabled() and self.use_gate:
            qkvz_weight = torch.cat(
                (
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                    self.g_proj.weight,
                ),
                dim=0,
            )
            qkvz = F.linear(hidden_states, qkvz_weight)
            q_input, k_input, v_input, gate_input = qkvz.split(
                (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
                dim=-1,
            )
            ba_weight = torch.cat(
                (self.b_proj.weight, self.a_proj.weight),
                dim=0,
            )
            b_input, a_input = F.linear(hidden_states, ba_weight).split(
                (self.num_v_heads, self.num_v_heads),
                dim=-1,
            )
            a_input = a_input.float()
        else:
            q_input = self.q_proj(hidden_states)
            k_input = self.k_proj(hidden_states)
            v_input = self.v_proj(hidden_states)
            a_input = self.a_proj(hidden_states).float()
            b_input = self.b_proj(hidden_states)
            gate_input = self.g_proj(hidden_states) if self.use_gate else None

        diagnostic_capture = getattr(self, "_diagnostic_capture_component", None)
        if diagnostic_capture is not None:
            diagnostic_capture("gdn_q_input", q_input)
            diagnostic_capture("gdn_k_input", k_input)
            diagnostic_capture("gdn_v_input", v_input)
            diagnostic_capture("gdn_a_input", a_input)
            diagnostic_capture("gdn_b_input", b_input)
            diagnostic_capture("gdn_gate_input", gate_input)

        if _conv_contract_enabled() and not self.use_short_conv:
            raise RuntimeError("XORL_GDN_CONV_CONTRACT is armed but this GatedDeltaNet has use_short_conv=False.")

        if self.use_short_conv and _conv_contract_enabled():
            if cp_context is not None:
                raise RuntimeError("XORL_GDN_CONV_CONTRACT does not support CP yet (conv prefix exchange).")
            if use_cache or last_state is not None:
                raise RuntimeError(
                    "XORL_GDN_CONV_CONTRACT covers prefill only; conv cache/decode comes with the decode contract."
                )
            q, k, v = causal_conv1d_qkv_contract(
                q_input,
                k_input,
                v_input,
                self.q_conv1d,
                self.k_conv1d,
                self.v_conv1d,
                cu_seqlens=cu_seqlens,
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

        if diagnostic_capture is not None:
            diagnostic_capture("gdn_conv_q", q)
            diagnostic_capture("gdn_conv_k", k)
            diagnostic_capture("gdn_conv_v", v)

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            repeat_factor = self.num_v_heads // self.num_heads
            q, k = (repeat(x, "... h d -> ... (h g) d", g=repeat_factor) for x in (q, k))

        if is_gdn_contract_enabled():
            # GDN contract: serving's fused_gdn_gating kernel kills the
            # 1-ULP g term (torch softplus vs tl.log(1+tl.exp)); beta is bitwise
            # either way.
            g, beta = bi_fused_gdn_gating(self.A_log, a_input, b_input, self.dt_bias)
        else:
            beta = _sglang_compatible_beta_gate(b_input)
            g = -self.A_log.float().exp() * F.softplus(a_input + self.dt_bias)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        if diagnostic_capture is not None:
            diagnostic_capture("gdn_g", g)
            diagnostic_capture("gdn_beta", beta)

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

            if _sglang_chunk_forward_diagnostic_enabled():
                if cp_context is not None:
                    raise RuntimeError("XORL_GDN_SGLANG_CHUNK_FORWARD does not support CP.")
                o, recurrent_state = _sglang_chunk_gated_delta_rule_forward_only(
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
            elif (
                _packed_segment_loop_enabled()
                and cu_seqlens is not None
                and cp_context is None
                and recurrent_state is None
                and not use_cache
            ):
                outputs: list[torch.Tensor] = []
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
                    if end <= start:
                        continue
                    seg_o, _ = chunk_fn(
                        q=q[:, start:end],
                        k=k[:, start:end],
                        v=v[:, start:end],
                        g=g[:, start:end],
                        beta=beta[:, start:end],
                        initial_state=None,
                        output_final_state=False,
                        cu_seqlens=None,
                        use_qk_l2norm_in_kernel=True,
                    )
                    outputs.append(seg_o)
                o = torch.cat(outputs, dim=1) if outputs else v.new_zeros(v.shape)
            else:
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

        if diagnostic_capture is not None:
            diagnostic_capture("gdn_scan_out", o)

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
        if diagnostic_capture is not None:
            diagnostic_capture("gdn_normed", o)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self._project_output_linear(o)
        if attention_mask is not None and indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
