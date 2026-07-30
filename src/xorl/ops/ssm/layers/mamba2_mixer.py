from __future__ import annotations

# Trainable Mamba2 (SSD) mixer matching transformers' NemotronHMamba2Mixer.
# Parameter names match the HF mixer 1:1 (in_proj, conv1d.{weight,bias}, A_log, D,
# dt_bias, norm.weight, out_proj) so checkpoint loading is a pass-through.
#
# Deviations from the HF torch_forward fallback (both deliberate):
# - dt is clamped to `time_step_limit` (floor AND ceiling), matching the HF cuda
#   kernel path and the shipped checkpoints' config; the HF torch fallback only
#   applies a `time_step_min` floor.
# - the chunked SSD scan uses the corrected inter-chunk recurrence (the HF
#   nemotron_h torch fallback drifts from the true SSD recurrence for
#   seq_len > chunk_size; see xorl.ops.ssm.ops.ssd).
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.ops.ssm.modules.gated_norm import GroupRMSNormGated
from xorl.ops.ssm.ops.conv import causal_depthwise_conv1d
from xorl.ops.ssm.ops.ssd import ssd_chunked


class Mamba2Mixer(nn.Module):
    """Mamba2 (SSD) mixer layer. Training-only for now: no inference cache support.

    Fused `in_proj` output layout: [gate d_inner | x d_inner | B n_groups*state | C n_groups*state | dt num_heads].
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        ssm_state_size: int,
        conv_kernel: int = 4,
        use_conv_bias: bool = True,
        chunk_size: int = 128,
        activation: str = "silu",
        time_step_limit: Sequence[float] = (0.0, float("inf")),
        layer_norm_epsilon: float = 1e-5,
        use_bias: bool = False,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if num_heads % n_groups != 0:
            raise ValueError(f"n_groups={n_groups} must divide num_heads={num_heads}.")
        if activation not in {"silu", "swish"}:
            raise ValueError(f"Unsupported activation: {activation}")
        if len(time_step_limit) != 2:
            raise ValueError(f"time_step_limit must be (min, max), got {time_step_limit}.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.activation = activation
        self.time_step_limit = (float(time_step_limit[0]), float(time_step_limit[1]))
        self.layer_idx = layer_idx

        self.intermediate_size = num_heads * head_dim
        self.conv_dim = self.intermediate_size + 2 * n_groups * ssm_state_size
        projection_size = self.intermediate_size + self.conv_dim + num_heads

        self.in_proj = nn.Linear(hidden_size, projection_size, bias=use_bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        self.dt_bias._no_weight_decay = True
        # S4D real initialization (not discretized); the model loads A_log from the checkpoint.
        A = torch.arange(1, num_heads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(num_heads))
        self.D._no_weight_decay = True

        self.norm = GroupRMSNormGated(
            self.intermediate_size, group_size=self.intermediate_size // n_groups, eps=layer_norm_epsilon
        )
        self.out_proj = nn.Linear(self.intermediate_size, hidden_size, bias=use_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any | None]:
        """SSD mixer forward.

        Args:
            hidden_states: ``[batch, seq_len, hidden_size]``; with ``cu_seqlens``, a packed row
                ``[1, total_len, hidden_size]``.
            attention_mask: optional ``[batch, seq_len]`` mask where 0 marks padding; padded
                positions are zeroed before the projection and after the convolution
                (HF Mamba2 semantics, correct for right padding).
            cu_seqlens: optional cumulative sequence boundaries ``[num_seqs + 1]`` for packed
                varlen input (flattened-pack convention, matching GatedDeltaNet); conv and SSM
                state are reset at every boundary.

        Returns:
            ``(output [batch, seq_len, hidden_size], None, past_key_values)``.
        """
        del output_attentions
        if use_cache or past_key_values is not None:
            raise NotImplementedError("Mamba2Mixer is training-only: inference caching is not supported yet.")
        if kwargs.get("cp_context") is not None:
            raise NotImplementedError("Mamba2Mixer does not support CP inputs yet.")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError("Expected `attention_mask` with shape [batch_size, seq_len] where 0 marks padding.")

        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        seq_idx = None
        if cu_seqlens is not None:
            seq_idx = self._build_seq_idx(cu_seqlens, batch_size, seq_len, hidden_states.device)

        if attention_mask is not None:
            # Tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        projected_states = self.in_proj(hidden_states)
        gate, hidden_states_b_c, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        hidden_states_b_c = causal_depthwise_conv1d(
            hidden_states_b_c, self.conv1d.weight, self.conv1d.bias, self.activation, seq_idx=seq_idx
        )
        if attention_mask is not None:
            hidden_states_b_c = (hidden_states_b_c * attention_mask[:, :, None]).to(hidden_states_b_c.dtype)

        groups_state_size = self.n_groups * self.ssm_state_size
        hidden_states, b, c = hidden_states_b_c.split(
            [self.intermediate_size, groups_state_size, groups_state_size], dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, *self.time_step_limit)
        A = -torch.exp(self.A_log.float())

        y = ssd_chunked(
            hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim),
            dt,
            A,
            b.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
            c.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
            self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
        )

        scan_output = self.norm(y.reshape(batch_size, seq_len, -1), gate)
        return self.out_proj(scan_output.to(dtype)), None, past_key_values

    @staticmethod
    def _build_seq_idx(
        cu_seqlens: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map ``cu_seqlens`` boundaries to a per-position sequence index ``[batch, seq_len]``."""
        if batch_size != 1:
            raise ValueError(f"Packed varlen input must have batch size 1, got {batch_size}.")
        cu = cu_seqlens.to(device=device, dtype=torch.long).flatten()
        if cu.numel() < 2 or cu[0] != 0 or cu[-1] > seq_len:
            raise ValueError(f"Invalid cu_seqlens for seq_len={seq_len}: {cu.tolist()}.")
        positions = torch.arange(seq_len, device=device)
        # Trailing positions beyond cu[-1] (right padding) land in their own segment.
        return (torch.searchsorted(cu, positions, right=True) - 1).unsqueeze(0)
