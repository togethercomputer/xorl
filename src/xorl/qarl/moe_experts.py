"""NVFP4 weight-only QARL fake-quant for MoE expert weights.

``MoEExperts`` stores its experts as two 3D ``nn.Parameter`` tensors in GKN
layout (``gate_up_proj [E, H, 2I]`` and ``down_proj [E, I, H]``), not as
``nn.Linear`` modules, so the dense ``inject_qarl_into_model`` Linear pass never
touches them. :class:`QARLMoEExperts` adds per-expert NVFP4 round-to-nearest
fake-quant with a straight-through backward.

It is installed **in place** (``module.__class__`` swap) so every existing
attribute is preserved: the two ``nn.Parameter`` masters (optimizer state, FSDP2
sharding, DCP checkpoint keys), the backend/EP settings, biases, gated flag, etc.
``forward`` fake-quantizes the two weight tensors then *shadows* the parameter
names around the inherited ``MoEExperts.forward`` so the unchanged dispatch path
(EP / triton / DeepEP / eager / native, gated or not) consumes the fake-quant
weights.

Weight fake-quant is **purely rank-local** under Expert Parallelism: each rank
holds only its local experts (``Shard(0)`` on the expert dim) and the per-expert
NVFP4 scales need no cross-rank reduction. FSDP2 all-gathers the (local-expert)
weight before ``forward`` runs, so the fake-quant operates on the full local
tensor. This composes with EP dispatch because the group GEMM receives the full
local expert stack — sidestepping the per-routed-expert eager path that blocked
EP in the standalone qat-moe wrapper.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

from xorl.models.layers.moe.common import split_gate_up_proj
from xorl.models.layers.moe.experts import MoEExperts
from xorl.ops.quantize.nvfp4_fake_quant import (
    _fake_quantize_3d_experts,
    _fake_quantize_3d_fused_gate_up,
    fake_quantize_activation_nvfp4,
)


_DEFAULT_QARL_GROUP_SIZE = 16

logger = logging.getLogger(__name__)


class QARLMoEExperts(MoEExperts):
    """``MoEExperts`` with NVFP4 weight-only STE fake-quant in the forward."""

    # Set by convert_moe_experts_to_qarl (class-swap installs no __init__).
    qarl_group_size: int = _DEFAULT_QARL_GROUP_SIZE
    qarl_quantize_weight: bool = True
    qarl_quantize_activation: bool = False  # W4A4: fake-quant the gate/up (w13) input activations
    qarl_format: str = "nvfp4"
    # One-time guard so the moe_act-incompatibility warning is logged once, not per step.
    _qarl_moe_act_warned: bool = False

    # Override the base gate/up views to drop the split-view ``.grad`` plumbing:
    # under fake-quant the shadowed ``gate_up_proj`` is a non-leaf tensor, and
    # touching its ``.grad`` would emit a (cosmetic) non-leaf warning every step.
    # Nothing external reads these views' ``.grad`` (only the base property set it).
    @property
    def gate_proj(self):
        if not self.gated:
            raise AttributeError("non-gated MoEExperts has no gate_proj")
        gate, _ = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        return gate

    @property
    def up_proj(self):
        if not self.gated:
            return self.gate_up_proj
        _, up = split_gate_up_proj(self.gate_up_proj, self.intermediate_size)
        return up

    def _qarl_fake_quant_weights(self):
        """STE fake-quant of the two 3D GKN expert tensors (per-expert, block=K)."""
        if self.gated:
            gate_up_fq = _fake_quantize_3d_fused_gate_up(
                self.gate_up_proj, self.intermediate_size, self.qarl_group_size
            )
        else:
            # Non-gated: gate_up_proj is [E, H, I] (GKN, contraction dim K=H).
            gate_up_fq = _fake_quantize_3d_experts(self.gate_up_proj, self.qarl_group_size)
        down_fq = _fake_quantize_3d_experts(self.down_proj, self.qarl_group_size)
        return gate_up_fq, down_fq

    @contextmanager
    def _qarl_shadow_moe_impl(self):
        """Temporarily shadow ``moe_implementation`` -> ``triton_w4a4`` for the EP down-quant.

        When activation quant is on and the backend is ``triton``, swap in the
        ``triton_w4a4`` EP compute (registered in ``moe.backend``) so ``_ep_forward``
        NVFP4-fake-quantizes the down-GEMM input (the SwiGLU intermediate). This
        completes W4A4 on the experts: the gate/up input is quantized pre-dispatch in
        ``forward`` and the down input is quantized inside the triton kernel.

        ``moe_implementation`` is a plain ``str`` attribute, so a direct
        assign/restore is safe. The swap is a no-op for any non-``triton`` backend (no
        ``triton_w4a4`` variant exists; gate/up stays quantized, down stays bf16 — the
        pre-existing partial-W4A4 behaviour) and when activation quant is off.

        Note: ``_ep_forward`` has a ``_moe_act`` recompute branch keyed on
        ``EP_EXPERT_COMPUTE_MOE_ACT``, which has no ``triton_w4a4`` entry — so with
        ``_moe_act`` on it falls through to the standard ``EP_EXPERT_COMPUTE['triton_w4a4']``.
        That still fake-quantizes the down input correctly, but the moe_act activation-
        checkpoint recompute is silently NOT applied (gate/up activations stay live), so we
        log a one-time warning rather than letting the memory regression pass unnoticed.
        """
        if not (self.qarl_quantize_activation and self.moe_implementation == "triton"):
            yield
            return
        if getattr(self, "_moe_act", False) and not type(self)._qarl_moe_act_warned:
            type(self)._qarl_moe_act_warned = True
            logger.warning(
                "QARL W4A4: moe_act (activation-checkpoint recompute) has no triton_w4a4 EP "
                "variant, so the down-quant path falls back to the non-recompute kernel. The "
                "down input is still NVFP4-fake-quantized, but gate/up activations are kept "
                "live -> higher activation memory than moe_act implies. Disable moe_act or add "
                "an EP_EXPERT_COMPUTE_MOE_ACT['triton_w4a4'] variant to recover the savings."
            )
        original = self.moe_implementation
        self.moe_implementation = "triton_w4a4"
        try:
            yield
        finally:
            self.moe_implementation = original

    @contextmanager
    def _qarl_shadow_weights(self, gate_up_fq, down_fq):
        """Temporarily shadow the parameter names with the fake-quant tensors.

        Writing into ``self.__dict__`` bypasses ``nn.Module.__setattr__`` (which
        forbids assigning a non-``Parameter`` to a registered parameter name) and
        takes precedence over ``nn.Module.__getattr__``, so the inherited forward
        and the ``gate_proj`` / ``up_proj`` properties read the fake-quant tensors.
        The real ``nn.Parameter`` objects are restored on exit.
        """
        self.__dict__["gate_up_proj"] = gate_up_fq
        self.__dict__["down_proj"] = down_fq
        try:
            yield
        finally:
            self.__dict__.pop("gate_up_proj", None)
            self.__dict__.pop("down_proj", None)

    def forward(self, *args, **kwargs):
        # W4A4: fake-quant the gate/up input (hidden_states = first positional arg, or kwarg)
        # before the inherited forward. This sits PRE-dispatch so it covers the routed
        # per-expert gate_up GEMM under EP. The down (w2) intermediate is fake-quantized
        # inside the triton EP kernel via the moe_implementation -> triton_w4a4 shadow
        # below (completing 100% W4A4 on the experts under the triton backend).
        if self.qarl_quantize_activation:
            if args and isinstance(args[0], torch.Tensor):
                args = (fake_quantize_activation_nvfp4(args[0], self.qarl_group_size),) + args[1:]
            elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
                kwargs = {
                    **kwargs,
                    "hidden_states": fake_quantize_activation_nvfp4(kwargs["hidden_states"], self.qarl_group_size),
                }
        # Shadow moe_implementation -> triton_w4a4 (no-op unless act-quant + triton) so the
        # EP down-GEMM input is NVFP4-fake-quantized; covers BOTH super().forward sites.
        with self._qarl_shadow_moe_impl():
            if not self.qarl_quantize_weight:
                return super().forward(*args, **kwargs)
            gate_up_fq, down_fq = self._qarl_fake_quant_weights()
            with self._qarl_shadow_weights(gate_up_fq, down_fq):
                return super().forward(*args, **kwargs)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, qarl_format={self.qarl_format}, qarl_group_size={self.qarl_group_size}"


def convert_moe_experts_to_qarl(
    experts: MoEExperts,
    *,
    group_size: int = _DEFAULT_QARL_GROUP_SIZE,
    quantize_weight: bool = True,
    quantize_activation: bool = False,
) -> QARLMoEExperts:
    """In-place class-swap an existing ``MoEExperts`` to :class:`QARLMoEExperts`.

    Preserves the two weight ``nn.Parameter`` masters, buffers, EP/FSDP2 wrapping,
    backend settings and biases — only the ``forward`` behaviour changes.
    """
    if isinstance(experts, QARLMoEExperts):
        return experts
    if not isinstance(experts, MoEExperts):
        raise TypeError(f"convert_moe_experts_to_qarl expects MoEExperts, got {type(experts).__name__}")
    experts.__class__ = QARLMoEExperts
    experts.qarl_group_size = int(group_size)
    experts.qarl_quantize_weight = bool(quantize_weight)
    experts.qarl_quantize_activation = bool(quantize_activation)
    experts.qarl_format = "nvfp4"
    return experts
