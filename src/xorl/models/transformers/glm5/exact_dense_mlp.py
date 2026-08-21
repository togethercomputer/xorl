"""Proposed exact active-LoRA TP1 dense MLP for GLM-5.2.

The composite is rooted at the logical MLP so its adapter factors keep the
canonical ``gate_proj.*``, ``up_proj.*``, and ``down_proj.*`` paths.  It is a
construction-independent component only: model replacement, checkpoint
loading, and distributed ownership must be admitted separately.
"""

from __future__ import annotations

import torch
from torch import Tensor

from xorl.models.transformers.glm5.exact_gate_up_qlora import (
    Glm52ExactTP1FusedGateUpBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_lora_contract import glm52_exact_lora_scaling
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.ops.exact.fused_silu_and_mul import one_round_swiglu


GLM52_EXACT_TP1_DENSE_MLP_CONTRACT_VERSION = "glm52_exact_tp1_dense_mlp_qlora_v2"


class Glm52ExactTP1DenseMLP(Glm52ExactTP1FusedGateUpBlockFP8QLoRA):
    """Compose the literal fused gate/up, production SwiGLU, and exact down.

    Inheriting the fused gate/up leaf places its logical ``gate_proj`` and
    ``up_proj`` factor holders directly below this MLP root.  ``down_proj`` is
    the existing exact generic projection, so the six trainable factor names
    match the canonical unfused GLM adapter inventory without aliases.
    """

    contract_version = GLM52_EXACT_TP1_DENSE_MLP_CONTRACT_VERSION
    logical_factor_names = (
        "gate_proj.lora_A",
        "gate_proj.lora_B",
        "up_proj.lora_A",
        "up_proj.lora_B",
        "down_proj.lora_A",
        "down_proj.lora_B",
    )

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        r: int = 1,
        lora_alpha: int = 1,
        bias: bool = False,
        device: torch.device | str | None = None,
        enable_aqn: bool = False,
        tp_size: int = 1,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            r=r,
            lora_alpha=lora_alpha,
            bias=bias,
            device=device,
            enable_aqn=enable_aqn,
            tp_size=tp_size,
        )
        self.hidden_size = int(hidden_size)
        self.down_proj = Glm52ExactTP1BlockFP8QLoRALinear(
            intermediate_size,
            hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            bias=False,
            device=device,
            enable_aqn=False,
        )
        self._exact_mlp_source_fqn: str | None = None
        self._exact_gate_source_fqn: str | None = None
        self._exact_up_source_fqn: str | None = None

    def bind_checkpoint_sources(self, mlp_fqn: str) -> None:
        """Bind the three canonical checkpoint sources without aliasing modules."""

        if not mlp_fqn or mlp_fqn.endswith("."):
            raise ValueError(f"Invalid GLM-5.2 dense MLP checkpoint prefix: {mlp_fqn!r}")
        if self._exact_mlp_source_fqn not in {None, mlp_fqn}:
            raise RuntimeError(
                "GLM-5.2 exact dense MLP checkpoint sources are immutable once bound: "
                f"existing={self._exact_mlp_source_fqn!r}, requested={mlp_fqn!r}"
            )
        self._exact_mlp_source_fqn = mlp_fqn
        self._exact_gate_source_fqn = f"{mlp_fqn}.gate_proj"
        self._exact_up_source_fqn = f"{mlp_fqn}.up_proj"
        self.down_proj._source_fqn = f"{mlp_fqn}.down_proj"
        self.down_proj._is_prequantized = True
        self.down_proj._source_quant_format = "block_fp8"
        self.down_proj._merge_sources = None
        self.down_proj._qlora_expected_skip_keys = {"weight", "weight_scale_inv"}

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        """Keep all three logical projections on the sole admitted adapter row."""

        self.scaling = glm52_exact_lora_scaling(lora_rank, lora_alpha)
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha
        self.down_proj.set_runtime_lora_config(lora_rank, lora_alpha)

    def _validate_dense_mlp_runtime_contract(self) -> None:
        gate_up_contract = (
            self.r,
            self.active_r,
            self.lora_alpha,
            self.active_lora_alpha,
            self.scaling,
        )
        down_contract = (
            self.down_proj.r,
            self.down_proj.active_r,
            self.down_proj.lora_alpha,
            self.down_proj.active_lora_alpha,
            self.down_proj._active_scaling(),
        )
        expected = (self.r, self.r, self.lora_alpha, self.lora_alpha, self.scaling)
        if gate_up_contract != expected or down_contract != expected:
            raise RuntimeError(
                "GLM-5.2 exact TP1 dense MLP requires one consistent adapter contract for gate, up, and down"
            )

    def forward(self, input: Tensor) -> Tensor:
        self._validate_dense_mlp_runtime_contract()
        gate_up = Glm52ExactTP1FusedGateUpBlockFP8QLoRA.forward(self, input)
        # Serving's exact mode computes the one-round FP32 SwiGLU
        # (SiluAndMul.forward_exact, xorl-sglang f10b907d8).
        activated = one_round_swiglu(gate_up)
        return self.down_proj(activated)


__all__ = [
    "GLM52_EXACT_TP1_DENSE_MLP_CONTRACT_VERSION",
    "Glm52ExactTP1DenseMLP",
]
