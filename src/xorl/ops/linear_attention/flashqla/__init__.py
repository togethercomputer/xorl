# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

__version__ = "0.1.0"

# Local (xorl) modification: the kernels call `T.gemm_v1`, which stock tilelang >=0.1.9 no
# longer exposes. Re-add it (fast tl::gemm_ss template) before the kernels are traced.
from xorl.ops.linear_attention.tilelang_gemm_v1 import patch as _patch_gemm_v1


_patch_gemm_v1()

from xorl.ops.linear_attention.flashqla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)


__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
]
