# Copyright 2025 xorl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Group GEMM kernels
from .group_gemm import group_gemm_same_mn, group_gemm_same_nk

# MoE operations
from .moe import (
    expert_histogram,
    moe_add_gather,
    moe_gather,
    moe_index_compute,
    moe_scatter,
)

# LoRA utilities
from .lora_utils import (
    init_lora_weights_stacked,
    compute_lora_scaling,
    merge_lora_weights_stacked,
    unmerge_lora_weights_stacked,
    get_lora_delta_weight_stacked,
)

__all__ = [
    # Group GEMM
    "group_gemm_same_mn",
    "group_gemm_same_nk",
    # MoE operations
    "expert_histogram",
    "moe_add_gather",
    "moe_gather",
    "moe_index_compute",
    "moe_scatter",
    # LoRA utilities
    "init_lora_weights_stacked",
    "compute_lora_scaling",
    "merge_lora_weights_stacked",
    "unmerge_lora_weights_stacked",
    "get_lora_delta_weight_stacked",
]
