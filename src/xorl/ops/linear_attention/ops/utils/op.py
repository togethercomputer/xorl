from __future__ import annotations

# Adapted from flash-linear-attention/fla/ops/utils/op.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from xorl.ops.linear_attention.utils import IS_GATHER_SUPPORTED

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":

    @triton.jit
    def exp(x):
        return tldevice.fast_expf(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tldevice.exp2(x.to(tl.float32))

else:

    @triton.jit
    def exp(x):
        return tl.exp(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tl.math.exp2(x.to(tl.float32))


if not IS_GATHER_SUPPORTED:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        del src, index, axis, _builder
        return None

else:
    gather = tl.gather


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:

    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        del base, shape, strides, block_shape, _builder
        return None
