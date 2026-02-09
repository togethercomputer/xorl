import triton
import triton.language as tl


@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask: tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)


@triton.jit
def load_with_pred_2d(
    ptr,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
    other=0,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_0 & mask_1, other=other)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, mask_0, other=other)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_1, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def store_with_pred_2d(
    ptr,
    value,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_0 & mask_1)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, mask_0)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_1)
    else:
        tl.store(ptr, value)


@triton.jit
def load_block_with_pred_2d(ptr, skip_boundary_check_0: tl.constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(1,))
    else:
        return tl.load(ptr)


@triton.jit
def store_block_with_pred_2d(ptr, value, skip_boundary_check_0: tl.constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(1,))
    else:
        tl.store(ptr, value)
