"""Shared FP4 E2M1 encode/decode helpers for MXFP4 and NVFP4 kernels."""
import triton
import triton.language as tl


FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0


@triton.jit
def _fp4_encode(val):
    """Hybrid encode: FMA for linear codes 0-3, comparisons for log codes 4-7."""
    sign = (val < 0.0).to(tl.int32) * 8
    a = tl.abs(val)
    lo = tl.minimum((a * 2.0 + 0.5).to(tl.int32), 3)
    hi = 4 + (a >= 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a >= 5.0).to(tl.int32)
    return tl.where(a >= 1.75, hi, lo) + sign


@triton.jit
def _fp4_decode(code):
    """FP4 E2M1 decode via IEEE754 bit construction (avoids exp2)."""
    c = (code & 0x7).to(tl.int32)
    exp_fp4 = c >> 1
    mant_fp4 = c & 1
    ieee_norm = ((exp_fp4 + 126) << 23) | (mant_fp4 << 22)
    ieee_sub = mant_fp4 * 0x3F000000
    ieee = tl.where(exp_fp4 > 0, ieee_norm, ieee_sub)
    return (ieee | ((code & 8) << 28)).to(tl.float32, bitcast=True)
