"""NF4 (NormalFloat4) codec: encode/decode helpers for NF4 quantization.

NF4 is an information-theoretically optimal 4-bit data type for normally
distributed weights (Dettmers et al., 2023). The 16 quantization levels
are the quantiles of the standard normal distribution that divide the
probability mass into 16 equal segments.

Encoding: comparison chain against midpoints (15 comparisons, branchless).
Decoding: LUT gather (pass pointer, tl.load with code as offset, L1-cached).
"""

import torch
import triton
import triton.language as tl


# NF4 quantization levels (sorted, normalized to [-1, 1])
NF4_TABLE = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

# Minimum gap between adjacent NF4 levels (between codes 7 and 8: 0.07958)
NF4_MIN_STEP = 0.07958029955625534


@triton.jit
def _nf4_encode(val):
    """Encode normalized [-1,1] value to NF4 4-bit code via comparison chain.

    Sums (val > midpoint_i) for 15 midpoints between adjacent NF4 levels.
    Branchless and SIMD-friendly; works on tensors of any shape.
    """
    code = (val > -0.8480964004993439).to(tl.int32)
    code += (val > -0.6106329262256622).to(tl.int32)
    code += (val > -0.4599952697753906).to(tl.int32)
    code += (val > -0.3396794348955154).to(tl.int32)
    code += (val > -0.2346074059605598).to(tl.int32)
    code += (val > -0.1379117332398891).to(tl.int32)
    code += (val > -0.0455250181257725).to(tl.int32)
    code += (val > 0.0397901497781277).to(tl.int32)
    code += (val > 0.1202552504837513).to(tl.int32)
    code += (val > 0.2035212516784668).to(tl.int32)
    code += (val > 0.2920137718319893).to(tl.int32)
    code += (val > 0.3893125355243683).to(tl.int32)
    code += (val > 0.5016634166240692).to(tl.int32)
    code += (val > 0.6427869200706482).to(tl.int32)
    code += (val > 0.8614784181118012).to(tl.int32)
    return code


@triton.jit
def _nf4_decode(code, LUT):
    """Decode NF4 4-bit code to float via LUT gather.

    LUT is a pointer to 16 float32 values (64 bytes, fits in L1 cache line).
    Vectorized gather: tl.load(LUT + code_tensor) works for any tensor shape.
    """
    return tl.load(LUT + (code & 0xF))


# Cache LUT tensors per device to avoid repeated allocation
_NF4_LUT_CACHE: dict = {}


def get_nf4_lut(device) -> torch.Tensor:
    """Get (or create) the NF4 decode LUT on the given device."""
    key = str(device)
    if key not in _NF4_LUT_CACHE:
        _NF4_LUT_CACHE[key] = torch.tensor(NF4_TABLE, dtype=torch.float32, device=device)
    return _NF4_LUT_CACHE[key]
