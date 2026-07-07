"""D64 mbarrier-ring TMA-feed probe (GEMM round-4 port to the long-D64 ring bodies).

The long-S D64 shapes (H256, S3072/4096/8192) run their wg GEMMs through the
MK_GEMM_MBAR_RING bodies promoted in 31dad00: op_gemm_wgmma (m64n64, 4-stage
ring, all four storage majors) and op_gemm_wgmma_n128 (m64n128, 3-stage ring,
A K-major only). This probe validates the n256-style TMA feed (9146c9c) on
EXACTLY those ring structures before any op-library edit:

  - tensormaps in GLOBAL memory (128B rows), acquired in-kernel with
    fence.proxy.tensormap::generic.acquire.gpu
  - per-CTA elected-thread cp.async.bulk.tensor.2d (no cluster/multicast)
  - bfull mbarrier count 1 + mbarrier.arrive.expect_tx by the issuing thread
  - ring depth, smem layout (SW128 == CU_TENSOR_MAP_SWIZZLE_128B), descriptors
    and epilogue-free fp32 store identical in structure to ops.cuh

Slab -> box map (all four majors are SW128 128B-row slabs, all TMA-expressible):
  m64n64  A !a_t: one {64k,128m} box    A a_t: two {64m,64k} boxes (MN slabs)
  m64n64  B  b_t: one {64k,64n} box     B !b_t: one {64n,64k} box (MN slab)
  m64n128 A: one {64k,128m} box
  m64n128 B  b_t: one {64k,128n} box    B !b_t: two {64n,64k} boxes 8KB apart

variant 0: per-thread cp.async SW128 + noinc arrivals (control == ops.cuh ring)
variant 1: elected-thread TMA feed (candidate)

Traps honored: timeout-guard every run (phase desync spins at 99% SM); util+mem
guards around timing (GPU 6 carries a parked out-of-container tenant).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

constexpr int BM = 128;
constexpr int BK = 64;
constexpr int SMEM_BYTES = 100 * 1024;

template <int BN, int STAGES>
struct RingSmem {
  bf16 A[STAGES][2][4096];   // per stage: two 64-row SW128 slabs (K- or MN-major)
  bf16 B[STAGES][BN * 64];   // per stage: BN x 64k SW128 slab(s)
  uint64_t bfull[STAGES];
  uint64_t bempty[STAGES];
  int claim;
};

__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}
__device__ __forceinline__ int mnoff_sw(int k, int mn8) {
  return k * 128 + ((((mn8 >> 3) ^ (k & 7)) << 4));
}
__device__ __forceinline__ uint64_t desc_ksw(const void* slab, int s) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(slab)) + s * 32;
  cute::GmmaDescriptor dsc;
  dsc.desc_ = 0;
  dsc.bitfield.start_address_ = (addr >> 4);
  dsc.bitfield.leading_byte_offset_ = 0;
  dsc.bitfield.stride_byte_offset_ = (1024 >> 4);
  dsc.bitfield.layout_type_ = 1;
  return dsc.desc_;
}
__device__ __forceinline__ uint64_t desc_mnsw(const void* slab, int s) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(slab)) + s * 2048;
  cute::GmmaDescriptor dsc;
  dsc.desc_ = 0;
  dsc.bitfield.start_address_ = (addr >> 4);
  dsc.bitfield.leading_byte_offset_ = 0;
  dsc.bitfield.stride_byte_offset_ = (1024 >> 4);
  dsc.bitfield.layout_type_ = 1;
  return dsc.desc_;
}
__device__ __forceinline__ uint64_t desc_mnsw128(const void* slab, int s) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(slab)) + s * 2048;
  cute::GmmaDescriptor dsc;
  dsc.desc_ = 0;
  dsc.bitfield.start_address_ = (addr >> 4);
  dsc.bitfield.leading_byte_offset_ = (8192 >> 4);
  dsc.bitfield.stride_byte_offset_ = (1024 >> 4);
  dsc.bitfield.layout_type_ = 1;
  return dsc.desc_;
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{.reg .b64 t; mbarrier.arrive.shared.b64 t, [%0];}" ::"r"(a));
}
__device__ __forceinline__ void mbar_arrive_cpasync(uint64_t* bar) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{.reg .b64 t; mbarrier.arrive.expect_tx.shared::cta.b64 t, [%0], %1;}"
               ::"r"(a), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p;}"
        : "=r"(done)
        : "r"(a), "r"(parity)
        : "memory");
  }
}

__device__ __forceinline__ void tmap_fence_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(map)
               : "memory");
}
__device__ __forceinline__ void tma_load_2d(const void* map, void* dst, int x, int y,
                                            uint64_t* bar) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(d), "l"(map), "r"(x), "r"(y), "r"(b)
      : "memory");
}

template <class MMA>
__device__ __forceinline__ void mma_n64(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                        float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
             d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
             d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
             SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}
template <class MMA>
__device__ __forceinline__ void mma_n128(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                         float (&d)[64]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
             d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
             d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
             d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
             d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49],
             d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59],
             d[60], d[61], d[62], d[63], SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

template <int BN, int STAGES>
__global__ void __launch_bounds__(256) ring_kernel(
    const bf16* __restrict__ A, const bf16* __restrict__ B, float* __restrict__ C, int M,
    int N, int K, int a_t, int b_t, int variant, const uint8_t* __restrict__ tmaps,
    int* __restrict__ claim) {
  extern __shared__ char raw[];
  char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + 1023) &
                                       ~uintptr_t(1023));
  RingSmem<BN, STAGES>& S = *reinterpret_cast<RingSmem<BN, STAGES>*>(smem);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wtid = tid & 127;
  const int m_tiles = M / BM;
  const int n_tiles = N / BN;
  const int tiles = m_tiles * n_tiles;
  const int iters = K / BK;
  constexpr int TX_BYTES = 16384 + BN * 128;
  const void* tmA = tmaps;
  const void* tmB = tmaps + 128;
  if (variant == 1 && tid == 0) {
    tmap_fence_acquire(tmA);
    tmap_fence_acquire(tmB);
  }

  while (true) {
    if (tid == 0) S.claim = atomicAdd(claim, 1);
    __syncthreads();
    const int tile = S.claim;
    __syncthreads();
    if (tile >= tiles) break;
    const int m0 = (tile / n_tiles) * BM;  // m-major, like the ops.cuh bodies
    const int n0 = (tile % n_tiles) * BN;

    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < STAGES; ++s) {
        mbar_init(&S.bfull[s], variant == 1 ? 1 : 256);
        mbar_init(&S.bempty[s], 256);
      }
    }
    __syncthreads();

    auto issue_stage_cp = [&](int k0, int st) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {  // A: 128r x 64k
        const int v = tid + i * 256;
        if (a_t) {
          const int h = v / 512, w_ = v % 512;
          const int k = w_ / 8, m8 = (w_ % 8) * 8;
          __pipeline_memcpy_async(
              reinterpret_cast<char*>(S.A[st][h]) + mnoff_sw(k, m8),
              &A[(int64_t)(k0 + k) * M + m0 + h * 64 + m8], 16);
        } else {
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(
              reinterpret_cast<char*>(S.A[st][r / 64]) + koff_sw(r % 64, k8),
              &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
        }
      }
#pragma unroll
      for (int i = 0; i < BN / 32; ++i) {  // B: BN x 64
        const int v = tid + i * 256;
        if (b_t) {
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + koff_sw(r, k8),
                                  &B[(int64_t)(n0 + r) * K + k0 + k8], 16);
        } else if (BN == 128) {  // [K,N] N-contig: two 64-mn MN slabs, 8KB apart
          const int k = v / 16, n8 = (v % 16) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + (n8 / 64) * 8192 +
                                      mnoff_sw(k, n8 % 64),
                                  &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
        } else {  // [K,N] N-contig: one 64-mn MN slab
          const int k = v / 8, n8 = (v % 8) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + mnoff_sw(k, n8),
                                  &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
        }
      }
      __pipeline_commit();
    };
    auto issue_stage_tma = [&](int k0, int st) {
      if (tid == 0) {
        mbar_expect_tx(&S.bfull[st], TX_BYTES);
        if (a_t) {
          tma_load_2d(tmA, S.A[st][0], m0, k0, &S.bfull[st]);
          tma_load_2d(tmA, S.A[st][1], m0 + 64, k0, &S.bfull[st]);
        } else {
          tma_load_2d(tmA, S.A[st][0], k0, m0, &S.bfull[st]);
        }
        if (b_t) {
          tma_load_2d(tmB, S.B[st], k0, n0, &S.bfull[st]);
        } else if (BN == 128) {
#pragma unroll
          for (int g = 0; g < 2; ++g)
            tma_load_2d(tmB, reinterpret_cast<char*>(S.B[st]) + g * 8192, n0 + g * 64,
                        k0, &S.bfull[st]);
        } else {
          tma_load_2d(tmB, S.B[st], n0, k0, &S.bfull[st]);
        }
      }
    };
    auto issue_stage_mb = [&](int t) {
      const int st = t % STAGES;
      if (variant == 1) {
        issue_stage_tma(t * BK, st);
      } else {
        issue_stage_cp(t * BK, st);
        mbar_arrive_cpasync(&S.bfull[st]);
      }
    };

    float d[BN / 2];
#pragma unroll
    for (int i = 0; i < BN / 2; ++i) d[i] = 0.0f;
    constexpr int LEAD = STAGES - 2;
    for (int p = 0; p < min(LEAD + 1, iters); ++p) issue_stage_mb(p);
    for (int t = 0; t < iters; ++t) {
      const int st = t % STAGES;
      mbar_wait(&S.bfull[st], (t / STAGES) & 1);
      uint64_t da[4], db[4];
      if constexpr (BN == 64) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          da[s] = a_t ? desc_mnsw(S.A[st][wg], s) : desc_ksw(S.A[st][wg], s);
          db[s] = b_t ? desc_ksw(S.B[st], s) : desc_mnsw(S.B[st], s);
        }
        float (&d64)[32] = reinterpret_cast<float (&)[32]>(d);
        if (!a_t && b_t)
          mma_n64<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d64);
        else if (!a_t && !b_t)
          mma_n64<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d64);
        else if (a_t && b_t)
          mma_n64<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>>(da, db, d64);
        else
          mma_n64<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(da, db, d64);
      } else {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          da[s] = desc_ksw(S.A[st][wg], s);
          db[s] = b_t ? desc_ksw(S.B[st], s) : desc_mnsw128(S.B[st], s);
        }
        float (&d128)[64] = reinterpret_cast<float (&)[64]>(d);
        if (b_t)
          mma_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d128);
        else
          mma_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d128);
      }
      mbar_arrive(&S.bempty[st]);
      const int tn = t + LEAD + 1;
      if (tn < iters) {
        if (tn >= STAGES) mbar_wait(&S.bempty[tn % STAGES], (tn / STAGES - 1) & 1);
        issue_stage_mb(tn);
      }
    }
    cute::warpgroup_wait<0>();
    __syncthreads();

    const int w = wtid / 32, l = wtid % 32;
    const int cb = (l & 3) * 2;
#pragma unroll
    for (int n8 = 0; n8 < BN / 8; ++n8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + n8 * 8 + cb;
        float2 out = make_float2(d[n8 * 4 + i * 2 + 0], d[n8 * 4 + i * 2 + 1]);
        *reinterpret_cast<float2*>(&C[idx]) = out;
      }
    }
    __syncthreads();
  }
}

static void encode_2d(uint8_t* out, const void* ptr, int64_t inner, int64_t outer,
                      int64_t stride_bytes, int box_inner, int box_outer) {
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t gstride[1] = {(cuuint64_t)stride_bytes};
  cuuint32_t bdim[2] = {(cuuint32_t)box_inner, (cuuint32_t)box_outer};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
      bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  memcpy(out, &map, 128);
}

torch::Tensor encode_tmaps(torch::Tensor A, torch::Tensor B, int64_t M, int64_t N,
                           int64_t K, int64_t a_t, int64_t b_t, int64_t bn) {
  auto out = torch::empty({2, 128}, torch::dtype(torch::kUInt8));
  uint8_t* p = out.data_ptr<uint8_t>();
  if (a_t) {  // A[K,M] M-contig: MN-major slabs, box {64 m, 64 k}
    encode_2d(p, A.data_ptr(), M, K, M * 2, 64, 64);
  } else {  // A[M,K] K-contig: K-major slab, box {64 k, 128 m}
    encode_2d(p, A.data_ptr(), K, M, K * 2, 64, 128);
  }
  if (b_t) {  // B[N,K] K-contig: K-major slab, box {64 k, BN n}
    encode_2d(p + 128, B.data_ptr(), K, N, K * 2, 64, (int)bn);
  } else {  // B[K,N] N-contig: MN-major slab(s), box {64 n, 64 k}
    encode_2d(p + 128, B.data_ptr(), N, K, N * 2, 64, 64);
  }
  return out;
}

void run_ring(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor tmaps,
              torch::Tensor claim, int64_t M, int64_t N, int64_t K, int64_t a_t,
              int64_t b_t, int64_t bn, int64_t variant, int64_t nblocks) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda() && tmaps.is_cuda());
  TORCH_CHECK(M % BM == 0 && N % bn == 0 && K % BK == 0);
  TORCH_CHECK(bn == 64 || bn == 128);
  TORCH_CHECK(!(a_t && bn == 128), "n128 body has no a_t path");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tmaps.data_ptr()) % 128 == 0);
  static int configured = 0;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)ring_kernel<64, 4>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        SMEM_BYTES));
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)ring_kernel<128, 3>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        SMEM_BYTES));
    configured = 1;
  }
  claim.zero_();
  if (bn == 64)
    ring_kernel<64, 4><<<(int)nblocks, 256, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(B.data_ptr()), C.data_ptr<float>(), (int)M, (int)N,
        (int)K, (int)a_t, (int)b_t, (int)variant, tmaps.data_ptr<uint8_t>(),
        claim.data_ptr<int>());
  else
    ring_kernel<128, 3><<<(int)nblocks, 256, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(B.data_ptr()), C.data_ptr<float>(), (int)M, (int)N,
        (int)K, (int)a_t, (int)b_t, (int)variant, tmaps.data_ptr<uint8_t>(),
        claim.data_ptr<int>());
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_ring", &run_ring, "D64 ring cp.async vs global-tensormap TMA feed");
  m.def("encode_tmaps", &encode_tmaps, "encode A/B CUtensorMaps into a 2x128 u8 tensor");
}
"""


def build():
    return load_inline(
        name="d64_tma_ring_probe",
        cpp_sources=[""],
        cuda_sources=[cuda_src],
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-I{CUTE_INC}",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def gpu_util() -> int:
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
         "-i", dev],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def make_inputs(M, N, K, a_t, b_t, seed):
    torch.manual_seed(seed)
    a = (torch.randn(K, M, device="cuda", dtype=torch.bfloat16) if a_t
         else torch.randn(M, K, device="cuda", dtype=torch.bfloat16)) * 0.05
    b = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) if b_t
         else torch.randn(K, N, device="cuda", dtype=torch.bfloat16)) * 0.05
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    return a, b, c


def _name(a_t, b_t):
    return {(0, 1): "NT", (0, 0): "NN", (1, 0): "TN", (1, 1): "TT"}[(a_t, b_t)]


def check(ext, M, N, K, a_t, b_t, bn, nblocks, claim):
    a, b, c = make_inputs(M, N, K, a_t, b_t, seed=11 + a_t * 2 + b_t)
    tmaps = ext.encode_tmaps(a, b, M, N, K, a_t, b_t, bn).cuda()
    rows = torch.randperm(M)[:128].sort().values
    am = (a.t() if a_t else a)[rows].float()
    bm = b.t().float() if b_t else b.float()
    ref = am @ bm
    outs = {}
    for variant, name in ((0, "cpasync-ring"), (1, "tma-ring")):
        c.fill_(float("nan"))
        ext.run_ring(a, b, c, tmaps, claim, M, N, K, a_t, b_t, bn, variant, nblocks)
        torch.cuda.synchronize()
        got = c[rows.cuda()]
        outs[variant] = c.clone()
        diff = (got - ref).abs()
        rel = (diff / ref.abs().clamp_min(0.25)).max().item()
        print(f"check {_name(a_t, b_t)} bn={bn} M={M} N={N} K={K} {name}: "
              f"max_abs={diff.max().item():.6e} rel={rel:.6e}", flush=True)
        assert rel < 3e-2, name
    bit = torch.equal(outs[0], outs[1])
    print(f"check {_name(a_t, b_t)} bn={bn}: tma vs cpasync bit-identical: {bit}",
          flush=True)
    assert bit, "TMA feed must be bit-identical to the cp.async ring"


def bench(ext, M, N, K, a_t, b_t, bn, nblocks, claim, reps, iters, order, label=""):
    a, b, c = make_inputs(M, N, K, a_t, b_t, seed=101 + a_t * 2 + b_t)
    tmaps = ext.encode_tmaps(a, b, M, N, K, a_t, b_t, bn).cuda()
    results = {}
    variants = [(0, "cpasync-ring"), (1, "tma-ring")]
    if order == "rev":
        variants = variants[::-1]
    for variant, name in variants:
        for _ in range(2):
            ext.run_ring(a, b, c, tmaps, claim, M, N, K, a_t, b_t, bn, variant, nblocks)
        torch.cuda.synchronize()
        vals = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                ext.run_ring(a, b, c, tmaps, claim, M, N, K, a_t, b_t, bn, variant,
                             nblocks)
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) * 1e3 / reps)
        vals.sort()
        results[name] = statistics.median(vals)
        tf = (2.0 * M * N * K) / (results[name] * 1e-6) / 1e12
        print(f"  {name:12s} med={results[name]:9.3f}us min={vals[0]:9.3f} "
              f"max={vals[-1]:9.3f} tf={tf:7.1f}", flush=True)
    delta = results["tma-ring"] - results["cpasync-ring"]
    pct = 100.0 * delta / results["cpasync-ring"]
    print(f"bench {label} {_name(a_t, b_t)} bn={bn} M={M} N={N} K={K} order={order}: "
          f"tma-minus-cpasync {delta:+9.3f}us ({pct:+.2f}%)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    u0 = gpu_util()
    ext = build()
    print(f"built d64_tma_ring_probe (pre-util {u0}%)", flush=True)
    dev = torch.cuda.current_device()
    nblocks = torch.cuda.get_device_properties(dev).multi_processor_count
    claim = torch.zeros(1, device="cuda", dtype=torch.int32)
    # parity: K=512 wraps ring parity for both stage counts; all body/major combos
    check(ext, 256, 512, 512, 0, 1, 64, nblocks, claim)   # m64n64 NT
    check(ext, 256, 512, 512, 0, 0, 64, nblocks, claim)   # m64n64 NN (MN-major B)
    check(ext, 256, 512, 512, 1, 0, 64, nblocks, claim)   # m64n64 TN (MN-major A)
    check(ext, 256, 512, 512, 0, 1, 128, nblocks, claim)  # m64n128 NT
    check(ext, 256, 512, 512, 0, 0, 128, nblocks, claim)  # m64n128 NN
    if mode == "long":
        order = sys.argv[2] if len(sys.argv) > 2 else "fwd"
        for S in (4096, 8192):
            r = 8192 // S * 8
            # n128 NT: gate_up fwd S x 1536 x 256
            bench(ext, S, 1536, 256, 0, 1, 128, nblocks, claim, r, 12, order, f"s{S}-gufwd")
            # n128 NN: mlp dX S x 768 x 256
            bench(ext, S, 768, 256, 0, 0, 128, nblocks, claim, r, 12, order, f"s{S}-dxmlp")
            # m64n64 NT: qkv(+qkrope) fwd S x 512 x 256
            bench(ext, S, 512, 256, 0, 1, 64, nblocks, claim, r, 12, order, f"s{S}-qkvfwd")
            # m64n64 NN: drow-class S x 256 x 256
            bench(ext, S, 256, 256, 0, 0, 64, nblocks, claim, r, 12, order, f"s{S}-drow")
            # m64n64 TN dW: gate_up dW 1536 x 256 x S (long-K sink class)
            bench(ext, 1536, 256, S, 1, 0, 64, nblocks, claim, r, 12, order, f"s{S}-dwgu")
            # m64n64 TN dW small-tile: wo dW 256 x 256 x S
            bench(ext, 256, 256, S, 1, 0, 64, nblocks, claim, r, 12, order, f"s{S}-dwwo")
    u1 = gpu_util()
    print(f"post-util {u1}% (pre {u0}%)", flush=True)
