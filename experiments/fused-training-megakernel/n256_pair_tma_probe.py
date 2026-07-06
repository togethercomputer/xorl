"""Production-shaped n256 paired-M cluster probe.

The minimal TMA multicast probe proved the primitive. This probe asks the next
question before interpreter integration: for the current qwen-style NT n256 body,
does pairing adjacent M tiles in a 2-CTA cluster and multicasting B into both CTAs'
local smem beat the current duplicated per-CTA B load?

Each cluster computes two adjacent 128x256 output tiles for one shared B tile:

  variant 0: both CTAs load A and B with the existing cp.async SW128 path
  variant 1: both CTAs load A locally; rank0 issues TMA multicast for B, with a
             conservative cluster sync before every TMA issue
  variant 2: same, but assumes lockstep paired CTAs after local mbarrier arming
             and skips the per-stage cluster sync

Run with a full grid of clusters to measure a real saturated body, but keep this
standalone until scheduler pairing and tensor-map ABI are ready in the interpreter.
"""

from __future__ import annotations

import os
import statistics
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

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int STAGES = 3;
constexpr int A_STAGE_BYTES = 16 * 1024;
constexpr int B_STAGE_BYTES = 32 * 1024;
constexpr int SMEM_BYTES = STAGES * (A_STAGE_BYTES + B_STAGE_BYTES) + 1024;

#define D4(i) d[(i) + 0], d[(i) + 1], d[(i) + 2], d[(i) + 3]
#define D16(i) D4(i), D4((i) + 4), D4((i) + 8), D4((i) + 12)
#define D64 D16(0), D16(16), D16(32), D16(48)
#define D128 D64, D16(64), D16(80), D16(96), D16(112)

struct Stage {
  bf16 A[2][4096];  // two 64x64 SW128 A slabs, one per warpgroup
  bf16 B[16384];    // one 256x64 SW128 B slab
};

struct PairSmem {
  Stage stage[STAGES];
  uint64_t bbar[STAGES];
};

__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
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

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{.reg .b64 t; mbarrier.arrive.expect_tx.shared::cta.b64 t, [%0], %1;}"
               ::"r"(a), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ bool mbar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done;
  do {
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p;}"
        : "=r"(done)
        : "r"(a), "r"(parity)
        : "memory");
  } while (!done);
  return true;
}

__device__ __forceinline__ void fence_mbarrier_init_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void tma_load_b_multicast(const CUtensorMap* map, void* dst,
                                                     int k0, uint64_t* bar) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes."
      "multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(d), "l"(map), "r"(b), "r"(k0), "r"(0), "h"(uint16_t(0x3))
      : "memory");
}

template <class MMA>
__device__ __forceinline__ void mma_n256(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                         float (&d)[128]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) MMA::fma(da[s], db[s], D128, SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

__global__ void n256_pair_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                 float* __restrict__ C, int K, int variant, int reps,
                                 const __grid_constant__ CUtensorMap tmB) {
  extern __shared__ char raw[];
  char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + 1023) &
                                       ~uintptr_t(1023));
  PairSmem& S = *reinterpret_cast<PairSmem*>(smem);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wtid = tid & 127;
  const int rank = static_cast<int>(cute::block_rank_in_cluster());
  const int cluster = static_cast<int>(cute::cluster_id_in_grid().x);
  const int m0 = (cluster * 2 + rank) * BM;
  const int iters = K / BK;

  for (int rep = 0; rep < reps; ++rep) {
  if (variant != 0 && tid == 0) {
#pragma unroll
      for (int st = 0; st < STAGES; ++st) mbar_init(&S.bbar[st], 1);
    }
    __syncthreads();
    if (variant != 0) {
      fence_mbarrier_init_cluster();
      cute::cluster_sync();
    }

    float d[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) d[i] = 0.0f;

    auto issue = [&](int kt) {
      const int k0 = kt * BK;
      const int st = kt % STAGES;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int v = tid + i * 256;
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(S.stage[st].A[r / 64]) + koff_sw(r % 64, k8),
            &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
      }
      if (variant == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int v = tid + i * 256;
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.stage[st].B) + koff_sw(r, k8),
                                  &B[(int64_t)r * K + k0 + k8], 16);
        }
      } else {
        if (tid == 0) mbar_expect_tx(&S.bbar[st], B_STAGE_BYTES);
      }
      __pipeline_commit();
      if (variant != 0) {
        __syncthreads();
        if (variant == 1) cute::cluster_sync();
        if (rank == 0 && tid == 0) tma_load_b_multicast(&tmB, S.stage[st].B, k0, &S.bbar[st]);
      }
    };

#pragma unroll
    for (int p = 0; p < STAGES - 1; ++p)
      if (p < iters) issue(p);
    for (int kt = 0; kt < iters; ++kt) {
      if (kt + STAGES - 1 < iters) issue(kt + STAGES - 1);
      __pipeline_wait_prior(min(STAGES - 1, iters - kt - 1));
      __syncthreads();
      if (variant != 0) {
        if (tid == 0) mbar_wait(&S.bbar[kt % STAGES], (kt / STAGES) & 1);
        __syncthreads();
      }
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = desc_ksw(S.stage[kt % STAGES].A[wg], s);
        db[s] = desc_ksw(S.stage[kt % STAGES].B, s);
      }
      mma_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
      __syncthreads();
    }

    const int w = wtid >> 5, l = wtid & 31;
    const int cb = (l & 3) * 2;
#pragma unroll
    for (int n8 = 0; n8 < 32; ++n8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        float2 out = make_float2(d[n8 * 4 + i * 2 + 0], d[n8 * 4 + i * 2 + 1]);
        *reinterpret_cast<float2*>(&C[(int64_t)(m0 + r) * BN + n8 * 8 + cb]) = out;
      }
    }
    __syncthreads();
  }
}

static CUtensorMap make_tmap_b(const void* ptr, int K) {
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)BN};
  cuuint64_t gstride[1] = {(cuuint64_t)K * 2};
  cuuint32_t bdim[2] = {(cuuint32_t)BK, (cuuint32_t)BN};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
      bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return map;
}

void run_pair(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t K,
              int64_t variant, int64_t reps, int64_t nclusters) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda());
  TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
  TORCH_CHECK(C.dtype() == torch::kFloat32);
  TORCH_CHECK(K % BK == 0);
  TORCH_CHECK(A.numel() == nclusters * 2 * BM * K);
  TORCH_CHECK(B.numel() == BN * K);
  TORCH_CHECK(C.numel() == nclusters * 2 * BM * BN);
  TORCH_CHECK(variant == 0 || variant == 1 || variant == 2);
  C10_CUDA_CHECK(cudaFuncSetAttribute((void*)n256_pair_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      SMEM_BYTES));
  CUtensorMap tmB = make_tmap_b(B.data_ptr(), (int)K);
  cudaLaunchConfig_t cfg = {0};
  cfg.gridDim = dim3((unsigned)nclusters * 2, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeCooperative;
  attrs[1].val.cooperative = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, n256_pair_kernel,
                                    reinterpret_cast<const bf16*>(A.data_ptr()),
                                    reinterpret_cast<const bf16*>(B.data_ptr()),
                                    C.data_ptr<float>(), (int)K, (int)variant,
                                    (int)reps, tmB));
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_pair", &run_pair, "n256 paired-M cp.async vs TMA multicast probe");
}
"""


def build():
    return load_inline(
        name="n256_pair_tma_probe",
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


def run_once(ext, variant: int, nclusters: int, k: int, reps: int, iters: int):
    m = nclusters * 2 * 128
    torch.manual_seed(100 + variant + k + nclusters)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(m, 256, device="cuda", dtype=torch.float32)
    for _ in range(3):
        ext.run_pair(a, b, c, k, variant, reps, nclusters)
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        ext.run_pair(a, b, c, k, variant, reps, nclusters)
        e.record()
        torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) * 1e3 / reps)
    vals.sort()
    return statistics.median(vals), vals[0], vals[-1], a, b, c


def check_correct(ext):
    nclusters, k = 2, 256
    m = nclusters * 2 * 128
    torch.manual_seed(7)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ b.float().T
    for variant, name in ((0, "cpasync"), (1, "tma-sync"), (2, "tma-nosync")):
        c = torch.empty(m, 256, device="cuda", dtype=torch.float32)
        ext.run_pair(a, b, c, k, variant, 1, nclusters)
        torch.cuda.synchronize()
        diff = (c - ref).abs()
        rel = (diff / ref.abs().clamp_min(0.5)).max().item()
        print(
            f"check {name}: max_abs={diff.max().item():.6e} rel={rel:.6e}",
            flush=True,
        )
        assert rel < 3e-2, name


def bench(ext, nclusters: int, k: int, reps: int, iters: int):
    rows = []
    for variant, name in ((0, "cpasync"), (1, "tma-sync"), (2, "tma-nosync")):
        med, lo, hi, *_ = run_once(ext, variant, nclusters, k, reps, iters)
        m = nclusters * 2 * 128
        tflops = (2.0 * m * 256 * k) / (med * 1e-6) / 1e12
        rows.append((name, med, lo, hi, tflops))
    base = rows[0][1]
    print(f"bench nclusters={nclusters} M={nclusters * 256} N=256 K={k} reps={reps}", flush=True)
    for name, med, lo, hi, tflops in rows:
        print(
            f"  {name:9s} med={med:8.3f}us min={lo:8.3f} max={hi:8.3f} "
            f"tf={tflops:7.1f} delta_vs_cpasync={med - base:+8.3f}us",
            flush=True,
        )


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("MK_PROBE_DEVICE", "0")))
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    ext = build()
    print("built n256_pair_tma_probe", flush=True)
    check_correct(ext)
    if mode == "qwen":
        bench(ext, nclusters=66, k=2560, reps=20, iters=12)
    else:
        bench(ext, nclusters=16, k=512, reps=20, iters=8)
