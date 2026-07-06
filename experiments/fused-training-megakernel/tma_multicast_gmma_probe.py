"""Standalone TMA multicast + GMMA probe.

This is the positive counterpart to `dsmem_gmma_probe.py`: instead of asking GMMA to
read a peer CTA's DSMEM, rank 0 issues one TMA tensor multicast that writes the same
B tile into both CTAs' local shared-memory slabs. Both CTAs then run ordinary local
GMMA descriptors. A rank-1 pass proves the intended cluster path for paired M tiles:
multicast B into each CTA, keep GMMA local.
"""

import os
import time

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

constexpr int PROBE_K = 64;
constexpr int PROBE_M = 64;
constexpr int PROBE_N = 128;
constexpr int PROBE_A_BYTES = 8192;
constexpr int PROBE_B_BYTES = 16384;
constexpr int PROBE_SMEM_BYTES = PROBE_A_BYTES + PROBE_B_BYTES + 1024;

#define FMA_ARGS d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], \
        d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], \
        d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], \
        d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], \
        d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], \
        d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]

__device__ __forceinline__ long long globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}

__device__ __forceinline__ uint64_t desc_k_sw(const void* slab, int s) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(slab)) + s * 32;
  cute::GmmaDescriptor dsc;
  dsc.desc_ = 0;
  dsc.bitfield.start_address_ = (addr >> 4);
  dsc.bitfield.leading_byte_offset_ = 0;
  dsc.bitfield.stride_byte_offset_ = (1024 >> 4);
  dsc.bitfield.layout_type_ = 1;
  return dsc.desc_;
}

__device__ __forceinline__ void fence_smem_async_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
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

__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done;
  asm volatile(
      "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
      "selp.u32 %0, 1, 0, p;}"
      : "=r"(done)
      : "r"(a), "r"(parity)
      : "memory");
  return done != 0;
}

__device__ __forceinline__ bool mbar_wait_timeout(uint64_t* bar, uint32_t parity,
                                                  long long timeout_ns) {
  const long long start = globaltimer();
  while (!mbar_try_wait(bar, parity)) {
    if (globaltimer() - start > timeout_ns) return false;
  }
  return true;
}

__device__ __forceinline__ void tma_load_2d_multicast(const CUtensorMap* map, void* dst,
                                                      int c0, int r0, uint64_t* bar,
                                                      uint16_t mask) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes."
      "multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(d), "l"(map), "r"(b), "r"(c0), "r"(r0), "h"(mask)
      : "memory");
}

__global__ void tma_multicast_gmma_kernel(const bf16* __restrict__ A,
                                          float* __restrict__ C,
                                          int64_t* __restrict__ diag,
                                          const __grid_constant__ CUtensorMap tmB) {
  extern __shared__ char smem[];
  char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(smem) + 1023) &
                                       ~uintptr_t(1023));
  bf16* sA = reinterpret_cast<bf16*>(base);
  bf16* sB = reinterpret_cast<bf16*>(base + PROBE_A_BYTES);
  uint64_t* bar = reinterpret_cast<uint64_t*>(base + PROBE_A_BYTES + PROBE_B_BYTES);
  __shared__ int s_wait_ok;

  const int tid = threadIdx.x;
  const int rank = static_cast<int>(cute::block_rank_in_cluster());

  if (tid == 0) {
    mbar_init(bar, 1);
    s_wait_ok = 0;
  }
  __syncthreads();
  fence_mbarrier_init_cluster();

  for (int v = tid; v < (PROBE_M * PROBE_K) / 8; v += blockDim.x) {
    const int r = v / 8;
    const int k8 = (v % 8) * 8;
    uint4 x = *reinterpret_cast<const uint4*>(&A[(rank * PROBE_M + r) * PROBE_K + k8]);
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(sA) + koff_sw(r, k8)) = x;
  }
  for (int v = tid; v < (PROBE_N * PROBE_K) / 8; v += blockDim.x) {
    const int r = v / 8;
    const int k8 = (v % 8) * 8;
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(sB) + koff_sw(r, k8)) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  fence_smem_async_cta();

  if (tid == 0) mbar_expect_tx(bar, PROBE_B_BYTES);
  __syncthreads();
  cute::cluster_sync();
  if (rank == 0 && tid == 0) tma_load_2d_multicast(&tmB, sB, 0, 0, bar, 0x3);

  if (tid == 0) {
    const bool ok = mbar_wait_timeout(bar, 0, 50000000LL);
    s_wait_ok = ok ? 1 : 0;
    const uint32_t baddr = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
    const uint32_t baraddr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    diag[rank * 8 + 0] = rank;
    diag[rank * 8 + 1] = s_wait_ok;
    diag[rank * 8 + 2] = static_cast<int64_t>(baddr);
    diag[rank * 8 + 3] = static_cast<int64_t>(baraddr);
    diag[rank * 8 + 4] = static_cast<int64_t>((baddr >> 4) & 0x3fff);
    diag[rank * 8 + 5] = static_cast<int64_t>(*reinterpret_cast<unsigned short*>(sB));
    diag[rank * 8 + 6] = 0;
    diag[rank * 8 + 7] = 0;
  }
  __syncthreads();
  if (!s_wait_ok) return;

  float d[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.0f;

  using MMA = SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>;
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    MMA::fma(desc_k_sw(sA, s), desc_k_sw(sB, s), FMA_ARGS,
             s == 0 ? SG::ScaleOut::Zero : SG::ScaleOut::One);
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();

  const int w = tid / 32;
  const int l = tid % 32;
  for (int n8 = 0; n8 < 16; ++n8) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int row = w * 16 + l / 4 + 8 * i;
        const int col = n8 * 8 + (l % 4) * 2 + j;
        C[(rank * PROBE_M + row) * PROBE_N + col] = d[n8 * 4 + i * 2 + j];
      }
    }
  }
}

static CUtensorMap make_tmap_b(const void* ptr) {
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)PROBE_K, (cuuint64_t)PROBE_N};
  cuuint64_t gstride[1] = {(cuuint64_t)PROBE_K * 2};
  cuuint32_t bdim[2] = {(cuuint32_t)PROBE_K, (cuuint32_t)PROBE_N};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
      bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return map;
}

void run_probe(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor diag) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda() && diag.is_cuda());
  TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
  TORCH_CHECK(C.dtype() == torch::kFloat32 && diag.dtype() == torch::kInt64);
  TORCH_CHECK(A.numel() == 2 * PROBE_M * PROBE_K);
  TORCH_CHECK(B.numel() == PROBE_N * PROBE_K);
  TORCH_CHECK(C.numel() == 2 * PROBE_M * PROBE_N);
  TORCH_CHECK(diag.numel() >= 16);

  C10_CUDA_CHECK(cudaFuncSetAttribute((void*)tma_multicast_gmma_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      PROBE_SMEM_BYTES));
  CUtensorMap tmB = make_tmap_b(B.data_ptr());
  cudaLaunchConfig_t cfg = {0};
  cfg.gridDim = dim3(2, 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = PROBE_SMEM_BYTES;
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
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tma_multicast_gmma_kernel,
                                    reinterpret_cast<const bf16*>(A.data_ptr()),
                                    C.data_ptr<float>(), diag.data_ptr<int64_t>(), tmB));
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_probe", &run_probe, "cluster TMA multicast GMMA probe");
}
"""


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("MK_PROBE_DEVICE", "0")))
    torch.manual_seed(1)
    ext = load_inline(
        name="tma_multicast_gmma_probe",
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

    A = torch.randn(2, 64, 64, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    C = torch.full((2, 64, 128), float("nan"), device="cuda", dtype=torch.float32)
    diag = torch.full((16,), -1, device="cuda", dtype=torch.int64)

    ext.run_probe(A, B, C, diag)
    torch.cuda.synchronize()
    ref = A.float() @ B.float().T
    err = (C - ref).abs().amax(dim=(1, 2)).detach().cpu()
    c_abs = C.abs().amax(dim=(1, 2)).detach().cpu()
    diag_cpu = diag.detach().cpu().tolist()

    print("diag rows:")
    for rank in range(2):
        row = diag_cpu[rank * 8 : (rank + 1) * 8]
        print(
            f"  rank={row[0]} wait_ok={row[1]} b_addr=0x{row[2]:x} "
            f"bar_addr=0x{row[3]:x} desc_start=0x{row[4]:x} first_b_bits=0x{row[5]:x}"
        )
    print(f"rank0_max_err={err[0].item():.6e} rank0_abs={c_abs[0].item():.6e}")
    print(f"rank1_max_err={err[1].item():.6e} rank1_abs={c_abs[1].item():.6e}")
    print(
        "tma_multicast_gmma="
        + ("PASS" if err.max().item() < 0.15 and diag_cpu[1] == 1 and diag_cpu[9] == 1 else "FAIL")
    )

    for _ in range(5):
        ext.run_probe(A, B, C, diag)
    torch.cuda.synchronize()
    t0 = time.time()
    reps = 2000
    for _ in range(reps):
        ext.run_probe(A, B, C, diag)
    torch.cuda.synchronize()
    print(f"launch_plus_probe_us={(time.time() - t0) * 1e6 / reps:.2f}")


if __name__ == "__main__":
    main()
