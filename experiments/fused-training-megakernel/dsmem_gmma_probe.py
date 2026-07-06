"""Standalone cluster DSMEM + GMMA probe.

This answers one narrow question before touching the megakernel body:

  Can a GMMA shared-memory descriptor consume a B operand from a peer CTA's DSMEM?

The kernel first proves regular cluster DSMEM addressing with `ld.shared::cluster`.
Then rank 0 stages B in shared memory, rank 1 leaves its local B slab zeroed, and both
CTAs run the same m64n128k64 GMMA. If rank 1 matches torch, GMMA accepted the remote
shared address. If rank 1 is near zero while the DSMEM read passed, GMMA descriptors
cannot carry the remote-rank bits and the GEMM path should use TMA multicast instead.
"""

import os
import time

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
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
constexpr int PROBE_DIAG_BYTES = 256;
constexpr int PROBE_SMEM_BYTES =
    PROBE_A_BYTES + PROBE_B_BYTES + PROBE_DIAG_BYTES + 1024;

#define FMA_ARGS d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], \
        d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], \
        d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], \
        d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], \
        d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], \
        d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]

__device__ __forceinline__ int koff(int r, int k) {
  return ((r >> 3) << 8) + ((k >> 3) << 7) + ((r & 7) << 4) + ((k & 7) << 1);
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t make_desc_addr(uint32_t addr) {
  cute::GmmaDescriptor dsc;
  dsc.desc_ = 0;
  dsc.bitfield.start_address_ = (addr >> 4);
  dsc.bitfield.leading_byte_offset_ = (128 >> 4);
  dsc.bitfield.stride_byte_offset_ = (256 >> 4);
  dsc.bitfield.layout_type_ = 0;
  return dsc.desc_;
}

__device__ __forceinline__ int ld_shared_cluster_i32(uint32_t addr) {
  int v;
  asm volatile("ld.shared::cluster.s32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ void fence_smem_async_cluster() {
  asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
}

__global__ void dsmem_gmma_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                  float* __restrict__ C, int64_t* __restrict__ diag) {
  extern __shared__ char smem[];
  char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(smem) + 1023) &
                                       ~uintptr_t(1023));
  bf16* sA = reinterpret_cast<bf16*>(base);
  bf16* sB = reinterpret_cast<bf16*>(base + PROBE_A_BYTES);
  int* s_probe = reinterpret_cast<int*>(base + PROBE_A_BYTES + PROBE_B_BYTES);
  const int tid = threadIdx.x;
  const int rank = static_cast<int>(cute::block_rank_in_cluster());

  if (tid == 0) {
    *s_probe = 1000 + rank;
  }

  for (int i = tid; i < PROBE_M * PROBE_K; i += blockDim.x) {
    const int r = i / PROBE_K;
    const int k = i % PROBE_K;
    reinterpret_cast<bf16*>(reinterpret_cast<char*>(sA) + (k / 16) * 2048 +
                            koff(r, k % 16))[0] =
        A[(rank * PROBE_M + r) * PROBE_K + k];
  }
  for (int i = tid; i < PROBE_N * PROBE_K; i += blockDim.x) {
    const int r = i / PROBE_K;
    const int k = i % PROBE_K;
    reinterpret_cast<bf16*>(reinterpret_cast<char*>(sB) + (k / 16) * 4096 +
                            koff(r, k % 16))[0] = __float2bfloat16(0.0f);
  }
  __syncthreads();

  if (rank == 0) {
    for (int i = tid; i < PROBE_N * PROBE_K; i += blockDim.x) {
      const int r = i / PROBE_K;
      const int k = i % PROBE_K;
      reinterpret_cast<bf16*>(reinterpret_cast<char*>(sB) + (k / 16) * 4096 +
                              koff(r, k % 16))[0] = B[r * PROBE_K + k];
    }
  }
  __syncthreads();
  fence_smem_async_cluster();
  cute::cluster_sync();

  const uint32_t probe_addr = smem_addr(s_probe);
  const uint32_t peer_probe_addr = cute::set_block_rank(probe_addr, 1 - rank);
  const int peer_probe = tid == 0 ? ld_shared_cluster_i32(peer_probe_addr) : 0;
  const uint32_t local_b_addr = smem_addr(sB);
  const uint32_t mapped_b_addr = cute::set_block_rank(local_b_addr, 0);

  if (tid == 0) {
    diag[rank * 8 + 0] = rank;
    diag[rank * 8 + 1] = peer_probe;
    diag[rank * 8 + 2] = static_cast<int64_t>(probe_addr);
    diag[rank * 8 + 3] = static_cast<int64_t>(peer_probe_addr);
    diag[rank * 8 + 4] = static_cast<int64_t>(local_b_addr);
    diag[rank * 8 + 5] = static_cast<int64_t>(mapped_b_addr);
    diag[rank * 8 + 6] = static_cast<int64_t>((local_b_addr >> 4) & 0x3fff);
    diag[rank * 8 + 7] = static_cast<int64_t>((mapped_b_addr >> 4) & 0x3fff);
  }

  float d[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.0f;

  using MMA = SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>;
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint64_t da = make_desc_addr(smem_addr(reinterpret_cast<char*>(sA) + s * 2048));
    const uint64_t db = make_desc_addr(mapped_b_addr + s * 4096);
    MMA::fma(da, db, FMA_ARGS, s == 0 ? SG::ScaleOut::Zero : SG::ScaleOut::One);
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

void run_probe(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor diag) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda() && diag.is_cuda());
  TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
  TORCH_CHECK(C.dtype() == torch::kFloat32 && diag.dtype() == torch::kInt64);
  TORCH_CHECK(A.numel() == 2 * PROBE_M * PROBE_K &&
              B.numel() == PROBE_N * PROBE_K &&
              C.numel() == 2 * PROBE_M * PROBE_N);
  TORCH_CHECK(diag.numel() >= 16);

  C10_CUDA_CHECK(cudaFuncSetAttribute((void*)dsmem_gmma_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      PROBE_SMEM_BYTES));
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
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsmem_gmma_kernel,
                                    reinterpret_cast<const bf16*>(A.data_ptr()),
                                    reinterpret_cast<const bf16*>(B.data_ptr()),
                                    C.data_ptr<float>(), diag.data_ptr<int64_t>()));
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_probe", &run_probe, "cluster DSMEM GMMA B-descriptor probe");
}
"""


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("MK_PROBE_DEVICE", "0")))
    torch.manual_seed(0)
    ext = load_inline(
        name="dsmem_gmma_probe",
        cpp_sources=[""],
        cuda_sources=[cuda_src],
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
            f"  rank={row[0]} peer_probe={row[1]} "
            f"probe_addr=0x{row[2]:x} peer_probe_addr=0x{row[3]:x} "
            f"local_b=0x{row[4]:x} mapped_b_rank0=0x{row[5]:x} "
            f"desc_start_local=0x{row[6]:x} desc_start_mapped=0x{row[7]:x}"
        )
    print(f"rank0_max_err={err[0].item():.6e} rank0_abs={c_abs[0].item():.6e}")
    print(f"rank1_max_err={err[1].item():.6e} rank1_abs={c_abs[1].item():.6e}")
    print(
        "remote_gmma="
        + ("PASS" if err[1].item() < 0.15 else "FAIL")
        + " dsmem_read="
        + ("PASS" if diag_cpu[1] == 1001 and diag_cpu[9] == 1000 else "FAIL")
    )

    # Keep a tiny timing datapoint so later body probes can tell if this path is viable.
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
