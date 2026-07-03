"""Standalone wgmma probe: validate smem descriptors + accumulator mapping vs torch.

One warpgroup computes C[64,128] = A[64,K] @ B[128,K]^T (both K-major, the Linear-fwd
layout) with m64n128k16 wgmma over K/16 steps. Run before porting into the megakernel.
"""

import time

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <cuda_bf16.h>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

// our K-major INTER arrangement: 8x8-bf16 core matrices; within a k16 step block:
//   offset_bytes(r, k) = (r/8)*256 + (k/8)*128 + (r%8)*16 + (k%8)*2
// (SBO = 256B between 8-row groups, LBO = 128B between the two k8 columns)
// step blocks stored contiguously: A step stride = 64*16*2 = 2048B, B = 128*16*2 = 4096B.
__device__ __forceinline__ int koff(int r, int k) {
  return ((r >> 3) << 8) + ((k >> 3) << 7) + ((r & 7) << 4) + ((k & 7) << 1);
}

__device__ __forceinline__ uint64_t make_desc(const void* smem_ptr) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (128 >> 4);
  d.bitfield.stride_byte_offset_ = (256 >> 4);
  d.bitfield.layout_type_ = 0;  // INTERLEAVED (no swizzle)
  return d.desc_;
}

// MN-major INTER arrangement (for operands stored with the MN dim contiguous):
// core matrices 8(mn) x 8(k), COLUMN-major within (mn fastest):
//   offset_bytes(mn, k) = (mn/8)*128 + (k/8)*SBO + (mn%8)*2 + (k%8)*16
// LBO = 128 (mn-group stride), SBO = mn_rows/8 * 128 (k-group stride).
#define FMA_ARGS d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], \
        d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], \
        d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], \
        d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], \
        d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], \
        d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]

__device__ __forceinline__ int mnoff(int mn, int k, int sbo) {
  return ((mn >> 3) << 7) + ((k >> 3) * sbo) + ((mn & 7) << 1) + ((k & 7) << 4);
}

// canonical MN-major INTER (mma_traits_sm90_gmma.hpp): ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
// -> SBO = mn-group (8-row) stride = 128B, LBO = k-group (8-col) stride (block-height dep.)
__device__ __forceinline__ uint64_t make_desc_mn(const void* smem_ptr, int kgo) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (kgo >> 4);
  d.bitfield.stride_byte_offset_ = (128 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

// NN: C[64,128] = A[64,K] (K-major) @ B[K,128] (row-major = MN-contiguous)
__global__ void probe_nn(const bf16* __restrict__ A, const bf16* __restrict__ B,
                         float* __restrict__ C, int K) {
  extern __shared__ char smem[];
  bf16* sA = reinterpret_cast<bf16*>(smem);                    // K/16 blocks of 2KB
  bf16* sB = reinterpret_cast<bf16*>(smem + (K / 16) * 2048);  // K/16 blocks of 4KB
  const int tid = threadIdx.x;
  for (int i = tid; i < 64 * K; i += blockDim.x) {
    const int r = i / K, k = i % K;
    reinterpret_cast<bf16*>(reinterpret_cast<char*>(sA) + (k / 16) * 2048 +
                            koff(r, k % 16))[0] = A[r * K + k];
  }
  for (int i = tid; i < 128 * K; i += blockDim.x) {
    const int k = i / 128, n = i % 128;  // B row-major [K, 128]
    reinterpret_cast<bf16*>(reinterpret_cast<char*>(sB) + (k / 16) * 4096 +
                            mnoff(n, k % 16, 2048))[0] = B[k * 128 + n];
  }
  __syncthreads();
  float d[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.0f;
  using MMA_NN = SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>;
  cute::warpgroup_arrive();
  for (int s = 0; s < K / 16; ++s) {
    MMA_NN::fma(make_desc(reinterpret_cast<char*>(sA) + s * 2048),
                make_desc_mn(reinterpret_cast<char*>(sB) + s * 4096, 2048), FMA_ARGS,
                SG::ScaleOut::One);
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
  const int w = tid / 32, l = tid % 32;
  for (int n8 = 0; n8 < 16; ++n8)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        C[(w * 16 + l / 4 + 8 * i) * 128 + n8 * 8 + (l % 4) * 2 + j] = d[n8 * 4 + i * 2 + j];
}

__global__ void probe(const bf16* __restrict__ A, const bf16* __restrict__ B,
                      float* __restrict__ C, int K, int REPS) {
  extern __shared__ char smem[];
  bf16* sA = reinterpret_cast<bf16*>(smem);                    // K/16 blocks of 2KB
  bf16* sB = reinterpret_cast<bf16*>(smem + (K / 16) * 2048);  // K/16 blocks of 4KB
  const int tid = threadIdx.x;

  for (int i = tid; i < 64 * K; i += blockDim.x) {
    const int r = i / K, k = i % K;
    reinterpret_cast<bf16*>(reinterpret_cast<char*>(sA) + (k / 16) * 2048 +
                            koff(r, k % 16))[0] = A[r * K + k];
  }
  for (int i = tid; i < 128 * K; i += blockDim.x) {
    const int r = i / K, k = i % K;
    reinterpret_cast<bf16*>(reinterpret_cast<char*>(sB) + (k / 16) * 4096 +
                            koff(r, k % 16))[0] = B[r * K + k];
  }
  __syncthreads();

  float d[64];
  using MMA = SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>;
  for (int rep = 0; rep < REPS; ++rep) {
    cute::warpgroup_arrive();
    MMA::fma(make_desc(sA), make_desc(reinterpret_cast<char*>(sB)), FMA_ARGS,
             SG::ScaleOut::Zero);
    for (int s = 1; s < K / 16; ++s) {
      MMA::fma(make_desc(reinterpret_cast<char*>(sA) + s * 2048),
               make_desc(reinterpret_cast<char*>(sB) + s * 4096), FMA_ARGS,
               SG::ScaleOut::One);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
  }

  // accumulator mapping (PTX m64nNk16 f32): warp w, lane l:
  //   d[n8*4 + i*2 + j] -> C[w*16 + l/4 + 8*i][n8*8 + (l%4)*2 + j]
  const int w = tid / 32, l = tid % 32;
  for (int n8 = 0; n8 < 16; ++n8)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        C[(w * 16 + l / 4 + 8 * i) * 128 + n8 * 8 + (l % 4) * 2 + j] = d[n8 * 4 + i * 2 + j];
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void run_nn(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t K) {
  const int smem = (int)(K / 16) * (2048 + 4096);
  cudaFuncSetAttribute((void*)probe_nn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  probe_nn<<<1, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()), C.data_ptr<float>(), (int)K);
  C10_CUDA_CHECK(cudaGetLastError());
}

void run(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t K, int64_t reps) {
  const int smem = (int)(K / 16) * (2048 + 4096);
  cudaFuncSetAttribute((void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  probe<<<1, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()), C.data_ptr<float>(), (int)K, (int)reps);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""

cpp_src = "void run(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t K, int64_t reps);\nvoid run_nn(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t K);"

ext = load_inline(
    name="wgmma_probe",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["run", "run_nn"],
    extra_cuda_cflags=["-O3", "-arch=sm_90a", f"-I{CUTE_INC}", "--expt-relaxed-constexpr"],
    verbose=False,
)

torch.cuda.set_device(0)
torch.manual_seed(0)
for K in (16, 64, 256):
    A = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(64, 128, device="cuda", dtype=torch.float32)
    ext.run(A, B, C, K, 1)
    torch.cuda.synchronize()
    ref = A.float() @ B.float().T
    err = (C - ref).abs().max().item()
    print(f"K={K:4d}: max_err={err:.4e} ref_max={ref.abs().max().item():.3f} {'OK' if err < 0.15 else 'FAIL'}")

# perf: time REPS wgmma sweeps over the K=256 tile (per-rep = 16 wgmma = 64x128x256 MACs)


K = 256
A = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)
B = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
C = torch.zeros(64, 128, device="cuda", dtype=torch.float32)
ext.run(A, B, C, K, 10)
torch.cuda.synchronize()
t0 = time.time()
ext.run(A, B, C, K, 20000)
torch.cuda.synchronize()
dt = time.time() - t0
fl = 2 * 64 * 128 * K * 20000
print(f"single-warpgroup sustained: {fl / dt * 1e-12:.1f} TF ({dt / 20000 * 1e9:.0f} ns per 16-step sweep)")
for K in (16, 64, 256):
    A = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)
    Bn = torch.randn(K, 128, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(64, 128, device="cuda", dtype=torch.float32)
    ext.run_nn(A, Bn, C, K)
    torch.cuda.synchronize()
    ref = A.float() @ Bn.float()
    err = (C - ref).abs().max().item()
    print(f"NN K={K:4d}: max_err={err:.4e} {'OK' if err < 0.15 else 'FAIL'}")
print("PROBE DONE")
