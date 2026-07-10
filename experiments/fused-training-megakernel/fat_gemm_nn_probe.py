"""Standalone WGMMA fat-tile probe for wide-N head-dX-style NN fp32 GEMMs.

This does not touch the production megakernel route. It compares the current
m64n128 NN fp32 route against a 100KB-compatible m64n256 NN direct-store tile for
the qwen lm-head backward dX shape.

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python fat_gemm_nn_probe.py
"""

from __future__ import annotations

import sys

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

#define BM 128
#define BK 64
#define BN128 128
#define BN256 256

#define D4(i) d[(i) + 0], d[(i) + 1], d[(i) + 2], d[(i) + 3]
#define D16(i) D4(i), D4((i) + 4), D4((i) + 8), D4((i) + 12)
#define FMA64 D16(0), D16(16), D16(32), D16(48)
#define FMA128 FMA64, D16(64), D16(80), D16(96), D16(112)

struct Smem128 {
  bf16 A[2][2][4096];   // [stage][wg][64x64 K-major SW128]
  bf16 B[2][8192];      // [stage][128 cols x 64 k elts MN-major SW128]
};

struct Smem256 {
  bf16 A[2][2][4096];   // [stage][wg][64x64 K-major SW128]
  bf16 B[2][16384];     // [stage][256 cols x 64 k elts MN-major SW128]
};

__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}

__device__ __forceinline__ int mnoff_sw(int k, int n8) {
  return k * 128 + ((((n8 >> 3) ^ (k & 7)) << 4));
}

__device__ __forceinline__ uint64_t desc_ksw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 32;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;
  return d.desc_;
}

__device__ __forceinline__ uint64_t desc_mnsw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 2048;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (8192 >> 4);
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;
  return d.desc_;
}

template <class MMA>
__device__ __forceinline__ void mma64(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                      float (&d)[64]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], FMA64, SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

template <class MMA>
__device__ __forceinline__ void mma128(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                       float (&d)[128]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], FMA128, SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

__global__ void gemm_n128_nn_f32(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K,
                                 int claim_sz, int* __restrict__ cursor) {
  extern __shared__ char raw[];
  char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + 1023) &
                                       ~uintptr_t(1023));
  Smem128& S = *reinterpret_cast<Smem128*>(smem);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wtid = tid & 127;
  const int n_tiles = N / BN128;
  const int total = (M / BM) * n_tiles;
  for (;;) {
    __shared__ int s_t0;
    if (tid == 0) s_t0 = atomicAdd(cursor, claim_sz);
    __syncthreads();
    const int t0 = s_t0;
    if (t0 >= total) return;
    const int t1 = min(t0 + claim_sz, total);
    for (int tile = t0; tile < t1; ++tile) {
      const int m0 = (tile / n_tiles) * BM;
      const int n0 = (tile % n_tiles) * BN128;
      float d[64];
#pragma unroll
      for (int i = 0; i < 64; ++i) d[i] = 0.0f;
      const int iters = K / BK;
      auto issue = [&](int k0, int st) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int v = tid + i * 256;
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.A[st][r / 64]) +
                                      koff_sw(r % 64, k8),
                                  &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int v = tid + i * 256;
          const int k = v / 16, n8 = (v % 16) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + (n8 / 64) * 8192 +
                                      mnoff_sw(k, n8 % 64),
                                  &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
        }
        __pipeline_commit();
      };
      issue(0, 0);
      for (int kt = 0; kt < iters; ++kt) {
        if (kt + 1 < iters) issue((kt + 1) * BK, (kt + 1) & 1);
        __pipeline_wait_prior(kt + 1 < iters ? 1 : 0);
        __syncthreads();
        uint64_t da[4], db[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          da[s] = desc_ksw(S.A[kt & 1][wg], s);
          db[s] = desc_mnsw(S.B[kt & 1], s);
        }
        mma64<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
        __syncthreads();
      }
      float* Cs = reinterpret_cast<float*>(smem);
      const int w = wtid >> 5, l = wtid & 31;
      {
        const int r = wg * 64 + w * 16 + l / 4;
        const int cb = (l & 3) * 2;
#pragma unroll
        for (int n8 = 0; n8 < 16; ++n8)
#pragma unroll
          for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
              Cs[(r + 8 * i) * BN128 + n8 * 8 + cb + j] =
                  d[n8 * 4 + i * 2 + j];
      }
      __syncthreads();
#pragma unroll
      for (int g = 0; g < 16; ++g) {
        const int gid = tid + g * 256;
        const int m = gid / 32, c4 = (gid % 32) * 4;
        float4 out = make_float4(Cs[m * BN128 + c4 + 0], Cs[m * BN128 + c4 + 1],
                                 Cs[m * BN128 + c4 + 2], Cs[m * BN128 + c4 + 3]);
        *reinterpret_cast<float4*>(&C[(int64_t)(m0 + m) * N + n0 + c4]) = out;
      }
      __syncthreads();
    }
  }
}

__global__ void gemm_n256_nn_f32(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K,
                                 int claim_sz, int* __restrict__ cursor) {
  extern __shared__ char raw[];
  char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + 1023) &
                                       ~uintptr_t(1023));
  Smem256& S = *reinterpret_cast<Smem256*>(smem);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wtid = tid & 127;
  const int n_tiles = N / BN256;
  const int total = (M / BM) * n_tiles;
  for (;;) {
    __shared__ int s_t0;
    if (tid == 0) s_t0 = atomicAdd(cursor, claim_sz);
    __syncthreads();
    const int t0 = s_t0;
    if (t0 >= total) return;
    const int t1 = min(t0 + claim_sz, total);
    for (int tile = t0; tile < t1; ++tile) {
      const int m0 = (tile / n_tiles) * BM;
      const int n0 = (tile % n_tiles) * BN256;
      float d[128];
#pragma unroll
      for (int i = 0; i < 128; ++i) d[i] = 0.0f;
      const int iters = K / BK;
      auto issue = [&](int k0, int st) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int v = tid + i * 256;
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.A[st][r / 64]) +
                                      koff_sw(r % 64, k8),
                                  &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int v = tid + i * 256;
          const int k = v / 32, n8 = (v % 32) * 8;
          __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + (n8 / 64) * 8192 +
                                      mnoff_sw(k, n8 % 64),
                                  &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
        }
        __pipeline_commit();
      };
      issue(0, 0);
      for (int kt = 0; kt < iters; ++kt) {
        if (kt + 1 < iters) issue((kt + 1) * BK, (kt + 1) & 1);
        __pipeline_wait_prior(kt + 1 < iters ? 1 : 0);
        __syncthreads();
        uint64_t da[4], db[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          da[s] = desc_ksw(S.A[kt & 1][wg], s);
          db[s] = desc_mnsw(S.B[kt & 1], s);
        }
        mma128<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
        __syncthreads();
      }
      const int w = wtid >> 5, l = wtid & 31;
      const int cb = (l & 3) * 2;
#pragma unroll
      for (int n8 = 0; n8 < 32; ++n8)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
          float2 out = make_float2(d[n8 * 4 + i * 2 + 0], d[n8 * 4 + i * 2 + 1]);
          *reinterpret_cast<float2*>(&C[(int64_t)(m0 + r) * N + n0 + n8 * 8 + cb]) = out;
        }
      __syncthreads();
    }
  }
}

#include <torch/extension.h>

void run_n128(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t M, int64_t N,
              int64_t K, int64_t claim, torch::Tensor cursor, int64_t nblocks) {
  const int smem = 100 * 1024;
  cudaFuncSetAttribute((const void*)gemm_n128_nn_f32,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  gemm_n128_nn_f32<<<(int)nblocks, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(A.data_ptr()),
      reinterpret_cast<const bf16*>(B.data_ptr()), reinterpret_cast<float*>(C.data_ptr()),
      (int)M, (int)N, (int)K, (int)claim, cursor.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_n256(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t M, int64_t N,
              int64_t K, int64_t claim, torch::Tensor cursor, int64_t nblocks) {
  const int smem = 100 * 1024;
  cudaFuncSetAttribute((const void*)gemm_n256_nn_f32,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  gemm_n256_nn_f32<<<(int)nblocks, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(A.data_ptr()),
      reinterpret_cast<const bf16*>(B.data_ptr()), reinterpret_cast<float*>(C.data_ptr()),
      (int)M, (int)N, (int)K, (int)claim, cursor.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

cpp_src = """
void run_n128(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t M, int64_t N,
              int64_t K, int64_t claim, torch::Tensor cursor, int64_t nblocks);
void run_n256(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t M, int64_t N,
              int64_t K, int64_t claim, torch::Tensor cursor, int64_t nblocks);
"""


def build():
    return load_inline(
        name="fat_gemm_nn_probe",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["run_n128", "run_n256"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-I{CUTE_INC}",
            "--expt-relaxed-constexpr",
        ],
        extra_include_paths=[CUTE_INC],
        verbose=False,
    )


def claim_for(tiles: int, nblocks: int) -> int:
    return max(1, min(8, (tiles + nblocks - 1) // nblocks))


def run_one(ext, fn_name: str, A, B, C, M: int, N: int, K: int, bn: int, iters: int, nblocks: int):
    tiles = (M // 128) * (N // bn)
    claim = claim_for(tiles, nblocks)
    cursor = torch.zeros(1, device="cuda", dtype=torch.int32)
    fn = getattr(ext, fn_name)
    for _ in range(4):
        cursor.zero_()
        fn(A, B, C, M, N, K, claim, cursor, nblocks)
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        cursor.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(A, B, C, M, N, K, claim, cursor, nblocks)
        e.record()
        torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) * 1e3)
    vals.sort()
    return vals[len(vals) // 2], vals[0], vals[-1], tiles, claim


def check_correct(ext):
    M, N, K = 256, 512, 256
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ref = A.float() @ B.float()
    for name, bn in (("run_n128", 128), ("run_n256", 256)):
        C = torch.empty(M, N, device="cuda", dtype=torch.float32)
        med, lo, hi, tiles, claim = run_one(ext, name, A, B, C, M, N, K, bn, 3, 132)
        diff = (C - ref).abs()
        denom = ref.abs().clamp_min(0.5)
        rel = (diff / denom).max().item()
        print(
            f"check {name}: rel={rel:.4e} max_abs={diff.max().item():.4e} time={med:.1f}us tiles={tiles} claim={claim}",
            flush=True,
        )
        assert rel < 3e-2, name


def bench_shape(ext, label: str, M: int, N: int, K: int, iters: int):
    assert M % 128 == 0 and N % 256 == 0 and K % 64 == 0
    torch.manual_seed(1)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    rows = []
    for name, bn in (("run_n128", 128), ("run_n256", 256)):
        med, lo, hi, tiles, claim = run_one(ext, name, A, B, C, M, N, K, bn, iters, 132)
        tf = (2.0 * M * N * K) / med / 1e6
        rows.append((name, med, lo, hi, tf, tiles, claim))
    print(label, flush=True)
    for name, med, lo, hi, tf, tiles, claim in rows:
        print(
            f"  {name:8s} med={med:8.1f}us min={lo:8.1f} max={hi:8.1f} tf={tf:6.1f} tiles={tiles:5d} claim={claim}",
            flush=True,
        )


if __name__ == "__main__":
    torch.cuda.set_device(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    iters = 8 if mode == "qwen" else 4
    ext = build()
    print("built fat_gemm_nn_probe", flush=True)
    check_correct(ext)
    shapes = [("small head_dx 1024x512x16384", 1024, 512, 16384)]
    if mode == "qwen":
        shapes = [("qwen4b head_dx 1024x2560x151936", 1024, 2560, 151936)]
    for shape in shapes:
        bench_shape(ext, *shape, iters=iters)
