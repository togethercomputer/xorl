"""pipe_probe.py — P4b gate 1: how much gemm throughput does pipeline depth buy?

The in-model wgmma gemm (op_gemm_wgmma) runs a 2-stage cp.async feed with
warpgroup_wait<0> per 64-K tile: the GMMA pipe drains every iteration and cp.async
has one mma-iteration (~0.3us) to cover DRAM latency (~0.5-0.7us). Measured in-model:
40-72 TF. This probe replicates the op's exact tile/claim/epilogue structure as a
standalone persistent kernel and parametrizes <STAGES, MMAS_IN_FLIGHT>:

  A  (2,0)  = the current op structure (control)
  B3 (3,0)  = deeper feed, draining mma
  B4 (4,0)  = deepest feed (96KB smem), draining mma
  P3 (3,1)  = 1 mma batch in flight across the sync, lead 1
  P4 (4,1)  = 1 mma batch in flight, lead 2  <- the cutlass sm90 mainloop shape

Load lead L = STAGES-1-W (max safe: stage reuse needs mma[t+L-S] retired, wait<W>
retires mma[t-W] at end of iter t). All variants: 132 blocks x 256 threads, 1
block/SM (smem-forced), in-model claim quantum, NT and NN majors, bf16 store
epilogue staged through smem. Gate for the round: >=1.5x variant A on the small
NT shapes.

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python pipe_probe.py [quick|full]
"""

import statistics
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
#define BN 64
#define BK 64
#define LDC 68

// ---- INTER (no-swizzle) arrangements + descriptors (validated: ops.cuh/wgmma_probe) --
__device__ __forceinline__ int koff(int r, int k) {
  return ((r >> 3) << 8) + ((k >> 3) << 7) + ((r & 7) << 4) + ((k & 7) << 1);
}
__device__ __forceinline__ int mnoff(int mn, int k) {
  return ((mn >> 3) << 7) + ((k >> 3) << 10) + ((mn & 7) << 1) + ((k & 7) << 4);
}
__device__ __forceinline__ uint64_t desc_k(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (128 >> 4);
  d.bitfield.stride_byte_offset_ = (256 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}
__device__ __forceinline__ uint64_t desc_mn(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (1024 >> 4);
  d.bitfield.stride_byte_offset_ = (128 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

template <class MMA, int W>
__device__ __forceinline__ void mma_ktile(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                          float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
             d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
             d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
             SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<W>();
}

// stage buffers: [stage][A: 2 halves x 4 k16 x 1024 | B: 4 k16 x 1024] = 24KB/stage.
// The SW128 variants reuse the same 24KB footprint as flat 64-row slabs:
//   A slab h: bytes [h*8192, h*8192+16384), B slab: the 8KB after A.
struct Stage {
  bf16 A[2][4][1024];  // 16KB
  bf16 B[4][1024];     // 8KB
};

// ---- SW128 (128B-swizzle) canonical layouts -------------------------------------------
// K-major slab: [64 rows][64 k-elts]; row = 128B = 8 x 16B chunks; chunk c of row r
// stored at chunk c ^ (r & 7). Store offset for the 16B vector at (r, k8):
__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}
// MN-major slab: [64 k-rows][64 mn-elts]; same swizzle with roles flipped.
__device__ __forceinline__ int mnoff_sw(int k, int mn8) {
  return k * 128 + ((((mn8 >> 3) ^ (k & 7)) << 4));
}
// K-major SW128 descriptor for the k16-atom s of a 64-row slab (deep_gemm recipe:
// layout_type=1(B128), LBO=0, SBO=1024B; per-atom start = base + s*32B).
__device__ __forceinline__ uint64_t desc_k_sw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 32;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;  // B128
  return d.desc_;
}
// MN-major SW128 descriptor: k16-atom s = 16 k-rows = 2KB step; SBO = 8-k-row group.
__device__ __forceinline__ uint64_t desc_mn_sw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 2048;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;  // B128
  return d.desc_;
}

template <int S, int SW>
__device__ __forceinline__ void issue_stage(Stage* St, const bf16* A, const bf16* B,
                                            int M, int N, int K, bool a_t, bool b_t,
                                            int m0, int n0, int k0, int st) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // A: 128r x 64k = 1024 16B vectors
    const int v = tid + i * 256;
    if (!a_t) {
      const int r = v / 8, k8 = (v % 8) * 8;
      char* dst = SW ? reinterpret_cast<char*>(St[st].A[r / 64]) + koff_sw(r % 64, k8)
                     : reinterpret_cast<char*>(St[st].A[r / 64][k8 / 16]) + koff(r % 64, k8 % 16);
      __pipeline_memcpy_async(dst, &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
    } else {
      const int h = v / 512, w_ = v % 512;
      const int k = w_ / 8, m8 = (w_ % 8) * 8;
      char* dst = SW ? reinterpret_cast<char*>(St[st].A[h]) + mnoff_sw(k, m8)
                     : reinterpret_cast<char*>(St[st].A[h][k / 16]) + mnoff(m8, k % 16);
      __pipeline_memcpy_async(dst, &A[(int64_t)(k0 + k) * M + m0 + h * 64 + m8], 16);
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {  // B: 64r x 64k = 512 16B vectors
    const int v = tid + i * 256;
    if (b_t) {
      const int r = v / 8, k8 = (v % 8) * 8;
      char* dst = SW ? reinterpret_cast<char*>(St[st].B) + koff_sw(r, k8)
                     : reinterpret_cast<char*>(St[st].B[k8 / 16]) + koff(r, k8 % 16);
      __pipeline_memcpy_async(dst, &B[(int64_t)(n0 + r) * K + k0 + k8], 16);
    } else {
      const int k = v / 8, n8 = (v % 8) * 8;
      char* dst = SW ? reinterpret_cast<char*>(St[st].B) + mnoff_sw(k, n8)
                     : reinterpret_cast<char*>(St[st].B[k / 16]) + mnoff(n8, k % 16);
      __pipeline_memcpy_async(dst, &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
    }
  }
  __pipeline_commit();
}

// One (S, W) mainloop instance. Persistent blocks claim tile batches from a cursor
// exactly like the in-model executor (claim = claim_sz tiles per atomicAdd).
template <int S, int W, int SW>
__global__ void gemm_pipe(const bf16* __restrict__ A, const bf16* __restrict__ B,
                          bf16* __restrict__ C, int M, int N, int K, int flags,
                          int claim_sz, int* __restrict__ cursor) {
  extern __shared__ char smem[];
  // SW128 swizzle phase = absolute smem address bits [7,10): slab bases must be
  // 1024B-aligned or the store-side XOR disagrees with the wgmma read pattern.
  char* smem_al = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem) + 1023) & ~uintptr_t(1023));
  Stage* St = reinterpret_cast<Stage*>(smem_al);
  float* Cs = reinterpret_cast<float*>(smem_al);  // epilogue overlays dead stages
  const bool a_t = flags & 1, b_t = flags & 2;
  const int n_tiles = N / BN;
  const int total = (M / BM) * n_tiles;
  const int tid = threadIdx.x;
  const int wg = tid / 128;
  const int wtid = tid % 128;
  constexpr int L = S - 1 - W;  // load lead (max safe for stage reuse)

  for (;;) {
    __shared__ int s_t0;
    if (tid == 0) s_t0 = atomicAdd(cursor, claim_sz);
    __syncthreads();
    const int t0 = s_t0;
    if (t0 >= total) return;
    const int t1 = min(t0 + claim_sz, total);
    for (int tile = t0; tile < t1; ++tile) {
      const int m0 = (tile / n_tiles) * BM;
      const int n0 = (tile % n_tiles) * BN;
      const int iters = K / BK;

      float d[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) d[i] = 0.0f;

#pragma unroll
      for (int p = 0; p < L + 1; ++p)  // prologue: stages for iters 0..L
        if (p < iters) issue_stage<S, SW>(St, A, B, M, N, K, a_t, b_t, m0, n0, p * BK, p % S);

      for (int t = 0; t < iters; ++t) {
        // outstanding commit groups after this wait: iters t+1..min(t+L, iters-1)
        const int out = max(0, min(t + L, iters - 1) - t);
        __pipeline_wait_prior(out);
        __syncthreads();
        uint64_t da[4], db[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          if (SW) {
            da[s] = a_t ? desc_mn_sw(St[t % S].A[wg], s) : desc_k_sw(St[t % S].A[wg], s);
            db[s] = b_t ? desc_k_sw(St[t % S].B, s) : desc_mn_sw(St[t % S].B, s);
          } else {
            da[s] = a_t ? desc_mn(St[t % S].A[wg][s]) : desc_k(St[t % S].A[wg][s]);
            db[s] = b_t ? desc_k(St[t % S].B[s]) : desc_mn(St[t % S].B[s]);
          }
        }
        if (!a_t && b_t)
          mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>, W>(da, db, d);
        else if (!a_t && !b_t)
          mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>, W>(da, db, d);
        else if (a_t && b_t)
          mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>, W>(da, db, d);
        else
          mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>, W>(da, db, d);
        __syncthreads();
        if (t + L + 1 < iters)
          issue_stage<S, SW>(St, A, B, M, N, K, a_t, b_t, m0, n0, (t + L + 1) * BK, (t + L + 1) % S);
      }
      cute::warpgroup_wait<0>();  // drain before the epilogue overlays stage smem
      __syncthreads();

      // epilogue: stage accs to smem, coalesced bf16 store (ops.cuh recipe)
      const int w = wtid / 32, l = wtid % 32;
      {
        const int r = wg * 64 + w * 16 + l / 4;
        const int cb = (l % 4) * 2;
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
          for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
              Cs[(r + 8 * i) * LDC + n8 * 8 + cb + j] = d[n8 * 4 + i * 2 + j];
      }
      __syncthreads();
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int gid = tid + g * 256;
        const int m = gid / 8, c8 = (gid % 8) * 8;
        const int64_t idx = (int64_t)(m0 + m) * N + n0 + c8;
        uint4 out_v;
        bf16* oe = reinterpret_cast<bf16*>(&out_v);
#pragma unroll
        for (int e = 0; e < 8; ++e) oe[e] = __float2bfloat16(Cs[m * LDC + c8 + e]);
        *reinterpret_cast<uint4*>(&C[idx]) = out_v;
      }
      __syncthreads();  // Cs dead before next tile's prologue reuses the smem
    }
  }
}

// host-facing: variant id selects the <S, W> instantiation
void run_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t M, int64_t N,
              int64_t K, int64_t flags, int64_t claim_sz, torch::Tensor cursor,
              int64_t variant, int64_t nblocks) {
  const bf16* a = reinterpret_cast<const bf16*>(A.data_ptr());
  const bf16* b = reinterpret_cast<const bf16*>(B.data_ptr());
  bf16* c = reinterpret_cast<bf16*>(C.data_ptr());
  int* cur = cursor.data_ptr<int>();
  const int smem = 100 * 1024;  // force 1 block/SM for every variant (in-model regime)
  auto launch = [&](auto kern) {
    static_assert(sizeof(Stage) == 24 * 1024, "stage layout");
    cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<(int)nblocks, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        a, b, c, (int)M, (int)N, (int)K, (int)flags, (int)claim_sz, cur);
  };
  switch (variant) {
    case 0: launch(gemm_pipe<2, 0, 0>); break;
    case 1: launch(gemm_pipe<3, 0, 0>); break;
    case 2: launch(gemm_pipe<4, 0, 0>); break;
    case 3: launch(gemm_pipe<3, 1, 0>); break;
    case 4: launch(gemm_pipe<4, 1, 0>); break;
    case 5: launch(gemm_pipe<2, 0, 1>); break;
    case 6: launch(gemm_pipe<3, 1, 1>); break;
    case 7: launch(gemm_pipe<4, 1, 1>); break;
    default: TORCH_CHECK(false, "bad variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

cpp_src = "void run_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t M, int64_t N, int64_t K, int64_t flags, int64_t claim_sz, torch::Tensor cursor, int64_t variant, int64_t nblocks);"

VARIANTS = ["S2W0(cur)", "S3W0", "S4W0", "S3W1", "S4W1", "S2W0-sw", "S3W1-sw", "S4W1-sw"]


def build():
    return load_inline(
        name="pipe_probe",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["run_gemm"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-I{CUTE_INC}", "--expt-relaxed-constexpr"],
        extra_include_paths=[CUTE_INC],
        verbose=False,
    )


def bench_shape(ext, M, N, K, a_t, b_t, iters=30, nblocks=132):
    """Times each variant; returns dict variant -> (us, tf)."""
    torch.manual_seed(0)
    # operands stored exactly as the op sees them (a_t: A[K,M]; b_t: B[N,K])
    A = torch.randn(K, M, device="cuda", dtype=torch.bfloat16) if a_t else torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) if b_t else torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    Am = A.t() if a_t else A
    Bm = B.t() if b_t else B
    ref = (Am.float() @ Bm.float()).to(torch.bfloat16)
    flags = (1 if a_t else 0) | (2 if b_t else 0)
    tiles = (M // 128) * (N // 64)
    claim = max(1, min(8, (tiles + nblocks - 1) // nblocks))
    cursor = torch.zeros(1, device="cuda", dtype=torch.int32)
    flop = 2.0 * M * N * K
    out = {}
    for v, name in enumerate(VARIANTS):
        # correctness first
        C.zero_()
        cursor.zero_()
        ext.run_gemm(A, B, C, M, N, K, flags, claim, cursor, v, nblocks)
        torch.cuda.synchronize()
        # denom floor 0.5: randn-K products give |ref| ~ sqrt(K); near-zero refs make a
        # pure-relative check blow up on ulp-level differences (measured 9.5e-5 abs).
        rel = ((C.float() - ref.float()).abs() / (ref.float().abs() + 0.5)).max().item()
        assert rel < 3e-2, f"{name} mismatch rel={rel}"
        # timing
        times = []
        for _ in range(iters):
            cursor.zero_()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            ext.run_gemm(A, B, C, M, N, K, flags, claim, cursor, v, nblocks)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e3)
        us = statistics.median(times)
        out[name] = (us, flop / us / 1e6)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    torch.cuda.set_device(0)
    ext = build()
    print("built ok")

    # (label, M, N, K, a_t, b_t) — the measured on-path/off-path heavy hitters
    shapes = [
        ("small NT gu      1024x3072x512 ", 1024, 3072, 512, False, True),
        ("small NT lm_head 1024x16384x512", 1024, 16384, 512, False, True),
        ("small NT down    1024x512x1536 ", 1024, 512, 1536, False, True),
        ("small NN dX-gu   1024x512x3072 ", 1024, 512, 3072, False, False),
        ("small NN dX-down 1024x1536x512 ", 1024, 1536, 512, False, False),
        ("small TN dW-gu   3072x512x1024 ", 3072, 512, 1024, True, False),
        ("nano  NT lm_head 512x8192x256  ", 512, 8192, 256, False, True),
        ("nano  NN dX-wo   512x256x256   ", 512, 256, 256, False, False),
    ]
    if mode == "quick":
        shapes = shapes[:2]

    hdr = "shape".ljust(34) + "".join(v.rjust(18) for v in VARIANTS)
    print(hdr)
    for label, M, N, K, a_t, b_t in shapes:
        r = bench_shape(ext, M, N, K, a_t, b_t)
        row = label.ljust(34)
        for v in VARIANTS:
            us, tf = r[v]
            row += f"{us:9.1f}us {tf:5.0f}TF".rjust(18)
        print(row)
