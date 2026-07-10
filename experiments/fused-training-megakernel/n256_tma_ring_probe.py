"""n256 mbarrier-ring TMA-feed probe (GEMM round-4 port precursor).

The qwen n256 NN/TN bodies now run the barrier-free mbarrier ring
(MK_GEMM_MBAR_RING, promoted 20260706T1835Z). Round 4 measured the TMA feed on
top of the ring at +3.5..7.1% on the dX family standalone — but with
__grid_constant__ tensormaps, which the interpreter cannot use (one persistent
kernel, program-dependent buffer set). This probe validates the EXACT protocol
intended for ops.cuh before any op-library edit:

  - tensormaps in GLOBAL memory (128B-aligned rows of a CUDA uint8 tensor),
    acquired in-kernel with fence.proxy.tensormap::generic.acquire.gpu
  - per-CTA elected-thread cp.async.bulk.tensor.2d (NO cluster, NO multicast —
    the 0307Z paired-multicast no-go is a different mechanism)
  - bfull mbarrier count 1 + mbarrier.arrive.expect_tx by the issuing thread
    (replaces the 256-count cp.async.mbarrier.arrive.noinc protocol)
  - ring structure, stage count, smem layout, epilogue identical to
    op_gemm_wgmma_n256_nn_f32_impl (claim-loop persistent CTAs included)

variant 0: current per-thread cp.async SW128 + noinc arrivals (control)
variant 1: elected-thread TMA feed (candidate)

Traps honored: timeout-guard every run (phase desync spins at 99% SM); SW128
slabs == CU_TENSOR_MAP_SWIZZLE_128B; no cp.reduce.async.bulk anywhere.
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
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int STAGES = 3;
constexpr int A_STAGE_BYTES = 16 * 1024;
constexpr int B_STAGE_BYTES = 32 * 1024;
constexpr int TX_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
constexpr int SMEM_BYTES = STAGES * TX_BYTES + 256 + 1024;

#define D4(i) d[(i) + 0], d[(i) + 1], d[(i) + 2], d[(i) + 3]
#define D16(i) D4(i), D4((i) + 4), D4((i) + 8), D4((i) + 12)
#define D64 D16(0), D16(16), D16(32), D16(48)
#define D128 D64, D16(64), D16(80), D16(96), D16(112)

struct Stage {
  bf16 A[2][4096];  // two 64-row (NN K-major) / 64-col (TN MN-major) SW128 slabs
  bf16 B[16384];    // four 64-col MN-major SW128 sub-slabs, 8KB apart
};
struct RingSmem {
  Stage stage[STAGES];
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

// The two new primitives under test for the interpreter port.
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
__device__ __forceinline__ void mma_n256(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                         float (&d)[128]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) MMA::fma(da[s], db[s], D128, SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

__global__ void __launch_bounds__(256) ring_kernel(
    const bf16* __restrict__ A, const bf16* __restrict__ B, float* __restrict__ C, int M,
    int N, int K, int a_t, int variant, const uint8_t* __restrict__ tmaps,
    int* __restrict__ claim) {
  extern __shared__ char raw[];
  char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + 1023) &
                                       ~uintptr_t(1023));
  RingSmem& S = *reinterpret_cast<RingSmem*>(smem);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wtid = tid & 127;
  const int m_tiles = M / BM;
  const int n_tiles = N / BN;
  const int tiles = m_tiles * n_tiles;
  const int iters = K / BK;
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
    const int mt = tile % m_tiles;  // n-major, like the promoted qwen route
    const int nt = tile / m_tiles;
    const int m0 = mt * BM;
    const int n0 = nt * BN;

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
              reinterpret_cast<char*>(S.stage[st].A[h]) + mnoff_sw(k, m8),
              &A[(int64_t)(k0 + k) * M + m0 + h * 64 + m8], 16);
        } else {
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(
              reinterpret_cast<char*>(S.stage[st].A[r / 64]) + koff_sw(r % 64, k8),
              &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {  // B[K,N] N-contig: four 64-col MN slabs
        const int v = tid + i * 256;
        const int k = v / 32, n8 = (v % 32) * 8;
        __pipeline_memcpy_async(reinterpret_cast<char*>(S.stage[st].B) +
                                    (n8 / 64) * 8192 + mnoff_sw(k, n8 % 64),
                                &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
      }
      __pipeline_commit();
    };
    auto issue_stage_tma = [&](int k0, int st) {
      if (tid == 0) {
        mbar_expect_tx(&S.bfull[st], TX_BYTES);
        if (a_t) {
          tma_load_2d(tmA, S.stage[st].A[0], m0, k0, &S.bfull[st]);
          tma_load_2d(tmA, S.stage[st].A[1], m0 + 64, k0, &S.bfull[st]);
        } else {
          tma_load_2d(tmA, S.stage[st].A[0], k0, m0, &S.bfull[st]);
        }
#pragma unroll
        for (int g = 0; g < 4; ++g)
          tma_load_2d(tmB, reinterpret_cast<char*>(S.stage[st].B) + g * 8192,
                      n0 + g * 64, k0, &S.bfull[st]);
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

    float d[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) d[i] = 0.0f;
    constexpr int LEAD = STAGES - 2;
    for (int p = 0; p < min(LEAD + 1, iters); ++p) issue_stage_mb(p);
    for (int t = 0; t < iters; ++t) {
      const int st = t % STAGES;
      mbar_wait(&S.bfull[st], (t / STAGES) & 1);
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = a_t ? desc_mnsw(S.stage[st].A[wg], s) : desc_ksw(S.stage[st].A[wg], s);
        db[s] = desc_mnsw128(S.stage[st].B, s);
      }
      if (a_t)
        mma_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(da, db,
                                                                                 d);
      else
        mma_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db,
                                                                                d);
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
    for (int n8 = 0; n8 < 32; ++n8) {
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
                           int64_t K, int64_t a_t) {
  auto out = torch::empty({2, 128}, torch::dtype(torch::kUInt8));
  uint8_t* p = out.data_ptr<uint8_t>();
  if (a_t) {  // A[K,M] M-contig: MN-major slabs, box {64 m, 64 k}
    encode_2d(p, A.data_ptr(), M, K, M * 2, 64, 64);
  } else {  // A[M,K] K-contig: K-major slab, box {64 k, 128 m}
    encode_2d(p, A.data_ptr(), K, M, K * 2, 64, 128);
  }
  // B[K,N] N-contig: MN-major sub-slabs, box {64 n, 64 k}
  encode_2d(p + 128, B.data_ptr(), N, K, N * 2, 64, 64);
  return out;
}

void run_ring(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor tmaps,
              torch::Tensor claim, int64_t M, int64_t N, int64_t K, int64_t a_t,
              int64_t variant, int64_t nblocks) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda() && tmaps.is_cuda());
  TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tmaps.data_ptr()) % 128 == 0);
  static int configured = 0;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)ring_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        SMEM_BYTES));
    configured = 1;
  }
  claim.zero_();
  ring_kernel<<<(int)nblocks, 256, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(A.data_ptr()),
      reinterpret_cast<const bf16*>(B.data_ptr()), C.data_ptr<float>(), (int)M, (int)N,
      (int)K, (int)a_t, (int)variant, tmaps.data_ptr<uint8_t>(),
      claim.data_ptr<int>());
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_ring", &run_ring, "n256 ring cp.async vs global-tensormap TMA feed");
  m.def("encode_tmaps", &encode_tmaps, "encode A/B CUtensorMaps into a 2x128 u8 tensor");
}
"""


def build():
    return load_inline(
        name="n256_tma_ring_probe",
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
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", dev],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def make_inputs(M, N, K, a_t, seed):
    torch.manual_seed(seed)
    if a_t:
        a = torch.randn(K, M, device="cuda", dtype=torch.bfloat16) * 0.05
    else:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    return a, b, c


def check(ext, M, N, K, a_t, nblocks, claim):
    a, b, c = make_inputs(M, N, K, a_t, seed=11 + a_t)
    tmaps = ext.encode_tmaps(a, b, M, N, K, a_t).cuda()
    rows = torch.randperm(M)[:128].sort().values
    am = (a.t() if a_t else a)[rows].float()
    ref = am @ b.float()
    for variant, name in ((0, "cpasync-ring"), (1, "tma-ring")):
        c.fill_(float("nan"))
        ext.run_ring(a, b, c, tmaps, claim, M, N, K, a_t, variant, nblocks)
        torch.cuda.synchronize()
        got = c[rows.cuda()]
        diff = (got - ref).abs()
        rel = (diff / ref.abs().clamp_min(0.25)).max().item()
        print(
            f"check {'TN' if a_t else 'NN'} M={M} N={N} K={K} {name}: max_abs={diff.max().item():.6e} rel={rel:.6e}",
            flush=True,
        )
        assert rel < 3e-2, name


def bench(ext, M, N, K, a_t, nblocks, claim, reps, iters, order):
    a, b, c = make_inputs(M, N, K, a_t, seed=101 + a_t)
    tmaps = ext.encode_tmaps(a, b, M, N, K, a_t).cuda()
    results = {}
    variants = [(0, "cpasync-ring"), (1, "tma-ring")]
    if order == "rev":
        variants = variants[::-1]
    for variant, name in variants:
        for _ in range(2):
            ext.run_ring(a, b, c, tmaps, claim, M, N, K, a_t, variant, nblocks)
        torch.cuda.synchronize()
        vals = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                ext.run_ring(a, b, c, tmaps, claim, M, N, K, a_t, variant, nblocks)
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) * 1e3 / reps)
        vals.sort()
        results[name] = statistics.median(vals)
        tf = (2.0 * M * N * K) / (results[name] * 1e-6) / 1e12
        print(
            f"  {name:12s} med={results[name]:9.3f}us min={vals[0]:9.3f} max={vals[-1]:9.3f} tf={tf:7.1f}", flush=True
        )
    delta = results["tma-ring"] - results["cpasync-ring"]
    pct = 100.0 * delta / results["cpasync-ring"]
    print(
        f"bench {'TN' if a_t else 'NN'} M={M} N={N} K={K} order={order}: "
        f"tma-minus-cpasync {delta:+9.3f}us ({pct:+.2f}%)",
        flush=True,
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    u0 = gpu_util()
    ext = build()
    print(f"built n256_tma_ring_probe (pre-util {u0}%)", flush=True)
    dev = torch.cuda.current_device()
    nblocks = torch.cuda.get_device_properties(dev).multi_processor_count
    claim = torch.zeros(1, device="cuda", dtype=torch.int32)
    check(ext, 1024, 2560, 2048, 0, nblocks, claim)
    check(ext, 2560, 2560, 1024, 1, nblocks, claim)
    if mode == "qwen":
        order = sys.argv[2] if len(sys.argv) > 2 else "fwd"
        # qwen head-dX NN 1024x2560x151936
        bench(ext, 1024, 2560, 151936, 0, nblocks, claim, reps=4, iters=12, order=order)
        # qwen mlp gate_up dW TN 19456x2560x1024
        bench(ext, 19456, 2560, 1024, 1, nblocks, claim, reps=8, iters=12, order=order)
        # qwen lm-head dW TN 151936x2560x1024
        bench(ext, 151936, 2560, 1024, 1, nblocks, claim, reps=2, iters=10, order=order)
    u1 = gpu_util()
    print(f"post-util {u1}% (pre {u0}%)", flush=True)
