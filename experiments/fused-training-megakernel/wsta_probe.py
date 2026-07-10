"""wsta_probe.py — weight-stationary (v4) gate: do resident W-slices beat streaming?

Every P4b measurement says the interpreter is LATENCY-bound (SM issue 19%, DRAM
<10%): a gemm tile's cost is dominated by its smem stage-fill latency, not math or
bandwidth. Weights are the operand we re-stream from gmem every time a tile is
claimed — but the host KNOWS the program, so it can PIN (instr, n-block) tiles to
specific blocks and keep each block's W slice RESIDENT in smem across all its
m-tiles and across steps, with the NEXT layer's slice prefetched during the current
layer's compute (the Diamos persistent-RNN lineage, P6 survey item 3).

Probe: a serial chain of L "layers", each an NT gemm y_l = x_l @ W_l^T at model
shapes; x_{l+1} depends on y_l (chain dependency enforced by reading y into the
next A). Three variants, same math, same 132 blocks x 256 threads x 1/SM:

  A  stream-both  : per tile, cp.async A-tile AND B(W)-tile per k-stage (the
                    current op_gemm_wgmma structure, SW128).
  B  W-resident   : block's W slice [64, K] preloaded ONCE (outside timing), only
                    A streams per k-stage; wgmma reads W from the resident slab.
  C  B + prefetch : during layer l's compute, a dedicated warp cp.asyncs layer
                    l+1's W slice into the other half of a double buffer (so cold
                    weights never stall the chain even on first touch).

Tiles: 128x64 (two warpgroups share the same 64 output cols, as in the op). Pinned
assignment: block b owns n-block (b % n_tiles) and walks m with stride
nblocks/n_tiles — every block computes with ONE W slice per layer.

Smem: A stages 2x16KB ping-pong + W resident 2 x [64 x K] slabs (K<=512: 64KB per
slab at K=512 -> K capped at 512 per slice; larger K splits into k-chunks with
split-K atomics in a real integration). Gate: B (or C) >= 1.3x variant A on the
chain time at nano/small gemm shapes.

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python wsta_probe.py
"""

import statistics

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

__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}
__device__ __forceinline__ uint64_t desc_k_sw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 32;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;
  return d.desc_;
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// grid-wide layer barrier: arrive with red.release, spin on acquire
__device__ __forceinline__ void layer_barrier(int* count, int l, int nblocks, int tid) {
  if (tid == 0) {
    int one = 1;
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(count + l), "r"(one)
                 : "memory");
  }
  if (tid == 0)
    while (ld_acquire_gpu(count + l) < nblocks) __nanosleep(128);
  __syncthreads();
}

// smem plan (dynamic, 1024B-aligned base):
//   A stages: 2 x 16KB ping-pong           [offset 0, 32KB)
//   W slabs : 2 x (K/64) x 8KB             [offset 32KB, ...)  (double buffer for C)
//   epilogue Cs overlays the A stages
// K <= 512 -> W slab <= 64KB -> total <= 32 + 128 = 160KB? K=512: slab = 8 k-chunks
// x 8KB = 64KB, x2 = 128KB + 32KB A = 160KB > 227? fits (max 227KB). K=256: 96KB.

template <int VARIANT>  // 0 = stream both, 1 = W resident, 2 = resident + prefetch
__global__ void wsta_chain(const bf16* __restrict__ X0, bf16* const* __restrict__ Ys,
                           const bf16* const* __restrict__ Ws, int L, int M, int N,
                           int K, int* __restrict__ bar) {
  extern __shared__ char smem[];
  char* base = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem) + 1023) & ~uintptr_t(1023));
  bf16* Ast = reinterpret_cast<bf16*>(base);              // [2][8192] elems
  // W slabs at +36KB: the epilogue Cst overlay is 128*LDC*4 = 34,816B — it
  // OVERRUNS the 32KB A stages by 2KB (racecheck-caught: the overlay trampled
  // the prefetched W half's first chunk).
  bf16* Wsl = reinterpret_cast<bf16*>(base + 36864);      // [2][K/64][4096] elems
  float* Cst = reinterpret_cast<float*>(base);            // epilogue overlay
  const int tid = threadIdx.x;
  const int wg = tid / 128;
  const int wtid = tid % 128;
  const int nblocks = gridDim.x;
  const int n_tiles = N / BN;
  const int m_tiles = M / BM;
  const int nb = blockIdx.x % n_tiles;         // pinned n-block
  const int mgroup = blockIdx.x / n_tiles;     // this block walks m with this offset
  const int mstride = (nblocks + n_tiles - 1) / n_tiles;
  const int kc = K / BK;                       // k-chunks in the W slice

  auto load_W = [&](int l, int half) {  // stage the block's W slice [BN, K]
    const bf16* W = Ws[l] + (int64_t)nb * BN * K;  // W stored [N, K] row-major
    for (int v = tid; v < BN * K / 8; v += 256) {
      const int r = v / (K / 8), k8 = (v % (K / 8)) * 8;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(Wsl + half * kc * 4096 + (k8 / BK) * 4096) +
              koff_sw(r, k8 % BK),
          &W[(int64_t)r * K + k8], 16);
    }
    __pipeline_commit();
  };

  if (VARIANT >= 1) {  // preload layer 0's W (outside the timed region caller-side
    load_W(0, 0);      // warmup runs make this hot anyway)
    __pipeline_wait_prior(0);
    __syncthreads();
  }

  for (int l = 0; l < L; ++l) {
    const bf16* A = (l == 0) ? X0 : Ys[l - 1];
    bf16* Y = Ys[l];
    const bf16* W = Ws[l] + (int64_t)nb * BN * K;
    const int half = (VARIANT == 2) ? (l & 1) : 0;
    bool prefetched = false;

    for (int mt = mgroup; mt < m_tiles; mt += mstride) {
      const int m0 = mt * BM;
      float d[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) d[i] = 0.0f;
      for (int t = 0; t < kc; ++t) {
        // A stage (ping-pong on t&1): 128r x 64k
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int v = tid + i * 256;
          const int r = v / 8, k8 = (v % 8) * 8;
          __pipeline_memcpy_async(
              reinterpret_cast<char*>(Ast + (t & 1) * 8192 + (r / 64) * 4096) +
                  koff_sw(r % 64, k8),
              &A[(int64_t)(m0 + r) * K + t * BK + k8], 16);
        }
        if (VARIANT == 0) {  // stream W k-chunk too (into slab half 0, chunk slot t&1)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int v = tid + i * 256;
            const int r = v / 8, k8 = (v % 8) * 8;
            __pipeline_memcpy_async(
                reinterpret_cast<char*>(Wsl + (t & 1) * 4096) + koff_sw(r, k8),
                &W[(int64_t)r * K + t * BK + k8], 16);
          }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        const bf16* wsrc = (VARIANT == 0) ? Wsl + (t & 1) * 4096
                                          : Wsl + half * kc * 4096 + t * 4096;
        uint64_t da[4], db[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          da[s] = desc_k_sw(Ast + (t & 1) * 8192 + wg * 4096, s);
          db[s] = desc_k_sw(wsrc, s);
        }
        cute::warpgroup_arrive();
#pragma unroll
        for (int s = 0; s < 4; ++s)
          SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>::fma(
              da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
              d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
              d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
              d[30], d[31], SG::ScaleOut::One);
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<0>();
        __syncthreads();
      }
      // after the first m-tile's mainloop, prefetch next layer's W (variant 2)
      if (VARIANT == 2 && !prefetched && l + 1 < L) {
        load_W(l + 1, (l + 1) & 1);  // joins the pipeline; drained by next waits
        prefetched = true;
      }
      // epilogue: stage + coalesced store (bf16), then A-stage smem is reused —
      // NOTE Cst overlays the A stages only (W slabs live above 32KB)
      const int w = wtid / 32, ln = wtid % 32;
      {
        const int r = wg * 64 + w * 16 + ln / 4;
        const int cb = (ln % 4) * 2;
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
          for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
              Cst[(r + 8 * i) * LDC + n8 * 8 + cb + j] = d[n8 * 4 + i * 2 + j];
      }
      __syncthreads();
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int gid = tid + g * 256;
        const int m = gid / 8, c8 = (gid % 8) * 8;
        uint4 out_v;
        bf16* oe = reinterpret_cast<bf16*>(&out_v);
#pragma unroll
        for (int e = 0; e < 8; ++e) oe[e] = __float2bfloat16(Cst[m * LDC + c8 + e]);
        *reinterpret_cast<uint4*>(&Y[(int64_t)(m0 + m) * N + nb * BN + c8]) = out_v;
      }
      __syncthreads();
    }
    // chain dependency: everyone waits for layer l to fully finish
    layer_barrier(bar, l, nblocks, tid);
    if (VARIANT == 1 && l + 1 < L) {  // resident, no overlap: reload after barrier
      load_W(l + 1, 0);
      __pipeline_wait_prior(0);
      __syncthreads();
    }
  }
}

void run_wsta(torch::Tensor X0, torch::Tensor Yptrs, torch::Tensor Wptrs, int64_t L,
              int64_t M, int64_t N, int64_t K, torch::Tensor bar, int64_t variant,
              int64_t nblocks) {
  const bf16* x0 = reinterpret_cast<const bf16*>(X0.data_ptr());
  bf16* const* ys = reinterpret_cast<bf16* const*>(Yptrs.data_ptr<int64_t>());
  const bf16* const* ws = reinterpret_cast<const bf16* const*>(Wptrs.data_ptr<int64_t>());
  int* b = bar.data_ptr<int>();
  const int kc = (int)K / BK;
  const int smem = 1024 + 36864 + 2 * kc * 8192;  // align + A/Cst region + 2 W slabs
  TORCH_CHECK(smem <= 227 * 1024, "smem over budget");
  auto launch = [&](auto kern) {
    cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<(int)nblocks, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        x0, ys, ws, (int)L, (int)M, (int)N, (int)K, b);
  };
  switch (variant) {
    case 0: launch(wsta_chain<0>); break;
    case 1: launch(wsta_chain<1>); break;
    case 2: launch(wsta_chain<2>); break;
    default: TORCH_CHECK(false, "bad variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

cpp_src = "void run_wsta(torch::Tensor X0, torch::Tensor Yptrs, torch::Tensor Wptrs, int64_t L, int64_t M, int64_t N, int64_t K, torch::Tensor bar, int64_t variant, int64_t nblocks);"

VAR = {0: "stream-both", 1: "W-resident", 2: "resident+prefetch"}


def main():
    torch.cuda.set_device(0)
    ext = load_inline(
        name="wsta_probe",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["run_wsta"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-I{CUTE_INC}", "--expt-relaxed-constexpr"],
        extra_include_paths=[CUTE_INC],
        verbose=False,
    )
    print("built ok")
    NB = 132
    # (label, L, M, N, K): square N==K chains so y_l feeds x_{l+1} directly
    shapes = [
        ("nano-ish  M512  NK256", 16, 512, 256, 256),
        ("small-ish M1024 NK512", 16, 1024, 512, 512),
    ]
    for label, L, M, N, K in shapes:
        torch.manual_seed(0)
        X0 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * (1.0 / K) for _ in range(L)]
        Ys = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(L)]
        Wp = torch.tensor([t.data_ptr() for t in Ws], dtype=torch.int64, device="cuda")
        Yp = torch.tensor([t.data_ptr() for t in Ys], dtype=torch.int64, device="cuda")
        bar = torch.zeros(L, dtype=torch.int32, device="cuda")

        # reference for the last layer
        ref = X0.float()
        for l in range(L):
            ref = ref @ Ws[l].float().T
            ref = torch.from_numpy(ref.to(torch.bfloat16).float().cpu().numpy()).cuda()  # round like the chain
        results = {}
        for v in (0, 1, 2):
            bar.zero_()
            ext.run_wsta(X0, Yp, Wp, L, M, N, K, bar, v, NB)
            torch.cuda.synchronize()
            err = (Ys[-1].float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item() + 1e-3
            assert err / scale < 3e-2, f"{VAR[v]} mismatch rel={err / scale}"
            times = []
            for _ in range(30):
                bar.zero_()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                ext.run_wsta(X0, Yp, Wp, L, M, N, K, bar, v, NB)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) * 1e3)
            results[v] = statistics.median(times)
        base = results[0]
        row = f"{label}  L={L}: "
        for v in (0, 1, 2):
            row += f"{VAR[v]} {results[v]:7.1f}us ({base / results[v]:4.2f}x)   "
        print(row)


if __name__ == "__main__":
    main()
