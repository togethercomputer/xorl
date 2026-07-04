// wspec_probe: warp-specialized paged-smem persistent-interpreter probe (mkv3 Phase 2).
//
// Measures per-hop latency of a serial chain of wgmma GEMM instructions
// (C_i = C_{i-1} @ B^T, 128x64x64 bf16 NT, one tile per instr) under:
//   mode 0: flat 256-thread block (milestone A — re-baseline of the interpreter path)
//   mode 1: 384-thread warp-specialized, volatile smem flags   (milestone B)
//   mode 2: 384-thread warp-specialized, mbarrier handoff      (milestone C)
//   mode 3: mode 1 + setmaxnreg 40/224                         (milestone D)
//   mode 4: mode 2 + setmaxnreg 40/224                         (milestone D)
// NN template variant: B stored [K,N] N-contiguous, MN-major descriptor (milestone E).
//
// wgmma tile structure copied from ops.cuh op_gemm_wgmma (validated by wgmma_probe.py):
// no-swizzle INTER K-major descriptors, m64n64k16 SS, branch-free ScaleOut::One over
// zeroed accumulators, smem-staged coalesced epilogue.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

#define WG_BM 128
#define WG_BN 64
#define WG_BK 64
#define WG_LDC 68

struct Page {
  union {
    struct {
      bf16 A[2][2][4][1024];  // [stage][row-half][k16-step][64x16 INTER block] = 32KB
      bf16 B[2][4][1024];     // [stage][k16-step][64x16 INTER block]           = 16KB
    } t;
    float Cs[WG_BM * WG_LDC];  // epilogue staging overlays the dead tile buffers
  };
};
static_assert(sizeof(Page) == 49152, "page must be 48KB");

// ---- descriptor recipes (ops.cuh / wgmma_probe.py) -----------------------------------
__device__ __forceinline__ int wg_koff(int r, int k) {  // bytes within a 64-row block
  return ((r >> 3) << 8) + ((k >> 3) << 7) + ((r & 7) << 4) + ((k & 7) << 1);
}
__device__ __forceinline__ int wg_mnoff(int mn, int k) {
  return ((mn >> 3) << 7) + ((k >> 3) << 10) + ((mn & 7) << 1) + ((k & 7) << 4);
}
__device__ __forceinline__ uint64_t wg_desc(const void* smem_ptr) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (128 >> 4);
  d.bitfield.stride_byte_offset_ = (256 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}
__device__ __forceinline__ uint64_t wg_desc_mn(const void* smem_ptr) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (1024 >> 4);
  d.bitfield.stride_byte_offset_ = (128 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

template <class MMA>
__device__ __forceinline__ void wg_mma_ktile(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                             float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
             d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
             d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
             cute::SM90::GMMA::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// ---- sync primitives -----------------------------------------------------------------
__device__ __forceinline__ int ld_acq(const int* p) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void st_rel(int* p, int v) {
  asm volatile("st.global.release.gpu.b32 [%0], %1;" ::"l"(p), "r"(v));
}
__device__ __forceinline__ void bar_sync(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t cnt) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  uint64_t tok;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(tok) : "r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  uint32_t rdy = 0;
  while (!rdy)
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(rdy)
        : "r"(a), "r"(parity));
}

// ---- tile building blocks --------------------------------------------------------------
// Fill one cp.async stage: A 128x64 (K-major) + B 64x64 (NT: K-major / NN: N-contiguous
// -> MN-major blocks). ft in [0, nft): the caller's lane within the fill group.
__device__ __forceinline__ void fill_a(Page& pg, const bf16* __restrict__ A, int ft,
                                       int nft, int K, int k0, int st) {
  for (int v = ft; v < 1024; v += nft) {  // A: 1024 16B vectors
    const int r = v >> 3, k8 = (v & 7) << 3;
    __pipeline_memcpy_async(
        reinterpret_cast<char*>(pg.t.A[st][r >> 6][k8 >> 4]) + wg_koff(r & 63, k8 & 15),
        &A[(int64_t)r * K + k0 + k8], 16);
  }
}

template <bool NN>
__device__ __forceinline__ void fill_b(Page& pg, const bf16* __restrict__ B, int ft,
                                       int nft, int K, int k0, int st) {
  if (!NN) {
    for (int v = ft; v < 512; v += nft) {  // B[N,K] K-contiguous: 512 16B vectors
      const int r = v >> 3, k8 = (v & 7) << 3;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(pg.t.B[st][k8 >> 4]) + wg_koff(r, k8 & 15),
          &B[(int64_t)r * K + k0 + k8], 16);
    }
  } else {
    for (int v = ft; v < 512; v += nft) {  // B[K,N] N-contiguous -> MN-major blocks
      const int k = v >> 3, n8 = (v & 7) << 3;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(pg.t.B[st][k >> 4]) + wg_mnoff(n8, k & 15),
          &B[(int64_t)(k0 + k) * WG_BN + n8], 16);
    }
  }
}

template <bool NN>
__device__ __forceinline__ void fill_tiles(Page& pg, const bf16* __restrict__ A,
                                           const bf16* __restrict__ B, int ft, int nft,
                                           int K, int k0, int st) {
  fill_a(pg, A, ft, nft, K, k0, st);
  fill_b<NN>(pg, B, ft, nft, K, k0, st);
  __pipeline_commit();
}

template <bool NN>
__device__ __forceinline__ void mma_tile(Page& pg, int st, int wg, float (&d)[32]) {
  uint64_t da[4], db[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    da[s] = wg_desc(pg.t.A[st][wg][s]);
    db[s] = NN ? wg_desc_mn(pg.t.B[st][s]) : wg_desc(pg.t.B[st][s]);
  }
  if (NN)
    wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
  else
    wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
}

// accumulator staging + coalesced bf16 store, 256 threads (ctid in [0,256)).
__device__ __forceinline__ void epilogue_stage(Page& pg, const float (&d)[32], int ctid) {
  const int wg = ctid / 128, wtid = ctid % 128;
  const int w = wtid / 32, l = wtid % 32;
  const int r = wg * 64 + w * 16 + l / 4, cb = (l % 4) * 2;
#pragma unroll
  for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        pg.Cs[(r + 8 * i) * WG_LDC + n8 * 8 + cb + j] = d[n8 * 4 + i * 2 + j];
}
__device__ __forceinline__ void epilogue_store(Page& pg, bf16* __restrict__ C, int ctid) {
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const int gid = ctid + g * 256;
    const int m = gid / 8, c8 = (gid % 8) * 8;
    uint4 out;
    bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = __float2bfloat16(pg.Cs[m * WG_LDC + c8 + e]);
    *reinterpret_cast<uint4*>(&C[(int64_t)m * WG_BN + c8]) = out;
  }
}

// ---- milestone A: flat 256-thread persistent chain kernel -----------------------------
template <bool NN>
__global__ void chain_flat_k(const bf16* __restrict__ A0, const bf16* __restrict__ Bm,
                             const int64_t* __restrict__ cs, int* done, int* cursor,
                             int n, int K) {
  extern __shared__ char smem[];
  Page& pg = *reinterpret_cast<Page*>(smem);
  __shared__ int s_i;
  const int tid = threadIdx.x;
  const int wg = tid / 128;
  for (;;) {
    if (tid == 0) s_i = atomicAdd(cursor, 1);
    __syncthreads();
    const int i = s_i;
    if (i >= n) return;
    // issue the pointer-array loads before the poll barrier so their L2 round-trip
    // overlaps the wait instead of delaying the cp.async prologue
    const bf16* A = (i == 0) ? A0 : reinterpret_cast<const bf16*>(cs[i - 1]);
    bf16* C = reinterpret_cast<bf16*>(cs[i]);
    if (tid == 0 && i > 0)
      while (ld_acq(&done[(i - 1) * 32]) == 0) __nanosleep(64);
    __syncthreads();
    float d[32];
#pragma unroll
    for (int q = 0; q < 32; ++q) d[q] = 0.0f;
    const int iters = K / WG_BK;
    fill_tiles<NN>(pg, A, Bm, tid, 256, K, 0, 0);
    for (int t = 0; t < iters; ++t) {  // 2-stage ping-pong (degenerate at K=64)
      if (t + 1 < iters) fill_tiles<NN>(pg, A, Bm, tid, 256, K, (t + 1) * WG_BK, (t + 1) & 1);
      __pipeline_wait_prior(t + 1 < iters ? 1 : 0);
      __syncthreads();
      mma_tile<NN>(pg, t & 1, wg, d);
      __syncthreads();
    }
    epilogue_stage(pg, d, tid);
    __syncthreads();
    epilogue_store(pg, C, tid);
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      st_rel(&done[i * 32], 1);
    }
    __syncthreads();  // protect s_i before the next claim
  }
}

// ---- milestones B/C/D: warp-specialized paged kernel -----------------------------------
// 384 threads. WG0 = threads 0-127: warps 0,2,3 (96 threads) front-end (claim ->
// page-reuse wait -> predecessor poll -> cp.async fill -> full[p]); warp 1 lane 0 =
// completion watcher (consumed[p] -> threadfence -> st.release done[i] -> freed[p]).
// WG1+2 = threads 128-383: consumers (wgmma + epilogue), consumer-only named barrier 1.
// Handoff flags are per-page monotone sequence numbers (u = per-page use count).
// SYNCM 0: volatile smem flags; SYNCM 1: mbarriers for full/consumed.
template <int SYNCM, bool REGCTL, bool NN>
__global__ void __launch_bounds__(384, 1)
    chain_ws_k(const bf16* __restrict__ A0, const bf16* __restrict__ Bm,
               const int64_t* __restrict__ cs, int* done, int* cursor, int n, int K) {
  extern __shared__ char smem[];
  Page* pages = reinterpret_cast<Page*>(smem);
  __shared__ volatile int s_pi;           // claim broadcast within the fill group
  __shared__ volatile int s_instr[2];     // instr id staged per page (-1 = quit)
  __shared__ volatile int s_full[2];      // seq: page filled       (SYNCM 0)
  __shared__ volatile int s_consumed[2];  // seq: consumers done    (SYNCM 0)
  __shared__ volatile int s_freed[2];     // seq: done published, page reusable
  __shared__ volatile int s_qi[4];        // watcher queue (instr ids, -1 = quit)
  __shared__ volatile int s_qtail;
  __shared__ __align__(8) uint64_t mb_full[2], mb_cons[2];

  const int tid = threadIdx.x;
  if (tid == 0) {
    s_instr[0] = s_instr[1] = 0;
    s_full[0] = s_full[1] = 0;
    s_consumed[0] = s_consumed[1] = 0;
    s_freed[0] = s_freed[1] = 0;
    s_qtail = 0;
    if (SYNCM == 1) {
      mbar_init(&mb_full[0], 1);
      mbar_init(&mb_full[1], 1);
      mbar_init(&mb_cons[0], 1);
      mbar_init(&mb_cons[1], 1);
    }
  }
  __syncthreads();  // the ONLY full-block barrier

  if (tid < 128) {  // ---------------- WG0: scheduler/producer ----------------
    if (REGCTL) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    const int warp = tid / 32;

    if (warp == 1) {  // completion watcher (lane 0)
      if (tid != 32) return;
      int t = 0;
      for (;;) {
        while (s_qtail <= t) __nanosleep(32);
        __threadfence_block();
        const int i = s_qi[t & 3];
        if (i < 0) return;
        const int p = t & 1, u = (t >> 1) + 1;
        if (SYNCM == 1)
          mbar_wait(&mb_cons[p], (u - 1) & 1);
        else
          while (s_consumed[p] < u) {
          }
        __threadfence();
        st_rel(&done[i * 32], 1);
        s_freed[p] = u;
        ++t;
      }
    }

    // fill front-end: warps 0,2,3 = 96 threads, named barrier 2
    const int lane = tid & 31;
    const int ft = (warp == 0) ? lane : (warp - 1) * 32 + lane;  // 0..95
    int t = 0;
    for (;;) {
      const int p = t & 1, u = (t >> 1) + 1;
      if (tid == 0) {
        s_pi = atomicAdd(cursor, 1);
        while (s_freed[p] < u - 1) __nanosleep(32);  // page's previous use retired
      }
      bar_sync(2, 96);
      const int i = s_pi;
      if (i >= n) {
        if (tid == 0) {
          s_instr[p] = -1;
          __threadfence_block();
          if (SYNCM == 1)
            mbar_arrive(&mb_full[p]);
          else
            s_full[p] = u;
          const int qt = s_qtail;  // quit marker for the watcher
          s_qi[qt & 3] = -1;
          __threadfence_block();
          s_qtail = qt + 1;
        }
        return;
      }
      // A-pointer load issued up front: its L2 round-trip overlaps fill_b + the poll
      const bf16* A = (i == 0) ? A0 : reinterpret_cast<const bf16*>(cs[i - 1]);
      // B-prefetch: B is chain-invariant, so its cp.async overlaps the predecessor poll
      // (page reuse is safe: the claim bar_sync above ordered tid0's freed-wait, folded
      // into the claim step, before any thread touches the page)
      fill_b<NN>(pages[p], Bm, ft, 96, K, 0, 0);
      __pipeline_commit();
      if (tid == 0) {
        if (i > 0)
          while (ld_acq(&done[(i - 1) * 32]) == 0) __nanosleep(64);
        s_instr[p] = i;
        __threadfence_block();
      }
      bar_sync(2, 96);  // predecessor done -> A readable
      fill_a(pages[p], A, ft, 96, K, 0, 0);
      __pipeline_commit();
      __pipeline_wait_prior(0);  // both commit groups landed
      bar_sync(2, 96);
      if (tid == 0) {
        __threadfence_block();
        if (SYNCM == 1)
          mbar_arrive(&mb_full[p]);
        else
          s_full[p] = u;
        const int qt = s_qtail;  // hand (i, p) to the watcher; page implicit in t order
        s_qi[qt & 3] = i;
        __threadfence_block();
        s_qtail = qt + 1;
      }
      ++t;
    }
  } else {  // ---------------- WG1+2: consumers ----------------
    if (REGCTL) asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
    const int ctid = tid - 128;
    const int wg = ctid / 128;
    int t = 0;
    for (;;) {
      const int p = t & 1, u = (t >> 1) + 1;
      if (SYNCM == 1)
        mbar_wait(&mb_full[p], (u - 1) & 1);
      else
        while (s_full[p] < u) {
        }
      __threadfence_block();
      const int i = s_instr[p];
      if (i < 0) return;
      // issue the output-pointer load now so its L2 round-trip hides under the mma
      bf16* C = reinterpret_cast<bf16*>(cs[i]);
      float d[32];
#pragma unroll
      for (int q = 0; q < 32; ++q) d[q] = 0.0f;
      mma_tile<NN>(pages[p], 0, wg, d);
      bar_sync(1, 256);  // both warpgroups done reading tiles before the overlay
      epilogue_stage(pages[p], d, ctid);
      bar_sync(1, 256);
      epilogue_store(pages[p], C, ctid);
      bar_sync(1, 256);
      // producer-side completion: consumers signal immediately; the __threadfence +
      // st.release for done[i] runs on the (otherwise idle) watcher warp instead.
      if (ctid == 0) {
        if (SYNCM == 1)
          mbar_arrive(&mb_cons[p]);
        else
          s_consumed[p] = u;
      }
      ++t;
    }
  }
}

// ---- host ------------------------------------------------------------------------------
#define SMEM_FLAT ((int)sizeof(Page))
#define SMEM_WS (2 * (int)sizeof(Page))

static void set_attr(const void* fn, int smem) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
}

void probe_init() {
  set_attr((const void*)chain_flat_k<false>, SMEM_FLAT);
  set_attr((const void*)chain_flat_k<true>, SMEM_FLAT);
  set_attr((const void*)chain_ws_k<0, false, false>, SMEM_WS);
  set_attr((const void*)chain_ws_k<1, false, false>, SMEM_WS);
  set_attr((const void*)chain_ws_k<0, true, false>, SMEM_WS);
  set_attr((const void*)chain_ws_k<1, true, false>, SMEM_WS);
  set_attr((const void*)chain_ws_k<0, false, true>, SMEM_WS);
  set_attr((const void*)chain_ws_k<1, false, true>, SMEM_WS);
  set_attr((const void*)chain_ws_k<0, true, true>, SMEM_WS);
  set_attr((const void*)chain_ws_k<1, true, true>, SMEM_WS);
}

// mode: 0 flat, 1 ws+volatile, 2 ws+mbar, 3 ws+volatile+setmaxnreg, 4 ws+mbar+setmaxnreg
void probe_run(int64_t mode, int64_t nn, torch::Tensor A0, torch::Tensor B, torch::Tensor cs,
               torch::Tensor done, torch::Tensor cursor, int64_t n, int64_t nblocks) {
  TORCH_CHECK(A0.is_cuda() && A0.dtype() == torch::kBFloat16);
  TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kBFloat16);
  TORCH_CHECK(cs.is_cuda() && cs.dtype() == torch::kInt64);
  TORCH_CHECK(done.is_cuda() && done.dtype() == torch::kInt32 && done.numel() >= n * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(done.data_ptr(), 0, (size_t)n * 32 * 4, stream));
  C10_CUDA_CHECK(cudaMemsetAsync(cursor.data_ptr(), 0, 4, stream));
  const bf16* a0 = reinterpret_cast<const bf16*>(A0.data_ptr());
  const bf16* b = reinterpret_cast<const bf16*>(B.data_ptr());
  const int64_t* csp = cs.data_ptr<int64_t>();
  int* dn = done.data_ptr<int>();
  int* cur = cursor.data_ptr<int>();
  const int N = (int)n, K = 64, G = (int)nblocks;
#define WS_LAUNCH(S, R, T) \
  chain_ws_k<S, R, T><<<G, 384, SMEM_WS, stream>>>(a0, b, csp, dn, cur, N, K)
  if (mode == 0) {
    if (nn)
      chain_flat_k<true><<<G, 256, SMEM_FLAT, stream>>>(a0, b, csp, dn, cur, N, K);
    else
      chain_flat_k<false><<<G, 256, SMEM_FLAT, stream>>>(a0, b, csp, dn, cur, N, K);
  } else if (mode == 1) {
    if (nn) WS_LAUNCH(0, false, true); else WS_LAUNCH(0, false, false);
  } else if (mode == 2) {
    if (nn) WS_LAUNCH(1, false, true); else WS_LAUNCH(1, false, false);
  } else if (mode == 3) {
    if (nn) WS_LAUNCH(0, true, true); else WS_LAUNCH(0, true, false);
  } else if (mode == 4) {
    if (nn) WS_LAUNCH(1, true, true); else WS_LAUNCH(1, true, false);
  } else {
    TORCH_CHECK(false, "bad mode");
  }
#undef WS_LAUNCH
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &probe_init, "set smem attributes");
  m.def("run", &probe_run, "run one chain");
}
