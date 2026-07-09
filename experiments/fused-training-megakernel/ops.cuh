// Device op library for the fused training megakernel.
//
// Every op is a __device__ function taking (Instr, tile, bufs, smem): `tile` is the
// block-level work item index and the WHOLE block (256 threads) executes it.
// Ops read operand buffer indices + shape ints from Instr.args.
//
// Conventions: activations/params bf16 row-major; every accumulation fp32; weight
// grads fp32. GEMM tiles are 64x64; batched row ops use MK_ROW_R rows per work item
// (one warp per row); CE/embed row ops use one row per work item.

#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

// Ops are executed by a 256-thread consumer group. Under the wave/df/df2 executors that
// group IS the whole block, so bar.sync 1,256 is exactly __syncthreads and
// MK_CONSUMERS equals the block width. Under megakernel_ws the block is 288 threads (warps 0-7 =
// consumers with unchanged mk_tid() semantics, warp 8 = scheduler): op code must
// NEVER use __syncthreads (the scheduler warp does not participate — instant hang) nor
// MK_CONSUMERS (wrong stride). consumer_sync() is a named barrier counting exactly the
// 256 arriving consumer threads.
#define MK_CONSUMERS 256
// mk_tid(): the op-local thread index. Ops are written against a 256-thread group;
// under the dual executor (v3 P4b round 3) a block carries TWO such groups (fat
// half threads 0-255, lean half 256-511), so ops index mk_tid() and sync on a
// half-specific named barrier. For the 256-thread executors both collapse to the
// old mk_tid() / bar.sync 1 exactly.
__device__ __forceinline__ int mk_tid() { return threadIdx.x & (MK_CONSUMERS - 1); }
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync %0, 256;" ::"r"(1 + (threadIdx.x >> 8)) : "memory");
}

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float lmhead_exp(float x) {
#ifdef MK_LMHEAD_EXP2_APPROX
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
#else
  return __expf(x);
#endif
}
__device__ __forceinline__ float ce_exp(float x) {
#ifdef MK_CE_EXP2_APPROX
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
#else
  return expf(x);
#endif
}
__device__ __forceinline__ float ce_bwd_exp(float x) {
#ifdef MK_CE_BWD_EXP2_APPROX
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
#else
  return expf(x);
#endif
}

// ---- trivial ops (skeleton validation) ------------------------------------------------

#define MK_CHUNK 16384  // elements per fill/cvt work item (mirrored in mk.py)

// args: {buf, n, value_bits}; tile = MK_CHUNK-element chunk index.
__device__ __forceinline__ void op_fill_f32(const Instr& I, int tile, void** bufs) {
  float* p = reinterpret_cast<float*>(bufs[I.args[0]]);
  const int n = I.args[1];
  const float v = __int_as_float(I.args[2]);
  const int base = tile * MK_CHUNK, end = min(base + MK_CHUNK, n);
  // vectorized main body + scalar tail (base is 16B-aligned: MK_CHUNK % 4 == 0)
  float4* p4 = reinterpret_cast<float4*>(p + base);
  const int quads = (end - base) / 4;
  const float4 v4 = make_float4(v, v, v, v);
  for (int i = mk_tid(); i < quads; i += MK_CONSUMERS) p4[i] = v4;
  for (int i = base + quads * 4 + mk_tid(); i < end; i += MK_CONSUMERS) p[i] = v;
}

// args: {y, x, n, alpha_bits}; tile = 4096-element chunk index. y += alpha * x
__device__ __forceinline__ void op_axpy_f32(const Instr& I, int tile, void** bufs) {
  float* y = reinterpret_cast<float*>(bufs[I.args[0]]);
  const float* x = reinterpret_cast<const float*>(bufs[I.args[1]]);
  const int n = I.args[2];
  const float a = __int_as_float(I.args[3]);
  const int base = tile * 4096;
  for (int i = base + mk_tid(); i < min(base + 4096, n); i += MK_CONSUMERS)
    y[i] += a * x[i];
}

#ifdef MK_HEAD_DX_SKR
// args: {ws, out, n, ks}; tile = 4096-element chunk index. out[i] = sum over ks
// per-K-slice fp32 partial slabs (round-12 SKR: the split gemm stores plain
// per-slice partials; this separate reduce replaces the fp32-atomic epilogue).
__device__ __forceinline__ void op_skr_reduce(const Instr& I, int tile, void** bufs) {
  const float* ws = reinterpret_cast<const float*>(bufs[I.args[0]]);
  float* out = reinterpret_cast<float*>(bufs[I.args[1]]);
  const int n = I.args[2];
  const int ks = I.args[3];
  const int base = tile * 4096, end = min(base + 4096, n);
  const int quads = (end - base) / 4;  // n is a whole fp32 matrix: n % 4 == 0
  const float4* w4 = reinterpret_cast<const float4*>(ws + base);
  float4* o4 = reinterpret_cast<float4*>(out + base);
  const long ns4 = (long)n / 4;  // slab stride in float4s
  for (int i = mk_tid(); i < quads; i += MK_CONSUMERS) {
    float4 acc = w4[i];
    for (int k = 1; k < ks; ++k) {
      const float4 p = w4[(long)k * ns4 + i];
      acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
    }
    o4[i] = acc;
  }
}
#endif

// ---- GEMM -----------------------------------------------------------------------------
// C[M,N] (+)= A[M,K] @ B[K,N].
//   flags bit0: A stored [K,M] row-major (use A^T)
//   flags bit1: B stored [N,K] row-major (use B^T)
//   flags bit2: accumulate into existing C (C += result)
//   flags bit3: C is fp32 (else bf16)
//   flags bit4: add bf16 residual buffer (args[7], same shape as C)
//   flags bit5: split-K (requires fp32 C, pre-zeroed): args[8] = #slices, each slice
//               computes a K-range and accumulates via fp32 atomicAdd. Rescues SM
//               occupancy for small dW matrices (few M*N tiles, large K).
//   flags bit6 (MK_GEMM_DX_TMA_RED builds, with bit5): NN dX split-K slices stage
//               fp32 C rows to smem, then drain each full 64x128 tile with
//               cp.reduce.async.bulk.add.f32 instead of per-element atomics.
//   flags bit15 (MK_HEAD_DX_SKR builds, with bit5): split-K slices write plain fp32
//               partials to per-slice slabs at C + slice*M*N (no zero-fill, no
//               atomics); OP_SKR_REDUCE sums the slabs (round-12 SKR structure).
//   flags bit14: direct m64n256 WGMMA tile; qwen-specific paths for NT lm-head with
//                CE/LSE partials (bit11) and NN fp32 head-dX.
//   flags bit17: qwen final head-dX emits per-row RMS-dot partials for the first
//                final RMS dX consumer. args[9]=partials, 10=nparts, 11=X, 12=wf.
//   flags bit25: opt-in 3-stage operand ring for the direct m64n256 tile. This needs
//                the qwen 148KB launch page and is never emitted by generic routes.
//   flags bit26: opt-in N-major tile order for exact qwen n256 routes. This groups
//                adjacent M bands for a shared 256-column B tile.
// args: {A, B, C, M, N, K, flags, res, sk}
// tile id = (m_tile * n_tiles + n_tile) * sk + k_slice.

#define GEMM_BM 64
#define GEMM_BN 128
#define GEMM_BK 32
#define GEMM_LDA (GEMM_BK + 8)  // bf16 smem strides (pad: bank conflicts + wmma align)
#define GEMM_LDB (GEMM_BN + 8)
#define GEMM_LDC (GEMM_BN + 4)  // fp32 staging (wmma: fp32 ld must be a multiple of 4)
#define GEMM_DX_TMA_RED_FLAG 64

struct GemmSmem {
  bf16 As[GEMM_BM][GEMM_LDA];
  bf16 Bs[GEMM_BK][GEMM_LDB];
  float Cs[GEMM_BM][GEMM_LDC];
};

__device__ __forceinline__ uint4 ldg16(const bf16* p) {
  return *reinterpret_cast<const uint4*>(p);
}

#if defined(MK_GEMM_DX_TMA_RED) || defined(MK_DW_TN_TMA_RED)
__device__ __forceinline__ void gemm_dx_tma_red_drain_tile(float* C, const GemmSmem& S,
                                                           int N, int m0, int n0) {
  const int tid = mk_tid();
  if (tid < GEMM_BM) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0;" : "=l"(policy));
    const uint32_t src =
        static_cast<uint32_t>(__cvta_generic_to_shared(&S.Cs[tid][0]));
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32"
        " [%0], [%1], %2, %3;"
        :
        : "l"(C + (int64_t)(m0 + tid) * N + n0), "r"(src), "r"(GEMM_BN * 4),
          "l"(policy)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}
#endif

#if defined(MK_GEMM_DX_TMA_RED) || defined(MK_DW_TN_TMA_RED)
__device__ __forceinline__ void op_gemm(const Instr& I, int tile, void** bufs,
                                        char* smem_raw) {
#else
__device__ void op_gemm(const Instr& I, int tile, void** bufs, char* smem_raw) {
#endif
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  void* Cp = bufs[I.args[2]];
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  const bool a_t = flags & 1, b_t = flags & 2, acc_c = flags & 4, c_f32 = flags & 8;
  const bf16* Res = (flags & 16) ? reinterpret_cast<const bf16*>(bufs[I.args[7]]) : nullptr;

  GemmSmem& S = *reinterpret_cast<GemmSmem*>(smem_raw);
  const int sk = (flags & 32) ? I.args[8] : 1;
  const int slice = tile % sk;
  const int mn = tile / sk;
  const int n_tiles = (N + GEMM_BN - 1) / GEMM_BN;
  const int m0 = (mn / n_tiles) * GEMM_BM;
  const int n0 = (mn % n_tiles) * GEMM_BN;
  // this slice's K range (BK-aligned chunks)
  const int kchunk = ((K + sk * GEMM_BK - 1) / (sk * GEMM_BK)) * GEMM_BK;
  const int k_lo = slice * kchunk;
  const int k_hi = min(K, k_lo + kchunk);
  const int tid = mk_tid();
  if (k_lo >= K) return;
  // Fast path: whole tile in bounds and every vectorized load 16B-aligned.
  const bool fast = (m0 + GEMM_BM <= M) && (n0 + GEMM_BN <= N) && (K % 8 == 0) &&
                    (M % 8 == 0) && (N % 8 == 0);
#if defined(MK_GEMM_DX_TMA_RED) || defined(MK_DW_TN_TMA_RED)
  bool tma_red = false;
#ifdef MK_GEMM_DX_TMA_RED
  tma_red = tma_red || ((flags & GEMM_DX_TMA_RED_FLAG) && (flags & 32) && c_f32 &&
                        !a_t && !b_t && !Res && (m0 + GEMM_BM <= M) &&
                        (n0 + GEMM_BN <= N));
#endif
#ifdef MK_DW_TN_TMA_RED
  tma_red = tma_red || ((flags & 32) && c_f32 && !acc_c && a_t && !b_t && !Res &&
                        (m0 + GEMM_BM <= M) && (n0 + GEMM_BN <= N));
#endif
#endif

  // 8 warps as 2(m) x 4(n): each computes a 32x32 warp tile = 2x2 wmma frags.
  const int warp = tid / 32;
  const int wm = warp / 4, wn = warp % 4;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(c_frag[i][j], 0.0f);

  // Each thread owns 1 A-vector + 2 B-vectors (uint4 = 8 bf16) per K-tile. Global loads
  // for tile t+1 are issued BEFORE the mma over tile t (register prefetch), hiding
  // global latency behind tensor-core work without doubling smem.
  const int a_m = tid / 4, a_k = (tid % 4) * 8;      // !a_t coords
  const int at_k = tid / 8, at_m = (tid % 8) * 8;    // a_t coords
  const int b_k0 = tid / 16, b_n0 = (tid % 16) * 8;  // !b_t coords (vector 1)
  const int b_k1 = (tid + 256) / 16, b_n1 = ((tid + 256) % 16) * 8;
  const int bt_n0 = tid / 4, bt_k0 = (tid % 4) * 8;  // b_t coords
  const int bt_n1 = (tid + 256) / 4, bt_k1 = ((tid + 256) % 4) * 8;

  auto issue_loads = [&](int k0, uint4& pa, uint4& pb0, uint4& pb1) {
    pa = a_t ? ldg16(&A[(int64_t)(k0 + at_k) * M + m0 + at_m])
             : ldg16(&A[(int64_t)(m0 + a_m) * K + k0 + a_k]);
    pb0 = b_t ? ldg16(&B[(int64_t)(n0 + bt_n0) * K + k0 + bt_k0])
              : ldg16(&B[(int64_t)(k0 + b_k0) * N + n0 + b_n0]);
    pb1 = b_t ? ldg16(&B[(int64_t)(n0 + bt_n1) * K + k0 + bt_k1])
              : ldg16(&B[(int64_t)(k0 + b_k1) * N + n0 + b_n1]);
  };
  auto stage_smem = [&](const uint4& pa, const uint4& pb0, const uint4& pb1) {
    if (!a_t) {
      *reinterpret_cast<uint4*>(&S.As[a_m][a_k]) = pa;
    } else {
      const bf16* e = reinterpret_cast<const bf16*>(&pa);
#pragma unroll
      for (int j = 0; j < 8; ++j) S.As[at_m + j][at_k] = e[j];
    }
    if (!b_t) {
      *reinterpret_cast<uint4*>(&S.Bs[b_k0][b_n0]) = pb0;
      *reinterpret_cast<uint4*>(&S.Bs[b_k1][b_n1]) = pb1;
    } else {
      const bf16* e0 = reinterpret_cast<const bf16*>(&pb0);
      const bf16* e1 = reinterpret_cast<const bf16*>(&pb1);
#pragma unroll
      for (int j = 0; j < 8; ++j) S.Bs[bt_k0 + j][bt_n0] = e0[j];
#pragma unroll
      for (int j = 0; j < 8; ++j) S.Bs[bt_k1 + j][bt_n1] = e1[j];
    }
  };
  auto mma_tile = [&]() {
#pragma unroll
    for (int kk = 0; kk < GEMM_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b_frag[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(a_frag[i], &S.As[wm * 32 + i * 16][kk], GEMM_LDA);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(b_frag[j], &S.Bs[kk][wn * 32 + j * 16], GEMM_LDB);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
    }
  };
  auto load_slow = [&](int k0) {
    for (int i = tid; i < GEMM_BM * GEMM_BK; i += MK_CONSUMERS) {
      const int m = i / GEMM_BK, k = i % GEMM_BK;
      const int gm = m0 + m, gk = k0 + k;
      bf16 v = f2bf(0.0f);
      if (gm < M && gk < K) v = a_t ? A[(int64_t)gk * M + gm] : A[(int64_t)gm * K + gk];
      S.As[m][k] = v;
    }
    for (int i = tid; i < GEMM_BK * GEMM_BN; i += MK_CONSUMERS) {
      const int k = i / GEMM_BN, n = i % GEMM_BN;
      const int gk = k0 + k, gn = n0 + n;
      bf16 v = f2bf(0.0f);
      if (gk < K && gn < N) v = b_t ? B[(int64_t)gn * K + gk] : B[(int64_t)gk * N + gn];
      S.Bs[k][n] = v;
    }
  };

  const int span = k_hi - k_lo;
  const int k_fast_end = fast ? k_lo + (span / GEMM_BK) * GEMM_BK : k_lo;
  if (k_fast_end > k_lo) {  // pipelined fast iterations
    uint4 pa, pb0, pb1;
    issue_loads(k_lo, pa, pb0, pb1);
    for (int k0 = k_lo; k0 < k_fast_end; k0 += GEMM_BK) {
      stage_smem(pa, pb0, pb1);
      const int k_next = k0 + GEMM_BK;
      consumer_sync();  // smem staged for everyone
      if (k_next < k_fast_end) issue_loads(k_next, pa, pb0, pb1);  // overlap with mma
      mma_tile();
      consumer_sync();  // everyone done reading smem before next stage
    }
  }
  for (int k0 = k_fast_end; k0 < k_hi; k0 += GEMM_BK) {  // guarded tail
    load_slow(k0);
    consumer_sync();
    mma_tile();
    consumer_sync();
  }

  // stage fp32 result, then epilogue with bounds guards
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&S.Cs[wm * 32 + i * 16][wn * 32 + j * 16], c_frag[i][j],
                              GEMM_LDC, wmma::mem_row_major);
#if defined(MK_GEMM_DX_TMA_RED) || defined(MK_DW_TN_TMA_RED)
  if (tma_red) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
  consumer_sync();

#if defined(MK_GEMM_DX_TMA_RED) || defined(MK_DW_TN_TMA_RED)
  if (tma_red) {
    gemm_dx_tma_red_drain_tile(reinterpret_cast<float*>(Cp), S, N, m0, n0);
    consumer_sync();
    return;
  }
#endif

  for (int i = tid; i < GEMM_BM * GEMM_BN; i += MK_CONSUMERS) {
    const int m = i / GEMM_BN, n = i % GEMM_BN;
    const int gm = m0 + m, gn = n0 + n;
    if (gm >= M || gn >= N) continue;
    float v = S.Cs[m][n];
    const int64_t idx = (int64_t)gm * N + gn;
    if (Res) v += bf2f(Res[idx]);
    if (flags & 32) {  // split-K: multiple slices accumulate concurrently
      atomicAdd(&reinterpret_cast<float*>(Cp)[idx], v);
    } else if (c_f32) {
      float* C = reinterpret_cast<float*>(Cp);
      C[idx] = acc_c ? C[idx] + v : v;
    } else {
      bf16* C = reinterpret_cast<bf16*>(Cp);
      C[idx] = f2bf(acc_c ? bf2f(C[idx]) + v : v);
    }
  }

  // Fused Drow epilogue (flags bit10, dOatt = dX @ Wo): drow[qh, s] += sum_d dO*O over
  // this tile's columns (fp32 atomics; drow pre-zeroed). Replaces OP_ATTN_DPRE — one
  // chain hop per layer. D divides GEMM_BN, so a tile covers whole heads. Values are
  // re-rounded through bf16 to match what the standalone op read from the dOatt buffer.
  // args: 9 = oatt, 10 = drow, 11 = D. Host guarantees no residual/split-K/acc co-use.
  if (flags & 1024) {
    const bf16* Oatt = reinterpret_cast<const bf16*>(bufs[I.args[9]]);
    float* drow = reinterpret_cast<float*>(bufs[I.args[10]]);
    const int D = I.args[11];
    const int lane = tid % 32;
    for (int m = warp; m < GEMM_BM; m += 8) {  // warp per row, striding
      const int gm = m0 + m;
      if (gm >= M) break;
      for (int hb = 0; hb < GEMM_BN && n0 + hb < N; hb += D) {
        float s = 0.0f;
        for (int d = lane; d < D; d += 32)
          s += bf2f(f2bf(S.Cs[m][hb + d])) * bf2f(Oatt[(int64_t)gm * N + n0 + hb + d]);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) atomicAdd(&drow[(int64_t)((n0 + hb) / D) * M + gm], s);
      }
    }
  }
}

// ---- wgmma GEMM (Hopper): NT layouts (A [M,K], B [N,K], both K-major) ---------------
// Tile 128x64: two warpgroups own 64-row halves of the SAME 64 output columns
// (B loads shared), m64n64k16 wgmma = 32 fp32 accumulators/thread so the interpreter
// keeps 2 blocks/SM. 2-stage cp.async feeds; epilogue stages through smem for fully
// coalesced vectorized stores. Descriptor layout = the no-swizzle INTER K-major
// arrangement validated by wgmma_probe.py. flags bit7 selects this path
// (host guarantees M%128==0, N%64==0, K%64==0).

#define WG_BM 128
#define WG_BN 64
#define WG_BK 64
#define GEMM_N256_STAGE3_FLAG (1 << 25)
#define GEMM_N256_NMAJOR_FLAG (1 << 26)
#define GEMM_N256_NT_SUPERTILE_FLAG (1 << 27)
#define GEMM_HEADDX_RMSDOT_FLAG (1 << 17)

template <int STAGES>
struct WgmmaSmemT {
  bf16 A[STAGES][2][4][1024];  // [stage][row-half][k16-step][64x16 block]
  bf16 B[STAGES][4][1024];     // [stage][k16-step][64x16 block]
};
using WgmmaSmem = WgmmaSmemT<2>;
template <int STAGES>
struct WgmmaSmemN128T {
  bf16 A[STAGES][2][4][1024];  // [stage][row-half][k16-step][64x64 SW128 slab]
  bf16 B[STAGES][8192];        // [stage][128 rows x 64 k elts SW128]
};
using WgmmaSmemN128 = WgmmaSmemN128T<2>;
template <int STAGES>
struct WgmmaSmemN256T {
  bf16 A[STAGES][2][4][1024];  // per stage: A 16KB
  bf16 B[STAGES][16384];       // per stage: B 32KB
};
using WgmmaSmemN256 = WgmmaSmemN256T<2>;
template <int STAGES>
struct WgmmaSmemN256NtSupertileT {
  bf16 A[STAGES][4][4096];  // per stage: two 128-row A strips, 32KB
  bf16 B[STAGES][8192];     // per stage: one shared 128-col NT B slab, 16KB
};
// epilogue staging overlays the (dead-by-then) stage buffers: 128 x 68 fp32 = 34.8KB
// (n128: 128 x 128 fp32 = 64KB over the 64KB n128 stages)
#define WG_LDC 68
#define WG_LDC_N128 128

__device__ __forceinline__ int wg_koff(int r, int k) {  // bytes within a 64-row block
  return ((r >> 3) << 8) + ((k >> 3) << 7) + ((r & 7) << 4) + ((k & 7) << 1);
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

// MN-major INTER arrangement for operands stored MN-contiguous (NN's B, TN's A):
// canonical ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)) -> SBO = 128B mn-group stride,
// LBO = 1024B k-group stride for our 64-row step blocks. Validated by wgmma_probe.py.
__device__ __forceinline__ int wg_mnoff(int mn, int k) {  // bytes in a 64-row block
  return ((mn >> 3) << 7) + ((k >> 3) << 10) + ((mn & 7) << 1) + ((k & 7) << 4);
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

// ---- SW128 (128B-swizzle) canonical layouts (v3 P4b) ----------------------------------
// The INTER arrangements above have an 8-way smem WRITE bank conflict: a warp's 16B
// cp.async stores hit only 4 bank-quads (bank index depends only on r&7), costing
// ~1.1us per 24KB stage — THE gemm limiter (probe: pipe_probe.py; 40-79TF -> 60-150TF
// from the swizzle alone, pipeline depth immaterial). SW128 spreads each 8-lane phase
// across all 32 banks. The swizzle phase is derived from ABSOLUTE smem address bits
// [7,10): slab bases MUST be 1024B-aligned (op_gemm_wgmma aligns its smem base;
// misalignment = silent garbage) and the store-side XOR uses the slab-relative row,
// identical mod 8. K-major slab: [64 rows][64 k-elts]; 128B row = 8 x 16B chunks;
// chunk c of row r stored at c ^ (r&7). MN-major slab: [64 k-rows][64 mn-elts], roles
// flipped. Attention keeps INTER: its one-arrangement-both-majors descriptor-swap
// trick has no SW128 analogue (the XOR breaks the symmetry).
__device__ __forceinline__ int wg_koff_sw(int r, int k8) {  // bytes; k8 = k in elts (x8)
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}
__device__ __forceinline__ int wg_mnoff_sw(int k, int mn8) {
  return k * 128 + ((((mn8 >> 3) ^ (k & 7)) << 4));
}
// K-major SW128 descriptor for k16-atom s of a 64-row slab (deep_gemm recipe: B128,
// LBO=0, SBO=1024B; atom start = base + s*32B — mid-row advance is legal).
__device__ __forceinline__ uint64_t wg_desc_ksw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 32;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;  // B128
  return d.desc_;
}
// MN-major SW128 descriptor for a 128-wide operand (n128 NN tiles): two 64-mn
// slabs stacked 8KB apart; canonical B128 MN layout ((T,8,n),(8,k)) uses LBO as
// the 64-mn group stride. k16-atom s = 16 k-rows = 2KB step within each slab.
__device__ __forceinline__ uint64_t wg_desc_mnsw128(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 2048;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (8192 >> 4);  // 64-mn group (slab) stride
  d.bitfield.stride_byte_offset_ = (1024 >> 4);   // 8-k-row group stride
  d.bitfield.layout_type_ = 1;  // B128
  return d.desc_;
}

// MN-major SW128 descriptor: k16-atom s = 16 k-rows = 2KB step; SBO = 8-row group.
__device__ __forceinline__ uint64_t wg_desc_mnsw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 2048;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;  // B128
  return d.desc_;
}

__device__ __forceinline__ void wg_mbar_init(uint64_t* bar, uint32_t count) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count));
}

__device__ __forceinline__ void wg_mbar_arrive(uint64_t* bar) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("{.reg .b64 t; mbarrier.arrive.shared.b64 t, [%0];}" ::"r"(a));
}

__device__ __forceinline__ void wg_mbar_arrive_cpasync(uint64_t* bar) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a));
}

__device__ __forceinline__ void wg_mbar_wait(uint64_t* bar, uint32_t phase) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p;}"
        : "=r"(done)
        : "r"(a), "r"(phase)
        : "memory");
  }
}

#if defined(MK_GEMM_N256_TMA) || defined(MK_GEMM_N256_NT_TMA) || defined(MK_GEMM_D64_TMA) || \
    defined(MK_GEMM_N256_TMA_STORE)
// GEMM round-4 TMA feed for the mbarrier-ring bodies: an elected thread
// issues cp.async.bulk.tensor.2d per stage instead of the per-thread cp.async
// slices, with mbarrier.arrive.expect_tx on a count-1 full barrier. Tensormaps
// live in GLOBAL memory (per-program table built by mk.py; the SW128 slabs are
// exactly CU_TENSOR_MAP_SWIZZLE_128B), which requires a tensormap-proxy acquire
// fence before first use. Validated by n256_tma_ring_probe.py (n256 bodies:
// parity bit-identical to the cp.async ring; qwen dX-head -7..-9% standalone)
// and d64_tma_ring_probe.py (m64n64/m64n128 ring bodies: bit-identical, all
// classes win standalone; TN long-K dW rows -16.5..-19.6%).
__device__ __forceinline__ void wg_tmap_fence_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(map)
               : "memory");
}
#endif
#ifdef MK_GEMM_N256_TMA_STORE
// DeepGEMM R3 store recipe (nt-storerecipe-standalone-a440677.md): bulk-tensor
// C store with an EVICT_FIRST L2 policy (0x12F0... = cute CacheHintSm90::
// EVICT_FIRST). Dirty C lines stream into DRAM writeback early instead of
// pooling in L2 — standalone -24% / 0.92->1.21 TB/s on the C-write-bound NT
// lm-head rows. Store-side hint only: the load-side EVICT_FIRST variant
// measured HARMFUL (kills B L2 reuse) — do not add it to the feed paths.
__device__ __forceinline__ void wg_tma_store_2d_ef(const void* map, const void* src,
                                                   int x, int y) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(src);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
      " [%0, {%1, %2}], [%3], %4;" ::"l"(map), "r"(x), "r"(y), "r"(s),
      "l"(0x12F0000000000000ull)
      : "memory");
}
#endif
#if defined(MK_GEMM_N256_TMA) || defined(MK_GEMM_N256_NT_TMA) || defined(MK_GEMM_D64_TMA)
__device__ __forceinline__ void wg_mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("{.reg .b64 t; mbarrier.arrive.expect_tx.shared::cta.b64 t, [%0], %1;}"
               ::"r"(a), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void wg_tma_load_2d(const void* map, void* dst, int x, int y,
                                               uint64_t* bar) {
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  const uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(d), "l"(map), "r"(x), "r"(y), "r"(b)
      : "memory");
}
#endif

#if defined(MK_ATTN_PDF_FEED) && !defined(MK_PDF_PRODUCER)
#error "MK_ATTN_PDF_FEED requires MK_PDF_PRODUCER (the pdf WG2 producer image)"
#endif

#ifdef MK_PDF_PRODUCER
// Producer-feed mailbox (producer-df executor, phase 2): consumer thread 0 posts
// one request per n256-TMA tile; the pdf executor's WG2 producer thread (thin
// setmaxnreg.dec region, pure issuer) replicates the stage schedule and issues
// every TMA load, gated only by ring empties — the GEMM round-5 producer
// topology in-model, motivated by the long-D64 finding that the elected-thread
// feed's fence+expect_tx serialization loses per row family. Requests are
// strictly serial per block: tile T+1's post happens program-order after tile
// T's last full-wait, so the producer is always done with T's barriers before
// T+1's re-init. File-scope __shared__: every kernel that references it gets a
// per-block instance; megakernel_pdf arms `active`, other executors clear it
// (smem is not guaranteed zero across launches).
// Attention stream requests (kind 4, MK_ATTN_PDF_FEED) reuse the fields with
// a cp.async-generic meaning: tmA/tmB = the two operand GMEM row-0 base
// pointers (dq: K/V; dkv: Q/dO), m0/n0 = their gmem row strides in BYTES,
// k_base/bk = their per-stage gmem byte steps, a0/b0 = the wga_off64 dst
// slabs with a_stride/b_stride per-stage slab strides. bfull is count-128
// (one cp.async.mbarrier.arrive per producer thread), bempty count-1
// (consumer tid0 arms it behind the end-of-stage consumer_sync). a1/a_t/b_t/
// expect_bytes are unused for kind 4.
struct MkPdfFeed {
  const char* tmA;
  const char* tmB;
  char* a0;                // S.A[0][0]
  char* a1;                // S.A[0][1] (a_t only)
  char* b0;                // S.B[0]
  uint64_t* bfull;
  uint64_t* bempty;
  int a_stride, b_stride;  // per-stage slab strides (bytes)
  int m0, n0, iters, stages, a_t, b_t, bk, k_base, kind;
  unsigned expect_bytes;
  int active;
  int halt;
  int seq;                 // release-published request counter
};
__shared__ MkPdfFeed g_pdf_feed;

// The feed functions for n256-TMA, and optionally for D64-TMA probe builds.
#ifdef MK_GEMM_N256_TMA
#define MK_PDF_N256_FEED 1
#endif
#ifdef MK_GEMM_N256_NT_TMA
#define MK_PDF_N256_NT_FEED 1
#endif
#if defined(MK_GEMM_D64_TMA) && defined(MK_PDF_D64_FEED)
#define MK_PDF_D64_TMA_FEED 1
#endif
// GEMM TMA replay needs the tensormap helpers; the attention cp.async stream
// (MK_ATTN_PDF_FEED) does not, so it arms the WG2 loop independently.
#if defined(MK_PDF_N256_FEED) || defined(MK_PDF_N256_NT_FEED) || defined(MK_PDF_D64_TMA_FEED)
#define MK_PDF_GEMM_FEED 1
#endif
#if defined(MK_PDF_GEMM_FEED) || defined(MK_ATTN_PDF_FEED)
#define MK_PDF_FEED 1
#endif

__device__ __forceinline__ void mk_pdf_st_release(int* p, int v) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ int mk_pdf_ld_acquire(const int* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  int v;
  asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
#endif

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


template <class MMA>
__device__ __forceinline__ void wg_mma_ktile_n128(const uint64_t (&da)[4], const uint64_t (&db)[4],
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
             d[60], d[61], d[62], d[63], cute::SM90::GMMA::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

#define WG_D4(i) d[(i) + 0], d[(i) + 1], d[(i) + 2], d[(i) + 3]
#define WG_D16(i) WG_D4(i), WG_D4((i) + 4), WG_D4((i) + 8), WG_D4((i) + 12)
#define WG_D64 WG_D16(0), WG_D16(16), WG_D16(32), WG_D16(48)
#define WG_D128 WG_D64, WG_D16(64), WG_D16(80), WG_D16(96), WG_D16(112)
template <class MMA>
__device__ __forceinline__ void wg_mma_ktile_n256(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                                  float (&d)[128]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], WG_D128, cute::SM90::GMMA::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}
#undef WG_D128
#undef WG_D64
#undef WG_D16
#undef WG_D4

#ifdef MK_GEMM_N256_NT_SUPERTILE
// Exact qwen lm-head NT+CE route: 256x128 supertile from pipe_probe_st.py.
// Two 128-row output strips share one 128-column B slab per K stage; the qwen
// vocab is divisible by 128, so this avoids the n256 body's final 128-col tail.
template <int STAGES>
__device__ __noinline__ void op_gemm_wgmma_n256_nt_supertile_impl(
    const Instr& I, int tile, void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  bf16* C = reinterpret_cast<bf16*>(bufs[I.args[2]]);
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  if (!(flags & 2) || I.args[20] <= 0) return;
  if (flags & (1 | 4 | 8 | 16 | 32 | 256 | 8192)) return;

  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  WgmmaSmemN256NtSupertileT<STAGES>& S =
      *reinterpret_cast<WgmmaSmemN256NtSupertileT<STAGES>*>(smem_raw);
  const int n_tiles = N / 128;
  const int m_tiles = M / 256;
  const bool nmajor = flags & GEMM_N256_NMAJOR_FLAG;
  const int mt = nmajor ? (tile % m_tiles) : (tile / n_tiles);
  const int nt = nmajor ? (tile / m_tiles) : (tile % n_tiles);
  const int m0 = mt * 256;
  const int n0 = nt * 128;
  const int tid = mk_tid();
  const int wg = tid / 128;

  const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
  const char* tmA = tbl + (int64_t)I.args[21] * 128;
  const char* tmB = tbl + (int64_t)I.args[22] * 128;
  uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
  uint64_t* bempty = bfull + STAGES;
  if (tid == 0) {
    wg_tmap_fence_acquire(tmA);
    wg_tmap_fence_acquire(tmB);
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      wg_mbar_init(&bfull[s], 1);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
#ifdef MK_PDF_N256_NT_FEED
#ifdef MK_GEMM_N256_NT_SUPERTILE_PDFONLY
  if (tid == 0) {
#else
  const bool pdf_feed = g_pdf_feed.active;
  if (pdf_feed && tid == 0) {
#endif
    MkPdfFeed& F = g_pdf_feed;
    F.tmA = tmA;
    F.tmB = tmB;
    F.a0 = reinterpret_cast<char*>(S.A[0][0]);
    F.a1 = reinterpret_cast<char*>(S.A[0][2]);
    F.b0 = reinterpret_cast<char*>(S.B[0]);
    F.a_stride = (STAGES > 1)
        ? (int)(reinterpret_cast<char*>(S.A[1][0]) - reinterpret_cast<char*>(S.A[0][0]))
        : 0;
    F.b_stride = (STAGES > 1)
        ? (int)(reinterpret_cast<char*>(S.B[1]) - reinterpret_cast<char*>(S.B[0]))
        : 0;
    F.bfull = bfull;
    F.bempty = bempty;
    F.m0 = m0;
    F.n0 = n0;
    F.iters = K / WG_BK;
    F.stages = STAGES;
    F.a_t = 0;
    F.b_t = 1;
    F.bk = WG_BK;
    F.k_base = 0;
    F.kind = 6;
    F.expect_bytes = 49152;
    mk_pdf_st_release(&F.seq, F.seq + 1);
  }
#else
#ifdef MK_GEMM_N256_NT_SUPERTILE_PDFONLY
  return;
#else
  const bool pdf_feed = false;
#endif
#endif

  float d0[64], d1[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    d0[i] = 0.0f;
    d1[i] = 0.0f;
  }
  const int iters = K / WG_BK;
  constexpr int LEAD = STAGES - 2;
#ifndef MK_GEMM_N256_NT_SUPERTILE_PDFONLY
  auto issue_stage_tma = [&](int t) {
    const int st = t % STAGES;
    const int k0 = t * WG_BK;
    uint64_t* bar = &bfull[st];
    wg_mbar_expect_tx(bar, 49152);  // two A 128x64 boxes + two B 64x64 boxes
    wg_tma_load_2d(tmA, S.A[st][0], k0, m0, bar);
    wg_tma_load_2d(tmA, S.A[st][2], k0, m0 + 128, bar);
#pragma unroll
    for (int g = 0; g < 2; ++g)
      wg_tma_load_2d(tmB, reinterpret_cast<char*>(S.B[st]) + g * 8192,
                     k0, n0 + g * 64, bar);
  };
  if (!pdf_feed && tid == 0) {
#pragma unroll
    for (int p = 0; p < min(LEAD + 1, iters); ++p) issue_stage_tma(p);
  }
#endif
  for (int t = 0; t < iters; ++t) {
    const int st = t % STAGES;
    wg_mbar_wait(&bfull[st], (t / STAGES) & 1);
#ifndef MK_GEMM_N256_NT_SUPERTILE_PDFONLY
    if (!pdf_feed && tid == 0) {
      const int tn = t + LEAD + 1;
      if (tn < iters) {
        if (tn >= STAGES)
          wg_mbar_wait(&bempty[tn % STAGES], (tn / STAGES - 1) & 1);
        issue_stage_tma(tn);
      }
    }
#endif
    uint64_t da0[4], da1[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da0[s] = wg_desc_ksw(S.A[st][wg], s);
      da1[s] = wg_desc_ksw(S.A[st][2 + wg], s);
      db[s] = wg_desc_ksw(S.B[st], s);
    }
    wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(
        da0, db, d0);
    wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(
        da1, db, d1);
    wg_mbar_arrive(&bempty[st]);
  }
  cute::warpgroup_wait<0>();
  consumer_sync();
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      wg_mbar_init(&bfull[s], 1);
      wg_mbar_init(&bempty[s], 256);
    }
  }
#ifndef MK_GEMM_N256_NT_SUPERTILE_POSTINIT_NOSYNC
  consumer_sync();
#endif

#ifdef MK_GEMM_N256_NT_SUPERTILE_REG_EPI
  const int wtid = tid % 128;
  const int w = wtid / 32, l = wtid % 32;
  const int cb = (l & 3) * 2;
#pragma unroll
  for (int strip = 0; strip < 2; ++strip) {
    const float(&d)[64] = strip == 0 ? d0 : d1;
    const int row_off = m0 + strip * 128;
#pragma unroll
    for (int n8 = 0; n8 < 16; ++n8) {
      const int c = n8 * 8 + cb;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(row_off + r) * N + n0 + c;
        __nv_bfloat162 out;
        out.x = f2bf(d[n8 * 4 + i * 2 + 0]);
        out.y = f2bf(d[n8 * 4 + i * 2 + 1]);
        *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
      }
    }

    if (flags & 2048) {
      float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
      const int nparts = I.args[10];
      const int row_base = wg * 64 + w * 16 + l / 4;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = row_base + 8 * i;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          float mx = -INFINITY, se = 0.0f;
#pragma unroll
          for (int n8 = half * 8; n8 < half * 8 + 8; ++n8) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const float zv = bf2f(f2bf(d[n8 * 4 + i * 2 + j]));
              if (zv > mx) {
                se = se * lmhead_exp(mx - zv) + 1.0f;
                mx = zv;
              } else {
                se += lmhead_exp(zv - mx);
              }
            }
          }
#pragma unroll
          for (int o = 1; o < 4; o <<= 1) {
            const float om = __shfl_xor_sync(0xffffffff, mx, o);
            const float os = __shfl_xor_sync(0xffffffff, se, o);
            const float Mx = fmaxf(mx, om);
            se = (mx == -INFINITY && om == -INFINITY) ? 0.0f
                                                      : se * lmhead_exp(mx - Mx) +
                                                            os * lmhead_exp(om - Mx);
            mx = Mx;
          }
          const int part = n0 / WG_BN + half;
          if ((l & 3) == 0 && part < nparts) {
            const int64_t o = ((int64_t)(row_off + r) * nparts + part) * 2;
            parts[o] = mx;
            parts[o + 1] = se;
          }
        }
      }
    }
  }
#else
  float* Cs = reinterpret_cast<float*>(smem_raw);
  const int wtid = tid % 128;
  const int w = wtid / 32, l = wtid % 32;
  const int cb = (l & 3) * 2;
#pragma unroll
  for (int strip = 0; strip < 2; ++strip) {
    const float(&d)[64] = strip == 0 ? d0 : d1;
    const int row_off = m0 + strip * 128;
    {
      const int r = wg * 64 + w * 16 + l / 4;
#pragma unroll
      for (int n8 = 0; n8 < 16; ++n8)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            Cs[(r + 8 * i) * WG_LDC_N128 + n8 * 8 + cb + j] =
                d[n8 * 4 + i * 2 + j];
    }
    consumer_sync();
    float* parts = (flags & 2048) ? reinterpret_cast<float*>(bufs[I.args[9]]) : nullptr;
    const int nparts = (flags & 2048) ? I.args[10] : 0;
    const int nb = n0 / WG_BN;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int gid = tid + g * 256;
      const int m = gid / 16, c8 = (gid % 16) * 8;
      const int64_t idx = (int64_t)(row_off + m) * N + n0 + c8;
      uint4 out;
      bf16* oe = reinterpret_cast<bf16*>(&out);
      float zv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        oe[e] = f2bf(Cs[m * WG_LDC_N128 + c8 + e]);
        zv[e] = bf2f(oe[e]);
      }
      *reinterpret_cast<uint4*>(&C[idx]) = out;

      if (parts) {
        float mx = zv[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) mx = fmaxf(mx, zv[e]);
        const int lane = tid & 31;
        const unsigned mask = 0xffu << (lane & 24);
#pragma unroll
        for (int off = 4; off > 0; off >>= 1)
          mx = fmaxf(mx, __shfl_xor_sync(mask, mx, off));
        float se = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) se += lmhead_exp(zv[e] - mx);
#pragma unroll
        for (int off = 4; off > 0; off >>= 1)
          se += __shfl_xor_sync(mask, se, off);
        if ((lane & 7) == 0) {
          const int half = c8 >= WG_BN;
          const int64_t o = ((int64_t)(row_off + m) * nparts + nb + half) * 2;
          parts[o] = mx;
          parts[o + 1] = se;
        }
      }
    }
    consumer_sync();
  }
#endif
}

#endif

// m64n256 NT direct-store tile (qwen giant-vocab follow-up to fat_gemm_probe.py).
// The staged 128x256 route needs 160KB and fails the current cooperative launch at 132
// blocks; this variant keeps the 100KB page by skipping the coalesced fp32 epilogue slab.
// It is deliberately narrow: NT only, bf16 output only, optional CE/LSE partials, and a
// final 128-column tail. Broad direct-store use regressed in the standalone probe.
template <int STAGES>
__device__ __noinline__ void op_gemm_wgmma_n256_direct_impl(const Instr& I, int tile,
                                                            void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  bf16* C = reinterpret_cast<bf16*>(bufs[I.args[2]]);
  const int N = I.args[4], K = I.args[5], flags = I.args[6];
  if (!(flags & 2)) return;
  const bf16* Res = (flags & 16) ? reinterpret_cast<const bf16*>(bufs[I.args[7]]) : nullptr;

  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  WgmmaSmemN256T<STAGES>& S = *reinterpret_cast<WgmmaSmemN256T<STAGES>*>(smem_raw);
  const int n_tiles = (N + 255) / 256;
  const int m_tiles = I.args[3] / WG_BM;
  const bool nmajor = flags & GEMM_N256_NMAJOR_FLAG;
  const int mt = nmajor ? (tile % m_tiles) : (tile / n_tiles);
  const int nt = nmajor ? (tile / m_tiles) : (tile % n_tiles);
  const int m0 = mt * WG_BM;
  const int n0 = nt * 256;
  const int valid_cols = min(256, N - n0);
  if (valid_cols <= 0) return;

  const int tid = mk_tid();
  const int wg = tid / 128;
  const int wtid = tid % 128;
  auto issue_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // A: 128r x 64k
      const int v = tid + i * 256;
      const int r = v / 8, k8 = (v % 8) * 8;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(S.A[st][r / 64]) + wg_koff_sw(r % 64, k8),
          &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {  // B: 256 x 64, K-major; invalid tail rows are ignored
      const int v = tid + i * 256;
      const int r = v / 8, k8 = (v % 8) * 8;
      const int br = (r < valid_cols) ? (n0 + r) : (N - 1);
      __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + wg_koff_sw(r, k8),
                              &B[(int64_t)br * K + k0 + k8], 16);
    }
    __pipeline_commit();
  };

  float d[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) d[i] = 0.0f;
  const int iters = K / WG_BK;
  bool did_tma = false;
#if defined(MK_GEMM_MBAR_RING) && defined(MK_GEMM_N256_NT_TMA)
  if (I.args[20] > 0 && valid_cols == 256) {
    did_tma = true;
    constexpr int WG_N256_MBAR_LEAD = STAGES - 2;
    const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
    const char* tmA = tbl + (int64_t)I.args[21] * 128;
    const char* tmB = tbl + (int64_t)I.args[22] * 128;
    uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
    uint64_t* bempty = bfull + STAGES;
    if (tid == 0) {
      wg_tmap_fence_acquire(tmA);
      wg_tmap_fence_acquire(tmB);
#pragma unroll
      for (int s = 0; s < STAGES; ++s) {
        wg_mbar_init(&bfull[s], 1);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
#ifdef MK_PDF_N256_NT_FEED
    const bool pdf_feed = g_pdf_feed.active;
    if (pdf_feed && tid == 0) {
      MkPdfFeed& F = g_pdf_feed;
      F.tmA = tmA;
      F.tmB = tmB;
      F.a0 = reinterpret_cast<char*>(S.A[0][0]);
      F.a1 = nullptr;
      F.b0 = reinterpret_cast<char*>(S.B[0]);
      F.a_stride = (STAGES > 1)
          ? (int)(reinterpret_cast<char*>(S.A[1][0]) - reinterpret_cast<char*>(S.A[0][0]))
          : 0;
      F.b_stride = (STAGES > 1)
          ? (int)(reinterpret_cast<char*>(S.B[1]) - reinterpret_cast<char*>(S.B[0]))
          : 0;
      F.bfull = bfull;
      F.bempty = bempty;
      F.m0 = m0;
      F.n0 = n0;
      F.iters = iters;
      F.stages = STAGES;
      F.a_t = 0;
      F.b_t = 1;
      F.bk = WG_BK;
      F.k_base = 0;
      F.kind = 3;
      F.expect_bytes = 49152;
      mk_pdf_st_release(&F.seq, F.seq + 1);
    }
#endif
    auto issue_stage_tma = [&](int t) {
      if (tid == 0) {
        const int st = t % STAGES;
        const int k0 = t * WG_BK;
        wg_mbar_expect_tx(&bfull[st], 49152);  // A 16KB + B 32KB per stage
        wg_tma_load_2d(tmA, S.A[st][0], k0, m0, &bfull[st]);
#pragma unroll
        for (int g = 0; g < 4; ++g)
          wg_tma_load_2d(tmB, reinterpret_cast<char*>(S.B[st]) + g * 8192,
                         k0, n0 + g * 64, &bfull[st]);
      }
    };
#ifdef MK_PDF_N256_NT_FEED
    if (!pdf_feed)
#endif
    {
#pragma unroll
      for (int p = 0; p < min(WG_N256_MBAR_LEAD + 1, iters); ++p)
        issue_stage_tma(p);
    }
    for (int t = 0; t < iters; ++t) {
      const int st = t % STAGES;
      wg_mbar_wait(&bfull[st], (t / STAGES) & 1);
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = wg_desc_ksw(S.A[st][wg], s);
        db[s] = wg_desc_ksw(S.B[st], s);
      }
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(
          da, db, d);
      wg_mbar_arrive(&bempty[st]);
#ifdef MK_PDF_N256_NT_FEED
      if (!pdf_feed)
#endif
      {
        const int tn = t + WG_N256_MBAR_LEAD + 1;
        if (tn < iters) {
          if (tn >= STAGES)
            wg_mbar_wait(&bempty[tn % STAGES], (tn / STAGES - 1) & 1);
          issue_stage_tma(tn);
        }
      }
    }
    cute::warpgroup_wait<0>();
    consumer_sync();
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < STAGES; ++s) {
        wg_mbar_init(&bfull[s], 1);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
  }
#endif
#if defined(MK_GEMM_MBAR_RING) && defined(MK_GEMM_N256_NT_MBAR)
  if (!did_tma) {
    constexpr int WG_N256_MBAR_LEAD = STAGES - 2;
    uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
    uint64_t* bempty = bfull + STAGES;
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < STAGES; ++s) {
        wg_mbar_init(&bfull[s], 256);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
    auto issue_stage_mb = [&](int t) {
      const int st = t % STAGES;
      issue_stage(t * WG_BK, st);
      wg_mbar_arrive_cpasync(&bfull[st]);
    };
#pragma unroll
    for (int p = 0; p < min(WG_N256_MBAR_LEAD + 1, iters); ++p)
      issue_stage_mb(p);
    for (int t = 0; t < iters; ++t) {
      const int st = t % STAGES;
      wg_mbar_wait(&bfull[st], (t / STAGES) & 1);
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = wg_desc_ksw(S.A[st][wg], s);
        db[s] = wg_desc_ksw(S.B[st], s);
      }
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(
          da, db, d);
      wg_mbar_arrive(&bempty[st]);
      const int tn = t + WG_N256_MBAR_LEAD + 1;
      if (tn < iters) {
        if (tn >= STAGES)
          wg_mbar_wait(&bempty[tn % STAGES], (tn / STAGES - 1) & 1);
        issue_stage_mb(tn);
      }
    }
    cute::warpgroup_wait<0>();
    consumer_sync();
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < STAGES; ++s) {
        wg_mbar_init(&bfull[s], 256);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
  }
#else
  if (!did_tma) {
#pragma unroll
    for (int p = 0; p < STAGES - 1; ++p)
      if (p < iters) issue_stage(p * WG_BK, p);
    for (int t = 0; t < iters; ++t) {
      if (t + STAGES - 1 < iters)
        issue_stage((t + STAGES - 1) * WG_BK, (t + STAGES - 1) % STAGES);
      __pipeline_wait_prior(min(STAGES - 1, iters - t - 1));
      consumer_sync();
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = wg_desc_ksw(S.A[t % STAGES][wg], s);
        db[s] = wg_desc_ksw(S.B[t % STAGES], s);
      }
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(
          da, db, d);
      consumer_sync();
    }
  }
#endif

  const int w = wtid / 32, l = wtid % 32;
  const int cb = (l & 3) * 2;
#ifdef MK_GEMM_N256_TMA_STORE
  // DeepGEMM R3 store recipe port: stage the 128x256 bf16 tile into the DEAD
  // mainloop ring smem (every mainloop variant ends warpgroup_wait<0> +
  // consumer_sync before this point) as four SW128 64-col slabs
  // (TMA_D_BLOCK_N=64), then drain via four EVICT_FIRST bulk-tensor stores on
  // tid<4's private bulk groups. The drain overlaps the ssq/CE partial
  // epilogues below; wait + consumer_sync run before op return so the next
  // op's smem writes cannot race the in-flight TMA reads. Host injects
  // args[19] = 1 + C tensormap row ({64n,128m} box, SW128) only on exact
  // eligible rows; unset rows take the byte-identical direct path.
  const bool tma_store = I.args[19] > 0 && I.args[20] > 0 && valid_cols == 256;
  if (tma_store) {
    bf16* cs = reinterpret_cast<bf16*>(&S);
#pragma unroll
    for (int n8 = 0; n8 < 32; ++n8) {
      const int slab = n8 >> 3, ch = n8 & 7;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        float v0 = d[n8 * 4 + i * 2 + 0];
        float v1 = d[n8 * 4 + i * 2 + 1];
        if (Res) {
          const int64_t idx = (int64_t)(m0 + r) * N + n0 + n8 * 8 + cb;
          v0 += bf2f(Res[idx + 0]);
          v1 += bf2f(Res[idx + 1]);
        }
        __nv_bfloat162 out;
        out.x = f2bf(v0);
        out.y = f2bf(v1);
        *reinterpret_cast<__nv_bfloat162*>(
            &cs[slab * 8192 + r * 64 + ((ch ^ (r & 7)) << 3) + cb]) = out;
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    consumer_sync();
    if (tid < 4) {
      const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
      const char* tmC = tbl + (int64_t)(I.args[19] - 1) * 128;
      wg_tmap_fence_acquire(tmC);
      wg_tma_store_2d_ef(tmC, cs + tid * 8192, n0 + tid * 64, m0);
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
  } else {
#endif
#pragma unroll
  for (int n8 = 0; n8 < 32; ++n8) {
    const int c = n8 * 8 + cb;
    if (c < valid_cols) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + c;
        __nv_bfloat162 out;
        float v0 = d[n8 * 4 + i * 2 + 0];
        float v1 = d[n8 * 4 + i * 2 + 1];
        if (Res) {
          v0 += bf2f(Res[idx + 0]);
          v1 += bf2f(Res[idx + 1]);
        }
        out.x = f2bf(v0);
        out.y = f2bf(v1);
        *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
      }
    }
  }
#ifdef MK_GEMM_N256_TMA_STORE
  }
#endif

  if (flags & 8192) {  // ssq partials from post-residual bf16-rounded output
    float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
    const int nparts = I.args[10];
    const int warp = tid / 32, lane = tid % 32;
    const int row_base = (warp / 4) * 64 + (warp & 3) * 16;
    const int lane_row = lane / 4;
    const int cb = (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int r = row_base + lane_row + 8 * i;
#pragma unroll
      for (int half = 0; half < 4; ++half) {
        if (half * WG_BN >= valid_cols) continue;
        float ss = 0.0f;
#pragma unroll
        for (int n8 = half * 8; n8 < half * 8 + 8; ++n8) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int c = n8 * 8 + cb + j;
            float v = d[n8 * 4 + i * 2 + j];
            if (Res) v += bf2f(Res[(int64_t)(m0 + r) * N + n0 + c]);
            const float zv = bf2f(f2bf(v));
            ss += zv * zv;
          }
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
        const int part = n0 / WG_BN + half;
        if ((lane & 3) == 0 && part < nparts)
          parts[(int64_t)(m0 + r) * nparts + part] = ss;
      }
    }
  }

#ifdef MK_GEMM_N256_TMA_STORE
  // Drain the EVICT_FIRST C stores (issued above, overlapped with the ssq/CE
  // partials) before any thread can leave the op and start the next op's smem
  // writes. tid<4 wait their private bulk groups; consumer_sync fans out.
  auto tma_store_drain = [&] {
    if (tma_store) {
      if (tid < 4) {
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
      consumer_sync();
    }
  };
  if (!(flags & 2048)) {
    tma_store_drain();
    return;
  }
#else
  if (!(flags & 2048)) return;
#endif

  float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
  const int nparts = I.args[10];
  const int warp = tid / 32, lane = tid % 32;
  const int row_base = (warp / 4) * 64 + (warp & 3) * 16;
  const int lane_row = lane / 4;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = row_base + lane_row + 8 * i;
#pragma unroll
    for (int half = 0; half < 4; ++half) {
      if (half * WG_BN >= valid_cols) continue;
      float mx = -INFINITY, se = 0.0f;
#pragma unroll
      for (int n8 = half * 8; n8 < half * 8 + 8; ++n8) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float zv = bf2f(f2bf(d[n8 * 4 + i * 2 + j]));
          if (zv > mx) {
            se = se * lmhead_exp(mx - zv) + 1.0f;
            mx = zv;
          } else {
            se += lmhead_exp(zv - mx);
          }
        }
      }
#pragma unroll
      for (int o = 1; o < 4; o <<= 1) {
        const float om = __shfl_xor_sync(0xffffffff, mx, o);
        const float os = __shfl_xor_sync(0xffffffff, se, o);
        const float Mx = fmaxf(mx, om);
        se = (mx == -INFINITY && om == -INFINITY) ? 0.0f
                                                  : se * lmhead_exp(mx - Mx) +
                                                        os * lmhead_exp(om - Mx);
        mx = Mx;
      }
      const int part = n0 / WG_BN + half;
      if ((lane & 3) == 0 && part < nparts) {
        const int64_t o = ((int64_t)(m0 + r) * nparts + part) * 2;
        parts[o] = mx;
        parts[o + 1] = se;
      }
    }
  }
#ifdef MK_GEMM_N256_TMA_STORE
  tma_store_drain();
#endif
}

// m64n256 direct-store tile for exact qwen giant-vocab NN head-dX / TN dW and
// scratch-gated BF16 NN dX follow-ups. This keeps the 100KB page by skipping the
// fp32 epilogue slab. It is deliberately narrow: no split-K/acc/residual.
template <int STAGES, bool DROW = false>
__device__ __noinline__ void op_gemm_wgmma_n256_nn_f32_impl(const Instr& I, int tile,
                                                            void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  void* Cp = bufs[I.args[2]];
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  if ((flags & 2) || (flags & (4 | 16 | 32 | 256 | 2048 | 8192))) {
    return;
  }
  if constexpr (DROW) {
    if (!(flags & 1024) || (flags & 8) || I.args[11] != 128) return;
  } else {
    if (flags & 1024) return;
  }

  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  WgmmaSmemN256T<STAGES>& S = *reinterpret_cast<WgmmaSmemN256T<STAGES>*>(smem_raw);
  const int n_tiles = N / 256;
  const int m_tiles = M / WG_BM;
  const bool nmajor = flags & GEMM_N256_NMAJOR_FLAG;
  const int mt = nmajor ? (tile % m_tiles) : (tile / n_tiles);
  const int nt = nmajor ? (tile / m_tiles) : (tile % n_tiles);
  const int m0 = mt * WG_BM;
  const int n0 = nt * 256;
  const int tid = mk_tid();
  const int wg = tid / 128;
  const int wtid = tid % 128;
  const bool a_t = flags & 1;
  const bool c_f32 = flags & 8;
  if (m0 >= M) return;

  auto issue_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // A: 128r x 64k
      const int v = tid + i * 256;
      if (a_t) {
        const int h = v / 512, w_ = v % 512;
        const int k = w_ / 8, m8 = (w_ % 8) * 8;
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(S.A[st][h]) + wg_mnoff_sw(k, m8),
            &A[(int64_t)(k0 + k) * M + m0 + h * 64 + m8], 16);
      } else {
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(S.A[st][r / 64]) + wg_koff_sw(r % 64, k8),
            &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {  // B[K,N] N-contiguous, 256 columns
      const int v = tid + i * 256;
      const int k = v / 32, n8 = (v % 32) * 8;
      __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + (n8 / 64) * 8192 +
                                  wg_mnoff_sw(k, n8 % 64),
                              &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
    }
    __pipeline_commit();
  };

  float d[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) d[i] = 0.0f;
  const int iters = K / WG_BK;
#ifdef MK_GEMM_MBAR_RING
  constexpr int WG_N256_MBAR_LEAD = STAGES - 2;
  uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
  uint64_t* bempty = bfull + STAGES;
#ifdef MK_GEMM_N256_TMA
  // TMA feed (round 4): args[20] = 1 + tmap-table buffer id (0 = cp.async
  // feed), args[21]/args[22] = 128B row indices of the A/B tensormaps. bfull
  // becomes a count-1 expect_tx barrier; bempty and the ring are unchanged.
  const bool use_tma = I.args[20] > 0;
  const char* tmA = nullptr;
  const char* tmB = nullptr;
  if (use_tma) {
    const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
    tmA = tbl + (int64_t)I.args[21] * 128;
    tmB = tbl + (int64_t)I.args[22] * 128;
    if (tid == 0) {
      wg_tmap_fence_acquire(tmA);
      wg_tmap_fence_acquire(tmB);
    }
  }
  const uint32_t bfull_cnt = use_tma ? 1u : 256u;
#else
  const uint32_t bfull_cnt = 256u;
#endif
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      wg_mbar_init(&bfull[s], bfull_cnt);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
#ifdef MK_PDF_N256_FEED
  // producer-df: hand the whole stage schedule to the WG2 producer; consumers
  // keep only wait-full -> mma -> arrive-empty. The post is release-ordered
  // AFTER the barrier init consumer_sync above, so the producer never sees a
  // request whose barriers are not yet initialized.
  const bool pdf_feed = use_tma && g_pdf_feed.active;
  if (pdf_feed && tid == 0) {
    MkPdfFeed& F = g_pdf_feed;
    F.tmA = tmA;
    F.tmB = tmB;
    F.a0 = reinterpret_cast<char*>(S.A[0][0]);
    F.a1 = reinterpret_cast<char*>(S.A[0][1]);
    F.b0 = reinterpret_cast<char*>(S.B[0]);
    F.a_stride = (STAGES > 1)
        ? (int)(reinterpret_cast<char*>(S.A[1][0]) - reinterpret_cast<char*>(S.A[0][0]))
        : 0;
    F.b_stride = (STAGES > 1)
        ? (int)(reinterpret_cast<char*>(S.B[1]) - reinterpret_cast<char*>(S.B[0]))
        : 0;
    F.bfull = bfull;
    F.bempty = bempty;
    F.m0 = m0;
    F.n0 = n0;
    F.iters = iters;
    F.stages = STAGES;
    F.a_t = a_t ? 1 : 0;
    F.b_t = 0;
    F.bk = WG_BK;
    F.k_base = 0;
    F.kind = 0;
    F.expect_bytes = 49152;
    mk_pdf_st_release(&F.seq, F.seq + 1);
  }
#endif
  auto issue_stage_mb = [&](int t) {
    const int st = t % STAGES;
#ifdef MK_GEMM_N256_TMA
    if (use_tma) {
      if (tid == 0) {
        const int k0 = t * WG_BK;
        wg_mbar_expect_tx(&bfull[st], 49152);  // A 16KB + B 32KB per stage
        if (a_t) {
          wg_tma_load_2d(tmA, S.A[st][0], m0, k0, &bfull[st]);
          wg_tma_load_2d(tmA, S.A[st][1], m0 + 64, k0, &bfull[st]);
        } else {
          wg_tma_load_2d(tmA, S.A[st][0], k0, m0, &bfull[st]);
        }
#pragma unroll
        for (int g = 0; g < 4; ++g)
          wg_tma_load_2d(tmB, reinterpret_cast<char*>(S.B[st]) + g * 8192,
                         n0 + g * 64, k0, &bfull[st]);
      }
      return;
    }
#endif
    issue_stage(t * WG_BK, st);
    wg_mbar_arrive_cpasync(&bfull[st]);
  };
#ifdef MK_PDF_N256_FEED
  if (!pdf_feed)
#endif
  {
#pragma unroll
    for (int p = 0; p < min(WG_N256_MBAR_LEAD + 1, iters); ++p)
      issue_stage_mb(p);
  }
  for (int t = 0; t < iters; ++t) {
    const int st = t % STAGES;
    wg_mbar_wait(&bfull[st], (t / STAGES) & 1);
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = a_t ? wg_desc_mnsw(S.A[st][wg], s) : wg_desc_ksw(S.A[st][wg], s);
      db[s] = wg_desc_mnsw128(S.B[st], s);
    }
    if (a_t)
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(
          da, db, d);
    else
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(
          da, db, d);
    wg_mbar_arrive(&bempty[st]);
#ifdef MK_PDF_N256_FEED
    if (!pdf_feed)
#endif
    {
      const int tn = t + WG_N256_MBAR_LEAD + 1;
      if (tn < iters) {
        if (tn >= STAGES)
          wg_mbar_wait(&bempty[tn % STAGES], (tn / STAGES - 1) & 1);
        issue_stage_mb(tn);
      }
    }
  }
  cute::warpgroup_wait<0>();
  consumer_sync();
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      wg_mbar_init(&bfull[s], bfull_cnt);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
#else
#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p)
    if (p < iters) issue_stage(p * WG_BK, p);
  for (int t = 0; t < iters; ++t) {
    if (t + STAGES - 1 < iters)
      issue_stage((t + STAGES - 1) * WG_BK, (t + STAGES - 1) % STAGES);
    __pipeline_wait_prior(min(STAGES - 1, iters - t - 1));
    consumer_sync();
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = a_t ? wg_desc_mnsw(S.A[t % STAGES][wg], s) : wg_desc_ksw(S.A[t % STAGES][wg], s);
      db[s] = wg_desc_mnsw128(S.B[t % STAGES], s);
    }
    if (a_t)
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(
          da, db, d);
    else
      wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(
          da, db, d);
    consumer_sync();
  }
#endif

  const int w = wtid / 32, l = wtid % 32;
  const int cb = (l & 3) * 2;
  if constexpr (DROW) {
    bf16* C = reinterpret_cast<bf16*>(Cp);
    const bf16* Oatt = reinterpret_cast<const bf16*>(bufs[I.args[9]]);
    float* drow = reinterpret_cast<float*>(bufs[I.args[10]]);
    const int D = I.args[11];
    float drow_sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
    for (int n8 = 0; n8 < 32; ++n8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + n8 * 8 + cb;
        __nv_bfloat162 out;
        const bf16 z0 = f2bf(d[n8 * 4 + i * 2 + 0]);
        const bf16 z1 = f2bf(d[n8 * 4 + i * 2 + 1]);
        out.x = z0;
        out.y = z1;
        *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
        const int h = n8 >> 4;
        drow_sum[i][h] += bf2f(z0) * bf2f(Oatt[idx]) + bf2f(z1) * bf2f(Oatt[idx + 1]);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float s = drow_sum[i][h];
        s += __shfl_xor_sync(0xffffffff, s, 1);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        if ((l & 3) == 0) {
          const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
          atomicAdd(&drow[(int64_t)(n0 / D + h) * M + m0 + r], s);
        }
      }
    }
  } else {
#pragma unroll
    for (int n8 = 0; n8 < 32; ++n8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + n8 * 8 + cb;
        if (c_f32) {
          float* C = reinterpret_cast<float*>(Cp);
          float2 out = make_float2(d[n8 * 4 + i * 2 + 0], d[n8 * 4 + i * 2 + 1]);
          *reinterpret_cast<float2*>(&C[idx]) = out;
        } else {
          bf16* C = reinterpret_cast<bf16*>(Cp);
          __nv_bfloat162 out;
          out.x = f2bf(d[n8 * 4 + i * 2 + 0]);
          out.y = f2bf(d[n8 * 4 + i * 2 + 1]);
          *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
        }
      }
    }
  }
}

#ifdef MK_GEMM_N256_HEAD_DX_EXACT
__device__ __noinline__ void op_gemm_wgmma_n256_head_dx_exact_impl(
    const Instr& I, int tile, void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  float* C = reinterpret_cast<float*>(bufs[I.args[2]]);
  const int flags = I.args[6];
  if (I.args[3] != 1024 || I.args[4] != 2560 || I.args[5] != 151936) return;
  if (!(flags & 8) || (flags & (1 | 2 | 4 | 16 | 32 | 256 | 1024 | 2048 | 8192))) {
    return;
  }
  if (!(flags & GEMM_N256_STAGE3_FLAG) || !(flags & GEMM_N256_NMAJOR_FLAG)) return;
  if (I.args[20] <= 0) return;

  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  constexpr int STAGES = 3;
  constexpr int M = 1024;
  constexpr int N = 2560;
  constexpr int K = 151936;
  constexpr int ITERS = K / WG_BK;
  WgmmaSmemN256T<STAGES>& S = *reinterpret_cast<WgmmaSmemN256T<STAGES>*>(smem_raw);
  const int mt = tile & 7;
  const int nt = tile >> 3;
  if (nt >= 10) return;
  const int m0 = mt * WG_BM;
  const int n0 = nt * 256;
  const int tid = mk_tid();
  const int wg = tid / 128;
  const int wtid = tid % 128;

  const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
  const char* tmA = tbl + (int64_t)I.args[21] * 128;
  const char* tmB = tbl + (int64_t)I.args[22] * 128;
  uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
  uint64_t* bempty = bfull + STAGES;
  if (tid == 0) {
    wg_tmap_fence_acquire(tmA);
    wg_tmap_fence_acquire(tmB);
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      wg_mbar_init(&bfull[s], 1);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();

#ifdef MK_PDF_N256_FEED
#ifdef MK_GEMM_N256_HEAD_DX_PDFONLY
  if (tid == 0) {
#else
  const bool pdf_feed = g_pdf_feed.active;
  if (pdf_feed && tid == 0) {
#endif
    MkPdfFeed& F = g_pdf_feed;
    F.tmA = tmA;
    F.tmB = tmB;
    F.a0 = reinterpret_cast<char*>(S.A[0][0]);
    F.a1 = reinterpret_cast<char*>(S.A[0][1]);
    F.b0 = reinterpret_cast<char*>(S.B[0]);
    F.a_stride = (int)(reinterpret_cast<char*>(S.A[1][0]) -
                       reinterpret_cast<char*>(S.A[0][0]));
    F.b_stride = (int)(reinterpret_cast<char*>(S.B[1]) -
                       reinterpret_cast<char*>(S.B[0]));
    F.bfull = bfull;
    F.bempty = bempty;
    F.m0 = m0;
    F.n0 = n0;
    F.iters = ITERS;
    F.stages = STAGES;
    F.a_t = 0;
    F.b_t = 0;
    F.bk = WG_BK;
    F.k_base = 0;
    F.kind = 0;
    F.expect_bytes = 49152;
    mk_pdf_st_release(&F.seq, F.seq + 1);
  }
#else
#ifndef MK_GEMM_N256_HEAD_DX_PDFONLY
  const bool pdf_feed = false;
#endif
#endif

  float d[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) d[i] = 0.0f;
  constexpr int WG_N256_MBAR_LEAD = STAGES - 2;
#ifndef MK_GEMM_N256_HEAD_DX_PDFONLY
  auto issue_stage_tma = [&](int t) {
    if (tid == 0) {
      const int st = t % STAGES;
      const int k0 = t * WG_BK;
      wg_mbar_expect_tx(&bfull[st], 49152);
      wg_tma_load_2d(tmA, S.A[st][0], k0, m0, &bfull[st]);
#pragma unroll
      for (int g = 0; g < 4; ++g)
      wg_tma_load_2d(tmB, reinterpret_cast<char*>(S.B[st]) + g * 8192,
                       n0 + g * 64, k0, &bfull[st]);
    }
  };
  if (!pdf_feed) {
#pragma unroll
    for (int p = 0; p < WG_N256_MBAR_LEAD + 1; ++p) issue_stage_tma(p);
  }
#endif
  for (int t = 0; t < ITERS; ++t) {
    const int st = t % STAGES;
    wg_mbar_wait(&bfull[st], (t / STAGES) & 1);
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = wg_desc_ksw(S.A[st][wg], s);
      db[s] = wg_desc_mnsw128(S.B[st], s);
    }
    wg_mma_ktile_n256<SG::MMA_64x256x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(
        da, db, d);
    wg_mbar_arrive(&bempty[st]);
#ifndef MK_GEMM_N256_HEAD_DX_PDFONLY
    if (!pdf_feed) {
      const int tn = t + WG_N256_MBAR_LEAD + 1;
      if (tn < ITERS) {
        if (tn >= STAGES)
          wg_mbar_wait(&bempty[tn % STAGES], (tn / STAGES - 1) & 1);
        issue_stage_tma(tn);
      }
    }
#endif
  }
  cute::warpgroup_wait<0>();
  consumer_sync();
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      wg_mbar_init(&bfull[s], 1);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();

  const int w = wtid / 32, l = wtid % 32;
  const int cb = (l & 3) * 2;
#ifdef MK_QWEN_HEADDX_RMS_DOT_PARTIALS
  const bool rmsdot_partials =
      (flags & GEMM_HEADDX_RMSDOT_FLAG) && I.args[10] == (N / 256);
  float rmsdot0 = 0.0f, rmsdot1 = 0.0f;
  const bf16* rmsdot_x =
      rmsdot_partials ? reinterpret_cast<const bf16*>(bufs[I.args[11]]) : nullptr;
  const bf16* rmsdot_w =
      rmsdot_partials ? reinterpret_cast<const bf16*>(bufs[I.args[12]]) : nullptr;
#endif
#pragma unroll
  for (int n8 = 0; n8 < 32; ++n8) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
      const int64_t idx = (int64_t)(m0 + r) * N + n0 + n8 * 8 + cb;
      float2 out = make_float2(d[n8 * 4 + i * 2 + 0], d[n8 * 4 + i * 2 + 1]);
      *reinterpret_cast<float2*>(&C[idx]) = out;
#ifdef MK_QWEN_HEADDX_RMS_DOT_PARTIALS
      if (rmsdot_partials) {
        const int col = n0 + n8 * 8 + cb;
        const bf16* xr = rmsdot_x + (int64_t)(m0 + r) * N;
        const float contrib =
            out.x * bf2f(rmsdot_w[col]) * bf2f(xr[col]) +
            out.y * bf2f(rmsdot_w[col + 1]) * bf2f(xr[col + 1]);
        if (i == 0)
          rmsdot0 += contrib;
        else
          rmsdot1 += contrib;
      }
#endif
    }
  }
#ifdef MK_QWEN_HEADDX_RMS_DOT_PARTIALS
  if (rmsdot_partials) {
    rmsdot0 += __shfl_xor_sync(0xffffffff, rmsdot0, 1);
    rmsdot1 += __shfl_xor_sync(0xffffffff, rmsdot1, 1);
    rmsdot0 += __shfl_xor_sync(0xffffffff, rmsdot0, 2);
    rmsdot1 += __shfl_xor_sync(0xffffffff, rmsdot1, 2);
    if ((l & 3) == 0) {
      float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
      const int nparts = I.args[10];
      const int row0 = wg * 64 + w * 16 + l / 4;
      const int row1 = row0 + 8;
      parts[(int64_t)(m0 + row0) * nparts + nt] = rmsdot0;
      parts[(int64_t)(m0 + row1) * nparts + nt] = rmsdot1;
    }
  }
#endif
}
#endif

// m64n128 NT tile (v3 P4b r3, generalized from the peer session's lm_head route):
// 64 fp32 accumulators/thread double the mma work per sync and halve B-traffic per
// FLOP — the dependent chain per FLOP shortens (the one lever the register-lifetime
// law allows). REG ~200 fits the 255 df budget; __noinline__ isolates the fat
// accumulator frame from the dispatch switch. Supports NT/NN + residual (bit16), fp32
// stores (bit3), fp32 split-K atomics (bits3+5), and CE partials (bit11); generic
// routing (flags bit12) still excludes split-K/acc/f32/qkrope/Drow except explicit
// head-dX routes.
__device__ __noinline__ void op_gemm_wgmma_n128(const Instr& I, int tile, void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  void* Cp = bufs[I.args[2]];
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  const bf16* Res = (flags & 16) ? reinterpret_cast<const bf16*>(bufs[I.args[7]]) : nullptr;

  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
#ifdef MK_GEMM_MBAR_RING
  constexpr int WG_N128_STAGES = 3;
  WgmmaSmemN128T<WG_N128_STAGES>& S =
      *reinterpret_cast<WgmmaSmemN128T<WG_N128_STAGES>*>(smem_raw);
#else
  WgmmaSmemN128& S = *reinterpret_cast<WgmmaSmemN128*>(smem_raw);
#endif
  const int sk = (flags & 32) ? I.args[8] : 1;
  const int slice = tile % sk;
  const int mn = tile / sk;
  const int n_tiles = N / 128;
  const int m0 = (mn / n_tiles) * WG_BM;
  const int n0 = (mn % n_tiles) * 128;
  const int kchunk = ((K + sk * WG_BK - 1) / (sk * WG_BK)) * WG_BK;
  const int k_lo = slice * kchunk;
  const int k_hi = min(K, k_lo + kchunk);
  const int tid = mk_tid();
  const int wg = tid / 128;
  const int wtid = tid % 128;
  if (k_lo >= K) return;

  const bool b_t = flags & 2;  // NT: B[N,K] K-contig; NN: B[K,N] N-contig (MN slabs)
  const bool c_f32 = flags & 8;
  auto issue_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // A: 128r x 64k
      const int v = tid + i * 256;
      const int r = v / 8, k8 = (v % 8) * 8;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(S.A[st][r / 64]) + wg_koff_sw(r % 64, k8),
          &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // B: 128 x 64
      const int v = tid + i * 256;
      if (b_t) {
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + wg_koff_sw(r, k8),
                                &B[(int64_t)(n0 + r) * K + k0 + k8], 16);
      } else {  // [K,N] N-contig: two 64-mn MN-major slabs, 8KB apart
        const int k = v / 16, n8 = (v % 16) * 8;
        __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + (n8 / 64) * 8192 +
                                    wg_mnoff_sw(k, n8 % 64),
                                &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
      }
    }
    __pipeline_commit();
  };

  float d[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.0f;
  const int iters = (k_hi - k_lo) / WG_BK;
#ifdef MK_GEMM_MBAR_RING
  constexpr int WG_N128_LEAD = WG_N128_STAGES - 2;
  uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
  uint64_t* bempty = bfull + WG_N128_STAGES;
#ifdef MK_GEMM_D64_TMA
  // D64 ring TMA feed (round-4 port): args[20] = 1 + tmap-table buffer id
  // (0 = cp.async feed), args[21]/args[22] = 128B row indices of the A/B
  // tensormaps — the SAME slots as the n256 port (the row sets are disjoint:
  // this body never sees bit14 rows). A = one {64k,128m} box; B = one
  // {64k,128n} box (NT) or two {64n,64k} MN boxes 8KB apart (NN). bfull is a
  // count-1 expect_tx barrier; bempty and the ring are unchanged. The TMA
  // path is a FULLY SEPARATE loop: a merged use_tma branch inside the shared
  // ring loop taxed the cp.async path ~30us/step at S3072 (codegen reflow)
  // even when never taken.
  if (I.args[20] > 0) {
    const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
    const char* tmA = tbl + (int64_t)I.args[21] * 128;
    const char* tmB = tbl + (int64_t)I.args[22] * 128;
    if (tid == 0) {
      wg_tmap_fence_acquire(tmA);
      wg_tmap_fence_acquire(tmB);
#pragma unroll
      for (int s = 0; s < WG_N128_STAGES; ++s) {
        wg_mbar_init(&bfull[s], 1);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
#ifdef MK_PDF_D64_TMA_FEED
    const bool pdf_feed = g_pdf_feed.active;
    if (pdf_feed && tid == 0) {
      MkPdfFeed& F = g_pdf_feed;
      F.tmA = tmA;
      F.tmB = tmB;
      F.a0 = reinterpret_cast<char*>(S.A[0][0]);
      F.a1 = nullptr;
      F.b0 = reinterpret_cast<char*>(S.B[0]);
      F.a_stride = (WG_N128_STAGES > 1)
          ? (int)(reinterpret_cast<char*>(S.A[1][0]) - reinterpret_cast<char*>(S.A[0][0]))
          : 0;
      F.b_stride = (WG_N128_STAGES > 1)
          ? (int)(reinterpret_cast<char*>(S.B[1]) - reinterpret_cast<char*>(S.B[0]))
          : 0;
      F.bfull = bfull;
      F.bempty = bempty;
      F.m0 = m0;
      F.n0 = n0;
      F.iters = iters;
      F.stages = WG_N128_STAGES;
      F.a_t = 0;
      F.b_t = b_t ? 1 : 0;
      F.bk = WG_BK;
      F.k_base = k_lo;
      F.kind = 1;
      F.expect_bytes = 32768;
      mk_pdf_st_release(&F.seq, F.seq + 1);
    }
#endif
    auto issue_stage_tma = [&](int t) {
      if (tid == 0) {
        const int st = t % WG_N128_STAGES;
        const int k0 = k_lo + t * WG_BK;
        wg_mbar_expect_tx(&bfull[st], 32768);  // A 16KB + B 16KB per stage
        wg_tma_load_2d(tmA, S.A[st][0], k0, m0, &bfull[st]);
        if (b_t) {
          wg_tma_load_2d(tmB, S.B[st], k0, n0, &bfull[st]);
        } else {
#pragma unroll
          for (int g = 0; g < 2; ++g)
            wg_tma_load_2d(tmB, reinterpret_cast<char*>(S.B[st]) + g * 8192,
                           n0 + g * 64, k0, &bfull[st]);
        }
      }
    };
#ifdef MK_PDF_D64_TMA_FEED
    if (!pdf_feed)
#endif
    {
      for (int p = 0; p < min(WG_N128_LEAD + 1, iters); ++p) issue_stage_tma(p);
    }
    for (int t = 0; t < iters; ++t) {
      const int st = t % WG_N128_STAGES;
      wg_mbar_wait(&bfull[st], (t / WG_N128_STAGES) & 1);
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = wg_desc_ksw(S.A[st][wg], s);
        db[s] = b_t ? wg_desc_ksw(S.B[st], s) : wg_desc_mnsw128(S.B[st], s);
      }
      if (b_t)
        wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
      else
        wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
      wg_mbar_arrive(&bempty[st]);
#ifdef MK_PDF_D64_TMA_FEED
      if (!pdf_feed)
#endif
      {
        const int tn = t + WG_N128_LEAD + 1;
        if (tn < iters) {
          if (tn >= WG_N128_STAGES)
            wg_mbar_wait(&bempty[tn % WG_N128_STAGES], (tn / WG_N128_STAGES - 1) & 1);
          issue_stage_tma(tn);
        }
      }
    }
    cute::warpgroup_wait<0>();
    consumer_sync();
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < WG_N128_STAGES; ++s) {
        wg_mbar_init(&bfull[s], 1);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
  } else {
#endif
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < WG_N128_STAGES; ++s) {
      wg_mbar_init(&bfull[s], 256);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
  auto issue_stage_mb = [&](int t) {
    const int st = t % WG_N128_STAGES;
    issue_stage(k_lo + t * WG_BK, st);
    wg_mbar_arrive_cpasync(&bfull[st]);
  };
  for (int p = 0; p < min(WG_N128_LEAD + 1, iters); ++p) issue_stage_mb(p);
  for (int t = 0; t < iters; ++t) {
    const int st = t % WG_N128_STAGES;
    wg_mbar_wait(&bfull[st], (t / WG_N128_STAGES) & 1);
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = wg_desc_ksw(S.A[st][wg], s);
      db[s] = b_t ? wg_desc_ksw(S.B[st], s) : wg_desc_mnsw128(S.B[st], s);
    }
    if (b_t)
      wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
    else
      wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
    wg_mbar_arrive(&bempty[st]);
    const int tn = t + WG_N128_LEAD + 1;
    if (tn < iters) {
      if (tn >= WG_N128_STAGES)
        wg_mbar_wait(&bempty[tn % WG_N128_STAGES], (tn / WG_N128_STAGES - 1) & 1);
      issue_stage_mb(tn);
    }
  }
  cute::warpgroup_wait<0>();
  consumer_sync();
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < WG_N128_STAGES; ++s) {
      wg_mbar_init(&bfull[s], 256);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
#ifdef MK_GEMM_D64_TMA
  }
#endif
#else
  issue_stage(k_lo, 0);
  for (int t = 0; t < iters; ++t) {
    if (t + 1 < iters) issue_stage(k_lo + (t + 1) * WG_BK, (t + 1) & 1);
    __pipeline_wait_prior(t + 1 < iters ? 1 : 0);
    consumer_sync();
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = wg_desc_ksw(S.A[t & 1][wg], s);
      db[s] = b_t ? wg_desc_ksw(S.B[t & 1], s) : wg_desc_mnsw128(S.B[t & 1], s);
    }
    if (b_t)
      wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
    else
      wg_mma_ktile_n128<SG::MMA_64x128x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
    consumer_sync();
  }
#endif

#ifdef MK_GEMM_DIRECT_BF16_EPILOGUE
  if (!(flags & (4 | 8 | 16 | 32 | 256 | 1024 | 2048 | 8192))) {
    bf16* C = reinterpret_cast<bf16*>(Cp);
    const int w = wtid / 32, l = wtid & 31;
    const int cb = (l & 3) * 2;
#pragma unroll
    for (int n8 = 0; n8 < 16; ++n8) {
      const int c = n8 * 8 + cb;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + c;
        __nv_bfloat162 out;
        out.x = f2bf(d[n8 * 4 + i * 2 + 0]);
        out.y = f2bf(d[n8 * 4 + i * 2 + 1]);
        *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
      }
    }
    return;
  }
#endif

  float* Cs = reinterpret_cast<float*>(smem_raw);
  const int w = wtid / 32, l = wtid % 32;
  {
    const int r = wg * 64 + w * 16 + l / 4;
    const int cb = (l % 4) * 2;
#pragma unroll
    for (int n8 = 0; n8 < 16; ++n8)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          Cs[(r + 8 * i) * WG_LDC_N128 + n8 * 8 + cb + j] = d[n8 * 4 + i * 2 + j];
  }
  consumer_sync();
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int gid = tid + g * 256;
    const int m = gid / 16, c8 = (gid % 16) * 8;
    const int64_t idx = (int64_t)(m0 + m) * N + n0 + c8;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = Cs[m * WG_LDC_N128 + c8 + e];
    if (Res) {
      const uint4 rv = *reinterpret_cast<const uint4*>(&Res[idx]);
      const bf16* re = reinterpret_cast<const bf16*>(&rv);
#pragma unroll
      for (int e = 0; e < 8; ++e) v[e] += bf2f(re[e]);
    }
    if (c_f32) {
      float* C = reinterpret_cast<float*>(Cp);
      if (flags & 32) {
#ifdef MK_HEAD_DX_SKR
        if (flags & 32768) {  // SKR: plain stores to this slice's partial slab
          float* Cs32 = C + (int64_t)slice * M * N;
          float4 o0 = make_float4(v[0], v[1], v[2], v[3]);
          float4 o1 = make_float4(v[4], v[5], v[6], v[7]);
          *reinterpret_cast<float4*>(&Cs32[idx]) = o0;
          *reinterpret_cast<float4*>(&Cs32[idx + 4]) = o1;
        } else
#endif
        {
#pragma unroll
          for (int e = 0; e < 8; ++e) atomicAdd(&C[idx + e], v[e]);
        }
      } else {
        float4 o0 = make_float4(v[0], v[1], v[2], v[3]);
        float4 o1 = make_float4(v[4], v[5], v[6], v[7]);
        *reinterpret_cast<float4*>(&C[idx]) = o0;
        *reinterpret_cast<float4*>(&C[idx + 4]) = o1;
      }
    } else {
      bf16* C = reinterpret_cast<bf16*>(Cp);
      uint4 out;
      bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
      for (int e = 0; e < 8; ++e) oe[e] = f2bf(v[e]);
      *reinterpret_cast<uint4*>(&C[idx]) = out;
    }
    if (flags & 8192) {  // ssq partials from post-residual v[] (see m64n64 version)
      float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
      const int nparts = I.args[10];
      float ss = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float zv = bf2f(f2bf(v[e]));
        ss += zv * zv;
      }
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if ((gid & 7) == 0) parts[(int64_t)(m0 + m) * nparts + n0 / WG_BN + (c8 >= 64)] = ss;
    }
  }

  if (flags & 2048) {  // CE/LSE partials over both 64-col halves (see m64n64 version)
    float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
    const int nparts = I.args[10];
    const int nb = n0 / WG_BN;
    const int warp = tid / 32, lane = tid % 32;
    for (int r = warp; r < WG_BM; r += 8) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float mx = -INFINITY, se = 0.0f;
        for (int cc = lane; cc < WG_BN; cc += 32) {
          const float zv = bf2f(f2bf(Cs[r * WG_LDC_N128 + half * WG_BN + cc]));
          if (zv > mx) {
            se = se * lmhead_exp(mx - zv) + 1.0f;
            mx = zv;
          } else {
            se += lmhead_exp(zv - mx);
          }
        }
        for (int o = 16; o > 0; o >>= 1) {
          const float om = __shfl_xor_sync(0xffffffff, mx, o);
          const float os = __shfl_xor_sync(0xffffffff, se, o);
          const float Mx = fmaxf(mx, om);
          se = (mx == -INFINITY && om == -INFINITY) ? 0.0f
                                                     : se * lmhead_exp(mx - Mx) +
                                                           os * lmhead_exp(om - Mx);
          mx = Mx;
        }
        if (lane == 0) {
          const int64_t o = ((int64_t)(m0 + r) * nparts + nb + half) * 2;
          parts[o] = mx;
          parts[o + 1] = se;
        }
      }
    }
  }

  if (flags & 256) {  // qk-RMSNorm + RoPE over the two 64-col heads in this n128 tile
    const int nq_ = I.args[16], nkv_ = I.args[17], D_ = I.args[18];
    if (D_ != WG_BN) return;
    const float eps = __int_as_float(I.args[19]);
    bf16* qkvr = reinterpret_cast<bf16*>(bufs[I.args[15]]);
    const int warp = tid / 32, lane = tid % 32;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int n_base = n0 + half * WG_BN;
      const int head = n_base / D_;
      if (head >= nq_ + nkv_) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int gid = tid + g * 256;
          const int m = gid / 8, c8 = (gid % 8) * 8;
          uint4 out;
          bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
          for (int e = 0; e < 8; ++e) oe[e] = f2bf(Cs[m * WG_LDC_N128 + half * WG_BN + c8 + e]);
          *reinterpret_cast<uint4*>(&qkvr[(int64_t)(m0 + m) * N + n_base + c8]) = out;
        }
      } else {
        const bool is_q = head < nq_;
        const bf16* w_ = reinterpret_cast<const bf16*>(bufs[is_q ? I.args[9] : I.args[10]]);
        float* rstd = reinterpret_cast<float*>(bufs[is_q ? I.args[11] : I.args[12]]);
        const float* cosr = reinterpret_cast<const float*>(bufs[I.args[13]]);
        const float* sinr = reinterpret_cast<const float*>(bufs[I.args[14]]);
        const int hd = is_q ? head : head - nq_;
        for (int r = warp; r < WG_BM; r += 8) {
          const int gm = m0 + r;
          const float x0 = Cs[r * WG_LDC_N128 + half * WG_BN + lane];
          const float x1 = Cs[r * WG_LDC_N128 + half * WG_BN + lane + 32];
          float ss = x0 * x0 + x1 * x1;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
          const float rs = rsqrtf(ss / D_ + eps);
          if (lane == 0) rstd[(int64_t)gm * (is_q ? nq_ : nkv_) + hd] = rs;
          const float a = x0 * rs * bf2f(w_[lane]);
          const float b = x1 * rs * bf2f(w_[lane + 32]);
          const float cv = cosr[(int64_t)gm * 32 + lane], sv = sinr[(int64_t)gm * 32 + lane];
          qkvr[(int64_t)gm * N + n_base + lane] = f2bf(a * cv - b * sv);
          qkvr[(int64_t)gm * N + n_base + lane + 32] = f2bf(b * cv + a * sv);
        }
      }
    }
  }
}

// Generic m64n64 (128x64-tile) wgmma body. Under MK_GEMM_D64_TMA it is
// __noinline__: the TMA additions otherwise reflow the one-giant-function
// dispatch frame (same-binary control measured ~+30us/step at S3072 from the
// compile-in alone — the STACK-IS-NOT-RUNTIME codegen-shape class); isolating
// the fat frame is the P1 __noinline__ law. Without the knob it stays
// __forceinline__ so the no-TMA binary keeps today's codegen exactly.
#ifdef MK_GEMM_D64_TMA
__device__ __noinline__
#else
__device__ __forceinline__
#endif
void op_gemm_wgmma_n64_impl(const Instr& I, int tile, void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  void* Cp = bufs[I.args[2]];
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  const bool acc_c = flags & 4, c_f32 = flags & 8;
  const bf16* Res = (flags & 16) ? reinterpret_cast<const bf16*>(bufs[I.args[7]]) : nullptr;

  // SW128 swizzle phase = absolute smem address bits [7,10): slab bases must be
  // 1024B-aligned (ws mode offsets opsmem by MK_WS_CTRL_BYTES; df base is unpadded).
  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
#ifdef MK_GEMM_MBAR_RING
  constexpr int WG_MBAR_STAGES = 4;
  WgmmaSmemT<WG_MBAR_STAGES>& S =
      *reinterpret_cast<WgmmaSmemT<WG_MBAR_STAGES>*>(smem_raw);
#else
  WgmmaSmem& S = *reinterpret_cast<WgmmaSmem*>(smem_raw);
#endif
  const bool a_t = flags & 1, b_t = flags & 2;  // storage: a_t -> A[K,M]; b_t -> B[N,K]
  const int sk = (flags & 32) ? I.args[8] : 1;
  const int slice = tile % sk;
  const int mn = tile / sk;
  const int n_tiles = N / WG_BN;
  const int m0 = (mn / n_tiles) * WG_BM;
  const int n0 = (mn % n_tiles) * WG_BN;
  const int kchunk = ((K + sk * WG_BK - 1) / (sk * WG_BK)) * WG_BK;
  const int k_lo = slice * kchunk;
  const int k_hi = min(K, k_lo + kchunk);
  const int tid = mk_tid();
  const int wg = tid / 128;  // warpgroup = row half
  const int wtid = tid % 128;
  if (k_lo >= K) return;

  auto issue_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // A: 128r x 64k = 1024 16B vectors
      const int v = tid + i * 256;
      if (!a_t) {  // A[M,K], K-contiguous -> SW128 K-major slab per 64-row half
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(S.A[st][r / 64]) + wg_koff_sw(r % 64, k8),
            &A[(int64_t)(m0 + r) * K + k0 + k8], 16);
      } else {  // A[K,M], M-contiguous -> SW128 MN-major slab
        const int h = v / 512, w_ = v % 512;
        const int k = w_ / 8, m8 = (w_ % 8) * 8;
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(S.A[st][h]) + wg_mnoff_sw(k, m8),
            &A[(int64_t)(k0 + k) * M + m0 + h * 64 + m8], 16);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // B: 64r x 64k = 512 16B vectors
      const int v = tid + i * 256;
      if (b_t) {  // B[N,K], K-contiguous -> SW128 K-major slab
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + wg_koff_sw(r, k8),
                                &B[(int64_t)(n0 + r) * K + k0 + k8], 16);
      } else {  // B[K,N], N-contiguous -> SW128 MN-major slab
        const int k = v / 8, n8 = (v % 8) * 8;
        __pipeline_memcpy_async(reinterpret_cast<char*>(S.B[st]) + wg_mnoff_sw(k, n8),
                                &B[(int64_t)(k0 + k) * N + n0 + n8], 16);
      }
    }
    __pipeline_commit();
  };

  // Branch-free wgmma accumulate chain (ScaleOut::One over zeroed regs; a data-
  // dependent scale made ptxas serialize every wgmma — see wgmma_probe.py).
  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.0f;
  const int iters = (k_hi - k_lo) / WG_BK;
#ifdef MK_GEMM_MBAR_RING
  constexpr int WG_MBAR_LEAD = WG_MBAR_STAGES - 2;
  uint64_t* bfull = reinterpret_cast<uint64_t*>(smem_raw + sizeof(S));
  uint64_t* bempty = bfull + WG_MBAR_STAGES;
#ifdef MK_GEMM_D64_TMA
  // D64 ring TMA feed (round-4 port): same args[20..22] contract as the n256
  // port (disjoint row sets — this body never sees bit14 rows). All four
  // storage majors are SW128 128B-row slabs: A = one {64k,128m} box (!a_t) or
  // two {64m,64k} MN boxes (a_t); B = one {64k,64n} box (b_t) or one {64n,64k}
  // MN box (!b_t). bfull is a count-1 expect_tx barrier. Fully separate loop —
  // see the n128 body's comment (merged-branch codegen tax).
  if (I.args[20] > 0) {
    const char* tbl = reinterpret_cast<const char*>(bufs[I.args[20] - 1]);
    const char* tmA = tbl + (int64_t)I.args[21] * 128;
    const char* tmB = tbl + (int64_t)I.args[22] * 128;
    if (tid == 0) {
      wg_tmap_fence_acquire(tmA);
      wg_tmap_fence_acquire(tmB);
#pragma unroll
      for (int s = 0; s < WG_MBAR_STAGES; ++s) {
        wg_mbar_init(&bfull[s], 1);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
#ifdef MK_PDF_D64_TMA_FEED
    const bool pdf_feed = g_pdf_feed.active;
    if (pdf_feed && tid == 0) {
      MkPdfFeed& F = g_pdf_feed;
      F.tmA = tmA;
      F.tmB = tmB;
      F.a0 = reinterpret_cast<char*>(S.A[0][0]);
      F.a1 = reinterpret_cast<char*>(S.A[0][1]);
      F.b0 = reinterpret_cast<char*>(S.B[0]);
      F.a_stride = (WG_MBAR_STAGES > 1)
          ? (int)(reinterpret_cast<char*>(S.A[1][0]) - reinterpret_cast<char*>(S.A[0][0]))
          : 0;
      F.b_stride = (WG_MBAR_STAGES > 1)
          ? (int)(reinterpret_cast<char*>(S.B[1]) - reinterpret_cast<char*>(S.B[0]))
          : 0;
      F.bfull = bfull;
      F.bempty = bempty;
      F.m0 = m0;
      F.n0 = n0;
      F.iters = iters;
      F.stages = WG_MBAR_STAGES;
      F.a_t = a_t ? 1 : 0;
      F.b_t = b_t ? 1 : 0;
      F.bk = WG_BK;
      F.k_base = k_lo;
      F.kind = 2;
      F.expect_bytes = 24576;
      mk_pdf_st_release(&F.seq, F.seq + 1);
    }
#endif
    auto issue_stage_tma = [&](int t) {
      if (tid == 0) {
        const int st = t % WG_MBAR_STAGES;
        const int k0 = k_lo + t * WG_BK;
        wg_mbar_expect_tx(&bfull[st], 24576);  // A 16KB + B 8KB per stage
        if (a_t) {
          wg_tma_load_2d(tmA, S.A[st][0], m0, k0, &bfull[st]);
          wg_tma_load_2d(tmA, S.A[st][1], m0 + 64, k0, &bfull[st]);
        } else {
          wg_tma_load_2d(tmA, S.A[st][0], k0, m0, &bfull[st]);
        }
        if (b_t) {
          wg_tma_load_2d(tmB, S.B[st], k0, n0, &bfull[st]);
        } else {
          wg_tma_load_2d(tmB, S.B[st], n0, k0, &bfull[st]);
        }
      }
    };
#ifdef MK_PDF_D64_TMA_FEED
    if (!pdf_feed)
#endif
    {
      for (int p = 0; p < min(WG_MBAR_LEAD + 1, iters); ++p) issue_stage_tma(p);
    }
    for (int t = 0; t < iters; ++t) {
      const int st = t % WG_MBAR_STAGES;
      wg_mbar_wait(&bfull[st], (t / WG_MBAR_STAGES) & 1);
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = a_t ? wg_desc_mnsw(S.A[st][wg], s) : wg_desc_ksw(S.A[st][wg], s);
        db[s] = b_t ? wg_desc_ksw(S.B[st], s) : wg_desc_mnsw(S.B[st], s);
      }
      if (!a_t && b_t)
        wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
      else if (!a_t && !b_t)
        wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
      else if (a_t && b_t)
        wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>>(da, db, d);
      else
        wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(da, db, d);
      wg_mbar_arrive(&bempty[st]);
#ifdef MK_PDF_D64_TMA_FEED
      if (!pdf_feed)
#endif
      {
        const int tn = t + WG_MBAR_LEAD + 1;
        if (tn < iters) {
          if (tn >= WG_MBAR_STAGES)
            wg_mbar_wait(&bempty[tn % WG_MBAR_STAGES], (tn / WG_MBAR_STAGES - 1) & 1);
          issue_stage_tma(tn);
        }
      }
    }
    cute::warpgroup_wait<0>();
    consumer_sync();
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < WG_MBAR_STAGES; ++s) {
        wg_mbar_init(&bfull[s], 1);
        wg_mbar_init(&bempty[s], 256);
      }
    }
    consumer_sync();
  } else {
#endif
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < WG_MBAR_STAGES; ++s) {
      wg_mbar_init(&bfull[s], 256);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
  auto issue_stage_mb = [&](int t) {
    const int st = t % WG_MBAR_STAGES;
    issue_stage(k_lo + t * WG_BK, st);
    wg_mbar_arrive_cpasync(&bfull[st]);
  };
  for (int p = 0; p < min(WG_MBAR_LEAD + 1, iters); ++p) issue_stage_mb(p);
  for (int t = 0; t < iters; ++t) {
    const int st = t % WG_MBAR_STAGES;
    wg_mbar_wait(&bfull[st], (t / WG_MBAR_STAGES) & 1);
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = a_t ? wg_desc_mnsw(S.A[st][wg], s) : wg_desc_ksw(S.A[st][wg], s);
      db[s] = b_t ? wg_desc_ksw(S.B[st], s) : wg_desc_mnsw(S.B[st], s);
    }
    if (!a_t && b_t)
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
    else if (!a_t && !b_t)
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
    else if (a_t && b_t)
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>>(da, db, d);
    else
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(da, db, d);
    wg_mbar_arrive(&bempty[st]);
    const int tn = t + WG_MBAR_LEAD + 1;
    if (tn < iters) {
      if (tn >= WG_MBAR_STAGES)
        wg_mbar_wait(&bempty[tn % WG_MBAR_STAGES], (tn / WG_MBAR_STAGES - 1) & 1);
      issue_stage_mb(tn);
    }
  }
  cute::warpgroup_wait<0>();
  consumer_sync();
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < WG_MBAR_STAGES; ++s) {
      wg_mbar_init(&bfull[s], 256);
      wg_mbar_init(&bempty[s], 256);
    }
  }
  consumer_sync();
#ifdef MK_GEMM_D64_TMA
  }
#endif
#else
  issue_stage(k_lo, 0);
  for (int t = 0; t < iters; ++t) {
    if (t + 1 < iters) issue_stage(k_lo + (t + 1) * WG_BK, (t + 1) & 1);
    __pipeline_wait_prior(t + 1 < iters ? 1 : 0);
    consumer_sync();
    uint64_t da[4], db[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      da[s] = a_t ? wg_desc_mnsw(S.A[t & 1][wg], s) : wg_desc_ksw(S.A[t & 1][wg], s);
      db[s] = b_t ? wg_desc_ksw(S.B[t & 1], s) : wg_desc_mnsw(S.B[t & 1], s);
    }
    if (!a_t && b_t)
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
    else if (!a_t && !b_t)
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
    else if (a_t && b_t)
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>>(da, db, d);
    else
      wg_mma_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>>(da, db, d);
    consumer_sync();  // both warpgroups done reading before the buffer is refilled
  }
#endif

#ifdef MK_GEMM_DIRECT_BF16_EPILOGUE
  if (!(flags & (4 | 8 | 16 | 32 | 256 | 1024 | 2048 | 8192))) {
    bf16* C = reinterpret_cast<bf16*>(Cp);
    const int w = wtid / 32, l = wtid & 31;
    const int cb = (l & 3) * 2;
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8) {
      const int c = n8 * 8 + cb;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + c;
        __nv_bfloat162 out;
        out.x = f2bf(d[n8 * 4 + i * 2 + 0]);
        out.y = f2bf(d[n8 * 4 + i * 2 + 1]);
        *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
      }
    }
    return;
  }
#endif

#ifdef MK_DROW_REG_EPILOGUE
  if ((flags & 1024) && I.args[11] == WG_BN) {
    bf16* C = reinterpret_cast<bf16*>(Cp);
    const bf16* Oatt = reinterpret_cast<const bf16*>(bufs[I.args[9]]);
    float* drow = reinterpret_cast<float*>(bufs[I.args[10]]);
    const int w = wtid / 32, l = wtid & 31;
    const int cb = (l & 3) * 2;
    float drow_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
        const int64_t idx = (int64_t)(m0 + r) * N + n0 + n8 * 8 + cb;
        const bf16 z0 = f2bf(d[n8 * 4 + i * 2 + 0]);
        const bf16 z1 = f2bf(d[n8 * 4 + i * 2 + 1]);
        __nv_bfloat162 out;
        out.x = z0;
        out.y = z1;
        *reinterpret_cast<__nv_bfloat162*>(&C[idx]) = out;
        drow_sum[i] += bf2f(z0) * bf2f(Oatt[idx]) + bf2f(z1) * bf2f(Oatt[idx + 1]);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float s = drow_sum[i];
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      if ((l & 3) == 0) {
        const int r = wg * 64 + w * 16 + l / 4 + 8 * i;
#ifdef MK_DROW_DIRECT_STORE
        if (M < 2048)
          drow[(int64_t)(n0 / WG_BN) * M + m0 + r] = s;
        else
#endif
          atomicAdd(&drow[(int64_t)(n0 / WG_BN) * M + m0 + r], s);
      }
    }
    return;
  }
#endif

  // stage accumulators to smem (over the dead A/B buffers), then coalesced epilogue
  float* Cs = reinterpret_cast<float*>(smem_raw);
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
          Cs[(r + 8 * i) * WG_LDC + n8 * 8 + cb + j] = d[n8 * 4 + i * 2 + j];
  }
  consumer_sync();
  // 128x64 outputs as 8-element groups: 1024 groups, 4 per thread, fully coalesced
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const int gid = tid + g * 256;
    const int m = gid / 8, c8 = (gid % 8) * 8;
    const int gm = m0 + m, gn = n0 + c8;
    const int64_t idx = (int64_t)gm * N + gn;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = Cs[m * WG_LDC + c8 + e];
    if (Res) {
      const uint4 rv = *reinterpret_cast<const uint4*>(&Res[idx]);
      const bf16* re = reinterpret_cast<const bf16*>(&rv);
#pragma unroll
      for (int e = 0; e < 8; ++e) v[e] += bf2f(re[e]);
    }
    if (flags & 32) {  // split-K: concurrent slices accumulate with fp32 atomics
      float* C = reinterpret_cast<float*>(Cp);
#pragma unroll
      for (int e = 0; e < 8; ++e) atomicAdd(&C[idx + e], v[e]);
    } else if (c_f32) {
      float* C = reinterpret_cast<float*>(Cp);
      if (acc_c) {
#pragma unroll
        for (int e = 0; e < 8; ++e) C[idx + e] += v[e];
      } else {
        float4 o0 = make_float4(v[0], v[1], v[2], v[3]);
        float4 o1 = make_float4(v[4], v[5], v[6], v[7]);
        *reinterpret_cast<float4*>(&C[idx]) = o0;
        *reinterpret_cast<float4*>(&C[idx + 4]) = o1;
      }
    } else {
      bf16* C = reinterpret_cast<bf16*>(Cp);
      if (acc_c) {
        const uint4 cv = *reinterpret_cast<const uint4*>(&C[idx]);
        const bf16* ce = reinterpret_cast<const bf16*>(&cv);
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] += bf2f(ce[e]);
      }
      uint4 out;
      bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
      for (int e = 0; e < 8; ++e) oe[e] = f2bf(v[e]);
      *reinterpret_cast<uint4*>(&C[idx]) = out;
    }
    if (flags & 8192) {
      // Fused ssq partials (bit13, wo/down gemms feeding rmsnorm): per-row sum of
      // squares of the bf16-ROUNDED POST-RESIDUAL values (= exactly what
      // rmsnorm_fwd would read back). 8 consecutive lanes share a row: butterfly
      // octet reduce, one plain store per (row, 64-col block).
      // args: 9 = ssq partials fp32 [M, nparts], 10 = nparts (= N/64).
      float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
      const int nparts = I.args[10];
      float ss = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float zv = bf2f(f2bf(v[e]));
        ss += zv * zv;
      }
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if ((gid & 7) == 0) parts[(int64_t)gm * nparts + n0 / WG_BN] = ss;
    }
  }

  // Fused Drow epilogue (flags bit10, dOatt = dX @ Wo): drow[qh, s] += sum_d dO*O over
  // this tile's 64 columns (fp32 atomics; drow pre-zeroed). WG_BN == 64 covers one
  // whole head at D=64 (a half-head partial at D=128 — the atomics accumulate both).
  // Values re-rounded through bf16 to match what the standalone op read back from the
  // dOatt buffer. args: 9 = oatt, 10 = drow, 11 = D. No residual/split-K/acc co-use.
  if (flags & 1024) {
    const bf16* Oatt = reinterpret_cast<const bf16*>(bufs[I.args[9]]);
    float* drow = reinterpret_cast<float*>(bufs[I.args[10]]);
    const int D = I.args[11];
    const int warp = tid / 32, lane = tid % 32;
    for (int r = warp; r < WG_BM; r += 8) {  // warp per row, block-wide
      float s = 0.0f;
      for (int d = lane; d < WG_BN; d += 32)
        s += bf2f(f2bf(Cs[r * WG_LDC + d])) * bf2f(Oatt[(int64_t)(m0 + r) * N + n0 + d]);
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
      if (lane == 0) {
#ifdef MK_DROW_DIRECT_STORE
        if (D == WG_BN && M < 2048)
          drow[(int64_t)(n0 / D) * M + m0 + r] = s;
        else
#endif
          atomicAdd(&drow[(int64_t)(n0 / D) * M + m0 + r], s);
      }
    }
  }

  // Fused CE/LSE partials (flags bit11, lm_head gemm): per-row online (max, sumexp)
  // over this tile's 64 columns, computed from the bf16-ROUNDED staged values so the
  // reduction sees exactly what OP_CE_FWD would have read back from the logits buffer.
  // args: 9 = partials [M, N/64, 2] fp32, 10 = nparts (= N/64). OP_CE_FWD then reduces
  // nparts pairs per row instead of rescanning V logits.
  if (flags & 2048) {
    float* parts = reinterpret_cast<float*>(bufs[I.args[9]]);
    const int nparts = I.args[10];
    const int nb = n0 / WG_BN;
    const int warp = tid / 32, lane = tid % 32;
    for (int r = warp; r < WG_BM; r += 8) {
      float mx = -INFINITY, se = 0.0f;
      for (int cc = lane; cc < WG_BN; cc += 32) {
        const float zv = bf2f(f2bf(Cs[r * WG_LDC + cc]));
        // __expf: register-lean SFU intrinsic — this runs inside the gemm's register
        // context, where libm expf spills the whole interpreter past 128 regs/thread
        // (1 block/SM). Precision loss is far below bf16 logit noise.
        if (zv > mx) {
          se = se * lmhead_exp(mx - zv) + 1.0f;
          mx = zv;
        } else {
          se += lmhead_exp(zv - mx);
        }
      }
      for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffff, mx, o);
        const float os = __shfl_xor_sync(0xffffffff, se, o);
        const float Mx = fmaxf(mx, om);
        se = (mx == -INFINITY && om == -INFINITY) ? 0.0f
                                                  : se * lmhead_exp(mx - Mx) +
                                                        os * lmhead_exp(om - Mx);
        mx = Mx;
      }
      if (lane == 0) {
        const int64_t o = ((int64_t)(m0 + r) * nparts + nb) * 2;
        parts[o] = mx;
        parts[o + 1] = se;
      }
    }
  }

  // Fused per-head qk-RMSNorm + RoPE epilogue (flags bit8): with WG_BN == 64 == D each
  // tile covers exactly one head, so the norm is tile-local over the fp32 staging.
  // args: 9=qw 10=kw 11=rstd_q 12=rstd_k 13=cos 14=sin 15=qkvr 16=nq 17=nkv 18=D 19=eps
  if (flags & 256) {
    const int nq_ = I.args[16], nkv_ = I.args[17], D_ = I.args[18];
    const float eps = __int_as_float(I.args[19]);
    bf16* qkvr = reinterpret_cast<bf16*>(bufs[I.args[15]]);
    const int head = n0 / D_;
    const int warp = tid / 32, lane = tid % 32;
    if (head >= nq_ + nkv_) {  // v head: pass through
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int gid = tid + g * 256;
        const int m = gid / 8, c8 = (gid % 8) * 8;
        uint4 out;
        bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
        for (int e = 0; e < 8; ++e) oe[e] = f2bf(Cs[m * WG_LDC + c8 + e]);
        *reinterpret_cast<uint4*>(&qkvr[(int64_t)(m0 + m) * N + n0 + c8]) = out;
      }
    } else {
      const bool is_q = head < nq_;
      const bf16* w_ = reinterpret_cast<const bf16*>(bufs[is_q ? I.args[9] : I.args[10]]);
      float* rstd = reinterpret_cast<float*>(bufs[is_q ? I.args[11] : I.args[12]]);
      const float* cosr = reinterpret_cast<const float*>(bufs[I.args[13]]);
      const float* sinr = reinterpret_cast<const float*>(bufs[I.args[14]]);
      const int hd = is_q ? head : head - nq_;
      for (int r = warp; r < WG_BM; r += 8) {  // warp per row; lane = rope pair (l, l+32)
        const int gm = m0 + r;
        const float x0 = Cs[r * WG_LDC + lane];
        const float x1 = Cs[r * WG_LDC + lane + 32];
        float ss = x0 * x0 + x1 * x1;
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
        const float rs = rsqrtf(ss / D_ + eps);
        if (lane == 0) rstd[(int64_t)gm * (is_q ? nq_ : nkv_) + hd] = rs;
        const float a = x0 * rs * bf2f(w_[lane]);
        const float b = x1 * rs * bf2f(w_[lane + 32]);
        const float cv = cosr[(int64_t)gm * 32 + lane], sv = sinr[(int64_t)gm * 32 + lane];
        qkvr[(int64_t)gm * N + n0 + lane] = f2bf(a * cv - b * sv);
        qkvr[(int64_t)gm * N + n0 + lane + 32] = f2bf(b * cv + a * sv);
      }
    }
  }
}

__device__ void op_gemm_wgmma(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int flags = I.args[6];
  if (flags & 16384) {
    if (flags & 2) {
#ifdef MK_GEMM_N256_NT_SUPERTILE
      if (flags & GEMM_N256_NT_SUPERTILE_FLAG) {
        op_gemm_wgmma_n256_nt_supertile_impl<3>(I, tile, bufs, smem_raw);
        return;
      }
#endif
      if (flags & GEMM_N256_STAGE3_FLAG)
        op_gemm_wgmma_n256_direct_impl<3>(I, tile, bufs, smem_raw);
      else
        op_gemm_wgmma_n256_direct_impl<2>(I, tile, bufs, smem_raw);
    } else {
#ifdef MK_GEMM_N256_HEAD_DX_EXACT
      if (I.args[3] == 1024 && I.args[4] == 2560 && I.args[5] == 151936 &&
          (flags & 8) && (flags & GEMM_N256_STAGE3_FLAG) &&
          (flags & GEMM_N256_NMAJOR_FLAG) && I.args[20] > 0 &&
          !(flags & (1 | 2 | 4 | 16 | 32 | 256 | 1024 | 2048 | 8192))) {
#ifdef MK_GEMM_N256_HEAD_DX_PDFONLY
        if (g_pdf_feed.active) {
          op_gemm_wgmma_n256_head_dx_exact_impl(I, tile, bufs, smem_raw);
          return;
        }
#else
        op_gemm_wgmma_n256_head_dx_exact_impl(I, tile, bufs, smem_raw);
        return;
#endif
      }
#endif
      if (flags & GEMM_N256_STAGE3_FLAG) {
        if (flags & 1024) {
          op_gemm_wgmma_n256_nn_f32_impl<3, true>(I, tile, bufs, smem_raw);
        } else {
          op_gemm_wgmma_n256_nn_f32_impl<3, false>(I, tile, bufs, smem_raw);
        }
      } else {
        if (flags & 1024) {
          op_gemm_wgmma_n256_nn_f32_impl<2, true>(I, tile, bufs, smem_raw);
        } else {
          op_gemm_wgmma_n256_nn_f32_impl<2, false>(I, tile, bufs, smem_raw);
        }
      }
    }
    return;
  }
  if (flags & 4096) {
    op_gemm_wgmma_n128(I, tile, bufs, smem_raw);
    return;
  }
  op_gemm_wgmma_n64_impl(I, tile, bufs, smem_raw);
}

// ---- fp32 -> bf16 convert (drains split/atomic fp32 workspaces) -------------------------
// args: {src_f32, dst_bf16, n}; tile = MK_CHUNK-element chunk.
__device__ __forceinline__ void op_cvt_f32_bf16(const Instr& I, int tile, void** bufs) {
  const float* src = reinterpret_cast<const float*>(bufs[I.args[0]]);
  bf16* dst = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  const int n = I.args[2];
  const int base = tile * MK_CHUNK;
  for (int i = base + mk_tid(); i < min(base + MK_CHUNK, n); i += MK_CONSUMERS)
    dst[i] = f2bf(src[i]);
}

// ---- batched row ops (v3 P6) -----------------------------------------------------------
// Row ops process MK_ROW_R rows per work item: warp-shuffle reductions (block_sum's 3
// block barriers per row are gone), uint4-vectorized bf16 IO (8 elems per lane per
// iteration) for memory-level parallelism — at 1 block/SM the old 1-row tiles were
// latency-bound (256 threads striding a 256..512-element row, serial row-to-row).
// v3 P4b: TWO rows per warp (r and r+8), loads interleaved — the nsys counters put
// the interpreter at SM-issue 19% with only 8 warps resident: each warp needs more
// independent load streams, not more claims. rmsnorm_bwd folds both rows into one
// smem dw slot, so its atomic tail halves too.
#define MK_ROW_R 8    // swiglu/qknorm rows per tile (one warp per row; mk.ROWOP_R)
#define MK_ROW_R2 16  // rmsnorm rows per tile: 2 rows/warp interleaved (mk.ROWOP_R2)

__device__ __forceinline__ void ld8bf(const bf16* p, float* f) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float2 v = __bfloat1622float2(h[j]);
    f[2 * j] = v.x;
    f[2 * j + 1] = v.y;
  }
}

__device__ __forceinline__ void st8bf(bf16* p, const float* f) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int j = 0; j < 4; j++) h[j] = __float22bfloat162_rn(make_float2(f[2 * j], f[2 * j + 1]));
  *reinterpret_cast<uint4*>(p) = u;
}

// 8-elem dy loader: bf16 activations or the fp32 atomic-workspace view (dy_f32 paths)
__device__ __forceinline__ void ld8dy(const bf16* pb, const float* pf, bool f32, int i,
                                      float* f) {
  if (f32) {
    const float4 a = *reinterpret_cast<const float4*>(pf + i);
    const float4 b = *reinterpret_cast<const float4*>(pf + i + 4);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
    f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
  } else {
    ld8bf(pb + i, f);
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <bool UseFma>
__device__ __forceinline__ float rms_dx_acc(float dx, float r, float g, float xh,
                                            float m) {
  if constexpr (UseFma) {
    return fmaf(r, fmaf(-xh, m, g), dx);
  }
  return dx + r * (g - xh * m);
}

// ---- RMSNorm ---------------------------------------------------------------------------
// fwd: y[r,:] = x[r,:] * rstd * w ; rstd = 1/sqrt(mean(x^2)+eps). Saves rstd (fp32).
// args: {x, w, y, rstd, H, eps_bits, S}; tile = MK_ROW_R-row group, one warp per row.
__device__ void op_rmsnorm_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int H = I.args[4], S = I.args[6];
  const float eps = __int_as_float(I.args[5]);
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  if (rowA >= S) return;  // barrier-free op: early exit is safe
  const bool hasB = rowA + 8 < S;
  const int rowB = hasB ? rowA + 8 : rowA;  // tail: compute A twice, store B never
  const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* xA = xb + (int64_t)rowA * H;
  const bf16* xB = xb + (int64_t)rowB * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  bf16* yb = reinterpret_cast<bf16*>(bufs[I.args[2]]);
  bf16* yA = yb + (int64_t)rowA * H;
  bf16* yB = yb + (int64_t)rowB * H;
  float* rstd = reinterpret_cast<float*>(bufs[I.args[3]]);

  float ssA = 0.0f, ssB = 0.0f;
  const int nparts = I.args[8];  // > 0: producer-gemm ssq partials (bit13 epilogue)
  if (nparts > 0) {
    // variance pass from partials: nparts (= H/64 <= 8ish) floats per row instead
    // of re-reading the H-wide row — half the op's loads and no long dot chain.
    const float* parts = reinterpret_cast<const float*>(bufs[I.args[7]]);
    if (lane < nparts) {
      ssA = parts[(int64_t)rowA * nparts + lane];
      ssB = parts[(int64_t)rowB * nparts + lane];
    }
  } else if ((H & 7) == 0) {
    for (int i = lane * 8; i < H; i += 32 * 8) {
      float a[8], b[8];
      ld8bf(xA + i, a);
      ld8bf(xB + i, b);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ssA += a[j] * a[j];
        ssB += b[j] * b[j];
      }
    }
  } else {
    for (int i = lane; i < H; i += 32) {
      const float a = bf2f(xA[i]), b = bf2f(xB[i]);
      ssA += a * a;
      ssB += b * b;
    }
  }
  const float rA = rsqrtf(warp_sum(ssA) / H + eps);
  const float rB = rsqrtf(warp_sum(ssB) / H + eps);
  if (lane == 0) {
    rstd[rowA] = rA;
    if (hasB) rstd[rowB] = rB;
  }
  if ((H & 7) == 0) {
    for (int i = lane * 8; i < H; i += 32 * 8) {
      float a[8], b[8], wv[8], ya[8], yv[8];
      ld8bf(xA + i, a);
      ld8bf(xB + i, b);
      ld8bf(w + i, wv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ya[j] = a[j] * rA * wv[j];
        yv[j] = b[j] * rB * wv[j];
      }
      st8bf(yA + i, ya);
      if (hasB) st8bf(yB + i, yv);
    }
  } else {
    for (int i = lane; i < H; i += 32) {
      yA[i] = f2bf(bf2f(xA[i]) * rA * bf2f(w[i]));
      if (hasB) yB[i] = f2bf(bf2f(xB[i]) * rB * bf2f(w[i]));
    }
  }
}

// bwd: with xhat = x*rstd, g = dy*w:
//   dx += rstd * (g - xhat * mean(g * xhat))       (accumulates into dx: residual stream)
//   dw += dy * xhat — staged in smem per tile, ONE global atomic per element per
//   MK_ROW_R rows (per-row global atomics serialize on the tiny [H] grad buffer: every
//   row hits the same addresses, so S atomic updates per address bound the op's span).
// args: {x, w, dy, dx, dw, rstd, H, dy_f32, S}; tile = MK_ROW_R-row group, warp per row.
// dy_f32 != 0 reads dy as fp32 (an atomic-accumulation workspace — no CVT chain hop).
__device__ void op_rmsnorm_bwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int H = I.args[6], S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  float* dw_rows = reinterpret_cast<float*>(smem_raw);  // [8, H] fp32 partials (A+B folded)
  float* dw_row = dw_rows + (int64_t)warp * H;
  if (rowA < S) {
    const bool hasB = rowA + 8 < S;
    const int rowB = hasB ? rowA + 8 : rowA;  // tail: compute A twice, fold/store B never
    const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
    const bf16* xA = xb + (int64_t)rowA * H;
    const bf16* xB = xb + (int64_t)rowB * H;
    const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
    const bf16* dybA = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowA * H;
    const float* dyfA = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowA * H;
    const bf16* dybB = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowB * H;
    const float* dyfB = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowB * H;
    bf16* dxb = reinterpret_cast<bf16*>(bufs[I.args[3]]);
    bf16* dxA = dxb + (int64_t)rowA * H;
    bf16* dxB = dxb + (int64_t)rowB * H;
    const float rA = reinterpret_cast<const float*>(bufs[I.args[5]])[rowA];
    const float rB = reinterpret_cast<const float*>(bufs[I.args[5]])[rowB];
    const float foldB = hasB ? 1.0f : 0.0f;  // dw fold guard (A==B at the tail)
    if ((H & 7) == 0) {
      float dotA = 0.0f, dotB = 0.0f;
      for (int i = lane * 8; i < H; i += 32 * 8) {
        float xa[8], xv[8], da[8], db[8], wv[8];
        ld8bf(xA + i, xa);
        ld8bf(xB + i, xv);
        ld8dy(dybA, dyfA, dy_f32, i, da);
        ld8dy(dybB, dyfB, dy_f32, i, db);
        ld8bf(w + i, wv);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          dotA += da[j] * wv[j] * xa[j];
          dotB += db[j] * wv[j] * xv[j];
        }
      }
      const float mA = warp_sum(dotA) * rA / H;  // = mean(g * xhat)
      const float mB = warp_sum(dotB) * rB / H;
      for (int i = lane * 8; i < H; i += 32 * 8) {
        float xa[8], xv[8], da[8], db[8], wv[8], dxa[8], dxv[8];
        ld8bf(xA + i, xa);
        ld8bf(xB + i, xv);
        ld8dy(dybA, dyfA, dy_f32, i, da);
        ld8dy(dybB, dyfB, dy_f32, i, db);
        ld8bf(w + i, wv);
        ld8bf(dxA + i, dxa);
        ld8bf(dxB + i, dxv);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const float xhA = xa[j] * rA, xhB = xv[j] * rB;
          dxa[j] += rA * (da[j] * wv[j] - xhA * mA);
          dxv[j] += rB * (db[j] * wv[j] - xhB * mB);
          dw_row[i + j] = da[j] * xhA + foldB * (db[j] * xhB);
        }
        st8bf(dxA + i, dxa);
        if (hasB) st8bf(dxB + i, dxv);
      }
    } else {
      auto dyA = [&](int i) { return dy_f32 ? dyfA[i] : bf2f(dybA[i]); };
      auto dyB = [&](int i) { return dy_f32 ? dyfB[i] : bf2f(dybB[i]); };
      float dotA = 0.0f, dotB = 0.0f;
      for (int i = lane; i < H; i += 32) {
        dotA += dyA(i) * bf2f(w[i]) * bf2f(xA[i]);
        dotB += dyB(i) * bf2f(w[i]) * bf2f(xB[i]);
      }
      const float mA = warp_sum(dotA) * rA / H;
      const float mB = warp_sum(dotB) * rB / H;
      for (int i = lane; i < H; i += 32) {
        const float xhA = bf2f(xA[i]) * rA, xhB = bf2f(xB[i]) * rB;
        dxA[i] = f2bf(bf2f(dxA[i]) + rA * (dyA(i) * bf2f(w[i]) - xhA * mA));
        if (hasB) dxB[i] = f2bf(bf2f(dxB[i]) + rB * (dyB(i) * bf2f(w[i]) - xhB * mB));
        dw_row[i] = dyA(i) * xhA + foldB * (dyB(i) * xhB);
      }
    }
  } else {
    for (int i = lane; i < H; i += 32) dw_row[i] = 0.0f;
  }
  consumer_sync();
  float* dw = reinterpret_cast<float*>(bufs[I.args[4]]);
  for (int i = mk_tid(); i < H; i += MK_CONSUMERS) {
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < 8; ++r) s += dw_rows[(int64_t)r * H + i];
    atomicAdd(&dw[i], s);
  }
}

// Four-row dx-only fold for long H256 shapes: fewer row tiles without touching the
// cold dw sink. The normal two-row op remains the default for wider H shapes.
__device__ void op_rmsnorm_bwd_dx_r4(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int H = I.args[6], S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int row0 = tile * 32 + warp;
  if (row0 >= S) return;  // barrier-free op: early exit is safe
  const bool has1 = row0 + 8 < S;
  const bool has2 = row0 + 16 < S;
  const bool has3 = row0 + 24 < S;
  const int row1 = row0 + 8;
  const int row2 = row0 + 16;
  const int row3 = row0 + 24;
  const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* x0 = xb + (int64_t)row0 * H;
  const bf16* x1 = xb + (int64_t)(has1 ? row1 : row0) * H;
  const bf16* x2 = xb + (int64_t)(has2 ? row2 : row0) * H;
  const bf16* x3 = xb + (int64_t)(has3 ? row3 : row0) * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const bf16* dyb0 = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)row0 * H;
  const bf16* dyb1 = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)(has1 ? row1 : row0) * H;
  const bf16* dyb2 = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)(has2 ? row2 : row0) * H;
  const bf16* dyb3 = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)(has3 ? row3 : row0) * H;
  const float* dyf0 = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)row0 * H;
  const float* dyf1 = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)(has1 ? row1 : row0) * H;
  const float* dyf2 = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)(has2 ? row2 : row0) * H;
  const float* dyf3 = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)(has3 ? row3 : row0) * H;
  bf16* dxb = reinterpret_cast<bf16*>(bufs[I.args[3]]);
  bf16* dx0 = dxb + (int64_t)row0 * H;
  bf16* dx1 = dxb + (int64_t)(has1 ? row1 : row0) * H;
  bf16* dx2 = dxb + (int64_t)(has2 ? row2 : row0) * H;
  bf16* dx3 = dxb + (int64_t)(has3 ? row3 : row0) * H;
  const float* rstd = reinterpret_cast<const float*>(bufs[I.args[5]]);
  const float r0 = rstd[row0];
  const float r1 = has1 ? rstd[row1] : 0.0f;
  const float r2 = has2 ? rstd[row2] : 0.0f;
  const float r3 = has3 ? rstd[row3] : 0.0f;
  if ((H & 7) == 0) {
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    for (int i = lane * 8; i < H; i += 32 * 8) {
      float x[8], d[8], wv[8];
      ld8bf(w + i, wv);
      ld8bf(x0 + i, x);
      ld8dy(dyb0, dyf0, dy_f32, i, d);
#pragma unroll
      for (int j = 0; j < 8; j++) dot0 += d[j] * wv[j] * x[j];
      if (has1) {
        ld8bf(x1 + i, x);
        ld8dy(dyb1, dyf1, dy_f32, i, d);
#pragma unroll
        for (int j = 0; j < 8; j++) dot1 += d[j] * wv[j] * x[j];
      }
      if (has2) {
        ld8bf(x2 + i, x);
        ld8dy(dyb2, dyf2, dy_f32, i, d);
#pragma unroll
        for (int j = 0; j < 8; j++) dot2 += d[j] * wv[j] * x[j];
      }
      if (has3) {
        ld8bf(x3 + i, x);
        ld8dy(dyb3, dyf3, dy_f32, i, d);
#pragma unroll
        for (int j = 0; j < 8; j++) dot3 += d[j] * wv[j] * x[j];
      }
    }
    const float m0 = warp_sum(dot0) * r0 / H;
    const float m1 = warp_sum(dot1) * r1 / H;
    const float m2 = warp_sum(dot2) * r2 / H;
    const float m3 = warp_sum(dot3) * r3 / H;
    for (int i = lane * 8; i < H; i += 32 * 8) {
      float x[8], d[8], wv[8], dxv[8];
      ld8bf(w + i, wv);
      ld8bf(x0 + i, x);
      ld8dy(dyb0, dyf0, dy_f32, i, d);
      ld8bf(dx0 + i, dxv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float xh = x[j] * r0;
        dxv[j] += r0 * (d[j] * wv[j] - xh * m0);
      }
      st8bf(dx0 + i, dxv);
      if (has1) {
        ld8bf(x1 + i, x);
        ld8dy(dyb1, dyf1, dy_f32, i, d);
        ld8bf(dx1 + i, dxv);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const float xh = x[j] * r1;
          dxv[j] += r1 * (d[j] * wv[j] - xh * m1);
        }
        st8bf(dx1 + i, dxv);
      }
      if (has2) {
        ld8bf(x2 + i, x);
        ld8dy(dyb2, dyf2, dy_f32, i, d);
        ld8bf(dx2 + i, dxv);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const float xh = x[j] * r2;
          dxv[j] += r2 * (d[j] * wv[j] - xh * m2);
        }
        st8bf(dx2 + i, dxv);
      }
      if (has3) {
        ld8bf(x3 + i, x);
        ld8dy(dyb3, dyf3, dy_f32, i, d);
        ld8bf(dx3 + i, dxv);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const float xh = x[j] * r3;
          dxv[j] += r3 * (d[j] * wv[j] - xh * m3);
        }
        st8bf(dx3 + i, dxv);
      }
    }
  } else {
    auto dy0 = [&](int i) { return dy_f32 ? dyf0[i] : bf2f(dyb0[i]); };
    auto dy1 = [&](int i) { return dy_f32 ? dyf1[i] : bf2f(dyb1[i]); };
    auto dy2 = [&](int i) { return dy_f32 ? dyf2[i] : bf2f(dyb2[i]); };
    auto dy3 = [&](int i) { return dy_f32 ? dyf3[i] : bf2f(dyb3[i]); };
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    for (int i = lane; i < H; i += 32) {
      const float wi = bf2f(w[i]);
      dot0 += dy0(i) * wi * bf2f(x0[i]);
      if (has1) dot1 += dy1(i) * wi * bf2f(x1[i]);
      if (has2) dot2 += dy2(i) * wi * bf2f(x2[i]);
      if (has3) dot3 += dy3(i) * wi * bf2f(x3[i]);
    }
    const float m0 = warp_sum(dot0) * r0 / H;
    const float m1 = warp_sum(dot1) * r1 / H;
    const float m2 = warp_sum(dot2) * r2 / H;
    const float m3 = warp_sum(dot3) * r3 / H;
    for (int i = lane; i < H; i += 32) {
      const float wi = bf2f(w[i]);
      const float xh0 = bf2f(x0[i]) * r0;
      dx0[i] = f2bf(bf2f(dx0[i]) + r0 * (dy0(i) * wi - xh0 * m0));
      if (has1) {
        const float xh1 = bf2f(x1[i]) * r1;
        dx1[i] = f2bf(bf2f(dx1[i]) + r1 * (dy1(i) * wi - xh1 * m1));
      }
      if (has2) {
        const float xh2 = bf2f(x2[i]) * r2;
        dx2[i] = f2bf(bf2f(dx2[i]) + r2 * (dy2(i) * wi - xh2 * m2));
      }
      if (has3) {
        const float xh3 = bf2f(x3[i]) * r3;
        dx3[i] = f2bf(bf2f(dx3[i]) + r3 * (dy3(i) * wi - xh3 * m3));
      }
    }
  }
}

// Split variant: dx-only half. This keeps the residual-gradient chain from waiting on
// the weight-gradient atomic drain; the dw-only half is emitted as a cold sink.
template <bool UseFma>
__device__ void op_rmsnorm_bwd_dx_impl(const Instr& I, int tile, void** bufs,
                                       char* smem_raw) {
  const int H = I.args[6], S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  if (rowA >= S) return;  // barrier-free op: early exit is safe
  const bool hasB = rowA + 8 < S;
  const int rowB = hasB ? rowA + 8 : rowA;
  const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* xA = xb + (int64_t)rowA * H;
  const bf16* xB = xb + (int64_t)rowB * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const bf16* dybA = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const float* dyfA = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const bf16* dybB = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  const float* dyfB = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  bf16* dxb = reinterpret_cast<bf16*>(bufs[I.args[3]]);
  bf16* dxA = dxb + (int64_t)rowA * H;
  bf16* dxB = dxb + (int64_t)rowB * H;
  const float rA = reinterpret_cast<const float*>(bufs[I.args[5]])[rowA];
  const float rB = reinterpret_cast<const float*>(bufs[I.args[5]])[rowB];
  if ((H & 7) == 0) {
    float dotA = 0.0f, dotB = 0.0f;
    for (int i = lane * 8; i < H; i += 32 * 8) {
      float xa[8], xv[8], da[8], db[8], wv[8];
      ld8bf(xA + i, xa);
      ld8bf(xB + i, xv);
      ld8dy(dybA, dyfA, dy_f32, i, da);
      ld8dy(dybB, dyfB, dy_f32, i, db);
      ld8bf(w + i, wv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        dotA += da[j] * wv[j] * xa[j];
        dotB += db[j] * wv[j] * xv[j];
      }
    }
    const float mA = warp_sum(dotA) * rA / H;
    const float mB = warp_sum(dotB) * rB / H;
    for (int i = lane * 8; i < H; i += 32 * 8) {
      float xa[8], xv[8], da[8], db[8], wv[8], dxa[8], dxv[8];
      ld8bf(xA + i, xa);
      ld8bf(xB + i, xv);
      ld8dy(dybA, dyfA, dy_f32, i, da);
      ld8dy(dybB, dyfB, dy_f32, i, db);
      ld8bf(w + i, wv);
      ld8bf(dxA + i, dxa);
      ld8bf(dxB + i, dxv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float xhA = xa[j] * rA, xhB = xv[j] * rB;
        dxa[j] = rms_dx_acc<UseFma>(dxa[j], rA, da[j] * wv[j], xhA, mA);
        dxv[j] = rms_dx_acc<UseFma>(dxv[j], rB, db[j] * wv[j], xhB, mB);
      }
      st8bf(dxA + i, dxa);
      if (hasB) st8bf(dxB + i, dxv);
    }
  } else {
    auto dyA = [&](int i) { return dy_f32 ? dyfA[i] : bf2f(dybA[i]); };
    auto dyB = [&](int i) { return dy_f32 ? dyfB[i] : bf2f(dybB[i]); };
    float dotA = 0.0f, dotB = 0.0f;
    for (int i = lane; i < H; i += 32) {
      dotA += dyA(i) * bf2f(w[i]) * bf2f(xA[i]);
      dotB += dyB(i) * bf2f(w[i]) * bf2f(xB[i]);
    }
    const float mA = warp_sum(dotA) * rA / H;
    const float mB = warp_sum(dotB) * rB / H;
    for (int i = lane; i < H; i += 32) {
      const float xhA = bf2f(xA[i]) * rA, xhB = bf2f(xB[i]) * rB;
      dxA[i] = f2bf(rms_dx_acc<UseFma>(bf2f(dxA[i]), rA, dyA(i) * bf2f(w[i]), xhA, mA));
      if (hasB) dxB[i] = f2bf(rms_dx_acc<UseFma>(bf2f(dxB[i]), rB, dyB(i) * bf2f(w[i]), xhB, mB));
    }
  }
}

#ifdef MK_RMS_DX_H2560
__device__ void op_rmsnorm_bwd_dx_h2560(const Instr& I, int tile, void** bufs,
                                        char* smem_raw) {
  const int S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  if (rowA >= S) return;  // barrier-free op: early exit is safe
  const bool hasB = rowA + 8 < S;
  const int rowB = hasB ? rowA + 8 : rowA;
  constexpr int H = 2560;
  constexpr float invH = 1.0f / H;
  const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* xA = xb + (int64_t)rowA * H;
  const bf16* xB = xb + (int64_t)rowB * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const bf16* dybA = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const float* dyfA = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const bf16* dybB = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  const float* dyfB = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  bf16* dxb = reinterpret_cast<bf16*>(bufs[I.args[3]]);
  bf16* dxA = dxb + (int64_t)rowA * H;
  bf16* dxB = dxb + (int64_t)rowB * H;
  const float rA = reinterpret_cast<const float*>(bufs[I.args[5]])[rowA];
  const float rB = reinterpret_cast<const float*>(bufs[I.args[5]])[rowB];

  float dotA = 0.0f, dotB = 0.0f;
#pragma unroll
  for (int i = lane * 8; i < H; i += 32 * 8) {
    float xa[8], xv[8], da[8], db[8], wv[8];
    ld8bf(xA + i, xa);
    ld8bf(xB + i, xv);
    ld8dy(dybA, dyfA, dy_f32, i, da);
    ld8dy(dybB, dyfB, dy_f32, i, db);
    ld8bf(w + i, wv);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      dotA += da[j] * wv[j] * xa[j];
      dotB += db[j] * wv[j] * xv[j];
    }
  }
  const float mA = warp_sum(dotA) * rA * invH;
  const float mB = warp_sum(dotB) * rB * invH;

#pragma unroll
  for (int i = lane * 8; i < H; i += 32 * 8) {
    float xa[8], xv[8], da[8], db[8], wv[8], dxa[8], dxv[8];
    ld8bf(xA + i, xa);
    ld8bf(xB + i, xv);
    ld8dy(dybA, dyfA, dy_f32, i, da);
    ld8dy(dybB, dyfB, dy_f32, i, db);
    ld8bf(w + i, wv);
    ld8bf(dxA + i, dxa);
    ld8bf(dxB + i, dxv);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xhA = xa[j] * rA, xhB = xv[j] * rB;
      dxa[j] = rms_dx_acc<false>(dxa[j], rA, da[j] * wv[j], xhA, mA);
      dxv[j] = rms_dx_acc<false>(dxv[j], rB, db[j] * wv[j], xhB, mB);
    }
    st8bf(dxA + i, dxa);
    if (hasB) st8bf(dxB + i, dxv);
  }
}

#ifdef MK_QWEN_HEADDX_RMS_DOT_PARTIALS
__device__ void op_rmsnorm_bwd_dx_h2560_dotparts(const Instr& I, int tile, void** bufs,
                                                 char* smem_raw) {
  const int S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  if (rowA >= S) return;  // barrier-free op: early exit is safe
  const bool hasB = rowA + 8 < S;
  const int rowB = hasB ? rowA + 8 : rowA;
  constexpr int H = 2560;
  constexpr float invH = 1.0f / H;
  const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* xA = xb + (int64_t)rowA * H;
  const bf16* xB = xb + (int64_t)rowB * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const bf16* dybA = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const float* dyfA = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const bf16* dybB = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  const float* dyfB = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  bf16* dxb = reinterpret_cast<bf16*>(bufs[I.args[3]]);
  bf16* dxA = dxb + (int64_t)rowA * H;
  bf16* dxB = dxb + (int64_t)rowB * H;
  const float rA = reinterpret_cast<const float*>(bufs[I.args[5]])[rowA];
  const float rB = reinterpret_cast<const float*>(bufs[I.args[5]])[rowB];
  const float* parts = reinterpret_cast<const float*>(bufs[I.args[9]]);
  const int nparts = I.args[10];

  float dotA = 0.0f, dotB = 0.0f;
  for (int p = lane; p < nparts; p += 32) {
    dotA += parts[(int64_t)rowA * nparts + p];
    if (hasB) dotB += parts[(int64_t)rowB * nparts + p];
  }
  const float mA = warp_sum(dotA) * rA * invH;
  const float mB = warp_sum(dotB) * rB * invH;

#pragma unroll
  for (int i = lane * 8; i < H; i += 32 * 8) {
    float xa[8], xv[8], da[8], db[8], wv[8], dxa[8], dxv[8];
    ld8bf(xA + i, xa);
    ld8bf(xB + i, xv);
    ld8dy(dybA, dyfA, dy_f32, i, da);
    ld8dy(dybB, dyfB, dy_f32, i, db);
    ld8bf(w + i, wv);
    ld8bf(dxA + i, dxa);
    ld8bf(dxB + i, dxv);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xhA = xa[j] * rA, xhB = xv[j] * rB;
      dxa[j] = rms_dx_acc<false>(dxa[j], rA, da[j] * wv[j], xhA, mA);
      dxv[j] = rms_dx_acc<false>(dxv[j], rB, db[j] * wv[j], xhB, mB);
    }
    st8bf(dxA + i, dxa);
    if (hasB) st8bf(dxB + i, dxv);
  }
}
#endif
#endif

__device__ void op_rmsnorm_bwd_dx(const Instr& I, int tile, void** bufs,
                                  char* smem_raw) {
#ifdef MK_RMS_DX_H2560
  if (I.args[6] == 2560) {
#ifdef MK_QWEN_HEADDX_RMS_DOT_PARTIALS
    if (I.args[10] == 10) {
      op_rmsnorm_bwd_dx_h2560_dotparts(I, tile, bufs, smem_raw);
      return;
    }
#endif
    op_rmsnorm_bwd_dx_h2560(I, tile, bufs, smem_raw);
    return;
  }
#endif
  op_rmsnorm_bwd_dx_impl<false>(I, tile, bufs, smem_raw);
}

__device__ void op_rmsnorm_bwd_dx_fma(const Instr& I, int tile, void** bufs,
                                      char* smem_raw) {
  op_rmsnorm_bwd_dx_impl<true>(I, tile, bufs, smem_raw);
}

__device__ void op_rmsnorm_bwd_dx_h256(const Instr& I, int tile, void** bufs,
                                       char* smem_raw) {
  const int S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  if (rowA >= S) return;  // barrier-free op: early exit is safe
  const bool hasB = rowA + 8 < S;
  const int rowB = hasB ? rowA + 8 : rowA;
  constexpr int H = 256;
  constexpr float invH = 1.0f / H;
  const int i = lane * 8;
  const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* xA = xb + (int64_t)rowA * H;
  const bf16* xB = xb + (int64_t)rowB * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const bf16* dybA = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const float* dyfA = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowA * H;
  const bf16* dybB = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  const float* dyfB = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowB * H;
  bf16* dxb = reinterpret_cast<bf16*>(bufs[I.args[3]]);
  bf16* dxA = dxb + (int64_t)rowA * H;
  bf16* dxB = dxb + (int64_t)rowB * H;
  const float rA = reinterpret_cast<const float*>(bufs[I.args[5]])[rowA];
  const float rB = reinterpret_cast<const float*>(bufs[I.args[5]])[rowB];

  float xa[8], xv[8], da[8], db[8], wv[8];
  ld8bf(xA + i, xa);
  ld8bf(xB + i, xv);
  ld8dy(dybA, dyfA, dy_f32, i, da);
  ld8dy(dybB, dyfB, dy_f32, i, db);
  ld8bf(w + i, wv);
  float dotA = 0.0f, dotB = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    dotA += da[j] * wv[j] * xa[j];
    dotB += db[j] * wv[j] * xv[j];
  }
  const float mA = warp_sum(dotA) * rA * invH;
  const float mB = warp_sum(dotB) * rB * invH;

  float dxa[8], dxv[8];
  ld8bf(dxA + i, dxa);
  ld8bf(dxB + i, dxv);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const float xhA = xa[j] * rA, xhB = xv[j] * rB;
    dxa[j] += rA * (da[j] * wv[j] - xhA * mA);
    dxv[j] += rB * (db[j] * wv[j] - xhB * mB);
  }
  st8bf(dxA + i, dxa);
  if (hasB) st8bf(dxB + i, dxv);
}

// Split variant: dw-only half. Same reduction/atomic policy as the combined op, but
// no dx math and no dependency on the residual-gradient buffer.
__device__ void op_rmsnorm_bwd_dw(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int H = I.args[6], S = I.args[8];
  const bool dy_f32 = I.args[7] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int rowA = tile * MK_ROW_R2 + warp;
  float* dw_rows = reinterpret_cast<float*>(smem_raw);
  float* dw_row = dw_rows + (int64_t)warp * H;
  if (rowA < S) {
    const bool hasB = rowA + 8 < S;
    const int rowB = hasB ? rowA + 8 : rowA;
    const bf16* xb = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
    const bf16* xA = xb + (int64_t)rowA * H;
    const bf16* xB = xb + (int64_t)rowB * H;
    const bf16* dybA = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowA * H;
    const float* dyfA = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowA * H;
    const bf16* dybB = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)rowB * H;
    const float* dyfB = reinterpret_cast<const float*>(bufs[I.args[2]]) + (int64_t)rowB * H;
    const float rA = reinterpret_cast<const float*>(bufs[I.args[5]])[rowA];
    const float rB = reinterpret_cast<const float*>(bufs[I.args[5]])[rowB];
    const float foldB = hasB ? 1.0f : 0.0f;
    if ((H & 7) == 0) {
      for (int i = lane * 8; i < H; i += 32 * 8) {
        float xa[8], xv[8], da[8], db[8];
        ld8bf(xA + i, xa);
        ld8bf(xB + i, xv);
        ld8dy(dybA, dyfA, dy_f32, i, da);
        ld8dy(dybB, dyfB, dy_f32, i, db);
#pragma unroll
        for (int j = 0; j < 8; j++) dw_row[i + j] = da[j] * (xa[j] * rA) + foldB * (db[j] * (xv[j] * rB));
      }
    } else {
      auto dyA = [&](int i) { return dy_f32 ? dyfA[i] : bf2f(dybA[i]); };
      auto dyB = [&](int i) { return dy_f32 ? dyfB[i] : bf2f(dybB[i]); };
      for (int i = lane; i < H; i += 32) {
        const float xhA = bf2f(xA[i]) * rA, xhB = bf2f(xB[i]) * rB;
        dw_row[i] = dyA(i) * xhA + foldB * (dyB(i) * xhB);
      }
    }
  } else {
    for (int i = lane; i < H; i += 32) dw_row[i] = 0.0f;
  }
  consumer_sync();
  float* dw = reinterpret_cast<float*>(bufs[I.args[4]]);
  for (int i = mk_tid(); i < H; i += MK_CONSUMERS) {
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < 8; ++r) s += dw_rows[(int64_t)r * H + i];
    atomicAdd(&dw[i], s);
  }
}

// ---- SwiGLU ----------------------------------------------------------------------------
// fwd: h[r,i] = silu(gate[r,i]) * up[r,i], gate/up = halves of gu[r, 2I] (gate first).
// args: {gu, h, S, Iw}; tile = MK_ROW_R-row group, one warp per row.
__device__ void op_swiglu_fwd(const Instr& I, int tile, void** bufs) {
  const int S = I.args[2], Iw = I.args[3];
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int row = tile * MK_ROW_R + warp;
  if (row >= S) return;  // barrier-free op: early exit is safe
  const bf16* gu = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * 2 * Iw;
  bf16* h = reinterpret_cast<bf16*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  bf16* sig_cache = nullptr;
#ifdef MK_SWIGLU_CACHE_SIG
  if (I.args[4] != 0) sig_cache = reinterpret_cast<bf16*>(bufs[I.args[4]]) + (int64_t)row * Iw;
#endif
  if ((Iw & 7) == 0) {
    for (int i = lane * 8; i < Iw; i += 32 * 8) {
      float g[8], u[8], hv[8], sv[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
#pragma unroll
      // __expf (SFU): libm expf is a multi-instruction software path that both
      // serializes the lane and bloats register pressure (see the CE epilogue note);
      // error is ~2 ulp, far below bf16 output rounding.
      for (int j = 0; j < 8; j++) {
        const float sig = 1.0f / (1.0f + __expf(-g[j]));
        sv[j] = sig;
        hv[j] = g[j] * sig * u[j];
      }
      st8bf(h + i, hv);
      if (sig_cache) st8bf(sig_cache + i, sv);
    }
  } else {
    for (int i = lane; i < Iw; i += 32) {
      const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]);
      const float sig = 1.0f / (1.0f + __expf(-g));
      h[i] = f2bf(g * sig * u);
      if (sig_cache) sig_cache[i] = f2bf(sig);
    }
  }
}

// bwd: dgate = dh * u * dsilu(g); dup = dh * silu(g). Writes dgu (bf16).
// args: {gu, dh, dgu, S, Iw, dy_f32}; tile = MK_ROW_R-row group, one warp per row.
// dy_f32 != 0 reads dh as fp32 (a split-K atomic workspace consumed directly).
__device__ void op_swiglu_bwd(const Instr& I, int tile, void** bufs) {
  const int S = I.args[3], Iw = I.args[4];
  const bool dy_f32 = I.args[5] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int row = tile * MK_ROW_R + warp;
  if (row >= S) return;  // barrier-free op: early exit is safe
  const bf16* gu = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * 2 * Iw;
  const bf16* dhb = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  const float* dhf = reinterpret_cast<const float*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  bf16* dgu = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)row * 2 * Iw;
  const bf16* sig_cache = nullptr;
#ifdef MK_SWIGLU_CACHE_SIG
  if (I.args[6] != 0) sig_cache = reinterpret_cast<const bf16*>(bufs[I.args[6]]) + (int64_t)row * Iw;
#endif
  if ((Iw & 7) == 0) {
    for (int i = lane * 8; i < Iw; i += 32 * 8) {
      float g[8], u[8], d[8], dg[8], du[8], sc[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
      ld8dy(dhb, dhf, dy_f32, i, d);
      if (sig_cache) ld8bf(sig_cache + i, sc);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float sig = sig_cache ? sc[j] : 1.0f / (1.0f + __expf(-g[j]));
        const float sg = g[j] * sig;
        // dsilu = sig + silu*(1-sig) = sig + sg - sg*sig.
#ifdef MK_SWIGLU_FMA_DERIV
        const float ds = fmaf(-sg, sig, sig + sg);
#else
        const float ds = sig + sg * (1.0f - sig);
#endif
        dg[j] = d[j] * u[j] * ds;
        du[j] = d[j] * sg;
      }
      st8bf(dgu + i, dg);
      st8bf(dgu + Iw + i, du);
    }
  } else {
    for (int i = lane; i < Iw; i += 32) {
      const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]);
      const float d = dy_f32 ? dhf[i] : bf2f(dhb[i]);
      const float sig = sig_cache ? bf2f(sig_cache[i]) : 1.0f / (1.0f + __expf(-g));
      const float sg = g * sig;
#ifdef MK_SWIGLU_FMA_DERIV
      const float ds = fmaf(-sg, sig, sig + sg);
#else
      const float ds = sig + sg * (1.0f - sig);
#endif
      dgu[i] = f2bf(d * u * ds);
      dgu[Iw + i] = f2bf(d * sg);
    }
  }
}

#ifdef MK_SWIGLU_BWD_2W
// Two warps per row for wide SwiGLU backward rows. Each row-local warp pair splits
// the feature dimension, so writes are disjoint and no inter-warp sync is required.
__device__ void op_swiglu_bwd_2w(const Instr& I, int tile, void** bufs) {
  const int S = I.args[3], Iw = I.args[4];
  const bool dy_f32 = I.args[5] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int row = tile * 4 + (warp >> 1);
  const int lane64 = ((warp & 1) << 5) + lane;
  if (row >= S) return;  // barrier-free op: early exit is safe
  const bf16* gu = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * 2 * Iw;
  const bf16* dhb = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  const float* dhf = reinterpret_cast<const float*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  bf16* dgu = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)row * 2 * Iw;
  const bf16* sig_cache = nullptr;
#ifdef MK_SWIGLU_CACHE_SIG
  if (I.args[6] != 0) sig_cache = reinterpret_cast<const bf16*>(bufs[I.args[6]]) + (int64_t)row * Iw;
#endif
  if ((Iw & 7) == 0) {
    for (int i = lane64 * 8; i < Iw; i += 64 * 8) {
      float g[8], u[8], d[8], dg[8], du[8], sc[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
      ld8dy(dhb, dhf, dy_f32, i, d);
      if (sig_cache) ld8bf(sig_cache + i, sc);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float sig = sig_cache ? sc[j] : 1.0f / (1.0f + __expf(-g[j]));
        const float sg = g[j] * sig;
#ifdef MK_SWIGLU_FMA_DERIV
        const float ds = fmaf(-sg, sig, sig + sg);
#else
        const float ds = sig + sg * (1.0f - sig);
#endif
        dg[j] = d[j] * u[j] * ds;
        du[j] = d[j] * sg;
      }
      st8bf(dgu + i, dg);
      st8bf(dgu + Iw + i, du);
    }
  } else {
    for (int i = lane64; i < Iw; i += 64) {
      const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]);
      const float d = dy_f32 ? dhf[i] : bf2f(dhb[i]);
      const float sig = sig_cache ? bf2f(sig_cache[i]) : 1.0f / (1.0f + __expf(-g));
      const float sg = g * sig;
#ifdef MK_SWIGLU_FMA_DERIV
      const float ds = fmaf(-sg, sig, sig + sg);
#else
      const float ds = sig + sg * (1.0f - sig);
#endif
      dgu[i] = f2bf(d * u * ds);
      dgu[Iw + i] = f2bf(d * sg);
    }
  }
}
#endif

#ifdef MK_SWIGLU_BWD_4W
// Four warps per row for very wide SwiGLU backward rows (qwen I=9728). Each
// row-local warp quad splits the feature dimension; writes are disjoint.
__device__ void op_swiglu_bwd_4w(const Instr& I, int tile, void** bufs) {
  const int S = I.args[3], Iw = I.args[4];
  const bool dy_f32 = I.args[5] != 0;
  const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
  const int row = tile * 2 + (warp >> 2);
  const int lane128 = ((warp & 3) << 5) + lane;
  if (row >= S) return;  // barrier-free op: early exit is safe
  const bf16* gu = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * 2 * Iw;
  const bf16* dhb = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  const float* dhf = reinterpret_cast<const float*>(bufs[I.args[1]]) + (int64_t)row * Iw;
  bf16* dgu = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)row * 2 * Iw;
  const bf16* sig_cache = nullptr;
#ifdef MK_SWIGLU_CACHE_SIG
  if (I.args[6] != 0) sig_cache = reinterpret_cast<const bf16*>(bufs[I.args[6]]) + (int64_t)row * Iw;
#endif
  if ((Iw & 7) == 0) {
    for (int i = lane128 * 8; i < Iw; i += 128 * 8) {
      float g[8], u[8], d[8], dg[8], du[8], sc[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
      ld8dy(dhb, dhf, dy_f32, i, d);
      if (sig_cache) ld8bf(sig_cache + i, sc);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float sig = sig_cache ? sc[j] : 1.0f / (1.0f + __expf(-g[j]));
        const float sg = g[j] * sig;
#ifdef MK_SWIGLU_FMA_DERIV
        const float ds = fmaf(-sg, sig, sig + sg);
#else
        const float ds = sig + sg * (1.0f - sig);
#endif
        dg[j] = d[j] * u[j] * ds;
        du[j] = d[j] * sg;
      }
      st8bf(dgu + i, dg);
      st8bf(dgu + Iw + i, du);
    }
  } else {
    for (int i = lane128; i < Iw; i += 128) {
      const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]);
      const float d = dy_f32 ? dhf[i] : bf2f(dhb[i]);
      const float sig = sig_cache ? bf2f(sig_cache[i]) : 1.0f / (1.0f + __expf(-g));
      const float sg = g * sig;
#ifdef MK_SWIGLU_FMA_DERIV
      const float ds = fmaf(-sg, sig, sig + sg);
#else
      const float ds = sig + sg * (1.0f - sig);
#endif
      dgu[i] = f2bf(d * u * ds);
      dgu[Iw + i] = f2bf(d * sg);
    }
  }
}
#endif

// ---- per-head RMSNorm (Qwen3 qk-norm) + RoPE, fused ------------------------------------
// Operates on the packed qkv buffer [S, (nq+2*nkv)*D]: normalizes+ropes q and k heads
// in place is NOT possible (bwd needs raw input), so reads qkv_raw and writes qkv_r.
// v passes through unchanged. rstd saved per (row, head) for q and k.
// cos/sin: fp32 [S, D/2].
// args: {qkv_raw, qkv_r, qw, kw, rstd_q, rstd_k, cos, sin, nq, nkv, D, eps_bits}
// tile = row; warp w handles heads w, w+8, ... (q heads, k heads, then v copies).
__device__ void op_qknorm_rope_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int nq = I.args[8], nkv = I.args[9], D = I.args[10];
  const float eps = __int_as_float(I.args[11]);
  const int row = tile;
  const int stride = (nq + 2 * nkv) * D;
  const bf16* src_row = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * stride;
  bf16* dst_row = reinterpret_cast<bf16*>(bufs[I.args[1]]) + (int64_t)row * stride;
  const float* cosr = reinterpret_cast<const float*>(bufs[I.args[6]]) + (int64_t)row * (D / 2);
  const float* sinr = reinterpret_cast<const float*>(bufs[I.args[7]]) + (int64_t)row * (D / 2);
  const int warp = mk_tid() / 32, lane = mk_tid() % 32, nwarp = MK_CONSUMERS / 32;

  for (int h = warp; h < nq + 2 * nkv; h += nwarp) {
    const bf16* src = src_row + h * D;
    bf16* dst = dst_row + h * D;
    if (h >= nq + nkv) {  // v head: pass through
      for (int i = lane; i < D; i += 32) dst[i] = src[i];
      continue;
    }
    const bool is_q = h < nq;
    const bf16* w = reinterpret_cast<const bf16*>(bufs[is_q ? I.args[2] : I.args[3]]);
    float ss = 0.0f;
    for (int i = lane; i < D; i += 32) {
      const float v = bf2f(src[i]);
      ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    const float r = rsqrtf(ss / D + eps);
    if (lane == 0)
      reinterpret_cast<float*>(bufs[is_q ? I.args[4] : I.args[5]])
          [(int64_t)row * (is_q ? nq : nkv) + (is_q ? h : h - nq)] = r;
    for (int i = lane; i < D / 2; i += 32) {
      const float a = bf2f(src[i]) * r * bf2f(w[i]);
      const float b = bf2f(src[i + D / 2]) * r * bf2f(w[i + D / 2]);
      const float c = cosr[i], s = sinr[i];
      dst[i] = f2bf(a * c - b * s);
      dst[i + D / 2] = f2bf(b * c + a * s);
    }
  }
}

// bwd: input d(qkv_r) (from attention bwd, v grad included), output d(qkv_raw).
// rope bwd = rotate by -theta; then per-head rmsnorm bwd (dw via fp32 atomics).
// v grads pass through. Writes dqkv_raw (bf16), OVERWRITING (not accumulating).
// args: {qkv_raw, dqkv_r, dqkv_raw, qw, kw, dqw, dkw, rstd_q, rstd_k, cos, sin, nq, nkv, D,
//        dy_f32, S, split_v}
// tile = MK_ROW_R-row group; warp w sweeps (head, row) tasks h = w mod nh (weight-vector
// locality). Per-tile dqw/dkw partials accumulate in smem (fast atomics), then ONE global
// atomicAdd per element per MK_ROW_R rows — the smem zero/flush and its two barriers are
// amortized over the row group. dy_f32 != 0 reads the incoming grad as fp32 (the
// attention-bwd atomic workspace, no CVT chain hop).
__device__ void op_qknorm_rope_bwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int nq = I.args[11], nkv = I.args[12], D = I.args[13];
  const bool dy_f32 = I.args[14] != 0;
  const int S = I.args[15];
  const bool split_v = I.args[16] != 0;
  const int stride = (nq + 2 * nkv) * D, nh = nq + (split_v ? nkv : 2 * nkv);
  const int warp = mk_tid() / 32, lane = mk_tid() % 32, nwarp = MK_CONSUMERS / 32;

#ifdef MK_QKBWD_VEC8
  // Vectorized-IO probe path: every global access in the task loop is 16B wide
  // (ld8bf/st8bf/ld8dy) instead of per-element bf16/fp32. Restructure: each
  // warp runs G = 32/(D/16) (head,row) tasks per iteration with C = D/16 lanes
  // per task; lane q of a task owns ONE 8-pair chunk — first-half elements
  // [8q, 8q+8) plus their rope partners [8q+D/2, 8q+D/2+8) — so both halves of
  // every pair are local to the lane (no cross-lane exchange) and the
  // per-element arithmetic (da/db/dx/dw formulas) is identical to the scalar
  // paths. The dot reduction becomes 8 serial adds per lane + an aligned
  // C-lane xor butterfly, so dx matches at tolerance level (reduction order),
  // not bitwise. dw partials stay plain (non-atomic) smem adds: each
  // (warp, task-slot) pair gets its own slice — G*D == 512 floats per warp per
  // weight for any admissible D — and the flush sums nwarp*G slices before the
  // one global atomicAdd per element (same tolerance class as before). Takes
  // precedence over the D64/D128 cache paths; falls back to the scalar body
  // unless D % 16 == 0 and (32 % (D/16)) == 0 (covers the (D & 7) == 0 guard
  // for every shape the model ships: D = 64, 128).
  if ((D % 16) == 0 && (32 % (D / 16)) == 0) {
    const int C = D / 16, G = 32 / C;
    const int g = lane / C, q = lane - g * C;  // task-in-warp, chunk-in-task
    float* dwq_s = reinterpret_cast<float*>(smem_raw);  // [nwarp][G][D] == [nwarp][512]
    float* dwk_s = dwq_s + (MK_CONSUMERS / 32) * 512;
    for (int i = mk_tid(); i < 2 * (MK_CONSUMERS / 32) * 512; i += MK_CONSUMERS) dwq_s[i] = 0.0f;
    consumer_sync();

    for (int t0 = warp * G; t0 < MK_ROW_R * nh; t0 += nwarp * G) {
      const int t = t0 + g;
      const bool in_range = t < MK_ROW_R * nh;
      const int h = in_range ? t % nh : 0, rr = in_range ? t / nh : 0;
      const int row = tile * MK_ROW_R + rr;
      const bool act = in_range && row < S;
      const int64_t off = (int64_t)row * stride + h * D;
      const bf16* xr = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + off;
      const bf16* dyb = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + off;
      const float* dyf = reinterpret_cast<const float*>(bufs[I.args[1]]) + off;
      bf16* dxr = reinterpret_cast<bf16*>(bufs[I.args[2]]) + off;
      const int i0 = q * 8;

      if (act && h >= nq + nkv) {  // v grads pass through: two 8-chunks per lane
        float f[8];
        ld8dy(dyb, dyf, dy_f32, i0, f);
        st8bf(dxr + i0, f);
        ld8dy(dyb, dyf, dy_f32, i0 + D / 2, f);
        st8bf(dxr + i0 + D / 2, f);
      }
      const bool is_qk = act && h < nq + nkv;
      const bool is_q = h < nq;
      float da[8], db[8], xh1[8], xh2[8], w1[8], w2[8];
      float r = 0.0f, dot = 0.0f;
      if (is_qk) {
        const bf16* w = reinterpret_cast<const bf16*>(bufs[is_q ? I.args[3] : I.args[4]]);
        r = reinterpret_cast<const float*>(bufs[is_q ? I.args[7] : I.args[8]])
            [(int64_t)row * (is_q ? nq : nkv) + (is_q ? h : h - nq)];
        const float* cosr =
            reinterpret_cast<const float*>(bufs[I.args[9]]) + (int64_t)row * (D / 2);
        const float* sinr =
            reinterpret_cast<const float*>(bufs[I.args[10]]) + (int64_t)row * (D / 2);
        float dy1[8], dy2[8], cv[8], sv[8];
        ld8dy(dyb, dyf, dy_f32, i0, dy1);
        ld8dy(dyb, dyf, dy_f32, i0 + D / 2, dy2);
        ld8dy(nullptr, cosr, true, i0, cv);
        ld8dy(nullptr, sinr, true, i0, sv);
        ld8bf(w + i0, w1);
        ld8bf(w + i0 + D / 2, w2);
        ld8bf(xr + i0, xh1);
        ld8bf(xr + i0 + D / 2, xh2);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          da[j] = dy1[j] * cv[j] + dy2[j] * sv[j];
          db[j] = -dy1[j] * sv[j] + dy2[j] * cv[j];
          xh1[j] *= r;
          xh2[j] *= r;
          dot += da[j] * w1[j] * xh1[j] + db[j] * w2[j] * xh2[j];
        }
      }
      // all 32 lanes execute the butterfly (aligned C-blocks; idle lanes carry 0)
      for (int o = C >> 1; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      dot /= D;
      if (is_qk) {
        float o1[8], o2[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          o1[j] = r * (da[j] * w1[j] - xh1[j] * dot);
          o2[j] = r * (db[j] * w2[j] - xh2[j] * dot);
        }
        st8bf(dxr + i0, o1);
        st8bf(dxr + i0 + D / 2, o2);
        float* dw_s = (is_q ? dwq_s : dwk_s) + (warp * G + g) * D;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          dw_s[i0 + j] += da[j] * xh1[j];
          dw_s[i0 + D / 2 + j] += db[j] * xh2[j];
        }
      }
    }
    consumer_sync();
    float* dqw = reinterpret_cast<float*>(bufs[I.args[5]]);
    float* dkw = reinterpret_cast<float*>(bufs[I.args[6]]);
    const int nsl = (MK_CONSUMERS / 32) * G;
    for (int i = mk_tid(); i < D; i += MK_CONSUMERS) {
      float aq = 0.0f, ak = 0.0f;
      for (int s2 = 0; s2 < nsl; ++s2) {
        aq += dwq_s[s2 * D + i];
        ak += dwk_s[s2 * D + i];
      }
      atomicAdd(&dqw[i], aq);
      atomicAdd(&dkw[i], ak);
    }
    return;
  }
#endif

  // per-warp dw partial slices: plain adds instead of block-wide smem atomics
  // (the old shared [D]+[D] arrays serialized every lane of every warp on 64
  // addresses — the dominant cost of this op at long S)
  float* dwq_s = reinterpret_cast<float*>(smem_raw);  // [nwarp][D] + [nwarp][D]
  float* dwk_s = dwq_s + (MK_CONSUMERS / 32) * D;
  for (int i = mk_tid(); i < 2 * (MK_CONSUMERS / 32) * D; i += MK_CONSUMERS) dwq_s[i] = 0.0f;
  consumer_sync();

  for (int t = warp; t < MK_ROW_R * nh; t += nwarp) {
    const int h = t % nh, rr = t / nh;
    const int row = tile * MK_ROW_R + rr;
    if (row >= S) continue;  // no barriers inside the task loop
    const bf16* x_row = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * stride;
    const bf16* dyb_row = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)row * stride;
    const float* dyf_row =
        reinterpret_cast<const float*>(bufs[I.args[1]]) + (int64_t)row * stride;
    bf16* dx_row = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)row * stride;
    const float* cosr = reinterpret_cast<const float*>(bufs[I.args[9]]) + (int64_t)row * (D / 2);
    const float* sinr = reinterpret_cast<const float*>(bufs[I.args[10]]) + (int64_t)row * (D / 2);
    const bf16* xr = x_row + h * D;
    const bf16* dyb = dyb_row + h * D;
    const float* dyf = dyf_row + h * D;
    auto dyr = [&](int i) { return dy_f32 ? dyf[i] : bf2f(dyb[i]); };
    bf16* dxr = dx_row + h * D;
    if (h >= nq + nkv) {  // v grads pass through
      for (int i = lane; i < D; i += 32) dxr[i] = f2bf(dyr(i));
      continue;
    }
    const bool is_q = h < nq;
    const bf16* w = reinterpret_cast<const bf16*>(bufs[is_q ? I.args[3] : I.args[4]]);
    float* dw_s = (is_q ? dwq_s : dwk_s) + warp * D;
    const float r = reinterpret_cast<const float*>(
        bufs[is_q ? I.args[7] : I.args[8]])[(int64_t)row * (is_q ? nq : nkv) + (is_q ? h : h - nq)];

#ifdef MK_QKBWD_D64_CACHE
    if (D == 64) {
      // D=64 model path: each lane owns one rope pair. Keep the intermediates live
      // across the dot reduction instead of reloading/recomputing them below.
      const int i = lane;
      const float c = cosr[i], s = sinr[i];
      const float dy1 = dyr(i), dy2 = dyr(i + 32);
      const float da = dy1 * c + dy2 * s;
      const float db = -dy1 * s + dy2 * c;
      const float w1 = bf2f(w[i]), w2 = bf2f(w[i + 32]);
      const float xh1 = bf2f(xr[i]) * r, xh2 = bf2f(xr[i + 32]) * r;
      float dot = da * w1 * xh1 + db * w2 * xh2;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
      dot *= 1.0f / 64.0f;
      dxr[i] = f2bf(r * (da * w1 - xh1 * dot));
      dxr[i + 32] = f2bf(r * (db * w2 - xh2 * dot));
      dw_s[i] += da * xh1;
      dw_s[i + 32] += db * xh2;
      continue;
    }
#endif

#ifdef MK_QKBWD_D128_CACHE
    if (D == 128 && S == 1024 && nq == 32 && nkv == 8) {
      // D=128 qwen path: each lane owns two rope pairs. Keep both pairs'
      // intermediates live across the dot reduction instead of reloading them
      // in the dx/dw pass below. The shape guard keeps unmeasured D=128 layouts
      // on the generic loop.
      const int i0 = lane;
      const int i1 = lane + 32;

      const float c0 = cosr[i0], s0 = sinr[i0];
      const float dy10 = dyr(i0), dy20 = dyr(i0 + 64);
      const float da0 = dy10 * c0 + dy20 * s0;
      const float db0 = -dy10 * s0 + dy20 * c0;
      const float w10 = bf2f(w[i0]), w20 = bf2f(w[i0 + 64]);
      const float xh10 = bf2f(xr[i0]) * r, xh20 = bf2f(xr[i0 + 64]) * r;

      const float c1 = cosr[i1], s1 = sinr[i1];
      const float dy11 = dyr(i1), dy21 = dyr(i1 + 64);
      const float da1 = dy11 * c1 + dy21 * s1;
      const float db1 = -dy11 * s1 + dy21 * c1;
      const float w11 = bf2f(w[i1]), w21 = bf2f(w[i1 + 64]);
      const float xh11 = bf2f(xr[i1]) * r, xh21 = bf2f(xr[i1 + 64]) * r;

      float dot = da0 * w10 * xh10 + db0 * w20 * xh20 +
                  da1 * w11 * xh11 + db1 * w21 * xh21;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
      dot *= 1.0f / 128.0f;

      dxr[i0] = f2bf(r * (da0 * w10 - xh10 * dot));
      dxr[i0 + 64] = f2bf(r * (db0 * w20 - xh20 * dot));
      dxr[i1] = f2bf(r * (da1 * w11 - xh11 * dot));
      dxr[i1 + 64] = f2bf(r * (db1 * w21 - xh21 * dot));
      dw_s[i0] += da0 * xh10;
      dw_s[i0 + 64] += db0 * xh20;
      dw_s[i1] += da1 * xh11;
      dw_s[i1 + 64] += db1 * xh21;
      continue;
    }
#endif

    // rope^-1 on the incoming grad: da = dy1*c + dy2*s ; db = -dy1*s + dy2*c;
    // then per-head rmsnorm bwd with x = raw.
    float dot = 0.0f;
    for (int i = lane; i < D / 2; i += 32) {
      const float c = cosr[i], s = sinr[i];
      const float dy1 = dyr(i), dy2 = dyr(i + D / 2);
      const float da = dy1 * c + dy2 * s;
      const float db = -dy1 * s + dy2 * c;
      dot += da * bf2f(w[i]) * bf2f(xr[i]) * r + db * bf2f(w[i + D / 2]) * bf2f(xr[i + D / 2]) * r;
    }
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
    dot /= D;
    for (int i = lane; i < D / 2; i += 32) {
      const float c = cosr[i], s = sinr[i];
      const float dy1 = dyr(i), dy2 = dyr(i + D / 2);
      const float da = dy1 * c + dy2 * s;
      const float db = -dy1 * s + dy2 * c;
      const float xh1 = bf2f(xr[i]) * r, xh2 = bf2f(xr[i + D / 2]) * r;
      dxr[i] = f2bf(r * (da * bf2f(w[i]) - xh1 * dot));
      dxr[i + D / 2] = f2bf(r * (db * bf2f(w[i + D / 2]) - xh2 * dot));
      dw_s[i] += da * xh1;
      dw_s[i + D / 2] += db * xh2;
    }
  }
  consumer_sync();
  float* dqw = reinterpret_cast<float*>(bufs[I.args[5]]);
  float* dkw = reinterpret_cast<float*>(bufs[I.args[6]]);
  for (int i = mk_tid(); i < D; i += MK_CONSUMERS) {
    float aq = 0.0f, ak = 0.0f;
#pragma unroll
    for (int w2 = 0; w2 < MK_CONSUMERS / 32; ++w2) {
      aq += dwq_s[w2 * D + i];
      ak += dwk_s[w2 * D + i];
    }
    atomicAdd(&dqw[i], aq);
    atomicAdd(&dkw[i], ak);
  }
}

// V heads do not need qk-norm or rope backward; they only convert the attention-bwd
// fp32 workspace into the packed bf16 qkv_raw gradient consumed by the next GEMM.
// args: {dqkv_f32, dqkv_raw, nq, nkv, D, S}; tile = MK_ROW_R rows.
__device__ void op_qkv_v_bwd(const Instr& I, int tile, void** bufs) {
  const int nq = I.args[2], nkv = I.args[3], D = I.args[4], S = I.args[5];
  const int stride = (nq + 2 * nkv) * D;
  const int rows0 = tile * MK_ROW_R;
  const int cols = nkv * D;
  const float* src = reinterpret_cast<const float*>(bufs[I.args[0]]);
  bf16* dst = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  const int total = min(MK_ROW_R, max(0, S - rows0)) * cols;
  const int v_off = (nq + nkv) * D;
  for (int i = mk_tid() * 8; i < total; i += MK_CONSUMERS * 8) {
    const int rr = i / cols;
    const int cc = i - rr * cols;
    float f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = cc + j;
      f[j] = c < cols ? src[(int64_t)(rows0 + rr) * stride + v_off + c] : 0.0f;
    }
    if (cc + 7 < cols) {
      st8bf(dst + (int64_t)(rows0 + rr) * stride + v_off + cc, f);
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = cc + j;
        if (c < cols) dst[(int64_t)(rows0 + rr) * stride + v_off + c] = f2bf(f[j]);
      }
    }
  }
}

// ---- embedding --------------------------------------------------------------------------
// fwd gather: x[r,:] = emb[tok[r],:]. args: {tok, emb, x, H}; tile = row.
__device__ void op_embed_fwd(const Instr& I, int tile, void** bufs) {
  const int H = I.args[3];
  const int t = reinterpret_cast<const int*>(bufs[I.args[0]])[tile];
  const bf16* e = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)t * H;
  bf16* x = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)tile * H;
  for (int i = mk_tid(); i < H; i += MK_CONSUMERS) x[i] = e[i];
}

// bwd scatter-add: demb[tok[r],:] += dx[r,:] (fp32 atomics). args: {tok, dx, demb, H}.
__device__ void op_embed_bwd(const Instr& I, int tile, void** bufs) {
  const int H = I.args[3];
  const int t = reinterpret_cast<const int*>(bufs[I.args[0]])[tile];
  const bf16* dx = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)tile * H;
  float* de = reinterpret_cast<float*>(bufs[I.args[2]]) + (int64_t)t * H;
  for (int i = mk_tid(); i < H; i += MK_CONSUMERS) atomicAdd(&de[i], bf2f(dx[i]));
}

// Sparse embedding-gradient clear. args: {prev_tok, tok, demb, H}; tile = row.
// Clears rows that may be nonzero from the previous step plus rows that will receive
// atomics in this step. Duplicate/current==previous rows are benign zero races.
__device__ void op_embed_zero_rows(const Instr& I, int tile, void** bufs) {
  const int H = I.args[3];
  const int* prev_tok = reinterpret_cast<const int*>(bufs[I.args[0]]);
  const int* tok = reinterpret_cast<const int*>(bufs[I.args[1]]);
  float* demb = reinterpret_cast<float*>(bufs[I.args[2]]);
  const int t_prev = prev_tok[tile];
  const int t_cur = tok[tile];
  float* de_prev = demb + (int64_t)t_prev * H;
  float* de_cur = demb + (int64_t)t_cur * H;
  for (int i = mk_tid(); i < H; i += MK_CONSUMERS) {
    de_prev[i] = 0.0f;
    de_cur[i] = 0.0f;
  }
}

// args: {src, dst, n}; tile = MK_CHUNK-element chunk index.
__device__ void op_copy_i32(const Instr& I, int tile, void** bufs) {
  const int* src = reinterpret_cast<const int*>(bufs[I.args[0]]);
  int* dst = reinterpret_cast<int*>(bufs[I.args[1]]);
  const int n = I.args[2];
  const int base = tile * MK_CHUNK, end = min(base + MK_CHUNK, n);
  for (int i = base + mk_tid(); i < end; i += MK_CONSUMERS) dst[i] = src[i];
}

// Valid-label reciprocal for CE. Keeping this inside the cooperative launch avoids a
// handful of host-side PyTorch kernels in MKQwen3.step(), while CE_FWD/BWD still read
// the same scalar contract: inv_valid = 1 / max(count(labels >= 0), 1).
// args: {labels, inv_valid, S}; tile = 0.
__device__ void op_inv_valid(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int S = I.args[2];
  const int* labels = reinterpret_cast<const int*>(bufs[I.args[0]]);
  float* inv_valid = reinterpret_cast<float*>(bufs[I.args[1]]);
  int count = 0;
  for (int i = mk_tid(); i < S; i += MK_CONSUMERS) count += labels[i] >= 0;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    count += __shfl_xor_sync(0xffffffffu, count, off);
  int* warp_counts = reinterpret_cast<int*>(smem_raw);
  if ((mk_tid() & 31) == 0) warp_counts[mk_tid() >> 5] = count;
  consumer_sync();
  if (mk_tid() == 0) {
    int total = 0;
#pragma unroll
    for (int w = 0; w < MK_CONSUMERS / 32; ++w) total += warp_counts[w];
    inv_valid[0] = 1.0f / float(max(total, 1));
  }
}

// ---- cross entropy ----------------------------------------------------------------------
// fwd over materialized logits [S, V]: per row lse; loss_sum += (lse - z_t) for valid rows.
// inv_valid is a device fp32 scalar (1/num_valid). loss is a device fp32 scalar (pre-zeroed).
// args: {logits, labels, lse, loss, inv_valid, V, parts, nparts}; tile = row. labels int32,
// -100 = ignore. nparts > 0: reduce the fused lm_head-epilogue (max, sumexp) partial pairs
// (args[6], [S, nparts, 2] fp32) instead of rescanning the V-wide logits row.
__device__ void op_ce_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int V = I.args[5];
  const bf16* z = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * V;
  const int label = reinterpret_cast<const int*>(bufs[I.args[1]])[tile];
  float* lse_out = reinterpret_cast<float*>(bufs[I.args[2]]);
  float* loss = reinterpret_cast<float*>(bufs[I.args[3]]);
  const float inv_valid = *reinterpret_cast<const float*>(bufs[I.args[4]]);
  const int nparts = I.args[7];
  float* scratch = reinterpret_cast<float*>(smem_raw);

#ifdef MK_CE_FWD_WARPROW
  if (nparts > 0) {
    // Warp-per-row partials reduce (MK_CONSUMERS/32 rows per tile, ZERO block
    // syncs): the per-row block-sync ladder below costs ~2us/row of pure
    // overhead at long S (8192 rows ~ 130us on-path); a warp shuffle-merge of
    // the (max,sumexp) pairs needs no cross-warp traffic at all.
    // args[8] = S (row count); tile covers rows [tile*8, tile*8+8).
    const int S = I.args[8];
    const int warp = mk_tid() >> 5, lane = mk_tid() & 31;
    const int row = tile * (MK_CONSUMERS / 32) + warp;
    if (row >= S) return;
    const bf16* zr = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * V;
    const int lab = reinterpret_cast<const int*>(bufs[I.args[1]])[row];
    const float* parts =
        reinterpret_cast<const float*>(bufs[I.args[6]]) + (int64_t)row * nparts * 2;
    float m = -INFINITY, s = 0.0f;
    for (int i = lane; i < nparts; i += 32) {
      const float om = parts[i * 2], os = parts[i * 2 + 1];
      const float M = fmaxf(m, om);
      s = (m == -INFINITY && om == -INFINITY) ? 0.0f
                                              : s * ce_exp(m - M) + os * ce_exp(om - M);
      m = M;
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float om = __shfl_xor_sync(0xffffffff, m, off);
      const float os = __shfl_xor_sync(0xffffffff, s, off);
      const float M = fmaxf(m, om);
      s = (m == -INFINITY && om == -INFINITY) ? 0.0f
                                              : s * ce_exp(m - M) + os * ce_exp(om - M);
      m = M;
    }
    if (lane == 0) {
      const float row_lse = m + logf(s);
      lse_out[row] = row_lse;
      if (lab >= 0) atomicAdd(loss, (row_lse - bf2f(zr[lab])) * inv_valid);
    }
    return;
  }
#endif

  // single-pass online (m, s) accumulation: one read of the logits row instead of two
  float mx = -INFINITY, se = 0.0f;
  if (nparts > 0) {
    const float* parts = reinterpret_cast<const float*>(bufs[I.args[6]]) + (int64_t)tile * nparts * 2;
    for (int i = mk_tid(); i < nparts; i += MK_CONSUMERS) {
      const float om = parts[i * 2], os = parts[i * 2 + 1];
      const float M = fmaxf(mx, om);
      se = (mx == -INFINITY && om == -INFINITY) ? 0.0f
                                                 : se * ce_exp(mx - M) + os * ce_exp(om - M);
      mx = M;
    }
  } else {
    for (int i = mk_tid(); i < V; i += MK_CONSUMERS) {
      const float zv = bf2f(z[i]);
      if (zv > mx) {
        se = se * ce_exp(mx - zv) + 1.0f;
        mx = zv;
      } else {
        se += ce_exp(zv - mx);
      }
    }
  }
  // merge (m, s) pairs across the warp, then across warps via smem
  for (int off = 16; off > 0; off >>= 1) {
    const float om = __shfl_xor_sync(0xffffffff, mx, off);
    const float os = __shfl_xor_sync(0xffffffff, se, off);
    const float M = fmaxf(mx, om);
    se = (mx == -INFINITY && om == -INFINITY) ? 0.0f
                                              : se * ce_exp(mx - M) + os * ce_exp(om - M);
    mx = M;
  }
  if ((mk_tid() & 31) == 0) {
    scratch[(mk_tid() >> 5) * 2] = mx;
    scratch[(mk_tid() >> 5) * 2 + 1] = se;
  }
  consumer_sync();
  if (mk_tid() == 0) {
    mx = scratch[0];
    se = scratch[1];
    for (int w = 1; w < MK_CONSUMERS / 32; ++w) {
      const float om = scratch[w * 2], os = scratch[w * 2 + 1];
      const float M = fmaxf(mx, om);
      se = se * ce_exp(mx - M) + os * ce_exp(om - M);
      mx = M;
    }
    scratch[0] = mx;
    scratch[1] = se;
  }
  consumer_sync();
  mx = scratch[0];
  se = scratch[1];
  consumer_sync();
  const float lse = mx + logf(se);
  if (mk_tid() == 0) {
    lse_out[tile] = lse;
    if (label >= 0) atomicAdd(loss, (lse - bf2f(z[label])) * inv_valid);
  }
}

// bwd: dlogits[r,v] = inv_valid * (softmax - onehot) for valid rows else 0. IN PLACE over
// the logits buffer. args: {logits, labels, lse, inv_valid, V}; tile = row.
__device__ void op_ce_bwd(const Instr& I, int tile, void** bufs) {
  const int V = I.args[4];
  bf16* z = reinterpret_cast<bf16*>(bufs[I.args[0]]) + (int64_t)tile * V;
  const int label = reinterpret_cast<const int*>(bufs[I.args[1]])[tile];
  const float lse = reinterpret_cast<const float*>(bufs[I.args[2]])[tile];
  const float inv_valid = *reinterpret_cast<const float*>(bufs[I.args[3]]);
  const float scale = (label >= 0) ? inv_valid : 0.0f;
#ifdef MK_CE_BWD_LABEL_FIXUP
  float label_out = 0.0f;
  if (label >= 0 && mk_tid() == 0) {
    const float p = ce_bwd_exp(bf2f(z[label]) - lse);
    label_out = scale * (p - 1.0f);
  }
#endif
  if ((V & 7) == 0) {  // uint4 IO (v3 P4b): the scalar loop was ~2KB/row/thread of
    // 2-byte accesses on the fattest activation buffer — latency-bound at 8 warps.
    // libm expf is the default: bitwise-identical dlogits vs the reference path (the
    // peer session measured __expf here and reverted it). MK_CE_BWD_EXP2_APPROX probes
    // whether the ex2.approx helper used by lm-head is accurate enough here.
    for (int i = mk_tid() * 8; i < V; i += MK_CONSUMERS * 8) {
      float zv[8];
      ld8bf(z + i, zv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float p = ce_bwd_exp(zv[j] - lse);
#ifdef MK_CE_BWD_LABEL_FIXUP
        zv[j] = scale * p;
#else
        zv[j] = scale * (p - (i + j == label ? 1.0f : 0.0f));
#endif
      }
      st8bf(z + i, zv);
    }
  } else {
    for (int i = mk_tid(); i < V; i += MK_CONSUMERS) {
      const float p = ce_bwd_exp(bf2f(z[i]) - lse);
#ifdef MK_CE_BWD_LABEL_FIXUP
      z[i] = f2bf(scale * p);
#else
      z[i] = f2bf(scale * (p - (i == label ? 1.0f : 0.0f)));
#endif
    }
  }
#ifdef MK_CE_BWD_LABEL_FIXUP
  consumer_sync();
  if (label >= 0 && mk_tid() == 0) z[label] = f2bf(label_out);
#endif
}
