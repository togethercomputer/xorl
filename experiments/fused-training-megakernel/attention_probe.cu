// Phase 5 probe: wgmma-based causal GQA flash attention (fwd + FA2 two-pass bwd).
// Standalone torch extension "xorl_attn_probe" — plain launches (grid = #tiles,
// 256 threads = 2 consumer warpgroups), NOT persistent. Drop-in op semantics:
//   fwd : qkv_r [S,(nq+2nkv)*D] bf16 -> O [S,nq*D] bf16, LSE [nq,S] f32; tile qt*nq+qh
//   dkv : (qkv_r, dO, LSE, Drow) -> fp32 atomicAdd into dqkv_f32 [S,stride] kv columns
//   dq  : (qkv_r, dO, LSE, Drow) -> fp32 atomicAdd into dqkv_f32 q columns
// Restrictions (probe): D == 64, S % 128 == 0 (model shapes: S=512/1024, D=64;
// D=128 falls back to the existing op at routing time).
//
// THE LAYOUT TRICK (validated by probe_views, milestone A): one 64x64 bf16 smem
// arrangement serves every operand role.
//   off64(r,c) = ((r>>3)<<10) + ((c>>3)<<7) + ((r&7)<<4) + ((c&7)<<1)      (8KB tile)
// Core matrices are the standard 8x8 INTER cores ((r&7)<<4 + (c&7)<<1); the same bytes
// admit two descriptor views (no swizzle, base_offset 0):
//   K-view : operand [mn=r, k=c] -> LBO=128B (c8-group), SBO=1024B (r8-group),
//            wgmma ktile step +256B.
//   MN-view: operand [k=r, mn=c] -> LBO=1024B (r8-group), SBO=128B (c8-group),
//            wgmma ktile step +2048B.  (byte-identical to ops.cuh wg_desc_mn blocks)
// So a row-major-loaded X[rows,cols] tile is simultaneously "X" (K-view A), "X^T as B"
// (K-view B), and "X as B / X^T as A" (MN-views) — descriptor-major transposes, zero
// data movement. Backward reuses each streamed Q/dO/K stage in two gemm roles this way.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

// Megakernel op-code invariants (ops must transplant verbatim into the ws executor):
// stride by MK_CONSUMERS, sync with consumer_sync() — never __syncthreads/blockDim.
#define MK_CONSUMERS 256
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16(v); }
// Generic-proxy smem stores (the register softmax writing P/dS) must be fenced into
// the async proxy before wgmma reads them. cp.async stores don't need this.
__device__ __forceinline__ void fence_smem_to_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ int off64(int r, int c) {
  return ((r >> 3) << 10) + ((c >> 3) << 7) + ((r & 7) << 4) + ((c & 7) << 1);
}

__device__ __forceinline__ uint64_t desc_k(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = addr >> 4;
  d.bitfield.leading_byte_offset_ = 128 >> 4;   // c8 (k) group stride
  d.bitfield.stride_byte_offset_ = 1024 >> 4;   // r8 (mn) group stride
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

__device__ __forceinline__ uint64_t desc_mn(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = addr >> 4;
  d.bitfield.leading_byte_offset_ = 1024 >> 4;  // r8 (k) group stride
  d.bitfield.stride_byte_offset_ = 128 >> 4;    // c8 (mn) group stride
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

#define FMA32 d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], \
    d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], \
    d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]

using MMA_KK = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>;
using MMA_KMN = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>;
using MMA_MNK = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>;
using MMA_MNMN = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>;

// d += A(view) @ B(view) over the 64-deep K dim (4 ktiles). Branch-free ScaleOut::One
// over pre-zeroed d (a data-dependent ScaleOut serializes wgmma ~60x).
template <class MMA, bool A_MN, bool B_MN>
__device__ __forceinline__ void mma_tile64(const bf16* A, const bf16* B, float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint64_t da =
        A_MN ? desc_mn((const char*)A + s * 2048) : desc_k((const char*)A + s * 256);
    const uint64_t db =
        B_MN ? desc_mn((const char*)B + s * 2048) : desc_k((const char*)B + s * 256);
    MMA::fma(da, db, FMA32, SG::ScaleOut::One);
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// two accumulations in one commit batch (bwd: dV then dK back-to-back)
template <class MMA, bool A_MN, bool B_MN>
__device__ __forceinline__ void mma_tile64_x2(const bf16* A1, const bf16* B1,
                                              float (&d1)[32], const bf16* A2,
                                              const bf16* B2, float (&d2)[32]) {
  cute::warpgroup_arrive();
  {
    float(&d)[32] = d1;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint64_t da =
          A_MN ? desc_mn((const char*)A1 + s * 2048) : desc_k((const char*)A1 + s * 256);
      const uint64_t db =
          B_MN ? desc_mn((const char*)B1 + s * 2048) : desc_k((const char*)B1 + s * 256);
      MMA::fma(da, db, FMA32, SG::ScaleOut::One);
    }
  }
  {
    float(&d)[32] = d2;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint64_t da =
          A_MN ? desc_mn((const char*)A2 + s * 2048) : desc_k((const char*)A2 + s * 256);
      const uint64_t db =
          B_MN ? desc_mn((const char*)B2 + s * 2048) : desc_k((const char*)B2 + s * 256);
      MMA::fma(da, db, FMA32, SG::ScaleOut::One);
    }
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// ---- milestone A: descriptor/view validation ------------------------------------------
// One warpgroup, A/B 64x64 bf16 row-major in gmem, loaded into the off64 arrangement
// with generic stores (also validates fence_smem_to_async before wgmma).
//   mode 0: C = A @ B^T   (K-view A, K-view B)
//   mode 1: C = A @ B     (K-view A, MN-view B)
//   mode 2: C = A^T @ B^T (MN-view A, K-view B)
//   mode 3: C = A^T @ B   (MN-view A, MN-view B)
__global__ void __maxnreg__(224) probe_views_kernel(const bf16* __restrict__ A,
                                                    const bf16* __restrict__ B,
                                                    float* __restrict__ C, int mode) {
  __shared__ __align__(16) bf16 tA[4096], tB[4096];
  const int tid = threadIdx.x;  // 128 threads
  for (int i = tid; i < 4096; i += 128) {
    const int r = i >> 6, c = i & 63;
    *(bf16*)((char*)tA + off64(r, c)) = A[i];
    *(bf16*)((char*)tB + off64(r, c)) = B[i];
  }
  fence_smem_to_async();
  __syncthreads();
  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.0f;
  if (mode == 0) mma_tile64<MMA_KK, false, false>(tA, tB, d);
  if (mode == 1) mma_tile64<MMA_KMN, false, true>(tA, tB, d);
  if (mode == 2) mma_tile64<MMA_MNK, true, false>(tA, tB, d);
  if (mode == 3) mma_tile64<MMA_MNMN, true, true>(tA, tB, d);
  // m64n64 f32 accumulator map: thread t=(w,l): d[n8*4+i*2+j] -> C[w*16+l/4+8i][n8*8+(l%4)*2+j]
  const int w = tid >> 5, l = tid & 31;
#pragma unroll
  for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        C[(w * 16 + (l >> 2) + 8 * i) * 64 + n8 * 8 + (l & 3) * 2 + j] = d[n8 * 4 + i * 2 + j];
}

// ---- forward ---------------------------------------------------------------------------
// Block tile = (head, 128 q rows), tile index = qt*nq + qh (qt-outer, drop-in order).
// WG w owns q rows [q0 + 64w, q0 + 64w + 64); K/V streamed in shared 64-row 2-stage
// cp.async ping-pong (same kv head for both WGs). Register online softmax on the wgmma
// accumulator layout: thread t of warp w holds rows w*16 + t/4 + {0,8}, cols
// (t%4)*2 + n8*8 + {0,1}; each row lives on the 4 lanes of a quad -> row max/sum are
// 2 shfl_xor steps (masks 1, 2). O accumulator stays in registers across stages.

struct __align__(1024) AttnFwdSmem {  // 64KB
  bf16 Q[2][4096];  // [wg]    q rows, K-view A of S = Q K^T
  bf16 P[2][4096];  // [wg]    P [q,kv], K-view A of O += P V
  bf16 K[2][4096];  // [stage] kv rows, K-view B (= K^T)
  bf16 V[2][4096];  // [stage] kv rows, MN-view B (= V)
};

__global__ void __maxnreg__(224) attn_fwd_kernel(const bf16* __restrict__ qkv,
                                                 bf16* __restrict__ O,
                                                 float* __restrict__ LSE, int S, int nq,
                                                 int nkv, float scale) {
  constexpr int D = 64;
  extern __shared__ char smem_raw[];
  AttnFwdSmem& sm = *reinterpret_cast<AttnFwdSmem*>(smem_raw);
  const int tile = blockIdx.x;
  const int qh = tile % nq;
  const int q0 = (tile / nq) * 128;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = threadIdx.x;
  const int wg = tid >> 7, wtid = tid & 127;
  const int q0wg = q0 + wg * 64;

  auto issue_kv_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // K: 512 16B vectors over 256 threads
      const int v = tid + i * 256;
      const int r = v >> 3, c8 = (v & 7) << 3;
      __pipeline_memcpy_async((char*)sm.K[st] + off64(r, c8),
                              &qkv[(int64_t)(k0 + r) * stride + (nq + kvh) * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // V
      const int v = tid + i * 256;
      const int r = v >> 3, c8 = (v & 7) << 3;
      __pipeline_memcpy_async(
          (char*)sm.V[st] + off64(r, c8),
          &qkv[(int64_t)(k0 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
    }
    __pipeline_commit();
  };

  // Q (both WG tiles, 1024 vectors) joins stage 0's commit group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (v >> 3) & 63, c8 = (v & 7) << 3;
    __pipeline_memcpy_async((char*)sm.Q[h] + off64(r, c8),
                            &qkv[(int64_t)(q0 + h * 64 + r) * stride + qh * D + c8], 16);
  }
  issue_kv_stage(0, 0);

  float o[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  const int w = wtid >> 5, ln = wtid & 31;
  const int r0 = w * 16 + (ln >> 2);  // local row (+8 for i=1)
  const int cb = (ln & 3) * 2;        // col base within an 8-col group

  const int n_stages = q0 / 64 + 2;
  for (int t = 0; t < n_stages; ++t) {
    const int k0 = t * 64;
    if (t + 1 < n_stages) issue_kv_stage((t + 1) * 64, (t + 1) & 1);
    __pipeline_wait_prior(t + 1 < n_stages ? 1 : 0);
    consumer_sync();
    const bool skip = k0 > q0wg + 63;  // WG0's tail stage: fully masked
    float alpha[2];
    if (!skip) {
      float s[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = 0.0f;
      mma_tile64<MMA_KK, false, false>(sm.Q[wg], sm.K[t & 1], s);  // S = Q K^T
      const bool masked = k0 + 63 > q0wg;                          // diagonal stage
      float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qr = q0wg + r0 + 8 * i;
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int idx = n8 * 4 + i * 2 + j;
            float sc = s[idx] * scale;
            if (masked && k0 + n8 * 8 + cb + j > qr) sc = -INFINITY;
            s[idx] = sc;
            rmax[i] = fmaxf(rmax[i], sc);
          }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffffu, rmax[i], 1));
        rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffffu, rmax[i], 2));
        const float mnew = fmaxf(m[i], rmax[i]);
        alpha[i] = __expf(m[i] - mnew);  // m=-inf only at stage 0, where mnew is finite
        m[i] = mnew;
      }
      float rsum[2] = {0.0f, 0.0f};
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const float p0 = __expf(s[idx] - m[i]);      // masked: exp(-inf)=0
          const float p1 = __expf(s[idx + 1] - m[i]);
          rsum[i] += p0 + p1;
          __nv_bfloat162 pv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.P[wg] +
                                             off64(r0 + 8 * i, n8 * 8 + cb)) = pv;
        }
        rsum[i] += __shfl_xor_sync(0xffffffffu, rsum[i], 1);
        rsum[i] += __shfl_xor_sync(0xffffffffu, rsum[i], 2);
        l[i] = l[i] * alpha[i] + rsum[i];
      }
    }
    fence_smem_to_async();
    consumer_sync();  // P visible to this WG's wgmma; V stage ready
    if (!skip) {
#pragma unroll
      for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j) o[n8 * 4 + i * 2 + j] *= alpha[i];
      mma_tile64<MMA_KMN, false, true>(sm.P[wg], sm.V[t & 1], o);  // O += P V
    }
    consumer_sync();  // both WGs done reading K/V stage t before refill
  }

  // epilogue: LSE + O from registers; O staged through smem for coalesced stores
  const float inv[2] = {1.0f / l[0], 1.0f / l[1]};
  if ((ln & 3) == 0) {
    LSE[(int64_t)qh * S + q0wg + r0] = m[0] + logf(l[0]);
    LSE[(int64_t)qh * S + q0wg + r0 + 8] = m[1] + logf(l[1]);
  }
  float* Cs = reinterpret_cast<float*>(smem_raw);  // overlay [128][68] over dead tiles
#pragma unroll
  for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        Cs[(wg * 64 + r0 + 8 * i) * 68 + n8 * 8 + cb + j] = o[n8 * 4 + i * 2 + j] * inv[i];
  consumer_sync();
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const int gid = tid + g * 256;
    const int r = gid >> 3, c8 = (gid & 7) << 3;
    uint4 out;
    bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = f2bf(Cs[r * 68 + c8 + e]);
    *reinterpret_cast<uint4*>(&O[(int64_t)(q0 + r) * (nq * D) + qh * D + c8]) = out;
  }
}

// ---- backward dK/dV pass -----------------------------------------------------------------
// Block tile = ((kvh, 128 kv rows, GQA member g), q-chunk c of C); qh = kvh*G + g.
// WG w owns kv rows [kv0 + 64w, kv0 + 64w + 64); Q/dO streamed in shared 64-row stages
// q0s = kv0 + (c + t*C)*64 (chunking trades C-fold owned-tile reloads for C-fold less
// serial latency; the fp32 atomic epilogue makes chunks race-free). P recomputed from
// LSE; dS from Drow (both inputs). dK/dV accumulate in registers; fp32 atomics at end.
// P/dS are stored once as [q,kv] tiles; the MN-view descriptor reads them as P^T/dS^T,
// and the streamed Q/dO stages are dual-viewed (K-view for S/dP, MN-view for dK/dV).

struct __align__(1024) AttnDkvSmem {  // 96KB
  bf16 K[2][4096];   // [wg] owned kv rows, K-view B (= K^T)
  bf16 V[2][4096];   // [wg] owned kv rows, K-view B (= V^T)
  bf16 P[2][4096];   // [wg] [q,kv]; MN-view A = P^T
  bf16 dS[2][4096];  // [wg] [q,kv]; MN-view A = dS^T
  bf16 Q[2][4096];   // [stage] K-view A (S = Q K^T) + MN-view B (dK += dS^T Q)
  bf16 dO[2][4096];  // [stage] K-view A (dP = dO V^T) + MN-view B (dV += P^T dO)
};

__global__ void __maxnreg__(224) attn_dkv_kernel(const bf16* __restrict__ qkv,
                                                 const bf16* __restrict__ dOg,
                                                 const float* __restrict__ LSE,
                                                 const float* __restrict__ Drow,
                                                 float* __restrict__ ws, int S, int nq,
                                                 int nkv, float scale, int C) {
  constexpr int D = 64;
  extern __shared__ char smem_raw[];
  AttnDkvSmem& sm = *reinterpret_cast<AttnDkvSmem*>(smem_raw);
  const int G = nq / nkv;
  const int n_kvt = S / 128;
  const int c = blockIdx.x % C;
  const int tile = blockIdx.x / C;
  const int kvh = tile / (n_kvt * G);
  const int rem = tile % (n_kvt * G);
  const int kv0 = (rem / G) * 128;
  const int g = rem % G;
  const int qh = kvh * G + g;
  const int stride = (nq + 2 * nkv) * D;
  const int tid = threadIdx.x;
  const int wg = tid >> 7, wtid = tid & 127;
  const int kv0wg = kv0 + wg * 64;
  const int n_stages = ((S - kv0) / 64 - c + C - 1) / C;
  if (n_stages <= 0) return;  // uniform: before any cp.async/barrier

  auto issue_qdo_stage = [&](int q0s, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 3, c8 = (v & 7) << 3;
      __pipeline_memcpy_async((char*)sm.Q[st] + off64(r, c8),
                              &qkv[(int64_t)(q0s + r) * stride + qh * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 3, c8 = (v & 7) << 3;
      __pipeline_memcpy_async((char*)sm.dO[st] + off64(r, c8),
                              &dOg[(int64_t)(q0s + r) * (nq * D) + qh * D + c8], 16);
    }
    __pipeline_commit();
  };

  // owned K/V tiles (both WGs) join stage 0's commit group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (v >> 3) & 63, c8 = (v & 7) << 3;
    __pipeline_memcpy_async(
        (char*)sm.K[h] + off64(r, c8),
        &qkv[(int64_t)(kv0 + h * 64 + r) * stride + (nq + kvh) * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (v >> 3) & 63, c8 = (v & 7) << 3;
    __pipeline_memcpy_async(
        (char*)sm.V[h] + off64(r, c8),
        &qkv[(int64_t)(kv0 + h * 64 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
  }
  issue_qdo_stage(kv0 + c * 64, 0);

  float dk[32], dv[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) dk[i] = dv[i] = 0.0f;
  const int w = wtid >> 5, ln = wtid & 31;
  const int r0 = w * 16 + (ln >> 2);
  const int cb = (ln & 3) * 2;

  for (int t = 0; t < n_stages; ++t) {
    const int q0s = kv0 + (c + t * C) * 64;
    if (t + 1 < n_stages) issue_qdo_stage(kv0 + (c + (t + 1) * C) * 64, (t + 1) & 1);
    __pipeline_wait_prior(t + 1 < n_stages ? 1 : 0);
    consumer_sync();
    const bool skip = q0s < kv0wg;  // WG1's first stage: fully masked
    if (!skip) {
      float s[32], s2[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = s2[i] = 0.0f;
      // rows of s/s2 = q stage rows; cols = this WG's kv rows
      mma_tile64_x2<MMA_KK, false, false>(sm.Q[t & 1], sm.K[wg], s,   // S  = Q  K^T
                                          sm.dO[t & 1], sm.V[wg], s2);  // dP = dO V^T
      const bool masked = q0s == kv0wg;  // diagonal stage
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qr = q0s + r0 + 8 * i;
        const float lse = LSE[(int64_t)qh * S + qr];
        const float dr = Drow[(int64_t)qh * S + qr];
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = kv0wg + n8 * 8 + cb;
          float p0 = __expf(s[idx] * scale - lse);
          float p1 = __expf(s[idx + 1] * scale - lse);
          if (masked && kr > qr) p0 = 0.0f;
          if (masked && kr + 1 > qr) p1 = 0.0f;
          const float ds0 = p0 * (s2[idx] - dr) * scale;
          const float ds1 = p1 * (s2[idx + 1] - dr) * scale;
          const int off = off64(r0 + 8 * i, n8 * 8 + cb);
          __nv_bfloat162 pv, dsv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          dsv.x = f2bf(ds0);
          dsv.y = f2bf(ds1);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.P[wg] + off) = pv;
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.dS[wg] + off) = dsv;
        }
      }
    }
    fence_smem_to_async();
    consumer_sync();  // P/dS visible; Q/dO stage still valid
    if (!skip)
      mma_tile64_x2<MMA_MNMN, true, true>(sm.P[wg], sm.dO[t & 1], dv,  // dV += P^T dO
                                          sm.dS[wg], sm.Q[t & 1], dk);  // dK += dS^T Q
    consumer_sync();  // both WGs done reading Q/dO stage t before refill
  }

  // epilogue: stage each 128x64 accumulator to smem, coalesced fp32 atomics
  float* Cs = reinterpret_cast<float*>(smem_raw);  // [128][68] overlay (K/V/P dead)
#pragma unroll
  for (int round = 0; round < 2; ++round) {
    const float(&acc)[32] = round == 0 ? dk : dv;
    const int col0 = (round == 0 ? (nq + kvh) : (nq + nkv + kvh)) * D;
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          Cs[(wg * 64 + r0 + 8 * i) * 68 + n8 * 8 + cb + j] = acc[n8 * 4 + i * 2 + j];
    consumer_sync();
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const int gid = tid + gq * 256;
      const int r = gid >> 3, c8 = (gid & 7) << 3;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        atomicAdd(&ws[(int64_t)(kv0 + r) * stride + col0 + c8 + e], Cs[r * 68 + c8 + e]);
    }
    consumer_sync();  // Cs reuse between rounds
  }
}

// ---- backward dQ pass ----------------------------------------------------------------------
// Block tile = ((qt*nq + qh), kv-chunk c of C): owns 128 q rows (64/WG); streams K/V
// stages k0 = (c + t*C)*64 (chunking as in dKV / the current op's Cq). LSE/Drow are
// register-resident (each thread's 2 rows are fixed). dQ += dS @ K reuses the K stage
// via its MN-view (dual view).

struct __align__(1024) AttnDqSmem {  // 80KB
  bf16 Q[2][4096];   // [wg] K-view A (S = Q K^T)
  bf16 dO[2][4096];  // [wg] K-view A (dP = dO V^T)
  bf16 dS[2][4096];  // [wg] [q,kv]; K-view A (dQ += dS K)
  bf16 K[2][4096];   // [stage] K-view B (= K^T) + MN-view B (= K)
  bf16 V[2][4096];   // [stage] K-view B (= V^T)
};

__global__ void __maxnreg__(224) attn_dq_kernel(const bf16* __restrict__ qkv,
                                                const bf16* __restrict__ dOg,
                                                const float* __restrict__ LSE,
                                                const float* __restrict__ Drow,
                                                float* __restrict__ ws, int S, int nq,
                                                int nkv, float scale, int C) {
  constexpr int D = 64;
  extern __shared__ char smem_raw[];
  AttnDqSmem& sm = *reinterpret_cast<AttnDqSmem*>(smem_raw);
  const int c = blockIdx.x % C;
  const int tile = blockIdx.x / C;
  const int qh = tile % nq;
  const int q0 = (tile / nq) * 128;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = threadIdx.x;
  const int wg = tid >> 7, wtid = tid & 127;
  const int q0wg = q0 + wg * 64;
  const int n_stages = (q0 / 64 + 2 - c + C - 1) / C;
  if (n_stages <= 0) return;  // uniform: before any cp.async/barrier

  auto issue_kv_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 3, c8 = (v & 7) << 3;
      __pipeline_memcpy_async((char*)sm.K[st] + off64(r, c8),
                              &qkv[(int64_t)(k0 + r) * stride + (nq + kvh) * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 3, c8 = (v & 7) << 3;
      __pipeline_memcpy_async(
          (char*)sm.V[st] + off64(r, c8),
          &qkv[(int64_t)(k0 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
    }
    __pipeline_commit();
  };

  // owned Q/dO tiles (both WGs) join stage 0's commit group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (v >> 3) & 63, c8 = (v & 7) << 3;
    __pipeline_memcpy_async((char*)sm.Q[h] + off64(r, c8),
                            &qkv[(int64_t)(q0 + h * 64 + r) * stride + qh * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (v >> 3) & 63, c8 = (v & 7) << 3;
    __pipeline_memcpy_async((char*)sm.dO[h] + off64(r, c8),
                            &dOg[(int64_t)(q0 + h * 64 + r) * (nq * D) + qh * D + c8], 16);
  }
  issue_kv_stage(c * 64, 0);

  float dq[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) dq[i] = 0.0f;
  const int w = wtid >> 5, ln = wtid & 31;
  const int r0 = w * 16 + (ln >> 2);
  const int cb = (ln & 3) * 2;
  const int qr[2] = {q0wg + r0, q0wg + r0 + 8};
  const float lse[2] = {LSE[(int64_t)qh * S + qr[0]], LSE[(int64_t)qh * S + qr[1]]};
  const float dr[2] = {Drow[(int64_t)qh * S + qr[0]], Drow[(int64_t)qh * S + qr[1]]};

  for (int t = 0; t < n_stages; ++t) {
    const int k0 = (c + t * C) * 64;
    if (t + 1 < n_stages) issue_kv_stage((c + (t + 1) * C) * 64, (t + 1) & 1);
    __pipeline_wait_prior(t + 1 < n_stages ? 1 : 0);
    consumer_sync();
    const bool skip = k0 > q0wg + 63;  // WG0's tail stage
    if (!skip) {
      float s[32], s2[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = s2[i] = 0.0f;
      mma_tile64_x2<MMA_KK, false, false>(sm.Q[wg], sm.K[t & 1], s,   // S  = Q  K^T
                                          sm.dO[wg], sm.V[t & 1], s2);  // dP = dO V^T
      const bool masked = k0 + 63 > q0wg;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = k0 + n8 * 8 + cb;
          float p0 = __expf(s[idx] * scale - lse[i]);
          float p1 = __expf(s[idx + 1] * scale - lse[i]);
          if (masked && kr > qr[i]) p0 = 0.0f;
          if (masked && kr + 1 > qr[i]) p1 = 0.0f;
          const float ds0 = p0 * (s2[idx] - dr[i]) * scale;
          const float ds1 = p1 * (s2[idx + 1] - dr[i]) * scale;
          __nv_bfloat162 dsv;
          dsv.x = f2bf(ds0);
          dsv.y = f2bf(ds1);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.dS[wg] +
                                             off64(r0 + 8 * i, n8 * 8 + cb)) = dsv;
        }
      }
    }
    fence_smem_to_async();
    consumer_sync();  // dS visible; K stage still valid
    if (!skip)
      mma_tile64<MMA_KMN, false, true>(sm.dS[wg], sm.K[t & 1], dq);  // dQ += dS K
    consumer_sync();  // both WGs done reading K/V stage t before refill
  }

  // epilogue: stage dQ, coalesced fp32 atomics into the q columns
  float* Cs = reinterpret_cast<float*>(smem_raw);  // [128][68] overlay
#pragma unroll
  for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        Cs[(wg * 64 + r0 + 8 * i) * 68 + n8 * 8 + cb + j] = dq[n8 * 4 + i * 2 + j];
  consumer_sync();
#pragma unroll
  for (int gq = 0; gq < 4; ++gq) {
    const int gid = tid + gq * 256;
    const int r = gid >> 3, c8 = (gid & 7) << 3;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      atomicAdd(&ws[(int64_t)(q0 + r) * stride + qh * D + c8 + e], Cs[r * 68 + c8 + e]);
  }
}

// ---- launchers -------------------------------------------------------------------------

static void set_smem(const void* kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      bytes));
}

void probe_views(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t mode) {
  probe_views_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(A.data_ptr()), reinterpret_cast<const bf16*>(B.data_ptr()),
      C.data_ptr<float>(), (int)mode);
  C10_CUDA_CHECK(cudaGetLastError());
}

void attn_fwd(torch::Tensor qkv, torch::Tensor O, torch::Tensor LSE, int64_t S, int64_t nq,
              int64_t nkv, double scale) {
  TORCH_CHECK(S % 128 == 0, "probe requires S % 128 == 0");
  static bool cfg = false;
  if (!cfg) {
    set_smem((const void*)attn_fwd_kernel, sizeof(AttnFwdSmem));
    cfg = true;
  }
  const int tiles = (int)(nq * (S / 128));
  attn_fwd_kernel<<<tiles, 256, sizeof(AttnFwdSmem), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), reinterpret_cast<bf16*>(O.data_ptr()),
      LSE.data_ptr<float>(), (int)S, (int)nq, (int)nkv, (float)scale);
  C10_CUDA_CHECK(cudaGetLastError());
}

void attn_dkv(torch::Tensor qkv, torch::Tensor dO, torch::Tensor LSE, torch::Tensor Drow,
              torch::Tensor ws, int64_t S, int64_t nq, int64_t nkv, double scale,
              int64_t C) {
  TORCH_CHECK(S % 128 == 0, "probe requires S % 128 == 0");
  static bool cfg = false;
  if (!cfg) {
    set_smem((const void*)attn_dkv_kernel, sizeof(AttnDkvSmem));
    cfg = true;
  }
  const int tiles = (int)(nkv * (S / 128) * (nq / nkv) * C);
  attn_dkv_kernel<<<tiles, 256, sizeof(AttnDkvSmem), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), reinterpret_cast<const bf16*>(dO.data_ptr()),
      LSE.data_ptr<float>(), Drow.data_ptr<float>(), ws.data_ptr<float>(), (int)S, (int)nq,
      (int)nkv, (float)scale, (int)C);
  C10_CUDA_CHECK(cudaGetLastError());
}

void attn_dq(torch::Tensor qkv, torch::Tensor dO, torch::Tensor LSE, torch::Tensor Drow,
             torch::Tensor ws, int64_t S, int64_t nq, int64_t nkv, double scale,
             int64_t C) {
  TORCH_CHECK(S % 128 == 0, "probe requires S % 128 == 0");
  static bool cfg = false;
  if (!cfg) {
    set_smem((const void*)attn_dq_kernel, sizeof(AttnDqSmem));
    cfg = true;
  }
  const int tiles = (int)(nq * (S / 128) * C);
  attn_dq_kernel<<<tiles, 256, sizeof(AttnDqSmem), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), reinterpret_cast<const bf16*>(dO.data_ptr()),
      LSE.data_ptr<float>(), Drow.data_ptr<float>(), ws.data_ptr<float>(), (int)S, (int)nq,
      (int)nkv, (float)scale, (int)C);
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_views", &probe_views, "milestone A: descriptor view validation");
  m.def("attn_fwd", &attn_fwd, "wgmma causal GQA attention forward");
  m.def("attn_dkv", &attn_dkv, "wgmma attention backward dK/dV pass");
  m.def("attn_dq", &attn_dq, "wgmma attention backward dQ pass");
}
