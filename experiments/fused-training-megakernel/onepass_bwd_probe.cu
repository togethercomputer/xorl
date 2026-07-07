// One-pass D64 attention-bwd STANDALONE feasibility probe (session b1d36305).
// Standalone torch extension "xorl_onepass_probe" — plain launches (grid = #tiles,
// 256 threads = 2 consumer warpgroups), NOT persistent. Conventions and the dual-view
// 64x64 layout are copied from attention_probe.cu (Phase-5 probe, validated).
//
// Kernels:
//   attn_dkv2 : two-pass dKV baseline at CURRENT-op feature level (exp2 prebias +
//               LSE/Drow smem prefetch) — mirrors op_attn_dkv_wg @2a41f6a.
//   attn_dq2  : two-pass dQ baseline at CURRENT-op feature level (exp2 prebias +
//               RS register-A dS feed + fp32-P dS + C==1 direct store) — mirrors
//               op_attn_dq_wg with the s4096/s8192 promoted gates.
//   attn_onepass<DQ_RS, DRAIN> : FA4-structure one-pass — CTA owns 128 kv rows
//               (64/WG persistent K/V + register dK/dV), streams 64-row Q/dO stages
//               from the diagonal; S+dP computed ONCE per stage; dV += P^T dO and
//               dK += dS^T Q via smem MN-view; per-stage dQp = dS @ K_own.
//               DQ_RS: dS feeds the dQ gemm from registers (the C-fragment IS the
//               K-major A fragment in our M=q orientation) vs from smem (K-view A).
//               DRAIN 0: per-thread float2 fp32 atomics from dqp regs (the OLD
//               refutation mechanism, correctness baseline).
//               DRAIN 1: stage dqp into a per-WG smem slab and issue ONE
//               cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 per WG
//               per stage (FA4-style zero-atomic drain; slab single-buffered with a
//               one-stage bulk-group flight). The slab uses a fragment-interleaved
//               [pair][thread] float2 order (bank-conflict-free store; natural
//               [64][64] row-major would be 8-way conflicted since the 256B row
//               stride aliases banks). dQaccum is a separate contiguous fp32
//               [nq, S, D] tensor whose per-stage 16KB blocks carry that internal
//               order for DRAIN==1 (host unscrambles); DRAIN==0 writes natural
//               [nq, S, D] order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

#define MK_CONSUMERS 256
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}
// per-warpgroup barrier (128 threads), ids 2/3 — used by the DRAIN==1 slab handshake
__device__ __forceinline__ void wg_sync(int wg) {
  asm volatile("bar.sync %0, 128;" ::"r"(2 + wg) : "memory");
}
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ void fence_smem_to_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

static constexpr float LOG2E = 1.4426950408889634f;
__device__ __forceinline__ float exp2_prebias(float score, float scale_l2,
                                              float neg_lse_l2) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(fmaf(score, scale_l2, neg_lse_l2)));
  return y;
}

// proven 1-D f32 bulk reduce idiom (cp-reduce-1d-unit-probe-2a41f6a.md: UBLKRED.G.S.ADD.F32.RN;
// multi-CTA same-row collisions exact per cp-reduce-collision-2a41f6a-k8s-20260707T0800Z.log)
__device__ __forceinline__ void bulk_reduce_add_f32(float* dst, const float* src_smem,
                                                    uint32_t bytes) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(src_smem);
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
      :
      : "l"(dst), "r"(s), "r"(bytes)
      : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ int off64(int r, int c) {
  return ((r >> 3) << 10) + ((c >> 3) << 7) + ((r & 7) << 4) + ((c & 7) << 1);
}

__device__ __forceinline__ uint64_t desc_k(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = addr >> 4;
  d.bitfield.leading_byte_offset_ = 128 >> 4;
  d.bitfield.stride_byte_offset_ = 1024 >> 4;
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

__device__ __forceinline__ uint64_t desc_mn(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = addr >> 4;
  d.bitfield.leading_byte_offset_ = 1024 >> 4;
  d.bitfield.stride_byte_offset_ = 128 >> 4;
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

#define FMA32 d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], \
    d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], \
    d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]

using MMA_KK = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>;
using MMA_KMN = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>;
using MMA_MNMN = SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::MN>;
using MMA_RS = SG::MMA_64x64x16_F32BF16BF16_RS<SG::Major::K, SG::Major::MN>;

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

// dQ RS feed (copied from wga_mma64_rs): dS is already in the C-fragment layout for a
// K-major A operand with q rows as M.
__device__ __forceinline__ void mma_tile64_rs(const uint32_t (&a)[16], const bf16* B,
                                              float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA_RS::fma(a[4 * s], a[4 * s + 1], a[4 * s + 2], a[4 * s + 3],
                desc_mn((const char*)B + s * 2048), FMA32, SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// one-pass phase-B batch: dV + dK (SS MN/MN) + dQp (RS from areg, or SS K/MN from smem
// dS) in ONE commit; mixing majors/RS inside a batch is fine (independent HGMMAs);
// the wait<0> retires all three.
template <bool DQ_RS>
__device__ __forceinline__ void mma_onepass_batch(const bf16* P, const bf16* dO,
                                                  float (&dv)[32], const bf16* dSs,
                                                  const bf16* Q, float (&dk)[32],
                                                  const uint32_t (&a)[16], const bf16* Kw,
                                                  float (&dqp)[32]) {
  cute::warpgroup_arrive();
  {
    float(&d)[32] = dv;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      MMA_MNMN::fma(desc_mn((const char*)P + s * 2048), desc_mn((const char*)dO + s * 2048),
                    FMA32, SG::ScaleOut::One);
  }
  {
    float(&d)[32] = dk;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      MMA_MNMN::fma(desc_mn((const char*)dSs + s * 2048), desc_mn((const char*)Q + s * 2048),
                    FMA32, SG::ScaleOut::One);
  }
  {
    float(&d)[32] = dqp;
    if (DQ_RS) {
#pragma unroll
      for (int s = 0; s < 4; ++s)
        MMA_RS::fma(a[4 * s], a[4 * s + 1], a[4 * s + 2], a[4 * s + 3],
                    desc_mn((const char*)Kw + s * 2048), FMA32, SG::ScaleOut::One);
    } else {
#pragma unroll
      for (int s = 0; s < 4; ++s)
        MMA_KMN::fma(desc_k((const char*)dSs + s * 256), desc_mn((const char*)Kw + s * 2048),
                     FMA32, SG::ScaleOut::One);
    }
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// ---- two-pass dKV baseline (current-op feature level) --------------------------------

struct __align__(1024) Dkv2Smem {  // 97KB
  bf16 K[2][4096];
  bf16 V[2][4096];
  bf16 P[2][4096];
  bf16 dS[2][4096];
  bf16 Q[2][4096];
  bf16 dO[2][4096];
  float LSEs[2][64];
  float Drows[2][64];
};

__global__ void __maxnreg__(224) attn_dkv2_kernel(const bf16* __restrict__ qkv,
                                                  const bf16* __restrict__ dOg,
                                                  const float* __restrict__ LSE,
                                                  const float* __restrict__ Drow,
                                                  float* __restrict__ ws, int S, int nq,
                                                  int nkv, float scale, int C) {
  constexpr int D = 64;
  extern __shared__ char smem_raw[];
  Dkv2Smem& sm = *reinterpret_cast<Dkv2Smem*>(smem_raw);
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
  if (n_stages <= 0) return;
  const float scale_l2 = scale * LOG2E;

  auto issue_qdo_stage = [&](int q0s, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.Q[st] + off64(r, c8),
                              &qkv[(int64_t)(q0s + r) * stride + qh * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.dO[st] + off64(r, c8),
                              &dOg[(int64_t)(q0s + r) * (nq * D) + qh * D + c8], 16);
    }
    if (tid < 16)
      __pipeline_memcpy_async(&sm.LSEs[st][tid * 4], &LSE[(int64_t)qh * S + q0s + tid * 4], 16);
    else if (tid < 32)
      __pipeline_memcpy_async(&sm.Drows[st][(tid - 16) * 4],
                              &Drow[(int64_t)qh * S + q0s + (tid - 16) * 4], 16);
    __pipeline_commit();
  };

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async(
        (char*)sm.K[h] + off64(r, c8),
        &qkv[(int64_t)(kv0 + h * 64 + r) * stride + (nq + kvh) * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
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
    const bool skip = q0s < kv0wg;
    if (!skip) {
      float s[32], s2[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = s2[i] = 0.0f;
      mma_tile64_x2<MMA_KK, false, false>(sm.Q[t & 1], sm.K[wg], s,     // S  = Q  K^T
                                          sm.dO[t & 1], sm.V[wg], s2);  // dP = dO V^T
      const bool masked = q0s == kv0wg;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qr = q0s + r0 + 8 * i;
        const float lse = sm.LSEs[t & 1][r0 + 8 * i];
        const float dr = sm.Drows[t & 1][r0 + 8 * i];
        const float neg_lse_l2 = -lse * LOG2E;
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = kv0wg + n8 * 8 + cb;
          float p0 = exp2_prebias(s[idx], scale_l2, neg_lse_l2);
          float p1 = exp2_prebias(s[idx + 1], scale_l2, neg_lse_l2);
          if (masked && kr > qr) p0 = 0.0f;
          if (masked && kr + 1 > qr) p1 = 0.0f;
          const int off = off64(r0 + 8 * i, n8 * 8 + cb);
          __nv_bfloat162 pv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.P[wg] + off) = pv;
          __nv_bfloat162 dsv;
          dsv.x = f2bf(bf2f(pv.x) * (s2[idx] - dr) * scale);
          dsv.y = f2bf(bf2f(pv.y) * (s2[idx + 1] - dr) * scale);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.dS[wg] + off) = dsv;
        }
      }
    }
    fence_smem_to_async();
    consumer_sync();
    if (!skip)
      mma_tile64_x2<MMA_MNMN, true, true>(sm.P[wg], sm.dO[t & 1], dv,   // dV += P^T dO
                                          sm.dS[wg], sm.Q[t & 1], dk);  // dK += dS^T Q
    consumer_sync();
  }

  float* Cs = reinterpret_cast<float*>(smem_raw);
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
    consumer_sync();
  }
}

// ---- two-pass dQ baseline (current-op feature level) ---------------------------------

struct __align__(1024) Dq2Smem {  // 80KB
  bf16 Q[2][4096];
  bf16 dO[2][4096];
  bf16 dS[2][4096];  // unused under RS feed; kept for layout parity
  bf16 K[2][4096];
  bf16 V[2][4096];
};

__global__ void __maxnreg__(224) attn_dq2_kernel(const bf16* __restrict__ qkv,
                                                 const bf16* __restrict__ dOg,
                                                 const float* __restrict__ LSE,
                                                 const float* __restrict__ Drow,
                                                 float* __restrict__ ws, int S, int nq,
                                                 int nkv, float scale, int C) {
  constexpr int D = 64;
  extern __shared__ char smem_raw[];
  Dq2Smem& sm = *reinterpret_cast<Dq2Smem*>(smem_raw);
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
  if (n_stages <= 0) return;
  const float scale_l2 = scale * LOG2E;

  auto issue_kv_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.K[st] + off64(r, c8),
                              &qkv[(int64_t)(k0 + r) * stride + (nq + kvh) * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async(
          (char*)sm.V[st] + off64(r, c8),
          &qkv[(int64_t)(k0 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
    }
    __pipeline_commit();
  };

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async((char*)sm.Q[h] + off64(r, c8),
                            &qkv[(int64_t)(q0 + h * 64 + r) * stride + qh * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
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
  const float neg_lse_l2[2] = {-lse[0] * LOG2E, -lse[1] * LOG2E};

  for (int t = 0; t < n_stages; ++t) {
    const int k0 = (c + t * C) * 64;
    if (t + 1 < n_stages) issue_kv_stage((c + (t + 1) * C) * 64, (t + 1) & 1);
    __pipeline_wait_prior(t + 1 < n_stages ? 1 : 0);
    consumer_sync();
    const bool skip = k0 > q0wg + 63;
    if (!skip) {
      float s[32], s2[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = s2[i] = 0.0f;
      mma_tile64_x2<MMA_KK, false, false>(sm.Q[wg], sm.K[t & 1], s,     // S  = Q  K^T
                                          sm.dO[wg], sm.V[t & 1], s2);  // dP = dO V^T
      const bool masked = k0 + 63 > q0wg;
      uint32_t areg[16];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = k0 + n8 * 8 + cb;
          float p0 = exp2_prebias(s[idx], scale_l2, neg_lse_l2[i]);
          float p1 = exp2_prebias(s[idx + 1], scale_l2, neg_lse_l2[i]);
          if (masked && kr > qr[i]) p0 = 0.0f;
          if (masked && kr + 1 > qr[i]) p1 = 0.0f;
          __nv_bfloat162 dsv;  // fp32-P dS (the promoted MK_ATTN_DQ_FP32_P path)
          dsv.x = f2bf(p0 * (s2[idx] - dr[i]) * scale);
          dsv.y = f2bf(p1 * (s2[idx + 1] - dr[i]) * scale);
          areg[4 * (n8 >> 1) + (n8 & 1) * 2 + i] = *reinterpret_cast<uint32_t*>(&dsv);
        }
      }
      mma_tile64_rs(areg, sm.K[t & 1], dq);  // dQ += dS K
    }
    consumer_sync();  // both WGs done reading K/V stage t before refill
  }

  if (C == 1) {  // one writer per q slice: direct store
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          ws[(int64_t)qr[i] * stride + qh * D + n8 * 8 + cb + j] = dq[n8 * 4 + i * 2 + j];
    return;
  }
  float* Cs = reinterpret_cast<float*>(smem_raw);
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

// ---- one-pass ------------------------------------------------------------------------

struct __align__(1024) OnepassSmem {  // 97KB (+32KB slabs when DRAIN==1)
  bf16 K[2][4096];
  bf16 V[2][4096];
  bf16 P[2][4096];
  bf16 dS[2][4096];
  bf16 Q[2][4096];
  bf16 dO[2][4096];
  float LSEs[2][64];
  float Drows[2][64];
  float dqslab[2][4096];  // [wg] 16KB fragment-interleaved [pair][thread] float2
};

template <bool DQ_RS, int DRAIN>
__global__ void __maxnreg__(224) attn_onepass_kernel(
    const bf16* __restrict__ qkv, const bf16* __restrict__ dOg,
    const float* __restrict__ LSE, const float* __restrict__ Drow,
    float* __restrict__ ws, float* __restrict__ dqa, int S, int nq, int nkv, float scale,
    int C) {
  constexpr int D = 64;
  extern __shared__ char smem_raw[];
  OnepassSmem& sm = *reinterpret_cast<OnepassSmem*>(smem_raw);
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
  if (n_stages <= 0) return;
  const float scale_l2 = scale * LOG2E;

  auto issue_qdo_stage = [&](int q0s, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.Q[st] + off64(r, c8),
                              &qkv[(int64_t)(q0s + r) * stride + qh * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.dO[st] + off64(r, c8),
                              &dOg[(int64_t)(q0s + r) * (nq * D) + qh * D + c8], 16);
    }
    if (tid < 16)
      __pipeline_memcpy_async(&sm.LSEs[st][tid * 4], &LSE[(int64_t)qh * S + q0s + tid * 4], 16);
    else if (tid < 32)
      __pipeline_memcpy_async(&sm.Drows[st][(tid - 16) * 4],
                              &Drow[(int64_t)qh * S + q0s + (tid - 16) * 4], 16);
    __pipeline_commit();
  };

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async(
        (char*)sm.K[h] + off64(r, c8),
        &qkv[(int64_t)(kv0 + h * 64 + r) * stride + (nq + kvh) * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * 256;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
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
    uint32_t areg[16];
    if (!skip) {
      float s[32], s2[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = s2[i] = 0.0f;
      mma_tile64_x2<MMA_KK, false, false>(sm.Q[t & 1], sm.K[wg], s,     // S  = Q  K^T
                                          sm.dO[t & 1], sm.V[wg], s2);  // dP = dO V^T
      const bool masked = q0s == kv0wg;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qr = q0s + r0 + 8 * i;
        const float lse = sm.LSEs[t & 1][r0 + 8 * i];
        const float dr = sm.Drows[t & 1][r0 + 8 * i];
        const float neg_lse_l2 = -lse * LOG2E;
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = kv0wg + n8 * 8 + cb;
          float p0 = exp2_prebias(s[idx], scale_l2, neg_lse_l2);
          float p1 = exp2_prebias(s[idx + 1], scale_l2, neg_lse_l2);
          if (masked && kr > qr) p0 = 0.0f;
          if (masked && kr + 1 > qr) p1 = 0.0f;
          const int off = off64(r0 + 8 * i, n8 * 8 + cb);
          __nv_bfloat162 pv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.P[wg] + off) = pv;
          __nv_bfloat162 dsv;  // fp32-P dS (FA4 computes dS from the fp32 P accumulator)
          dsv.x = f2bf(p0 * (s2[idx] - dr) * scale);
          dsv.y = f2bf(p1 * (s2[idx + 1] - dr) * scale);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.dS[wg] + off) = dsv;
          if (DQ_RS) areg[4 * (n8 >> 1) + (n8 & 1) * 2 + i] = *reinterpret_cast<uint32_t*>(&dsv);
        }
      }
    }
    fence_smem_to_async();
    consumer_sync();  // P/dS visible; Q/dO stage still valid
    float dqp[32];
    if (!skip) {
#pragma unroll
      for (int i = 0; i < 32; ++i) dqp[i] = 0.0f;
      mma_onepass_batch<DQ_RS>(sm.P[wg], sm.dO[t & 1], dv,   // dV += P^T dO
                               sm.dS[wg], sm.Q[t & 1], dk,   // dK += dS^T Q
                               areg, sm.K[wg], dqp);         // dQp = dS K_own
    }
    consumer_sync();  // both WGs done reading Q/dO stage t before refill
    if (!skip) {
      if (DRAIN == 0) {
        // old-refutation mechanism: float2 fp32 atomics straight from registers
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
          for (int i = 0; i < 2; ++i)
            atomicAdd(reinterpret_cast<float2*>(
                          &dqa[((int64_t)qh * S + q0s + r0 + 8 * i) * D + n8 * 8 + cb]),
                      make_float2(dqp[n8 * 4 + i * 2], dqp[n8 * 4 + i * 2 + 1]));
      } else {
        // FA4-style: slab + one bulk reduce per WG per stage (skip is WG-uniform, so
        // the per-WG barrier stays consistent). Slab single-buffered: wait out the
        // previous stage's in-flight bulk group before overwriting.
        if (wtid == 0) bulk_wait0();
        wg_sync(wg);
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
          for (int i = 0; i < 2; ++i)
            *reinterpret_cast<float2*>(&sm.dqslab[wg][((n8 * 2 + i) * 128 + wtid) * 2]) =
                make_float2(dqp[n8 * 4 + i * 2], dqp[n8 * 4 + i * 2 + 1]);
        fence_smem_to_async();
        wg_sync(wg);
        if (wtid == 0)
          bulk_reduce_add_f32(&dqa[((int64_t)qh * S + q0s) * D], sm.dqslab[wg], 64 * D * 4);
      }
    }
  }

  // dK/dV epilogue (as dkv): Cs overlay lives in the K/V/P/dS region, disjoint from
  // dqslab, so an in-flight final bulk reduce is safe; wait it out before exit.
  float* Cs = reinterpret_cast<float*>(smem_raw);
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
    consumer_sync();
  }
  if (DRAIN == 1 && wtid == 0) bulk_wait0();
}

// ---- launchers -------------------------------------------------------------------------

static void set_smem(const void* kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      bytes));
}

void attn_dkv2(torch::Tensor qkv, torch::Tensor dO, torch::Tensor LSE, torch::Tensor Drow,
               torch::Tensor ws, int64_t S, int64_t nq, int64_t nkv, double scale,
               int64_t C) {
  TORCH_CHECK(S % 128 == 0);
  static bool cfg = false;
  if (!cfg) {
    set_smem((const void*)attn_dkv2_kernel, sizeof(Dkv2Smem));
    cfg = true;
  }
  const int tiles = (int)(nkv * (S / 128) * (nq / nkv) * C);
  attn_dkv2_kernel<<<tiles, 256, sizeof(Dkv2Smem), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), reinterpret_cast<const bf16*>(dO.data_ptr()),
      LSE.data_ptr<float>(), Drow.data_ptr<float>(), ws.data_ptr<float>(), (int)S, (int)nq,
      (int)nkv, (float)scale, (int)C);
  C10_CUDA_CHECK(cudaGetLastError());
}

void attn_dq2(torch::Tensor qkv, torch::Tensor dO, torch::Tensor LSE, torch::Tensor Drow,
              torch::Tensor ws, int64_t S, int64_t nq, int64_t nkv, double scale,
              int64_t C) {
  TORCH_CHECK(S % 128 == 0);
  static bool cfg = false;
  if (!cfg) {
    set_smem((const void*)attn_dq2_kernel, sizeof(Dq2Smem));
    cfg = true;
  }
  const int tiles = (int)(nq * (S / 128) * C);
  attn_dq2_kernel<<<tiles, 256, sizeof(Dq2Smem), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), reinterpret_cast<const bf16*>(dO.data_ptr()),
      LSE.data_ptr<float>(), Drow.data_ptr<float>(), ws.data_ptr<float>(), (int)S, (int)nq,
      (int)nkv, (float)scale, (int)C);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <bool DQ_RS, int DRAIN>
static void launch_onepass(torch::Tensor& qkv, torch::Tensor& dO, torch::Tensor& LSE,
                           torch::Tensor& Drow, torch::Tensor& ws, torch::Tensor& dqa,
                           int64_t S, int64_t nq, int64_t nkv, double scale, int64_t C) {
  static bool cfg = false;
  if (!cfg) {
    set_smem((const void*)attn_onepass_kernel<DQ_RS, DRAIN>, sizeof(OnepassSmem));
    cfg = true;
  }
  const int tiles = (int)(nkv * (S / 128) * (nq / nkv) * C);
  attn_onepass_kernel<DQ_RS, DRAIN>
      <<<tiles, 256, sizeof(OnepassSmem), at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const bf16*>(qkv.data_ptr()),
          reinterpret_cast<const bf16*>(dO.data_ptr()), LSE.data_ptr<float>(),
          Drow.data_ptr<float>(), ws.data_ptr<float>(), dqa.data_ptr<float>(), (int)S,
          (int)nq, (int)nkv, (float)scale, (int)C);
  C10_CUDA_CHECK(cudaGetLastError());
}

void attn_onepass(torch::Tensor qkv, torch::Tensor dO, torch::Tensor LSE,
                  torch::Tensor Drow, torch::Tensor ws, torch::Tensor dqa, int64_t S,
                  int64_t nq, int64_t nkv, double scale, int64_t C, int64_t dq_rs,
                  int64_t drain) {
  TORCH_CHECK(S % 128 == 0);
  if (dq_rs && drain == 0)
    launch_onepass<true, 0>(qkv, dO, LSE, Drow, ws, dqa, S, nq, nkv, scale, C);
  else if (dq_rs && drain == 1)
    launch_onepass<true, 1>(qkv, dO, LSE, Drow, ws, dqa, S, nq, nkv, scale, C);
  else if (!dq_rs && drain == 0)
    launch_onepass<false, 0>(qkv, dO, LSE, Drow, ws, dqa, S, nq, nkv, scale, C);
  else
    launch_onepass<false, 1>(qkv, dO, LSE, Drow, ws, dqa, S, nq, nkv, scale, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_dkv2", &attn_dkv2, "two-pass dKV baseline (current-op feature level)");
  m.def("attn_dq2", &attn_dq2, "two-pass dQ baseline (current-op feature level)");
  m.def("attn_onepass", &attn_onepass, "one-pass D64 bwd (dq_rs, drain variants)");
}
