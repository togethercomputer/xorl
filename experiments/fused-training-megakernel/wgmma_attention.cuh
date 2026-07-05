// wgmma (Hopper) causal GQA flash attention ops — fwd + FA2 two-pass bwd.
//
// Ported from the Phase-5 probe (attention_probe.cu; gates: fwd 3.3x/5.6x, dKV
// 2.1x/3.3x, dQ 2.2x/4.9x vs the WMMA ops at nano/small — results/mkv3-p5-attnprobe.md).
// Requirements (host routes to the WMMA ops otherwise): D == 64, S % 128 == 0.
// Contracts are drop-in: O bf16 [S,nq*D]; LSE fp32 [nq,S]; Drow fp32 [nq,S] input;
// dqkv_f32 pre-zeroed fp32 [S,stride] accumulated with atomicAdd.
//
// THE LAYOUT (validated to 3.8e-6 in all four major combos): ONE 64x64 bf16 no-swizzle
// smem arrangement serves every operand role.
//   off64(r,c) = ((r>>3)<<10) + ((c>>3)<<7) + ((r&7)<<4) + ((c&7)<<1)      (8KB tile)
// The same bytes admit two descriptor views:
//   K-view : operand [mn=r, k=c] -> LBO=128B, SBO=1024B, wgmma ktile step +256B
//   MN-view: operand [k=r, mn=c] -> LBO=1024B, SBO=128B, wgmma ktile step +2048B
// so a row-major-loaded X[rows,cols] tile is simultaneously "X" (K-view A), "X^T as B"
// (K-view B), and "X as B / X^T as A" (MN-views): descriptor-major transposes, zero
// data movement. Backward reuses each streamed Q/dO/K stage in two gemm roles this way.
//
// Block tile = 128 rows split across the two consumer warpgroups (64 rows each); the
// counterpart operand streams in shared 64-row 2-stage cp.async ping-pong. Online
// softmax / P / dS run on the wgmma m64n64 f32 accumulator layout in REGISTERS:
// thread t of warp w holds rows w*16 + t/4 + {0,8}, cols n8*8 + (t%4)*2 + {0,1};
// row max/sum = 2 shfl_xor steps (masks 1,2) within the quad.
//
// Op-code invariants: consumer_sync()/MK_CONSUMERS only (never __syncthreads/
// blockDim.x) so the 384-thread ws executor stays correct. Register footprints
// (probe build): fwd 109 / dq 140 / dkv 168 — all under the 224 ws consumer budget.

#pragma once

// generic-proxy smem stores (the register softmax writing P/dS) must be fenced into
// the async proxy before wgmma reads them; cp.async-written tiles don't need this.
__device__ __forceinline__ void wga_fence_smem_to_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ int wga_off64(int r, int c) {
  return ((r >> 3) << 10) + ((c >> 3) << 7) + ((r & 7) << 4) + ((c & 7) << 1);
}

__device__ __forceinline__ float wga_lse_log(float x) {
#ifdef MK_ATTN_FAST_LOG
  return __logf(x);
#else
  return logf(x);
#endif
}

__device__ __forceinline__ float wga_exp(float x) {
#ifdef MK_ATTN_EXP2_APPROX
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
#else
  return __expf(x);
#endif
}

// Loader lane mapping (v3 P4b): off64's bank quad depends only on r&7, so the old
// v -> (r = v/8, c8 = v%8*8) assignment put each 8-lane store phase on ONE row =
// one bank quad = 8-way conflict (the same pathology SW128 fixed in the gemm; here
// the dual-view trick pins the byte layout, so fix the ASSIGNMENT instead):
// v -> (r = v%8 | (v/64)*8, c8 = (v/8)%8*8) spans 8 rows per phase — all 32 banks —
// while each row still reads 64B contiguous gmem per warp (same 16 L2 sectors).

__device__ __forceinline__ uint64_t wga_desc_k(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (128 >> 4);   // c8 (k) group stride
  d.bitfield.stride_byte_offset_ = (1024 >> 4);   // r8 (mn) group stride
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

__device__ __forceinline__ uint64_t wga_desc_mn(const void* p) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (1024 >> 4);  // r8 (k) group stride
  d.bitfield.stride_byte_offset_ = (128 >> 4);    // c8 (mn) group stride
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

#define WGA_FMA32 d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10],     \
    d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22],  \
    d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]

using WGA_MMA_KK =
    cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::K,
                                                  cute::SM90::GMMA::Major::K>;
using WGA_MMA_KMN =
    cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::K,
                                                  cute::SM90::GMMA::Major::MN>;
using WGA_MMA_MNMN =
    cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN,
                                                  cute::SM90::GMMA::Major::MN>;

// issue-only variant (v3 P4b r3 pipelining): arrive+fma+commit, NO wait — the caller
// overlaps register/softmax work with the flying batch and waits explicitly.
// Retirement is IN ORDER: a wait that retires batch k retires everything older.
template <class MMA, bool A_MN, bool B_MN>
__device__ __forceinline__ void wga_mma64_issue(const bf16* A, const bf16* B,
                                                float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint64_t da =
        A_MN ? wga_desc_mn((const char*)A + s * 2048) : wga_desc_k((const char*)A + s * 256);
    const uint64_t db =
        B_MN ? wga_desc_mn((const char*)B + s * 2048) : wga_desc_k((const char*)B + s * 256);
    MMA::fma(da, db, WGA_FMA32, cute::SM90::GMMA::ScaleOut::One);
  }
  cute::warpgroup_commit_batch();
}

// d += A(view) @ B(view) over the 64-deep K dim (4 ktiles). Branch-free ScaleOut::One
// over pre-zeroed d (a data-dependent ScaleOut serializes wgmma ~60x).
template <class MMA, bool A_MN, bool B_MN>
__device__ __forceinline__ void wga_mma64(const bf16* A, const bf16* B, float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint64_t da =
        A_MN ? wga_desc_mn((const char*)A + s * 2048) : wga_desc_k((const char*)A + s * 256);
    const uint64_t db =
        B_MN ? wga_desc_mn((const char*)B + s * 2048) : wga_desc_k((const char*)B + s * 256);
    MMA::fma(da, db, WGA_FMA32, cute::SM90::GMMA::ScaleOut::One);
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// two accumulations in one commit batch (bwd pairs: S+dP, dV+dK)
template <class MMA, bool A_MN, bool B_MN>
__device__ __forceinline__ void wga_mma64_x2(const bf16* A1, const bf16* B1,
                                             float (&d1)[32], const bf16* A2,
                                             const bf16* B2, float (&d2)[32]) {
  cute::warpgroup_arrive();
  {
    float(&d)[32] = d1;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint64_t da = A_MN ? wga_desc_mn((const char*)A1 + s * 2048)
                               : wga_desc_k((const char*)A1 + s * 256);
      const uint64_t db = B_MN ? wga_desc_mn((const char*)B1 + s * 2048)
                               : wga_desc_k((const char*)B1 + s * 256);
      MMA::fma(da, db, WGA_FMA32, cute::SM90::GMMA::ScaleOut::One);
    }
  }
  {
    float(&d)[32] = d2;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint64_t da = A_MN ? wga_desc_mn((const char*)A2 + s * 2048)
                               : wga_desc_k((const char*)A2 + s * 256);
      const uint64_t db = B_MN ? wga_desc_mn((const char*)B2 + s * 2048)
                               : wga_desc_k((const char*)B2 + s * 256);
      MMA::fma(da, db, WGA_FMA32, cute::SM90::GMMA::ScaleOut::One);
    }
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// ---- forward ---------------------------------------------------------------------------
// args: {qkv_r, O, LSE, S, nq, nkv, D(=64), scale_bits}; tile = qt*nq + qh (qt-outer,
// 128-row tiles -> O/LSE complete in row order; band = nq tiles per 128 rows).

struct __align__(16) AttnWgFwdSmem {  // 64KB (80KB under MK_ATTN_PIPE: P ping-pong)
  bf16 Q[2][4096];  // [wg]    q rows, K-view A of S = Q K^T
#ifdef MK_ATTN_PIPE
  bf16 P[2][2][4096];  // [wg][stage&1]  P ping-pong: O-mma(t) reads P[t&1] while
                       // softmax(t+1) writes P[(t+1)&1] — both fly with the pipe
  bf16 K[4][4096];     // [stage%4] 4-ring: loads(t+2) issue while O(t-1) still
  bf16 V[4][4096];     //          reads (t-1)%4 — a 2-ring would alias (112KB total)
#else
  bf16 P[2][4096];  // [wg]    P [q,kv], K-view A of O += P V
  bf16 K[2][4096];  // [stage] kv rows, K-view B (= K^T)
  bf16 V[2][4096];  // [stage] kv rows, MN-view B (= V)
#endif
};

#ifdef MK_ATTN_PIPE
// FA2-shape software-pipelined forward (v3 P4b r3). Batch ledger (in-order wgmma
// retirement makes wait<1> exact): per stage t, commit order is S(t) then O(t-1);
//   step4 wait<1> retires O(t-2) [S(t) flies], step5 wait<1> retires S(t)
//   [O(t-1) flies] -> softmax(t) OVERLAPS the O(t-1) mma, the FA2 win.
// s-accumulator ping-pong is bound by reference at two call sites (no dynamic
// indexing -> no local-memory spill). Register cost ~+40 (fwd ~150, fits 224/255).
__device__ void op_attn_fwd_wg_pipe(const Instr& I, int tile, void** bufs,
                                    char* smem_raw) {
  constexpr int D = 64;
  const int S = I.args[3], nq = I.args[4], nkv = I.args[5];
  const float scale = __int_as_float(I.args[7]);
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  bf16* O = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  float* LSE = reinterpret_cast<float*>(bufs[I.args[2]]);
  AttnWgFwdSmem& sm = *reinterpret_cast<AttnWgFwdSmem*>(smem_raw);

  const int qh = tile % nq;
  const int q0 = (tile / nq) * 128;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();
  const int wg = tid >> 7, wtid = tid & 127;
  const int q0wg = q0 + wg * 64;

  auto issue_kv_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.K[st] + wga_off64(r, c8),
                              &qkv[(int64_t)(k0 + r) * stride + (nq + kvh) * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async(
          (char*)sm.V[st] + wga_off64(r, c8),
          &qkv[(int64_t)(k0 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
    }
    __pipeline_commit();
  };

#pragma unroll
  for (int i = 0; i < 4; ++i) {  // Q joins stage 0's group
    const int v = tid + i * MK_CONSUMERS;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async((char*)sm.Q[h] + wga_off64(r, c8),
                            &qkv[(int64_t)(q0 + h * 64 + r) * stride + qh * D + c8], 16);
  }
  issue_kv_stage(0, 0);
  const int n_stages = q0 / 64 + 2;
  if (n_stages > 1) issue_kv_stage(64, 1);

  float o[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  float alpha_prev[2] = {1.0f, 1.0f};
  const int w = wtid >> 5, ln = wtid & 31;
  const int r0 = w * 16 + (ln >> 2);
  const int cb = (ln & 3) * 2;
  float sA[32], sB[32];

  // softmax of stage t from s into P[wg][t&1]; updates m/l, returns alpha in a_out
  auto softmax_stage = [&](float (&sv)[32], int t, float (&a_out)[2]) {
    const int k0 = t * 64;
    const bool masked = k0 + 63 > q0wg;
    float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int qr = q0wg + r0 + 8 * i;
#pragma unroll
      for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int idx = n8 * 4 + i * 2 + j;
          float sc = sv[idx] * scale;
          if (masked && k0 + n8 * 8 + cb + j > qr) sc = -INFINITY;
          sv[idx] = sc;
          rmax[i] = fmaxf(rmax[i], sc);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffffu, rmax[i], 1));
      rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffffu, rmax[i], 2));
      const float mnew = fmaxf(m[i], rmax[i]);
      a_out[i] = wga_exp(m[i] - mnew);
      m[i] = mnew;
    }
    float rsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int n8 = 0; n8 < 8; ++n8) {
        const int idx = n8 * 4 + i * 2;
        const float p0 = wga_exp(sv[idx] - m[i]);
        const float p1 = wga_exp(sv[idx + 1] - m[i]);
        rsum[i] += p0 + p1;
        __nv_bfloat162 pv;
        pv.x = f2bf(p0);
        pv.y = f2bf(p1);
        *reinterpret_cast<__nv_bfloat162*>((char*)sm.P[wg][t & 1] +
                                           wga_off64(r0 + 8 * i, n8 * 8 + cb)) = pv;
      }
      rsum[i] += __shfl_xor_sync(0xffffffffu, rsum[i], 1);
      rsum[i] += __shfl_xor_sync(0xffffffffu, rsum[i], 2);
      l[i] = l[i] * a_out[i] + rsum[i];
    }
  };

  // one pipeline iteration; sCur/sNxt bound by reference per parity (no spill)
  auto stage_iter = [&](int t, float (&sCur)[32], float (&sNxt)[32]) {
    (void)sNxt;
    const bool liveS = t < n_stages && !(t * 64 > q0wg + 63);
    const bool liveO = t > 0 && !((t - 1) * 64 > q0wg + 63);
    if (t < n_stages) {
      __pipeline_wait_prior(t + 1 < n_stages ? 1 : 0);  // loads(t) landed
    }
    consumer_sync();  // stage t visible + P[(t-1)&1] visible for the O-mma
    if (liveS) {
#pragma unroll
      for (int i = 0; i < 32; ++i) sCur[i] = 0.0f;
      wga_mma64_issue<WGA_MMA_KK, false, false>(sm.Q[wg], sm.K[t & 3], sCur);
    }
    if (liveS)
      cute::warpgroup_wait<1>();  // O(t-2) retired (S(t) flies)
    else
      cute::warpgroup_wait<0>();  // no S(t) committed: drain O(t-2) directly
    if (liveO) {
#pragma unroll
      for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j) o[n8 * 4 + i * 2 + j] *= alpha_prev[i];
      wga_mma64_issue<WGA_MMA_KMN, false, true>(sm.P[wg][(t - 1) & 1], sm.V[(t - 1) & 3], o);
    }
    if (liveO)
      cute::warpgroup_wait<1>();  // S(t) retired (O(t-1) flies)
    else
      cute::warpgroup_wait<0>();  // no O(t-1) committed: drain S(t) directly
    if (liveS) {
      float a[2];
      softmax_stage(sCur, t, a);
      alpha_prev[0] = a[0];
      alpha_prev[1] = a[1];
    } else if (t < n_stages) {
      alpha_prev[0] = 1.0f;
      alpha_prev[1] = 1.0f;
    }
    wga_fence_smem_to_async();
    if (t + 2 < n_stages) issue_kv_stage((t + 2) * 64, (t + 2) & 3);
  };

  for (int t = 0; t <= n_stages; t += 2) {  // t == n_stages issues the final O
    stage_iter(t, sA, sB);
    if (t + 1 <= n_stages) stage_iter(t + 1, sB, sA);
  }
  cute::warpgroup_wait<0>();  // drain the last O

  const float inv[2] = {1.0f / l[0], 1.0f / l[1]};
  if ((ln & 3) == 0) {
    LSE[(int64_t)qh * S + q0wg + r0] = m[0] + wga_lse_log(l[0]);
    LSE[(int64_t)qh * S + q0wg + r0 + 8] = m[1] + wga_lse_log(l[1]);
  }
  float* Cs = reinterpret_cast<float*>(smem_raw);
  consumer_sync();  // all wgmma reads of Q/P/K/V done before the overlay
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
    const int gid = tid + g * MK_CONSUMERS;
    const int r = gid >> 3, c8 = (gid & 7) << 3;
    uint4 out;
    bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = f2bf(Cs[r * 68 + c8 + e]);
    *reinterpret_cast<uint4*>(&O[(int64_t)(q0 + r) * (nq * D) + qh * D + c8]) = out;
  }
}
#endif  // MK_ATTN_PIPE

__device__ void op_attn_fwd_wg(const Instr& I, int tile, void** bufs, char* smem_raw) {
#ifdef MK_ATTN_PIPE
  op_attn_fwd_wg_pipe(I, tile, bufs, smem_raw);
#else

  constexpr int D = 64;  // host routes D!=64 to op_attn_fwd
  const int S = I.args[3], nq = I.args[4], nkv = I.args[5];
  const float scale = __int_as_float(I.args[7]);
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  bf16* O = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  float* LSE = reinterpret_cast<float*>(bufs[I.args[2]]);
  AttnWgFwdSmem& sm = *reinterpret_cast<AttnWgFwdSmem*>(smem_raw);

  const int qh = tile % nq;
  const int q0 = (tile / nq) * 128;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();
  const int wg = tid >> 7, wtid = tid & 127;
  const int q0wg = q0 + wg * 64;

  auto issue_kv_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // K: 512 16B vectors over the 256 consumers
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.K[st] + wga_off64(r, c8),
                              &qkv[(int64_t)(k0 + r) * stride + (nq + kvh) * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // V
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async(
          (char*)sm.V[st] + wga_off64(r, c8),
          &qkv[(int64_t)(k0 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
    }
    __pipeline_commit();
  };

  // Q (both WG tiles, 1024 vectors) joins stage 0's commit group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * MK_CONSUMERS;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async((char*)sm.Q[h] + wga_off64(r, c8),
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
      wga_mma64<WGA_MMA_KK, false, false>(sm.Q[wg], sm.K[t & 1], s);  // S = Q K^T
      const bool masked = k0 + 63 > q0wg;                             // diagonal stage
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
        alpha[i] = wga_exp(m[i] - mnew);  // m=-inf only at stage 0, where mnew is finite
        m[i] = mnew;
      }
      float rsum[2] = {0.0f, 0.0f};
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const float p0 = wga_exp(s[idx] - m[i]);  // masked: exp(-inf)=0
          const float p1 = wga_exp(s[idx + 1] - m[i]);
          rsum[i] += p0 + p1;
          __nv_bfloat162 pv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.P[wg] +
                                             wga_off64(r0 + 8 * i, n8 * 8 + cb)) = pv;
        }
        rsum[i] += __shfl_xor_sync(0xffffffffu, rsum[i], 1);
        rsum[i] += __shfl_xor_sync(0xffffffffu, rsum[i], 2);
        l[i] = l[i] * alpha[i] + rsum[i];
      }
    }
    wga_fence_smem_to_async();
    consumer_sync();  // P visible to this WG's wgmma; V stage ready
    if (!skip) {
#pragma unroll
      for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j) o[n8 * 4 + i * 2 + j] *= alpha[i];
      wga_mma64<WGA_MMA_KMN, false, true>(sm.P[wg], sm.V[t & 1], o);  // O += P V
    }
    consumer_sync();  // both WGs done reading K/V stage t before refill
  }

  // epilogue: LSE + O from registers; O staged through smem for coalesced stores
  const float inv[2] = {1.0f / l[0], 1.0f / l[1]};
  if ((ln & 3) == 0) {
    LSE[(int64_t)qh * S + q0wg + r0] = m[0] + wga_lse_log(l[0]);
    LSE[(int64_t)qh * S + q0wg + r0 + 8] = m[1] + wga_lse_log(l[1]);
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
    const int gid = tid + g * MK_CONSUMERS;
    const int r = gid >> 3, c8 = (gid & 7) << 3;
    uint4 out;
    bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = f2bf(Cs[r * 68 + c8 + e]);
    *reinterpret_cast<uint4*>(&O[(int64_t)(q0 + r) * (nq * D) + qh * D + c8]) = out;
  }
#endif  // MK_ATTN_PIPE
}

// ---- backward dK/dV pass -----------------------------------------------------------------
// args: {qkv_r, dO, LSE, Drow, dqkv_f32, S, nq, nkv, D(=64), scale_bits, C}.
// tile = ((kvh*n_kvt + kvt)*G + g)*C + c: block owns 128 kv rows (64/WG) for one GQA
// member qh = kvh*G + g; streams Q/dO stages q0s = kv0 + (c + t*C)*64 (q-chunk c of C
// trades C-fold owned-tile reloads for C-fold less serial latency; the fp32 atomic
// epilogue makes chunks race-free). dK/dV accumulate in registers.

struct __align__(16) AttnWgDkvSmem {  // 96KB
  bf16 K[2][4096];   // [wg] owned kv rows, K-view B (= K^T)
  bf16 V[2][4096];   // [wg] owned kv rows, K-view B (= V^T)
  bf16 P[2][4096];   // [wg] [q,kv]; MN-view A = P^T
  bf16 dS[2][4096];  // [wg] [q,kv]; MN-view A = dS^T
  bf16 Q[2][4096];   // [stage] K-view A (S = Q K^T) + MN-view B (dK += dS^T Q)
  bf16 dO[2][4096];  // [stage] K-view A (dP = dO V^T) + MN-view B (dV += P^T dO)
};

__device__ void op_attn_dkv_wg(const Instr& I, int tile, void** bufs, char* smem_raw) {
  constexpr int D = 64;
  const int S = I.args[5], nq = I.args[6], nkv = I.args[7];
  const float scale = __int_as_float(I.args[9]);
  const int C = I.args[10];
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* dOg = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const float* LSE = reinterpret_cast<const float*>(bufs[I.args[2]]);
  const float* Drow = reinterpret_cast<const float*>(bufs[I.args[3]]);
  float* ws = reinterpret_cast<float*>(bufs[I.args[4]]);
  AttnWgDkvSmem& sm = *reinterpret_cast<AttnWgDkvSmem*>(smem_raw);

  const int G = nq / nkv;
  const int n_kvt = S / 128;
  const int c = tile % C;
  const int t128 = tile / C;
  const int kvh = t128 / (n_kvt * G);
  const int rem = t128 % (n_kvt * G);
  const int kv0 = (rem / G) * 128;
  const int g = rem % G;
  const int qh = kvh * G + g;
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();
  const int wg = tid >> 7, wtid = tid & 127;
  const int kv0wg = kv0 + wg * 64;
  const int n_stages = ((S - kv0) / 64 - c + C - 1) / C;
  if (n_stages <= 0) return;  // uniform: before any cp.async/barrier

  auto issue_qdo_stage = [&](int q0s, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.Q[st] + wga_off64(r, c8),
                              &qkv[(int64_t)(q0s + r) * stride + qh * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.dO[st] + wga_off64(r, c8),
                              &dOg[(int64_t)(q0s + r) * (nq * D) + qh * D + c8], 16);
    }
    __pipeline_commit();
  };

  // owned K/V tiles (both WGs) join stage 0's commit group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * MK_CONSUMERS;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async(
        (char*)sm.K[h] + wga_off64(r, c8),
        &qkv[(int64_t)(kv0 + h * 64 + r) * stride + (nq + kvh) * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * MK_CONSUMERS;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async(
        (char*)sm.V[h] + wga_off64(r, c8),
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
      // Register diet (the interpreter shares one 255-reg allocation across ALL ops;
      // spills tax every op uniformly — the v3 P1 lesson): never hold the P(fp32) and
      // dP(fp32) banks at once. P goes to smem as bf16 immediately and is read BACK
      // (same thread, same addresses) for dS — the dV gemm consumes bf16 P anyway, so
      // this costs no precision. Live fp32 banks: dk, dv, s (96) instead of 128.
      float s[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = 0.0f;
      // rows of s = q stage rows; cols = this WG's kv rows
      wga_mma64<WGA_MMA_KK, false, false>(sm.Q[t & 1], sm.K[wg], s);  // S = Q K^T
      const bool masked = q0s == kv0wg;  // diagonal stage
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qr = q0s + r0 + 8 * i;
        const float lse = LSE[(int64_t)qh * S + qr];
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = kv0wg + n8 * 8 + cb;
          float p0 = wga_exp(s[idx] * scale - lse);
          float p1 = wga_exp(s[idx + 1] * scale - lse);
          if (masked && kr > qr) p0 = 0.0f;
          if (masked && kr + 1 > qr) p1 = 0.0f;
          __nv_bfloat162 pv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          *reinterpret_cast<__nv_bfloat162*>(
              (char*)sm.P[wg] + wga_off64(r0 + 8 * i, n8 * 8 + cb)) = pv;
        }
      }
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = 0.0f;
      wga_mma64<WGA_MMA_KK, false, false>(sm.dO[t & 1], sm.V[wg], s);  // dP = dO V^T
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float dr = Drow[(int64_t)qh * S + q0s + r0 + 8 * i];
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int off = wga_off64(r0 + 8 * i, n8 * 8 + cb);
          const __nv_bfloat162 pv =
              *reinterpret_cast<const __nv_bfloat162*>((char*)sm.P[wg] + off);
          __nv_bfloat162 dsv;
          dsv.x = f2bf(bf2f(pv.x) * (s[idx] - dr) * scale);
          dsv.y = f2bf(bf2f(pv.y) * (s[idx + 1] - dr) * scale);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.dS[wg] + off) = dsv;
        }
      }
    }
    wga_fence_smem_to_async();
    consumer_sync();  // P/dS visible; Q/dO stage still valid
    if (!skip)
      wga_mma64_x2<WGA_MMA_MNMN, true, true>(sm.P[wg], sm.dO[t & 1], dv,   // dV += P^T dO
                                             sm.dS[wg], sm.Q[t & 1], dk);  // dK += dS^T Q
    consumer_sync();  // both WGs done reading Q/dO stage t before refill
  }

  // epilogue: stage each 128x64 accumulator to smem, coalesced fp32 atomics
#ifdef MK_ATTN_DKV_DIRECT_ATOMIC
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
          atomicAdd(&ws[(int64_t)(kv0wg + r0 + 8 * i) * stride + col0 + n8 * 8 + cb + j],
                    acc[n8 * 4 + i * 2 + j]);
  }
#else
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
      const int gid = tid + gq * MK_CONSUMERS;
      const int r = gid >> 3, c8 = (gid & 7) << 3;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        atomicAdd(&ws[(int64_t)(kv0 + r) * stride + col0 + c8 + e], Cs[r * 68 + c8 + e]);
    }
    consumer_sync();  // Cs reuse between rounds
  }
#endif
}

// ---- backward dQ pass ----------------------------------------------------------------------
// args: {qkv_r, dO, LSE, Drow, dqkv_f32, S, nq, nkv, D(=64), scale_bits, C}.
// tile = (qt*nq + qh)*C + c: block owns 128 q rows (64/WG); streams K/V stages
// k0 = (c + t*C)*64. LSE/Drow are register-resident (each thread's 2 rows are fixed).
// dQ += dS @ K reuses the K stage via its MN-view (dual view).

struct __align__(16) AttnWgDqSmem {  // 80KB
  bf16 Q[2][4096];   // [wg] K-view A (S = Q K^T)
  bf16 dO[2][4096];  // [wg] K-view A (dP = dO V^T)
  bf16 dS[2][4096];  // [wg] [q,kv]; K-view A (dQ += dS K)
  bf16 K[2][4096];   // [stage] K-view B (= K^T) + MN-view B (= K)
  bf16 V[2][4096];   // [stage] K-view B (= V^T)
};

__device__ void op_attn_dq_wg(const Instr& I, int tile, void** bufs, char* smem_raw) {
  constexpr int D = 64;
  const int S = I.args[5], nq = I.args[6], nkv = I.args[7];
  const float scale = __int_as_float(I.args[9]);
  const int C = I.args[10];
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* dOg = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const float* LSE = reinterpret_cast<const float*>(bufs[I.args[2]]);
  const float* Drow = reinterpret_cast<const float*>(bufs[I.args[3]]);
  float* ws = reinterpret_cast<float*>(bufs[I.args[4]]);
  AttnWgDqSmem& sm = *reinterpret_cast<AttnWgDqSmem*>(smem_raw);

  const int c = tile % C;
  const int t128 = tile / C;
  const int qh = t128 % nq;
  const int q0 = (t128 / nq) * 128;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();
  const int wg = tid >> 7, wtid = tid & 127;
  const int q0wg = q0 + wg * 64;
  const int n_stages = (q0 / 64 + 2 - c + C - 1) / C;
  if (n_stages <= 0) return;  // uniform: before any cp.async/barrier

  auto issue_kv_stage = [&](int k0, int st) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async((char*)sm.K[st] + wga_off64(r, c8),
                              &qkv[(int64_t)(k0 + r) * stride + (nq + kvh) * D + c8], 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * MK_CONSUMERS;
      const int r = ((v >> 6) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
      __pipeline_memcpy_async(
          (char*)sm.V[st] + wga_off64(r, c8),
          &qkv[(int64_t)(k0 + r) * stride + (nq + nkv + kvh) * D + c8], 16);
    }
    __pipeline_commit();
  };

  // owned Q/dO tiles (both WGs) join stage 0's commit group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * MK_CONSUMERS;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async((char*)sm.Q[h] + wga_off64(r, c8),
                            &qkv[(int64_t)(q0 + h * 64 + r) * stride + qh * D + c8], 16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = tid + i * MK_CONSUMERS;
    const int h = v >> 9;
    const int r = (((v >> 6) & 7) << 3) | (v & 7), c8 = ((v >> 3) & 7) << 3;
    __pipeline_memcpy_async((char*)sm.dO[h] + wga_off64(r, c8),
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
      // Register diet as in dkv: P parks in the dS tile (bf16) across the dP gemm and
      // is read back (same thread, same addresses), then overwritten with dS. Live
      // fp32 banks: dq, s (64) instead of 96.
      float s[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = 0.0f;
      wga_mma64<WGA_MMA_KK, false, false>(sm.Q[wg], sm.K[t & 1], s);  // S = Q K^T
      const bool masked = k0 + 63 > q0wg;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int kr = k0 + n8 * 8 + cb;
          float p0 = wga_exp(s[idx] * scale - lse[i]);
          float p1 = wga_exp(s[idx + 1] * scale - lse[i]);
          if (masked && kr > qr[i]) p0 = 0.0f;
          if (masked && kr + 1 > qr[i]) p1 = 0.0f;
          __nv_bfloat162 pv;
          pv.x = f2bf(p0);
          pv.y = f2bf(p1);
          *reinterpret_cast<__nv_bfloat162*>(
              (char*)sm.dS[wg] + wga_off64(r0 + 8 * i, n8 * 8 + cb)) = pv;
        }
      }
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = 0.0f;
      wga_mma64<WGA_MMA_KK, false, false>(sm.dO[wg], sm.V[t & 1], s);  // dP = dO V^T
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int idx = n8 * 4 + i * 2;
          const int off = wga_off64(r0 + 8 * i, n8 * 8 + cb);
          const __nv_bfloat162 pv =
              *reinterpret_cast<const __nv_bfloat162*>((char*)sm.dS[wg] + off);
          __nv_bfloat162 dsv;
          dsv.x = f2bf(bf2f(pv.x) * (s[idx] - dr[i]) * scale);
          dsv.y = f2bf(bf2f(pv.y) * (s[idx + 1] - dr[i]) * scale);
          *reinterpret_cast<__nv_bfloat162*>((char*)sm.dS[wg] + off) = dsv;
        }
      }
    }
    wga_fence_smem_to_async();
    consumer_sync();  // dS visible; K stage still valid
    if (!skip)
      wga_mma64<WGA_MMA_KMN, false, true>(sm.dS[wg], sm.K[t & 1], dq);  // dQ += dS K
    consumer_sync();  // both WGs done reading K/V stage t before refill
  }

  // epilogue: C=1 has one writer per q slice, so write the accumulator layout directly
  // to the fp32 workspace and skip the smem stage/drain used by chunked atomics.
  if (C == 1) {
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          ws[(int64_t)qr[i] * stride + qh * D + n8 * 8 + cb + j] =
              dq[n8 * 4 + i * 2 + j];
    return;
  }

  // Chunked C>1 routes have multiple writers per q slice, so stage dQ then drain with
  // coalesced fp32 atomics.
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
    const int gid = tid + gq * MK_CONSUMERS;
    const int r = gid >> 3, c8 = (gid & 7) << 3;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      atomicAdd(&ws[(int64_t)(q0 + r) * stride + qh * D + c8 + e], Cs[r * 68 + c8 + e]);
  }
}
