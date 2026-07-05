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
// args: {A, B, C, M, N, K, flags, res, sk}
// tile id = (m_tile * n_tiles + n_tile) * sk + k_slice.

#define GEMM_BM 64
#define GEMM_BN 128
#define GEMM_BK 32
#define GEMM_LDA (GEMM_BK + 8)  // bf16 smem strides (pad: bank conflicts + wmma align)
#define GEMM_LDB (GEMM_BN + 8)
#define GEMM_LDC (GEMM_BN + 4)  // fp32 staging (wmma: fp32 ld must be a multiple of 4)

struct GemmSmem {
  bf16 As[GEMM_BM][GEMM_LDA];
  bf16 Bs[GEMM_BK][GEMM_LDB];
  float Cs[GEMM_BM][GEMM_LDC];
};

__device__ __forceinline__ uint4 ldg16(const bf16* p) {
  return *reinterpret_cast<const uint4*>(p);
}

__device__ void op_gemm(const Instr& I, int tile, void** bufs, char* smem_raw) {
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
  consumer_sync();

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

struct WgmmaSmem {
  bf16 A[2][2][4][1024];  // [stage][row-half][k16-step][64x16 INTER block] = 32KB
  bf16 B[2][4][1024];     // [stage][k16-step][64x16 INTER block]           = 16KB
};
struct WgmmaSmemN128 {
  bf16 A[2][2][4][1024];  // [stage][row-half][k16-step][64x64 SW128 slab] = 32KB
  bf16 B[2][8192];        // [stage][128 rows x 64 k elts SW128]           = 32KB
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

// m64n128 NT tile (v3 P4b r3, generalized from the peer session's lm_head route):
// 64 fp32 accumulators/thread double the mma work per sync and halve B-traffic per
// FLOP — the dependent chain per FLOP shortens (the one lever the register-lifetime
// law allows). REG ~200 fits the 255 df budget; __noinline__ isolates the fat
// accumulator frame from the dispatch switch. Supports NT + residual (bit16) + CE
// partials (bit11); routing (flags bit12) excludes split-K/acc/f32/qkrope/Drow.
__device__ __noinline__ void op_gemm_wgmma_n128(const Instr& I, int tile, void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  bf16* C = reinterpret_cast<bf16*>(bufs[I.args[2]]);
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  const bf16* Res = (flags & 16) ? reinterpret_cast<const bf16*>(bufs[I.args[7]]) : nullptr;

  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  WgmmaSmemN128& S = *reinterpret_cast<WgmmaSmemN128*>(smem_raw);
  const int n_tiles = N / 128;
  const int m0 = (tile / n_tiles) * WG_BM;
  const int n0 = (tile % n_tiles) * 128;
  const int tid = mk_tid();
  const int wg = tid / 128;
  const int wtid = tid % 128;

  const bool b_t = flags & 2;  // NT: B[N,K] K-contig; NN: B[K,N] N-contig (MN slabs)
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
  const int iters = K / WG_BK;
  issue_stage(0, 0);
  for (int t = 0; t < iters; ++t) {
    if (t + 1 < iters) issue_stage((t + 1) * WG_BK, (t + 1) & 1);
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
    uint4 out;
    bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = f2bf(v[e]);
    *reinterpret_cast<uint4*>(&C[idx]) = out;
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
            se = se * __expf(mx - zv) + 1.0f;
            mx = zv;
          } else {
            se += __expf(zv - mx);
          }
        }
        for (int o = 16; o > 0; o >>= 1) {
          const float om = __shfl_xor_sync(0xffffffff, mx, o);
          const float os = __shfl_xor_sync(0xffffffff, se, o);
          const float Mx = fmaxf(mx, om);
          se = (mx == -INFINITY && om == -INFINITY) ? 0.0f : se * __expf(mx - Mx) + os * __expf(om - Mx);
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
}

__device__ void op_gemm_wgmma(const Instr& I, int tile, void** bufs, char* smem_raw) {
  namespace SG = cute::SM90::GMMA;
  const bf16* A = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* B = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  void* Cp = bufs[I.args[2]];
  const int M = I.args[3], N = I.args[4], K = I.args[5], flags = I.args[6];
  if (flags & 4096) {
    op_gemm_wgmma_n128(I, tile, bufs, smem_raw);
    return;
  }
  const bool acc_c = flags & 4, c_f32 = flags & 8;
  const bf16* Res = (flags & 16) ? reinterpret_cast<const bf16*>(bufs[I.args[7]]) : nullptr;

  // SW128 swizzle phase = absolute smem address bits [7,10): slab bases must be
  // 1024B-aligned (ws mode offsets opsmem by MK_WS_CTRL_BYTES; df base is unpadded).
  smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  WgmmaSmem& S = *reinterpret_cast<WgmmaSmem*>(smem_raw);
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
          se = se * __expf(mx - zv) + 1.0f;
          mx = zv;
        } else {
          se += __expf(zv - mx);
        }
      }
      for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffff, mx, o);
        const float os = __shfl_xor_sync(0xffffffff, se, o);
        const float Mx = fmaxf(mx, om);
        se = (mx == -INFINITY && om == -INFINITY) ? 0.0f : se * __expf(mx - Mx) + os * __expf(om - Mx);
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
  if ((H & 7) == 0) {
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

__device__ void op_rmsnorm_bwd_dx(const Instr& I, int tile, void** bufs,
                                  char* smem_raw) {
  op_rmsnorm_bwd_dx_impl<false>(I, tile, bufs, smem_raw);
}

__device__ void op_rmsnorm_bwd_dx_fma(const Instr& I, int tile, void** bufs,
                                      char* smem_raw) {
  op_rmsnorm_bwd_dx_impl<true>(I, tile, bufs, smem_raw);
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
  if ((Iw & 7) == 0) {
    for (int i = lane * 8; i < Iw; i += 32 * 8) {
      float g[8], u[8], hv[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
#pragma unroll
      // __expf (SFU): libm expf is a multi-instruction software path that both
      // serializes the lane and bloats register pressure (see the CE epilogue note);
      // error is ~2 ulp, far below bf16 output rounding.
      for (int j = 0; j < 8; j++) hv[j] = g[j] / (1.0f + __expf(-g[j])) * u[j];
      st8bf(h + i, hv);
    }
  } else {
    for (int i = lane; i < Iw; i += 32) {
      const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]);
      h[i] = f2bf(g / (1.0f + __expf(-g)) * u);
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
  if ((Iw & 7) == 0) {
    for (int i = lane * 8; i < Iw; i += 32 * 8) {
      float g[8], u[8], d[8], dg[8], du[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
      ld8dy(dhb, dhf, dy_f32, i, d);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float sig = 1.0f / (1.0f + __expf(-g[j]));  // SFU, see swiglu_fwd note
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
      const float sig = 1.0f / (1.0f + __expf(-g));
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
  if ((Iw & 7) == 0) {
    for (int i = lane64 * 8; i < Iw; i += 64 * 8) {
      float g[8], u[8], d[8], dg[8], du[8];
      ld8bf(gu + i, g);
      ld8bf(gu + Iw + i, u);
      ld8dy(dhb, dhf, dy_f32, i, d);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float sig = 1.0f / (1.0f + __expf(-g[j]));
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
      const float sig = 1.0f / (1.0f + __expf(-g));
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
//        dy_f32, S}
// tile = MK_ROW_R-row group; warp w sweeps (head, row) tasks h = w mod nh (weight-vector
// locality). Per-tile dqw/dkw partials accumulate in smem (fast atomics), then ONE global
// atomicAdd per element per MK_ROW_R rows — the smem zero/flush and its two barriers are
// amortized over the row group. dy_f32 != 0 reads the incoming grad as fp32 (the
// attention-bwd atomic workspace, no CVT chain hop).
__device__ void op_qknorm_rope_bwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int nq = I.args[11], nkv = I.args[12], D = I.args[13];
  const bool dy_f32 = I.args[14] != 0;
  const int S = I.args[15];
  const int stride = (nq + 2 * nkv) * D, nh = nq + 2 * nkv;
  const int warp = mk_tid() / 32, lane = mk_tid() % 32, nwarp = MK_CONSUMERS / 32;

  float* dwq_s = reinterpret_cast<float*>(smem_raw);  // [D] + [D] fp32 partials
  float* dwk_s = dwq_s + D;
  for (int i = mk_tid(); i < 2 * D; i += MK_CONSUMERS) dwq_s[i] = 0.0f;
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
    float* dw_s = is_q ? dwq_s : dwk_s;
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
      atomicAdd(&dw_s[i], da * xh1);
      atomicAdd(&dw_s[i + 32], db * xh2);
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
      atomicAdd(&dw_s[i], da * xh1);
      atomicAdd(&dw_s[i + D / 2], db * xh2);
    }
  }
  consumer_sync();
  float* dqw = reinterpret_cast<float*>(bufs[I.args[5]]);
  float* dkw = reinterpret_cast<float*>(bufs[I.args[6]]);
  for (int i = mk_tid(); i < D; i += MK_CONSUMERS) {
    atomicAdd(&dqw[i], dwq_s[i]);
    atomicAdd(&dkw[i], dwk_s[i]);
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

  // single-pass online (m, s) accumulation: one read of the logits row instead of two
  float mx = -INFINITY, se = 0.0f;
  if (nparts > 0) {
    const float* parts = reinterpret_cast<const float*>(bufs[I.args[6]]) + (int64_t)tile * nparts * 2;
    for (int i = mk_tid(); i < nparts; i += MK_CONSUMERS) {
      const float om = parts[i * 2], os = parts[i * 2 + 1];
      const float M = fmaxf(mx, om);
      se = (mx == -INFINITY && om == -INFINITY) ? 0.0f : se * expf(mx - M) + os * expf(om - M);
      mx = M;
    }
  } else {
    for (int i = mk_tid(); i < V; i += MK_CONSUMERS) {
      const float zv = bf2f(z[i]);
      if (zv > mx) {
        se = se * expf(mx - zv) + 1.0f;
        mx = zv;
      } else {
        se += expf(zv - mx);
      }
    }
  }
  // merge (m, s) pairs across the warp, then across warps via smem
  for (int off = 16; off > 0; off >>= 1) {
    const float om = __shfl_xor_sync(0xffffffff, mx, off);
    const float os = __shfl_xor_sync(0xffffffff, se, off);
    const float M = fmaxf(mx, om);
    se = (mx == -INFINITY && om == -INFINITY) ? 0.0f : se * expf(mx - M) + os * expf(om - M);
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
      se = se * expf(mx - M) + os * expf(om - M);
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
  if ((V & 7) == 0) {  // uint4 IO (v3 P4b): the scalar loop was ~2KB/row/thread of
    // 2-byte accesses on the fattest activation buffer — latency-bound at 8 warps.
    // libm expf kept: bitwise-identical dlogits vs the reference path (the peer
    // session measured __expf here and reverted it).
    for (int i = mk_tid() * 8; i < V; i += MK_CONSUMERS * 8) {
      float zv[8];
      ld8bf(z + i, zv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float p = expf(zv[j] - lse);
        zv[j] = scale * (p - (i + j == label ? 1.0f : 0.0f));
      }
      st8bf(z + i, zv);
    }
  } else {
    for (int i = mk_tid(); i < V; i += MK_CONSUMERS) {
      const float p = expf(bf2f(z[i]) - lse);
      z[i] = f2bf(scale * (p - (i == label ? 1.0f : 0.0f)));
    }
  }
}
