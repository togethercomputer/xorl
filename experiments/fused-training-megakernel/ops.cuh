// Device op library for the fused training megakernel.
//
// Every op is a __device__ function taking (Instr, tile, bufs, smem): `tile` is the
// block-level work item index and the WHOLE block (256 threads) executes it.
// Ops read operand buffer indices + shape ints from Instr.args.
//
// Conventions: activations/params bf16 row-major; every accumulation fp32; weight
// grads fp32. GEMM tiles are 64x64; row ops use one row per work item.

#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

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
  for (int i = threadIdx.x; i < quads; i += blockDim.x) p4[i] = v4;
  for (int i = base + quads * 4 + threadIdx.x; i < end; i += blockDim.x) p[i] = v;
}

// args: {y, x, n, alpha_bits}; tile = 4096-element chunk index. y += alpha * x
__device__ __forceinline__ void op_axpy_f32(const Instr& I, int tile, void** bufs) {
  float* y = reinterpret_cast<float*>(bufs[I.args[0]]);
  const float* x = reinterpret_cast<const float*>(bufs[I.args[1]]);
  const int n = I.args[2];
  const float a = __int_as_float(I.args[3]);
  const int base = tile * 4096;
  for (int i = base + threadIdx.x; i < min(base + 4096, n); i += blockDim.x)
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
  const int tid = threadIdx.x;
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
    for (int i = tid; i < GEMM_BM * GEMM_BK; i += blockDim.x) {
      const int m = i / GEMM_BK, k = i % GEMM_BK;
      const int gm = m0 + m, gk = k0 + k;
      bf16 v = f2bf(0.0f);
      if (gm < M && gk < K) v = a_t ? A[(int64_t)gk * M + gm] : A[(int64_t)gm * K + gk];
      S.As[m][k] = v;
    }
    for (int i = tid; i < GEMM_BK * GEMM_BN; i += blockDim.x) {
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
      __syncthreads();  // smem staged for everyone
      if (k_next < k_fast_end) issue_loads(k_next, pa, pb0, pb1);  // overlap with mma
      mma_tile();
      __syncthreads();  // everyone done reading smem before next stage
    }
  }
  for (int k0 = k_fast_end; k0 < k_hi; k0 += GEMM_BK) {  // guarded tail
    load_slow(k0);
    __syncthreads();
    mma_tile();
    __syncthreads();
  }

  // stage fp32 result, then epilogue with bounds guards
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&S.Cs[wm * 32 + i * 16][wn * 32 + j * 16], c_frag[i][j],
                              GEMM_LDC, wmma::mem_row_major);
  __syncthreads();

  for (int i = tid; i < GEMM_BM * GEMM_BN; i += blockDim.x) {
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
}

// ---- fp32 -> bf16 convert (drains split/atomic fp32 workspaces) -------------------------
// args: {src_f32, dst_bf16, n}; tile = MK_CHUNK-element chunk.
__device__ __forceinline__ void op_cvt_f32_bf16(const Instr& I, int tile, void** bufs) {
  const float* src = reinterpret_cast<const float*>(bufs[I.args[0]]);
  bf16* dst = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  const int n = I.args[2];
  const int base = tile * MK_CHUNK;
  for (int i = base + threadIdx.x; i < min(base + MK_CHUNK, n); i += blockDim.x)
    dst[i] = f2bf(src[i]);
}

// ---- block-wide row reduction helper ---------------------------------------------------
// Sums `val` across the block (256 threads). Uses 32 floats of scratch.
__device__ __forceinline__ float block_sum(float val, float* scratch) {
  for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = val;
  __syncthreads();
  float total = 0.0f;
  if (threadIdx.x < blockDim.x / 32) total = scratch[threadIdx.x];
  for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(0xffffffff, total, off);
  total = __shfl_sync(0xffffffff, total, 0);
  if (threadIdx.x == 0) scratch[0] = total;
  __syncthreads();
  total = scratch[0];
  __syncthreads();
  return total;
}

// ---- RMSNorm ---------------------------------------------------------------------------
// fwd: y[r,:] = x[r,:] * rstd * w ; rstd = 1/sqrt(mean(x^2)+eps). Saves rstd (fp32).
// args: {x, w, y, rstd, H, eps_bits}; tile = row.
__device__ void op_rmsnorm_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const bf16* x = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * I.args[4];
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  bf16* y = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)tile * I.args[4];
  float* rstd = reinterpret_cast<float*>(bufs[I.args[3]]);
  const int H = I.args[4];
  const float eps = __int_as_float(I.args[5]);
  float* scratch = reinterpret_cast<float*>(smem_raw);

  float ss = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const float v = bf2f(x[i]);
    ss += v * v;
  }
  ss = block_sum(ss, scratch);
  const float r = rsqrtf(ss / H + eps);
  if (threadIdx.x == 0) rstd[tile] = r;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    y[i] = f2bf(bf2f(x[i]) * r * bf2f(w[i]));
}

// bwd: with xhat = x*rstd, g = dy*w:
//   dx += rstd * (g - xhat * mean(g * xhat))       (accumulates into dx: residual stream)
//   dw += dy * xhat                                 (fp32 atomics)
// args: {x, w, dy, dx, dw, rstd, H}; tile = row.
__device__ void op_rmsnorm_bwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int H = I.args[6];
  const bf16* x = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * H;
  const bf16* w = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const bf16* dy = reinterpret_cast<const bf16*>(bufs[I.args[2]]) + (int64_t)tile * H;
  bf16* dx = reinterpret_cast<bf16*>(bufs[I.args[3]]) + (int64_t)tile * H;
  float* dw = reinterpret_cast<float*>(bufs[I.args[4]]);
  const float r = reinterpret_cast<const float*>(bufs[I.args[5]])[tile];
  float* scratch = reinterpret_cast<float*>(smem_raw);

  float dot = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const float xhat = bf2f(x[i]) * r;
    dot += bf2f(dy[i]) * bf2f(w[i]) * xhat;
  }
  dot = block_sum(dot, scratch) / H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const float xhat = bf2f(x[i]) * r;
    const float g = bf2f(dy[i]) * bf2f(w[i]);
    dx[i] = f2bf(bf2f(dx[i]) + r * (g - xhat * dot));
    atomicAdd(&dw[i], bf2f(dy[i]) * xhat);
  }
}

// ---- SwiGLU ----------------------------------------------------------------------------
// fwd: h[r,i] = silu(gate[r,i]) * up[r,i], gate/up = halves of gu[r, 2I] (gate first).
// args: {gu, h, S, Iw}; tile = row.
__device__ void op_swiglu_fwd(const Instr& I, int tile, void** bufs) {
  const int Iw = I.args[3];
  const bf16* gu = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * 2 * Iw;
  bf16* h = reinterpret_cast<bf16*>(bufs[I.args[1]]) + (int64_t)tile * Iw;
  for (int i = threadIdx.x; i < Iw; i += blockDim.x) {
    const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]);
    const float sg = g / (1.0f + expf(-g));
    h[i] = f2bf(sg * u);
  }
}

// bwd: dgate = dh * u * dsilu(g); dup = dh * silu(g). Writes dgu (bf16).
// args: {gu, dh, dgu, S, Iw}; tile = row.
__device__ void op_swiglu_bwd(const Instr& I, int tile, void** bufs) {
  const int Iw = I.args[4];
  const bf16* gu = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * 2 * Iw;
  const bf16* dh = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)tile * Iw;
  bf16* dgu = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)tile * 2 * Iw;
  for (int i = threadIdx.x; i < Iw; i += blockDim.x) {
    const float g = bf2f(gu[i]), u = bf2f(gu[Iw + i]), d = bf2f(dh[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    const float sg = g * sig;
    dgu[i] = f2bf(d * u * (sig + sg * (1.0f - sig)));  // dsilu = sig + silu*(1-sig)
    dgu[Iw + i] = f2bf(d * sg);
  }
}

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
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nwarp = blockDim.x / 32;

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
// args: {qkv_raw, dqkv_r, dqkv_raw, qw, kw, dqw, dkw, rstd_q, rstd_k, cos, sin, nq, nkv, D}
// tile = row; warp w handles heads w, w+8, ... Per-block dqw/dkw partials accumulate in
// smem (fast atomics), then ONE global atomicAdd per element per row — instead of one
// per (row, head), which serialized on the tiny [D] grad buffers.
__device__ void op_qknorm_rope_bwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int nq = I.args[11], nkv = I.args[12], D = I.args[13];
  const int row = tile;
  const int stride = (nq + 2 * nkv) * D;
  const bf16* x_row = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)row * stride;
  const bf16* dy_row = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)row * stride;
  bf16* dx_row = reinterpret_cast<bf16*>(bufs[I.args[2]]) + (int64_t)row * stride;
  const float* cosr = reinterpret_cast<const float*>(bufs[I.args[9]]) + (int64_t)row * (D / 2);
  const float* sinr = reinterpret_cast<const float*>(bufs[I.args[10]]) + (int64_t)row * (D / 2);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nwarp = blockDim.x / 32;

  float* dwq_s = reinterpret_cast<float*>(smem_raw);  // [D] + [D] fp32 partials
  float* dwk_s = dwq_s + D;
  for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) dwq_s[i] = 0.0f;
  __syncthreads();

  for (int h = warp; h < nq + 2 * nkv; h += nwarp) {
    const bf16* xr = x_row + h * D;
    const bf16* dyr = dy_row + h * D;
    bf16* dxr = dx_row + h * D;
    if (h >= nq + nkv) {  // v grads pass through
      for (int i = lane; i < D; i += 32) dxr[i] = dyr[i];
      continue;
    }
    const bool is_q = h < nq;
    const bf16* w = reinterpret_cast<const bf16*>(bufs[is_q ? I.args[3] : I.args[4]]);
    float* dw_s = is_q ? dwq_s : dwk_s;
    const float r = reinterpret_cast<const float*>(
        bufs[is_q ? I.args[7] : I.args[8]])[(int64_t)row * (is_q ? nq : nkv) + (is_q ? h : h - nq)];

    // rope^-1 on the incoming grad: da = dy1*c + dy2*s ; db = -dy1*s + dy2*c;
    // then per-head rmsnorm bwd with x = raw.
    float dot = 0.0f;
    for (int i = lane; i < D / 2; i += 32) {
      const float c = cosr[i], s = sinr[i];
      const float dy1 = bf2f(dyr[i]), dy2 = bf2f(dyr[i + D / 2]);
      const float da = dy1 * c + dy2 * s;
      const float db = -dy1 * s + dy2 * c;
      dot += da * bf2f(w[i]) * bf2f(xr[i]) * r + db * bf2f(w[i + D / 2]) * bf2f(xr[i + D / 2]) * r;
    }
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
    dot /= D;
    for (int i = lane; i < D / 2; i += 32) {
      const float c = cosr[i], s = sinr[i];
      const float dy1 = bf2f(dyr[i]), dy2 = bf2f(dyr[i + D / 2]);
      const float da = dy1 * c + dy2 * s;
      const float db = -dy1 * s + dy2 * c;
      const float xh1 = bf2f(xr[i]) * r, xh2 = bf2f(xr[i + D / 2]) * r;
      dxr[i] = f2bf(r * (da * bf2f(w[i]) - xh1 * dot));
      dxr[i + D / 2] = f2bf(r * (db * bf2f(w[i + D / 2]) - xh2 * dot));
      atomicAdd(&dw_s[i], da * xh1);
      atomicAdd(&dw_s[i + D / 2], db * xh2);
    }
  }
  __syncthreads();
  float* dqw = reinterpret_cast<float*>(bufs[I.args[5]]);
  float* dkw = reinterpret_cast<float*>(bufs[I.args[6]]);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
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
  for (int i = threadIdx.x; i < H; i += blockDim.x) x[i] = e[i];
}

// bwd scatter-add: demb[tok[r],:] += dx[r,:] (fp32 atomics). args: {tok, dx, demb, H}.
__device__ void op_embed_bwd(const Instr& I, int tile, void** bufs) {
  const int H = I.args[3];
  const int t = reinterpret_cast<const int*>(bufs[I.args[0]])[tile];
  const bf16* dx = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)tile * H;
  float* de = reinterpret_cast<float*>(bufs[I.args[2]]) + (int64_t)t * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) atomicAdd(&de[i], bf2f(dx[i]));
}

// ---- cross entropy ----------------------------------------------------------------------
// fwd over materialized logits [S, V]: per row lse; loss_sum += (lse - z_t) for valid rows.
// inv_valid is a device fp32 scalar (1/num_valid). loss is a device fp32 scalar (pre-zeroed).
// args: {logits, labels, lse, loss, inv_valid, V}; tile = row. labels int32, -100 = ignore.
__device__ void op_ce_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int V = I.args[5];
  const bf16* z = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * V;
  const int label = reinterpret_cast<const int*>(bufs[I.args[1]])[tile];
  float* lse_out = reinterpret_cast<float*>(bufs[I.args[2]]);
  float* loss = reinterpret_cast<float*>(bufs[I.args[3]]);
  const float inv_valid = *reinterpret_cast<const float*>(bufs[I.args[4]]);
  float* scratch = reinterpret_cast<float*>(smem_raw);

  // single-pass online (m, s) accumulation: one read of the logits row instead of two
  float mx = -INFINITY, se = 0.0f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float zv = bf2f(z[i]);
    if (zv > mx) {
      se = se * expf(mx - zv) + 1.0f;
      mx = zv;
    } else {
      se += expf(zv - mx);
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
  if ((threadIdx.x & 31) == 0) {
    scratch[(threadIdx.x >> 5) * 2] = mx;
    scratch[(threadIdx.x >> 5) * 2 + 1] = se;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    mx = scratch[0];
    se = scratch[1];
    for (int w = 1; w < blockDim.x / 32; ++w) {
      const float om = scratch[w * 2], os = scratch[w * 2 + 1];
      const float M = fmaxf(mx, om);
      se = se * expf(mx - M) + os * expf(om - M);
      mx = M;
    }
    scratch[0] = mx;
    scratch[1] = se;
  }
  __syncthreads();
  mx = scratch[0];
  se = scratch[1];
  __syncthreads();
  const float lse = mx + logf(se);
  if (threadIdx.x == 0) {
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
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float p = expf(bf2f(z[i]) - lse);
    z[i] = f2bf(scale * (p - (i == label ? 1.0f : 0.0f)));
  }
}
