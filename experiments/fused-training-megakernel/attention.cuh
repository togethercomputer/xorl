// Causal GQA flash attention (fwd + two-pass bwd) for the fused training megakernel.
//
// Uniform 32x32 tiling (QT=KT=32), WMMA bf16 mma with fp32 accumulation, smem-resident
// running state. q/k/v live packed in qkv_r [S, (nq+2*nkv)*D]; attention output O is
// [S, nq*D]; LSE and Drow are [nq, S] fp32. Backward is the standard FA2 two-pass
// (dKV pass over kv tiles, dQ pass over q tiles) with P recomputed from LSE — no atomics.

#pragma once

#define AT_T 32  // q/kv tile rows

// smem plan (worst case D=128):
//   fwd : Q 32x136 + K 32x136 + V 32x136 bf16 (8.5KB ea) + O 32x129 f32 (16.5KB)
//         + S 32x33 f32 + P 32x40 bf16 + row state  ~= 48KB
//   dkv : K,V,Q,dO tiles + dKacc,dVacc 32x129 f32 + small S/P/dP/dS ~= 81KB
//   dq  : K,V,Q,dO tiles + dQacc + small ~= 65KB
struct AttnSmem {
  bf16 Qs[AT_T][136];
  bf16 Ks[AT_T][136];
  bf16 Vs[AT_T][136];
  bf16 Ds[AT_T][136];   // dO tile
  float Acc1[AT_T][132];  // fwd: O accum | dkv: dK accum | dq: dQ accum (fp32 wmma ld: mult of 4)
  float Acc2[AT_T][132];  // dkv: dV accum
  float Ss[AT_T][36];     // scores / dP staging
  bf16 Ps[AT_T][40];      // P (bf16 for mma)
  bf16 dSs[AT_T][40];     // dS (bf16 for mma)
  float row_m[AT_T], row_l[AT_T], row_alpha[AT_T];
};

// ---- shared mma helpers -----------------------------------------------------------------

// scores[0:QT,0:KT] (fp32, in Ss) = Arow[32,D] @ Brow[32,D]^T  (A=Qs-like, B=Ks-like)
// Warps 0..3 compute the four 16x16 output tiles.
__device__ __forceinline__ void mma_ab_t(AttnSmem& S, const bf16 (*A)[136],
                                         const bf16 (*B)[136], int D) {
  const int warp = threadIdx.x / 32;
  if (warp < 4) {
    const int wm = warp / 2, wn = warp % 2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int kk = 0; kk < D; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
      wmma::load_matrix_sync(a, &A[wm * 16][kk], 136);
      wmma::load_matrix_sync(b, &B[wn * 16][kk], 136);  // col_major: (k,n) at [n][k]
      wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(&S.Ss[wm * 16][wn * 16], c, 36, wmma::mem_row_major);
  }
  __syncthreads();
}

// Acc[0:32,0:D] += Abf[32,32] @ Brow[32,D]  where Abf is bf16 [32][40] (row or col view).
// a_col: load A as col_major (i.e. use A^T). All 8 warps cover [32, D] (2m x 4n of 16x16
// for D=64; two n-frags each for D=128).
template <bool A_COL>
__device__ __forceinline__ void mma_acc(float (*Acc)[132], const bf16 (*A)[40],
                                        const bf16 (*B)[136], int D) {
  const int warp = threadIdx.x / 32;
  const int wm = warp / 4, wn = warp % 4;  // 2 x 4
  const int nfr = D / 64;                  // 16-col frags per warp
  for (int f = 0; f < nfr; ++f) {
    const int n0 = (wn + f * 4) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::load_matrix_sync(c, &Acc[wm * 16][n0], 132, wmma::mem_row_major);
    for (int kk = 0; kk < 32; kk += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
      wmma::load_matrix_sync(b, &B[kk][n0], 136);
      if (A_COL) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
        wmma::load_matrix_sync(a, &A[kk][wm * 16], 40);  // (m,k) at [k][m]
        wmma::mma_sync(c, a, b, c);
      } else {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::load_matrix_sync(a, &A[wm * 16][kk], 40);
        wmma::mma_sync(c, a, b, c);
      }
    }
    wmma::store_matrix_sync(&Acc[wm * 16][n0], c, 132, wmma::mem_row_major);
  }
  __syncthreads();
}

// load a [rows, D] slice of a packed [S, row_stride] bf16 buffer into smem (zero-fill OOB)
__device__ __forceinline__ void load_tile(bf16 (*dst)[136], const bf16* base, int r0,
                                          int rows_total, int col_off, int row_stride,
                                          int D) {
  // D and all row/col offsets are multiples of 8 -> 16B-vectorized loads. OOB rows zero.
  const int vpr = D / 8;  // uint4 vectors per row
  for (int v = threadIdx.x; v < AT_T * vpr; v += blockDim.x) {
    const int r = v / vpr, c = (v % vpr) * 8;
    const int gr = r0 + r;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (gr < rows_total)
      val = *reinterpret_cast<const uint4*>(&base[(int64_t)gr * row_stride + col_off + c]);
    *reinterpret_cast<uint4*>(&dst[r][c]) = val;
  }
}

// ---- forward ------------------------------------------------------------------------------
// args: {qkv_r, O, LSE, S, nq, nkv, D, scale_bits}; tile = qh * n_qtiles + qtile
__device__ void op_attn_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int S = I.args[3], nq = I.args[4], nkv = I.args[5], D = I.args[6];
  const float scale = __int_as_float(I.args[7]);
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  bf16* O = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  float* LSE = reinterpret_cast<float*>(bufs[I.args[2]]);
  AttnSmem& sm = *reinterpret_cast<AttnSmem*>(smem_raw);

  const int n_qt = (S + AT_T - 1) / AT_T;
  const int qh = tile / n_qt, q0 = (tile % n_qt) * AT_T;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = threadIdx.x;

  load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
  for (int i = tid; i < AT_T * D; i += blockDim.x) sm.Acc1[i / D][i % D] = 0.0f;
  if (tid < AT_T) {
    sm.row_m[tid] = -INFINITY;
    sm.row_l[tid] = 0.0f;
  }
  __syncthreads();

  for (int k0 = 0; k0 <= q0; k0 += AT_T) {
    load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
    load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
    __syncthreads();
    mma_ab_t(sm, sm.Qs, sm.Ks, D);  // Ss = Q K^T

    // rowwise online softmax update (thread t < 32 owns row t)
    if (tid < AT_T) {
      const int qr = q0 + tid;
      float m_new = sm.row_m[tid];
      for (int j = 0; j < AT_T; ++j) {
        const int kr = k0 + j;
        float s = (kr <= qr && qr < S && kr < S) ? sm.Ss[tid][j] * scale : -INFINITY;
        sm.Ss[tid][j] = s;
        m_new = fmaxf(m_new, s);
      }
      const float alpha = expf(sm.row_m[tid] - m_new);
      float lsum = 0.0f;
      for (int j = 0; j < AT_T; ++j) {
        const float p = expf(sm.Ss[tid][j] - m_new);
        sm.Ps[tid][j] = f2bf(p);
        lsum += p;
      }
      sm.row_m[tid] = m_new;
      sm.row_l[tid] = sm.row_l[tid] * alpha + lsum;
      sm.row_alpha[tid] = alpha;
    }
    __syncthreads();
    // rescale O rows by alpha, then O += P @ V
    for (int i = tid; i < AT_T * D; i += blockDim.x)
      sm.Acc1[i / D][i % D] *= sm.row_alpha[i / D];
    __syncthreads();
    mma_acc<false>(sm.Acc1, sm.Ps, sm.Vs, D);
  }

  // epilogue: O/l -> global, LSE = m + log l
  for (int i = tid; i < AT_T * D; i += blockDim.x) {
    const int r = i / D, c = i % D;
    const int gr = q0 + r;
    if (gr < S) O[(int64_t)gr * (nq * D) + qh * D + c] = f2bf(sm.Acc1[r][c] / sm.row_l[r]);
  }
  if (tid < AT_T && q0 + tid < S)
    LSE[(int64_t)qh * S + q0 + tid] = sm.row_m[tid] + logf(sm.row_l[tid]);
}

// ---- backward preprocess: Drow[qh,s] = sum_d dO[s,qh,d] * O[s,qh,d] ------------------------
// args: {dO, O, Drow, S, nq, D}; tile = s. Warp w handles heads w, w+8, ...
__device__ void op_attn_dpre(const Instr& I, int tile, void** bufs) {
  const int S = I.args[3], nq = I.args[4], D = I.args[5];
  const bf16* dO = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * nq * D;
  const bf16* O = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)tile * nq * D;
  float* Drow = reinterpret_cast<float*>(bufs[I.args[2]]);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int h = warp; h < nq; h += blockDim.x / 32) {
    float acc = 0.0f;
    for (int d = lane; d < D; d += 32) acc += bf2f(dO[h * D + d]) * bf2f(O[h * D + d]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Drow[(int64_t)h * S + tile] = acc;
  }
}

// ---- shared: recompute P and dS for a (q-tile, kv-tile) pair -------------------------------
// Requires Qs/Ks/Vs/Ds loaded. Produces Ps (P^bf16) and dSs (dS^bf16). Thread t<32 = q row t.
__device__ __forceinline__ void recompute_p_ds(AttnSmem& sm, const float* LSE,
                                               const float* Drow, int qh, int q0, int k0,
                                               int S, float scale, int D) {
  mma_ab_t(sm, sm.Qs, sm.Ks, D);  // Ss = Q K^T
  const int tid = threadIdx.x;
  if (tid < AT_T) {
    const int qr = q0 + tid;
    const float lse = (qr < S) ? LSE[(int64_t)qh * S + qr] : 0.0f;
    for (int j = 0; j < AT_T; ++j) {
      const int kr = k0 + j;
      const bool ok = (kr <= qr && qr < S && kr < S);
      sm.Ps[tid][j] = f2bf(ok ? expf(sm.Ss[tid][j] * scale - lse) : 0.0f);
    }
  }
  __syncthreads();
  mma_ab_t(sm, sm.Ds, sm.Vs, D);  // Ss = dO V^T   (dP)
  if (tid < AT_T) {
    const int qr = q0 + tid;
    const float dr = (qr < S) ? Drow[(int64_t)qh * S + qr] : 0.0f;
    for (int j = 0; j < AT_T; ++j) {
      const float p = bf2f(sm.Ps[tid][j]);
      sm.dSs[tid][j] = f2bf(p * (sm.Ss[tid][j] - dr) * scale);
    }
  }
  __syncthreads();
}

// ---- backward dK/dV pass --------------------------------------------------------------------
// args: {qkv_r, dO, LSE, Drow, dqkv_r, S, nq, nkv, D, scale_bits}
// tile = kvh * n_kvtiles + kvtile. Loops all q heads in the GQA group x q tiles >= kv tile.
__device__ void op_attn_dkv(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int S = I.args[5], nq = I.args[6], nkv = I.args[7], D = I.args[8];
  const float scale = __int_as_float(I.args[9]);
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* dO = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const float* LSE = reinterpret_cast<const float*>(bufs[I.args[2]]);
  const float* Drow = reinterpret_cast<const float*>(bufs[I.args[3]]);
  bf16* dqkv = reinterpret_cast<bf16*>(bufs[I.args[4]]);
  AttnSmem& sm = *reinterpret_cast<AttnSmem*>(smem_raw);

  const int n_kvt = (S + AT_T - 1) / AT_T;
  const int kvh = tile / n_kvt, k0 = (tile % n_kvt) * AT_T;
  const int G = nq / nkv;
  const int stride = (nq + 2 * nkv) * D;
  const int tid = threadIdx.x;

  load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
  load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
  for (int i = tid; i < AT_T * D; i += blockDim.x) {
    sm.Acc1[i / D][i % D] = 0.0f;  // dK
    sm.Acc2[i / D][i % D] = 0.0f;  // dV
  }
  __syncthreads();

  for (int g = 0; g < G; ++g) {
    const int qh = kvh * G + g;
    for (int q0 = k0; q0 < S; q0 += AT_T) {
      load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
      load_tile(sm.Ds, dO, q0, S, qh * D, nq * D, D);
      __syncthreads();
      recompute_p_ds(sm, LSE, Drow, qh, q0, k0, S, scale, D);
      mma_acc<true>(sm.Acc2, sm.Ps, sm.Ds, D);   // dV += P^T dO
      mma_acc<true>(sm.Acc1, sm.dSs, sm.Qs, D);  // dK += dS^T Q
    }
  }

  for (int i = tid; i < AT_T * D; i += blockDim.x) {
    const int r = i / D, c = i % D;
    const int gr = k0 + r;
    if (gr < S) {
      dqkv[(int64_t)gr * stride + (nq + kvh) * D + c] = f2bf(sm.Acc1[r][c]);
      dqkv[(int64_t)gr * stride + (nq + nkv + kvh) * D + c] = f2bf(sm.Acc2[r][c]);
    }
  }
}

// ---- backward dQ pass -------------------------------------------------------------------------
// args: {qkv_r, dO, LSE, Drow, dqkv_r, S, nq, nkv, D, scale_bits}
// tile = qh * n_qtiles + qtile. Loops kv tiles <= q tile.
__device__ void op_attn_dq(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int S = I.args[5], nq = I.args[6], nkv = I.args[7], D = I.args[8];
  const float scale = __int_as_float(I.args[9]);
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const bf16* dO = reinterpret_cast<const bf16*>(bufs[I.args[1]]);
  const float* LSE = reinterpret_cast<const float*>(bufs[I.args[2]]);
  const float* Drow = reinterpret_cast<const float*>(bufs[I.args[3]]);
  bf16* dqkv = reinterpret_cast<bf16*>(bufs[I.args[4]]);
  AttnSmem& sm = *reinterpret_cast<AttnSmem*>(smem_raw);

  const int n_qt = (S + AT_T - 1) / AT_T;
  const int qh = tile / n_qt, q0 = (tile % n_qt) * AT_T;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = threadIdx.x;

  load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
  load_tile(sm.Ds, dO, q0, S, qh * D, nq * D, D);
  for (int i = tid; i < AT_T * D; i += blockDim.x) sm.Acc1[i / D][i % D] = 0.0f;  // dQ
  __syncthreads();

  for (int k0 = 0; k0 <= q0; k0 += AT_T) {
    load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
    load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
    __syncthreads();
    recompute_p_ds(sm, LSE, Drow, qh, q0, k0, S, scale, D);
    mma_acc<false>(sm.Acc1, sm.dSs, sm.Ks, D);  // dQ += dS K
  }

  for (int i = tid; i < AT_T * D; i += blockDim.x) {
    const int r = i / D, c = i % D;
    const int gr = q0 + r;
    if (gr < S) dqkv[(int64_t)gr * stride + qh * D + c] = f2bf(sm.Acc1[r][c]);
  }
}
