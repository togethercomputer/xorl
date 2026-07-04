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
  const int warp = mk_tid() / 32;
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
  consumer_sync();
}

// Acc[0:32,0:D] += Abf[32,32] @ Brow[32,D]  where Abf is bf16 [32][40] (row or col view).
// a_col: load A as col_major (i.e. use A^T). All 8 warps cover [32, D] (2m x 4n of 16x16
// for D=64; two n-frags each for D=128).
template <bool A_COL>
__device__ __forceinline__ void mma_acc(float (*Acc)[132], const bf16 (*A)[40],
                                        const bf16 (*B)[136], int D) {
  const int warp = mk_tid() / 32;
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
  consumer_sync();
}

// load a [rows, D] slice of a packed [S, row_stride] bf16 buffer into smem (zero-fill OOB)
__device__ __forceinline__ void load_tile(bf16 (*dst)[136], const bf16* base, int r0,
                                          int rows_total, int col_off, int row_stride,
                                          int D) {
  // D and all row/col offsets are multiples of 8 -> 16B-vectorized loads. OOB rows zero.
  const int vpr = D / 8;  // uint4 vectors per row
  for (int v = mk_tid(); v < AT_T * vpr; v += MK_CONSUMERS) {
    const int r = v / vpr, c = (v % vpr) * 8;
    const int gr = r0 + r;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (gr < rows_total)
      val = *reinterpret_cast<const uint4*>(&base[(int64_t)gr * row_stride + col_off + c]);
    *reinterpret_cast<uint4*>(&dst[r][c]) = val;
  }
}

// ---- forward ------------------------------------------------------------------------------
// args: {qkv_r, O, LSE, S, nq, nkv, D, scale_bits}; tile = qtile * nq + qh.
// qt-OUTER tile order: with causal masking, tile t only reads qkvr rows < (qt+1)*AT_T,
// so the df2 region gate on qkvr is plain row-linear (k = nq * REGION_ROWS/AT_T), and
// O/LSE complete in row order, making attention a row-linear PRODUCER for o-proj.
__device__ void op_attn_fwd(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int S = I.args[3], nq = I.args[4], nkv = I.args[5], D = I.args[6];
  const float scale = __int_as_float(I.args[7]);
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  bf16* O = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  float* LSE = reinterpret_cast<float*>(bufs[I.args[2]]);
  AttnSmem& sm = *reinterpret_cast<AttnSmem*>(smem_raw);

  const int qh = tile % nq, q0 = (tile / nq) * AT_T;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();

  load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) sm.Acc1[i / D][i % D] = 0.0f;
  for (int i = tid; i < AT_T * AT_T; i += MK_CONSUMERS) sm.Ps[i / AT_T][i % AT_T] = f2bf(0.0f);
  if (tid < AT_T) {
    sm.row_m[tid] = -INFINITY;
    sm.row_l[tid] = 0.0f;
  }
  consumer_sync();

  for (int k0 = 0; k0 <= q0; k0 += AT_T) {
    load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
    load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
    consumer_sync();
    mma_ab_t(sm, sm.Qs, sm.Ks, D);  // Ss = Q K^T

    // rowwise online softmax update: warp w owns rows 4w..4w+3, lanes = the 32 columns
    {
      const int lane = tid % 32, warp = tid / 32;
      for (int r = warp * 4; r < warp * 4 + 4; ++r) {
        const int qr = q0 + r;
        if (qr >= S) continue;  // OOB q rows: never stored, keep state clean
        const int kr = k0 + lane;
        float sc = (kr <= qr && kr < S) ? sm.Ss[r][lane] * scale : -INFINITY;
        float mx = sc;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        const float m_new = fmaxf(sm.row_m[r], mx);
        const float pv = (sc == -INFINITY) ? 0.0f : expf(sc - m_new);
        float lsum = pv;
        for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
        sm.Ps[r][lane] = f2bf(pv);
        if (lane == 0) {
          const float alpha = expf(sm.row_m[r] - m_new);
          sm.row_alpha[r] = alpha;
          sm.row_m[r] = m_new;
          sm.row_l[r] = sm.row_l[r] * alpha + lsum;
        }
      }
    }
    consumer_sync();
    // rescale O rows by alpha, then O += P @ V
    for (int i = tid; i < AT_T * D; i += MK_CONSUMERS)
      if (q0 + i / D < S) sm.Acc1[i / D][i % D] *= sm.row_alpha[i / D];
    consumer_sync();
    mma_acc<false>(sm.Acc1, sm.Ps, sm.Vs, D);
  }

  // epilogue: O/l -> global, LSE = m + log l
  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) {
    const int r = i / D, c = i % D;
    const int gr = q0 + r;
    if (gr < S) O[(int64_t)gr * (nq * D) + qh * D + c] = f2bf(sm.Acc1[r][c] / sm.row_l[r]);
  }
  if (tid < AT_T && q0 + tid < S)
    LSE[(int64_t)qh * S + q0 + tid] = sm.row_m[tid] + logf(sm.row_l[tid]);
}

// ---- split-KV forward (flash-decoding style) ----------------------------------------
// tile = (qh, qtile, chunk c of C): partial attention over kv tiles c, c+C, ... <= qtile
// with chunk-local online softmax; writes O_c (locally normalized), m_c, l_c. A combine
// op merges the C partials. Empty chunks (c > qtile) write zero-weight partials.
// args: {qkv_r, Opart, Mpart, Lpart, S, nq, nkv, D, scale_bits, C}
// Opart: [C, S, nq*D] bf16; Mpart/Lpart: [C, nq, S] fp32.
__device__ void op_attn_fwd_split(const Instr& I, int tile, void** bufs, char* smem_raw) {
  const int S = I.args[4], nq = I.args[5], nkv = I.args[6], D = I.args[7];
  const float scale = __int_as_float(I.args[8]);
  const int C = I.args[9];
  const bf16* qkv = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  bf16* Opart = reinterpret_cast<bf16*>(bufs[I.args[1]]);
  float* Mpart = reinterpret_cast<float*>(bufs[I.args[2]]);
  float* Lpart = reinterpret_cast<float*>(bufs[I.args[3]]);
  AttnSmem& sm = *reinterpret_cast<AttnSmem*>(smem_raw);

  const int n_qt = (S + AT_T - 1) / AT_T;
  const int qh = tile / (n_qt * C);
  const int rem = tile % (n_qt * C);
  const int q0 = (rem / C) * AT_T;
  const int chunk = rem % C;
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();

  load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) sm.Acc1[i / D][i % D] = 0.0f;
  for (int i = tid; i < AT_T * AT_T; i += MK_CONSUMERS) sm.Ps[i / AT_T][i % AT_T] = f2bf(0.0f);
  if (tid < AT_T) {
    sm.row_m[tid] = -INFINITY;
    sm.row_l[tid] = 0.0f;
  }
  consumer_sync();

  for (int k0 = chunk * AT_T; k0 <= q0; k0 += C * AT_T) {
    load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
    load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
    consumer_sync();
    mma_ab_t(sm, sm.Qs, sm.Ks, D);
    {
      const int lane = tid % 32, warp = tid / 32;
      for (int r = warp * 4; r < warp * 4 + 4; ++r) {
        const int qr = q0 + r;
        if (qr >= S) continue;
        const int kr = k0 + lane;
        float sc = (kr <= qr && kr < S) ? sm.Ss[r][lane] * scale : -INFINITY;
        float mx = sc;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        const float m_new = fmaxf(sm.row_m[r], mx);
        const float pv = (sc == -INFINITY) ? 0.0f : expf(sc - m_new);
        float lsum = pv;
        for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
        sm.Ps[r][lane] = f2bf(pv);
        if (lane == 0) {
          const float alpha = expf(sm.row_m[r] - m_new);
          sm.row_alpha[r] = alpha;
          sm.row_m[r] = m_new;
          sm.row_l[r] = sm.row_l[r] * alpha + lsum;
        }
      }
    }
    consumer_sync();
    for (int i = tid; i < AT_T * D; i += MK_CONSUMERS)
      if (q0 + i / D < S) sm.Acc1[i / D][i % D] *= sm.row_alpha[i / D];
    consumer_sync();
    mma_acc<false>(sm.Acc1, sm.Ps, sm.Vs, D);
  }

  // write locally-normalized partial + (m_c, l_c); empty chunks yield l=0, m=-inf
  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) {
    const int r = i / D, c = i % D;
    const int gr = q0 + r;
    if (gr < S) {
      const float l = sm.row_l[r];
      Opart[(int64_t)chunk * S * nq * D + (int64_t)gr * (nq * D) + qh * D + c] =
          f2bf(l > 0.0f ? sm.Acc1[r][c] / l : 0.0f);
    }
  }
  if (tid < AT_T && q0 + tid < S) {
    Mpart[((int64_t)chunk * nq + qh) * S + q0 + tid] = sm.row_m[tid];
    Lpart[((int64_t)chunk * nq + qh) * S + q0 + tid] = sm.row_l[tid];
  }
}

// combine: O[s,qh,:] = sum_c w_c O_c, w_c = l_c exp(m_c - m*) / l*; LSE = m* + ln l*.
// args: {Opart, Mpart, Lpart, O, LSE, S, nq, D, C}; tile = s; warp w handles head w, w+8...
__device__ void op_attn_combine(const Instr& I, int tile, void** bufs) {
  const int S = I.args[5], nq = I.args[6], D = I.args[7], C = I.args[8];
  const bf16* Opart = reinterpret_cast<const bf16*>(bufs[I.args[0]]);
  const float* Mpart = reinterpret_cast<const float*>(bufs[I.args[1]]);
  const float* Lpart = reinterpret_cast<const float*>(bufs[I.args[2]]);
  bf16* O = reinterpret_cast<bf16*>(bufs[I.args[3]]);
  float* LSE = reinterpret_cast<float*>(bufs[I.args[4]]);
  const int warp = mk_tid() / 32, lane = mk_tid() % 32;
  const int s = tile;
  for (int h = warp; h < nq; h += MK_CONSUMERS / 32) {
    float mstar = -INFINITY;
    for (int c = 0; c < C; ++c)
      mstar = fmaxf(mstar, Mpart[((int64_t)c * nq + h) * S + s]);
    float w[8];  // C <= 8
    float lstar = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float lc = Lpart[((int64_t)c * nq + h) * S + s];
      const float mc = Mpart[((int64_t)c * nq + h) * S + s];
      w[c] = (lc > 0.0f) ? lc * expf(mc - mstar) : 0.0f;
      lstar += w[c];
    }
    for (int d0 = lane * 2; d0 < D; d0 += 64) {  // lanes cover D in float2 steps
      float acc0 = 0.0f, acc1 = 0.0f;
      for (int c = 0; c < C; ++c) {
        if (w[c] == 0.0f) continue;
        const int64_t base = (int64_t)c * S * nq * D + (int64_t)s * (nq * D) + h * D + d0;
        acc0 += w[c] * bf2f(Opart[base]);
        acc1 += w[c] * bf2f(Opart[base + 1]);
      }
      O[(int64_t)s * (nq * D) + h * D + d0] = f2bf(acc0 / lstar);
      O[(int64_t)s * (nq * D) + h * D + d0 + 1] = f2bf(acc1 / lstar);
    }
    if (lane == 0) LSE[(int64_t)h * S + s] = mstar + logf(lstar);
  }
}

// ---- backward preprocess: Drow[qh,s] = sum_d dO[s,qh,d] * O[s,qh,d] ------------------------
// args: {dO, O, Drow, S, nq, D}; tile = s. Warp w handles heads w, w+8, ...
__device__ void op_attn_dpre(const Instr& I, int tile, void** bufs) {
  const int S = I.args[3], nq = I.args[4], D = I.args[5];
  const bf16* dO = reinterpret_cast<const bf16*>(bufs[I.args[0]]) + (int64_t)tile * nq * D;
  const bf16* O = reinterpret_cast<const bf16*>(bufs[I.args[1]]) + (int64_t)tile * nq * D;
  float* Drow = reinterpret_cast<float*>(bufs[I.args[2]]);
  const int warp = mk_tid() / 32, lane = mk_tid() % 32;
  for (int h = warp; h < nq; h += MK_CONSUMERS / 32) {
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
  const int tid = mk_tid();
  const int lane = tid % 32, warp = tid / 32;
  for (int r = warp * 4; r < warp * 4 + 4; ++r) {
    const int qr = q0 + r;
    const float lse = (qr < S) ? LSE[(int64_t)qh * S + qr] : 0.0f;
    const int kr = k0 + lane;
    const bool ok = (kr <= qr && qr < S && kr < S);
    sm.Ps[r][lane] = f2bf(ok ? expf(sm.Ss[r][lane] * scale - lse) : 0.0f);
  }
  consumer_sync();
  mma_ab_t(sm, sm.Ds, sm.Vs, D);  // Ss = dO V^T   (dP)
  for (int r = warp * 4; r < warp * 4 + 4; ++r) {
    const int qr = q0 + r;
    const float dr = (qr < S) ? Drow[(int64_t)qh * S + qr] : 0.0f;
    const float p = bf2f(sm.Ps[r][lane]);
    sm.dSs[r][lane] = f2bf(p * (sm.Ss[r][lane] - dr) * scale);
  }
  consumer_sync();
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
  const int G = nq / nkv;
  const int kvh = tile / (n_kvt * G);
  const int rem = tile % (n_kvt * G);
  const int k0 = (rem / G) * AT_T;
  const int g = rem % G;  // one GQA group member per tile (parallel, fp32-atomic reduce)
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();
  float* ws = reinterpret_cast<float*>(dqkv);  // fp32 [S, stride] workspace (pre-zeroed)

  load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
  load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) {
    sm.Acc1[i / D][i % D] = 0.0f;  // dK
    sm.Acc2[i / D][i % D] = 0.0f;  // dV
  }
  consumer_sync();

  const int qh = kvh * G + g;
  for (int q0 = k0; q0 < S; q0 += AT_T) {
    load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
    load_tile(sm.Ds, dO, q0, S, qh * D, nq * D, D);
    consumer_sync();
    recompute_p_ds(sm, LSE, Drow, qh, q0, k0, S, scale, D);
    mma_acc<true>(sm.Acc2, sm.Ps, sm.Ds, D);   // dV += P^T dO
    mma_acc<true>(sm.Acc1, sm.dSs, sm.Qs, D);  // dK += dS^T Q
  }

  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) {
    const int r = i / D, c = i % D;
    const int gr = k0 + r;
    if (gr < S) {
      atomicAdd(&ws[(int64_t)gr * stride + (nq + kvh) * D + c], sm.Acc1[r][c]);
      atomicAdd(&ws[(int64_t)gr * stride + (nq + nkv + kvh) * D + c], sm.Acc2[r][c]);
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

  const int C = I.args[10];  // kv-chunk parallelism
  const int n_qt = (S + AT_T - 1) / AT_T;
  const int qh = tile / (n_qt * C);
  const int rem = tile % (n_qt * C);
  const int q0 = (rem / C) * AT_T;
  const int chunk = rem % C;  // this tile handles kv tiles chunk, chunk+C, ...
  const int kvh = qh / (nq / nkv);
  const int stride = (nq + 2 * nkv) * D;
  const int tid = mk_tid();
  float* ws = reinterpret_cast<float*>(dqkv);  // fp32 [S, stride] workspace (pre-zeroed)

  load_tile(sm.Qs, qkv, q0, S, qh * D, stride, D);
  load_tile(sm.Ds, dO, q0, S, qh * D, nq * D, D);
  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) sm.Acc1[i / D][i % D] = 0.0f;  // dQ
  consumer_sync();

  for (int k0 = chunk * AT_T; k0 <= q0; k0 += C * AT_T) {
    load_tile(sm.Ks, qkv, k0, S, (nq + kvh) * D, stride, D);
    load_tile(sm.Vs, qkv, k0, S, (nq + nkv + kvh) * D, stride, D);
    consumer_sync();
    recompute_p_ds(sm, LSE, Drow, qh, q0, k0, S, scale, D);
    mma_acc<false>(sm.Acc1, sm.dSs, sm.Ks, D);  // dQ += dS K
  }

  for (int i = tid; i < AT_T * D; i += MK_CONSUMERS) {
    const int r = i / D, c = i % D;
    const int gr = q0 + r;
    if (gr < S) atomicAdd(&ws[(int64_t)gr * stride + qh * D + c], sm.Acc1[r][c]);
  }
}
