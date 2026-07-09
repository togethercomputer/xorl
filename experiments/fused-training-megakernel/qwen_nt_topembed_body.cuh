{
  const Instr& _mk_nttop_I = MK_NT_TOPEMBED_I;
  const int _mk_nttop_tile = MK_NT_TOPEMBED_TILE;
  void** _mk_nttop_bufs = MK_NT_TOPEMBED_BUFS;
  char* _mk_nttop_smem_raw = MK_NT_TOPEMBED_SMEM;
  bf16* _mk_nttop_C = reinterpret_cast<bf16*>(_mk_nttop_bufs[_mk_nttop_I.args[2]]);
  const int _mk_nttop_M = _mk_nttop_I.args[3];
  const int _mk_nttop_N = _mk_nttop_I.args[4];
  const int _mk_nttop_K = _mk_nttop_I.args[5];
  const int _mk_nttop_flags = _mk_nttop_I.args[6];

  _mk_nttop_smem_raw = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(_mk_nttop_smem_raw) + 1023) & ~uintptr_t(1023));
  WgmmaSmemN256NtSupertileT<3>& _mk_nttop_S =
      *reinterpret_cast<WgmmaSmemN256NtSupertileT<3>*>(_mk_nttop_smem_raw);
  const int _mk_nttop_n_tiles = _mk_nttop_N / 128;
  const int _mk_nttop_m_tiles = _mk_nttop_M / 256;
  const bool _mk_nttop_nmajor = _mk_nttop_flags & GEMM_N256_NMAJOR_FLAG;
  const int _mk_nttop_mt = _mk_nttop_nmajor
      ? (_mk_nttop_tile % _mk_nttop_m_tiles)
      : (_mk_nttop_tile / _mk_nttop_n_tiles);
  const int _mk_nttop_nt = _mk_nttop_nmajor
      ? (_mk_nttop_tile / _mk_nttop_m_tiles)
      : (_mk_nttop_tile % _mk_nttop_n_tiles);
  const int _mk_nttop_m0 = _mk_nttop_mt * 256;
  const int _mk_nttop_n0 = _mk_nttop_nt * 128;
  const int _mk_nttop_tid = mk_tid();
  const int _mk_nttop_wg = _mk_nttop_tid / 128;

  const char* _mk_nttop_tbl =
      reinterpret_cast<const char*>(_mk_nttop_bufs[_mk_nttop_I.args[20] - 1]);
  const char* _mk_nttop_tmA = _mk_nttop_tbl + (int64_t)_mk_nttop_I.args[21] * 128;
  const char* _mk_nttop_tmB = _mk_nttop_tbl + (int64_t)_mk_nttop_I.args[22] * 128;
  uint64_t* _mk_nttop_bfull =
      reinterpret_cast<uint64_t*>(_mk_nttop_smem_raw + sizeof(_mk_nttop_S));
  uint64_t* _mk_nttop_bempty = _mk_nttop_bfull + 3;
  if (_mk_nttop_tid == 0) {
    wg_tmap_fence_acquire(_mk_nttop_tmA);
    wg_tmap_fence_acquire(_mk_nttop_tmB);
#pragma unroll
    for (int _mk_nttop_s = 0; _mk_nttop_s < 3; ++_mk_nttop_s) {
      wg_mbar_init(&_mk_nttop_bfull[_mk_nttop_s], 1);
      wg_mbar_init(&_mk_nttop_bempty[_mk_nttop_s], 256);
    }
  }
  consumer_sync();
  if (_mk_nttop_tid == 0) {
    MkPdfFeed& _mk_nttop_F = g_pdf_feed;
    _mk_nttop_F.tmA = _mk_nttop_tmA;
    _mk_nttop_F.tmB = _mk_nttop_tmB;
    _mk_nttop_F.a0 = reinterpret_cast<char*>(_mk_nttop_S.A[0][0]);
    _mk_nttop_F.a1 = reinterpret_cast<char*>(_mk_nttop_S.A[0][2]);
    _mk_nttop_F.b0 = reinterpret_cast<char*>(_mk_nttop_S.B[0]);
    _mk_nttop_F.a_stride =
        (int)(reinterpret_cast<char*>(_mk_nttop_S.A[1][0]) -
              reinterpret_cast<char*>(_mk_nttop_S.A[0][0]));
    _mk_nttop_F.b_stride =
        (int)(reinterpret_cast<char*>(_mk_nttop_S.B[1]) -
              reinterpret_cast<char*>(_mk_nttop_S.B[0]));
    _mk_nttop_F.bfull = _mk_nttop_bfull;
    _mk_nttop_F.bempty = _mk_nttop_bempty;
    _mk_nttop_F.m0 = _mk_nttop_m0;
    _mk_nttop_F.n0 = _mk_nttop_n0;
    _mk_nttop_F.iters = _mk_nttop_K / WG_BK;
    _mk_nttop_F.stages = 3;
    _mk_nttop_F.a_t = 0;
    _mk_nttop_F.b_t = 1;
    _mk_nttop_F.bk = WG_BK;
    _mk_nttop_F.k_base = 0;
    _mk_nttop_F.kind = 6;
    _mk_nttop_F.expect_bytes = 49152;
    mk_pdf_st_release(&_mk_nttop_F.seq, _mk_nttop_F.seq + 1);
  }

  float _mk_nttop_d0[64], _mk_nttop_d1[64];
#pragma unroll
  for (int _mk_nttop_i = 0; _mk_nttop_i < 64; ++_mk_nttop_i) {
    _mk_nttop_d0[_mk_nttop_i] = 0.0f;
    _mk_nttop_d1[_mk_nttop_i] = 0.0f;
  }
  const int _mk_nttop_iters = _mk_nttop_K / WG_BK;
  for (int _mk_nttop_k = 0; _mk_nttop_k < _mk_nttop_iters; ++_mk_nttop_k) {
    const int _mk_nttop_st = _mk_nttop_k % 3;
    wg_mbar_wait(&_mk_nttop_bfull[_mk_nttop_st], (_mk_nttop_k / 3) & 1);
    uint64_t _mk_nttop_da0[4], _mk_nttop_da1[4], _mk_nttop_db[4];
#pragma unroll
    for (int _mk_nttop_s = 0; _mk_nttop_s < 4; ++_mk_nttop_s) {
      _mk_nttop_da0[_mk_nttop_s] =
          wg_desc_ksw(_mk_nttop_S.A[_mk_nttop_st][_mk_nttop_wg], _mk_nttop_s);
      _mk_nttop_da1[_mk_nttop_s] =
          wg_desc_ksw(_mk_nttop_S.A[_mk_nttop_st][2 + _mk_nttop_wg], _mk_nttop_s);
      _mk_nttop_db[_mk_nttop_s] = wg_desc_ksw(_mk_nttop_S.B[_mk_nttop_st], _mk_nttop_s);
    }

#define MK_NTTOP_D4(_d, _i) (_d)[(_i) + 0], (_d)[(_i) + 1], (_d)[(_i) + 2], (_d)[(_i) + 3]
#define MK_NTTOP_D16(_d, _i) \
  MK_NTTOP_D4(_d, _i), MK_NTTOP_D4(_d, (_i) + 4), \
      MK_NTTOP_D4(_d, (_i) + 8), MK_NTTOP_D4(_d, (_i) + 12)
#define MK_NTTOP_D64(_d) \
  MK_NTTOP_D16(_d, 0), MK_NTTOP_D16(_d, 16), \
      MK_NTTOP_D16(_d, 32), MK_NTTOP_D16(_d, 48)
    using MkNtTopMma =
        cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<
            cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>;
    cute::warpgroup_arrive();
#pragma unroll
    for (int _mk_nttop_s = 0; _mk_nttop_s < 4; ++_mk_nttop_s)
      MkNtTopMma::fma(_mk_nttop_da0[_mk_nttop_s], _mk_nttop_db[_mk_nttop_s],
                      MK_NTTOP_D64(_mk_nttop_d0),
                      cute::SM90::GMMA::ScaleOut::One);
#pragma unroll
    for (int _mk_nttop_s = 0; _mk_nttop_s < 4; ++_mk_nttop_s)
      MkNtTopMma::fma(_mk_nttop_da1[_mk_nttop_s], _mk_nttop_db[_mk_nttop_s],
                      MK_NTTOP_D64(_mk_nttop_d1),
                      cute::SM90::GMMA::ScaleOut::One);
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
#undef MK_NTTOP_D64
#undef MK_NTTOP_D16
#undef MK_NTTOP_D4
    wg_mbar_arrive(&_mk_nttop_bempty[_mk_nttop_st]);
  }
  cute::warpgroup_wait<0>();
  consumer_sync();
  if (_mk_nttop_tid == 0) {
#pragma unroll
    for (int _mk_nttop_s = 0; _mk_nttop_s < 3; ++_mk_nttop_s) {
      wg_mbar_init(&_mk_nttop_bfull[_mk_nttop_s], 1);
      wg_mbar_init(&_mk_nttop_bempty[_mk_nttop_s], 256);
    }
  }
#ifndef MK_GEMM_N256_NT_SUPERTILE_POSTINIT_NOSYNC
  consumer_sync();
#endif

  const int _mk_nttop_wtid = _mk_nttop_tid % 128;
  const int _mk_nttop_w = _mk_nttop_wtid / 32;
  const int _mk_nttop_l = _mk_nttop_wtid % 32;
  const int _mk_nttop_cb = (_mk_nttop_l & 3) * 2;
#pragma unroll
  for (int _mk_nttop_strip = 0; _mk_nttop_strip < 2; ++_mk_nttop_strip) {
    const float* _mk_nttop_d =
        _mk_nttop_strip == 0 ? _mk_nttop_d0 : _mk_nttop_d1;
    const int _mk_nttop_row_off = _mk_nttop_m0 + _mk_nttop_strip * 128;
#pragma unroll
    for (int _mk_nttop_n8 = 0; _mk_nttop_n8 < 16; ++_mk_nttop_n8) {
      const int _mk_nttop_c = _mk_nttop_n8 * 8 + _mk_nttop_cb;
#pragma unroll
      for (int _mk_nttop_i = 0; _mk_nttop_i < 2; ++_mk_nttop_i) {
        const int _mk_nttop_r =
            _mk_nttop_wg * 64 + _mk_nttop_w * 16 + _mk_nttop_l / 4 + 8 * _mk_nttop_i;
        const int64_t _mk_nttop_idx =
            (int64_t)(_mk_nttop_row_off + _mk_nttop_r) * _mk_nttop_N +
            _mk_nttop_n0 + _mk_nttop_c;
        __nv_bfloat162 _mk_nttop_out;
        _mk_nttop_out.x = f2bf(_mk_nttop_d[_mk_nttop_n8 * 4 + _mk_nttop_i * 2 + 0]);
        _mk_nttop_out.y = f2bf(_mk_nttop_d[_mk_nttop_n8 * 4 + _mk_nttop_i * 2 + 1]);
        *reinterpret_cast<__nv_bfloat162*>(&_mk_nttop_C[_mk_nttop_idx]) =
            _mk_nttop_out;
      }
    }

    if (_mk_nttop_flags & 2048) {
      float* _mk_nttop_parts = reinterpret_cast<float*>(_mk_nttop_bufs[_mk_nttop_I.args[9]]);
      const int _mk_nttop_nparts = _mk_nttop_I.args[10];
      const int _mk_nttop_row_base =
          _mk_nttop_wg * 64 + _mk_nttop_w * 16 + _mk_nttop_l / 4;
#pragma unroll
      for (int _mk_nttop_i = 0; _mk_nttop_i < 2; ++_mk_nttop_i) {
        const int _mk_nttop_r = _mk_nttop_row_base + 8 * _mk_nttop_i;
#pragma unroll
        for (int _mk_nttop_half = 0; _mk_nttop_half < 2; ++_mk_nttop_half) {
          float _mk_nttop_mx = -INFINITY, _mk_nttop_se = 0.0f;
#pragma unroll
          for (int _mk_nttop_n8 = _mk_nttop_half * 8;
               _mk_nttop_n8 < _mk_nttop_half * 8 + 8; ++_mk_nttop_n8) {
#pragma unroll
            for (int _mk_nttop_j = 0; _mk_nttop_j < 2; ++_mk_nttop_j) {
              const float _mk_nttop_zv =
                  bf2f(f2bf(_mk_nttop_d[_mk_nttop_n8 * 4 +
                                         _mk_nttop_i * 2 + _mk_nttop_j]));
              if (_mk_nttop_zv > _mk_nttop_mx) {
                _mk_nttop_se =
                    _mk_nttop_se * lmhead_exp(_mk_nttop_mx - _mk_nttop_zv) + 1.0f;
                _mk_nttop_mx = _mk_nttop_zv;
              } else {
                _mk_nttop_se += lmhead_exp(_mk_nttop_zv - _mk_nttop_mx);
              }
            }
          }
#pragma unroll
          for (int _mk_nttop_o = 1; _mk_nttop_o < 4; _mk_nttop_o <<= 1) {
            const float _mk_nttop_om =
                __shfl_xor_sync(0xffffffff, _mk_nttop_mx, _mk_nttop_o);
            const float _mk_nttop_os =
                __shfl_xor_sync(0xffffffff, _mk_nttop_se, _mk_nttop_o);
            const float _mk_nttop_Mx = fmaxf(_mk_nttop_mx, _mk_nttop_om);
            _mk_nttop_se =
                (_mk_nttop_mx == -INFINITY && _mk_nttop_om == -INFINITY)
                    ? 0.0f
                    : _mk_nttop_se * lmhead_exp(_mk_nttop_mx - _mk_nttop_Mx) +
                          _mk_nttop_os * lmhead_exp(_mk_nttop_om - _mk_nttop_Mx);
            _mk_nttop_mx = _mk_nttop_Mx;
          }
          const int _mk_nttop_part = _mk_nttop_n0 / 64 + _mk_nttop_half;
          if ((_mk_nttop_l & 3) == 0 && _mk_nttop_part < _mk_nttop_nparts) {
            const int64_t _mk_nttop_po =
                ((int64_t)(_mk_nttop_row_off + _mk_nttop_r) * _mk_nttop_nparts +
                 _mk_nttop_part) * 2;
            _mk_nttop_parts[_mk_nttop_po] = _mk_nttop_mx;
            _mk_nttop_parts[_mk_nttop_po + 1] = _mk_nttop_se;
          }
        }
      }
    }
  }
}
