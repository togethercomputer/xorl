// Fused training megakernel — persistent cooperative kernel + in-kernel interpreter.
//
// One kernel executes an entire training fwd+bwd as an instruction stream. Blocks are
// persistent (one per SM); instructions are grouped into dependency-free "waves";
// grid.sync() separates waves; within a wave blocks self-schedule (instr, tile) work
// items. Buffers are referenced through a pointer table so instructions are plain ints.
//
// See NOTES.md for the design; ops.cuh for the device op library.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdlib>
#include <cstring>

namespace cg = cooperative_groups;

#define MK_MAX_ARGS 23
struct Instr {
  int op;
  int tile_off;  // first work-item id of this instr within its wave
  int ntiles;    // work items (block-level tiles) of this instr
  int args[MK_MAX_ARGS];
};
static_assert(sizeof(Instr) == 104, "keep Instr layout in sync with mk.py");

// ---- op enum (mirrored in mk.py; keep in sync) -------------------------------------
enum Op : int {
  OP_NOP = 0,
  OP_FILL_F32 = 1,
  OP_AXPY_F32 = 2,
  OP_GEMM = 3,
  OP_RMSNORM_FWD = 4,
  OP_RMSNORM_BWD = 5,
  OP_SWIGLU_FWD = 6,
  OP_SWIGLU_BWD = 7,
  OP_QKNORM_ROPE_FWD = 8,
  OP_QKNORM_ROPE_BWD = 9,
  OP_EMBED_FWD = 10,
  OP_EMBED_BWD = 11,
  OP_CE_FWD = 12,
  OP_CE_BWD = 13,
  OP_ATTN_FWD = 14,
  OP_ATTN_DPRE = 15,
  OP_ATTN_DKV = 16,
  OP_ATTN_DQ = 17,
  OP_CVT_F32BF16 = 18,
  OP_ATTN_FWD_SPLIT = 19,
  OP_ATTN_COMBINE = 20,
  OP_ATTN_FWD_WG = 21,
  OP_ATTN_DKV_WG = 22,
  OP_ATTN_DQ_WG = 23,
  OP_RMSNORM_BWD_DX = 24,
  OP_RMSNORM_BWD_DW = 25,
  OP_RMSNORM_BWD_DX_R4 = 26,
  OP_INV_VALID = 27,
  OP_RMSNORM_BWD_DX_FMA = 28,
  OP_SWIGLU_BWD_2W = 29,
  OP_QKV_V_BWD = 30,
  OP_RMSNORM_BWD_DX_H256 = 31,
  OP_ATTN_FWD_WG128 = 32,
  OP_ATTN_DKV_WG128 = 33,
  OP_ATTN_DQ_WG128 = 34,
  OP_EMBED_ZERO_ROWS = 35,
  OP_COPY_I32 = 36,
  OP_SWIGLU_BWD_4W = 37,
  OP_SKR_REDUCE = 38,
};

__device__ __forceinline__ long long mk_globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;  // globally-synchronized ns counter (clock64 is per-SM: incomparable stamps)
}

#include "ops.cuh"
#include "attention.cuh"
#include "wgmma_attention.cuh"

// MK_OCC2 (mk.py MK_OCC2=1): cap the 256-thread executors at 128 regs so 2 blocks
// co-reside per SM. The P4b nsys counters showed the 1-block interpreter is
// latency-bound (SM issue 19%, 8 warps in flight, DRAM <10%): doubling resident
// warps buys latency-hiding at the price of ptxas spilling the fat op paths.
#ifdef MK_OCC2
#define MK_LB __launch_bounds__(256, 2)
#else
#define MK_LB
#endif

// idle ready-ring poll cadence (ns). 256 was tuned pre-P4b when spans dominated;
// with waits now ~20% of nano, faster discovery may pay. MK_IDLE_NS build knob.
#ifndef MK_IDLE_NS
#define MK_IDLE_NS 256
#endif

// ---- interpreter --------------------------------------------------------------------
__device__ __forceinline__ void dispatch(const Instr& I, int tile, void** bufs,
                                         char* smem) {
  switch (I.op) {
    case OP_NOP:
      break;
    case OP_FILL_F32:
      op_fill_f32(I, tile, bufs);
      break;
    case OP_AXPY_F32:
      op_axpy_f32(I, tile, bufs);
      break;
    case OP_GEMM:
      if (I.args[6] & 128)
        op_gemm_wgmma(I, tile, bufs, smem);
      else
        op_gemm(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_FWD:
      op_rmsnorm_fwd(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD:
      op_rmsnorm_bwd(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD_DX:
      op_rmsnorm_bwd_dx(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD_DX_FMA:
      op_rmsnorm_bwd_dx_fma(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD_DX_H256:
      op_rmsnorm_bwd_dx_h256(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD_DW:
      op_rmsnorm_bwd_dw(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD_DX_R4:
      op_rmsnorm_bwd_dx_r4(I, tile, bufs, smem);
      break;
    case OP_SWIGLU_FWD:
      op_swiglu_fwd(I, tile, bufs);
      break;
    case OP_SWIGLU_BWD:
      op_swiglu_bwd(I, tile, bufs);
      break;
    case OP_SWIGLU_BWD_2W:
#ifdef MK_SWIGLU_BWD_2W
      op_swiglu_bwd_2w(I, tile, bufs);
#else
      asm volatile("trap;");
#endif
      break;
    case OP_SWIGLU_BWD_4W:
#ifdef MK_SWIGLU_BWD_4W
      op_swiglu_bwd_4w(I, tile, bufs);
#else
      asm volatile("trap;");
#endif
      break;
    case OP_SKR_REDUCE:
#ifdef MK_HEAD_DX_SKR
      op_skr_reduce(I, tile, bufs);
#else
      asm volatile("trap;");
#endif
      break;
    case OP_QKNORM_ROPE_FWD:
      op_qknorm_rope_fwd(I, tile, bufs, smem);
      break;
    case OP_QKNORM_ROPE_BWD:
      op_qknorm_rope_bwd(I, tile, bufs, smem);
      break;
    case OP_QKV_V_BWD:
      op_qkv_v_bwd(I, tile, bufs);
      break;
    case OP_EMBED_FWD:
      op_embed_fwd(I, tile, bufs);
      break;
    case OP_EMBED_BWD:
      op_embed_bwd(I, tile, bufs);
      break;
    case OP_EMBED_ZERO_ROWS:
      op_embed_zero_rows(I, tile, bufs);
      break;
    case OP_COPY_I32:
      op_copy_i32(I, tile, bufs);
      break;
    case OP_INV_VALID:
      op_inv_valid(I, tile, bufs, smem);
      break;
    case OP_CE_FWD:
      op_ce_fwd(I, tile, bufs, smem);
      break;
    case OP_CE_BWD:
      op_ce_bwd(I, tile, bufs);
      break;
    case OP_ATTN_FWD:
      op_attn_fwd(I, tile, bufs, smem);
      break;
    case OP_ATTN_DPRE:
      op_attn_dpre(I, tile, bufs);
      break;
    case OP_ATTN_DKV:
      op_attn_dkv(I, tile, bufs, smem);
      break;
    case OP_ATTN_DQ:
      op_attn_dq(I, tile, bufs, smem);
      break;
    case OP_CVT_F32BF16:
      op_cvt_f32_bf16(I, tile, bufs);
      break;
    case OP_ATTN_FWD_SPLIT:
      op_attn_fwd_split(I, tile, bufs, smem);
      break;
    case OP_ATTN_COMBINE:
      op_attn_combine(I, tile, bufs);
      break;
    case OP_ATTN_FWD_WG:
      op_attn_fwd_wg(I, tile, bufs, smem);
      break;
    case OP_ATTN_FWD_WG128:
      op_attn_fwd_wg128(I, tile, bufs, smem);
      break;
    case OP_ATTN_DKV_WG128:
      op_attn_dkv_wg128(I, tile, bufs, smem);
      break;
    case OP_ATTN_DQ_WG128:
      op_attn_dq_wg128(I, tile, bufs, smem);
      break;
    case OP_ATTN_DKV_WG:
      op_attn_dkv_wg(I, tile, bufs, smem);
      break;
    case OP_ATTN_DQ_WG:
      op_attn_dq_wg(I, tile, bufs, smem);
      break;
    default:
      break;
  }
}

extern "C" __global__ void megakernel(const Instr* __restrict__ instrs,
                                      const int* __restrict__ wave_start,  // [nwaves+1]
                                      const int* __restrict__ wave_tiles,  // [nwaves]
                                      int nwaves, void** bufs,
                                      long long* wave_clk /* nullable [nwaves+1] */) {
  extern __shared__ char smem[];
  __shared__ Instr s_I;  // owner instr staged in smem (dispatch spill tax)
  cg::grid_group grid = cg::this_grid();
  const int nblocks = gridDim.x;

#if defined(MK_PDF_PRODUCER) && defined(MK_ATTN_PDF_FEED)
  // The attention bodies read g_pdf_feed.active; smem is not zeroed across
  // launches, so every executor image must clear it (ordered for the block by
  // the pre-dispatch __syncthreads).
  if (threadIdx.x == 0) g_pdf_feed.active = 0;
#endif
  if (wave_clk && blockIdx.x == 0 && threadIdx.x == 0) wave_clk[0] = clock64();
  for (int w = 0; w < nwaves; ++w) {
    const int i0 = wave_start[w], i1 = wave_start[w + 1];
    const int total = wave_tiles[w];
    for (int work = blockIdx.x; work < total; work += nblocks) {
      // locate the instruction owning this work item: scan offsets only (2 ints),
      // stage the full Instr in smem just for the owner (every thread scans the same
      // globals, so the branch is block-uniform).
      for (int i = i0; i < i1; ++i) {
        if (work < instrs[i].tile_off + instrs[i].ntiles) {
          if (threadIdx.x < 3 + MK_MAX_ARGS)
            reinterpret_cast<int*>(&s_I)[threadIdx.x] =
                reinterpret_cast<const int*>(instrs + i)[threadIdx.x];
          __syncthreads();
          dispatch(s_I, work - s_I.tile_off, bufs, smem);
          break;
        }
      }
      __syncthreads();  // smem reuse safety between work items within a wave
    }
    grid.sync();
    if (wave_clk && blockIdx.x == 0 && threadIdx.x == 0) wave_clk[w + 1] = clock64();
  }
}

// ---- dataflow executor ----------------------------------------------------------------
// No waves: each instruction has a dependency count; finished instructions decrement
// their dependents' counts; instructions with count 0 enter a ready ring. Blocks claim
// (instr, tile-range) work items via per-instruction atomic cursors. Tiles of
// independent instructions (e.g. the backward dX chain and the dW GEMMs) co-execute.
//
// state layout: pending[n] | cursor[n] | done[n] | ready[n] | ctrl[4]
//   ctrl[0]=ready tail, ctrl[1]=finished instr count.
// MK_DF_MAXNREG (build knob): entry register ceiling for the 240/24 producer-df
// study — phase 1 measures the consumer-ceiling tax alone (no producer WG).
// MK_DF_PRODUCER (build knob, step A executor shell): 384-thread megakernel_df.
// WG0+WG1 (threads 0-255) run the df body unchanged at setmaxnreg.inc 240 (same
// consumer codegen budget as MK_DF_MAXNREG=240); WG2 (threads 256-383) decs to 24
// and PARKS on ctrl[1] (finished-instr count) until the run completes — step B
// makes it the TMA mailbox producer. Entry __maxnreg__(168): any block >256
// threads is charged 12 warps -> 65536/384 hard ceiling (see the megakernel_ws
// comment block), and ptxas ignores setmaxnreg without an entry maxnreg (C7508).
// Balance: 256*240 + 128*24 = 64512 <= 65536. Under this variant the executor
// loop must sync exactly the 256 consumers (WG2 never reaches the loop): the
// ops' named barrier (bar.sync 1, 256 == consumer_sync for threads 0-255)
// replaces __syncthreads, which would count all 384 threads and hang.
// Round 2: the split sits at KERNEL ENTRY, before any shared code — the round-1
// shell split after init and every consumer instruction (init, claim loop, the
// whole op library) was allocated at the 168 entry ceiling (STACK 112 vs plain
// 48; measured +156.8us small / +45.8us nano on GPU 3, 14x the flat-240 ceiling
// tax). With the split first, the consumer path is dominated by the inc-240 so
// ptxas allocates it at 240; the WG2 path (2 grid.syncs + a relaxed poll) must
// fit its dec-24 budget. The paths NEVER reconverge (the CUTLASS/ws rule):
// WG2 pairs with the consumers' two init grid.syncs from its own call sites
// (cg grid.sync's internal CTA barrier is call-site agnostic), so init work
// loops are MK_CONSUMERS-wide (identical arithmetic for all 256-thread builds).
#if defined(MK_DF_PRODUCER) && defined(MK_DF_MAXNREG)
#error "MK_DF_PRODUCER and MK_DF_MAXNREG are mutually exclusive (producer sets entry maxnreg 168)"
#endif
#if defined(MK_DF_PRODUCER) && defined(MK_OCC2)
#error "MK_DF_PRODUCER and MK_OCC2 are mutually exclusive (384 threads vs __launch_bounds__(256,2))"
#endif
// MK_DF_NAMED_BAR (attribution knob): plain 256-thread df with the four
// executor-loop __syncthreads replaced by the producer variant's bar.sync 1,256
// (consumer_sync) — isolates the named-barrier cost from the producer topology
// (WG2 residency, setmaxnreg partition, entry ceiling). Redundant under
// MK_DF_PRODUCER, which already syncs on the named barrier.
#if defined(MK_DF_NAMED_BAR) && defined(MK_DF_PRODUCER)
#error "MK_DF_NAMED_BAR is redundant under MK_DF_PRODUCER (it already uses bar.sync 1,256)"
#endif
#ifdef MK_DF_PRODUCER
#define MK_DF_NR __maxnreg__(168)
#define MK_DF_THREADS 384
#define mk_df_sync() consumer_sync()
#else
#ifdef MK_DF_MAXNREG
#define MK_DF_NR __maxnreg__(MK_DF_MAXNREG)
#else
#define MK_DF_NR
#endif
#define MK_DF_THREADS 256
#ifdef MK_DF_NAMED_BAR
#define mk_df_sync() consumer_sync()
#else
#define mk_df_sync() __syncthreads()
#endif
#endif
extern "C" __global__ void MK_LB MK_DF_NR megakernel_df(const Instr* __restrict__ instrs, int n_instr,
                                         const int* __restrict__ dep_cnt,
                                         const int* __restrict__ adj_off,
                                         const int* __restrict__ adj,
                                         const int* __restrict__ claim_sz,
                                         const int* __restrict__ crit, int cold_cap,
                                         int release_at, int* state, void** bufs,
                                         long long* iclk /* nullable [2*n] */,
                                         int bind0, unsigned long long ptr0,
                                         int bind1, unsigned long long ptr1) {
  extern __shared__ char smem[];
  cg::grid_group grid = cg::this_grid();
#ifdef MK_DF_PRODUCER
  // Entry warpgroup split (see the knob comment above). The branch is
  // warpgroup-uniform, so each setmaxnreg is converged across its warpgroup as
  // .sync.aligned requires; WG2's dec is unconditional and first, so the
  // consumers' inc (which blocks until registers are free) cannot deadlock.
  // WG2 skips the init WORK (the loops below are MK_CONSUMERS-wide) but calls
  // grid.sync() twice to pair with the consumers' init grid.syncs, then parks:
  // ld.relaxed.gpu on ctrl[1] every 8192ns keeps it off the L2 hot path. The
  // poll starts only after the second grid.sync — before that, ctrl[1] still
  // holds n_instr from the PREVIOUS launch (state is reused across steps) and
  // an early read would exit WG2 into step B's producer role prematurely.
  if (threadIdx.x >= MK_CONSUMERS) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    const int* fin = state + 5 * n_instr + 1;  // ctrl[1] = finished instr count
    grid.sync();
    grid.sync();
    for (;;) {
      int f;
      asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(f) : "l"(fin) : "memory");
      if (f >= n_instr) break;
      __nanosleep(8192);
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");  // blocks until WG2's dec frees regs
#endif
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (bind0 >= 0) bufs[bind0] = reinterpret_cast<void*>(ptr0);
    if (bind1 >= 0) bufs[bind1] = reinterpret_cast<void*>(ptr1);
  }
#ifdef MK_PDF_PRODUCER
  if (threadIdx.x == 0) g_pdf_feed.active = 0;  // smem is not zeroed across launches
#endif
  int* pending = state;
  int* cursor = state + n_instr;
  int* done = state + 2 * n_instr;
  // two-class ready rings (v3 P6, the Whippletree/MPK lesson): HOT = instrs something
  // depends on (the chain); COLD = sinks (dW gemms, embed_bwd) + wave-0 fills. Idle
  // blocks drain hot first, so chain consumers start within ~a claim batch of their
  // producer finishing instead of behind hundreds of sticky off-path tile claims.
  int* ready_hot = state + 3 * n_instr;
  int* ready_cold = state + 4 * n_instr;
  int* ctrl = state + 5 * n_instr;

  // init, then one sync, then seed roots (visible before anyone consumes: the tail
  // increment in the seeding phase publishes them). MK_CONSUMERS-wide strides,
  // NOT blockDim.x: under MK_DF_PRODUCER only the 256 consumers run these loops
  // (WG2 split off above); for every other df build blockDim.x == MK_CONSUMERS.
  for (int i = blockIdx.x * MK_CONSUMERS + threadIdx.x; i < n_instr;
       i += gridDim.x * MK_CONSUMERS) {
    pending[i] = dep_cnt[i];
    cursor[i] = 0;
    done[i] = 0;
    ready_hot[i] = -1;
    ready_cold[i] = -1;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ctrl[0] = 0;  // hot ready tail
    ctrl[1] = 0;  // finished count
    ctrl[2] = 0;  // hot consumed head
    ctrl[3] = 0;  // cold ready tail
    ctrl[4] = 0;  // cold consumed head
    ctrl[5] = 0;  // blocks currently working cold entries (cold_cap limiter)
  }
  grid.sync();
  for (int i = blockIdx.x * MK_CONSUMERS + threadIdx.x; i < n_instr;
       i += gridDim.x * MK_CONSUMERS) {
    if (dep_cnt[i] == 0) {
      if (crit[i]) {
        const int t = atomicAdd(&ctrl[0], 1);
        atomicExch(&ready_hot[t], i);
      } else {
        const int t = atomicAdd(&ctrl[3], 1);
        atomicExch(&ready_cold[t], i);
      }
    }
  }
  grid.sync();

  __shared__ int s_ins, s_t0, s_t1;
  __shared__ Instr s_I;  // claimed instr staged in smem (see the dispatch-loop note)
  int last_ins = -1;   // sticky: retry the previous instruction before scanning
  int last_cold = 0;   // sticky instr's class: cold stickiness yields to fresh hot work
  int seen_hot = 0;    // hot tail at the last failed hot scan (held back at invisible
                       // slots so a not-yet-published push still forces a rescan)
  volatile int* vhot = ready_hot;
  volatile int* vcold = ready_cold;
  volatile int* vctrl = ctrl;
  for (;;) {
    if (threadIdx.x == 0) {
      s_ins = -1;
      // fast path: most blocks keep pulling tiles from the same big instruction —
      // unless it is cold work and the hot ring has news
      if (last_ins >= 0 && !(last_cold && vctrl[0] != seen_hot)) {
        const int nt = instrs[last_ins].ntiles;
        if (cursor[last_ins] < nt) {
          const int bs = claim_sz[last_ins];
          const int t0 = atomicAdd(&cursor[last_ins], bs);
          if (t0 < nt) {
            s_ins = last_ins;
            s_t0 = t0;
            s_t1 = min(t0 + bs, nt);
          }
        }
      }
      while (s_ins < 0 && vctrl[1] < n_instr) {
        // hot ring first; ctrl[2] = consumed head (entries below are fully claimed)
        const int htail = vctrl[0];
        int hvis = htail;
        for (int q = vctrl[2]; q < htail; ++q) {
          const int ins = vhot[q];
          if (ins < 0) {  // slot reserved, payload not yet visible
            if (q < hvis) hvis = q;
            continue;
          }
          const int nt = instrs[ins].ntiles;
          if (cursor[ins] >= nt) {
            if (q == vctrl[2]) atomicCAS(&ctrl[2], q, q + 1);  // advance consumed head
            continue;
          }
          const int bs = claim_sz[ins];
          const int t0 = atomicAdd(&cursor[ins], bs);
          if (t0 < nt) {
            s_ins = ins;
            s_t0 = t0;
            s_t1 = min(t0 + bs, nt);
            if (last_cold) atomicSub(&ctrl[5], 1);
            last_cold = 0;
            break;
          }
        }
        if (s_ins >= 0) break;
        seen_hot = hvis;
        // cold_cap bounds how many blocks work cold entries concurrently (bandwidth
        // headroom for the chain); a block already on cold keeps its slot — the cap
        // must not lock out every worker or capped cold work could never finish.
        // phase-adaptive cold-cap release (f30b8c0c): once the hot consumed
        // head crosses release_at (host: n_hot - MK_COLD_RELEASE_TAIL), the
        // chain is nearly drained and the cap lifts so sink work (dW, the
        // EMBED_BWD join feeders) finishes with the freed blocks. ctrl[5]
        // accounting is admission-only, so exceeding the cap after release
        // is benign. release_at = INT_MAX when the knob is off.
        if (last_cold || vctrl[5] < cold_cap || vctrl[2] >= release_at) {
          const int ctail = vctrl[3];
          for (int q = vctrl[4]; q < ctail; ++q) {
            const int ins = vcold[q];
            if (ins < 0) continue;
            const int nt = instrs[ins].ntiles;
            if (cursor[ins] >= nt) {
              if (q == vctrl[4]) atomicCAS(&ctrl[4], q, q + 1);
              continue;
            }
            const int bs = claim_sz[ins];
            const int t0 = atomicAdd(&cursor[ins], bs);
            if (t0 < nt) {
              s_ins = ins;
              s_t0 = t0;
              s_t1 = min(t0 + bs, nt);
              if (!last_cold) atomicAdd(&ctrl[5], 1);
              last_cold = 1;
              break;
            }
          }
        }
        if (s_ins >= 0) break;
        __nanosleep(MK_IDLE_NS);
      }
      if (s_ins < 0 && last_cold) atomicSub(&ctrl[5], 1);  // exiting: release the slot
      last_ins = s_ins;
    }
    mk_df_sync();
    const int ins = s_ins;
    if (ins < 0) break;  // everything finished
    const int t0 = s_t0, t1 = s_t1;
    // stage the claimed Instr in static smem: a register/stack copy of the 104-byte
    // struct is live across every dispatch call site — exactly where the P5/P6
    // spill-tax STL/LDL sit (ptxas spills caller state around the inlined switch)
    if (threadIdx.x < 3 + MK_MAX_ARGS)
      reinterpret_cast<int*>(&s_I)[threadIdx.x] =
          reinterpret_cast<const int*>(instrs + ins)[threadIdx.x];
    mk_df_sync();
    if (iclk && threadIdx.x == 0 && t0 == 0) iclk[2 * ins] = mk_globaltimer();
    for (int t = t0; t < t1; ++t) {
      dispatch(s_I, t, bufs, smem);
      mk_df_sync();
    }
    if (threadIdx.x == 0) {
      __threadfence();  // publish this instr's writes before enabling dependents
      const int d = atomicAdd(&done[ins], t1 - t0) + (t1 - t0);
      if (d == s_I.ntiles) {  // last tile: enable dependents
        if (iclk) iclk[2 * ins + 1] = mk_globaltimer();
        int hint = -1;  // completion hint (v3 P4b, the ws lesson): the finisher
        // adopts a hot dependent it just enabled as its own sticky claim — the
        // chain's next hop skips ring rediscovery on a warm block.
        for (int e = adj_off[ins]; e < adj_off[ins + 1]; ++e) {
          const int dep = adj[e];
          if (atomicSub(&pending[dep], 1) == 1) {
            if (crit[dep]) {
              const int t = atomicAdd(&ctrl[0], 1);
              atomicExch(&ready_hot[t], dep);
              if (hint < 0) hint = dep;
            } else {
              const int t = atomicAdd(&ctrl[3], 1);
              atomicExch(&ready_cold[t], dep);
            }
          }
        }
        atomicAdd(&ctrl[1], 1);
        if (hint >= 0) {
          if (last_cold) atomicSub(&ctrl[5], 1);  // leaving a cold sticky: free the slot
          last_ins = hint;
          last_cold = 0;
        }
      }
    }
    mk_df_sync();
  }
}

// ---- producer-df executor (MK_PDF): the register-point probe --------------------------
// The df protocol EXACTLY (claim loop, hot/cold rings, sticky claims, completion hints —
// keep this body in lockstep with megakernel_df above), launched at 384 threads with the
// ws register plumbing: entry __maxnreg__(168) (the 12-warp charge ceiling), consumer
// warpgroups 0-1 setmaxnreg.inc -> MK_PDF_REGS (default 240; 8*MK_PDF_REGS + 4*24 must
// stay <= 2048 warp-regs — 240/24 is the exact-balance point: 128*(168-24) ==
// 256*(240-168)), warpgroup 2 setmaxnreg.dec -> 24 and exits after init (PHASE 1: this
// isolates the pure register-point tax vs df's 255 regs at an otherwise identical
// protocol; the ws-vs-df comparison could never do that because ws also changes the
// scheduling protocol). PHASE 2 keeps WG2 resident as a pure TMA producer for GEMM
// mbarrier-ring rows (the GEMM round-5 escape, in-model). Block-wide __syncthreads is
// replaced by consumer_sync() — WG2 does not participate below the grid.syncs (the ws
// invariant; ops already sync on bar.sync 1,256).
#ifdef MK_PDF
#ifndef MK_PDF_REGS
#define MK_PDF_REGS 240
#endif
#define MK_PDF_THREADS 384

extern "C" __global__ void __maxnreg__(168) megakernel_pdf(
    const Instr* __restrict__ instrs, int n_instr, const int* __restrict__ dep_cnt,
    const int* __restrict__ adj_off, const int* __restrict__ adj,
    const int* __restrict__ claim_sz, const int* __restrict__ crit, int cold_cap,
    int release_at, int* state, void** bufs, long long* iclk /* nullable [2*n] */, int bind0,
    unsigned long long ptr0, int bind1, unsigned long long ptr1) {
  extern __shared__ char smem[];
  cg::grid_group grid = cg::this_grid();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (bind0 >= 0) bufs[bind0] = reinterpret_cast<void*>(ptr0);
    if (bind1 >= 0) bufs[bind1] = reinterpret_cast<void*>(ptr1);
  }
#ifdef MK_PDF_PRODUCER
  if (threadIdx.x == 0) {
    g_pdf_feed.active = 1;  // ordered for the whole block by the init grid.sync
    g_pdf_feed.seq = 0;
    g_pdf_feed.halt = 0;
  }
#endif
  int* pending = state;
  int* cursor = state + n_instr;
  int* done = state + 2 * n_instr;
  int* ready_hot = state + 3 * n_instr;
  int* ready_cold = state + 4 * n_instr;
  int* ctrl = state + 5 * n_instr;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    pending[i] = dep_cnt[i];
    cursor[i] = 0;
    done[i] = 0;
    ready_hot[i] = -1;
    ready_cold[i] = -1;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ctrl[0] = 0;
    ctrl[1] = 0;
    ctrl[2] = 0;
    ctrl[3] = 0;
    ctrl[4] = 0;
    ctrl[5] = 0;
  }
  grid.sync();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    if (dep_cnt[i] == 0) {
      if (crit[i]) {
        const int t = atomicAdd(&ctrl[0], 1);
        atomicExch(&ready_hot[t], i);
      } else {
        const int t = atomicAdd(&ctrl[3], 1);
        atomicExch(&ready_cold[t], i);
      }
    }
  }
  grid.sync();

  // -------- specialization (the ws register dance; no full-block barrier below) --------
  // STRUCTURE MATTERS: ptxas honors the inc region ONLY in the ws idiom — the inc'd
  // consumer body self-contained inside the taken branch, ending in return, with the
  // dec path on the fallthrough. With the inc on the fallthrough (dec branch first),
  // ptxas silently keeps the whole kernel at the 168 entry cap (max SASS reg R165,
  // STACK:208) for EVERY inc value 224/232/240 — measured on CUDA 13.1.
  __shared__ int s_ins, s_t0, s_t1;
  __shared__ Instr s_I;
  if (threadIdx.x < MK_CONSUMERS) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(MK_PDF_REGS));

  int last_ins = -1;
  int last_cold = 0;
  int seen_hot = 0;
  volatile int* vhot = ready_hot;
  volatile int* vcold = ready_cold;
  volatile int* vctrl = ctrl;
  for (;;) {
    if (threadIdx.x == 0) {
      s_ins = -1;
      if (last_ins >= 0 && !(last_cold && vctrl[0] != seen_hot)) {
        const int nt = instrs[last_ins].ntiles;
        if (cursor[last_ins] < nt) {
          const int bs = claim_sz[last_ins];
          const int t0 = atomicAdd(&cursor[last_ins], bs);
          if (t0 < nt) {
            s_ins = last_ins;
            s_t0 = t0;
            s_t1 = min(t0 + bs, nt);
          }
        }
      }
      while (s_ins < 0 && vctrl[1] < n_instr) {
        const int htail = vctrl[0];
        int hvis = htail;
        for (int q = vctrl[2]; q < htail; ++q) {
          const int ins = vhot[q];
          if (ins < 0) {
            if (q < hvis) hvis = q;
            continue;
          }
          const int nt = instrs[ins].ntiles;
          if (cursor[ins] >= nt) {
            if (q == vctrl[2]) atomicCAS(&ctrl[2], q, q + 1);
            continue;
          }
          const int bs = claim_sz[ins];
          const int t0 = atomicAdd(&cursor[ins], bs);
          if (t0 < nt) {
            s_ins = ins;
            s_t0 = t0;
            s_t1 = min(t0 + bs, nt);
            if (last_cold) atomicSub(&ctrl[5], 1);
            last_cold = 0;
            break;
          }
        }
        if (s_ins >= 0) break;
        seen_hot = hvis;
        // phase-adaptive cold-cap release (f30b8c0c): once the hot consumed
        // head crosses release_at (host: n_hot - MK_COLD_RELEASE_TAIL), the
        // chain is nearly drained and the cap lifts so sink work (dW, the
        // EMBED_BWD join feeders) finishes with the freed blocks. ctrl[5]
        // accounting is admission-only, so exceeding the cap after release
        // is benign. release_at = INT_MAX when the knob is off.
        if (last_cold || vctrl[5] < cold_cap || vctrl[2] >= release_at) {
          const int ctail = vctrl[3];
          for (int q = vctrl[4]; q < ctail; ++q) {
            const int ins = vcold[q];
            if (ins < 0) continue;
            const int nt = instrs[ins].ntiles;
            if (cursor[ins] >= nt) {
              if (q == vctrl[4]) atomicCAS(&ctrl[4], q, q + 1);
              continue;
            }
            const int bs = claim_sz[ins];
            const int t0 = atomicAdd(&cursor[ins], bs);
            if (t0 < nt) {
              s_ins = ins;
              s_t0 = t0;
              s_t1 = min(t0 + bs, nt);
              if (!last_cold) atomicAdd(&ctrl[5], 1);
              last_cold = 1;
              break;
            }
          }
        }
        if (s_ins >= 0) break;
        __nanosleep(MK_IDLE_NS);
      }
      if (s_ins < 0 && last_cold) atomicSub(&ctrl[5], 1);
      last_ins = s_ins;
    }
    consumer_sync();
#ifdef MK_PDF_SHELL_SMEM
    // Local-spill fix for the 240-reg consumer region (pdfld lane, claim
    // 20260708T0140Z-realclock): the fat noinline op bodies clobber ~the whole
    // 240-reg pool at the dispatch call, so hoisting ins/t0/t1 into registers
    // once per claim (the df idiom, fine at 255 regs where R249+ survive every
    // call) makes ptxas park them in LOCAL and reload per tile — measured
    // 4.83M local-LD + 6.49M local-ST sectors/launch at s2048 default-pdf,
    // 63% on these shell lines (df image at the same shape: 198K/2.3M).
    // Re-reading s_ins/s_t0/s_t1 from SHARED across the call turns each
    // rehydrate into an LDS (the call is a smem barrier, so the loads stay in
    // the loop by construction). Writes to s_* can't race: thread 0 only
    // rewrites them in the claim phase, which every thread's tile loop + the
    // completion consumer_sync separate from these reads. The loop counter t
    // stays a REGISTER on purpose: a shared tile cursor advanced by thread 0
    // after dispatch races with other threads' loop-top reads inside the same
    // barrier interval (tile skip / barrier desync), and the race-free
    // variant costs a second consumer_sync per tile — t's remaining local
    // round-trip (~2.4M sectors/launch at s2048) is the documented residual.
    // __noinline__ on the inlined row/CE ops was tried first and REFUTED
    // (LD 4.83M -> 6.32M): the traffic is the ABI boundary itself, not
    // inline pressure.
    if (s_ins < 0) break;
    if (threadIdx.x < 3 + MK_MAX_ARGS)
      reinterpret_cast<int*>(&s_I)[threadIdx.x] =
          reinterpret_cast<const int*>(instrs + s_ins)[threadIdx.x];
    consumer_sync();
    if (iclk && threadIdx.x == 0 && s_t0 == 0) iclk[2 * s_ins] = mk_globaltimer();
    for (int t = s_t0; t < s_t1; ++t) {
      dispatch(s_I, t, bufs, smem);
      consumer_sync();
    }
    if (threadIdx.x == 0) {
      __threadfence();
      const int ins = s_ins;
      const int nt_done = s_t1 - s_t0;
      const int d = atomicAdd(&done[ins], nt_done) + nt_done;
      if (d == s_I.ntiles) {
#else
    const int ins = s_ins;
    if (ins < 0) break;
    const int t0 = s_t0, t1 = s_t1;
    if (threadIdx.x < 3 + MK_MAX_ARGS)
      reinterpret_cast<int*>(&s_I)[threadIdx.x] =
          reinterpret_cast<const int*>(instrs + ins)[threadIdx.x];
    consumer_sync();
    if (iclk && threadIdx.x == 0 && t0 == 0) iclk[2 * ins] = mk_globaltimer();
    for (int t = t0; t < t1; ++t) {
      dispatch(s_I, t, bufs, smem);
      consumer_sync();
    }
    if (threadIdx.x == 0) {
      __threadfence();
      const int d = atomicAdd(&done[ins], t1 - t0) + (t1 - t0);
      if (d == s_I.ntiles) {
#endif  // MK_PDF_SHELL_SMEM
        if (iclk) iclk[2 * ins + 1] = mk_globaltimer();
        int hint = -1;
        for (int e = adj_off[ins]; e < adj_off[ins + 1]; ++e) {
          const int dep = adj[e];
          if (atomicSub(&pending[dep], 1) == 1) {
            if (crit[dep]) {
              const int t = atomicAdd(&ctrl[0], 1);
              atomicExch(&ready_hot[t], dep);
              if (hint < 0) hint = dep;
            } else {
              const int t = atomicAdd(&ctrl[3], 1);
              atomicExch(&ready_cold[t], dep);
            }
          }
        }
        atomicAdd(&ctrl[1], 1);
        if (hint >= 0) {
          if (last_cold) atomicSub(&ctrl[5], 1);
          last_ins = hint;
          last_cold = 0;
        }
      }
    }
    consumer_sync();
  }
#ifdef MK_PDF_PRODUCER
  if (threadIdx.x == 0) mk_pdf_st_release(&g_pdf_feed.halt, 1);
#endif
  return;
  }  // threadIdx.x < MK_CONSUMERS

#ifndef MK_PDF_DEC
#define MK_PDF_DEC 24
#endif
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(MK_PDF_DEC));
#ifdef MK_PDF_FEED
  // -------- WG2 producer (thread 256; 257-383 exit) --------------------------------
  // Pure TMA issuer at the dec register point: replays each posted tile's full stage
  // schedule, gated only by ring empties — never by the consumers' mma cadence. One
  // request is in flight at a time by construction (tile T+1 posts program-order
  // after tile T's last full-wait on the same consumer thread).
#ifdef MK_ATTN_PDF_FEED
  // Attention stream requests (kind 4) are served WARPGROUP-WIDE: each of the
  // 128 producer threads issues 4 cp.async 16B slices per operand tile with the
  // exact consumer-loader slice geometry (v = ptid + i*128 -> r += 16, c8
  // fixed, wga_off64 += 2048 per i), so the destination bytes are identical to
  // the consumer path (the layout-preserving Branch A of the feasibility note).
  // GEMM TMA replay (kinds 0-3) stays on the elected thread 256; threads
  // 257-383 skip those requests but track seq. The serial-request invariant
  // holds for kind 4 too: the consumer's last bfull wait needs all 128
  // producer-thread arrivals, so every producer thread finished issuing tile T
  // before tile T+1 posts.
  if (threadIdx.x >= MK_CONSUMERS) {
#else
  if (threadIdx.x == MK_CONSUMERS) {
#endif
    volatile MkPdfFeed* Fv = &g_pdf_feed;
    int seen = 0;
    for (;;) {
      const int s = mk_pdf_ld_acquire(const_cast<const int*>(&Fv->seq));
      if (s == seen) {
        if (Fv->halt) break;
        __nanosleep(64);
        continue;
      }
      ++seen;
#ifdef MK_ATTN_PDF_FEED
      if (Fv->kind == 4) {
        // Attention cp.async stream: operand GMEM row bases in tmA/tmB, gmem
        // row strides (bytes) in m0/n0, per-stage gmem byte steps in
        // k_base/bk; dst wga_off64 slabs a0/b0 with a_stride/b_stride
        // per-stage slab strides (see the MkPdfFeed kind-4 comment).
        const char* gA = Fv->tmA;
        const char* gB = Fv->tmB;
        char* a0 = Fv->a0;
        char* b0 = Fv->b0;
        uint64_t* bfull = Fv->bfull;
        uint64_t* bempty = Fv->bempty;
        const int a_st = Fv->a_stride, b_st = Fv->b_stride;
        const int a_gs = Fv->m0, b_gs = Fv->n0;
        const int stepA = Fv->k_base, stepB = Fv->bk;
        const int iters = Fv->iters, stages = Fv->stages;
        const int ptid = threadIdx.x - MK_CONSUMERS;
        const int r = ((ptid >> 6) << 3) | (ptid & 7);
        const int c8 = ((ptid >> 3) & 7) << 3;
        const int doff = wga_off64(r, c8);
        gA += (int64_t)r * a_gs + c8 * 2;
        gB += (int64_t)r * b_gs + c8 * 2;
        for (int t = 0; t < iters; ++t) {
          const int st = t % stages;
          if (t >= stages) wg_mbar_wait(&bempty[st], (t / stages - 1) & 1);
          char* dA = a0 + st * a_st + doff;
          char* dB = b0 + st * b_st + doff;
#pragma unroll
          for (int i = 0; i < 4; ++i)
            __pipeline_memcpy_async(dA + i * 2048, gA + i * (a_gs << 4), 16);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            __pipeline_memcpy_async(dB + i * 2048, gB + i * (b_gs << 4), 16);
          wg_mbar_arrive_cpasync(&bfull[st]);
          gA += stepA;
          gB += stepB;
        }
        continue;
      }
      if (threadIdx.x != MK_CONSUMERS) continue;
#endif
#ifdef MK_PDF_GEMM_FEED
      const char* tmA = Fv->tmA;
      const char* tmB = Fv->tmB;
      char* a0 = Fv->a0;
      char* a1 = Fv->a1;
      char* b0 = Fv->b0;
      uint64_t* bfull = Fv->bfull;
      uint64_t* bempty = Fv->bempty;
      const int a_st = Fv->a_stride, b_st = Fv->b_stride;
      const int m0 = Fv->m0, n0 = Fv->n0, iters = Fv->iters, stages = Fv->stages;
      const int a_t = Fv->a_t, b_t = Fv->b_t, bk = Fv->bk;
      const int k_base = Fv->k_base, kind = Fv->kind;
      const unsigned xb = Fv->expect_bytes;
      wg_tmap_fence_acquire(tmA);
      wg_tmap_fence_acquire(tmB);
      for (int t = 0; t < iters; ++t) {
        const int st = t % stages;
        if (t >= stages) wg_mbar_wait(&bempty[st], (t / stages - 1) & 1);
        wg_mbar_expect_tx(&bfull[st], xb);
        const int k0 = k_base + t * bk;
        if (kind == 3) {
          wg_tma_load_2d(tmA, a0 + st * a_st, k0, m0, &bfull[st]);
#pragma unroll
          for (int g = 0; g < 4; ++g)
            wg_tma_load_2d(tmB, b0 + st * b_st + g * 8192, k0, n0 + g * 64,
                           &bfull[st]);
        } else if (kind == 1) {
          wg_tma_load_2d(tmA, a0 + st * a_st, k0, m0, &bfull[st]);
          if (b_t) {
            wg_tma_load_2d(tmB, b0 + st * b_st, k0, n0, &bfull[st]);
          } else {
#pragma unroll
            for (int g = 0; g < 2; ++g)
              wg_tma_load_2d(tmB, b0 + st * b_st + g * 8192, n0 + g * 64, k0,
                             &bfull[st]);
          }
        } else if (a_t) {
          wg_tma_load_2d(tmA, a0 + st * a_st, m0, k0, &bfull[st]);
          wg_tma_load_2d(tmA, a1 + st * a_st, m0 + 64, k0, &bfull[st]);
        } else {
          wg_tma_load_2d(tmA, a0 + st * a_st, k0, m0, &bfull[st]);
        }
        if (kind == 2) {
          if (b_t)
            wg_tma_load_2d(tmB, b0 + st * b_st, k0, n0, &bfull[st]);
          else
            wg_tma_load_2d(tmB, b0 + st * b_st, n0, k0, &bfull[st]);
        } else if (kind == 0) {
#pragma unroll
          for (int g = 0; g < 4; ++g)
            wg_tma_load_2d(tmB, b0 + st * b_st + g * 8192, n0 + g * 64, k0,
                           &bfull[st]);
        }
      }
#endif  // MK_PDF_GEMM_FEED
    }
  }
#endif  // MK_PDF_FEED
}
#endif  // MK_PDF

// ---- dataflow executor v2: region watermarks (tile-granular producer/consumer) -------
// Like megakernel_df, plus: producers with gated out-edges count completed tiles per
// 128-row REGION (band_tiles[i] tiles each, m-major tile order required); a completed
// region prefix advances frontier[i]; each gated edge publishes watermark[consumer] =
// frontier * gate_k (precomputed: consumer tiles enabled per producer region) via
// atomicMax. Gated consumers enter the ready ring when their REDUCED pending (gated
// in-edge removed) hits zero, but tile claims are BOUNDED by the watermark (CAS loop
// instead of the unconditional atomicAdd). On full completion the producer also
// publishes an unbounded watermark — belt-and-suspenders against any frontier race.
__global__ void megakernel_df2(const Instr* __restrict__ instrs, const int n_instr,
                               const int ring_cap, const int* __restrict__ dep_cnt,
                               const int* __restrict__ adj_off, const int* __restrict__ adj,
                               const int* __restrict__ claim_sz, const int* __restrict__ gated_in,
                               const int* __restrict__ band_tiles, const int* __restrict__ region_off,
                               const int* __restrict__ region_cnt0, const int* __restrict__ gate_off,
                               const int* __restrict__ gate_cons, const int* __restrict__ gate_k,
                               int* __restrict__ state, void** __restrict__ bufs,
                               long long* __restrict__ iclk) {
  extern __shared__ char smem[];
  // state: pending|cursor|done|queued|watermark|frontier (n each) | ready[ring_cap]
  //        | ctrl[4] | region_cnt[R]
  int* pending = state;
  int* cursor = state + n_instr;
  int* done = state + 2 * n_instr;
  int* queued = state + 3 * n_instr;  // instr has a live ring slot (wakeup dedupe)
  int* wmark = state + 4 * n_instr;
  int* frontier = state + 5 * n_instr;
  int* ready = state + 6 * n_instr;
  int* ctrl = state + 6 * n_instr + ring_cap;
  int* rcnt = ctrl + 4;
  const int R = region_off[n_instr];

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
#if defined(MK_PDF_PRODUCER) && defined(MK_ATTN_PDF_FEED)
  if (threadIdx.x == 0) g_pdf_feed.active = 0;  // smem is not zeroed across launches
#endif
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    pending[i] = dep_cnt[i];
    cursor[i] = 0;
    done[i] = 0;
    queued[i] = dep_cnt[i] == 0 ? 1 : 0;
    wmark[i] = gated_in[i] ? 0 : instrs[i].ntiles;
    frontier[i] = 0;
  }
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < ring_cap;
       r += gridDim.x * blockDim.x)
    ready[r] = -1;
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < R; r += gridDim.x * blockDim.x)
    rcnt[r] = region_cnt0[r];
  if (blockIdx.x == 0 && threadIdx.x < 4) ctrl[threadIdx.x] = 0;
  grid.sync();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    if (dep_cnt[i] == 0) {
      const int t = atomicAdd(&ctrl[0], 1);
      atomicExch(&ready[t], i);
    }
  }
  grid.sync();

  __shared__ int s_ins, s_t0, s_t1;
  __shared__ Instr s_I;  // claimed instr staged in smem (dispatch spill tax)
  int last_ins = -1;  // sticky: retry the previous instruction before scanning
  volatile int* vready = ready;
  volatile int* vctrl = ctrl;
  volatile int* vwmark = wmark;
  volatile int* vrcnt = rcnt;
  volatile int* vfrontier = frontier;
  volatile int* vpending = pending;

  auto push = [&](int i) {
    const int t = atomicAdd(&ctrl[0], 1);
    atomicExch(&ready[t], i);
  };

  // watermark-bounded claim (thread 0). Returns true and fills s_* on success.
  auto try_claim = [&](int ins) -> bool {
    const int nt = instrs[ins].ntiles;
    const int bound = gated_in[ins] ? min(vwmark[ins], nt) : nt;
    const int bs = claim_sz[ins];
    if (bound == nt) {
      // unbounded (ungated, or gated fully published): one-shot fetch-add — a CAS
      // loop here degrades quadratically when ~all blocks claim from one big instr
      const int t0 = atomicAdd(&cursor[ins], bs);
      if (t0 < nt) {
        s_ins = ins;
        s_t0 = t0;
        s_t1 = min(t0 + bs, nt);
        return true;
      }
      return false;
    }
    int c = cursor[ins];
    while (c < bound) {
      const int t1 = min(c + bs, bound);
      const int prev = atomicCAS(&cursor[ins], c, t1);
      if (prev == c) {
        s_ins = ins;
        s_t0 = c;
        s_t1 = t1;
        return true;
      }
      c = prev;
    }
    return false;
  };

  for (;;) {
    if (threadIdx.x == 0) {
      s_ins = -1;
      if (last_ins >= 0 && cursor[last_ins] < instrs[last_ins].ntiles) try_claim(last_ins);
      while (s_ins < 0 && vctrl[1] < n_instr) {
        const int gq0 = vctrl[2];
        const int tail = vctrl[0];
        for (int q = gq0; q < tail; ++q) {
          const int ins = vready[q];
          if (ins == -1) continue;  // slot reserved, payload not yet visible
          if (ins == -3 || cursor[ins] >= instrs[ins].ntiles) {
            if (q == vctrl[2]) atomicCAS(&ctrl[2], q, q + 1);  // dead/drained: advance head
            continue;
          }
          if (try_claim(ins)) break;
          // gated instr exhausted below its watermark: PARK it — kill the slot and
          // transfer re-push duty to the watermark raisers (a parked entry would
          // otherwise pin the consumed head and every idle block would rescan it)
          if (gated_in[ins] && atomicExch(&queued[ins], 0) == 1) {
            __threadfence();
            const int nt = instrs[ins].ntiles;
            if (cursor[ins] < min(vwmark[ins], nt)) {
              // became claimable while parking: re-arm (lost-wakeup guard)
              if (atomicExch(&queued[ins], 1) == 1)
                atomicExch(&ready[q], -3);  // a raiser re-armed + pushed a fresh slot
              // else our slot stays live; claim succeeds on a later pass
            } else {
              atomicExch(&ready[q], -3);
            }
          }
        }
        if (s_ins >= 0) break;
        __nanosleep(MK_IDLE_NS);
      }
      last_ins = s_ins;
    }
    __syncthreads();
    const int ins = s_ins;
    if (ins < 0) break;  // everything finished
    const int t0 = s_t0, t1 = s_t1;
    if (threadIdx.x < 3 + MK_MAX_ARGS)  // smem-stage the Instr (dispatch spill tax)
      reinterpret_cast<int*>(&s_I)[threadIdx.x] =
          reinterpret_cast<const int*>(instrs + ins)[threadIdx.x];
    __syncthreads();
    if (iclk && threadIdx.x == 0 && t0 == 0) iclk[2 * ins] = mk_globaltimer();
    for (int t = t0; t < t1; ++t) {
      dispatch(s_I, t, bufs, smem);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      __threadfence();  // publish this batch's writes before any watermark/dep signal
      const int ro = region_off[ins];
      const int nr = region_off[ins + 1] - ro;
      if (nr > 0) {  // producer with gated out-edges: per-region accounting
        const int bt = band_tiles[ins];
        bool raised = false;
        for (int r = t0 / bt; r <= (t1 - 1) / bt; ++r) {
          const int take = min(t1, (r + 1) * bt) - max(t0, r * bt);
          if (atomicSub(&rcnt[ro + r], take) == take) {  // this batch zeroed region r
            // advance frontier over the completed prefix (volatile reads: atomicSub
            // results live in L2; a stale L1 read here could strand the frontier)
            int f = vfrontier[ins];
            while (f < nr && vrcnt[ro + f] == 0) {
              atomicCAS(&frontier[ins], f, f + 1);
              f = vfrontier[ins];
            }
            for (int e = gate_off[ins]; e < gate_off[ins + 1]; ++e)
              atomicMax(&wmark[gate_cons[e]], f * gate_k[e]);
            raised = true;
          }
        }
        if (raised) {  // wake parked consumers (after ALL publishes; fence orders them)
          __threadfence();
          for (int e = gate_off[ins]; e < gate_off[ins + 1]; ++e) {
            const int cons = gate_cons[e];
            if (vpending[cons] == 0 && cursor[cons] < instrs[cons].ntiles &&
                atomicExch(&queued[cons], 1) == 0)
              push(cons);
          }
        }
      }
      const int d = atomicAdd(&done[ins], t1 - t0) + (t1 - t0);
      if (d == s_I.ntiles) {  // last tile: enable dependents
        if (iclk) iclk[2 * ins + 1] = mk_globaltimer();
        for (int e = gate_off[ins]; e < gate_off[ins + 1]; ++e)
          atomicMax(&wmark[gate_cons[e]], 0x3fffffff);  // final: everything enabled
        __threadfence();
        for (int e = gate_off[ins]; e < gate_off[ins + 1]; ++e) {
          const int cons = gate_cons[e];
          if (vpending[cons] == 0 && cursor[cons] < instrs[cons].ntiles &&
              atomicExch(&queued[cons], 1) == 0)
            push(cons);
        }
        for (int e = adj_off[ins]; e < adj_off[ins + 1]; ++e) {
          const int dep = adj[e];
          if (atomicSub(&pending[dep], 1) == 1) {
            atomicExch(&queued[dep], 1);
            push(dep);
          }
        }
        atomicAdd(&ctrl[1], 1);
      }
    }
    __syncthreads();
  }
}

// ---- warp-specialized dataflow executor (ws) ------------------------------------------
// Scheduling comes COMPLETELY off the consumer critical path. 384 threads: threads
// 0-255 (warpgroups 0-1) are the consumer group running ops EXACTLY as the other
// executors (threadIdx.x semantics unchanged; ops stride by MK_CONSUMERS and sync on
// bar.sync 1,256). Warpgroup 2 lane 0 (threadIdx.x == 256) is the scheduler: it
// pre-claims the NEXT (instr, tile-range) batch into a second smem control slot while
// consumers execute the current one, and does ALL completion accounting
// (__threadfence, done atomicAdd, dependent decrements, ready-ring pushes, iclk
// stamps) while consumers immediately start the pre-claimed batch. Threads 257-383
// exit after init (no full-block barrier exists after specialization). NEVER
// __syncthreads below the grid.syncs: warpgroup 2 does not participate in op
// execution.
//
// Register plumbing (measured, do not "simplify"): H100 allocates registers at 4-WARP
// granularity, so ANY block over 256 threads is charged 12 warps' worth -> a hard
// 65536/384 = 168-reg ceiling. The plan's original 288-thread shape compiled to
// REG:168 STACK:544 (ptxas spilled the op hot paths) and ran a uniform +14% on BOTH
// configs. Fix = the ws_probe recipe: entry __maxnreg__(168) (also required for ptxas
// to honor setmaxnreg at all, C7508), then per-warpgroup setmaxnreg — scheduler
// warpgroup dec->56, consumer warpgroups inc->224 (feasible: 128*(168-56) ==
// 256*(224-168); 256*224 + 128*56 = 64512 <= 64K). The 240/24 split was ALSO measured
// and is WORSE at both configs: the dec-24 scheduler spills its claim/accounting path
// and the slower handoff costs more than 16 extra consumer registers recover, even
// though the register-fat WMMA/attention ops profile +4-12% over df's 255-reg build at
// 224 (the ~126-reg wgmma path is unaffected). Requires the explicit
// -gencode=arch=compute_90a,code=sm_90a flag in mk.py (CUDA 13.1's -arch=sm_90a
// silently also emits compute_90 PTX where setmaxnreg is rejected).
//
// Handoff: monotone smem sequence counters. full_seq = batches staged (st.release.cta
// by the scheduler after writing the slot); done_seq = batches finished (st.release.cta
// by consumer thread 0 after the batch's last consumer_sync — which CTA-orders all 256
// consumers' writes before the flag; the scheduler's __threadfence then publishes them
// device-wide by cumulativity, the producer-side-completion pattern the wsprobe
// measured 13% faster than consumer-side). Slot of batch k = k&1; staging batch k
// requires k < acct + lookahead, and accounting always drains first, so a slot's
// previous occupant is finished AND accounted before reuse. Halt = staged Instr with
// op < 0.
//
// state layout (pad = ints per entry; 32 = one 128B line per instr to kill the false
// sharing the wspec replication identified; ready rings stay packed for cheap scans):
//   cursor[n*pad] | done[n*pad] | pending[n*pad] | ready_hot[n] | ready_cold[n]
//   | ctrl[6*pad]  (0=hot tail, 1=finished, 2=hot head, 3=cold tail, 4=cold head)
// v3 P4b: hot/cold criticality rings ported from df (the P6 win that made ws lose
// its P5 lead): idle schedulers drain chain work before sink/fill work.
#define MK_WS_CTRL_BYTES 256  // control carve-out at the base of dynamic smem

struct WsCtrl {
  Instr ins[2];  // staged instruction per slot
  int t0[2], t1[2];
  int full_seq;
  int done_seq;
};
static_assert(sizeof(WsCtrl) <= MK_WS_CTRL_BYTES, "ws control region overflow");

__device__ __forceinline__ void mk_st_release_cta(int* p, int v) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ int mk_ld_acquire_cta(const int* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  int v;
  asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}

#define MK_WS_THREADS 384

extern "C" __global__ void __maxnreg__(168) megakernel_ws(
    const Instr* __restrict__ instrs, int n_instr, const int* __restrict__ dep_cnt,
    const int* __restrict__ adj_off, const int* __restrict__ adj,
    const int* __restrict__ claim_sz, const int* __restrict__ crit, int* state, int pad,
    int lookahead, void** bufs, long long* iclk /* nullable [2*n] */) {
  extern __shared__ char smem[];
  __shared__ Instr sQI[2];  // consumer-owned Instr snapshots (dispatch spill tax: the
  // 104B register copy live across dispatch was df's biggest P6 single win; the
  // earlier ws attempt read through a reference INTO THE CONTROL SLOT and hung —
  // this snapshot is consumer-owned, complete behind a consumer_sync before any
  // dispatch, and the slot itself is never read during execution).
  WsCtrl* C = reinterpret_cast<WsCtrl*>(smem);
  char* opsmem = smem + MK_WS_CTRL_BYTES;  // ops get the rest (16B-aligned)
  cg::grid_group grid = cg::this_grid();
#ifdef MK_PDF_PRODUCER
  if (threadIdx.x == 0) g_pdf_feed.active = 0;  // smem is not zeroed across launches
#endif
  int* cursor = state;
  int* done = state + n_instr * pad;
  int* pending = state + 2 * n_instr * pad;
  int* ready_hot = state + 3 * n_instr * pad;
  int* ready_cold = ready_hot + n_instr;
  int* ctrl = ready_cold + n_instr;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    cursor[i * pad] = 0;
    done[i * pad] = 0;
    pending[i * pad] = dep_cnt[i];
    ready_hot[i] = -1;
    ready_cold[i] = -1;
  }
  if (blockIdx.x == 0 && threadIdx.x < 6) ctrl[threadIdx.x * pad] = 0;
  if (threadIdx.x == 0) {
    C->full_seq = 0;
    C->done_seq = 0;
  }
  grid.sync();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    if (dep_cnt[i] == 0) {
      if (crit[i]) {
        const int t = atomicAdd(&ctrl[0], 1);
        atomicExch(&ready_hot[t], i);
      } else {
        const int t = atomicAdd(&ctrl[3 * pad], 1);
        atomicExch(&ready_cold[t], i);
      }
    }
  }
  grid.sync();

  if (threadIdx.x < MK_CONSUMERS) {
    // ---------------- consumer group (threads 0-255) ----------------
    asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");  // blocks until WG2's dec frees them
    for (int seen = 0;; ++seen) {
      while (mk_ld_acquire_cta(&C->full_seq) <= seen) __nanosleep(32);
      const int slot = seen & 1;
      // snapshot the staged Instr into consumer-owned static smem (26-int parallel
      // copy; every thread has done its own acquire, so the slot bytes are visible).
      // Slot reuse needs batch seen+1 ACCOUNTED, which needs it FINISHED — no writer
      // can touch C->ins[slot] while this batch executes, and dispatch reads sQI.
      // MK_WS_REGCOPY rebuilds with the old per-thread register copy (A/B: barrier
      // cost per batch vs the 104B copy's spill; STACK 304 vs 544).
#ifdef MK_WS_REGCOPY
      const Instr I = C->ins[slot];  // ordered after the acquire
      const int t0 = C->t0[slot], t1 = C->t1[slot];
      if (I.op < 0) break;  // halt sentinel
#else
      if (threadIdx.x < 3 + MK_MAX_ARGS)
        reinterpret_cast<int*>(&sQI[slot])[threadIdx.x] =
            reinterpret_cast<const int*>(&C->ins[slot])[threadIdx.x];
      const int t0 = C->t0[slot], t1 = C->t1[slot];
      consumer_sync();  // snapshot complete before first dispatch
      const Instr& I = sQI[slot];
      if (I.op < 0) break;  // halt sentinel
#endif
      for (int t = t0; t < t1; ++t) {
        dispatch(I, t, bufs, opsmem);
        consumer_sync();  // smem reuse safety + orders all consumer writes
      }
      if (threadIdx.x == 0) mk_st_release_cta(&C->done_seq, seen + 1);
    }
    return;
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");  // whole warpgroup 2, converged here
  if (threadIdx.x != MK_CONSUMERS) return;  // scheduler warpgroup: only thread 256 stays

  // ---------------- scheduler (thread 256) ----------------
  // Claim discipline (each rule measured — see results/mkv3-p4a.md):
  //  * COMMIT LATE, DISCOVER EARLY (lookahead=1): while consumers compute, the
  //    scheduler only CACHES a ring candidate (no cursor atomicAdd). At the done_seq
  //    flip it claims sticky/candidate with one atomic, stages, and releases full_seq
  //    BEFORE doing the completion accounting — consumers restart in ~one L2 round
  //    trip, and the fence + done/pending/ring atomics overlap the next op. Eagerly
  //    pre-COMMITTING a batch into slot B (lookahead=2) measured NET NEGATIVE: a
  //    committed long batch (attn-dq quantum ~40us) rides out the current op on THIS
  //    block while other blocks idle at the instr tail (ATTN_DQ span +740us at small);
  //    gating pre-claims by abundance also measured negative (lost pre-claims cost
  //    more than the tail imbalance they avoid).
  //  * While consumers are busy the ring is rescanned ONLY when its tail moved: 132
  //    schedulers continuously rescanning (cursor probes = L2 round trips) contend
  //    with the latency-bound ops (the df2 lesson). New claimable work for a busy
  //    block can only appear via a push (cursors only grow), so the tail check is
  //    exact. When consumers are idle, scan every pass like df.
  //  * Completion accounting that drops a dependent's pending to 0 claims THAT instr
  //    directly (hint) — the successor of a chain hop skips ring rediscovery.
  volatile int* vhot = ready_hot;
  volatile int* vcold = ready_cold;
  volatile int* vctrl = ctrl;
  int last_ins = -1;   // sticky: retry the previous instruction before scanning
  int cand = -1;       // pre-discovered ring candidate (uncommitted)
  int staged = 0, acct = 0;
  int seen_hot = -1, seen_cold = -1;  // ring tails at the last completed scan (held
                                      // back at invisible slots; busy-rescan gate)
  int q_ins0 = -1, q_t00 = 0, q_t10 = 0;  // outstanding batch info per slot (registers,
  int q_ins1 = -1, q_t01 = 0, q_t11 = 0;  // not an array: dynamic indexing would spill)
  unsigned lazy = 0;  // busy-side discovery runs every 32nd pass (~1us cadence): 132
                      // schedulers probing cursors/tail at the 32ns poll cadence is
                      // measurable L2 contention against the latency-bound ops

  int c_ins = -1, c_t0 = 0, c_t1 = 0;  // out-params of try_claim
  // NOTE: gating slot-B pre-claims by tile abundance (skip work whose remaining tiles
  // <= nblocks*bs, or <= nblocks) was measured repeatedly and always NET NEGATIVE
  // (+90..+140 vs ungated): the tail imbalance a committed batch causes costs less
  // than the pre-claims the gate forfeits. `gated` retained for documentation; unused.
  auto try_claim = [&](int i, bool /*gated*/) {
    const int nt = instrs[i].ntiles;
    const int cur = cursor[i * pad];
    if (cur >= nt) return false;
    const int bs = claim_sz[i];
    const int c0 = atomicAdd(&cursor[i * pad], bs);
    if (c0 >= nt) return false;
    c_ins = i;
    c_t0 = c0;
    c_t1 = min(c0 + bs, nt);
    return true;
  };
  auto stage = [&]() {
    const bool s = staged & 1;
    C->ins[s] = instrs[c_ins];
    C->t0[s] = c_t0;
    C->t1[s] = c_t1;
    if (s) {
      q_ins1 = c_ins, q_t01 = c_t0, q_t11 = c_t1;
    } else {
      q_ins0 = c_ins, q_t00 = c_t0, q_t10 = c_t1;
    }
    if (iclk && c_t0 == 0) iclk[2 * c_ins] = mk_globaltimer();
    mk_st_release_cta(&C->full_seq, staged + 1);
    ++staged;
    last_ins = c_ins;
  };
  // hot-first two-ring pass (P6 criticality rings): claim (do_claim) or discover a
  // candidate. Sets c_ins / cand respectively; advances the consumed heads past
  // drained entries; updates seen_* (held back at invisible slots so a mid-push
  // entry still forces a rescan).
  auto ring_pass = [&](bool pre, bool do_claim) {
    const int htail = vctrl[0];
    int vis = htail;
    for (int q = vctrl[2 * pad]; q < htail; ++q) {
      const int r = vhot[q];
      if (r < 0) {  // slot reserved, payload not yet visible
        if (q < vis) vis = q;
        continue;
      }
      if (cursor[r * pad] >= instrs[r].ntiles) {
        if (q == vctrl[2 * pad]) atomicCAS(&ctrl[2 * pad], q, q + 1);
        continue;
      }
      if (do_claim ? try_claim(r, pre) : (cand = r) >= 0) return;
    }
    seen_hot = vis;
    const int ctail = vctrl[3 * pad];
    vis = ctail;
    for (int q = vctrl[4 * pad]; q < ctail; ++q) {
      const int r = vcold[q];
      if (r < 0) {
        if (q < vis) vis = q;
        continue;
      }
      if (cursor[r * pad] >= instrs[r].ntiles) {
        if (q == vctrl[4 * pad]) atomicCAS(&ctrl[4 * pad], q, q + 1);
        continue;
      }
      if (do_claim ? try_claim(r, pre) : (cand = r) >= 0) return;
    }
    seen_cold = vis;
  };

  for (;;) {
    const int ds = mk_ld_acquire_cta(&C->done_seq);
    bool progress = false;
    // 1) FAST PATH at the flip: consumers drained everything staged — restage from
    //    sticky/candidate BEFORE accounting (safe: ready-ring entries and same-instr
    //    tiles never read the just-finished batch's output). ds - acct < 2 guard
    //    (v3 P4b): at lookahead=2 the flip can find TWO finished un-accounted
    //    batches; staging here would overwrite slot (ds&1)'s q_ins* bookkeeping
    //    while batch acct (same parity) still needs it for accounting -> done[]
    //    corruption -> lost dependents (the likely historical la=2 hang). At
    //    lookahead=1 the guard is always true (no behavior change).
    if (ds > acct && staged == ds && ds - acct < 2) {
      c_ins = -1;
      if (last_ins >= 0) try_claim(last_ins, false);
      if (c_ins < 0 && cand >= 0) {
        try_claim(cand, false);
        cand = -1;
      }
      if (c_ins >= 0) {
        stage();
        progress = true;
      }
    }
    // 2) completion accounting for every batch consumers have finished
    int hint = -1;
    while (acct < ds) {
      const bool s = acct & 1;
      const int ins = s ? q_ins1 : q_ins0;
      const int t0 = s ? q_t01 : q_t00;
      const int t1 = s ? q_t11 : q_t10;
      __threadfence();  // publish the consumers' writes before enabling dependents
      const int d = atomicAdd(&done[ins * pad], t1 - t0) + (t1 - t0);
      if (d == instrs[ins].ntiles) {  // last tile anywhere: enable dependents
        if (iclk) iclk[2 * ins + 1] = mk_globaltimer();
        for (int e = adj_off[ins]; e < adj_off[ins + 1]; ++e) {
          const int dep = adj[e];
          if (atomicSub(&pending[dep * pad], 1) == 1) {
            if (crit[dep]) {
              const int t = atomicAdd(&ctrl[0], 1);
              atomicExch(&ready_hot[t], dep);
            } else {
              const int t = atomicAdd(&ctrl[3 * pad], 1);
              atomicExch(&ready_cold[t], dep);
            }
            if (hint < 0) hint = dep;  // chain fast path: claim it ourselves too
          }
        }
        atomicAdd(&ctrl[1 * pad], 1);
      }
      ++acct;
      progress = true;
    }
    ++lazy;
    // 3) stage (JIT when consumers wait; eager slot-B pre-claim only if lookahead=2)
    if (staged < acct + lookahead) {
      const bool pre = staged > acct;
      c_ins = -1;
      if (hint >= 0) try_claim(hint, pre);
      if (c_ins < 0 && last_ins >= 0) try_claim(last_ins, pre);
      if (c_ins < 0 && cand >= 0) {
        try_claim(cand, pre);
        cand = -1;
      }
      if (c_ins < 0 &&
          (!pre || vctrl[0] != seen_hot || vctrl[3 * pad] != seen_cold))  // busy-gated
        ring_pass(pre, /*do_claim=*/true);
      if (c_ins >= 0) {
        stage();
        progress = true;
      } else if (acct == staged && vctrl[1 * pad] >= n_instr) {  // all finished: halt
        C->ins[staged & 1].op = -1;
        mk_st_release_cta(&C->full_seq, staged + 1);
        break;
      }
    } else if (staged > acct && cand < 0 && (lazy & 31u) == 0) {
      // 4) busy + nothing committed ahead: DISCOVER the next candidate (no atomics on
      //    the claim path at the flip). Sticky covers the same-instr case; the rings
      //    are scanned only when a tail moved.
      const bool sticky_live = last_ins >= 0 && cursor[last_ins * pad] < instrs[last_ins].ntiles;
      if (!sticky_live && (vctrl[0] != seen_hot || vctrl[3 * pad] != seen_cold))
        ring_pass(/*pre=*/true, /*do_claim=*/false);
    }
    if (!progress) __nanosleep(staged > acct ? 32 : 256);
  }
}


// ---- host launcher ------------------------------------------------------------------
static int mk_cluster_x() {
  const char* env = std::getenv("MK_CLUSTER_X");
  if (!env || !*env) return 1;
  const int x = std::atoi(env);
  TORCH_CHECK(x == 1 || x == 2, "MK_CLUSTER_X currently supports only 1 or 2, got ", x);
  return x;
}

template <typename Kernel, typename... Args>
static void mk_launch_cooperative(Kernel kernel, dim3 grid, dim3 block, void** legacy_args,
                                  size_t smem_bytes, cudaStream_t stream, Args... args) {
  const int cluster_x = mk_cluster_x();
  if (cluster_x == 1) {
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, grid, block, legacy_args,
                                               smem_bytes, stream));
    return;
  }

  TORCH_CHECK(grid.x % cluster_x == 0,
              "cluster launch requires grid.x divisible by MK_CLUSTER_X; grid.x=",
              grid.x, " MK_CLUSTER_X=", cluster_x);
  cudaLaunchConfig_t cfg = {0};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster_x;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeCooperative;
  attrs[1].val.cooperative = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

static int g_nblocks = -1;

void mk_run(torch::Tensor instrs, torch::Tensor wave_start, torch::Tensor wave_tiles,
            torch::Tensor bufs, int64_t smem_bytes,
            c10::optional<torch::Tensor> wave_clk) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  TORCH_CHECK(wave_start.is_cuda() && wave_tiles.is_cuda() && bufs.is_cuda());
  TORCH_CHECK(bufs.dtype() == torch::kInt64);
  const int nwaves = (int)wave_tiles.numel();

  // Re-configure when a larger carveout is requested (mixed-carveout processes:
  // e.g. a 100KB default model and a 112KB D=128-attention model side by side).
  static int smem_configured = 0;
  if ((int)smem_bytes > smem_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    smem_configured = (int)smem_bytes;
    g_nblocks = -1;
  }
  if (g_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel, 256, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel does not fit an SM with smem=", smem_bytes);
    g_nblocks = sms * per_sm;
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_ws = wave_start.data_ptr<int>();
  const int* d_wt = wave_tiles.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      wave_clk.has_value() ? reinterpret_cast<long long*>(wave_clk->data_ptr<int64_t>()) : nullptr;
  void* args[] = {(void*)&d_instrs, (void*)&d_ws, (void*)&d_wt, (void*)&nwaves,
                  (void*)&d_bufs, (void*)&d_clk};
  auto stream = at::cuda::getCurrentCUDAStream();
  mk_launch_cooperative(megakernel, dim3(g_nblocks), dim3(256), args,
                        (size_t)smem_bytes, stream.stream(), d_instrs, d_ws, d_wt,
                        nwaves, d_bufs, d_clk);
}

int64_t mk_nblocks() { return g_nblocks; }

void mk_run_df(torch::Tensor instrs, torch::Tensor dep_cnt, torch::Tensor adj_off,
               torch::Tensor adj, torch::Tensor claim_sz, torch::Tensor crit,
               int64_t cold_cap64, torch::Tensor state, torch::Tensor bufs,
               int64_t smem_bytes, c10::optional<torch::Tensor> iclk,
               int64_t bind0_64, int64_t ptr0_64, int64_t bind1_64, int64_t ptr1_64) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  const int n_instr = (int)(instrs.numel() / (3 + MK_MAX_ARGS));
  TORCH_CHECK(state.numel() >= 5 * (int64_t)n_instr + 8, "df state tensor too small");

  static int df_configured = 0;
  static int df_nblocks = -1;
  if ((int)smem_bytes > df_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel_df,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    df_configured = (int)smem_bytes;
    df_nblocks = -1;
  }
  if (df_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel_df, MK_DF_THREADS, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel_df does not fit an SM with smem=", smem_bytes);
    df_nblocks = sms * per_sm;
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_dc = dep_cnt.data_ptr<int>();
  const int* d_ao = adj_off.data_ptr<int>();
  const int* d_ad = adj.data_ptr<int>();
  const int* d_cs = claim_sz.data_ptr<int>();
  const int* d_cr = crit.data_ptr<int>();
  // low 20 bits = cap; high bits = hot-consumed-head release threshold
  // (phase-adaptive cold-cap release; 0 = knob off -> INT_MAX)
  int cold_cap = (int)(cold_cap64 & 0xFFFFF);
  int release_at = (int)(cold_cap64 >> 20);
  if (release_at <= 0) release_at = INT_MAX;
  if (cold_cap <= 0) cold_cap = df_nblocks;  // 0 = uncapped
  int* d_state = state.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      iclk.has_value() ? reinterpret_cast<long long*>(iclk->data_ptr<int64_t>()) : nullptr;
  int bind0 = (int)bind0_64;
  int bind1 = (int)bind1_64;
  unsigned long long ptr0 = (unsigned long long)ptr0_64;
  unsigned long long ptr1 = (unsigned long long)ptr1_64;
  void* args[] = {(void*)&d_instrs, (void*)&n_instr, (void*)&d_dc,    (void*)&d_ao,
                  (void*)&d_ad,     (void*)&d_cs,    (void*)&d_cr,    (void*)&cold_cap,
                  (void*)&release_at, (void*)&d_state, (void*)&d_bufs, (void*)&d_clk,
                  (void*)&bind0,    (void*)&ptr0,    (void*)&bind1,   (void*)&ptr1};
  auto stream = at::cuda::getCurrentCUDAStream();
  mk_launch_cooperative(megakernel_df, dim3(df_nblocks), dim3(MK_DF_THREADS), args,
                        (size_t)smem_bytes, stream.stream(), d_instrs, n_instr,
                        d_dc, d_ao, d_ad, d_cs, d_cr, cold_cap, release_at, d_state,
                        d_bufs, d_clk, bind0, ptr0, bind1, ptr1);
}

#ifdef MK_PDF
void mk_run_pdf(torch::Tensor instrs, torch::Tensor dep_cnt, torch::Tensor adj_off,
                torch::Tensor adj, torch::Tensor claim_sz, torch::Tensor crit,
                int64_t cold_cap64, torch::Tensor state, torch::Tensor bufs,
                int64_t smem_bytes, c10::optional<torch::Tensor> iclk,
                int64_t bind0_64, int64_t ptr0_64, int64_t bind1_64, int64_t ptr1_64) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  const int n_instr = (int)(instrs.numel() / (3 + MK_MAX_ARGS));
  TORCH_CHECK(state.numel() >= 5 * (int64_t)n_instr + 8, "pdf state tensor too small");

  static int pdf_configured = 0;
  static int pdf_nblocks = -1;
  if ((int)smem_bytes > pdf_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel_pdf,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    pdf_configured = (int)smem_bytes;
    pdf_nblocks = -1;
  }
  if (pdf_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel_pdf, MK_PDF_THREADS, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel_pdf does not fit an SM with smem=", smem_bytes);
    pdf_nblocks = sms;  // 1 block/SM by design (384 threads at entry 168 regs)
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_dc = dep_cnt.data_ptr<int>();
  const int* d_ao = adj_off.data_ptr<int>();
  const int* d_ad = adj.data_ptr<int>();
  const int* d_cs = claim_sz.data_ptr<int>();
  const int* d_cr = crit.data_ptr<int>();
  int cold_cap = (int)(cold_cap64 & 0xFFFFF);
  int release_at = (int)(cold_cap64 >> 20);
  if (release_at <= 0) release_at = INT_MAX;
  if (cold_cap <= 0) cold_cap = pdf_nblocks;  // 0 = uncapped
  int* d_state = state.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      iclk.has_value() ? reinterpret_cast<long long*>(iclk->data_ptr<int64_t>()) : nullptr;
  int bind0 = (int)bind0_64;
  int bind1 = (int)bind1_64;
  unsigned long long ptr0 = (unsigned long long)ptr0_64;
  unsigned long long ptr1 = (unsigned long long)ptr1_64;
  void* args[] = {(void*)&d_instrs, (void*)&n_instr, (void*)&d_dc,    (void*)&d_ao,
                  (void*)&d_ad,     (void*)&d_cs,    (void*)&d_cr,    (void*)&cold_cap,
                  (void*)&release_at, (void*)&d_state, (void*)&d_bufs, (void*)&d_clk,
                  (void*)&bind0,    (void*)&ptr0,    (void*)&bind1,   (void*)&ptr1};
  auto stream = at::cuda::getCurrentCUDAStream();
  mk_launch_cooperative(megakernel_pdf, dim3(pdf_nblocks), dim3(MK_PDF_THREADS), args,
                        (size_t)smem_bytes, stream.stream(), d_instrs, n_instr,
                        d_dc, d_ao, d_ad, d_cs, d_cr, cold_cap, release_at, d_state,
                        d_bufs, d_clk, bind0, ptr0, bind1, ptr1);
}
#endif  // MK_PDF

void mk_run_ws(torch::Tensor instrs, torch::Tensor dep_cnt, torch::Tensor adj_off,
               torch::Tensor adj, torch::Tensor claim_sz, torch::Tensor crit,
               torch::Tensor state, int64_t pad64, int64_t lookahead64,
               torch::Tensor bufs, int64_t smem_bytes,
               c10::optional<torch::Tensor> iclk) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  const int n_instr = (int)(instrs.numel() / (3 + MK_MAX_ARGS));
  const int pad = (int)pad64;
  const int lookahead = (int)lookahead64;
  TORCH_CHECK(pad >= 1 && lookahead >= 1 && lookahead <= 2);
  TORCH_CHECK(state.numel() >= 3 * (int64_t)n_instr * pad + 2 * n_instr + 6 * pad,
              "ws state tensor too small for pad=", pad);

  static int ws_configured = 0;
  static int ws_nblocks = -1;
  if ((int)smem_bytes > ws_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel_ws,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    ws_configured = (int)smem_bytes;
    ws_nblocks = -1;
  }
  if (ws_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel_ws, MK_WS_THREADS, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel_ws does not fit an SM with smem=", smem_bytes);
    ws_nblocks = sms;  // 1 block/SM by design (384 threads at entry 168 regs, ~100KB smem)
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_dc = dep_cnt.data_ptr<int>();
  const int* d_ao = adj_off.data_ptr<int>();
  const int* d_ad = adj.data_ptr<int>();
  const int* d_cs = claim_sz.data_ptr<int>();
  const int* d_cr = crit.data_ptr<int>();
  int* d_state = state.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      iclk.has_value() ? reinterpret_cast<long long*>(iclk->data_ptr<int64_t>()) : nullptr;
  void* args[] = {(void*)&d_instrs, (void*)&n_instr,   (void*)&d_dc,    (void*)&d_ao,
                  (void*)&d_ad,     (void*)&d_cs,      (void*)&d_cr,    (void*)&d_state,
                  (void*)&pad,      (void*)&lookahead, (void*)&d_bufs,  (void*)&d_clk};
  auto stream = at::cuda::getCurrentCUDAStream();
  mk_launch_cooperative(megakernel_ws, dim3(ws_nblocks), dim3(MK_WS_THREADS), args,
                        (size_t)smem_bytes, stream.stream(), d_instrs, n_instr,
                        d_dc, d_ao, d_ad, d_cs, d_cr, d_state, pad, lookahead,
                        d_bufs, d_clk);
}

void mk_run_df2(torch::Tensor instrs, torch::Tensor dep_cnt, torch::Tensor adj_off,
                torch::Tensor adj, torch::Tensor claim_sz, torch::Tensor gated_in,
                torch::Tensor band_tiles, torch::Tensor region_off, torch::Tensor region_cnt0,
                torch::Tensor gate_off, torch::Tensor gate_cons, torch::Tensor gate_k,
                int64_t ring_cap64, torch::Tensor state, torch::Tensor bufs,
                int64_t smem_bytes, c10::optional<torch::Tensor> iclk) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  const int n_instr = (int)(instrs.numel() / (3 + MK_MAX_ARGS));
  const int ring_cap = (int)ring_cap64;

  static int df2_configured = 0;
  static int df2_nblocks = -1;
  if ((int)smem_bytes > df2_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel_df2,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    df2_configured = (int)smem_bytes;
    df2_nblocks = -1;
  }
  if (df2_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel_df2, 256, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel_df2 does not fit an SM with smem=", smem_bytes);
    df2_nblocks = sms * per_sm;
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_dc = dep_cnt.data_ptr<int>();
  const int* d_ao = adj_off.data_ptr<int>();
  const int* d_ad = adj.data_ptr<int>();
  const int* d_cs = claim_sz.data_ptr<int>();
  const int* d_gi = gated_in.data_ptr<int>();
  const int* d_bt = band_tiles.data_ptr<int>();
  const int* d_ro = region_off.data_ptr<int>();
  const int* d_rc = region_cnt0.data_ptr<int>();
  const int* d_go = gate_off.data_ptr<int>();
  const int* d_gc = gate_cons.data_ptr<int>();
  const int* d_gk = gate_k.data_ptr<int>();
  int* d_state = state.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      iclk.has_value() ? reinterpret_cast<long long*>(iclk->data_ptr<int64_t>()) : nullptr;
  void* args[] = {(void*)&d_instrs, (void*)&n_instr, (void*)&ring_cap, (void*)&d_dc,
                  (void*)&d_ao,     (void*)&d_ad,    (void*)&d_cs,     (void*)&d_gi,
                  (void*)&d_bt,     (void*)&d_ro,    (void*)&d_rc,     (void*)&d_go,
                  (void*)&d_gc,     (void*)&d_gk,    (void*)&d_state,  (void*)&d_bufs,
                  (void*)&d_clk};
  auto stream = at::cuda::getCurrentCUDAStream();
  mk_launch_cooperative(megakernel_df2, dim3(df2_nblocks), dim3(256), args,
                        (size_t)smem_bytes, stream.stream(), d_instrs, n_instr,
                        ring_cap, d_dc, d_ao, d_ad, d_cs, d_gi, d_bt, d_ro, d_rc,
                        d_go, d_gc, d_gk, d_state, d_bufs, d_clk);
}

// Host-side CUtensorMap encoder for the n256 TMA feed (MK_GEMM_N256_TMA).
// Returns one 128B tensormap as a CPU uint8 tensor; mk.py concatenates rows
// into a per-program GPU table (rows stay 128B-aligned) referenced through the
// buffer table. bf16 2D row-major only; SWIZZLE_128B matches the SW128 slabs.
torch::Tensor mk_encode_tmap_2d(int64_t ptr, int64_t inner, int64_t outer,
                                int64_t stride_bytes, int64_t box_inner,
                                int64_t box_outer) {
  TORCH_CHECK(ptr % 16 == 0, "tensormap base must be 16B-aligned");
  TORCH_CHECK(stride_bytes % 16 == 0, "tensormap row stride must be 16B-aligned");
  auto out = torch::empty({128}, torch::dtype(torch::kUInt8));
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t gstride[1] = {(cuuint64_t)stride_bytes};
  cuuint32_t bdim[2] = {(cuuint32_t)box_inner, (cuuint32_t)box_outer};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, reinterpret_cast<void*>(ptr), gdim,
      gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap ABI");
  std::memcpy(out.data_ptr<uint8_t>(), &map, 128);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &mk_run, "run megakernel program (wave mode)");
  m.def("encode_tmap_2d", &mk_encode_tmap_2d,
        "encode a 128B CUtensorMap (bf16 2D row-major, SW128) as a CPU u8 tensor");
  m.def("run_df", &mk_run_df, "run megakernel program (dataflow mode)");
  m.def("run_ws", &mk_run_ws, "run megakernel program (warp-specialized dataflow mode)");
#ifdef MK_PDF
  m.def("run_pdf", &mk_run_pdf, "run megakernel program (producer-df register-point mode)");
#endif
  m.def("run_df2", &mk_run_df2, "run megakernel program (region-watermark dataflow mode)");
  m.def("nblocks", &mk_nblocks, "resolved persistent block count");
}
