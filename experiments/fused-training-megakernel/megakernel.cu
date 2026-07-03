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
#include <cuda_bf16.h>

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
};

__device__ __forceinline__ long long mk_globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;  // globally-synchronized ns counter (clock64 is per-SM: incomparable stamps)
}

#include "ops.cuh"
#include "attention.cuh"

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
      op_gemm(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_FWD:
      op_rmsnorm_fwd(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD:
      op_rmsnorm_bwd(I, tile, bufs, smem);
      break;
    case OP_SWIGLU_FWD:
      op_swiglu_fwd(I, tile, bufs);
      break;
    case OP_SWIGLU_BWD:
      op_swiglu_bwd(I, tile, bufs);
      break;
    case OP_QKNORM_ROPE_FWD:
      op_qknorm_rope_fwd(I, tile, bufs, smem);
      break;
    case OP_QKNORM_ROPE_BWD:
      op_qknorm_rope_bwd(I, tile, bufs, smem);
      break;
    case OP_EMBED_FWD:
      op_embed_fwd(I, tile, bufs);
      break;
    case OP_EMBED_BWD:
      op_embed_bwd(I, tile, bufs);
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
  cg::grid_group grid = cg::this_grid();
  const int nblocks = gridDim.x;

  if (wave_clk && blockIdx.x == 0 && threadIdx.x == 0) wave_clk[0] = clock64();
  for (int w = 0; w < nwaves; ++w) {
    const int i0 = wave_start[w], i1 = wave_start[w + 1];
    const int total = wave_tiles[w];
    for (int work = blockIdx.x; work < total; work += nblocks) {
      // locate the instruction owning this work item: scan offsets only (2 ints),
      // copy the full Instr struct just for the owner.
      for (int i = i0; i < i1; ++i) {
        if (work < instrs[i].tile_off + instrs[i].ntiles) {
          const Instr I = instrs[i];
          dispatch(I, work - I.tile_off, bufs, smem);
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
extern "C" __global__ void megakernel_df(const Instr* __restrict__ instrs, int n_instr,
                                         const int* __restrict__ dep_cnt,
                                         const int* __restrict__ adj_off,
                                         const int* __restrict__ adj,
                                         const int* __restrict__ claim_sz, int* state,
                                         void** bufs,
                                         long long* iclk /* nullable [2*n] */) {
  extern __shared__ char smem[];
  cg::grid_group grid = cg::this_grid();
  int* pending = state;
  int* cursor = state + n_instr;
  int* done = state + 2 * n_instr;
  int* ready = state + 3 * n_instr;
  int* ctrl = state + 4 * n_instr;

  // init, then one sync, then seed roots (visible before anyone consumes: the tail
  // increment in the seeding phase publishes them).
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_instr;
       i += gridDim.x * blockDim.x) {
    pending[i] = dep_cnt[i];
    cursor[i] = 0;
    done[i] = 0;
    ready[i] = -1;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ctrl[0] = 0;  // ready tail
    ctrl[1] = 0;  // finished count
    ctrl[2] = 0;  // consumed head
  }
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
  int last_ins = -1;  // sticky: retry the previous instruction before scanning
  volatile int* vready = ready;
  volatile int* vctrl = ctrl;
  for (;;) {
    if (threadIdx.x == 0) {
      s_ins = -1;
      // fast path: most blocks keep pulling tiles from the same big instruction
      if (last_ins >= 0) {
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
        // ctrl[2] = global consumed head: ring entries below it are fully claimed
        const int gq0 = vctrl[2];
        const int tail = vctrl[0];
        for (int q = gq0; q < tail; ++q) {
          const int ins = vready[q];
          if (ins < 0) continue;  // slot reserved, payload not yet visible
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
            break;
          }
        }
        if (s_ins >= 0) break;
        __nanosleep(256);
      }
      last_ins = s_ins;
    }
    __syncthreads();
    const int ins = s_ins;
    if (ins < 0) break;  // everything finished
    const int t0 = s_t0, t1 = s_t1;
    __syncthreads();
    if (iclk && threadIdx.x == 0 && t0 == 0) iclk[2 * ins] = mk_globaltimer();
    const Instr I = instrs[ins];
    for (int t = t0; t < t1; ++t) {
      dispatch(I, t, bufs, smem);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      __threadfence();  // publish this instr's writes before enabling dependents
      const int d = atomicAdd(&done[ins], t1 - t0) + (t1 - t0);
      if (d == I.ntiles) {  // last tile: enable dependents
        if (iclk) iclk[2 * ins + 1] = mk_globaltimer();
        for (int e = adj_off[ins]; e < adj_off[ins + 1]; ++e) {
          const int dep = adj[e];
          if (atomicSub(&pending[dep], 1) == 1) {
            const int t = atomicAdd(&ctrl[0], 1);
            atomicExch(&ready[t], dep);
          }
        }
        atomicAdd(&ctrl[1], 1);
      }
    }
    __syncthreads();
  }
}

// ---- host launcher ------------------------------------------------------------------
static int g_nblocks = -1;

void mk_run(torch::Tensor instrs, torch::Tensor wave_start, torch::Tensor wave_tiles,
            torch::Tensor bufs, int64_t smem_bytes,
            c10::optional<torch::Tensor> wave_clk) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  TORCH_CHECK(wave_start.is_cuda() && wave_tiles.is_cuda() && bufs.is_cuda());
  TORCH_CHECK(bufs.dtype() == torch::kInt64);
  const int nwaves = (int)wave_tiles.numel();

  static bool smem_configured = false;
  if (!smem_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    smem_configured = true;
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
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)megakernel, dim3(g_nblocks),
                                             dim3(256), args, (size_t)smem_bytes,
                                             stream.stream()));
}

int64_t mk_nblocks() { return g_nblocks; }

void mk_run_df(torch::Tensor instrs, torch::Tensor dep_cnt, torch::Tensor adj_off,
               torch::Tensor adj, torch::Tensor claim_sz, torch::Tensor state,
               torch::Tensor bufs, int64_t smem_bytes,
               c10::optional<torch::Tensor> iclk) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  const int n_instr = (int)(instrs.numel() / (3 + MK_MAX_ARGS));

  static bool df_configured = false;
  if (!df_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel_df,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    df_configured = true;
  }
  static int df_nblocks = -1;
  if (df_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel_df, 256, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel_df does not fit an SM with smem=", smem_bytes);
    df_nblocks = sms * per_sm;
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_dc = dep_cnt.data_ptr<int>();
  const int* d_ao = adj_off.data_ptr<int>();
  const int* d_ad = adj.data_ptr<int>();
  const int* d_cs = claim_sz.data_ptr<int>();
  int* d_state = state.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      iclk.has_value() ? reinterpret_cast<long long*>(iclk->data_ptr<int64_t>()) : nullptr;
  void* args[] = {(void*)&d_instrs, (void*)&n_instr, (void*)&d_dc, (void*)&d_ao,
                  (void*)&d_ad,     (void*)&d_cs,    (void*)&d_state, (void*)&d_bufs,
                  (void*)&d_clk};
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)megakernel_df, dim3(df_nblocks),
                                             dim3(256), args, (size_t)smem_bytes,
                                             stream.stream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &mk_run, "run megakernel program (wave mode)");
  m.def("run_df", &mk_run_df, "run megakernel program (dataflow mode)");
  m.def("nblocks", &mk_nblocks, "resolved persistent block count");
}
